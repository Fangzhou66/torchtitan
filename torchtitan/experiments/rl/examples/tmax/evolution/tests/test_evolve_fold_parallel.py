# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""fold() prepares a batch's packages in parallel and commits serially: the
published mix is the same as a serial fold's, and one bad package fails alone."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evolve_ondella as od
import test_evolve_lineage as tl

N = 8
PREP_SEC = 0.3


def _batch(tmp_path, monkeypatch, name: str):
    """A root with N tasks and one accepted, not yet folded, rewrite of each."""
    root = tl._root(tmp_path / name, monkeypatch)
    monkeypatch.setenv("SWE_POLICY_CANARY_K", "0")
    tids = ["tw_a"] + [f"tw_{i}" for i in range(1, N)]
    for tid in tids[1:]:
        tl._add_task(root, tid)
    tl._stub(monkeypatch)
    batch = []
    for group, tid in enumerate(tids, start=7):
        tl._signal(root, task=tid, group=group)
        h = od.handle(root, od._replay_signal(root, f"{tl.RUN}/{tid}--g{group}"), declared={})
        assert h["status"] == "accepted"
        batch.append(h)
    return root, batch


def _slow(monkeypatch, fail: str | None = None):
    real = od.candidate_row

    def candidate_row(package, input_row, meta):
        time.sleep(PREP_SEC)
        if meta["task"] == fail:
            raise RuntimeError("malformed package")
        return real(package, input_row, meta)

    monkeypatch.setattr(od, "candidate_row", candidate_row)


def _fold(tmp_path, monkeypatch, workers: int, fail: str | None = None):
    root, batch = _batch(tmp_path, monkeypatch, f"w{workers}-{fail}")
    monkeypatch.setenv("EVOLVE_FOLD_WORKERS", str(workers))
    _slow(monkeypatch, fail)
    t0 = time.monotonic()
    version = od.fold(root, batch)
    return root, batch, version, time.monotonic() - t0


def test_parallel_fold_matches_serial_and_is_faster(tmp_path, monkeypatch):
    serial_root, serial_batch, sv, serial_sec = _fold(tmp_path, monkeypatch, 1)
    par_root, par_batch, pv, par_sec = _fold(tmp_path, monkeypatch, 4)

    assert serial_sec >= N * PREP_SEC
    # N / workers rounds of PREP_SEC, plus slack for the commit.
    assert par_sec < (N / 4) * PREP_SEC + 1.0, (par_sec, serial_sec)
    assert sv == pv
    assert par_root.mix.live.read_text() == serial_root.mix.live.read_text()
    labels = [json.loads(l)["label"] for l in par_root.mix.live.read_text().splitlines()]
    assert labels == ["tw_a"] + [f"tw_{i}" for i in range(1, N)]
    for s, p in zip(serial_batch, par_batch):
        assert (s["status"], s["meta"]["result_rev"]) == (p["status"], p["meta"]["result_rev"]) == ("accepted", 1)
        assert par_root.evolution.task(p["meta"]["task"]).rev(1).is_dir()
        assert not (par_root.evolution.task(p["meta"]["task"]).rev(1) / "AGENTS.md").exists()


def test_one_failed_prep_does_not_touch_the_others(tmp_path, monkeypatch):
    root, batch, version, _ = _fold(tmp_path, monkeypatch, 4, fail="tw_3")

    assert version is not None
    by_task = {h["meta"]["task"]: h for h in batch}
    bad = by_task.pop("tw_3")
    assert bad["status"] == "failed"
    assert bad["meta"]["stage"] == "fold" and "malformed package" in bad["meta"]["reason"]
    assert json.loads(bad["rewrite"].meta.read_text())["status"] == "failed"
    assert not root.evolution.task("tw_3").rev(1).exists()
    assert all(h["status"] == "accepted" for h in by_task.values())
    live = {json.loads(l)["label"]: json.loads(l) for l in root.mix.live.read_text().splitlines()}
    assert live["tw_3"]["metadata"]["rev"] == 0
    assert all(live[t]["metadata"]["rev"] == 1 for t in by_task)
