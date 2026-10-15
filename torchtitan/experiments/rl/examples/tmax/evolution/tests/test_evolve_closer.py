# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The coordinator does not wait for folds: the closer thread folds and closes
finished rewrites while the coordinator keeps collecting and refilling, a task
stays busy until its ledger line is written, and the discovery reads that every
refill repeats are made once."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from threading import Event

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evolve_ondella as od
import test_evolve_lineage as tl
from torchtitan.experiments.rl.examples.tmax import layout


def test_a_rewrite_finishing_during_a_fold_is_collected_and_its_slot_refilled(
    tmp_path, monkeypatch
):
    """One worker, three tasks. tw_b runs while tw_a's fold is held and
    finishes during it; it is collected then, and its slot takes tw_c before
    that fold is over. A coordinator that folded inline collected tw_b only
    once the fold returned, so tw_c could not start in time."""
    root = tl._root(tmp_path, monkeypatch)
    for tid in ("tw_b", "tw_c"):
        tl._add_task(root, tid)
    before = root.mix.live_version()[0]
    for group, (tid, created) in enumerate(
        (("tw_a", "20260904-190000Z"), ("tw_b", "20260904-180000Z"), ("tw_c", "20260904-170000Z")),
        start=7,
    ):
        tl._signal(root, task=tid, group=group, created=created)
    seen = tl._stub(monkeypatch)
    c_collected = Event()
    real_submit = od._Closer.submit

    def submit(self, batch):
        real_submit(self, batch)
        if any(h["meta"]["task"] == "tw_c" for h in batch):
            c_collected.set()

    real_fold = od.fold
    held: list[bool] = []

    def slow_fold(root_, accepted):
        if not held:
            held.append(c_collected.wait(10))
        return real_fold(root_, accepted)

    monkeypatch.setattr(od._Closer, "submit", submit)
    monkeypatch.setattr(od, "fold", slow_fold)
    monkeypatch.setattr(od, "FREE_SLOT_POLL_SEC", 0.01)
    r = od.run_round(root, workers=1)
    assert held == [True], "tw_c waited for tw_a's fold"
    assert [s["signal"]["task"] for s in seen] == ["tw_a", "tw_b", "tw_c"]
    assert (r["handled"], r["accepted"]) == (3, 3), r
    # tw_a alone, then tw_b and tw_c, queued behind it, in one version.
    assert [v for v, _ in root.mix.versions() if v > before] == [before + 1, before + 2]
    live = {json.loads(l)["label"]: json.loads(l) for l in root.mix.live.read_text().splitlines()}
    assert {t: live[t]["metadata"]["rev"] for t in live} == {"tw_a": 1, "tw_b": 1, "tw_c": 1}
    assert [line["outcome"] for line in tl._ledger(root)] == ["handled"] * 3


def test_a_task_being_folded_is_not_started_again(tmp_path, monkeypatch):
    """A signal arriving for a task whose fold is still running waits for the
    close, then is judged against the new revision."""
    root = tl._root(tmp_path, monkeypatch)
    tl._signal(root, group=7)
    seen = tl._stub(monkeypatch)
    real_fold = od.fold
    folding, release = Event(), Event()

    def held_fold(root_, accepted):
        # Written while the fold runs, about the revision being replaced.
        tl._signal(root, group=8, created="20260904-190000Z")
        folding.set()
        assert release.wait(10)
        return real_fold(root_, accepted)

    real_select = od.select_work
    selects_during_fold: list[int] = []

    def select(*a, **kw):
        out = real_select(*a, **kw)
        if folding.is_set() and not release.is_set():
            selects_during_fold.append(len(seen))
            if len(selects_during_fold) >= 3:
                release.set()
        return out

    monkeypatch.setattr(od, "fold", held_fold)
    monkeypatch.setattr(od, "select_work", select)
    monkeypatch.setattr(od, "FREE_SLOT_POLL_SEC", 0.01)
    r = od.run_round(root, workers=2)
    assert selects_during_fold and set(selects_during_fold) == {1}, selects_during_fold
    assert len(seen) == 1 and r["handled"] == 1 and r["accepted"] == 1
    lines = {line["signal"]: line for line in tl._ledger(root)}
    late = layout.signal_id(tl.RUN, "tw_a", 8)
    assert lines[late]["outcome"] == "superseded"
    assert "not the task's current rev 1" in lines[late]["reason"]


def test_a_failed_fold_ends_the_round_and_leaves_the_signal_open(tmp_path, monkeypatch):
    root = tl._root(tmp_path, monkeypatch)
    sid = tl._signal(root)
    tl._stub(monkeypatch)

    def broken(root_, accepted):
        raise OSError("mix storage unavailable")

    monkeypatch.setattr(od, "fold", broken)
    with pytest.raises(OSError, match="mix storage unavailable"):
        od.run_round(root, workers=1)
    assert tl._ledger(root) == []
    [rewrite] = root.evolution.task("tw_a").rewrite_dirs()
    assert json.loads(rewrite.meta.read_text())["status"] == "running"
    assert sid in {s.sid for s in od.discover(root, od.load_ledger(root))}


def test_a_pending_signal_is_read_once(tmp_path, monkeypatch):
    root = tl._root(tmp_path, monkeypatch)
    tl._signal(root)
    calls = []
    real = od.read_signal
    monkeypatch.setattr(od, "read_signal", lambda p: calls.append(p) or real(p))
    first = od.discover(root, {})
    second = od.discover(root, {})
    assert len(calls) == 1
    assert first[0].data == second[0].data and first[0].data is not second[0].data


def test_an_unreadable_young_signal_is_read_again(tmp_path, monkeypatch):
    root = tl._root(tmp_path, monkeypatch)
    path = root.run(tl.RUN).signal("tw_a", 3)
    path.parent.mkdir(parents=True)
    path.write_text("")
    assert od.discover(root, {}) == []
    tl._signal(root, group=3)
    assert [s.group for s in od.discover(root, {})] == [3]


def test_settled_rollout_headers_are_read_once(tmp_path, monkeypatch):
    root = tl._root(tmp_path, monkeypatch)
    tl._second_seed(root)
    tl._signal(root, task="tw_a", group=7)
    tl._signal(root, task="tw_b", group=8)
    tl._later_group(root, "tw_b", 9, [1.0, 1.0], rev=1)  # read, then skipped
    reads = []
    real = od.rollout_record.read_header
    monkeypatch.setattr(
        od.rollout_record, "read_header", lambda p: reads.append(p) or real(p)
    )
    signals = [s for s in od.discover(root, {}) if s.task == "tw_b"]
    for _ in range(3):
        assert od._resolved_by(root, signals[0], {}) is None
    assert len(reads) == 2


def test_publish_records_the_written_bytes_digest(tmp_path):
    mix = layout.MixDir(tmp_path / "mix")
    version, path = mix.publish(['{"label": "café"}'])
    manifest = json.loads(mix.manifest_of(path).read_text())
    assert manifest["sha256"] == layout.sha256_file(path)
    assert mix.live_version()[0] == version


def test_status_refresh_runs_beside_the_coordinator(tmp_path, monkeypatch):
    """A slow status rebuild does not hold a finished rewrite's fold."""
    root = tl._root(tmp_path, monkeypatch)
    tl._signal(root)
    tl._stub(monkeypatch)
    real_rebuild = od.rebuild_status
    folded = Event()

    def slow(root_, *, active_tasks=None):
        if active_tasks is not None:
            assert folded.wait(10), "the fold waited for the status refresh"
        return real_rebuild(root_, active_tasks=active_tasks)

    real_fold = od.fold

    def fold(root_, accepted):
        version = real_fold(root_, accepted)
        folded.set()
        return version

    monkeypatch.setattr(od, "rebuild_status", slow)
    monkeypatch.setattr(od, "fold", fold)
    monkeypatch.setattr(od, "STATUS_REFRESH_SEC", 0.0001)
    process = od.fb.process_one

    def lagging(*a, **kw):
        time.sleep(0.05)  # long enough for a status refresh to start
        return process(*a, **kw)

    monkeypatch.setattr(od.fb, "process_one", lagging)
    r = od.run_round(root, workers=1)
    assert r["accepted"] == 1 and folded.is_set()
