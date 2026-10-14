# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import time

import pytest

from torchtitan.experiments.rl.examples.tmax import layout
from torchtitan.experiments.rl.examples.tmax.evolution import evolve_ondella as od
from torchtitan.experiments.rl.examples.tmax.evolution.rewrite_admission import (
    admission_is_open,
)


@pytest.mark.parametrize(
    "payload",
    [
        None,
        "bad json",
        {},
        {"allow_admission": True, "expires_at": 0},
        {"allow_admission": False, "expires_at": 1e20},
    ],
)
def test_missing_expired_or_closed_lease_pauses(monkeypatch, tmp_path, payload):
    path = tmp_path / "admission.json"
    monkeypatch.setenv("EVOLVE_ADMISSION_STATUS_FILE", str(path))
    if payload is not None:
        path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    assert not admission_is_open()
    assert admission_is_open(dry=True)


def test_fresh_lease_and_unconfigured_workflow(monkeypatch, tmp_path):
    monkeypatch.delenv("EVOLVE_ADMISSION_STATUS_FILE", raising=False)
    assert admission_is_open()
    path = tmp_path / "admission.json"
    path.write_text(
        json.dumps({"allow_admission": True, "expires_at": time.time() + 60})
    )
    monkeypatch.setenv("EVOLVE_ADMISSION_STATUS_FILE", str(path))
    assert admission_is_open()


def test_paused_round_does_not_select_or_close_signals(monkeypatch, tmp_path):
    monkeypatch.setenv("EVOLVE_ADMISSION_STATUS_FILE", str(tmp_path / "missing"))
    monkeypatch.setattr(
        od, "select_work", lambda *_a, **_kw: pytest.fail("selected work")
    )
    root = layout.Root(tmp_path / "root")
    result = od.run_round(root)
    assert result["reason"] == "training admission paused"
    assert result["handled"] == 0


def test_paused_wait_does_not_spin_on_pending_signals(monkeypatch, tmp_path):
    monkeypatch.setenv("EVOLVE_ADMISSION_STATUS_FILE", str(tmp_path / "missing"))
    monkeypatch.setattr(od, "_pending", lambda *_a: pytest.fail("read pending signals"))
    clock = iter([0, 0, 5, 10])
    sleeps = []
    monkeypatch.setattr(od.time, "time", lambda: next(clock))
    monkeypatch.setattr(od.time, "sleep", sleeps.append)
    od._wait_for_signals(layout.Root(tmp_path / "root"), max_wait=10)
    assert sleeps == [5, 5]
