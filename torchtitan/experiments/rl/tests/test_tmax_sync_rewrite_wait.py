# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import threading

from torchtitan.experiments.rl.examples.tmax import layout, rollouter


def _root(tmp_path):
    root = layout.Root(tmp_path)
    run = layout.Run(root.runs / "tmax-9b--20261018-000000Z")
    run.signals.mkdir(parents=True)
    root.evolution.path.mkdir(parents=True)
    return root, run


def _ledger(root, sid, outcome="handled"):
    with open(root.evolution.ledger, "a") as f:
        f.write(json.dumps({"signal": sid, "outcome": outcome}) + "\n")


def test_pending_until_every_signal_has_a_ledger_line(tmp_path):
    root, run = _root(tmp_path)
    assert rollouter._pending_signals(root, run.name) == []
    run.signal("task-a", 0).write_text("{}")
    run.signal("task-a", 1).write_text("{}")
    sid0 = layout.signal_id(run.name, "task-a", 0)
    sid1 = layout.signal_id(run.name, "task-a", 1)
    assert rollouter._pending_signals(root, run.name) == [sid0, sid1]
    _ledger(root, sid0)
    assert rollouter._pending_signals(root, run.name) == [sid1]
    _ledger(root, sid1, outcome="deferred")
    assert rollouter._pending_signals(root, run.name) == []


def test_wait_returns_once_the_loop_writes_the_ledger(tmp_path, monkeypatch):
    root, run = _root(tmp_path)
    monkeypatch.setenv("TRL_BASE", str(tmp_path))
    monkeypatch.setenv("TRL_RUN_DIR", str(run.path))
    assert rollouter._wait_for_rewrites() is False
    run.signal("task-a", 0).write_text("{}")
    monkeypatch.setattr(rollouter.time, "sleep", lambda s: _ledger(
        root, layout.signal_id(run.name, "task-a", 0)
    ))
    assert rollouter._wait_for_rewrites() is True
    assert rollouter._pending_signals(root, run.name) == []


def test_wait_gives_up_after_the_timeout(tmp_path, monkeypatch):
    root, run = _root(tmp_path)
    monkeypatch.setenv("TRL_BASE", str(tmp_path))
    monkeypatch.setenv("TRL_RUN_DIR", str(run.path))
    monkeypatch.setattr(rollouter, "_SYNC_REWRITE_TIMEOUT_SEC", 0.0)
    run.signal("task-a", 0).write_text("{}")
    calls = threading.Event()
    monkeypatch.setattr(rollouter.time, "sleep", lambda s: calls.set())
    assert rollouter._wait_for_rewrites() is True
    assert not calls.is_set()


def test_earlier_runs_and_other_attempts(tmp_path):
    root, run = _root(tmp_path)
    old = layout.Run(root.runs / "tmax-9b--20261017-000000Z")
    old.signals.mkdir(parents=True)
    old.signal("task-a", 0).write_text("{}")
    assert rollouter._pending_signals(root, run.name) == []
    retry = layout.Run(root.runs / f"{run.name}--attempt-20261018-010000Z--abc")
    retry.signals.mkdir(parents=True)
    retry.signal("task-a", 3).write_text("{}")
    attempt = f"{run.name}--attempt-20261018-020000Z--def"
    assert rollouter._pending_signals(root, attempt) == [
        layout.signal_id(retry.name, "task-a", 3)
    ]
