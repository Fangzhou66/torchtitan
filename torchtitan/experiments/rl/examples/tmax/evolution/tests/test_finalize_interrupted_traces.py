# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import finalize_interrupted_traces as fit
from torchtitan.experiments.rl.examples.tmax import layout


def _rewrite(
    root: layout.Root, stamp_: str, *, status: str, sessions: dict[str, str]
) -> layout.RewriteDir:
    rw = root.evolution.task("tw_a").rewrite("harder", stamp_)
    layout.write_json_atomic(
        rw.meta,
        {
            "task": "tw_a",
            "status": status,
            "finished": None,
            "signal": "run/tw_a--g1",
            "job": "harder",
        },
    )
    for kind, st in sessions.items():
        sd = rw.session(kind, stamp_)
        layout.write_json_atomic(
            sd.meta, {"kind": kind, "status": st, "finished": None}
        )
    return rw


def test_finalize_marks_only_running_sessions_and_rewrites(tmp_path) -> None:
    root = layout.Root(tmp_path / "root")
    live = _rewrite(
        root,
        "20260904-100000Z",
        status="running",
        sessions={"agent": "completed", "repair": "running"},
    )
    done = _rewrite(
        root, "20260904-090000Z", status="rejected", sessions={"agent": "completed"}
    )

    counts = fit.finalize_interrupted(root, stopped_loop_pid=123)

    assert counts == {"marked": 2, "skipped": 3, "failed": 0}
    meta = json.loads(live.meta.read_text())
    assert meta["status"] == "interrupted" and meta["stopped_loop_pid"] == 123
    assert meta["finished"] and "pid 123" in meta["error"]
    events = layout.read_jsonl(root.evolution.run_outcomes("run"))
    assert len(events) == 2
    assert {event["status"] for event in events} == {"interrupted", "rejected"}
    by_kind = {
        s.path.name.split("--")[1]: json.loads(s.meta.read_text())
        for s in live.session_dirs()
    }
    assert (
        by_kind["repair"]["status"] == "interrupted" and by_kind["repair"]["finished"]
    )
    assert (
        by_kind["agent"]["status"] == "completed"
        and by_kind["agent"]["finished"] is None
    )
    assert json.loads(done.meta.read_text())["status"] == "rejected"

    assert fit.finalize_interrupted(root, stopped_loop_pid=123) == {
        "marked": 0,
        "skipped": 5,
        "failed": 0,
    }
    assert len(layout.read_jsonl(root.evolution.run_outcomes("run"))) == 2


def test_finalize_over_an_empty_root_is_nothing(tmp_path) -> None:
    assert fit.finalize_interrupted(
        layout.Root(tmp_path / "root"), stopped_loop_pid=1
    ) == {"marked": 0, "skipped": 0, "failed": 0}


def test_startup_recovery_records_orphaned_rewrite_once(tmp_path) -> None:
    root = layout.Root(tmp_path / "root")
    rewrite = _rewrite(root, "20260904-100000Z", status="running", sessions={})

    assert fit.finalize_interrupted(root) == {"marked": 1, "skipped": 0, "failed": 0}
    assert fit.finalize_interrupted(root) == {"marked": 0, "skipped": 1, "failed": 0}
    meta = json.loads(rewrite.meta.read_text())
    assert meta["status"] == "interrupted"
    assert meta["error"] == "evolve loop exited before this rewrite finished"
    assert "stopped_loop_pid" not in meta
    assert len(layout.read_jsonl(root.evolution.run_outcomes("run"))) == 1


def test_startup_recovery_repairs_missing_outcome_after_status_write(tmp_path) -> None:
    root = layout.Root(tmp_path / "root")
    _rewrite(root, "20260904-100000Z", status="interrupted", sessions={})

    assert fit.finalize_interrupted(root) == {"marked": 0, "skipped": 1, "failed": 0}
    events = layout.read_jsonl(root.evolution.run_outcomes("run"))
    assert len(events) == 1
    assert events[0]["status"] == "interrupted"


def test_finalize_removes_stopped_session_auth_link_without_touching_account(
    tmp_path,
) -> None:
    root = layout.Root(tmp_path / "root")
    rw = _rewrite(
        root, "20260904-100000Z", status="running", sessions={"agent": "running"}
    )
    account = tmp_path / "account-auth.json"
    account.write_text('{"token": "test-only"}')
    session = next(iter(rw.session_dirs()))
    session.codex_home.mkdir()
    auth = session.codex_home / "auth.json"
    auth.symlink_to(account)

    fit.finalize_interrupted(root, stopped_loop_pid=123)

    assert not auth.is_symlink()
    assert account.read_text() == '{"token": "test-only"}'
    assert json.loads(session.meta.read_text())["status"] == "interrupted"


def _rewrite_in(
    root: layout.Root, task: str, stamp_: str, *, status: str, sessions: dict[str, str]
) -> layout.RewriteDir:
    rw = root.evolution.task(task).rewrite("harder", stamp_)
    layout.write_json_atomic(
        rw.meta,
        {"task": task, "status": status, "signal": f"run/{task}--g1", "job": "harder"},
    )
    for kind, st in sessions.items():
        layout.write_json_atomic(rw.session(kind, stamp_).meta, {"status": st})
    return rw


def _close_in_ledger(root: layout.Root, rws: list[layout.RewriteDir]) -> None:
    root.evolution.ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(root.evolution.ledger, "a") as f:
        for rw in rws:
            ref = str(rw.path.relative_to(root.evolution.path))
            f.write(json.dumps({"outcome": "handled", "rewrite": ref}) + "\n")
        # Lines that close nothing: another outcome, a torn tail.
        f.write(json.dumps({"outcome": "junk", "signal": "run/x--g1"}) + "\n")
        f.write('{"outcome": "hand')


def test_finalize_reads_only_rewrites_the_ledger_left_open(
    tmp_path, monkeypatch
) -> None:
    root = layout.Root(tmp_path / "root")
    closed = [
        _rewrite_in(
            root,
            f"tw_{i % 7}",
            f"20260904-{100000 + i:06d}Z",
            status="accepted",
            sessions={"agent": "completed", "verify": "completed"},
        )
        for i in range(60)
    ]
    # A closed rewrite is final by construction; one that still read
    # `running` would expose a read of it.
    tricky = _rewrite_in(
        root, "tw_0", "20260904-090000Z", status="running", sessions={"agent": "running"}
    )
    _close_in_ledger(root, closed + [tricky])
    running = [
        _rewrite_in(
            root,
            f"tw_{i}",
            f"20260905-{100000 + i:06d}Z",
            status="running",
            sessions={"agent": "completed", "repair": "running"},
        )
        for i in range(3)
    ]
    finished_open = _rewrite_in(
        root, "tw_9", "20260905-120000Z", status="failed", sessions={}
    )

    touched: list[Path] = []
    real_read_text = Path.read_text
    real_exists = Path.exists

    def counting_read_text(self, *a, **kw):
        touched.append(self)
        return real_read_text(self, *a, **kw)

    def counting_exists(self, *a, **kw):
        touched.append(self)
        return real_exists(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", counting_read_text)
    monkeypatch.setattr(Path, "exists", counting_exists)
    counts = fit.finalize_interrupted(root)

    assert counts == {"marked": 6, "skipped": 4, "failed": 0}
    closed_paths = {rw.path for rw in closed + [tricky]}
    assert touched
    assert not [p for p in touched if closed_paths & {p, *p.parents}]
    assert json.loads(real_read_text(tricky.meta))["status"] == "running"
    for rw in running:
        assert json.loads(real_read_text(rw.meta))["status"] == "interrupted"
        repair = next(s for s in rw.session_dirs() if s.path.name.endswith("repair"))
        assert json.loads(real_read_text(repair.meta))["status"] == "interrupted"
    events = layout.read_jsonl(root.evolution.run_outcomes("run"))
    assert sorted(e["rewrite"] for e in events) == sorted(
        str(rw.path.relative_to(root.path)) for rw in running + [finished_open]
    )

    # `include` is examined even when the ledger closed it.
    assert fit.finalize_interrupted(root, include=[tricky])["marked"] == 2
    assert json.loads(real_read_text(tricky.meta))["status"] == "interrupted"


def test_parallel_recover_publishes_every_workspace(tmp_path, monkeypatch) -> None:
    import rewrite_workspace as rw_workspace

    root = layout.Root(tmp_path / "root")
    monkeypatch.setenv("EVOLVE_WORK_ROOT", str(tmp_path / "node-local"))
    monkeypatch.setattr(rw_workspace, "RECOVER_WORKERS", 4)
    source = tmp_path / "source"
    source.mkdir()
    durables = []
    for i in range(10):
        durable = root.evolution.task(f"tw_{i}").rewrite("harder", "20260904-190000Z")
        workspace = rw_workspace.prepare_rewrite(root, durable, source)
        meta = {
            "task": f"tw_{i}",
            "status": "running",
            "signal": f"run/tw_{i}--g1",
            "job": "harder",
        }
        layout.write_json_atomic(workspace.active.meta, meta)
        workspace.claim(meta)
        session = workspace.active.session("agent", "20260904-190001Z")
        session.path.mkdir(parents=True)
        layout.write_json_atomic(session.meta, {"status": "running"})
        session.stderr.write_text(f"cut short {i}\n")
        durables.append(durable)

    published = rw_workspace.recover(root)

    assert sorted(d.path for d in published) == sorted(d.path for d in durables)
    for i, durable in enumerate(durables):
        session = durable.session("agent", "20260904-190001Z")
        assert session.stderr.read_text() == f"cut short {i}\n"
        assert "active_path" not in json.loads(durable.meta.read_text())
    assert not list((tmp_path / "node-local").rglob("rewrite.json"))
    assert fit.finalize_interrupted(root, include=published) == {
        "marked": 20,
        "skipped": 0,
        "failed": 0,
    }
