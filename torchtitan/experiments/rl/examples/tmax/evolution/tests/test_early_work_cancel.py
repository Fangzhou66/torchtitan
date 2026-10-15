# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Early work that will not be used is stopped, the input hashes that gate
reusing it cover symlinks, and cache deletion is per rewrite."""

from __future__ import annotations

import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evolve_codex as ec
import feedback_loop as fb

from rewrite_deadline import RewriteDeadline
from test_blind_verifier import _rewrite, SEED
from test_feedback_loop_gates import _fake_ec, _rewrite as _gate_rewrite, SIGNAL
from test_rewrite_speedups import _harder, _probes, _replays, _revalidate, BOX


@pytest.fixture(autouse=True)
def stable_rewrite_deadline(monkeypatch):
    deadline = RewriteDeadline(0, 10**12, 0)
    monkeypatch.setattr(ec, "rewrite_deadline", lambda _rewrite: deadline)


# -- cancelling dropped early probes --------------------------------------------


def _held_probes(monkeypatch):
    """Probes that run until cancelled; records each call's cancel event."""
    events: list = []

    def probe(package, shortcut=None, resources=None, require_paths=None,
              pretest_file=None, deadline=None, cancel=None):
        events.append(cancel)
        assert cancel is not None and cancel.wait(5), "early probe never cancelled"
        return {"ok": False, "stage": "cancelled"}

    monkeypatch.setattr(fb, "daytona_probe", probe)
    return events


def _wait_for(events, n):
    for _ in range(500):
        if len(events) >= n:
            return
        threading.Event().wait(0.01)
    raise AssertionError("early probes did not start")


@pytest.mark.parametrize("how", ["repair", "box", "close", "restart"])
def test_dropped_early_probes_are_cancelled(tmp_path, monkeypatch, how):
    work, task, orig = _harder(tmp_path)
    events = _held_probes(monkeypatch)
    early = fb.EarlyProbes(work, orig, BOX, None, None)
    early(task)
    _wait_for(events, 2)
    future = early._future
    if how == "repair":
        early(None)
    elif how == "box":
        resources = fb._probe_box(task, {**BOX, "mem_gb": 4})
        assert early.take(work, resources, [], None) is None
    elif how == "close":
        early.close()
    else:
        early(task)  # a second handoff replaces the first run
    assert all(e.is_set() for e in events[:2])
    future.result(timeout=5)
    assert early.dropped
    if how == "restart":
        _wait_for(events, 4)
        early.close()
        assert all(e.is_set() for e in events)


def test_process_one_cancels_unused_early_probes_and_records_it(
    tmp_path, monkeypatch
):
    rw, r0 = _gate_rewrite(tmp_path, monkeypatch)
    events = _held_probes(monkeypatch)
    harder = "#!/bin/sh\ncd /app\ntrue\n"

    def fake_evolve_agentic(rewrite, agent_task, job, **kwargs):
        (rewrite.package / "solution/solve.sh").write_text(harder)
        new = {**agent_task, "solve_sh": harder, "_support_changed": []}
        kwargs["handoff"](new)
        raise RuntimeError("agent died after the handoff")

    monkeypatch.setitem(sys.modules, "evolve_codex",
                        _fake_ec(evolve_agentic=fake_evolve_agentic))
    monkeypatch.setenv("SWE_RETUNE_AGENT", "codex")
    rec = fb.process_one(rw, SIGNAL, job="harder", seed_dir=r0, resources=BOX)
    assert rec["status"] == "failed"
    assert rec["timings"]["daytona_early"] is False
    assert rec["timings"]["daytona_early_dropped"] is True
    _wait_for(events, 2)
    assert all(e.is_set() for e in events)


def test_used_early_probes_are_not_reported_dropped(tmp_path, monkeypatch):
    rw, r0 = _gate_rewrite(tmp_path, monkeypatch)
    _probes(monkeypatch)
    harder = "#!/bin/sh\ncd /app\ntrue\n"

    def fake_evolve_agentic(rewrite, agent_task, job, **kwargs):
        (rewrite.package / "solution/solve.sh").write_text(harder)
        new = {**agent_task, "solve_sh": harder, "_support_changed": []}
        kwargs["handoff"](new)
        return new

    monkeypatch.setitem(sys.modules, "evolve_codex",
                        _fake_ec(evolve_agentic=fake_evolve_agentic))
    monkeypatch.setenv("SWE_RETUNE_AGENT", "codex")
    rec = fb.process_one(rw, SIGNAL, job="harder", seed_dir=r0, resources=BOX)
    assert rec["status"] == "accepted", rec
    assert rec["timings"]["daytona_early"] is True
    assert rec["timings"]["daytona_early_dropped"] is False


_CHILD = r"""
import signal, sys, time
mode, marker = sys.argv[1], sys.argv[2]
def on_int(*_):
    if mode == "ignore":
        return
    open(marker, "w").write("cleaned up")
    sys.exit(130)
signal.signal(signal.SIGINT, on_int)
signal.signal(signal.SIGTERM, signal.SIG_IGN if mode == "ignore" else signal.SIG_DFL)
print("started", flush=True)
time.sleep(60)
"""


@pytest.mark.parametrize("mode", ["cleanup", "ignore"])
def test_cancelled_probe_subprocess_is_interrupted_then_killed_and_reaped(
    tmp_path, monkeypatch, mode
):
    monkeypatch.setattr(fb, "DAYTONA_CANCEL_GRACE_SEC", 1)
    monkeypatch.setattr(fb, "DAYTONA_KILL_GRACE_SEC", 1)
    marker = tmp_path / "marker"
    cancel = threading.Event()
    popens = []
    real_popen = fb.subprocess.Popen

    def popen(*args, **kwargs):
        popens.append(real_popen(*args, **kwargs))
        return popens[-1]

    monkeypatch.setattr(fb.subprocess, "Popen", popen)
    threading.Timer(1.0, cancel.set).start()
    got = fb._run_probe(
        [sys.executable, "-c", _CHILD, mode, str(marker)], 60, cancel
    )
    assert got is None
    (p,) = popens
    assert p.returncode is not None  # reaped, not a zombie
    if mode == "cleanup":
        assert marker.read_text() == "cleaned up"  # SIGINT let it clean up
        assert p.returncode == 130
    else:
        assert p.returncode == -9  # ignored SIGINT and SIGTERM: killed


def test_cancelled_daytona_probe_does_not_retry(tmp_path, monkeypatch):
    env = tmp_path / "daytona.env"
    env.write_text("")
    monkeypatch.setattr(fb, "DAYTONA_VENV_PY", sys.executable)
    monkeypatch.setattr(fb, "DAYTONA_ENV_FILE", str(env))
    runs = []

    def run_probe(cmd, timeout, cancel):
        runs.append(cancel)
        return None

    monkeypatch.setattr(fb, "_run_probe", run_probe)
    cancel = threading.Event()
    cancel.set()
    verdict = fb.daytona_probe(tmp_path, cancel=cancel)
    assert verdict["stage"] == "cancelled" and not verdict["ok"]
    assert len(runs) == 1


def test_uncancelled_probe_subprocess_returns_its_output():
    got = fb._run_probe(
        [sys.executable, "-c", "print('{\"ok\": true}')"], 30, threading.Event()
    )
    assert got.returncode == 0 and got.stdout.strip() == '{"ok": true}'


# -- symlinks in the input hashes ------------------------------------------------


def test_probe_digest_hashes_symlink_targets(tmp_path):
    work = tmp_path / "pkg"
    work.mkdir()
    (work / "a.txt").write_text("same\n")
    (work / "b.txt").write_text("same\n")
    (work / "link").symlink_to("a.txt")
    before = fb._probe_inputs_digest(work, None)
    (work / "link").unlink()
    (work / "link").symlink_to("b.txt")
    assert fb._probe_inputs_digest(work, None) != before


def test_directory_symlink_never_reuses_early_probes(tmp_path, monkeypatch):
    work, task, orig = _harder(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "data.txt").write_text("x\n")
    (work / "environment/shared").symlink_to(outside)
    assert fb._probe_inputs_digest(work, None) is None
    calls = _probes(monkeypatch)
    early = fb.EarlyProbes(work, orig, BOX, None, None)
    early(task)
    assert not early.started
    verdict = _revalidate(work, task, orig, early)
    assert verdict["ok"] and not early.used
    assert len(calls) == 2  # the final probes only


def test_replay_inputs_see_exec_bits_symlinks_and_refuse_dir_symlinks(tmp_path):
    vpkg, probe = tmp_path / "vpkg", tmp_path / "probe"
    (vpkg / "tests").mkdir(parents=True)
    (probe / "run/verifier-probes").mkdir(parents=True)
    script = vpkg / "tests/test.sh"
    script.write_text("true\n")
    (vpkg / "tests/other.sh").write_text("true\n")
    first = ec._replay_inputs(vpkg, probe)
    script.chmod(0o755)
    assert ec._replay_inputs(vpkg, probe) != first
    (vpkg / "tests/link.sh").symlink_to("test.sh")
    linked = ec._replay_inputs(vpkg, probe)
    (vpkg / "tests/link.sh").unlink()
    (vpkg / "tests/link.sh").symlink_to("other.sh")
    assert ec._replay_inputs(vpkg, probe) != linked
    (vpkg / "data").symlink_to(probe, target_is_directory=True)
    assert ec._replay_inputs(vpkg, probe) is None


def test_early_replay_is_rerun_when_inputs_cannot_be_hashed(tmp_path):
    vpkg, probe = tmp_path / "vpkg", tmp_path / "probe"
    vpkg.mkdir()
    (probe / "run/verifier-probes").mkdir(parents=True)
    (vpkg / "d").symlink_to(probe, target_is_directory=True)
    with ThreadPoolExecutor(max_workers=1) as pool:
        early = pool.submit(lambda: (({}, {}), None))
    assert ec._early_outcome(early, vpkg, probe) is ec._RERUN


# -- per-rewrite cache deletion --------------------------------------------------


def test_drain_waits_only_for_its_own_rewrites_deletions(tmp_path, monkeypatch):
    gate = threading.Event()
    real_rmtree = shutil.rmtree

    def rmtree(path, ignore_errors=False):
        if "held" in str(path):
            assert gate.wait(10)
        real_rmtree(path, ignore_errors=ignore_errors)

    monkeypatch.setattr(ec.shutil, "rmtree", rmtree)
    held = tmp_path / "held/.codex-cache-1"
    mine = [tmp_path / f"mine/.codex-cache-{i}" for i in range(3)]
    for d in [held, *mine]:
        (d / "x").mkdir(parents=True)
        ec._delete_in_background(d)
    # The other rewrite's deletion is still held; this one's drain returns.
    ec.drain_cache_cleanup(tmp_path / "mine")
    assert not any(d.exists() for d in mine)
    assert held.exists()
    gate.set()
    ec.drain_cache_cleanup(tmp_path / "held")
    assert not held.exists()


# -- a rejected original replay skips the early replay --------------------------


def test_rejected_original_replay_skips_the_early_replay(tmp_path, monkeypatch):
    rw = _rewrite(tmp_path, monkeypatch)
    _started, replays = _replays(monkeypatch)
    probe_gate = threading.Event()
    real_author = ec._author_independent_probes

    def slow_author(*args, **kwargs):
        assert probe_gate.wait(5)
        return real_author(*args, **kwargs)

    def rejected(vsession, rewrite=None, deadline=None):
        # The probe author finishes only after the rejection has propagated.
        threading.Timer(0.5, probe_gate.set).start()
        raise RuntimeError("original controls rejected")

    monkeypatch.setattr(ec, "_author_independent_probes", slow_author)
    monkeypatch.setattr(ec, "_verify_original_probes", rejected)
    with pytest.raises(RuntimeError, match="original controls rejected"):
        ec._blind_verifier(rw, dict(SEED), ec.ev.file_map(SEED))
    assert replays == []
