"""Fast replay of recorded solutions: start a turn once the terminal is idle,
never later than the recorded gap, and fall back to the recorded timing."""

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import terminus_validation as tv
from torchtitan.experiments.rl.harness.agents.spec import AgentRun


def action(step, offset, keys="true\n", kind="terminal_action"):
    return {
        "step_id": step,
        "offset_sec": offset,
        "keystrokes": keys if kind == "terminal_action" else "",
        "duration": 0.1 if kind == "terminal_action" else 0,
        "kind": kind,
    }


def recording():
    return [
        action(1, 100.0),
        action(2, 300.0),
        action(3, 400.0, kind="completion_marker"),
        action(4, 410.0, kind="completion_marker"),
    ]


def state(idle, screen="s", background=()):
    return {"idle": idle, "screen": screen, "background": frozenset(background)}


class FakeProbe:
    """Returns the scripted states in order, the last one forever; each read costs 0.1 s."""

    def __init__(self, clock, states):
        self.clock = clock
        self.states = list(states)
        self.calls = 0

    async def sample(self):
        self.calls += 1
        self.clock[0] += 0.1
        item = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]

    async def sleep(delay):
        assert delay >= 0
        now[0] += delay

    monkeypatch.setattr(tv.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(tv.asyncio, "sleep", sleep)
    monkeypatch.delenv(tv.FAST_REPLAY_ENV, raising=False)
    return now


def turn(adapter):
    return asyncio.run(adapter.complete("s", {"messages": [{"content": "pane"}]}))


def test_waits_for_idle_not_the_recorded_offset(clock):
    probe = FakeProbe(clock, [state(False), state(False), state(True), state(True)])
    adapter = tv.RecordedActionsAdapter(recording(), probe=probe)
    turn(adapter)
    wait = adapter.waits[0]
    assert wait["reason"] == "idle" and wait["recorded_sec"] == 100.0
    assert wait["waited_sec"] < 3
    assert adapter.transcript[0]["replay_wait"] == wait


def test_screen_must_be_still_for_idle(clock):
    probe = FakeProbe(
        clock,
        [state(True, "a"), state(True, "b"), state(True, "c"), state(True, "c")],
    )
    adapter = tv.RecordedActionsAdapter(recording(), probe=probe)
    turn(adapter)
    assert adapter.waits[0]["reason"] == "idle" and probe.calls == 4


def test_busy_terminal_waits_exactly_the_recorded_gap(clock):
    probe = FakeProbe(clock, [state(False)])
    adapter = tv.RecordedActionsAdapter(recording(), probe=probe)
    turn(adapter)
    assert adapter.waits[0]["reason"] == "gap"
    assert adapter.waits[0]["waited_sec"] == pytest.approx(100.0, abs=0.2)


def test_later_turns_are_bounded_by_their_own_recorded_gap(clock):
    probe = FakeProbe(clock, [state(True), state(True), state(False)])
    adapter = tv.RecordedActionsAdapter(recording(), probe=probe)
    turn(adapter)  # idle after ~0.7 s instead of 100 s
    assert adapter.waits[0]["reason"] == "idle"
    clock[0] += 0.1  # the turn's command duration
    turn(adapter)  # busy: capped at the recorded 200 s gap, not 300 s
    assert adapter.waits[1]["reason"] == "gap"
    assert adapter.waits[1]["recorded_sec"] == pytest.approx(200.0 - 0.1, abs=0.01)


def test_new_background_process_keeps_the_recorded_gap(clock):
    probe = FakeProbe(
        clock,
        [
            state(True, background={7}),
            state(True, background={7}),
            # Turn 2 started a server (pid 9) that is still running.
            state(True, background={7, 9}),
            state(True, background={7, 9}),
            # Turn 3 changed nothing in the background.
            state(True, background={7, 9}),
        ],
    )
    adapter = tv.RecordedActionsAdapter(recording(), probe=probe)
    turn(adapter)
    turn(adapter)
    turn(adapter)
    assert [w["reason"] for w in adapter.waits] == ["idle", "background", "idle"]
    assert adapter.waits[1]["waited_sec"] == pytest.approx(
        adapter.waits[1]["recorded_sec"]
    )


def test_probe_failure_falls_back_to_recorded_timing(clock):
    probe = FakeProbe(clock, [RuntimeError("no tmux socket"), state(True)])
    adapter = tv.RecordedActionsAdapter(recording(), probe=probe)
    turn(adapter)
    turn(adapter)
    assert [w["reason"] for w in adapter.waits] == ["fallback", "fallback"]
    assert adapter.waits[0]["waited_sec"] == pytest.approx(100.0)
    assert adapter.waits[1]["waited_sec"] == pytest.approx(200.0, abs=0.2)
    assert probe.calls == 1 and "no tmux socket" in adapter.summary()["fallback"]


def test_disabled_by_env_keeps_original_timing(clock, monkeypatch):
    monkeypatch.setenv(tv.FAST_REPLAY_ENV, "0")
    probe = FakeProbe(clock, [state(True)])
    adapter = tv.RecordedActionsAdapter(recording(), probe=probe)
    sleeps = []
    for _ in range(4):
        start = clock[0]
        turn(adapter)
        sleeps.append(clock[0] - start)
        clock[0] += 0.1
    assert probe.calls == 0
    assert sleeps == pytest.approx([100.0, 199.9, 99.9, 9.9])
    assert adapter.summary()["reasons"] == {"recorded": 4}


def test_submission_and_actions_are_unchanged_by_fast_replay(clock):
    fast = tv.RecordedActionsAdapter(recording(), probe=FakeProbe(clock, [state(True)]))
    slow = tv.RecordedActionsAdapter(recording())
    for _ in range(4):
        a, b = turn(fast), turn(slow)
        assert a["content"][0]["text"] == b["content"][0]["text"]
    assert fast.exit_code == slow.exit_code == 0


def stat(pid, comm, state_, ppid, pgrp, tpgid):
    rest = " ".join(["0"] * 44)
    return f"{pid} ({comm}) {state_} {ppid} {pgrp} {pgrp} 34816 {tpgid} {rest}"


def probe_output(stats, pane=10, student=""):
    lines = ["SOCK /var/tmp/terminus-x/tmux-0/default", "SELF 500", f"PANE {pane}"]
    lines.append("SCREEN 123 45")
    if student:
        lines.append(f"STUDENT {student}")
    return "\n".join(lines + ["STAT"] + stats) + "\n"


def test_read_terminal_idle_shell_with_background_server():
    out = probe_output(
        [
            stat(1, "daytona", "S", 0, 1, -1),
            stat(10, "bash", "S", 5, 10, 10),
            stat(11, "python3 server", "S", 10, 11, 10),
            stat(500, "sh", "S", 1, 500, -1),
            stat(501, "cat", "R", 500, 500, -1),
        ]
    )
    seen = tv.read_terminal(out)
    assert seen["idle"] and seen["screen"] == "123 45"
    assert seen["background"] == {11}


def test_read_terminal_busy_foreground_and_programs():
    busy = tv.read_terminal(
        probe_output(
            [stat(10, "bash", "S", 5, 10, 12), stat(12, "make", "R", 10, 12, 12)]
        )
    )
    assert not busy["idle"] and busy["background"] == frozenset()
    repl = tv.read_terminal(
        probe_output(
            [stat(10, "bash", "S", 5, 10, 12), stat(12, "python3", "S", 10, 12, 12)]
        )
    )
    assert not repl["idle"]
    running = tv.read_terminal(probe_output([stat(10, "bash", "R", 5, 10, 10)]))
    assert not running["idle"]
    nested = tv.read_terminal(
        probe_output(
            [stat(10, "bash", "S", 5, 10, 12), stat(12, "bash", "S", 10, 12, 12)]
        )
    )
    assert nested["idle"]


def test_read_terminal_counts_student_cgroup_and_skips_zombies():
    seen = tv.read_terminal(
        probe_output(
            [
                stat(10, "bash", "S", 5, 10, 10),
                stat(20, "nginx", "S", 1, 20, -1),
                stat(21, "defunct", "Z", 10, 21, 10),
            ],
            student="10 20 21",
        )
    )
    assert seen["idle"] and seen["background"] == {20}


def test_read_terminal_rejects_output_without_pane():
    with pytest.raises(ValueError):
        tv.read_terminal("SELF 1\nSTAT\n")


def test_run_reference_probes_the_session_terminus_runs(monkeypatch):
    sb = SimpleNamespace(
        exec=AsyncMock(return_value=(0, "tmux 3.2", "")),
        write_file=AsyncMock(),
        read_file=AsyncMock(return_value="pane"),
    )

    async def run(task, *, terminal_session_name, parser_name):
        assert task.adapter.probe.session == terminal_session_name
        assert task.adapter.probe.sb is sb
        task.adapter.exit_code = 0
        return AgentRun(turns=4, submitted=True, finish_reason="submit", pane_path="/p")

    monkeypatch.setattr(tv, "terminus_agent", run)
    monkeypatch.delenv(tv.FAST_REPLAY_ENV, raising=False)
    result = asyncio.run(tv.run_reference(sb, "", 30, actions=recording()))
    assert result["solve_exit"] == 0 and result["replay"]["fast"] is True


@pytest.mark.skipif(not shutil.which("tmux"), reason="needs tmux")
def test_probe_reads_a_real_tmux_pane():
    directory = Path(tempfile.mkdtemp(prefix="terminus-", dir="/var/tmp"))
    session = "oracle-" + uuid.uuid4().hex
    env = {**os.environ, "TMUX_TMPDIR": str(directory)}

    def tmux(*args):
        subprocess.run(["tmux", *args], env=env, check=True, capture_output=True)

    class LocalSandbox:
        async def exec(self, cmd, check=False, timeout=30):
            p = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
            return p.returncode, p.stdout, p.stderr

    tmux("new-session", "-d", "-s", session, "-x", "80", "-y", "24", "bash --norc")
    try:
        probe = tv.TerminalProbe(LocalSandbox(), session)

        def until(predicate):
            for _ in range(50):
                seen = asyncio.run(probe.sample())
                if predicate(seen):
                    return seen
                time.sleep(0.1)
            raise AssertionError(seen)

        until(lambda s: s["idle"])
        assert probe.socket.startswith(str(directory))
        tmux("send-keys", "-t", session, "sleep 30", "Enter")
        until(lambda s: not s["idle"])
        tmux("send-keys", "-t", session, "C-c", "")
        tmux("send-keys", "-t", session, "sleep 30 &", "Enter")
        seen = until(lambda s: s["idle"] and s["background"])
        assert len(seen["background"]) == 1
    finally:
        subprocess.run(["tmux", "kill-server"], env=env, capture_output=True)
        shutil.rmtree(directory, ignore_errors=True)


def _verdict(ok, fast_used):
    return {
        "ok": ok,
        "stage": "daytona_oracle",
        "reward": float(ok),
        "solve_exit": 0,
        "replay": {"fast_used": fast_used},
    }


@pytest.mark.parametrize(
    "first,shortcut,reruns",
    [
        (_verdict(False, True), None, True),
        (_verdict(True, True), None, False),
        (_verdict(False, False), None, False),
        (_verdict(False, True), "cat /answer", False),
    ],
)
def test_revalidate_reruns_a_failed_fast_replay_at_recorded_timing(
    monkeypatch, first, shortcut, reruns
):
    import daytona_revalidate as dr

    calls = []

    async def once(pkg, sc, *args, fast=None, **kwargs):
        calls.append(fast)
        return first if len(calls) == 1 else _verdict(True, False)

    monkeypatch.setattr(dr, "_probe_once", once)
    result = asyncio.run(dr.probe(Path("/pkg"), shortcut, 900))
    assert calls == ([None, False] if reruns else [None])
    assert result["fast_replay_fallback"] is reruns
    if reruns:
        assert result["ok"] and not result["fast_replay_used"]
        assert result["fast_replay_first"]["reward"] == 0.0
    else:
        assert result["fast_replay_used"] is first["replay"]["fast_used"]


def test_sandbox_check_reruns_a_failed_fast_replay_on_a_fresh_container(
    monkeypatch,
):
    import agent_sandbox as asb

    events = []
    monkeypatch.setattr(asb, "cmd_down", lambda pkg: events.append("down"))
    monkeypatch.setattr(
        asb, "cmd_up", lambda pkg, at_max=False: events.append(("up", at_max)) or 0
    )
    monkeypatch.setattr(asb, "_read_state", lambda pkg: {"sock": "/s"})

    def request(state, op, **fields):
        events.append((op, fields.get("fast", "default")))
        if op == "grade":
            return {"ok": True, "reward": 0.0}
        return {"ok": True, "reward": 1.0, "solve_exit": 0, "replay": {}}

    monkeypatch.setattr(asb, "_request", request)
    failed = {"ok": True, "reward": 0.0, "solve_exit": 0, "replay": {"fast_used": True}}
    r = asb._with_recorded_rerun(Path("/pkg"), failed, 900, True, null_probe=True)
    assert events == ["down", ("up", True), ("grade", "default"), ("oracle", False)]
    assert r["reward"] == 1.0 and r["fast_replay_fallback"]
    assert r["fast_replay_first"] == {"reward": 0.0, "solve_exit": 0}

    events.clear()
    passed = {"ok": True, "reward": 1.0, "solve_exit": 0, "replay": {"fast_used": True}}
    r = asb._with_recorded_rerun(Path("/pkg"), passed, 900, False, null_probe=True)
    assert events == [] and r["fast_replay_used"] and not r["fast_replay_fallback"]
