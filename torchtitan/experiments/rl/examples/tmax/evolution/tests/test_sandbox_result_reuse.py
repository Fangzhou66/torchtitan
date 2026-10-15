# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""./sandbox oracle/check reuse a passing result only for byte-identical inputs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent_sandbox as asb
import evolve_codex as ec
from test_agent_sandbox import _FakeServer

HERE = Path(__file__).resolve().parents[1]


def _package(tmp_path: Path) -> Path:
    pkg = tmp_path / "pkg"
    for d in ("run", "tests", "solution", "environment"):
        (pkg / d).mkdir(parents=True, exist_ok=True)
    (pkg / "instruction.md").write_text("Write /app/out.txt.\n")
    (pkg / "task.toml").write_text("[agent]\ntimeout_sec = 600\n")
    (pkg / "environment" / "Dockerfile").write_text("FROM scratch\nCOPY data.txt /\n")
    (pkg / "environment" / "data.txt").write_text("input\n")
    (pkg / "tests" / "test.sh").write_text("test -s /app/out.txt\n")
    (pkg / "solution" / "solve.sh").write_text("echo ok > /app/out.txt\n")
    (pkg / "run" / "resources.json").write_text(
        '{"cpu": 1, "mem_gb": 2, "disk_gb": 2, "source": "row"}\n'
    )
    return pkg


class _Box:
    """A fake sandbox: counts requests, answers like a passing task."""

    def __init__(self, pkg: Path, monkeypatch, *, oracle_reward=1.0):
        self.pkg = pkg
        self.ops: list[str] = []
        self.oracle_reward = oracle_reward
        self.boots = 0
        self.server = _FakeServer(self.handle)
        monkeypatch.setattr(asb, "cmd_up", self.up)
        monkeypatch.setattr(asb, "cmd_down", lambda _pkg: 0)

    def handle(self, req):
        self.ops.append(req["op"])
        if req["op"] == "grade":
            return {"ok": True, "reward": 0.0}
        if req["op"] == "oracle":
            return {
                "ok": True,
                "reward": self.oracle_reward,
                "solve_exit": 0,
                "tail": "solved",
                "measured": {"mem_peak_mb": 100.0, "solve_secs": 3.0},
            }
        return {"ok": True}

    def up(self, _pkg, at_max=False):
        self.boots += 1
        asb._write_state(
            self.pkg,
            {
                "status": "ready",
                "sock": self.server.path,
                "sandbox_id": f"sb-{self.boots}",
                "resources": asb._box(self.pkg, at_max),
            },
        )
        return 0

    def close(self):
        self.server.close()


def _records(pkg: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (pkg / "run" / "checks.jsonl").read_text().splitlines()
    ]


@pytest.fixture
def box(tmp_path, monkeypatch):
    monkeypatch.delenv("EVOLVE_SANDBOX_REUSE", raising=False)
    b = _Box(_package(tmp_path), monkeypatch)
    yield b
    b.close()


def test_check_reuses_a_pass_on_identical_inputs(box, capsys):
    assert asb.cmd_check(box.pkg, 30) == 0
    assert box.ops == ["grade", "oracle"]
    first = _records(box.pkg)[-1]
    capsys.readouterr()

    assert asb.cmd_check(box.pkg, 30) == 0
    assert box.ops == ["grade", "oracle"]  # nothing rebuilt or replayed
    out = capsys.readouterr().out
    assert "REUSED" in out and "--fresh" in out and "VERDICT: pass" in out
    again = _records(box.pkg)[-1]
    assert again["verdict"] == "pass" and again["reused_from"]
    for key in ("measured", "resources", "null_reward", "reward", "solve_exit"):
        assert again[key] == first[key]


def test_instruction_edit_is_not_an_oracle_input(box):
    assert asb.cmd_check(box.pkg, 30) == 0
    (box.pkg / "instruction.md").write_text("Write the report to /app/out.txt.\n")
    assert asb.cmd_check(box.pkg, 30) == 0
    assert box.ops == ["grade", "oracle"]


def _edit_solution(pkg):
    (pkg / "solution" / "solve.sh").write_text("echo ok2 > /app/out.txt\n")


def _edit_tests(pkg):
    (pkg / "tests" / "test.sh").write_text("grep ok /app/out.txt\n")


def _add_test_fixture(pkg):
    (pkg / "tests" / "expected.txt").write_text("ok\n")


def _edit_dockerfile(pkg):
    (pkg / "environment" / "Dockerfile").write_text("FROM scratch\n")


def _edit_build_context(pkg):
    (pkg / "environment" / "data.txt").write_text("other\n")


def _edit_task_toml(pkg):
    (pkg / "task.toml").write_text("[agent]\ntimeout_sec = 900\n")


def _edit_pretest(pkg):
    (pkg / "run" / "pretest.json").write_text('{"pre_test_sh": "true"}\n')


def _chmod_solution(pkg):
    os.chmod(pkg / "solution" / "solve.sh", 0o755)


def _symlink_in_context(pkg):
    os.symlink("data.txt", pkg / "environment" / "alias.txt")


def _resize_box(pkg):
    (pkg / "run" / "resources.json").write_text(
        '{"cpu": 2, "mem_gb": 4, "disk_gb": 2, "source": "row"}\n'
    )


@pytest.mark.parametrize(
    "change",
    [
        _edit_solution,
        _edit_tests,
        _add_test_fixture,
        _edit_dockerfile,
        _edit_build_context,
        _edit_task_toml,
        _edit_pretest,
        _chmod_solution,
        _symlink_in_context,
        _resize_box,
    ],
)
def test_check_reruns_after_any_input_change(box, change):
    assert asb.cmd_check(box.pkg, 30) == 0
    change(box.pkg)
    assert asb.cmd_check(box.pkg, 30) == 0
    assert box.ops == ["grade", "oracle", "grade", "oracle"]
    assert "reused_from" not in _records(box.pkg)[-1]


def test_check_reruns_at_another_timeout_or_the_ceiling(box):
    assert asb.cmd_check(box.pkg, 30) == 0
    assert asb.cmd_check(box.pkg, 60) == 0
    assert asb.cmd_check(box.pkg, 30, at_max=True) == 0
    assert box.ops.count("oracle") == 3


def test_fresh_forces_a_rerun(box):
    assert asb.cmd_check(box.pkg, 30) == 0
    assert asb.cmd_check(box.pkg, 30, fresh=True) == 0
    assert box.ops.count("oracle") == 2
    assert asb.cmd_oracle(box.pkg, 30, fresh=True) == 0
    assert box.ops.count("oracle") == 3


def test_reuse_can_be_switched_off(box, monkeypatch):
    monkeypatch.setenv("EVOLVE_SANDBOX_REUSE", "0")
    assert asb.cmd_check(box.pkg, 30) == 0
    assert asb.cmd_check(box.pkg, 30) == 0
    assert box.ops.count("oracle") == 2


def test_a_failed_rerun_drops_the_earlier_pass(box):
    assert asb.cmd_check(box.pkg, 30) == 0
    box.oracle_reward = 0.0
    assert asb.cmd_check(box.pkg, 30, fresh=True) == 1
    box.oracle_reward = 1.0
    assert asb.cmd_check(box.pkg, 30) == 0
    assert box.ops.count("oracle") == 3  # the third one ran again
    assert "reused_from" not in _records(box.pkg)[-1]


def test_a_failed_check_is_never_reused(box):
    box.oracle_reward = 0.0
    assert asb.cmd_check(box.pkg, 30) == 1
    assert asb.cmd_check(box.pkg, 30) == 1
    assert box.ops.count("oracle") == 2


def test_oracle_reuses_a_passing_check_and_says_nothing_ran(box, capsys):
    assert asb.cmd_check(box.pkg, 30) == 0
    capsys.readouterr()
    assert asb.cmd_oracle(box.pkg, 30) == 0
    assert box.ops.count("oracle") == 1
    out = capsys.readouterr().out
    assert "REUSED" in out and "Nothing ran" in out and "reward=1.0" in out
    _edit_solution(box.pkg)
    assert asb.cmd_oracle(box.pkg, 30) == 0
    assert box.ops.count("oracle") == 2


def test_check_never_reuses_a_standalone_oracle(box):
    box.up(box.pkg)
    assert asb.cmd_oracle(box.pkg, 30) == 0
    assert asb.cmd_check(box.pkg, 30) == 0
    assert box.ops == ["oracle", "grade", "oracle"]


def test_standalone_oracle_reuse_is_bound_to_its_container(box):
    box.up(box.pkg)
    assert asb.cmd_oracle(box.pkg, 30) == 0
    assert asb.cmd_oracle(box.pkg, 30) == 0
    assert box.ops.count("oracle") == 1
    box.up(box.pkg)  # a reset: another container
    assert asb.cmd_oracle(box.pkg, 30) == 0
    assert box.ops.count("oracle") == 2


def test_cli_offers_fresh():
    for cmd in ("check", "oracle"):
        out = subprocess.run(
            [sys.executable, str(HERE / "agent_sandbox.py"), cmd, "--help"],
            capture_output=True,
            text=True,
            cwd=HERE,
        ).stdout
        assert "--fresh" in out


def test_harness_clears_reuse_where_revalidation_failed():
    source = (HERE / "evolve_codex.py").read_text()
    assert source.count('"checks.jsonl", "result-cache.json"') == 2
    assert asb.REUSE_FILE == "result-cache.json"


def test_prompts_carry_the_wait_and_grade_guidance():
    runtime = ec._RUNTIME_INTERFACE
    assert "yield_time_ms 600000" in runtime
    assert "re-check verifier edits with grade rather than check" in runtime
    assert "--fresh" in runtime
    role = ec.SPEC.read_text()
    assert "yield_time_ms 600000" in role
    assert "re-check verifier\nedits with `grade` rather than `check`" in role


class _ReplayBox(_Box):
    """A recorded solution: the fast replay fails, the recorded-timing one
    answers ``recorded_reward``; the null probe answers ``null_rewards`` in turn."""

    def __init__(self, pkg, monkeypatch, *, recorded_reward, null_rewards=(0.0,)):
        super().__init__(pkg, monkeypatch)
        self.recorded_reward = recorded_reward
        self.null_rewards = list(null_rewards)

    def handle(self, req):
        self.ops.append(req["op"] + ("/recorded" if req.get("fast") is False else ""))
        if req["op"] == "grade":
            reward = (
                self.null_rewards.pop(0)
                if len(self.null_rewards) > 1
                else self.null_rewards[0]
            )
            return {"ok": True, "reward": reward}
        if req["op"] == "oracle":
            fast = req.get("fast") is not False
            return {
                "ok": True,
                "reward": 0.0 if fast else self.recorded_reward,
                "solve_exit": 0,
                "tail": "",
                "measured": {"mem_peak_mb": 100.0, "solve_secs": 3.0},
                "replay": {
                    "fast_used": fast,
                    "waited_sec": 1.0,
                    "recorded_wait_sec": 9.0,
                },
            }
        return {"ok": True}


def test_fallback_pass_is_remembered_with_its_replay_fields(tmp_path, monkeypatch):
    monkeypatch.delenv("EVOLVE_SANDBOX_REUSE", raising=False)
    b = _ReplayBox(_package(tmp_path), monkeypatch, recorded_reward=1.0)
    try:
        assert asb.cmd_check(b.pkg, 30) == 0
        assert b.ops == ["grade", "oracle", "grade", "oracle/recorded"]
        rec = _records(b.pkg)[-1]
        assert rec["verdict"] == "pass"
        assert rec["fast_replay_used"] is False and rec["fast_replay_fallback"] is True
        (entry,) = asb._read_reuse(b.pkg)
        assert entry["sandbox_id"] == "sb-2"  # the rerun's container
        assert entry["oracle"]["fast_replay_fallback"] is True
        # A hit runs nothing and carries the replay fields into the record.
        assert asb.cmd_check(b.pkg, 30) == 0
        assert len(b.ops) == 4
        again = _records(b.pkg)[-1]
        assert again["reused_from"] and again["fast_replay_fallback"] is True
    finally:
        b.close()


def test_fallback_fail_forgets_the_earlier_pass(tmp_path, monkeypatch):
    monkeypatch.delenv("EVOLVE_SANDBOX_REUSE", raising=False)
    b = _ReplayBox(_package(tmp_path), monkeypatch, recorded_reward=1.0)
    try:
        assert asb.cmd_check(b.pkg, 30) == 0
        b.recorded_reward = 0.0
        assert asb.cmd_check(b.pkg, 30, fresh=True) == 1
        assert asb._read_reuse(b.pkg) == []
        assert _records(b.pkg)[-1]["fast_replay_fallback"] is True
    finally:
        b.close()


def test_rerun_null_probe_pass_fails_the_check(tmp_path, monkeypatch):
    monkeypatch.delenv("EVOLVE_SANDBOX_REUSE", raising=False)
    b = _ReplayBox(
        _package(tmp_path), monkeypatch, recorded_reward=1.0, null_rewards=(0.0, 1.0)
    )
    try:
        assert asb.cmd_check(b.pkg, 30) == 1
        assert _records(b.pkg)[-1]["stage"] == "null_probe"
        assert asb._read_reuse(b.pkg) == []
    finally:
        b.close()

