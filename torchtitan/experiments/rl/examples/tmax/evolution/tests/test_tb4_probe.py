# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The probe runs a TB4 row the way the rollouter does.

It boots with the row's GPU/compose keywords, runs the healthcheck, uploads
the solution's bytes, runs the reference as the image's own user (harness as
root), and grades with the task's declared verifier budget; what collection
and the verifier left survives a failed grade. A non-TB4 row takes none of it.
"""

from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import daytona_revalidate as dr  # noqa: E402

_BINARY = b"\x89PNG\r\n\x1a\n\xff\x00"


class _Sandbox:
    def __init__(self, calls):
        self.calls = calls
        self.sandbox_id = "sb-1"

    async def exec(self, cmd, *, user="root", env=None, **kw):
        self.calls.append(("exec", user, cmd))
        if cmd.startswith("getent passwd"):
            return 0, "/home/agent\n", ""
        return 0, "", ""

    async def write_file(self, path, content, *, user="root"):
        self.calls.append(("write", user, path, content))

    async def read_file(self, path, *, user="root"):
        return ""


def _package(tmp_path: Path) -> Path:
    pkg = tmp_path / "pkg"
    (pkg / "solution").mkdir(parents=True)
    (pkg / "solution" / "solve.sh").write_text("echo solved\n")
    (pkg / "solution" / "plate.png").write_bytes(_BINARY)
    (pkg / "task.toml").write_text("")
    return pkg


def _tb4_row() -> dict:
    return {
        "prompt": "do it",
        "metadata": {
            "instance_id": "t",
            "image": "harborframework/terminal-bench:t@sha256:" + "a" * 64,
            "workdir": "/app",
            "verifier_timeout_sec": 7200.0,
            "daytona_gpu": 1,
            "daytona_target": "earth",
            "healthcheck": {"command": "probe-health", "retries": 1},
            "agent_user": "agent",
            "tmax": {
                "test_sh": "",
                "fixtures": {},
                "separate_verifier": {"image": "v", "artifacts": ["/app/out"]},
            },
        },
    }


def _install(monkeypatch, calls, graded, *, grade_error=None):
    booted: list = []

    @asynccontextmanager
    async def boot(_image, **kw):
        booted.append(kw)
        yield _Sandbox(calls)

    async def run_reference(sb, cmd, timeout, **_kw):
        await sb.exec(cmd, check=False, timeout=timeout)
        return {
            "solve_exit": 0,
            "stdout": "",
            "stderr": "",
            "submitted": True,
            "terminal": {},
            "transcript": [],
            "execution_harness": "terminus",
        }

    async def grade_separate(_sb, _tmax, **kw):
        graded.update(kw)
        kw["diagnostics"]["separate_verifier"] = {"verifier": "ran"}
        if grade_error is not None:
            raise grade_error
        return 1.0

    async def grade_tmax(_sb, _tmax, **kw):
        graded.update(kw, plain=True)
        return 1.0

    async def nothing(*_a, **_kw):
        return None

    async def measure(*_a, **_kw):
        return {}

    monkeypatch.setattr(dr, "boot_agent_sandbox", boot)
    monkeypatch.setattr(dr, "run_reference", run_reference)
    monkeypatch.setattr(dr, "grade_separate", grade_separate)
    monkeypatch.setattr(dr, "grade_tmax", grade_tmax)
    monkeypatch.setattr(dr, "capture_baseline", nothing)
    monkeypatch.setattr(dr, "seed_workspace", nothing)
    monkeypatch.setattr(dr, "measure", measure)
    monkeypatch.delenv("TMAX_EVAL_TIMEOUT_SEC", raising=False)
    return booted


def test_a_tb4_row_is_probed_as_the_rollouter_runs_it(monkeypatch, tmp_path):
    calls: list = []
    graded: dict = {}
    booted = _install(monkeypatch, calls, graded)
    result = asyncio.run(
        dr._probe_once(_package(tmp_path), None, 900, prepared_row=_tb4_row())
    )
    assert result["reward"] == 1.0 and result["ok"]
    assert booted[0]["sandbox_options"] == {"gpu": 1, "target": "earth"}
    execs = [(user, cmd) for kind, user, cmd, *_ in calls if kind == "exec"]
    assert ("root", "probe-health") in execs  # healthcheck, as root
    assert execs.index(("root", "probe-health")) < execs.index(
        ("agent", "bash /solution/solve.sh")
    )
    uploads = {path: (user, content) for kind, user, path, content in
               (c for c in calls if c[0] == "write")}
    assert uploads["/solution/plate.png"] == ("root", _BINARY)  # bytes, as root
    assert graded["timeout_sec"] == 7200  # the declared budget, not 900
    assert "plain" not in graded


def test_the_floor_still_applies_and_a_failed_grade_keeps_its_evidence(
    monkeypatch, tmp_path
):
    calls: list = []
    graded: dict = {}
    _install(monkeypatch, calls, graded, grade_error=RuntimeError("verifier boot"))
    row = _tb4_row()
    row["metadata"]["verifier_timeout_sec"] = 60.0
    with pytest.raises(RuntimeError, match="verifier boot") as caught:
        asyncio.run(dr._probe_once(_package(tmp_path), None, 900, prepared_row=row))
    assert graded["timeout_sec"] == 900
    assert caught.value.validation == {
        "grading": {"separate_verifier": {"verifier": "ran"}}
    }


def test_a_plain_row_is_probed_as_before(monkeypatch, tmp_path):
    calls: list = []
    graded: dict = {}
    booted = _install(monkeypatch, calls, graded)
    row = {
        "prompt": "do it",
        "metadata": {
            "instance_id": "t",
            "image": "img",
            "workdir": "/app",
            "tmax": {"test_sh": "echo 1 > /logs/verifier/reward.txt"},
        },
    }
    asyncio.run(dr._probe_once(_package(tmp_path), None, 900, prepared_row=row))
    assert "sandbox_options" not in booted[0] and "compose" not in booted[0]
    assert all(user == "root" for kind, user, *_ in calls if kind == "exec")
    uploads = {c[2]: c[3] for c in calls if c[0] == "write"}
    assert isinstance(uploads["/solution/plate.png"], str)  # text, as before
    assert graded["plain"] is True and "timeout_sec" not in graded
