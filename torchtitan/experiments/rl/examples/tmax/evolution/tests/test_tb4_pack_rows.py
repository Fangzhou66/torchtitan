# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A packaged (rewritten) TB4 task gets both halves of what its Harbor task
declares: the verifier block, built from tests/, and the agent's compose,
healthcheck, user and GPU -- as a benchmark row does. A package without a
separate verifier gets neither."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pack_to_dataset as pack  # noqa: E402

_PIN = "@sha256:" + "a" * 64
_TOML = """schema_version = "1.2"
artifacts = [{ source = "/audit/a.json", service = "api" }]

[task]
name = "terminal-bench/demo-task"

[verifier]
timeout_sec = 600.0
environment_mode = "separate"

[agent]
timeout_sec = 3600.0

[environment]
cpus = 2
memory_mb = 4096
storage_mb = 10240
gpus = 1
gpu_types = ["H100"]

[environment.healthcheck]
command = "true"
"""


@pytest.fixture(autouse=True)
def _checkout(monkeypatch):
    monkeypatch.setenv("TRL_TT", str(Path(__file__).resolve().parents[7]))


def _package(tmp_path, *, toml=_TOML, user="agent", sidecar=f"image: x{_PIN}"):
    pkg = tmp_path / "pkg"
    for rel, text in {
        "environment/Dockerfile": f"FROM ubuntu:24.04\nWORKDIR /app\nUSER {user}\n",
        "environment/docker-compose.yaml": (
            f"services:\n  main:\n    image: m{_PIN}\n  api:\n    {sidecar}\n"
        ),
        "tests/test.sh": "echo 1 > /logs/verifier/reward.txt\n",
        "tests/Dockerfile": "FROM python:3.12-slim\nCOPY . /tests/\n",
        "solution/solve.sh": "true\n",
        "instruction.md": "do it\n",
    }.items():
        (pkg / rel).parent.mkdir(parents=True, exist_ok=True)
        (pkg / rel).write_text(text)
    if toml is not None:
        (pkg / "task.toml").write_text(toml)
    return pkg


def test_a_packaged_tb4_task_gets_the_agent_half_too(tmp_path):
    md = pack.to_row(str(_package(tmp_path)), task_id="t")["metadata"]
    sv = md["tmax"]["separate_verifier"]
    assert sv["dockerfile"].startswith("FROM python:3.12-slim")
    assert (sv["daytona_gpu"], sv["daytona_target"]) == (1, "earth")  # inherited
    assert (md["daytona_gpu"], md["daytona_gpu_type"], md["daytona_target"]) == (
        1,
        ["H100"],
        "earth",
    )
    assert md["compose"]["docker_compose"].startswith("services:")
    assert md["healthcheck"]["command"] == "true"
    assert md["agent_user"] == "agent"
    # The image is left root for the harness: USER root before the tmux step.
    block = pack._rts_module()._AGENT_RUNTIME_BLOCK
    assert md["dockerfile"].endswith("\nUSER root" + block)
    assert md["dockerfile"].index("USER agent") < md["dockerfile"].index("USER root")


def test_a_root_image_is_not_touched_and_a_built_sidecar_is_refused(tmp_path):
    md = pack.to_row(str(_package(tmp_path / "a", user="root")), task_id="t")["metadata"]
    assert "agent_user" not in md and "USER root\n# harbor-agent-runtime" not in md["dockerfile"]
    with pytest.raises(ValueError, match="t: compose services \\['api'\\] build"):
        pack.to_row(str(_package(tmp_path / "b", sidecar="build: ./api")), task_id="t")


def test_a_package_without_a_separate_verifier_gains_nothing(tmp_path):
    md = pack.to_row(str(_package(tmp_path, toml=None)), task_id="t")["metadata"]
    assert "separate_verifier" not in md["tmax"]
    for key in ("compose", "healthcheck", "agent_user", "daytona_gpu"):
        assert key not in md
    assert "USER root" not in md["dockerfile"]
