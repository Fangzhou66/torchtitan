# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""boot_agent_sandbox: TB4 options are opt-in; existing callers boot as before."""

from __future__ import annotations

import asyncio

import pytest

from torchtitan.experiments.rl.harness.agents import claude_code
from torchtitan.experiments.rl.harness.sandbox import compose as compose_mod


class _Sb:
    sandbox_id = "sb"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return None


def _run(**kwargs):
    async def go():
        async with claude_code.boot_agent_sandbox(
            "img", install_claude=False, **kwargs
        ) as sb:
            return sb

    return asyncio.run(go())


def test_existing_callers_pass_exactly_the_old_arguments(monkeypatch):
    calls = []
    monkeypatch.setattr(
        claude_code,
        "make_sandbox",
        lambda image, **kw: calls.append((image, kw)) or _Sb(),
    )
    _run(cpu=2, memory=4, disk_gb=10, labels={"task": "t"})
    image, kw = calls[0]
    assert image == "img"
    assert set(kw) == {
        "dockerfile",
        "build_context",
        "cpu",
        "memory",
        "disk_gb",
        "issue_tracker",
        "failure_diagnostics_dir",
        "labels",
    }


def test_sandbox_options_pass_through(monkeypatch):
    calls = []
    monkeypatch.setattr(
        claude_code, "make_sandbox", lambda image, **kw: calls.append(kw) or _Sb()
    )
    _run(sandbox_options={"gpu": 1, "gpu_type": ["H100"], "target": "earth"})
    assert calls[0]["gpu"] == 1 and calls[0]["target"] == "earth"


def test_compose_boots_a_compose_sandbox_with_the_row_image_as_main(monkeypatch):
    made = []

    class _FakeCompose(_Sb):
        def __init__(self, **kw):
            made.append(kw)

    monkeypatch.setattr(compose_mod, "DaytonaComposeSandbox", _FakeCompose)
    monkeypatch.setattr(
        claude_code, "make_sandbox", lambda *a, **k: pytest.fail("no plain sandbox")
    )
    sb = _run(
        cpu=2,
        dockerfile="FROM img\nRUN tmux -V\n",
        compose={"docker_compose": "services: {}\n", "project": "p"},
        sandbox_options={"target": "earth"},
    )
    assert isinstance(sb, _FakeCompose)
    kw = made[0]
    assert kw["project"] == "p" and kw["cpu"] == 2 and kw["target"] == "earth"
    assert kw["main_image"] == "img"
    assert kw["main_dockerfile"] == "FROM img\nRUN tmux -V\n"
