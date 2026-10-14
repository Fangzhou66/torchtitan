# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests of policy-probe routing and the optimization-complete boundary."""

import asyncio
from types import SimpleNamespace
from typing import cast

import pytest

from torchtitan.experiments.rl.controller import Controller
from torchtitan.experiments.rl.examples.tmax import layout
from torchtitan.experiments.rl.examples.tmax import policy_canary_controller as pc
from torchtitan.experiments.rl.examples.tmax.policy_canary import (
    PolicyCanaryConfig,
    PolicyCanaryQueue,
)


def _row(rev):
    return {
        "prompt": "Fix the tool",
        "metadata": {
            "instance_id": "t",
            "rev": rev,
            "image": "fixture-image",
            "tmax": {"test_sh": "grade"},
        },
    }


def _controller(tmp_path, monkeypatch):
    owner = pc.PolicyCanaryController.__new__(pc.PolicyCanaryController)
    monkeypatch.setattr(
        owner, "config", SimpleNamespace(dump_folder=str(tmp_path)), raising=False
    )
    owner._next_validation_group_id = -1
    owner._generator_policy_version = 3
    monkeypatch.setattr(
        owner,
        "_sampling",
        SimpleNamespace(temperature=1.0, top_p=1.0, max_tokens=32),
        raising=False,
    )
    monkeypatch.setattr(
        owner,
        "renderer",
        SimpleNamespace(_tokenizer=SimpleNamespace(decode=lambda ids: "")),
        raising=False,
    )
    return owner


def test_probes_use_training_generators_with_disjoint_negative_ids(
    tmp_path, monkeypatch
):
    owner = _controller(tmp_path, monkeypatch)
    calls = []

    generate = object()
    monkeypatch.setattr(owner, "_make_generate_fn", lambda **kwargs: generate)

    async def run_group(**kwargs):
        calls.append(kwargs)
        rollouts = [
            SimpleNamespace(
                rollout_id=i,
                reward=float(i % 2),
                diagnostics={},
                turns=[SimpleNamespace(min_policy_version=3, max_policy_version=3)],
            )
            for i in range(kwargs["group_size"])
        ]
        group = SimpleNamespace(group_id=kwargs["group_id"], rollouts=rollouts)
        return group

    monkeypatch.setattr(
        owner,
        "_rollouter",
        SimpleNamespace(run_group_rollouts=run_group),
        raising=False,
    )
    owner._eval_rollout_workers = [SimpleNamespace()]
    records = []
    recorder = SimpleNamespace(
        record=lambda **kw: (records.append(kw), SimpleNamespace(report_dir="traces"))[
            1
        ]
    )
    monkeypatch.setattr(
        pc,
        "ValidationTraceRecorder",
        SimpleNamespace(
            Config=lambda **kw: SimpleNamespace(build=lambda **kwargs: recorder)
        ),
    )
    request = {
        "task": "t",
        "input_rev": 0,
        "group_size": 4,
        "original": _row(0),
        "candidate": _row(1),
    }
    result = asyncio.run(owner._evaluate_policy_canary(request, "identity"))
    assert [call["group_id"] for call in calls] == [-1, -2]
    assert all(call["generate_fn"] is generate for call in calls)
    assert all(call["sampling"] is owner._sampling for call in calls)
    assert [call["sample"].rev for call in calls] == [0, 1]
    assert result["before"]["num_scored"] == result["after"]["num_scored"] == 4
    assert result["sampling"]["max_tokens"] == 32
    assert records[0]["expected_num_trials"] == 8
    assert owner._next_validation_group_id == -3


def test_failed_paired_group_cancels_its_sibling(tmp_path, monkeypatch):
    owner = _controller(tmp_path, monkeypatch)
    stopped = []
    monkeypatch.setattr(owner, "_make_generate_fn", lambda **kwargs: object())

    async def run_group(**kwargs):
        if kwargs["sample"].rev == 0:
            await asyncio.sleep(0)
            raise RuntimeError("sandbox unavailable")
        try:
            await asyncio.Event().wait()
        finally:
            stopped.append(True)

    monkeypatch.setattr(
        owner,
        "_rollouter",
        SimpleNamespace(run_group_rollouts=run_group),
        raising=False,
    )
    request = {
        "task": "t",
        "input_rev": 0,
        "group_size": 4,
        "original": _row(0),
        "candidate": _row(1),
    }
    with pytest.raises(RuntimeError, match="sandbox unavailable"):
        asyncio.run(owner._evaluate_policy_canary(request, "identity"))
    assert stopped == [True]


def test_optimization_completion_stops_probes_before_final_eval(tmp_path, monkeypatch):
    async def exercise():
        owner = _controller(tmp_path, monkeypatch)
        config = cast(
            pc.PolicyCanaryController.Config,
            object.__new__(pc.PolicyCanaryController.Config),
        )
        config.policy_canary = PolicyCanaryConfig(group_size=4, poll_sec=1)
        config.policy_canary_root = str(tmp_path)
        owner.config = config
        queue = PolicyCanaryQueue(layout.Root(tmp_path))
        queue.submit(
            rewrite="rw",
            task="t",
            input_rev=0,
            route="simplify",
            original=_row(0),
            candidate=_row(1),
            config=config.policy_canary,
            expires_at=10**12,
        )
        started = asyncio.Event()
        stopped = []

        async def evaluate(request, identity):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.append(True)

        async def trainer(self, *args, **kwargs):
            await asyncio.wait_for(started.wait(), 3)

        async def parent_run(self):
            await self._trainer_loop(None, num_training_steps=1)
            assert stopped == [True]
            assert self._policy_canary_task.done()
            assert not list(queue.responses.glob("*.json"))

        monkeypatch.setattr(owner, "_evaluate_policy_canary", evaluate)
        monkeypatch.setattr(Controller, "_trainer_loop", trainer)
        monkeypatch.setattr(Controller, "run", parent_run)
        await owner.run()

    asyncio.run(exercise())
