# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks for paired student measurements and publication admission."""

import asyncio
import os
import time
from types import SimpleNamespace

import pytest

from torchtitan.experiments.rl.examples.tmax import layout
from torchtitan.experiments.rl.examples.tmax.evolution_metrics import (
    EvolutionMetrics,
    record_outcome,
)
from torchtitan.experiments.rl.examples.tmax.policy_canary import (
    PolicyCanaryConfig,
    PolicyCanaryQueue,
    PolicyCanaryService,
    admission,
    content_hash,
    summarize_group,
)


def _group(rewards, *, infra=False, version=3):
    rollouts = [
        SimpleNamespace(
            rollout_id=index,
            reward=reward,
            diagnostics={"infra_failed": infra},
            turns=[
                SimpleNamespace(min_policy_version=version, max_policy_version=version)
            ],
        )
        for index, reward in enumerate(rewards)
    ]
    return summarize_group(
        SimpleNamespace(rollouts=rollouts, group_id=-1), group_size=4
    )


def _measurement(route="simplify", before=(0, 0, 0, 0), after=(0, 1, 0, 1)):
    request = {"route": route, "group_size": 4, "harder_ratio": 0.9}
    response = {
        "request_sha256": content_hash(request),
        "status": "complete",
        "before": _group(before),
        "after": _group(after),
    }
    return request, response


@pytest.mark.parametrize(
    "route,before,after,ok,stage",
    [
        ("simplify", [0] * 4, [0, 1, 0, 1], True, "policy_canary_pass"),
        ("evolve", [1] * 4, [0, 1, 0, 1], True, "policy_canary_pass"),
        ("simplify", [0] * 4, [0] * 4, False, "policy_too_hard"),
        ("simplify", [0] * 4, [1] * 4, False, "policy_too_easy"),
        ("simplify", [0, 1, 0, 0], [0, 1, 0, 1], False, "invalid_signal"),
        ("evolve", [0, 1, 1, 1], [0, 1, 0, 1], False, "invalid_signal"),
        ("repair_spec", [0] * 4, [0] * 4, True, "policy_repair_measured"),
        ("repair_spec", [0] * 4, [1] * 4, True, "policy_repair_measured"),
    ],
)
def test_admission_routes(route, before, after, ok, stage):
    request, response = _measurement(route, before, after)
    verdict = admission(request, response, max_version_span=5)
    assert verdict["ok"] == ok and verdict["stage"] == stage


@pytest.mark.parametrize(
    "failure", ["infra", "nan", "missing", "version", "no_generation"]
)
def test_unreliable_measurement_cannot_admit(failure):
    request, response = _measurement()
    if failure == "infra":
        response["after"] = _group([0, 1, 0, 1], infra=True)
    elif failure == "nan":
        response["after"] = _group([0, float("nan"), 0, 1])
    elif failure == "missing":
        response["after"]["trials"].pop()
    elif failure == "version":
        response["after"] = _group([0, 1, 0, 1], version=9)
    else:
        response["after"]["trials"][0]["policy_version_min"] = None
    verdict = admission(request, response, max_version_span=5)
    assert not verdict["ok"] and verdict["stage"] == "invalid_signal"


def test_old_candidate_response_cannot_admit():
    request, response = _measurement()
    response["request_sha256"] = "different revision"
    with pytest.raises(ValueError, match="does not match"):
        admission(request, response, max_version_span=5)


def test_api_error_cannot_admit():
    request, response = _measurement()
    response.update(status="error", error="sandbox unavailable")
    assert (
        admission(request, response, max_version_span=5)["stage"]
        == "policy_canary_error"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"group_size": 1},
        {"max_concurrency": 0},
        {"timeout_sec": 0},
        {"max_version_span": -1},
    ],
)
def test_canary_config_contract(kwargs):
    with pytest.raises(ValueError):
        PolicyCanaryConfig(**kwargs)


def _submit(queue, config, index=0, *, expires_at=None):
    return queue.submit(
        rewrite=f"evolution/tasks/t/rewrites/{index}",
        task="t",
        input_rev=0,
        route="simplify",
        original={"version": 0},
        candidate={"version": 1},
        config=config,
        expires_at=expires_at or time.time() + 60,
    )


def test_queue_binds_candidate_and_preserves_production_mix(tmp_path):
    root = layout.Root(tmp_path)
    root.mix.publish(['{"metadata":{"instance_id":"t"}}'])
    original = root.mix.live.read_bytes()
    queue = PolicyCanaryQueue(root)
    config = PolicyCanaryConfig(group_size=4)
    identity, request = _submit(queue, config)
    response = {
        "request_sha256": identity,
        "status": "complete",
        "before": _group([0] * 4),
        "after": _group([0, 1, 0, 1]),
    }
    layout.write_json_atomic(queue.responses / f"{identity}.json", response)
    verdict = queue.wait(identity, request, config=config)
    assert verdict["ok"] and verdict["candidate_sha256"] == content_hash(
        request["candidate"]
    )
    assert verdict["expires_at"] == request["expires_at"]
    assert root.mix.live.read_bytes() == original


def test_expired_request_never_uses_a_late_result(tmp_path):
    queue = PolicyCanaryQueue(layout.Root(tmp_path))
    config = PolicyCanaryConfig(group_size=4)
    identity, request = _submit(queue, config, expires_at=time.time() - 1)
    verdict = queue.wait(identity, request, config=config)
    assert not verdict["ok"] and verdict["stage"] == "policy_canary_timeout"


def test_service_bounds_concurrency_and_cancels_children(tmp_path):
    async def exercise():
        queue = PolicyCanaryQueue(layout.Root(tmp_path))
        config = PolicyCanaryConfig(group_size=4, max_concurrency=2, poll_sec=1)
        for index in range(4):
            _submit(queue, config, index)
        active = 0
        maximum = 0
        stopped = 0
        started = asyncio.Event()

        async def evaluate(request, identity):
            nonlocal active, maximum, stopped
            active += 1
            maximum = max(maximum, active)
            if active == 2:
                started.set()
            try:
                await asyncio.Event().wait()
                return {}
            finally:
                active -= 1
                stopped += 1

        task = asyncio.create_task(
            PolicyCanaryService(queue, config=config, evaluate=evaluate).run()
        )
        await asyncio.wait_for(started.wait(), 3)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert maximum == 2 and active == 0 and stopped == 2
        assert not list(queue.responses.glob("*.json"))

    asyncio.run(exercise())


def test_service_response_roundtrip(tmp_path):
    async def exercise():
        queue = PolicyCanaryQueue(layout.Root(tmp_path))
        config = PolicyCanaryConfig(group_size=4, poll_sec=1)
        identity, request = _submit(queue, config)

        async def evaluate(request, identity):
            return {"before": _group([0] * 4), "after": _group([0, 1, 0, 1])}

        task = asyncio.create_task(
            PolicyCanaryService(queue, config=config, evaluate=evaluate).run()
        )
        try:
            verdict = await asyncio.wait_for(
                asyncio.to_thread(queue.wait, identity, request, config=config), 3
            )
            assert verdict["ok"]
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(exercise())


def test_service_admits_oldest_requests_before_hash_order(tmp_path):
    async def exercise():
        queue = PolicyCanaryQueue(layout.Root(tmp_path))
        config = PolicyCanaryConfig(group_size=4, max_concurrency=1, poll_sec=1)
        identities = [_submit(queue, config, index)[0] for index in range(3)]
        oldest_first = sorted(identities, reverse=True)
        for index, identity in enumerate(oldest_first):
            os.utime(queue.requests / f"{identity}.json", ns=(index + 1, index + 1))
        calls = []
        finished = asyncio.Event()

        async def evaluate(request, identity):
            calls.append(identity)
            if len(calls) == 3:
                finished.set()
            return {"before": _group([0] * 4), "after": _group([0, 1, 0, 1])}

        task = asyncio.create_task(
            PolicyCanaryService(queue, config=config, evaluate=evaluate).run()
        )
        try:
            await asyncio.wait_for(finished.wait(), 3)
            assert calls == oldest_first
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(exercise())


def test_spec_repairs_are_outside_simplification_denominator(tmp_path):
    root = layout.Root(tmp_path)
    run = root.run("training")
    for index, (route, status) in enumerate(
        [
            ("simplify", "accepted"),
            ("simplify", "kept"),
            ("repair_spec", "accepted"),
            ("repair_spec", "kept"),
            ("invalid_signal", "kept"),
        ]
    ):
        record_outcome(
            root,
            root.evolution.task("t").rewrite("easier", f"rw{index}"),
            {
                "signal": f"training/t--g{index}",
                "task": "t",
                "job": "easier",
                "route": route,
                "status": status,
            },
        )
    metrics = EvolutionMetrics(run, root).poll()
    assert metrics["evolution/simplify/attempted"] == 2
    assert metrics["evolution/simplify/acceptance_rate"] == 0.5
    assert metrics["evolution/repair_spec/attempted"] == 2
    assert metrics["evolution/rewrites/accepted_easier"] == 1
    assert metrics["evolution/rewrites/accepted_repair_spec"] == 1
    assert metrics["evolution/flow/rewrites_accepted"] == 2
    assert metrics["evolution/invalid_signal/kept"] == 1


def test_repairs_made_inside_rewrites_are_counted(tmp_path):
    root = layout.Root(tmp_path)
    run = root.run("training")
    fix = {"defect": "d", "evidence": "e", "change": "c", "kind": "hidden_requirement"}
    hack = {**fix, "kind": "reward_hack"}
    for index, (job, status, repair) in enumerate(
        [
            ("harder", "accepted", [hack, fix]),
            ("harder", "rejected", [hack]),
            ("easier", "accepted", [{**fix, "kind": "broken_setup"}]),
            ("easier", "accepted", None),
        ]
    ):
        meta = {
            "signal": f"training/t--g{index}",
            "task": "t",
            "job": job,
            "status": status,
        }
        if repair:
            meta["repair"] = repair
        record_outcome(root, root.evolution.task("t").rewrite(job, f"rw{index}"), meta)
    metrics = EvolutionMetrics(run, root).poll()
    assert metrics["evolution/repairs/rewrites"] == 3
    assert metrics["evolution/repairs/fixes"] == 4
    assert metrics["evolution/repairs/accepted_rewrites"] == 2
    assert metrics["evolution/repairs/harder"] == 2
    assert metrics["evolution/repairs/easier"] == 1
    assert metrics["evolution/repairs/kind_reward_hack"] == 2
    assert metrics["evolution/repairs/kind_hidden_requirement"] == 1
    assert metrics["evolution/repairs/kind_broken_setup"] == 1
    assert metrics["evolution/repairs/kind_other"] == 0
    # Each fix becomes one report row with the rewriter's own text.
    flow = EvolutionMetrics(run, root)
    flow.poll(step=7)
    assert [row[:5] for row in flow.repair_rows] == [
        [7, "t", "harder", "accepted", "reward_hack"],
        [7, "t", "harder", "accepted", "hidden_requirement"],
        [7, "t", "harder", "rejected", "reward_hack"],
        [7, "t", "easier", "accepted", "broken_setup"],
    ]
    assert flow.repair_rows[0][5:] == ["d", "e", "c"]
