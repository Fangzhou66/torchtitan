# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from torchtitan.experiments.rl.controller import Controller
from torchtitan.experiments.rl.training_lineage import TrainingLineageRecorder


@dataclass
class _Sample:
    instance_id: str
    prompt: str


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_recorder_appends_events_and_deduplicates_complete_inputs(tmp_path) -> None:
    recorder = TrainingLineageRecorder(dump_dir=str(tmp_path))
    sample = _Sample(instance_id="task-a", prompt="full prompt")
    lineage = recorder.describe_sample(sample=sample, group_id=4)

    recorder.record_sample(lineage=lineage, sample=sample)
    recorder.record_sample(lineage=lineage, sample=sample)
    recorder.record_event("admitted", lineage=lineage)
    recorder.record_event("trained", lineage=lineage, train_step=9)

    samples = _jsonl(recorder.samples_path)
    events = _jsonl(recorder.events_path)
    assert len(samples) == 1
    assert samples[0]["input"] == {
        "instance_id": "task-a",
        "prompt": "full prompt",
    }
    assert [event["event"] for event in events] == [
        "session_started",
        "admitted",
        "trained",
    ]
    assert events[-1]["train_step"] == 9
    assert events[-1]["occurrence_id"] == lineage["occurrence_id"]


def test_recorder_atomically_marks_optimization_complete(tmp_path) -> None:
    recorder = TrainingLineageRecorder(dump_dir=str(tmp_path))

    recorder.record_optimization_complete(
        policy_version=150,
        requested_training_steps=150,
    )

    marker = json.loads(recorder.optimization_complete_path.read_text())
    assert marker["event"] == "optimization_complete"
    assert marker["record_type"] == "optimization_complete"
    assert marker["policy_version"] == 150
    assert marker["requested_training_steps"] == 150
    assert not list(recorder.root.glob(".optimization_complete.json.*.tmp"))


@pytest.mark.parametrize("crash", [False, True])
def test_controller_heartbeat_closes_on_success_or_interruption(tmp_path, crash):
    async def run():
        controller = object.__new__(Controller)
        controller.config = SimpleNamespace(training_heartbeat_interval_sec=0.01)
        controller.training_lineage = TrainingLineageRecorder(dump_dir=str(tmp_path))
        controller._training_activity_phase = "starting"
        controller._trainer_policy_version = 3

        async def training():
            controller._training_activity_phase = "training"
            await asyncio.sleep(0.04)
            heartbeat = json.loads(
                controller.training_lineage.training_activity_path.read_text()
            )
            assert heartbeat["phase"] == "training"
            assert heartbeat["policy_version"] == 3
            if crash:
                raise RuntimeError("preempted")
            controller._training_activity_phase = "complete"

        controller._run = training
        if crash:
            with pytest.raises(RuntimeError, match="preempted"):
                await controller.run()
        else:
            await controller.run()
        heartbeat = json.loads(
            controller.training_lineage.training_activity_path.read_text()
        )
        assert heartbeat["phase"] == ("interrupted" if crash else "complete")
        assert not list(
            controller.training_lineage.root.glob(".training_activity.json.*.tmp")
        )

    asyncio.run(run())
