# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU restart proofs for prompt recovery and full training checkpoints."""

from __future__ import annotations

import asyncio
import json
import pickle
import threading
from collections import deque
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import default_adamw
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.controller import Controller
from torchtitan.experiments.rl.examples.tmax.data import TMaxDataset
from torchtitan.experiments.rl.training_lineage import TrainingLineageRecorder


def _dataset(path, task_ids, *, shuffle=False):
    path.write_text(
        "".join(
            json.dumps(
                {
                    "metadata": {
                        "instance_id": task,
                        "image": "example/image",
                        "tmax": {"test_sh": "true"},
                    }
                }
            )
            + "\n"
            for task in task_ids
        )
    )
    return TMaxDataset(
        TMaxDataset.Config(data_path=str(path), seed=42, shuffle=shuffle)
    )


def _controller(path, dataset, *, version=3):
    controller = object.__new__(Controller)
    controller.training_lineage = TrainingLineageRecorder(
        dump_dir=str(path), track_pending_inputs=True
    )
    controller._training_data_lock = asyncio.Lock()
    controller._next_training_group_id = (
        controller.training_lineage.next_unused_group_id
    )
    controller._replay_training_inputs = deque()
    controller._trainer_policy_version = version
    controller._generator_policy_version = version
    controller.start_step = version
    controller._rollouter = SimpleNamespace(
        get_training_sample=lambda: next(dataset),
        drain_training_data_events=dataset.drain_lineage_events,
        training_data_state_dict=dataset.state_dict,
        load_training_data_state_dict=dataset.load_state_dict,
    )
    return controller


class _Buffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.work = []

    async def wait_for_slot(self):
        return len(self.work) < self.capacity

    async def add_work(self, work):
        work.admitted_ts = 1.0
        self.work.append(work)


def test_pending_prompts_replay_before_the_restored_cursor(tmp_path):
    async def run():
        dataset = _dataset(tmp_path / "old.jsonl", list("abcd"))
        original = _controller(tmp_path / "original", dataset)
        buffer = _Buffer(3)
        await original._data_input_loop(buffer)
        original._record_training_event("trained", lineage=buffer.work[0].lineage)
        original._record_training_event("dropped", lineage=buffer.work[1].lineage)
        payload = await original._snapshot_training_data_state()
        assert list(original.training_lineage.pending_inputs) == [2]

        restarted_dataset = _dataset(tmp_path / "new.jsonl", list("abcd"))
        restarted = _controller(tmp_path / "restarted", restarted_dataset)
        restarted._restore_training_data_state(payload)
        new_buffer = _Buffer(2)
        await restarted._data_input_loop(new_buffer)
        assert [work.group_id for work in new_buffer.work] == [2, 3]
        assert [work.sample.instance_id for work in new_buffer.work] == ["c", "d"]
        assert new_buffer.work[0].sample.lineage == buffer.work[2].sample.lineage
        assert restarted_dataset.state_dict()["stream_position"] == 4

    asyncio.run(run())


def test_fresh_run_issues_group_ids_from_zero(tmp_path):
    async def run():
        dataset = _dataset(tmp_path / "data.jsonl", list("abc"))
        controller = _controller(tmp_path / "run", dataset)
        assert controller.training_lineage.next_unused_group_id == 0
        buffer = _Buffer(3)
        await controller._data_input_loop(buffer)
        assert [work.group_id for work in buffer.work] == [0, 1, 2]

    asyncio.run(run())


def test_resume_issues_new_group_ids_above_every_earlier_session(tmp_path):
    async def run():
        dataset = _dataset(tmp_path / "old.jsonl", list("abcdef"))
        original = _controller(tmp_path / "run", dataset)
        buffer = _Buffer(2)
        await original._data_input_loop(buffer)
        original._record_training_event("trained", lineage=buffer.work[0].lineage)
        payload = await original._snapshot_training_data_state()
        # Groups 2 and 3 are issued after the checkpoint, then the run crashes.
        buffer.capacity = 4
        await original._data_input_loop(buffer)
        assert [work.group_id for work in buffer.work] == [0, 1, 2, 3]

        restarted_dataset = _dataset(tmp_path / "new.jsonl", list("abcdef"))
        restarted = _controller(tmp_path / "run", restarted_dataset)
        assert restarted.training_lineage.next_unused_group_id == 4
        restarted._restore_training_data_state(payload)
        new_buffer = _Buffer(3)
        await restarted._data_input_loop(new_buffer)
        # Pending group 1 replays under its own id; new groups skip 2 and 3.
        assert [work.group_id for work in new_buffer.work] == [1, 4, 5]
        assert [work.sample.instance_id for work in new_buffer.work] == list("bcd")

        # Without checkpointed controller data the counter still starts above.
        plain_dataset = _dataset(tmp_path / "plain.jsonl", list("ab"))
        plain = _controller(tmp_path / "run", plain_dataset)
        assert plain._next_training_group_id == 6

    asyncio.run(run())


def test_snapshot_waits_for_dataset_draw_and_prompt_registration(tmp_path):
    async def run():
        dataset = _dataset(tmp_path / "data.jsonl", list("abc"))
        controller = _controller(tmp_path / "controller", dataset)
        entered, release = threading.Event(), threading.Event()

        def draw():
            sample = next(dataset)
            entered.set()
            assert release.wait(5)
            return sample

        controller._rollouter.get_training_sample = draw
        admission = asyncio.create_task(controller._data_input_loop(_Buffer(1)))
        assert await asyncio.to_thread(entered.wait, 5)
        snapshot = asyncio.create_task(controller._snapshot_training_data_state())
        await asyncio.sleep(0.01)
        assert not snapshot.done()
        release.set()
        await admission
        saved = pickle.loads(await snapshot)
        assert saved["dataset"]["pos"] == 1
        assert len(saved["pending_inputs"]) == 1
        assert saved["pending_inputs"][0][1].instance_id == "a"

    asyncio.run(run())


def test_dataset_reconciles_changed_mix_by_task_id(tmp_path):
    old = _dataset(tmp_path / "old.jsonl", list("abcd"))
    next(old)
    next(old)
    state = old.state_dict()
    new = _dataset(tmp_path / "new.jsonl", list("dcbe"))
    new.load_state_dict(state)
    assert [next(new).instance_id for _ in range(3)] == list("cde")
    assert new.state_dict()["stream_id"] == state["stream_id"]


def test_dataset_shuffle_rng_and_legacy_state_roundtrip(tmp_path):
    old = _dataset(tmp_path / "old.jsonl", list("abcdef"), shuffle=True)
    for _ in range(4):
        next(old)
    state = old.state_dict()
    fresh = _dataset(tmp_path / "new.jsonl", list("abcdef"), shuffle=True)
    fresh.load_state_dict(state)
    assert [next(old).instance_id for _ in range(20)] == [
        next(fresh).instance_id for _ in range(20)
    ]
    legacy = dict(state)
    legacy.pop("ordered_task_ids")
    fresh.load_state_dict(legacy)
    assert fresh.state_dict()["pos"] == state["pos"]


class _TrainState:
    state_dict = PolicyTrainer.state_dict
    load_state_dict = PolicyTrainer.load_state_dict

    def __init__(self):
        self.config = SimpleNamespace(checkpoint_training_data=True)
        self.policy_version = 0
        self._controller_training_data = b""


def _training(path):
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 2).double()
    optimizers = replace(default_adamw(lr=1e-3), implementation="for-loop").build(
        model_parts=[model]
    )
    schedulers = LRSchedulersContainer.Config(warmup_steps=1, min_lr_factor=1).build(
        optimizers=optimizers,
        training_steps=8,
    )
    state = _TrainState()
    manager = CheckpointManager.Config(
        enable=True,
        folder=str(path),
        interval=1,
        async_mode="disabled",
        keep_latest_k=0,
        initial_load_path=None,
        initial_load_model_only=False,
        last_save_model_only=False,
    ).build(
        dataloader=None,
        model_parts=[model],
        optimizers=optimizers,
        lr_schedulers=schedulers,
        states={"train_state": state},
        sd_adapter=None,
    )
    return model, optimizers, schedulers, state, manager


def _steps(training, start, stop):
    model, optimizers, schedulers, state, _ = training
    measurements = []
    for step in range(start, stop):
        features = torch.tensor([[step / 10, 0.2, 0.3]], dtype=torch.float64)
        loss = model(features).square().mean()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        measurements.append((loss.item(), grad_norm.item()))
        optimizers.step()
        schedulers.step()
        optimizers.zero_grad()
        state.policy_version += 1
    return measurements


def test_full_dcp_restart_preserves_loss_grad_norm_optimizer_and_variable_queue(
    tmp_path,
):
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    managers = []
    try:
        reference = _training(tmp_path / "reference")
        interrupted = _training(tmp_path / "recovery")
        managers.extend([reference[-1], interrupted[-1]])
        expected = _steps(reference, 0, 8)
        actual = _steps(interrupted, 0, 3)
        sample = next(_dataset(tmp_path / "data.jsonl", list("ab")))
        snapshot = pickle.dumps(
            {
                "schema_version": 1,
                "policy_version": 3,
                "pending_inputs": [(7, sample)],
                "dataset": {"pos": 1},
                "next_group_id": 8,
            }
        )
        interrupted[3]._controller_training_data = snapshot
        assert interrupted[-1].save(3)
        # An unfinished, newer write must not hide the latest complete checkpoint.
        (tmp_path / "recovery" / "step-9").mkdir()
        resumed = _training(tmp_path / "recovery")
        managers.append(resumed[-1])
        assert resumed[-1]._find_load_step() == 3
        assert resumed[-1].load()
        assert resumed[3].policy_version == 3
        recovered = pickle.loads(resumed[3]._controller_training_data)
        assert recovered["pending_inputs"] == [(7, sample)]
        actual.extend(_steps(resumed, 3, 8))
        assert actual == expected
        torch.testing.assert_close(
            resumed[0].state_dict(), reference[0].state_dict(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            resumed[1].state_dict(), reference[1].state_dict(), rtol=0, atol=0
        )
        assert resumed[2].state_dict() == reference[2].state_dict()
    finally:
        for manager in managers:
            manager.close()
        torch.use_deterministic_algorithms(previous)


@pytest.mark.parametrize("version", [2, 4])
def test_snapshot_must_match_checkpoint_policy_version(tmp_path, version):
    fake = _TrainState()
    fake.policy_version = 3
    fake.checkpointer = SimpleNamespace(_should_save=lambda *_a, **_kw: True)
    snapshot = pickle.dumps({"policy_version": version})
    with pytest.raises(ValueError, match="disagrees"):
        asyncio.run(
            PolicyTrainer.save_checkpoint._method(fake, 3, training_data_state=snapshot)
        )


def test_first_step_checkpoint_uses_the_existing_save_policy(tmp_path):
    training = _training(tmp_path / "first-step")
    manager = training[-1]
    try:
        manager.interval = 2
        manager.enable_first_step_checkpoint = True
        assert [step for step in range(1, 7) if manager._should_save(step)] == [
            1,
            2,
            4,
            6,
        ]
        assert manager._should_save(7, last_step=True)
        _steps(training, 0, 1)
        training[3]._controller_training_data = pickle.dumps(
            {"policy_version": 1, "pending_inputs": []}
        )
        assert manager.save(1)
        assert (tmp_path / "first-step" / "step-1" / ".metadata").exists()
        assert manager._find_load_step() == 1
    finally:
        manager.close()
