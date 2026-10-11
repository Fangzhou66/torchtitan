# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
from types import SimpleNamespace

import pytest
import torch

from torchtitan.experiments.rl.actors import trainer as trainer_module
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def step(self) -> None:
        self.calls.append("step")

    def zero_grad(self) -> None:
        self.calls.append("zero_grad")


def _fake_trainer():
    optimizers = _Recorder()
    scheduler = SimpleNamespace(get_last_lr=lambda: [3e-6])
    schedulers = SimpleNamespace(schedulers=[scheduler], step=lambda: None)
    return SimpleNamespace(
        lr_schedulers=schedulers,
        optimizers=optimizers,
        model_parts=[torch.nn.Linear(2, 2)],
        config=SimpleNamespace(training=SimpleNamespace(max_norm=1.0)),
        parallel_dims=SimpleNamespace(
            get_optional_mesh=lambda name: None, ep_enabled=False
        ),
        policy_version=7,
    )


@pytest.mark.parametrize(
    "norm,stepped", [(0.08, True), (float("nan"), False), (float("inf"), False)]
)
def test_optim_step_skips_the_update_on_a_nonfinite_grad_norm(
    monkeypatch, norm, stepped
):
    monkeypatch.setattr(
        trainer_module.dist_utils,
        "clip_grad_norm_",
        lambda *args, **kwargs: torch.tensor(norm),
    )
    fake = _fake_trainer()
    out = asyncio.run(PolicyTrainer.optim_step._method(fake))
    assert fake.optimizers.calls == (["step", "zero_grad"] if stepped else ["zero_grad"])
    assert out.policy_version == 8
    assert out.metrics["train/skipped_nonfinite_grad"] == float(not stepped)
