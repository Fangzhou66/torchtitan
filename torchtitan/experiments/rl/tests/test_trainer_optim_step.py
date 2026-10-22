# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
from types import SimpleNamespace

import pytest
import torch

from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.experiments.rl.actors import trainer as trainer_module
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.examples.tmax.config_registry import _tmax_9b_adamw


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def step(self) -> None:
        self.calls.append("step")

    def zero_grad(self) -> None:
        self.calls.append("zero_grad")


def _fake_trainer(skip_zero_grad_step: bool = False):
    optimizers = _Recorder()
    scheduler = SimpleNamespace(get_last_lr=lambda: [3e-6])
    schedulers = SimpleNamespace(schedulers=[scheduler], step=lambda: None)
    return SimpleNamespace(
        lr_schedulers=schedulers,
        optimizers=optimizers,
        model_parts=[torch.nn.Linear(2, 2)],
        config=SimpleNamespace(
            training=SimpleNamespace(max_norm=1.0), skip_zero_grad_step=skip_zero_grad_step,
        ),
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


@pytest.mark.parametrize(
    "norm,skip_knob,stepped",
    [
        (0.0, False, True),
        (0.0, True, False),
        (1e-30, True, True),
        (float("nan"), True, False),
    ],
)
def test_optim_step_skips_the_update_on_a_zero_grad_norm_only_when_enabled(
    monkeypatch, norm, skip_knob, stepped
):
    monkeypatch.setattr(
        trainer_module.dist_utils,
        "clip_grad_norm_",
        lambda *args, **kwargs: torch.tensor(norm),
    )
    fake = _fake_trainer(skip_zero_grad_step=skip_knob)
    out = asyncio.run(PolicyTrainer.optim_step._method(fake))
    assert fake.optimizers.calls == (["step", "zero_grad"] if stepped else ["zero_grad"])
    assert out.policy_version == 8
    assert out.metrics["train/skipped_zero_grad"] == float(skip_knob and norm == 0.0)
    assert out.metrics["train/skipped_nonfinite_grad"] == float(norm != norm)


@pytest.mark.parametrize("skip_knob", [True, False])
def test_zero_grad_steps_leave_weights_alone_only_with_the_knob(skip_knob):
    """One real step, then zero-advantage steps through the tmax AdamW recipe.

    With the knob off this documents the old behaviour: AdamW's moment
    estimates keep moving the weights although every gradient is exactly 0.
    """
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 3)
    optimizers = _tmax_9b_adamw(lr=1e-3).build(model_parts=[model])
    lr_schedulers = LRSchedulersContainer.Config(warmup_steps=0).build(
        optimizers=optimizers, training_steps=10
    )
    fake = SimpleNamespace(
        lr_schedulers=lr_schedulers,
        optimizers=optimizers,
        model_parts=[model],
        config=SimpleNamespace(
            training=SimpleNamespace(max_norm=1.0),
            skip_zero_grad_step=skip_knob,
        ),
        parallel_dims=SimpleNamespace(
            get_optional_mesh=lambda name: None, ep_enabled=False
        ),
        policy_version=0,
    )
    x = torch.randn(5, 4)

    model(x).pow(2).sum().backward()
    out = asyncio.run(PolicyTrainer.optim_step._method(fake))
    assert out.metrics["train/grad_norm/mean"] > 0.0
    assert out.metrics["train/skipped_zero_grad"] == 0.0
    after_real_step = [p.detach().clone() for p in model.parameters()]

    num_zero_steps = 3
    for _ in range(num_zero_steps):
        # A zero-advantage batch: the loss is scaled by 0, so the backward pass
        # yields exact zeros rather than no gradient at all.
        (model(x).pow(2).sum() * 0.0).backward()
        out = asyncio.run(PolicyTrainer.optim_step._method(fake))
        assert out.metrics["train/grad_norm/mean"] == 0.0
        assert out.metrics["train/skipped_zero_grad"] == float(skip_knob)

    unchanged = all(
        torch.equal(before, p) for before, p in zip(after_real_step, model.parameters())
    )
    assert unchanged == skip_knob
    assert out.policy_version == 1 + num_zero_steps
    assert lr_schedulers.schedulers[0].last_epoch == 1 + num_zero_steps
