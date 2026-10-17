# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-format-error reward penalty: shapes the advantage, never the reported reward."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from torchtitan.experiments.rl.components.training_sample_builder import (
    TrainingSampleBuilder,
)
from torchtitan.experiments.rl.controller_metrics import compute_rollout_metrics
from torchtitan.experiments.rl.examples.tmax import rollouter as tmax_rollouter
from torchtitan.experiments.rl.examples.tmax.rollouter import TMaxRollouter
from torchtitan.experiments.rl.rollout.advantage import AdvantageEstimator
from torchtitan.experiments.rl.rollout.types import (
    Rollout,
    RolloutGroup,
    RolloutStatus,
    RolloutTurn,
    task_reward,
)
from torchtitan.experiments.rl.types import RolloutTurnID


def _rollout(reward: float, format_errors: int, idx: int) -> Rollout:
    return Rollout(
        group_id=0,
        rollout_id=idx,
        turns=[
            RolloutTurn(
                rollout_id=RolloutTurnID(group_id=0, rollout_id=idx, turn_id=0),
                prompt_token_ids=[1],
                completion_token_ids=[2],
                completion_logprobs=[0.0],
                min_policy_version=0,
                max_policy_version=0,
            )
        ],
        status=RolloutStatus.COMPLETED,
        reward=reward,
        diagnostics={"format_errors": format_errors},
    )


def _apply(rollouts, penalty=0.05, cap=0.3):
    rollouter = SimpleNamespace(
        _format_error_penalty=penalty, _format_error_penalty_cap=cap
    )
    return TMaxRollouter._apply_format_error_penalty(rollouter, rollouts)


def _group(rollouts) -> RolloutGroup:
    group = RolloutGroup(group_id=0, rollouts=rollouts)
    advantages = AdvantageEstimator(AdvantageEstimator.Config())(group)
    for rollout, advantage in zip(rollouts, advantages, strict=True):
        rollout.advantage = advantage
    return group


def _mean(metrics, key):
    (metric,) = [x for x in metrics if x.key == key]
    return metric.value.value / metric.value.count


def test_penalty_is_per_error_and_capped():
    rollouts = [_rollout(1.0, 0, 0), _rollout(1.0, 2, 1), _rollout(0.0, 20, 2)]
    metrics = _apply(rollouts)
    assert [r.reward for r in rollouts] == pytest.approx([1.0, 0.9, -0.3])
    assert [task_reward(r) for r in rollouts] == [1.0, 1.0, 0.0]
    assert _mean(metrics, "rollout/format_error_penalty_mean") == pytest.approx(0.4 / 3)
    assert _mean(metrics, "rollout/shaped_reward_mean") == pytest.approx(1.6 / 3)


def test_fewer_errors_get_positive_advantage_in_an_all_fail_group():
    # Every sibling failed, so the raw group is zero-std; the penalty separates them.
    rollouts = [_rollout(0.0, 0, 0), _rollout(0.0, 1, 1), _rollout(0.0, 1, 2)]
    _apply(rollouts)
    advantages = AdvantageEstimator(AdvantageEstimator.Config())(
        RolloutGroup(group_id=0, rollouts=rollouts)
    )
    assert advantages[0] > 0 > advantages[1]


def test_unscored_siblings_are_left_alone():
    infra = _rollout(math.nan, 3, 0)
    ok = _rollout(1.0, 1, 1)
    _apply([infra, ok])
    assert math.isnan(infra.reward)
    assert "format_error_penalty" not in infra.diagnostics
    assert ok.reward == pytest.approx(0.95)


def test_reported_metrics_use_the_unshaped_reward():
    rollouts = [_rollout(1.0, 4, 0), _rollout(0.0, 0, 1)]
    _apply(rollouts)
    stats = [x for x in compute_rollout_metrics("rollout", rollouts) if x.key == "rollout_reward"]
    assert stats[0].value.total == pytest.approx(1.0)

    # Mixed group (kept): avg_train_reward is the solve rate, not the shaped mean.
    group = TrainingSampleBuilder(TrainingSampleBuilder.Config()).build_from_group(
        rollout_group=_group(rollouts)
    )
    assert _mean(group.metrics, "rollout_reward/avg_train_reward") == pytest.approx(0.5)


def test_zero_std_drop_sees_the_shaped_reward():
    builder = TrainingSampleBuilder(TrainingSampleBuilder.Config())
    # All failed, different error counts: shaped variance keeps the group.
    kept = [_rollout(0.0, 0, 0), _rollout(0.0, 2, 1)]
    _apply(kept)
    out = builder.build_from_group(rollout_group=_group(kept))
    assert out.training_samples
    assert _mean(out.metrics, "rollout_reward/avg_train_reward") == 0.0
    # All failed, same error count: still zero-std, dropped, reported as all-failed.
    dropped = [_rollout(0.0, 1, 0), _rollout(0.0, 1, 1)]
    _apply(dropped)
    out = builder.build_from_group(
        rollout_group=_group(dropped)
    )
    assert not out.training_samples
    assert _mean(out.metrics, "rollout_reward/group_all_failed_frac") == 1.0


def test_evolution_signal_counts_raw_solves(monkeypatch):
    written = []
    monkeypatch.setattr(
        tmax_rollouter, "_run_dir", lambda: SimpleNamespace(name="r", signal=lambda *a: a)
    )
    monkeypatch.setattr(
        tmax_rollouter.layout, "write_json_atomic", lambda path, data: written.append(data)
    )
    monkeypatch.setenv("SWE_EVOLUTION_SIGNALS", "1")
    # All solved; the capped penalty may not turn any of them into an unsolved verdict.
    rollouts = [_rollout(1.0, 0, 0), _rollout(1.0, 50, 1)]
    _apply(rollouts, penalty=0.5, cap=0.9)
    rollouter = SimpleNamespace(
        _reward_mode="sparse", _evolution_harder_ratio=1.0, _evolution_easier_ratio=0.0
    )
    sample = SimpleNamespace(instance_id="t", rev=0)
    TMaxRollouter._maybe_emit_evolution_signal(rollouter, sample, rollouts)
    assert written and written[0]["direction"] == "harder"
    assert written[0]["solved"] == 2


def test_config_validation():
    with pytest.raises(ValueError, match="format_error_penalty must"):
        TMaxRollouter(TMaxRollouter.Config(format_error_penalty=-0.1))
    with pytest.raises(ValueError, match="format_error_penalty_cap"):
        TMaxRollouter(TMaxRollouter.Config(format_error_penalty_cap=1.0))
    with pytest.raises(ValueError, match="format_error_penalty_cap"):
        TMaxRollouter(
            TMaxRollouter.Config(
                format_error_penalty_cap=0.95, length_penalty=True, length_penalty_max=0.1
            )
        )
