# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-turn advantage offset on format-error turns (SWE_FORMAT_ERROR_TURN_ADV)."""

from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace

import pytest

from torchtitan.experiments.rl.components.batcher import Batcher, BatchConfig
from torchtitan.experiments.rl.components.training_sample_builder import (
    TrainingSampleBuilder,
)
from torchtitan.experiments.rl.examples.tmax.rollouter import (
    _rollout_turn_indices,
    TMaxRollouter,
)
from torchtitan.experiments.rl.harness.adapters.anthropic import CapturedTurn
from torchtitan.experiments.rl.rollout.advantage import AdvantageEstimator
from torchtitan.experiments.rl.rollout.types import (
    Rollout,
    RolloutGroup,
    RolloutStatus,
    RolloutTurn,
    TURN_ADVANTAGE_OFFSETS_KEY,
)
from torchtitan.experiments.rl.types import RolloutTurnID

_C = 0.5


def _rollout(reward: float, error_turns: list[int], idx: int, *, group_id=0):
    """Three append-only turns of 2 completion tokens each."""
    turns = []
    prefix: list[int] = []
    for t in range(3):
        prompt = prefix + [100 + t]
        completion = [10 * (t + 1), 10 * (t + 1) + 1]
        turns.append(
            RolloutTurn(
                rollout_id=RolloutTurnID(group_id=group_id, rollout_id=idx, turn_id=t),
                prompt_token_ids=prompt,
                completion_token_ids=completion,
                completion_logprobs=[-0.1, -0.2],
                min_policy_version=0,
                max_policy_version=0,
            )
        )
        prefix = prompt + completion
    return Rollout(
        group_id=group_id,
        rollout_id=idx,
        turns=turns,
        status=RolloutStatus.COMPLETED,
        reward=reward,
        diagnostics={
            "format_errors": len(error_turns),
            "format_error_turns": list(error_turns),
        },
    )


def _apply(rollouts, c=_C):
    rollouter = SimpleNamespace(_format_error_turn_advantage=c)
    return TMaxRollouter._apply_format_error_turn_offsets(rollouter, rollouts)


def _group(rollouts, group_id=0) -> RolloutGroup:
    group = RolloutGroup(group_id=group_id, rollouts=rollouts)
    advantages = AdvantageEstimator(AdvantageEstimator.Config())(group)
    for rollout, advantage in zip(rollouts, advantages, strict=True):
        rollout.advantage = advantage
    return group


def _turn_advantages(sample) -> list[float]:
    """The advantage of each turn's completion (2 tokens each), in order."""
    trained = [a for a, keep in zip(sample.advantage, sample.loss_mask) if keep]
    assert len(trained) == 6
    assert trained[0::2] == trained[1::2]
    return trained[0::2]


def _mean(metrics, key):
    (metric,) = [x for x in metrics if x.key == key]
    return metric.value.value / metric.value.count


def test_offset_only_on_error_turns():
    rollouts = [_rollout(1.0, [], 0), _rollout(0.0, [0, 2], 1)]
    _apply(rollouts)
    out = TrainingSampleBuilder(TrainingSampleBuilder.Config()).build_from_group(
        rollout_group=_group(rollouts)
    )
    clean, erred = out.training_samples
    a0, a1 = rollouts[0].advantage, rollouts[1].advantage
    assert a0 > 0 > a1
    assert _turn_advantages(clean) == [a0, a0, a0]
    assert _turn_advantages(erred) == [a1 - _C, a1, a1 - _C]
    # 2 of 6 trained turns and tokens carry the offset.
    assert _mean(out.metrics, "training_sample_builder/advantage_offset_turn_frac") == (
        pytest.approx(2 / 6)
    )
    assert _mean(
        out.metrics, "training_sample_builder/advantage_offset_token_frac"
    ) == pytest.approx(2 / 6)


def test_all_error_zero_std_group_still_trains_its_error_turns():
    # Every sibling failed and made the turn-0 error: rollout advantages are all 0.
    rollouts = [_rollout(0.0, [0], i) for i in range(4)]
    metrics = _apply(rollouts)
    assert _mean(metrics, "rollout/format_error_turn_frac") == pytest.approx(4 / 12)
    group = _group(rollouts)
    assert all(r.advantage == 0.0 for r in rollouts)

    out = TrainingSampleBuilder(TrainingSampleBuilder.Config()).build_from_group(
        rollout_group=group
    )
    assert len(out.training_samples) == 4
    keys = {x.key for x in out.metrics}
    assert "training_sample_builder/num_zero_std_groups_kept_for_offsets" in keys
    assert "training_sample_builder/num_groups_dropped_zero_std" not in keys
    for sample in out.training_samples:
        assert _turn_advantages(sample) == [-_C, 0.0, 0.0]

    # The batcher's zero-advantage skip sees them as nonzero-advantage samples.
    batch = (
        Batcher.Config(
            batch=BatchConfig(local_batch_size=4, seq_len=64),
            skip_zero_advantage_samples=True,
            zero_advantage_tokens_in_loss_denominator=False,
        )
        .build(num_groups_per_train_step=1, dp_degree=1, pad_id=0, initial_policy_version=0)
        .add_training_samples(training_sample_group=out)
    )
    assert batch is not None
    assert batch.num_packed_valid_tokens == batch.num_global_valid_tokens > 0
    assert batch.num_loss_denominator_tokens == batch.num_global_valid_tokens


def test_zero_std_group_without_offsets_is_still_dropped():
    rollouts = [_rollout(0.0, [], i) for i in range(3)]
    _apply(rollouts)
    out = TrainingSampleBuilder(TrainingSampleBuilder.Config()).build_from_group(
        rollout_group=_group(rollouts)
    )
    assert not out.training_samples


def test_off_by_default_is_bit_identical():
    def build(apply_c):
        rollouts = [_rollout(1.0, [1], 0), _rollout(0.0, [0, 2], 1), _rollout(0.0, [], 2)]
        if apply_c is None:
            # No per-turn attribution at all (e.g. another harness or example).
            for rollout in rollouts:
                del rollout.diagnostics["format_error_turns"]
        else:
            _apply(rollouts, c=apply_c)
        out = TrainingSampleBuilder(TrainingSampleBuilder.Config()).build_from_group(
            rollout_group=_group(rollouts)
        )
        return [s.advantage for s in out.training_samples], rollouts

    baseline, _ = build(None)
    # The rollouter only attaches offsets when the knob is on.
    assert TMaxRollouter.Config().format_error_turn_advantage == 0.0
    off, rollouts = build(0.0)
    assert all(
        not r.diagnostics.get(TURN_ADVANTAGE_OFFSETS_KEY) for r in rollouts
    )
    assert off == baseline
    assert [[math.copysign(1, a) for a in s] for s in off] == [
        [math.copysign(1, a) for a in s] for s in baseline
    ]


def test_validation_and_unscored_rollouts_get_no_offset():
    # Validation group: offsets present in diagnostics are ignored by the builder.
    val = [_rollout(1.0, [0], 0, group_id=-1), _rollout(0.0, [0], 1, group_id=-1)]
    _apply(val)
    builder = TrainingSampleBuilder(TrainingSampleBuilder.Config())
    for rollout in val:
        rollout.advantage = 0.25
        for sample in builder.rollout_to_training_samples(rollout):
            assert _turn_advantages(sample) == [0.25, 0.25, 0.25]
    # Infra-failed (NaN) sibling: no offsets attached.
    infra = _rollout(math.nan, [0], 0)
    ok = _rollout(0.0, [1], 1)
    _apply([infra, ok])
    assert TURN_ADVANTAGE_OFFSETS_KEY not in infra.diagnostics
    assert ok.diagnostics[TURN_ADVANTAGE_OFFSETS_KEY] == {1: -_C}


def _captured(num_tokens: int) -> CapturedTurn:
    return CapturedTurn(
        prompt_token_ids=[1],
        completion_token_ids=[2] * num_tokens,
        completion_logprobs=[0.0] * num_tokens,
        min_policy_version=0,
        max_policy_version=0,
        finish_reason="stop",
        extends_previous=True,
    )


def test_capture_indices_map_past_dropped_empty_turns():
    # Capture 1 is an over-budget empty turn, which _captured_to_turns drops.
    captured = [_captured(2), _captured(0), _captured(3), _captured(1)]
    assert _rollout_turn_indices(captured, [0, 2, 3]) == [0, 1, 2]
    assert _rollout_turn_indices(captured, [1]) == []
    assert _rollout_turn_indices(captured, []) == []


def test_terminus_attributes_errors_to_the_captured_turn():
    from harbor.llms.base import OutputLengthExceededError

    from torchtitan.experiments.rl.harness.agents.terminus import (
        _AdapterLLM,
        _CountingParser,
    )

    replies = [
        ("<analysis>a</analysis>", "end_turn"),  # 0: error
        ("ok", "end_turn"),  # 1: parsed into a command
        None,  # adapter exhausted: nothing captured
        ("<response><commands>ls", "max_tokens"),  # 2: truncated, parsed for warnings
        ("bad", "end_turn"),  # 3: error
    ]

    class _Adapter:
        def session_max_tokens(self, _sid):
            return None

        async def complete(self, _sid, _payload):
            reply = replies.pop(0)
            if reply is None:
                return None
            text, stop = reply
            return {
                "content": [{"type": "text", "text": text}],
                "stop_reason": stop,
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }

    class _Inner:
        def parse_response(self, content):
            commands = ["ls"] if content == "ok" else []
            return SimpleNamespace(commands=commands, is_task_complete=False, error="")

    llm = _AdapterLLM(_Adapter(), session_id="s", max_context=0, turn_max_tokens=16)
    parser = _CountingParser(_Inner(), llm)

    async def drive():
        for _ in range(2):
            parser.parse_response((await llm.call(prompt="p")).content)
        with pytest.raises(RuntimeError, match="no completion"):
            await llm.call(prompt="p")
        # A fallback string Terminus-2 parses after a failed call: not attributed.
        parser.parse_response("Technical difficulties.")
        with pytest.raises(OutputLengthExceededError) as excinfo:
            await llm.call(prompt="p")
        parser.parse_response(excinfo.value.truncated_response)
        parser.parse_response((await llm.call(prompt="p")).content)

    asyncio.run(drive())
    assert llm.captured_turns == 4
    assert parser.format_errors == 4
    assert parser.format_error_turns == [0, 2, 3]


def test_config_validation():
    with pytest.raises(ValueError, match="format_error_turn_advantage must"):
        TMaxRollouter(TMaxRollouter.Config(format_error_turn_advantage=-0.1))
