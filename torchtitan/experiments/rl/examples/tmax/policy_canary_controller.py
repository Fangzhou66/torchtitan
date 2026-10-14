# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Admission probes for the TMax experiment's terminal-task mixtures."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

from torchtitan.experiments.rl.controller import Controller
from torchtitan.experiments.rl.eval_trace_recorder import ValidationTraceRecorder
from torchtitan.experiments.rl.examples.tmax import layout
from torchtitan.experiments.rl.examples.tmax.data import _parse_sample_row
from torchtitan.experiments.rl.examples.tmax.policy_canary import (
    PolicyCanaryConfig,
    PolicyCanaryQueue,
    PolicyCanaryService,
    summarize_group,
)
from torchtitan.experiments.rl.observability import metrics as m

logger = logging.getLogger(__name__)


class _TokenizerDecoder(Protocol):
    def decode(self, token_ids: list[int]) -> str: ...


class _TokenizingRenderer(Protocol):
    _tokenizer: _TokenizerDecoder


class PolicyCanaryController(Controller):
    @dataclass(kw_only=True, slots=True)
    class Config(Controller.Config):
        policy_canary: PolicyCanaryConfig = field(default_factory=PolicyCanaryConfig)
        policy_canary_root: str = ""

        def __post_init__(self) -> None:
            super(PolicyCanaryController.Config, self).__post_init__()
            if self.policy_canary.group_size < 2:
                raise ValueError(
                    "PolicyCanaryController requires an enabled policy canary"
                )
            if not Path(self.policy_canary_root).is_absolute():
                raise ValueError(
                    "policy_canary_root must be an absolute experiment root"
                )
            if os.environ.get("TMAX_AGENT", "") != "terminus":
                raise ValueError(
                    "policy canaries require the training Terminus scaffold"
                )

    async def _evaluate_policy_canary(self, request: dict, identity: str) -> dict:
        samples = [_parse_sample_row(request[key]) for key in ("original", "candidate")]
        if any(sample.instance_id != request["task"] for sample in samples):
            raise ValueError("policy canary rows do not match the requested task")
        if [sample.rev for sample in samples] != [
            request["input_rev"],
            request["input_rev"] + 1,
        ]:
            raise ValueError("policy canary rows do not match the requested revisions")
        group_ids = self._allocate_validation_group_ids(2)

        async def run_group(index: int):
            # Own the orchestration locally so cancellation drains sandbox children.
            return await self._rollouter.run_group_rollouts(
                sample=samples[index],
                group_id=group_ids[index],
                group_size=request["group_size"],
                sampling=self._sampling,
                generate_fn=self._make_generate_fn(
                    metrics_prefix="policy_canary_generator"
                ),
                renderer=self.renderer,
            )

        tasks = [asyncio.create_task(run_group(index)) for index in range(2)]
        try:
            groups = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                if not task.done() and not task.cancelling():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        before, after = [
            summarize_group(group, group_size=request["group_size"]) for group in groups
        ]
        versions = [group["policy_version_max"] for group in (before, after)]
        # Report this as an online policy span, not a frozen-checkpoint benchmark.
        recorder = ValidationTraceRecorder.Config(
            enable=True, dirname=f"policy_canary_traces/{identity}"
        ).build(dump_dir=self.config.dump_folder)
        # Controller requires an HF-backed renderer, but Renderer omits this attribute.
        tokenizer = cast(_TokenizingRenderer, self.renderer)._tokenizer
        summary = recorder.record(
            policy_version=max(
                (v for v in versions if v is not None),
                default=self._generator_policy_version,
            ),
            groups=groups,
            task_ids=[f"{request['task']}--original", f"{request['task']}--candidate"],
            decode=tokenizer.decode,
            expected_num_tasks=2,
            expected_num_trials=2 * request["group_size"],
            valid=all(
                group["num_scored"] == request["group_size"]
                for group in (before, after)
            ),
        )
        assert summary is not None
        return {
            "before": before,
            "after": after,
            "traces": summary.report_dir,
            "scaffold": "terminus",
            "sampling": {
                "temperature": self._sampling.temperature,
                "top_p": self._sampling.top_p,
                "max_tokens": self._sampling.max_tokens,
            },
        }

    async def run(self) -> None:
        config = self.config
        assert isinstance(config, PolicyCanaryController.Config)
        queue = PolicyCanaryQueue(layout.Root(Path(config.policy_canary_root)))
        self._policy_canary_service = PolicyCanaryService(
            queue,
            config=config.policy_canary,
            evaluate=self._evaluate_policy_canary,
        )
        self._policy_canary_task = asyncio.create_task(
            self._policy_canary_service.run(), name="policy_canaries"
        )

        def report_failure(task: asyncio.Task) -> None:
            if not task.cancelled() and task.exception() is not None:
                logger.error(
                    "policy canary service stopped; candidates cannot publish",
                    exc_info=task.exception(),
                )

        self._policy_canary_task.add_done_callback(report_failure)
        try:
            await super().run()
        finally:
            self._policy_canary_task.cancel()
            await asyncio.gather(self._policy_canary_task, return_exceptions=True)

    def _evolution_metrics(self, step: int | None = None) -> list[m.Metric]:
        return [
            *super()._evolution_metrics(step),
            m.Metric(
                "evolution/policy_canary/active",
                m.NoReduce(float(self._policy_canary_service.active_count)),
            ),
            m.Metric(
                "evolution/policy_canary/waiting",
                m.NoReduce(float(self._policy_canary_service.waiting_count)),
            ),
        ]

    async def _trainer_loop(self, *args, **kwargs) -> None:
        try:
            await super()._trainer_loop(*args, **kwargs)
        finally:
            self._policy_canary_task.cancel()
            await asyncio.gather(self._policy_canary_task, return_exceptions=True)
