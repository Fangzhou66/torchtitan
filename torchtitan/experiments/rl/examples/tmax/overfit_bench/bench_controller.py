# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The controller of an overfit-bench job under the scheduler (SWE_BENCH_SCHEDULER=1).

It owns the BenchScheduler: started before the rollout workers are spawned (so
the controller's rollouter can draw and a checkpoint can restore the pool),
stopped when the run ends. After each training step's weight pull it emits
``step_done``; the pull at startup is not a step and emits nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from torchtitan.experiments.rl.controller import Controller
from torchtitan.experiments.rl.examples.tmax import layout
from torchtitan.experiments.rl.examples.tmax.overfit_bench import events
from torchtitan.experiments.rl.examples.tmax.overfit_bench.scheduler import (
    BenchScheduler,
)

logger = logging.getLogger(__name__)


class BenchController(Controller):
    @dataclass(kw_only=True, slots=True)
    class Config(Controller.Config):
        bench_root: str = ""
        """The job directory ($TRL_BASE): its data/mix is the pool's."""

        bench_hook_timeout_sec: float = 1.0
        """How long a draw waits for the rewriter's which_version answer."""

        def __post_init__(self) -> None:
            super(BenchController.Config, self).__post_init__()
            if not Path(self.bench_root).is_absolute():
                raise ValueError("bench_root must be an absolute job directory")

    _bench_scheduler: BenchScheduler | None = None
    _bench_last_step: int | None = None

    async def setup_async(self, **kwargs) -> None:
        config = self.config
        assert isinstance(config, BenchController.Config)
        self._bench_scheduler = BenchScheduler(
            layout.Root(Path(config.bench_root)),
            run=layout.Run.from_env(),
            seed=config.generator.debug.seed or 0,
            hook_timeout_sec=config.bench_hook_timeout_sec,
        )
        self._bench_scheduler.start()
        await super().setup_async(**kwargs)

    async def _pull_generator_weights(
        self, *, policy_version: int, include_eval: bool = True
    ) -> None:
        await super()._pull_generator_weights(
            policy_version=policy_version, include_eval=include_eval
        )
        # policy_version counts completed optimizer steps, so it is the step
        # number; the startup pull is at start_step and is not a step.
        if policy_version > self.start_step and policy_version != self._bench_last_step:
            self._bench_last_step = policy_version
            events.emit(events.STEP_DONE, step=policy_version, policy_version=policy_version)

    async def run(self) -> None:
        try:
            await super().run()
        finally:
            if self._bench_scheduler is not None:
                self._bench_scheduler.stop()
