# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A stalled RolloutWorker leaves its thread stacks on local disk, and the
controller reports a worker that stops answering pings together with them."""

from __future__ import annotations

import asyncio
import faulthandler
import logging
import os
import time
from types import SimpleNamespace

from torchtitan.experiments.rl.actors.rollout_worker import RolloutWorker
from torchtitan.experiments.rl.controller import Controller


def test_stalled_loop_dumps_stacks_and_reports_them_once_it_recovers(caplog):
    worker = SimpleNamespace(
        config=SimpleNamespace(rollout_worker_ping_timeout_sec=0.4),
        _stall_dump_path="",
        _stall_watch=None,
    )

    async def scenario() -> tuple[str, int]:
        RolloutWorker._start_stall_dump(worker)
        await asyncio.sleep(0)  # arm the timer
        time.sleep(0.6)  # block the loop past the 0.3 s dump deadline
        with open(worker._stall_dump_path) as f:
            during = f.read()
        await asyncio.sleep(0.3)  # the loop runs again and reports the stall
        size = os.path.getsize(worker._stall_dump_path)
        worker._stall_watch.cancel()
        faulthandler.cancel_dump_traceback_later()
        return during, size

    with caplog.at_level(logging.WARNING, logger="torchtitan"):
        during, size_after = asyncio.run(scenario())
    assert "test_rollout_worker_watch.py" in during, during
    assert size_after == 0
    assert any("event loop stalled" in r.getMessage() for r in caplog.records)


class _Worker:
    def __init__(self, answer: str | None):
        async def call_one():
            if answer is None:
                await asyncio.sleep(10)
            return answer

        self.ping = SimpleNamespace(call_one=call_one)


def test_silent_worker_is_reported_once_with_its_dump(tmp_path, caplog):
    dump = tmp_path / "stall.txt"
    dump.write_text('Thread 0x1 (most recent call first):\n  File "x.py", line 1\n')
    healthy, silent = _Worker(str(dump)), _Worker(str(dump))
    controller = SimpleNamespace(
        config=SimpleNamespace(
            rollout_worker_ping_interval_sec=0.01,
            rollout_worker_ping_timeout_sec=0.05,
        ),
        _rollout_workers=[healthy, silent],
        _eval_rollout_workers=[],
    )
    controller._ping_rollout_worker = lambda w: Controller._ping_rollout_worker(
        controller, w
    )

    async def scenario():
        task = asyncio.create_task(Controller._watch_rollout_workers(controller))
        await asyncio.sleep(0.05)
        # Learn the dump path from a first answer, then fall silent.
        silent.ping = _Worker(None).ping
        await asyncio.sleep(0.4)
        task.cancel()

    with caplog.at_level(logging.WARNING):
        asyncio.run(scenario())
    reports = [r.getMessage() for r in caplog.records if "did not answer" in r.getMessage()]
    assert len(reports) == 1, reports
    assert "rollout_worker_1" in reports[0] and 'File "x.py"' in reports[0]
