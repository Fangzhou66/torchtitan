# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""cpu_staged_weight_pull: a failed RDMA pull is retried over gloo."""

import asyncio
import os
from types import SimpleNamespace

import pytest

from torchtitan.experiments.rl.actors.generator import VLLMGenerator


def _generator(fetch):
    gen = object.__new__(VLLMGenerator)
    gen.config = SimpleNamespace(cpu_staged_weight_pull=True)
    gen._dp_weight_broadcaster = None
    gen._backend = "torchtitan_wrapper"
    gen._cpu_pull_buffer = {"w": object()}
    gen._fetch_model_state_dict = fetch
    gen.applied = []
    gen._apply_model_state_dict = lambda version, staged, **_: gen.applied.append(
        (version, staged)
    )
    return gen


def test_failed_rdma_pull_retries_over_gloo_and_restores_env(monkeypatch) -> None:
    monkeypatch.setenv("TORCHSTORE_RDMA_ENABLED", "1")
    seen = []

    async def fetch(version, *, overlap_safe, staged):
        seen.append(os.environ.get("TORCHSTORE_RDMA_ENABLED"))
        if len(seen) == 1:
            raise RuntimeError("RdmaAction.submit failed: 964/964 ops failed")
        return staged, 1.0, 17.5

    gen = _generator(fetch)
    asyncio.run(gen._pull_model_state_dict(3))
    assert seen == ["1", "0"]
    assert gen.applied == [(3, gen._cpu_pull_buffer)]
    assert os.environ["TORCHSTORE_RDMA_ENABLED"] == "1"


def test_failed_gloo_retry_still_raises(monkeypatch) -> None:
    monkeypatch.delenv("TORCHSTORE_RDMA_ENABLED", raising=False)

    async def fetch(version, *, overlap_safe, staged):
        raise RuntimeError("store down")

    gen = _generator(fetch)
    with pytest.raises(RuntimeError, match="store down"):
        asyncio.run(gen._pull_model_state_dict(3))
    assert "TORCHSTORE_RDMA_ENABLED" not in os.environ
    assert gen.applied == []


def test_failed_gpu_direct_pull_falls_back_to_gloo_into_cpu_copy(monkeypatch) -> None:
    monkeypatch.setenv("TORCHSTORE_RDMA_ENABLED", "1")
    calls = []

    async def fetch(version, *, overlap_safe, staged):
        calls.append((os.environ.get("TORCHSTORE_RDMA_ENABLED"), staged))
        if len(calls) == 1:
            raise RuntimeError("RdmaAction.submit failed: 964/964 ops failed")
        return staged, 1.0, 17.5

    gen = _generator(fetch)
    gen.config.cpu_staged_weight_pull = False
    asyncio.run(gen._pull_model_state_dict(4))
    assert calls[0] == ("1", None)  # GPU-direct first
    assert calls[1] == ("0", gen._cpu_pull_buffer)  # then gloo into the CPU copy
    assert gen.applied == [(4, gen._cpu_pull_buffer)]
