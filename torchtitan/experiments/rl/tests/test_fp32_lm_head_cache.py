# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The vllm_native fp32 lm_head cache is built under inference mode (inside vLLM's
forward) and refreshed after each weight sync, outside it."""

import torch

from torchtitan.experiments.rl.actors.generator import _refresh_fp32_lm_head_cache


def test_refresh_updates_an_inference_mode_cache_in_place():
    lm_head = torch.nn.Linear(8, 16, bias=False).to(torch.bfloat16)
    with torch.inference_mode():
        # What _install_fp32_native_lm_head does on the first lm_head call.
        object.__setattr__(lm_head, "_tt_w32", lm_head.weight.detach().float())
    cache = lm_head._tt_w32
    assert cache.is_inference()

    with torch.no_grad():
        lm_head.weight.copy_(torch.randn(16, 8).to(torch.bfloat16))
    _refresh_fp32_lm_head_cache(torch.nn.Sequential(lm_head))

    assert lm_head._tt_w32 is cache  # updated in place, so CUDA graphs see it
    assert torch.equal(cache, lm_head.weight.float())
