# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3.5 norms that keep only their bf16 inputs for backward.

The eager norms upcast to fp32 and autograd saves several fp32 intermediates.
For GDN's RMSNormGated at 65536 tokens x 64 heads x 128 that is ~10 GiB per
layer, held during the block's FullAC recompute. A nested non-reentrant
checkpoint saves the inputs and replays the same eager ops in backward, so
outputs and gradients are bitwise unchanged; the replay is a few elementwise
kernels.

Activate on the trainer with
``trainer.override.imports=["torchtitan.experiments.rl.models.lean_norms"]``.
"""

from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint

from torchtitan.config import derive, override
from torchtitan.models.qwen3_5.model import OffsetRMSNorm, RMSNormGated


class LeanOffsetRMSNorm(OffsetRMSNorm):
    @dataclass(kw_only=True, slots=True)
    class Config(OffsetRMSNorm.Config):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return super().forward(x)
        return checkpoint(super().forward, x, use_reentrant=False)


class LeanRMSNormGated(RMSNormGated):
    @dataclass(kw_only=True, slots=True)
    class Config(RMSNormGated.Config):
        pass

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return super().forward(x, gate)
        return checkpoint(super().forward, x, gate, use_reentrant=False)


@override(
    "lean_offset_rmsnorm",
    target=OffsetRMSNorm.Config,
    exact=True,
    description="Qwen3.5 OffsetRMSNorm saving only its input for backward.",
)
def lean_offset_rmsnorm(cfg: OffsetRMSNorm.Config) -> LeanOffsetRMSNorm.Config:
    return derive(cfg, LeanOffsetRMSNorm.Config)


@override(
    "lean_gated_rmsnorm",
    target=RMSNormGated.Config,
    exact=True,
    description="Qwen3.5 GDN RMSNormGated saving only its inputs for backward.",
)
def lean_gated_rmsnorm(cfg: RMSNormGated.Config) -> LeanRMSNormGated.Config:
    return derive(cfg, LeanRMSNormGated.Config)
