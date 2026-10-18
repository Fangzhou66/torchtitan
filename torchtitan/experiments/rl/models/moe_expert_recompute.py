# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Routed experts that keep only their input rows for backward.

Under FullAC the block recompute holds every grouped-mm intermediate
(w1/w3 outputs, SiLU, product, w2 output) until the block's backward. On an EP
rank that receives ~1.8x the average rows (hot experts) that is 10+ GiB at 65536
tokens. A nested non-reentrant checkpoint around the expert compute saves the
routed input and replays the same grouped mms in backward: gradients are
unchanged, the cost is one more expert forward.

Activate on the trainer with
``trainer.override.imports=["torchtitan.experiments.rl.models.moe_expert_recompute"]``.
"""

from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint

from torchtitan.config import derive, override
from torchtitan.models.common.moe import GroupedExperts


class RecomputeGroupedExperts(GroupedExperts):
    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        pass

    def _experts_forward(
        self, x_RD: torch.Tensor, num_tokens_per_expert_E: torch.Tensor
    ) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return super()._experts_forward(x_RD, num_tokens_per_expert_E)
        return checkpoint(
            super()._experts_forward, x_RD, num_tokens_per_expert_E, use_reentrant=False
        )


@override(
    "moe_expert_recompute",
    target=GroupedExperts.Config,
    exact=True,
    description="Routed experts saving only their input rows for backward.",
)
def moe_expert_recompute(cfg: GroupedExperts.Config) -> RecomputeGroupedExperts.Config:
    return derive(cfg, RecomputeGroupedExperts.Config)
