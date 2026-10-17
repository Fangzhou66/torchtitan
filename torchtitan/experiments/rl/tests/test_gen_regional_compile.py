# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generator-side regional compile compiles only feed-forward and norms."""

import torch
from torch import nn

from torchtitan.experiments.rl.models.vllm_wrapper import maybe_regional_compile


class _Block(nn.Module):
    def __init__(self, with_ffn: bool = True):
        super().__init__()
        self.attn = nn.Linear(4, 4)
        self.attention_norm = nn.LayerNorm(4)
        self.ffn_norm = nn.LayerNorm(4)
        if with_ffn:
            self.feed_forward = nn.Linear(4, 4)
        else:
            self.moe = nn.Linear(4, 4)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleDict({"0": _Block(), "1": _Block(with_ffn=False)})


def _compiled(m: nn.Module) -> bool:
    return getattr(m, "_compiled_call_impl", None) is not None


def test_off_by_default(monkeypatch) -> None:
    monkeypatch.delenv("SWE_GEN_REGIONAL_COMPILE", raising=False)
    model = _Model()
    assert maybe_regional_compile(model) == 0
    assert not any(_compiled(m) for m in model.modules())


def test_compiles_only_ffn_and_norms(monkeypatch) -> None:
    monkeypatch.setenv("SWE_GEN_REGIONAL_COMPILE", "1")
    model = _Model()
    assert maybe_regional_compile(model) == 5  # 2 + 2 norms, 1 feed_forward
    b0, b1 = model.layers["0"], model.layers["1"]
    assert _compiled(b0.feed_forward) and _compiled(b0.attention_norm)
    assert _compiled(b0.ffn_norm) and _compiled(b1.attention_norm)
    assert not _compiled(b0.attn) and not _compiled(b1.attn)
    assert not _compiled(b1.moe)  # MoE blocks keep their experts eager
    torch._dynamo.reset()
