# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Partial (local-expert) weight pull for vllm_native expert-parallel ranks."""

from types import SimpleNamespace

import torch

from torchtitan.experiments.rl.native_ep_partial_pull import (
    local_expert_range,
    NativeEPPartialPull,
)

E, F, D = 16, 6, 4  # experts, expert hidden, model dim
KEY = "layers.3.moe.experts.w1_EFD"


class _FusedMoE(torch.nn.Module):
    def __init__(self, local: range | None):
        super().__init__()
        self.global_num_experts = E
        if local is None:
            self.expert_map = None
        else:
            m = torch.full((E,), -1)
            m[local.start : local.stop] = torch.arange(len(local))
            self.expert_map = m

    def _map_global_expert_id_to_local_expert_id(self, expert_id):
        return int(self.expert_map[expert_id])


def _model(local):
    model = torch.nn.Module()
    model.moe = _FusedMoE(local)
    return model


def test_local_expert_range():
    assert local_expert_range(_model(range(4, 8))) == (4, 8, E)
    assert local_expert_range(_model(None)) is None
    assert local_expert_range(_model(range(0, E))) is None
    assert local_expert_range(torch.nn.Linear(2, 2)) is None


def test_requests_fetch_only_overlapping_trainer_shards():
    """Trainer FSDP shards experts on dim 0 (4 experts per shard here); a rank
    holding experts [4, 8) must read exactly the shard covering them."""
    from torchstore.client import LocalClient
    from torchstore.transport.types import TensorSlice

    pull = NativeEPPartialPull(4, 8, E)
    full = torch.randn(E, F, D)
    pull.record({KEY: full, "norm.weight": torch.ones(D)})
    requests, inplace = pull.build_requests("model_state_dict", [KEY, "norm.weight"])
    expert_req = requests[0]
    assert expert_req.tensor_slice.offsets == (4, 0, 0)
    assert expert_req.tensor_slice.local_shape == (4, F, D)
    assert requests[1].tensor_slice is None  # dense keys are fetched whole

    stored = [
        TensorSlice((s, 0, 0), (i,), (E, F, D), (4, F, D), (4,))
        for i, s in enumerate(range(0, E, 4))
    ]
    sub = LocalClient._expand_tensor_slices(
        None, expert_req, SimpleNamespace(tensor_slices=stored), use_inplace=True
    )
    assert [r.tensor_slice.offsets for r in sub] == [(4, 0, 0)]
    # the fetched rows land in the whole local buffer
    assert sub[0].tensor_val.data_ptr() == inplace[expert_req.key].data_ptr()


def test_embed_hf_and_vllm_style_load_match_full_pull():
    """Embedding the local slice and loading by global expert id (skipping
    non-local ids before reading, like vLLM's FusedMoE.weight_loader) gives the
    same local experts as loading the full tensor."""
    start, stop = 8, 12
    pull = NativeEPPartialPull(start, stop, E)
    full = torch.randn(E, 2 * F, D)
    hf_key = "model.language_model.layers.3.mlp.experts.gate_up_proj"
    embedded = pull.embed_hf({hf_key: full[start:stop].clone(), "x.weight": torch.ones(1)})
    assert embedded["x.weight"].shape == (1,)

    def load(weight):
        moe = _FusedMoE(range(start, stop))
        param = torch.zeros(stop - start, 2 * F, D)
        for expert_id in range(E):
            local = moe._map_global_expert_id_to_local_expert_id(expert_id)
            if local == -1:
                continue  # never reads weight[expert_id]
            param[local].copy_(weight[expert_id])
        return param

    assert torch.equal(load(embedded[hf_key]), load(full))
    # the buffer is reused across pulls
    again = pull.embed_hf({hf_key: full[start:stop].clone()})
    assert again[hf_key].data_ptr() == embedded[hf_key].data_ptr()


def test_record_rejects_unexpected_expert_count():
    pull = NativeEPPartialPull(0, 4, E)
    pull.record({KEY: torch.zeros(E + 1, F, D)})
    assert not pull.ready
