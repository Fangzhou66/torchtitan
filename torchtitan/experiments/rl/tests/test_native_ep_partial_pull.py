# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Partial (local-expert) weight pull for vllm_native expert-parallel ranks."""

from types import SimpleNamespace

import torch

from torchtitan.experiments.rl.native_ep_partial_pull import (
    expert_dims,
    local_expert_range,
    NativeEPPartialPull,
)

E, F, D = 16, 6, 4  # experts, expert hidden, model dim
KEY = "layers.3.moe.experts.w1_EFD"


class _FusedMoE(torch.nn.Module):
    def __init__(self, local: range | None, tp_size: int = 1):
        super().__init__()
        self.global_num_experts = E
        self.tp_size = tp_size
        self.hidden_size = D
        self.intermediate_size_per_partition = F
        self.params_dtype = torch.bfloat16
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
    pull.last_fetch_partial = True  # as after fetch()
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


def test_first_full_pull_is_not_embedded():
    """record() runs on the first, full pull; that same pull's full HF experts
    must pass through embed_hf unchanged (regression: it tried to copy all E
    experts into the local slice)."""
    pull = NativeEPPartialPull(8, 12, E)
    tt_key = "layers.3.moe.experts.w1_EFD"
    pull.record({tt_key: torch.randn(E, F, D)})
    assert pull.ready and not pull.last_fetch_partial
    hf_key = "model.language_model.layers.3.mlp.experts.gate_up_proj"
    full_hf = torch.randn(E, 2 * F, D)
    out = pull.embed_hf({hf_key: full_hf})
    assert out[hf_key] is full_hf


def test_dims_give_record_shapes_without_a_full_pull():
    """Shapes read from the vLLM layer must equal what record() learns from a
    full pull, so the first pull can already be partial (a full first pull holds
    the whole checkpoint once per EP rank in host memory)."""
    dims = expert_dims(_model(range(4, 8)))
    assert dims == (D, F, torch.bfloat16)
    tp2 = torch.nn.Module()
    tp2.moe = _FusedMoE(range(4, 8), tp_size=2)
    assert expert_dims(tp2) is None

    keys = [f"layers.3.moe.experts.{n}" for n in ("w1_EFD", "w2_EDF", "w3_EFD")]
    recorded = NativeEPPartialPull(4, 8, E)
    recorded.record(
        {
            keys[0]: torch.zeros(E, F, D, dtype=torch.bfloat16),
            keys[1]: torch.zeros(E, D, F, dtype=torch.bfloat16),
            keys[2]: torch.zeros(E, F, D, dtype=torch.bfloat16),
        }
    )
    from_dims = NativeEPPartialPull(4, 8, E, dims=dims)
    assert from_dims.ready
    from_dims._meta_from_dims(keys + ["norm.weight"])
    assert from_dims._expert_meta == recorded._expert_meta
