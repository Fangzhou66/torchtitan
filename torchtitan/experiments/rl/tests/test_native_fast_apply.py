# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SWE_NATIVE_FAST_APPLY: the fast apply must load bitwise the same vLLM
parameters as the default ``to_hf`` (+ EP ``embed_hf``) + ``load_weights`` path.

Uses the TP partial-pull test's torchstore fake and vLLM parameter tree, with
each layer's ``mlp.experts`` a FusedMoE whose ``weight_loader`` is vLLM's own,
under fake EP / MoE-TP placements.
"""

import asyncio
import types

import pytest
import torch

from torchtitan.experiments.rl.native_ep_partial_pull import (
    local_expert_range,
    NativeEPPartialPull,
)
from torchtitan.experiments.rl.native_fast_apply import NativeFastApply
from torchtitan.experiments.rl.native_tp_partial_pull import NativeTPPartialPull

pytest.importorskip("vllm")
pytest.importorskip("torchstore")

from torchtitan.experiments.rl.tests import (  # noqa: E402
    test_native_tp_partial_pull as tpt,
)

E, FE = tpt.E, 24  # global experts, expert hidden
DT, DIM = tpt.DT, tpt.DIM


def _state_dict(seed: int) -> dict[str, torch.Tensor]:
    sd = tpt._tt_state_dict(seed)
    g = torch.Generator().manual_seed(seed + 100)
    for i in range(len(tpt.LAYER_TYPES)):
        for name, shape in (("w1_EFD", (E, FE, DIM)), ("w3_EFD", (E, FE, DIM)), ("w2_EDF", (E, DIM, FE))):
            sd[f"layers.{i}.moe.experts.{name}"] = torch.randn(*shape, generator=g).to(DT)
    return sd


def _fused_moe(local: range | None, moe_tp: tuple[int, int] = (0, 1)):
    """A FusedMoE with vLLM's weight_loader, holding experts ``local`` (None:
    all, no EP), its intermediate dim split ``moe_tp = (rank, size)`` ways."""
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod,
    )

    m = FusedMoE.__new__(FusedMoE)
    torch.nn.Module.__init__(m)
    tp_rank, tp_size = moe_tp
    par = types.SimpleNamespace(tp_size=tp_size, tp_rank=tp_rank, ep_size=1, ep_rank=0)
    f = FE // tp_size
    m.moe_parallel_config = par
    m.moe_config = types.SimpleNamespace(
        is_act_and_mul=True, moe_parallel_config=par, hidden_dim=DIM,
        intermediate_size_per_partition=f,
    )
    m.quant_config = None
    quant = UnquantizedFusedMoEMethod.__new__(UnquantizedFusedMoEMethod)
    torch.nn.Module.__init__(quant)  # a CustomOp; only its class name is read
    m.quant_method = quant
    m.global_num_experts = E
    m.params_dtype = DT
    m.rocm_aiter_fmoe_enabled = False
    m.local_num_experts = E if local is None else len(local)
    if local is None:
        m._expert_map = None
        n = E
        to_local = lambda g: g  # noqa: E731
    else:
        m._expert_map = torch.full((E,), -1, dtype=torch.int32)
        m._expert_map[local.start : local.stop] = torch.arange(len(local), dtype=torch.int32)
        n = len(local)
        to_local = lambda g: int(m._expert_map[g])  # noqa: E731
    m.expert_map_manager = types.SimpleNamespace(map_global_to_local=to_local)
    m.w13_weight = torch.nn.Parameter(torch.empty(n, 2 * f, DIM, dtype=DT), requires_grad=False)
    m.w2_weight = torch.nn.Parameter(torch.empty(n, DIM, f, dtype=DT), requires_grad=False)
    for p in (m.w13_weight, m.w2_weight):
        p.weight_loader = m.weight_loader
    return m


def _model(shared_tp: bool, local: range | None, moe_tp=(0, 1)):
    from vllm.model_executor.models.qwen3_5 import Qwen3_5Model

    top = tpt._vllm_text_model(shared_tp)
    top.model.load_fused_expert_weights = types.MethodType(
        Qwen3_5Model.load_fused_expert_weights, top.model
    )
    for layer in top.model.layers:
        layer.mlp.experts = _fused_moe(local, moe_tp)
    for p in top.parameters():
        p.data.fill_(float("nan"))
    return top


def _loader(top):
    from vllm.model_executor.models.qwen3_5 import Qwen3_5Model

    def load(items):
        inner = []
        for name, value in items:
            if name == "lm_head.weight":
                top.lm_head.weight_loader(top.lm_head.weight, value)
            else:
                inner.append((name.removeprefix("model.language_model."), value))
        Qwen3_5Model.load_weights(top.model, inner)

    return load


def _params(top) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in top.state_dict().items()}


def _adapter():
    from torchtitan.models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter

    return Qwen35StateDictAdapter(types.SimpleNamespace(), None)


def _assert_equal(loaded, reference):
    assert loaded.keys() == reference.keys()
    for name in reference:
        assert not reference[name].isnan().any(), name
        assert torch.equal(loaded[name], reference[name]), name


@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("ep", [(0, 4), (4, 8), None])
def test_fast_apply_bitwise_equal_with_partial_pulls(monkeypatch, tp_rank, ep):
    tpt._fake_tp(monkeypatch, tp_rank, tpt.TP)
    torch.set_default_dtype(DT)
    try:
        shared_tp = True
        local = None if ep is None else range(*ep)
        adapter = _adapter()
        top_fast = _model(shared_tp, local)
        top_default = _model(shared_tp, local)
        if ep is not None:
            assert local_expert_range(top_fast) == (ep[0], ep[1], E)
        rules = tpt.tpp.tp_rules_from_vllm_model(top_fast)
        tp = NativeTPPartialPull(rules)
        epp = (
            None
            if ep is None
            else NativeEPPartialPull(*ep, E, dims=(DIM, FE, DT), reuse_buffers=True)
        )
        fast = NativeFastApply(top_fast, ep=epp, tp=tp)
        assert len(fast._experts) == len(tpt.LAYER_TYPES)
        for version in range(3):  # new weights every sync; buffers are reused
            sd = _state_dict(version)
            client = tpt._FakeClient(sd, "model_state_dict")
            monkeypatch.setattr(
                "torchstore.client", lambda c=client: tpt._async(c), raising=False
            )
            full = tpt._full_fetch(client, list(sd))
            # ground truth: the whole HF state dict through vLLM's loaders
            top_ref = _model(shared_tp, local)
            _loader(top_ref)(_to_hf_items(adapter, full))
            reference = _params(top_ref)
            if version == 0:
                # first sync: full fetch (records the TP shapes)
                staged = full
                tp.record(full)
            else:
                for buf in [*tp._full.values(), *fast._fused_buffers.values()]:
                    buf.fill_(float("nan"))  # rows never fetched stay NaN
                staged = asyncio.run(tp.fetch("model_state_dict", ep=epp))
            # default path, as the generator runs it today
            hf = adapter.to_hf(dict(staged))
            if epp is not None and version > 0:
                hf = epp.embed_hf(hf)
            _loader(top_default)(hf.items())
            _assert_equal(_params(top_default), reference)
            # fast path
            fast.apply(adapter, staged, _loader(top_fast))
            _assert_equal(_params(top_fast), reference)
    finally:
        torch.set_default_dtype(torch.float32)


def _to_hf_items(adapter, sd):
    return adapter.to_hf(dict(sd)).items()


@pytest.mark.parametrize("moe_tp_rank", [0, 1])
def test_fast_apply_moe_tensor_parallel(monkeypatch, moe_tp_rank):
    """Experts TP-split on the intermediate dim (no EP): the slab load narrows
    the same rows the per-expert loader does."""
    tpt._fake_tp(monkeypatch, moe_tp_rank, tpt.TP)
    torch.set_default_dtype(DT)
    try:
        adapter = _adapter()
        sd = _state_dict(7)
        top_ref = _model(False, None, (moe_tp_rank, 2))
        _loader(top_ref)(_to_hf_items(adapter, sd))
        top_fast = _model(False, None, (moe_tp_rank, 2))
        NativeFastApply(top_fast).apply(adapter, sd, _loader(top_fast))
        _assert_equal(_params(top_fast), _params(top_ref))
    finally:
        torch.set_default_dtype(torch.float32)


def test_non_contiguous_placement_keeps_per_expert_loader(monkeypatch):
    """A round-robin expert map is not a slab: those layers fall back to the
    default HF path and still load the same values."""
    tpt._fake_tp(monkeypatch, 0, tpt.TP)
    torch.set_default_dtype(DT)
    try:
        adapter = _adapter()
        sd = _state_dict(3)

        def model():
            top = _model(False, range(0, 4))
            for layer in top.model.layers:
                moe = layer.mlp.experts
                moe._expert_map = torch.tensor([0, -1, 1, -1, 2, -1, 3, -1], dtype=torch.int32)
            return top

        top_ref = model()
        _loader(top_ref)(_to_hf_items(adapter, sd))
        top_fast = model()
        fast = NativeFastApply(top_fast)
        assert not fast._experts
        fast.apply(adapter, sd, _loader(top_fast))
        _assert_equal(_params(top_fast), _params(top_ref))
    finally:
        torch.set_default_dtype(torch.float32)


def test_reused_expert_buffers_keep_their_storage():
    ep = NativeEPPartialPull(2, 6, E, dims=(DIM, FE, DT), reuse_buffers=True)
    ep.prepare(["layers.0.moe.experts.w1_EFD"])
    a = ep.expert_request("k", "layers.0.moe.experts.w1_EFD")[1]
    b = ep.expert_request("k", "layers.0.moe.experts.w1_EFD")[1]
    assert a.data_ptr() == b.data_ptr() and a.shape == (4, FE, DIM)
    fresh = NativeEPPartialPull(2, 6, E, dims=(DIM, FE, DT))
    fresh.prepare(["layers.0.moe.experts.w1_EFD"])
    c = fresh.expert_request("k", "layers.0.moe.experts.w1_EFD")[1]
    d = fresh.expert_request("k", "layers.0.moe.experts.w1_EFD")[1]
    assert c.data_ptr() != d.data_ptr()
