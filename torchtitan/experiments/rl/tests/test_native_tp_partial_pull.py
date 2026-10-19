# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TP-sliced weight pull for vllm_native generators.

The end-to-end test stores a small Qwen3.5 MoE state dict the way an FSDP x TP
trainer puts it (one TorchStore volume per trainer rank), fetches it through
torchstore's own request expansion / assembly code, converts it with the real
adapter and loads it with vLLM's real ``Qwen3_5Model.load_weights`` and layer
weight loaders under a TP rank. The partial pull must load bitwise the same
parameters as the full pull while the never-fetched rows hold NaN.
"""

import asyncio
import math
import types

import pytest
import torch

from torchtitan.experiments.rl import native_tp_partial_pull as tpp
from torchtitan.experiments.rl.native_ep_partial_pull import NativeEPPartialPull
from torchtitan.experiments.rl.native_tp_partial_pull import (
    NativeTPPartialPull,
    TPSliceRules,
)

vllm = pytest.importorskip("vllm")
pytest.importorskip("torchstore")
from torchstore.client import LocalClient  # noqa: E402

DIM, HEAD, N_HEADS, N_KV = 64, 16, 4, 2
NK, NV, KHD, VHD, CONV = 4, 8, 8, 8, 4
KD, VD = NK * KHD, NV * VHD
SHARED_F, E, VOCAB = 32, 8, 1000  # vocab 1000 pads to 1024: uneven vocab split
LAYER_TYPES = ["linear_attention", "linear_attention", "full_attention"]
FSDP, TP = 4, 2
DT = torch.bfloat16


def _tt_state_dict(seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)

    def r(*shape):
        return torch.randn(*shape, generator=g).to(DT)

    sd = {"tok_embeddings.weight": r(VOCAB, DIM), "norm.weight": r(DIM)}
    for i, kind in enumerate(LAYER_TYPES):
        p = f"layers.{i}."
        if kind == "linear_attention":
            for name, rows in (("q", KD), ("k", KD), ("v", VD), ("z", VD)):
                sd[f"{p}attn.in_proj_{name}.weight"] = r(rows, DIM)
            sd[f"{p}attn.in_proj_a.weight"] = r(NV, DIM)
            sd[f"{p}attn.in_proj_b.weight"] = r(NV, DIM)
            for name, rows in (("q", KD), ("k", KD), ("v", VD)):
                sd[f"{p}attn.conv_{name}.weight"] = r(rows, 1, CONV)
            sd[f"{p}attn.A_log"] = r(NV)
            sd[f"{p}attn.dt_bias"] = r(NV)
            sd[f"{p}attn.norm.weight"] = r(VHD)
            sd[f"{p}attn.out_proj.weight"] = r(DIM, VD)
        else:
            sd[f"{p}attn.wq.weight"] = r(N_HEADS * HEAD * 2, DIM)  # q + output gate
            sd[f"{p}attn.wk.weight"] = r(N_KV * HEAD, DIM)
            sd[f"{p}attn.wv.weight"] = r(N_KV * HEAD, DIM)
            sd[f"{p}attn.wo.weight"] = r(DIM, N_HEADS * HEAD)
            sd[f"{p}attn.q_norm.weight"] = r(HEAD)
            sd[f"{p}attn.k_norm.weight"] = r(HEAD)
        sd[f"{p}attention_norm.weight"] = r(DIM)
        sd[f"{p}ffn_norm.weight"] = r(DIM)
        sd[f"{p}moe.router.gate.weight"] = r(E, DIM)
        sd[f"{p}moe.shared_experts.w1.weight"] = r(SHARED_F, DIM)
        sd[f"{p}moe.shared_experts.w3.weight"] = r(SHARED_F, DIM)
        sd[f"{p}moe.shared_experts.w2.weight"] = r(DIM, SHARED_F)
        sd[f"{p}moe.shared_experts.gate.weight"] = r(1, DIM)
    sd["lm_head.weight"] = r(VOCAB, DIM)
    return sd


# --- an FSDP x TP trainer's TorchStore layout, read through torchstore's client ---

_ROWWISE_TT = ("attn.wo.", "attn.out_proj.", "shared_experts.w2.")
_REPLICATED_TT = ("norm.weight", "router.gate", "shared_experts.gate", "q_norm", "k_norm")


def _chunks(n: int, k: int) -> list[tuple[int, int]]:
    c = math.ceil(n / k)
    return [(i * c, min(c, n - i * c)) for i in range(k) if n - i * c > 0]


def _trainer_slices(key: str, shape: tuple[int, ...]):
    """``[(volume, offsets, local_shape)]`` as an FSDP-4 x TP-2 trainer stores
    ``key``: colwise = TP halves of dim 0 each chunked by FSDP, rowwise = FSDP
    rows x TP columns, replicated = FSDP rows stored by both TP ranks."""
    rest = list(shape[1:])
    out = []
    if any(s in key for s in _ROWWISE_TT):
        cols = shape[1] // TP
        for d, (r0, n) in enumerate(_chunks(shape[0], FSDP)):
            for t in range(TP):
                out.append((f"v{d}{t}", (r0, t * cols), (n, cols)))
    elif any(s in key for s in _REPLICATED_TT):
        for d, (r0, n) in enumerate(_chunks(shape[0], FSDP)):
            for t in range(TP):
                out.append((f"v{d}{t}", (r0, *[0] * len(rest)), (n, *rest)))
    else:
        half = shape[0] // TP
        for t in range(TP):
            for d, (r0, n) in enumerate(_chunks(half, FSDP)):
                out.append((f"v{d}{t}", (t * half + r0, *[0] * len(rest)), (n, *rest)))
    return out


class _FakeClient:
    """Minimal TorchStore client over in-memory volumes, reusing LocalClient's
    request expansion, in-place destination views and assembly."""

    def __init__(self, state_dict: dict[str, torch.Tensor], prefix: str):
        from torch.distributed.checkpoint._nested_dict import flatten_state_dict
        from torchstore.controller import ObjectType, StorageInfo
        from torchstore.transport.types import TensorSlice

        self.prefix = prefix
        _, self.mapping = flatten_state_dict(state_dict)
        self.volumes: dict[str, dict] = {}  # key -> {vid: StorageInfo}
        self.data: dict[tuple, torch.Tensor] = {}
        for fk, value in state_dict.items():
            vmap = {}
            for vid, offsets, local_shape in _trainer_slices(fk, tuple(value.shape)):
                ts_slice = TensorSlice(
                    offsets, (vid,), tuple(value.shape), local_shape, (FSDP, TP)
                )
                info = vmap.setdefault(vid, StorageInfo(ObjectType.TENSOR_SLICE))
                info.tensor_slices.add(ts_slice)
                region = value
                for dim, (o, n) in enumerate(zip(offsets, local_shape)):
                    region = region.narrow(dim, o, n)
                self.data[(f"{prefix}/{fk}", vid, offsets)] = region.clone()
            self.volumes[f"{prefix}/{fk}"] = vmap
        self.num_sub_requests = 0
        self.bytes_read = 0

    async def get(self, key):
        assert key == f"{self.prefix}/MAPPING"
        return self.mapping

    async def _fetch(self, requests):
        volume_maps = {r.key: self.volumes[r.key] for r in requests}
        buffers = {
            vid: types.SimpleNamespace(supports_inplace_resharding=True)
            for vm in volume_maps.values()
            for vid in vm
        }
        volume_requests, whole = LocalClient._build_volume_requests(
            self, requests, volume_maps, buffers
        )
        pairs = []
        for vid, subs in volume_requests.items():
            for sub in subs:
                stored = next(
                    (k, s)
                    for s in self.volumes[sub.key][vid].tensor_slices
                    for k in [(sub.key, vid, s.offsets)]
                    if all(
                        so <= o and o + n <= so + sn
                        for so, sn, o, n in zip(
                            s.offsets,
                            s.local_shape,
                            sub.tensor_slice.offsets,
                            sub.tensor_slice.local_shape,
                        )
                    )
                )
                src = self.data[stored[0]]
                for dim, (o, n, so) in enumerate(
                    zip(
                        sub.tensor_slice.offsets,
                        sub.tensor_slice.local_shape,
                        stored[1].offsets,
                    )
                ):
                    src = src.narrow(dim, o - so, n)
                self.num_sub_requests += 1
                self.bytes_read += src.numel() * src.element_size()
                if sub.tensor_val is not None:  # RDMA writes into the view
                    sub.tensor_val.copy_(src)
                    pairs.append((sub, sub.tensor_val))
                else:
                    pairs.append((sub, src.clone()))
        return LocalClient._assemble_results(self, requests, pairs, whole)

    _expand_tensor_slices = LocalClient._expand_tensor_slices

    def _apply_inplace(self, fetched, inplace, request):
        return LocalClient._apply_inplace(self, fetched, inplace, request)


def _full_fetch(client: _FakeClient, flat_keys: list[str]) -> dict[str, torch.Tensor]:
    from torchstore.transport.types import Request

    requests = [Request.from_any(f"{client.prefix}/{fk}", None) for fk in flat_keys]
    results = asyncio.run(client._fetch(requests))
    return {fk: results[r.key] for fk, r in zip(flat_keys, requests)}


# --- vLLM side: real layers + real Qwen3_5Model.load_weights under a TP rank ---


def _fake_tp(monkeypatch, rank: int, size: int) -> None:
    from vllm.distributed import parallel_state as ps

    monkeypatch.setattr(
        ps, "_TP", types.SimpleNamespace(world_size=size, rank_in_group=rank, rank=rank)
    )


def _vllm_text_model(shared_tp: bool):
    """The parameter tree vLLM's Qwen3.5 MoE text model builds (same layer
    classes, names and weight loaders), without the attention / MoE kernels."""
    from torch import nn
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNormGated
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        MergedColumnParallelLinear,
        QKVParallelLinear,
        ReplicatedLinear,
        RowParallelLinear,
    )
    from vllm.model_executor.layers.mamba.mamba_mixer2 import (
        mamba_v2_sharded_weight_loader,
    )
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )
    from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP
    from vllm.model_executor.utils import set_weight_attrs
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()

    def gdn():
        m = nn.Module()
        m.conv1d = ColumnParallelLinear(CONV, 2 * KD + VD, bias=False)
        m.conv1d.weight.data = m.conv1d.weight.data.unsqueeze(1)
        m.in_proj_qkvz = MergedColumnParallelLinear(DIM, [KD, KD, VD, VD], bias=False)
        m.in_proj_ba = MergedColumnParallelLinear(DIM, [NV, NV], bias=False)
        m.conv1d.weight.weight_loader = mamba_v2_sharded_weight_loader(
            [(KD, 0, False), (KD, 0, False), (VD, 0, False)], tp_size, tp_rank
        )
        m.dt_bias = nn.Parameter(torch.ones(NV // tp_size))
        m.A_log = nn.Parameter(torch.empty(NV // tp_size, dtype=torch.float32))
        set_weight_attrs(m.A_log, {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(m.dt_bias, {"weight_loader": sharded_weight_loader(0)})
        m.norm = RMSNormGated(VHD, norm_before_gate=True, device="cpu")
        m.out_proj = RowParallelLinear(VD, DIM, bias=False, input_is_parallel=True)
        return m

    def attn():
        m = nn.Module()
        m.qkv_proj = QKVParallelLinear(DIM, HEAD, N_HEADS * 2, N_KV, bias=False)
        m.o_proj = RowParallelLinear(N_HEADS * HEAD, DIM, bias=False)
        m.q_norm = GemmaRMSNorm(HEAD)
        m.k_norm = GemmaRMSNorm(HEAD)
        return m

    def layer(kind):
        m = nn.Module()
        if kind == "linear_attention":
            m.linear_attn = gdn()
        else:
            m.self_attn = attn()
        mlp = nn.Module()
        mlp.gate = ReplicatedLinear(DIM, E, bias=False)
        mlp.shared_expert_gate = ReplicatedLinear(DIM, 1, bias=False)
        mlp.shared_expert = Qwen2MoeMLP(
            DIM, SHARED_F, "silu", reduce_results=False, is_sequence_parallel=not shared_tp
        )
        m.mlp = mlp
        m.input_layernorm = GemmaRMSNorm(DIM)
        m.post_attention_layernorm = GemmaRMSNorm(DIM)
        return m

    from vllm.config import set_current_vllm_config, VllmConfig

    model = nn.Module()
    with set_current_vllm_config(VllmConfig()):
        model.embed_tokens = VocabParallelEmbedding(VOCAB, DIM)
        model.layers = nn.ModuleList([layer(k) for k in LAYER_TYPES])
        model.norm = GemmaRMSNorm(DIM)
        lm_head = ParallelLMHead(VOCAB, DIM)
    model.config = types.SimpleNamespace(num_experts=E)
    model.get_expert_mapping = lambda: []
    top = nn.Module()
    top.model = model
    top.lm_head = lm_head
    for p in top.parameters():
        p.data.fill_(float("nan"))  # every loaded value must be overwritten
    return top


def _load(top, hf_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    from vllm.model_executor.models.qwen3_5 import Qwen3_5Model

    inner, lm_head = [], None
    for name, value in hf_state_dict.items():
        if name == "lm_head.weight":
            lm_head = value
        else:
            inner.append((name.removeprefix("model.language_model."), value))
    Qwen3_5Model.load_weights(top.model, inner)
    top.lm_head.weight_loader(top.lm_head.weight, lm_head)
    return {k: v.detach().clone() for k, v in top.state_dict().items()}


def _to_hf(tt: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    from torchtitan.models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter

    adapter = Qwen35StateDictAdapter(types.SimpleNamespace(), None)
    return adapter.to_hf(tt)


@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("shared_tp", [False, True])
def test_partial_pull_loads_bitwise_equal_params(monkeypatch, tp_rank, shared_tp):
    _fake_tp(monkeypatch, tp_rank, TP)
    torch.set_default_dtype(DT)
    try:
        sd = _tt_state_dict()
        client = _FakeClient(sd, "model_state_dict")
        monkeypatch.setattr("torchstore.client", lambda: _async(client), raising=False)
        flat_keys = list(sd)

        full = _full_fetch(client, flat_keys)
        assert all(torch.equal(full[k], sd[k]) for k in sd)
        full_requests, full_bytes = client.num_sub_requests, client.bytes_read
        reference = _load(_vllm_text_model(shared_tp), _to_hf(full))

        rules = tpp.tp_rules_from_vllm_model(_vllm_text_model(shared_tp))
        assert rules.tp_rank == tp_rank and rules.tp_size == TP
        assert rules.num_kv_heads == N_KV
        assert rules.split_shared_experts == shared_tp
        start = 0 if tp_rank == 0 else 512
        assert rules.vocab_ranges == {
            "tok_embeddings.weight": (start, 512 if tp_rank == 0 else VOCAB),
            "lm_head.weight": (start, 512 if tp_rank == 0 else VOCAB),
        }
        pull = NativeTPPartialPull(rules)
        pull.record(full)
        for _ in range(2):  # the second pull reuses the full-size buffers
            for buf in pull._full.values():
                buf.fill_(float("nan"))  # never-fetched rows stay NaN
            client.num_sub_requests = client.bytes_read = 0
            partial = asyncio.run(pull.fetch("model_state_dict"))
            loaded = _load(_vllm_text_model(shared_tp), _to_hf(partial))
            assert loaded.keys() == reference.keys()
            for name in reference:
                assert not reference[name].isnan().any(), name
                assert torch.equal(loaded[name], reference[name]), name
        assert client.num_sub_requests < full_requests
        assert client.bytes_read < 0.75 * full_bytes
        # the fake client also counts the duplicate reads of TP-replicated keys
        assert 0 < pull.last_fetch_bytes <= client.bytes_read
    finally:
        torch.set_default_dtype(torch.float32)


async def _async(value):
    return value


def test_combined_with_ep_pull_fetches_local_experts(monkeypatch):
    """The TP fetch also routes stacked expert keys through the EP slice."""
    sd = _tt_state_dict()
    experts = torch.randn(E, SHARED_F, DIM).to(DT)
    sd["layers.0.moe.experts.w1_EFD"] = experts
    client = _FakeClient(sd, "model_state_dict")
    monkeypatch.setattr("torchstore.client", lambda: _async(client), raising=False)
    full = _full_fetch(client, list(sd))
    rules = TPSliceRules(tp_rank=1, tp_size=TP, num_kv_heads=N_KV)
    pull = NativeTPPartialPull(rules)
    pull.record(full)
    ep = NativeEPPartialPull(2, 6, E, dims=(DIM, SHARED_F, DT))
    out = asyncio.run(pull.fetch("model_state_dict", ep=ep))
    assert ep.last_fetch_partial
    assert torch.equal(out["layers.0.moe.experts.w1_EFD"], experts[2:6])
    q = out["layers.0.attn.in_proj_q.weight"]
    assert torch.equal(q[KD // 2 :], sd["layers.0.attn.in_proj_q.weight"][KD // 2 :])
    assert torch.equal(out["norm.weight"], sd["norm.weight"])  # replicated: whole


def test_rules_on_122b_shapes():
    rules = TPSliceRules(
        tp_rank=1,
        tp_size=2,
        num_kv_heads=2,
        vocab_ranges={"tok_embeddings.weight": (124160, 248320)},
    )
    s = rules.slice_for
    assert s("layers.3.attn.wq.weight", (16384, 3072)) == (0, 8192, 16384)
    assert s("layers.3.attn.wk.weight", (512, 3072)) == (0, 256, 512)
    assert s("layers.3.attn.wo.weight", (3072, 8192)) == (1, 4096, 8192)
    assert s("layers.0.attn.in_proj_v.weight", (8192, 3072)) == (0, 4096, 8192)
    assert s("layers.0.attn.in_proj_a.weight", (64, 3072)) == (0, 32, 64)
    assert s("layers.0.attn.conv_k.weight", (2048, 1, 4)) == (0, 1024, 2048)
    assert s("layers.0.attn.A_log", (64,)) == (0, 32, 64)
    assert s("layers.0.attn.out_proj.weight", (3072, 8192)) == (1, 4096, 8192)
    assert s("tok_embeddings.weight", (248320, 3072)) == (0, 124160, 248320)
    for whole in (
        "layers.0.attn.norm.weight",
        "layers.3.attn.q_norm.weight",
        "layers.0.attention_norm.weight",
        "layers.0.moe.router.gate.weight",
        "layers.0.moe.shared_experts.w1.weight",  # replicated under SP MoE
        "layers.0.moe.experts.w1_EFD",
        "vision_encoder.layers.0.attn.wq.weight",
        "lm_head.weight",  # no vocab range given
    ):
        assert s(whole, (64, 64)) is None, whole


def test_kv_heads_replicated_when_tp_exceeds_them():
    """vLLM: shard_rank = tp_rank // (tp / num_kv_heads)."""
    for rank, expect in ((0, (0, 0, 16)), (1, (0, 0, 16)), (2, (0, 16, 32)), (3, (0, 16, 32))):
        rules = TPSliceRules(tp_rank=rank, tp_size=4, num_kv_heads=2)
        assert rules.slice_for("layers.3.attn.wk.weight", (32, 64)) == expect
    assert TPSliceRules(0, 4, 2).slice_for("layers.3.attn.wq.weight", (6, 64)) is None


def test_generator_first_pull_full_then_tp_partial_with_ep(monkeypatch):
    """Generator wiring: the first native pull is the EP pull and records the TP
    shapes; later pulls go through the TP fetch with the EP puller attached and
    report only the fetched bytes."""
    from torchtitan.experiments.rl.actors.generator import VLLMGenerator

    calls = []
    tt = {"layers.0.attn.in_proj_q.weight": torch.zeros(KD, DIM, dtype=DT)}

    class _EP:
        ready = True

        async def fetch(self, key):
            calls.append("ep")
            return dict(tt)

        def record(self, sd):
            raise AssertionError("EP pull already ready")

    ep = _EP()
    tp = NativeTPPartialPull(TPSliceRules(tp_rank=0, tp_size=TP, num_kv_heads=N_KV))

    async def tp_fetch(key, ep=None):
        calls.append(("tp", ep))
        tp.last_fetch_bytes = 3 * 2**30
        return dict(tt)

    monkeypatch.setattr(tp, "fetch", tp_fetch)
    gen = object.__new__(VLLMGenerator)
    gen._backend = "vllm_native"
    gen._rank = 0
    gen._ep_partial_pull = ep
    gen._tp_partial_pull = tp
    _, _, gib = asyncio.run(gen._fetch_model_state_dict(1, overlap_safe=False))
    assert calls == ["ep"] and tp.ready
    assert gib == pytest.approx(KD * DIM * 2 / 2**30)
    _, _, gib = asyncio.run(gen._fetch_model_state_dict(2, overlap_safe=False))
    assert calls[1] == ("tp", ep) and gib == 3
