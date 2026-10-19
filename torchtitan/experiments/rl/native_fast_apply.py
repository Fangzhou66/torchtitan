# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cheaper apply step of a vllm_native weight sync (SWE_NATIVE_FAST_APPLY=1).

The default apply converts the whole fetched state dict to HF and hands it to
``load_weights``. For a Qwen3.5 MoE that does three host copies the vLLM
parameters never needed: ``to_hf`` concatenates each layer's w1/w3 experts into
``gate_up_proj`` (a fresh tensor, so every page faults), the EP partial pull
copies the local experts again into a full-size tensor, and vLLM's fused-expert
loader immediately chunks ``gate_up_proj`` back into w1/w3 and walks all global
expert ids one at a time. On a 122B-A10B EP-8 rank that is ~48 GB of host
copies per sync.

Here the stacked expert tensors skip ``to_hf``: each local w1 / w3 / w2 slab
goes through the layer's own FusedMoE ``weight_loader`` once (its 3-D
"full load" path), which copies it into ``w13_weight`` / ``w2_weight`` exactly
where the per-expert loop would. The GatedDeltaNet q/k/v and conv fusions are
written into reused buffers, copying only the rows a TP rank loads. Everything
else goes through ``to_hf`` + ``load_weights`` unchanged, so loaded parameters
are bitwise those of the default path.
"""

import re
from collections.abc import Callable, Iterable
from typing import Any

import torch

from torchtitan.tools.logging import logger

_TT_EXPERT = re.compile(r"^layers\.(\d+)\.moe\.experts\.(w1_EFD|w3_EFD|w2_EDF)$")
_VLLM_EXPERTS = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.experts$")
_GDN_PART = re.compile(
    r"^layers\.(\d+)\.attn\.(in_proj_q|in_proj_k|in_proj_v|conv_q|conv_k|conv_v)\.weight$"
)
_GDN_FUSED = {
    "in_proj": (
        ("in_proj_q", "in_proj_k", "in_proj_v"),
        "model.language_model.layers.{}.linear_attn.in_proj_qkv.weight",
    ),
    "conv": (
        ("conv_q", "conv_k", "conv_v"),
        "model.language_model.layers.{}.linear_attn.conv1d.weight",
    ),
}
# torchtitan shard -> (vLLM param, FusedMoE shard id)
_SHARD = {"w1_EFD": ("w13_weight", "w1"), "w3_EFD": ("w13_weight", "w3"), "w2_EDF": ("w2_weight", "w2")}


def _batched_load_ok(module: torch.nn.Module) -> str | None:
    """Why ``module`` cannot take a whole-slab expert load, or None if it can."""
    quant = type(getattr(module, "quant_method", None)).__name__
    if not quant.startswith("Unquantized") or getattr(module, "quant_config", None):
        return f"quant method {quant}"
    for name in ("w13_weight", "w2_weight"):
        param = getattr(module, name, None)
        if not isinstance(param, torch.nn.Parameter) or param.dim() != 3:
            return f"no 3-D {name}"
        if getattr(param, "is_transposed", False) or getattr(
            param, "use_bitsandbytes_4bit", False
        ):
            return f"{name} has a special layout"
    if getattr(module, "w13_bias", None) is not None:
        return "expert biases"
    return None


class NativeFastApply:
    """Apply a fetched torchtitan state dict to a native vLLM Qwen3.5 model."""

    def __init__(self, model: torch.nn.Module, ep: Any = None, tp: Any = None) -> None:
        self._ep = ep
        self._tp = tp
        # layer index -> (FusedMoE module, first local global expert id, local count)
        self._experts: dict[int, tuple[torch.nn.Module, int, int]] = {}
        self._fused_buffers: dict[str, torch.Tensor] = {}
        reasons = set()
        for name, module in model.named_modules():
            m = _VLLM_EXPERTS.search(name)
            if m is None or "mtp" in name or not hasattr(
                module, "_map_global_expert_id_to_local_expert_id"
            ):
                continue
            reason = _batched_load_ok(module) or self._local_order(module)
            if isinstance(reason, str):
                reasons.add(reason)
                continue
            self._experts[int(m.group(1))] = (module, *reason)
        if reasons:
            logger.warning(
                f"[fast-apply] {len(reasons)} expert layout(s) keep the per-expert "
                f"loader: {sorted(reasons)}"
            )
        logger.info(f"[fast-apply] whole-slab expert load for {len(self._experts)} layers")

    @staticmethod
    def _local_order(module: torch.nn.Module) -> tuple[int, int] | str:
        """``(first global id, count)`` when local expert ``i`` is global expert
        ``first + i`` for every local slot, else the reason it is not."""
        num = int(module.global_num_experts)
        local = int(module.w13_weight.shape[0])
        to_local = module._map_global_expert_id_to_local_expert_id
        ids = [to_local(g) for g in range(num)]
        held = [g for g, i in enumerate(ids) if i >= 0]
        if len(held) != local or not held:
            return f"{len(held)} mapped experts for {local} local slots"
        first = held[0]
        if any(ids[g] != g - first for g in held) or held[-1] != first + local - 1:
            return "non-contiguous expert placement"
        return first, local

    def _load_experts(self, staged: dict[str, Any]) -> dict[str, Any]:
        """Load the stacked expert tensors of every eligible layer; return the
        rest of ``staged`` for the HF path."""
        rest = {}
        for key, value in staged.items():
            m = _TT_EXPERT.match(key)
            entry = self._experts.get(int(m.group(1))) if m else None
            if entry is None or not isinstance(value, torch.Tensor):
                rest[key] = value
                continue
            module, first, local = entry
            if value.shape[0] == local:
                slab = value  # the EP partial pull fetched only the local experts
            elif value.shape[0] == module.global_num_experts:
                slab = value[first : first + local]
            else:
                raise ValueError(
                    f"[fast-apply] {key} has {value.shape[0]} experts; the vLLM layer "
                    f"holds {local} of {module.global_num_experts}"
                )
            param_name, shard_id = _SHARD[m.group(2)]
            param = getattr(module, param_name)
            ok = module.weight_loader(
                param,
                slab,
                f"layers.{m.group(1)}.mlp.experts.{param_name}",
                shard_id=shard_id,
                expert_id=first,
                return_success=True,
            )
            if not ok:
                raise RuntimeError(f"[fast-apply] FusedMoE refused the {key} slab")
        return rest

    def _rows(self, key: str, n: int) -> tuple[int, int]:
        """Rows of tt ``key`` this rank's vLLM loader reads (all, without TP)."""
        meta = self._tp._meta.get(key) if self._tp is not None else None
        if meta is None or meta[2] != 0:
            return 0, n
        return meta[3], meta[4]

    def _fuse_gdn(self, staged: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Write each GatedDeltaNet q/k/v (and conv q/k/v) triple into a reused
        HF-layout buffer. Rows outside this TP rank's slice are never written or
        read. Returns ``(rest of staged, fused HF tensors)``."""
        parts: dict[str, dict[str, torch.Tensor]] = {}
        rest = {}
        for key, value in staged.items():
            m = _GDN_PART.match(key)
            if m is None or not isinstance(value, torch.Tensor):
                rest[key] = value
                continue
            parts.setdefault(m.group(1), {})[m.group(2)] = value
        fused = {}
        for layer, got in parts.items():
            for names, hf_template in _GDN_FUSED.values():
                pieces = [got.get(n) for n in names]
                if any(p is None for p in pieces):
                    # Incomplete triple: leave it to to_hf, which drops it as well.
                    for n, p in zip(names, pieces):
                        if p is not None:
                            rest[f"layers.{layer}.attn.{n}.weight"] = p
                    continue
                hf_key = hf_template.format(layer)
                shape = (sum(p.shape[0] for p in pieces), *pieces[0].shape[1:])
                buf = self._fused_buffers.get(hf_key)
                if buf is None or tuple(buf.shape) != shape or buf.dtype != pieces[0].dtype:
                    buf = torch.empty(shape, dtype=pieces[0].dtype)
                    self._fused_buffers[hf_key] = buf
                offset = 0
                for n, p in zip(names, pieces):
                    a, b = self._rows(f"layers.{layer}.attn.{n}.weight", p.shape[0])
                    buf[offset + a : offset + b].copy_(p[a:b])
                    offset += p.shape[0]
                fused[hf_key] = buf
        return rest, fused

    def apply(
        self,
        adapter: Any,
        staged: dict[str, Any],
        load_weights: Callable[[Iterable[tuple[str, torch.Tensor]]], Any],
    ) -> tuple[float, float, float]:
        """Load ``staged`` into the model. Returns ``(to_hf, load_weights,
        experts)`` seconds."""
        import time

        t0 = time.perf_counter()
        rest, fused = self._fuse_gdn(staged)
        experts = {k: v for k, v in rest.items() if _TT_EXPERT.match(k)}
        others = {k: v for k, v in rest.items() if k not in experts}
        hf_state_dict = adapter.to_hf(others)
        hf_state_dict.update(fused)
        t1 = time.perf_counter()
        load_weights(hf_state_dict.items())
        t2 = time.perf_counter()
        leftover = self._load_experts(experts)
        if leftover:
            # Layers whose FusedMoE layout keeps the per-expert loader.
            hf_left = adapter.to_hf(leftover)
            if self._ep is not None and self._ep.ready:
                hf_left = self._ep.embed_hf(hf_left)
            load_weights(hf_left.items())
        t3 = time.perf_counter()
        return t1 - t0, t2 - t1, t3 - t2
