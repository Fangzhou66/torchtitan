# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pull only this rank's experts on a vllm_native expert-parallel generator.

With EP the native generator holds ``E / ep`` experts per rank, but the default
pull fetches the whole torchtitan state dict (all ``E`` experts, ~93% of a
35B-A3B checkpoint), converts it to HF, and lets ``load_weights`` drop the
non-local experts. This module requests only the local expert slice of each
stacked expert tensor from TorchStore and converts only that slice.

vLLM's fused-expert loader indexes the stacked HF tensor by global expert id and
returns before reading a non-local expert, so the local slice is embedded into a
full-size, never-written CPU tensor (its pages stay unallocated) at the global
offset; ``load_weights`` runs unchanged.
"""

import re
from typing import Any

import torch

from torchtitan.tools.logging import logger

_TT_EXPERT_KEY = re.compile(r"\.moe\.experts\.(w1_EFD|w3_EFD|w2_EDF)$")
_HF_EXPERT_KEY = re.compile(r"\.mlp\.experts\.(gate_up_proj|down_proj)$")


def local_expert_range(model: torch.nn.Module) -> tuple[int, int, int] | None:
    """``(start, stop, num_experts)`` of the experts this rank holds, or None.

    Read from the first FusedMoE layer's ``expert_map`` (global -> local id, -1
    elsewhere). None when there is no expert parallelism or the local experts are
    not one contiguous range.
    """
    for module in model.modules():
        expert_map = getattr(module, "expert_map", None)
        num_experts = getattr(module, "global_num_experts", None)
        if num_experts is None or not hasattr(
            module, "_map_global_expert_id_to_local_expert_id"
        ):
            continue
        if expert_map is None:
            return None
        local = (expert_map >= 0).nonzero().flatten().tolist()
        if not local or local != list(range(local[0], local[-1] + 1)):
            return None
        if len(local) == num_experts:
            return None
        return local[0], local[-1] + 1, int(num_experts)
    return None


def expert_dims(model: torch.nn.Module) -> tuple[int, int, torch.dtype] | None:
    """``(dim, hidden_dim, dtype)`` of the experts, read from the first FusedMoE
    layer, or None when they cannot be read unsharded (TP > 1)."""
    for module in model.modules():
        if not hasattr(module, "_map_global_expert_id_to_local_expert_id"):
            continue
        if getattr(module, "tp_size", 1) != 1:
            return None
        return (
            int(module.hidden_size),
            int(module.intermediate_size_per_partition),
            module.params_dtype,
        )
    return None


class NativeEPPartialPull:
    """Partial weight pull for one vllm_native EP rank.

    Expert shapes come from ``dims`` when given; otherwise the first pull is a full
    fetch and ``record`` learns each expert tensor's global shape and dtype from it.
    """

    def __init__(
        self,
        start: int,
        stop: int,
        num_experts: int,
        dims: tuple[int, int, torch.dtype] | None = None,
    ) -> None:
        self.start, self.stop, self.num_experts = start, stop, num_experts
        # With ``dims`` the expert shapes are known up front, so even the first
        # fetch is partial: a full first pull holds the whole checkpoint once per
        # EP rank in host memory (8 x 245 GB on 122B-A10B, which OOMs the host).
        self._dims = dims
        self._expert_meta: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
        self._full_hf: dict[str, torch.Tensor] = {}
        # Whether the most recent fetch was partial; a full fetch (the first pull,
        # which also calls record) must not be embedded.
        self.last_fetch_partial = False

    @property
    def ready(self) -> bool:
        return bool(self._expert_meta) or self._dims is not None

    def _meta_from_dims(self, flat_keys: list[str]) -> None:
        dim, hidden_dim, dtype = self._dims
        shapes = {
            "w1_EFD": (self.num_experts, hidden_dim, dim),
            "w3_EFD": (self.num_experts, hidden_dim, dim),
            "w2_EDF": (self.num_experts, dim, hidden_dim),
        }
        for fk in flat_keys:
            m = _TT_EXPERT_KEY.search(fk)
            if m:
                self._expert_meta[fk] = (shapes[m.group(1)], dtype)
        logger.info(
            f"[ep-partial-pull] pulling experts [{self.start}, {self.stop}) of "
            f"{self.num_experts} for {len(self._expert_meta)} expert tensors "
            "(shapes from the vLLM config)"
        )

    def record(self, state_dict: dict[str, Any]) -> None:
        """Learn the stacked expert tensors from a full fetch."""
        self.last_fetch_partial = False
        for key, value in state_dict.items():
            if _TT_EXPERT_KEY.search(key) and isinstance(value, torch.Tensor):
                if value.shape[0] != self.num_experts:
                    logger.warning(
                        f"[ep-partial-pull] {key} has {value.shape[0]} experts, "
                        f"expected {self.num_experts}; keeping full pulls"
                    )
                    self._expert_meta.clear()
                    return
                self._expert_meta[key] = (tuple(value.shape), value.dtype)
        if self._expert_meta:
            logger.info(
                f"[ep-partial-pull] pulling experts [{self.start}, {self.stop}) of "
                f"{self.num_experts} for {len(self._expert_meta)} expert tensors"
            )

    def prepare(self, flat_keys: list[str]) -> None:
        """Fill the expert shapes from ``dims`` before the first partial fetch."""
        if not self._expert_meta and self._dims is not None:
            self._meta_from_dims(flat_keys)

    def expert_request(self, store_key: str, flat_key: str):
        """``(request, local buffer)`` for the local expert slice of a stacked
        expert key, or None for any other key."""
        from torchstore.transport.types import Request, TensorSlice

        meta = self._expert_meta.get(flat_key)
        if meta is None:
            return None
        global_shape, dtype = meta
        local_shape = (self.stop - self.start, *global_shape[1:])
        local = torch.empty(local_shape, dtype=dtype)
        spec = TensorSlice(
            offsets=(self.start, *(0 for _ in global_shape[1:])),
            coordinates=None,
            global_shape=global_shape,
            local_shape=local_shape,
            mesh_shape=None,
        )
        return Request.from_any(store_key, local, spec), local

    def build_requests(self, key: str, flat_keys: list[str]):
        """TorchStore requests: whole tensors for dense keys, the local expert
        slice (into a preallocated CPU tensor) for stacked expert keys."""
        from torchstore.transport.types import Request

        delim = "/"
        requests, inplace = [], {}
        for fk in flat_keys:
            store_key = f"{key}{delim}{fk}"
            expert = self.expert_request(store_key, fk)
            if expert is None:
                requests.append(Request.from_any(store_key, None))
                continue
            requests.append(expert[0])
            inplace[store_key] = expert[1]
        return requests, inplace

    async def fetch(self, key: str = "model_state_dict") -> dict[str, Any]:
        """Fetch the state dict with only this rank's experts (``E / ep`` rows of
        each stacked expert tensor, global offset ``start``)."""
        import torchstore as ts
        from torchstore.state_dict_utils import DELIM, MAPPING, unflatten_state_dict

        client = await ts.client()
        mapping = await client.get(f"{key}{DELIM}{MAPPING}")
        flat_keys = list(mapping.keys())
        self.prepare(flat_keys)
        requests, inplace = self.build_requests(key, flat_keys)
        results = await client._fetch(requests)
        flat = {}
        for fk, request in zip(flat_keys, requests):
            value = client._apply_inplace(
                results[request.key], inplace.get(request.key), request
            )
            flat[fk] = value
        self.last_fetch_partial = True
        return unflatten_state_dict(flat, mapping)

    def embed_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Place each local-slice HF expert tensor at its global offset in a
        reused full-size CPU tensor. Rows outside ``[start, stop)`` are never
        written or read (vLLM skips non-local expert ids before touching them).
        A full fetch passes through unchanged."""
        if not self.last_fetch_partial:
            return hf_state_dict
        out = dict(hf_state_dict)
        for key, value in hf_state_dict.items():
            if not _HF_EXPERT_KEY.search(key):
                continue
            full_shape = (self.num_experts, *value.shape[1:])
            full = self._full_hf.get(key)
            if full is None or tuple(full.shape) != full_shape or full.dtype != value.dtype:
                full = torch.empty(full_shape, dtype=value.dtype, device=value.device)
                self._full_hf[key] = full
            full[self.start : self.stop].copy_(value)
            out[key] = full
        return out
