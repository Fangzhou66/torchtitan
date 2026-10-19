# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pull only this rank's tensor-parallel slice of the non-expert weights.

A vllm_native generator with TP > 1 fetches every non-expert tensor whole and
lets ``load_weights`` keep its 1 / tp share. Each vLLM weight loader narrows the
global HF tensor on one dimension (column-parallel: dim 0 per fused segment,
row-parallel: dim 1, vocab-parallel: the rank's vocab range), and every HF tensor
involved is either a torchtitan tensor unchanged or a dim-0 concatenation of
torchtitan tensors (GatedDeltaNet q/k/v and conv). So the slice is cut per
torchtitan key: this module requests only that slice from TorchStore, writes it
at its global offset in a reused full-size CPU tensor whose other rows are never
read, and ``to_hf`` + ``load_weights`` run unchanged. Values loaded into vLLM are
bitwise those of the full pull.

Keys not listed in ``TPSliceRules`` (norms, routers, replicated shared experts,
the vision tower) are fetched whole, as before.
"""

import re
from dataclasses import dataclass, field
from typing import Any

import torch

from torchtitan.tools.logging import logger

# Column-parallel (dim 0, even split): attention q (with its output gate),
# GatedDeltaNet input projections, conv and per-head params, dense FFN w1/w3.
_COLWISE = re.compile(
    r"^layers\.\d+\.("
    r"attn\.(wq|in_proj_q|in_proj_k|in_proj_v|in_proj_z|conv_q|conv_k|conv_v)\.weight"
    r"|attn\.(A_log|dt_bias)"
    r"|feed_forward\.(w1|w3)\.weight"
    r")$"
)
_BA = re.compile(r"^layers\.\d+\.attn\.in_proj_[ab]\.weight$")
# Row-parallel (dim 1, even split).
_ROWWISE = re.compile(
    r"^layers\.\d+\.(attn\.(wo|out_proj)\.weight|feed_forward\.w2\.weight)$"
)
# Attention k/v: dim 0, replicated across ranks when tp > num_kv_heads.
_KV = re.compile(r"^layers\.\d+\.attn\.(wk|wv)\.weight$")
_SHARED_COLWISE = re.compile(r"^layers\.\d+\.moe\.shared_experts\.(w1|w3)\.weight$")
_SHARED_ROWWISE = re.compile(r"^layers\.\d+\.moe\.shared_experts\.w2\.weight$")


@dataclass
class TPSliceRules:
    """Which rows / columns of each torchtitan key a vLLM TP rank loads.

    ``vocab_ranges`` maps ``tok_embeddings.weight`` / ``lm_head.weight`` to the
    rank's ``[start, stop)`` vocab rows (vLLM's ``shard_indices``).
    """

    tp_rank: int
    tp_size: int
    num_kv_heads: int
    vocab_ranges: dict[str, tuple[int, int]] = field(default_factory=dict)
    split_shared_experts: bool = False
    split_ba: bool = True

    def _even(self, dim: int, shape: tuple[int, ...]) -> tuple[int, int, int] | None:
        n = shape[dim]
        if n % self.tp_size:
            return None
        size = n // self.tp_size
        return dim, self.tp_rank * size, (self.tp_rank + 1) * size

    def slice_for(self, key: str, shape: tuple[int, ...]) -> tuple[int, int, int] | None:
        """``(dim, start, stop)`` this rank loads from ``key``, or None to fetch
        the whole tensor."""
        if self.tp_size == 1 or not shape:
            return None
        if _COLWISE.match(key) or (self.split_ba and _BA.match(key)):
            return self._even(0, shape)
        if self.split_shared_experts and _SHARED_COLWISE.match(key):
            return self._even(0, shape)
        if len(shape) >= 2 and (
            _ROWWISE.match(key)
            or (self.split_shared_experts and _SHARED_ROWWISE.match(key))
        ):
            return self._even(1, shape)
        if _KV.match(key):
            # vLLM QKVParallelLinear: shard_rank = tp_rank // num_kv_head_replicas.
            if self.tp_size >= self.num_kv_heads:
                if self.tp_size % self.num_kv_heads or shape[0] % self.num_kv_heads:
                    return None
                size = shape[0] // self.num_kv_heads
                shard = self.tp_rank // (self.tp_size // self.num_kv_heads)
                return 0, shard * size, (shard + 1) * size
            return self._even(0, shape)
        vocab = self.vocab_ranges.get(key)
        if vocab is not None:
            return 0, vocab[0], vocab[1]
        return None


def tp_rules_from_vllm_model(model: torch.nn.Module) -> TPSliceRules | None:
    """Rules for a native vLLM Qwen3.5 model, or None when partial pulls are
    unsafe (TP 1, quantized weights, no attention layer to read the kv heads)."""
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    tp_size = get_tensor_model_parallel_world_size()
    if tp_size == 1:
        return None
    num_kv_heads = None
    vocab_ranges: dict[str, tuple[int, int]] = {}
    split_shared = False
    split_ba = True
    for name, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method is not None and not type(quant_method).__name__.startswith(
            "Unquantized"
        ):
            logger.warning(
                f"[tp-partial-pull] {name} is quantized "
                f"({type(quant_method).__name__}); pulling full weights"
            )
            return None
        # The text model's attention (the vision tower has its own qkv).
        if num_kv_heads is None and name.endswith("self_attn.qkv_proj"):
            num_kv_heads = int(module.total_num_kv_heads)
        if getattr(module, "disable_tp_for_ba_proj", False):
            split_ba = False
        shared = getattr(module, "shared_expert", None)
        if shared is not None and getattr(shared.gate_up_proj, "tp_size", 1) > 1:
            split_shared = True
        indices = getattr(module, "shard_indices", None)
        if indices is not None:
            span = (
                int(indices.org_vocab_start_index),
                int(indices.org_vocab_end_index),
            )
            if name.endswith("embed_tokens"):
                vocab_ranges["tok_embeddings.weight"] = span
            elif name.endswith("lm_head"):
                vocab_ranges["lm_head.weight"] = span
    if num_kv_heads is None:
        return None
    return TPSliceRules(
        tp_rank=get_tensor_model_parallel_rank(),
        tp_size=tp_size,
        num_kv_heads=num_kv_heads,
        vocab_ranges=vocab_ranges,
        split_shared_experts=split_shared,
        split_ba=split_ba,
    )


class NativeTPPartialPull:
    """TP-sliced weight pull for one vllm_native rank.

    The first pull is full; ``record`` learns each sliced key's global shape and
    dtype from it. Later pulls fetch only the slices (and, given an
    ``NativeEPPartialPull``, only the local experts).
    """

    def __init__(self, rules: TPSliceRules) -> None:
        self.rules = rules
        # flat key -> (global shape, dtype, dim, start, stop)
        self._meta: dict[str, tuple[tuple[int, ...], torch.dtype, int, int, int]] = {}
        self._full: dict[str, torch.Tensor] = {}
        self.last_fetch_bytes = 0

    @property
    def ready(self) -> bool:
        return bool(self._meta)

    def record(self, state_dict: dict[str, Any]) -> None:
        """Learn the sliced keys from a full fetch."""
        for key, value in state_dict.items():
            if not isinstance(value, torch.Tensor):
                continue
            shape = tuple(value.shape)
            spec = self.rules.slice_for(key, shape)
            if spec is not None:
                self._meta[key] = (shape, value.dtype, *spec)
        if self._meta:
            sliced = sum(
                (stop - start) * torch.Size(s).numel() // s[dim] * dt.itemsize
                for s, dt, dim, start, stop in self._meta.values()
            )
            full = sum(
                torch.Size(s).numel() * dt.itemsize for s, dt, *_ in self._meta.values()
            )
            logger.info(
                f"[tp-partial-pull] tp rank {self.rules.tp_rank}/{self.rules.tp_size}: "
                f"{len(self._meta)} keys sliced, {sliced / 2**30:.2f} of "
                f"{full / 2**30:.2f} GiB"
            )

    def _full_buffer(self, key: str, shape: tuple[int, ...], dtype: torch.dtype):
        full = self._full.get(key)
        if full is None or tuple(full.shape) != shape or full.dtype != dtype:
            # Rows outside this rank's slice are never written or read.
            full = torch.empty(shape, dtype=dtype)
            self._full[key] = full
        return full

    def slice_request(self, store_key: str, flat_key: str):
        """``(request, destination, full)`` for a sliced key, or None.

        A dim-0 slice is fetched straight into its (contiguous) rows of the
        full-size buffer; other dims go through a contiguous local tensor that
        ``finish`` copies in.
        """
        from torchstore.transport.types import Request, TensorSlice

        meta = self._meta.get(flat_key)
        if meta is None:
            return None
        shape, dtype, dim, start, stop = meta
        full = self._full_buffer(flat_key, shape, dtype)
        local_shape = list(shape)
        local_shape[dim] = stop - start
        if dim == 0:
            dest = full[start:stop]
        else:
            dest = torch.empty(local_shape, dtype=dtype)
        offsets = [0] * len(shape)
        offsets[dim] = start
        spec = TensorSlice(
            offsets=tuple(offsets),
            coordinates=None,
            global_shape=shape,
            local_shape=tuple(local_shape),
            mesh_shape=None,
        )
        return Request.from_any(store_key, dest, spec), dest, full

    def finish(self, flat_key: str, dest: torch.Tensor) -> torch.Tensor:
        """The full-size tensor holding this rank's slice of ``flat_key``."""
        shape, _, dim, start, stop = self._meta[flat_key]
        full = self._full[flat_key]
        if dim != 0:
            full.narrow(dim, start, stop - start).copy_(dest)
        return full

    async def fetch(
        self, key: str = "model_state_dict", ep: Any | None = None
    ) -> dict[str, Any]:
        """Fetch the state dict with this rank's TP slice of each sliced key
        (embedded in a full-size tensor) and, when ``ep`` is a ready
        ``NativeEPPartialPull``, only its local experts."""
        import torchstore as ts
        from torchstore.state_dict_utils import DELIM, MAPPING, unflatten_state_dict
        from torchstore.transport.types import Request

        client = await ts.client()
        mapping = await client.get(f"{key}{DELIM}{MAPPING}")
        flat_keys = list(mapping.keys())
        if ep is not None:
            ep.prepare(flat_keys)
        requests, dests, kinds = [], {}, {}
        for fk in flat_keys:
            store_key = f"{key}{DELIM}{fk}"
            expert = ep.expert_request(store_key, fk) if ep is not None else None
            if expert is not None:
                requests.append(expert[0])
                dests[store_key] = expert[1]
                kinds[fk] = "expert"
                continue
            sliced = self.slice_request(store_key, fk)
            if sliced is not None:
                requests.append(sliced[0])
                dests[store_key] = sliced[1]
                kinds[fk] = "tp"
                continue
            requests.append(Request.from_any(store_key, None))
        results = await client._fetch(requests)
        flat, nbytes = {}, 0
        for fk, request in zip(flat_keys, requests):
            value = client._apply_inplace(
                results[request.key], dests.get(request.key), request
            )
            if isinstance(value, torch.Tensor):
                nbytes += value.numel() * value.element_size()
            if kinds.get(fk) == "tp":
                value = self.finish(fk, value)
            flat[fk] = value
        self.last_fetch_bytes = nbytes
        if ep is not None:
            ep.last_fetch_partial = True
        return unflatten_state_dict(flat, mapping)
