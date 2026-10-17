# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fetch each replicated tensor region from one TorchStore volume, not all.

A trainer with data-parallel replication (HSDP ``dp_replicate > 1``) puts the
same shard from every replica rank, so TorchStore holds each region once per
replica. ``LocalClient._build_volume_requests`` requests every stored slice, so
a generator reads each region ``dp_replicate`` times and overwrites the same
destination bytes (torchstore notes this as a TODO in ``_expand_tensor_slices``).
Replicas are bitwise identical after the all-reduced optimizer step, so reading
one copy is exact. The copy is picked per region and per process so concurrent
readers spread over the replicas instead of all hitting the first one.
"""

import hashlib
import os
import socket
from dataclasses import replace

from torchtitan.tools.logging import logger

_installed = False


def _reader_seed() -> int:
    ident = f"{socket.gethostname()}:{os.environ.get('RL_GPU_INDEX', '')}:{os.getpid()}"
    return int.from_bytes(hashlib.blake2b(ident.encode(), digest_size=8).digest(), "big")


def dedup_volume_map(volume_map: dict, seed: int) -> dict:
    """Keep one (volume, slice) per distinct region of a TENSOR_SLICE key.

    ``volume_map`` is ``{volume_id: StorageInfo}``. Object and whole-tensor keys,
    and regions stored once, pass through unchanged.
    """
    from torchstore.controller import ObjectType

    infos = list(volume_map.values())
    if not infos or any(i.object_type != ObjectType.TENSOR_SLICE for i in infos):
        return volume_map
    regions: dict[tuple, list[tuple[str, object]]] = {}
    for volume_id in sorted(volume_map):
        for tensor_slice in volume_map[volume_id].tensor_slices:
            if tensor_slice is None:
                return volume_map
            key = (tuple(tensor_slice.offsets), tuple(tensor_slice.local_shape))
            regions.setdefault(key, []).append((volume_id, tensor_slice))
    if all(len(c) == 1 for c in regions.values()):
        return volume_map
    chosen: dict[str, set] = {}
    for key, candidates in regions.items():
        pick = (seed + hash(key)) % len(candidates)
        volume_id, tensor_slice = candidates[pick]
        chosen.setdefault(volume_id, set()).add(tensor_slice)
    return {
        volume_id: replace(volume_map[volume_id], tensor_slices=slices)
        for volume_id, slices in chosen.items()
    }


def install() -> None:
    """Patch torchstore's LocalClient to skip duplicate replica reads (idempotent)."""
    global _installed
    if _installed:
        return
    from torchstore.client import LocalClient

    original = LocalClient._build_volume_requests
    seed = _reader_seed()

    def _build_volume_requests(self, requests, volume_maps, transport_buffer_map):
        volume_maps = {
            key: dedup_volume_map(volume_map, seed)
            for key, volume_map in volume_maps.items()
        }
        return original(self, requests, volume_maps, transport_buffer_map)

    LocalClient._build_volume_requests = _build_volume_requests
    _installed = True
    logger.info("[torchstore] reading each replicated region from one volume")
