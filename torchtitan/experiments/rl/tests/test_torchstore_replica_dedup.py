# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""torchstore_replica_dedup: one read per replicated region."""

from types import SimpleNamespace

import torch
from torchstore.client import LocalClient
from torchstore.controller import ObjectType, StorageInfo
from torchstore.transport.types import Request, TensorSlice

from torchtitan.experiments.rl import torchstore_replica_dedup as dedup

ROWS, COLS, SHARDS, REPLICAS = 64, 4, 8, 3


def _slice(shard: int, replica: int) -> TensorSlice:
    rows = ROWS // SHARDS
    return TensorSlice(
        offsets=(shard * rows, 0),
        coordinates=(replica, shard),
        global_shape=(ROWS, COLS),
        local_shape=(rows, COLS),
        mesh_shape=(REPLICAS, SHARDS),
    )


def _hsdp_volume_map() -> dict:
    # One volume per trainer rank, rank = replica * SHARDS + shard (HSDP 3 x 8).
    return {
        f"v{replica * SHARDS + shard:02d}": StorageInfo(
            object_type=ObjectType.TENSOR_SLICE,
            tensor_slices={_slice(shard, replica)},
        )
        for replica in range(REPLICAS)
        for shard in range(SHARDS)
    }


def _regions(volume_map: dict) -> list[tuple]:
    return sorted(
        (tuple(s.offsets), tuple(s.local_shape))
        for info in volume_map.values()
        for s in info.tensor_slices
    )


def test_one_volume_per_region_and_every_region_kept() -> None:
    full = _hsdp_volume_map()
    kept = dedup.dedup_volume_map(full, seed=7)
    assert len(_regions(full)) == SHARDS * REPLICAS
    assert _regions(kept) == sorted(set(_regions(full)))  # each region once, none lost


def test_readers_spread_over_replicas() -> None:
    replicas_used = set()
    for seed in range(32):
        for volume_id in dedup.dedup_volume_map(_hsdp_volume_map(), seed):
            replicas_used.add(int(volume_id[1:]) // SHARDS)
    assert replicas_used == set(range(REPLICAS))


def test_unreplicated_and_object_keys_unchanged() -> None:
    single = {
        "v0": StorageInfo(object_type=ObjectType.TENSOR_SLICE, tensor_slices={_slice(0, 0)}),
        "v1": StorageInfo(object_type=ObjectType.TENSOR_SLICE, tensor_slices={_slice(1, 0)}),
    }
    assert dedup.dedup_volume_map(single, seed=3) is single
    obj = {"v0": StorageInfo(object_type=ObjectType.OBJECT)}
    assert dedup.dedup_volume_map(obj, seed=3) is obj


def test_patched_client_requests_each_region_once() -> None:
    dest = torch.empty(ROWS, COLS)
    key = "model_state_dict/w"
    volume_maps = {key: _hsdp_volume_map()}
    transport = {
        vid: SimpleNamespace(supports_inplace_resharding=True)
        for vid in volume_maps[key]
    }
    client = object.__new__(LocalClient)

    def count(build):
        reqs, _ = build(client, [Request.from_any(key, dest)], volume_maps, transport)
        return sum(len(v) for v in reqs.values())

    original = LocalClient._build_volume_requests
    try:
        before = count(LocalClient._build_volume_requests)
        dedup._installed = False
        dedup.install()
        after = count(LocalClient._build_volume_requests)
    finally:
        LocalClient._build_volume_requests = original
        dedup._installed = False
    assert before == SHARDS * REPLICAS
    assert after == SHARDS
