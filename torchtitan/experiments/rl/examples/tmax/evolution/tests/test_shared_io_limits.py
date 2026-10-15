# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import evolve_codex as ec
import rewrite_workspace as rw


def _peak(fn_wrapper):
    lock = threading.Lock()
    state = {"now": 0, "peak": 0}

    def work(_item):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.02)
        with lock:
            state["now"] -= 1

    fn_wrapper(work)
    return state["peak"]


def test_concurrent_publishes_share_one_pool(monkeypatch) -> None:
    monkeypatch.setattr(rw, "PUBLISH_WORKERS", 4)
    monkeypatch.setattr(rw, "_PUBLISH_POOL", None)

    def run(work):
        with ThreadPoolExecutor(5) as callers:
            list(callers.map(lambda _: rw._pool_map(work, list(range(8))), range(5)))

    assert _peak(run) <= 4


def test_cache_deletions_are_capped(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(ec, "_TRASH_SLOTS", threading.BoundedSemaphore(2))
    lock = threading.Lock()
    state = {"now": 0, "peak": 0}

    def slow_rmtree(path, ignore_errors=False):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.05)
        with lock:
            state["now"] -= 1

    monkeypatch.setattr(ec.shutil, "rmtree", slow_rmtree)
    for i in range(6):
        ec._delete_in_background(tmp_path / f"t{i}")
    ec.drain_cache_cleanup(tmp_path)
    assert state["peak"] <= 2


def test_live_mix_read_retries_a_publish_swap(monkeypatch, tmp_path) -> None:
    import evolve_ondella as od
    from types import SimpleNamespace

    live = tmp_path / "live.jsonl"
    calls = []
    real = od.layout.read_jsonl

    def flaky(path, *, missing_ok=True):
        calls.append(path)
        if len(calls) == 1:
            raise FileNotFoundError(path)
        return [{"metadata": {"instance_id": "t"}}]

    monkeypatch.setattr(od.layout, "read_jsonl", flaky)
    monkeypatch.setattr(od.time, "sleep", lambda _s: None)
    root = SimpleNamespace(mix=SimpleNamespace(live=live))
    assert od._read_live_mix(root) == [{"metadata": {"instance_id": "t"}}]
    assert len(calls) == 2
    monkeypatch.setattr(od.layout, "read_jsonl", real)
