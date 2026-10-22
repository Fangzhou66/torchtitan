# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The overfit bench scheduler's events: their schema, and how a process emits one.

Four kinds, training only (validation, which is the readout, emits nothing):

    rollout_done  version task rev group idx reward finish_reason record_path policy_versions
    group_done    version task rev group passed scored policy_version
    step_done     step policy_version
    pool_drained  task

``reward`` is the verifier's verdict, None for an attempt the harness could not
score; ``passed`` counts the scored attempts that solved it, ``scored`` the
attempts that count (not infra-failed, not ended by a strict format rule), the
same set the evolution signal judges. ``record_path`` is relative to ``run``.
Every event also carries ``event``, ``run`` and ``stamp``.

``emit`` is the producer side and never raises or blocks: in the controller the
scheduler takes the event directly; anywhere else (a RolloutWorker process) a
background thread sends it over the scheduler's socket, found lazily from
``SWE_BENCH_SCHEDULER_SOCKET`` or ``$TRL_BASE/scheduler.json``. With no
scheduler to send to, an event is dropped and counted.
"""

from __future__ import annotations

import collections
import json
import logging
import os
import socket
import threading
import time
from collections.abc import Callable

from torchtitan.experiments.rl.examples.tmax import layout

logger = logging.getLogger(__name__)

ROLLOUT_DONE = "rollout_done"
GROUP_DONE = "group_done"
STEP_DONE = "step_done"
POOL_DRAINED = "pool_drained"

FIELDS: dict[str, tuple[str, ...]] = {
    ROLLOUT_DONE: (
        "version",
        "task",
        "rev",
        "group",
        "idx",
        "reward",
        "finish_reason",
        "record_path",
        "policy_versions",
    ),
    GROUP_DONE: (
        "version",
        "task",
        "rev",
        "group",
        "passed",
        "scored",
        "policy_version",
    ),
    STEP_DONE: ("step", "policy_version"),
    POOL_DRAINED: ("task",),
}

SOCKET_ENV = "SWE_BENCH_SCHEDULER_SOCKET"
# Events held while the socket is down; the oldest go first.
_QUEUE_SIZE = 4096
_RECONNECT_SEC = 5.0


def make(kind: str, **fields) -> dict:
    """One event of ``kind`` with exactly its schema's fields."""
    expected = FIELDS.get(kind)
    if expected is None:
        raise ValueError(f"unknown event {kind!r}")
    if set(fields) != set(expected):
        raise ValueError(
            f"{kind} takes {sorted(expected)}, got {sorted(fields)}"
        )
    run = layout.Run.from_env()
    return {
        "event": kind,
        **fields,
        "run": run.name if run is not None else None,
        "stamp": layout.stamp(),
    }


def socket_path() -> str | None:
    """The scheduler's socket: the env override, else the job's scheduler.json."""
    path = os.environ.get(SOCKET_ENV)
    if path:
        return path
    base = os.environ.get("TRL_BASE")
    if not base:
        return None
    try:
        with open(layout.Root(base).scheduler_json) as f:
            return json.load(f).get("socket") or None
    except (OSError, ValueError):
        return None


# Set by a scheduler running in this process (the controller's), which then
# receives events without a socket round trip.
_local_sink: Callable[[dict], None] | None = None


def set_local_sink(sink: Callable[[dict], None] | None) -> None:
    global _local_sink
    _local_sink = sink


class _Producer:
    """Sends events to the scheduler from a daemon thread, connecting lazily.

    Bounded: while disconnected the queue keeps the newest ``_QUEUE_SIZE``
    events, and a missing scheduler costs one stat per ``_RECONNECT_SEC``."""

    def __init__(self) -> None:
        self._queue: collections.deque[dict] = collections.deque(maxlen=_QUEUE_SIZE)
        self._cv = threading.Condition()
        self._thread: threading.Thread | None = None
        self.dropped = 0

    def put(self, event: dict) -> None:
        with self._cv:
            if len(self._queue) == self._queue.maxlen:
                self.dropped += 1
            self._queue.append(event)
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="bench-events", daemon=True
                )
                self._thread.start()
            self._cv.notify()

    def _connect(self) -> socket.socket | None:
        path = socket_path()
        if path is None:
            return None
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        try:
            sock.connect(path)
            sock.sendall(
                (json.dumps({"op": "hello", "role": "producer", "pid": os.getpid()}) + "\n").encode()
            )
        except OSError:
            sock.close()
            return None
        return sock

    def _run(self) -> None:
        sock: socket.socket | None = None
        while True:
            with self._cv:
                while not self._queue:
                    self._cv.wait()
            if sock is None:
                sock = self._connect()
                if sock is None:
                    time.sleep(_RECONNECT_SEC)
                    continue
            with self._cv:
                event = self._queue.popleft() if self._queue else None
            if event is None:
                continue
            try:
                sock.sendall((json.dumps({"op": "event", "event": event}) + "\n").encode())
            except OSError:
                # Lost with the connection; the next one starts from the queue.
                self.dropped += 1
                sock.close()
                sock = None


_producer: _Producer | None = None
_producer_lock = threading.Lock()


def emit(kind: str, **fields) -> None:
    """Send one event to the scheduler. Never raises, never blocks a rollout."""
    global _producer
    try:
        event = make(kind, **fields)
        sink = _local_sink
        if sink is not None:
            sink(event)
            return
        with _producer_lock:
            if _producer is None:
                if socket_path() is None:
                    return
                _producer = _Producer()
        _producer.put(event)
    except Exception as e:  # noqa: BLE001 -- bookkeeping must not fail a rollout
        logger.warning(f"[bench] event {kind} not sent: {type(e).__name__}: {e}")
