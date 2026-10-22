# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The overfit bench scheduler: owns the version pool and talks to the rewriter.

It runs as threads inside the controller process, which is the only process
that draws training samples; RolloutWorkers only send it events. It listens on
a unix socket (JSON lines) whose path it writes to ``<job>/scheduler.json``.
The socket sits under a node-local ``/tmp/<user>-bench-<id>/`` because a unix
socket path is limited to ~104 bytes and a job directory on GPFS is longer.

Protocol. A client's first line is ``{"op": "hello", "role": ...}``.

- ``role: "producer"`` (a RolloutWorker): then ``{"op": "event", "event": {...}}``
  lines, never answered.
- ``role: "rewriter"``: may send ``{"op": "pool", "id": n, "call": <add |
  reweight | retire | get_state>, "args": {...}}`` and gets ``{"type": "reply",
  "id": n, "ok": true, "result": ...}`` or ``"ok": false, "error": ...``. It
  receives every event as ``{"type": "event", "event": {...}}``. With
  ``"hooks": ["which_version"]`` in its hello it is also asked, at every
  training draw, ``{"type": "which_version", "req": k, "candidates": [...]}``
  and answers ``{"op": "answer", "req": k, "version": <id or null>}``. An
  answer later than ``hook_timeout_sec`` (1 s), or naming no active version,
  falls back to the pool's weighted draw. A dead or absent rewriter never holds
  a draw back.

Events go through one bounded queue (drop-oldest, counted) to
``events.jsonl`` and to each rewriter's own bounded queue (drop-oldest,
counted): a rewriter that stops reading loses events, never stalls training,
and can catch up from the file. ``pool_drained`` is emitted when a task's
active weight reaches 0.

A rewriter that publishes ``data/mix/live.jsonl`` the file way (``publish_mix.py``,
or the reference loop's fold) is imported: the file names the whole active
set, so new rows become versions (parent: the version of their task the same
publish drops) and dropped ones are retired.
"""

from __future__ import annotations

import collections
import getpass
import hashlib
import json
import logging
import os
import random
import shutil
import socket
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path

from torchtitan.experiments.rl.examples.tmax import layout
from torchtitan.experiments.rl.examples.tmax.data import _parse_sample_row, TMaxSample
from torchtitan.experiments.rl.examples.tmax.overfit_bench import events
from torchtitan.experiments.rl.examples.tmax.overfit_bench.pool import (
    PoolError,
    VersionPool,
)
from torchtitan.experiments.rl.types import SampleLineage

logger = logging.getLogger(__name__)

PROTOCOL = 1
_EVENT_QUEUE = 4096
_CLIENT_QUEUE = 4096
_POOL_CALLS = ("add", "reweight", "retire", "get_state")

_current: "BenchScheduler | None" = None


def current() -> "BenchScheduler | None":
    """The scheduler running in this process, if any (only the controller's)."""
    return _current


def require_current() -> "BenchScheduler":
    if _current is None:
        raise RuntimeError(
            "SWE_BENCH_SCHEDULER=1 but no bench scheduler runs in this process; "
            "it is started by BenchController (config_registry wraps the "
            "controller in it when the switch is on)"
        )
    return _current


class _Client:
    """One connection. Replies and hook requests go out before queued events."""

    def __init__(self, sock: socket.socket, name: str) -> None:
        self.sock = sock
        self.name = name
        self.role = ""
        self.hooks: set[str] = set()
        self.urgent: collections.deque[dict] = collections.deque()
        self.events: collections.deque[dict] = collections.deque(maxlen=_CLIENT_QUEUE)
        self.dropped = 0
        self.cv = threading.Condition()
        self.closed = False

    def send(self, message: dict, *, urgent: bool = False) -> None:
        with self.cv:
            if self.closed:
                return
            if urgent:
                self.urgent.append(message)
            else:
                if len(self.events) == self.events.maxlen:
                    self.dropped += 1
                self.events.append(message)
            self.cv.notify()

    def writer(self) -> None:
        while True:
            with self.cv:
                while not (self.urgent or self.events or self.closed):
                    self.cv.wait()
                if self.closed:
                    return
                message = self.urgent.popleft() if self.urgent else self.events.popleft()
            try:
                self.sock.sendall((json.dumps(message, ensure_ascii=False) + "\n").encode())
            except OSError:
                self.close()
                return

    def close(self) -> None:
        with self.cv:
            self.closed = True
            self.cv.notify_all()
        try:
            self.sock.close()
        except OSError:
            pass


class BenchScheduler:
    def __init__(
        self,
        root: layout.Root,
        *,
        run: layout.Run | None = None,
        seed: int = 0,
        hook_timeout_sec: float = 1.0,
        import_interval_sec: float = 2.0,
        socket_dir: Path | None = None,
    ) -> None:
        self.root = root
        self.run = run
        self.hook_timeout_sec = hook_timeout_sec
        self.import_interval_sec = import_interval_sec
        logs = run.scheduler if run is not None else None
        self._events_path = logs / "events.jsonl" if logs is not None else None
        self._draws_path = logs / "draws.jsonl" if logs is not None else None
        self.pool = VersionPool(
            log_path=logs / "pool.jsonl" if logs is not None else None,
            state_path=root.mix.pool_state,
            publish=self._publish,
            on_drained=lambda task: self.submit(
                events.make(events.POOL_DRAINED, task=task)
            ),
        )
        self._rng = random.Random(seed)
        self._draws = 0
        self._stream_id = uuid.uuid4().hex
        self._samples: dict[str, TMaxSample] = {}
        self._queue: collections.deque[dict] = collections.deque(maxlen=_EVENT_QUEUE)
        self._queue_cv = threading.Condition()
        self.dropped = 0
        self._clients: list[_Client] = []
        self._clients_lock = threading.Lock()
        self._answers: dict[int, str | None] = {}
        self._answer_cv = threading.Condition()
        self._next_req = 0
        self._live_seen: tuple[int, int] | None = None
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._server: socket.socket | None = None
        if socket_dir is None:
            tag = hashlib.sha256(f"{root.path.resolve()}:{os.getpid()}".encode()).hexdigest()[:10]
            socket_dir = Path("/tmp") / f"{getpass.getuser()}-bench-{tag}"
        self.socket_dir = socket_dir
        self.socket_path = socket_dir / "sched.sock"

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Load the pool, listen, and become this process's scheduler."""
        global _current
        state = self.root.mix.pool_state
        if state.is_file():
            self.pool.load_state_dict(json.loads(state.read_text()))
        self._import_live(source="boot")
        if not self.pool.active():
            raise RuntimeError(f"{self.root.mix.live} holds no row to train on")
        self.socket_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.socket_path.exists():
            self.socket_path.unlink()
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(self.socket_path))
        server.listen(64)
        server.settimeout(0.5)
        self._server = server
        layout.write_json_atomic(
            self.root.scheduler_json,
            {
                "socket": str(self.socket_path),
                "protocol": PROTOCOL,
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "run": self.run.name if self.run is not None else None,
                "started": layout.stamp(),
            },
        )
        for target, name in (
            (self._accept_loop, "bench-accept"),
            (self._dispatch_loop, "bench-dispatch"),
            (self._import_loop, "bench-import"),
        ):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        _current = self
        events.set_local_sink(self.submit)
        logger.info(
            f"[bench] scheduler up on {self.socket_path}: "
            f"{len(self.pool.active())} active version(s)"
        )

    def stop(self) -> None:
        global _current
        if _current is self:
            _current = None
            events.set_local_sink(None)
        self._stop.set()
        with self._queue_cv:
            self._queue_cv.notify_all()
        if self._server is not None:
            self._server.close()
        with self._clients_lock:
            for c in self._clients:
                c.close()
        for t in self._threads:
            t.join(timeout=5)
        try:
            if json.loads(self.root.scheduler_json.read_text()).get("socket") == str(
                self.socket_path
            ):
                self.root.scheduler_json.unlink()
        except (OSError, ValueError):
            pass
        shutil.rmtree(self.socket_dir, ignore_errors=True)
        logger.info(f"[bench] scheduler stopped; {self.dropped} event(s) dropped")

    # -- draws -------------------------------------------------------------

    def draw(self) -> TMaxSample:
        """The next training sample: the rewriter's pick if it answers in time,
        else a weighted draw from the pool. Never waits longer than the hook
        timeout."""
        started = time.monotonic()
        version = None
        how = "weighted"
        picked = self._ask_rewriter()
        if picked is not None and self.pool.is_active(picked):
            version = self.pool.get(picked)
            how = "rewriter"
        elif picked is not None:
            how = "rewriter_invalid"
        if version is None:
            version, fallback = self.pool.sample(self._rng)
            if fallback != "weighted":
                how = fallback
                logger.warning(f"[bench] draw fell back to {fallback}: {version.label}")
        sample = self._samples.get(version.id)
        if sample is None:
            sample = replace(_parse_sample_row(version.row), version=version.id)
            self._samples[version.id] = sample
        position = self._draws
        self._draws += 1
        if self._draws_path is not None:
            layout.append_jsonl(
                self._draws_path,
                {
                    "stamp": layout.stamp(),
                    "draw": position,
                    "version": version.id,
                    "label": version.label,
                    "by": how,
                    "ms": round(1000 * (time.monotonic() - started)),
                },
            )
        return replace(
            sample,
            lineage=SampleLineage(
                occurrence_id=f"{self._stream_id}:{position}",
                task_id=version.task,
                sample_revision=version.id,
                mix_revision=self.pool.revision(),
                dataset_epoch=0,
                dataset_position=position,
                stream_position=position,
                stream_id=self._stream_id,
            ),
        )

    def _ask_rewriter(self) -> str | None:
        with self._clients_lock:
            hooked = [c for c in self._clients if "which_version" in c.hooks and not c.closed]
        if not hooked:
            return None
        client = hooked[-1]
        with self._answer_cv:
            # Draws are serial, so any answer still here was too late for its draw.
            self._answers.clear()
            req = self._next_req
            self._next_req += 1
        client.send(
            {
                "type": "which_version",
                "req": req,
                "candidates": [v.info() for v in self.pool.active()],
            },
            urgent=True,
        )
        deadline = time.monotonic() + self.hook_timeout_sec
        with self._answer_cv:
            while req not in self._answers:
                left = deadline - time.monotonic()
                if left <= 0:
                    return None
                self._answer_cv.wait(left)
            return self._answers.pop(req)

    # -- events ------------------------------------------------------------

    def submit(self, event: dict) -> None:
        """Queue an event for events.jsonl and every rewriter. Never blocks."""
        with self._queue_cv:
            if len(self._queue) == self._queue.maxlen:
                self.dropped += 1
            self._queue.append(event)
            self._queue_cv.notify()

    def _dispatch_loop(self) -> None:
        while True:
            with self._queue_cv:
                while not self._queue and not self._stop.is_set():
                    self._queue_cv.wait()
                if not self._queue:
                    return
                event = self._queue.popleft()
            try:
                if self._events_path is not None:
                    layout.append_jsonl(self._events_path, event)
            except OSError as e:
                logger.warning(f"[bench] events.jsonl line lost: {e}")
            with self._clients_lock:
                rewriters = [c for c in self._clients if c.role == "rewriter"]
            for c in rewriters:
                c.send({"type": "event", "event": event})

    # -- the socket --------------------------------------------------------

    def _accept_loop(self) -> None:
        n = 0
        while not self._stop.is_set():
            try:
                sock, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            sock.settimeout(None)
            n += 1
            client = _Client(sock, f"client-{n}")
            threading.Thread(
                target=self._serve, args=(client,), name=f"bench-{client.name}", daemon=True
            ).start()

    def _serve(self, client: _Client) -> None:
        with self._clients_lock:
            self._clients.append(client)
        writer = threading.Thread(
            target=client.writer, name=f"bench-{client.name}-out", daemon=True
        )
        writer.start()
        try:
            for line in client.sock.makefile("r", encoding="utf-8"):
                if not line.strip():
                    continue
                try:
                    message = json.loads(line)
                except ValueError:
                    logger.warning(f"[bench] {client.name}: unparseable line dropped")
                    continue
                try:
                    self._handle(client, message)
                except (TypeError, ValueError) as e:
                    logger.warning(f"[bench] {client.name}: bad message dropped ({e})")
        except OSError:
            pass
        finally:
            client.close()
            with self._clients_lock:
                if client in self._clients:
                    self._clients.remove(client)
            if client.role == "rewriter":
                logger.info(
                    f"[bench] rewriter {client.name} disconnected "
                    f"({client.dropped} event(s) dropped for it)"
                )

    def _handle(self, client: _Client, message: dict) -> None:
        op = message.get("op")
        if op == "hello":
            client.role = str(message.get("role") or "")
            client.hooks = set(message.get("hooks") or ()) & {"which_version"}
            if client.role == "rewriter":
                logger.info(f"[bench] rewriter {client.name} connected, hooks={sorted(client.hooks)}")
        elif op == "event":
            event = message.get("event")
            if isinstance(event, dict) and event.get("event") in events.FIELDS:
                self.submit(event)
        elif op == "answer":
            with self._answer_cv:
                self._answers[int(message.get("req", -1))] = message.get("version")
                self._answer_cv.notify_all()
        elif op == "pool" and client.role == "rewriter":
            client.send(self._pool_call(message), urgent=True)

    def _pool_call(self, message: dict) -> dict:
        reply = {"type": "reply", "id": message.get("id")}
        call = message.get("call")
        args = message.get("args") or {}
        try:
            if call not in _POOL_CALLS:
                raise PoolError(f"unknown call {call!r}; one of {list(_POOL_CALLS)}")
            if call == "add":
                result = self.pool.add(
                    args["row"],
                    weight=args.get("weight", 1.0),
                    parent=args.get("parent"),
                    source="rewriter",
                )
            elif call == "reweight":
                result = self.pool.reweight(args["version"], args["weight"], source="rewriter")
            elif call == "retire":
                result = self.pool.retire(args["version"], source="rewriter")
            else:
                result = self.pool.get_state(with_rows=bool(args.get("with_rows")))
        except (PoolError, KeyError, TypeError) as e:
            return {**reply, "ok": False, "error": f"{type(e).__name__}: {e}"}
        return {**reply, "ok": True, "result": result}

    # -- the mix files -----------------------------------------------------

    def _publish(self, rows: list[str]) -> None:
        if not rows:
            # The pool keeps drawing (see VersionPool.sample); the mix keeps
            # its last rows, so a restart still has something to load.
            logger.warning("[bench] no active version left; data/mix not republished")
            return
        self.root.mix.publish(rows)
        self._live_seen = self._live_id()

    def _live_id(self) -> tuple[int, int] | None:
        try:
            st = os.stat(self.root.mix.live)
        except OSError:
            return None
        return st.st_ino, st.st_mtime_ns

    def _import_live(self, *, source: str) -> None:
        """Take an external publish of live.jsonl into the pool (see module doc)."""
        seen = self._live_id()
        if seen is None or seen == self._live_seen:
            return
        self._live_seen = seen
        try:
            text = self.root.mix.read_live()
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
            if not rows:
                raise PoolError("no rows")
            result = self.pool.reconcile(rows, source=source)
        except (OSError, ValueError) as e:
            logger.warning(f"[bench] live.jsonl not imported, the pool keeps its state: {e}")
            return
        if result["added"] or result["retired"]:
            logger.info(f"[bench] imported live.jsonl ({source}): {result}")

    def _import_loop(self) -> None:
        while not self._stop.wait(self.import_interval_sec):
            try:
                self._import_live(source="file")
            except Exception:  # noqa: BLE001 -- the importer must outlive one bad file
                logger.exception("[bench] live.jsonl import failed")

    # -- checkpoint --------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "pool": self.pool.state_dict(),
            "rng": self._rng.getstate(),
            "draws": self._draws,
            "stream_id": self._stream_id,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore the draw stream; the pool only if the checkpoint's is not
        older than what pool.json held (a rewriter may have changed it since)."""
        self._rng.setstate(state["rng"])
        self._draws = int(state["draws"])
        self._stream_id = state["stream_id"]
        if state["pool"]["seq"] >= self.pool.seq:
            self.pool.load_state_dict(state["pool"])
            self._samples.clear()
        else:
            logger.info(
                f"[bench] checkpoint pool (seq {state['pool']['seq']}) is older than "
                f"pool.json (seq {self.pool.seq}); keeping pool.json"
            )
