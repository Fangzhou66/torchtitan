#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A rewriter's side of the bench scheduler's socket, and the reference loop on it.

    client = RewriterClient(job, which_version=pick)   # pick(candidates) -> id | None
    client.connect()
    client.add(row, parent=old_id); client.retire(old_id); client.get_state()
    event = client.wait_for({"group_done"}, timeout=60)

    rewriter_client.py reference <job> [evolve_ondella.py arguments]

``reference`` runs evolution/evolve_ondella.py unchanged except for two
things: it wakes on ``group_done`` instead of polling for signal files every
5 s, and each fold it publishes is also given to the pool as add(new row,
parent) + retire(parent). Every file the loop writes is still written. With no
scheduler to connect to it runs exactly as the file-based loop does.
"""

from __future__ import annotations

import collections
import json
import logging
import socket
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[6]))

from torchtitan.experiments.rl.examples.tmax import layout  # noqa: E402
from torchtitan.experiments.rl.training_lineage import content_revision  # noqa: E402

log = logging.getLogger("rewriter_client")


class RewriterError(RuntimeError):
    """The scheduler refused a pool call, or could not be reached."""


class RewriterClient:
    def __init__(
        self,
        job: Path | str,
        *,
        which_version: Callable[[list[dict]], str | None] | None = None,
        call_timeout_sec: float = 30.0,
        max_events: int = 10000,
    ) -> None:
        self.root = layout.Root(Path(job))
        self._which_version = which_version
        self._call_timeout_sec = call_timeout_sec
        self._sock: socket.socket | None = None
        self._send_lock = threading.Lock()
        self._events: collections.deque[dict] = collections.deque(maxlen=max_events)
        self._events_cv = threading.Condition()
        self._replies: dict[int, dict] = {}
        self._replies_cv = threading.Condition()
        self._next_id = 0

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def connect(self) -> None:
        try:
            path = json.loads(self.root.scheduler_json.read_text())["socket"]
        except (OSError, ValueError, KeyError) as e:
            raise RewriterError(f"no scheduler for {self.root.path}: {e}") from e
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(path)
        except OSError as e:
            sock.close()
            raise RewriterError(f"scheduler at {path} is not listening: {e}") from e
        self._sock = sock
        hooks = ["which_version"] if self._which_version is not None else []
        self._send({"op": "hello", "role": "rewriter", "hooks": hooks})
        threading.Thread(target=self._read, name="rewriter-client", daemon=True).start()

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    # -- pool calls --------------------------------------------------------

    def add(self, row: dict, *, weight: float = 1.0, parent: str | None = None) -> dict:
        return self.call("add", row=row, weight=weight, parent=parent)

    def reweight(self, version: str, weight: float) -> dict:
        return self.call("reweight", version=version, weight=weight)

    def retire(self, version: str) -> dict:
        return self.call("retire", version=version)

    def get_state(self, *, with_rows: bool = False) -> dict:
        return self.call("get_state", with_rows=with_rows)

    def call(self, name: str, **args):
        with self._replies_cv:
            rid = self._next_id
            self._next_id += 1
        self._send({"op": "pool", "id": rid, "call": name, "args": args})
        deadline = time.monotonic() + self._call_timeout_sec
        with self._replies_cv:
            while rid not in self._replies:
                left = deadline - time.monotonic()
                if left <= 0 or self._sock is None:
                    raise RewriterError(f"no reply to {name} from the scheduler")
                self._replies_cv.wait(min(left, 1.0))
            reply = self._replies.pop(rid)
        if not reply.get("ok"):
            raise RewriterError(reply.get("error") or f"{name} refused")
        return reply.get("result")

    # -- events ------------------------------------------------------------

    def next_event(self, timeout: float | None = None) -> dict | None:
        return self.wait_for(None, timeout)

    def wait_for(self, kinds: set[str] | None, timeout: float | None) -> dict | None:
        """The next event (of one of ``kinds``, others discarded), or None at the
        timeout. Without a connection this just waits out the timeout."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._events_cv:
            while True:
                while self._events:
                    event = self._events.popleft()
                    if kinds is None or event.get("event") in kinds:
                        return event
                left = None if deadline is None else deadline - time.monotonic()
                if left is not None and left <= 0:
                    return None
                self._events_cv.wait(left)

    # -- internals ---------------------------------------------------------

    def _send(self, message: dict) -> None:
        if self._sock is None:
            raise RewriterError("not connected")
        data = (json.dumps(message, ensure_ascii=False) + "\n").encode()
        with self._send_lock:
            try:
                self._sock.sendall(data)
            except OSError as e:
                self.close()
                raise RewriterError(f"scheduler connection lost: {e}") from e

    def _read(self) -> None:
        sock = self._sock
        try:
            for line in sock.makefile("r", encoding="utf-8"):
                if not line.strip():
                    continue
                message = json.loads(line)
                kind = message.get("type")
                if kind == "event":
                    with self._events_cv:
                        self._events.append(message["event"])
                        self._events_cv.notify_all()
                elif kind == "reply":
                    with self._replies_cv:
                        self._replies[message["id"]] = message
                        self._replies_cv.notify_all()
                elif kind == "which_version":
                    threading.Thread(
                        target=self._answer, args=(message,), daemon=True
                    ).start()
        except (OSError, ValueError) as e:
            log.warning("scheduler connection closed: %s", e)
        finally:
            self.close()
            with self._replies_cv:
                self._replies_cv.notify_all()

    def _answer(self, request: dict) -> None:
        try:
            version = self._which_version(request["candidates"])
        except Exception:  # noqa: BLE001 -- the scheduler falls back on its own
            log.exception("which_version hook failed")
            version = None
        try:
            self._send({"op": "answer", "req": request["req"], "version": version})
        except RewriterError:
            pass


def run_reference(job: Path, argv: list[str]) -> None:
    """evolve_ondella.main() on this job, woken by group_done, folds mirrored
    into the pool. See the module docstring."""
    evolution = Path(__file__).resolve().parent.parent / "evolution"
    sys.path.insert(0, str(evolution))
    import evolve_ondella as ev  # noqa: E402

    client = RewriterClient(job)
    try:
        client.connect()
    except RewriterError as e:
        log.warning("running file-based only: %s", e)

    def wait_for_signals(root: layout.Root, max_wait: int, tick: float = 5.0) -> None:
        ledger = ev.load_ledger(root)
        deadline = time.time() + max_wait
        while time.time() < deadline:
            if ev.admission_is_open() and ev._pending(root, ledger):
                return
            # group_done is emitted after its signal file is written, so a
            # wake-up here finds the file; the tick still bounds a missed event.
            client.wait_for({"group_done"}, timeout=min(tick, max(deadline - time.time(), 0)))

    file_fold = ev.fold

    def fold(root: layout.Root, accepted: list[dict]) -> int | None:
        parents = {}
        for line in root.mix.read_live().splitlines():
            if line.strip():
                row = json.loads(line)
                parents[row["metadata"]["instance_id"]] = row
        version = file_fold(root, accepted)
        if version is None or not client.connected:
            return version
        published = {}
        for line in dict(root.mix.versions())[version].read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                published[row["metadata"]["instance_id"]] = row
        for h in accepted:
            task = h["meta"]["task"]
            if h.get("status") != "accepted" or task not in published:
                continue
            parent = parents.get(task)
            parent_id = content_revision(parent) if parent is not None else None
            try:
                client.add(published[task], parent=parent_id)
                if parent_id is not None:
                    client.retire(parent_id)
            except RewriterError as e:
                # The importer takes the same publish from live.jsonl.
                log.warning("fold of %s not sent to the pool: %s", task, e)
        return version

    ev._wait_for_signals = wait_for_signals
    ev.fold = fold
    sys.argv = [str(evolution / "evolve_ondella.py"), *argv]
    ev.main()


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] != "reference":
        raise SystemExit(__doc__)
    # No logging setup here: evolve_ondella.main() installs its loop.log handler.
    run_reference(Path(sys.argv[2]), sys.argv[3:])


if __name__ == "__main__":
    main()
