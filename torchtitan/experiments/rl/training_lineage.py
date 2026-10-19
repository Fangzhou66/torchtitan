# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Append-only audit records for the asynchronous RL training-data path."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import queue
import re
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from torchtitan.experiments.rl.types import SampleLineage


_SCHEMA_VERSION = 1
# Training group ids in canonical_json events; validation ids are negative.
_TRAINING_GROUP_ID = re.compile(rb'"group_id":(\d+)')


def canonical_json(value: object) -> str:
    """Return the stable JSON representation used for content revisions."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def content_revision(value: object) -> str:
    """SHA-256 of a complete JSON-serializable input."""
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def sample_payload(sample: object) -> object:
    """Serialize the complete environment input while excluding occurrence metadata."""
    if dataclasses.is_dataclass(sample):
        payload: object = dataclasses.asdict(sample)
    elif isinstance(sample, dict):
        payload = dict(sample)
    else:
        payload = {"type": type(sample).__qualname__, "repr": repr(sample)}
    if isinstance(payload, dict):
        payload.pop("lineage", None)
    return payload


def max_recorded_group_id(events_path: Path) -> int:
    """Highest non-negative ``group_id`` in an events file, or -1 if none."""
    best = -1
    if not events_path.exists():
        return best
    tail = b""
    with events_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            data = tail + chunk
            cut = data.rfind(b"\n") + 1
            tail = data[cut:]
            for match in _TRAINING_GROUP_ID.finditer(data, 0, cut):
                best = max(best, int(match.group(1)))
    for match in _TRAINING_GROUP_ID.finditer(tail):
        best = max(best, int(match.group(1)))
    return best


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class TrainingLineageRecorder:
    """Durable JSONL source of truth for sample identity and lifecycle events.

    Each call opens and flushes one append. A controller crash can lose at most a
    partially written final line; prior events and the deduplicated sample catalog
    remain intact and are reused when the same output directory is resumed.
    """

    def __init__(self, *, dump_dir: str, track_pending_inputs: bool = False) -> None:
        self.root = Path(dump_dir) / "training_lineage"
        self.root.mkdir(parents=True, exist_ok=True)
        self.events_path = self.root / "events.jsonl"
        self.samples_path = self.root / "samples.jsonl"
        self.optimization_complete_path = self.root / "optimization_complete.json"
        self.training_activity_path = self.root / "training_activity.json"
        self.session_id = uuid.uuid4().hex
        self._sequence = 0
        self.track_pending_inputs = track_pending_inputs
        self.pending_inputs: dict[int, object] = {}
        self._known_sample_revisions = self._read_known_sample_revisions()
        # First training group id no earlier session of this run has used, so
        # a resumed run never reissues ids recorded after its checkpoint.
        self.next_unused_group_id = max_recorded_group_id(self.events_path) + 1
        # Background writer (enable_background_writes); None appends synchronously.
        self._write_queue: queue.Queue | None = None
        self._writer: threading.Thread | None = None
        self._writer_error: BaseException | None = None
        self.record_event("session_started", pid=os.getpid())

    def _read_known_sample_revisions(self) -> set[str]:
        revisions: set[str] = set()
        if not self.samples_path.exists():
            return revisions
        with self.samples_path.open() as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                revision = record.get("sample_revision")
                if isinstance(revision, str):
                    revisions.add(revision)
        return revisions

    def _append(self, path: Path, record: dict[str, Any]) -> None:
        line = canonical_json(record) + "\n"
        if self._write_queue is None:
            with path.open("a") as f:
                f.write(line)
                f.flush()
            return
        if self._writer_error is not None:
            raise RuntimeError("training lineage writer failed") from self._writer_error
        self._write_queue.put((path, line))

    def enable_background_writes(self) -> None:
        """Append from a background thread instead of the caller's thread.

        Lines are serialized when recorded, so file contents and order are the same
        as synchronous appends; each batch the thread drains opens a file once.
        Records still queued at a crash are lost (synchronous mode loses at most a
        partial line). ``flush`` waits for the queue; ``close`` also stops the thread.
        """
        if self._write_queue is not None:
            return
        self._write_queue = queue.Queue()
        self._writer = threading.Thread(
            target=self._write_loop, name="training-lineage-writer", daemon=True
        )
        self._writer.start()

    def _write_loop(self) -> None:
        assert self._write_queue is not None
        while True:
            items = [self._write_queue.get()]
            while True:
                try:
                    items.append(self._write_queue.get_nowait())
                except queue.Empty:
                    break
            stop = any(item is None for item in items)
            batches: dict[Path, list[str]] = {}
            for item in items:
                if item is not None:
                    batches.setdefault(item[0], []).append(item[1])
            try:
                for path, lines in batches.items():
                    with path.open("a") as f:
                        f.write("".join(lines))
                        f.flush()
            except BaseException as error:  # surfaced by the next _append/flush
                self._writer_error = error
            finally:
                for _ in items:
                    self._write_queue.task_done()
            if stop:
                return

    def flush(self) -> None:
        """Block until every queued record is written (no-op when synchronous)."""
        if self._write_queue is not None:
            self._write_queue.join()
        if self._writer_error is not None:
            raise RuntimeError("training lineage writer failed") from self._writer_error

    def close(self) -> None:
        """Write everything queued and stop the background writer."""
        if self._write_queue is None:
            return
        self._write_queue.put(None)
        if self._writer is not None:
            self._writer.join()
        self._write_queue = None
        self._writer = None
        if self._writer_error is not None:
            raise RuntimeError("training lineage writer failed") from self._writer_error

    def describe_sample(self, *, sample: object, group_id: int) -> dict[str, Any]:
        """Return a stable occurrence identity, with a generic fallback."""
        declared = getattr(sample, "lineage", None)
        if isinstance(declared, SampleLineage):
            lineage = dataclasses.asdict(declared)
        elif dataclasses.is_dataclass(declared):
            lineage = dataclasses.asdict(declared)
        elif isinstance(declared, dict):
            lineage = dict(declared)
        else:
            payload = sample_payload(sample)
            task_id = (
                getattr(sample, "instance_id", None)
                or getattr(sample, "id", None)
                or (sample.get("instance_id") if isinstance(sample, dict) else None)
                or f"group-{group_id}"
            )
            lineage = {
                "occurrence_id": f"{self.session_id}:{group_id}",
                "task_id": str(task_id),
                "sample_revision": content_revision(payload),
                "mix_revision": None,
                "dataset_epoch": None,
                "dataset_position": None,
                "stream_position": group_id,
                "stream_id": self.session_id,
            }
        lineage["group_id"] = group_id
        return lineage

    def record_sample(self, *, lineage: dict[str, Any], sample: object) -> None:
        """Store a complete input once per content revision."""
        if self.track_pending_inputs:
            self.pending_inputs[lineage["group_id"]] = sample
        revision = lineage.get("sample_revision")
        if not isinstance(revision, str) or revision in self._known_sample_revisions:
            return
        self._append(
            self.samples_path,
            {
                "schema_version": _SCHEMA_VERSION,
                "record_type": "sample",
                "sample_revision": revision,
                "task_id": lineage.get("task_id"),
                "recorded_at": _utc_now(),
                "input": sample_payload(sample),
            },
        )
        self._known_sample_revisions.add(revision)

    def record_event(
        self,
        event: str,
        *,
        lineage: dict[str, Any] | None = None,
        **fields: Any,
    ) -> None:
        """Append one timestamped lifecycle or dataset event."""
        self._sequence += 1
        record = {
            "schema_version": _SCHEMA_VERSION,
            "record_type": "event",
            "event": event,
            "timestamp": _utc_now(),
            "time_unix_ns": time.time_ns(),
            "session_id": self.session_id,
            "sequence": self._sequence,
        }
        if lineage:
            record.update(lineage)
            if self.track_pending_inputs and event in {"trained", "dropped"}:
                self.pending_inputs.pop(lineage["group_id"], None)
        record.update(fields)
        self._append(self.events_path, record)

    def record_dataset_event(self, event: dict[str, Any]) -> None:
        """Append an event emitted by a hot-reloadable dataset."""
        payload = dict(event)
        event_name = str(payload.pop("event"))
        self.record_event(event_name, **payload)

    def record_optimization_complete(
        self, *, policy_version: int, requested_training_steps: int
    ) -> None:
        """Atomically publish that the optimizer will perform no more steps."""
        self.flush()
        record = {
            "schema_version": _SCHEMA_VERSION,
            "record_type": "optimization_complete",
            "event": "optimization_complete",
            "timestamp": _utc_now(),
            "time_unix_ns": time.time_ns(),
            "session_id": self.session_id,
            "policy_version": policy_version,
            "requested_training_steps": requested_training_steps,
        }
        self._atomic_snapshot(self.optimization_complete_path, record)

    def record_training_activity(self, *, phase: str, policy_version: int) -> None:
        """Heartbeat for consumers that must pause during controller restarts."""
        self._atomic_snapshot(
            self.training_activity_path,
            {
                "session_id": self.session_id,
                "phase": phase,
                "policy_version": policy_version,
                "time_unix_ns": time.time_ns(),
            },
        )

    def _atomic_snapshot(self, path: Path, record: dict[str, Any]) -> None:
        temporary = self.root / f".{path.name}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("x") as output:
                output.write(canonical_json(record) + "\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
