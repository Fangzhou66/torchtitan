# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The overfit bench's version pool: what training draws from.

A version is one task row, identified by its content sha (``content_revision``
of the row, which is also the lineage's ``sample_revision``); ``task@r<rev>``
is only its display label, so two different rows of one task at one rev are
two versions. Each carries a weight, its parent version, when it was added
and whether it is still active. Training draws an active version with
probability proportional to its weight (the scheduler may ask the rewriter
first). Weight, parent and status live beside the row, never in it, so they
do not move the sha.

Every change is one line in ``pool.jsonl`` (the run's ``scheduler/``), the
whole pool is rewritten to ``data/mix/pool.json``, and a change to the active
set republishes the active rows as the next mix version, so ``live.jsonl``,
``bench.py readout`` and any file reader keep seeing what training draws from.
"""

from __future__ import annotations

import json
import math
import random
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from torchtitan.experiments.rl.examples.tmax import layout
from torchtitan.experiments.rl.examples.tmax.data import _parse_sample_row
from torchtitan.experiments.rl.training_lineage import content_revision

ACTIVE = "active"
RETIRED = "retired"


@dataclass(kw_only=True)
class Version:
    id: str
    task: str
    rev: int
    row: dict
    weight: float
    parent: str | None
    added_at: str
    status: str = ACTIVE

    @property
    def label(self) -> str:
        return f"{self.task}@r{self.rev}"

    def info(self, *, with_row: bool = False) -> dict:
        out = asdict(self)
        if not with_row:
            del out["row"]
        out["label"] = self.label
        return out


class PoolError(ValueError):
    """A pool call the pool refuses (unknown version, bad weight, bad row)."""


def _weight(value) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError):
        raise PoolError(f"weight must be a number, got {value!r}") from None
    if not math.isfinite(weight) or weight < 0:
        raise PoolError(f"weight must be finite and >= 0, got {value!r}")
    return weight


class VersionPool:
    """Thread-safe: the scheduler's socket threads and the draw thread share it."""

    def __init__(
        self,
        *,
        log_path: Path | None = None,
        state_path: Path | None = None,
        publish: Callable[[list[str]], None] | None = None,
        on_drained: Callable[[str], None] | None = None,
    ) -> None:
        self._versions: dict[str, Version] = {}
        self._seq = 0
        self._lock = threading.RLock()
        self._log_path = log_path
        self._state_path = state_path
        self._publish = publish
        self._on_drained = on_drained

    # -- mutations ---------------------------------------------------------

    def add(
        self,
        row: dict,
        *,
        weight: float = 1.0,
        parent: str | None = None,
        source: str = "client",
        publish: bool = True,
    ) -> dict:
        """Add a row as an active version. Adding an active version again
        changes nothing (so the socket and the file importer can both report
        one fold); a retired one is made active again at ``weight``.

        The row must parse the way the trainer parses it. ``parent`` must be a
        version this pool knows (active or retired), or None."""
        if not isinstance(row, dict):
            raise PoolError("row must be a JSON object")
        try:
            sample = _parse_sample_row(row)
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            raise PoolError(f"row does not parse: {e}") from e
        weight = _weight(weight)
        vid = content_revision(row)
        with self._lock:
            if parent is not None and parent not in self._versions:
                raise PoolError(f"unknown parent version {parent}")
            before = self._task_weights()
            existing = self._versions.get(vid)
            if existing is not None and existing.status == ACTIVE:
                return existing.info()
            if existing is not None:
                existing.status = ACTIVE
                existing.weight = weight
                version = existing
            else:
                version = Version(
                    id=vid,
                    task=sample.instance_id,
                    rev=sample.rev,
                    row=row,
                    weight=weight,
                    parent=parent,
                    added_at=layout.stamp(),
                )
                self._versions[vid] = version
            self._changed("add", version, source=source, before=before, publish=publish)
            return version.info()

    def reweight(self, version: str, weight: float, *, source: str = "client") -> dict:
        weight = _weight(weight)
        with self._lock:
            v = self._active(version)
            before = self._task_weights()
            v.weight = weight
            self._changed("reweight", v, source=source, before=before, publish=False)
            return v.info()

    def retire(
        self, version: str, *, source: str = "client", publish: bool = True
    ) -> dict:
        with self._lock:
            v = self._versions.get(version)
            if v is None:
                raise PoolError(f"unknown version {version}")
            if v.status == RETIRED:
                return v.info()
            before = self._task_weights()
            v.status = RETIRED
            self._changed("retire", v, source=source, before=before, publish=publish)
            return v.info()

    def reconcile(self, rows: list[dict], *, source: str) -> dict:
        """Make the active set exactly ``rows`` (a file-based publish names the
        whole mix). A new version's parent is the version of its task that the
        same publish retires, and it inherits that parent's weight. Every row
        is checked before anything changes; nothing is republished, since the
        file already says it."""
        parsed = []
        for i, row in enumerate(rows, 1):
            try:
                parsed.append((content_revision(row), _parse_sample_row(row), row))
            except (ValueError, TypeError, AttributeError, KeyError) as e:
                raise PoolError(f"row {i} does not parse: {e}") from e
        wanted = {vid for vid, _, _ in parsed}
        with self._lock:
            leaving = [
                v for v in self._versions.values() if v.status == ACTIVE and v.id not in wanted
            ]
            added = retired = 0
            for vid, sample, row in parsed:
                current = self._versions.get(vid)
                if current is not None and current.status == ACTIVE:
                    continue
                parents = [v for v in leaving if v.task == sample.instance_id]
                parent = parents[-1] if parents else None
                self.add(
                    row,
                    weight=parent.weight if parent is not None else 1.0,
                    parent=parent.id if parent is not None else None,
                    source=source,
                    publish=False,
                )
                added += 1
            for v in leaving:
                self.retire(v.id, source=source, publish=False)
                retired += 1
            return {"added": added, "retired": retired}

    # -- reads -------------------------------------------------------------

    def get(self, version: str) -> Version:
        with self._lock:
            v = self._versions.get(version)
            if v is None:
                raise PoolError(f"unknown version {version}")
            return v

    def active(self) -> list[Version]:
        with self._lock:
            return [v for v in self._versions.values() if v.status == ACTIVE]

    def is_active(self, version: str) -> bool:
        with self._lock:
            v = self._versions.get(version)
            return v is not None and v.status == ACTIVE

    def get_state(self, *, with_rows: bool = False) -> dict:
        with self._lock:
            return {
                "seq": self._seq,
                "versions": [v.info(with_row=with_rows) for v in self._versions.values()],
            }

    def rows(self) -> list[str]:
        """The active rows as JSON lines, in the order they were added."""
        with self._lock:
            return [json.dumps(v.row, ensure_ascii=False) for v in self.active()]

    def revision(self) -> str:
        """One sha for the active set and its weights (the lineage's mix_revision)."""
        with self._lock:
            return content_revision(sorted((v.id, v.weight) for v in self.active()))

    def sample(self, rng: random.Random) -> tuple[Version, str]:
        """An active version, proportional to weight, and how it was chosen.

        Training never waits on the pool: with every weight at 0 it draws
        uniformly among the active versions ("zero_weight"), and with none
        active it reuses the most recently retired one ("no_active")."""
        with self._lock:
            active = self.active()
            weighted = [v for v in active if v.weight > 0]
            if weighted:
                return rng.choices(weighted, weights=[v.weight for v in weighted])[0], "weighted"
            if active:
                return rng.choice(active), "zero_weight"
            if not self._versions:
                raise PoolError("the version pool is empty")
            return list(self._versions.values())[-1], "no_active"

    # -- checkpoint --------------------------------------------------------

    @property
    def seq(self) -> int:
        return self._seq

    def state_dict(self) -> dict:
        with self._lock:
            return {
                "seq": self._seq,
                "versions": [v.info(with_row=True) for v in self._versions.values()],
            }

    def load_state_dict(self, state: dict) -> None:
        with self._lock:
            self._versions = {}
            for info in state["versions"]:
                fields = {k: info[k] for k in Version.__dataclass_fields__}
                self._versions[info["id"]] = Version(**fields)
            self._seq = int(state["seq"])
            self._write_state()

    # -- internals ---------------------------------------------------------

    def _active(self, version: str) -> Version:
        v = self._versions.get(version)
        if v is None or v.status != ACTIVE:
            raise PoolError(f"{version} is not an active version")
        return v

    def _task_weights(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for v in self._versions.values():
            out.setdefault(v.task, 0.0)
            if v.status == ACTIVE:
                out[v.task] += v.weight
        return out

    def _changed(
        self, op: str, v: Version, *, source: str, before: dict[str, float], publish: bool
    ) -> None:
        self._seq += 1
        if self._log_path is not None:
            layout.append_jsonl(
                self._log_path,
                {
                    "stamp": layout.stamp(),
                    "seq": self._seq,
                    "op": op,
                    "source": source,
                    "version": v.id,
                    "label": v.label,
                    "task": v.task,
                    "rev": v.rev,
                    "weight": v.weight,
                    "parent": v.parent,
                    "status": v.status,
                },
            )
        self._write_state()
        if publish and op in ("add", "retire") and self._publish is not None:
            self._publish(self.rows())
        if self._on_drained is not None:
            after = self._task_weights()
            if before.get(v.task, 0.0) > 0 and after.get(v.task, 0.0) == 0:
                self._on_drained(v.task)

    def _write_state(self) -> None:
        if self._state_path is not None:
            layout.write_json_atomic(self._state_path, self.state_dict())
