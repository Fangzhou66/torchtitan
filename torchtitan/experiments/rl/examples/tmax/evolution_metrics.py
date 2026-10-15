# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run-scoped evolution flow, sampled at each training step.

Every value is cumulative for this training run, across its controller
attempts (``runs/<name>--attempt-*`` hold each lifetime's signals and outcomes;
``runs/<name>/trainer`` holds the lineage of all of them), and counted when the
trainer observes it. Three of them form a cumulative flow diagram over training steps:
`signals_issued` (the trainer wrote the signal), `signals_consumed` (the loop
wrote its ledger decision) and `rewrites_accepted` (a rewrite was accepted,
which means folded into the mix). At one step, the vertical gap between issued
and consumed is the work still open; at one height, the horizontal gap is how
many steps a signal waited. `accept_latency_steps` states that wait for
accepted rewrites. `rewrites_trained` counts accepted rewrites whose new
revision has been trained on at least once, and `train_latency_steps` is the
wait from a signal's issue to that first training step: an accepted revision
reaches training only when the epoch order next reaches its task.

`stale/*` counts the groups trained on a task's original while its rewrite was
pending: claimed after one of this run's signals asked to rewrite that exact
revision and before the loop consumed the signal. A group claimed before the
rewrite started counts as `while_waiting`, after it as `while_rewriting`.
Each signal is classified when it is consumed, so the counts only grow.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

from torchtitan.experiments.rl.examples.tmax import layout


SIGNAL_OUTCOMES = ("handled", "deferred", "superseded", "resolved", "junk")
REWRITE_OUTCOMES = (
    "accepted_harder", "accepted_easier", "accepted_repair_spec", "kept", "rejected", "failed"
)
REWRITE_ROUTES = ("evolve", "simplify", "repair_spec", "invalid_signal")
FLOW_KEYS = ("signals_issued", "signals_consumed", "rewrites_accepted", "rewrites_trained")
# Merges the latency median looks back over: recent enough to move when the
# loop speeds up or stalls, wide enough that one outlier does not set it.
LATENCY_WINDOW = 20


def record_outcome(root: layout.Root, rewrite: layout.RewriteDir, meta: dict) -> None:
    if meta.get("dry") or not meta.get("signal"):
        return
    run_name = meta["signal"].split("/", 1)[0]
    layout.append_jsonl(
        root.evolution.run_outcomes(run_name),
        {
            "rewrite": str(rewrite.path.relative_to(root.path)),
            "task": meta["task"],
            "direction": meta["job"],
            "status": meta["status"],
            "route": meta.get("route") or (
                "repair_spec" if meta.get("spec_repair") else
                "invalid_signal" if meta.get("stage") == "invalid_signal" else
                "simplify" if meta["job"] == "easier" else "evolve"
            ),
            "stage": meta.get("stage"),
            "started": meta.get("started"),
            "finished": meta.get("finished"),
            "signal": meta["signal"],
        },
    )


def _new_lines(path: Path, offset: int) -> tuple[list[dict], int]:
    """Complete JSON lines appended after ``offset``; a line the writer is
    still appending is left for the next poll."""
    if not path.exists():
        return [], offset
    events = []
    with path.open("rb") as stream:
        stream.seek(offset)
        while line := stream.readline():
            if not line.endswith(b"\n"):
                break
            events.append(json.loads(line))
            offset = stream.tell()
    return events, offset


class EvolutionMetrics:
    def __init__(self, run: layout.Run, root: layout.Root) -> None:
        self.run = run
        self.base = run.base
        self.root = root
        self.claim_offset = 0
        self.ledger_offset = 0
        self.outcome_offsets: dict[str, int] = {}
        # The policy version at group claim is the step a signal was issued at.
        # Group ids restart with every controller attempt, so a claim is keyed by
        # its lineage session; (None, group) is the latest claim of that group,
        # for a run without attempts.
        self.claimed_origin: dict[tuple[str | None, int], int] = {}
        self.issued: dict[str, int] = {}
        self.issued_by_direction: Counter[str] = Counter()
        self.closed: dict[str, str] = {}
        self.rewrites: set[str] = set()
        self.outcomes: Counter[str] = Counter()
        self.routes: dict[str, Counter[str]] = {route: Counter() for route in REWRITE_ROUTES}
        self.latencies: list[int] = []
        # Why rewrites ended where they did, and how long each took.
        self.stages: Counter[tuple[str, str]] = Counter()
        self.rewrite_secs: list[float] = []
        self.history: dict[int, tuple[int, ...]] = {}
        # First training on an accepted revision: trained groups by task, and
        # accepted rewrites still waiting for one.
        self.trained: dict[str, list[tuple[float, str, int]]] = {}
        self.awaiting_training: dict[str, tuple[str, str, float, int]] = {}
        self.train_latencies: list[int] = []
        # Staleness: every claim by task, each signal's revision and lifetime.
        self.claims: dict[str, list[tuple[float, str, str]]] = {}
        self.group_revision: dict[tuple[str | None, int], str] = {}
        self.signal_span: dict[str, tuple[str, str, float]] = {}
        self.closed_at: dict[str, float] = {}
        self.rewrite_started: dict[str, float] = {}
        self.classified: set[str] = set()
        self.stale_claims: set[str] = set()
        self.stale: Counter[str] = Counter()

    def _poll_claims(self) -> None:
        events, self.claim_offset = _new_lines(
            self.base.trainer / "training_lineage/events.jsonl", self.claim_offset
        )
        for event in events:
            if event["event"] == "trained" and "sample_revision" in event:
                self.trained.setdefault(event["task_id"], []).append(
                    (
                        event["time_unix_ns"] / 1e9,
                        event["sample_revision"],
                        event["train_step"],
                    )
                )
            if event["event"] == "claimed":
                keys = ((event.get("session_id"), event["group_id"]), (None, event["group_id"]))
                for key in keys:
                    self.claimed_origin[key] = event["generator_policy_version"]
                # Lineage written before claims carried their sample is skipped.
                if "sample_revision" in event and "task_id" in event:
                    for key in keys:
                        self.group_revision[key] = event["sample_revision"]
                    self.claims.setdefault(event["task_id"], []).append(
                        (
                            event["time_unix_ns"] / 1e9,
                            event["sample_revision"],
                            event["occurrence_id"],
                        )
                    )

    def _poll_signals(self) -> None:
        for attempt in self.base.attempts():
            for path in attempt.signal_files():
                self._poll_signal(attempt, path)

    def _poll_signal(self, attempt: layout.Run, path: Path) -> None:
        identity = f"{attempt.name}/{path.stem}"
        if identity in self.issued:
            return
        signal = json.loads(path.read_text())
        key = (attempt.session_id, signal["group"])
        origin = self.claimed_origin.get(key)
        # Counted once its claim is visible, so its issue step is known.
        if origin is not None:
            self.issued[identity] = origin
            self.issued_by_direction[signal.get("direction") or "unknown"] += 1
            revision = self.group_revision.get(key)
            if revision is not None and signal.get("created"):
                self.signal_span[identity] = (
                    signal["task"],
                    revision,
                    layout.parse_stamp(signal["created"]),
                )

    def _poll_ledger(self) -> None:
        events, self.ledger_offset = _new_lines(
            self.root.evolution.ledger, self.ledger_offset
        )
        for event in events:
            signal = event.get("signal", "")
            if self._ours(signal):
                self.closed[signal] = event["outcome"]
                if event.get("stamp"):
                    self.closed_at.setdefault(signal, layout.parse_stamp(event["stamp"]))

    def _ours(self, signal: str) -> bool:
        """A signal id names the attempt directory that issued it."""
        name, slash, _ = signal.partition("/")
        return bool(slash) and layout.Run(self.base.path.with_name(name)).base == self.base

    def _poll_outcomes(self, step: int | None) -> None:
        events = []
        for attempt in self.base.attempts():
            new, self.outcome_offsets[attempt.name] = _new_lines(
                self.root.evolution.run_outcomes(attempt.name),
                self.outcome_offsets.get(attempt.name, 0),
            )
            events += new
        for event in events:
            if event["rewrite"] in self.rewrites:
                continue
            self.rewrites.add(event["rewrite"])
            if event.get("started") and event.get("signal"):
                started = layout.parse_stamp(event["started"])
                # A retried signal waited until its first attempt began.
                self.rewrite_started[event["signal"]] = min(
                    started, self.rewrite_started.get(event["signal"], started)
                )
            status = event["status"]
            self.stages[("failed" if status == "interrupted" else status, event.get("stage") or "none")] += 1
            if event.get("started") and event.get("finished"):
                try:
                    self.rewrite_secs.append(
                        layout.parse_stamp(event["finished"]) - layout.parse_stamp(event["started"])
                    )
                except (TypeError, ValueError):
                    pass
            route = event.get("route") or ("simplify" if event["direction"] == "easier" else "evolve")
            if route in self.routes:
                self.routes[route]["attempted"] += 1
                self.routes[route]["failed" if status == "interrupted" else status] += 1
            if status == "accepted":
                outcome = "accepted_repair_spec" if route == "repair_spec" else f"accepted_{event['direction']}"
                self.outcomes[outcome] += 1
                origin = self.issued.get(event.get("signal"))
                if step is not None and origin is not None:
                    self.latencies.append(step - origin)
                span = self.signal_span.get(event.get("signal"))
                if origin is not None and span is not None and event.get("finished"):
                    self.awaiting_training[event["rewrite"]] = (
                        span[0],
                        span[1],
                        layout.parse_stamp(event["finished"]),
                        origin,
                    )
            elif status == "interrupted":
                self.outcomes["failed"] += 1
            else:
                self.outcomes[status] += 1

    def _classify_stale(self) -> None:
        for signal, closed_at in self.closed_at.items():
            if signal in self.classified or signal not in self.signal_span:
                continue
            self.classified.add(signal)
            task, revision, created = self.signal_span[signal]
            started = self.rewrite_started.get(signal, closed_at)
            for claimed_at, claim_revision, occurrence in self.claims.get(task, ()):
                if (
                    claim_revision != revision
                    or not created <= claimed_at <= closed_at
                    or occurrence in self.stale_claims
                ):
                    continue
                self.stale_claims.add(occurrence)
                key = "while_waiting" if claimed_at < started else "while_rewriting"
                self.stale[key] += 1

    def _match_training(self) -> None:
        """An accepted rewrite reaches training at the first trained group of
        its task on a revision other than the one its signal measured."""
        for rewrite, (task, old, accepted_at, origin) in list(
            self.awaiting_training.items()
        ):
            first = min(
                (
                    train_step
                    for trained_at, revision, train_step in self.trained.get(task, ())
                    if revision != old and trained_at >= accepted_at
                ),
                default=None,
            )
            if first is not None:
                self.train_latencies.append(first - origin)
                del self.awaiting_training[rewrite]

    def poll(self, *, step: int | None = None) -> dict[str, float]:
        self._poll_claims()
        self._poll_signals()
        self._poll_ledger()
        self._poll_outcomes(step)
        self._classify_stale()
        self._match_training()
        closed = Counter(o for s, o in self.closed.items() if s in self.issued)
        flow = (
            len(self.issued),
            sum(closed.values()),
            self.outcomes["accepted_harder"] + self.outcomes["accepted_easier"] + self.outcomes["accepted_repair_spec"],
            len(self.train_latencies),
        )
        if step is not None:
            self.history[step] = flow
        values = {
            **{f"evolution/flow/{k}": float(v) for k, v in zip(FLOW_KEYS, flow, strict=True)},
            **{f"evolution/signals/{o}": float(closed[o]) for o in SIGNAL_OUTCOMES},
            **{
                f"evolution/rewrites/{o}": float(self.outcomes[o])
                for o in REWRITE_OUTCOMES
            },
            **{
                f"evolution/stale/{k}": float(self.stale[k])
                for k in ("while_waiting", "while_rewriting")
            },
            **{
                f"evolution/{route}/{outcome}": float(counts[outcome])
                for route, counts in self.routes.items()
                for outcome in ("attempted", "accepted", "kept", "rejected", "failed")
            },
        }
        for direction, count in self.issued_by_direction.items():
            values[f"evolution/flow/signals_issued_{direction}"] = float(count)
        for (status, stage), count in self.stages.items():
            values[f"evolution/rewrites_by_stage/{status}/{stage}"] = float(count)
        if self.rewrite_secs:
            values["evolution/flow/rewrite_secs"] = float(
                statistics.median(self.rewrite_secs[-LATENCY_WINDOW:])
            )
        for route, counts in self.routes.items():
            if counts["attempted"]:
                values[f"evolution/{route}/acceptance_rate"] = counts["accepted"] / counts["attempted"]
        if self.latencies:
            values["evolution/flow/accept_latency_steps"] = float(
                statistics.median(self.latencies[-LATENCY_WINDOW:])
            )
        if self.train_latencies:
            values["evolution/flow/train_latency_steps"] = float(
                statistics.median(self.train_latencies[-LATENCY_WINDOW:])
            )
        return values

    def flow_series(self) -> tuple[list[int], list[list[int]], list[str]]:
        """The flow counts at every logged step, for one overlay chart."""
        steps = sorted(self.history)
        return (
            steps,
            [[self.history[s][i] for s in steps] for i in range(len(FLOW_KEYS))],
            list(FLOW_KEYS),
        )
