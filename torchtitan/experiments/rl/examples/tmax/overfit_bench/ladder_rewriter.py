#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The ladder rewriter: several difficulty versions of one task, side by side.

    ladder_rewriter.py <job> [--workers 16] [--help-levels none,vague,specific]

A reference rewriter on the scheduler socket, built from the evolve loop's own
parts: every rewrite is evolve_ondella.handle (the same sessions, oracle,
probes, verifier, rewrite.json, sessions and traces), every finished rewrite
is closed with its lineage, outcome and ledger lines, and loop.log, status.json
and the lineage snapshot are kept as the loop keeps them. What differs is the
policy, which is this file's and not the bench's:

- A trainer signal about a version (``version`` in the signal file) is handled
  against that version, not against "the task's current revision". An
  all-failed group starts one easier rewrite per help level, in parallel;
  an all-solved group (the trainer's SWE_EVOLUTION_HARDER_RATIO) starts one
  harder rewrite, the evolve loop's withdraw-help hardening.
- The help levels are the simplify prompt's own (simplify_operators.prompt):
  none, vague and specific, small to large help. Each rewrite of one signal
  carries its level in the signal (``simplify_hint``), which feedback_loop
  reads before SWE_SIMPLIFY_HINT.
- Every child that passes validation becomes its own version, with the
  version it was rewritten from as parent, at a fresh revision r<max+1> of the
  task. Nothing is retired on a fold: the parent stays in the pool.
- One request per version at a time, and none for a version that already has
  an active child in that direction; any number of versions of one task may
  be rewritten at once, up to --workers rewrites in all.
- Weights, recomputed from the version's training groups at every
  group_done: p(1-p) for p the solve rate of its last --history groups
  (smoothed by --prior on both sides), --explore for a version with no group
  yet, times --harder-parent-factor while the version has an active harder
  child, and never below --original-floor for the original (rev 0). A
  version other than the original is retired once it has an active harder
  child and its last --retire-after groups since that child was folded each
  solved at least --mastered of their scored attempts.

It reads what any rewriter may read -- signals, training rollout records, the
scheduler's events and the pool -- and never the readout: the attempt index
the author's record is built from is limited to training records
(evolve_ondella.ATTEMPT_RECORD_DIRS). It runs no student rollout. It needs the
scheduler (it waits for it, as RewriterClient.connect does), and it takes the
evolve loop's lock, so it never runs beside the reference rewriter on one job.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[6]))

from torchtitan.experiments.rl.examples.tmax import layout  # noqa: E402
from torchtitan.experiments.rl.examples.tmax.overfit_bench.rewriter_client import (  # noqa: E402
    RewriterClient,
    RewriterError,
)

log = logging.getLogger("ladder")

EASIER, HARDER = "easier", "harder"
# simplify_operators.prompt's levels, from the least help to the most.
HELP_LEVELS = ("none", "vague", "specific")


@dataclass(frozen=True, kw_only=True)
class Policy:
    help_levels: tuple[str, ...] = HELP_LEVELS
    history: int = 3
    prior: float = 0.5
    explore: float = 0.25
    original_floor: float = 0.05
    harder_parent_factor: float = 0.5
    mastered: float = 0.9
    retire_after: int = 2


# -- the policy, as functions of what training measured ---------------------


def is_original(version: dict) -> bool:
    """The job's own task: the row training started from is revision 0."""
    return int(version["rev"]) == 0


def solve_rate(groups: list[tuple], policy: Policy) -> float | None:
    """The smoothed solve rate of the last ``history`` (passed, scored, ...)
    groups, None before the first."""
    recent = groups[-policy.history :]
    if not recent:
        return None
    passed = sum(g[0] for g in recent)
    scored = sum(g[1] for g in recent)
    return (passed + policy.prior) / (scored + 2 * policy.prior)


def weight(
    version: dict,
    groups: list[tuple],
    *,
    has_harder_child: bool,
    policy: Policy,
) -> float:
    p = solve_rate(groups, policy)
    w = policy.explore if p is None else p * (1 - p)
    if has_harder_child:
        w *= policy.harder_parent_factor
    if is_original(version):
        w = max(w, policy.original_floor)
    return w


def mastered(groups: list[tuple], policy: Policy) -> bool:
    """The last ``retire_after`` (passed, scored, ...) groups each solved at
    least ``mastered``."""
    recent = groups[-policy.retire_after :]
    return len(recent) == policy.retire_after and all(
        g[1] > 0 and g[0] / g[1] >= policy.mastered for g in recent
    )


# -- the loop ----------------------------------------------------------------


@dataclass
class Request:
    """One signal's rewrites of one version: K easier, or one harder."""

    parent: str
    label: str
    task: str
    direction: str
    signal: str
    pending: int
    added: list[str] = field(default_factory=list)
    junk_written: bool = False


class Ladder:
    """The coordinator. ``ev`` is evolution/evolve_ondella (a stand-in in tests)."""

    def __init__(
        self,
        root: layout.Root,
        client: RewriterClient,
        ev,
        *,
        policy: Policy = Policy(),
        workers: int = 16,
    ) -> None:
        self.root, self.client, self.ev, self.policy = root, client, ev, policy
        # The readout is not an input: the record each rewrite's author reads
        # indexes training attempts only.
        ev.ATTEMPT_RECORD_DIRS = ("rollouts",)
        # version -> [(passed, scored, stamp)], oldest first.
        self.groups: dict[str, list[tuple[int, int, str]]] = collections.defaultdict(list)
        self.latest_group: dict[tuple[str, str | None], int] = {}
        self._seen_groups: set[tuple] = set()
        # parent version -> [(child version, direction, how many of the
        # parent's groups were recorded when the child was folded)].
        self.children: dict[str, list[tuple[str, str, int]]] = collections.defaultdict(list)
        self.inflight: dict[str, Request] = {}
        self.taken: set[str] = set()
        self.versions: dict[str, dict] = {}
        self._ex = ThreadPoolExecutor(workers, thread_name_prefix="ladder")
        self._futs: dict[Future, tuple[Request, object, str | None]] = {}

    # -- state ----------------------------------------------------------

    def start(self) -> None:
        """The pool, the training groups so far, and the children folded so far."""
        self.refresh(with_rows=True)
        for run in self.root.run_dirs():
            for event in layout.read_jsonl(run.scheduler / "events.jsonl"):
                if event.get("event") == "group_done":
                    self.on_group_done(event, rebalance=False)
        for _task, _rw, meta in self.ev._rewrite_metas(self.root):
            ladder = meta.get("ladder") or {}
            if ladder.get("version") and ladder.get("parent"):
                # The parent's groups up to the fold's second came before it.
                folded = meta.get("finished") or ""
                before = sum(1 for g in self.groups[ladder["parent"]] if g[2] <= folded)
                self.children[ladder["parent"]].append((ladder["version"], meta["job"], before))
        for task in sorted({v["task"] for v in self.versions.values()}):
            self.rebalance(task)

    def refresh(self, *, with_rows: bool = False) -> None:
        state = self.client.get_state(with_rows=with_rows)
        rows = {vid: v.get("row") for vid, v in self.versions.items()}
        self.versions = {v["id"]: v for v in state["versions"]}
        if not with_rows:
            for vid, v in self.versions.items():
                if rows.get(vid) is not None:
                    v["row"] = rows[vid]

    def _active(self, vid: str) -> bool:
        v = self.versions.get(vid)
        return v is not None and v["status"] == "active"

    def _has_active_child(self, vid: str, direction: str) -> bool:
        return self._active_child_since(vid, direction) is not None

    def _active_child_since(self, vid: str, direction: str) -> int | None:
        """How many of ``vid``'s groups came before its earliest active child
        in ``direction``, None without one."""
        stamps = [
            since for c, d, since in self.children.get(vid, ())
            if d == direction and self._active(c)
        ]
        return min(stamps) if stamps else None

    def on_group_done(self, event: dict, *, rebalance: bool = True) -> None:
        vid = event.get("version")
        if not vid:
            return
        key = (event.get("run"), vid, event.get("group"))
        if key in self._seen_groups:
            return
        self._seen_groups.add(key)
        if isinstance(event.get("group"), int):
            at = (vid, event.get("run"))
            self.latest_group[at] = max(self.latest_group.get(at, event["group"]), event["group"])
        if event.get("scored"):
            self.groups[vid].append(
                (int(event["passed"]), int(event["scored"]), str(event.get("stamp") or ""))
            )
        if rebalance:
            self.rebalance(event["task"])

    def rebalance(self, task: str) -> None:
        """Retire what the policy retires, then reweight every active version
        of the task whose weight moved."""
        try:
            self.refresh()
            active = [
                v for v in self.versions.values()
                if v["task"] == task and v["status"] == "active"
            ]
            for v in active:
                since = self._active_child_since(v["id"], HARDER)
                if (
                    since is not None
                    and not is_original(v)
                    and mastered(self.groups[v["id"]][since:], self.policy)
                ):
                    self.client.retire(v["id"])
                    v["status"] = "retired"
                    log.info(
                        "retire %s: last %d groups since its harder child at >= %.2f",
                        v["label"], self.policy.retire_after, self.policy.mastered,
                    )
            for v in active:
                if v["status"] != "active":
                    continue
                w = weight(
                    v,
                    self.groups[v["id"]],
                    has_harder_child=self._has_active_child(v["id"], HARDER),
                    policy=self.policy,
                )
                if abs(w - float(v["weight"])) > 1e-9:
                    self.client.reweight(v["id"], w)
                    log.info("reweight %s: %.4f -> %.4f", v["label"], float(v["weight"]), w)
                    v["weight"] = w
        except RewriterError as e:
            log.warning("rebalance of %s not applied: %s", task, e)

    # -- requests -------------------------------------------------------

    def _version_of(self, data: dict) -> str | None:
        if data.get("version"):
            return str(data["version"])
        # A signal written before the trainer named versions: the one active
        # version of the task at that rev, if there is exactly one.
        match = [
            vid for vid, v in self.versions.items()
            if v["task"] == str(data["task"]) and int(v["rev"]) == int(data["rev"])
            and v["status"] == "active"
        ]
        return match[0] if len(match) == 1 else None

    def judge(self, sig) -> tuple[tuple[str, str] | None, dict | None]:
        """(outcome, reason) for a signal that starts nothing, else the version
        it asks to rewrite."""
        d = sig.data
        direction = d["direction"]
        vid = self._version_of(d)
        v = self.versions.get(vid) if vid else None
        if v is None:
            return ("superseded", f"version {vid} is not in the pool"), None
        if v["status"] != "active":
            return ("superseded", f"{v['label']} is retired"), None
        latest = self.latest_group.get((vid, sig.run.name))
        if latest is not None and sig.group is not None and latest > sig.group:
            return ("superseded", f"{v['label']} has a later group g{latest}"), None
        if vid in self.inflight:
            r = self.inflight[vid]
            return ("superseded", f"a {r.direction} request for {v['label']} is running"), None
        if self._has_active_child(vid, direction):
            return ("superseded", f"{v['label']} already has an active {direction} child"), None
        why = self.ev._defer_reason(self.root, sig)
        if why is not None:
            return ("deferred", why), None
        return None, v

    def dispatch(self) -> int:
        """Start a request for every pending signal the policy takes; close
        the rest in the ledger. Returns the number of rewrites started."""
        ev = self.ev
        if not ev.admission_is_open():
            return 0
        pending = [
            (run, path, sid)
            for run, path, sid in ev._pending(self.root, ev.load_ledger(self.root))
            if sid not in self.taken
        ]
        if not pending:
            return 0
        self.refresh(with_rows=True)
        lines, signals = [], []
        for run, path, sid in pending:
            data, junk = ev.read_signal(path)
            if data is None and junk is None:
                continue  # still being written
            sig = ev.Signal(run, path, sid, data, junk)
            if junk:
                self.taken.add(sid)
                lines.append(ev._ledger_entry(sig, "junk", reason=junk))
                continue
            signals.append(sig)
        started = 0
        # Newest first: of two signals about one version, the newer one runs.
        for sig in sorted(signals, key=lambda s: (str(s.data.get("created", "")), s.path.name), reverse=True):
            closed, version = self.judge(sig)
            if closed is not None:
                self.taken.add(sig.sid)
                log.info("%s %s %s: %s", sig.task, sig.sid, *closed)
                lines.append(ev._ledger_entry(sig, closed[0], reason=closed[1]))
                continue
            started += self.submit(sig, version)
        ev._append_ledger(self.root, lines)
        return started

    def submit(self, sig, version: dict) -> int:
        ev, root = self.ev, self.root
        row, tid = version["row"], version["task"]
        md = row["metadata"]
        # The box, pin hook and protected lists of the version being rewritten,
        # read off its own row (the live mix holds several rows of the task).
        declared = {tid: {k: md[rk] for k, rk in ev.ROW_KEYS.items() if rk in md}}
        hook = ev.row_pretest(md)
        lists = ev.pack.Protected.from_tmax(md.get("tmax") or {})
        direction = sig.data["direction"]
        levels = list(self.policy.help_levels) if direction == EASIER else [None]
        data = dict(sig.data)
        if direction == HARDER:
            data["student_feedback"] = self._student_feedback(data, version)
        req = Request(
            parent=version["id"], label=version["label"], task=tid,
            direction=direction, signal=sig.sid, pending=len(levels),
        )
        self.inflight[req.parent] = req
        self.taken.add(sig.sid)
        if ev.rw_workspace.configured_root(root) is not None:
            ev.ec.rewrite_budget_sec(root, refresh=True)
        for level in levels:
            child = ev.Signal(
                sig.run, sig.path, sig.sid,
                {**data, "simplify_hint": level} if level else data, None,
            )
            fut = self._ex.submit(
                ev.handle, root, child, declared=declared,
                pretests={tid: hook} if hook else {},
                protecteds={tid: lists} if lists is not None else {},
                input_row=row,
            )
            self._futs[fut] = (req, child, level)
        log.info(
            "%s %s: %s %d/%d -> %d %s rewrite(s)%s",
            req.label, sig.sid, direction, sig.data.get("solved"), sig.data.get("total"),
            len(levels), direction, f" at help {','.join(levels)}" if levels != [None] else "",
        )
        return len(levels)

    def _student_feedback(self, data: dict, version: dict) -> dict:
        """What evolve_ondella._attach_student_feedback gives a harder rewrite,
        with the parent version as the previous revision rather than rev - 1."""
        out = {
            "measurement": {k: data[k] for k in ("run", "group", "rev", "solved", "total")},
            "measurement_scope": "One rollout group from the named training run.",
        }
        parent = self.versions.get(version.get("parent") or "")
        if parent is not None:
            path = self.root.evolution.task(version["task"]).rev(int(parent["rev"])) / "instruction.md"
            if path.is_file():
                out["previous_revision"] = {"rev": int(parent["rev"]), "instruction": path.read_text()}
        return out

    # -- finished rewrites ---------------------------------------------

    def collect(self) -> int:
        """Fold and close every finished rewrite. Returns how many closed."""
        done = [f for f in self._futs if f.done()]
        tasks, closed = set(), 0
        for fut in done:
            req, sig, level = self._futs.pop(fut)
            req.pending -= 1
            try:
                h = fut.result()
            except self.ev.NoSeed as e:
                log.warning("junk signal %s: %s", sig.sid, e)
                if not req.junk_written:
                    req.junk_written = True
                    self.ev._ledger_line(self.root, sig, "junk", reason=str(e)[:200])
            except Exception as e:  # noqa: BLE001 -- one rewrite, not the loop
                # No rewrite exists and no ledger line is written: the signal
                # is tried again by the next ladder process, not by this one.
                log.exception("%s: handling failed before a rewrite existed: %s", sig.sid, e)
            else:
                self._settle(req, h, level)
                closed += 1
            if req.pending == 0:
                del self.inflight[req.parent]
                tasks.add(req.task)
                log.info(
                    "%s %s request done: %d version(s) added",
                    req.label, req.direction, len(req.added),
                )
        for task in tasks:
            self.rebalance(task)
        if closed:
            try:
                self.ev.rebuild_status(self.root)
            except Exception as e:  # noqa: BLE001 -- status is a derived snapshot
                log.warning("status rebuild failed: %s: %s", type(e).__name__, e)
            self.ev._snapshot_lineage(self.root, f"ladder: {closed} rewrite(s) closed")
        return closed

    def _settle(self, req: Request, h: dict, level: str | None) -> None:
        meta = h["meta"]
        meta["ladder"] = {"parent": req.parent, "help": level, "version": None}
        if h["status"] == "accepted":
            self._fold(req, h)
        else:
            layout.write_json_atomic(h["rewrite"].meta, meta)
        log.info(
            "%s %s r%s %s help=%s -> %s (%s%s)",
            meta["task"], req.signal, meta["input_rev"], meta["job"], level,
            h["status"], meta.get("stage") or "-",
            f": {meta['reason'][:200]}" if meta.get("reason") else "",
        )
        self.ev._close(self.root, h, dry=False)

    def _fold(self, req: Request, h: dict) -> None:
        """The accepted package as the task's next free revision, and as a new
        version of the pool whose parent stays active."""
        ev, meta = self.ev, h["meta"]
        task = self.root.evolution.task(meta["task"])
        parent_row = self.versions[req.parent]["row"]
        try:
            row = ev._prepare_fold_row(
                h["rewrite"].package, json.dumps(parent_row, ensure_ascii=False), meta, {}
            )
            # Siblings share an input revision, so the next free one, not
            # input_rev + 1. Only this thread folds, so the number is ours.
            to_rev = max(task.revs(), default=0) + 1
            row["metadata"]["rev"] = to_rev
            os.rename(h["rewrite"].package, task.rev(to_rev))
        except Exception as e:  # noqa: BLE001 -- one malformed package, not the loop
            ev._finish(h, "failed", stage="fold", reason=f"{type(e).__name__}: {e}")
            return
        try:
            info = self.client.add(row, weight=self.policy.explore, parent=req.parent)
        except RewriterError as e:
            meta["ladder"]["revision_unused"] = to_rev
            ev._finish(h, "failed", stage="pool", reason=f"r{to_rev} not added to the pool: {e}")
            return
        meta.update({"status": "accepted", "result_rev": to_rev, "finished": layout.stamp()})
        meta["ladder"]["version"] = info["id"]
        layout.write_json_atomic(h["rewrite"].meta, meta)
        h["status"] = "accepted"
        layout.append_jsonl(
            task.lineage,
            {
                "stamp": layout.stamp(),
                "event": "fold",
                "from_rev": meta["input_rev"],
                "to_rev": to_rev,
                "version": info["id"],
                "parent_version": req.parent,
                "help": meta["ladder"]["help"],
                "rewrite": f"rewrites/{h['rewrite'].path.name}",
            },
        )
        self.children[req.parent].append((info["id"], meta["job"], len(self.groups[req.parent])))
        self.versions[info["id"]] = {**info, "row": row}
        req.added.append(info["id"])
        log.info(
            "fold %s: r%s -> r%d (%s, help %s) as %s; parent %s stays active",
            meta["task"], meta["input_rev"], to_rev, meta["job"],
            meta["ladder"]["help"], info["id"][:12], req.label,
        )

    # -- driving ---------------------------------------------------------

    def step(self, timeout: float) -> None:
        """Take the events that arrived (waiting up to ``timeout`` for one),
        then fold what finished and start what is pending."""
        event = self.client.wait_for({"group_done"}, timeout=timeout)
        while event is not None:
            self.on_group_done(event)
            event = self.client.wait_for({"group_done"}, timeout=0)
        self.collect()
        self.dispatch()

    def run(self, tick: float = 5.0) -> None:
        while True:
            if not self.client.connected:
                # A trainer restart stops its scheduler; rewrites keep running
                # and are folded once a scheduler is back.
                try:
                    self.client.connect()
                    self.refresh(with_rows=True)
                except RewriterError as e:
                    log.warning("%s", e)
                    continue
            try:
                self.step(tick)
            except RewriterError as e:
                log.warning("scheduler call failed: %s", e)


def _evolve():
    evolution = Path(__file__).resolve().parent.parent / "evolution"
    if str(evolution) not in sys.path:
        sys.path.insert(0, str(evolution))
    import evolve_ondella as ev  # noqa: PLC0415

    return ev


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job", type=Path)
    ap.add_argument("--workers", type=int, default=16, help="concurrent rewrites in all")
    ap.add_argument("--help-levels", default=",".join(HELP_LEVELS),
                    help="one easier rewrite per level, small help to large")
    ap.add_argument("--history", type=int, default=Policy.history, help="groups the solve rate is over")
    ap.add_argument("--prior", type=float, default=Policy.prior)
    ap.add_argument("--explore", type=float, default=Policy.explore, help="weight of a version with no group yet")
    ap.add_argument("--original-floor", type=float, default=Policy.original_floor)
    ap.add_argument("--harder-parent-factor", type=float, default=Policy.harder_parent_factor)
    ap.add_argument("--mastered", type=float,
                    default=float(os.environ.get("SWE_EVOLUTION_HARDER_RATIO", Policy.mastered)))
    ap.add_argument("--retire-after", type=int, default=Policy.retire_after)
    ap.add_argument("--tick", type=float, default=5.0, help="seconds between pending-signal scans")
    args = ap.parse_args()
    levels = tuple(x for x in args.help_levels.split(",") if x)
    policy = Policy(
        help_levels=levels, history=args.history, prior=args.prior, explore=args.explore,
        original_floor=args.original_floor, harder_parent_factor=args.harder_parent_factor,
        mastered=args.mastered, retire_after=args.retire_after,
    )
    job = args.job.resolve()
    os.environ.setdefault("TRL_BASE", str(job))
    root = layout.Root(job)
    ev = _evolve()
    # The simplify prompt refuses an unknown level only when a session starts;
    # refuse it here, before anything runs.
    import simplify_operators  # noqa: PLC0415 -- on the path _evolve() set

    for level in levels:
        simplify_operators.prompt(level)
    budget = ev.ec.rewrite_budget_sec(root, refresh=True)
    if budget <= ev.ec.FINAL_VALIDATION_RESERVE_SEC:
        raise SystemExit(
            f"rewrite budget {budget:.0f}s does not exceed the "
            f"{ev.ec.FINAL_VALIDATION_RESERVE_SEC}s final-validation reserve"
        )
    tool_bin = ev.rw_workspace.prepare_tool_bin(root)
    if tool_bin != root.bin:
        os.environ[ev.rw_workspace.TOOL_BIN_ENV] = str(tool_bin)
    _lock_fd = ev.acquire_singleton(root.evolution.loop_lock)  # noqa: F841 -- held for life
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(root.evolution.loop_log)],
    )
    published = ev.rw_workspace.recover(root)
    recovered = ev.finalize_interrupted(root, include=published)
    if recovered["failed"]:
        raise RuntimeError(f"could not finalize orphaned rewrites: {recovered}")
    if not ev.SIMPLIFY_ENABLED:
        log.warning("SWE_EVOLVE_SIMPLIFY is off: easier signals are deferred, no easier child is made")
    log.info("ladder up: pid=%d root=%s workers=%d policy=%s", os.getpid(), root.path, args.workers, policy)
    client = RewriterClient(job)
    client.connect()
    ladder = Ladder(root, client, ev, policy=policy, workers=args.workers)
    ladder.start()
    ladder.run(args.tick)


if __name__ == "__main__":
    main()
