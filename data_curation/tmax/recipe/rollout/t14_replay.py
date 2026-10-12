#!/usr/bin/env python3
"""Gate-2 replay of every t14 reward-1 episode (user's rule 2026-10-03 via the lead; list coordinated with tb21-coverage):
a TMAX row is clean only if its solution replays to 1.0 in a fresh sandbox. One replay per task. No model calls.

REUSE. The replay itself is rollout_smoke/replay_actions.py::replay_one, unchanged: the same sandbox, terminal, seeding,
integrity baseline, entry gate (the training repo's parse_actions), sizing (1.3x uncensored peaks, the episode's own
rung as floor, clamped at 8 GiB / 10 GiB) and grading. What this file adds:

  * the EPISODE: the task's counted reward-1 attempt in this line -- pass 1 (runs/t14r*) if it scored 1, else the
    first reward-1 counted attempt of pass 2 (runs/t14m*) -- pinned by run/trial/attempt, never re-picked from
    rollout_smoke's tables (which do not hold these runs). replay_actions' module global HERE is pointed at this
    directory so its runs/ paths resolve here.
  * CPU AT THE RUNG: REPLAY_CPU_FROM_RUNG=1, the opt-in of replay_actions.sandbox_kwargs (fbd328a8; before it the
    constructor kept cpu=1 for every replay). resources.cpu_applied records what the sandbox actually got. The
    10-episode validation replays (replays/t14rpval_2) ran before that commit through a sandbox subclass here that
    applied the same rung cpu.
  * the MARKER TRANSFORM, gate-2 rule: a sequence the entry gate rejects is retried once with
    drop_between_last_markers, and replayed only if the gate then accepts it; transform is recorded per row.
  * RESUME + DURABILITY: one JSON line per finished replay, appended + fsync'd to <out>/results.jsonl; a task with a
    finished line in any --out of this tag family is skipped.

  python t14_replay.py --out replays/t14rp_<ts> [--concurrency 100] [--ids F] [--plan-only]
"""
from __future__ import annotations
import argparse, asyncio, csv, glob, json, os, sys, time
from pathlib import Path

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
KIT = Path("/data/users/zekaili/fangzhou/tb_check/reaudit_398/reviews/overfilter_20260916/v13/tb_training_mix/rollout_smoke")
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(KIT))
import t14_driver as drv                    # noqa: E402  (the line's attempt classification)
from ladder import peaks_of                 # noqa: E402
import replay_actions as ra                 # noqa: E402

ra.HERE = HERE                              # runs/<run>/... resolve in this directory
TERMINAL = {"GRADED", "NO_TRAJECTORY", "PARSER_REJECTED", "INFRA_OR_REPLAY_ERROR"}


P1, P2 = "t14r", "t14m"     # tag families of pass 1 / the retries (--p1-prefix / --p2-prefix; CF: cfrp / cfrm)


def episodes(retries: int) -> list[dict]:
    """task -> its replay target: the counted reward-1 attempt (pass 1, else pass 2), with the fields replay_one reads."""
    order = drv.population(True, None)
    sec = drv.security_ids()
    att1, att2 = drv.attempts(P1), drv.attempts(P2)
    full = {}
    for prefix in (P1, P2):
        for rp in glob.glob(str(drv.RUNS) + f"/{prefix}*-*_*/results.jsonl"):
            run = os.path.basename(os.path.dirname(rp))
            for line in open(rp):
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                full[(run, d["task_id"])] = d
    out = []
    for t in order:
        c1 = drv.counted(att1.get(t, []))
        ep, pas = None, ""
        if c1 is not None and c1["reward"] >= 1:
            ep, pas = c1, "pass1"
        elif c1 is not None:
            real2 = [x for x in att2.get(t, []) if x["real"]][:2]
            hit = next((i for i, x in enumerate(real2) if x["reward"] >= 1), None)
            if hit is not None:
                ep, pas = real2[hit], f"retry{hit + 1}"
        if ep is None:
            continue
        d = full[(ep["run"], t)]
        res = d.get("resources") or {}
        pk = peaks_of(res)
        out.append({"task_id": t, "security": int(t in sec), "pass": pas, "run": ep["run"],
                    "trial": d.get("trial", 0), "attempt": d.get("attempt", 1), "rung": ep["rung"],
                    "peaks_usable": "1" if pk["peaks_usable"] else "0",
                    "peak_ram_mb": pk["peak_ram_mb"] or "", "peak_disk_mb": pk["peak_disk_mb"] or "",
                    "wall_s": d.get("seconds") or 0})
    # security last (lead's rule), the rest in population order
    return sorted(out, key=lambda e: e["security"])


def done_keys(out_root: Path, retry_ungraded: bool = False) -> set[tuple[str, str]]:
    """(task, episode run) pairs with a finished replay in any sibling --out of this family."""
    keys = set()
    for rp in out_root.parent.glob(out_root.name.split("_")[0] + "_*/results.jsonl"):
        for line in open(rp):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") in TERMINAL and not (retry_ungraded and r.get("status") == "INFRA_OR_REPLAY_ERROR"):
                keys.add((r["task_id"], r["source_run"]))
    if retry_ungraded:   # an episode with ANY graded replay is done; only never-graded ones are retried
        for rp in out_root.parent.glob(out_root.name.split("_")[0] + "_*/results.jsonl"):
            for line in open(rp):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("status") == "GRADED":
                    keys.add((r["task_id"], r["source_run"]))
    return keys




async def run(a) -> int:
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    eps = episodes(a.retries)
    if a.ids:
        want = {l.split()[0] for l in open(a.ids) if l.strip()}
        eps = [e for e in eps if e["task_id"] in want]
    done = done_keys(out, a.retry_ungraded)
    todo = [e for e in eps if (e["task_id"], e["run"]) not in done]
    print(json.dumps({"event": "plan", "episodes": len(eps), "done": len(eps) - len(todo), "todo": len(todo),
                      "by_pass": {p: sum(1 for e in todo if e["pass"] == p) for p in ("pass1", "retry1", "retry2")},
                      "security_todo": sum(e["security"] for e in todo), "concurrency": a.concurrency}), flush=True)
    if a.plan_only:
        return 0
    sys.path.insert(0, str(ra.SNAP))
    os.environ["REPLAY_CPU_FROM_RUNG"] = "1"
    if not hasattr(ra, "sandbox_kwargs"):
        raise SystemExit("refusing: replay_actions.py predates the rung-cpu opt-in (fbd328a8)")
    mod, psha = ra.load_parse_actions()
    sem = asyncio.Semaphore(a.concurrency)
    fh = open(out / "results.jsonl", "a")

    async def one(e: dict, seq: int) -> dict:
        r = await ra.replay_one(e["task_id"], e, sem, out, mod.parse_actions, seq)
        if r.get("status") == "PARSER_REJECTED":
            r2 = await ra.replay_one(e["task_id"], e, sem, out, mod.parse_actions, seq, "drop_between_last_markers")
            r2["untransformed_rejection"] = r.get("error")
            r = r2
        r.setdefault("transform", {"transform": "none"})
        return {**r, "security": e["security"], "episode_pass": e["pass"], "replay_run": out.name,
                "parse_actions_sha256": psha, "driver_sha256": ra.DRIVER_SHA, "t": time.time()}

    tasks = [asyncio.ensure_future(one(e, 1)) for e in todo]
    n = 0
    for fut in asyncio.as_completed(tasks):
        r = await fut
        fh.write(json.dumps(r) + "\n"); fh.flush(); os.fsync(fh.fileno())   # durable as produced
        n += 1
        if n % 50 == 0 or n == len(tasks):
            print(json.dumps({"event": "progress", "done": n, "of": len(tasks), "t": time.strftime("%H:%M:%SZ", time.gmtime())}),
                  flush=True)
    fh.close()
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids")
    ap.add_argument("--concurrency", type=int, default=100)
    ap.add_argument("--retries", type=int, default=8)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--p1-prefix", default="t14r")
    ap.add_argument("--p2-prefix", default="t14m")
    ap.add_argument("--retry-ungraded", action="store_true",
                    help="re-replay episodes whose every replay so far ended INFRA_OR_REPLAY_ERROR (no verdict yet)")
    a = ap.parse_args()
    global P1, P2
    P1, P2 = a.p1_prefix, a.p2_prefix
    sys.exit(asyncio.run(run(a)))


if __name__ == "__main__":
    main()
