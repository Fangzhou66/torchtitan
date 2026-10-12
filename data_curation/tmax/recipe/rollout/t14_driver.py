#!/usr/bin/env python3
"""tmax-15k never-graded rollouts (corpus t14, lead 2026-10-03): one counted rollout per task.

Population: class 4_never_graded of tmax_k3/tmax15k_partition.tsv (11,484 ids), in the seeded order of
rows/t14_population.tsv (non-security first) then rows/t14_population.security.tsv (only with
--include-security-domain). Rows: rows/t14_rows.jsonl (prepare_tmax_rows.py, the pinned converter).

The rollout itself is the rollout_smoke line unchanged: this directory holds symlinks to its ladder.py,
set_rung.py, run_retain.sh, retain_trajectory.py (so `HERE`, and with it runs/, is THIS directory), and
each lane is one `ladder.py run` (rung 1/1/1 first, a ceiling re-runs once at 4/8/10).

What this adds is the scheduling, for a population 30x the cfn line's:
  * LANES. Up to --lanes ladders run at once, each over a chunk of --chunk ids at --lane-conc. A lane
    that ends is replaced at once by the next chunk, so one chunk's tail (its slowest tasks) overlaps
    the other lanes' work instead of idling the whole width.
  * PLAN FROM DISK. Every plan re-reads runs/<prefix>*/results.jsonl (cached by size+mtime). A task is
    DONE when it has a REAL attempt (cf_pass2's rule: not interrupted, not harness/infra-killed, not
    censored below the top rung), or when 1 + --retries attempts were killed (gateway_exhausted).
    An interrupted attempt (slice wall, stop) never counts. A task whose last attempt hit a ceiling
    below the top starts at 4/8/10.
  * QUOTA HALT. Each tick scans the active lanes' provider_errors.jsonl: a 429 quota_exceeded stops
    every lane (ladder SIGSTOP, launcher SIGINT, then ladder SIGKILL) and exits 5. Other 429/5xx are
    counted, never acted on (owner's order for the gateway lines: back-off off).
  * SIGINT (the slice wall, sbatch --signal=B:INT): the same stop, exit 0; interrupted attempts re-plan.

  python t14_driver.py --out slices/sliceN [--prefix t14p] [--lanes 4 --lane-conc 32 --chunk 400]
                       [--ids-file F] [--include-security-domain] [--plan-only]

Exit: 0 done (or stopped by the wall), 5 quota exhausted, 2 refused.
"""
from __future__ import annotations
import argparse, collections, csv, datetime as dt, glob, json, os, shutil, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = ("/data/users/zekaili/fangzhou/tb_check/reaudit_398/reviews/overfilter_20260916/v13/"
       "tb_training_mix/rollout_smoke")
sys.path.insert(0, KIT)
from reduce_attempts import failure_class, xml_entities  # noqa: E402  (the line's own definitions)
from ladder import censored                               # noqa: E402  (the ladder's own ceiling rule)

LADDER = ["1/1/1", "2/2/2", "4/4/4", "4/8/10"]
TOP = LADDER[-1]
# T14_ROWS: another corpus's rows (the CalibForge rest line runs this driver from cf/, rows cf/rows/cfn_rollout_rows.jsonl)
ROWS = os.environ.get("T14_ROWS") or os.path.join(HERE, "rows", "tmax_rows_all14601.jsonl")
POPMETA: dict[str, dict] = {}   # task -> its population row (start_rung, pass2_skip when the corpus has them)
TOP_IDS: set[str] = set()       # --top-ids: start at the top rung (2026-10-11 Kimi-max line: tasks DeepSeek needed 4/8/10 for)
RUNS = os.path.join(HERE, "runs")
STOP = {"flag": False}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Log:
    def __init__(self, out: str):
        self.fh = open(os.path.join(out, "driver_events.jsonl"), "a")

    def __call__(self, **kw) -> None:
        line = json.dumps({"t": now(), **kw})
        self.fh.write(line + "\n"); self.fh.flush(); os.fsync(self.fh.fileno())
        print(line, flush=True)


POP = "t14s_population"   # written by t14_scope.py (reviewed clean/repair scope, lead 2026-10-03); re-read every plan


def population(include_sec: bool, ids_file: str | None) -> list[str]:
    files = [f"{POP}.tsv"] + ([f"{POP}.security.tsv"] if include_sec else [])
    rows = [r for f in files for r in csv.DictReader(open(os.path.join(HERE, "rows", f)), delimiter="\t")]
    POPMETA.clear()
    POPMETA.update({r["task_id"]: r for r in rows})
    # non-security first, then security (lead 2026-10-03); within each, the scope's seeded order
    order = [r["task_id"] for r in sorted(rows, key=lambda r: (int(r["security"]), int(r["order"])))]
    if ids_file:
        want = {l.strip() for l in open(ids_file) if l.strip()}
        missing = want - set(order)
        if missing:
            raise SystemExit(f"refusing: {len(missing)} id(s) of {ids_file} are not in the allowed population")
        order = [t for t in order if t in want]
    return order


_CACHE: dict[str, tuple[tuple[int, float], list[dict]]] = {}


VOID = os.path.join(HERE, "runs_void.tsv")


def _void_lines():
    for path in (VOID, VOID.replace(".tsv", ".security.tsv")):   # the .security file is gitignored (security ids)
        if os.path.exists(path):
            for line in open(path):
                if line.strip() and not line.startswith("#"):
                    yield line.rstrip("\n").split("\t")


def void_runs() -> dict[str, str]:
    """run dir name -> reason, from runs_void.tsv (run<TAB>reason). A voided run stays on disk but none of its attempts
    count: not as a kill, not as real (lead 2026-10-05: kills caused by our own config error, e.g. the kimi_smoke2
    top_p 400s, must not spend a task's kill budget)."""
    return {f[0]: f[1] for f in _void_lines() if len(f) == 2}


def void_pairs() -> dict[tuple[str, str], str]:
    """(run, task) -> reason, from 3-field lines run<TAB>task<TAB>reason in runs_void.tsv / runs_void.security.tsv: one
    task's attempt in a shared lane counts for nothing (lead 2026-10-06: the 7 CF tasks that exhausted their kill budget
    on infrastructure get a fresh one)."""
    return {(f[0], f[1]): f[2] for f in _void_lines() if len(f) == 3}


def attempts(prefix: str) -> dict[str, list[dict]]:
    """task -> its attempts under runs/<prefix>*, in time order, classified as cf_pass2 does (voided runs skipped)."""
    out: dict[str, list[dict]] = {}
    void, vpairs = void_runs(), void_pairs()
    for rd in sorted(glob.glob(os.path.join(RUNS, f"{prefix}*-*_*"))):
        if os.path.basename(rd) in void:
            continue
        rp = os.path.join(rd, "results.jsonl")
        if not os.path.exists(rp):
            continue
        st = os.stat(rp)
        key = (st.st_size, st.st_mtime)
        if rp in _CACHE and _CACHE[rp][0] == key:
            recs = _CACHE[rp][1]
        else:
            recs = []
            for line in open(rp):
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                res = d.get("resources") or {}
                sz = res.get("size") or {}
                rung = f"{sz.get('cpu')}/{sz.get('mem_gb')}/{sz.get('disk_gb')}"
                ad = os.path.join(rd, "tasks", d["task_id"], f"trial-{d.get('trial', 0)}",
                                  f"attempt-{d.get('attempt', 1)}")
                term_kill = (d.get("finish_reason") == "terminal_exited"
                             and xml_entities(ad).get("keystrokes_xml_entities") == 1)
                interrupted = d.get("outcome") == "interrupted"
                killed = interrupted or term_kill or failure_class(d) in ("infra", "harness")
                ceil = bool(censored(res)) and rung != TOP
                recs.append({"run": os.path.basename(rd), "task_id": d["task_id"], "rung": rung,
                             "t": d.get("finished") or d.get("started") or 0,
                             "reward": d.get("reward") or 0, "interrupted": interrupted,
                             "killed": killed, "ceiling_below_top": ceil,
                             "real": not killed and not ceil})
            _CACHE[rp] = (key, recs)
        for r in recs:
            if (r["run"], r["task_id"]) not in vpairs:
                out.setdefault(r["task_id"], []).append(r)
    for v in out.values():
        v.sort(key=lambda r: r["t"])
    return out


def state(a: list[dict], retries: int) -> tuple[str, str]:
    """(status, start rung): status is 'real' | 'gateway_exhausted' | 'todo'."""
    if any(x["real"] for x in a):
        return "real", ""
    kills = sum(1 for x in a if x["killed"] and not x["interrupted"])
    if kills > retries:
        return "gateway_exhausted", ""
    ceil = any(x["ceiling_below_top"] for x in a)
    return "todo", (TOP if ceil else LADDER[0])


def plan(order: list[str], prefix: str, retries: int, owned: set[str]) -> tuple[list[tuple[str, str]], dict]:
    att = attempts(prefix)
    todo, counts = [], {"real": 0, "gateway_exhausted": 0, "todo": 0, "in_flight": 0}
    for t in order:
        if t in owned:
            counts["in_flight"] += 1
            continue
        s, rung = state(att.get(t, []), retries)
        counts[s] += 1
        if s == "todo":
            if (POPMETA.get(t) or {}).get("start_rung") == TOP or t in TOP_IDS:   # declared 8G (CF staging) / --top-ids
                rung = TOP
            todo.append((t, rung))
    return todo, counts


def counted(a: list[dict]) -> dict | None:
    """The counted attempt of one pass: the first real one (t14_tally's rule)."""
    real = [x for x in a if x["real"]]
    return real[0] if real else None


def plan2(order: list[str], src_prefix: str, prefix: str, retries: int, per_task: int,
          owned: set[str]) -> tuple[list[tuple[str, str]], dict]:
    """PASS 2 (user 2026-10-03 via the lead): a task whose pass-1 counted attempt scored 0 gets up to `per_task` more
    counted attempts (effort max), stopping at the first reward 1; still 0 after them -> repair. Kills and interrupts
    are retried as in pass 1 and use no slot. Start rung: 4/8/10 if any attempt of the task reached it, else 1/1/1."""
    att1, att2 = attempts(src_prefix), attempts(prefix)
    todo, counts = [], collections.Counter()
    for t in order:
        if t in owned:
            counts["in_flight"] += 1
            continue
        c1 = counted(att1.get(t, []))
        if c1 is None:
            counts["awaiting_pass1"] += 1
            continue
        if c1["reward"] >= 1:
            counts["solved_pass1"] += 1
            continue
        if str((POPMETA.get(t) or {}).get("pass2_skip") or "0") == "1":   # expected-0 (cfn rule): one roll, no retry
            counts["expected_zero_no_retry"] += 1
            continue
        a2 = att2.get(t, [])
        real2 = [x for x in a2 if x["real"]]
        if any(x["reward"] >= 1 for x in real2):
            counts[f"solved_pass2_try{1 + next(i for i, x in enumerate(real2) if x['reward'] >= 1)}"] += 1
            continue
        if len(real2) >= per_task:
            counts["repair_0_after_3"] += 1
            continue
        if sum(1 for x in a2 if x["killed"] and not x["interrupted"]) > retries:
            counts["gateway_exhausted_pass2"] += 1
            continue
        top = any(x["rung"] == TOP or x["ceiling_below_top"] for x in att1.get(t, []) + a2) \
            or (POPMETA.get(t) or {}).get("start_rung") == TOP or t in TOP_IDS
        counts["todo"] += 1
        todo.append((t, TOP if top else LADDER[0]))
    return todo, dict(counts)


def security_ids() -> set[str]:
    return {r["task_id"] for r in csv.DictReader(open(os.path.join(HERE, "rows", f"{POP}.security.tsv")), delimiter="\t")}


def launcher_pids(tag: str) -> list[int]:
    """retain_trajectory.py processes whose --out is one of THIS tag's run dirs (never a pattern kill)."""
    out = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            argv = [x.decode(errors="replace") for x in open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0") if x]
        except OSError:
            continue
        if any(x.endswith("retain_trajectory.py") for x in argv) and "--out" in argv:
            o = argv[argv.index("--out") + 1]
            if os.path.dirname(os.path.abspath(o)) == RUNS and os.path.basename(o).startswith(tag + "-"):
                out.append(int(pid))
    return out


def lane_rows(tag: str) -> int:
    n = 0
    for rp in glob.glob(os.path.join(RUNS, f"{tag}-*_*", "results.jsonl")):
        n += sum(1 for _ in open(rp))
    return n


def foreign_owned(mine: set[str]) -> set[str]:
    """Ids a live `ladder.py run` on OUR rows owns that is not one of this driver's lanes (another slice, the smoke):
    never planned twice. Read from /proc cmdline (python does not rewrite its argv)."""
    own = set()
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            argv = [x.decode(errors="replace") for x in open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0") if x]
        except OSError:
            continue
        if not any(x.endswith("ladder.py") for x in argv) or "run" not in argv or "--ids" not in argv:
            continue
        opt = lambda k: argv[argv.index(k) + 1] if k in argv and argv.index(k) + 1 < len(argv) else ""
        if os.path.abspath(opt("--rows")) == ROWS and opt("--tag") not in mine and os.path.exists(opt("--ids")):
            own |= {l.strip() for l in open(opt("--ids")) if l.strip()}
    return own


def kill_share(prefix: str, now: float, window: float) -> tuple[int, int]:
    """(killed, finished) over the attempts that finished in (now - window, now]; interrupted ones are not counted."""
    recent = [x for v in attempts(prefix).values() for x in v
              if not x["interrupted"] and now - window < x["t"] <= now]
    return sum(1 for x in recent if x["killed"]), len(recent)


def create_per_lane(a) -> int:
    # fill mode: lanes start at most one per tick, so about 4 lanes are ever creating at once; 64 total / 4
    return 16 if a.target else max(1, -(-a.create_total // a.lanes))


def inflight(lanes: dict) -> int:
    """Attempts started and not yet finished across our live lanes: per lane, its NEWEST run dir's task dirs minus
    its result rows (in fill mode every id of a run starts at once, so task dirs = started). A lane between rungs
    (its run ended, the climb not yet started) counts as its own size until the climb's run dir appears."""
    n = 0
    for tag, L in lanes.items():
        rds = sorted(glob.glob(os.path.join(RUNS, f"{tag}-*_*")), key=os.path.getmtime)
        if not rds:
            n += L["n"]
            continue
        rd = rds[-1]
        try:
            started = len(os.listdir(os.path.join(rd, "tasks")))
        except OSError:
            started = L["n"]
        rp = os.path.join(rd, "results.jsonl")
        done = sum(1 for _ in open(rp)) if os.path.exists(rp) else 0
        n += max(0, started - done)
    return n


def run_scope(cmd: str, log) -> None:
    r = subprocess.run(cmd, shell=True, cwd=HERE, capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or [""])[-1]
    try:
        info = json.loads(last)
    except json.JSONDecodeError:
        info = {"stdout_tail": last[:300]}
    log(event="scope", rc=r.returncode, **({"error": r.stderr.strip()[-300:]} if r.returncode else info))


COMPRESS = "/data/users/zekaili/fangzhou/tb_check/rst_data/tools/compress_finished_attempts.py"


def compress(tag: str, log) -> None:
    """gzip conversation.jsonl/commands.jsonl of a finished lane's attempts (rst_data tool; verified, ledgered).
    One pass: --watch-pid names a pid that cannot exist, so the tool returns after its first pass."""
    for rd in glob.glob(os.path.join(RUNS, f"{tag}-*_*")):
        r = subprocess.run([sys.executable, COMPRESS, "--run", rd, "--min-free-gb", "0", "--watch-pid", "999999999"],
                           capture_output=True, text=True)
        log(event="compress", run=os.path.basename(rd), rc=r.returncode,
            tail=(r.stdout.strip().splitlines() or [""])[-1][:300])


def preflight() -> str:
    """'' if the interpreter imports what the launcher needs; else the error (a damaged venv)."""
    r = subprocess.run([sys.executable, "-c", "from harbor.agents.terminus_2 import Terminus2; import daytona, litellm"],
                       capture_output=True, text=True)
    return "" if r.returncode == 0 else (r.stderr.strip().splitlines() or ["?"])[-1][:300]


def stop_lanes(lanes: dict, log: Log, why: str) -> None:
    ladders = [L["proc"].pid for L in lanes.values()]
    launchers = sorted({p for tag in lanes for p in launcher_pids(tag)})
    log(event="stop", why=why, ladders=ladders, launchers=launchers)
    for p in ladders:
        try: os.kill(p, signal.SIGSTOP)
        except ProcessLookupError: pass
    for p in launchers:
        try: os.kill(p, signal.SIGINT)
        except ProcessLookupError: pass
    t0 = time.time()
    while any(os.path.exists(f"/proc/{p}") for p in launchers) and time.time() - t0 < 150:
        time.sleep(5)
    left = [p for p in launchers if os.path.exists(f"/proc/{p}")]
    for p in ladders:
        try: os.kill(p, signal.SIGKILL)
        except ProcessLookupError: pass
    for L in lanes.values():
        try: L["proc"].wait(timeout=30)
        except subprocess.TimeoutExpired: pass
    log(event="stopped", launchers_still_alive=left)


def provider_events(tags: list[str], seen: dict) -> tuple[int, int, dict]:
    """(new quota events, new rate events, status counts of the new rate events); incremental by file size."""
    quota = rate = 0
    by = {}
    for tag in tags:
        for pe in glob.glob(os.path.join(RUNS, f"{tag}-*_*", "tasks", "*", "trial-*", "attempt-*",
                                         "provider_errors.jsonl")):
            sz = os.path.getsize(pe)
            off = seen.get(pe, 0)
            if sz <= off:
                continue
            with open(pe) as fh:
                fh.seek(off)
                chunk = fh.read()
            seen[pe] = off + len(chunk.encode())
            for line in chunk.splitlines():
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                st = d.get("http_status") or 0
                if st == 429 and "quota_exceeded" in (d.get("body") or ""):
                    quota += 1
                elif st == 429 or st >= 500:
                    rate += 1
                    by[str(st)] = by.get(str(st), 0) + 1
    return quota, rate, by


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="t14r", help="tag family of this line's run dirs")
    ap.add_argument("--lanes", type=int, default=4)
    ap.add_argument("--lane-conc", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=400)
    ap.add_argument("--retries", type=int, default=3, help="killed (gateway/harness/infra) retries per task")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--pass2-prefix", default="", help="enable PASS 2 under this tag family (e.g. t14m)")
    ap.add_argument("--pass2-effort", default="max")
    ap.add_argument("--pass2-per-task", type=int, default=2)
    ap.add_argument("--no-pass1", action="store_true", help="plan pass 2 only")
    ap.add_argument("--ids-file")
    ap.add_argument("--top-ids", help="file of task ids that start at the top rung (pass 1 and pass 2)")
    ap.add_argument("--include-security-domain", action="store_true")
    ap.add_argument("--lane-start", type=int, default=1, help="first lane number (tags never collide)")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--target", type=int, default=0,
                    help="FILL MODE (2026-10-03): keep this many attempts in flight across all lanes; each lane is a "
                         "batch of <= --chunk ids launched all at once (concurrency = its size), and a new one starts "
                         "whenever measured in-flight work leaves room for >= --min-batch. --lanes then caps the number "
                         "of live lanes. 0 = the fixed-lanes mode.")
    ap.add_argument("--min-batch", type=int, default=20)
    ap.add_argument("--launch-until", type=float, default=0,
                    help="epoch seconds: launch no new lane after this; once the live lanes end, exit 0 (the slice "
                         "ends early and its successor starts). Short slices (1 h, lead 2026-10-04) otherwise interrupt "
                         "every in-flight episode at the wall and spend its tokens for nothing.")
    ap.add_argument("--kill-share", type=float, default=0.30,
                    help="pause (exit 7) when this share of the attempts finished in the last --kill-window s were "
                         "killed (gateway 5xx/harness/infra); 2026-10-03 13:10-13:35Z Cloudflare 502/520/524 burst")
    ap.add_argument("--kill-window", type=float, default=600)
    ap.add_argument("--kill-min", type=int, default=40)
    ap.add_argument("--create-total", type=int, default=64,
                    help="TT_DAYTONA_CREATE_CONCURRENCY summed over lanes (user's rule: 64; 32 on create failures)")
    ap.add_argument("--min-free-gb", type=float, default=50, help="stop every lane below this free space on runs/")
    ap.add_argument("--scope-cmd", help="shell command re-building the population (t14_scope.py ...), run every --scope-every s")
    ap.add_argument("--scope-every", type=float, default=600)
    ap.add_argument("--compress", action="store_true", help="gzip conversation/commands of each finished lane")
    ap.add_argument("--tick", type=float, default=60)
    a = ap.parse_args()

    # installed FIRST: a SIGINT during the opening scope scan / preflight was lost on 2026-10-03 13:43Z (slice 3 kept
    # launching lanes); the loop checks the flag before launching anything
    def on_int(signum, frame):
        STOP["flag"] = True
    signal.signal(signal.SIGINT, on_int)
    signal.signal(signal.SIGTERM, on_int)
    os.makedirs(a.out, exist_ok=True)
    log = Log(a.out)
    ROUTES = {("deepseek-flash", "https://gateway.huanzhimao.com/v1"),
              ("moonshotai/Kimi-K3", "https://api2.inferact.dev/v1")}       # Kimi: user 2026-10-05, CF rest
    if (os.environ.get("MODEL"), os.environ.get("ROLLOUT_API_BASE", "").rstrip("/")) not in ROUTES:
        log(event="refused", why="MODEL/ROLLOUT_API_BASE is not an allowed route", allowed=sorted(map(list, ROUTES)))
        return 2
    log(event="route", model=os.environ.get("MODEL"), api_base=os.environ.get("ROLLOUT_API_BASE"),
        effort_pass1=a.effort, effort_pass2=a.pass2_effort if a.pass2_prefix else None,
        top_p_override=os.environ.get("T14_TOP_P") or None)
    if a.scope_cmd:
        run_scope(a.scope_cmd, log)
    if a.top_ids:
        TOP_IDS.update(l.strip() for l in open(a.top_ids) if l.strip())
    order = population(a.include_security_domain, a.ids_file)
    todo, counts = plan(order, a.prefix, a.retries, foreign_owned(set()))
    if a.pass2_prefix:
        todo2, counts2 = plan2(order, a.prefix, a.pass2_prefix, a.retries, a.pass2_per_task, foreign_owned(set()))
        log(event="plan_pass2", todo=len(todo2), counts=counts2, effort=a.pass2_effort, prefix=a.pass2_prefix,
            per_task=a.pass2_per_task)
    log(event="plan", population=len(order), todo=len(todo), counts=counts,
        start_rungs={r: sum(1 for _, x in todo if x == r) for r in sorted({x for _, x in todo})},
        lanes=a.lanes, lane_conc=a.lane_conc, chunk=a.chunk, effort=a.effort, prefix=a.prefix,
        include_security_domain=a.include_security_domain, slurm_job=os.environ.get("SLURM_JOB_ID"),
        create_per_lane=create_per_lane(a), min_free_gb=a.min_free_gb)
    if a.plan_only:
        return 0
    err = preflight()
    if err:
        log(event="refused", why="preflight import failed", error=err, python=sys.executable)
        return 2


    lanes: dict[str, dict] = {}
    seen: dict[str, int] = {}
    n = a.lane_start
    rate_total = 0
    last_scope = time.time()
    idle_logged = False
    while True:
        free_gb = shutil.disk_usage(RUNS).free / 1e9
        if free_gb < a.min_free_gb:
            log(event="disk_low", free_gb=round(free_gb, 1), min_free_gb=a.min_free_gb)
            stop_lanes(lanes, log, "disk below the floor")
            return 6
        if a.scope_cmd and time.time() - last_scope >= a.scope_every:
            run_scope(a.scope_cmd, log)
            last_scope = time.time()
        if STOP["flag"]:
            stop_lanes(lanes, log, "SIGINT (slice wall or operator)")
            return 0
        # reap finished lanes
        for tag in [t for t, L in lanes.items() if L["proc"].poll() is not None]:
            L = lanes.pop(tag)
            rows = lane_rows(tag)
            log(event="lane_end", tag=tag, rc=L["proc"].returncode, ids=L["n"], rung=L["rung"], result_rows=rows,
                minutes=round((time.time() - L["t0"]) / 60, 1))
            if rows == 0:
                # a lane that wrote no result row at all is a launcher that never ran (2026-10-03: a damaged venv
                # re-launched 4 empty lanes in 2 s). Never re-plan into it: stop everything and refuse.
                log(event="lane_empty", tag=tag, note="no result row: the launcher did not run; see the lane log")
                stop_lanes(lanes, log, "empty lane")
                return 2
            if a.compress:
                compress(tag, log)
        # a gateway 5xx burst kills episodes and spends each task's bounded retries: pause and let the health wait resume
        k, nfin = kill_share(a.prefix, time.time(), a.kill_window)
        if a.pass2_prefix:
            k2, n2 = kill_share(a.pass2_prefix, time.time(), a.kill_window)
            k, nfin = k + k2, nfin + n2
        if nfin >= a.kill_min and k / nfin > a.kill_share:
                log(event="kill_share_pause", killed=k, finished=nfin, window_s=a.kill_window)
                stop_lanes(lanes, log, "killed share above the floor (gateway degradation)")
                return 7
        # quota / rate scan over the active lanes
        q, r, by = provider_events(list(lanes), seen)
        if r:
            rate_total += r
            log(event="rate_events_recorded", new=r, total=rate_total, by_status=by)
        if q:
            log(event="quota_exhausted", n=q)
            stop_lanes(lanes, log, "429 quota_exceeded")
            return 5
        if a.launch_until and time.time() >= a.launch_until:
            if not lanes:
                log(event="drained", note="launch window over and every lane ended; the successor slice continues")
                return 0
            t0 = time.time()
            while time.time() - t0 < a.tick and not STOP["flag"]:
                time.sleep(2)
                if any(L["proc"].poll() is not None for L in lanes.values()):
                    break
            continue
        # fill: fixed lanes, or (fill mode) up to --target attempts in flight, at most one new lane per tick
        room = a.target - inflight(lanes) if a.target else 0
        if (not a.target and len(lanes) < a.lanes) or (a.target and len(lanes) < a.lanes and room >= a.min_batch) \
                or (a.target and not lanes):
            owned = set().union(*(L["ids"] for L in lanes.values())) if lanes else set()
            owned |= foreign_owned(set(lanes))
            order = population(a.include_security_domain, a.ids_file)
            todo, counts = plan(order, a.prefix, a.retries, owned)
            todo = [] if a.no_pass1 else [(t, r_, "p1") for t, r_ in todo]
            if a.pass2_prefix:
                t2, c2 = plan2(order, a.prefix, a.pass2_prefix, a.retries, a.pass2_per_task, owned)
                counts = {**counts, "pass2": c2}
                pos, sec = {t: i for i, t in enumerate(order)}, security_ids()
                todo = sorted(todo + [(t, r_, "p2") for t, r_ in t2],
                              key=lambda x: (x[0] in sec, x[2], pos[x[0]]))     # security last; pass 1 before pass 2
            if not todo and not lanes:
                # the judge line may still add reviewed ids: idle (re-scoping) unless told the judge is done
                if not a.scope_cmd or os.path.exists(os.path.join(HERE, "rows", "JUDGE_DONE")):
                    log(event="done", counts=counts)
                    return 0
                if not idle_logged:
                    log(event="idle", counts=counts, note="nothing to roll; waiting for newly reviewed ids")
                    idle_logged = True
            if todo:
                idle_logged = False
            launched = 0
            while todo and len(lanes) < a.lanes and not (a.target and (launched or room < a.min_batch)):
                rung, kind = todo[0][1], todo[0][2]
                size = min(a.chunk, room) if a.target else a.chunk
                chunk = [t for t, r_, k_ in todo if r_ == rung and k_ == kind][:size]
                launched += 1
                todo = [x for x in todo if x[0] not in set(chunk)]
                fam = a.pass2_prefix if kind == "p2" else a.prefix
                # never reuse a tag: ladder.py would overwrite runs/<tag>_ladder.json (2026-10-03: a second chain's
                # slice 1 would have re-used t14rb1001.. of the first)
                while glob.glob(os.path.join(RUNS, f"{fam}b{n:04d}[-_]*")):
                    n += 1
                tag = f"{fam}b{n:04d}"
                n += 1
                idf = os.path.join(a.out, f"{tag}_ids.txt")
                with open(idf, "w") as fh:
                    fh.write("".join(t + "\n" for t in chunk))
                conc = len(chunk) if a.target else min(a.lane_conc, len(chunk))
                env = {**os.environ, "EFFORT": a.pass2_effort if kind == "p2" else a.effort,
                       "CREATE_CONCURRENCY": str(create_per_lane(a))}
                lf = open(os.path.join(a.out, f"{tag}_ladder.log"), "w")
                p = subprocess.Popen([sys.executable, "-u", os.path.join(HERE, "ladder.py"), "run",
                                      "--rows", ROWS, "--ids", idf, "--tag", tag, "--start-rung", rung,
                                      "--concurrency", str(conc), "--include-security",
                                      "--i-understand-this-spends"],
                                     cwd=HERE, env=env, stdout=lf, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
                lanes[tag] = {"proc": p, "ids": set(chunk), "n": len(chunk), "rung": rung, "t0": time.time()}
                log(event="lane_start", tag=tag, pid=p.pid, ids=len(chunk), rung=rung, conc=conc, kind=kind,
                    effort=env["EFFORT"],
                    create=create_per_lane(a), room=room if a.target else None,
                    remaining_todo=len(todo), counts=counts)
        t0 = time.time()
        while time.time() - t0 < a.tick and not STOP["flag"]:
            time.sleep(2)
            if any(L["proc"].poll() is not None for L in lanes.values()):
                break


if __name__ == "__main__":
    sys.exit(main())
