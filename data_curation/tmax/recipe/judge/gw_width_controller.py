#!/usr/bin/env python3
"""Adaptive-width controller for a judge run on the huanzhimao gateway (owner's order 2026-09-24 ~20:00Z: V4.1-Flash
max over every CalibForge id without a valid V4.1 row).

Why a controller and not run_glm_judges --concurrency N: the width must change while the run goes, and a launcher's
pool is fixed at its start. So this keeps W one-task launchers (run_glm_judges, --attempts 1) running, each an attempt
of one task, and:
  * ORDER: every worker without a valid row at start, sorted, then random.Random(seed).shuffle -- recorded.
  * RETRY: a task whose attempt ends without a valid row (cut, gateway failure, watchdog kill, invalid row) goes back
    to the FRONT of the queue until it has had --max-attempts launches; an invalid row is renamed aside
    (<task>.jsonl.invalid.N), never deleted. Each retry is logged to --events with its reason: the launcher's exit
    line and any watchdog or gateway event for the task since the attempt began.
  * WIDTH: every poll it reads the shim log it is given. A request is a gateway call: every final "POST /v1/messages"
    line except 402/404/499 (ours or the worker's), plus every "pre-stream failure" line (a failed call the shim
    retried). A bad request is a pre-stream failure line, a final 524, or a final 5xx with 0 bytes streamed.
    Mid-stream cuts are requests, not bad ones. Rungs --rungs (64,48,32; owner's refinement ~20:10Z): the run starts
    on the first and drops ONE rung only when, over the last --window seconds (600), with at least --min-requests
    requests, the bad share exceeds --drop-share (0.50) AND fewer than --min-rows (4) valid rows were written in that
    window (row-file mtimes); after a drop the next test waits a full window. Anything below that is logged at most
    every 5 minutes (bad share over 20%), not acted on. No automatic step back up. Shrinking never stops a running
    attempt, it only starts fewer. Every change is recorded, with its trigger, in --caps-file (width_history) and
    <ctl>/width.jsonl.
  * THROUGHPUT (lead, 20:3xZ): successful calls (final 200s) per --tp-window (300 s) against the run's best such
    window, read from the whole shim log so a restart keeps it. Under --tp-drop (25%) of the best -> one rung down
    (then a full --tp-window before the next test). Recorded like every change. --start-width starts on a lower rung
    (a restart that is itself a drop).
  * STEP UP (lead ~21:45Z; replaces the best-window step-up, which a 32-wide run could never reach against a best
    set at 64): the failure share -- pre-stream failures, final 524s, 0-byte 5xx AND mid-stream cuts, over all
    requests -- measured over the last --up-window (300 s) at every poll, under --up-share (15%) at every poll for
    --up-after (900 s), with at least --min-requests requests each time -> one rung up, never above the first rung.
    Not while the gate holds the run paused.
  * GATE (lead ~20:40Z): <ctl>/gate.json {"paused": true} holds the width at 0 -- nothing new launches, attempts in
    flight finish; {"paused": false, "width": W} resumes on rung W. Read every poll; each change is recorded. The
    gateway probe (gw_gate_probe.py) writes it.
  * PACKAGES (2026-09-25): the prompts must point judges at the run's read-only copy (<run-dir>/pkg, harbor/
    judge_copy.py), never at the shipped tree. --prompt-dir names prompts rewritten for that copy. When the run ends,
    if <run-dir>/pkg_manifest.json exists, the copy is checked against it and the result recorded in --caps-file
    (staged_tree_check). Then the judges' sessions of this run (--cfg, since its first launch) are scanned for
    executed task content (harbor/scan_judge_exec.py run) and the flagged rows recorded in --caps-file (exec_scan);
    a flagged row whose reason rests on the executed result is marked hold_before_build.
  * ADOPTION: a restarted controller takes over the attempts its predecessor left running: a task's launcher is
    found by its per-task exclude file in the cmdline, and a task's attempt count is the number of <ctl>/logs/<task>_N.log
    files, so nothing in flight is restarted and no attempt is counted twice.
  * HALT: only a quota halt of the shim (/__caps halted with quota in the reason) stops the run: nothing more is
    started, the controller signals its own process group (its launchers and their workers) and exits. No relaunch.
The watchdog runs beside it with --launcher-pid-file naming this controller, so the watchdog's own halt action reaches
the same group. Exits when the queue is empty and nothing runs; <ctl>/status.json is rewritten every poll.

    gw_width_controller.py --run-dir <cf_h3_gwmax> --work <work dir> --cfg <cfg dir> --shim-log <log>
                           --caps-file <caps json> --events <gateway_aborts.jsonl> [--dry-run]
"""
import argparse, bisect, collections, datetime as _dt, json, os, random, re, signal, subprocess, sys, time, urllib.request

V13 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAUNCHER = V13 + "/tb21_top10/run_glm_judges.py"
SCHEMA = V13 + "/harbor/output_schema_v13_harbor3.json"
GKEY = "/home/zekaili/fangzhou/tb_check/mdpsk.txt"
POST = re.compile(r"^\[(\d\d:\d\d:\d\d)\] POST /v1/messages -> (\d+) (\d+)B")
PRE = re.compile(r"^\[(\d\d:\d\d:\d\d)\]\s+pre-stream failure")


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def out(m):
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {m}", flush=True)


def hms_epoch(hms, now):
    """A log line's HH:MM:SS as an epoch: today's date, or yesterday's if that would lie in the future."""
    d = _dt.datetime.fromtimestamp(now, _dt.timezone.utc)
    h, m, s = map(int, hms.split(":"))
    t = d.replace(hour=h, minute=m, second=s, microsecond=0).timestamp()
    return t - 86400 if t > now + 60 else t


class Meter:
    """Requests and bad requests seen in the shim log, as (epoch, bad) pairs."""
    def __init__(self, path):
        self.path, self.pos, self.ev, self.ok, self.best, self.best_t = path, 0, collections.deque(), [], 0, None

    def read(self, now):
        if not os.path.exists(self.path):
            return
        with open(self.path) as fh:
            fh.seek(self.pos)
            for l in fh:
                m = PRE.match(l)
                if m:
                    self.ev.append((hms_epoch(m.group(1), now), True, False))
                    continue
                m = POST.match(l)
                if m:
                    st, nb = int(m.group(2)), int(m.group(3))
                    if st in (402, 404, 499):
                        continue
                    self.ev.append((hms_epoch(m.group(1), now), st == 524 or (st >= 500 and nb == 0),
                                    st >= 500 and st != 524 and nb > 0))
                    if st == 200:
                        self.ok.append(hms_epoch(m.group(1), now))
            self.pos = fh.tell()

    def throughput(self, now, window):
        """(successful calls in the last window, the best count over any window so far, sliding by 60 s)."""
        ok = sorted(self.ok)
        cnt = lambda end: bisect.bisect_left(ok, end) - bisect.bisect_left(ok, end - window)
        cur = cnt(now + 1e-6)
        if ok:
            t = self.best_t or ok[0] + window
            while t <= now:
                self.best = max(self.best, cnt(t))
                t += 60
            self.best_t = t
        return cur, max(self.best, cur)

    def share(self, now, window):
        while self.ev and self.ev[0][0] < now - 3 * 3600:
            self.ev.popleft()
        w = [e[1] for e in self.ev if e[0] >= now - window]
        return (sum(w) / len(w) if w else 0.0), len(w)

    def fail_share(self, now, window):
        """Pre-stream failures, 524s, 0-byte 5xx and mid-stream cuts over all requests in the window."""
        w = [e[1] or e[2] for e in self.ev if e[0] >= now - window]
        return (sum(w) / len(w) if w else 0.0), len(w)


def validator(path=SCHEMA):
    import jsonschema
    s = json.load(open(path))
    return (getattr(jsonschema, "Draft202012Validator", None) or jsonschema.Draft7Validator)(s)


def row_file_state(rows_dir, t, val, check_task_id=True):
    """'none' | 'valid' | 'invalid' for <rows_dir>/<t>.jsonl. Valid = exactly one non-blank JSON line that passes the
    schema and -- check_task_id, default on since 2026-10-03 -- whose task_id equals t: a judge once wrote another
    task's (schema-valid) row into the right filename (4 of 10,625 in tmax_k3/k3_rest), which no line can use."""
    p = f"{rows_dir}/{t}.jsonl"
    if not os.path.exists(p):
        return "none"
    try:
        lines = [l for l in open(p).read().splitlines() if l.strip()]
        if len(lines) != 1:
            return "invalid"
        r = json.loads(lines[0])
        if list(val.iter_errors(r)) or (check_task_id and not (isinstance(r, dict) and r.get("task_id") == t)):
            return "invalid"
        return "valid"
    except Exception:
        return "invalid"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--shim-log", required=True)
    ap.add_argument("--caps-file", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--stats-url", default="http://127.0.0.1:8800")
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--grant", default="", help="file of task ids that get ONE attempt beyond --max-attempts (owner's "
                    "order 2026-09-25: ids out of attempts under a lifted output ceiling)")
    ap.add_argument("--rungs", default="64,48,32", help="widths, widest first; start on the first")
    ap.add_argument("--window", type=int, default=600)
    ap.add_argument("--drop-share", type=float, default=0.50)
    ap.add_argument("--min-rows", type=int, default=4)
    ap.add_argument("--note-share", type=float, default=0.20)
    ap.add_argument("--min-requests", type=int, default=20)
    ap.add_argument("--prompt-dir", default="", help="prompt dir (default <run-dir>/prompts)")
    ap.add_argument("--ctl-dir", default="", help="control dir (logs, excludes, status, gate, order); default "
                    "<run-dir>/ctl. A new run over the same rows dir takes its own, so attempt counts start at 0")
    ap.add_argument("--order-file", default="", help="use this recorded order (a reprioritised one) instead of "
                    "<ctl>/order_seed<seed>.txt; ids with a valid row are skipped, running attempts adopted")
    ap.add_argument("--start-width", type=int, default=0, help="start on this rung instead of the first")
    ap.add_argument("--tp-window", type=int, default=300)
    ap.add_argument("--tp-drop", type=float, default=0.25)
    ap.add_argument("--up-share", type=float, default=0.15)
    ap.add_argument("--up-window", type=int, default=300)
    ap.add_argument("--up-after", type=int, default=900)
    ap.add_argument("--tp-min-best", type=int, default=40, help="no throughput rule until the best window has this many")
    ap.add_argument("--poll", type=int, default=10)
    ap.add_argument("--schema", default=SCHEMA, help="the row schema that decides a valid row (default harbor3's; the "
                    "Xiaomi run passes output_schema_v13_harbor4xm.json)")
    ap.add_argument("--max-turns", type=int, default=60, help="Claude Code turns per session (default 60, as before)")
    ap.add_argument("--attempt-timeout", type=int, default=5400, help="seconds per attempt (default 5400, as before)")
    ap.add_argument("--continue-unfinished", type=int, default=0, help="passed to the launcher: resume a session that "
                    "ended cleanly without a valid row, up to N times (Kimi K3 run, 2026-10-02; default 0 = unchanged)")
    ap.add_argument("--no-task-id-check", action="store_true", help="accept a schema-valid row whose task_id is not "
                    "its filename's task (default: such a row is invalid, renamed aside and retried)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    R = a.run_dir
    ctl = a.ctl_dir or os.path.join(R, "ctl")
    os.makedirs(ctl + "/excl", exist_ok=True); os.makedirs(ctl + "/logs", exist_ok=True)
    val = validator(a.schema)

    row_state = lambda t: row_file_state(R + "/rows", t, val, check_task_id=not a.no_task_id_check)

    PROMPTS = a.prompt_dir or R + "/prompts"
    workers = sorted(os.path.splitext(f)[0] for f in os.listdir(PROMPTS))
    # worker = <corpus>__<task>; the corpus prefix is whatever the stager used (cf__, cfn__, tw__), never assumed
    tid_of = lambda w: w.split("__", 1)[1]
    worker_of = {tid_of(w): w for w in workers}
    order_file = a.order_file or os.path.join(ctl, f"order_seed{a.seed}.txt")
    if a.order_file and not os.path.exists(order_file):
        sys.exit(f"{order_file}: no such order file")
    if os.path.exists(order_file):                       # a restart keeps the recorded order
        todo = [w for w in (l.strip() for l in open(order_file)) if w and row_state(tid_of(w)) != "valid"]
    else:
        todo = [w for w in workers if row_state(tid_of(w)) != "valid"]
        random.Random(a.seed).shuffle(todo)
        open(order_file, "w").write("\n".join(todo) + "\n")
    out(f"{len(workers)} workers, {len(workers) - len(todo)} with a valid row, {len(todo)} to judge; order {order_file}")
    if a.dry_run:
        out(f"DRY RUN: first 5 {todo[:5]}; rungs {a.rungs}")
        return 0

    def record_width(entry):
        with open(ctl + "/width.jsonl", "a") as fh:
            fh.write(json.dumps(entry) + "\n")
        try:
            caps = json.load(open(a.caps_file))
        except Exception:
            caps = {}
        caps.setdefault("width_history", []).append(entry)
        json.dump(caps, open(a.caps_file, "w"), indent=1)

    def log_event(ev):
        with open(a.events, "a") as fh:
            fh.write(json.dumps(ev) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def reason(t, since, logf):
        why = []
        try:
            ex = [l.strip() for l in open(logf) if "exit=" in l or "TIMEOUT" in l]
            if ex:
                why.append(ex[-1][:160])
        except OSError:
            pass
        for f in sorted(os.listdir(R)):
            if f.startswith("abandon_events_") and f.endswith(".jsonl"):
                for l in open(os.path.join(R, f)):
                    e = json.loads(l)
                    if e.get("task") == t and e.get("t", 0) >= since:
                        why.append("watchdog " + str(e.get("detail")))
        if os.path.exists(a.events):
            for l in open(a.events):
                e = json.loads(l)
                if e.get("task") == t and e.get("class") in ("cut", "provider_abort_recorded", "shim_prestream_retry"):
                    try:
                        if _dt.datetime.strptime(e["utc"], "%Y-%m-%dT%H:%M:%SZ").replace(
                                tzinfo=_dt.timezone.utc).timestamp() >= since:
                            why.append(f"gateway {e['class']} {e.get('status')} ({e.get('bytes')} B)")
                    except (KeyError, ValueError):
                        pass
        return "; ".join(why) or "attempt ended without a valid row"

    def launch(t, n):
        excl = f"{ctl}/excl/{t}.txt"
        if not os.path.exists(excl):
            open(excl, "w").write("\n".join(w for w in workers if w != worker_of[t]) + "\n")
        logf = f"{ctl}/logs/{t}_{n}.log"
        env = {k: v for k, v in os.environ.items() if k != "INCO_API_KEY"}
        p = subprocess.Popen(["python3", LAUNCHER, "--prompt-dir", PROMPTS, "--out-dir", R + "/results",
                              "--work-dir", a.work, "--config-dir", a.cfg, "--concurrency", "1",
                              "--base-url", a.stats_url, "--model", "deepseek/deepseek-v4.1-flash",
                              "--small-model", "deepseek/deepseek-v4.1-flash", "--key-file", GKEY, "--effort", "max",
                              "--max-output-tokens", "64000", "--allowed-tools", "Read", "Write", "Bash",
                              "--max-turns", str(a.max_turns), "--timeout", str(a.attempt_timeout), "--attempts", "1", "--exclude-workers", excl,
                              "--add-dir", V13, "--rows-root", R]
                             + (["--continue-unfinished", str(a.continue_unfinished), "--row-schema", a.schema]
                                if a.continue_unfinished else []),
                             cwd=os.path.dirname(os.path.dirname(R)),
                             stdout=open(logf, "w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)
        return p, time.time(), logf

    rungs = [int(x) for x in a.rungs.split(",")]
    attempts, running, done, gave_up, invalid_n = collections.Counter(), {}, [], [], collections.Counter()
    for f in os.listdir(ctl + "/logs"):                  # launches so far, this controller's or a predecessor's
        m = re.match(r"(.+)_(\d+)\.log$", f)
        if m:
            attempts[m.group(1)] = max(attempts[m.group(1)], int(m.group(2)))
    for p in os.listdir("/proc"):                        # attempts a predecessor left running: adopt, do not restart
        if not p.isdigit():
            continue
        try:
            cmd = open(f"/proc/{p}/cmdline", "rb").read().decode("utf-8", "replace").split("\0")
        except OSError:
            continue
        if any(x.endswith("run_glm_judges.py") for x in cmd) and "--exclude-workers" in cmd:
            ex = cmd[cmd.index("--exclude-workers") + 1]
            if ex.startswith(ctl + "/excl/"):
                t = os.path.basename(ex)[:-len(".txt")]
                logf = f"{ctl}/logs/{t}_{attempts[t]}.log"
                t0 = os.path.getmtime(logf) if os.path.exists(logf) else time.time()
                running[t] = (int(p), os.getpgid(int(p)), t0, logf)
    adopted_groups = {v[1] for v in running.values()}
    grant = {l.strip() for l in open(a.grant) if l.strip()} if a.grant else set()
    limit = lambda t: a.max_attempts + (1 if t in grant else 0)
    # a restart must not relaunch ids that have used their attempts (it re-queued every id without a valid row)
    queue = collections.deque(tid_of(w) for w in todo if tid_of(w) not in running and attempts[tid_of(w)] < limit(tid_of(w)))
    width = a.start_width if a.start_width in rungs else rungs[0]
    meter, last_change, last_note, tp_change, tp_high_since = Meter(a.shim_log), time.time(), 0.0, time.time(), None
    was_paused = False
    record_width({"utc": utc(), "width": width, "trigger": f"controller start (rungs {rungs}); adopted {len(running)} "
                  f"running attempts, {sum(attempts.values())} launches so far"})
    out(f"adopted {len(running)} running attempts; {len(queue)} queued; width {width}")
    while queue or running:
        now = time.time()
        # the quota halt: nothing more starts; this group goes
        try:
            with urllib.request.urlopen(a.stats_url + "/__caps", timeout=10) as r:
                caps = json.loads(r.read())
            if caps.get("halted") and "quota" in str(caps.get("halt_reason")):
                out(f"QUOTA HALT: {caps.get('halt_reason')}; stopping this group, no relaunch")
                record_width({"utc": utc(), "width": 0, "trigger": f"quota halt: {caps.get('halt_reason')}"})
                for g in adopted_groups:
                    try:
                        os.killpg(g, signal.SIGTERM)
                    except OSError:
                        pass
                os.killpg(0, signal.SIGTERM)
                return 1
        except (OSError, ValueError):
            pass
        # reap
        for t, (p, g, t0, logf) in list(running.items()):
            if isinstance(p, int):                       # adopted: not our child
                try:
                    os.kill(p, 0)
                    continue
                except OSError:
                    pass
            elif p.poll() is None:
                continue
            del running[t]
            st = row_state(t)
            if st == "valid":
                done.append(t)
                continue
            if st == "invalid":
                invalid_n[t] += 1
                os.rename(f"{R}/rows/{t}.jsonl", f"{R}/rows/{t}.jsonl.invalid.{invalid_n[t]}")
            why = ("invalid row (renamed .invalid.%d); " % invalid_n[t] if st == "invalid" else "") + reason(t, t0, logf)
            if attempts[t] < limit(t):
                queue.appendleft(t)
                ev = {"utc": utc(), "task": t, "class": "retry", "attempt": attempts[t] + 1, "of": limit(t),
                      "reason": why}
            else:
                gave_up.append(t)
                ev = {"utc": utc(), "task": t, "class": "attempts_exhausted", "attempts": attempts[t], "reason": why}
            log_event(ev)
            out(f"{ev['class'].upper()} {json.dumps(ev)[:300]}")
        # width
        meter.read(now)
        share, n = meter.share(now, a.window)
        rows_win = sum(1 for f in os.listdir(R + "/rows")
                       if f.endswith(".jsonl") and os.path.getmtime(f"{R}/rows/{f}") >= now - a.window)
        if n >= a.min_requests:
            nxt = [r for r in rungs if r < width]
            if share > a.drop_share and rows_win < a.min_rows and nxt and now - last_change >= a.window:
                width, last_change = nxt[0], now
                record_width({"utc": utc(), "width": width, "trigger": f"bad share {share:.1%} of {n} requests in the "
                              f"last {a.window}s > {a.drop_share:.0%} and {rows_win} rows written in it < {a.min_rows}"})
                out(f"WIDTH -> {width} ({share:.1%} of {n}, {rows_win} rows)")
        cur, best = meter.throughput(now, a.tp_window)
        if best >= a.tp_min_best:
            down = [r for r in rungs if r < width]
            up = [r for r in rungs if r > width]
            if cur < a.tp_drop * best and down and now - tp_change >= a.tp_window:
                width, tp_change, tp_high_since = down[0], now, None
                record_width({"utc": utc(), "width": width, "trigger": f"throughput: {cur} successful calls in the last "
                              f"{a.tp_window}s < {a.tp_drop:.0%} of the best window ({best})"})
                out(f"WIDTH -> {width} (throughput {cur} of best {best})")
        fs, fn = meter.fail_share(now, a.up_window)
        up = [r for r in rungs if r > width]
        if up and not was_paused and fn >= a.min_requests and fs < a.up_share:
            tp_high_since = tp_high_since or now
            if now - tp_high_since >= a.up_after:
                width, tp_change, tp_high_since = up[-1], now, None
                record_width({"utc": utc(), "width": width, "trigger": f"failure share under {a.up_share:.0%} (last "
                              f"{a.up_window}s, incl. mid-stream cuts) at every poll for {a.up_after}s; last {fs:.1%} "
                              f"of {fn}"})
                out(f"WIDTH -> {width} (failure share {fs:.1%})")
        else:
            tp_high_since = None
        if n >= a.min_requests and share > a.note_share and now - last_note >= 300:
            last_note = now
            with open(ctl + "/width.jsonl", "a") as fh:
                fh.write(json.dumps({"utc": utc(), "note": f"bad share {share:.1%} of {n} requests, {rows_win} rows "
                                     f"in the last {a.window}s: below the drop rule, not acted on", "width": width}) + "\n")
        # the gate
        try:
            gate = json.load(open(ctl + "/gate.json"))
        except (OSError, ValueError):
            gate = {}
        paused = bool(gate.get("paused"))
        if paused != was_paused:
            if not paused and gate.get("width") in rungs:
                width, tp_change, tp_high_since = gate["width"], now, None
            record_width({"utc": utc(), "width": 0 if paused else width,
                          "trigger": ("gate paused: " if paused else "gate resumed: ") + str(gate.get("reason", ""))})
            out(f"GATE {'paused' if paused else 'resumed at %d' % width}")
            was_paused = paused
        # launch
        while queue and len(running) < (0 if paused else width):
            t = queue.popleft()
            if row_state(t) == "valid":
                done.append(t)
                continue
            attempts[t] += 1
            p, t0, logf = launch(t, attempts[t])
            running[t] = (p, None, t0, logf)
        json.dump({"utc": utc(), "width": width, "running": len(running), "queue": len(queue),
                   "done_this_run": len(done), "attempts_exhausted": gave_up, "bad_share": round(share, 4),
                   "paused": paused, "fail_share_up_window": round(fs, 4), "window_requests": n, "rows_in_window": rows_win, "tp_window_ok": cur, "tp_best": best, "retries": sum(v - 1 for v in attempts.values() if v > 1)},
                  open(ctl + "/status.json", "w"), indent=1)
        time.sleep(a.poll)
    out(f"END: {len(done)} valid rows this run; attempts exhausted without a row: {gave_up}")
    if os.path.exists(os.path.join(R, "pkg_manifest.json")):
        sys.path.insert(0, os.path.join(V13, "harbor"))
        import judge_copy
        r = judge_copy.check(R, a.caps_file, label=f"end of run ({os.path.basename(ctl)})")
        out(f"staged tree check: ok={r['ok']} {({k: len(v) for k, v in r['differences'].items()})} "
            f"security {r['security_differences_counts']}")
    sys.path.insert(0, os.path.join(V13, "harbor"))
    import scan_judge_exec
    since = None                               # this control dir's first controller start (width.jsonl), 5 min early
    try:
        t0 = json.loads(open(os.path.join(ctl, "width.jsonl")).readline())["utc"]
        since = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(
            _dt.datetime.strptime(t0, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=_dt.timezone.utc).timestamp() - 300))
    except (OSError, ValueError, KeyError):
        logs = [os.path.join(ctl, "logs", f) for f in os.listdir(os.path.join(ctl, "logs"))]
        since = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(min(map(os.path.getmtime, logs)) - 6000)) if logs else None
    xs = scan_judge_exec.scan_run(a.cfg, os.path.join(R, "rows"), a.caps_file, label=f"end of run ({os.path.basename(ctl)})",
                                  since=since)
    out(f"exec scan: {len(xs['flagged_rows'])} flagged, hold before build {xs['hold_before_build']}; security flagged "
        f"{xs['security_flagged_count']}, held {xs['security_hold_count']} (ids in exec_scan.security.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
