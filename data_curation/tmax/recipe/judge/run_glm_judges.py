#!/usr/bin/env python3
"""
run_glm_workers.py - fan out a directory of prompt files to N headless Claude Code
workers, each of which runs on GLM-5.3 via the owner's OpenAI-compatible endpoint.

Design (1) "headless workers" from GLM53_WITH_CLAUDE_CODE.md:
    Claude (or a human) orchestrates -> this launcher -> M x `claude -p` processes
    -> ANTHROPIC_BASE_URL=https://api.inco.ai -> glm-5.3-flash:fast

Key handling
------------
* The API key is read ONCE into memory (from $INCO_API_KEY, or from the file named
  by --key-file) and passed to each child only through its environment block.
* It is never written to disk, never placed on a command line (so it cannot be
  seen in `ps`), and never logged. `mask()` scrubs it from anything we print.

Prompts
-------
Every regular file in --prompt-dir is one worker. The file's whole content is the
prompt; it is delivered on the worker's **stdin**, so even a 500-line rubric never
touches argv.

Results
-------
For each worker <name> we write, under --out-dir:
    <name>.result.json   raw `--output-format json` envelope from Claude Code
    <name>.stderr.txt    stderr, only when non-empty
and one machine-readable line per worker into <out-dir>/summary.jsonl:
    {"launch","worker","exit_code","ok","duration_s","model","input_tokens","output_tokens",
     "num_turns","result_text","error"}

summary.jsonl is APPENDED, never truncated, and every line carries the `launch` stamp of the run that wrote
it. It used to be opened "w": a later launch sharing an --out-dir replaced the earlier one's summary with its
own records, and on 2026-09-22 a one-worker re-run group erased a 162-record band that way. Select a launch
with its stamp; the file is a log of the directory, not of one run.

Usage
-----
    INCO_API_KEY="$(cat /home/zekaili/fangzhou/tb_check/4.1.txt)" \
    python3 run_glm_workers.py \
        --prompt-dir  ./prompts \
        --out-dir     ./results \
        --work-dir    ./scratch \
        --concurrency 16 \
        --allowed-tools Read Write Bash \
        --max-turns 12
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

DEFAULT_BASE_URL = "https://api.inco.ai"
DEFAULT_MODEL = "glm-5.3"

_print_lock = threading.Lock()
LAUNCH = ""                       # set in main(); stamps every record so an appended summary stays separable
_SECRETS: list[str] = []


def mask(text: str) -> str:
    """Remove any known secret from text before it is printed or stored."""
    for s in _SECRETS:
        if s:
            text = text.replace(s, "***MASKED***")
    return text


def log(msg: str) -> None:
    with _print_lock:
        print(mask(msg), file=sys.stderr, flush=True)


def load_key(key_file: str | None) -> str:
    """Return the API key from the environment, or from key_file. Never logged."""
    key = os.environ.get("INCO_API_KEY", "").strip()
    if not key and key_file:
        key = pathlib.Path(key_file).read_text().strip()
    if not key:
        sys.exit(
            "No API key. Set INCO_API_KEY in the environment, or pass --key-file.\n"
            "  e.g. INCO_API_KEY=\"$(cat /path/to/key.txt)\" python3 run_glm_workers.py ..."
        )
    _SECRETS.append(key)
    return key


CONTINUE_NUDGE = ("You stopped before finishing: your row has not been written (or does not validate). Continue the "
                  "procedure now using your tools -- read what you still need, write the ONE JSON line to the output "
                  "path, validate it with the given command -- and only then reply with the labels.")


def schema_errors(row, schema_path: str) -> list:
    """['<field path>: <validator keyword>', ...] for a row against the schema; [] when it validates, has no schema,
    or does not exist. Values are never included."""
    if not schema_path or row is None or not row.exists():
        return []
    import jsonschema
    try:
        r = json.loads(row.read_text().splitlines()[0])
    except Exception:
        return ["(row): not one JSON line"]
    s = json.load(open(schema_path))
    V = getattr(jsonschema, "Draft202012Validator", None) or jsonschema.Draft7Validator
    return sorted({f"{'/'.join(str(x) for x in e.path) or '(row)'}: {e.validator}" for e in V(s).iter_errors(r)})


def run_one(
    prompt_path: pathlib.Path,
    out_dir: pathlib.Path,
    work_root: pathlib.Path,
    child_env: dict,
    allowed_tools: list[str],
    max_turns: int,
    timeout_s: int,
    add_dirs: list[str] | None = None,
    summary_path: "pathlib.Path | None" = None,
    keys: "list[tuple[str, str]] | None" = None,
    attempts: int = 1,
    rows_root: str = "",
    key_index: int = 0,
    continue_unfinished: int = 0,
    row_schema: str = "",
) -> dict:
    name = prompt_path.stem
    # Each worker gets its own cwd so parallel workers cannot collide on relative paths.
    cwd = work_root / name
    cwd.mkdir(parents=True, exist_ok=True)

    cmd = ["claude", "-p", "--output-format", "json", "--max-turns", str(max_turns)]
    if allowed_tools:
        cmd += ["--allowedTools", *allowed_tools]
    for d in add_dirs or []:
        cmd += ["--add-dir", d]

    prompt = prompt_path.read_text()
    row = row_path_for(name, rows_root)
    if row_ok(row):
        log(f"[{name}] SKIP, row already written")
        return {"worker": name, "prompt_file": str(prompt_path), "exit_code": 0, "ok": True, "skipped": True}

    rec: dict = {}
    for attempt in range(1, max(attempts, 1) + 1):
        if keys:                                   # round-robin the keys, and rotate again on every retry
            kname, kval = keys[(key_index + attempt - 1) % len(keys)]
            child_env = dict(child_env); child_env["ANTHROPIC_AUTH_TOKEN"] = kval
        else:
            kname = "default"
        if row is not None and row.exists() and not row_ok(row):
            row.unlink()                           # a partial row would make the judge refuse to write
        rec = _run_once(prompt_path, name, cmd, prompt, out_dir, cwd, child_env, timeout_s, kname, attempt)
        rec["row_written_file"] = bool(row_ok(row))
        # --continue-unfinished (opt-in, 2026-10-02; default 0 = unchanged): a session that ENDED CLEANLY with no valid
        # row (a model that narrates its next step and stops) is resumed in the same session with a fixed nudge that
        # carries no task content, up to N times, before the attempt counts as failed.
        # With --row-schema a row that fails the schema counts as unfinished too: it is renamed aside (never deleted) and
        # the nudge names only the failing field paths and validator keywords (no values, no task text).
        n_cont = 0
        while (continue_unfinished and n_cont < continue_unfinished and rec.get("ok") and rec.get("session_id")
               and not (rec["row_written_file"] and not schema_errors(row, row_schema))):
            n_cont += 1
            nudge = CONTINUE_NUDGE
            errs = schema_errors(row, row_schema) if rec["row_written_file"] else []
            if row is not None and row.exists():
                row.rename(row.with_name(row.name + f".unvalidated.{attempt}.{n_cont}"))
            if errs:
                nudge += (" Your last row failed the schema on: " + "; ".join(errs) +
                          ". Rewrite the whole row with those fields corrected (a reason is at most 600 characters).")
            ccmd = cmd + ["--resume", rec["session_id"]]
            log(f"[{name}] session ended without a valid row ({'schema: ' + str(len(errs)) + ' error(s)' if errs else 'no row'}); "
                f"resuming it ({n_cont}/{continue_unfinished})")
            nxt = _run_once(prompt_path, name, ccmd, nudge, out_dir, cwd, child_env, timeout_s, kname, attempt)
            nxt["continued"] = n_cont
            nxt["row_written_file"] = bool(row_ok(row))
            rec = nxt
        rec["continuations"] = n_cont
        if rec.get("ok") and rec["row_written_file"]:
            break
        if attempt < max(attempts, 1):
            log(f"[{name}] attempt {attempt} failed (key={kname}, error={str(rec.get('error'))[:60]}); retrying on the next key")
            time.sleep(5 * attempt)
    rec["launch"] = LAUNCH
    if summary_path is not None:
        with _print_lock:
            with open(summary_path, "a") as fh:
                fh.write(mask(json.dumps(rec, ensure_ascii=False)) + "\n")
    return rec


def _run_once(prompt_path, name, cmd, prompt, out_dir, cwd, child_env, timeout_s, key_name, attempt) -> dict:
    started = time.time()
    rec: dict = {"worker": name, "prompt_file": str(prompt_path), "key": key_name, "attempt": attempt}

    try:
        proc = subprocess.run(
            cmd,
            input=prompt,              # prompt on stdin, never on argv
            env=child_env,             # key lives only here
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        rec["exit_code"] = proc.returncode
        stdout, stderr = proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        rec.update(exit_code=124, ok=False, error=f"timeout after {timeout_s}s",
                   duration_s=round(time.time() - started, 2))
        log(f"[{name}] TIMEOUT after {timeout_s}s (key={key_name}, attempt={attempt})")
        return rec

    rec["duration_s"] = round(time.time() - started, 2)

    if stderr.strip():
        (out_dir / f"{name}.stderr.txt").write_text(mask(stderr))

    envelope = None
    if stdout.strip():
        (out_dir / f"{name}.result.json").write_text(mask(stdout))
        try:
            envelope = json.loads(stdout)
        except json.JSONDecodeError as exc:
            rec["error"] = f"stdout was not JSON: {exc}"

    if isinstance(envelope, dict):
        usage = envelope.get("usage") or {}
        model_usage = envelope.get("modelUsage") or {}
        rec.update(
            ok=(rec["exit_code"] == 0 and not envelope.get("is_error")),
            model=",".join(model_usage.keys()) or None,     # which model really answered
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            cache_read_tokens=usage.get("cache_read_input_tokens"),
            num_turns=envelope.get("num_turns"),
            session_id=envelope.get("session_id"),
            result_text=envelope.get("result"),
        )
        if envelope.get("is_error"):
            rec["error"] = envelope.get("api_error_status") or "is_error=true"
    else:
        rec["ok"] = False
        rec.setdefault("error", "no JSON envelope on stdout")

    log(f"[{name}] exit={rec['exit_code']} ok={rec.get('ok')} "
        f"{rec['duration_s']}s key={key_name} attempt={attempt} model={rec.get('model')}")
    return rec


def row_path_for(worker: str, rows_root: str) -> "pathlib.Path | None":
    """<set>__<task_id> -> <rows_root>/rows_<set>/<task_id>.jsonl (the file the judge is told to write)."""
    if not rows_root or "__" not in worker:
        return None
    label, task = worker.split("__", 1)
    flat = pathlib.Path(rows_root) / "rows" / f"{task}.jsonl"      # this run keeps every row in one rows/ directory
    return flat if flat.parent.is_dir() else pathlib.Path(rows_root) / f"rows_{label}" / f"{task}.jsonl"


def row_ok(path: "pathlib.Path | None") -> bool:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return False
    try:
        json.loads(path.read_text().splitlines()[0])
        return True
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt-dir", required=True, help="one file per worker; content = prompt")
    ap.add_argument("--out-dir", required=True, help="where result JSON and summary.jsonl go")
    ap.add_argument("--work-dir", default=None,
                    help="parent of each worker's cwd (default: <out-dir>/work)")
    ap.add_argument("--concurrency", type=int, default=8, help="max workers in flight")
    ap.add_argument("--model", default=os.environ.get("GLM_MODEL", DEFAULT_MODEL))
    ap.add_argument("--base-url", default=os.environ.get("GLM_BASE_URL", DEFAULT_BASE_URL))
    ap.add_argument("--key-file", default=None,
                    help="file holding the API key; preferred: set INCO_API_KEY instead")
    ap.add_argument("--allowed-tools", nargs="*", default=["Read"],
                    help="passed to --allowedTools; keep it minimal")
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--effort", default=None, help="low|medium|high|xhigh|max -> CLAUDE_CODE_EFFORT_LEVEL -> output_config.effort")
    ap.add_argument("--add-dir", nargs="*", default=[], help="extra directories the workers may read and write")
    ap.add_argument("--max-output-tokens", type=int, default=None, help="CLAUDE_CODE_MAX_OUTPUT_TOKENS for each worker")
    ap.add_argument("--small-model", default=None, help="model id for Claude Code's background/haiku requests")
    ap.add_argument("--key-files", nargs="*", default=[], help="several key files; workers round-robin over them and a retry rotates to the next")
    ap.add_argument("--attempts", type=int, default=1, help="attempts per worker; each retry uses the next key")
    ap.add_argument("--row-schema", default="", help="with --continue-unfinished: a row failing this schema is unfinished")
    ap.add_argument("--continue-unfinished", type=int, default=0,
                    help="resume a session that ended cleanly without a valid row, up to N times (default 0: never)")
    ap.add_argument("--rows-root", default="", help="<root>/rows/<task>.jsonl decides whether a worker is already done")
    ap.add_argument("--exclude-workers", default="", help="file of worker names to skip (e.g. still running elsewhere)")
    ap.add_argument("--order-file", default="",
                    help="file of worker names, one per line: submit in this order instead of by file name (names it "
                         "does not list follow, sorted). A ceiling stop then leaves a sample spread the way the file is.")
    ap.add_argument("--timeout", type=int, default=900, help="per-worker seconds")
    ap.add_argument("--max-context-tokens", type=int, default=195000,
                    help="what Claude Code should assume the window is")
    ap.add_argument("--config-dir", default=None,
                    help="CLAUDE_CONFIG_DIR for the workers; keeps ~/.claude untouched")
    args = ap.parse_args()

    key = load_key(args.key_file)

    prompt_dir = pathlib.Path(args.prompt_dir).resolve()
    out_dir = pathlib.Path(args.out_dir).resolve()
    work_root = pathlib.Path(args.work_dir).resolve() if args.work_dir else out_dir / "work"
    global LAUNCH
    LAUNCH = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    prompts = sorted(p for p in prompt_dir.iterdir() if p.is_file())
    if args.exclude_workers:
        skip = {l.strip() for l in open(args.exclude_workers) if l.strip()}
        before = len(prompts)
        prompts = [p for p in prompts if p.stem not in skip]
        log(f"excluded {before - len(prompts)} workers listed in {args.exclude_workers}")
    if args.rows_root:
        before = len(prompts)
        prompts = [p for p in prompts if not row_ok(row_path_for(p.stem, args.rows_root))]
        log(f"skipped {before - len(prompts)} workers whose row is already written")
    if args.order_file:
        rank = {}
        for l in open(args.order_file):
            if l.strip() and l.strip() not in rank:
                rank[l.strip()] = len(rank)
        prompts.sort(key=lambda p: (rank.get(p.stem, len(rank)), p.stem))
        listed = sum(p.stem in rank for p in prompts)
        log(f"order from {args.order_file}: {listed} listed first, {len(prompts) - listed} unlisted after, sorted")
    if not prompts:
        sys.exit(f"no prompt files in {prompt_dir}")

    # Minimal environment for the children: no inherited ANTHROPIC_* from this shell.
    child_env = {
        "HOME": os.environ.get("HOME", ""),
        "PATH": os.environ.get("PATH", ""),
        "TERM": "dumb",
        "ANTHROPIC_BASE_URL": args.base_url,
        "ANTHROPIC_AUTH_TOKEN": key,            # -> Authorization: Bearer (overridden per worker when --key-files is used)
        "ANTHROPIC_MODEL": args.model,
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(args.max_context_tokens),
        # The endpoint is not Anthropic; don't send pre-release capabilities at it.
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_TELEMETRY": "1",
    }
    if args.effort:
        child_env["CLAUDE_CODE_EFFORT_LEVEL"] = args.effort
    if args.max_output_tokens:
        child_env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(args.max_output_tokens)
    child_env["CLAUDE_CODE_SUBAGENT_MODEL"] = args.model
    small = args.small_model or args.model
    child_env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = small
    child_env["ANTHROPIC_SMALL_FAST_MODEL"] = small
    child_env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = args.model
    child_env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = args.model
    if args.config_dir:
        child_env["CLAUDE_CONFIG_DIR"] = str(pathlib.Path(args.config_dir).resolve())

    keys: list[tuple[str, str]] = []
    for kf in args.key_files:
        v = pathlib.Path(kf).read_text().strip()
        if not v:
            sys.exit(f"empty key file: {kf}")
        keys.append((pathlib.Path(kf).stem, v))
        _SECRETS.append(v)
    if not keys:
        keys = [("env", key)]
    log(f"{len(prompts)} workers, concurrency={args.concurrency}, model={args.model}, "
        f"base_url={args.base_url}, keys={[n for n, _ in keys]}, attempts={args.attempts}")

    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(run_one, p, out_dir, work_root, child_env,
                        args.allowed_tools, args.max_turns, args.timeout, args.add_dir,
                        out_dir / "summary_live.jsonl", keys, args.attempts, args.rows_root, i,
                        args.continue_unfinished, args.row_schema): p
            for i, p in enumerate(prompts)
        }
        for fut in as_completed(futures):
            try:
                records.append(fut.result())
            except Exception as exc:                        # never lose a worker silently
                records.append({"worker": futures[fut].stem, "ok": False,
                                "exit_code": -1, "error": f"launcher exception: {exc}"})

    records.sort(key=lambda r: r["worker"])
    with (out_dir / "summary.jsonl").open("a") as fh:      # append: a launch may not erase another's records
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok = sum(1 for r in records if r.get("ok"))
    # A judgment exists when its ROW exists, not when its worker exited zero: a worker can write a valid row
    # and then be stopped -- a watchdog TERM, a per-task cost cap -- and exit non-zero. That has under-counted
    # a finished band by one twice (172 rows reported as 171 ok, then 162 as 161), while the resume check,
    # which trusts rows, had it right both times. Report both, with the number that means "we have a
    # judgment" first.
    if args.rows_root:
        rows = sum(1 for r in records if row_ok(row_path_for(r["worker"], args.rows_root)))
        log(f"DONE {rows}/{len(records)} rows ({ok} clean exits) -> {out_dir/'summary.jsonl'}")
    else:
        log(f"DONE {ok}/{len(records)} ok -> {out_dir/'summary.jsonl'}")
    return 0 if ok == len(records) else 1


if __name__ == "__main__":
    sys.exit(main())
