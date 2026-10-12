#!/bin/bash
# GLM-5.3 (low) judging over the agent-reranked TB2.1 top-10, through OpenRouter pinned to baidu/fp8.
#
# THIS IS THE CONFIGURATION THAT PRODUCED THE ROWS OF 2026-09-21 (272 TMAX + 65 RST), not a description of
# one: every default below is the value that ran. The measurements behind them are in
# glm_claudecode/OPENROUTER_GLM_JUDGE.md sections 13-18, and the money in openrouter/final/CAMPAIGN_SPEND.json.
#
# Route:  Claude Code -> or_pin_proxy.py on 127.0.0.1 (adds the `provider` object OpenRouter needs in the
#         body, which Claude Code cannot send) -> OpenRouter /api/v1/messages (Anthropic skin) -> baidu/fp8.
# run_glm_judges.py is UNCHANGED; only flags differ from the inco line.
#
# TMAX runs under the frozen v13 prompt, RST under v13-rst, one task per headless worker. A task whose row is
# already written is skipped, so re-running this script finishes a partial run; a row left half-written by a
# stop is invalid JSON and the launcher deletes it before re-judging that task.
#
# WHY baidu/fp8 AND NOT io-net. Measured 2026-09-21: 5,271 calls to baidu at concurrency 16, 32 and 48
# returned zero 429s, while io-net refused 6 of 6 at concurrency 1 that same afternoon and never produced a
# single RST row in six attempts. baidu was also cheaper on both token axes -- but its listed price moved
# twice inside that one day, so re-price before spending on the cost argument.
#
# WHY --max-output-tokens 64000 AND NOT 32000. A turn stops at exactly the number configured here, and when
# GLM spends the whole budget thinking the turn returns stop_reason=max_tokens with an empty final message:
# the judgment dies having written nothing, and you pay for all of it. 32,000 was the binding constraint on
# this workload; at 64,000 zero of 46 TMAX turns in the check were capped. baidu allows 131,072, so the
# ceiling is ours. Do not lower it to save money -- the tokens are spent either way, and a capped turn spends
# them for nothing.
#
# NOTE ON --effort low. Claude Code does send it (output_config.effort=low) and OpenRouter accepts it. On
# io-net it was measured to change nothing at all (thinking-token medians 504 default / 748 low / 538
# native-low, one band). THAT FINDING IS ABOUT io-net AND HAS NOT BEEN REPEATED ON baidu. It is kept here
# because it is the production setting, not because it is known to throttle anything on this route.
#
# Start:  ./launch_openrouter.sh [--arm tmax|rst] [--ranks N] [--concurrency N] [--cap USD] [--port N]
# Stop:   ./stop_openrouter.sh --arm <arm> [--port N]   -- it CAPTURES the shim's counters before killing it,
#         because the run's spend exists only in the shim's memory until then.
set -u

V13=/data/users/zekaili/fangzhou/tb_check/reaudit_398/reviews/overfilter_20260916/v13
RUN=$V13/tb21_agentic_top10
EXT=/data/users/zekaili/fangzhou/tb_check/glm_claudecode/tb21_agentic_top10_run
SHIM=$V13/openrouter          # the committed kit: the shim, the watchdog, the probes, the captures
RUNDIR=$EXT/run               # pids and shim logs: run state, deliberately OUTSIDE the repo
KEY=/home/zekaili/fangzhou/tb_check/openrouter.txt
PROVIDER=baidu/fp8

ARM=tmax; RANKS=3; CONC=16; RUNCAP=135; PORT=8787
while [ $# -gt 0 ]; do
  case "$1" in
    --arm)         ARM=$2; shift 2;;
    --ranks)       RANKS=$2; shift 2;;   # TMAX only: best rank across neighbours, read from the source CSV.
                                         # The band is CUMULATIVE (rank <= N). Rows from a lower band are
                                         # skipped by the resume check, so --ranks 5 finishes ranks 4-5.
    --concurrency) CONC=$2; shift 2;;    # 16, 32 and 48 all measured clean on baidu
    --cap)         RUNCAP=$2; shift 2;;  # ENFORCED by the shim, not merely reported: it refuses at this number
    --port)        PORT=$2; shift 2;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done

case "$ARM" in
  tmax) P=$RUN/tmax/glm_low; CEIL=95000;  PERROW=0.48; ADD=("$RUN/tmax" "$V13");;
  # 190,000 on RST and not 95,000: an RST session that legitimately recovered from a capped turn reached
  # 124,403 output tokens, so the TMAX ceiling would have abandoned healthy work. One ceiling cannot serve
  # both arms.
  rst)  P=$RUN/rst/glm_low;  CEIL=190000; PERROW=0.62; ADD=("$RUN/rst" "$V13" "$V13/rst");;
  *) echo "--arm must be tmax or rst" >&2; exit 2;;
esac
WORK=$EXT/work_$ARM; CFG=$EXT/cfg_$ARM
# One events file per launch. Appending every run to one abandon_events.jsonl produced a 516-line file with 13
# real events in it, separable only by cutting on timestamps -- a record that gets misread rather than read.
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
die() { echo "FATAL: $*" >&2; exit 2; }

# ---------------------------------------------------------------- 0. preflight
[ -r "$KEY" ] || die "no readable OpenRouter key at $KEY"
command -v curl   >/dev/null || die "curl not found"
command -v claude >/dev/null || die "claude not found on PATH"
mkdir -p "$P/rows" "$P/results" "$WORK" "$CFG" "$RUNDIR" || die "cannot create run dirs"

# A launcher still running on this arm would double-judge whatever is in flight (two workers, one row path).
if [ -f "$P/launcher.pid" ]; then
  p=$(cat "$P/launcher.pid" 2>/dev/null || true)
  if [ -n "${p:-}" ] && kill -0 "$p" 2>/dev/null &&
     tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q run_glm_judges.py; then
    die "$ARM launcher $p is still running; ./stop_openrouter.sh --arm $ARM first (or let it finish)"
  fi
fi

# The resume contract. run_glm_judges.row_path_for() resolves <rows-root>/rows/<task>.jsonl only while that
# directory exists; without it the skip check silently reads rows_<label>/ instead and every finished task is
# judged again. Assert that the rendered prompt writes where the skip check looks.
[ -d "$P/prompts" ] || die "$ARM: no prompt directory at $P/prompts (run build_run.py first)"
first=$(find "$P/prompts" -maxdepth 1 -type f | sort | head -1)
[ -n "$first" ] || die "$ARM: $P/prompts holds no prompts"
grep -q "$P/rows/" "$first" || die "$ARM: prompts do not write into $P/rows -- --rows-root would not resume"

# Population. TMAX judges one rank band and excludes the rest; RST judges every prompt it has.
# The "already judged" count must be taken INSIDE the selected population: this rows directory holds every
# band judged so far, so counting all of it understates the work left (and can go negative).
EXCL=""
sel=$(find "$P/prompts" -maxdepth 1 -type f | wc -l)
have=$(find "$P/rows" -maxdepth 1 -name '*.jsonl' -size +0 | wc -l)
if [ "$ARM" = "tmax" ]; then
  EXCL=$P/exclude_not_rank$RANKS.txt
  counts=$(python3 - "$RUN" "$P" "$RANKS" <<'PY'
import csv, collections, os, sys
RUN, P, ranks = sys.argv[1], sys.argv[2], int(sys.argv[3])
best = collections.defaultdict(lambda: 10**9)
for r in csv.DictReader(open(f"{RUN}/tb21_agentic_top10.source.csv")):
    try:
        best[r["task_id"]] = min(best[r["task_id"]], int(float(r["rank"])))
    except Exception:
        pass
# a worker name is the prompt filename WITHOUT its extension (top10__task_x.txt -> top10__task_x) and the
# task id is that minus the top10__ prefix. --exclude-workers takes worker names; a row is named by task id.
workers = {os.path.splitext(f)[0] for f in os.listdir(f"{P}/prompts")}
band = {w for w in workers if best[w[len("top10__"):]] <= ranks}
# A task that has exhausted its attempts must not be picked up again by a later, wider band: the resume check
# only skips tasks that produced a ROW, so without this a permanent failure is retried every time the band
# grows. Worker names in no_more_attempts.txt (# comments allowed) are held back, and the reason belongs in
# that file beside them.
held = set()
if os.path.exists(f"{P}/no_more_attempts.txt"):
    for line in open(f"{P}/no_more_attempts.txt"):
        line = line.split("#")[0].strip()
        if line:
            held.add(line)
band -= held
with open(f"{P}/exclude_not_rank{ranks}.txt", "w") as fh:
    fh.write("\n".join(sorted(workers - band)) + "\n")
done = sum(1 for w in band if os.path.exists(f"{P}/rows/{w[len('top10__'):]}.jsonl"))
if held:
    sys.stderr.write(f"  held back by no_more_attempts.txt: {len(held & workers)} of {len(held)} listed\n")
print(f"{len(band)} {done}")
PY
) || die "could not build the rank band"
  sel=${counts% *}; have=${counts#* }
fi
echo "$ARM: $sel selected, $have already judged, $((sel - have)) to run"
[ "$((sel - have))" -gt 0 ] || { echo "nothing to do: every selected row is already written"; exit 0; }

# ---------------------------------------------------------------- 1. the pin shim
# Loopback only. The key is read into this subshell's environment and inherited by the shim; it never reaches
# a command line, a log or a config. Anything already on the port must be OUR shim, or the run would go to an
# unpinned (or differently pinned) upstream without saying so.
#
# THE CAPS LIVE HERE because the shim is the only layer that sees money: $1.50 per task across all of its
# attempts (a retry cannot get around it) and $RUNCAP for the run, at which the shim answers 402 to everything
# and the watchdog stops the launcher. Both fired in the 2026-09-21 runs. The test is made when a call is
# ADMITTED, against the spend so far, so a task that crosses its cap mid-call still finishes that call: the
# two tasks actually refused had reached $1.55 and $1.56 against the $1.50 cap. Budget the cap plus one call.
shim_is_ours() {
  local p; p=$(cat "$RUNDIR/proxy_$ARM.pid" 2>/dev/null || true)
  [ -n "${p:-}" ] || return 1
  kill -0 "$p" 2>/dev/null || return 1
  tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q "or_pin_proxy.py .*--port $PORT" || return 1
  echo "$p"
}
if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/__stats" >/dev/null 2>&1; then
  shim_pid=$(shim_is_ours) || die "something other than this arm's shim is listening on 127.0.0.1:$PORT"
  echo "shim already listening (pid $shim_pid)"
else
  # The redirections go on the SUBSHELL and `exec` makes it become the shim: with them on the inner command
  # only, the backgrounded subshell survives holding this script's stdout, and `./launch_openrouter.sh | tee`
  # then never returns even though the run started (measured).
  ( cd "$SHIM" && export OPENROUTER_API_KEY="$(cat "$KEY")" &&
    exec setsid nohup python3 or_pin_proxy.py --port "$PORT" --provider "$PROVIDER" \
         --pid-file "$RUNDIR/proxy_$ARM.pid" \
         --task-cost-cap 1.50 --run-cost-cap "$RUNCAP" --provider-log "$P/provider.jsonl"
  ) > "$RUNDIR/proxy_$ARM.log" 2>&1 < /dev/null &
  for _ in $(seq 1 20); do
    curl -fsS --max-time 2 "http://127.0.0.1:$PORT/__stats" >/dev/null 2>&1 && break
    sleep 0.5
  done
  shim_pid=$(shim_is_ours) || die "shim did not come up on 127.0.0.1:$PORT; see $RUNDIR/proxy_$ARM.log"
  echo "shim started (pid $shim_pid)"
fi
# Confirm the pin on the process, not on our intention: a shim we did not start -- or one started without
# --provider -- would send this run to or_pin_proxy.py's DEFAULT, which is still io-net/fp8, and nothing in
# the rows would say so until the provider log was read afterwards.
pin=$(tr '\0' ' ' < "/proc/$shim_pid/cmdline" | grep -o -- '--provider [^ ]*' || true)
[ "$pin" = "--provider $PROVIDER" ] \
  || die "the shim on $PORT carries '${pin:-no --provider at all (its default is io-net/fp8)}', not --provider $PROVIDER"
echo "shim ok: $pin | caps \$1.50 per task, \$$RUNCAP per run | $(curl -fsS "http://127.0.0.1:$PORT/__caps")"

# ---------------------------------------------------------------- 2. the judge line
# INCO_API_KEY must not reach the children: run_glm_judges.load_key reads it BEFORE --key-file, so a stale
# inco key in this shell would be sent to OpenRouter and 401 every worker. env -u removes it for the child.
#
# --attempts 2 is one retry, and the retry is worth having because the failure is stochastic: two of three
# tasks recorded as permanently failed succeeded when tried a third time by hand. Nothing measured justifies
# more than one automatic retry -- a third attempt is a decision to spend, so it is made explicitly.
# --max-turns 60 is a ceiling, not a budget (completions ran 9-19 turns). A cap the work does not fit under
# produces no row and still pays for the turns, which is worse than the tail cost of a generous one.
common=(--base-url "http://127.0.0.1:$PORT"
        --model z-ai/glm-5.3 --small-model z-ai/glm-5.3
        --key-file "$KEY"
        --effort low --max-output-tokens 64000
        --allowed-tools Read Write Bash
        --max-turns 60 --timeout 2400 --attempts 2)

# The pid must be the process that ends up BEING the launcher: `setsid` forks when the caller is already a
# process-group leader (an interactive shell), and then $! is a wrapper that exits at once and the stop script
# kills nothing. Having the child write its own pid is exact in both cases.
cd "$RUN" || die "cannot cd $RUN"
env -u INCO_API_KEY setsid nohup \
  bash -c 'echo $$ > "$1"; shift 1; exec "$@"' _ "$P/launcher.pid" \
  python3 "$V13/tb21_top10/run_glm_judges.py" \
    --prompt-dir "$P/prompts" --out-dir "$P/results" --work-dir "$WORK" --config-dir "$CFG" \
    --concurrency "$CONC" "${common[@]}" \
    ${EXCL:+--exclude-workers "$EXCL"} --add-dir "${ADD[@]}" --rows-root "$P" \
  > "$P/launch.log" 2>&1 < /dev/null &
sleep 2
lp=$(cat "$P/launcher.pid" 2>/dev/null || echo '?')
{ [ "$lp" != "?" ] && kill -0 "$lp" 2>/dev/null; } || echo "WARNING: launcher did not start; see $P/launch.log"

# ---------------------------------------------------------------- 3. the watchdog
# A doomed worker cannot be told from a slow one by the think/output ratio: completions ran 0.80-0.94 and the
# workers that produced nothing ran 0.95-0.99, and the bands touch. WASTE can be bounded even so, which is
# what this does -- it abandons a task whose cumulative output passes the arm's ceiling with no row written,
# capping a non-completion near $0.30-$0.50 instead of letting it run to the 2,400 s timeout. In production it
# abandoned two doomed workers at 96k and 99k tokens and left a healthy 0.85-ratio worker alone.
# It also stops the launcher's whole process group when the shim reports the run cap reached.
# It writes its own pid for the same reason the launcher does, and because setsid puts it in a session of
# its own: stopping the launcher's process group does NOT reach it.
setsid nohup \
  bash -c 'echo $$ > "$1"; shift 1; exec "$@"' _ "$P/watchdog.pid" \
  python3 "$SHIM/abandon_doomed.py" --stats "http://127.0.0.1:$PORT" \
    --work-dir "$WORK" --rows-dir "$P/rows" --max-output-tokens "$CEIL" --interval 20 \
    --launcher-pid-file "$P/launcher.pid" --events "$P/abandon_events_$STAMP.jsonl" \
  > "$P/watchdog.log" 2>&1 < /dev/null &
sleep 1

echo "launched: $ARM launcher $lp | concurrency $CONC | watchdog ceiling $CEIL out-tokens | run cap \$$RUNCAP"
echo "estimate: ~\$$(awk "BEGIN{printf \"%.2f\", ($sel - $have) * $PERROW}") at the measured \$$PERROW per row" \
     "($ARM, campaign mean), wall ~$(awk "BEGIN{printf \"%.1f\", ($sel - $have) * 724 / $CONC / 3600}")h" \
     "at 724 s per judgment -- that rate is from the saturated 172-task TMAX run; the RST and rank-4-5 runs" \
     "never saturated their concurrency, so they give no cleaner figure."
echo "watch:   curl -s http://127.0.0.1:$PORT/__stats  and  /__caps  and  /__tasks"
echo "         ls $P/rows | wc -l ; tail -f $P/launch.log ; tail -f $P/watchdog.log"
echo "STOP:    ./stop_openrouter.sh --arm $ARM --port $PORT"
echo "         It captures the shim's counters FIRST -- they are the only record of what the run cost -- and"
echo "         then stops the shim: while it listens, any local user on this box can spend the key through it"
echo "         without ever holding it."
