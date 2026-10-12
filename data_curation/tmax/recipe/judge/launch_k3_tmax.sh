#!/usr/bin/env bash
# Kimi K3 (moonshotai/Kimi-K3 on api2.inferact.dev) over the TMAX re-judge scope (tmax_k3/k3_inferact, 1,106 ids: every
# TMax-15K task whose recorded rollout reward is not directly 1) -- owner's order via the lead, 2026-10-02. The xm/cfn
# machinery (launch_v41_xm.sh), pointed at inferact:
#   * the shim gw_messages_proxy.py with --upstream https://api2.inferact.dev/v1/chat/completions --gateway-model
#     moonshotai/Kimi-K3, the inferact key from its file in the shim's environment only, its own day ledger
#     (final/inferact_day_ledger.jsonl), price table = OpenRouter's moonshotai/kimi-k3 list price (an ESTIMATE);
#     port 8801 so the huanzhimao shim can keep 8800;
#   * prompt v13_prompt.md 96fbc03b, as for the shipped TMAX rows; schema = the run-local copy
#     tmax_k3/output_schema_v13_k3.json 0af911a6 (output_schema_v13.json f8a47506 with reason.maxLength 600 -> 4000 only;
#     lead 2026-10-02 "这个开大啊"); the controller validates with --schema;
#   * effort low, forced at the shim (--force-effort low; lead "effort开个low就行"); no spend limit (lead "不管花销这个key搞就行
#     了"): the shim's caps are set out of reach, its ledger is kept;
#   * width: rungs 824,512,256,128,64,32,16,8,4,2 (up to one worker per task), started at WIDTH (default 2); one rung up
#     (a doubling) after 300 s with the failure share under 15% at every poll; down one rung when more than half the
#     requests fail over 10 minutes; by hand through <ctl>/gate.json (pause, then resume on a rung);
#   * MODE smoke: the ids of <ctl_smoke>/order.txt only; MODE full: every id without a valid row.
#   * TAG (env) names a fresh ctl/work/caps/shim-log set for either mode (resume 2026-10-02: TAG=_r2); a full run with
#     <ctl>/order.txt takes that order (ids with a valid row are still skipped); RUNGS / UP_AFTER / TP_DROP (env) set the
#     ladder, the step-up hold and the throughput drop rule (defaults as before). Run it INSIDE a Slurm allocation (user 2026-10-02: every judge process in an srun shell);
#     it refuses to start without SLURM_JOB_ID.
#   * RUNPATH (env, absolute) runs any staged run dir (the rest of CalibForge, 2026-10-03: harbor/cfr_h3) and
#     SCHEMA_PATH its schema (default the TMAX run-local copy); every prompt must name <run dir>/pkg.
#   * RUNDIR (env, default k3_inferact) picks the run dir under tmax_k3 (its rows/, prompts/, ctl dirs); EFFORT (env,
#     default low) is the effort forced at the shim (user 2026-10-02: the no-row tasks retried at medium, in
#     k3_inferact_med so the low rows stay as they are). UNIQUE_TOOL_IDS=1 adds --unique-tool-ids; PORT (env) moves the
#     shim off 8801 (two arms side by side); FILL_READ_STUBS=1 (env) adds the shim's --fill-read-stubs (the
#     file-unchanged Read stub replaced by the earlier Read result; trial 2026-10-02 in k3_inferact_fill).
#   bash launch_k3_tmax.sh smoke|full [WIDTH]
set -euo pipefail
MODE="${1:?smoke|full}"; WIDTH="${2:-2}"; TAG="${TAG:-}"
RUNGS="${RUNGS:-824,512,256,128,64,32,16,8,4,2}"; UP_AFTER="${UP_AFTER:-300}"; TP_DROP="${TP_DROP:-0}"
[ -n "${SLURM_JOB_ID:-}" ] || { echo "not inside a Slurm allocation (no SLURM_JOB_ID): srun --pty first"; exit 2; }
V13=/data/users/zekaili/fangzhou/tb_check/reaudit_398/reviews/overfilter_20260916/v13; SHIM=$V13/openrouter
R="${RUNPATH:-$V13/tmax_k3/${RUNDIR:-k3_inferact}}"; PORT="${PORT:-8801}"; EFFORT="${EFFORT:-low}"
SHIMX=(); [ "${FILL_READ_STUBS:-0}" = 1 ] && SHIMX=(--fill-read-stubs)
[ -n "${STUB_DUMP:-}" ] && SHIMX+=(--dump-unmatched-stubs "$STUB_DUMP")     # diagnosis: content-free skeletons
[ "${UNIQUE_TOOL_IDS:-0}" = 1 ] && SHIMX+=(--unique-tool-ids)
EXT=/data/users/zekaili/fangzhou/tb_check/glm_claudecode/tb21_agentic_top10_run; RD=$EXT/run
IKEY=/home/zekaili/fangzhou/tb_check/inferact.txt; SCHEMA="${SCHEMA_PATH:-$V13/tmax_k3/output_schema_v13_k3.json}"
LEDGER=$SHIM/final/inferact_day_ledger.jsonl; STAMP=$(date -u +%Y%m%dT%H%M%SZ)
PRICE='{"cache_miss": 1.125, "cache_hit": 0.29, "output": 11.2, "source": "OpenRouter list price for moonshotai/kimi-k3 (openrouter.ai/api/v1/models), read 2026-10-02T03:37Z; inferact publishes none"}'
case "$MODE" in
  smoke) C=$R/ctl_smoke$TAG; WORK=$EXT/work_k3_tmax_smoke$TAG; CFG=$EXT/cfg_k3_tmax; LOG=$RD/gwproxy_k3_tmax_smoke$TAG.log
         EXTRA=(--order-file "$C/order.txt"); CAPS=$R/caps_k3_smoke.json ;;
  full)  C=$R/ctl$TAG; WORK=$EXT/work_k3_tmax$TAG; CFG=$EXT/cfg_k3_tmax; LOG=$RD/gwproxy_k3_tmax$TAG.log; CAPS=$R/caps_k3$TAG.json
         EXTRA=(); [ -s "$C/order.txt" ] && EXTRA=(--order-file "$C/order.txt") ;;
  *) echo "MODE is smoke or full"; exit 2 ;;
esac
[ -f "$R/pkg_manifest.json" ] && [ -d "$R/prompts" ] || { echo "not staged: run tmax_k3/stage_tmax_k3.py"; exit 2; }
if grep -lq "tmax_k3/tasks" "$R"/prompts/*; then echo "a prompt points at the source tree"; exit 2; fi
[ "$(grep -L "$R/pkg" "$R"/prompts/* | head -1)" = "" ] || { echo "a prompt does not name the run's read-only copy $R/pkg"; exit 2; }
if [ "$MODE" = smoke ] && [ ! -s "$C/order.txt" ]; then echo "$C/order.txt: write the smoke workers first"; exit 2; fi
if curl -fsS --max-time 2 http://127.0.0.1:$PORT/__caps >/dev/null 2>&1; then echo "port $PORT is taken"; exit 2; fi
[ -e "$WORK" ] && { echo "$WORK exists: a run gets a fresh work dir"; exit 2; }
mkdir -p "$C" "$WORK" "$CFG"
( cd "$SHIM" && export GATEWAY_API_KEY="$(cat "$IKEY")" && exec setsid nohup nice -n 5 python3 "$SHIM/gw_messages_proxy.py" --port $PORT \
    --upstream https://api2.inferact.dev/v1/chat/completions --gateway-model moonshotai/Kimi-K3 --force-effort "$EFFORT" --price "$PRICE" \
    --pid-file "$RD/gwproxy_k3_tmax_$MODE$TAG.pid" --task-cost-cap 1000000 --run-cost-cap 1000000000 --day-cap 1000000000 \
    --day-ledger "$LEDGER" --log-fields "${SHIMX[@]}" ) > "$LOG" 2>&1 < /dev/null & disown
for i in $(seq 1 30); do curl -fsS --max-time 1 http://127.0.0.1:$PORT/__caps >/dev/null 2>&1 && break; sleep 0.3; done
curl -fsS --max-time 2 http://127.0.0.1:$PORT/__caps >/dev/null || { echo "shim did not come up"; head -3 "$LOG"; exit 3; }
cd "$V13/tmax_k3" && env -u INCO_API_KEY setsid nohup bash -c 'echo $$ > "$1"; shift 1; exec nice -n 5 "$@"' _ "$C/controller.pid" \
  python3 "$SHIM/gw_width_controller.py" --run-dir "$R" --ctl-dir "$C" --work "$WORK" --cfg "$CFG" --shim-log "$LOG" \
    --caps-file "$CAPS" --events "$R/gateway_aborts.jsonl" --schema "$SCHEMA" --stats-url http://127.0.0.1:$PORT "${EXTRA[@]}" \
    --rungs "$RUNGS" --start-width "$WIDTH" --tp-drop "$TP_DROP" --drop-share 0.50 --window 600 --min-rows 1000000 \
    --up-after "$UP_AFTER" --up-window "$UP_AFTER" --up-share 0.15 --continue-unfinished 3 --max-turns 200 --attempt-timeout 7200 > "$C/controller.log" 2>&1 < /dev/null & disown
sleep 3
setsid nohup bash -c 'echo $$ > "$1"; shift 1; exec "$@"' _ "$C/watchdog.pid" \
  python3 "$SHIM/abandon_doomed.py" --stats http://127.0.0.1:$PORT --work-dir "$WORK" --rows-dir "$R/rows" \
    --max-output-tokens 1000000000 --spin-tokens 1000000000 --stall-from stream --stall-sec 1200 --interval 20 --launcher-pid-file "$C/controller.pid" \
    --events "$R/abandon_events_$STAMP.jsonl" > "$C/watchdog.log" 2>&1 < /dev/null & disown
GW_STATS_URL=http://127.0.0.1:$PORT GW_RUN_DIR="$R" setsid nohup bash -c 'echo $$ > "$1"; shift 1; exec "$@"' _ "$C/abort_monitor.pid" \
  python3 "$SHIM/gw_abort_monitor.py" "$LOG" "$C/controller.pid" > "$C/abort_monitor.log" 2>&1 < /dev/null & disown
sleep 4
echo "k3 $MODE start $STAMP | shim $(cat "$RD/gwproxy_k3_tmax_$MODE$TAG.pid") | controller $(cat "$C/controller.pid") | watchdog $(cat "$C/watchdog.pid") | abort monitor $(cat "$C/abort_monitor.pid")"
tail -2 "$C/controller.log"; head -1 "$C/watchdog.log"
