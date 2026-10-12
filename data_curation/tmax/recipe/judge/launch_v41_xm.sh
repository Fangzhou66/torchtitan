#!/usr/bin/env bash
# V4.1-Flash at effort max over the Xiaomi MiMo tasks of the TB2.1 top-10 (31 judged; the 1 security-category id is
# counted only and not staged) -- lead's order 2026-09-29, after harbor-prompt-reviewer's GO. The Terminal-Lego run's
# machinery (launch_v41_tl.sh, 8621e3fa), unchanged but for:
#   * packages built by harbor/fetch_xiaomi.py + harbor/xm_git_history.py into harbor/xm; staged by
#     tb21_agentic_top10/stage_xiaomi.py into harbor/xm_h4 with a READ-ONLY judge copy (pkg/) every prompt points at;
#   * prompt v13_harbor4xm_prompt.md and schema output_schema_v13_harbor4xm.json (pins in stage_xiaomi.py); the
#     controller validates rows with that schema (--schema);
#   * MODE smoke: the two ids of <ctl_smoke>/order.txt only, own ctl dir; MODE full: every id without a valid row,
#     seeded order, width 64 (more than the population: all at once).
# Same as TL: effort max, max output 64000, max turns 60, 5400 s per attempt; a fresh shim, no token ceiling, $6 per
# task, run and day caps $480; up to 3 attempts per task, only a quota 429 halts; the watchdog with only the stall kill
# (1200 s); at the end the staged-tree check and the exec scan into caps_xm.json.
#   bash launch_v41_xm.sh smoke|full
set -euo pipefail
MODE="${1:?smoke|full}"
V13=/data/users/zekaili/fangzhou/tb_check/reaudit_398/reviews/overfilter_20260916/v13; SHIM=$V13/openrouter
# XM_RUN / XM_SCHEMA (optional): the opensource-code run (2026-09-29) uses XM_RUN=xm_h4c and the v13-harbor4xm-c schema;
# unset they are the terminal_bench run's, byte for byte as before
XM_RUN="${XM_RUN:-xm_h4}"
R=$V13/tb21_agentic_top10/harbor/$XM_RUN
EXT=/data/users/zekaili/fangzhou/tb_check/glm_claudecode/tb21_agentic_top10_run; RD=$EXT/run
GKEY=/home/zekaili/fangzhou/tb_check/mdpsk.txt; SCHEMA="${XM_SCHEMA:-$V13/harbor/output_schema_v13_harbor4xm.json}"
LEDGER=$SHIM/final/gateway_day_ledger.jsonl; STAMP=$(date -u +%Y%m%dT%H%M%SZ)
case "$MODE" in
  smoke) C=$R/ctl_smoke; WORK=$EXT/work_${XM_RUN}_smoke; CFG=$EXT/cfg_${XM_RUN}; LOG=$RD/gwproxy_${XM_RUN}_smoke.log
         EXTRA=(--order-file "$C/order.txt"); CAPS=$R/caps_xm_smoke.json ;;
  full)  C=$R/ctl; WORK=$EXT/work_${XM_RUN}; CFG=$EXT/cfg_${XM_RUN}; LOG=$RD/gwproxy_${XM_RUN}.log; EXTRA=(); CAPS=$R/caps_xm.json ;;
  *) echo "MODE is smoke or full"; exit 2 ;;
esac
[ -f "$R/pkg_manifest.json" ] && [ -d "$R/prompts" ] || { echo "not staged: run stage_xiaomi.py"; exit 2; }
if grep -lq "harbor/xm/tasks\|xm_packages" "$R"/prompts/*; then echo "a prompt points at the shipped tree"; exit 2; fi
if [ "$MODE" = smoke ] && [ ! -s "$C/order.txt" ]; then echo "$C/order.txt: write the two smoke workers first"; exit 2; fi
if curl -fsS --max-time 2 http://127.0.0.1:8800/__caps >/dev/null 2>&1; then echo "port 8800 is taken: stop it first"; exit 2; fi
[ -e "$WORK" ] && { echo "$WORK exists: a run gets a fresh work dir"; exit 2; }
mkdir -p "$C" "$WORK" "$CFG"
( cd "$SHIM" && export GATEWAY_API_KEY="$(cat "$GKEY")" && exec setsid nohup nice -n 5 python3 "$SHIM/gw_messages_proxy.py" --port 8800 \
    --pid-file "$RD/gwproxy_${XM_RUN}_$MODE.pid" --task-cost-cap 6 --run-cost-cap 480 --day-cap 480 \
    --day-ledger "$LEDGER" --log-fields ) > "$LOG" 2>&1 < /dev/null & disown
for i in $(seq 1 30); do curl -fsS --max-time 1 http://127.0.0.1:8800/__caps >/dev/null 2>&1 && break; sleep 0.3; done
curl -fsS --max-time 2 http://127.0.0.1:8800/__caps >/dev/null || { echo "shim did not come up"; head -3 "$LOG"; exit 3; }
cd "$V13/tb21_agentic_top10" && env -u INCO_API_KEY setsid nohup bash -c 'echo $$ > "$1"; shift 1; exec nice -n 5 "$@"' _ "$C/controller.pid" \
  python3 "$SHIM/gw_width_controller.py" --run-dir "$R" --ctl-dir "$C" --work "$WORK" --cfg "$CFG" --shim-log "$LOG" \
    --caps-file "$CAPS" --events "$R/gateway_aborts.jsonl" --schema "$SCHEMA" "${EXTRA[@]}" \
    --rungs 64,32 --start-width 64 --tp-drop 0 --drop-share 0.50 --window 600 --min-rows 1000000 > "$C/controller.log" 2>&1 < /dev/null & disown
sleep 3
setsid nohup bash -c 'echo $$ > "$1"; shift 1; exec "$@"' _ "$C/watchdog.pid" \
  python3 "$SHIM/abandon_doomed.py" --stats http://127.0.0.1:8800 --work-dir "$WORK" --rows-dir "$R/rows" \
    --max-output-tokens 1000000000 --spin-tokens 1000000000 --stall-from stream --stall-sec 1200 --interval 20 --launcher-pid-file "$C/controller.pid" \
    --events "$R/abandon_events_$STAMP.jsonl" > "$C/watchdog.log" 2>&1 < /dev/null & disown
GW_RUN_DIR="$R" setsid nohup python3 "$SHIM/gw_abort_monitor.py" "$LOG" "$C/controller.pid" > "$C/abort_monitor.log" 2>&1 < /dev/null & disown
sleep 4
echo "$XM_RUN $MODE start $STAMP | shim $(cat "$RD/gwproxy_${XM_RUN}_$MODE.pid") | controller $(cat "$C/controller.pid") | watchdog $(cat "$C/watchdog.pid")"
tail -2 "$C/controller.log"; head -1 "$C/watchdog.log"
