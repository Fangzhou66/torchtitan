#!/usr/bin/env bash
# V4.1-Flash at effort max over the 91 Terminal-Lego ids of results_v2 (gate_results_v2_tl_ids.txt; 4 security-tagged,
# counts only) -- lead's order 2026-09-26 on the owner's word, after harbor-prompt-reviewer's GO. Same machinery as the
# cfn run (0480fcf3), with the owner's cfn changes from the start:
#   * packages fetched WITHOUT solution/ by harbor/fetch_terminal_lego.py (Lego-X/Terminal-Lego-15k@9c197f1c, every
#     file verified against the Hub) into harbor/tl; staged by stage_harbor_197.py --restage --corpus tl into
#     harbor/tl_h3tl with a READ-ONLY judge copy (pkg/, judge_copy.py) that every prompt points at;
#   * prompt v13_harbor3tl_prompt.md eee58586 (3df1dfe8), schema d8d695a6, effort max, max output 64000, max turns 60,
#     5400 s per attempt;
#   * a fresh shim from HEAD, no token ceiling, $6 per task (cumulative over attempts), run and day caps $480;
#   * gw_width_controller.py: seeded order (20260924), up to 3 attempts per task, only a quota 429 halts; width 64 from
#     the start and held (rungs 64,32: no step up); down to 32 only when more than half the requests fail over a
#     10-minute window (no row condition: --min-rows out of reach); the throughput drop is OFF (--tp-drop 0);
#   * the watchdog with ONLY the stall kill (1200 s from the last streamed byte): no output-token ceiling, no spin guard;
#   * at the end: the staged-tree check and the exec scan into caps_tl.json, security counted through
#     harbor/tl/security_screened_ids.txt (the ids carry no prefix).
set -euo pipefail
V13=/data/users/zekaili/fangzhou/tb_check/reaudit_398/reviews/overfilter_20260916/v13; SHIM=$V13/openrouter
R=$V13/tb21_agentic_top10/harbor/tl_h3tl; C=$R/ctl
EXT=/data/users/zekaili/fangzhou/tb_check/glm_claudecode/tb21_agentic_top10_run; RD=$EXT/run
WORK=$EXT/work_tl_h3tl; CFG=$EXT/cfg_tl_h3tl; GKEY=/home/zekaili/fangzhou/tb_check/mdpsk.txt
LEDGER=$SHIM/final/gateway_day_ledger.jsonl; LOG=$RD/gwproxy_tl_h3tl.log; STAMP=$(date -u +%Y%m%dT%H%M%SZ)
[ -f "$R/pkg_manifest.json" ] && [ -d "$R/prompts" ] || { echo "not staged: run stage_harbor_197.py --restage --corpus tl"; exit 2; }
if grep -lq "harbor/tl/tasks" "$R"/prompts/*; then echo "a prompt points at the shipped tree"; exit 2; fi
if curl -fsS --max-time 2 http://127.0.0.1:8800/__caps >/dev/null 2>&1; then echo "port 8800 is taken: stop it first"; exit 2; fi
[ -e "$WORK" ] && { echo "$WORK exists: a run gets a fresh work dir"; exit 2; }
mkdir -p "$C" "$WORK" "$CFG"
( cd "$SHIM" && export GATEWAY_API_KEY="$(cat "$GKEY")" && exec setsid nohup python3 "$SHIM/gw_messages_proxy.py" --port 8800 \
    --pid-file "$RD/gwproxy_tl_h3tl.pid" --task-cost-cap 6 --run-cost-cap 480 --day-cap 480 \
    --day-ledger "$LEDGER" --log-fields ) > "$LOG" 2>&1 < /dev/null & disown
for i in $(seq 1 30); do curl -fsS --max-time 1 http://127.0.0.1:8800/__caps >/dev/null 2>&1 && break; sleep 0.3; done
curl -fsS --max-time 2 http://127.0.0.1:8800/__caps >/dev/null || { echo "shim did not come up"; head -3 "$LOG"; exit 3; }
cd "$V13/tb21_agentic_top10" && env -u INCO_API_KEY setsid nohup bash -c 'echo $$ > "$1"; shift 1; exec "$@"' _ "$C/controller.pid" \
  python3 "$SHIM/gw_width_controller.py" --run-dir "$R" --ctl-dir "$C" --work "$WORK" --cfg "$CFG" --shim-log "$LOG" \
    --caps-file "$R/caps_tl.json" --events "$R/gateway_aborts.jsonl" \
    --rungs 64,32 --start-width 64 --tp-drop 0 --drop-share 0.50 --window 600 --min-rows 1000000 > "$C/controller.log" 2>&1 < /dev/null & disown
sleep 3
setsid nohup bash -c 'echo $$ > "$1"; shift 1; exec "$@"' _ "$C/watchdog.pid" \
  python3 "$SHIM/abandon_doomed.py" --stats http://127.0.0.1:8800 --work-dir "$WORK" --rows-dir "$R/rows" \
    --max-output-tokens 1000000000 --spin-tokens 1000000000 --stall-from stream --stall-sec 1200 --interval 20 --launcher-pid-file "$C/controller.pid" \
    --events "$R/abandon_events_$STAMP.jsonl" > "$C/watchdog.log" 2>&1 < /dev/null & disown
GW_RUN_DIR="$R" setsid nohup python3 "$SHIM/gw_abort_monitor.py" "$LOG" "$C/controller.pid" > "$C/abort_monitor.log" 2>&1 < /dev/null & disown
sleep 4
echo "tl start $STAMP | shim $(cat "$RD/gwproxy_tl_h3tl.pid") | controller $(cat "$C/controller.pid") | watchdog $(cat "$C/watchdog.pid")"
tail -2 "$C/controller.log"; head -1 "$C/watchdog.log"
