#!/usr/bin/env bash
# V4.1-Flash at effort max over the per-source top-10's NEW CalibForge ids (312 not among the judged 454; 40 of them
# security) -- lead's order 2026-09-25 on the owner's word. Same machinery as the V4.1 run behind bf7e91db:
#   * packages fetched by harbor/fetch_calibforge.py --ids-tsv ... --only-new (AweAI-Team/CalibForge@fb1e7544, every
#     file verified against the Hub) into harbor/cfn; staged by stage_harbor_197.py --restage --corpus cfn into
#     harbor/cfn_h3 with a READ-ONLY judge copy (pkg/, judge_copy.py) that every prompt points at;
#   * prompt 8ebb7fcf, schema d8d695a6, effort max, max output 64000, max turns 60, 5400 s per attempt;
#   * a fresh shim from HEAD (pre-stream retry 357ec505), no token ceiling, $2 per task, run and day caps $480;
#   * gw_width_controller.py: seeded order (20260924), up to 3 attempts per task, only a quota 429 halts; width starts
#     at 100 on rungs 100/64/32 and steps down only on the owner's sustained rule (> 50% of requests in a 10-minute
#     window failing before a byte AND fewer than 4 rows written in it) -- the throughput drop is OFF (--tp-drop 0),
#     so a 524/502 burst moves nothing; steps back up after 15 minutes under 15% failures;
#   * the watchdog at HEAD (stall 1200 s from the last streamed byte, 95,000 output tokens per attempt);
#   * at the end: the staged-tree check and the exec scan (hold_before_build) into caps_cfn.json.
set -euo pipefail
V13=/data/users/zekaili/fangzhou/tb_check/reaudit_398/reviews/overfilter_20260916/v13; SHIM=$V13/openrouter
R=$V13/tb21_agentic_top10/harbor/cfn_h3; C=$R/ctl
EXT=/data/users/zekaili/fangzhou/tb_check/glm_claudecode/tb21_agentic_top10_run; RD=$EXT/run
WORK=$EXT/work_cfn_h3; CFG=$EXT/cfg_cfn_h3; GKEY=/home/zekaili/fangzhou/tb_check/mdpsk.txt
LEDGER=$SHIM/final/gateway_day_ledger.jsonl; LOG=$RD/gwproxy_cfn_h3.log; STAMP=$(date -u +%Y%m%dT%H%M%SZ)
[ -f "$R/pkg_manifest.json" ] && [ -d "$R/prompts" ] || { echo "not staged: run stage_harbor_197.py --restage --corpus cfn"; exit 2; }
if grep -lq "harbor/cfn/tasks" "$R"/prompts/*; then echo "a prompt points at the shipped tree"; exit 2; fi
if curl -fsS --max-time 2 http://127.0.0.1:8800/__caps >/dev/null 2>&1; then echo "port 8800 is taken: stop it first"; exit 2; fi
[ -e "$WORK" ] && { echo "$WORK exists: a run gets a fresh work dir"; exit 2; }
mkdir -p "$C" "$WORK" "$CFG"
( cd "$SHIM" && export GATEWAY_API_KEY="$(cat "$GKEY")" && exec setsid nohup python3 "$SHIM/gw_messages_proxy.py" --port 8800 \
    --pid-file "$RD/gwproxy_cfn_h3.pid" --task-cost-cap 2 --run-cost-cap 480 --day-cap 480 \
    --day-ledger "$LEDGER" --log-fields ) > "$LOG" 2>&1 < /dev/null & disown
for i in $(seq 1 30); do curl -fsS --max-time 1 http://127.0.0.1:8800/__caps >/dev/null 2>&1 && break; sleep 0.3; done
curl -fsS --max-time 2 http://127.0.0.1:8800/__caps >/dev/null || { echo "shim did not come up"; head -3 "$LOG"; exit 3; }
cd "$V13/tb21_agentic_top10" && env -u INCO_API_KEY setsid nohup bash -c 'echo $$ > "$1"; shift 1; exec "$@"' _ "$C/controller.pid" \
  python3 "$SHIM/gw_width_controller.py" --run-dir "$R" --ctl-dir "$C" --work "$WORK" --cfg "$CFG" --shim-log "$LOG" \
    --caps-file "$R/caps_cfn.json" --events "$R/gateway_aborts.jsonl" \
    --rungs 100,64,32 --start-width 100 --tp-drop 0 > "$C/controller.log" 2>&1 < /dev/null & disown
sleep 3
setsid nohup bash -c 'echo $$ > "$1"; shift 1; exec "$@"' _ "$C/watchdog.pid" \
  python3 "$SHIM/abandon_doomed.py" --stats http://127.0.0.1:8800 --work-dir "$WORK" --rows-dir "$R/rows" \
    --max-output-tokens 95000 --stall-from stream --stall-sec 1200 --interval 20 --launcher-pid-file "$C/controller.pid" \
    --events "$R/abandon_events_$STAMP.jsonl" > "$C/watchdog.log" 2>&1 < /dev/null & disown
GW_RUN_DIR="$R" setsid nohup python3 "$SHIM/gw_abort_monitor.py" "$LOG" "$C/controller.pid" > "$C/abort_monitor.log" 2>&1 < /dev/null & disown
sleep 4
echo "cfn start $STAMP | shim $(cat "$RD/gwproxy_cfn_h3.pid") | controller $(cat "$C/controller.pid") | watchdog $(cat "$C/watchdog.pid")"
tail -2 "$C/controller.log"; head -1 "$C/watchdog.log"
