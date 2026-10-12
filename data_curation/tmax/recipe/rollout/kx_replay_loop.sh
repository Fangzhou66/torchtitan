#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -J kx_replay
# Gate-2 replays of the Kimi-max line (lead 2026-10-11: the 977 TMAX SEED/CLEAN tasks DeepSeek left unsolved after 3): replay every
# reward-1 episode, same Slurm setup). 1 h jobs (the length that backfills here): each replays what has landed (resumable
# family kxrp_*), stops by SIGINT 5 min before the wall, and resubmits itself until the Kimi-max line has logged line_done AND
# nothing is left to replay.
set -u
CF=/data/users/zekaili/fangzhou/tb_check/reaudit_398/reviews/overfilter_20260916/v13/tmax14k_rollout   # the TMAX line dir
PY=/data/users/zekaili/fangzhou/tb_check/envs/rollout_env_t14/bin/python
cd "$CF"
# successor first (afterany), so a wall kill (TIMEOUT) cannot end the chain (2026-10-06: job 14780 timed out in its idle
# sleep before the line's line_done and nothing resubmitted it); a finished loop cancels it below
NEXT=$(sbatch --parsable --dependency=afterany:"${SLURM_JOB_ID:-0}" -o "$CF/replays/kxloop_$(date -u +%H%M%S).out" "$CF/kx_replay_loop.sh")
WALL_END=$(date -d "$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %e 2>/dev/null)" +%s 2>/dev/null || echo 0)
O=$CF/replays/kxrp_$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$O"
export T14_HOME=$CF
timeout -s INT 3300 bash "$CF/t14_replay_sbatch.sh" "$O" --p1-prefix kxa --p2-prefix kxr --concurrency 100
sleep 30
DONE=$(grep -c '"event":"line_done"' "$CF/slices/kimi_max/line_events.jsonl" 2>/dev/null); DONE=${DONE:-0}
TODO=$("$PY" t14_replay.py --out "$CF/replays/kxrp_plan" --p1-prefix kxa --p2-prefix kxr --plan-only 2>/dev/null \
       | python3 -c "import sys,json; print(json.loads(sys.stdin.read().strip().splitlines()[-1])['todo'])" 2>/dev/null || echo 1)
echo "{\"t\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"run\":\"$(basename "$O")\",\"cf_line_done\":$DONE,\"todo_after\":$TODO}" >> "$CF/replays/kx_replay_loop.jsonl"
if [ "$DONE" -gt 0 ] && [ "$TODO" -eq 0 ]; then
  echo "{\"t\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"event\":\"replays_done\"}" >> "$CF/replays/kx_replay_loop.jsonl"
  [ -n "$NEXT" ] && scancel "$NEXT"
  exit 0
fi
# nothing to replay yet: wait inside this job instead of spinning 1-minute jobs (2026-10-05: the loop resubmitted every
# minute), but never past the wall (the successor starts when this job ends, however it ends)
if [ "$TODO" -eq 0 ]; then
  LEFT=$(( ${WALL_END:-0} - $(date +%s) - 120 )); [ "${WALL_END:-0}" -gt 0 ] || LEFT=600
  S=$(( LEFT < 2400 ? LEFT : 2400 )); [ "$S" -gt 0 ] && sleep "$S"
fi
