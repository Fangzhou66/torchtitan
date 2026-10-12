#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH -J t14_rollout
#SBATCH --signal=B:INT@300
# 3 h slices (lead 2026-10-04): a 12 h limit sat in PENDING (Priority) behind other users' jobs with the node idle;
# backfill starts the shorter job. Successors are submitted without -t, so this header decides their limit.
# The tmax-15k never-graded rollout line (corpus t14, lead 2026-10-03): a chain of sbatch slices, each running
# t14_driver.py as a child the script waits on (no nohup/disown/setsid: the job's wall must be able to stop it).
#
#   sbatch -o <root>/slice<N>.out t14_slice.sh <N> <root> <max slices> [driver args...]
#
# Interpreter: envs/rollout_env_t14, a rebuild (2026-10-03) of the locked rollout_env, whose /tmp copy lost its files to the 10-day
# /tmp cleanup; same 20 pinned dists + the gated parser patch, 6,786/6,787 files hash-equal to the old RECORDs.
# Each slice: (1) asks the gateway (gateway_probe.py); a quota 429 pauses the line via t14_quota_wait.sh;
# (2) submits the NEXT slice first (afterany), so the chain survives a wall kill; (3) runs the driver, which
# plans from disk; (4) at the wall (SIGINT 300 s before) forwards SIGINT to the driver, which stops its lanes
# gracefully (interrupted attempts re-plan); (5) driver rc 5 (quota): cancel the next slice, submit the wait.
# Driver rc 0 with nothing left: the next slice plans 0 and ends the chain.
set -u
N="$1"; ROOT="$2"; MAX="$3"; shift 3
HERE="${T14_HOME:-/data/users/zekaili/fangzhou/tb_check/reaudit_398/reviews/overfilter_20260916/v13/tmax14k_rollout}"   # T14_HOME: another line dir (cf/)
cd "$HERE"
export PY=/data/users/zekaili/fangzhou/tb_check/envs/rollout_env_t14/bin/python
# route: the gateway's deepseek-flash by default; T14_MODEL / T14_API_BASE / T14_KEY_FILE switch it (2026-10-05, user: CF
# rest on Kimi K3 via inferact). The adapter's OpenAI-compatible chat path ("deepseek" provider) serves both.
export MODEL="${T14_MODEL:-deepseek-flash}"
export ROLLOUT_PROVIDER=deepseek
export ROLLOUT_API_BASE="${T14_API_BASE:-https://gateway.huanzhimao.com/v1}"
export ROLLOUT_KEY_FILE="${T14_KEY_FILE:-/home/zekaili/fangzhou/tb_check/mdpsk.txt}"
export T14_TOP_P="${T14_TOP_P:-}"   # top_p override (t14_gateway_retry.py); Kimi K3 on inferact accepts only 0.95
export LIMITER_RPM=2000 LIMITER_BURST=100 LIMITER_TPM=1000000000
export TMAX_GATEWAY_STREAM=1
export T14_GATEWAY_RETRY=1   # call-level retry of gateway 5xx / dropped streams (t14_gateway_retry.py via the venv's .pth)
export HF_HUB_OFFLINE=1
unset TMAX_OPENROUTER_PROVIDER
L="$ROOT/slice$N"; mkdir -p "$L"
say() { echo "slice $N job ${SLURM_JOB_ID:-?} $(date -u +%H:%M:%SZ): $*"; }
ev() { echo "{\"t\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",$1}" >> "$ROOT/line_events.jsonl"; }
pause() {   # pause <slice to resume> <driver args...>
  local S="$1"; shift
  local W; W=$(sbatch --parsable -o "$ROOT/quota_wait_for_slice$S.out" "$HERE/t14_quota_wait.sh" "$S" "$ROOT" "$MAX" "$@")
  say "QUOTA PAUSE: wait job $W resubmits slice $S after the reset"; ev "\"event\":\"quota_pause\",\"resume_slice\":$S,\"wait_job\":\"$W\""
}
"$PY" gateway_probe.py --key-file "$ROLLOUT_KEY_FILE" --api-base "$ROLLOUT_API_BASE" --model "$MODEL" >> "$ROOT/probe.log"; P=$?
if [ "$P" -eq 5 ]; then pause "$N" "$@"; exit 0; fi
[ "$P" -eq 0 ] || say "probe inconclusive (rc $P); running anyway -- the driver halts on a real quota 429"
NEXT=""
if [ "$N" -lt "$MAX" ]; then
  NEXT=$(sbatch --parsable --dependency=afterany:"${SLURM_JOB_ID}" -o "$ROOT/slice$((N + 1)).out" \
         "$HERE/t14_slice.sh" $((N + 1)) "$ROOT" "$MAX" "$@")
  say "next slice $((N + 1)) submitted as $NEXT"
fi
ev "\"event\":\"slice_start\",\"slice\":$N,\"job\":\"${SLURM_JOB_ID:-}\",\"next\":\"$NEXT\""
sha256sum "${T14_ROWS:-rows/tmax_rows_all14601.jsonl}" rows/t14s_population.tsv t14_driver.py t14_slice.sh \
  "$(readlink -f ladder.py)" "$(readlink -f set_rung.py)" "$(readlink -f run_retain.sh)" \
  "$(readlink -f retain_trajectory.py)" "$(readlink -f launcher-settings-final.json)" > "$L/inputs.sha256"
DP=""
trap 'say "SIGINT: forwarding to the driver"; [ -n "$DP" ] && kill -INT "$DP"' INT
# drain: no new lane in the last DRAIN_S (default 25 min) of the allocation, so short slices do not cut episodes at the wall
END=$(date -d "$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %e 2>/dev/null)" +%s 2>/dev/null || echo 0)
LU=""; [ "${END:-0}" -gt 0 ] && LU="--launch-until $((END - ${DRAIN_S:-1500}))"
"$PY" -u t14_driver.py --out "$L" --lane-start $(( ${LANE_BASE:-0} + N * 1000 + 1)) $LU "$@" > "$L/driver.log" 2>&1 < /dev/null &
DP=$!
ev "\"event\":\"driver_start\",\"slice\":$N,\"pid\":$DP"
wait "$DP"; RC=$?
# a trapped signal returns wait early: wait again for the driver's graceful stop
while kill -0 "$DP" 2>/dev/null; do wait "$DP"; RC=$?; done
say "driver exited rc=$RC"; ev "\"event\":\"driver_end\",\"slice\":$N,\"rc\":$RC"
if [ "$RC" -eq 0 ] && tail -1 "$L/driver_events.jsonl" 2>/dev/null | grep -q '"event": "done"'; then
  # nothing left: end the chain here (2026-10-03 20:30Z four empty slices ran before this check existed)
  [ -n "$NEXT" ] && scancel "$NEXT" && say "line done: cancelled next slice $NEXT"
  ev "\"event\":\"line_done\",\"slice\":$N"
fi
if [ "$RC" -eq 2 ] || [ "$RC" -eq 6 ]; then   # 6 = disk below the floor   # the driver refused (preflight or an empty lane): the chain stops loudly
  [ -n "$NEXT" ] && scancel "$NEXT" && say "REFUSED: cancelled next slice $NEXT"
  ev "\"event\":\"line_refused\",\"slice\":$N"
fi
if [ "$RC" -eq 7 ]; then   # gateway degradation: pause, resume via the health wait (3 healthy rounds)
  [ -n "$NEXT" ] && scancel "$NEXT" && say "cancelled next slice $NEXT"
  W=$(sbatch --parsable -o "$ROOT/health_wait_for_slice$((N + 1)).out" --export=ALL,HEALTH_ROUNDS=3,PROBE_EVERY=600 \
      "$HERE/t14_health_wait.sh" $((N + 1)) "$ROOT" "$MAX" "$@")
  say "HEALTH PAUSE: wait job $W"; ev "\"event\":\"health_pause\",\"resume_slice\":$((N + 1)),\"wait_job\":\"$W\""
fi
if [ "$RC" -eq 5 ]; then
  [ -n "$NEXT" ] && scancel "$NEXT" && say "cancelled next slice $NEXT"
  pause $((N + 1)) "$@"
fi
