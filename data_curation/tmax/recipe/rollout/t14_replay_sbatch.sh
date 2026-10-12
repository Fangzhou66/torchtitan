#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH -J t14_replay
#SBATCH --signal=B:INT@120
# Gate-2 replays of the t14 reward-1 episodes (t14_replay.py), the driver a child the job waits on (no nohup/disown).
#   sbatch -o <out>/slurm.out t14_replay_sbatch.sh <out> [t14_replay.py args...]
# Resumable: a re-submission with a new <out> of the same family skips every finished replay.
set -u
OUT="$1"; shift
HERE="${T14_HOME:-/data/users/zekaili/fangzhou/tb_check/reaudit_398/reviews/overfilter_20260916/v13/tmax14k_rollout}"   # T14_HOME: another line dir (cf/)
PY=/data/users/zekaili/fangzhou/tb_check/envs/rollout_env_t14/bin/python
cd "$HERE"; mkdir -p "$OUT"
export DAYTONA_API_KEY="$(tr -d '\r\n' < /home/zekaili/fangzhou/tb_check/mdaytona.txt)"
export TT_DAYTONA_EPHEMERAL=1 TT_DAYTONA_AUTO_STOP_MIN=60 TT_DAYTONA_HEARTBEAT_SEC=180
export TT_DAYTONA_CREATE_CONCURRENCY="${TT_DAYTONA_CREATE_CONCURRENCY:-16}" TT_DAYTONA_MAX_MEM_GB=8 TT_DAYTONA_CREATE_RETRIES=8
sha256sum t14_replay.py "$(readlink -f "$HERE/../tb_training_mix/rollout_smoke/replay_actions.py")" > "$OUT/inputs.sha256"
"$PY" -u t14_replay.py --out "$OUT" "$@" > "$OUT/console.log" 2>&1 < /dev/null &
PID=$!
trap 'kill -INT "$PID"' INT
echo "{\"slurm_job_id\":\"${SLURM_JOB_ID:-}\",\"driver_pid\":$PID,\"started_utc\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" > "$OUT/allocation.json"
wait "$PID"; RC=$?
while kill -0 "$PID" 2>/dev/null; do wait "$PID"; RC=$?; done
echo "driver exited rc=$RC $(date -u +%H:%M:%SZ)"
exit "$RC"
