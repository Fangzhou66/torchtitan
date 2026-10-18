#!/bin/bash
# Train one overfit-bench job in the foreground; a restart resumes from the
# job's newest checkpoint (launch_9b.sh finds it under SWE_CKPT_FOLDER).
#
#   TRL_PROFILE=<name> RUNNER_ENV=<hardware env> ./start_training.sh <job dir> [--dry-run]
#
# recipe.env is the bench's fixed training recipe and is loaded first, so a
# runner file cannot change it; RUNNER_ENV holds only hardware placement
# (RL_GPUS, SWE_DP_SHARD, SWE_GEN_DP, SWE_NUM_GENERATORS, SWE_GPU_MEM_LIMIT, ...).
# A value exported by the caller still wins over both, as everywhere in
# launch_9b.sh: do not export recipe keys.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB="$(cd "${1:?usage: start_training.sh <job dir>}" && pwd)"
[ -f "$JOB/job.json" ] || { echo "$JOB has no job.json; make it with bench.py new" >&2; exit 2; }

load_env_file() {
    while IFS='=' read -r k v; do
        case "$k" in ''|\#*) continue ;; esac
        [ -n "${!k-}" ] || export "$k=$v"
    done < "$1"
}
load_env_file "$HERE/recipe.env"
[ -n "${RUNNER_ENV:-}" ] && load_env_file "$RUNNER_ENV"

tag=$(python3 -c 'import json,sys; j=json.load(open(sys.argv[1])); print(j["bench"]+"-"+j["task"]+"-"+j["rewriter"])' "$JOB/job.json")
export TRL_BASE="$JOB"
export TRL_TT="${TRL_TT:-$(cd "$HERE/../../../../../.." && pwd)}"
export SWE_TB2_VAL_DATA="$JOB/data/evalsets/original.jsonl"
export SWE_CKPT_FOLDER="${SWE_CKPT_FOLDER:-$JOB/ckpt}"
export TT_DAYTONA_LABEL="${TT_DAYTONA_LABEL:-overfit-$tag}"
exec "$HERE/../runbook/launch_9b.sh" "${@:2}"
