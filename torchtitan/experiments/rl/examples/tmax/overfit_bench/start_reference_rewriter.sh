#!/bin/bash
# Run the reference rewriter (this repository's evolve loop) over one job, in
# the foreground. Any other rewriter replaces this script; README.md says what
# it reads and how it publishes. It runs through rewriter_client.py, which
# wakes the loop on the scheduler's group_done and gives each fold to the pool;
# with no scheduler (SWE_BENCH_SCHEDULER=0) it is the plain file-based loop.
#
#   REWRITER_ENV=<env file> EVOLVE_WORK_ROOT=<node-local dir> ./start_reference_rewriter.sh <job dir>
#
# REWRITER_ENV holds the loop's settings and credentials; see
# reference_rewriter.env.example.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB="$(cd "${1:?usage: start_reference_rewriter.sh <job dir>}" && pwd)"
[ -f "$JOB/job.json" ] || { echo "$JOB has no job.json; make it with bench.py new" >&2; exit 2; }
: "${REWRITER_ENV:?set REWRITER_ENV (see reference_rewriter.env.example)}"
: "${EVOLVE_WORK_ROOT:?set EVOLVE_WORK_ROOT to a node-local scratch directory}"
set -a; . "$REWRITER_ENV"; set +a
export TRL_BASE="$JOB"
export TRL_TT="${TRL_TT:-$(cd "$HERE/../../../../../.." && pwd)}"
export PYTHONPATH="$TRL_TT"
: "${TRL_VENV:=/scratch/gpfs/TRIDAO/al9080/titan-rl}"
mkdir -p "$EVOLVE_WORK_ROOT"
exec "$TRL_VENV/bin/python" "$TRL_TT/torchtitan/experiments/rl/examples/tmax/overfit_bench/rewriter_client.py" \
    reference "$JOB" --interval 120 --workers 16 >> "$JOB/evolution/loop.log" 2>&1
