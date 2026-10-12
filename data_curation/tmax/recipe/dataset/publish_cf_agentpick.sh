#!/usr/bin/env bash
# The r3 CalibForge publish in one command, once the owner has sent the two switch values (lead, 2026-09-30):
#   bash build_agentpick_release9_cf/publish_cf.sh <canary off|scrub|drop> <siblings off|repair|drop> [--push]
# 1. Read the summary pins from build_dataset.py (R3_PINS, the only place they live) and their commit times.
# 2. make_cf_columns.py for both populations at those pins, under the rule and the switches, then combine them into
#    build/columns/cf_<tag>/<canary>_<siblings>/.
# 3. build_dataset.py --preset calibforge_split_r3 with the same switches.
# 4. publish_cf.py: the card, the full gate against the Hub head, and with --push the commit, the read-back and the
#    record. Without --push it is a dry run.
# Every step's exit status is checked, and a failed step stops the script.
set -euo pipefail
C=${1:?canary}; S=${2:?siblings}; PUSH=${3:-}
HERE=$(cd "$(dirname "$0")" && pwd); MIX=$(dirname "$HERE")
PY=/data/users/zekaili/fangzhou/tb_check/terminus_env/bin/python
cd "$MIX"
PINS_JSON=$("$PY" - <<'EOF'
import json, subprocess, sys
sys.argv = ["x"]
import build_dataset as b
p = b.R3_PINS
t = {k: subprocess.run(["git", "log", "-1", "--format=%cI", v], capture_output=True, text=True, check=True).stdout.strip()
     for k, v in p.items()}
print(json.dumps({"cf_rev": p["cf"], "cfn_rev": p["cfn"], "cf_time": t["cf"], "cfn_time": t["cfn"], "tag": b._R3_TAG}))
EOF
)
echo "pins: $PINS_JSON"
CF_REV=$(echo "$PINS_JSON" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["cf_rev"])')
CFN_REV=$(echo "$PINS_JSON" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["cfn_rev"])')
TAG=$(echo "$PINS_JSON" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["tag"])')
mkdir -p /data/users/zekaili/fangzhou/tb_check/reaudit_398/scratch/cf_r3_publish
RUN=$(mktemp -d /data/users/zekaili/fangzhou/tb_check/reaudit_398/scratch/cf_r3_publish/run.XXXXXX)
echo "$PINS_JSON" > "$RUN/pins.json"
for pop in cf cfn; do
  mkdir -p "$RUN/$pop"
  if [ $pop = cf ]; then extra=(--summary-rev "$CF_REV"); else extra=(--population cfn --summary-rev "$CFN_REV" --repair-needs-verified); fi
  (cd build/columns && "$PY" make_cf_columns.py "${extra[@]}" --split-prefix calibforge_ --unsolved-to-repair \
      --canary "$C" --siblings "$S" --pending-dir "$RUN/pending" --out-dir "$RUN/$pop" > "$RUN/$pop.log" 2>&1) \
      || { echo "make_cf_columns $pop failed: $RUN/$pop.log"; exit 1; }
done
OUT="build/columns/cf_$TAG/${C}_${S}"
mkdir -p "$OUT"
"$PY" - "$RUN" "$OUT" <<'EOF'
import sys, pyarrow as pa, pyarrow.parquet as pq
run, out = sys.argv[1:3]
pq.write_table(pa.concat_tables([pq.read_table(f"{run}/{p}/cf_columns.parquet") for p in ("cf", "cfn")],
                                promote_options="default"), f"{out}/cf_columns.parquet")
a = open(f"{run}/cf/split_assignment_cf.tsv").read().splitlines(True)
b = open(f"{run}/cfn/split_assignment_cf.tsv").read().splitlines(True)
assert a[0] == b[0]
open(f"{out}/split_assignment_cf.tsv", "w").write("".join(a + b[1:]))
EOF
"$PY" build_dataset.py --preset calibforge_split_r3 --public --cf-canary "$C" --cf-siblings "$S" > "$RUN/build.log" 2>&1 \
  || { echo "build failed: $RUN/build.log"; tail -5 "$RUN/build.log"; exit 1; }
BUILD="build_calibforge_split_${TAG}_${C}_${S}"
: "${HF_TOKEN:?export HF_TOKEN first}"
"$PY" "$HERE/publish_cf.py" "$BUILD" "$OUT/split_assignment_cf.tsv" "$OUT/cf_columns.parquet" "$C" "$S" "$RUN/pins.json" $PUSH
cp "$RUN/pins.json" "$HERE/pins.json"
echo "run dir: $RUN"
