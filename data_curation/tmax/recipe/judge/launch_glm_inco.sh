#!/bin/bash
# GLM-5.3 at low effort over the top-10 run (the ids the top-3 run and earlier V13-level judgments did not cover):
# TMAX under the frozen v13 prompt, RST under v13-rst. One task per headless worker, three keys round-robin, a retry
# rotates the key, rows already written are skipped. 14 + 10 = 24 workers in flight.
V13=/data/users/zekaili/fangzhou/tb_check/reaudit_398/reviews/overfilter_20260916/v13
RUN=$V13/tb21_top10_glm
EXT=/data/users/zekaili/fangzhou/tb_check/glm_claudecode/tb21_top10_glm_run
KEYS="/home/zekaili/fangzhou/tb_check/4.1.txt /home/zekaili/fangzhou/tb_check/testin.txt /home/zekaili/fangzhou/tb_check/infe.txt"
export INCO_API_KEY="$(cat /home/zekaili/fangzhou/tb_check/4.1.txt)"
mkdir -p "$EXT" "$RUN/tmax/glm_low/results" "$RUN/rst/glm_low/results"
cd "$RUN" || exit 1
common="--model glm-5.3 --small-model glm-5.3-flash:fast --effort low --max-output-tokens 65536 --allowed-tools Read Write Bash --max-turns 60 --timeout 3600 --attempts 3"
setsid nohup python3 "$V13/tb21_top10/run_glm_judges.py" --prompt-dir "$RUN/tmax/glm_low/prompts" --out-dir "$RUN/tmax/glm_low/results" \
  --work-dir "$EXT/work_tmax" --config-dir "$EXT/cfg" --concurrency 14 $common --add-dir "$RUN/tmax" "$V13" \
  --key-files $KEYS --rows-root "$RUN/tmax/glm_low" > "$RUN/tmax/glm_low/launch.log" 2>&1 < /dev/null &
echo $! > "$RUN/tmax/glm_low/launcher.pid"
setsid nohup python3 "$V13/tb21_top10/run_glm_judges.py" --prompt-dir "$RUN/rst/glm_low/prompts" --out-dir "$RUN/rst/glm_low/results" \
  --work-dir "$EXT/work_rst" --config-dir "$EXT/cfg" --concurrency 10 $common --add-dir "$RUN/rst" "$V13/rst" \
  --key-files $KEYS --rows-root "$RUN/rst/glm_low" > "$RUN/rst/glm_low/launch.log" 2>&1 < /dev/null &
echo $! > "$RUN/rst/glm_low/launcher.pid"
