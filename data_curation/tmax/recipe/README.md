# Data-cleaning recipe: TMAX, CalibForge and the top-10 (agentpick) mix

This recipe records how three sets of splits were cleaned:
- `Fzz1/tb-full-data` @ `d8da251d`: `tmax_*` and `calibforge_*`;
- `Fzz1/tb-training-mix-agentpick` @ `071b1706`: `clean`/`repair` (TMAX, RST, TerminalWorld), `calibforge_*`,
  `terminallego_*` and `xiaomi_*`.

It contains every judge prompt and schema those splits used, the launch script for every stage, and the runtime scripts
those launchers start. Each file is a byte copy of what ran; `SHA256SUMS` lists them.

The scripts still carry the absolute paths of the machine they ran on, so read them as a record of what ran. Keys are
always read from a file and never appear here:
- `inferact.txt`: Kimi K3;
- `mdpsk.txt`: DeepSeek gateway;
- `openrouter.txt`: GLM-5.3;
- `mdaytona.txt`: Daytona sandboxes;
- `hf.txt`: Hub.

## Stage order

0. **Earlier TMAX filtering.** These stages are recorded in each row's `verdict_history` and are not re-run here.
   1. Gate 1 empty-submission sandbox run: 23 tasks scored without doing anything and were killed.
   2. Legacy build and the v1r1 / v1r2 audits.
   3. Gate 2 solve gate: gpt-5.6-luna, keep if 1 of 3 attempts passes.
   4. Over-filter, then the V7, V8 and V12 audits.
   5. The 452-corpus static audit.
   6. V12 → v13 tier crosswalk.
1. **Judge.** One task per headless Claude Code worker (`claude -p`, tools `Read Write Bash`) behind a local shim. The
   worker writes one JSON row, which must pass the schema and carry the right `task_id`. Judges read a read-only package
   copy and never execute task code.
2. **Rollout.** Up to 3 attempts per SEED/CLEAN task. Attempt 1 runs at the line's effort; retries run at effort max and
   stop at the first reward 1.
3. **Replay.** One fresh-sandbox replay of the first reward-1 episode decides whether the solution is verified.
4. **Classify.**
   - clean: SEED/CLEAN with a solution replayed at 1.0;
   - repair: every other judged task, ANSWER-LEAK included, plus SEED/CLEAN tasks unsolved after 3 attempts or not
     reproduced on replay;
   - killed: killed by an earlier stage (audit-only split);
   - excluded: no review anywhere.
5. **Kimi-max.** The 977 SEED/CLEAN tasks DeepSeek failed 3 of 3 were re-rolled on Kimi K3 max: 140 replayed at 1.0 and
   moved to clean, 837 stay in repair.
6. **Build and publish.** Each publish is one atomic Hub commit on top of the previous head, gated before upload.

## Which prompt and script made which split

| split | judge (verdict_source) | prompt | judge launch | rollout / replay | build / publish |
|---|---|---|---|---|---|
| tb-full-data `tmax_*` | Kimi K3 low: `v13_k3_rest` 10,625, `v13_k3_rejudge`/`fix`/`fill` 824 | `prompts/v13_prompt.md` | `judge/launch_k3_tmax.sh` | `rollout/t14_slice.sh` → `t14_driver.py`; `t14_replay_sbatch.sh` → `t14_replay.py`; Kimi-max `kx_replay_loop.sh` | `dataset/build_superset.py` → `apply_kimi_max.py` → `publish_superset.py` |
| | `v13_prior` 1,463: GLM-5.3 low 723; Opus 4.8 medium (V12 crosswalk) 619; top-3 GLM-5.3 + Opus 4.8, adjudicated by Opus 5 xhigh, 91; Opus 4.8 medium 29; 1 pending | `prompts/v13_prompt.md` | `judge/launch_openrouter.sh`, `judge/launch_glm_inco.sh` | same | same |
| | `pre_v13:*` 1,659 and `killed:*` 23: stage 0 labels | — | — | — | same |
| tb-full-data `calibforge_*` | Kimi K3 low: `cfr_h3` 4,665 | `prompts/v13_harbor3_prompt.md` | `judge/launch_k3_tmax.sh` (`RUNPATH=…/cfr_h3`) | `rollout/t14_slice.sh` (`T14_HOME=…/cf`) | `dataset/build_superset.py` → `publish_superset.py` |
| | DeepSeek V4.1 Flash max: `cfn_v41max` 312, `cf_v41max` 176 | `prompts/v13_harbor3_prompt.md` | `judge/launch_v41_cfn.sh` | agentpick CF line | carried from agentpick |
| | GLM-5.3 low: `cf_h3` 278 | `prompts/v13_harbor3_prompt.md` | GLM pin shim (`judge/or_pin_proxy.py` + `run_glm_judges.py`) | agentpick CF line | carried from agentpick |
| agentpick `clean`/`repair` | GLM-5.3 low: TMAX, RST, TerminalWorld | `prompts/v13_prompt.md`, `v13_rst_prompt.md`, `v13_harbor211_prompt.md` | `judge/launch_openrouter.sh --arm tmax\|rst` | agentic attempts + gate-2 replay; RST/TW oracle gate | `build_dataset.py --preset agentpick_tw` |
| agentpick `calibforge_*` | DeepSeek V4.1 Flash max; GLM-5.3 low where no V4.1 row exists | `prompts/v13_harbor3_prompt.md` | `judge/launch_v41_cfn.sh` | rollouts + majority replay | `dataset/publish_cf_agentpick.sh off repair --push` |
| agentpick `terminallego_*` | DeepSeek V4.1 Flash max | `prompts/v13_harbor3tl_prompt.md` | `judge/launch_v41_tl.sh full` | oracle gate | `build_dataset.py --preset tl_split` |
| agentpick `xiaomi_*` | DeepSeek V4.1 Flash max | `prompts/v13_harbor4xm_prompt.md`, `v13_harbor4xm_c_prompt.md` | `judge/launch_v41_xm.sh full` | rollout + network-blocked replay | `build_xm_release.py` |

**Schemas.** `python schemas/materialize.py` writes all 8 schemas from the one-line `*.min.json` files and refuses on
any sha256 mismatch:
- `output_schema_v13`, `_rst`, `_harbor21`, `_harbor3`, `_harbor4xm`, `_harbor4xm_c`;
- the Kimi K3 variants `_k3` and `_harbor3_k3`, which differ from their bases only in `reason.maxLength`, 600 → 4000.

## Commands

Run `python schemas/materialize.py` first; the commands below name the schema files it writes.

**Judge, Kimi K3 effort low** (inferact).
- Run inside an allocation: `srun -p gpu -c 48 --mem=96G -t 12:00:00 --pty bash`.
- `launch_k3_tmax.sh` starts the shim (`gw_messages_proxy.py --gateway-model moonshotai/Kimi-K3 --force-effort
  $EFFORT [--unique-tool-ids] [--fill-read-stubs]`), the width controller (`gw_width_controller.py`: 3 attempts, 3
  session resumes, schema plus `task_id` check), a stall watchdog and an abort monitor.
- `--unique-tool-ids` is required for Kimi. Kimi repeats tool-call ids across turns, and Claude Code then drops the
  repeated calls. On 40 tasks: 16/40 rows without the flag, 40/40 with it.

```bash
RUNDIR=k3_rest EFFORT=low TAG=_rest PORT=8803 UNIQUE_TOOL_IDS=1 FILL_READ_STUBS=1 \
  RUNGS=128,100,64,32 UP_AFTER=120 TP_DROP=0.25 bash judge/launch_k3_tmax.sh full 64                 # TMAX 10,625
RUNDIR=k3_inferact_rejudge EFFORT=low TAG=_rejudge PORT=8803 UNIQUE_TOOL_IDS=1 FILL_READ_STUBS=1 \
  RUNGS=128,100,64,32 UP_AFTER=120 TP_DROP=0.25 bash judge/launch_k3_tmax.sh full 128                # + _fix, _fill: the 824
RUNPATH=<harbor>/cfr_h3 SCHEMA_PATH=schemas/output_schema_v13_harbor3_k3.json EFFORT=low TAG=_cfr PORT=8803 \
  UNIQUE_TOOL_IDS=1 FILL_READ_STUBS=1 RUNGS=128,100,64,32 UP_AFTER=120 TP_DROP=0.25 bash judge/launch_k3_tmax.sh full 64   # CalibForge 4,665
```

**Judge, DeepSeek V4.1 Flash effort max** (the same shim at its defaults: `deepseek-flash` on the gateway).
- Max output 64,000 tokens, 60 turns, 5,400 s per attempt.
- 3 attempts, seeded order 20260924; only a quota 429 halts the run.

```bash
bash judge/launch_v41_cfn.sh full     # CalibForge (cfn)
bash judge/launch_v41_tl.sh full      # Terminal-Lego
bash judge/launch_v41_xm.sh full      # Xiaomi terminal_bench
XM_RUN=xm_h4c XM_SCHEMA=schemas/output_schema_v13_harbor4xm_c.json bash judge/launch_v41_xm.sh full   # Xiaomi opensource-code
```

**Judge, GLM-5.3 effort low** (OpenRouter pinned to `baidu/fp8`).
- `or_pin_proxy.py` adds the provider pin and a $1.50 per-task cap.
- Each worker runs `run_glm_judges.py --model z-ai/glm-5.3 --effort low --max-output-tokens 64000 --max-turns 60
  --timeout 2400 --attempts 2`.
- The earlier inco route is `launch_glm_inco.sh`.

```bash
bash judge/launch_openrouter.sh --arm tmax --ranks N --concurrency 16
bash judge/launch_openrouter.sh --arm rst --concurrency 16
```

**Rollout and replay** (`rollout/`; interpreter `envs/rollout_env_t14`).
- Each `t14_slice.sh` slice is a 1 h sbatch job. It submits its successor first, then runs `t14_driver.py` in fill mode.
  It stops launching new work 25 min before the wall time, and pauses on a quota limit or when 30% or more of attempts
  are killed.
- `T14_GATEWAY_RETRY=1` installs `t14_gateway_retry.py`, via the `.pth` line in `t14_gateway_retry.pth.installed`. It
  retries each call on 5xx or a dropped stream (4 retries, after 15/30/60/120 s) and sends `T14_TOP_P`. inferact accepts
  Kimi only with top_p 0.95.

```bash
# TMAX: DeepSeek V4.1 Flash, attempt 1 at effort high; retries at max (stop at the first reward 1)
sbatch rollout/t14_slice.sh 2 <root> 40 --include-security-domain --target 300 --chunk 75 --lanes 40 --min-batch 20 \
  --create-total 64 --compress --min-free-gb 50 --scope-cmd "$PY t14_scope.py --out-dir rows" --scope-every 600
sbatch --export=ALL,LANE_BASE=100000 rollout/t14_slice.sh 1 <root> 40 --include-security-domain --target 300 --chunk 75 \
  --lanes 40 --min-batch 20 --create-total 64 --compress --min-free-gb 50 --retries 8 --pass2-prefix t14m \
  --pass2-effort max --pass2-per-task 2 --scope-cmd "$PY t14_scope.py --include-tier-only --out-dir rows" --scope-every 600
# CalibForge: Kimi K3, attempt 1 at effort low, retries at max, top_p 0.95
sbatch --export=ALL,T14_HOME=<line>/cf,T14_ROWS=<line>/cf/rows/cfn_rollout_rows.jsonl,LANE_BASE=0,T14_MODEL=moonshotai/Kimi-K3,T14_API_BASE=https://api2.inferact.dev/v1,T14_KEY_FILE=<keys>/inferact.txt,T14_TOP_P=0.95 \
  rollout/t14_slice.sh 2 <root> 400 --include-security-domain --prefix cfrp --target 128 --chunk 75 --lanes 40 --min-batch 20 \
  --create-total 64 --compress --min-free-gb 25 --retries 8 --effort low --pass2-prefix cfrm --pass2-effort max --pass2-per-task 2 \
  --scope-cmd "$PY cf_scope.py" --scope-every 600
# Kimi-max: the 977 SEED/CLEAN tasks DeepSeek failed 3 of 3; all attempts at effort max
sbatch --export=ALL,LANE_BASE=70000,T14_MODEL=moonshotai/Kimi-K3,T14_API_BASE=https://api2.inferact.dev/v1,T14_KEY_FILE=<keys>/inferact.txt,T14_TOP_P=0.95 \
  rollout/t14_slice.sh 1 <root> 200 --ids-file <977 ids> --top-ids <ids that started at 4/8/10> --include-security-domain \
  --prefix kxa --target 128 --chunk 75 --lanes 40 --min-batch 20 --create-total 64 --compress --min-free-gb 25 --retries 8 \
  --effort max --pass2-prefix kxr --pass2-effort max --pass2-per-task 2
# deciding replays
sbatch -c 8 --mem=32G -t 12:00:00 rollout/t14_replay_sbatch.sh <replays>/t14rp_<ts> --concurrency 100
sbatch -c 4 --mem=8G -t 03:00:00 rollout/t14_replay_sbatch.sh <replays>/t14rp_<ts>_ungraded --retry-ungraded --concurrency 50
sbatch rollout/kx_replay_loop.sh          # replay loop: submits its successor first, writes replays_done
```

**Build and publish tb-full-data** (`dataset/`; interpreter `terminus_env`; `HF_TOKEN` read from `hf.txt`).
- `publish_superset.py` without `--push` is a dry run. Its gate refuses when any of these fails:
  - the schema matches;
  - row counts match the split file;
  - each id appears once across all splits;
  - inputs are sha-equal to the Hub's;
  - each package sha equals its row's;
  - every clean row has a solution replayed at 1.0;
  - package integrity holds;
  - the leak scan is clean;
  - after upload, every file reads back equal.

```bash
srun -n 1 -t 60  $PY dataset/build_superset.py --out-dir <S>/superset_tc2
srun -n 1 -t 60  $PY dataset/apply_kimi_max.py --base <S>/superset_tc2 --out-dir <S>/superset_kx2 --rev 7fd8cd9f
srun -n 1 -t 180 $PY dataset/publish_superset.py <S>/superset_kx2 --push --parent 0bc1288b047922307e1fe75e57afbb26145347e6   # -> d8da251d
```

The agentpick `calibforge_*` publish was `timeout 2400 bash dataset/publish_cf_agentpick.sh off repair --push`. That
script runs three steps: make columns, build, then gated publish.

## Not included

These are named here but not copied:
- the per-line builders that `build_superset.py` imports (`build_tmax_full.py`, `build_cf_full.py`) and the agentpick
  builders (`build_dataset.py`, `compose_release.py`, `build_xm_release.py`);
- the rollout harness the driver calls (`run_retain.sh`, `ladder.py`, `replay_actions.py`, `trajectory_to_actions.py`);
- the gate driver for the oracle/empty gates (`gate_driver.py`);
- the stall watchdog and abort monitor (`abandon_doomed.py`, `gw_abort_monitor.py`);
- per-run staging and fetch scripts, tallies, tests and logs.
