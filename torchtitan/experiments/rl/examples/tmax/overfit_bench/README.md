# Overfit bench: does the rewriter bring the policy to solve the task?

One job trains the policy on one benchmark task and nothing else, for a fixed
number of steps. The rewriter may change what the policy trains on at any time;
the readout is always the original task, graded by its own unchanged verifier.
The policy, the recipe and the readout are the same for every job, so what
differs between two runs of a task is the rewriter, and that is what is
measured. Learning the task by heart counts: the question is only whether the
original task gets solved.

Tasks: every Terminal-Bench 2.1 task (89) and every Terminal-Bench 4 task (66).

## What the bench fixes

- **Policy and recipe**: `recipe.env` (group size 8, LR 5e-7, 800 steps,
  2400 s per attempt on every task). Hardware placement is the runner's.
- **Start**: the job's training mix is the original task alone (mix v1).
- **Readout**: the original task row, held in `data/evalsets/original.jsonl`
  and never served from the mix, 16 attempts every 5 steps. Reported as
  pass@5 per validation, the first step with any pass, and the first step from
  which pass@5 stays at or above 0.5 for three validations in a row.
- **Mix format**: a published mix whose rows the trainer cannot parse is
  ignored and training keeps the previous version. A row whose environment
  fails at rollout time is unscored (excluded from the advantage), and
  training continues.

Everything else belongs to the rewriter.

## What the rewriter decides

When to rewrite, which tasks the mix holds, how many rows, whether the
original stays in, what a rewritten task contains, which model and prompts
produce it, and whether and how it is checked before publishing. There is no
trigger from the bench: the trainer's `signals/` (written for groups whose
solved fraction crosses a threshold) exist for the reference rewriter and may
be ignored.

## Interface

A job is an experiment root (`../LAYOUT.md`). The rewriter is any process run
with that root; it reads and writes only these:

| | path | |
|---|---|---|
| read | `job.json` | bench, task, rewriter name |
| read | `data/sources/<corpus>/tasks/<task>/` | the original task package (instruction, environment, solution, tests) |
| read | `data/mix/live.jsonl`, `data/mix/history/` | the mix being trained on, and every version published |
| read | `runs/*/rollouts/<task>/g<group>-r<idx>.jsonl` | every training attempt as it completes: reward, finish reason, every turn (format in LAYOUT.md) |
| write | `evolution/` | the rewriter's own state and logs |
| publish | `publish_mix.py --job <job> --rows <file>` | the next mix version; refused whole unless every row parses |

It must not read `data/evalsets/` or `runs/*/validation_rollouts/` and
`runs/*/trainer/validation_traces/`: those are the readout.

A row is the trainer's task schema (`../data.py`, `_parse_sample_row`): the
prompt, and under `metadata` the instance id, an image or Dockerfile, and the
`tmax` grading block (`test_sh`, `fixtures`, `reward_path`).

## Running a job

```
# once per benchmark: one single-row file per task
bench.py split --evalset <bench eval jsonl> --bench tb21 --out <tasks dir>

# one job per (task, rewriter)
bench.py new --tasks <tasks dir> --task <id> --sources <bench task tree> \
    --rewriter <name> --jobs-root <dir> [--bin <codex/jq dir>]

# two processes, side by side
TRL_PROFILE=<p> RUNNER_ENV=<hardware env> ./start_training.sh <job>
<your rewriter> <job>        # or: ./start_reference_rewriter.sh <job>

# the reference solution through the job's own row, on Daytona (a check that
# the task runs as training serves it; nothing of the bench depends on it)
evolution/daytona_revalidate.py <task package> --row <tasks dir>/<id>.jsonl

# results, one line per job
bench.py readout <job> [<job> ...]
```

For TB2.1 the eval JSONL is `data/evalsets/tb2_1_tmux_f1640f428d48.jsonl`
(built by `../prepare_tb2_1_data.py`, tmux preinstalled) and the task tree is
`terminal-bench-2-1/`. TB4: `data/evalsets/tb4_tmux_493ef156d.jsonl` (66 tasks,
the same builder pointed at the TB4 tree with `--expect-tasks 66
--verify-runtime <dir> --keep-going`). Every TB4 task grades in a separate
verifier environment (`tmax.separate_verifier`), which the rollouter boots
after the agent submits and fills with the task's declared artifacts. Sandbox
requests above Daytona's per-sandbox limits (4 vCPU, 8 GiB, 10 GiB disk) run at
those limits; fp8-rmsnorm-gemm, jax-speedrun-gpu and math-eval-grader declare
an H100, which a Daytona sandbox does not have.
