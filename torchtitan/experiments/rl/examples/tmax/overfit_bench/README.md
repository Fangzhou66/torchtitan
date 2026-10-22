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

- **Policy and recipe**: `recipe.env` (group size 12, LR 3e-6, 200 steps of
  one group each, 8 groups in flight, groups up to 8 steps old still trained).
  Hardware placement is the runner's.
- **A training attempt** is 60 Terminus turns and nothing else
  (`SWE_BENCH_MAX_TURNS`): the agent has no wall-clock deadline, no budget
  floor applies, and the rollout's outer guard is only a 12 h sanity bound
  (`SWE_BENCH_TURN_CAP_GUARD_SEC`). Grading keeps the task's own verifier
  timeout. The recipe used to say "2400 s per attempt on every task"; that was
  never what ran. `SWE_TIME_BUDGET_SEC` applies only to a row that declares no
  agent timeout, and every benchmark row declares one, so an attempt got
  max(declared, `SWE_AGENT_TIMEOUT_FLOOR_SEC`) -- with the 7200 s floor of
  `runbook/rltrain.env`, 7200 s for 88 of the 89 TB2.1 tasks, 12000 s for the
  other, 28800 s for every TB4 task -- and `TMAX_TERMINUS_MAX_TURNS` (120)
  turns.
- **Sandbox**: every sandbox, training and readout, and a separate verifier's,
  gets 4 vCPU (`SWE_BENCH_SANDBOX_CPU`), whatever the row declares; memory and
  disk are the row's, clamped to Daytona's per-sandbox caps (8 GiB, 10 GiB;
  with a GPU 192 GiB, 512 GiB). A GPU task that declares 16 vCPU therefore
  runs on 4.
- **Start**: the job's training mix is the original task alone (mix v1).
- **Readout**: the original task row, held in `data/evalsets/original.jsonl`
  and never served from the mix, 16 attempts every 5 steps. A readout attempt
  has the same budget as a training attempt (60 turns, no wall-clock
  deadline), so the original is read under the conditions the policy trains in.
  Reported as
  pass@5 per validation, the first step with any pass, and the first step from
  which pass@5 stays at or above 0.5 for three validations in a row.
- **Mix format**: a published mix whose rows the trainer cannot parse is
  ignored and training keeps the previous version. A row whose environment
  fails at rollout time is unscored (excluded from the advantage), and
  training continues.

Everything else belongs to the rewriter.

## What the rewriter decides

When to rewrite, which versions the pool holds and at what weight, whether
the original stays in, which version each group trains on, what a rewritten
task contains, which model and prompts produce it, and whether and how it is
checked before publishing. There is no trigger from the bench: the trainer's
`signals/` (written for groups whose solved fraction crosses a threshold) and
its events exist for the rewriter and may be ignored.

Two things it may not do, whatever else it decides:

- **Read the readout**: `data/evalsets/`, `runs/*/validation_rollouts/` and
  `runs/*/trainer/validation_traces/` are the measurement, not an input.
- **Run extra student rollouts**: the policy is sampled only by training (and
  the readout). A rewriter learns how the policy does from the training
  attempts the bench already records, not by running the policy itself; the
  policy canary controller is refused in bench mode for this reason.

Training never waits for the rewriter. A draw asks it at most 1 s for a
version and otherwise draws from the pool; a rewrite in progress, a slow hook
or a dead rewriter changes what is drawn, never when.

## Interface

What training draws from is a **version pool**. A version is one task row,
identified by its content sha (the row's `content_revision`, also the
lineage's `sample_revision`; `task@r<rev>` is only a label), with a weight, a
parent version, when it was added, and whether it is active. Each group draws
one active version, by default with probability proportional to its weight.
The pool lives in the controller (`scheduler.py`), records every change in
`runs/<run>/scheduler/pool.jsonl`, keeps its whole state in
`data/mix/pool.json`, and republishes its active rows as the next mix version
whenever the active set changes, so `data/mix/` always holds what training
draws from. A rewriter talks to it through either interface below, or both.

### Socket

`<job>/scheduler.json` names a unix socket (under `/tmp`, node-local: the
rewriter runs on the trainer's host). JSON lines; `rewriter_client.py` is the
client, `example_rewriter.py` a minimal rewriter on it, and `scheduler.py`'s
docstring the wire format. The scheduler starts at the trainer's first draw;
`RewriterClient.connect()` waits for it (30 min by default, logging while it
waits), past a `scheduler.json` a killed earlier run left behind.

| call | |
|---|---|
| `add(row, weight=1, parent=None)` | a row, checked as the trainer parses it, becomes an active version |
| `reweight(version, weight)` | weight >= 0; 0 keeps it active but undrawn |
| `retire(version)` | no longer drawn |
| `get_state()` | every version: id, task, rev, weight, parent, added_at, status |
| hook `which_version(candidates)` | asked at every draw; answer an active version id, or null; no valid answer within 1 s means the weighted draw |

Events, training groups only (never the readout), each also in
`runs/<run>/scheduler/events.jsonl`:

| event | fields |
|---|---|
| `rollout_done` | version, task, rev, group, idx, reward (None = unscored), finish_reason, record_path, policy_versions |
| `group_done` | version, task, rev, group, passed, scored, policy_version |
| `step_done` | step, policy_version (after the step's weights reach the generator) |
| `pool_drained` | task (its active weight just reached 0) |

A rewriter that stops reading loses its oldest events (counted), never stalls
training; `events.jsonl` has them all. With every weight at 0 the pool draws
uniformly among active versions, and with none active it reuses the last one
retired, so a drained pool still trains.

### Files

The rewriter is any process run with the job root; it reads and writes only these:

| | path | |
|---|---|---|
| read | `job.json` | bench, task, rewriter name |
| read | `data/sources/<corpus>/tasks/<task>/` | the original task package (instruction, environment, solution, tests) |
| read | `data/mix/live.jsonl`, `data/mix/history/` | the mix being trained on, and every version published |
| read | `runs/*/rollouts/<task>/g<group>-r<idx>.jsonl` | every training attempt as it completes: reward, finish reason, every turn (format in LAYOUT.md) |
| write | `evolution/` | the rewriter's own state and logs |
| read | `runs/*/scheduler/events.jsonl`, `data/mix/pool.json` | the socket's events and the pool, as files |
| publish | `publish_mix.py --job <job> --rows <file>` | the next mix version; refused whole unless every row parses |

A file publish names the whole active set: the scheduler imports it within
2 s, adding each new row as a version (its parent is the version of the same
task the publish drops, whose weight it inherits) and retiring every active
version the file leaves out.

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
                             # or: ./example_rewriter.py <job>

# the reference solution through the job's own row, on Daytona (a check that
# the task runs as training serves it; nothing of the bench depends on it)
evolution/daytona_revalidate.py <task package> --row <tasks dir>/<id>.jsonl

# results, one line per job
bench.py readout <job> [<job> ...]
```

For TB2.1 the eval JSONL is `data/evalsets/tb2_1_tmux_f1640f428d48.jsonl`
(built by `../prepare_tb2_1_data.py`, tmux preinstalled) and the task tree is
`terminal-bench-2-1/`. TB4: `data/evalsets/tb4_tmux_493ef156d.jsonl` (66 tasks,
the same builder pointed at the TB4 release tree -- `terminal-bench-prebuilt-v4.0.0`
from the v4.0.0 GitHub release, where every image is digest-pinned -- with
`--expect-tasks 66 --verify-runtime <dir> --keep-going`). Every TB4 task grades in
a separate verifier environment (`tmax.separate_verifier`), which the rollouter
boots after the agent submits and fills with the task's declared artifacts. A
task that declares more carries it in its row (see `../harbor_env.py`): an H100
in region earth (`--gpu-target`) for fp8-rmsnorm-gemm, jax-speedrun-gpu and
math-eval-grader; its docker-compose project, run on one docker-in-docker
sandbox (11 tasks); the image's own non-root user for the agent (3); a
healthcheck, collect hooks, verifier env and no-internet. That eval file
predates these fields; rebuild it to get them. A sandbox request above Daytona's
per-sandbox limits (4 vCPU, 8 GiB, 10 GiB disk; with a GPU 16 vCPU, 192 GiB,
512 GiB) is clamped to them; 20 tasks have one. For a compose task that clamps
the docker-in-docker host only: the compose project keeps the task's declared
cpu/memory reservations, which can then exceed the host's.

The rewriter can be started before the trainer: `example_rewriter.py` and any
rewriter on `RewriterClient.connect()` wait for the scheduler, and the
reference rewriter runs file-based until it can connect.

The reference rewriter (`start_reference_rewriter.sh`) is the evolve loop run
through `rewriter_client.py reference`: it wakes on `group_done` rather than
polling for signal files, and gives each fold to the pool as add(new) +
retire(parent) as well as publishing it.

`SYNC_OVERFIT.md` describes a strictly synchronous variant of a job (one group
in flight, the next group waiting for the rewrites it triggered) and how it is run.
