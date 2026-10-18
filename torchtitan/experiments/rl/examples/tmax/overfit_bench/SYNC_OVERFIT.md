# A synchronous overfit run: rewrite, then brute-force RL

What this run asks: can a rewriter teach the 9B a task it never solves, by
rewriting the task into something it can reach, letting RL fit that, and moving
back toward the original, until the original task is solved? The readout is
the original task, graded by its own verifier, as in `README.md`.

First task: TB2.1 `torch-tensor-parallelism`. The base 9B solved it 0 of 55
times across the eleven TB2.1 inline evals of an earlier 200-step run, so any
solve is the rewriter's doing.

## How it differs from `recipe.env`

The bench recipe is asynchronous (two active groups); this run is strictly
synchronous so that every rollout is on the newest policy and on the newest
task version. Everything not listed keeps `recipe.env`.

| setting | value | why |
|---|---|---|
| `SWE_MAX_ACTIVE_GROUPS`, `SWE_INITIAL_ACTIVE_GROUPS` | 1 | one group in flight |
| `SWE_NUM_ROLLOUT_WORKERS` | 1 | one active group needs one worker to keep its gate supplied |
| `SWE_ALLOW_IDLE_ROLLOUT_SLOTS` | 1 | allows one active group (the work-conserving check otherwise asks for 2) |
| `SWE_SYNC_WAIT_FOR_REWRITES` | 1 | the next group waits for this run's pending rewrites (below) |
| `SWE_GROUP_SIZE`, `SWE_ROLLOUT_CONCURRENCY` | 12 | more attempts per task version; 32 had too long a tail |
| `SWE_TB2_VAL_K` | 5 | readout = 5 attempts at the original every 5 steps |
| `SWE_TMAX_CONTEXT` / `SWE_MAX_CONTEXT_LEN` | 122880 / 120832 | 24% of the first rollouts ended at the 63.5K wall |
| `SWE_AGENT_TIMEOUT_FLOOR_SEC` | 2400 | see "budget" below |
| `SWE_EVOLUTION_HARDER_RATIO` | 0.9 | harder signal at >= 90% solved; easier at 0 |

Hardware: one 8-GPU host, everything collocated: trainer FSDP-6, one
generator, one eval generator.

## What "synchronous" means here

A step is: 12 rollouts of the current task version, all finished -> train ->
weights pushed to the generator -> the next group is drawn. Two pieces make it
so, both in this example, none in the core loop:

- The controller releases a group slot only after the weight pull, and draws
  the next sample only once it holds a slot. With one active group this alone
  gives rollout -> train -> sync -> rollout, policy age 0.
- `TMaxRollouter.get_training_sample` (runs in a thread after the slot frees)
  waits, with `SWE_SYNC_WAIT_FOR_REWRITES=1`, until every evolution signal
  this run wrote has a line in `evolution/ledger.jsonl`, then forces a mix
  reload. Rewriting therefore overlaps training but never the next rollout,
  and the next group samples the version the rewrite produced. Signals of
  earlier runs in the same job dir do not count; launcher retries of the
  same run (`--attempt-...` suffixed directories) do.
  `SWE_SYNC_REWRITE_TIMEOUT_SEC` (default 4 h) bounds the wait.

A group that solves 0/12 writes an "easier" signal, one at >= 90% a "harder"
signal; in between nothing is rewritten and steps run back to back. The curve
this produces is the back-and-forth of rewrites around a task the policy can
just reach.

## The rewriter

The reference rewriter (`evolution/evolve_ondella.py`, codex + GPT-5.6) with
`reference_rewriter.env.example`, two changes:

- `SYNTH_EFFORT=medium` (the example says high): shorter rewrites, and each one
  sits on the critical path of a synchronous run.
- `EVOLVE_PROBE_COMMAND_TIMEOUT_SEC=1200`: caps each semantic-probe command.
  The first rewrite of this task wrote a "wrong solution" witness that ran
  under `torchrun`, deadlocked on a mismatched gloo collective, and held its
  case (and the sandbox server, so even `down` hung) for the rest of the
  rewrite budget.

The second rewrite shows the probes doing their job: a wrong solution passed
the rewritten grader, and a probe-repair session tightened it before anything
was published.

## Things that bit, in order

- **Budget.** TB2.1 rows declare `agent_timeout_sec` (900 for this task) and
  the rollouter takes `max(declared, SWE_AGENT_TIMEOUT_FLOOR_SEC)`, so the
  recipe's "2400 s per attempt" holds only with the floor set to 2400.
- **Env forwarding.** A launcher that forwards a whitelist of variables to the
  remote job must include `TMAX_CONTEXT_NOTICE_STYLE`/`TMAX_CONTEXT_WARN_FRAC`
  and the three switches above, or the run silently gets the defaults.
- **Lock on a network filesystem.** `evolution/loop.lock` needs `ftruncate`;
  on a filesystem without it, link the lock to a local file.
- **Host compiler path.** Activating the conda env on the submitting machine
  puts `-ccbin=<that machine's conda c++>` into `NVCC_PREPEND_FLAGS`. Forwarded
  to the job, every JIT nvcc call there fails; the first one was the tilelang
  GDN backward at 120K. Strip `-ccbin` before forwarding.
- **Counting passes.** Grep for "PASS" in a validation `INDEX.md` also matches
  verifier output; use `bench.py`'s column regex.

## Watching it

```
bench.py readout <job dir>
plot_curve.py <job dir> --since <first run directory of this attempt> --out curve.png
```

`plot_curve.py` draws the training task's per-step solve rate (training
lineage), the original task's readout (validation traces) and each accepted
rewrite as a harder/easier marker at the step it reached training. `--since`
skips earlier runs left in the job dir.

Status: running; the result (the first step at which the original is solved,
and how many rewrites it took) goes here when there is one.
