#!/usr/bin/env python3
"""Gate, card and publish for the tb-full-data SUPERSET commit (build_superset.py; parent ded8a2a5): TMAX and
CalibForge only (user 2026-10-09).

Gate, every check refuses:
  - the build is the superset build, full (no --limit);
  - each of the 6 split parquets has the schema agentpick's 70 columns + build_tmax_full.EXTRA + `origin` +
    `verdict_history`;
  - every *_killed row is a kill in metadata/killed_tasks.tsv, and no killed task is in a clean or repair split;
  - row counts equal the split files'; every corpus split holds only its corpus; EACH ID APPEARS ONCE across all
    splits and all split files (this replaces the release-overlap check: tb-full-data now holds the releases' rows);
  - every input the build carried rows from is the published bytes: the local build files are sha256-equal to the
    Hub's LFS objects of Fzz1/tb-full-data@ded8a2a5, Fzz1/tb-training-mix-agentpick@071b1706 and
    Fzz1/tb-training-mix@2523ef7c;
  - tars hold only tasks/<id>/ regular files for exactly their split's ids, and each package's sha equals the row's;
  - clean rows: SEED/CLEAN with a verified solution (solution_replay_reward 1.0); every solution with a recorded
    sha256 has exactly those bytes;
  - package_integrity.py, unmodified, with the cfr fetch manifests added to its CalibForge manifests, over every
    corpus here (tmax, calibforge): 0 deviations; (when Xiaomi splits exist) the Xiaomi packages equal
    agentpick's published Xiaomi packages byte for byte (that release checked them against Xiaomi's MANIFEST.tsv);
  - ExtScanner (keys + identifiers): a planted needle fires first, then 0 hits over every file uploaded;
  - canary scan: report-only.
With --push: commit on --parent, read back every uploaded file's sha256 and write PUBLISH_RECORD.json.

  HF_TOKEN=... python publish_superset.py <superset build dir> [--push --parent SHA]
"""
import collections, csv, hashlib, importlib.util, json, os, re, subprocess, sys, tarfile, tempfile

REPO_ID = "Fzz1/tb-full-data"
HERE = os.path.dirname(os.path.abspath(__file__))
MIX = os.path.dirname(HERE)
V13 = os.path.dirname(MIX)
REPO = os.path.normpath(os.path.join(V13, "..", "..", ".."))
CORPORA = ("tmax", "calibforge")          # user 2026-10-09: TMAX and CalibForge only
LABEL = {"tmax": "tmax", "calibforge": "calibforge", "rst": "rst", "terminalworld": "terminalworld",
         "terminallego": "terminallego", "xiaomi": "MiMo-V2.6-RL-oss"}
SPLITS = tuple(f"{c}_{k}" for c in CORPORA for k in (("clean", "repair", "killed") if c in ("tmax", "calibforge")
                                                      else ("clean", "repair")))
SOURCES = {  # local dir -> (Hub repo, revision): the published bytes the build carried rows from
    "scratch/full_data/final_fe662354": ("Fzz1/tb-full-data", "ded8a2a591c8ad181f8626ed60f08400db7370a3"),
    "scratch/full_data/cf_final_d13e845e": ("Fzz1/tb-full-data", "ded8a2a591c8ad181f8626ed60f08400db7370a3"),
    "reviews/overfilter_20260916/v13/tb_training_mix/build_agentpick_release6_tl": ("Fzz1/tb-training-mix-agentpick", "071b170639c63cd9d5bdad5837fd57fa2e0454c6"),
    "reviews/overfilter_20260916/v13/tb_training_mix/build_calibforge_split_r3c_fbb40fa8_fbb40fa8_off_repair": ("Fzz1/tb-training-mix-agentpick", "071b170639c63cd9d5bdad5837fd57fa2e0454c6"),
    "reviews/overfilter_20260916/v13/tb_training_mix/build_tl_split": ("Fzz1/tb-training-mix-agentpick", "071b170639c63cd9d5bdad5837fd57fa2e0454c6"),
    "reviews/overfilter_20260916/v13/tb_training_mix/build_agentpick_release8_xm": ("Fzz1/tb-training-mix-agentpick", "071b170639c63cd9d5bdad5837fd57fa2e0454c6"),
    "reviews/overfilter_20260916/v13/tb_training_mix/build": ("Fzz1/tb-training-mix", "2523ef7c7194375fa844d042e34d899ac9164e10"),
}
CFR_MANIFESTS = [os.path.join(V13, "tb21_agentic_top10/harbor/cfr/package_manifest.tsv"),
                 os.path.join(V13, "tb21_agentic_top10/harbor/cfr/package_manifest.security.tsv")]
INTEGRITY_CACHE = "/data/scratch/zekaili/cache/huggingface/hub"
BARE_FROM = re.compile(rb"^FROM aweaiteam/calibforge@sha256:[0-9a-f]{64}\n?$")
PY = sys.executable


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, [path]
    spec.loader.exec_module(m)
    sys.argv = argv
    return m


def tsv(p):
    return list(csv.DictReader(open(p), delimiter="\t"))


def card(assign: dict, rows_meta: dict, canary: int, killed: list) -> str:
    n = {s: rows_meta[s]["rows"] for s in SPLITS}
    sol = {s: rows_meta[s]["with_solution"] for s in SPLITS}
    sec = {s: rows_meta[s]["security"] for s in SPLITS}
    total = sum(n.values())
    excl = {c: collections.Counter(r["reason"].split(":")[0] for r in assign[c] if r["split"] == "none") for c in CORPORA}
    tm = assign["tmax"]
    pre = collections.Counter(r["verdict_source"].split(":", 1)[1] for r in tm if r["verdict_source"].startswith("pre_v13:"))
    leak = {c: sum(1 for r in assign[c] if r["split"] == f"{c}_repair" and r["verdict"] == "ANSWER-LEAK") for c in CORPORA}
    ks = collections.Counter(k["kill_stage"] for k in killed if k["status"] == "killed" and k["corpus"] == "tmax"
                             and k["split_now"] == "tmax_killed" and k["kill_stage"] == k["all_kill_stages"].split("+")[0])
    sup = sum(1 for k in killed if k["status"] == "superseded")
    sib = sum(1 for r in assign["calibforge"] if r["reason"].startswith("TB2.1 sibling"))
    kimi_ok = sum(1 for r in tm if "(Kimi K3 max" in r["reason"] and r["split"] == "tmax_clean")
    kimi_no = sum(1 for r in tm if "unsolved after 3 DeepSeek + 3 Kimi" in r["reason"])
    kimi_txt = (f"""- {kimi_ok + kimi_no:,} SEED/CLEAN tasks of this line that DeepSeek left unsolved after 3 attempts were rolled
  again on Kimi K3 (effort max, top_p 0.95, up to 3 attempts, stopping at the first reward 1, then one deciding
  replay). {kimi_ok:,} were solved and replay-verified and are in `tmax_clean` (`solution_model` `moonshotai/Kimi-K3`,
  `solution_source` `kimi_max_replay_verified`, `rollout_best_attempt` `kimi_max_attempt<k>`; `rollout_attempts`
  counts both models' attempts, and the DeepSeek reward columns keep the DeepSeek attempts). {kimi_no:,} remain in
  repair, unsolved after 3 DeepSeek + 3 Kimi attempts.
""" if kimi_ok + kimi_no else "")
    yaml = "\n".join(f"  - split: {s}\n    path: data/{s}.parquet" for s in SPLITS)
    split_rows = "\n".join(f"| `{s}` | **{n[s]:,}** | {sol[s]:,} | {sec[s]:,} |" for s in SPLITS)
    return f"""---
configs:
- config_name: default
  data_files:
{yaml}
task_categories:
- reinforcement-learning
language:
- en
tags:
- terminal-bench
- agentic-rl
- verifier-audit
size_categories:
- 10K<n<100K
---

# tb-full-data

**Every TMAX and CalibForge task we have reviewed with one of our prompt rubrics, in one dataset.** Each task appears
exactly once, with its most recent verdict, where it came from (`origin`), every earlier verdict
(`verdict_history`), and, where one exists, a solution verified the way the line that produced it verified it.
{total:,} tasks in {len(SPLITS)} splits: clean, repair and killed for each corpus.

| split | rows | with a solution | security-domain rows |
|---|---:|---:|---:|
{split_rows}

```python
from datasets import load_dataset
ds = load_dataset("{REPO_ID}")   # ds["tmax_clean"], ds["tmax_repair"], ds["calibforge_clean"], ...
```

## The rule

- **`<corpus>_clean`**: tier SEED, verdict CLEAN, **and** a verified solution (`solution_replay_reward` 1.0).
- **`<corpus>_repair`**: every other judged task that no earlier review killed. None is held back.
- **`<corpus>_killed`: kept for audit, NOT training data.** Every task an earlier review or gate of ours removed:
  TMAX {n['tmax_killed']:,} (first removed by: {', '.join(f'{k} {v:,}' for k, v in ks.most_common())}), CalibForge
  {n['calibforge_killed']:,} (answer leaks in `Fzz1/tb-training-mix-agentpick`'s review). The stage, the label it
  recorded and the reason are in `origin`, `verdict_history` and the split file.
- **Answer leaks found by the latest review are in repair**, with the verdict visible (TMAX {leak['tmax']:,},
  CalibForge {leak['calibforge']:,}). **Filter on `verdict` before training on a repair split**:
  `ds["tmax_repair"].filter(lambda r: r["verdict"] != "ANSWER-LEAK")`.
- **One verdict per task, the most recent of ours.** Earlier verdicts and kills are in `verdict_history` and in the
  split files (`metadata/split_assignment_<corpus>.tsv`).
- **Nothing was re-judged or re-rolled for this release.** Rows come from the lines that produced them, as those lines
  published or recorded them; `origin` names the line, dataset and revision of every row.
- **A verified solution means verified by its own line:**
  - this dataset's own TMAX and CalibForge lines: one replay of the first reward-1 attempt in a fresh sandbox graded
    1.0 (`solution_source` `t14_replay_verified`, and `kimi_max_replay_verified` for the TMAX Kimi-max retries);
  - rows from `Fzz1/tb-training-mix-agentpick`: its gate-2 replay of an agent attempt (TMAX,
    `agentpick_replay_verified`) or a strict majority of graded replays (CalibForge, `agentpick_cf_replay_majority`);
  - 24 TMAX rows from `Fzz1/Tmax-Tasks-Clean` reaudit: its published `solve.sh`, carried only where that release's
    instruction, setup and tests are byte-identical to upstream, it adds no verifier file and runs no pre-test hook
    (`ttc_reaudit_gold:identical_verifier`).
- **`metadata/killed_tasks.tsv`** lists every kill: task id, upstream and revision, the killing stage, the label as
  recorded, the reason, and the record it comes from in our working repository (`record_file`, the path with
  internal directory prefixes removed, and `record_commit`). It also lists, as `superseded`, {sup:,} TMAX tasks an
  early stage removed and a later review or release took back; those are in the split their later verdict gives.

## TMAX

`allenai/tmax-15k-open-instruct` at `7b090eca`: {n['tmax_clean'] + n['tmax_repair'] + n['tmax_killed']:,} of its
14,601 tasks.
- The tasks of this dataset's TMAX line: judged by Kimi K3 under rubric v13 (some earlier v13 verdicts), rolled out by
  DeepSeek V4.1 Flash (effort high, retries at max), replayed once.
{kimi_txt}- The tasks earlier lines had already reviewed or released: in `Fzz1/Tmax-Tasks-Clean`,
  `Fzz1/tb-training-mix-agentpick` or `Fzz1/tb-training-mix`, killed by an earlier review or gate, or reviewed under an
  earlier rubric. Where a v13 verdict exists it decides. Otherwise the task's latest pre-v13 verdict is kept **in its
  own vocabulary** ({', '.join(f'{k} {v:,}' for k, v in pre.most_common())}): `rubric` and `tier_source` name the
  stage, `verdict` holds its label, `tier` is empty, and the row is in repair (`pre_v13_verdict_only`), or in
  `tmax_killed` if an earlier stage killed it. These earlier rubrics are not mapped to v13 tiers because they
  disagree with v13 too often to be read as one: of tasks judged by both, v8 PASS became v13 SEED on 40 of 57.
- A solution an earlier line verified against **different** verifier bytes is not carried (most of
  `Fzz1/Tmax-Tasks-Clean`'s reaudit and longlongcheck solutions); those tasks are in repair unless another verified
  solution exists.
- Not here: {excl['tmax'].get('excluded_no_prompt_review', 0):,} tasks with no prompt review on record (listed by a
  release with no verdict anywhere, including tasks the 452-corpus audit held without reviewing them); each is in the
  split file with the reason.
- `audit452` is the 452-corpus static semantic audit of `Fzz1/Tmax-Tasks-Clean`'s reaudit corpus (protocol 2.2,
  independent A/B reviews with an adjudicating C on disagreement); its label is its assessment (`STATIC_RISK`,
  `SUPPORTED_WITHIN_SCOPE`). It killed no task.

## CalibForge

`AweAI-Team/CalibForge` at `fb1e7544`: all 5,431 tasks.
- 4,665 from this dataset's CalibForge line: judged by Kimi K3 (v13-harbor3); pass 1 on DeepSeek V4.1 Flash (high,
  top_p 1.0) for 544 tasks and Kimi K3 (low, top_p 0.95) for the rest; retries on Kimi K3 (max, top_p 0.95).
- 766 from `Fzz1/tb-training-mix-agentpick`: judged by DeepSeek V4.1-Flash (max), GLM-5.3 (low) where no valid V4.1
  row exists; DeepSeek Flash rollouts. Its {n['calibforge_killed']} answer-leak kills are in `calibforge_killed`.
- {sib} tasks that reuse a Terminal-Bench 2.1 task's test scaffold stay in `calibforge_repair` (reason "TB2.1
  sibling (copied test scaffold)"), as in that release.

## Not in this dataset

Other corpora we reviewed or curated are in their own datasets and not repeated here: RST, TerminalWorld,
Terminal-Lego and Xiaomi MiMo (`Fzz1/tb-training-mix-agentpick`, `Fzz1/tb-training-mix`), and
`Fzz1/SWE-Smith-Seeds-Clean`, `Fzz1/SWE-Rebench-Tasks-Clean`, `Fzz1/TB2-Vendor-38`.

## Columns

The first 70 columns are `Fzz1/tb-training-mix-agentpick`'s, with the same names, order and types; the next 50 are
this dataset's rollout and replay columns; the last two are `origin` and `verdict_history`.

- `origin`: where the row and its verdict come from: this dataset's own line, or the earlier dataset and revision it
  was published in (or the split file that listed it without shipping it), the verdict's source, the solution's
  source, and, for a killed row, the stage that killed it.
- `verdict_history`: a JSON list of every earlier verdict and kill of the task, oldest first (stage, label, and
  where a line recorded one, its class). The split files carry the same.
- `solution_source`: how the shipped solution was verified (see the rule above); empty when there is no solution.
- The 50 rollout and replay columns describe this dataset's own TMAX and CalibForge rollouts (`rollout_*`,
  `replay_*`). **They are null on every row this dataset did not roll out** (all rows from earlier lines);
  `rollout_attempts` is 0 there. Those lines' own solution and sizing facts are in the first 70 columns as they
  published them.
- Rows from earlier releases keep those releases' values in the first 70 columns, including `column_gaps`; rows a
  release listed without shipping are filled from that release's own side tables and the package, and
  `column_gaps` names every column with no source.
- CalibForge rows mix models: `solution_model` (`deepseek-flash`, `moonshotai/Kimi-K3`) and
  `solution_reasoning_effort` are the replayed attempt's. The split file gives model, effort and top_p of every
  attempt of this dataset's CalibForge rollouts.

## Packages

`data/tasks-<split>.tar` holds each task as `tasks/<task_id>/`, byte-identical to its upstream at the revisions
above, except:
- TMAX: `instruction.md`, `setup.sh`, `tests/test.sh`, plus `environment/Dockerfile` (`FROM <the task's image>` with
  the setup script inlined as comments; equal to the `dockerfile` column);
- CalibForge rows with `dockerfile_repaired` true: environment/Dockerfile is a single
  `FROM aweaiteam/calibforge@sha256:<digest>` line (what the rollout ran).

## Canary markers

The canary scan found {canary} benchmark canary marker(s) across the uploaded files. They are kept as they are.
"""


def main():
    B = os.path.abspath(sys.argv[1])
    push = "--push" in sys.argv
    parent = sys.argv[sys.argv.index("--parent") + 1] if "--parent" in sys.argv else ""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    sys.path[:0] = [MIX, HERE]
    import canary_gate
    import build_tmax_full as BT
    bd = load_mod("bd", os.path.join(MIX, "build_dataset.py"))
    api = HfApi(token=os.environ["HF_TOKEN"])
    rec = {"repo": REPO_ID, "build_dir": B, "checks": [], "files": {}}

    def ok(cond, msg):
        if not cond:
            raise SystemExit("GATE FAIL: " + msg)
        rec["checks"].append(msg)
        print("OK  ", msg, flush=True)

    brec = json.load(open(f"{B}/metadata/BUILD_RECORD.json"))
    ok(brec.get("kind") == "superset" and brec.get("limit") == 0, "the build is the full superset build")
    schema = pa.schema(list(pq.read_schema(BT.TEMPLATE)) + [pa.field(n, t) for n, t in BT.EXTRA]
                       + [pa.field("origin", pa.string()), pa.field("verdict_history", pa.string())])
    killed = tsv(f"{B}/metadata/killed_tasks.tsv")
    assign = {c: tsv(f"{B}/metadata/split_assignment_{c}.tsv") for c in CORPORA}
    ids, tbls, meta = {}, {}, {}
    for s in SPLITS:
        c = s.rsplit("_", 1)[0]
        t = pq.read_table(f"{B}/data/{s}.parquet")
        tbls[s] = t
        ok(t.schema.remove_metadata().equals(schema), f"{s}: schema = 70 + {len(BT.EXTRA)} + origin + verdict_history")
        ok(all(isinstance(json.loads(x), list) for x in t.column("verdict_history").to_pylist()),
           f"{s}: every verdict_history is a JSON list")
        ids[s] = t.column("task_id").to_pylist()
        ok(len(ids[s]) == len(set(ids[s])) == sum(1 for r in assign[c] if r["split"] == s),
           f"{s}: {len(ids[s])} unique rows = the split file's count")
        ok(set(t.column("corpus").to_pylist()) <= {LABEL[c]}, f"{s}: every row is corpus {LABEL[c]}")
        ok(all(o for o in t.column("origin").to_pylist()), f"{s}: every row names its origin")
        meta[s] = {"rows": t.num_rows, "with_solution": sum(1 for x in t.column("solution").to_pylist() if x),
                   "security": sum(1 for x in t.column("security").to_pylist() if x)}
    allids = [t for s in SPLITS for t in ids[s]]
    ok(len(allids) == len(set(allids)), f"each of the {len(allids)} task ids appears once across the {len(SPLITS)} splits")
    kill_ids = {(k["corpus"], k["task_id"]) for k in killed if k["status"] == "killed"}
    in_killed = {(s.rsplit("_", 1)[0], t) for s in SPLITS if s.endswith("_killed") for t in ids[s]}
    ok(in_killed <= kill_ids, f"every *_killed row ({len(in_killed)}) is a kill in killed_tasks.tsv")
    ok(not ({(c, t) for c in CORPORA for t in ids[f"{c}_clean"] + ids[f"{c}_repair"]} & kill_ids),
       "no killed task is in a clean or repair split")
    ok({(c, t) for c, t in kill_ids if c in ("tmax", "calibforge")} == in_killed,
       "every TMAX and CalibForge kill is in its killed split")
    ok(all(k["record_file"] and k["record_commit"] and k["kill_stage"] and k["label"] for k in killed),
       f"killed_tasks.tsv: {len(kill_ids)} killed tasks, {len(killed)} rows, each with stage, label, record and commit")
    listed = [r["task_id"] for c in CORPORA for r in assign[c]]
    ok(len(listed) == len(set(listed)), f"each of the {len(listed)} ids appears once across the {len(CORPORA)} split files")

    # the carried inputs are the published bytes
    hub = {}
    for d, (repo, rev) in SOURCES.items():
        if (repo, rev) not in hub:
            info = api.dataset_info(repo, revision=rev, files_metadata=True)
            hub[(repo, rev)] = {s.rfilename: (s.lfs.sha256 if s.lfs else None) for s in info.siblings}
    bad, n_src = [], 0
    for rel, digest in brec["inputs_sha256"].items():
        for d, (repo, rev) in SOURCES.items():
            if rel.startswith(d + "/data/"):
                n_src += 1
                hp = "data/" + rel[len(d + "/data/"):]
                if hub[(repo, rev)].get(hp) != digest:
                    bad.append(rel)
    ok(n_src >= 12 and not bad, f"{n_src} carried data files are sha256-equal to their Hub LFS objects {bad[:3]}")

    # tars
    xm_pub = {}
    for s in SPLITS:
        rowsha = dict(zip(ids[s], tbls[s].column("task_content_sha256").to_pylist()))
        have = collections.defaultdict(dict)
        with tarfile.open(f"{B}/data/tasks-{s}.tar") as tf:
            for m in tf:
                if not (m.isfile() and m.name.startswith("tasks/") and m.name.count("/") >= 2):
                    raise SystemExit(f"GATE FAIL: {s} tar has a non-file or misplaced member")
                tid, _, rel = m.name[len("tasks/"):].partition("/")
                have[tid][rel] = tf.extractfile(m).read()
        ok(set(have) == set(ids[s]), f"tasks-{s}.tar holds exactly its {len(ids[s])} ids")
        ok(not [t for t, m in have.items() if BT.MT.package_sha256(m) != rowsha[t]],
           f"tasks-{s}.tar: every package sha equals task_content_sha256")
        if s.startswith("xiaomi"):
            xm_pub.update(have)
    # Xiaomi: equal to agentpick's published packages
    pubx = {}
    for sp in ("xiaomi_clean", "xiaomi_repair"):
        with tarfile.open(os.path.join(MIX, "build_agentpick_release8_xm/data", f"tasks-{sp}.tar")) as tf:
            for m in tf:
                if m.isfile():
                    tid, _, rel = m.name[len("tasks/"):].partition("/")
                    pubx.setdefault(tid, {})[rel] = tf.extractfile(m).read()
    if "xiaomi" in CORPORA:
        ok(all(pubx.get(t) == m for t, m in xm_pub.items()) and len(xm_pub) == 8,
           "the 8 Xiaomi packages equal agentpick's published ones byte for byte")

    # solutions
    for s in SPLITS:
        cols = tbls[s].select(["tier", "verdict", "solution", "solution_replay_reward", "solution_actions_sha256"]).to_pylist()
        ok(not [r for r in cols if r["solution"] and r["solution_actions_sha256"] and hashlib.sha256(
            r["solution"].encode("utf-8")).hexdigest() != r["solution_actions_sha256"]],
           f"{s}: every solution with a recorded sha256 has exactly those bytes")
        if s.endswith("_clean"):
            ok(all(r["tier"] == "SEED" and r["verdict"] == "CLEAN" and r["solution"] and r["solution_replay_reward"] == 1.0
                   for r in cols), f"{s}: every row SEED/CLEAN with a verified solution")

    # upstream integrity: package_integrity.py over a view without the Xiaomi splits
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(f"{td}/view/data")
        for s in SPLITS:
            if not s.startswith("xiaomi"):
                for f in (f"{s}.parquet", f"tasks-{s}.tar"):
                    os.symlink(f"{B}/data/{f}", f"{td}/view/data/{f}")
        rp = os.path.join(td, "integrity.json")
        code = ("import sys; sys.path.insert(0, %r); import package_integrity as P; P.CF_MANIFESTS += %r; "
                "sys.argv = ['package_integrity.py', 'tars', %r, %r, '--report', %r]; P.main()"
                % (MIX, CFR_MANIFESTS, f"{td}/view", INTEGRITY_CACHE, rp))
        pr = subprocess.run([PY, "-c", code], capture_output=True, text=True)
        integ = json.load(open(rp)) if os.path.exists(rp) else None
    ok(integ is not None, "package_integrity.py produced a report" + ("" if integ else f" ({pr.stderr[-600:]})"))
    want = {LABEL[c]: sum(meta[s]["rows"] for s in SPLITS if s.rsplit("_", 1)[0] == c) for c in CORPORA if c != "xiaomi"}
    pcs = integ["per_corpus"]
    ok(set(pcs) == set(want) and all(pcs[c]["packages"] == want[c] and not (pcs[c]["extra"] or pcs[c]["missing"]
                                                                           or pcs[c]["changed"]) for c in want)
       and integ["security_deviations"] == 0,
       f"package_integrity: {sum(p['packages'] for p in pcs.values())} packages, "
       f"{sum(p['files_checked'] for p in pcs.values())} files, 0 deviations (by design: {integ['by_design']}); "
       f"{ {c: pcs[c]['packages'] for c in pcs} }" + ("" if not integ.get("deviations") else f" FIRST {integ['deviations'][:3]}"))
    rec["integrity"] = {"per_corpus": pcs, "by_design": integ["by_design"], "upstream_read": integ["upstream_read"]}

    files = ["metadata/killed_tasks.tsv"] + [f"metadata/split_assignment_{c}.tsv" for c in CORPORA] + \
            [f"data/{s}.parquet" for s in SPLITS] + \
            [f"data/tasks-{s}.tar" for s in SPLITS]
    can = canary_gate.gate([f"{B}/{f}" for f in files])
    rec["canary"] = {"total": can["total"], "selftest": can["selftest"],
                     "files": {os.path.relpath(p, B): {"total": r["total"], "hits": r["hits"]} for p, r in can["files"].items()}}
    ok(True, f"canary scan (report-only): {can['total']} marker(s) in the data and metadata files")
    open(f"{B}/README.md", "w").write(card(assign, meta, can["total"], killed))
    rec["canary"]["readme_total"] = canary_gate.gate([f"{B}/README.md"])["total"]
    files = ["README.md"] + files

    # extended leak scan with a positive control first
    sc = bd.ExtScanner()
    plant = sc.keys[0] if sc.keys else b""
    ok(plant and sum(sc.hits(b"x" + plant + b"y").values()) >= 1, "leak scanner fires on a planted key")
    ok(sum(sc.hits(b"/data/users/someone").values()) >= 1, "leak scanner fires on a planted home path")
    upstream = {}
    for s in SPLITS:
        with tarfile.open(f"{B}/data/tasks-{s}.tar") as tf:
            for m in tf:
                tid, _, rel = m.name[len("tasks/"):].partition("/")
                b = tf.extractfile(m).read()
                if s.startswith("tmax") and rel not in BT.MT_MEMBERS:
                    continue
                if rel == "environment/Dockerfile" and BARE_FROM.match(b):
                    continue
                upstream[tid] = upstream.get(tid, b"") + b.lower() + b"\0"
    ident = [(c, n) for c, ns in sc.ident.items() for n in ns]

    def scan(b: bytes, tid: str):
        lo = b.lower()
        keys = sum(1 for n in sc.keys if n in b)
        found = [(c, n) for c, n in ident if n in lo]
        found += [("scratch_and_session", u) for u in set(bd._UUID.findall(lo)) & sc.sessions]
        ex = [(c, n) for c, n in found if tid and n in upstream.get(tid, b"")]
        return keys, [f for f in found if f not in ex], ex

    tot, exempt, where = 0, collections.Counter(), []
    for f in files:
        path, units, hits = f"{B}/{f}", 0, 0
        if f.endswith(".parquet"):
            t = pq.read_table(path)
            tids = t.column("task_id").to_pylist()
            for col in t.column_names:
                for i, v in enumerate(t.column(col).to_pylist()):
                    units += 1
                    if v is None:
                        continue
                    b = v.encode() if isinstance(v, str) else json.dumps(v, default=str).encode()
                    k, bad_, ex = scan(b, tids[i])
                    hits += k + len(bad_)
                    exempt.update(f"{c}:{col}" for c, _ in ex)
                    if k or bad_:
                        where.append((f, col, [c for c, _ in bad_], k))
        elif f.endswith(".tar"):
            with tarfile.open(path) as tf:
                for m in tf:
                    units += 1
                    tid = m.name[len("tasks/"):].partition("/")[0]
                    k, bad_, ex = scan(m.name.encode() + b"\0" + tf.extractfile(m).read(), tid)
                    hits += k + len(bad_)
                    exempt.update(f"{c}:tar_member" for c, _ in ex)
                    if k or bad_:
                        where.append((f, m.name.split("/")[-1], [c for c, _ in bad_], k))
        else:
            units = 1
            k, bad_, _ = scan(open(path, "rb").read(), "")
            hits += k + len(bad_)
            if k or bad_:
                where.append((f, [c for c, _ in bad_], k))
        rec["files"][f] = {"bytes": os.path.getsize(path), "sha256": sha(path), "scan_units": units, "scan_hits": hits}
        tot += hits
    rec["leak_scan_exempt_upstream"] = dict(exempt)
    ok(tot == 0, f"extended leak scan: {tot} hits over {len(files)} files; identifier needles found only in the "
                 f"task's own upstream text exempted: {dict(exempt)}" + (f" FIRST: {where[:3]}" if tot else ""))
    host = sc.ident["hostname"][0]
    some = next(t for t, u in upstream.items() if host not in u)
    ok(scan(b"/data/users/x", "")[1] and scan(b"x " + host + b" y", some)[1],
       "the exemption does not cover a needle outside the task's own upstream text")

    def hub_shas(rev):
        info = api.dataset_info(REPO_ID, revision=rev, files_metadata=True)
        out = {}
        with tempfile.TemporaryDirectory() as td:
            for s in info.siblings:
                out[s.rfilename] = s.lfs.sha256 if s.lfs else sha(hf_hub_download(
                    REPO_ID, s.rfilename, repo_type="dataset", revision=rev, cache_dir=td))
        return info, out

    head0 = api.dataset_info(REPO_ID).sha
    if not push:
        json.dump(rec, open(f"{B}/metadata/PUBLISH_DRYRUN.json", "w"), indent=1, default=str)
        print("dry run: gate passed; nothing pushed; Hub head", head0)
        return
    from huggingface_hub import CommitOperationAdd
    ok(parent and head0 == parent, f"Hub head is the named parent {parent[:8]}")
    _, before = hub_shas(parent)
    ok(set(before) - {".gitattributes"} <= set(files), "every file at the parent is replaced by this commit (none left stale)")
    ops = [CommitOperationAdd(path_in_repo=f, path_or_fileobj=f"{B}/{f}") for f in files]
    ci = api.create_commit(REPO_ID, repo_type="dataset", operations=ops, parent_commit=parent,
                           commit_message="Superset: every prompt-reviewed task, 12 splits, "
                                          + ", ".join(f"{s} {meta[s]['rows']}" for s in SPLITS))
    head = api.dataset_info(REPO_ID).sha
    rec["parent"], rec["head"] = parent, head
    ok(head == ci.oid, f"Hub head is the new commit {head[:8]}")
    info, remote = hub_shas(head)
    rec["private"] = info.private
    ok(info.private is False, "repo is public")
    for f in files:
        ok(remote.get(f) == rec["files"][f]["sha256"], f"read-back {f}: sha256 equals local")
    ok(set(remote) - {".gitattributes"} == set(files), "the Hub holds exactly the uploaded files")
    json.dump(rec, open(f"{B}/metadata/PUBLISH_RECORD.json", "w"), indent=1, default=str)
    print("published", REPO_ID, head)


if __name__ == "__main__":
    main()
