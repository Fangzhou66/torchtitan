#!/usr/bin/env python3
"""Fzz1/tb-full-data as the SUPERSET -- TMAX and CalibForge only (user 2026-10-09, "这几个没必要啊"; the other corpora's
code is kept behind OTHER_CORPORA) (user, relayed by the lead 2026-10-09: "把prompt review过的之前的也都放到最全的也就是我们
最新的dataset"): every task that went through one of our prompt reviews, in one dataset, each id exactly once.

THE INPUTS, read, never re-judged or re-rolled:
  tmax_*        the published TMAX build (build_tmax_full.py, rollouts at fe662354), plus the 3,073 TMAX ids it excluded:
                2,969 with a prompt-review record, 104 without one (excluded, reason excluded_no_prompt_review)
  calibforge_*  the published CF build (build_cf_full.py, rollouts at d13e845e), plus agentpick's 766 CF ids: its 627
                published rows as published, and its 139 leak kills from its own side columns and the fetched packages
  rst_*         RST ids judged under v13-rst: agentpick's 85 and tb-training-mix's 214 (279; 1 never judged, excluded)
  terminalworld_*, terminallego_*, xiaomi_*   agentpick's judged TerminalWorld (99), Terminal-Lego (91) and Xiaomi
                terminal_bench (8) ids: published rows as published, the rest from agentpick's side columns + packages
Not prompt-reviewed, so not here: SWE-Smith, SWE-Rebench, xiaomi_swe, the unjudged Xiaomi opensource-code ids.

KILLS (user 2026-10-09: "kill的还是不用放", then "kill的就放在kill split吧新开两个cal和tmax被kill的", "kill的带上原因放进metadata"):
a task an earlier line killed never enters a clean/repair split. TMAX kills (the partition's killed-earlier stage,
agentpick's TMAX kills, tb-training-mix's TMAX leaks) go to tmax_killed and agentpick's CalibForge leak kills to
calibforge_killed, for audit, not training. RST / TerminalWorld / Terminal-Lego kills are in no split.
metadata/killed_tasks.tsv lists every kill with its stage, recorded label, reason and the local record (file + commit),
plus, as status superseded, every TMAX id whose earlier partition kill a later review or release took back.
The 38 CalibForge TB2.1 siblings stay in calibforge_repair (the owner's decontamination ruling for agentpick).

THE RULE (as tb-full-data's): clean = tier SEED, verdict CLEAN and a verified solution, verified as the line that
produced it verified it; every other judged id -> repair, never held. A verdict is the most recent of our rubric
verdicts. A pre-v13 label (v1-v6, V7, overfilter, v8, v12) is kept in its own vocabulary with NO tier mapping
(measured: v8 PASS -> v13 SEED on 40/57, v1-v6 CLEAN on 3/9), so its row is repair (pre_v13_verdict_only).

TMAX packages are the upstream bytes plus the rendered Dockerfile (build_tmax_full.package). A solution is carried
onto an added TMAX row only if it was verified against byte-identical instruction.md / setup.sh / tests/test.sh:
agentpick's replay-verified action sequences (its gate-2 replay; package sha must equal ours), else
Tmax-Tasks-Clean's reaudit solve.sh where its verifier files equal upstream, it adds no verifier file and runs no
pre-test hook.

SCHEMA: the published 120 columns + `origin` (which line/dataset/revision the row and its verdict come from) +
`verdict_history` (JSON: every earlier verdict and kill of the task, oldest first). The split files repeat both. Task text is copied as bytes, never printed;
security rows get no hint and no pins; reports give counts.

  $PY build_superset.py --out-dir <fresh dir>
"""
from __future__ import annotations

import argparse, collections, csv, glob, hashlib, io, json, os, re, subprocess, sys, tarfile, time

import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
import build_tmax_full as BT                                             # noqa: E402
sys.path.insert(0, BT.MIX)
import build_rst_columns as BR                                           # noqa: E402

MIX, V13, REPO = BT.MIX, BT.V13, BT.REPO
OF = os.path.dirname(V13)
SCR = os.path.join(REPO, "scratch", "full_data")
TMAX_BUILD = os.path.join(SCR, "final_fe662354")
CF_BUILD = os.path.join(SCR, "cf_final_d13e845e")
AP = os.path.join(MIX, "build_agentpick_release6_tl")                    # clean/repair as published (sha-matched)
AP_CF = os.path.join(MIX, "build_calibforge_split_r3c_fbb40fa8_fbb40fa8_off_repair")
AP_TL = os.path.join(MIX, "build_tl_split")
AP_XM = os.path.join(MIX, "build_agentpick_release8_xm")
TM = os.path.join(MIX, "build")                                          # Fzz1/tb-training-mix@2523ef7c
CF_SIDE = os.path.join(MIX, "build/columns/cf_r3c_fbb40fa8_fbb40fa8/off_repair/cf_columns.parquet")
RST_SIDE = os.path.join(MIX, "build/rst_columns.parquet")
TW_SIDE = os.path.join(MIX, "build/columns/agentic_v211_columns.parquet")
TL_SIDE = os.path.join(MIX, "build/columns/tl_8621e3fa/tl_columns.parquet")
TOP3 = os.path.join(V13, "tb21_top3", "top3_final.tsv")
TTC = "Fzz1/Tmax-Tasks-Clean"
TTC_REV = "bb6cf8fd646d753ab2c0c14330bfb32759143645"
HEADS = {"agentpick": "Fzz1/tb-training-mix-agentpick@071b1706", "mix": "Fzz1/tb-training-mix@2523ef7c",
         "ttc": "Fzz1/Tmax-Tasks-Clean@bb6cf8fd"}
PKG_DIRS = {"rst": BR.P_POOLS,
            "terminalworld": ["/data/users/zekaili/fangzhou/tb_check/reaudit_398/scratch/gate197/TerminalWorld-Seeds-Clean/tasks"],
            "terminallego": ["/data/users/zekaili/fangzhou/tb_check/reaudit_398/scratch/gate_results_v2_tl/Terminal-Lego-15k/tasks"],
            "calibforge": [os.path.join(V13, "tb21_agentic_top10/harbor/cf/tasks"),
                           os.path.join(V13, "tb21_agentic_top10/harbor/cfn/tasks")]}
ROOT_FILES = ("instruction.md", "task.toml")
SUBTREES = ("environment", "tests", "solution")
CORPORA = ("tmax", "calibforge")          # user 2026-10-09 ("这几个没必要啊"): TMAX and CalibForge only
OTHER_CORPORA = ()                        # ("rst", "terminalworld", "terminallego", "xiaomi") when wanted again
KILLED_SPLIT = ("tmax", "calibforge")                                  # user 2026-10-09: kills get splits of their own
SPLITS = tuple(f"{c}_{k}" for c in CORPORA for k in (("clean", "repair", "killed") if c in KILLED_SPLIT
                                                      else ("clean", "repair")))
_OFR = "reviews/overfilter_20260916"
KILL_RECORD = {   # killing stage -> the local record that says so (repo-relative)
    "gate1": "unified/all_verdicts_by_task.jsonl", "v1r1": "unified/all_verdicts_by_task.jsonl",
    "v1r2": f"{_OFR}/original852_scope_manifest.jsonl", "gate2": f"{_OFR}/historical_stage_attrition_20260916.json",
    "withdrawn27": f"{_OFR}/historical_stage_attrition_20260916.json", "overfilter": f"{_OFR}/verdicts.jsonl",
    "v8": f"{_OFR}/v8/runs", "v12": f"{_OFR}/v12/production/disposition.csv",
    "drop452": "reviews/rebuild_452_20260907/drop_ids_452.txt",
    "partition": f"{_OFR}/v13/tmax_k3/tmax_scope_partition.tsv",
    "audit452": "reviews/full_semantic_audit_runs/20260907T111702Z_452_v22/task_table.tsv",
    "agentpick": f"{_OFR}/v13/tb_training_mix/build_agentpick_release6_tl/metadata/split_assignment.tsv",
    "mix": f"{_OFR}/v13/tb_training_mix/build/metadata/split_assignment.tsv",
    "agentpick_cf": f"{_OFR}/v13/tb_training_mix/build_calibforge_split_r3c_fbb40fa8_fbb40fa8_off_repair/metadata/split_assignment_calibforge.tsv",
    "agentpick_tl": f"{_OFR}/v13/tb_training_mix/build_tl_split/metadata/split_assignment.tsv"}
PUBLISHED_AS = {"agentpick_cf": "Fzz1/tb-training-mix-agentpick@071b1706:metadata/split_assignment_calibforge.tsv",
                "agentpick": "Fzz1/tb-training-mix-agentpick@071b1706:metadata/split_assignment.tsv",
                "mix": "Fzz1/tb-training-mix@2523ef7c:metadata/split_assignment.tsv"}


def public_path(rel: str) -> str:
    """A record's path with our working tree's internal directory prefixes removed (the leak scan treats those
    names as private identifiers); with the commit it still finds the file."""
    for pre in (f"{_OFR}/v13/tb_training_mix/", f"{_OFR}/v13/", f"{_OFR}/", "reviews/"):
        if rel.startswith(pre):
            return rel[len(pre):]
    return rel


def record_ref(key: str) -> tuple[str, str]:
    """(published path, commit) for a kill record: the file's last commit, or why there is none."""
    rel = KILL_RECORD[key]
    if subprocess.run(["git", "-C", REPO, "check-ignore", "-q", rel]).returncode == 0:
        return public_path(rel), "gitignored (holds security ids)"
    c = subprocess.run(["git", "-C", REPO, "log", "-1", "--format=%h", "--", rel], capture_output=True, text=True).stdout.strip()
    if c:
        return public_path(rel), c
    return public_path(rel), "untracked" + (f"; published as {PUBLISHED_AS[key]}" if key in PUBLISHED_AS else "")
PRE_STAGES = ("v1r1_review", "v1r1_rejudge", "v1r2_opus", "v1r2_fable", "v2", "v3", "v4", "v4_b2", "v4_disp", "v5",
              "v6_b1", "v6_cal", "v7", "audit452", "overfilter", "v8", "v12")       # chronological; gate1/gate2 are runtime gates
SOL_COLS = ("solution", "solution_format", "solution_model", "solution_reasoning_effort", "solution_episode_id",
            "solution_phase", "solution_replay_reward", "solution_actions_sha256", "solution_terminal_action_count",
            "solution_replay_resources_json", "solution_is_normalized_shell", "has_reference_solution")
RUN_COLS = ("req_cpus", "req_memory_mb", "est_disk_mb", "runtime_peak_ram_mb", "runtime_peak_disk_mb",
            "runtime_cpu_seconds", "runtime_sandbox_size", "runtime_peak_source")
die = BT.die


def tsv(p):
    return list(csv.DictReader(open(p, newline="", encoding="utf-8"), delimiter="\t"))


def jl(p):
    for line in open(p):
        line = line.strip()
        if line:
            yield json.loads(line)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_tar(path) -> dict[str, dict[str, bytes]]:
    out = collections.defaultdict(dict)
    with tarfile.open(path) as tf:
        for m in tf:
            if m.isfile():
                tid, _, rel = m.name[len("tasks/"):].partition("/")
                out[tid][rel] = tf.extractfile(m).read()
    return out


def rows_of(path, where=None):
    return [r for r in pq.read_table(path).to_pylist() if where is None or where(r)]


def harbor_package(tid: str, corpus: str, bare_from: str = "") -> dict[str, bytes]:
    """build_dataset.package_members for a harbor-shape corpus: the root files and every regular file under
    environment/, tests/, solution/ of the task's one pool dir; environment/Dockerfile replaced by a bare FROM."""
    roots = [os.path.join(d, tid) for d in PKG_DIRS[corpus] if os.path.isdir(os.path.join(d, tid))]
    if len(roots) != 1:
        die(f"a {corpus} package is not in exactly one pool dir")
    m = {}
    for f in ROOT_FILES:
        p = os.path.join(roots[0], f)
        if os.path.isfile(p):
            m[f] = open(p, "rb").read()
    for sub in SUBTREES:
        for dp, _dn, fs in os.walk(os.path.join(roots[0], sub)):
            for f in fs:
                ap = os.path.join(dp, f)
                if os.path.isfile(ap) and not os.path.islink(ap):
                    m[os.path.relpath(ap, roots[0])] = open(ap, "rb").read()
    if bare_from:
        m["environment/Dockerfile"] = bare_from.encode("utf-8")
    return m


def pkg_stats(m: dict[str, bytes]) -> dict:
    return {"task_content_sha256": BT.MT.package_sha256(m), "file_count": len(m), "archive_entry_count": len(m),
            "uncompressed_bytes": sum(len(b) for b in m.values())}


# ------------------------------------------------------------------------------------------------ TMAX history ---
AUDIT452 = os.path.join(REPO, "reviews/full_semantic_audit_runs/20260907T111702Z_452_v22/task_table.tsv")


def tmax_history() -> dict[str, list]:
    """{tid: [(stage, label), ...]} in funnel/chronological order, every prompt-review stage we ran before v13."""
    h = collections.defaultdict(list)
    allv = {}
    for r in jl(os.path.join(REPO, "unified/all_verdicts_by_task.jsonl")):
        allv[r["task_id"]] = r
        for k in PRE_STAGES[:12]:
            if r.get(k):
                h[r["task_id"]].append((k, str(r[k])))
    for p in sorted(glob.glob(os.path.join(REPO, "judge/fullrun_v31*_20260902/judgements_V7_bucket*.jsonl"))):
        for r in jl(p):
            if r.get("verdict") or r.get("tier"):
                h[r["task_id"]].append(("v7", f"{r.get('verdict') or ''}"))
    # the 452-corpus static semantic audit (protocol 2.2, 2026-09-07; independent A/B reviews, C on disagreement):
    # its assessment is the label; a held task (provider-limited/blocked) was not reviewed and gets no entry here
    holds = {}
    for r in csv.DictReader(open(AUDIT452), delimiter="\t"):
        if r["disposition"].startswith("HOLD"):
            holds[r["task_id"]] = r["disposition"]
        else:
            h[r["task_id"]].append(("audit452", r["assessment"]))
    for r in jl(os.path.join(OF, "verdicts.jsonl")):
        if r.get("codex_verdict"):
            h[r["task_id"]].append(("overfilter", r["codex_verdict"]))
    v8 = collections.defaultdict(set)
    for p in glob.glob(os.path.join(OF, "v8/runs/*/rows/*.jsonl")):
        for r in jl(p):
            if r.get("tier") and not r.get("skipped_security"):
                v8[r["task_id"]].add(r["tier"])
    for t, s in v8.items():
        h[t].append(("v8", "+".join(sorted(s))))
    for r in csv.DictReader(open(os.path.join(OF, "v12/production/disposition.csv"))):
        if r.get("v12_tier"):
            h[r["task_id"]].append(("v12", r["v12_tier"]))
    return h, allv, holds


def kill_label(t: str, stage: str, allv: dict, hist: dict) -> str:
    """The label the killing stage recorded for a task."""
    u = allv.get(t, {})
    pre = dict(hist.get(t, []))
    return {"gate1": u.get("gate1") or "no gate-1 record",
            "v1r1": u.get("v1r1_rejudge") or u.get("v1r1_review") or "",
            "v1r2": "audited, not usable" + "".join(f"; {k} {u[k]}" for k in ("v1r2_opus", "v1r2_fable") if u.get(k)),
            "gate2": f"solve gate {u.get('gate2') or '0/3'}", "withdrawn27": "withdrawn after the adversarial review",
            "overfilter": pre.get("overfilter", "REJECT"), "v8": pre.get("v8", "REJECT-PROVEN"),
            "v12": pre.get("v12", "REJECT"), "drop452": "dropped from the 452 corpus"}.get(stage, stage)


# ------------------------------------------------------------------------------------------------ build ----------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out_dir):
        die(f"{a.out_dir} exists; builds go to a fresh directory")
    t0 = time.time()
    from huggingface_hub import hf_hub_download
    template = pq.read_schema(BT.TEMPLATE)
    schema = pa.schema(list(template) + [pa.field(n, t) for n, t in BT.EXTRA] + [pa.field("origin", pa.string()),
                                                                                  pa.field("verdict_history", pa.string())])
    killed_index = []                    # every kill, across all lines (metadata/killed_tasks.tsv)
    names70 = template.names
    extra = [n for n, _ in BT.EXTRA]
    rows: dict[str, dict] = {}            # tid -> row
    pkgs: dict[str, dict] = {}            # tid -> members
    assign: dict[str, dict] = {}          # tid -> split-file row
    corpus_of: dict[str, str] = {}
    inputs = {}

    def put(tid, corpus, row, members, srow):
        if tid in rows or tid in assign:
            die("an id would appear twice")
        corpus_of[tid] = corpus
        assign[tid] = dict(srow, task_id=tid)
        if row is not None:
            rows[tid] = row
            pkgs[tid] = members

    def widen(r70: dict, origin: str, solution_source: str = "") -> dict:
        r = {c: r70.get(c) for c in names70}
        for c in extra:
            r[c] = None
        r["rollout_attempts"] = 0
        r["solution_source"] = solution_source if r.get("solution") else ""
        r["origin"] = origin
        r["verdict_history"] = "[]"
        return r

    def vh(entries):
        return json.dumps([e for e in entries if e], sort_keys=True)

    def note(path):
        inputs[os.path.relpath(path, REPO) if path.startswith(REPO) else path] = sha_file(path)
        return path

    # ================================================================ TMAX
    hist, allv, holds452 = tmax_history()
    ap_sa = {r["task_id"]: r for r in tsv(note(os.path.join(AP, "metadata/split_assignment.tsv")))}
    tm_sa = {r["task_id"]: r for r in tsv(note(os.path.join(TM, "metadata/split_assignment.tsv")))}
    tl_sa = {r["task_id"]: r for r in tsv(note(os.path.join(AP_TL, "metadata/split_assignment.tsv")))}
    part = {r["task_id"]: r for r in tsv(BT.PARTITION)}
    refs = {k: record_ref(k) for k in KILL_RECORD}
    t_assign = {r["task_id"]: r for r in tsv(note(os.path.join(TMAX_BUILD, "metadata/split_assignment_tmax.tsv")))}
    for sp in ("tmax_clean", "tmax_repair"):
        tar = read_tar(note(os.path.join(TMAX_BUILD, f"data/tasks-{sp}.tar")))
        for r in rows_of(note(os.path.join(TMAX_BUILD, f"data/{sp}.parquet"))):
            r["origin"] = "tb-full-data TMAX line (judge: v13 K3; rollouts and replays: tmax14k-rollout fe662354)"
            r["verdict_history"] = vh([{"stage": k, "label": l} for k, l in hist.get(r["task_id"], [])]
                                      + [{"stage": r["tier_source"], "label": f"{r['tier']}/{r['verdict']}"}])
            put(r["task_id"], "tmax", r, tar[r["task_id"]], dict(t_assign[r["task_id"]], origin=r["origin"]))
    verdicts = BT.load_verdicts()
    images = json.load(open(BT.MT.IMAGE_MAP))
    gate1 = {r["task_id"]: r for r in json.load(open(BT.MT.GATE1))}
    gen = pq.read_table(BT.MT.GENERIC_PARQUET, columns=["task_id", "domain"])
    domain = dict(zip(gen.column("task_id").to_pylist(), gen.column("domain").to_pylist()))
    excl = {t: r for t, r in t_assign.items() if r["split"] == "none"}
    want = set(excl)
    prepared = {}
    with open(BT.PREPARED) as fh:
        for line in fh:
            d = json.loads(line)
            if d["label"] in want:
                prepared[d["label"]] = line.rstrip("\n")
    # carried solutions: agentpick's replay-verified TMAX rows, then Tmax-Tasks-Clean reaudit on identical verifiers
    ap_tmax = {r["task_id"]: r for sp in ("clean", "repair") for r in rows_of(note(os.path.join(AP, f"data/{sp}.parquet")))
               if r["corpus"] == "tmax"}
    ttc_tar = read_tar(hf_hub_download(TTC, "data/tasks-reaudit-00000.tar", repo_type="dataset", revision=TTC_REV))
    ttc_rows = {r["task_id"]: r for r in pq.read_table(hf_hub_download(
        TTC, "splits/reaudit.parquet", repo_type="dataset", revision=TTC_REV)).to_pylist()}
    ttc_ok = set()
    for t, m in ttc_tar.items():
        up_ok = all(m.get(rel) == open(os.path.join(BT.MT.CFG["snapshot"], t, rel), "rb").read()
                    for rel in BT.MT_MEMBERS if os.path.isfile(os.path.join(BT.MT.CFG["snapshot"], t, rel)))
        if up_ok and set(m) <= set(BT.MT_MEMBERS) | {"environment/Dockerfile", "solution/solve.sh"} \
                and not (ttc_rows[t].get("pre_test_sh") or ""):
            ttc_ok.add(t)
    rel_names = {"Tmax-Tasks-Clean": HEADS["ttc"], "tb-training-mix-agentpick": HEADS["agentpick"],
                 "tb-training-mix": HEADS["mix"]}
    tcount = collections.Counter()
    for t in sorted(excl):
        x = excl[t]
        v = verdicts.get(t)
        if bool(v) != bool(x["tier"]) or (v and (v["tier"], v["verdict"]) != (x["tier"], x["verdict"])):
            die("a TMAX verdict differs from the published split file")
        pre = hist.get(t, [])
        was = x["reason"]
        hist_cols = {f"hist_{s}": "+".join(l for k, l in pre if k == s) for s in PRE_STAGES}
        where = "+".join(rel_names[n] for n in ("Tmax-Tasks-Clean", "tb-training-mix-agentpick", "tb-training-mix")
                         if re.search(r"(^|[ +:])%s([ +]|$)" % re.escape(n), was))
        base = {"partition_class": x["partition_class"], "security": x["security"], "was": was, **hist_cols}
        ks = []                                   # (stage, label, reason, record key), oldest first
        if was.startswith("excluded_killed_earlier"):
            st = was.split(": ", 1)[1]
            ks.append((st, kill_label(t, st, allv, hist), f"removed by the {st} stage",
                       st if st in KILL_RECORD else "partition"))
        if t in tm_sa and tm_sa[t]["class"] == "leak":
            ks.append(("v13 tb-training-mix", f"{tm_sa[t]['tier']} ({tm_sa[t]['blocking_fields']})",
                       "excluded as a leak: a leak field fired", "mix"))
        if t in ap_sa and ap_sa[t]["class"] == "kill":
            ks.append(("v13 tb-training-mix-agentpick", f"{ap_sa[t]['tier']}/{ap_sa[t]['verdict']}", ap_sa[t]["reason"],
                       "agentpick"))
        killed = bool(ks)
        kstage_part = part.get(t, {}).get("killed_stage", "")
        if not killed and kstage_part:
            ref = refs[kstage_part if kstage_part in KILL_RECORD else "partition"]
            killed_index.append({"task_id": t, "corpus": "tmax", "upstream": BT.MT.SOURCE_DATASET,
                                 "upstream_revision": BT.MT.TMAX_UPSTREAM_REVISION, "status": "superseded",
                                 "kill_stage": kstage_part, "label": kill_label(t, kstage_part, allv, hist),
                                 "reason": "an earlier stage removed it; a later review or release took it back",
                                 "record_file": ref[0], "record_commit": ref[1], "security": x["security"],
                                 "split_now": "(see split file)", "all_kill_stages": kstage_part})
        if not v and not pre and not killed:
            why = (f"the 452-corpus audit held it ({holds452[t]}) without a review; " if t in holds452 else "")
            put(t, "tmax", None, None, dict(base, split="none", reason=f"excluded_no_prompt_review: {why}{was}",
                                            tier="", verdict="", verdict_source="", origin=""))
            tcount["excluded_no_prompt_review"] += 1
            continue
        m, df = BT.package(t, images)
        img = (images.get(t) or {}).get("image", "")
        g1 = gate1.get(t)
        sec = x["security"] == "1"
        gaps = []
        gap = lambda c, why: gaps.append({"column": c, "reason": why})   # noqa: E731
        r = {c: None for c in schema.names}
        r.update(task_id=t, task_group_id=t, **pkg_stats(m), member_prefix=f"tasks/{t}",
                 instruction=m["instruction.md"].decode("utf-8", errors="replace"), dockerfile=df,
                 terminal_domain="", tw_source_type=BT.MT.TW_SOURCE_TYPE, base_image="", verdict_flipped=False,
                 reference_partial=False, dockerfile_repaired=False, pre_test_sh="", pre_test_env_identity="",
                 protected_paths="[]", protected_cmds="[]", corpus_revision=BT.MT.TMAX_UPSTREAM_REVISION,
                 prepared_row_json=prepared.get(t, ""), repair_selection="NOT_REPAIRED", whole_task_certified=False,
                 corpus="tmax", source_dataset=BT.MT.SOURCE_DATASET, security=sec, image=img,
                 tmax_domain=domain.get(t, "") or "", tb21_best_rank=None, tb21_neighbour_of=[], rst_oracle="",
                 rst_training_class="", rst_pool="", rst_category="", runtime_empty_run="",
                 runtime_empty_run_sandbox="", runtime_sandbox_size="", runtime_peak_source="")
        if not img:
            gap("dockerfile", "no_image_in_image_map_15k")
        if not r["prepared_row_json"]:
            gap("prepared_row_json", "converter_emitted_no_row")
        gap("terminal_domain", "tmax15k_domain_is_a_different_taxonomy_see_tmax_domain")
        if g1 and g1.get("status") == "ok" and str(g1.get("empty_reward")) == "0":
            r.update(validation_status="EMPTY_RUN_0_NO_REFERENCE_RUN", reward_verdict="EMPTY_RUN_0",
                     runtime_empty_run="ordinary empty run (reward 0)", runtime_empty_run_sandbox=g1.get("res") or "")
        else:
            r.update(validation_status="", reward_verdict="")
            gap("validation_status", "no_measured_empty_run")
        if not v and not pre:                        # killed by a runtime gate before any review
            st, lab = ks[0][0], ks[0][1]
            r.update(rubric=f"runtime_gate:{st}", tier="", verdict=lab, tier_source=f"killed:{st}",
                     verdict_source=f"killed:{st}", blocking_fields=[], note_fields=[])
            gap("tier", "killed_by_a_runtime_gate_before_any_review")
            hint, pins = "", []
        elif v:
            jf = BT.judge_fields(v["row_path"])
            if not jf:
                gap("blocking_fields", "earlier_v13_verdict_without_a_judgment_row_here")
            r.update(rubric="v13", tier=v["tier"], verdict=v["verdict"], tier_source=BT.public_source(v["source"]),
                     verdict_source=BT.public_source(v["source"]),
                     blocking_fields=[k for k in BT.LEAK + BT.BSIDE if jf.get(k)],
                     note_fields=[k for k in BT.NOTES if jf.get(k)])
            hint = jf.get("overspecific_check") or ""
            pins = jf.get("protected_paths") if not sec else []
        else:
            stage, label = pre[-1]
            r.update(rubric=stage, tier="", verdict=label, tier_source=f"pre_v13:{stage}",
                     verdict_source=f"pre_v13:{stage}", blocking_fields=[], note_fields=[])
            gap("tier", "pre_v13_label_not_mapped_to_a_v13_tier")
            gap("blocking_fields", "pre_v13_rubric_has_no_v13_fields")
            hint, pins = "", []
        r["v13_pin_candidates"] = json.dumps([str(p) for p in pins] if isinstance(pins, list) else [])
        src = ""
        apr = ap_tmax.get(t)
        if apr and apr["solution"] and str(apr["solution_phase"]).startswith("replay"):
            if apr["task_content_sha256"] != r["task_content_sha256"]:
                die("an agentpick TMAX package differs from ours; its replay does not carry over")
            for c in SOL_COLS + RUN_COLS:
                r[c] = apr[c]
            src = "agentpick_replay_verified" + (":drop_between_last_markers" if "marker" in str(apr["solution_phase"]) else "")
        elif t in ttc_ok:
            body = ttc_rows[t]["solution"]
            r.update(solution=body, solution_format="tmax_reaudit_solve_sh", solution_model="",
                     solution_reasoning_effort="", solution_episode_id="",
                     solution_phase=f"{HEADS['ttc']} reaudit (reward_verdict pass)", solution_replay_reward=1.0,
                     solution_actions_sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
                     solution_terminal_action_count=None, solution_replay_resources_json="",
                     solution_is_normalized_shell=False, has_reference_solution=True,
                     req_cpus=ttc_rows[t]["req_cpus"], req_memory_mb=ttc_rows[t]["req_memory_mb"],
                     est_disk_mb=ttc_rows[t]["est_disk_mb"])
            src = "ttc_reaudit_gold:identical_verifier"
        else:
            r.update(solution="", solution_format="", solution_model="", solution_reasoning_effort="",
                     solution_episode_id="", solution_phase="", solution_actions_sha256="",
                     solution_replay_resources_json="", solution_is_normalized_shell=False,
                     has_reference_solution=False)
            gap("solution", "no_solution_verified_against_these_files")
        if r["req_cpus"] is None:
            gap("req_cpus", "never_rolled_out_by_tb-full-data")
        r["runtime_cpu_seconds"] = r["runtime_cpu_seconds"]
        r["rollout_attempts"] = 0
        r["solution_source"] = src
        verified = bool(r["solution"]) and r["solution_replay_reward"] == 1.0
        clean = bool(v) and v["tier"] == "SEED" and v["verdict"] == "CLEAN" and verified and not killed
        sp = "tmax_killed" if killed else ("tmax_clean" if clean else "tmax_repair")
        r["repair_hint"] = hint if (sp == "tmax_repair" and not sec) else ""
        if killed:
            reason = "killed: " + "; then ".join(f"{k[0]} ({k[1]}): {k[2]}" for k in ks)
        elif clean:
            reason = f"tier SEED, verdict CLEAN, and a verified solution ({src})"
        elif v and v["tier"] == "SEED" and v["verdict"] == "CLEAN":
            reason = "repair_no_verified_solution: v13 SEED/CLEAN; not rolled out by tb-full-data, no solution verified against these files"
        elif v:
            reason = f"repair_verdict: {v['tier']}/{v['verdict'] or '(no verdict)'}" + ("; with a verified solution" if verified else "")
        else:
            reason = f"pre_v13_verdict_only: {pre[-1][0]} {pre[-1][1]}"
        reason += f" (earlier: {was})"
        origin = ("verdict " + (r["tier_source"]) + (f"; solution {src}" if src else "")
                  + (f"; in {where}" if where else "")
                  + ("; KILLED by " + "; then ".join(f"{k[0]} ({k[1]})" for k in ks) if killed else "")
                  + (f"; an earlier {kstage_part} removal was superseded" if (kstage_part and not killed) else ""))
        r["origin"] = origin
        r["verdict_history"] = vh([{"stage": "gate1", "label": allv.get(t, {}).get("gate1", "")}]
                                  + [{"stage": k, "label": l} for k, l in pre]
                                  + ([{"stage": "gate2", "label": allv[t]["gate2"]}] if allv.get(t, {}).get("gate2") else [])
                                  + ([{"stage": "partition_kill", "label": kstage_part}] if kstage_part else [])
                                  + ([{"stage": "v13 tb-training-mix", "label": tm_sa[t]["tier"], "class": tm_sa[t]["class"]}] if t in tm_sa else [])
                                  + ([{"stage": "v13 tb-training-mix-agentpick", "label": f"{ap_sa[t]['tier']}/{ap_sa[t]['verdict']}",
                                       "class": ap_sa[t]["class"]}] if t in ap_sa else [])
                                  + ([{"stage": r["tier_source"], "label": f"{v['tier']}/{v['verdict']}"}] if v else []))
        for k in ks:
            ref = refs[k[3]]
            killed_index.append({"task_id": t, "corpus": "tmax", "upstream": BT.MT.SOURCE_DATASET,
                                 "upstream_revision": BT.MT.TMAX_UPSTREAM_REVISION, "status": "killed",
                                 "kill_stage": k[0], "label": k[1], "reason": k[2], "record_file": ref[0],
                                 "record_commit": ref[1], "security": x["security"], "split_now": sp,
                                 "all_kill_stages": "+".join(z[0] for z in ks)})
        r["split"], r["shard"] = sp, f"data/tasks-{sp}.tar"
        r["column_gaps"] = json.dumps(gaps, sort_keys=True)
        put(t, "tmax", r, m, dict(base, split=sp, reason=reason, tier=r["tier"], verdict=r["verdict"],
                                  verdict_source=r["verdict_source"], origin=origin,
                                  replay_verified=int(verified), rollout_attempts=0,
                                  kill_stages="+".join(k[0] for k in ks), partition_kill_stage=kstage_part))
        tcount[sp] += 1
        tcount[f"{sp}:{src or 'no_solution'}"] += 1

    # ================================================================ CalibForge
    c_assign = {r["task_id"]: r for r in tsv(note(os.path.join(CF_BUILD, "metadata/split_assignment_calibforge.tsv")))}
    for sp in ("calibforge_clean", "calibforge_repair"):
        tar = read_tar(note(os.path.join(CF_BUILD, f"data/tasks-{sp}.tar")))
        for r in rows_of(note(os.path.join(CF_BUILD, f"data/{sp}.parquet"))):
            r["origin"] = "tb-full-data CalibForge line (judge: K3 cfr_h3; rollouts and replays: tmax14k-rollout d13e845e)"
            r["verdict_history"] = vh([{"stage": r["tier_source"], "label": f"{r['tier']}/{r['verdict']}"}])
            put(r["task_id"], "calibforge", r, tar[r["task_id"]], dict(c_assign[r["task_id"]], origin=r["origin"]))
    apc = {r["task_id"]: r for r in tsv(note(os.path.join(AP_CF, "metadata/split_assignment_calibforge.tsv")))}
    side = {r["task_id"]: r for r in rows_of(note(CF_SIDE))}
    pub = {}
    for sp in ("calibforge_clean", "calibforge_repair"):
        tar = read_tar(note(os.path.join(AP_CF, f"data/tasks-{sp}.tar")))
        for r in rows_of(note(os.path.join(AP_CF, f"data/{sp}.parquet"))):
            pub[r["task_id"]] = (r, tar[r["task_id"]])
    if set(apc) != set(c_assign) - {t for t, x in c_assign.items() if x["split"] != "none"} \
            or len(apc) != 766:
        die("agentpick's 766 CalibForge ids are not exactly tb-full-data's excluded CalibForge ids")
    ccount = collections.Counter()
    for t in sorted(apc):
        s = apc[t]
        if t in pub:
            r0, m = pub[t]
            r = widen(r0, "", "agentpick_cf_replay_majority" if r0["solution"] else "")
        else:
            sd = side[t]
            bare = sd["dockerfile"] if sd["dockerfile_repaired"] else ""
            m = harbor_package(t, "calibforge", bare)
            r0 = {c: None for c in names70}
            tpl = next(iter(pub.values()))[0]
            for c in ("corpus", "source_dataset", "corpus_revision", "rubric", "tw_source_type", "terminal_domain",
                      "base_image", "validation_status", "reward_verdict", "pre_test_sh", "pre_test_env_identity",
                      "protected_paths", "protected_cmds", "prepared_row_json", "repair_selection", "rst_oracle",
                      "rst_training_class", "rst_pool", "rst_category", "image", "tmax_domain", "runtime_empty_run",
                      "runtime_empty_run_sandbox", "verdict_flipped", "reference_partial", "whole_task_certified",
                      "solution_is_normalized_shell"):
                r0[c] = tpl[c]
            for c in names70:
                if c in sd and c not in ("security",):
                    r0[c] = sd[c]
            r0.update(task_id=t, task_group_id=t, member_prefix=f"tasks/{t}", **pkg_stats(m),
                      security=bool(sd["security"]), tier=s["tier"], verdict=s["verdict"], tier_source=s["tier_source"],
                      verdict_source=s["tier_source"], blocking_fields=[], note_fields=[], repair_hint="",
                      v13_pin_candidates="[]", has_reference_solution=bool(sd["solution"]))
            r0["column_gaps"] = json.dumps([{"column": "blocking_fields", "reason": "not_filled_for_calibforge_as_in_agentpick"}])
            r = widen(r0, "", "agentpick_cf_replay_majority" if r0["solution"] else "")
        verified = bool(r["solution"]) and r["solution_replay_reward"] == 1.0
        killed = s["class"] == "kill"
        sibling = bool(s["cf_tb21_sibling"])
        clean = r["tier"] == "SEED" and r["verdict"] == "CLEAN" and verified and not killed and not sibling
        sp = "calibforge_killed" if killed else ("calibforge_clean" if clean else "calibforge_repair")
        r["split"], r["shard"] = sp, f"data/tasks-{sp}.tar"
        if r["security"] or sp == "calibforge_clean":
            r["repair_hint"] = ""
        origin = (f"{HEADS['agentpick']} CalibForge ({'published ' + s['split'] if s['split'] != 'none' else 'reviewed, not shipped: ' + s['class']}); "
                  f"judge {s['tier_source']}" + (f"; KILLED by v13 tb-training-mix-agentpick ({s['tier']}/{s['verdict']})" if killed else ""))
        r["origin"] = origin
        r["verdict_history"] = vh([{"stage": "agentpick cf judge V4.1-Flash max", "label": s["cf_verdict_v41"]},
                                   {"stage": "agentpick cf judge GLM-5.3 low", "label": s["cf_verdict_glm"]},
                                   {"stage": s["tier_source"], "label": f"{s['tier']}/{s['verdict']}", "class": s["class"],
                                    "judge_of_record": s["cf_judge"]}])
        if killed:
            reason = f"killed: v13 tb-training-mix-agentpick ({s['tier']}/{s['verdict']}): {s['reason']}"
            ref = refs["agentpick_cf"]
            killed_index.append({"task_id": t, "corpus": "calibforge", "upstream": "AweAI-Team/CalibForge",
                                 "upstream_revision": "fb1e75441a94b8bb0ced08acd6b59e711704d70a", "status": "killed",
                                 "kill_stage": "v13 tb-training-mix-agentpick (CalibForge)",
                                 "label": f"{s['tier']}/{s['verdict']}", "reason": s["reason"], "record_file": ref[0],
                                 "record_commit": ref[1], "security": s["security"], "split_now": sp,
                                 "all_kill_stages": "v13 tb-training-mix-agentpick"})
        elif sibling and r["tier"] == "SEED" and r["verdict"] == "CLEAN" and verified:
            reason = f"TB2.1 sibling (copied test scaffold): {s['cf_tb21_sibling']}; agentpick: {s['reason']}"
        elif clean:
            reason = "tier SEED, verdict CLEAN, and a replay-verified solution (agentpick: strict majority of graded replays)"
        else:
            reason = f"repair_from_agentpick: {s['reason']}"
        put(t, "calibforge", r, m, {"split": sp, "reason": reason, "security": s["security"], "tier": s["tier"],
                                    "verdict": s["verdict"], "verdict_source": s["tier_source"], "origin": origin,
                                    "replay_verified": int(verified), "rollout_attempts": 0,
                                    "agentpick_split": s["split"], "agentpick_class": s["class"],
                                    "cf_judge": s["cf_judge"], "cf_verdict_v41": s["cf_verdict_v41"],
                                    "cf_verdict_glm": s["cf_verdict_glm"], "cf_tb21_sibling": s["cf_tb21_sibling"]})
        ccount[sp] += 1

    # ================================================================ RST, TerminalWorld, Terminal-Lego, Xiaomi
    if OTHER_CORPORA:
        published = {}                                   # tid -> (line, row, members)
        for line, d, sps in (("agentpick", AP, ("clean", "repair")), ("mix", TM, ("pass", "repair")),
                             ("agentpick", AP_TL, ("terminallego_clean", "terminallego_repair")),
                             ("agentpick", AP_XM, ("xiaomi_clean", "xiaomi_repair"))):
            for sp in sps:
                tar = read_tar(note(os.path.join(d, f"data/tasks-{sp}.tar")))
                for r in rows_of(note(os.path.join(d, f"data/{sp}.parquet"))):
                    if r["corpus"] in ("rst", "terminalworld", "terminallego", "MiMo-V2.6-RL-oss"):
                        if r["task_id"] in published and published[r["task_id"]][0] == "agentpick":
                            continue
                        published[r["task_id"]] = (line, r, tar[r["task_id"]], sp)
        rst_ids = {t for t, s in ap_sa.items() if s["corpus"] == "RST"} | {t for t, s in tm_sa.items() if s["corpus"] == "RST"}
        tw_ids = {t for t, s in ap_sa.items() if s["corpus"] == "TW"}
        tl_ids = set(tl_sa)
        xm_ids = {t for t, (line, r, _m, sp) in published.items() if r["corpus"] == "MiMo-V2.6-RL-oss" and r["tier"]}
        side_rst = {r["task_id"]: r for r in rows_of(note(RST_SIDE))}
        side_tw = {r["task_id"]: r for r in rows_of(note(TW_SIDE))}
        side_tl = {r["task_id"]: r for r in rows_of(note(TL_SIDE))}
        top3 = {r["task_id"]: r for r in tsv(TOP3)}
        judged = BR.judged_rows(rst_ids)
        gate = {}
        for tag, path in BR.P_RESULTS:
            for line in open(path):
                i = line.find('"task_id": "')
                t = line[i + 12:line.find('"', i + 12)]
                if t not in rst_ids:
                    continue
                d = json.loads(line)
                if d.get("terminal") and d.get("arm") in ("oracle", "empty"):
                    key = (t, d["arm"])
                    prev = gate.get(key)
                    if prev is None or (BR.ORDER[tag], d.get("attempt", 0)) >= (BR.ORDER[prev["_src"]], prev.get("attempt", 0)):
                        d["_src"] = tag
                        gate[key] = d
        bulls = set()
        for d in BR.P_BULLSEYE:
            if os.path.isdir(d):
                bulls |= set(os.listdir(d))
        ocount = collections.Counter()

        def corpus_consts(corpus_label):
            rs = [r for (_l, r, _m, _s) in published.values() if r["corpus"] == corpus_label]
            return {c: rs[0][c] for c in names70 if all(json.dumps(x[c], sort_keys=True, default=str)
                                                        == json.dumps(rs[0][c], sort_keys=True, default=str) for x in rs)}

        def other_row(t, key, label, decider, sa_row, verdict, tier, tier_source, blocking, sec):
            """A judged id our releases did not ship: package from its pool dir, agentpick's side columns where they
            exist, the corpus's constant columns, the decider's verdict; never clean (not shipped by its line)."""
            consts = corpus_consts(label)
            side_t = {"rst": side_rst, "terminalworld": side_tw, "terminallego": side_tl}[key].get(t, {})
            bare = ""
            m = harbor_package(t, key, bare)
            r0 = {c: None for c in names70}
            for c, val in consts.items():
                r0[c] = val
            for c in names70:
                if c in side_t and c not in ("split",):
                    r0[c] = side_t[c]
            gaps = []
            if not r0.get("instruction"):
                r0["instruction"] = m["instruction.md"].decode("utf-8", errors="replace")
            if not r0.get("dockerfile"):
                r0["dockerfile"] = m.get("environment/Dockerfile", b"").decode("utf-8", errors="replace")
            if key == "rst":
                r0["dockerfile_repaired"] = t in bulls
                if not side_t:
                    o, e = gate.get((t, "oracle")), gate.get((t, "empty"))
                    oracle = float(o["reward"]) if o and o.get("reward") is not None else None
                    r0["solution"] = m.get("solution/solve.sh", b"").decode("utf-8", errors="replace")
                    r0.update(solution_format="harbor_solve_sh" if r0["solution"] else "", solution_phase="rst_gate_oracle_arm" if o else "",
                              solution_replay_reward=oracle,
                              solution_actions_sha256=hashlib.sha256(r0["solution"].encode("utf-8")).hexdigest() if r0["solution"] else "",
                              validation_status=("RST_GATE_ORACLE_1_EMPTY_0" if oracle == 1.0 else "RST_GATE_ORACLE_0"
                                                 if oracle == 0.0 else "RST_NOT_GATED"),
                              reward_verdict=("ORACLE_1_EMPTY_0" if oracle == 1.0 else "ORACLE_0" if oracle == 0.0 else "NOT_GATED"),
                              rst_oracle=oracle, runtime_empty_run=(f"empty run reward {e.get('reward')}" if e else ""))
                    gaps.append({"column": "prepared_row_json", "reason": "not_built_for_a_row_its_line_did_not_ship"})
            r0.update(task_id=t, member_prefix=f"tasks/{t}", **pkg_stats(m), security=sec, tier=tier, verdict=verdict,
                      tier_source=tier_source, verdict_source=tier_source, blocking_fields=blocking, note_fields=r0.get("note_fields") or [],
                      repair_hint="", v13_pin_candidates="[]", has_reference_solution=bool(r0.get("solution")))
            if not r0.get("task_group_id"):
                r0["task_group_id"] = t
            for c in names70:
                if r0[c] is not None and pa.types.is_string(template.field(c).type) and not isinstance(r0[c], str):
                    r0[c] = str(r0[c])                     # e.g. rst_oracle: a double in the side table, a string here
                if r0[c] is None and pa.types.is_string(template.field(c).type):
                    r0[c] = ""
                    gaps.append({"column": c, "reason": "no_source_for_a_row_its_line_did_not_ship"})
                if r0[c] is None and pa.types.is_list(template.field(c).type):
                    r0[c] = []
                if r0[c] is None and pa.types.is_boolean(template.field(c).type):
                    r0[c] = False
            r0["column_gaps"] = json.dumps(sorted(gaps, key=lambda g: g["column"]), sort_keys=True)
            return r0, m

        UPSTREAM = {"rst": ("Zhongzhi1228/Recursive-Task-Synthesis", "be44f968"),
                    "terminalworld": ("andylizf/TerminalWorld-Seeds-Clean", "ef0a100cd93cd2ecd316adb30eb3819ec3f37201"),
                    "terminallego": ("Lego-X/Terminal-Lego-15k", "9c197f1c2e87b64cc316b1a5bfcef57b584929f0"),
                    "xiaomi": ("XiaomiMiMo/MiMo-V2.6-RL-oss", "639865fd3374018d6cb29b9fb82dd531406fcf5f")}

        def history(t):
            out = []
            if t in tm_sa:
                out.append({"stage": f"v13-rst tb-training-mix ({tm_sa[t]['tier_source']})", "label": tm_sa[t]["tier"],
                            "class": tm_sa[t]["class"]})
            if t in ap_sa:
                out.append({"stage": f"tb-training-mix-agentpick ({ap_sa[t]['tier_source']})",
                            "label": f"{ap_sa[t]['tier']}/{ap_sa[t]['verdict']}", "class": ap_sa[t]["class"]})
            if t in tl_sa:
                out.append({"stage": f"tb-training-mix-agentpick ({tl_sa[t]['tier_source']})",
                            "label": f"{tl_sa[t]['tier']}/{tl_sa[t]['verdict']}", "class": tl_sa[t]["class"]})
            return out

        def place(t, key, label, decider_sa, decider_line, tier, verdict, tier_source, blocking, line_clean, sec, kills=()):
            if kills:                                     # user 2026-10-09: kills stay out; the index lists them
                for st, lab, why, refk in kills:
                    ref = refs[refk]
                    killed_index.append({"task_id": t, "corpus": key, "upstream": UPSTREAM[key][0],
                                         "upstream_revision": UPSTREAM[key][1], "status": "killed", "kill_stage": st,
                                         "label": lab, "reason": why, "record_file": ref[0], "record_commit": ref[1],
                                         "security": int(bool(sec)), "split_now": "none (not in a split)",
                                         "all_kill_stages": "+".join(k[0] for k in kills)})
                put(t, key, None, None, {"split": "none", "reason": "killed: " + "; then ".join(f"{k[0]} ({k[1]}): {k[2]}" for k in kills),
                                         "security": int(bool(sec)), "tier": tier, "verdict": verdict,
                                         "verdict_source": tier_source, "origin": "", "kill_stages": "+".join(k[0] for k in kills),
                                         "verdict_history": vh(history(t))})
                ocount[f"{key}:killed"] += 1
                return
            pubr = published.get(t)
            if pubr:
                line, r0, m, psp = pubr
                # rst / terminalworld / terminallego ship the package's own solve.sh, graded by that line's gate oracle run;
                # xiaomi's solution is our rollout's action sequence, verified by its deciding replay
                r = widen(r0, "", f"{line}_{'replay_verified' if key == 'xiaomi' else 'gate_oracle'}")
                if (r["tier"], r["verdict"]) != (tier, verdict):
                    r.update(tier=tier, verdict=verdict, tier_source=tier_source, verdict_source=tier_source,
                             blocking_fields=blocking)
                src = f"{HEADS[line]} {psp}"
            else:
                r0, m = other_row(t, key, label, decider_line, decider_sa, verdict, tier, tier_source, blocking, sec)
                r = widen(r0, "", f"{key}_package_solution_gate" if r0.get("solution") else "")
                src = f"{HEADS[decider_line]} split file only (reviewed, not shipped: {decider_sa.get('class', '')})"
            verified = bool(r["solution"]) and r["solution_replay_reward"] == 1.0
            clean = line_clean and tier == "SEED" and verdict == "CLEAN" and verified and bool(pubr)
            sp = f"{key}_{'clean' if clean else 'repair'}"
            r["split"], r["shard"] = sp, f"data/tasks-{sp}.tar"
            if r["security"] or clean:
                r["repair_hint"] = ""
            if r["security"]:
                r["v13_pin_candidates"] = "[]"
            origin = f"{src}; verdict {tier_source} ({decider_line})"
            r["origin"] = origin
            r["verdict_history"] = vh(history(t) or [{"stage": tier_source, "label": f"{tier}/{verdict}"}])
            reason = (f"tier SEED, verdict CLEAN, and a verified solution ({r['solution_source']})" if clean else
                      f"repair_from_{decider_line}: class {decider_sa.get('class', '')}" + (f"; {decider_sa['reason']}" if decider_sa.get("reason") else ""))
            put(t, key, r, m, {"split": sp, "reason": reason, "security": int(bool(sec)), "tier": tier, "verdict": verdict,
                               "verdict_source": tier_source, "origin": origin, "replay_verified": int(verified),
                               "rollout_attempts": 0, "agentpick_split": ap_sa.get(t, {}).get("split", tl_sa.get(t, {}).get("split", "")),
                               "agentpick_class": ap_sa.get(t, {}).get("class", tl_sa.get(t, {}).get("class", "")),
                               "mix_split": tm_sa.get(t, {}).get("split", ""), "mix_class": tm_sa.get(t, {}).get("class", ""),
                               "mix_tier": tm_sa.get(t, {}).get("tier", "")})
            ocount[sp] += 1

        for t in sorted(rst_ids):
            a_, m_ = ap_sa.get(t), tm_sa.get(t)
            if a_ and a_["tier"] not in ("", "NOT JUDGED"):
                dec, line, tier, verdict, ts = a_, "agentpick", a_["tier"], a_["verdict"], a_["tier_source"]
                blocking = [b for b in a_["blocking_fields"].split("+") if b]
                line_clean = a_["split"] == "clean"
            elif m_ and m_["tier"] not in ("", "NOT JUDGED"):
                dec, line, tier, ts = m_, "mix", m_["tier"], m_["tier_source"]
                if ts == "top3_final":
                    f3 = top3[t]
                    verdict = (f3["glm_low_verdict"] if f3["final_tier"] == f3["glm_low_tier"] else
                               f3["opus48_verdict"] if f3["final_tier"] == f3["opus48_tier"] else "")
                else:
                    verdict = (judged.get(t, ({}, ""))[0] or {}).get("verdict", "")
                blocking = [b for b in m_["blocking_fields"].split("+") if b]
                line_clean = m_["split"] == "pass"
            else:
                put(t, "rst", None, None, {"split": "none", "reason": "excluded_not_judged: no v13-rst judgment row",
                                           "security": (a_ or m_)["security"], "tier": "", "verdict": "", "verdict_source": "",
                                           "origin": "", "agentpick_class": (a_ or {}).get("class", ""),
                                           "mix_class": (m_ or {}).get("class", "")})
                ocount["rst:excluded_not_judged"] += 1
                continue
            kills = []
            if m_ and m_["class"] == "leak":
                kills.append(("v13-rst tb-training-mix", f"{m_['tier']} ({m_['blocking_fields']})",
                              "excluded as a leak: a leak field fired", "mix"))
            if a_ and a_["class"] == "kill":
                kills.append(("v13-rst tb-training-mix-agentpick", f"{a_['tier']}/{a_['verdict']}", a_["reason"], "agentpick"))
            place(t, "rst", "rst", dec, line, tier, verdict, ts, blocking, line_clean, (dec["security"] == "1"), kills)
        for t in sorted(tw_ids):
            s = ap_sa[t]
            place(t, "terminalworld", "terminalworld", s, "agentpick", s["tier"], s["verdict"], s["tier_source"],
                  [b for b in s["blocking_fields"].split("+") if b], s["split"] == "clean", s["security"] == "1",
                  [("v13-harbor2 tb-training-mix-agentpick", f"{s['tier']}/{s['verdict']}", s["reason"], "agentpick")]
                  if s["class"] == "kill" else [])
        for t in sorted(tl_ids):
            s = tl_sa[t]
            place(t, "terminallego", "terminallego", s, "agentpick", s["tier"], s["verdict"], s["tier_source"],
                  [b for b in s["blocking_fields"].split("+") if b], s["split"] == "terminallego_clean", s["security"] == "1",
                  [("v13-harbor3 tb-training-mix-agentpick (Terminal-Lego)", f"{s['tier']}/{s['verdict']}", s["reason"],
                    "agentpick_tl")] if s["class"] == "kill" else [])
        for t in sorted(xm_ids):
            line, r0, m, psp = published[t]
            place(t, "xiaomi", "MiMo-V2.6-RL-oss", {"class": psp, "split": psp}, "agentpick", r0["tier"], r0["verdict"],
                  r0["tier_source"], list(r0["blocking_fields"] or []), psp == "xiaomi_clean", bool(r0["security"]))

    # ================================================================ checks that refuse
    if len(set(rows)) != len(rows):
        die("duplicate ids")
    for t, r in rows.items():
        if set(r) != set(schema.names):
            die(f"row columns differ from the schema: {sorted(set(r) ^ set(schema.names))[:6]}")
        if r["split"].endswith("_killed") and not any(k["task_id"] == t and k["status"] == "killed" for k in killed_index):
            die("a killed-split row has no kill in the index")
        if r["split"].endswith("_clean") and not (r["tier"] == "SEED" and r["verdict"] == "CLEAN" and r["solution"]
                                                  and r["solution_replay_reward"] == 1.0):
            die("a clean row lacks SEED/CLEAN or a verified solution")
        if r["security"] and (r["repair_hint"] or r["v13_pin_candidates"] != "[]"):
            die("a security row carries a hint or pin candidates")
        if r["task_content_sha256"] != BT.MT.package_sha256(pkgs[t]):
            die("a row's task_content_sha256 differs from its package")
        if r["solution"] and r["solution_actions_sha256"] and \
                hashlib.sha256(r["solution"].encode("utf-8")).hexdigest() != r["solution_actions_sha256"]:
            die("a shipped solution's sha256 differs from solution_actions_sha256")

    # ================================================================ write
    os.makedirs(os.path.join(a.out_dir, "data"))
    os.makedirs(os.path.join(a.out_dir, "metadata"))
    for sp in SPLITS:
        sub = [rows[t] for t in sorted(rows) if rows[t]["split"] == sp]
        pq.write_table(pa.Table.from_pylist(sub, schema=schema), os.path.join(a.out_dir, "data", f"{sp}.parquet"))
        BT.write_tar(os.path.join(a.out_dir, "data", f"tasks-{sp}.tar"), {t: pkgs[t] for t in rows if rows[t]["split"] == sp})
    for c in CORPORA:
        ids = sorted(t for t in assign if corpus_of[t] == c)
        cols = []
        for t in ids:
            for k in assign[t]:
                if k not in cols:
                    cols.append(k)
        first = ["task_id", "split", "reason", "security", "tier", "verdict", "verdict_source", "origin"]
        cols = first + [k for k in cols if k not in first]
        for t in ids:                                  # the earlier verdicts, for rows too (excluded ones included)
            if "verdict_history" not in assign[t]:
                assign[t]["verdict_history"] = rows[t]["verdict_history"] if t in rows else ""
        if "verdict_history" not in cols:
            cols.append("verdict_history")
        with open(os.path.join(a.out_dir, "metadata", f"split_assignment_{c}.tsv"), "w") as fh:
            w = csv.DictWriter(fh, cols, delimiter="\t", lineterminator="\n", restval="")
            w.writeheader()
            w.writerows(assign[t] for t in ids)
    for k in killed_index:
        if k["split_now"] == "(see split file)":
            k["split_now"] = assign[k["task_id"]]["split"]
    kcols = ["task_id", "corpus", "upstream", "upstream_revision", "status", "kill_stage", "label", "reason",
             "record_file", "record_commit", "security", "split_now", "all_kill_stages"]
    with open(os.path.join(a.out_dir, "metadata", "killed_tasks.tsv"), "w") as fh:
        w = csv.DictWriter(fh, kcols, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(sorted(killed_index, key=lambda k: (k["corpus"], k["task_id"], k["kill_stage"])))
    counts = {"split": dict(collections.Counter(r["split"] for r in rows.values())),
              "killed_index": dict(collections.Counter(f"{k['corpus']}:{k['status']}" for k in killed_index)),
              "killed_index_tasks": len({(k["corpus"], k["task_id"]) for k in killed_index if k["status"] == "killed"}),
              "excluded": dict(collections.Counter(f"{corpus_of[t]}:{x['reason'].split(':')[0]}" for t, x in assign.items()
                                                   if x["split"] == "none")),
              "security_by_split": dict(collections.Counter(r["split"] for r in rows.values() if r["security"])),
              "with_solution_by_split": dict(collections.Counter(r["split"] for r in rows.values() if r["solution"])),
              "tmax_added": dict(tcount), "calibforge_added": dict(ccount),
              "ttc_identical_verifier_ids": len(ttc_ok)}
    rec = {"built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t0, 1),
           "kind": "superset", "limit": 0, "schema_columns": len(schema), "heads": HEADS, "ttc_rev": TTC_REV,
           "inputs_sha256": inputs, "counts": counts}
    json.dump(rec, open(os.path.join(a.out_dir, "metadata", "BUILD_RECORD.json"), "w"), indent=1, sort_keys=True)
    print(json.dumps(counts, indent=1, sort_keys=True))
    print("wrote", a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
