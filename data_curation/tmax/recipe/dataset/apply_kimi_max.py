#!/usr/bin/env python3
"""Apply tmax14k-rollout's Kimi-max line (lead 2026-10-11) to the published tb-full-data superset build.

Scope: the 977 tmax_repair rows with reason repair_unsolved (SEED/CLEAN, DeepSeek 0 in 3), re-rolled on Kimi K3 at
effort max, top_p 0.95, up to 3 attempts, then one fresh-sandbox deciding replay of the solving episode
(tmax14k_rollout/rows/kimi_max, commit given by --rev).
  final replay_verified  -> tmax_clean; the solution is rebuilt from the solving episode's retained trajectory exactly
                            as the replay rebuilt it (build_tmax_full.rebuild_solution) and its sha256 must equal the
                            replay's actions_sha256; solution_* and replay_* describe the Kimi attempt and its replay
  final unsolved_after_3 -> stays in tmax_repair, reason "unsolved after 3 DeepSeek + 3 Kimi"
The DeepSeek rollout columns (rollout_reward_pass1 / retry1 / retry2, pass-1 metrics) stay as they were;
rollout_attempts adds the counted Kimi attempts, and rollout_best_attempt names the solving Kimi attempt.
Every other row and file is copied unchanged. The base build must be the published one (sha256 = Hub LFS at --base-rev).

  $PY apply_kimi_max.py --base <superset build> --out-dir <fresh dir> --rev 7fd8cd9f --base-rev cd5e0f9a...
"""
from __future__ import annotations

import argparse, collections, csv, hashlib, json, os, shutil, subprocess, sys, tarfile, time

import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_tmax_full as BT                                             # noqa: E402

K = os.path.join(BT.T14, "rows", "kimi_max")
REPLAYS = os.path.join(BT.T14, "replays")
die = BT.die


def tsv(p):
    return list(csv.DictReader(open(p, newline="", encoding="utf-8"), delimiter="\t"))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_tar(path):
    out = collections.defaultdict(dict)
    with tarfile.open(path) as tf:
        for m in tf:
            if m.isfile():
                tid, _, rel = m.name[len("tasks/"):].partition("/")
                out[tid][rel] = tf.extractfile(m).read()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rev", required=True, help="tmax14k-rollout's commit of the kimi_max tables")
    ap.add_argument("--base-rev", default="", help="the Hub revision the base build was published as ('' = a rebuild)")
    a = ap.parse_args()
    if os.path.exists(a.out_dir):
        die(f"{a.out_dir} exists")
    rev = subprocess.run(["git", "-C", BT.REPO, "rev-parse", a.rev], capture_output=True, text=True, check=True).stdout.strip()
    B = os.path.abspath(a.base)

    # the base is the published build
    import os as _os
    from huggingface_hub import HfApi
    api = HfApi(token=_os.environ["HF_TOKEN"])
    if a.base_rev:                      # empty: the base is a fresh superset rebuild (checked by a keyed diff instead)
        info = api.dataset_info("Fzz1/tb-full-data", revision=a.base_rev, files_metadata=True)
        hub = {s.rfilename: (s.lfs.sha256 if s.lfs else None) for s in info.siblings}
        for f in hub:
            if f.startswith("data/") and sha(os.path.join(B, f)) != hub[f]:
                die(f"the base build's {f} is not the published bytes")

    # the kimi tables: committed (non-security) at --rev, security from disk
    for f in ("kx_final.tsv", "kx_replay_results.tsv"):
        if open(os.path.join(K, f)).read() != BT.git_show(rev, os.path.join(K, f)):
            die(f"{f} on disk differs from {a.rev}")
    final = {r["task_id"]: r for f in ("kx_final.tsv", "kx_final.security.tsv") for r in tsv(os.path.join(K, f))}
    tab = {r["task_id"]: r for f in ("kx_replay_results.tsv", "kx_replay_results.security.tsv")
           for r in tsv(os.path.join(K, f))}
    assign = tsv(os.path.join(B, "metadata", "split_assignment_tmax.tsv"))
    scope = {r["task_id"] for r in assign if r["split"] == "tmax_repair" and r["reason"].startswith("repair_unsolved")}
    if set(final) != scope or len(scope) != 977:
        die(f"kx_final's ids are not exactly the base build's repair_unsolved rows ({len(set(final) ^ scope)} differ)")
    solved = {t for t, r in final.items() if r["final"] == "replay_verified"}
    if any(r["final"] not in ("replay_verified", "unsolved_after_3") for r in final.values()):
        die("an unknown kimi final state")

    # deciding replays: read from the replay run dirs, cross-checked against the owner's table
    runs = sorted({os.path.join(REPLAYS, r["replay_run"]) for r in tab.values()})
    replays = BT.load_replays(runs)
    rrows = {t: x for t, x in BT.load_replay_rows(runs).items() if t in final}
    replays = {t: x for t, x in replays.items() if t in final}
    ver_tab = {t for t, r in tab.items() if r["status"] == "GRADED" and float(r["replay_reward"] or "nan") == 1.0}
    if not (ver_tab == set(replays) == solved):
        die("verified sets differ between kx_final, kx_replay_results and the replay run dirs")
    for t in solved:
        x, r = replays[t], tab[t]
        if (x["actions_sha256"], x["episode_run"], x["replay_run"]) != (r["actions_sha256"], r["episode_run"], r["replay_run"]) \
                or final[t]["episode_run"] != r["episode_run"]:
            die("a verified task's episode, replay run or actions sha differs between the tables and the run dirs")

    # rows
    os.makedirs(os.path.join(a.out_dir, "data"))
    os.makedirs(os.path.join(a.out_dir, "metadata"))
    tbl = {sp: pq.read_table(os.path.join(B, "data", f"{sp}.parquet")) for sp in ("tmax_clean", "tmax_repair")}
    schema = tbl["tmax_clean"].schema
    rows = {sp: tbl[sp].to_pylist() for sp in tbl}
    tars = {sp: read_tar(os.path.join(B, "data", f"tasks-{sp}.tar")) for sp in tbl}
    moved, kept = [], []
    new_repair = []
    for r in rows["tmax_repair"]:
        t = r["task_id"]
        if t not in final:
            new_repair.append(r)
            continue
        f = final[t]
        n_k = int(f["counted_attempts"])
        hist = json.loads(r["verdict_history"])
        hist.append({"stage": f"kimi-max retry (tmax14k-rollout {a.rev[:8]})", "label": f["final"],
                     **({"solved_attempt": f["solved_attempt"]} if f["solved_attempt"] else {})})
        r["verdict_history"] = json.dumps(hist, sort_keys=True)
        r["rollout_attempts"] = (r["rollout_attempts"] or 0) + n_k
        if t in solved:
            rp = replays[t]
            body, n_act = BT.rebuild_solution(t, rp)
            k = f["solved_attempt"]
            if (f[f"attempt{k}_model"], f[f"attempt{k}_effort"], f[f"attempt{k}_top_p"]) != ("moonshotai/Kimi-K3", "max", "0.95"):
                die("a solving attempt is not Kimi K3 max top_p 0.95")
            r.update(solution=body, solution_format="terminus_tmux_actions_v1", solution_model=f[f"attempt{k}_model"],
                     solution_reasoning_effort="max",
                     solution_episode_id=f"{t}__{rp['episode_run']}__trial-{rp['trial']}/attempt-{rp['attempt']}",
                     solution_phase=rp["replay_run"], solution_replay_reward=1.0,
                     solution_actions_sha256=rp["actions_sha256"], solution_terminal_action_count=n_act,
                     solution_replay_resources_json=rp.get("replay_resources_json") or "",
                     solution_is_normalized_shell=False, has_reference_solution=True,
                     solution_source="kimi_max_replay_verified" + (":" + rp["transform"] if rp.get("transform") else ""),
                     rollout_reward_best=1.0, rollout_best_attempt=f"kimi_max_attempt{k}", rollout_best_effort="max",
                     repair_hint="")
            r.update(rrows[t])
            r["column_gaps"] = json.dumps([g for g in json.loads(r["column_gaps"]) if g["column"] != "solution"],
                                          sort_keys=True)
            r["origin"] += f"; solution Kimi K3 max attempt {k} (tmax14k-rollout {a.rev[:8]}), deciding replay {rp['replay_run']}"
            r["split"], r["shard"] = "tmax_clean", "data/tasks-tmax_clean.tar"
            if not (r["tier"] == "SEED" and r["verdict"] == "CLEAN"):
                die("a moved row is not SEED/CLEAN")
            rows["tmax_clean"].append(r)
            tars["tmax_clean"][t] = tars["tmax_repair"].pop(t)
            moved.append(t)
        else:
            new_repair.append(r)
            kept.append(t)
    rows["tmax_repair"] = new_repair
    for sp in rows:
        rows[sp].sort(key=lambda x: x["task_id"])
        pq.write_table(pa.Table.from_pylist(rows[sp], schema=schema), os.path.join(a.out_dir, "data", f"{sp}.parquet"))
        BT.write_tar(os.path.join(a.out_dir, "data", f"tasks-{sp}.tar"), tars[sp])
    for f in os.listdir(os.path.join(B, "data")):
        if not f.startswith(("tmax_clean", "tmax_repair", "tasks-tmax_clean", "tasks-tmax_repair")):
            shutil.copy2(os.path.join(B, "data", f), os.path.join(a.out_dir, "data", f))
    for f in os.listdir(os.path.join(B, "metadata")):
        if f not in ("split_assignment_tmax.tsv", "BUILD_RECORD.json", "PUBLISH_RECORD.json", "PUBLISH_DRYRUN.json"):
            shutil.copy2(os.path.join(B, "metadata", f), os.path.join(a.out_dir, "metadata", f))

    # split file
    byid = {r["task_id"]: r for sp in rows for r in rows[sp]}
    cols = list(assign[0].keys()) + ["kimi_final", "kimi_solved_attempt", "kimi_episode_run", "kimi_replay_run",
                                     "kimi_attempts"]
    for r in assign:
        t = r["task_id"]
        if t in final:
            f = final[t]
            r.update(kimi_final=f["final"], kimi_solved_attempt=f["solved_attempt"], kimi_episode_run=f["episode_run"],
                     kimi_replay_run=f["replay_run"], kimi_attempts=f["counted_attempts"],
                     verdict_history=byid[t]["verdict_history"], origin=byid[t]["origin"], split=byid[t]["split"],
                     rollout_attempts=byid[t]["rollout_attempts"])
            if t in solved:
                r.update(reason=f"tier SEED, verdict CLEAN, and a replay-verified solution (Kimi K3 max, top_p 0.95, "
                                f"attempt {f['solved_attempt']}, after DeepSeek scored 0 in 3)", replay_verified=1)
            else:
                r.update(reason="repair_unsolved: unsolved after 3 DeepSeek + 3 Kimi (Kimi K3 max, top_p 0.95)")
    with open(os.path.join(a.out_dir, "metadata", "split_assignment_tmax.tsv"), "w") as fh:
        w = csv.DictWriter(fh, cols, delimiter="\t", lineterminator="\n", restval="")
        w.writeheader()
        w.writerows(assign)

    rec = json.load(open(os.path.join(B, "metadata", "BUILD_RECORD.json")))
    rec["counts"]["split"] = {sp: len(rows[sp]) if sp in rows else rec["counts"]["split"][sp] for sp in rec["counts"]["split"]}
    rec["kimi_max"] = {"rev": rev, "base_rev": a.base_rev, "moved_to_clean": len(moved), "unsolved_after_3": len(kept),
                       "solved_attempt": dict(collections.Counter(final[t]["solved_attempt"] for t in moved)),
                       "replay_runs": [os.path.basename(d) for d in runs],
                       "tables_sha256": {f: sha(os.path.join(K, f)) for f in sorted(os.listdir(K)) if f.endswith(".tsv")}}
    rec["built_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(rec, open(os.path.join(a.out_dir, "metadata", "BUILD_RECORD.json"), "w"), indent=1, sort_keys=True)
    print(json.dumps({"moved_to_clean": len(moved), "unsolved_after_3": len(kept),
                      "tmax_clean": len(rows["tmax_clean"]), "tmax_repair": len(rows["tmax_repair"]),
                      "moved_security": sum(1 for t in moved if byid[t]["security"])}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
