# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bake tmux into a TB2 eval JSONL so sandboxes no longer install it at boot.

The TB2 images ship without tmux, so harbor's Terminus-2 runs
``apt-get update && apt-get install -y tmux`` (120 s exec timeout) in every
sandbox. When a whole eval pass starts at once the mirrors slow down, the
install times out, and the trial fails with "Failed to start tmux session"
before turn 0, scored 0.

``bake`` gives every row a Dockerfile: its existing one (env patches) or
``FROM <image>``, plus a RUN that installs tmux the way harbor does at runtime.
Daytona builds and caches it server-side, so the install is paid once per task.
The apt lists are kept, as harbor's runtime install leaves them, so an agent's
bare ``apt-get install`` behaves as before. ``verify`` boots every row from its
Dockerfile (which also warms Daytona's build cache) and checks tmux and $HOME
against the original image.

    python -m torchtitan.experiments.rl.examples.tmax.bake_tmux_into_tb2 \
        bake SRC.jsonl DST.jsonl
    python -m torchtitan.experiments.rl.examples.tmax.bake_tmux_into_tb2 \
        verify DST.jsonl REPORT.jsonl [concurrency]
"""

import argparse
import asyncio
import json
import time

TMUX_BLOCK = (
    "# tmux for the Terminus-2 agent, installed the way harbor does it at runtime\n"
    "RUN if ! command -v tmux >/dev/null 2>&1; then "
    "if command -v apt-get >/dev/null 2>&1; then "
    "DEBIAN_FRONTEND=noninteractive apt-get update && "
    "DEBIAN_FRONTEND=noninteractive apt-get install -y tmux; "
    "elif command -v apk >/dev/null 2>&1; then apk add --no-cache tmux; "
    "elif command -v dnf >/dev/null 2>&1; then dnf install -y tmux; "
    "elif command -v yum >/dev/null 2>&1; then yum install -y tmux; "
    "else echo 'no supported package manager for tmux' >&2; exit 1; fi; "
    "fi && tmux -V\n"
)

_PROBE = (
    "echo TMUX=$(tmux -V 2>&1); echo HOME=$HOME; "
    "echo APT_LISTS=$(ls /var/lib/apt/lists 2>/dev/null | wc -l)"
)


def bake(src: str, dst: str) -> None:
    with open(src) as f_in, open(dst, "w") as f_out:
        for line in f_in:
            row = json.loads(line)
            md = row["metadata"]
            base = md.get("dockerfile") or f"FROM {md['image']}\n"
            md["dockerfile"] = base.rstrip("\n") + "\n" + TMUX_BLOCK
            f_out.write(json.dumps(row) + "\n")


async def _probe(md: dict, use_dockerfile: bool) -> dict:
    from torchtitan.experiments.rl.harness.sandbox.daytona import DaytonaSandbox

    sb = DaytonaSandbox(
        md["image"],
        dockerfile=md["dockerfile"] if use_dockerfile else None,
        cpu=md.get("daytona_cpu"),
        memory=md.get("daytona_mem_gb"),
        disk_gb=md.get("daytona_disk_gb"),
        labels={"run": "tb2-bake-tmux-verify"},
    )
    t0 = time.monotonic()
    async with sb:
        boot = time.monotonic() - t0
        _rc, out, _err = await sb.exec(_PROBE, timeout=60)
    kv = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    kv["boot_secs"] = round(boot, 1)
    return kv


async def _verify_row(md: dict, sem: asyncio.Semaphore) -> dict:
    async with sem:
        rec = {"task": md["instance_id"]}
        for key, use_dockerfile in (("new", True), ("orig", False)):
            try:
                rec[key] = await _probe(md, use_dockerfile)
            except Exception as e:  # noqa: BLE001 -- report every row
                rec[f"{key}_error"] = f"{type(e).__name__}: {e}"[:500]
        new, orig = rec.get("new", {}), rec.get("orig", {})
        rec["ok"] = new.get("TMUX", "").startswith("tmux ") and new.get(
            "HOME"
        ) == orig.get("HOME")
        print(json.dumps(rec), flush=True)
        return rec


async def verify(src: str, report: str, concurrency: int) -> int:
    rows = [json.loads(line)["metadata"] for line in open(src)]
    sem = asyncio.Semaphore(concurrency)
    recs = await asyncio.gather(*(_verify_row(md, sem) for md in rows))
    with open(report, "w") as f:
        for rec in recs:
            f.write(json.dumps(rec) + "\n")
    bad = [rec["task"] for rec in recs if not rec["ok"]]
    print(f"{len(recs) - len(bad)}/{len(recs)} ok; not ok: {bad}", flush=True)
    return 1 if bad else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_bake = sub.add_parser("bake")
    p_bake.add_argument("src")
    p_bake.add_argument("dst")
    p_verify = sub.add_parser("verify")
    p_verify.add_argument("src")
    p_verify.add_argument("report")
    p_verify.add_argument("concurrency", nargs="?", type=int, default=8)
    args = parser.parse_args()
    if args.cmd == "bake":
        bake(args.src, args.dst)
    else:
        raise SystemExit(asyncio.run(verify(args.src, args.report, args.concurrency)))


if __name__ == "__main__":
    main()
