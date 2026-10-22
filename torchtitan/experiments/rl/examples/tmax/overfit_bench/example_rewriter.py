#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A minimal rewriter on the scheduler socket: it rewrites nothing, it steers.

    example_rewriter.py <job>

Every training draw asks it which version to train on (the ``which_version``
hook); it picks the active version with the fewest groups so far. When a group
solves every scored attempt, it halves that version's weight, which matters
only to draws it does not answer in time. It is a template for the interface,
not a recipe: a real rewriter adds versions (``client.add``) and retires them.
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[6]))

from torchtitan.experiments.rl.examples.tmax.overfit_bench.rewriter_client import (  # noqa: E402
    RewriterClient,
)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    groups: collections.Counter[str] = collections.Counter()

    def which_version(candidates: list[dict]) -> str | None:
        return min(candidates, key=lambda c: groups[c["id"]])["id"] if candidates else None

    client = RewriterClient(sys.argv[1], which_version=which_version)
    client.connect()
    while client.connected:
        event = client.wait_for({"group_done"}, timeout=60)
        if event is None or event["version"] is None:
            continue
        groups[event["version"]] += 1
        if event["scored"] and event["passed"] == event["scored"]:
            state = client.get_state()
            weight = next(
                (
                    v["weight"]
                    for v in state["versions"]
                    if v["id"] == event["version"] and v["status"] == "active"
                ),
                None,
            )
            if weight is not None:
                client.reweight(event["version"], weight / 2)


if __name__ == "__main__":
    main()
