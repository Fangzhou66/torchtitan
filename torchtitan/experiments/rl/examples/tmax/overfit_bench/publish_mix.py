#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Publish a rewriter's training mix into a bench job.

    publish_mix.py --job <job dir> --rows <mix.jsonl>

Every row must parse the way the trainer parses it (TMaxDataset); a file with
any row that does not is refused whole and nothing is published. On success
the rows become the next version under data/mix/history/ and live.jsonl moves
to it; the trainer picks it up at its next draw. Prints the published version.
Any number of rows, any task ids: which tasks the mix holds is the rewriter's.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import new_root  # noqa: E402

layout = new_root.layout


def check_rows(text: str) -> list[str]:
    from torchtitan.experiments.rl.examples.tmax.data import _parse_sample_row

    rows = []
    for i, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            _parse_sample_row(json.loads(line))
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            raise SystemExit(f"row {i}: {e}") from e
        rows.append(line.strip())
    if not rows:
        raise SystemExit("no rows")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--job", type=Path, required=True)
    ap.add_argument("--rows", type=Path, required=True)
    a = ap.parse_args()
    rows = check_rows(a.rows.read_text(encoding="utf-8"))
    version, path = layout.Root(a.job).mix.publish(rows)
    print(f"v{version} {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
