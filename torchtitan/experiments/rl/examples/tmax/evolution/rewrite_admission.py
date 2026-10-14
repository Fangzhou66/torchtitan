# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Optional, expiring admission lease for an external training lifecycle watcher."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


def admission_is_open(*, dry: bool = False) -> bool:
    """Pause new authors on restart; already-running rewrites may finish."""
    path = os.environ.get("EVOLVE_ADMISSION_STATUS_FILE")
    if dry or not path:
        return True
    try:
        status = json.loads(Path(path).read_text())
        return status["allow_admission"] is True and status["expires_at"] > time.time()
    except (OSError, KeyError, TypeError, ValueError):
        return False
