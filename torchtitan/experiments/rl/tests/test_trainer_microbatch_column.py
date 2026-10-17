# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""PolicyTrainer picks its microbatch from a full row or a one-element column."""

import pytest

from torchtitan.experiments.rl.actors.trainer import PolicyTrainer


def _trainer(dp_rank: int, dp_size: int) -> PolicyTrainer:
    trainer = object.__new__(PolicyTrainer)
    trainer.dp_rank = dp_rank
    trainer.dp_size = dp_size
    return trainer


def test_full_row_picks_own_dp_rank() -> None:
    assert _trainer(2, 4)._select_local_microbatch(["a", "b", "c", "d"]) == "c"


def test_column_send_uses_the_single_microbatch() -> None:
    assert _trainer(2, 4)._select_local_microbatch(["c"]) == "c"


def test_dp1_row_and_column_agree() -> None:
    assert _trainer(0, 1)._select_local_microbatch(["a"]) == "a"


def test_wrong_length_rejected() -> None:
    with pytest.raises(ValueError, match="expected 4"):
        _trainer(0, 4)._select_local_microbatch(["a", "b"])
