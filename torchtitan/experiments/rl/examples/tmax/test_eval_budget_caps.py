# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from unittest import mock

from torchtitan.experiments.rl.examples.tmax import config_registry


def _rollouter_config(**env):
    base = {"SWE_VAL_SAMPLES": "0"}
    with mock.patch.dict(os.environ, {**base, **env}):
        return config_registry._tmax_rollouter()


def test_eval_budget_caps_default_off():
    cfg = _rollouter_config()
    assert cfg.eval_agent_budget_cap_sec == 0
    assert cfg.eval_verifier_budget_cap_sec == 0


def test_eval_budget_caps_read_from_env():
    cfg = _rollouter_config(
        SWE_EVAL_AGENT_BUDGET_CAP_SEC="4800", SWE_EVAL_VERIFIER_BUDGET_CAP_SEC="900"
    )
    assert cfg.eval_agent_budget_cap_sec == 4800
    assert cfg.eval_verifier_budget_cap_sec == 900
