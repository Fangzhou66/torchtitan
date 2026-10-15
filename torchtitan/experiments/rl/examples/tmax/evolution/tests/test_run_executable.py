# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import errno
import subprocess

import pytest
import verifier_probes as vp


def test_run_executable_retries_text_file_busy(monkeypatch) -> None:
    calls = []

    def flaky(argv, **kwargs):
        calls.append(argv)
        if len(calls) < 3:
            raise OSError(errno.ETXTBSY, "Text file busy")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(vp.subprocess, "run", flaky)
    monkeypatch.setattr(vp.time, "sleep", lambda _s: None)
    assert vp.run_executable(["./sandbox", "up"]).returncode == 0
    assert len(calls) == 3


def test_run_executable_does_not_retry_other_errors(monkeypatch) -> None:
    def missing(argv, **kwargs):
        raise OSError(errno.ENOENT, "No such file")

    monkeypatch.setattr(vp.subprocess, "run", missing)
    with pytest.raises(FileNotFoundError):
        vp.run_executable(["./sandbox"])
