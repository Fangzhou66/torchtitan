# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pack_to_dataset import patch_rebench_test_sh  # noqa: E402

# Fzz1/SWE-Rebench-Tasks-Clean@a6d8f8ea, tests/test.sh up to the line before
# `git apply test.patch`: repo lookup and test-surface reset.
_UPSTREAM = '#!/usr/bin/env bash\n# Grader: reward 1 iff every FAIL_TO_PASS passes and the PASS_TO_PASS sample\n# holds. The graded tests are materialized here, from a patch the agent has\n# never seen: the test surface is first restored to HEAD, so nothing the agent\n# planted in a graded test, a fixture beside it, or a conftest that feeds it\n# survives to be graded.\nset -uo pipefail\n\n# The repo is wherever the image put it: the shallowest .git whose HEAD is the\n# task\'s base commit. Depth-limited so a stray vendored .git cannot win.\nfind_repo() {\n  want="$1"\n  for d in $(find / -maxdepth 2 -name .git -type d 2>/dev/null | sort); do\n    r="$(dirname "$d")"\n    if [ "$(git -C "$r" rev-parse HEAD 2>/dev/null)" = "$want" ]; then\n      echo "$r"; return 0\n    fi\n  done\n  # No fallback. Grading "any repo at all" is worse than not grading: the old\n  # fallback ran the graded tests against whatever .git turned up first, and\n  # whatever it scored was recorded as a fact about this task.\n  echo "no repository at base commit $want" >&2\n  return 1\n}\n\nHERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\nmkdir -p /logs/verifier\n# Fail closed. A grader with no repository at the task\'s base commit has\n# nothing to say about the task, and saying 0 without saying why is how a\n# broken box gets recorded as a wrong fix.\nR="$(find_repo 69d24e856d04abbf44a81786b67cc26e98d43ee5)" || {\n  echo "error=no_repo_at_base_commit" > /logs/verifier/pytest_rc.txt\n  echo 0 > /logs/verifier/reward.txt\n  exit 0\n}\ncd "$R"\n# `HEAD --`, never a bare `--`: `git checkout -- <path>` restores from the\n# INDEX, so an agent who edited a graded test and `git add`ed it would have\n# their own version restored by the very step meant to undo it.\nwhile IFS= read -r f; do\n  [ -z "$f" ] && continue\n  git checkout -q HEAD -- "$f" 2>/dev/null || rm -f -- "$f"\ndone <<\'TPFILES\'\ntests/test_a.py\nTPFILES\n# And the whole test surface, not only the paths test_patch happens to touch:\n# a graded test usually imports helpers, fixtures and a conftest that the patch\n# never mentions, and each of those is a place to make a failing test pass.\n# Restricted to what a source fix never touches — measured over all 1,093 gold\n# patches: none touches a tests/ tree, a conftest.py or pytest.ini, while 34\n# touch pyproject.toml, 7 tox.ini and 4 setup.cfg, which is why those three are\n# reported below rather than restored. `git clean` without -x, so the image\'s\n# own ignored artifacts (a generated _version.py, an egg-info) stay: removing\n# them leaves a repository that cannot import itself.\n# The test surface, listed out of HEAD and restored in one pass. Not\n# `git checkout -- :(glob)...`: one pathspec that matches nothing makes git\n# refuse the WHOLE command, so in any repository without a test/ directory the\n# entire reset silently did nothing — which is how this was found. Listing from\n# HEAD also restores a graded test the agent deleted outright, which a worktree\n# listing would have missed.\ngit ls-tree -r -z --name-only HEAD \\\n  | grep -zE \'(^|/)(tests?|testing)/|(^|/)conftest\\.py$|(^|/)pytest\\.ini$\' \\\n  | xargs -0 -r git checkout -q HEAD -- 2>/dev/null\n# and anything the agent ADDED in there that HEAD never had. --exclude-standard\n# so the image\'s own ignored artifacts (a generated _version.py, an egg-info)\n# stay: removing those leaves a repository that cannot import itself.\ngit ls-files -z --others --exclude-standard \\\n  | grep -zE \'(^|/)(tests?|testing)/|(^|/)conftest\\.py$|(^|/)pytest\\.ini$\' \\\n  | xargs -0 -r rm -f -- 2>/dev/null\ntouched="$(git diff --name-only HEAD -- \':(glob)**/pyproject.toml\' \':(glob)**/setup.cfg\' \':(glob)**/tox.ini\' 2>/dev/null)"\n'

_REPORT = """
echo "repo=$R"
echo "src=$(cat src.py)"
echo "test_a=$(cat tests/test_a.py)"
echo "test_b=$([ -e tests/test_b.py ] && cat tests/test_b.py || echo MISSING)"
echo "conftest=$(cat tests/conftest.py)"
echo "evil=$([ -e tests/test_evil.py ] && echo PRESENT || echo ABSENT)"
"""


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _repo(root):
    repo = root / "proj"
    (repo / "tests").mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "a@b")
    _git(repo, "config", "user.name", "a")
    (repo / "src.py").write_text("BUGGY")
    (repo / "tests/test_a.py").write_text("ORIG_A")
    (repo / "tests/test_b.py").write_text("ORIG_B")
    (repo / "tests/conftest.py").write_text("ORIG_CONF")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    base = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return repo, base


def _fix_and_tamper(repo):
    (repo / "src.py").write_text("FIXED")
    (repo / "tests/test_a.py").write_text("TAMPERED")
    _git(repo, "rm", "-q", "tests/test_b.py")
    (repo / "tests/conftest.py").write_text("CHEAT")
    (repo / "tests/test_evil.py").write_text("EVIL")


def _grade(script, root, base):
    script = re.sub(r"\b[0-9a-f]{40}\b", base, script)
    script = script.replace("find / ", f"find {root} ")
    script = script.replace("/logs/verifier", str(root / "logs"))
    path = root / "test.sh"
    path.write_text(script + _REPORT)
    out = subprocess.run(["bash", str(path)], capture_output=True, text=True, cwd=root)
    return dict(l.split("=", 1) for l in out.stdout.splitlines() if "=" in l)


_CLEAN = {
    "src": "FIXED",
    "test_a": "ORIG_A",
    "test_b": "ORIG_B",
    "conftest": "ORIG_CONF",
    "evil": "ABSENT",
}


def test_patch_matches_upstream_once_and_is_idempotent_noop_elsewhere():
    patched = patch_rebench_test_sh(_UPSTREAM)
    assert patched != _UPSTREAM
    assert patch_rebench_test_sh(patched) == patched
    assert patch_rebench_test_sh("#!/bin/bash\npytest /tests\n") == (
        "#!/bin/bash\npytest /tests\n"
    )


def test_drifted_grader_fails_loudly():
    drifted = _UPSTREAM.replace("git checkout -q HEAD -- \"$f\"", "git restore \"$f\"")
    with pytest.raises(ValueError, match="occurs 0 times"):
        patch_rebench_test_sh(drifted)


@pytest.mark.parametrize("commit", [False, True])
def test_fix_survives_and_tampering_is_undone(tmp_path, commit):
    repo, base = _repo(tmp_path)
    _fix_and_tamper(repo)
    if commit:
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "fix")
    got = _grade(patch_rebench_test_sh(_UPSTREAM), tmp_path, base)
    assert got.pop("repo") == str(repo)
    assert got == _CLEAN


def test_upstream_and_patched_agree_when_nothing_is_committed(tmp_path):
    repo, base = _repo(tmp_path)
    _fix_and_tamper(repo)
    got = _grade(_UPSTREAM, tmp_path, base)
    assert got.pop("repo") == str(repo)
    assert got == _CLEAN


def test_upstream_scores_a_committed_fix_without_grading(tmp_path):
    repo, base = _repo(tmp_path)
    _fix_and_tamper(repo)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "fix")
    assert _grade(_UPSTREAM, tmp_path, base) == {}
    assert (tmp_path / "logs/reward.txt").read_text().strip() == "0"


def test_unrelated_checkout_still_finds_no_repo(tmp_path):
    repo, base = _repo(tmp_path)
    _git(repo, "checkout", "-q", "--orphan", "other")
    _git(repo, "rm", "-rqf", ".")
    (repo / "src.py").write_text("ELSEWHERE")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "other")
    assert _grade(patch_rebench_test_sh(_UPSTREAM), tmp_path, base) == {}
    assert (tmp_path / "logs/reward.txt").read_text().strip() == "0"
