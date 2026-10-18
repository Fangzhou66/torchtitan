# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""grade_separate for TB4: Harbor's artifact entries, collect hooks and reward rule."""

from __future__ import annotations

import asyncio
import contextlib
import re

import pytest

from torchtitan.experiments.rl.examples.tmax import grading


class _Verifier:
    """A verifier sandbox whose test.sh writes the files a case asks for."""

    def __init__(self, writes: dict[str, str]):
        self.writes = writes
        self.files: dict[str, str] = {}
        self.cmds: list[tuple[str, dict]] = []

    async def exec(self, cmd, *, user="root", check=False, timeout=60, env=None):
        self.cmds.append((cmd, {"env": env, "timeout": timeout}))
        nonce = re.search(r"tmax-sentinel-[0-9a-f]+", cmd)
        if nonce:
            if "rm -f /logs/verifier/reward.json" in cmd:
                self.files.pop("/logs/verifier/reward.json", None)
            self.files["/logs/verifier/reward.txt"] = nonce.group(0)
        if "bash /tests/test.sh" in cmd:
            self.files.update(self.writes)
            return 0, "tests ran", ""
        if cmd.startswith("test -e "):
            return (0 if cmd.split()[-1] in self.files else 1), "", ""
        return 0, "", ""

    async def read_file(self, path, *, user="root"):
        return self.files.get(path, "")

    async def write_file(self, path, content, *, user="root"):
        self.files[path] = content


def _grade(writes, *, harbor_reward=True, env=None, stale_json=False):
    sb = _Verifier(writes)
    if stale_json:
        sb.files["/logs/verifier/reward.json"] = '{"reward": 1.0}'
    diagnostics: dict = {}
    reward = asyncio.run(
        grading.grade_tmax(
            sb,
            {"test_sh": ""},
            workdir="/app",
            timeout_sec=60,
            diagnostics=diagnostics,
            upload_tests=False,
            harbor_reward=harbor_reward,
            env=env,
        )
    )
    return reward, diagnostics, sb


J, T = "/logs/verifier/reward.json", "/logs/verifier/reward.txt"


@pytest.mark.parametrize(
    "writes, reward, source",
    [
        ({J: '{"reward": 0.25, "partial": 0.5}'}, 0.25, "reward.json"),
        ({J: '{"score": 1.0}'}, 1.0, "reward.json"),  # one value, any key
        ({J: '{"reward": 1.0}', T: "0"}, 1.0, "reward.json"),  # Harbor: json wins
        ({T: "0.5\n"}, 0.5, "reward.txt"),
        ({J: "not json"}, 0.0, "reward.json"),
        ({J: '{"a": 1, "b": 0}'}, 0.0, "reward.json"),
        ({}, 0.0, "reward.txt"),  # neither: the sentinel is left, 0
    ],
)
def test_harbor_reward_rule(writes, reward, source):
    got, diagnostics, _sb = _grade(writes)
    assert got == reward
    assert diagnostics["reward_file"] == source
    assert diagnostics["exit_code"] == 0 and diagnostics["output_tail"] == "tests ran"


def test_a_stale_reward_json_is_removed_before_the_verifier_runs():
    got, _d, _sb = _grade({}, stale_json=True)
    assert got == 0.0


def test_the_separate_verifier_can_write_its_outputs_as_its_own_user():
    """Harbor opens /logs/verifier (777) before the verifier runs; a verifier
    that drops privileges (live-database-cutover's postgres) writes there. The
    sentinel is opened too, so such a verifier can also replace it."""
    _reward, _diag, sb = _grade({T: "1"})
    pre = next(cmd for cmd, _ in sb.cmds if "tmax-sentinel-" in cmd)
    assert pre.endswith("; chmod 777 /logs/verifier; chmod 666 /logs/verifier/reward.txt")
    _reward, _diag, plain = _grade({T: "1"}, harbor_reward=False)
    assert not any("chmod 777 /logs/verifier" in cmd for cmd, _ in plain.cmds)


def test_without_harbor_reward_json_is_ignored_and_no_env_is_passed():
    got, diagnostics, sb = _grade({J: '{"reward": 1.0}'}, harbor_reward=False)
    assert got == 0.0 and "reward_file" not in diagnostics
    test_sh = next(kw for cmd, kw in sb.cmds if "bash /tests/test.sh" in cmd)
    assert test_sh["env"] is None


def test_verifier_env_reaches_test_sh():
    _got, _d, sb = _grade({T: "1"}, env={"PORT": "8000"})
    test_sh = next(kw for cmd, kw in sb.cmds if "bash /tests/test.sh" in cmd)
    assert test_sh["env"] == {"PORT": "8000"}


# --------------------------------------------------------------------------- #
# grade_separate orchestration
# --------------------------------------------------------------------------- #
class _ComposeAgent:
    def __init__(self, log, *, fail_hook=False):
        self.log = log
        self.fail_hook = fail_hook

    async def exec(self, cmd, *, user="root", check=False, timeout=60, env=None):
        self.log.append(("exec", cmd[:40], user, timeout))
        return 0, "", ""

    async def service_exec(self, cmd, *, service, user=None, timeout=60, env=None):
        self.log.append(("service_exec", service, cmd, timeout))
        if self.fail_hook:
            raise RuntimeError("hook boom")
        return 0, "", ""

    async def stop_service(self, service):
        self.log.append(("stop", service))

    async def copy_out(self, service, path):
        self.log.append(("copy_out", service, path))
        if path == "/gone":
            raise RuntimeError("no running container for service")
        return f"{service}:{path}".encode()


def _install(monkeypatch, log):
    async def collect(sb, paths):
        log.append(("collect", list(paths)))
        return b"main-tar"

    async def excluding(sb, source, exclude):
        log.append(("excluding", source, list(exclude)))
        return b"dir-tar"

    async def restore(sb, data, into=None):
        log.append(("restore", data, into))

    async def grade(sb, tmax, **kw):
        log.append(("grade", kw["harbor_reward"], kw["env"], kw["upload_tests"]))
        return 1.0

    monkeypatch.setattr(grading, "collect_artifacts", collect)
    monkeypatch.setattr(grading, "_collect_excluding", excluding)
    monkeypatch.setattr(grading, "restore_artifacts", restore)
    monkeypatch.setattr(grading, "grade_tmax", grade)
    booted: list = []

    @contextlib.asynccontextmanager
    async def boot(image, **kwargs):
        booted.append((image, kwargs))
        yield object()

    return boot, booted


_SV = {
    "image": "verifier/image",
    "artifacts": [
        "/app/out.json",
        {"source": "/audit/decisions.json", "service": "portal"},
        {"source": "/workspace/gen", "exclude": ["node_modules", "*.pyc"]},
    ],
    "convention_dir": "/logs/artifacts",
    "collect": [
        {"service": "portal", "command": "dump-db", "timeout_sec": 10.0},
        {"service": "main", "command": "git diff", "timeout_sec": 60.0},
    ],
    "env": {"PORT": "8000"},
    "daytona_cpu": 2,
    "daytona_gpu": 1,
    "daytona_gpu_type": ["H100"],
    "daytona_target": "earth",
    "network_block_all": True,
}


def test_collection_follows_harbors_phases(monkeypatch):
    log: list = []
    boot, booted = _install(monkeypatch, log)
    diagnostics: dict = {}
    reward = asyncio.run(
        grading.grade_separate(
            _ComposeAgent(log),
            {"separate_verifier": _SV},
            workdir="/app",
            boot=boot,
            timeout_sec=7200,
            diagnostics=diagnostics,
        )
    )
    assert reward == 1.0
    assert log == [
        ("exec", "git diff", "root", 60),  # main hooks first
        ("collect", ["/app/out.json", "/logs/artifacts"]),  # then main artifacts
        ("excluding", "/workspace/gen", ["node_modules", "*.pyc"]),
        ("stop", "main"),  # main stops before any sidecar is touched
        ("service_exec", "portal", "dump-db", 10),
        ("copy_out", "portal", "/audit/decisions.json"),
        ("restore", b"main-tar", None),
        ("restore", b"dir-tar", "/workspace/gen"),
        ("restore", b"portal:/audit/decisions.json", None),
        ("grade", True, {"PORT": "8000"}, False),
    ]
    image, kwargs = booted[0]
    assert image == "verifier/image" and kwargs["cpu"] == 2
    assert kwargs["sandbox_options"] == {
        "gpu": 1,
        "gpu_type": ["H100"],
        "target": "earth",
        "network_block_all": True,
    }
    record = diagnostics["separate_verifier"]
    assert record["timeout_sec"] == 7200
    assert record["sidecar_artifacts"] == {"portal:/audit/decisions.json": 28}
    assert [h["service"] for h in record["hooks"]] == ["main", "portal"]


def test_a_failing_hook_is_recorded_not_fatal(monkeypatch):
    log: list = []
    boot, _booted = _install(monkeypatch, log)
    diagnostics: dict = {}
    sv = {
        **_SV,
        "artifacts": [
            {"source": "/gone", "service": "portal"},
            {"source": "/audit/x", "service": "portal"},
        ],
    }
    asyncio.run(
        grading.grade_separate(
            _ComposeAgent(log, fail_hook=True),
            {"separate_verifier": sv},
            workdir="/app",
            boot=boot,
            diagnostics=diagnostics,
        )
    )
    record = diagnostics["separate_verifier"]
    assert "hook boom" in record["hooks"][1]["error"]
    # A sidecar artifact that cannot be copied is recorded; the next one still is.
    assert "no running container" in record["sidecar_artifacts"]["portal:/gone"]
    assert record["sidecar_artifacts"]["portal:/audit/x"] == len(b"portal:/audit/x")
    assert ("restore", b"portal:/audit/x", None) in log


def test_a_plain_row_boots_and_collects_exactly_as_before(monkeypatch):
    log: list = []
    boot, booted = _install(monkeypatch, log)
    asyncio.run(
        grading.grade_separate(
            _ComposeAgent(log),
            {"separate_verifier": {"image": "v", "artifacts": ["/app/out.step"]}},
            workdir="/app",
            boot=boot,
        )
    )
    assert log[0] == ("collect", ["/app/out.step"])
    assert "sandbox_options" not in booted[0][1]
    assert not any(entry[0] in ("stop", "service_exec", "copy_out") for entry in log)


def test_sidecar_artifacts_need_a_compose_sandbox(monkeypatch):
    log: list = []
    boot, _booted = _install(monkeypatch, log)

    class _Plain:
        async def exec(self, *a, **k):
            return 0, "", ""

    sv = {"image": "v", "artifacts": [{"source": "/a", "service": "db"}]}
    with pytest.raises(ValueError, match="compose services"):
        asyncio.run(
            grading.grade_separate(
                _Plain(), {"separate_verifier": sv}, workdir="/app", boot=boot
            )
        )


def test_restore_into_empties_the_directory_first():
    class _Sb:
        def __init__(self):
            self.cmds = []

        async def write_file(self, path, content, *, user="root"):
            pass

        async def exec(self, cmd, **kw):
            self.cmds.append(cmd)
            return 0, "", ""

    sb = _Sb()
    asyncio.run(grading.restore_artifacts(sb, b"x", into="/workspace/gen"))
    assert sb.cmds[0].startswith(
        grading._empty_dir_command("/workspace/gen") + " && tar -C /workspace/gen -xzf "
    )
    sb = _Sb()
    asyncio.run(grading.restore_artifacts(sb, b"x"))
    assert sb.cmds[0].startswith("tar -C / -xzf ")


def _tgz(entries: dict[str, bytes | None]) -> bytes:
    import io
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in entries.items():
            info = tarfile.TarInfo(name)
            if data is None:
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
            else:
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_directory_artifacts_are_emptied_before_restore_as_harbor_does():
    data = _tgz({"app/output": None, "app/output/model.pt": b"x", "app/out.json": b"{}"})
    assert grading._dir_artifacts(data, ["/app/output", "/app/out.json", "/missing"]) == [
        "/app/output"
    ]
    assert grading._dir_artifacts(b"not a tar", ["/app/output"]) == []

    class _Sb:
        def __init__(self):
            self.cmds = []

        async def write_file(self, path, content, *, user="root"):
            pass

        async def exec(self, cmd, **kw):
            self.cmds.append(cmd)
            return 0, "", ""

    sb = _Sb()
    asyncio.run(grading.restore_artifacts(sb, data, empty=["/app/output"]))
    assert sb.cmds[0].startswith(
        grading._empty_dir_command("/app/output") + " && tar -C / -xzf "
    )


def test_emptying_keeps_the_directory_a_shell_is_standing_in(tmp_path):
    """Harbor's empty_dirs empties in place. Our exec runs in a long-lived
    session shell; if that shell stands in the directory (a verifier image's
    WORKDIR that is also a directory artifact, as medical-claims-processing's
    /shared is), replacing the directory leaves every later command in a
    deleted one. Emptying in place does not."""
    import os
    import subprocess

    d = tmp_path / "shared"
    (d / "sub").mkdir(parents=True)
    (d / "old.txt").write_text("x")
    (d / ".hidden").write_text("x")
    inode = d.stat().st_ino
    getcwd = 'python3 -c "import os; print(os.getcwd())"'

    def in_shell_standing_in_d(command):
        return subprocess.run(
            ["sh", "-c", f"{command} && {getcwd}"],
            cwd=d,
            capture_output=True,
            text=True,
        )

    ok = in_shell_standing_in_d(grading._empty_dir_command(str(d)))
    assert ok.returncode == 0, ok.stderr
    assert ok.stdout.strip() == str(d)
    assert list(d.iterdir()) == [] and d.stat().st_ino == inode
    assert oct(d.stat().st_mode & 0o777) == "0o777"

    # The replace-the-root form this replaces: the shell's cwd is gone.
    bad = in_shell_standing_in_d(f"rm -rf {d} && mkdir -p {d}")
    assert bad.returncode != 0 and "FileNotFoundError" in bad.stderr

    # A symlink or a file where the directory should be is replaced, as Harbor does.
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "elsewhere")
    assert subprocess.run(["sh", "-c", grading._empty_dir_command(str(link))]).returncode == 0
    assert link.is_dir() and not link.is_symlink()
    assert not os.path.exists(tmp_path / "elsewhere")


def test_grade_separate_empties_the_directory_artifacts_it_restores(monkeypatch):
    log: list = []
    boot, _booted = _install(monkeypatch, log)
    data = _tgz({"app/output": None, "app/output/model.pt": b"x"})

    async def collect(sb, paths):
        return data

    monkeypatch.setattr(grading, "collect_artifacts", collect)

    async def restore(sb, data, **kw):
        log.append(("restore", kw))

    monkeypatch.setattr(grading, "restore_artifacts", restore)
    sv = {"image": "v", "artifacts": ["/app/output", "/app/out.json"]}
    asyncio.run(
        grading.grade_separate(
            _ComposeAgent(log), {"separate_verifier": sv}, workdir="/app", boot=boot
        )
    )
    assert ("restore", {"empty": ["/app/output"]}) in log


def test_collection_budget_covers_hooks_stop_and_every_transfer():
    # hooks 10 + 60, stop main 60, 3 transfers x 600 (plain, excluded, sidecar),
    # and the sidecar's docker cp 630.
    assert grading.separate_collect_budget_sec(_SV) == 70 + 60 + 3 * 600 + 630
    assert grading.separate_collect_budget_sec({"artifacts": ["/app/x"]}) == 600
