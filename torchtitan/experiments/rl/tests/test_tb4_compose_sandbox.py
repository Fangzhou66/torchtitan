# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DaytonaComposeSandbox runs a released TB4 compose project as Harbor's DinD does."""

from __future__ import annotations

import asyncio
import base64
import json
import shlex

import pytest

from torchtitan.experiments.rl.harness.sandbox import (
    compose as compose_mod,
    daytona as daytona_mod,
)

# What ``docker compose ps -q`` prints: the full 64-hex container id.
CID = "3f2a" * 16
PINNED = "harborframework/terminal-bench:demo-environment-1@sha256:" + "a" * 64
COMPOSE = (
    "services:\n"
    f"  main:\n    image: {PINNED}\n    depends_on: [api]\n"
    f"  api:\n    image: harborframework/terminal-bench:demo-sidecar-api@sha256:{'b' * 64}\n"
)


class _FakeHost:
    instances: list = []

    def __init__(self, image, **kwargs):
        self.image = image
        self.kwargs = kwargs
        self.cmds: list[tuple[str, dict | None]] = []
        self.files: dict[str, bytes | str] = {}
        self.sandbox_id = "dind-1"
        self.allocated_disk_gb = kwargs.get("disk_gb")
        self.exited = None
        _FakeHost.instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.exited = exc_type

    async def exec(self, cmd, *, user="root", env=None, timeout=120, check=False):
        self.cmds.append((cmd, env))
        return self.reply(cmd)

    def reply(self, cmd):
        if cmd == f"echo {compose_mod._EXEC_PROBE}":
            return 0, compose_mod._EXEC_PROBE + "\n", ""
        if " ps -q " in cmd:
            return 0, CID + "\n", ""
        return 0, "", ""

    async def write_file(self, path, content, *, user="root"):
        self.files[path] = content

    async def read_file(self, path, *, user="root"):
        if path.endswith(".b64"):
            return base64.b64encode(b"tgz-bytes").decode()
        return "file-text"


def _boot(monkeypatch, **kw):
    _FakeHost.instances.clear()
    monkeypatch.setattr(daytona_mod, "DaytonaSandbox", _FakeHost)
    sb = compose_mod.DaytonaComposeSandbox(
        docker_compose=COMPOSE,
        project="Ctr.Optimization",
        main_image=PINNED,
        cpu=2,
        memory=8,
        disk_gb=10,
        **kw,
    )
    asyncio.run(sb.__aenter__())
    return sb, _FakeHost.instances[-1]


def _command(host, suffix):
    return next((c, e) for c, e in host.cmds if c.endswith(suffix))


def test_host_is_dind_with_bash_and_carries_the_resources(monkeypatch):
    _sb, host = _boot(monkeypatch, target="earth")
    assert host.image == ""
    assert host.kwargs["dockerfile"] == compose_mod.DIND_DOCKERFILE
    assert "docker:28.3.3-dind" in compose_mod.DIND_DOCKERFILE
    assert (host.kwargs["cpu"], host.kwargs["memory"], host.kwargs["disk_gb"]) == (
        2,
        8,
        10,
    )
    assert host.kwargs["target"] == "earth"


def test_compose_files_and_env_match_harbors_prebuilt_project(monkeypatch):
    _sb, host = _boot(monkeypatch)
    up, env = _command(host, " up -d")
    argv = shlex.split(up)
    assert argv[:6] == [
        "docker",
        "compose",
        "-p",
        "ctr-optimization",
        "--project-directory",
        "/harbor/environment",
    ]
    files = [argv[i + 1] for i, a in enumerate(argv) if a == "-f"]
    assert files == [
        "/harbor/compose/docker-compose-resources.json",
        "/harbor/compose/docker-compose-prebuilt.yaml",
        "/harbor/compose/docker-compose-mounts.json",
        "/harbor/environment/docker-compose.yaml",
        "/harbor/compose/docker-compose-runtime.json",
        "/harbor/compose/docker-compose-environment.json",
    ]
    assert env["PREBUILT_IMAGE_NAME"] == PINNED
    assert env["CPUS"] == "2" and env["MEMORY"] == "8192M"
    assert host.files["/harbor/environment/docker-compose.yaml"] == COMPOSE
    resources = json.loads(host.files["/harbor/compose/docker-compose-resources.json"])
    assert resources["services"]["main"]["deploy"]["resources"]["reservations"] == {
        "cpus": "2",
        "memory": "8192M",
    }
    mounts = json.loads(host.files["/harbor/compose/docker-compose-mounts.json"])
    assert [m["target"] for m in mounts["services"]["main"]["volumes"]] == list(
        compose_mod.LOG_MOUNTS
    )
    # Without a row Dockerfile main is built FROM its own image: the image itself.
    assert host.files["/harbor/compose/main-context/.tmax-main.Dockerfile"] == (
        f"FROM {PINNED}\n"
    )
    order = [c.split(" ")[-1] for c, _ in host.cmds if c.startswith("docker compose")]
    assert order[:3] == ["--ignore-buildable", "build", "-d"]


def test_a_row_dockerfile_builds_main_in_its_own_context(monkeypatch):
    dockerfile = f"FROM {PINNED}\nCOPY app.py /app/\nUSER root\nRUN tmux -V\n"
    context = {"app.py": base64.b64encode(b"print(1)\n").decode()}
    _sb, host = _boot(monkeypatch, main_dockerfile=dockerfile, main_build_context=context)
    assert host.files["/harbor/compose/main-context/.tmax-main.Dockerfile"] == dockerfile
    assert host.files["/harbor/compose/main-context/app.py"] == b"print(1)\n"
    assert json.loads(host.files[compose_mod._RUNTIME_OVERRIDE]) == {
        "services": {
            "main": {
                "image": "hb__ctr-optimization",
                "build": {
                    "context": "/harbor/compose/main-context",
                    "dockerfile": ".tmax-main.Dockerfile",
                },
                "pull_policy": "build",
            }
        }
    }
    with pytest.raises(ValueError, match="not allowed"):
        _boot(monkeypatch, main_dockerfile=dockerfile, main_build_context={"../x": "eA=="})


def test_exec_routes_main_through_a_login_shell_and_sidecars_through_sh(monkeypatch):
    sb, host = _boot(monkeypatch)
    asyncio.run(sb.exec("ls /app"))
    assert shlex.split(host.cmds[-1][0]) == [
        "docker",
        "exec",
        CID,
        "bash",
        "-lc",
        "ls /app",
    ]
    asyncio.run(sb.exec("whoami", user="agent", env={"A": "1"}))
    assert shlex.split(host.cmds[-1][0]) == [
        "docker",
        "exec",
        "-u",
        "agent",
        "-e",
        "A=1",
        CID,
        "bash",
        "-lc",
        "whoami",
    ]
    asyncio.run(sb.service_exec("redis-cli save", service="redis"))
    assert shlex.split(host.cmds[-1][0])[-3:] == ["sh", "-c", "redis-cli save"]


def test_file_transfer_goes_through_docker_cp(monkeypatch):
    sb, host = _boot(monkeypatch)
    asyncio.run(sb.write_file("/app/x.txt", "hi", user="agent"))
    joined = "\n".join(c for c, _ in host.cmds)
    assert "docker cp /tmp/tt_wf_" in joined and f"{CID}:/app/x.txt" in joined
    assert "chown agent:agent /app/x.txt" in joined
    assert asyncio.run(sb.read_file("/app/x.txt")) == "file-text"


def test_copy_out_tars_a_service_path_rooted_at_slash(monkeypatch):
    sb, host = _boot(monkeypatch)
    assert asyncio.run(sb.copy_out("api", "/audit/decisions.json")) == b"tgz-bytes"
    cp = next(c for c, _ in host.cmds if " docker cp " in c and "tt_cp_" in c)
    assert f"{CID}:/audit/decisions.json" in cp
    assert "tar -czf" in cp and cp.split("tar -czf ")[1].split(" && ")[0].endswith(
        " audit/decisions.json"
    )
    assert host.cmds[-1][0].startswith("rm -rf /tmp/tt_cp_")
    # A path the service does not have is left out, not an error.
    base = host.reply
    host.reply = lambda cmd: (1, "", "no such file") if "tt_cp_" in cmd and "docker cp" in cmd else base(cmd)
    assert asyncio.run(sb.copy_out("api", "/missing")) is None


def test_teardown_runs_compose_down_then_deletes_the_host(monkeypatch):
    sb, host = _boot(monkeypatch)
    asyncio.run(sb.__aexit__(None, None, None))
    assert host.cmds[-1][0].endswith("down --remove-orphans")
    assert host.exited is None


@pytest.mark.parametrize("step", [" --ignore-buildable", " build", " up -d"])
def test_a_failed_compose_step_deletes_the_host(monkeypatch, step):
    class _Failing(_FakeHost):
        def reply(self, cmd):
            return (1, "", "boom") if cmd.endswith(step) else super().reply(cmd)

    monkeypatch.setattr(daytona_mod, "DaytonaSandbox", _Failing)
    sb = compose_mod.DaytonaComposeSandbox(
        docker_compose=COMPOSE, project="p", main_image=PINNED
    )
    with pytest.raises(RuntimeError, match="docker compose .* failed: .*boom"):
        asyncio.run(sb.__aenter__())
    assert _FakeHost.instances[-1].exited is RuntimeError


def test_host_image_carries_the_exec_wrappers_tools():
    # stdbuf (coreutils) and grep -b (GNU grep): without them the exec wrapper
    # captures no output on Alpine, and ps -q returns no container id.
    run = compose_mod.DIND_DOCKERFILE.splitlines()[1]
    assert run.startswith("RUN apk add --no-cache ")
    assert {"bash", "coreutils", "grep"} <= set(run.split()[4:])


def test_uncaptured_host_output_fails_the_boot_naming_the_cause(monkeypatch):
    class _Blind(_FakeHost):
        def reply(self, cmd):
            return 0, daytona_mod._MISSING_OUTPUT_MESSAGE, ""

    monkeypatch.setattr(daytona_mod, "DaytonaSandbox", _Blind)
    sb = compose_mod.DaytonaComposeSandbox(
        docker_compose=COMPOSE, project="p", main_image=PINNED
    )
    with pytest.raises(RuntimeError, match="output is not captured"):
        asyncio.run(sb.__aenter__())
    host = _FakeHost.instances[-1]
    assert host.exited is RuntimeError
    assert not any("docker compose" in c for c, _ in host.cmds)


def test_a_non_container_id_from_ps_is_refused_with_its_output(monkeypatch):
    sb, host = _boot(monkeypatch)
    host.reply = lambda cmd: (0, daytona_mod._MISSING_OUTPUT_MESSAGE, "")
    with pytest.raises(RuntimeError, match="no running container.*unavailable"):
        asyncio.run(sb.exec("true"))


def test_upload_mkdirs_as_root_and_surfaces_docker_cp_errors(monkeypatch):
    sb, host = _boot(monkeypatch)
    base = host.reply

    def reply(cmd):
        if cmd.startswith("docker cp /tmp/tt_wf_"):
            return 1, "Error response from daemon: no space left on device\n", ""
        return base(cmd)

    host.reply = reply
    with pytest.raises(RuntimeError, match="main:/app/x.txt failed.*no space left"):
        asyncio.run(sb.write_file("/app/x.txt", "hi"))
    assert ["docker", "exec", "-u", "0", CID, "mkdir", "-p", "/app"] in [
        shlex.split(c) for c, _ in host.cmds
    ]
    assert host.cmds[-1][0].startswith("rm -f /tmp/tt_wf_")

    host.reply = lambda cmd: (
        (1, "mkdir: permission denied\n", "") if " mkdir -p " in cmd else base(cmd)
    )
    with pytest.raises(RuntimeError, match="mkdir -p /app in main failed"):
        asyncio.run(sb.write_file("/app/x.txt", "hi"))


def test_constructor_needs_a_project_and_a_main():
    with pytest.raises(ValueError, match="docker-compose.yaml"):
        compose_mod.DaytonaComposeSandbox(docker_compose=" ", project="p", main_image=PINNED)
    with pytest.raises(ValueError, match="main's image"):
        compose_mod.DaytonaComposeSandbox(docker_compose=COMPOSE, project="p", main_image="")
    with pytest.raises(ValueError, match="build_context is only meaningful"):
        compose_mod.DaytonaComposeSandbox(
            docker_compose=COMPOSE, project="p", main_image=PINNED, main_build_context={"a": ""}
        )
