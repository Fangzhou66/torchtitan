# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A docker-compose task as one ``Sandbox``: Harbor's Daytona DinD strategy, ported.

Multi-container tasks (11 of Terminal-Bench 4's 66: sidecar databases, mock APIs,
a browser) cannot run as one image. Harbor 0.22.0 runs them on Daytona by booting
a single sandbox from ``docker:*-dind``, starting dockerd in it and running the
task's compose project there; the agent then talks to the ``main`` service only
(``harbor/environments/daytona/environment.py``, ``_DaytonaDinD``). This module
does the same on top of ``DaytonaSandbox`` (its retries, heartbeats, diagnostics
and labels).

The project is the one Harbor runs for a released task, whose every service is a
digest-pinned image:

* compose files in Harbor's order: resources, Harbor's prebuilt ``main``
  template, log mounts, the task's ``docker-compose.yaml``, our runtime
  override, the startup env;
* the same infra env (``PREBUILT_IMAGE_NAME``, ``CPUS``/``MEMORY``, legacy
  log-mount vars), built with Harbor's own helpers;
* ``main`` commands run through a login shell (``bash -lc``), sidecar commands
  through ``sh -c``, files move through ``docker cp`` via the host.

Deliberate differences, all mechanical:

* ``main`` is built from the row's Dockerfile -- ``FROM <pinned image>`` plus
  the tmux layer the Terminus agent drives, as ``prepare_tb2_1_data
  --verify-runtime`` writes it, or a rewritten package's own -- in a context of
  its own holding only that Dockerfile's ``build_context``. A row without one
  builds ``FROM <its image>``, which is the image itself. Either way main's
  ``build:`` in the task's compose file is overridden.
* Images are pulled before ``up`` under the build timeout. Harbor pulls them
  inside ``up -d``, whose 120 s budget a cold DinD host can spend on pulls alone.
* Commands reach a container with ``docker exec <id>`` rather than ``docker
  compose exec``: the same non-tty exec, without the compose CLI's per-call
  startup, which an agent would pay on every command.

The host runs ``docker:28.3.3-dind`` (Harbor's default) plus what our exec
wrapper needs and Alpine lacks: ``bash``, GNU ``coreutils`` (``stdbuf``) and
``grep`` (``grep -b``). Without the last two the wrapper still runs each command
but captures none of its output, so ``docker compose ps -q`` comes back empty and
every exec or ``docker cp`` into main fails with its error lost.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import shlex
import tempfile
import uuid
from pathlib import Path, PurePosixPath

from torchtitan.experiments.rl.harness.sandbox.base import (
    ExecResult,
    FileContent,
    SandboxIssueTracker,
)

logger = logging.getLogger(__name__)

MAIN_SERVICE = "main"
DIND_IMAGE = "docker:28.3.3-dind"
DIND_DOCKERFILE = f"FROM {DIND_IMAGE}\nRUN apk add --no-cache bash coreutils grep\n"
# A round trip through the host's exec at boot: if its output is not captured,
# fail there, naming the cause, rather than later in a docker cp with no error.
_EXEC_PROBE = "__tt_dind_exec_probe__"
_CONTAINER_ID = re.compile(r"[0-9a-f]{12,64}")
_COMPOSE_DIR = "/harbor/compose"
_ENVIRONMENT_DIR = "/harbor/environment"
_MAIN_CONTEXT = f"{_COMPOSE_DIR}/main-context"
# Named apart from anything a build context may carry (a COPY'd "Dockerfile").
_MAIN_DOCKERFILE = ".tmax-main.Dockerfile"
_RUNTIME_OVERRIDE = f"{_COMPOSE_DIR}/docker-compose-runtime.json"
_MOUNTS_COMPOSE = f"{_COMPOSE_DIR}/docker-compose-mounts.json"
# Harbor's agent environment self-binds these into main (trial.py
# _agent_env_mounts): verifier logs, agent logs, the artifacts publish dir.
LOG_MOUNTS = ("/logs/verifier", "/logs/agent", "/logs/artifacts")
_DOCKER_DAEMON_TIMEOUT_SEC = 60
_UP_TIMEOUT_SEC = 120
_MAIN_WAIT_SEC = 60


class DaytonaComposeSandbox:
    """``Sandbox`` over the ``main`` service of a compose project on a DinD host."""

    def __init__(
        self,
        *,
        docker_compose: str,
        project: str,
        main_image: str,
        main_dockerfile: str | None = None,
        main_build_context: dict[str, str] | None = None,
        cpu: int | None = None,
        memory: int | None = None,
        disk_gb: int | None = None,
        build_timeout_sec: int = 600,
        issue_tracker: SandboxIssueTracker | None = None,
        failure_diagnostics_dir: Path | None = None,
        labels: dict[str, str] | None = None,
        target: str | None = None,
    ) -> None:
        if not docker_compose.strip():
            raise ValueError("compose sandbox needs the task's docker-compose.yaml")
        if not main_image and not main_dockerfile:
            raise ValueError("compose sandbox needs main's image or Dockerfile")
        if main_build_context and not main_dockerfile:
            raise ValueError("main_build_context is only meaningful with a Dockerfile")
        self.docker_compose = docker_compose
        self.project = re.sub(r"[^a-z0-9_-]", "-", project.lower())
        self.main_image = main_image
        self.main_dockerfile = main_dockerfile or f"FROM {main_image}\n"
        self.main_build_context = dict(main_build_context or {})
        self.cpu = cpu
        self.memory = memory
        self.build_timeout_sec = int(build_timeout_sec)
        self.issue_tracker = issue_tracker or SandboxIssueTracker()
        self._host_kwargs = dict(
            cpu=cpu,
            memory=memory,
            disk_gb=disk_gb,
            issue_tracker=self.issue_tracker,
            failure_diagnostics_dir=failure_diagnostics_dir,
            labels=labels,
            target=target,
        )
        self._host = None
        self._container_ids: dict[str, str] = {}
        self.sandbox_id = ""
        self.allocated_disk_gb: int | None = None

    # -- compose plumbing (Harbor's file order and env) ----------------------

    def _compose_files(self) -> list[str]:
        from harbor.environments.docker import ENV_COMPOSE_NAME, RESOURCES_COMPOSE_NAME

        return [
            f"{_COMPOSE_DIR}/{RESOURCES_COMPOSE_NAME}",
            f"{_COMPOSE_DIR}/docker-compose-prebuilt.yaml",
            _MOUNTS_COMPOSE,
            f"{_ENVIRONMENT_DIR}/docker-compose.yaml",
            _RUNTIME_OVERRIDE,
            f"{_COMPOSE_DIR}/{ENV_COMPOSE_NAME}",
        ]

    def _main_image_name(self) -> str:
        from harbor.environments.docker.docker import _sanitize_docker_image_name

        return _sanitize_docker_image_name(f"hb__{self.project}")

    def _compose_env(self) -> dict[str, str]:
        from harbor.environments.docker.compose_env import (
            ComposeInfraEnvVars,
            legacy_log_mount_env_vars,
            merge_compose_env,
        )

        infra = ComposeInfraEnvVars(
            main_image_name=self._main_image_name(),
            context_dir=_ENVIRONMENT_DIR,
            prebuilt_image_name=self.main_image or None,
            cpus=self.cpu,
            memory=f"{self.memory * 1024}M" if self.memory else None,
        ).to_env_dict()
        infra.update(legacy_log_mount_env_vars(self._volumes(), host_value="target"))
        return merge_compose_env(user_env={}, infra_env=infra, logger=logger)

    def _volumes(self) -> list[dict]:
        return [{"type": "bind", "source": p, "target": p} for p in LOG_MOUNTS]

    def _runtime_override(self) -> dict:
        """Our one change to the project: main is built from the row's Dockerfile."""
        return {
            "services": {
                MAIN_SERVICE: {
                    "image": self._main_image_name(),
                    "build": {"context": _MAIN_CONTEXT, "dockerfile": _MAIN_DOCKERFILE},
                    "pull_policy": "build",
                }
            }
        }

    def _compose(self, *subcommand: str) -> str:
        flags = [x for f in self._compose_files() for x in ("-f", f)]
        return shlex.join(
            [
                "docker",
                "compose",
                "-p",
                self.project,
                "--project-directory",
                _ENVIRONMENT_DIR,
                *flags,
                *subcommand,
            ]
        )

    async def _host_exec(self, cmd: str, *, timeout: int, env=None) -> ExecResult:
        return await self._host.exec(cmd, user="root", env=env, timeout=timeout)

    async def _compose_exec(self, *subcommand: str, timeout: int) -> ExecResult:
        return await self._host_exec(
            self._compose(*subcommand), timeout=timeout, env=self._compose_env()
        )

    # -- lifecycle -----------------------------------------------------------

    async def __aenter__(self) -> DaytonaComposeSandbox:
        from torchtitan.experiments.rl.harness.sandbox.daytona import DaytonaSandbox

        self._host = DaytonaSandbox("", dockerfile=DIND_DOCKERFILE, **self._host_kwargs)
        await self._host.__aenter__()
        self.sandbox_id = self._host.sandbox_id
        self.allocated_disk_gb = self._host.allocated_disk_gb
        try:
            await self._start()
        except BaseException as error:
            await self._host.__aexit__(type(error), error, error.__traceback__)
            self._host = None
            raise
        return self

    async def _start(self) -> None:
        from harbor.environments.docker import (
            COMPOSE_PREBUILT_PATH,
            ENV_COMPOSE_NAME,
            RESOURCES_COMPOSE_NAME,
            write_env_compose_file,
            write_mounts_compose_file,
            write_resources_compose_file,
        )

        # The DinD image's entrypoint is not run by Daytona; start dockerd as
        # Harbor does, detached so it outlives this exec.
        await self._host_exec(
            "setsid nohup dockerd-entrypoint.sh dockerd > /var/log/dockerd.log 2>&1 "
            "< /dev/null &",
            timeout=30,
        )
        await self._wait_for_docker()
        rc, out, err = await self._host_exec(f"echo {_EXEC_PROBE}", timeout=30)
        if rc != 0 or _EXEC_PROBE not in (out or ""):
            raise RuntimeError(
                "DinD host exec output is not captured (rc="
                f"{rc}, output {((out or '') + (err or ''))[-300:]!r}); the host "
                "image needs bash, coreutils (stdbuf) and GNU grep"
            )

        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp)
            write_resources_compose_file(
                local / RESOURCES_COMPOSE_NAME,
                cpu_request=self.cpu,
                memory_request_mb=self.memory * 1024 if self.memory else None,
            )
            write_mounts_compose_file(local / "mounts.json", self._volumes())
            write_env_compose_file(local / ENV_COMPOSE_NAME, {})
            staged = {
                f"{_COMPOSE_DIR}/{COMPOSE_PREBUILT_PATH.name}": COMPOSE_PREBUILT_PATH,
                f"{_COMPOSE_DIR}/{RESOURCES_COMPOSE_NAME}": local
                / RESOURCES_COMPOSE_NAME,
                _MOUNTS_COMPOSE: local / "mounts.json",
                f"{_COMPOSE_DIR}/{ENV_COMPOSE_NAME}": local / ENV_COMPOSE_NAME,
            }
            for dest, src in staged.items():
                await self._host.write_file(dest, Path(src).read_bytes())
        await self._host.write_file(
            f"{_ENVIRONMENT_DIR}/docker-compose.yaml", self.docker_compose
        )
        for rel, b64 in sorted(self.main_build_context.items()):
            rel_path = PurePosixPath(rel)
            if rel_path.is_absolute() or ".." in rel_path.parts or rel == _MAIN_DOCKERFILE:
                raise ValueError(f"main build context path {rel!r} is not allowed")
            await self._host.write_file(
                f"{_MAIN_CONTEXT}/{rel_path}", base64.b64decode(b64)
            )
        await self._host.write_file(
            f"{_MAIN_CONTEXT}/{_MAIN_DOCKERFILE}", self.main_dockerfile
        )
        await self._host.write_file(
            _RUNTIME_OVERRIDE, json.dumps(self._runtime_override(), indent=2)
        )
        binds = " ".join(shlex.quote(p) for p in LOG_MOUNTS)
        await self._host_exec(f"mkdir -p {binds} && chmod 777 {binds}", timeout=30)

        for step in (("pull", "--ignore-buildable"), ("build",)):
            rc, out, err = await self._compose_exec(
                *step, timeout=self.build_timeout_sec
            )
            if rc != 0:
                raise RuntimeError(
                    f"docker compose {step[0]} failed: {out[-4000:]} {err[-4000:]}"
                )
        rc, out, err = await self._compose_exec("up", "-d", timeout=_UP_TIMEOUT_SEC)
        if rc != 0:
            raise RuntimeError(f"docker compose up failed: {out[-4000:]} {err[-4000:]}")
        await self._wait_for_main()

    async def _wait_for_docker(self) -> None:
        last = ""
        for _ in range(_DOCKER_DAEMON_TIMEOUT_SEC // 2):
            rc, out, err = await self._host_exec("docker info", timeout=10)
            if rc == 0:
                return
            last = (out or "") + (err or "")
            await asyncio.sleep(2)
        raise RuntimeError(
            f"Docker daemon not ready after {_DOCKER_DAEMON_TIMEOUT_SEC}s: {last[-500:]}"
        )

    async def _wait_for_main(self) -> None:
        for _ in range(_MAIN_WAIT_SEC // 2):
            rc, _out, _err = await self._compose_exec(
                "exec", "-T", MAIN_SERVICE, "true", timeout=10
            )
            if rc == 0:
                return
            await asyncio.sleep(2)
        raise RuntimeError(f"main container not running after {_MAIN_WAIT_SEC}s")

    async def __aexit__(self, exc_type, exc, tb) -> None:
        host, self._host = self._host, None
        if host is None:
            return
        try:
            await host.exec(
                self._compose("down", "--remove-orphans"),
                user="root",
                env=self._compose_env(),
                timeout=60,
            )
        except Exception as error:  # noqa: BLE001 -- the sandbox is deleted anyway
            logger.warning("docker compose down failed: %s", error)
        await host.__aexit__(exc_type, exc, tb)

    # -- containers ----------------------------------------------------------

    async def _container(self, service: str) -> str:
        if service not in self._container_ids:
            rc, out, err = await self._compose_exec("ps", "-q", service, timeout=30)
            lines = (out or "").strip().splitlines() if rc == 0 else []
            cid = lines[-1].strip() if lines else ""
            if not _CONTAINER_ID.fullmatch(cid):
                raise RuntimeError(
                    f"no running container for service {service!r} "
                    f"(docker compose ps -q rc={rc}): {(out or '')[-500:]} {err}"
                )
            self._container_ids[service] = cid
        return self._container_ids[service]

    async def service_exec(
        self,
        cmd: str,
        *,
        service: str = MAIN_SERVICE,
        user: str | int | None = None,
        env: dict[str, str] | None = None,
        timeout: int = 120,
    ) -> ExecResult:
        """Run ``cmd`` in a service: ``bash -lc`` in main, ``sh -c`` elsewhere."""
        argv = ["docker", "exec"]
        if user not in (None, "root"):
            argv += ["-u", str(user)]
        for key, value in (env or {}).items():
            argv += ["-e", f"{key}={value}"]
        argv.append(await self._container(service))
        argv += ["bash", "-lc", cmd] if service == MAIN_SERVICE else ["sh", "-c", cmd]
        return await self._host_exec(shlex.join(argv), timeout=timeout)

    async def stop_service(self, service: str) -> None:
        rc, out, err = await self._compose_exec("stop", service, timeout=60)
        if rc != 0:
            raise RuntimeError(f"docker compose stop {service} failed: {out} {err}")

    async def copy_out(self, service: str, path: str) -> bytes | None:
        """``path`` (file or dir) from a service, as a gzipped tar rooted at ``/``.

        The same shape ``grading.collect_artifacts`` makes in a single sandbox,
        so ``restore_artifacts`` unpacks it at the same path. None when the
        service has no such path."""
        staging = f"/tmp/tt_cp_{uuid.uuid4().hex}"
        cid = await self._container(service)
        rel = path.strip("/")
        parent = str(PurePosixPath(rel).parent)
        rc, out, err = await self._host_exec(
            f"mkdir -p {staging}/{shlex.quote(parent)} && docker cp "
            f"{shlex.quote(f'{cid}:/{rel}')} {staging}/{shlex.quote(parent)}/ && "
            f"tar -czf {staging}.tgz -C {staging} {shlex.quote(rel)} && "
            f"base64 -w0 {staging}.tgz > {staging}.b64",
            timeout=600,
        )
        try:
            if rc != 0:
                logger.info("artifact %s:%s not collected: %s %s", service, path, out, err)
                return None
            text = (await self._host.read_file(f"{staging}.b64") or "").strip()
            return base64.b64decode(text) if text else None
        finally:
            await self._host_exec(
                f"rm -rf {staging} {staging}.tgz {staging}.b64", timeout=30
            )

    # -- Sandbox protocol (main service) --------------------------------------

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
    ) -> ExecResult:
        # "root" is the container's own default user, exactly as DaytonaSandbox
        # runs "root" as the image's default user (docker exec without -u).
        rc, out, err = await self.service_exec(
            cmd, service=MAIN_SERVICE, user=user, env=env, timeout=timeout
        )
        if check and rc != 0:
            raise RuntimeError(f"command failed rc={rc}: {cmd[:200]!r}: {err[-500:]}")
        return rc, out, err

    async def write_file(
        self, sandbox_path: str, content: FileContent, *, user: str = "root"
    ) -> None:
        staging = f"/tmp/tt_wf_{uuid.uuid4().hex}"
        await self._host.write_file(staging, content)
        cid = await self._container(MAIN_SERVICE)
        parent = str(PurePosixPath(sandbox_path).parent)
        try:
            # As root, like the docker cp that follows (and DaytonaSandbox's own
            # write_file), whatever user the image runs as.
            rc, out, err = await self._host_exec(
                shlex.join(["docker", "exec", "-u", "0", cid, "mkdir", "-p", parent]),
                timeout=60,
            )
            if rc != 0:
                raise RuntimeError(
                    f"mkdir -p {parent} in main failed (rc={rc}): {out} {err}"
                )
            rc, out, err = await self._host_exec(
                f"docker cp {staging} {shlex.quote(f'{cid}:{sandbox_path}')}",
                timeout=600,
            )
            if rc != 0:
                raise RuntimeError(
                    f"docker cp into main:{sandbox_path} failed (rc={rc}): {out} {err}"
                )
            if user and user != "root":
                await self._host_exec(
                    shlex.join(
                        [
                            "docker",
                            "exec",
                            "-u",
                            "0",
                            cid,
                            "chown",
                            f"{user}:",  # its login group (nobody's is nogroup)
                            sandbox_path,
                        ]
                    ),
                    timeout=60,
                )
        finally:
            await self._host_exec(f"rm -f {staging}", timeout=30)

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        staging = f"/tmp/tt_rf_{uuid.uuid4().hex}"
        cid = await self._container(MAIN_SERVICE)
        try:
            rc, _out, _err = await self._host_exec(
                f"docker cp {shlex.quote(f'{cid}:{sandbox_path}')} {staging}",
                timeout=600,
            )
            if rc != 0:
                return ""
            return await self._host.read_file(staging)
        finally:
            await self._host_exec(f"rm -f {staging}", timeout=30)
