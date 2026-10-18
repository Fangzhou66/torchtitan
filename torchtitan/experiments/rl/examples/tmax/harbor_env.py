# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The parts of a Harbor task's environment a Terminal-Bench 4 row carries.

``prepare_tb2_1_data`` writes these row fields only for a task that declares
them, and every function here returns at once (or returns its input) for a row
without them, so no other row boots or runs any differently:

* ``daytona_gpu`` / ``daytona_gpu_type`` / ``daytona_target``: a GPU sandbox in
  the region that has the GPU quota (and the same three on
  ``tmax.separate_verifier`` for its verifier);
* ``compose``: the task's released ``docker-compose.yaml``, run as a
  ``DaytonaComposeSandbox`` whose ``main`` is the row's image;
* ``healthcheck``: Harbor's ``[environment].healthcheck``, run before the agent;
* ``agent_user``: the non-root user the image declares; the agent runs as it
  while the harness keeps root, as in Harbor.

Imported by the rollouter and by ``evolution/daytona_revalidate.py`` alike, so
the probe boots and enters a row exactly as a training rollout does.
"""

from __future__ import annotations

import asyncio
import shlex
import time

# Row field -> DaytonaSandbox keyword. The same names sit on the row's metadata
# (agent environment) and on its tmax.separate_verifier block (verifier).
_SANDBOX_OPTIONS = (
    ("daytona_gpu", "gpu"),
    ("daytona_gpu_type", "gpu_type"),
    ("daytona_target", "target"),
    ("network_block_all", "network_block_all"),
)

# Harbor's agent-side mount targets (trial.py _agent_env_mounts), which its
# environments create world-writable (mkdir -p + chmod 777) before the agent
# starts, so an unprivileged agent can write its logs and artifacts.
AGENT_LOG_DIRS = ("/logs/agent", "/logs/verifier", "/logs/artifacts")


def sandbox_options(fields: dict) -> dict:
    """``boot_agent_sandbox(sandbox_options=...)`` from a row's fields; {} if none."""
    return {key: fields[name] for name, key in _SANDBOX_OPTIONS if fields.get(name)}


def boot_kwargs(md: dict) -> dict:
    """The extra ``boot_agent_sandbox`` keywords a row's metadata asks for."""
    out: dict = {}
    options = sandbox_options(md)
    if options:
        out["sandbox_options"] = options
    compose = md.get("compose")
    if compose:
        out["compose"] = {
            "docker_compose": compose["docker_compose"],
            "project": md.get("instance_id") or "task",
            "build_timeout_sec": int(compose.get("build_timeout_sec") or 600),
        }
    return out


async def run_healthcheck(sb, healthcheck: dict | None) -> None:
    """Harbor's ``BaseEnvironment.run_healthcheck``: Docker HEALTHCHECK semantics.

    Failures inside the start period do not count; after it, ``retries``
    consecutive failures raise. A row without a healthcheck returns at once.
    """
    if not healthcheck:
        return
    start_period_end = time.monotonic() + float(healthcheck.get("start_period_sec", 0))
    retries = int(healthcheck.get("retries", 3))
    failures = 0
    while True:
        in_start = time.monotonic() < start_period_end
        rc, _out, _err = await sb.exec(
            healthcheck["command"],
            user="root",
            check=False,
            timeout=max(1, int(float(healthcheck.get("timeout_sec", 30)))),
        )
        if rc == 0:
            return
        if in_start:
            await asyncio.sleep(float(healthcheck.get("start_interval_sec", 5)))
            continue
        failures += 1
        if failures >= retries:
            raise RuntimeError(
                f"healthcheck failed after {retries} consecutive retries: "
                f"{healthcheck['command'][:200]!r}"
            )
        await asyncio.sleep(float(healthcheck.get("interval_sec", 5)))


class AgentUserSandbox:
    """The agent's view of a sandbox whose image declares a non-root user.

    Harbor runs the agent as that user and everything else (uploads, setup,
    grading) as root. Our agents ask for ``user="root"``, which on every other
    row means the image's default user, so this maps that one request to the
    agent user, with the ``HOME``/``USER``/``LOGNAME`` a login as it would set.
    An explicit other user passes through; reads stay root.
    """

    def __init__(self, inner, user: str, home: str = "") -> None:
        self._inner = inner
        self.user = user
        self._env = {"USER": user, "LOGNAME": user, **({"HOME": home} if home else {})}

    def _as(self, user: str | None) -> str:
        return self.user if user in (None, "root") else user

    async def exec(self, cmd: str, *, user: str = "root", env=None, **kwargs):
        as_user = self._as(user)
        if as_user == self.user:
            env = {**self._env, **(env or {})}
        return await self._inner.exec(cmd, user=as_user, env=env, **kwargs)

    async def write_file(self, sandbox_path: str, content, *, user: str = "root"):
        return await self._inner.write_file(sandbox_path, content, user=self._as(user))

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        return await self._inner.read_file(sandbox_path, user=user)

    def __getattr__(self, name):
        return getattr(self._inner, name)


async def enter_as_agent_user(sb, agent_user: str):
    """``sb`` itself for a row without ``agent_user``. Otherwise Harbor's log
    dirs are made world-writable (as root) and the agent's view comes back."""
    if not agent_user:
        return sb
    dirs = " ".join(AGENT_LOG_DIRS)
    rc, out, err = await sb.exec(
        f"mkdir -p {dirs} && chmod 777 {dirs}", user="root", check=False, timeout=60
    )
    if rc != 0:
        raise RuntimeError(
            f"creating {dirs} as root failed (rc={rc}): {(out or '')[-300:]} "
            f"{(err or '')[-300:]}"
        )
    _rc, home, _ = await sb.exec(
        f"getent passwd {shlex.quote(agent_user)} | cut -d: -f6",
        user="root",
        check=False,
        timeout=60,
    )
    lines = (home or "").strip().splitlines()
    return AgentUserSandbox(sb, agent_user, lines[-1].strip() if lines else "")
