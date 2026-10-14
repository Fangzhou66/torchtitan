# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run-scoped student-policy checks before candidate tasks enter the live mix.

The evolution sidecar writes self-contained task rows; the training controller
scores both rows on fresh sandboxes with the training generators. Requests and
responses are content-bound, so a prior run or revision cannot admit a candidate.
Neither these trajectories nor their measurements enter the optimizer or TB eval.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from torchtitan.experiments.rl.examples.tmax import layout


def content_hash(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


@dataclass(frozen=True, kw_only=True, slots=True)
class PolicyCanaryConfig:
    group_size: int = 0
    max_concurrency: int = 4
    max_version_span: int = 5
    timeout_sec: int = 7200
    poll_sec: int = 10

    def __post_init__(self) -> None:
        if self.group_size != 0 and self.group_size < 2:
            raise ValueError("SWE_POLICY_CANARY_K must be 0 (disabled) or at least 2")
        if self.max_concurrency < 1 or self.timeout_sec < 1 or self.poll_sec < 1:
            raise ValueError(
                "policy canary concurrency, timeout and poll must be positive"
            )
        if self.max_version_span < 0:
            raise ValueError("SWE_POLICY_CANARY_MAX_VERSION_SPAN must be nonnegative")

    @classmethod
    def from_env(cls) -> PolicyCanaryConfig:
        return cls(
            group_size=int(os.environ.get("SWE_POLICY_CANARY_K", "0")),
            max_concurrency=int(
                os.environ.get("SWE_POLICY_CANARY_MAX_CONCURRENCY", "4")
            ),
            max_version_span=int(
                os.environ.get("SWE_POLICY_CANARY_MAX_VERSION_SPAN", "5")
            ),
            timeout_sec=int(os.environ.get("SWE_POLICY_CANARY_TIMEOUT_SEC", "7200")),
            poll_sec=int(os.environ.get("SWE_POLICY_CANARY_POLL_SEC", "10")),
        )


def summarize_group(group, *, group_size: int) -> dict:
    """Preserve each verdict and policy span; infrastructure is never a failed solve."""
    trials = []
    versions = []
    for rollout in group.rollouts:
        reward = rollout.reward
        scored = (
            reward is not None
            and math.isfinite(reward)
            and not rollout.diagnostics.get("infra_failed", False)
        )
        spans = [
            (turn.min_policy_version, turn.max_policy_version)
            for turn in rollout.turns
            if turn.min_policy_version is not None
            and turn.max_policy_version is not None
        ]
        versions.extend(version for span in spans for version in span)
        trials.append(
            {
                "rollout_id": rollout.rollout_id,
                "reward": float(reward) if scored else None,
                "scored": scored,
                "infra_failed": bool(rollout.diagnostics.get("infra_failed", False)),
                "policy_version_min": min((span[0] for span in spans), default=None),
                "policy_version_max": max((span[1] for span in spans), default=None),
            }
        )
    return {
        "group_id": group.group_id,
        "requested": group_size,
        "num_scored": sum(trial["scored"] for trial in trials),
        "num_solved": sum(trial["scored"] and trial["reward"] > 0 for trial in trials),
        "policy_version_min": min(versions, default=None),
        "policy_version_max": max(versions, default=None),
        "trials": trials,
    }


def admission(request: dict, response: dict, *, max_version_span: int) -> dict:
    """Admit useful adjustments, but measure repairs without calling them simplifications."""
    if response.get("request_sha256") != content_hash(request):
        raise ValueError("policy canary response does not match the request")
    if response.get("status") != "complete":
        return {
            "ok": False,
            "stage": "policy_canary_error",
            "why": response.get("error", "probe failed"),
        }
    before, after = response["before"], response["after"]
    k = request["group_size"]
    for group in (before, after):
        trials = group["trials"]
        if len(trials) != k or any(
            not trial["scored"]
            or trial["infra_failed"]
            or trial["reward"] is None
            or not math.isfinite(trial["reward"])
            or trial["policy_version_min"] is None
            or trial["policy_version_max"] is None
            for trial in trials
        ):
            return {
                "ok": False,
                "stage": "invalid_signal",
                "why": "policy canary has incomplete or unscored trials",
            }
    minima = [group["policy_version_min"] for group in (before, after)]
    maxima = [group["policy_version_max"] for group in (before, after)]
    if any(version is None for version in minima + maxima):
        return {
            "ok": False,
            "stage": "invalid_signal",
            "why": "policy canary has no generated policy versions",
        }
    if max(maxima) - min(minima) > max_version_span:
        return {
            "ok": False,
            "stage": "invalid_signal",
            "why": "policy changed too far during the paired canary",
        }
    old = sum(trial["reward"] > 0 for trial in before["trials"])
    new = sum(trial["reward"] > 0 for trial in after["trials"])
    if request["route"] == "repair_spec":
        # Semantic validation, not a success-rate target, establishes a repair.
        # The repaired revision must be measured afresh before a difficulty edit.
        return {
            "ok": True,
            "stage": "policy_repair_measured",
            "why": f"repair measured: {old}/{k} -> {new}/{k}",
        }
    if (request["route"] == "simplify" and old != 0) or (
        request["route"] == "evolve" and old / k < request["harder_ratio"]
    ):
        return {
            "ok": False,
            "stage": "invalid_signal",
            "why": f"fresh original no longer triggers {request['route']}: {old}/{k}",
        }
    if new == 0 or new == k:
        return {
            "ok": False,
            "stage": "policy_too_hard" if new == 0 else "policy_too_easy",
            "why": f"candidate remains outside the useful band: {new}/{k}",
        }
    return {
        "ok": True,
        "stage": "policy_canary_pass",
        "why": f"paired student trials: {old}/{k} -> {new}/{k}",
    }


class PolicyCanaryQueue:
    def __init__(self, root: layout.Root) -> None:
        self.path = root.evolution.path / "policy_canaries"
        self.requests = self.path / "requests"
        self.responses = self.path / "responses"

    def submit(
        self,
        *,
        rewrite: str,
        task: str,
        input_rev: int,
        route: str,
        original: dict,
        candidate: dict,
        config: PolicyCanaryConfig,
        expires_at: float,
    ) -> tuple[str, dict]:
        if route not in ("simplify", "evolve", "repair_spec"):
            raise ValueError(f"unknown policy canary route: {route}")
        if config.group_size < 2:
            raise ValueError("cannot submit a disabled policy canary")
        request = {
            "schema_version": 1,
            "rewrite": rewrite,
            "task": task,
            "input_rev": input_rev,
            "route": route,
            "group_size": config.group_size,
            "harder_ratio": float(os.environ.get("SWE_EVOLUTION_HARDER_RATIO", "1")),
            "original": original,
            "candidate": candidate,
            "created_at": time.time(),
            "expires_at": min(expires_at, time.time() + config.timeout_sec),
        }
        identity = content_hash(request)
        layout.write_json_atomic(self.requests / f"{identity}.json", request)
        return identity, request

    def wait(self, identity: str, request: dict, *, config: PolicyCanaryConfig) -> dict:
        response_path = self.responses / f"{identity}.json"
        while time.time() < request["expires_at"]:
            if response_path.is_file():
                response = json.loads(response_path.read_text())
                if time.time() >= request["expires_at"]:
                    break
                verdict = admission(
                    request, response, max_version_span=config.max_version_span
                )
                return {
                    **verdict,
                    "request_id": identity,
                    "expires_at": request["expires_at"],
                    "response": response,
                    "candidate_sha256": content_hash(request["candidate"]),
                }
            time.sleep(
                min(config.poll_sec, max(0, request["expires_at"] - time.time()))
            )
        return {
            "ok": False,
            "stage": "policy_canary_timeout",
            "why": "no current-policy result before the rewrite deadline",
            "request_id": identity,
        }


class PolicyCanaryService:
    """Bound controller-side probes and cancel all of them when optimization ends."""

    def __init__(
        self,
        queue: PolicyCanaryQueue,
        *,
        config: PolicyCanaryConfig,
        evaluate: Callable[[dict, str], Awaitable[dict]],
    ) -> None:
        self.queue = queue
        self.config = config
        self.evaluate = evaluate
        self.active_count = 0
        self.waiting_count = 0

    async def _evaluate(self, path: Path) -> None:
        identity = path.stem
        response: dict = {"request_sha256": identity, "status": "error"}
        try:
            request = await asyncio.to_thread(lambda: json.loads(path.read_text()))
            if content_hash(request) != identity:
                raise ValueError(f"policy canary request hash mismatch: {path}")
            if (
                request["schema_version"] != 1
                or request["group_size"] != self.config.group_size
            ):
                raise ValueError(
                    "policy canary request does not match the controller configuration"
                )
            remaining = request["expires_at"] - time.time()
            if remaining <= 0:
                raise TimeoutError("policy canary request expired before evaluation")
            result = await asyncio.wait_for(self.evaluate(request, identity), remaining)
            response.update(result, status="complete")
        except Exception as error:  # noqa: BLE001 -- reject the candidate, keep training
            response["error"] = f"{type(error).__name__}: {error}"[:300]
        response["finished_at"] = time.time()
        await asyncio.to_thread(
            layout.write_json_atomic, self.queue.responses / path.name, response
        )

    async def run(self) -> None:
        active: dict[Path, asyncio.Task] = {}
        try:
            while True:

                def pending() -> list[Path]:
                    paths = [
                        path
                        for path in self.queue.requests.glob("*.json")
                        if path not in active
                        and not (self.queue.responses / path.name).is_file()
                    ]
                    # Hash filenames are random order; FIFO avoids starving old probes.
                    return sorted(
                        paths, key=lambda path: (path.stat().st_mtime_ns, path.name)
                    )

                paths = await asyncio.to_thread(pending)
                admitted = paths[: self.config.max_concurrency - len(active)]
                for path in admitted:
                    active[path] = asyncio.create_task(self._evaluate(path))
                self.active_count = len(active)
                self.waiting_count = len(paths) - len(admitted)
                if active:
                    done, _ = await asyncio.wait(
                        active.values(),
                        timeout=self.config.poll_sec,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for path, task in list(active.items()):
                        if task in done:
                            await task
                            del active[path]
                    self.active_count = len(active)
                else:
                    await asyncio.sleep(self.config.poll_sec)
        finally:
            for task in active.values():
                task.cancel()
            await asyncio.gather(*active.values(), return_exceptions=True)
            self.active_count = 0
