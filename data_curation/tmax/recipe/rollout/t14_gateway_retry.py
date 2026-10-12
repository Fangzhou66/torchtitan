"""Call-level retry for transient gateway failures (t14 line, 2026-10-03). Active only when T14_GATEWAY_RETRY=1.

Why: between 13:10Z and 15:20Z the huanzhimao gateway had short windows (a few minutes) in which Cloudflare returned
502/520/524 or dropped the stream mid-body ("peer closed connection"). The adapter (api_rollout_provider.APIAdapter)
raises on the first such response, so ONE failed call ended a whole multi-minute episode (outcome error, a retry of the
task spent); at 15:15Z 111 of 115 finishing attempts died that way. The limiter wrapper retries only 429s.

What: when the provider module is imported, wrap APIAdapter._post_stream so that a 5xx status or an httpx transport
error re-sends the IDENTICAL payload after 15/30/60/120 s (+-20% jitter); a 4xx (incl. 429, which the limiter owns)
and every other error propagate unchanged. The episode sees only the successful response, exactly as if the first call
had succeeded; each failure is still in provider_errors.jsonl (the original writes it) and each retry is journalled in
the attempt's gateway_retries.jsonl. Nothing on disk in any shared tool changes: this module is loaded by a .pth file in
the t14 venv (envs/rollout_env_t14), and does nothing unless the env var is set.
"""
from __future__ import annotations

import asyncio
import importlib.abc
import importlib.util
import os
import random
import re
import sys
import time

DELAYS = (15, 30, 60, 120)
_STATUS = re.compile(r"HTTP (\d{3})$")
TARGET = "api_rollout_provider"


def retryable(exc: BaseException) -> str:
    """'' if not retryable, else a short reason."""
    if isinstance(exc, RuntimeError):
        m = _STATUS.search(str(exc))
        if m and m.group(1).startswith("5"):
            return f"http_{m.group(1)}"
        return ""
    try:
        import httpx
    except ImportError:
        return ""
    if isinstance(exc, httpx.TransportError):
        return type(exc).__name__
    return ""


def wrap(original, delays=DELAYS, sleep=asyncio.sleep):
    async def _post_stream(self, payload):
        for i in range(len(delays) + 1):
            try:
                return await original(self, payload)
            except Exception as exc:  # noqa: BLE001
                why = retryable(exc)
                if not why or i == len(delays):
                    raise
                delay = delays[i] * random.uniform(0.8, 1.2)
                journal = getattr(self, "journal", None)
                if journal is not None:
                    journal.append("gateway_retries.jsonl", {"retry": i + 1, "reason": why, "sleep_sec": round(delay, 1),
                                                             "time": time.time()})
                await sleep(delay)
    _post_stream._t14_retry = True
    return _post_stream


def override_top_p(adapter, payload):
    """T14_TOP_P (2026-10-05): inferact's Kimi K3 rejects every request whose top_p is not 0.95 (HTTP 400 "Kimi request
    field top_p must be one of: 0.95"); the shared sampling config sends 1.0 and has no knob. When the env var is set,
    the payload is sent with that top_p (a copy; the caller's dict is untouched) and the attempt's
    sampling_overrides.jsonl records the first override (field, sent, configured). Unset: payload unchanged."""
    v = os.environ.get("T14_TOP_P", "")
    if not v or not isinstance(payload, dict) or "top_p" not in payload:
        return payload
    sent = float(v)
    if payload["top_p"] == sent:
        return payload
    journal = getattr(adapter, "journal", None)
    if journal is not None and not getattr(adapter, "_t14_top_p_logged", False):
        journal.append("sampling_overrides.jsonl", {"field": "top_p", "sent": sent, "configured": payload["top_p"],
                                                    "time": time.time()})
        adapter._t14_top_p_logged = True
    return {**payload, "top_p": sent}


def wrap_nonstream(original, delays=DELAYS, sleep=asyncio.sleep):
    """_post for NON-streamed payloads (2026-10-05: Kimi K3 on inferact is called without streaming, which the
    _post_stream wrap never sees). A streamed payload passes straight through: _post hands it to the wrapped
    _post_stream, so it is never retried twice."""
    inner = wrap(original, delays, sleep)

    async def _post(self, payload):
        payload = override_top_p(self, payload)
        if isinstance(payload, dict) and payload.get("stream"):
            return await original(self, payload)
        return await inner(self, payload)
    _post._t14_retry = True
    return _post


def patch(module) -> None:
    cls = getattr(module, "APIAdapter", None)
    if cls is None or getattr(cls._post_stream, "_t14_retry", False):
        return
    cls._post_stream = wrap(cls._post_stream)
    cls._post = wrap_nonstream(cls._post)
    print(f"[t14_gateway_retry] APIAdapter._post_stream and non-streamed _post wrapped: retry 5xx/transport errors after {DELAYS} s",
          file=sys.stderr, flush=True)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.rsplit(".", 1)[-1] != TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader
        orig_exec = loader.exec_module

        def exec_module(module):
            orig_exec(module)
            patch(module)
        loader.exec_module = exec_module
        return spec


def install() -> None:
    if os.environ.get("T14_GATEWAY_RETRY") == "1" and not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
