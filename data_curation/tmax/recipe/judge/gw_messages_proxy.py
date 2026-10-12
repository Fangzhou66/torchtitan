#!/usr/bin/env python3
"""Loopback shim: Anthropic /v1/messages in -> OpenAI /v1/chat/completions out, for the huanzhimao gateway.

Why this exists. The gateway (https://gateway.huanzhimao.com/v1, key mdpsk.txt) serves `deepseek-flash`
(DeepSeek-V4.1-Flash, GATEWAY_HUANZHIMAO_PROBE.md §9) only on the OpenAI Chat protocol; its /v1/messages returns
501 protocol_not_configured. The judge is `claude -p`, which speaks Anthropic Messages. So this shim TRANSLATES,
where or_pin_proxy.py only passes through. Everything else is or_pin_proxy.py's contract, unchanged, so the
launcher, the watchdog (abandon_doomed.py) and the stop procedure work as they do there:
  * /__stats, /__caps, /__tasks with the same shapes; capture them before stopping;
  * a per-task and a per-run cost cap, refused with 402 at admission; retries of 429/5xx while nothing has reached the
    client; a mid-stream failure ends the body instead of hanging the worker;
  * /__tasks also carries `last_byte` per task: the time the shim last forwarded translated stream output to that
    task's worker, touched at most once a second. abandon_doomed.py --stall-from stream measures its stall from
    this, so a max-effort turn that streams for ten minutes is live work, not a stall (the 2026-09-24 pilot lost
    all 7 of its abandoned turns that way, each still streaming at 605-630 s);
  * a failure WRITING to the worker (it was killed or hung up) is a client abort, logged as status 499 and counted
    in `client_aborts`; `mid_stream_aborts` and `upstream_errors` count the gateway's side only. The rate rule
    reads the latter, and a watchdog kill is not the provider's fault;
  * --token-cap N --token-since ISO: a ceiling on provider-counted tokens (prompt cache miss + hit + completion, from
    the gateway's usage fields) summed over the day ledger from that instant plus this process's calls. At the
    ceiling every new request is refused with 402 and /__caps reports halted, so the watchdog stops the launcher
    cleanly. A stream killed before its usage chunk is not counted (the gateway reported none);
  * a 429 whose body says quota_exceeded is not retried: it halts the shim at once the same way (the key is out,
    and no retry recovers that).
  * a failure BEFORE the first body byte -- an error status (Cloudflare's 524 when the first token takes past its
    125 s origin limit, 5xx, a non-quota 429), a connection error, a timeout, or a 2xx whose body ends or breaks
    before a byte -- is retried here, transparently, up to PRESTREAM_RETRIES times (backoff PRESTREAM_BACKOFF),
    before the worker sees an error; nothing has reached the worker, so nothing is duplicated. Once a body byte has
    arrived the request is never retried by the shim (the worker's own retry, or the task-level retry, covers it).
    Each such retry is logged: "pre-stream failure (<status or error>) after <s>s; shim retry i/N in <w>s";
  * SIGUSR1 drains: stop accepting, close the port (a successor can bind it at once), finish every request in
    flight, then exit. A deploy then loses nothing.
  * the task is named by the row path in the body (TASK_RE); no message content is ever logged.
Added for this key: a per-UTC-day ledger (the gateway has no usage endpoint, so this file is the record) with a
day cap, and costs derived from a price table because the gateway publishes no prices.

Measured facts this translation rests on (2026-09-24, direct calls):
  * the gateway needs a User-Agent that is not Python-urllib's (Cloudflare 1010 otherwise);
  * tools / tool_calls / role:tool round-trip; ids look like call_00_...;
  * in thinking mode DeepSeek REFUSES a tool-loop follow-up unless each assistant message carries its
    reasoning_content back (400 "The reasoning_content in the thinking mode must be passed back") -- so thinking
    blocks are emitted to the client and mapped back to reasoning_content on the next request;
  * reasoning_effort accepts none | minimal | low | medium | high | xhigh | ultra | max (422 otherwise);
  * streaming: delta.reasoning_content, delta.content, delta.tool_calls, usage in the last chunk with
    stream_options.include_usage; usage carries prompt_cache_hit_tokens / prompt_cache_miss_tokens and
    completion_tokens_details.reasoning_tokens.

--fill-read-stubs (opt-in, default off; the Kimi K3 line, 2026-10-02): Claude Code answers a Read of a file unchanged
since an earlier Read with the stub "Wasted call — file unchanged since your last Read. Refer to that earlier
tool_result instead." Kimi K3 reads that as an interrupted read, re-reads, and ends on a thinking-only turn with no
row. With the flag, before translation, each such stub tool_result is replaced by the content of the most recent
earlier non-stub, non-error Read result in the same request for the same (file_path, offset, limit) -- absent fields
match only absent fields; with no such earlier result the stub is left as is. Counted per request in the fields log
line (read_stubs=filled/stubs) and in /__stats (read_stub_filled, read_stub_unmatched).

--unique-tool-ids (opt-in, default off; the Kimi K3 line, 2026-10-02): Kimi K3 on inferact numbers its tool calls per
turn (Read_0, Read_1, Bash_0, ...), so ids repeat across turns (8,566 tool calls in 120 sessions used 672 distinct
ids). Claude Code drops a tool_use/tool_result pair whose id it has already seen when it builds the next request, so
the model's own reads vanish from its history; it re-reads, gets the file-unchanged stub, and ends with no row. With
the flag each tool_use id handed to Claude Code is "<upstream id>_<12 hex>" (unique); it goes back upstream as the
tool_call_id unchanged.

Key handling: $GATEWAY_API_KEY in this process only; never logged, never written, never on a command line.

    export GATEWAY_API_KEY="$(cat /home/zekaili/fangzhou/tb_check/mdpsk.txt)"
    python3 gw_messages_proxy.py --port 8799 --pid-file run.pid --day-ledger final/gateway_day_ledger.jsonl
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import signal
import sys
import threading
import uuid
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = "https://gateway.huanzhimao.com/v1/chat/completions"
USER_AGENT = "v13-harbor-gw-shim/1.0"
GATEWAY_MODEL = "deepseek-flash"
KEY = ""
UPSTREAM_READ_TIMEOUT = 1800
PRESTREAM_RETRIES = 3
PRESTREAM_BACKOFF = (2, 5, 10)
ACTIVE = 0                     # requests in progress (the drain waits for 0)
DRAINING = False
TASK_RE = re.compile(r"/rows/([A-Za-z0-9_.-]+)\.jsonl")
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "ultra", "max")
FORCE_EFFORT = None            # set by --force-effort: overrides whatever the client sent
FILL_READ_STUBS = False        # set by --fill-read-stubs (see the docstring)
READ_STUB = "Wasted call — file unchanged since your last Read."
UNIQUE_TOOL_IDS = False       # set by --unique-tool-ids (see the docstring)
STUB_DUMP_DIR = None           # set by --dump-unmatched-stubs: content-free skeletons of requests with an unmatched stub
STUB_DUMPS = 0
# Price per MILLION tokens. The gateway publishes none: DeepSeek's own list price for V4.1-Flash, as OpenRouter lists
# its DeepSeek endpoint on 2026-09-24 (full rate; DeepSeek's off-peak discount windows are ignored, so this errs
# high). Every dollar this shim reports is therefore an ESTIMATE; the token counts are exact.
PRICE = {"cache_miss": 0.30, "cache_hit": 0.006, "output": 1.20,
         "source": "DeepSeek list price for deepseek-v4.1-flash via openrouter.ai/api/v1/models/deepseek/"
                   "deepseek-v4.1-flash/endpoints (DeepSeek endpoint, full rate), read 2026-09-24T07:19Z"}
TASKS = {}
TASK_COST_CAP = 2.00
RUN_COST_CAP = 120.00
DAY_CAP = 480.00               # the gateway key's $500 per UTC day minus a $20 reserve
DAY_LEDGER = None
DAY_SPENT = {}                 # utc day -> $ from the ledger file plus this process
TOKEN_CAP = 0                  # 0 = no ceiling; else provider-counted tokens since TOKEN_SINCE
TOKEN_SINCE = 0.0
TOKENS_SINCE = 0               # ledger lines with t >= TOKEN_SINCE, plus this process
HALT_REASON = None             # set once: quota_exceeded, or the token ceiling
REFUSALS = []
LOCK = threading.Lock()
LOG_FIELDS = False
STATS = {"requests": 0, "translated": 0, "upstream_errors": 0, "by_status": {}, "cost_is_estimate": True}


def log(msg: str) -> None:
    with LOCK:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def utc_day(t: float | None = None) -> str:
    return _dt.datetime.fromtimestamp(t or time.time(), _dt.timezone.utc).strftime("%Y-%m-%d")


# --------------------------------------------------------------------------------------------- request translation
def _text_of(content) -> str:
    """The text of an Anthropic content value (string, or list of blocks); non-text blocks become markers."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for b in content:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text":
            parts.append(b.get("text", ""))
        elif b.get("type") == "image":
            parts.append("[image omitted by the gateway shim]")
    return "\n".join(parts)


def fill_read_stubs(payload: dict) -> tuple[dict, int, int]:
    """(payload with each file-unchanged Read stub replaced by the most recent earlier Read result for the same
    (file_path, offset, limit), number filled, number left as stubs). The input is not modified: changed messages and
    blocks are copies, the rest are shared."""
    key_of, last, filled, unmatched, out = {}, {}, 0, 0, []
    for m in payload.get("messages") or []:
        content = m.get("content")
        if not isinstance(content, list):
            out.append(m)
            continue
        if m.get("role") == "assistant":
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") == "Read":
                    inp = b.get("input") or {}
                    key_of[b.get("id")] = (inp.get("file_path"), inp.get("offset"), inp.get("limit"))
            out.append(m)
            continue
        new, changed = [], False
        for b in content:
            k = key_of.get(b.get("tool_use_id")) if isinstance(b, dict) and b.get("type") == "tool_result" else None
            if k is not None:
                txt = _text_of(b.get("content"))
                if READ_STUB in txt and len(txt) < 4000:     # the stub, possibly with a reminder appended
                    if k in last:
                        b = dict(b, content=last[k])
                        filled, changed = filled + 1, True
                    else:
                        unmatched += 1
                elif not b.get("is_error"):
                    last[k] = b.get("content")
            new.append(b)
        out.append(dict(m, content=new) if changed else m)
    return dict(payload, messages=out), filled, unmatched


def stub_skeleton(payload: dict) -> list:
    """The shape of a request's messages without content: roles, block types, Read inputs (path/offset/limit),
    tool_result ids, lengths, is_error, and whether each is the file-unchanged stub."""
    out = []
    for m in payload.get("messages") or []:
        c = m.get("content")
        if not isinstance(c, list):
            out.append({"role": m.get("role"), "str_len": len(c or "")})
            continue
        blocks = []
        for b in c:
            t = b.get("type") if isinstance(b, dict) else type(b).__name__
            if t == "tool_use":
                inp = b.get("input") or {}
                blocks.append({"t": t, "id": b.get("id"), "name": b.get("name"), "keys": sorted(inp),
                               **({k: inp.get(k) for k in ("file_path", "offset", "limit")} if b.get("name") == "Read" else {})})
            elif t == "tool_result":
                txt = _text_of(b.get("content"))
                blocks.append({"t": t, "id": b.get("tool_use_id"), "len": len(txt), "err": bool(b.get("is_error")),
                               "stub": READ_STUB in txt, "ctype": type(b.get("content")).__name__})
            else:
                blocks.append({"t": t, "len": len(json.dumps(b))})
        out.append({"role": m.get("role"), "blocks": blocks})
    return out


def tool_id(upstream_id: str | None, fallback: str) -> str:
    """The tool_use id handed to the client: the upstream's, made unique under --unique-tool-ids."""
    base = upstream_id or fallback
    return f"{base}_{uuid.uuid4().hex[:12]}" if UNIQUE_TOOL_IDS else base


def map_effort(payload: dict) -> str | None:
    if FORCE_EFFORT:
        return FORCE_EFFORT
    th = payload.get("thinking") or {}
    if isinstance(th, dict) and th.get("type") == "disabled":
        return "none"
    eff = (payload.get("output_config") or {}).get("effort")
    return eff if eff in EFFORTS else None


def anthropic_to_openai(payload: dict, model: str = GATEWAY_MODEL) -> dict:
    """Anthropic Messages request -> OpenAI Chat request (the gateway's dialect)."""
    msgs = []
    system = payload.get("system")
    if system:
        msgs.append({"role": "system", "content": _text_of(system)})
    for m in payload.get("messages") or []:
        role, content = m.get("role"), m.get("content")
        if role == "user":
            if isinstance(content, str):
                msgs.append({"role": "user", "content": content})
                continue
            texts = []
            for b in content or []:
                if b.get("type") == "tool_result":
                    txt = _text_of(b.get("content"))
                    if b.get("is_error"):
                        txt = "[tool error] " + txt
                    msgs.append({"role": "tool", "tool_call_id": b.get("tool_use_id"), "content": txt})
                elif b.get("type") in ("text", "image"):
                    texts.append(_text_of([b]))
            if texts:
                msgs.append({"role": "user", "content": "\n".join(texts)})
        elif role == "assistant":
            out = {"role": "assistant", "content": "", "reasoning_content": ""}
            if isinstance(content, str):
                out["content"] = content
            else:
                reasoning, texts, calls = [], [], []
                for b in content or []:
                    t = b.get("type")
                    if t == "thinking":
                        reasoning.append(b.get("thinking", ""))
                    elif t == "text":
                        texts.append(b.get("text", ""))
                    elif t == "tool_use":
                        calls.append({"id": b.get("id"), "type": "function",
                                      "function": {"name": b.get("name"),
                                                   "arguments": json.dumps(b.get("input") or {})}})
                out["content"] = "".join(texts)
                out["reasoning_content"] = "".join(reasoning)
                if calls:
                    out["tool_calls"] = calls
            msgs.append(out)
    body = {"model": model, "messages": msgs}
    if payload.get("max_tokens"):
        body["max_tokens"] = payload["max_tokens"]
    tools = payload.get("tools") or []
    if tools:
        body["tools"] = [{"type": "function", "function": {"name": t.get("name"), "description": t.get("description", ""),
                                                           "parameters": t.get("input_schema") or {"type": "object"}}}
                         for t in tools]
    tc = payload.get("tool_choice")
    if isinstance(tc, dict):
        body["tool_choice"] = {"auto": "auto", "any": "required", "none": "none"}.get(tc.get("type"), "auto") \
            if tc.get("type") != "tool" else {"type": "function", "function": {"name": tc.get("name")}}
    eff = map_effort(payload)
    if eff:
        body["reasoning_effort"] = eff
    for a, b in (("temperature", "temperature"), ("top_p", "top_p"), ("stop_sequences", "stop")):
        if payload.get(a) is not None:
            body[b] = payload[a]
    if payload.get("stream"):
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return body


# -------------------------------------------------------------------------------------------- response translation
STOP = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use", "content_filter": "end_turn",
        "function_call": "tool_use"}


def map_usage(u: dict | None) -> dict:
    u = u or {}
    hit = int(u.get("prompt_cache_hit_tokens") or (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    prompt = int(u.get("prompt_tokens") or 0)
    miss = int(u.get("prompt_cache_miss_tokens") if u.get("prompt_cache_miss_tokens") is not None else prompt - hit)
    out = int(u.get("completion_tokens") or 0)
    think = int((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
    return {"input_tokens": miss, "cache_read_input_tokens": hit, "cache_creation_input_tokens": 0,
            "output_tokens": out, "output_tokens_details": {"thinking_tokens": think}}


def cost_of(u: dict) -> float:
    """$ from an Anthropic-shaped usage (map_usage) and the price table -- an estimate."""
    return (u["input_tokens"] * PRICE["cache_miss"] + u["cache_read_input_tokens"] * PRICE["cache_hit"]
            + u["output_tokens"] * PRICE["output"]) / 1e6


def _sig(text: str) -> str:
    return "gw-" + hashlib.sha256(text.encode()).hexdigest()[:24]


def _tool_input(args: str):
    try:
        v = json.loads(args or "{}")
        return v if isinstance(v, dict) else {"value": v}
    except ValueError:
        return None


def openai_to_anthropic(resp: dict, model: str) -> dict:
    """OpenAI Chat non-stream response -> Anthropic Messages response."""
    ch = (resp.get("choices") or [{}])[0]
    m = ch.get("message") or {}
    content = []
    if m.get("reasoning_content"):
        content.append({"type": "thinking", "thinking": m["reasoning_content"], "signature": _sig(m["reasoning_content"])})
    if m.get("content"):
        content.append({"type": "text", "text": m["content"]})
    for tc in m.get("tool_calls") or []:
        fn = tc.get("function") or {}
        inp = _tool_input(fn.get("arguments"))
        if inp is None:
            raise ValueError(f"tool call {fn.get('name')} has arguments that are not JSON")
        content.append({"type": "tool_use", "id": tool_id(tc.get("id"), "call_gw"), "name": fn.get("name"), "input": inp})
    return {"id": "msg_" + str(resp.get("id") or "gw"), "type": "message", "role": "assistant", "model": model,
            "content": content, "stop_reason": STOP.get(ch.get("finish_reason"), "end_turn"),
            "stop_sequence": None, "usage": map_usage(resp.get("usage"))}


def _ev(name: str, data: dict) -> bytes:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()


class StreamTranslator:
    """OpenAI Chat SSE -> Anthropic Messages SSE. Reasoning and text stream live; tool calls are buffered per index
    and emitted whole when the turn ends (their order is then fixed and interleaved fragments cannot break it).
    feed(line) takes one SSE line from upstream and returns the bytes to send; finish() returns the closing events."""

    def __init__(self, model: str):
        self.model, self.started, self.idx = model, False, -1
        self.open = None            # "thinking" | "text" | None
        self.think_buf = ""
        self.tools = {}             # openai index -> {id, name, args}
        self.finish = None
        self.usage = None
        self.text_deltas = 0
        self.done = False

    def _start(self, rid) -> bytes:
        self.started = True
        return _ev("message_start", {"type": "message_start", "message": {
            "id": "msg_" + str(rid or "gw"), "type": "message", "role": "assistant", "model": self.model, "content": [],
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}}})

    def _close(self) -> bytes:
        out = b""
        if self.open == "thinking":
            out += _ev("content_block_delta", {"type": "content_block_delta", "index": self.idx,
                                               "delta": {"type": "signature_delta", "signature": _sig(self.think_buf)}})
        if self.open:
            out += _ev("content_block_stop", {"type": "content_block_stop", "index": self.idx})
        self.open = None
        return out

    def _open(self, kind: str) -> bytes:
        out = self._close()
        self.idx += 1
        self.open = kind
        block = {"type": "thinking", "thinking": "", "signature": ""} if kind == "thinking" else {"type": "text", "text": ""}
        return out + _ev("content_block_start", {"type": "content_block_start", "index": self.idx, "content_block": block})

    def feed(self, line: str) -> bytes:
        line = line.strip()
        if not line.startswith("data:"):
            return b""
        data = line[5:].strip()
        if data == "[DONE]":
            self.done = True
            return b""
        try:
            chunk = json.loads(data)
        except ValueError:
            return b""
        out = b"" if self.started else self._start(chunk.get("id"))
        if chunk.get("usage"):
            self.usage = chunk["usage"]
        for ch in chunk.get("choices") or []:
            d = ch.get("delta") or {}
            rc = d.get("reasoning_content")
            if rc:
                if self.open != "thinking":
                    out += self._open("thinking")
                self.think_buf += rc
                out += _ev("content_block_delta", {"type": "content_block_delta", "index": self.idx,
                                                   "delta": {"type": "thinking_delta", "thinking": rc}})
            txt = d.get("content")
            if txt:
                if self.open != "text":
                    out += self._open("text")
                self.text_deltas += 1
                out += _ev("content_block_delta", {"type": "content_block_delta", "index": self.idx,
                                                   "delta": {"type": "text_delta", "text": txt}})
            for tc in d.get("tool_calls") or []:
                slot = self.tools.setdefault(tc.get("index", len(self.tools)), {"id": None, "name": None, "args": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["args"] += fn["arguments"]
            if ch.get("finish_reason"):
                self.finish = ch["finish_reason"]
        return out

    def finish_events(self) -> bytes:
        out = b"" if self.started else self._start(None)
        out += self._close()
        for _i, slot in sorted(self.tools.items()):
            inp = _tool_input(slot["args"])
            if inp is None:
                raise ValueError(f"tool call {slot['name']} has arguments that are not JSON")
            self.idx += 1
            out += _ev("content_block_start", {"type": "content_block_start", "index": self.idx, "content_block": {
                "type": "tool_use", "id": tool_id(slot["id"], f"call_gw_{self.idx}"), "name": slot["name"], "input": {}}})
            out += _ev("content_block_delta", {"type": "content_block_delta", "index": self.idx,
                                               "delta": {"type": "input_json_delta", "partial_json": json.dumps(inp)}})
            out += _ev("content_block_stop", {"type": "content_block_stop", "index": self.idx})
        u = map_usage(self.usage)
        out += _ev("message_delta", {"type": "message_delta",
                                     "delta": {"stop_reason": STOP.get(self.finish, "tool_use" if self.tools else "end_turn"),
                                               "stop_sequence": None},
                                     "usage": u})
        out += _ev("message_stop", {"type": "message_stop"})
        return out


# ------------------------------------------------------------------------------------------------- accounting
def _call_tokens(prompt_miss, prompt_hit, completion) -> int:
    return int(prompt_miss or 0) + int(prompt_hit or 0) + int(completion or 0)


def _day_load():
    global TOKENS_SINCE
    if not DAY_LEDGER or not os.path.exists(DAY_LEDGER):
        return
    for l in open(DAY_LEDGER):
        try:
            r = json.loads(l)
            DAY_SPENT[r["day"]] = DAY_SPENT.get(r["day"], 0.0) + float(r.get("cost_est", 0.0))
            if TOKEN_CAP and float(r.get("t", 0)) >= TOKEN_SINCE:
                TOKENS_SINCE += _call_tokens(r.get("prompt_miss"), r.get("prompt_hit"), r.get("completion"))
        except (ValueError, KeyError, TypeError):
            continue


def halt(reason: str) -> None:
    """Refuse everything from now on and tell the watchdog (/__caps halted). The caller holds LOCK."""
    global HALT_REASON
    if HALT_REASON is None:
        HALT_REASON = reason
        STATS["halted"] = True
        STATS["halt_reason"] = reason


class ClientGone(Exception):
    """Writing to the worker failed: it hung up (a watchdog SIGTERM, a timeout), not the gateway."""


def _task_entry(task: str) -> dict:
    """The /__tasks record for a task. The caller holds LOCK."""
    return TASKS.setdefault(task, {"turns": 0, "out": 0, "think": 0, "text": 0, "tool": 0, "cost": 0.0,
                                   "first": round(time.time(), 1)})


def touch(task) -> None:
    """Stream output for this task just reached its worker: the watchdog's evidence that the turn is live."""
    if task:
        with LOCK:
            _task_entry(task)["last_byte"] = round(time.time(), 1)


def account(task, usage_a: dict, text_n: int, tool_n: int, want_max: int) -> float:
    global TOKENS_SINCE
    c = cost_of(usage_a)
    day = utc_day()
    with LOCK:
        STATS["cost_usd"] = round(STATS.get("cost_usd", 0.0) + c, 6)
        STATS["out_tokens"] = STATS.get("out_tokens", 0) + usage_a["output_tokens"]
        STATS["thinking_tokens"] = STATS.get("thinking_tokens", 0) + usage_a["output_tokens_details"]["thinking_tokens"]
        STATS["prompt_miss_tokens"] = STATS.get("prompt_miss_tokens", 0) + usage_a["input_tokens"]
        STATS["prompt_hit_tokens"] = STATS.get("prompt_hit_tokens", 0) + usage_a["cache_read_input_tokens"]
        DAY_SPENT[day] = round(DAY_SPENT.get(day, 0.0) + c, 6)
        TOKENS_SINCE += _call_tokens(usage_a["input_tokens"], usage_a["cache_read_input_tokens"], usage_a["output_tokens"])
        STATS["tokens_since"] = TOKENS_SINCE
        if TOKEN_CAP and TOKENS_SINCE >= TOKEN_CAP:
            halt(f"token ceiling reached: {TOKENS_SINCE} of {TOKEN_CAP} provider-counted tokens")
        if task:
            e = _task_entry(task)
            e["turns"] += 1
            e["out"] += usage_a["output_tokens"]
            e["think"] += usage_a["output_tokens_details"]["thinking_tokens"]
            e["cost"] = round(e["cost"] + c, 6)
            e["text"] += text_n
            e["tool"] += tool_n
            e["last"] = round(time.time(), 1)
            if want_max and usage_a["output_tokens"] >= want_max and text_n == 0:
                e["capped_turns"] = e.get("capped_turns", 0) + 1
                STATS["capped_turns"] = STATS.get("capped_turns", 0) + 1
        if DAY_LEDGER:
            with open(DAY_LEDGER, "a") as fh:
                fh.write(json.dumps({"t": round(time.time(), 1), "day": day, "task": task, "cost_est": round(c, 6),
                                     "prompt_miss": usage_a["input_tokens"], "prompt_hit": usage_a["cache_read_input_tokens"],
                                     "completion": usage_a["output_tokens"],
                                     "reasoning": usage_a["output_tokens_details"]["thinking_tokens"]}) + "\n")
    return c


# ---------------------------------------------------------------------------------------------------- server
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "gw-messages-shim"

    def log_message(self, fmt, *args):
        pass

    def _json(self, status: int, obj: dict):
        payload = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _err(self, status: int, etype: str, msg: str):
        self._json(status, {"type": "error", "error": {"type": etype, "message": msg}})

    def _send(self, b: bytes, flush: bool = False):
        """Write to the worker. A failure here is the worker's side (ClientGone), never the gateway's."""
        try:
            self.wfile.write(b)
            if flush:
                self.wfile.flush()
        except OSError as e:
            raise ClientGone(type(e).__name__) from e

    def _open_upstream(self, data: bytes, stream: bool, started: float, task=None):
        """Open the gateway and read its first body bytes: (resp, first bytes, error, ttft). Everything that can fail
        before a body byte exists is retried here (see the module notes); a quota 429 halts and is returned at once.
        For a non-stream call the whole body is the first read."""
        tries, err = 0, None
        while True:
            t0 = time.time()
            try:
                resp = urllib.request.urlopen(self._upstream_req(data), timeout=UPSTREAM_READ_TIMEOUT)
                try:
                    first = resp.read1(4096) if stream else resp.read()
                except Exception as e:  # noqa: BLE001
                    resp.close()
                    why, err = f"{type(e).__name__} before the first byte", e
                else:
                    if first:
                        return resp, first, None, time.time() - started
                    resp.close()
                    why, err = "empty body", RuntimeError("gateway: the body ended before its first byte")
            except urllib.error.HTTPError as e:
                try:
                    e.shim_body = e.read()[:2000]
                except Exception:  # noqa: BLE001
                    e.shim_body = b""
                if e.code == 429 and b"quota_exceeded" in e.shim_body:
                    with LOCK:
                        halt("gateway quota_exceeded (429): the key is out")
                    return None, b"", e, None
                why, err = str(e.code), e
            except Exception as e:  # noqa: BLE001
                why, err = type(e).__name__, e
            if tries >= PRESTREAM_RETRIES or HALT_REASON:
                return None, b"", err, None
            wait = PRESTREAM_BACKOFF[min(tries, len(PRESTREAM_BACKOFF) - 1)]
            with LOCK:
                STATS["prestream_retries"] = STATS.get("prestream_retries", 0) + 1
                STATS.setdefault("prestream_retried", {})
                STATS["prestream_retried"][why] = STATS["prestream_retried"].get(why, 0) + 1
            log(f"  pre-stream failure ({why}) after {time.time() - t0:.1f}s; shim retry {tries + 1}/{PRESTREAM_RETRIES} "
                f"in {wait}s task={task or '-'}")
            time.sleep(wait)
            tries += 1

    def _upstream_req(self, body: bytes):
        req = urllib.request.Request(UPSTREAM, data=body, method="POST")
        req.add_header("Authorization", "Bearer " + KEY)
        req.add_header("Content-Type", "application/json")
        req.add_header("User-Agent", USER_AGENT)
        return req

    def _messages(self, raw: bytes):
        started, status, n = time.time(), 0, 0
        try:
            payload = json.loads(raw)
        except ValueError:
            return self._err(400, "invalid_request_error", "shim: body is not JSON")
        task = TASK_RE.search(raw.decode("utf-8", "ignore"))
        task = task.group(1) if task else None
        model = payload.get("model") or GATEWAY_MODEL
        want_max = int(payload.get("max_tokens") or 0)
        stream = bool(payload.get("stream"))
        stubs = None
        if FILL_READ_STUBS:
            payload, nf, nu = fill_read_stubs(payload)
            stubs = f"{nf}/{nf + nu}"
            global STUB_DUMPS
            if nu and STUB_DUMP_DIR and STUB_DUMPS < 30:
                STUB_DUMPS += 1
                with open(os.path.join(STUB_DUMP_DIR, f"stub_{task}_{int(time.time() * 1000)}.json"), "w") as fh:
                    json.dump(stub_skeleton(payload), fh)
            with LOCK:
                STATS["read_stub_filled"] = STATS.get("read_stub_filled", 0) + nf
                STATS["read_stub_unmatched"] = STATS.get("read_stub_unmatched", 0) + nu
        body = anthropic_to_openai(payload, GATEWAY_MODEL)
        if LOG_FIELDS:
            log("  fields: " + ",".join(sorted(payload)) + " | model=" + str(payload.get("model")) +
                " | max_tokens=" + str(want_max) + " | stream=" + str(stream) +
                " | output_config=" + json.dumps(payload.get("output_config")) +
                " | thinking=" + json.dumps(payload.get("thinking")) + " | tools=" + str(len(payload.get("tools") or [])) +
                " | msgs=" + str(len(payload.get("messages") or [])) +
                " || upstream: model=" + body["model"] + " | reasoning_effort=" + str(body.get("reasoning_effort")) +
                " | tools=" + str(len(body.get("tools") or [])) + " | msgs=" + str(len(body["messages"])) +
                " | tool_msgs=" + str(sum(1 for m in body["messages"] if m["role"] == "tool")) +
                " | assistant_with_reasoning=" + str(sum(1 for m in body["messages"]
                                                         if m["role"] == "assistant" and m.get("reasoning_content"))) +
                ("" if stubs is None else " | read_stubs=" + stubs))
        stop = None
        with LOCK:
            spent, tspent = STATS.get("cost_usd", 0.0), (TASKS.get(task) or {}).get("cost", 0.0) if task else 0.0
            dspent = DAY_SPENT.get(utc_day(), 0.0)
        if HALT_REASON:
            stop = f"halted: {HALT_REASON}"
        elif spent >= RUN_COST_CAP:
            stop = f"run cost cap reached: ${spent:.2f} of ${RUN_COST_CAP:.2f} (estimated)"
        elif dspent >= DAY_CAP:
            stop = f"day cap reached: ${dspent:.2f} of ${DAY_CAP:.2f} for UTC {utc_day()} (estimated)"
        elif task and tspent >= TASK_COST_CAP:
            stop = f"task cost cap reached: {task} has spent ${tspent:.2f} of ${TASK_COST_CAP:.2f} (estimated)"
        if stop:
            self._err(402, "api_error", "shim cap: " + stop)
            with LOCK:
                STATS["refused_by_cap"] = STATS.get("refused_by_cap", 0) + 1
                STATS["halted"] = bool(HALT_REASON or spent >= RUN_COST_CAP or dspent >= DAY_CAP)
                REFUSALS.append({"t": round(time.time(), 1), "task": task, "reason": stop})
                STATS["requests"] += 1
                STATS["by_status"]["402"] = STATS["by_status"].get("402", 0) + 1
            log(f"  REFUSED: {stop}")
            return
        data = json.dumps(body).encode()
        resp, first, err, ttft = self._open_upstream(data, stream, started, task)
        sent = False
        try:
            if resp is None:
                raise err if err is not None else RuntimeError("upstream failed")
            with resp as up:
                if not stream:
                    obj = json.loads(first)
                    a = openai_to_anthropic(obj, model)
                    text_n = sum(1 for b in a["content"] if b["type"] == "text")
                    tool_n = sum(1 for b in a["content"] if b["type"] == "tool_use")
                    account(task, a["usage"], text_n, tool_n, want_max)
                    self._json(200, a)
                    status, n = 200, 0
                else:
                    self.send_response(200)
                    sent = True
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    st = StreamTranslator(model)
                    buf = b""
                    touched = 0.0
                    chunk = first                # the first bytes, read inside the pre-stream retry
                    while True:
                        if not chunk:
                            break
                        buf += chunk
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            out = st.feed(line.decode("utf-8", "replace"))
                            if out:
                                n += len(out)
                                self._send(b"%x\r\n%s\r\n" % (len(out), out), flush=True)
                                if time.time() - touched >= 1.0:
                                    touch(task)
                                    touched = time.time()
                        # read1, not read: read(4096) blocks until 4096 bytes have arrived, so output reached the
                        # worker in bursts and ttft / last_byte measured the burst, not the gateway
                        chunk = up.read1(4096)
                    if buf:
                        out = st.feed(buf.decode("utf-8", "replace"))
                        if out:
                            self._send(b"%x\r\n%s\r\n" % (len(out), out))
                    out = st.finish_events()
                    n += len(out)
                    self._send(b"%x\r\n%s\r\n" % (len(out), out))
                    self._send(b"0\r\n\r\n", flush=True)
                    if st.usage is None:
                        with LOCK:
                            STATS["streams_without_usage"] = STATS.get("streams_without_usage", 0) + 1
                    account(task, map_usage(st.usage), st.text_deltas, len(st.tools), want_max)
                    status = 200
            with LOCK:
                STATS["translated"] += 1
        except ClientGone as e:
            status = 499                   # nginx's "client closed request": the worker left, the gateway did not fail
            self.close_connection = True
            log(f"  worker gone mid-stream ({e}); stopped reading the gateway")
            with LOCK:
                STATS["client_aborts"] = STATS.get("client_aborts", 0) + 1
        except urllib.error.HTTPError as e:
            raw_err = (getattr(e, "shim_body", None) or e.read()[:2000]).decode("utf-8", "replace")
            try:
                msg = json.loads(raw_err).get("error", {}).get("message") or raw_err
            except ValueError:
                msg = raw_err
            status = e.code
            etype = "rate_limit_error" if e.code == 429 else ("invalid_request_error" if 400 <= e.code < 500 else "api_error")
            self._err(e.code, etype, "gateway: " + str(msg)[:500])
            with LOCK:
                STATS["upstream_errors"] += 1
        except Exception as e:  # noqa: BLE001
            status = 502
            if sent:
                try:
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except OSError:
                    pass
                self.close_connection = True
                log(f"  upstream {type(e).__name__} mid-stream; closed the body and the connection")
            else:
                try:
                    self._err(502, "api_error", f"shim: {type(e).__name__}: {str(e)[:200]}")
                except OSError:
                    self.close_connection = True
            with LOCK:
                STATS["upstream_errors"] += 1
                STATS["mid_stream_aborts"] = STATS.get("mid_stream_aborts", 0) + int(sent)
        with LOCK:
            STATS["requests"] += 1
            STATS["by_status"][str(status)] = STATS["by_status"].get(str(status), 0) + 1
            if ttft is not None:
                STATS.setdefault("ttft_s", []).append(round(ttft, 2))
                STATS["ttft_s"] = STATS["ttft_s"][-500:]
        log(f"POST /v1/messages -> {status} {n}B {time.time() - started:.2f}s {'stream' if stream else 'json'} "
            f"ttft={'%.2fs' % ttft if ttft is not None else '-'} task={task or '-'}")

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError):
            self.close_connection = True

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        path = self.path.split("?")[0]
        if path.endswith("/count_tokens"):       # as with OpenRouter today: Claude Code tolerates it
            with LOCK:
                STATS["requests"] += 1
                STATS["by_status"]["404"] = STATS["by_status"].get("404", 0) + 1
            return self._err(404, "not_found_error", "count_tokens is not served by the gateway shim")
        if path.endswith("/v1/messages") or path == "/messages":
            global ACTIVE
            with LOCK:
                ACTIVE += 1
            try:
                return self._messages(raw)
            finally:
                with LOCK:
                    ACTIVE -= 1
                if DRAINING:                     # the worker's next request goes to the successor
                    self.close_connection = True
        self._err(404, "not_found_error", f"shim: no route {path}")

    def do_GET(self):
        with LOCK:
            if self.path == "/__caps":
                obj = {"task_cost_cap": TASK_COST_CAP, "run_cost_cap": RUN_COST_CAP, "spent": STATS.get("cost_usd", 0.0),
                       "day_cap": DAY_CAP, "day_spent": DAY_SPENT.get(utc_day(), 0.0), "utc_day": utc_day(),
                       "halted": bool(STATS.get("halted")), "halt_reason": HALT_REASON,
                       "token_cap": TOKEN_CAP, "token_since": TOKEN_SINCE, "tokens_since": TOKENS_SINCE,
                       "refusals": REFUSALS[-50:], "cost_is_estimate": True,
                       "price_per_mtok": PRICE}
            elif self.path == "/__tasks":
                obj = TASKS
            elif self.path == "/__stats":
                obj = STATS
            else:
                obj = None
            payload = json.dumps(obj).encode() if obj is not None else None
        if payload is None:
            return self._err(404, "not_found_error", "shim: no route")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_HEAD(self):
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()


def main() -> int:
    global KEY, LOG_FIELDS, TASK_COST_CAP, RUN_COST_CAP, DAY_CAP, DAY_LEDGER, GATEWAY_MODEL, FORCE_EFFORT, FILL_READ_STUBS
    global TOKEN_CAP, TOKEN_SINCE, UPSTREAM
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8799)
    ap.add_argument("--upstream", default=UPSTREAM, help="the gateway's chat/completions URL (tests point it at a fake)")
    ap.add_argument("--pid-file")
    ap.add_argument("--gateway-model", default=GATEWAY_MODEL)
    ap.add_argument("--force-effort", choices=EFFORTS, help="send this reasoning_effort whatever the client asked")
    ap.add_argument("--task-cost-cap", type=float, default=2.00)
    ap.add_argument("--run-cost-cap", type=float, default=120.0)
    ap.add_argument("--day-cap", type=float, default=480.0, help="$ per UTC day on this key (the $500 limit minus reserve)")
    ap.add_argument("--day-ledger", required=True, help="append-only JSONL of every call's estimated cost and tokens")
    ap.add_argument("--log-fields", action="store_true")
    ap.add_argument("--dump-unmatched-stubs", help="dir: write a content-free skeleton (stub_skeleton) of up to 30 "
                    "requests that carry a stub --fill-read-stubs could not match (diagnosis)")
    ap.add_argument("--unique-tool-ids", action="store_true", help="suffix each upstream tool call id with 12 random hex "
                    "so ids never repeat within a session (opt-in; see the docstring)")
    ap.add_argument("--fill-read-stubs", action="store_true", help="replace Claude Code's file-unchanged Read stub with the "
                    "earlier Read result for the same file and range (opt-in; see the docstring)")
    ap.add_argument("--price", help="JSON {cache_miss, cache_hit, output (dollars per MILLION tokens), source}: the price "
                    "table for an upstream other than the huanzhimao gateway (default: DeepSeek V4.1-Flash, unchanged)")
    ap.add_argument("--token-cap", type=int, default=0,
                    help="ceiling on provider-counted tokens (prompt miss + hit + completion) since --token-since; 0 = none")
    ap.add_argument("--token-since", help="UTC instant the ceiling counts from, YYYY-MM-DDTHH:MM:SSZ (required with --token-cap)")
    args = ap.parse_args()
    if args.price:
        p = json.loads(args.price)
        assert {"cache_miss", "cache_hit", "output", "source"} <= set(p), "--price needs cache_miss, cache_hit, output, source"
        PRICE.clear(); PRICE.update(p)
    KEY = os.environ.get("GATEWAY_API_KEY", "").strip()
    if not KEY:
        sys.exit("GATEWAY_API_KEY is not set in this process's environment")
    for p in (args.day_ledger, args.pid_file):
        if p and not os.path.isabs(p):
            sys.exit(f"{p}: give an absolute path (a relative one resolves from this process's cwd)")
    UPSTREAM = args.upstream
    GATEWAY_MODEL, FORCE_EFFORT, LOG_FIELDS = args.gateway_model, args.force_effort, bool(args.log_fields)
    FILL_READ_STUBS = bool(args.fill_read_stubs)
    global STUB_DUMP_DIR, UNIQUE_TOOL_IDS
    STUB_DUMP_DIR = args.dump_unmatched_stubs
    UNIQUE_TOOL_IDS = bool(args.unique_tool_ids)
    if UNIQUE_TOOL_IDS:
        log("--unique-tool-ids: tool_use ids are suffixed with 12 random hex (unique within a session)")
    if FILL_READ_STUBS:
        log("--fill-read-stubs: file-unchanged Read stubs are replaced by the earlier Read result (same file and range)")
    TASK_COST_CAP, RUN_COST_CAP, DAY_CAP, DAY_LEDGER = args.task_cost_cap, args.run_cost_cap, args.day_cap, args.day_ledger
    if args.token_cap:
        if not args.token_since:
            sys.exit("--token-cap needs --token-since")
        TOKEN_CAP = args.token_cap
        TOKEN_SINCE = _dt.datetime.strptime(args.token_since, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=_dt.timezone.utc).timestamp()
    os.makedirs(os.path.dirname(DAY_LEDGER), exist_ok=True)
    _day_load()
    if TOKEN_CAP:
        STATS["tokens_since"] = TOKENS_SINCE
        if TOKENS_SINCE >= TOKEN_CAP:
            halt(f"token ceiling reached before start: {TOKENS_SINCE} of {TOKEN_CAP}")
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)

    def _drain(signum, frame):
        global DRAINING
        if not DRAINING:
            DRAINING = True
            log("SIGUSR1: draining -- no new connections; in-flight requests finish, then exit")
            threading.Thread(target=srv.shutdown, daemon=True).start()
    signal.signal(signal.SIGUSR1, _drain)
    if args.pid_file:
        with open(args.pid_file, "w") as fh:
            fh.write(str(os.getpid()) + "\n")
    log(f"caps: ${TASK_COST_CAP:.2f} per task, ${RUN_COST_CAP:.2f} per run, ${DAY_CAP:.2f} per UTC day "
        f"(day {utc_day()} so far ${DAY_SPENT.get(utc_day(), 0.0):.2f} per {DAY_LEDGER}); ALL ESTIMATED from {PRICE}")
    if TOKEN_CAP:
        log(f"token ceiling: {TOKEN_CAP} provider-counted tokens since {args.token_since}; "
            f"{TOKENS_SINCE} already in the ledger{' -- HALTED' if HALT_REASON else ''}")
    log(f"listening on 127.0.0.1:{args.port} -> {UPSTREAM} model={GATEWAY_MODEL} force_effort={FORCE_EFFORT} "
        f"(key from the environment, {len(KEY)} chars, never logged)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    srv.server_close()                           # the port is free: a successor can bind it now
    if DRAINING:
        log(f"drain: port closed; waiting for {ACTIVE} request(s) in flight")
        waited = 0
        while ACTIVE > 0 and waited < UPSTREAM_READ_TIMEOUT * 4:
            time.sleep(1)
            waited += 1
            if waited % 60 == 0:
                log(f"drain: {ACTIVE} still in flight after {waited}s")
        log(f"drain: done ({ACTIVE} left)")
    log(f"stopping; stats {json.dumps(STATS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
