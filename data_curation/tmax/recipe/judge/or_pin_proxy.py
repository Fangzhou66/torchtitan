#!/usr/bin/env python3
"""Loopback shim: Anthropic /v1/messages in -> OpenRouter's Anthropic skin out, with the
provider pin injected into the body.

Why this exists. OpenRouter's Anthropic skin (POST https://openrouter.ai/api/v1/messages)
already speaks exactly the protocol Claude Code speaks, and it HONOURS OpenRouter's
`provider` routing object in the body (a bogus slug there returns 404 "No allowed providers
are available"). Claude Code cannot add a body field. So this shim adds that one field and
nothing else: it does NOT translate formats (no Anthropic->OpenAI round trip, so cache_control,
thinking blocks, tool_use and the SSE shape reach Claude Code exactly as OpenRouter emits them).

Key handling: read from $OPENROUTER_API_KEY in this process at startup, held in memory only.
Never logged, never written, never on a command line. The client's own Authorization header is
ignored and replaced.

Usage (bind is always 127.0.0.1):
    export OPENROUTER_API_KEY="$(cat /home/zekaili/fangzhou/tb_check/openrouter.txt)"
    python3 or_pin_proxy.py --port 8787 --pid-file /path/run.pid --log /path/proxy.log
Stop:
    kill "$(cat /path/run.pid)"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = "https://openrouter.ai/api"
KEY = ""
PIN = {"only": ["io-net/fp8"], "allow_fallbacks": False}
UPSTREAM_READ_TIMEOUT = 1800
# Retry a REJECTED call here, before the worker ever learns of it: a worker-level retry throws away a whole
# judgment (minutes, tens of thousands of tokens) to recover from one refused HTTP call. Only safe while
# nothing has been written to the client -- once bytes flow, a retry would duplicate a partial stream.
RETRY_STATUS = {429, 500, 502, 503, 504}
RETRY_WINDOW_SEC = 60          # total per request; beyond this the provider is saturated, let --attempts take over
RETRY_BACKOFF = (1, 2, 4, 8, 16, 32)
TASK_RE = re.compile(r"/rows/([A-Za-z0-9_.-]+)\.jsonl")   # the row path names the task; a label, never content
PROVIDER_RE = re.compile(r'"provider"\s*:\s*"([^"]+)"')
TEXT_RE = re.compile(rb'"type"\s*:\s*"text_delta"')      # the answer being written
TOOL_RE = re.compile(rb'"type"\s*:\s*"tool_use"')        # the judge acting (Read/Write/Bash)
TASKS = {}                                                # task -> cumulative productivity, read by the watchdog
TASK_COST_CAP = 1.50      # a single task may not spend more than this across ALL its attempts
RUN_COST_CAP = 100.00     # the whole run stops here; enforced, not reported
REFUSALS = []             # {t, task, reason} -- so an abandoned task is distinguishable from a transport death
LOCK = threading.Lock()
LOG_FIELDS = False
PROVIDER_LOG = "/tmp/or_pin_provider.jsonl"
STATS = {"requests": 0, "injected": 0, "upstream_errors": 0, "by_status": {}}


def log(msg: str) -> None:
    with LOCK:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _cost_of(tail: bytes):
    """(cost_usd, output_tokens, thinking_tokens) scraped from the tail of a response.
    Numbers only -- this never looks at, stores or logs message content."""
    try:
        text = tail.decode("utf-8", "ignore")
        cost = re.findall(r'"cost"\s*:\s*([0-9.eE+-]+)', text)
        if not cost:
            return None
        out = re.findall(r'"output_tokens"\s*:\s*([0-9]+)', text)
        thk = re.findall(r'"thinking_tokens"\s*:\s*([0-9]+)', text)
        return (float(cost[-1]), int(out[-1]) if out else None, int(thk[-1]) if thk else None)
    except Exception:  # noqa: BLE001 -- accounting must never affect the proxied response
        return None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "or-pin-shim"

    def log_message(self, fmt, *args):      # silence the default access log (it prints headers)
        pass

    def _forward(self, body: bytes | None):
        path = self.path
        url = UPSTREAM + path if path.startswith("/v1/") else UPSTREAM + "/v1" + path
        injected = False
        if body and path.startswith("/v1/messages") and not path.split("?")[0].endswith("count_tokens"):
            try:
                payload = json.loads(body)
                if isinstance(payload, dict):
                    if LOG_FIELDS:
                        # field NAMES and enum-valued knobs only -- never message content
                        log("  fields: " + ",".join(sorted(payload)) +
                            " | model=" + str(payload.get("model")) +
                            " | max_tokens=" + str(payload.get("max_tokens")) +
                            " | stream=" + str(payload.get("stream")) +
                            " | output_config=" + json.dumps(payload.get("output_config")) +
                            " | thinking=" + json.dumps(payload.get("thinking")) +
                            " | tools=" + str(len(payload.get("tools") or [])) +
                            " | msgs=" + str(len(payload.get("messages") or [])))
                    payload["provider"] = json.loads(json.dumps(PIN))
                    body = json.dumps(payload).encode()
                    injected = True
            except ValueError:
                pass                                   # not JSON: forward untouched
        req = urllib.request.Request(url, data=body, method=self.command)
        req.add_header("Authorization", "Bearer " + KEY)
        req.add_header("Content-Type", "application/json")
        for h in ("anthropic-version", "anthropic-beta", "accept"):
            v = self.headers.get(h)
            if v:
                req.add_header(h, v)
        started = time.time()
        sent = False                      # have response headers gone out? decides how an error ends
        task = (TASK_RE.search(body.decode("utf-8", "ignore")) if body else None)
        task = task.group(1) if task else None
        want_max = 0
        if body:
            m = re.search(rb'"max_tokens"\s*:\s*(\d+)', body)
            want_max = int(m.group(1)) if m else 0
        # --- hard caps, enforced here because this is the only layer that sees the money ------------
        stop = None
        with LOCK:
            spent = STATS.get("cost_usd", 0.0)
            tspent = (TASKS.get(task) or {}).get("cost", 0.0) if task else 0.0
        if spent >= RUN_COST_CAP:
            stop = f"run cost cap reached: ${spent:.2f} of ${RUN_COST_CAP:.2f}"
        elif task and tspent >= TASK_COST_CAP:
            stop = f"task cost cap reached: {task} has spent ${tspent:.2f} of ${TASK_COST_CAP:.2f}"
        if stop:
            payload = json.dumps({"type": "error", "error": {"type": "api_error", "message": "shim cap: " + stop}}).encode()
            self.send_response(402)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            with LOCK:
                STATS["refused_by_cap"] = STATS.get("refused_by_cap", 0) + 1
                STATS["halted"] = bool(spent >= RUN_COST_CAP)
                REFUSALS.append({"t": round(time.time(), 1), "task": task, "reason": stop})
                STATS["requests"] = STATS.get("requests", 0) + 1
                STATS["by_status"]["402"] = STATS["by_status"].get("402", 0) + 1
            log(f"  REFUSED: {stop}")
            return
        # --- retry the refusals, transparently, while nothing has reached the client -----------------
        attempt = 0
        while True:
            try:
                # NOT a `with`: the response must stay open for the streaming loop below. Reading the first
                # chunk here is what proves the call succeeded, so a retry can still be transparent.
                resp = urllib.request.urlopen(req, timeout=UPSTREAM_READ_TIMEOUT)
                first = resp.read(4096)
                upstream = resp
                break
            except urllib.error.HTTPError as e:
                if (e.code not in RETRY_STATUS or attempt >= len(RETRY_BACKOFF)
                        or time.time() - started + RETRY_BACKOFF[attempt] > RETRY_WINDOW_SEC):
                    upstream, first = None, None
                    err = e
                    break
                with LOCK:
                    STATS["retries"] = STATS.get("retries", 0) + 1
                    STATS["retried_status"] = STATS.get("retried_status", {})
                    STATS["retried_status"][str(e.code)] = STATS["retried_status"].get(str(e.code), 0) + 1
                log(f"  upstream {e.code}; retrying in {RETRY_BACKOFF[attempt]}s "
                    f"(attempt {attempt + 1}, {time.time() - started:.0f}s of {RETRY_WINDOW_SEC}s)")
                time.sleep(RETRY_BACKOFF[attempt])
                attempt += 1
                req = urllib.request.Request(url, data=body, method=self.command)
                req.add_header("Authorization", "Bearer " + KEY)
                req.add_header("Content-Type", "application/json")
                for h in ("anthropic-version", "anthropic-beta", "accept"):
                    v = self.headers.get(h)
                    if v:
                        req.add_header(h, v)
                continue
            except Exception:
                upstream, first = None, None
                err = None
                break
        try:
            # A thinking-heavy turn sends NOTHING for minutes (40k thinking tokens at ~50 tok/s is 800 s),
            # and this timeout is per socket read: 900 s turned legitimate long turns into mid-stream
            # aborts. Keep it well above the longest observed silence.
            if upstream is None:
                raise err if err is not None else RuntimeError("upstream failed")
            with upstream as up:
                self.send_response(up.status)
                sent = True
                for k, v in up.headers.items():
                    if k.lower() in ("content-type", "cache-control"):
                        self.send_header(k, v)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                n = 0
                tail = first or b""
                prov = PROVIDER_RE.search((first or b"").decode("utf-8", "ignore"))
                if prov:
                    with LOCK:
                        STATS.setdefault("by_provider", {})
                        STATS["by_provider"][prov.group(1)] = STATS["by_provider"].get(prov.group(1), 0) + 1
                    if task:
                        with LOCK, open(PROVIDER_LOG, "a") as fh:
                            fh.write(json.dumps({"t": round(time.time(), 1), "task": task,
                                                 "provider": prov.group(1)}) + "\n")
                text_n = len(TEXT_RE.findall(first or b""))
                tool_n = len(TOOL_RE.findall(first or b""))
                if first:
                    n += len(first)
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(first), first))
                    self.wfile.flush()
                while True:
                    chunk = up.read(4096)
                    if not chunk:
                        break
                    n += len(chunk)
                    text_n += len(TEXT_RE.findall(chunk))
                    tool_n += len(TOOL_RE.findall(chunk))
                    tail = (tail + chunk)[-4096:]
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                    self.wfile.flush()
                cost = _cost_of(tail)
                if cost:
                    with LOCK:
                        STATS["cost_usd"] = round(STATS.get("cost_usd", 0.0) + cost[0], 6)
                        STATS["out_tokens"] = STATS.get("out_tokens", 0) + (cost[1] or 0)
                        STATS["thinking_tokens"] = STATS.get("thinking_tokens", 0) + (cost[2] or 0)
                # Per-task productivity, for the watchdog. A judgment that is working writes text and calls
                # tools; the failure signature of 2026-09-21 is output that is all thinking and produces
                # neither. Numbers only -- no message content is read or kept.
                if task:
                    with LOCK:
                        e = TASKS.setdefault(task, {"turns": 0, "out": 0, "think": 0, "text": 0, "tool": 0,
                                                    "cost": 0.0, "first": round(time.time(), 1)})
                        e["turns"] += 1
                        e["out"] += (cost[1] or 0) if cost else 0
                        e["think"] += (cost[2] or 0) if cost else 0
                        e["cost"] = round(e["cost"] + (cost[0] if cost else 0.0), 6)
                        e["text"] += text_n
                        e["tool"] += tool_n
                        e["last"] = round(time.time(), 1)
                        # A response that used the whole max_tokens and emitted no text is a turn the
                        # budget killed: the model thought until the ceiling and never answered.
                        if want_max and cost and (cost[1] or 0) >= want_max and text_n == 0:
                            e["capped_turns"] = e.get("capped_turns", 0) + 1
                            STATS["capped_turns"] = STATS.get("capped_turns", 0) + 1
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
                status = up.status
        except urllib.error.HTTPError as e:
            payload = e.read()
            status = e.code
            self.send_response(e.code)
            self.send_header("Content-Type", e.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            n = len(payload)
            with LOCK:
                STATS["upstream_errors"] += 1
        except Exception as e:                          # noqa: BLE001 -- report, never crash the shim
            # A failure AFTER the headers went out cannot become a 502: a second status line written into
            # an open chunked body leaves the client waiting for a terminator that never comes, and a
            # Claude Code worker then sits in ep_poll until its own timeout (measured: six workers wedged
            # for 19 minutes when a provider 429'd mid-stream). Terminate the body and drop the connection.
            status = 502
            n = 0
            if sent:
                try:
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except OSError:
                    pass
                self.close_connection = True
                log(f"  upstream {type(e).__name__} mid-stream; closed the body and the connection")
            else:
                payload = json.dumps({"type": "error", "error": {"type": "api_error",
                                                                 "message": f"shim: {type(e).__name__}"}}).encode()
                try:
                    self.send_response(502)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    n = len(payload)
                except OSError:
                    self.close_connection = True
            with LOCK:
                STATS["upstream_errors"] += 1
                STATS["mid_stream_aborts"] = STATS.get("mid_stream_aborts", 0) + int(sent)
        with LOCK:
            STATS["requests"] += 1
            STATS["injected"] += int(injected)
            STATS["by_status"][str(status)] = STATS["by_status"].get(str(status), 0) + 1
        log(f"{self.command} {path} -> {status} {n}B {time.time()-started:.2f}s pin={'yes' if injected else 'no'}")

    def handle_one_request(self):
        """A client that resets the connection is normal here (Claude Code aborts a stalled stream);
        it is not a stack trace."""
        try:
            super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError):
            self.close_connection = True

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self._forward(self.rfile.read(length) if length else None)

    def do_GET(self):
        if self.path == "/__caps":
            with LOCK:
                payload = json.dumps({"task_cost_cap": TASK_COST_CAP, "run_cost_cap": RUN_COST_CAP,
                                      "spent": STATS.get("cost_usd", 0.0),
                                      "halted": bool(STATS.get("halted")),
                                      "refusals": REFUSALS[-50:]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path == "/__tasks":
            with LOCK:
                payload = json.dumps(TASKS).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path == "/__stats":
            payload = json.dumps(STATS).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self._forward(None)

    def do_HEAD(self):                                   # Claude Code's startup HEAD /api/hello
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()


def main() -> int:
    global KEY, PIN, LOG_FIELDS, PROVIDER_LOG, TASK_COST_CAP, RUN_COST_CAP
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--provider", default="io-net/fp8")
    ap.add_argument("--allow-fallbacks", action="store_true")
    ap.add_argument("--pid-file", default=None)
    ap.add_argument("--task-cost-cap", type=float, default=1.50, help="max $ one task may spend across all attempts")
    ap.add_argument("--run-cost-cap", type=float, default=100.0, help="max $ for the whole run; the shim then refuses everything")
    ap.add_argument("--provider-log", default=None, help="JSONL of {t, task, provider} per response")
    ap.add_argument("--log-fields", action="store_true",
                    help="log the request's top-level field names and enum knobs (never content)")
    args = ap.parse_args()
    KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not KEY:
        sys.exit("OPENROUTER_API_KEY is not set in this process's environment")
    PIN = {"only": [args.provider], "allow_fallbacks": bool(args.allow_fallbacks)}
    LOG_FIELDS = bool(args.log_fields)
    TASK_COST_CAP = args.task_cost_cap
    RUN_COST_CAP = args.run_cost_cap
    if args.provider_log:
        PROVIDER_LOG = args.provider_log
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    if args.pid_file:
        with open(args.pid_file, "w") as fh:
            fh.write(str(os.getpid()) + "\n")
    log(f"caps: ${TASK_COST_CAP:.2f} per task, ${RUN_COST_CAP:.2f} per run")
    log(f"listening on 127.0.0.1:{args.port} -> {UPSTREAM}/v1/messages, provider pin {PIN} "
        f"(key loaded from the environment, {len(KEY)} chars, never logged)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    log(f"stopping; stats {json.dumps(STATS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
