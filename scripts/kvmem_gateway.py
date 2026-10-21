#!/usr/bin/env python3
"""KVMem entry gateway: trims long histories before they reach vLLM.

Why this exists: vLLM indexes KV blocks by logical position and accounts sequence length
monotonically, so a KV layer cannot truncate a live request (measured: scheduler assertion, then a
hang with spec decoding off). The entry is the one place selection is compatible with the engine -
send less history, and the engine prefills exactly what it is given.

Run:
    python scripts/kvmem_gateway.py --upstream http://127.0.0.1:8001 --port 8100 \
        --kvmem-budget 16384 --kvmem-sink-tokens 256 --kvmem-recent-tokens 1024

Only /v1/chat/completions is rewritten; everything else is forwarded untouched.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0] + "/src")
from kvmem_entry_policy import (  # noqa: E402
    ContextTooLong,
    EntryPolicy,
    rewrite_payload,
)

ARGS = None
CACHE = None
CLAMPS = {"count": 0}


POLICY = None
CACHE = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep the console readable
        if ARGS.verbose:
            sys.stderr.write("gateway: " + (fmt % args) + "\n")

    CORS = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Max-Age": "86400",
    }

    def _send_cors(self):
        for key, value in self.CORS.items():
            self.send_header(key, value)

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _forward(self, body: bytes, content_type: str):
        request = urllib.request.Request(
            ARGS.upstream.rstrip("/") + self.path, data=body, method=self.command,
            headers={"Content-Type": content_type or "application/json"})
        try:
            response = urllib.request.urlopen(request, timeout=ARGS.timeout)
        except urllib.error.HTTPError as exc:
            payload = exc.read()
            ctype = exc.headers.get("Content-Type", "application/json")
            self.send_response(exc.code)
            self.send_header("Content-Type", ctype)
            self._send_cors()
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        except Exception as exc:
            payload = json.dumps({"error": {"message": f"gateway upstream error: {exc}"}}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self._send_cors()
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        ctype = response.headers.get("Content-Type", "application/json")
        streaming = "text/event-stream" in ctype
        if streaming:
            # Server-sent events must be relayed as they arrive: buffering the whole stream turns
            # token-by-token output into one final blob.
            try:
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except Exception:
                pass
            self.send_response(response.status)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self._send_cors()
            self.end_headers()
            try:
                while True:
                    chunk = response.read(512)
                    if not chunk:
                        break
                    self.wfile.write(b"%X\r\n" % len(chunk) + chunk + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except Exception:
                pass
            finally:
                response.close()
            return

        payload = response.read()
        response.close()
        self.send_response(response.status)
        self.send_header("Content-Type", ctype)
        self._send_cors()
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        self._forward(None, None)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        if "/v1/chat/completions" in self.path and body:
            try:
                payload = json.loads(body)
                changed, stats = rewrite_payload(payload, POLICY, CACHE)
                if changed:
                    if stats:
                        sys.stderr.write("gateway: " + json.dumps(
                            {"kind": "entry_rewrite", **(stats or {})},
                            separators=(",", ":")) + "\n")
                    body = json.dumps(payload).encode()
            except ContextTooLong as exc:
                out = json.dumps({"error": {"message": str(exc), "type": "BadRequestError",
                                            "code": 400}}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
                return
            except Exception as exc:
                sys.stderr.write(f"gateway: rewrite skipped ({exc})\n")
        self._forward(body, self.headers.get("Content-Type"))


def main(argv=None) -> int:
    global ARGS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0",
                        help="listen address; 0.0.0.0 serves the LAN like the engine does")
    parser.add_argument("--upstream", default="http://127.0.0.1:8001")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--kvmem-budget", type=int, default=16384,
                        help="B: token budget for the history sent to the engine")
    parser.add_argument("--kvmem-ctx", type=int, default=524288,
                        help="CTX: kvmem logical context ceiling (independent of the engine window)")
    parser.add_argument("--kvmem-gen-reserve", type=int, default=8192,
                        help="R: hard ceiling on generation; max_tokens is clamped to it")
    parser.add_argument("--kvmem-sink-tokens", type=int, default=256)
    parser.add_argument("--kvmem-recent-tokens", type=int, default=1024)
    parser.add_argument("--chars-per-token", type=int, default=3)
    parser.add_argument("--kvmem-select", choices=("retrieval", "recency"), default="retrieval",
                        help="retrieval scores blocks against the question; recency keeps sink+tail only")
    parser.add_argument("--question-tail-chars", type=int, default=600,
                        help="tail of the user message treated as the question when scoring")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--kvmem-history-mib", type=int, default=0,
                        help="H budget for the entry-side history block cache (0 disables it)")
    parser.add_argument("--kvmem-history-kib", type=int, default=0,
                        help="H budget in KiB (overrides --kvmem-history-mib when > 0)")
    parser.add_argument("--kvmem-history-disk-mib", type=int, default=0,
                        help="D budget for the history cache; requires --kvmem-history-disk-path")
    parser.add_argument("--kvmem-history-disk-path", default="/tmp/kvmem_history_tier")
    parser.add_argument("--verbose", action="store_true")
    ARGS = parser.parse_args(argv)
    global CACHE, POLICY
    host_kib = ARGS.kvmem_history_kib if ARGS.kvmem_history_kib > 0 \
        else ARGS.kvmem_history_mib * 1024
    POLICY = EntryPolicy(
        budget=ARGS.kvmem_budget, gen_reserve=ARGS.kvmem_gen_reserve, ctx=ARGS.kvmem_ctx,
        sink_tokens=ARGS.kvmem_sink_tokens, recent_tokens=ARGS.kvmem_recent_tokens,
        select=ARGS.kvmem_select, chars_per_token=max(1, ARGS.chars_per_token),
        question_tail_chars=ARGS.question_tail_chars,
        history_kib=host_kib, history_disk_mib=ARGS.kvmem_history_disk_mib,
        history_disk_path=ARGS.kvmem_history_disk_path)
    CACHE = POLICY.make_cache()
    for issue in POLICY.warnings():
        print(f"[kvmem gateway] warning: {issue}", file=sys.stderr, flush=True)
    server = ThreadingHTTPServer((ARGS.host, ARGS.port), Handler)
    print(f"[kvmem gateway] {ARGS.upstream} <- {ARGS.host}:{ARGS.port} "
          f"CTX={ARGS.kvmem_ctx} B={ARGS.kvmem_budget} R={ARGS.kvmem_gen_reserve} "
          f"sink={ARGS.kvmem_sink_tokens} recent={ARGS.kvmem_recent_tokens}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
