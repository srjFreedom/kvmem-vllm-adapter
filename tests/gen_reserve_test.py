#!/usr/bin/env python3
"""R is a hard ceiling on generation: max_tokens is clamped to the generation reserve."""
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

checks = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    checks.append(bool(cond))


class Echo(BaseHTTPRequestHandler):
    """Pretends to be the engine: reports the max_tokens it actually received."""
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        out = json.dumps({"max_tokens": body.get("max_tokens"),
                          "max_completion_tokens": body.get("max_completion_tokens")}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main():
    upstream_port = free_port()
    engine = ThreadingHTTPServer(("127.0.0.1", upstream_port), Echo)
    threading.Thread(target=engine.serve_forever, daemon=True).start()

    gw_port = free_port()
    gw = subprocess.Popen([
        sys.executable, os.path.join(os.path.dirname(__file__), "..", "scripts", "kvmem_gateway.py"),
        "--upstream", f"http://127.0.0.1:{upstream_port}", "--port", str(gw_port),
        "--kvmem-gen-reserve", "64", "--kvmem-budget", "4096",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{gw_port}/health", timeout=1)
                break
            except Exception:
                time.sleep(0.1)

        def ask(max_tokens, field="max_tokens"):
            body = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}],
                               field: max_tokens}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{gw_port}/v1/chat/completions",
                                         data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read())

        got = ask(500)
        check(got["max_tokens"] == 64, f"a larger client budget is reduced to R (got {got['max_tokens']})")
        got = ask(32)
        check(got["max_tokens"] == 32, f"a smaller client budget is honoured (got {got['max_tokens']})")
        got = ask(500, field="max_completion_tokens")
        check(got["max_completion_tokens"] == 64,
              f"max_completion_tokens gets R too (got {got['max_completion_tokens']})")

        # Omitting the field must not let the engine's own default decide: R is stamped on.
        body = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}]}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{gw_port}/v1/chat/completions",
                                     data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            got = json.loads(resp.read())
        check(got["max_tokens"] == 64, f"an omitted budget is stamped with R (got {got['max_tokens']})")
    finally:
        gw.terminate()
        engine.shutdown()

    print(("PASS" if all(checks) else "FAILED") + " gen_reserve_test")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
