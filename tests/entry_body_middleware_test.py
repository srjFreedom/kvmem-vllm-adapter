#!/usr/bin/env python3
"""Standard body rewriting: Starlette middleware + the `request._body` cache.

This is the documented way to change a request body in Starlette/FastAPI: read it once, write the new
bytes back into the cache, and let `call_next` replay from the cache. Replacing the ASGI `receive`
channel instead leaves downstream readers waiting on a drained stream (measured: the request entered
the handler and never returned).
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "plugin"))
sys.path.insert(0, os.path.join(ROOT, "src"))

os.environ.update({
    "KVMEM_ENTRY_B": "4096",
    "KVMEM_ENTRY_R": "64",
    "KVMEM_ENTRY_CTX": "100000",
    "KVMEM_ENTRY_SINK": "256",
    "KVMEM_ENTRY_RECENT": "512",
    "KVMEM_ENTRY_HISTORY_KIB": "0",
})

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

import kvmem_entry_middleware as entry  # noqa: E402

checks = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    checks.append(bool(cond))


def make_client():
    seen = {}
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        seen["max"] = body.get("max_tokens")
        seen["content"] = body["messages"][0]["content"]
        seen["n"] = len(body["messages"])
        return {"ok": True}

    @app.get("/health")
    async def health():
        return {"ok": True}

    # Same order as vLLM's build_app: our middleware first, CORS second (so CORS is outer).
    app.add_middleware(entry._middleware())
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    # A malformed body is forwarded untouched, so the handler's own parse error must surface rather
    # than being swallowed by the test client.
    return TestClient(app, raise_server_exceptions=False), seen


def main():
    client, seen = make_client()

    response = client.post("/v1/chat/completions",
                           json={"max_tokens": 5000,
                                 "messages": [{"role": "user", "content": "hi"}]})
    check(response.status_code == 200, "the request still reaches the handler")
    check(seen["max"] == 64, "the rewritten body is what the handler parses (R cap visible)")

    history = "\n".join(f"Record {i}: the code is K{i:05d}." for i in range(1, 3000))
    question = "\n\nWhat is the code for record 1500?"
    response = client.post("/v1/chat/completions",
                           json={"messages": [{"role": "user", "content": history + question}]})
    check(response.status_code == 200, "a long history is still accepted")
    check(len(seen["content"]) < len(history) // 3,
          f"the handler receives the selected history ({len(seen['content'])} chars)")
    check("record 1500" in seen["content"], "the question survives the rewrite")

    response = client.post("/v1/chat/completions",
                           json={"max_tokens": 32,
                                 "messages": [{"role": "user", "content": "hi"}]})
    check(seen["max"] == 32, "a smaller client budget is not rewritten")

    response = client.post("/v1/chat/completions",
                           json={"messages": [{"role": "user", "content": "A" * 400000}]})
    check(response.status_code == 400, "a history above CTX is rejected with 400")
    check("CTX" in response.text, "the 400 explains the CTX ceiling")

    response = client.post("/v1/chat/completions", content=b"{not json",
                           headers={"Content-Type": "application/json"})
    check(response.status_code >= 400,
          f"a malformed body is passed through untouched (status {response.status_code})")

    response = client.options("/v1/chat/completions",
                              headers={"Origin": "http://example.com",
                                       "Access-Control-Request-Method": "POST"})
    check(response.status_code == 200, "CORS preflight still works (CORS stays outermost)")
    check(response.headers.get("access-control-allow-origin") == "*",
          "the preflight carries the CORS header")

    print(("PASS" if all(checks) else "FAILED") + " entry_body_middleware_test")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
