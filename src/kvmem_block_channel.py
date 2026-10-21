"""Worker -> EngineCore channel: "give me one free aligned block for request R".

Why this cannot be done in the worker: block ownership lives in the scheduler's `block_pool` and
`req_to_blocks`; only EngineCore can hand out a page. vLLM's `scheduler_output` is one-way
(scheduler -> worker), so a restore request needs a side channel.

Scope: recency selection never needs this (a block that slides out of the window can never come
back). Retrieval selection does, because a parked page can become relevant again. The channel is
implemented and tested here so the retrieval path has its control plane ready.
"""
import json
import socket
import threading


DEFAULT_PATH = "/tmp/kvmem_block_channel.sock"


class BlockChannelScheduler:
    """Runs inside EngineCore: serves block requests from the allocator supplied by the plugin.

    Also serves ``compact`` messages (in-request compaction, Route A): the worker tells the
    engine which pages survived a reselect, and the engine truncates the request's
    ``_all_token_ids`` to those pages and rewinds ``num_computed_tokens`` so the scheduler sees no
    gap (the three-book reconciliation). ``compactor(request_id, pages, page_tokens)`` performs it.
    """

    def __init__(self, allocator, path=DEFAULT_PATH, backlog=8, compactor=None):
        self._allocator = allocator      # callable(request_id, count) -> list[int]
        self._compactor = compactor      # callable(request_id, pages, page_tokens) -> bool
        self._path = path
        self._server = None
        self._thread = None
        self._stop = threading.Event()
        self.requests_served = 0
        self.compacts_applied = 0

    def start(self):
        if self._server is not None:
            return self
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            import os
            os.unlink(self._path)
        except FileNotFoundError:
            pass
        self._server.bind(self._path)
        self._server.listen(8)
        self._server.settimeout(0.5)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        return self

    def _serve(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with conn:
                try:
                    payload = conn.recv(65536)
                    if not payload:
                        continue
                    req = json.loads(payload.decode())
                    kind = req.get("kind", "alloc")
                    if kind == "compact":
                        # In-request compaction: the worker selected a working set; the engine
                        # truncates _all_token_ids to those pages and rewinds num_computed_tokens.
                        if self._compactor is None:
                            raise RuntimeError("no compactor installed")
                        ok = bool(self._compactor(req.get("request"),
                                                  [int(p) for p in req.get("pages", [])],
                                                  int(req.get("page_tokens", 0))))
                        self.compacts_applied += 1
                        conn.sendall(json.dumps({"ok": ok}).encode())
                    else:
                        blocks = self._allocator(req.get("request"), int(req.get("count", 1)))
                        self.requests_served += 1
                        conn.sendall(json.dumps({"blocks": list(blocks)}).encode())
                except Exception as exc:                      # never kill the scheduler
                    conn.sendall(json.dumps({"error": str(exc)}).encode())

    def stop(self):
        self._stop.set()
        if self._server is not None:
            self._server.close()
            self._server = None
        try:
            import os
            os.unlink(self._path)
        except OSError:
            pass


class BlockChannel:
    """Runs inside the worker: asks EngineCore for free blocks on demand."""

    def __init__(self, path=DEFAULT_PATH, timeout=5.0):
        self._path = path
        self._timeout = float(timeout)

    def request(self, request_id, count=1):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(self._timeout)
            sock.connect(self._path)
            sock.sendall(json.dumps({"request": request_id, "count": int(count)}).encode())
            data = sock.recv(4096)
        reply = json.loads(data.decode())
        if "error" in reply:
            raise RuntimeError(f"block request failed: {reply['error']}")
        return [int(b) for b in reply["blocks"]]

    def compact(self, request_id, pages, page_tokens):
        """Tell EngineCore that `request_id` compacted to `pages` (in compacted order).

        The engine truncates the request's `_all_token_ids` to the tokens of those pages and
        rewinds `num_computed_tokens`, so the scheduler sees no gap and does not recompute the
        spilled tokens. Returns True when the engine applied it.
        """
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(self._timeout)
            sock.connect(self._path)
            sock.sendall(json.dumps({"kind": "compact", "request": request_id,
                                      "pages": [int(p) for p in pages],
                                      "page_tokens": int(page_tokens)}).encode())
            data = sock.recv(4096)
        reply = json.loads(data.decode())
        if "error" in reply:
            raise RuntimeError(f"compact request failed: {reply['error']}")
        return bool(reply.get("ok", False))
