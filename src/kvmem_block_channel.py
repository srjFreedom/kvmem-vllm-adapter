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

    Compact messages carry original logical page IDs and physical device slots separately.
    Ordinary compaction reconciles resident allocation without changing the logical token book.
    Explicit reconstruction queues a replacement prompt for the scheduler's safe output boundary.
    """

    def __init__(self, allocator, path=DEFAULT_PATH, backlog=8, compactor=None,
                 session_sink=None, replayer=None):
        self._allocator = allocator      # callable(request_id, count) -> list[int]
        self._compactor = compactor      # callable(request_id, pages, page_tokens) -> bool
        # callable(payload) -- a finished session was parked in the worker; EngineCore keeps the
        # routing metadata so it can decide a later resume without touching any KV.
        self._session_sink = session_sink
        self._replayer = replayer        # callable(request_id, visible, n) -> dict
        self._path = path
        self._server = None
        self._thread = None
        self._stop = threading.Event()
        self.requests_served = 0
        self.compacts_applied = 0
        self.replays_applied = 0
        self.sessions_parked = 0

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
                    if kind == "replay":
                        # KVMem QueryReplay: rewind BOTH books by N so the scheduler re-schedules
                        # the last N tokens against the freshly installed working set.
                        if self._replayer is None:
                            raise RuntimeError("no replayer installed")
                        reply = self._replayer(req.get("request"),
                                               int(req.get("visible_tokens", 0) or 0),
                                               int(req.get("replay_tokens", 0) or 0))
                        self.replays_applied += 1
                        conn.sendall(json.dumps(reply if isinstance(reply, dict)
                                                else {"ok": bool(reply)}).encode())
                        continue
                    if kind == "compact":
                        # Preserve typed reconstruction replies; bool(dict) hides refusals and
                        # drops the queued/tokens fields needed by both TP workers.
                        if self._compactor is None:
                            raise RuntimeError("no compactor installed")
                        ok = self._compactor(req.get("request"),
                                                  [int(p) for p in req.get("pages", [])],
                                                  int(req.get("page_tokens", 0)),
                                                  int(req.get("visible_tokens", 0) or 0),
                                                  bool(req.get("rebuild", False)),
                                                  int(req.get("block_pages", 0) or 0),
                                                  req.get("physical_pages", []))
                        self.compacts_applied += 1
                        # `ok` = the engine's book is now consistent with the worker's view;
                        # `truncated` = it actually shortened (so installing the view is needed).
                        if isinstance(ok, dict):
                            conn.sendall(json.dumps(ok).encode())
                        else:
                            conn.sendall(json.dumps({"ok": bool(ok),
                                                    "truncated": bool(ok)}).encode())
                    elif kind == "session_park":
                        if self._session_sink is not None:
                            self._session_sink(req)
                        self.sessions_parked += 1
                        conn.sendall(json.dumps({"ok": True}).encode())
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

    def session_park(self, payload):
        """Announce a parked session (block hashes + geometry) to EngineCore."""
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(self._timeout)
            sock.connect(self._path)
            body = dict(payload)
            body["kind"] = "session_park"
            sock.sendall(json.dumps(body).encode())
            data = sock.recv(4096)
        reply = json.loads(data.decode())
        if "error" in reply:
            raise RuntimeError(f"session park failed: {reply['error']}")
        return bool(reply.get("ok", False))

    def replay(self, request_id, visible_tokens, replay_tokens):
        """Ask EngineCore to re-schedule the last `replay_tokens` tokens (KVMem QueryReplay).

        Prefill-time compaction keeps only sink+newest (no query exists yet), so the tokens that were
        prefilled while the middle was evicted saw an approximate history. After the query-driven
        reselect installs the real working set, rewinding BOTH books by N makes the scheduler
        prefilled those N tokens again - this time against the full B - which refreshes their KV and
        the recurrent state right before sampling.
        """
        n = int(replay_tokens or 0)
        if n <= 0:
            return None
        body = {"kind": "replay", "request": request_id,
                "visible_tokens": int(visible_tokens), "replay_tokens": n}
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(self._timeout)
            sock.connect(self._path)
            sock.sendall(json.dumps(body).encode())
            data = sock.recv(4096)
        reply = json.loads(data.decode())
        if "error" in reply:
            raise RuntimeError(f"replay request failed: {reply['error']}")
        return reply

    def compact(self, request_id, pages, page_tokens, visible_tokens=0, rebuild=False,
                block_pages=0, physical_pages=None):
        """Tell EngineCore that `request_id` compacted to `pages` (in compacted order).

        Logical ``pages`` select original tokens; ``physical_pages`` identifies allocator slots.
        Ordinary compaction rewinds residency. Reconstruction queues a fresh prompt admission.
        Return the full acknowledgement, including explicit refusals and pending status.
        """
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(self._timeout)
            sock.connect(self._path)
            sock.sendall(json.dumps({"kind": "compact", "request": request_id,
                                      "pages": [int(p) for p in pages],
                                      "page_tokens": int(page_tokens),
                                      "visible_tokens": int(visible_tokens or 0),
                                      "rebuild": bool(rebuild),
                                      "block_pages": int(block_pages or 0),
                                       "physical_pages": list(physical_pages or [])}).encode())
            data = sock.recv(4096)
        reply = json.loads(data.decode())
        if "error" in reply:
            raise RuntimeError(f"compact request failed: {reply['error']}")
        # `ok` also covers the "nothing to truncate" case; `truncated` is the signal the worker
        # needs to decide whether its compacted view may be installed at all.
        return reply
