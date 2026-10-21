"""KVMem entry policy inside the vLLM API server process.

One service, one port, no sidecar. Registered as a `vllm.endpoint_plugins` entry point (see
`plugin/pyproject.toml`), so vLLM loads it into the API server process and calls
`attach_router(app)`.

Why the entry and not the KV layer: vLLM indexes KV blocks by logical position and accounts
sequence length monotonically, so truncating a live request desynchronises its scheduler (measured:
`num_scheduled_tokens >= num_logits` assertion, then a hang with speculative decoding off). Choosing
how much history to send *before* the engine sees it is compatible with that accounting, and it is
the same decision kvmem-llama / kvmem-ninfer make inside their engines.

Why a serving-class hook rather than ASGI middleware: rewriting the request body means consuming the
ASGI `receive` channel and replaying it, and in this app that hangs the request once the router hands
it to the chat handler (measured: the middleware logs "calling downstream", the handler is entered,
and no response ever leaves; without the middleware the same request answers in 0.24 s). Applying
the policy to the *parsed* request needs no ASGI plumbing at all: streaming, CORS and the error paths
stay exactly as vLLM ships them.

The hook is one wrapper on `OpenAIServingChat.create_chat_completion`, installed by our plugin - the
same pattern the adapter already uses for the engine-side runner hooks.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace

_ADAPTER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (os.path.join(_ADAPTER_ROOT, "build", "python"), os.path.join(_ADAPTER_ROOT, "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from kvmem_entry_policy import (  # noqa: E402
    ContextTooLong,
    EntryPolicy,
    rewrite_payload,
)

try:  # vLLM configures the `vllm.*` loggers; fall back to stdlib when imported standalone.
    from vllm.logger import init_logger

    _log = init_logger("vllm.kvmem.entry")
except Exception:  # pragma: no cover - only when vLLM is unavailable
    import logging

    _log = logging.getLogger("kvmem.entry")

STATE = {"calls": 0, "rewrites": 0, "ctx_rejected": 0, "errors": 0}
_POLICY = None
_CACHE = None


def apply_to_messages(messages, policy: EntryPolicy, cache=None) -> dict | None:
    """B selection over a list of chat messages, in place. Returns stats or None."""
    if not isinstance(messages, list) or not messages:
        return None
    payload = {"messages": [dict(m) if isinstance(m, dict) else m for m in messages]}
    _, selection = rewrite_payload(payload, policy, cache)
    if isinstance(selection, dict) and selection.get("kind") != "prompt_selected":
        selection = selection.get("selection")
    if not (isinstance(selection, dict) and selection.get("kind") == "prompt_selected"):
        return None
    for index, rewritten in enumerate(payload["messages"]):
        if isinstance(rewritten, dict) and isinstance(messages[index], dict):
            messages[index]["content"] = rewritten.get("content")
    return selection


def _budget_field(request):
    """vLLM reads `max_completion_tokens` when set and falls back to the deprecated `max_tokens`."""
    if getattr(request, "max_completion_tokens", None) is not None:
        return request.max_completion_tokens
    return getattr(request, "max_tokens", None)


def apply_policy(request, policy: EntryPolicy, cache=None) -> dict | None:
    """Apply B selection and the R ceiling to an already parsed chat request.

    Raises ContextTooLong when the history is above CTX (a configuration error, not a silent trim).
    """
    STATE["calls"] += 1
    stats: dict = {}

    if policy.gen_reserve > 0:
        requested = _budget_field(request)
        if requested is None or requested > policy.gen_reserve:
            # R is kvmem's ceiling, not a client preference. Written to the field vLLM actually
            # reads, and only that one, so a smaller `max_tokens` can never be silently overridden.
            request.max_completion_tokens = policy.gen_reserve
            stats["gen_budget"] = {"field": "max_completion_tokens", "requested": requested,
                                   "applied": policy.gen_reserve}

    selection = apply_to_messages(getattr(request, "messages", None) or [], policy, cache)
    if selection:
        stats["selection"] = selection

    if stats:
        STATE["rewrites"] += 1
        _log.info("kvmem entry: %s", json.dumps(
            {"kind": "entry_rewrite", **STATE, **stats}, separators=(",", ":"), default=str))
    return stats or None


# ============================================================================
# Standard path: Starlette middleware that rewrites the body through the
# `request._body` cache (the documented way - a replacement receive would leave
# downstream readers waiting on an already-drained stream).
# ============================================================================
CHAT_SUFFIX = "/v1/chat/completions"


def _resolve_tokenizer(app):
    """Find the model tokenizer reachable from the app state (None when unavailable).

    B is a token budget, so estimating it with a fixed chars/token ratio over-sends by however far
    that ratio is off (measured: 1.83x on code-like history). When the tokenizer is reachable we
    calibrate the ratio per request from the actual history.
    """
    state = getattr(app, "state", None)
    for path in (("openai_serving_chat", "renderer", "tokenizer"),
                 ("openai_serving_completion", "renderer", "tokenizer"),
                 ("engine_client", "renderer", "tokenizer")):
        obj = state
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and callable(getattr(obj, "encode", None)):
            return obj
    return None


def _calibrated(policy: EntryPolicy, messages, counter):
    """Return a policy whose chars/token ratio is measured from this request's history."""
    if counter is None or not isinstance(messages, list):
        return policy
    longest = ""
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str) and len(content) > len(longest):
            longest = content
    threshold = max(1, policy.budget) * max(1e-9, policy.chars_per_token)
    if len(longest) <= threshold:
        return policy
    try:
        tokens = len(counter.encode(longest, add_special_tokens=False))
    except TypeError:
        tokens = len(counter.encode(longest))
    except Exception:
        return policy
    if tokens <= 0:
        return policy
    return replace(policy, chars_per_token=len(longest) / tokens)


def _install_middleware_class():
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse

    class KvmemEntryMiddleware(BaseHTTPMiddleware):
        def __init__(self, app):
            super().__init__(app)
            self.policy = EntryPolicy.from_env()
            self.cache = self.policy.make_cache()
            self._counter = None
            _log.info("kvmem entry policy (middleware): %s", json.dumps(self.policy.as_dict()))
            for issue in self.policy.warnings():
                _log.warning("kvmem entry policy: %s", issue)

        @staticmethod
        def _matches(request) -> bool:
            if request.method != "POST":
                return False
            if not (request.url.path or "").endswith(CHAT_SUFFIX):
                return False
            ctype = (request.headers.get("content-type") or "").lower()
            return "application/json" in ctype or ctype == ""

        async def dispatch(self, request, call_next):
            if not self._matches(request):
                return await call_next(request)
            body = await request.body()  # consumed once, and cached on this Request
            try:
                payload = json.loads(body)
            except Exception:
                return await call_next(request)  # the cache still holds the original bytes
            if self._counter is None:
                self._counter = _resolve_tokenizer(request.app)
            policy = _calibrated(self.policy, payload.get("messages"), self._counter)
            try:
                changed, stats = rewrite_payload(payload, policy, self.cache)
            except ContextTooLong as exc:
                STATE["ctx_rejected"] += 1
                _log.warning("kvmem entry: rejected request: %s", exc)
                return JSONResponse({"error": {"message": str(exc), "type": "BadRequestError",
                                               "code": 400}}, status_code=400)
            except Exception:
                STATE["errors"] += 1
                _log.warning("kvmem entry: policy skipped for this request", exc_info=True)
                return await call_next(request)

            if not changed:
                return await call_next(request)

            new_body = json.dumps(payload).encode()
            # Hand the rewritten bytes to downstream readers through the cache, and keep
            # content-length honest (the body no longer matches the original header).
            request._body = new_body
            request.scope["headers"] = [
                (k, v) for k, v in request.scope["headers"]
                if k not in (b"content-length", b"transfer-encoding")
            ] + [(b"content-length", str(len(new_body)).encode())]
            STATE["rewrites"] += 1
            if stats:
                _log.info("kvmem entry: %s", json.dumps(
                    {"kind": "entry_rewrite", **STATE, **(stats or {})},
                    separators=(",", ":"), default=str))
            return await call_next(request)

    return KvmemEntryMiddleware


KvmemEntryMiddleware = None


def _middleware():
    global KvmemEntryMiddleware
    if KvmemEntryMiddleware is None:
        KvmemEntryMiddleware = _install_middleware_class()
    return KvmemEntryMiddleware


def install(serving_cls=None) -> bool:
    """Wrap the chat serving class so every chat request passes through the entry policy."""
    global _POLICY, _CACHE
    try:
        if serving_cls is None:
            from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat

            serving_cls = OpenAIServingChat
    except Exception as exc:  # pragma: no cover - vLLM not importable
        _log.warning("kvmem entry: cannot install (no vLLM serving class): %s", exc)
        return False

    original = serving_cls.create_chat_completion
    if getattr(original, "_kvmem_entry_patched", False):
        return True

    _POLICY = EntryPolicy.from_env()
    _CACHE = _POLICY.make_cache()
    _log.info("kvmem entry policy (in-process): %s", json.dumps(_POLICY.as_dict()))
    for issue in _POLICY.warnings():
        _log.warning("kvmem entry policy: %s", issue)

    async def create_chat_completion(self, request, raw_request=None, *args, **kwargs):
        try:
            counter = getattr(getattr(self, "renderer", None), "tokenizer", None)
            apply_policy(request, _calibrated(_POLICY, getattr(request, "messages", None), counter),
                         _CACHE)
        except ContextTooLong as exc:
            STATE["ctx_rejected"] += 1
            _log.warning("kvmem entry: rejected request: %s", exc)
            create_error = getattr(self, "create_error_response", None)
            if callable(create_error):
                # The signature wants an HTTPStatus, not an int.
                from http import HTTPStatus

                return create_error(message=str(exc), err_type="BadRequestError",
                                    status_code=HTTPStatus.BAD_REQUEST)
            raise
        except Exception:
            STATE["errors"] += 1
            _log.warning("kvmem entry: policy skipped for this request", exc_info=True)
        return await original(self, request, raw_request, *args, **kwargs)

    create_chat_completion._kvmem_entry_patched = True
    serving_cls.create_chat_completion = create_chat_completion
    _log.info("kvmem entry: chat serving hook installed on %s", serving_cls.__name__)
    return True


class KvmemEntryPlugin:
    """`vllm.endpoint_plugins` entry point: install the entry policy, keep the HTTP surface intact.

    No routes and no middleware: the policy rides on the parsed request, so streaming, CORS and the
    error paths stay exactly as vLLM ships them.
    """

    name = "kvmem_entry"
    required_tasks = ("generate",)

    def attach_router(self, app) -> None:  # noqa: D401 - protocol hook
        # KVMem Web UI: the static build talks to the API with relative URLs (/v1/models,
        # ./v1/chat/completions), so serving it from the same port needs no proxy. The mount is
        # added last, after every core router, so /v1/*, /health and /docs keep priority.
        ui_dir = os.environ.get("KVMEM_UI_DIR", "/home/srj/kvmem-ui")
        if ui_dir and os.path.isdir(ui_dir):
            try:
                from starlette.staticfiles import StaticFiles

                app.mount("/", StaticFiles(directory=ui_dir, html=True), name="kvmem-ui")
                _log.info("kvmem entry: static UI mounted from %s at /", ui_dir)
            except Exception:
                _log.warning("kvmem entry: cannot mount the UI from %s", ui_dir, exc_info=True)

        mode = os.environ.get("KVMEM_ENTRY_MODE", "middleware").strip().lower()
        if mode == "hook":
            install()
            return
        # Standard path: the framework-sanctioned interception point (a Starlette middleware),
        # added from the official vLLM endpoint-plugin hook.
        app.add_middleware(_middleware())
        _log.info("kvmem entry: body-rewriting middleware installed (mode=middleware)")

    async def init_state(self, engine_client, state, args) -> None:  # noqa: D401 - protocol hook
        _log.info("kvmem entry plugin ready (in-process, no sidecar)")
