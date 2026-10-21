"""KVMem entry policy, shared by the in-process middleware and the optional gateway.

vLLM indexes KV blocks by logical position and accounts sequence length monotonically, so a KV
layer cannot truncate a live request (measured: scheduler assertion, then a hang once speculative
decoding is off). The entry is the one place selection is compatible with the engine: send less
history and the engine prefills exactly what it is given.

Two rules live here, and both are kvmem's, not the client's:

* ``B`` (budget) - how much history is sent; longer histories are reduced to sink + retrieved
  blocks + recent tail.
* ``R`` (generation reserve) - the output ceiling. ``max_tokens`` is capped at R (and stamped when
  the caller omits it); the residency bound ``B + R`` only holds if a request never generates past
  R. With in-request compaction on, the engine recycles the pool while generating, so the ceiling
  becomes the logical budget ``CTX - B`` and R is only the reselect cadence (see ``generation_ceil``).

``CTX`` is kvmem's logical context ceiling, unrelated to the engine window: histories above it are
a configuration error and are rejected rather than silently trimmed.

Keeping this in one module means the in-process middleware, the optional sidecar gateway and the
tests all exercise the same code.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

from kvmem_history_cache import HistoryCache
from kvmem_prompt_select import _blocks, question_tail, select_history

ENV_PREFIX = "KVMEM_ENTRY_"


class ContextTooLong(Exception):
    """The history exceeds ``CTX``, the logical context KVMem was sized for."""


@dataclass
class EntryPolicy:
    budget: int = 65536  # B: history tokens sent to the engine
    gen_reserve: int = 32768  # R: generation ceiling / max_tokens
    ctx: int = 524288  # CTX: logical context ceiling
    sink_tokens: int = 8192
    recent_tokens: int = 2048
    select: str = "retrieval"  # retrieval | recency
    chars_per_token: float = 3.0
    question_tail_chars: int = 2000
    question_paragraphs: int = 3
    history_kib: int = 0  # H tier for parsed history blocks
    history_disk_mib: int = 0  # D tier
    history_disk_path: str = "/tmp/kvmem_history_tier"
    # Route A: in-request compaction. When on, the generation ceiling is the LOGICAL budget
    # (CTX - B), not R: the request must be allowed to generate past R so the engine can reselect
    # (spill/restore) and keep going. R stays the reselect cadence on the engine side.
    inrequest_compaction: bool = False

    @classmethod
    def from_env(cls, env=None) -> "EntryPolicy":
        env = os.environ if env is None else env
        base = cls()
        get_int = lambda name, default: int(env.get(ENV_PREFIX + name, default) or default)
        return cls(
            budget=get_int("B", base.budget),
            gen_reserve=get_int("R", base.gen_reserve),
            ctx=get_int("CTX", base.ctx),
            sink_tokens=get_int("SINK", base.sink_tokens),
            recent_tokens=get_int("RECENT", base.recent_tokens),
            select=str(env.get(ENV_PREFIX + "SELECT", base.select) or base.select),
            chars_per_token=float(env.get(ENV_PREFIX + "CHARS_PER_TOKEN", base.chars_per_token)
                                  or base.chars_per_token),
            question_tail_chars=get_int("QUESTION_TAIL_CHARS", base.question_tail_chars),
            question_paragraphs=get_int("QUESTION_PARAGRAPHS", base.question_paragraphs),
            history_kib=get_int("HISTORY_KIB", base.history_kib),
            history_disk_mib=get_int("HISTORY_DISK_MIB", base.history_disk_mib),
            history_disk_path=str(env.get(ENV_PREFIX + "HISTORY_DISK_PATH", base.history_disk_path)),
            inrequest_compaction=str(env.get(ENV_PREFIX + "INREQUEST_COMPACTION", "")).strip()
                                 .lower() in ("1", "true", "yes", "on"),
        )

    def as_dict(self) -> dict:
        return asdict(self)

    def generation_ceil(self) -> int:
        """Output ceiling in tokens. Normally R; in compaction mode the engine recycles the pool
        while generating, so the ceiling is the logical budget CTX - B (0 disables the cap, which
        leaves the engine default in place)."""
        if self.inrequest_compaction:
            ctx = int(self.ctx or 0)
            return ctx - int(self.budget) if ctx > 0 else 0
        return int(self.gen_reserve)

    def warnings(self) -> list[str]:
        """Configuration coherence checks: an impossible budget silently keeps everything."""
        issues = []
        if self.budget <= 0:
            issues.append("B (KVMEM_ENTRY_B) is 0: no history would be selected")
        elif self.sink_tokens + self.recent_tokens >= self.budget:
            issues.append(
                f"B={self.budget} <= sink+recent={self.sink_tokens + self.recent_tokens}: "
                "the fixed windows alone exceed the budget, so selection cannot reduce anything")
        if self.ctx and self.ctx <= self.budget:
            issues.append(f"CTX={self.ctx} <= B={self.budget}: every request would be rejected")
        if self.select not in ("retrieval", "recency"):
            issues.append(f"unknown select method {self.select!r} (retrieval|recency)")
        return issues

    def make_cache(self) -> HistoryCache | None:
        """H/D tier for parsed history blocks; None when H is zero."""
        host_bytes = int(self.history_kib) * 1024
        if host_bytes <= 0:
            return None
        return HistoryCache(host_bytes=host_bytes,
                            disk_bytes=int(self.history_disk_mib) * 1024 * 1024,
                            disk_dir=self.history_disk_path or None)


def apply_generation_budget(payload: dict, reserve: int) -> dict | None:
    """Cap (or stamp) ``max_tokens`` at R. Returns a stats dict when the payload changed.

    Only ONE field is ever written. vLLM prefers ``max_completion_tokens`` over ``max_tokens``, so
    stamping both would silently override a smaller client budget.
    """
    reserve = int(reserve or 0)
    if reserve <= 0:
        return None
    for field_name in ("max_completion_tokens", "max_tokens"):
        if field_name not in payload:
            continue
        requested = payload.get(field_name)
        if not isinstance(requested, int):
            continue
        if requested <= reserve:
            return None  # the client asked for less: honour it, change nothing
        payload[field_name] = reserve
        return {"kind": "gen_budget", "field": field_name, "requested": requested,
                "applied": reserve}
    # Neither field present: the engine default could exceed R, so impose the cap.
    payload["max_tokens"] = reserve
    return {"kind": "gen_budget", "field": "max_tokens", "requested": None, "applied": reserve}


def _check_ctx(text: str, policy: EntryPolicy) -> None:
    limit = int(policy.ctx or 0)
    if limit <= 0:
        return
    estimated = int(len(text) / max(1e-9, policy.chars_per_token))
    if estimated > limit:
        raise ContextTooLong(
            f"history is ~{estimated} tokens, above kvmem CTX={limit}; "
            f"raise {ENV_PREFIX}CTX or shorten the history")


def rewrite_payload(payload: dict, policy: EntryPolicy,
                    cache: HistoryCache | None = None) -> tuple[bool, dict | None]:
    """Apply the generation budget and, when needed, the history selection.

    Returns ``(changed, stats)``; a rewrite that throws must be caught by the caller and the
    original payload forwarded untouched.
    """
    changed = False
    stats = apply_generation_budget(payload, policy.generation_ceil())
    if stats is not None:
        changed = True

    selection = _select_in_messages(payload, policy, cache)
    if selection is not None:
        changed = True
        stats = _merge(stats, selection)
    return changed, stats


def _merge(first: dict | None, second: dict | None) -> dict | None:
    if first and second:
        merged = dict(first)
        merged.update({k: v for k, v in second.items() if k not in merged})
        merged["selection"] = second
        return merged
    return first or second


def _question_for(messages: list, content_index: int, content: str,
                  policy: EntryPolicy) -> str:
    """The query used to rank blocks.

    Normal client shape: the history is earlier messages and the question is the last user message.
    There the whole last message is the query - no heuristics. When history and question share one
    message (older clients), fall back to the trailing paragraphs of that message.
    """
    last_user = None
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            last_user = message
            break
    if last_user is not None and messages.index(last_user) != content_index:
        text = last_user.get("content")
        if isinstance(text, str) and text.strip():
            return text[-policy.question_tail_chars:] if len(text) > policy.question_tail_chars else text
    return question_tail(content, policy.question_tail_chars, policy.question_paragraphs)


def _select_in_messages(payload: dict, policy: EntryPolicy,
                        cache: HistoryCache | None) -> dict | None:
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    threshold = max(1, policy.budget) * max(1e-9, policy.chars_per_token)
    best = None
    for index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, str) or len(content) <= threshold:
            continue
        _check_ctx(content, policy)
        if cache is not None:
            cached = cache.get_blocks(content)
            if cached is not None:
                content = "\n".join(cached)
            else:
                cache.put_blocks(content, _blocks(content))
        selected, stats = select_history(
            content, _question_for(messages, index, content, policy),
            policy.budget,
            sink_tokens=policy.sink_tokens,
            recent_tokens=policy.recent_tokens,
            chars_per_token=policy.chars_per_token,
            method=policy.select)
        if stats.get("trimmed") and selected.strip():
            message["content"] = selected
            stats["message_index"] = index
            stats["kind"] = "prompt_selected"
            if best is None or stats["chars_after"] < best["chars_after"]:
                best = stats
    return best
