"""Entry-layer history selection.

vLLM keeps a request's KV blocks indexed by logical position and accounts sequence length
monotonically, so truncating the KV mid-request desynchronises its scheduler (measured: the
`num_scheduled_tokens >= num_logits` assertion, and a hang once spec decoding is off). The one
place a KV cache layer can bound the context without touching that accounting is the ENTRY:
decide how much history to send, and let the engine prefill exactly that.

This module implements that decision. It is deliberately dependency-free and side-effect free so
the gateway and the tests can share it.
"""

from __future__ import annotations

import re


_DEFAULT_SINK = 256      # tokens of the opening context that are always kept
_DEFAULT_RECENT = 1024   # tokens of the tail that are always kept
_CHARS_PER_TOKEN = 3     # conservative estimate for this code/markup-heavy corpus


def _blocks(text: str):
    """Split history into selectable blocks.

    Records of the form `Record 1234: ...` (the shape these workloads use) become one block each;
    anything else falls back to blank-line / line grouping.
    """
    lines = text.splitlines()
    blocks, current = [], []
    record_re = re.compile(r"^\s*(Record|Entry|Item|Row)\s+\d+\s*:")
    for line in lines:
        if record_re.match(line) and current:
            blocks.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    if len(blocks) <= 1:
        # No record structure: group every 8 lines so selection is still possible.
        lines = text.splitlines()
        blocks = ["\n".join(lines[i:i + 8]) for i in range(0, len(lines), 8)]
    return blocks


def question_tail(text: str, keep_chars: int = 600, paragraphs: int = 1) -> str:
    """The part of the last user message that names what to look for.

    History and question often share one message, so scoring the whole message turns every record
    into a "question term" and the ranking degenerates. Taking only the *last paragraph* fixes that
    for one-line questions, but loses the topic when the question spans several paragraphs (measured:
    the topic sat in paragraph 1 and the final paragraph was pure formatting, so no block scored and
    the records that held the answers were dropped).

    So: use up to `paragraphs` trailing paragraphs, capped at `keep_chars` from the end, and use the
    whole message when it is already short enough.
    """
    if len(text) <= keep_chars:
        return text
    parts = [part for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not parts:
        return text[-keep_chars:]
    take = parts[-max(1, int(paragraphs)):]
    joined = "\n\n".join(take)
    return joined[-keep_chars:] if len(joined) > keep_chars else joined


def _score(block: str, question_terms) -> float:
    """Lexical overlap with the question's identifiers.

    Numbers and alphanumeric ids dominate: "record 250" and a six-letter key are exactly the
    tokens a retrieval score must match on. Longer common words are ignored.
    """
    if not question_terms:
        return 0.0
    hits = 0
    for term in question_terms:
        if term in block:
            hits += 1
    # Prefer blocks that match more distinct terms; ties break by shorter block (cheaper).
    return hits / (1.0 + len(block) / 4000.0)


def select_history(text: str, question: str, budget_tokens: int,
                   sink_tokens: int = _DEFAULT_SINK,
                   recent_tokens: int = _DEFAULT_RECENT,
                   chars_per_token: float = _CHARS_PER_TOKEN,
                   method: str = "retrieval"):
    """Return (selected_text, stats).

    Keeps the sink prefix, the recent tail, and the best-scoring middle blocks that fit the
    remaining budget. `stats` reports what was dropped so callers can log/verify the decision.
    """
    blocks = _blocks(text)
    if not blocks:
        return text, {"blocks": 0, "kept": 0, "chars_before": len(text), "chars_after": len(text)}
    chars_budget = max(1, int(budget_tokens)) * max(1, chars_per_token)
    if len(text) <= chars_budget:
        return text, {"blocks": len(blocks), "kept": len(blocks),
                      "chars_before": len(text), "chars_after": len(text), "trimmed": False}

    sink_chars = sink_tokens * chars_per_token
    recent_chars = recent_tokens * chars_per_token

    head, tail = [], []
    used = 0
    for block in blocks:                      # sink prefix
        if used + len(block) > sink_chars:
            break
        head.append(block)
        used += len(block)
    used = 0
    for block in reversed(blocks):            # recent tail
        if used + len(block) > recent_chars:
            break
        tail.insert(0, block)
        used += len(block)

    fixed = "\n".join(head + tail)
    remaining = chars_budget - len(fixed)
    middle = blocks[len(head):len(blocks) - len(tail)] if tail else blocks[len(head):]

    # `recency` keeps the question out of the ranking entirely, which is exactly what a pure
    # recency policy would do; comparing the two makes the retrieval effect observable.
    question_terms = [] if method == "recency" else [
        t for t in re.findall(r"[A-Za-z0-9]{3,}", str(question))
        if not t.lower() in {"the", "and", "for", "with", "answer", "exactly", "keys",
                             "key", "record", "separated", "single", "space"}]
    ranked = sorted(middle, key=lambda b: _score(b, question_terms), reverse=True)

    kept = []
    if remaining > 0:
        for block in ranked:
            if len(block) > remaining:
                continue
            kept.append(block)
            remaining -= len(block)
            if remaining <= 0:
                break

    # Preserve the original order among kept middle blocks.
    kept_set = set(id(b) for b in kept)
    ordered_middle = [b for b in middle if id(b) in kept_set]

    selected = "\n".join(head + ordered_middle + tail)
    stats = {
        "blocks": len(blocks),
        "kept": len(head) + len(ordered_middle) + len(tail),
        "dropped": len(blocks) - (len(head) + len(ordered_middle) + len(tail)),
        "chars_before": len(text),
        "chars_after": len(selected),
        "budget_tokens": int(budget_tokens),
        "estimated_tokens_after": int(len(selected) / max(1e-9, chars_per_token)),
        "trimmed": True,
        "method": method,
    }
    return selected, stats
