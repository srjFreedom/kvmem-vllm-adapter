#!/usr/bin/env python3
"""Entry-layer selection: long histories shrink, asked-about records survive."""
import json
import os
import random
import string
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from kvmem_prompt_select import select_history  # noqa: E402

checks = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    checks.append(bool(cond))


def _fixture(records=900, asked=(250, 800)):
    """Build a self-contained history: one key per record, two of them asked about."""
    import random
    random.seed(11)
    keys, lines = {}, []
    for index in range(1, records + 1):
        key = "".join(random.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6))
        keys[index] = key
        lines.append(f"Record {index}: the access key for record {index} is {key}.")
    body = "You are given a list of records.\n\n" + "\n".join(lines)
    question = (f"Answer with exactly two keys separated by a single space: "
                f"the key for record {asked[0]}, and the key for record {asked[1]}.")
    return body, question, keys


def main():
    body, question, keys = _fixture()
    text = body + "\n\n" + question

    sel, stats = select_history(body, question, 4000)
    check(stats["trimmed"], "a long history is trimmed")
    check(len(sel) < len(body) // 3, f"selection shrank to {stats['estimated_tokens_after']} tokens")
    check(keys[250] in sel and keys[800] in sel,
          "the records named by the question survive the selection")

    # A question about a record far from both ends must still pull its block in.
    sel2, _ = select_history(body, "the key for record 500?", 4000)
    check("Record 500:" in sel2, "a mid-history record is retrieved by name")

    # Short prompts are untouched.
    short = "hello, how are you?"
    sel3, stats3 = select_history(short, short, 4000)
    check(sel3 == short and not stats3.get("trimmed"), "short prompts pass through unchanged")

    print(("PASS" if all(checks) else "FAILED") + " prompt_select_test")
    return 0 if all(checks) else 1


def test_oversized_single_block_never_empties_the_prompt():
    """A pasted file / one minified line is a single huge block.

    Regression: the sink/recent loops used to break on that block and return "", which made the
    entry policy replace the whole prompt with an empty string (engine saw only the question).
    """
    rng = random.Random(11)
    blob = " ".join("".join(rng.choices(string.ascii_lowercase, k=9)) for _ in range(20000))
    text = blob + "\n\n只输出：OK"
    selected, stats = select_history(text, "只输出：OK", 53248, sink_tokens=8192, recent_tokens=2048,
                                     chars_per_token=2.0, method="retrieval")
    assert selected.strip(), "selection must never be empty"
    assert selected.rstrip().endswith("只输出：OK"), "question tail must survive"
    assert len(selected) <= 53248 * 2.0 * 1.02, f"budget exceeded: {len(selected)} chars"
    assert stats["trimmed"] is True
    # the blob is chunked so the middle is rankable and the budget is actually used
    assert len(selected) > 53248 * 2.0 * 0.8, f"budget under-used: {len(selected)} chars"
    print("ok   oversized single block keeps tail and uses budget")


test_oversized_single_block_never_empties_the_prompt()


if __name__ == "__main__":
    raise SystemExit(main())

