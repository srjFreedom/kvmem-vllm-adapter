#!/usr/bin/env python3
"""Entry-layer selection: long histories shrink, asked-about records survive."""
import json
import os
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


if __name__ == "__main__":
    raise SystemExit(main())
