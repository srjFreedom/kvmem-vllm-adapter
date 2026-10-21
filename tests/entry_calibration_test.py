#!/usr/bin/env python3
"""B is a token budget: calibrate the chars/token ratio from the request's real history.

Measured before this fix: a code-like history was estimated at 65,611 tokens with the fixed
3.0 chars/token ratio but actually tokenized to 120,321 - a 1.8x over-send that also breaks the
N x (B + R) residency math.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "plugin"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import kvmem_entry_middleware as entry  # noqa: E402
from kvmem_entry_policy import EntryPolicy  # noqa: E402

checks = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    checks.append(bool(cond))


class TwoCharTokenizer:
    """Fake tokenizer: exactly two characters per token."""

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text) // 2))


class BrokenTokenizer:
    def encode(self, *a, **kw):
        raise RuntimeError("no tokenizer")


def main():
    policy = EntryPolicy(budget=4096, sink_tokens=256, recent_tokens=512, ctx=10 ** 9)
    history = "\n".join(f"Record {i}: the code is K{i:05d}." for i in range(1, 4000))
    messages = [{"role": "user", "content": history + "\n\nWhat is the code for record 2000?"}]

    check(entry._calibrated(policy, messages, None) is policy,
          "without a tokenizer the policy is used unchanged (estimate fallback)")

    calibrated = entry._calibrated(policy, messages, TwoCharTokenizer())
    check(abs(calibrated.chars_per_token - 2.0) < 0.05,
          f"the ratio is measured from the history ({calibrated.chars_per_token:.2f} ~ 2.0)")

    check(entry._calibrated(policy, messages, BrokenTokenizer()) is policy,
          "a failing tokenizer falls back to the estimate instead of raising")

    short = [{"role": "user", "content": "hi"}]
    check(entry._calibrated(policy, short, TwoCharTokenizer()) is policy,
          "a short history is not worth a tokenizer pass")

    # The calibrated run must land near the budget, the uncalibrated one overshoots.
    def sent_tokens(p):
        payload = json.loads(json.dumps({"messages": messages}))
        _, stats = entry.rewrite_payload(payload, p, None)
        content = payload["messages"][0]["content"]
        return len(content) // 2, stats

    estimated_chars, _ = sent_tokens(policy)
    measured_tokens, _ = sent_tokens(calibrated)
    check(measured_tokens <= policy.budget + policy.sink_tokens + policy.recent_tokens,
          f"the calibrated request fits B+sink+recent ({measured_tokens} tokens)")
    check(estimated_chars > measured_tokens * 1.3,
          f"the fixed ratio over-sent before the fix ({estimated_chars} vs {measured_tokens} tokens)")

    print(("PASS" if all(checks) else "FAILED") + " entry_calibration_test")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
