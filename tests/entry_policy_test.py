#!/usr/bin/env python3
"""Entry policy: B selection, R as the max_tokens cap, CTX ceiling - one implementation.

The middleware inside the vLLM API server and the optional sidecar gateway both call this module,
so these three rules cannot drift apart between deployment shapes.
"""
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kvmem_entry_policy import (  # noqa: E402
    ContextTooLong,
    EntryPolicy,
    apply_generation_budget,
    rewrite_payload,
)

checks = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    checks.append(bool(cond))


def main():
    # --- R is kvmem's max_tokens cap -------------------------------------------------
    pl = {"max_tokens": 500}
    stats = apply_generation_budget(pl, 64)
    check(pl["max_tokens"] == 64 and stats["applied"] == 64, "a larger client budget is capped to R")

    pl = {"max_tokens": 32}
    check(apply_generation_budget(pl, 64) is None and pl["max_tokens"] == 32,
          "a smaller client budget is honoured untouched")

    pl = {}
    stats = apply_generation_budget(pl, 64)
    check(pl.get("max_tokens") == 64 and stats["requested"] is None,
          "an omitted budget is stamped with R")

    pl = {"max_completion_tokens": 5000, "max_tokens": 5000}
    apply_generation_budget(pl, 64)
    check(pl["max_completion_tokens"] == 64,
          "the field vLLM actually reads (max_completion_tokens) is the one capped")

    pl = {"max_completion_tokens": 8, "max_tokens": 5000}
    apply_generation_budget(pl, 64)
    check(pl["max_completion_tokens"] == 8 and pl["max_tokens"] == 5000,
          "a small max_completion_tokens wins and nothing else is rewritten")

    # --- B: history selection --------------------------------------------------------
    history = "\n".join(f"Record {i}: the access key for record {i} is K{i:06d}." for i in range(1, 3000))
    question = "\n\nAnswer with exactly two keys: record 250 and record 2800."
    policy = EntryPolicy(budget=4096, sink_tokens=256, recent_tokens=512, ctx=200000)
    payload = {"messages": [{"role": "user", "content": history + question}]}
    changed, stats = rewrite_payload(payload, policy)
    text = payload["messages"][0]["content"]
    check(changed and stats is not None, "a long history is rewritten")
    check("K000250" in text and "K002800" in text,
          "the records the question names survive the selection")
    check(len(text) < len(history) // 3, f"selection shrank the prompt ({len(text)} chars)")
    check("record 2800" in text, "the question itself is preserved")

    # --- CTX is a rejection, not a silent trim ---------------------------------------
    policy = EntryPolicy(budget=4096, sink_tokens=256, recent_tokens=512, ctx=1000)
    big = {"messages": [{"role": "user", "content": "A" * 60000}]}
    try:
        rewrite_payload(big, policy)
        check(False, "a history above CTX is rejected")
    except ContextTooLong:
        check(True, "a history above CTX is rejected")

    # --- policy coherence warnings ---------------------------------------------------
    check(EntryPolicy().warnings() == [], "the default policy is coherent")
    bad = EntryPolicy(budget=4096, sink_tokens=8192, recent_tokens=2048)
    check(any("sink+recent" in w for w in bad.warnings()),
          "B smaller than sink+recent is reported as incoherent")
    check(any("rejected" in w for w in EntryPolicy(budget=8192, ctx=4096).warnings()),
          "CTX <= B is reported as incoherent")

    # --- env driven ------------------------------------------------------------------
    policy = EntryPolicy.from_env({"KVMEM_ENTRY_B": "8192", "KVMEM_ENTRY_R": "1024",
                                   "KVMEM_ENTRY_CTX": "65536", "KVMEM_ENTRY_SELECT": "recency"})
    check((policy.budget, policy.gen_reserve, policy.ctx, policy.select)
          == (8192, 1024, 65536, "recency"), "the middleware reads its policy from KVMEM_ENTRY_*")

    # --- in-request compaction: the generation ceiling becomes CTX - B -----------------
    plain = EntryPolicy(budget=36864, gen_reserve=16384, ctx=524288)
    check(plain.inrequest_compaction is False and plain.generation_ceil() == 16384,
          "without compaction the ceiling stays R")
    comp = EntryPolicy(budget=36864, gen_reserve=16384, ctx=524288, inrequest_compaction=True)
    check(comp.generation_ceil() == 524288 - 36864,
          "with compaction the ceiling is the logical budget CTX - B")
    check(EntryPolicy(budget=36864, gen_reserve=16384, ctx=0,
                      inrequest_compaction=True).generation_ceil() == 0,
          "compaction with no CTX releases the cap (engine default applies)")
    pl = {"max_tokens": 999999}
    _, stats = rewrite_payload(pl, comp)
    check(pl["max_tokens"] == 524288 - 36864 and stats.get("kind") == "gen_budget"
          and stats.get("applied") == 524288 - 36864,
          "rewrite_payload caps at CTX - B, not R, when compaction is on")
    pl = {"max_tokens": 1000}
    rewrite_payload(pl, comp)
    check(pl["max_tokens"] == 1000, "a smaller client budget is still honoured under compaction")
    pol = EntryPolicy.from_env({"KVMEM_ENTRY_INREQUEST_COMPACTION": "1"})
    check(pol.inrequest_compaction is True,
          "KVMEM_ENTRY_INREQUEST_COMPACTION turns the flag on")
    check(EntryPolicy.from_env({}).inrequest_compaction is False,
          "the flag defaults off")

    # The last user message is the query when history and question are separate messages: the
    # records the question names must survive even when the topic sits in the question's first
    # paragraph (the trailing-paragraph heuristic alone dropped them - measured).
    random.seed(3)
    lines = []
    for i in range(1, 12000):
        if i in (4200, 7800):
            key = "".join(random.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6))
            if i == 4200:
                first = key
            else:
                second = key
            lines.append(f"Record {i}: ZX-9 firewall migration note. The access key is {key}.")
        else:
            lines.append(f"Record {i}: unrelated maintenance log entry for cluster {i % 97}.")
    topic = ("The ZX-9 firewall migration project is the subject here. " * 8)
    boiler = ("Formatting rules: keep the answer to two short lines. " * 4)
    question = topic + "\n" + boiler + "\n\nReturn the two codes for it now."
    policy = EntryPolicy(budget=65536, sink_tokens=8192, recent_tokens=2048, chars_per_token=1.64)
    payload = {"messages": [{"role": "user", "content": "\n".join(lines)},
                            {"role": "user", "content": question}]}
    rewrite_payload(payload, policy)
    selected = payload["messages"][0]["content"]
    check(first in selected and second in selected,
          "a separate last user message is used as the query (topic in its first paragraph)")

    print(("PASS" if all(checks) else "FAILED") + " entry_policy_test")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
