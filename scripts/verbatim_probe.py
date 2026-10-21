#!/usr/bin/env python3
"""Verbatim-recall probe: the only quality test that can see through the GDN state.

The earlier "random key" probe FAILED to detect sparsity because 48 of the model's 64 layers are
GDN (linear attention): their recurrent state summarises the whole history during prefill, so a
dropped middle block is still "known". This probe asks for a VERBATIM quote from the middle of the
context (a line that cannot be inferred), which only the attention window can supply.
"""
import json
import random
import sys

NOUNS = ["fjord", "tundra", "basalt", "lichen", "gravel", "sorrel", "wicker", "tallow"]


def build(n_lines=4000, probes=(100, 2000, 3800), seed=7):
    rng = random.Random(seed)
    lines, answers = [], {}
    for i in range(1, n_lines + 1):
        # A sentence whose wording is unique and cannot be derived from the index.
        sent = (f"Note {i}: the surveyor wrote that the {rng.choice(NOUNS)} near marker {i} "
                f"was {rng.choice(['crumbling', 'sodden', 'sun-bleached', 'overgrown'])} "
                f"after {rng.randint(3, 19)} winters of {rng.choice(NOUNS)}.")
        lines.append(sent)
        answers[i] = sent
    asked = [i for i in probes if i <= n_lines]
    q = ("\n\nQuote verbatim (exact words, no paraphrase) the sentence for note "
         + ", ".join(f"{i}" for i in asked[:-1]) + (f" and {asked[-1]}" if len(asked) > 1 else "")
         + ". Separate the quotes with a blank line.")
    return {"model": "KVMEM-SPARSE",
            "messages": [{"role": "user", "content": "\n".join(lines) + q}],
            "max_tokens": 160,
            "chat_template_kwargs": {"enable_thinking": False}}, answers, asked


def score(response, answers, asked):
    out = []
    for i in asked:
        gold = answers[i]
        # Ragged but honest: compare the distinctive tail of the quoted sentence.
        key = gold.split(" wrote that the ")[-1][:40]
        out.append({"note": i, "hit": key in response, "probe": key})
    return out


if __name__ == "__main__":
    lines = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    req, answers, asked = build(lines)
    path = f"/tmp/verbatim_{lines}.json"
    json.dump(req, open(path, "w"))
    print(f"wrote {path} (~{sum(len(m['content']) for m in req['messages']) // 3} tokens)")
    for i in asked:
        print(f"  note {i}: {answers[i][:80]}...")
