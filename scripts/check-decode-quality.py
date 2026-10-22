#!/usr/bin/env python3
"""Decode-quality gate for the live KVMem+vLLM engine (needs a running server).

Why this exists: every earlier acceptance test asked a question whose answer was a few
tokens long (a needle key, a number). Those pass even when the DECODE path is corrupt,
because the first step's logits are still produced from a correct prompt KV. The failure
that reached the user (a 5,000-character repetition loop in the chat UI) is invisible to
them: the model degenerates from the *second* decode step on, and only free-form output
(thinking + prose) shows it.

Gate: greedy output must stay coherent and must not repeat an n-gram.
    python3 scripts/check-decode-quality.py [port] [max_tokens]

Exit code 0 = all cases passed, 1 = at least one degenerate case.
"""
import json
import sys
import time
import urllib.request
from collections import Counter

PORT = sys.argv[1] if len(sys.argv) > 1 else "18200"
MAX_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 400
BASE = f"http://127.0.0.1:{PORT}/v1/chat/completions"
MODEL = "Qwen3.8-27B-Coder390-EfficientThink-FP8-MTP"

CASES = [
    ("short-zh", "用一句话介绍长江。", 0.0),
    ("essay-zh", "写一段 200 字的说明文，介绍长江的水文特征。", 0.0),
    ("code-py", "写一个 Python 函数，返回列表中出现次数最多的 3 个元素，并给出用例。", 0.0),
    ("short-sampled", "用一句话介绍长江。", 0.7),
]

# A healthy answer repeats no long window; a degenerate loop repeats one window many times.
WINDOW = 12
MAX_REPEATS = 10


def call(prompt, max_tokens, temperature):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature,
            "top_p": 1.0 if temperature == 0.0 else 0.95}
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        out = json.loads(resp.read())
    msg = out["choices"][0]["message"]
    text = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
    return text, time.time() - t0, (out.get("usage") or {}).get("completion_tokens"), \
        out["choices"][0].get("finish_reason")


def worst_repeat(text):
    if len(text) < WINDOW * 2:
        return 0, ""
    grams = Counter(text[i:i + WINDOW] for i in range(len(text) - WINDOW))
    gram, count = grams.most_common(1)[0]
    return count, gram


def main():
    failed = 0
    for name, prompt, temp in CASES:
        text, dt, ct, fr = call(prompt, MAX_TOKENS, temp)
        count, gram = worst_repeat(text)
        ok = count <= MAX_REPEATS and ct and ct > 0
        failed += 0 if ok else 1
        print(json.dumps({
            "case": name, "temperature": temp, "ok": ok,
            "wall_s": round(dt, 1), "completion_tokens": ct, "finish": fr,
            "max_window_repeats": count, "repeated": gram,
            "head": text[:120].replace("\n", "\\n"),
        }, ensure_ascii=False))
    print(f"\n{'PASS' if not failed else 'FAIL'} {len(CASES) - failed}/{len(CASES)} "
          f"(window={WINDOW}, max_repeats={MAX_REPEATS})")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
