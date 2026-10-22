# The entry policy must bound the prompt in REAL tokens, and it must see text that arrives as
# OpenAI content parts. Measured live before this fix: a Chinese history was trimmed with the
# 3.0 chars/token estimate, so the engine received 143,216 tokens on a B=53,248 configuration and
# every in-request compaction was then refused for cutting below the prompt (garbled answers).
import json
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kvmem_entry_policy import EntryPolicy, rewrite_payload  # noqa: E402

checks = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    checks.append(bool(cond))


class ChineseTokenizer:
    """Dense Chinese: ~1.1 characters per token (the ratio that broke the live run)."""

    def __init__(self, chars_per_token=1.1):
        self.ratio = float(chars_per_token)

    def encode(self, text, add_special_tokens=False):
        return list(range(max(1, int(len(text) / self.ratio))))


def history(chars):
    line = "第{i}条记录：节点状态正常，磁盘水位稳定，延迟没有明显变化，偶尔有重试。\n"
    out, i = [], 0
    total = 0
    while total < chars:
        i += 1
        piece = line.format(i=i)
        out.append(piece)
        total += len(piece)
    return "".join(out)


def tokens(counter, text):
    return len(counter.encode(text))


def main():
    counter = ChineseTokenizer()
    policy = EntryPolicy(budget=2048, gen_reserve=512, ctx=10 ** 6,
                         sink_tokens=128, recent_tokens=128, chars_per_token=3.0)

    text = history(3 * 2048 * 3)                 # ~3x B in ESTIMATED tokens, ~8x in real ones

    # 1) A plain string history is trimmed to B REAL tokens (not just to the estimate).
    payload = {"messages": [{"role": "user", "content": text},
                            {"role": "user", "content": "总结上面的记录。"}]}
    changed, stats = rewrite_payload(payload, policy, None, counter)
    selected = payload["messages"][0]["content"]
    check(changed and stats and stats.get("trimmed"), "a long history is trimmed")
    check(isinstance(selected, str), "a string content stays a string")
    check(tokens(counter, selected) <= policy.budget,
          f"the selected history fits B in REAL tokens ({tokens(counter, selected)} <= {policy.budget})")

    # Without the tokenizer the estimate is used: this is the old (overshooting) behaviour, and it
    # is exactly why the engine must also refuse to compact a prompt above B.
    payload2 = {"messages": [{"role": "user", "content": text}]}
    rewrite_payload(payload2, policy, None, None)
    est = payload2["messages"][0]["content"]
    check(tokens(counter, est) > policy.budget,
          "without a tokenizer the estimate still over-sends (engine guard is required)")

    # 2) Multipart content (text + image) must be trimmed too, and the image must survive.
    payload3 = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]}]}
    changed3, stats3 = rewrite_payload(payload3, policy, None, counter)
    parts = payload3["messages"][0]["content"]
    check(changed3 and stats3 is not None, "a multipart content is trimmed")
    check(isinstance(parts, list) and len(parts) == 2, f"the image part survives ({len(parts)} parts)")
    check(parts[0].get("type") == "text", "the selected text is the first part")
    check(parts[1].get("type") == "image_url", "the image part is preserved verbatim")
    check(tokens(counter, parts[0]["text"]) <= policy.budget,
          f"the multipart text fits B in REAL tokens ({tokens(counter, parts[0]['text'])})")

    # 3) A short multipart message is left alone.
    short = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "你好"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]}]}
    changed4, _ = rewrite_payload(short, policy, None, counter)
    check(not changed4 or short["messages"][0]["content"][0]["text"] == "你好",
          "a short multipart message is not touched")

    print(("PASS" if all(checks) else "FAILED") + " entry_parts_test")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
