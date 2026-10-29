#!/usr/bin/env python3
"""Prepare/run a centered-needle chat request and enforce measured beyond-pool admission."""
import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--model", default="Qwen3.5-Coder-NVFP4-MTP")
    parser.add_argument("--target-tokens", type=int, default=196608)
    parser.add_argument("--pool-tokens", type=int, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18200")
    parser.add_argument("--output", default="/tmp/kvmem-reconstruction-needle")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    filler = "Archive entry: ordinary maintenance completed; no special access code recorded.\n"
    needle = "\nThe unique record KEY-MID has access code ZX-4417-QJ7.\n"
    question = "\nWhat is the access code for KEY-MID? Reply with only the exact code."
    system = "Read the archive and answer the question using its unique record."

    def messages(n):
        return [{"role": "system", "content": system},
                {"role": "user", "content": filler * n + needle + filler * n + question}]

    def encode(msg):
        encoded = tok.apply_chat_template(msg, tokenize=True, add_generation_prompt=True,
                                          enable_thinking=False)
        return encoded["input_ids"] if hasattr(encoded, "keys") else encoded

    per_line = max(1, len(tok.encode(filler, add_special_tokens=False)))
    lo, hi = 1, args.target_tokens // (2 * per_line) + 64
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if len(encode(messages(mid))) <= args.target_tokens:
            lo = mid
        else:
            hi = mid
    msg = messages(lo)
    length = len(encode(msg))
    prefix = len(tok.encode(filler * lo, add_special_tokens=False))
    if length <= args.pool_tokens:
        raise SystemExit(f"refusing in-pool request: {length} <= {args.pool_tokens}")
    request = {"model": args.model, "messages": msg, "temperature": 0,
               "max_tokens": 64, "stream": False,
               "chat_template_kwargs": {"enable_thinking": False}}
    base = Path(args.output)
    base.parent.mkdir(parents=True, exist_ok=True)
    request_path = base.with_suffix(".request.json")
    request_path.write_text(json.dumps(request))
    info = {"tokenizer_prompt_tokens": length, "pool_tokens": args.pool_tokens,
            "needle_prefix_tokens": prefix, "needle_fraction": prefix / length,
            "expected": "ZX-4417-QJ7", "request": str(request_path)}
    print(json.dumps(info), flush=True)
    if args.prepare_only:
        base.with_suffix(".metadata.json").write_text(json.dumps(info, indent=2))
        return
    start = time.monotonic()
    req = urllib.request.Request(args.url + "/v1/chat/completions",
                                 data=json.dumps(request).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3600) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        info.update(seconds=round(time.monotonic() - start, 2),
                    http_status=exc.code, error=body[:1000])
        base.with_suffix(".response.json").write_text(body)
        base.with_suffix(".metadata.json").write_text(json.dumps(info, indent=2))
        print(json.dumps(info), flush=True)
        raise SystemExit(1) from exc
    base.with_suffix(".response.json").write_text(json.dumps(result, indent=2))
    content = result["choices"][0]["message"].get("content") or ""
    usage = result.get("usage", {})
    info.update(seconds=round(time.monotonic() - start, 2), usage=usage, content=content,
                correct=content.strip() == info["expected"],
                beyond_pool=usage.get("prompt_tokens", 0) > args.pool_tokens)
    base.with_suffix(".metadata.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info), flush=True)
    if not info["correct"] or not info["beyond_pool"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
