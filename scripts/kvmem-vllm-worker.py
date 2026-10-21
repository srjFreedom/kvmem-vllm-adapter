#!/usr/bin/env python3
"""KVMem entry point for the vLLM backend.

KVMem owns the CLI: the `--kvmem-*` flags are the authority (same names and meanings as the
llama.cpp / NInfer workers), and this worker translates them into the vLLM backend's
configuration, then starts the backend. `--dry-run` prints the translation instead of starting
anything, which is how the mapping is tested.

    kvmem-vllm-worker.py --model /path/to/model --kvmem-budget 32768 \
        --kvmem-gen-reserve 16384 --kvmem-sink-tokens 8192 --kvmem-recent-tokens 1024 \
        --kvmem-block-tokens 128 --kvmem-host-mib 16384 --kvmem-sessions 4 \
        --kvmem-disk-mib 0 --no-kvmem-lrfu --tensor-parallel-size 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys

DEFAULTS = {
    "budget": 32768,
    "gen_reserve": 16384,
    "sink": 12288,
    "recent": 1024,
    "sessions": 4,
    "host_mib": 16384,
    "disk_mib": 0,
    "block_tokens": None,
    "lrfu": True,
}


def parse(argv=None):
    ap = argparse.ArgumentParser(prog="kvmem-vllm-worker", add_help=True,
                                 description="KVMem worker for the vLLM backend")
    # --- KVMem's own surface (authoritative) ------------------------------------------------
    ap.add_argument("--kvmem", action="store_true", default=True)
    ap.add_argument("--no-kvmem", dest="kvmem", action="store_false")
    ap.add_argument("--kvmem-budget", type=int, default=DEFAULTS["budget"],
                    help="history tokens selected per request, excludes R")
    ap.add_argument("--kvmem-gen-reserve", type=int, default=DEFAULTS["gen_reserve"])
    ap.add_argument("--kvmem-sink-tokens", type=int, default=DEFAULTS["sink"])
    ap.add_argument("--kvmem-recent-tokens", type=int, default=DEFAULTS["recent"])
    ap.add_argument("--kvmem-sessions", type=int, default=DEFAULTS["sessions"],
                    help="retained inactive histories (active requests are not counted)")
    ap.add_argument("--kvmem-host-mib", type=int, default=DEFAULTS["host_mib"],
                    help="per-rank KV payload RAM budget")
    ap.add_argument("--kvmem-disk-mib", type=int, default=DEFAULTS["disk_mib"])
    ap.add_argument("--kvmem-disk-path", default=None)
    ap.add_argument("--kvmem-capture-tokens", type=int, default=512,
                    help="query rows captured per step for KVMem retrieval (KVMem's query_max_tokens "
                         "default is 512; larger windows cost GPU memory and a wider copy)")
    ap.add_argument("--kvmem-block-tokens", type=int, default=DEFAULTS["block_tokens"],
                    help="selection granularity; must be a whole number of native pages")
    ap.add_argument("--no-kvmem-lrfu", dest="lrfu", action="store_false")
    ap.add_argument("--kvmem-lrfu-lambda", type=float, default=0.5)
    ap.add_argument("--kvmem-lrfu-top-frac", type=float, default=0.9)
    ap.add_argument("--kvmem-score-aggregation", choices=("mean", "max"), default="mean")
    ap.add_argument("--kvmem-inrequest-compaction", action="store_true", default=False,
                    help="opt in to truncating the context while a request is alive; OFF by default "
                         "because vLLM accounts sequence length monotonically")
    ap.add_argument("--kvmem-retrieval", action="store_true",
                    help="content-based (mean-K) selection instead of the recency fallback; "
                         "requires the pre-RoPE capture path")
    # --- backend surface (passed through to vLLM) -------------------------------------------
    ap.add_argument("--model", required=False)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-model-len", type=int, default=0,
                    help="engine window; 0 = auto = max_num_seqs x (B + R), the exact KV need")
    ap.add_argument("--tensor-parallel-size", type=int, default=2)
    ap.add_argument("--max-num-seqs", type=int, default=1)
    ap.add_argument("--served-name", default="",
                    help="model id clients see; keep it identical to the previous backend")
    ap.add_argument("--speculative-tokens", type=int, default=-1,
                    help="override SPECULATIVE_TOKENS (fewer drafts = fewer logits rows per step)")
    ap.add_argument("--max-num-batched-tokens", type=int, default=0,
                    help="per-step token budget; raise it for concurrency N>2 with long prefills")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.98)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def build_config(a) -> dict:
    """Translate the KVMem CLI into the vLLM backend's plugin configuration."""
    if a.kvmem_disk_mib and not a.kvmem_disk_path:
        raise SystemExit("--kvmem-disk-mib > 0 requires --kvmem-disk-path")
    # A selection block must be a whole number of native pages (16 tokens on this model).
    block = a.kvmem_block_tokens or 128
    if block % 16:
        raise SystemExit(f"--kvmem-block-tokens {block} is not a multiple of the native page (16)")
    if a.kvmem_sink_tokens % 16 or a.kvmem_recent_tokens % 16:
        raise SystemExit("--kvmem-sink-tokens/--kvmem-recent-tokens must be page aligned (16)")
    if a.kvmem_budget <= 0 or a.kvmem_gen_reserve <= 0:
        raise SystemExit("--kvmem-budget and --kvmem-gen-reserve must be positive")
    cfg = {
        "enabled": bool(a.kvmem),
        "sparse": bool(a.kvmem),
        "use_kvmem_plan": True,
        "install": True,
        "budget_tokens": a.kvmem_budget,
        "gen_reserve_tokens": a.kvmem_gen_reserve,
        "sink_tokens": a.kvmem_sink_tokens,
        "recent_tokens": a.kvmem_recent_tokens,
        "sessions": a.kvmem_sessions,
        "host_mib": a.kvmem_host_mib,
        "disk_mib": a.kvmem_disk_mib,
        "disk_path": a.kvmem_disk_path,
        "select_block_tokens": block,
        "lrfu": bool(a.lrfu),
        "lrfu_lambda": a.kvmem_lrfu_lambda,
        "lrfu_top_fraction": a.kvmem_lrfu_top_frac,
        "score_aggregation": a.kvmem_score_aggregation,
        "retrieval": bool(getattr(a, "kvmem_retrieval", False)),
        # Reselect cadence follows KVMem's reserve: recompute once R new tokens exist.
        # R is the KVMem cadence; the token budget mirrors it as a safety net so reselection cannot
        # be disabled by a missing planner.
        # Cadence belongs to KVMem's generation reserve (the planner's R); the adapter must not run
        # its own timer, so the fallback budget stays 0. With the native planner present, R drives
        # every reselect; without it, sparsity degrades gracefully instead of inventing a cadence.
        "reselect_token_budget": 0,
        "reselect_interval": 1_000_000,
    }
    return cfg


def launch_plan(a, cfg) -> dict:
    return {"worker": "vllm",
            "backend_args": {"model": a.model, "port": a.port,
                             "max_model_len": a.max_model_len,
                             "tensor_parallel_size": a.tensor_parallel_size,
                             "max_num_seqs": a.max_num_seqs,
                             "gpu_memory_utilization": a.gpu_memory_utilization},
            "plugin_config": cfg}


def main(argv=None) -> int:
    a = parse(argv)
    cfg = build_config(a)
    plan = launch_plan(a, cfg)
    if a.dry_run:
        json.dump(plan, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0
    # Real start: write the plugin config, then hand off to the vLLM launcher.
    import subprocess
    path = "/tmp/kvmem_plugin.json"
    with open(path, "w") as fh:
        json.dump(cfg, fh)
    adapter = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.environ.get("KVMEM_LAUNCH_SCRIPT") or os.path.join(
        adapter, "deploy", "run-kvmem-engine.sh")
    print(f"[kvmem-vllm-worker] backend=vllm port={a.port} budget={a.kvmem_budget} "
          f"R={a.kvmem_gen_reserve} sink={a.kvmem_sink_tokens} recent={a.kvmem_recent_tokens} "
          f"sessions={a.kvmem_sessions} H={a.kvmem_host_mib}MiB block={cfg['select_block_tokens']}")
    return subprocess.call([script, "--port", str(a.port)])


if __name__ == "__main__":
    sys.exit(main())
