#!/usr/bin/env python3
"""KVMem backend worker for vLLM — the `-Backend vllm` counterpart of llamacpp / ninfer.

KVMem owns the CLI (same flag names and meanings as `llama-kvmem-server` and `ninfer-serve`);
this worker translates them into vLLM's official configuration channel and starts the engine:

    --kvmem-*  ->  --additional-config '{"kvmem": {...}}'  ->  VllmConfig.additional_config
                                                          ->  serialised to every TP worker
                                                          ->  the plugin reads it back

No side-channel file, no environment variable: on a TP>1 deployment the configuration reaches
each worker the same way vLLM delivers the rest of its config.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTER = os.path.dirname(os.path.dirname(HERE))
CLI = os.path.join(ADAPTER, "scripts", "kvmem-vllm-worker.py")


def _cli_module():
    """The CLI/parse/build logic lives in scripts/; this worker reuses it verbatim."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("kvmem_vllm_worker", CLI)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    module = _cli_module()
    ns = module.parse(argv)
    cfg = module.build_config(ns)
    cfg['inrequest_compaction'] = bool(getattr(ns, 'kvmem_inrequest_compaction', False))
    payload = json.dumps({"kvmem": cfg}, separators=(",", ":"))
    if "--dry-run" in argv:
        json.dump({"additional_config": payload, "plugin_config": cfg}, sys.stdout,
                  indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0
    env = dict(os.environ)
    # KVMem retrieval needs pre-RoPE statistics, and under `torch.compile` + CUDA Graph no Python
    # code runs per step (measured: capture.calls == 0 in the default config). The eager path is
    # therefore mandatory for retrieval only; recency keeps the compiled/graph path.
    # The sparse path rewrites the block table when a request is compacted. vLLM's ASYNC scheduler
    # plans the next step before the current one's outputs are accounted for, so that rewrite breaks
    # `assert request.num_output_placeholders >= 0` (async_scheduler.py:63) and kills the engine.
    # Sync scheduling keeps every step self-contained, which is exactly the invariant kvmem needs.
    # Measured: 2-way and 4-way concurrency both return 200 with sync scheduling, and both crash
    # with async scheduling.
    # Only pay for it when concurrency is actually requested: single-request mode keeps the async
    # scheduler (measured 3.38-3.45 s for 882+300 vs 4.1-4.5 s with sync scheduling).
    _seqs = int(env.get("MAX_NUM_SEQS", "1") or "1")
    if _seqs > 1 and env.get("NO_ASYNC_SCHEDULING") != "1":
        env["NO_ASYNC_SCHEDULING"] = "1"
        print(f"[kvmem backends/vllm] MAX_NUM_SEQS={_seqs} -> --no-async-scheduling "
              "(async scheduling pre-plans steps and conflicts with per-step page compaction; "
              "single-request mode keeps async for ~20% more decode throughput)")
    if cfg.get("retrieval"):
        # The fork bridge (vllm/kvmem_bridge.py) exposes the query through a device copy that is
        # part of the compiled graph, so retrieval no longer needs the slow eager path. Keep the
        # operation available only as an explicit fallback.
        env.setdefault("KVMEM_CAPTURE", "1")
        if env.get("KVMEM_FORCE_EAGER") == "1":
            env.setdefault("VLLM_USE_AOT_COMPILE", "0")
            env.setdefault("ENFORCE_EAGER", "1")
            print("[kvmem backends/vllm] KVMEM_FORCE_EAGER=1 -> eager path "
                  "(measured cost ~+40% decode throughput)")
        else:
            print("[kvmem backends/vllm] retrieval=on -> original compiled path + query bridge "
                  "(KVMEM_CAPTURE=1)")
    env["ADDITIONAL_CONFIG_JSON"] = payload
    env["KVMEM_PLUGIN_CONFIG"] = ""          # no side-channel file
    script = ("/home/srj/run-kvmem-4way.sh" if ns.max_num_seqs > 1
              else "/home/srj/run-kvmem-sparse.sh")
    print(f"[kvmem backends/vllm] port={ns.port} budget={ns.kvmem_budget} "
          f"R={ns.kvmem_gen_reserve} sink={ns.kvmem_sink_tokens} "
          f"recent={ns.kvmem_recent_tokens} sessions={ns.kvmem_sessions} "
          f"H={ns.kvmem_host_mib}MiB block={cfg['select_block_tokens']} "
          f"lrfu={cfg['lrfu']} score={cfg['score_aggregation']}")
    args = [script, "--port", str(ns.port)]
    # The launcher admits concurrency as `N x MAX_MODEL_LEN <= KV pool`. With KVMem the per-request
    # residency is B + R (only that much enters the model), so the admission basis must be B + R -
    # otherwise a large max_model_len (the logical window) is rejected for no reason.
    try:
        _effective = int(ns.kvmem_budget) + int(ns.kvmem_gen_reserve)
        if _effective > 0:
            env["KV_CONCURRENCY_EFFECTIVE_LEN"] = str(_effective)
    except Exception:
        pass
    # The launcher scripts pin MAX_MODEL_LEN themselves (the 4-way one uses 32768). Only override
    # when the caller explicitly asks for a different value, so the default path stays exactly the
    # one the launcher was tuned for.
    # Engine window sizing: vLLM needs its KV pool to hold ONE full max_model_len sequence, and N
    # concurrent KVMem requests need N x (B + R). Setting the window to that product makes the two
    # requirements equal - never larger than the real need, and large enough to serve all N.
    _wanted = int(getattr(ns, "max_model_len", 0) or 0)
    if _wanted <= 0:
        _wanted = max(1, int(ns.max_num_seqs)) * (int(ns.kvmem_budget) + int(ns.kvmem_gen_reserve))
        print(f"[kvmem backends/vllm] max_model_len auto = {ns.max_num_seqs} x "
              f"({ns.kvmem_budget} + {ns.kvmem_gen_reserve}) = {_wanted}")
        ns.max_model_len = _wanted
    if _wanted:
        # Always explicit: `run-kvmem-4way.sh` pins MAX_MODEL_LEN=32768 and would reject any prompt
        # above that, which is exactly the long-context concurrency case.
        args += ["--set", f"MAX_MODEL_LEN={_wanted}"]
        env["MAX_MODEL_LEN"] = str(_wanted)
    # The profile pins max_num_batched_tokens (4096 with spec decode); `--set` overrides the
    # profile, which is what concurrency N>2 with long prefills needs.
    # Query capture window: the bridge reads it at import time and allocates its GPU buffer once, so
    # it has to travel as an env var into every process (the launcher passes its own env through).
    _capture = int(getattr(ns, "kvmem_capture_tokens", 0) or 0)
    if _capture > 0:
        env["KVMEM_CAPTURE_TOKENS"] = str(_capture)
        args += ["--set", f"KVMEM_CAPTURE_TOKENS={_capture}"]
        print(f"[kvmem backends/vllm] KVMEM_CAPTURE_TOKENS={_capture}")

    # The launcher scripts pin --set MAX_NUM_SEQS=1; concurrency must come from the caller, and the
    # engine window above was sized for exactly this N.
    _seqs = int(getattr(ns, "max_num_seqs", 0) or 0)
    if _seqs > 0:
        args += ["--set", f"MAX_NUM_SEQS={_seqs}"]
        env["MAX_NUM_SEQS"] = str(_seqs)
    _spec = int(getattr(ns, "speculative_tokens", -1) or -1)
    served = str(getattr(ns, "served_name", "") or "")
    if served:
        # Keep the id stable across backends: clients written against the llama.cpp / ninfer service
        # send that name, and a rename would look like "model not found".
        args += ["--set", f"SERVED_NAME={served}"]
        env["SERVED_NAME"] = served
        print(f"[kvmem backends/vllm] served_model_name={served}")
    if _spec >= 0:
        args += ["--set", f"SPECULATIVE_TOKENS={_spec}"]
        env["SPECULATIVE_TOKENS"] = str(_spec)
        print(f"[kvmem backends/vllm] SPECULATIVE_TOKENS={_spec}")
    if int(getattr(ns, "max_num_batched_tokens", 0) or 0) > 0:
        env["MAX_BATCHED_TOKENS"] = str(int(ns.max_num_batched_tokens))
        args += ["--set", f"MAX_BATCHED_TOKENS={int(ns.max_num_batched_tokens)}"]
    return subprocess.call(args, env=env)


if __name__ == "__main__":
    sys.exit(main())
