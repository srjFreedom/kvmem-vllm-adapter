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


def async_scheduling_mode(cfg, override=None):
    """Decide the scheduler mode (``"1"`` = sync) for this engine.

    In-request compaction rewrites the block table mid-request, which the async scheduler cannot
    absorb: it plans step k+1 before step k's output has been accounted, so the rewrite trips
    ``assert request.num_output_placeholders >= 0`` (async_scheduler.py:63). Sync scheduling keeps
    every step self-contained, so compaction FORCES it — an explicit ``0`` cannot be honoured there
    (that configuration is not merely slower, it is broken). With compaction off the caller's
    explicit choice stands, and async is the default.
    """
    override = "" if override is None else str(override).strip()
    if cfg.get("inrequest_compaction"):
        return "1"
    return override or "0"



def entry_compaction_env(cfg, env):
    """Keep the ENTRY-side compaction switch in sync with the engine-side one (one knob).

    In-request compaction is a single feature spread over two processes: the generation ceiling is
    raised inside the API server (the entry policy reads ``KVMEM_ENTRY_*``) while the compaction
    itself runs in EngineCore / the TP workers (CLI -> ``--additional-config``). ``worker.py`` is
    the process that spawns the engine and the API server inherits its environment, so the CLI
    switch is the single authority here and the entry flag is derived from it. The env var stays
    overridable only for deployments that run the entry policy without this worker (gateway).

    Returns ``(value, note)``; ``note`` is ``"entry was off"``/``"entry was on"`` when the incoming
    value contradicted the CLI.
    """
    on = bool(cfg.get("inrequest_compaction"))
    current = str(env.get("KVMEM_ENTRY_INREQUEST_COMPACTION", "")).strip().lower()
    truthy = current in ("1", "true", "yes", "on")
    falsy = current in ("0", "false", "no", "off")
    if on:
        return "1", ("entry was off" if falsy else None)
    if truthy:
        # A raised ceiling without engine-side compaction overflows the pool (B + R cannot hold
        # the output), so the CLI wins here as well.
        return "0", "entry was on"
    return "0", None


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    module = _cli_module()
    ns = module.parse(argv)
    cfg = module.build_config(ns)
    cfg['inrequest_compaction'] = bool(getattr(ns, 'kvmem_inrequest_compaction', False))
    # Route A: the three-book reconciliation rides on the worker->EngineCore block channel, so it
    # must be enabled whenever in-request compaction is. The EngineCore starts the socket server
    # from this flag; the worker's compact signal connects to it.
    if cfg['inrequest_compaction']:
        cfg['block_channel'] = True
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
    # Scheduler mode. Under entry-side selection the engine only ever sees B+R tokens and the block
    # table is never rewritten mid-request, so the async scheduler is safe and is the default here
    # (measured: identical decode throughput to sync scheduling in this configuration). Only
    # in-request compaction (opt-in) rewrites the block table: vLLM's async scheduler plans the next
    # step before the current one's outputs are accounted for, so that rewrite breaks
    # `assert request.num_output_placeholders >= 0` (async_scheduler.py:63) and kills the engine.
    # Sync scheduling keeps every step self-contained, so compaction FORCES it back on; without
    # compaction an explicit NO_ASYNC_SCHEDULING still wins and async remains the default.
    _seqs = int(env.get("MAX_NUM_SEQS", "1") or "1")
    # One switch for the whole feature: the CLI flag also drives the entry-side ceiling.
    _entry_flag, _entry_note = entry_compaction_env(cfg, env)
    env["KVMEM_ENTRY_INREQUEST_COMPACTION"] = _entry_flag
    # The logical context cap (kvmem CTX) is not the engine window any more: with the resident
    # model the engine holds B+R of KV while the conversation may grow to CTX. The fork reads this
    # to size its request-length check, so export it (falling back to the site's CTX).
    if _entry_flag == "1":
        _ctx = str(env.get("CTX") or env.get("KVMEM_ENTRY_CTX") or "").strip()
        if _ctx.isdigit() and int(_ctx) > 0:
            env["KVMEM_ENTRY_CTX"] = _ctx
    if _entry_note == "entry was off":
        print("[kvmem backends/vllm] --kvmem-inrequest-compaction overrides "
              "KVMEM_ENTRY_INREQUEST_COMPACTION=0: the entry ceiling must match the engine")
    elif _entry_note == "entry was on":
        print("[kvmem backends/vllm] KVMEM_ENTRY_INREQUEST_COMPACTION was on but "
              "--kvmem-inrequest-compaction is off -> forcing the entry ceiling back to R "
              "(a raised ceiling without engine compaction overflows B + R)")
    _async_override = env.get("NO_ASYNC_SCHEDULING", "").strip()
    env["NO_ASYNC_SCHEDULING"] = async_scheduling_mode(cfg, _async_override)
    if cfg.get("inrequest_compaction"):
        # Prefix caching hashes the LOGICAL token stream while the KV layout is compacted, so a hit
        # would hand back pages that no longer hold those tokens. Force it off and say so.
        env["DISABLE_PREFIX_CACHING"] = "1"
        print("[kvmem backends/vllm] inrequest_compaction=on -> prefix caching disabled "
              "(block hashes are over logical tokens, the KV layout is compacted)")
        if _async_override == "0":
            print("[kvmem backends/vllm] inrequest_compaction=on overrides NO_ASYNC_SCHEDULING=0: "
                  "the compaction rewrite requires sync scheduling")
        print("[kvmem backends/vllm] inrequest_compaction=on -> --no-async-scheduling "
              "(block-table rewrite requires sync scheduling)")
    elif _async_override == "":
        print(f"[kvmem backends/vllm] MAX_NUM_SEQS={_seqs} -> async scheduling "
              "(entry-side selection, no mid-request block-table rewrite)")
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
    # One launcher ships with the adapter (deploy/): concurrency N and the engine window travel as
    # --set values, so a single script serves any N. KVMEM_LAUNCH_SCRIPT overrides it.
    script = os.environ.get("KVMEM_LAUNCH_SCRIPT") or os.path.join(
        ADAPTER, "deploy", "run-kvmem-engine.sh")
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
    _resident = str(env.get("KVMEM_ENTRY_RESIDENT_PREFILL", "")).strip().lower() in (
        "1", "true", "yes", "on")
    if _wanted <= 0 and _resident:
        # Lossless long prompts: the engine window is the PHYSICAL pool (auto-fitted to the card),
        # not N x (B+R). A whole prompt up to that window stays resident, so no compaction has to
        # happen while the prompt is still being prefilled (that path is not reliable yet); the
        # entry's CTX remains the logical ceiling.
        # A concrete window is required: the launcher validates N x MAX_MODEL_LEN against the
        # profiled KV pool and refuses -1. 180K fits this model's pool (~189K tokens) with margin.
        _wanted = int(env.get("KVMEM_RESIDENT_MAX_MODEL_LEN", "180000") or "180000")
        env["KV_CONCURRENCY_EFFECTIVE_LEN"] = str(int(ns.kvmem_budget))
        print(f"[kvmem backends/vllm] KVMEM_ENTRY_RESIDENT_PREFILL=1 -> "
              f"MAX_MODEL_LEN={_wanted} (engine window = physical pool), prefill stays resident, "
              f"compaction is decode-only")
    if _wanted <= 0 and not _resident:
        _wanted = max(1, int(ns.max_num_seqs)) * (int(ns.kvmem_budget) + int(ns.kvmem_gen_reserve))
        print(f"[kvmem backends/vllm] max_model_len auto = {ns.max_num_seqs} x "
              f"({ns.kvmem_budget} + {ns.kvmem_gen_reserve}) = {_wanted}")
        ns.max_model_len = _wanted
    if _wanted:
        # Always explicit: the launcher's default MAX_MODEL_LEN (32768) would reject any prompt
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
