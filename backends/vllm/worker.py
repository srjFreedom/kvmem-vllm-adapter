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


def _env_flag(name):
    return str(os.environ.get(name, "")).strip().lower() in ("1", "true", "yes", "on")


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
        # Cross-request prefix reuse: park a finished conversation's session (KV pages in the H
        # tier + recurrent-state snapshot) keyed by native block hashes, and resume the next turn
        # from it instead of prefilling the whole conversation again. Opt-in: each parked session
        # keeps its share of the host budget until it is evicted.
        if _env_flag("KVMEM_SESSION_REUSE"):
            cfg['session_reuse'] = True
        # The logical ceiling must travel WITH the config: the fork's scheduler and the runtime both
        # need it to distinguish "resident KV length" from "served position" (a 199K prompt stalled
        # at 180,000 = max_model_len because ctx was missing and the cap fell back to the window).
        try:
            _ctx = int(str(os.environ.get("CTX") or os.environ.get("KVMEM_ENTRY_CTX") or "0"))
        except Exception:
            _ctx = 0
        if _ctx > 0:
            cfg['ctx'] = _ctx
    # Experimental switches travel inside the plugin config: the worker processes do NOT inherit
    # KVMEM_* from the API server (the harness filters them), so an env read inside the runtime is
    # silently always-false. `ADDITIONAL_CONFIG_JSON` is the channel that does survive.
    for _k, _cfgk in (("KVMEM_PROMPT_END_RESELECT", "prompt_end_reselect"),
                      ("KVMEM_CHECKPOINT_REPLAY", "checkpoint_replay"),
                      ("KVMEM_COMPACT_REBUILD", "compact_rebuild")):
        if os.environ.get(_k, "") in ("1", "true", "yes", "on"):
            cfg[_cfgk] = True
    if os.environ.get("KVMEM_REPLAY_WINDOW", "").strip():
        cfg["replay_window"] = int(os.environ["KVMEM_REPLAY_WINDOW"])
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
        # Compaction and resident prefill are one feature: compaction alone trims the entry to B and
        # compacts while the prompt is still being prefilled, which is the combination that answered
        # with noise. Turn the resident path on with it unless the operator explicitly disabled it.
        _res = str(env.get("KVMEM_ENTRY_RESIDENT_PREFILL", "")).strip().lower()
        if _res in ("", "1", "true", "yes", "on"):
            env["KVMEM_ENTRY_RESIDENT_PREFILL"] = "1"
        else:
            print("[kvmem backends/vllm] KVMEM_ENTRY_RESIDENT_PREFILL is explicitly off while "
                  "--kvmem-inrequest-compaction is on: the prompt will be trimmed to B and compacted "
                  "during prefill (the path that produced wrong answers); keep it on unless you are "
                  "debugging that path")
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
        # Resident entry leaves the prompt intact. Auto-fit may hold a prompt inside the
        # physical pool; beyond-pool experiments explicitly set KVMEM_MAX_MODEL_LEN and
        # compact during prefill before the pool runs out. CTX remains the logical cap.
        # Auto-fit (-1) is used only without that explicit ceiling; the launcher skips
        # its N x window check for this resident scheduling scheme.
        env["KV_CONCURRENCY_EFFECTIVE_LEN"] = str(int(ns.kvmem_budget))
        # An explicit ceiling wins: vLLM's auto-fit lands BELOW the KV pool (measured 177,760 on a
        # 203,616-token pool), and every buffer sized by max_model_len - including the rotary cache -
        # then rejects a longer prompt with an illegal access instead of a clean error.
        _mm = int(env.get("KVMEM_MAX_MODEL_LEN", "0") or 0)
        if _mm > 0:
            args += ["--set", f"MAX_MODEL_LEN={_mm}"]
            env["MAX_MODEL_LEN"] = str(_mm)
            print(f"[kvmem backends/vllm] MAX_MODEL_LEN pinned to {_mm} (engine window)")
        else:
            args += ["--set", "MAX_MODEL_LEN=-1"]
            env["MAX_MODEL_LEN"] = "-1"
        print("[kvmem backends/vllm] KVMEM_ENTRY_RESIDENT_PREFILL=1 -> "
              f"MAX_MODEL_LEN={env['MAX_MODEL_LEN']} (full prompt enters the engine; "
              "beyond-pool mode requires prefill compaction)")
        _wanted = 0
    # Request a smaller block size for KVMem's 16-token pages. Hybrid models can still align
    # full attention blocks to the Mamba page size (measured: 1616 tokens = 101 KVMem pages),
    # so compaction must use the runner's actual blocks_per_kv_block and EngineCore ownership.
    _bs = int(os.environ.get("KVMEM_BLOCK_SIZE", "0") or 0)
    if _bs > 0:
        args += ["--set", f"BLOCK_SIZE={_bs}"]
        print(f"[kvmem backends/vllm] BLOCK_SIZE requested as {_bs} "
              "(hybrid attention may align it upward)")
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
