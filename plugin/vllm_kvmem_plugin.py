# The out-of-tree vLLM plugin: loaded by vLLM's `load_general_plugins()` in every process
# (engine core + each TP worker), so it can wrap GPUModelRunner.execute_model without patching
# the vLLM source. This fork runs the V2 model runner (vllm.v1.worker.gpu.model_runner), so both
# runner classes are patched.
#
# Config lives in a FILE, not the environment: vLLM spawns the EngineCore and the TP workers with
# a filtered environment, so env vars set on the API server never reach them.
#   /tmp/kvmem_plugin.json (override with KVMEM_PLUGIN_CONFIG):
#     {"enabled": true, "probe_log": "...", "smoke_log": "...", "adapter_root": "...",
#      "resident_tokens": 65536, "pool_bytes": 268435456, "async_transfer": false}
# With no config file (or enabled=false) the plugin is a no-op, so installing it cannot affect
# normal serving.
import functools
import json
import os
import sys
import traceback

_PATCHED = False


_ACTIVE_CONFIG: dict = {}


def _config_from_runner(runner):
    """Take the KVMem configuration from vLLM's own config object.

    `--additional-config '{"kvmem": {...}}'` is vLLM's official channel: it is part of VllmConfig
    and is therefore serialised to every TP worker. That is the equivalent of kvmem-llama passing
    its CLI straight to the worker (no side-channel file, no environment variable).
    """
    global _ACTIVE_CONFIG
    try:
        cfg = getattr(getattr(runner, "vllm_config", None), "additional_config", None) or {}
        kvmem_cfg = cfg.get("kvmem") if isinstance(cfg, dict) else None
        if isinstance(kvmem_cfg, dict) and kvmem_cfg:
            if kvmem_cfg != _ACTIVE_CONFIG:
                _ACTIVE_CONFIG = dict(kvmem_cfg)
                try:                      # live proof that the config came from vLLM itself
                    _append(kvmem_cfg.get("probe_log") or "/tmp/kvmem_sparse.jsonl",
                            {"kind": "config_source", "from": "additional_config",
                             "pid": os.getpid(), "keys": sorted(kvmem_cfg)})
                except Exception:
                    pass
            return _ACTIVE_CONFIG
    except Exception:
        pass
    return None


def _env_flag(name: str) -> bool:
    import os
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _resident_prefill_on() -> bool:
    return _env_flag("KVMEM_ENTRY_RESIDENT_PREFILL")


def _config():
    """KVMem settings for this process: additional_config first, a file only as a legacy fallback."""
    if _ACTIVE_CONFIG:
        return _ACTIVE_CONFIG
    path = os.environ.get("KVMEM_PLUGIN_CONFIG", "")
    if not path:
        return {}
    try:
        with open(path) as handle:
            return json.load(handle)
    except Exception:
        return {}


def _install_watchdog(log_path):
    """Dump every thread's stack on SIGUSR1 so a hang can be localised without extra tooling."""
    try:
        import faulthandler
        import signal as _signal
        path = f"/tmp/kvmem_watchdog_{os.getpid()}.log"
        handle = open(path, "a")
        faulthandler.register(_signal.SIGUSR1, file=handle, all_threads=True, chain=False)
        _append(log_path, {"kind": "watchdog_armed", "pid": os.getpid(), "path": path})
    except Exception:
        _append(log_path, {"kind": "watchdog_error", "error": traceback.format_exc()})


def _boot(message):
    """Diagnostic breadcrumb: the filtered worker env makes silent no-ops hard to see."""
    try:
        with open("/tmp/kvmem_plugin_boot.log", "a") as handle:
            handle.write(f"{os.getpid()} {message}\n")
    except Exception:
        pass


# --- live console -------------------------------------------------------------------------
# Iterating on the engine used to cost a full restart (80 s) per one-line change. The workers are
# Python, so they can be inspected and patched in place: drop code into /tmp/kvmem_console.py while
# /tmp/kvmem_console.arm exists, and the next step executes it INSIDE the worker with the live
# runtime/orchestrator/bridge in scope and writes the result to /tmp/kvmem_console.out. One stat()
# per step when it is not armed. Disarm by removing the arm file.
CONSOLE_ARM = "/tmp/kvmem_console.arm"
CONSOLE_CODE = "/tmp/kvmem_console.py"
CONSOLE_OUT = "/tmp/kvmem_console.out"


def _console_step(context):
    """Execute the pending console snippet in this worker process. Never raises."""
    import os
    import io
    import traceback
    try:
        if not os.path.exists(CONSOLE_ARM) or not os.path.exists(CONSOLE_CODE):
            return
        try:
            code = open(CONSOLE_CODE).read()
        except Exception:
            return
        try:
            os.remove(CONSOLE_CODE)          # one-shot: a snippet never runs twice
        except Exception:
            pass
        buffer = io.StringIO()
        namespace = dict(context)
        namespace["__name__"] = "kvmem_console"
        try:
            import contextlib
            with contextlib.redirect_stdout(buffer):
                exec(compile(code, CONSOLE_CODE, "exec"), namespace)
        except Exception:
            buffer.write("EXC:\n" + traceback.format_exc())
        with open(CONSOLE_OUT, "a") as handle:
            handle.write(f"===== pid={os.getpid()} step={context.get('step')} "
                         f"req={context.get('req')} =====\n")
            handle.write(buffer.getvalue())
            handle.write("\n")
    except Exception:
        pass


HOT_REQUEST = "/tmp/kvmem_reload.request"


def _hot_apply(runner, runtime, log_path):
    """Apply pending Python changes at a REQUEST BOUNDARY (no capture in flight). Never raises.

    Python code does not need an engine restart: modules are reloaded and their methods rebound on
    the live runtime class, and the bridge geometry is rebuilt in place. Only compiled extensions
    (the pybind .so) would need a restart - nothing here touches those.
    """
    import importlib
    import os
    import traceback
    out = []
    try:
        reloaded = {}
        for name in ("kvmem_statistics", "kvmem_recall", "kvmem_sparse_orchestrator",
                     "kvmem_sparse_runtime"):
            try:
                module = importlib.import_module(name)
                reloaded[name] = importlib.reload(module)
            except Exception as exc:
                out.append(f"reload {name}: {exc}")
        mod = reloaded.get("kvmem_sparse_runtime")
        if mod is not None:
            cls = type(runtime)
            rebound = []
            for attr in ("harvest_key_statistics", "attach_statistics", "statistics_stats",
                         "publish_position_offset", "_skip"):
                if hasattr(mod.VllmSparseRuntime, attr):
                    setattr(cls, attr, getattr(mod.VllmSparseRuntime, attr))
                    rebound.append(attr)
            out.append(f"rebound {rebound}")
        # Re-derive the PER-RANK key geometry from the config (the hf count is global under TP) and
        # rebuild the bridge buffers at that width.
        from vllm import kvmem_bridge as _bridge
        cfg = getattr(getattr(runner, "vllm_config", None), "model_config", None)
        hf = getattr(cfg, "hf_config", None)
        text = getattr(hf, "text_config", None)
        def _int(obj, name):
            value = getattr(obj, name, None)
            return int(value) if isinstance(value, int) and value > 0 else 0
        heads = _int(hf, "num_attention_heads") or _int(text, "num_attention_heads")
        kv_global = _int(hf, "num_key_value_heads") or _int(text, "num_key_value_heads") or heads
        head_dim = _int(hf, "head_dim") or _int(text, "head_dim") or 256
        try:
            tp = int(getattr(getattr(runner, "vllm_config", None), "parallel_config",
                             None).tensor_parallel_size)
        except Exception:
            tp = 1
        per_rank_heads = max(1, heads // max(1, tp)) if heads else 0
        per_rank_kv = max(1, kv_global // max(1, tp)) if kv_global else 0
        layers = _int(hf, "num_hidden_layers") or _int(text, "num_hidden_layers") or 64
        ok = _bridge.ensure(layers=layers, heads=per_rank_heads, head_dim=head_dim,
                            device="cuda", dtype=torch_dtype(runner),
                            kv_heads=per_rank_kv, force=True)
        runtime.kv_heads = per_rank_kv
        for table in ("_statistics", "_stats_next", "_stats_written", "_stats_skipped",
                      "_compare_done"):
            getattr(runtime, table).clear()
        out.append(f"bridge force-ensure ok={ok} heads/rank={per_rank_heads} "
                   f"kv/rank={per_rank_kv} width={getattr(_bridge, '_KEY_WIDTH', None)}")
    except Exception:
        out.append("EXC:\n" + traceback.format_exc()[-800:])
    try:
        _append(log_path, {"kind": "hot_apply", "pid": os.getpid(), "lines": out})
    except Exception:
        pass
    try:
        os.remove(HOT_REQUEST)
    except Exception:
        pass


def torch_dtype(runner):
    import torch as _t
    cfg = getattr(getattr(runner, "vllm_config", None), "model_config", None)
    dtype = getattr(cfg, "dtype", None)
    return dtype if getattr(dtype, "is_floating_point", False) else _t.bfloat16


def _append(path, record):
    try:
        with open(path, "a") as handle:
            handle.write(json.dumps(record) + "\n")
    except Exception:
        pass


def _ensure_import_path():
    adapter_root = _config().get("adapter_root") or os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))
    for path in (os.path.join(adapter_root, "build", "python"),
                 os.path.join(adapter_root, "src")):
        if path not in sys.path:
            sys.path.insert(0, path)


def _paged_attn_tensors(runner):
    """The paged FULL-ATTENTION KV tensors, with their cache group id.

    In `mamba_cache_mode=align` the GDN layers carry Mamba state blocks and the attention layers
    carry the token payload, but the attention block is padded to the aligned size (e.g. 1616
    tokens = 101 pages of 16), so a tensor's first dim is PAGES, not blocks.
    """
    caches = getattr(runner, "kv_caches", None) or []
    config = getattr(runner, "kv_cache_config", None)
    attn_groups = set()
    for gi, group in enumerate(getattr(config, "kv_cache_groups", None) or []):
        spec = getattr(group, "kv_cache_spec", None)
        if spec is not None and type(spec).__name__ == "FullAttentionSpec":
            attn_groups.add(gi)
    group_ids = getattr(runner, "kv_cache_group_ids", None) or []
    paged = []
    for i, tensor in enumerate(caches):
        group = group_ids[i] if i < len(group_ids) else None
        if group in attn_groups and tensor.dim() == 4:
            paged.append((i, tensor, group))
    return paged, attn_groups, config


def _state_groups(runner):
    """KV-cache groups that are NOT paged attention (GDN / Mamba) as {group: [tensors]}.

    Their state is per-request recurrent data, NOT in the paged store: a resume must carry a
    snapshot of it or the resumed prefill reads a stale state (the documented failure where draft
    acceptance collapses and the generation repeats).
    """
    out = {}
    try:
        caches = getattr(runner, "kv_caches", None) or []
        gids = getattr(runner, "kv_cache_group_ids", None) or []
        groups = getattr(getattr(runner, "kv_cache_config", None), "kv_cache_groups", None) or []
        for i, cache in enumerate(caches):
            g = gids[i] if i < len(gids) else None
            if g is None or int(g) >= len(groups):
                continue
            spec = getattr(groups[int(g)], "kv_cache_spec", None)
            if type(spec).__name__ == "FullAttentionSpec":
                continue
            items = cache if isinstance(cache, (list, tuple)) else (cache,)
            out.setdefault(int(g), []).extend(t for t in items if hasattr(t, "shape"))
    except Exception:
        return {}
    return out


def _has_recurrent_groups(runner):
    return bool(_state_groups(runner))


def _state_slot(runner, row, group):
    try:
        return int(runner.block_tables.block_tables[int(group)].gpu[int(row), 0].item())
    except Exception:
        return None


def _capture_state_snapshot(runner, runtime, req):
    """Copy this request's recurrent state to RAM (the parked session's RAM tier)."""
    groups = _state_groups(runner)
    st = runtime._state.get(req)
    row = getattr(st, "row_index", None) if st is not None else None
    if not groups or row is None:
        return None
    snap = {}
    for group, tensors in groups.items():
        slot = _state_slot(runner, row, group)
        if slot is None or slot < 0:
            continue
        try:
            snap[int(group)] = (slot, [t[slot].detach().to("cpu").clone() for t in tensors])
        except Exception:
            continue
    return snap or None


def _restore_state_snapshot(runner, runtime, req, snapshot):
    """Put the parked recurrent state back into the request's own state slots."""
    if not snapshot:
        return False
    groups = _state_groups(runner)
    st = runtime._state.get(req)
    row = getattr(st, "row_index", None) if st is not None else None
    if not groups or row is None:
        return False
    restored = False
    for group, (_src_slot, saved) in snapshot.items():
        tensors = groups.get(int(group))
        if not tensors or len(tensors) != len(saved):
            continue
        dst = _state_slot(runner, row, group)
        if dst is None or dst < 0:
            continue
        for tensor, value in zip(tensors, saved):
            try:
                tensor[dst].copy_(value.to(tensor.device, non_blocking=False))
                restored = True
            except Exception:
                pass
    return restored


def _probe_runner(runner, log_path):
    record = {"kind": "probe", "pid": os.getpid()}
    try:
        caches = getattr(runner, "kv_caches", None) or []
        record["num_kv_cache_tensors"] = len(caches)
        group_ids = getattr(runner, "kv_cache_group_ids", None) or []
        num_blocks = getattr(runner, "kv_cache_num_blocks", None) or []
        record["tensors"] = [
            {"index": i, "shape": list(t.shape), "dtype": str(t.dtype),
             "group": group_ids[i] if i < len(group_ids) else None,
             "num_blocks": num_blocks[i] if i < len(num_blocks) else None}
            for i, t in enumerate(caches)
        ]
        config = getattr(runner, "kv_cache_config", None)
        record["groups"] = [
            {"group": gi, "spec": type(getattr(g, "kv_cache_spec", None)).__name__,
             "block_size": getattr(getattr(g, "kv_cache_spec", None), "block_size", None)}
            for gi, g in enumerate(getattr(config, "kv_cache_groups", None) or [])
        ]
    except Exception:
        record["error"] = traceback.format_exc()
    _append(log_path, record)


def _attach_adapter(runner, log_path):
    """Build the KVMem adapter for this rank and return a per-step callable (or None)."""
    _ensure_import_path()
    try:
        import torch
        import vllm_kvmem_ext as ext
        from vllm_integration import KvmemAdapter
    except Exception:
        _append(log_path, {"kind": "attach_error", "pid": os.getpid(),
                           "error": traceback.format_exc()})
        return None

    paged, attn_groups, config = _paged_attn_tensors(runner)
    if not paged:
        _append(log_path, {"kind": "attach_skip", "pid": os.getpid(),
                           "reason": "no paged full-attention KV tensors",
                           "groups": sorted(attn_groups)})
        return None

    sample = paged[0][1]
    itemsize = sample.element_size()
    pages, heads, page_tokens, content = sample.shape
    head_dim = content // (2 * itemsize) if content else 0
    # The engine's block table addresses the tensor's FIRST DIM: with mamba_cache_mode=align the
    # attention pool is allocated in aligned blocks (e.g. 1616 tokens) but the table stores PAGE
    # ids in [0, pages). So the KVMem block unit is one page (16 tokens), not the aligned block.
    block_tokens = page_tokens
    pages_per_block = 1
    aligned_tokens = None
    for gi in sorted(attn_groups):
        if config is not None and gi < len(config.kv_cache_groups):
            spec = getattr(config.kv_cache_groups[gi], "kv_cache_spec", None)
            if spec is not None and getattr(spec, "block_size", None):
                aligned_tokens = int(spec.block_size)
                break
    record_bytes = block_tokens * heads * head_dim * itemsize * 2 * len(paged)

    cfg = _config()
    budget = cfg.get("resident_tokens")
    budget = int(budget) if budget else None
    _append(log_path, {"kind": "geometry", "pid": os.getpid(),
                       "paged_tensors": len(paged), "shape": list(sample.shape),
                       "block_tokens": block_tokens, "page_tokens": page_tokens,
                       "pages_per_block": pages_per_block, "num_kv_heads": heads,
                       "head_dim": head_dim, "dtype_bytes": itemsize,
                       "record_bytes": record_bytes, "resident_tokens": budget,
                       "aligned_tokens": aligned_tokens,
                       "block_table_unit": "page"})

    adapter = KvmemAdapter(
        runner=runner,
        pool_capacity_bytes=int(cfg.get("pool_bytes", max(1 << 30, 8 * record_bytes))),
        # The core rejects a zero session id; both ranks may share it (their cartridges and host
        # pools are independent, and an identical id keeps the lockstep decision identical).
        session_id=max(1, int(cfg.get("session_id", 1))),
        block_tokens=block_tokens,
        num_full_attn_layers=len(paged),
        num_kv_heads=heads,
        head_dim=head_dim,
        dtype_bytes=itemsize,
        resident_budget_tokens=budget,
        async_transfer=bool(cfg.get("async_transfer", False)),
        pages_per_block=pages_per_block,
        ctx_tokens=int(cfg.get("ctx") or 0),
        sink_tokens=int(cfg.get("sink_tokens", 0) or 0),
        # Lossless long prompts default to "no compaction while prefilling": that path is verified
        # (80K/170K/N=2 answers exact) while compacting mid-prefill is not (measured: noise from
        # the first token). The prompt then stays resident up to the engine window, which the
        # worker sizes to the physical pool. Opt in to the old behaviour with
        # KVMEM_RESIDENT_PREFILL_COMPACT=1 when working on that path.
        no_prefill_compact=(
            _resident_prefill_on()
            and not _env_flag("KVMEM_RESIDENT_PREFILL_COMPACT")),
        kv_caches=[tensor for _, tensor, _ in paged],
    ).bind()
    _append(log_path, {"kind": "attached", "pid": os.getpid(),
                       "pages_per_block": pages_per_block, "block_tokens": block_tokens})

    stats = {"steps": 0, "needle_done": False}

    def refresh_handle():
        """Track the live conversation: catalog = the blocks the engine currently references.

        The device always holds every block (the adapter copies bytes out but never frees a
        vLLM block), so "all resident" is the truthful device state each step.
        """
        ids, seen = [], set()
        for group, raw in KvmemAdapter._block_table_entries(runner):
            if group not in attn_groups:
                continue          # Mamba state groups use a different block numbering
            rows = raw.tolist() if hasattr(raw, "tolist") else raw
            for row in rows:
                for b in row:
                    b = int(b)
                    if b >= 0 and b not in seen:
                        seen.add(b)
                        ids.append(b)
        if not ids:
            return []
        catalog = adapter._handle._catalog
        catalog[:] = []
        for i, b in enumerate(ids):
            cb = ext.CatalogBlock()
            si = ext.SessionIdentity(); si.id = adapter._session_id; si.incarnation = 1
            cb.session = si
            cb.id = b
            cb.original_start = i * block_tokens
            cb.valid_tokens = block_tokens
            cb.content_version = 1
            cb.statistics_version = 1
            catalog.append(cb)
        adapter._handle._resident = list(ids)
        adapter._handle._stamp.frontier = len(ids) * block_tokens
        return ids

    def step(scheduler_output):
        try:
            ids = refresh_handle()
            if not ids:
                return None
            stats["steps"] += 1
            # The needle cycle on LIVE KV: evict the oldest settled block to the host pool and
            # restore it immediately (nothing writes in between), then check byte equality.
            if stats["steps"] == 2:
                _append(log_path, {"kind": "catalog", "pid": os.getpid(),
                                   "count": len(ids), "min": min(ids), "max": max(ids),
                                   "pages": pages, "pages_per_block": pages_per_block,
                                   "tables": [(g, list(raw.shape) if hasattr(raw, "shape") else None)
                                              for g, raw in
                                              KvmemAdapter._block_table_entries(runner)],
                                   "attn_groups": sorted(attn_groups)})
            if not stats["needle_done"] and len(ids) >= 2 and stats["steps"] >= 2:
                stats["needle_done"] = True
                target = ids[0]
                _append(log_path, {"kind": "needle_pre", "pid": os.getpid(),
                                   "target": target, "catalog": len(ids),
                                   "max_block_for_pages": pages // pages_per_block - 1})
                span = slice(target * pages_per_block, (target + 1) * pages_per_block)
                ref = [t[span].clone() for _, t, _ in paged]
                # Only the newest block is mandatory (what a decode step actually needs), so the
                # policy evicts the oldest one and the spill really runs.
                forced = [ids[-1]]
                real_step_ids = adapter._step_block_ids
                adapter._step_block_ids = lambda output, catalog: set(forced)
                try:
                    adapter._resident_budget_tokens = (len(ids) - 1) * block_tokens
                    view = adapter.on_execute_model_done(scheduler_output)
                finally:
                    adapter._step_block_ids = real_step_ids
                evicted = sorted(set(ids) - {b.block.id for b in view.blocks}) if view else []
                adapter._resident_budget_tokens = len(ids) * block_tokens
                adapter._step_block_ids = lambda output, catalog: set(forced)
                try:
                    view2 = adapter.on_execute_model_done(scheduler_output)
                finally:
                    adapter._step_block_ids = real_step_ids
                restored = len(view2.blocks) if view2 else 0
                same = all(torch.equal(t[span], r) for (_, t, _), r in zip(paged, ref))
                _append(log_path, {"kind": "needle", "pid": os.getpid(),
                                   "catalog": len(ids), "ids": [min(ids), max(ids)],
                                   "pages": pages, "target": target,
                                   "slice_pages": span.stop - span.start,
                                   "evicted": evicted, "restored_blocks": restored,
                                   "bytes_identical": bool(same)})
            return None
        except Exception:
            _append(log_path, {"kind": "cycle_error", "pid": os.getpid(),
                               "error": traceback.format_exc()})
            return None

    return step


def _attach_sparse(runner, log_path):
    """Wire the sparse-view runtime (KVMem A mode) onto a live V2 runner.

    Two hook points per step:
      * prepare_pos_seq_lens() — after the engine computes positions/seq_lens, install the
        request's sparse view (block-table row, seq_len, compacted query positions);
      * execute_model() — after KV writes, commit the new pages and reselect when due.
    """
    _ensure_import_path()
    try:
        import vllm.v1.worker.gpu.model_runner as mr

        import vllm_kvmem_ext as ext
        from kvmem_live_view import resolve_live_view
        from kvmem_sparse_runtime import VllmSparseRuntime
    except Exception:
        _append(log_path, {"kind": "sparse_attach_error", "pid": os.getpid(),
                           "error": traceback.format_exc()})
        return None

    cfg = _config()
    view = resolve_live_view(runner, _paged_attn_tensors)
    if view is None:
        _append(log_path, {"kind": "sparse_skip", "pid": os.getpid(),
                           "reason": "no paged attention KV"})
        return None

    # Engine allocation quantum in kernel pages: with a hybrid model under `mamba align` the
    # attention block is widened (e.g. 1616 tokens = 101 pages of 16) so the block table row is
    # written a whole block at a time. The runtime must pad the resident row to that quantum.
    pages_per_block = 1
    try:
        bt = getattr(runner, "block_tables", None)
        bpk = getattr(bt, "blocks_per_kv_block", None)
        if bpk:
            pages_per_block = int(max(int(v) for v in bpk))
    except Exception:
        pages_per_block = 1

    sel_cfg = ext.SparseSelectorConfig()
    sel_cfg.select_block_tokens = int(cfg.get("select_block_tokens", 128))
    sel_cfg.budget_tokens = int(cfg.get("budget_tokens", 32768))
    sel_cfg.gen_budget_tokens = int(cfg.get("gen_reserve_tokens", 16384))
    sel_cfg.sink_tokens = int(cfg.get("sink_tokens", 8192))
    sel_cfg.recent_tokens = int(cfg.get("recent_tokens", 1024))
    sel_cfg.page_tokens = int(view.page_tokens)
    sel_cfg.lrfu = bool(cfg.get("lrfu", False))
    selector = ext.VllmSparseSelector(sel_cfg)

    runtime = VllmSparseRuntime(
        view, selector, view.page_tokens, int(cfg.get("head_dim", 256)),
        int(cfg.get("rotary_dim", 64)), float(cfg.get("rope_theta", 1e7)),
        k_slice=slice(0, int(cfg.get("head_dim", 256))),
        reselect_interval=int(cfg.get("reselect_interval", 64)),
        log=lambda record: _append(log_path, record),
        # The kvmem contract parameters: the planner needs B/R verbatim (defaults would plan a
        # 32768-token view and drive the R cadence from 16384, both wrong for this site).
        select_block_tokens=int(cfg.get("select_block_tokens", 128)),
        budget_tokens=int(cfg.get("budget_tokens", 32768)),
        gen_reserve=int(cfg.get("gen_reserve_tokens", 16384)),
        recent_tokens=int(cfg.get("recent_tokens", 1024)),
        host_mib=int(cfg.get("host_mib", 16384)),
        use_kvmem_plan=bool(cfg.get("use_kvmem_plan", False)),
        sessions=int(cfg.get("sessions", 4)),
        pages_per_block=pages_per_block,
        ctx_tokens=int(cfg.get("ctx") or 0),
        sink_tokens=int(cfg.get("sink_tokens", 0) or 0),
        # Lossless long prompts default to "no compaction while prefilling": that path is verified
        # (80K/170K/N=2 answers exact) while compacting mid-prefill is not (measured: noise from
        # the first token). The prompt then stays resident up to the engine window, which the
        # worker sizes to the physical pool. Opt in to the old behaviour with
        # KVMEM_RESIDENT_PREFILL_COMPACT=1 when working on that path.
        no_prefill_compact=(
            _resident_prefill_on()
            and not _env_flag("KVMEM_RESIDENT_PREFILL_COMPACT")),
        # vLLM's NULL block occupies one aligned unit at the front of the pool.
        first_page=int(cfg.get("first_page", 0)) or
        (int(cfg.get("select_block_tokens", 128)) // max(1, int(view.page_tokens))))
    runtime.install = bool(cfg.get("install", True))
    # In-request compaction rewrites the block table mid-request; vLLM accounts sequence length
    # monotonically, so this is opt-in and comes from --kvmem-inrequest-compaction.
    runtime.inrequest_compaction = bool(cfg.get("inrequest_compaction", False))
    # Route A three-book reconciliation: the worker->EngineCore channel that delivers each
    # reselect so the engine truncates `_all_token_ids` and rewinds `num_computed_tokens`. It
    # reuses the block-channel socket, so it is only available when that channel is enabled.
    if runtime.inrequest_compaction:
        try:
            from kvmem_block_channel import BlockChannel
            runtime.compact_channel = BlockChannel(
                path=cfg.get("block_channel_path", "/tmp/kvmem_block_channel.sock"))
        except Exception:
            runtime.compact_channel = None
    else:
        runtime.compact_channel = None
    # --- cross-request prefix reuse: park finished sessions, adopt them on a later turn --------
    runtime.has_recurrent = _has_recurrent_groups(runner)
    # The recurrent-state hooks are needed by TWO features, not just session reuse:
    #   * parking a session across requests, and
    #   * QueryReplay, which must restore the GDN state to the replay boundary or the recomputed
    #     tail is applied to that state twice.
    # They used to be wired only inside the session-reuse branch, so a compaction-only run had
    # capture_state=None and the replay silently corrupted the state.
    def _set_block_cursor(req, page_count):
        """Reset the worker's block-table append cursor for a compacted request.

        `append_block_ids` writes each new engine block's pages at
        `num_blocks.np[group, req]` and advances that cursor. Installing a shorter row left the cursor
        at the pre-compaction value, so the NEXT block was written at the old slot
        (measured: 'row capacity (11514 > 11312)' = 114 blocks * 101 pages while the row had 22
        blocks). Only groups that map several native pages per engine block are touched.
        """
        try:
            st = runtime._state.get(req)
            idx = getattr(st, "row_index", None) if st is not None else None
            if idx is None:
                return False
            bt = runner.block_tables
            nb = bt.num_blocks.np
            bpk = list(getattr(bt, "blocks_per_kv_block", []) or [])
            done = False
            for g in range(nb.shape[0]):
                if g < len(bpk) and int(bpk[g]) > 1:
                    nb[g, int(idx)] = int(page_count)
                    done = True
            return done
        except Exception:
            return False

    runtime.set_block_cursor = _set_block_cursor
    if runtime.has_recurrent:
        runtime.capture_state = lambda req: _capture_state_snapshot(runner, runtime, req)
        runtime.restore_state = lambda req, snap: _restore_state_snapshot(runner, runtime, req, snap)

        def _reset_state(req):
            """Zero this request's recurrent state (prep for a reconstruction pass)."""
            groups = _state_groups(runner)
            st = runtime._state.get(req)
            row = getattr(st, "row_index", None) if st is not None else None
            if not groups or row is None:
                return False
            done = False
            for group, tensors in groups.items():
                slot = _state_slot(runner, row, group)
                if slot is None or slot < 0:
                    continue
                for tensor in tensors:
                    try:
                        tensor[slot].zero_()
                        done = True
                    except Exception:
                        pass
            return done

        runtime.reset_state = _reset_state
    if bool(cfg.get("session_reuse", False)) and runtime.inrequest_compaction:
        try:
            from kvmem_session_store import get_session_store
            _bs = int(getattr(getattr(getattr(runner, "vllm_config", None), "cache_config", None),
                               "block_size", 0) or 0)
            runtime.enable_session_reuse(get_session_store("worker"), _bs)
            runtime.capture_state = (
                lambda req: _capture_state_snapshot(runner, runtime, req))
            runtime.restore_state = (
                lambda req, snap: _restore_state_snapshot(runner, runtime, req, snap))
            _ch = getattr(runtime, "compact_channel", None)
            runtime.session_park = getattr(_ch, "session_park", None)
            _append(log_path, {"kind": "session_reuse_on", "pid": os.getpid(),
                               "block_size": _bs, "recurrent": runtime.has_recurrent,
                               "channel": runtime.session_park is not None})
        except Exception:
            _append(log_path, {"kind": "session_reuse_error", "pid": os.getpid(),
                               "error": traceback.format_exc()})
    # Reconstruction on compaction (opt-in): rewrite the engine's token book to the compacted
    # sequence, zero the recurrent state and re-prefill the selected pages. This is what removes the
    # approximation that in-prefill compaction leaves in a hybrid model's state.
    try:
        # Native pages per ENGINE block for the full-attention group (1616-token blocks over
        # 16-token pages -> 101). The compaction needs it to keep the paged table a faithful
        # reflection of the engine's block list.
        _bpk = list(getattr(runner.block_tables, "blocks_per_kv_block", []) or [])
        runtime.block_pages = int(max(_bpk) if _bpk else 1)
    except Exception:
        runtime.block_pages = 1
    try:
        runner.kvmem_runtime = runtime      # diagnostic hooks in the runner read the books here
    except Exception:
        pass
    # Experimental: rewriting the engine's token book stalls the request today ("no sampled
    # token"), so it stays OFF unless explicitly enabled.
    runtime.compact_rebuild = (os.environ.get("KVMEM_COMPACT_REBUILD", "0") == "1")
    # The rebuild rewrites ONE engine-side token book, so only rank 0 may ask for it. Both ranks
    # still send their ordinary compact message (each owns its KV), but a second rebuild would
    # re-derive the token list from an already-rewritten book (measured: 7 rebuild records and
    # garbage output).
    try:
        runtime.tp_rank = int(getattr(getattr(runner.vllm_config, "parallel_config", None),
                                      "rank", 0) or 0)
    except Exception:
        runtime.tp_rank = 0
    # Experimental switches arrive through the plugin config (env does not reach the workers).
    try:
        if cfg.get("prompt_end_reselect"):
            runtime.reselect_at_prompt_end = True
        if cfg.get("checkpoint_replay"):
            runtime.checkpoint_replay = True
        if cfg.get("replay_window"):
            runtime.replay_window = int(cfg.get("replay_window") or 2048)
        _append(log_path, {"kind": "checkpoint_flags", "pid": os.getpid(),
                           "prompt_end_reselect": bool(runtime.reselect_at_prompt_end),
                           "checkpoint_replay": bool(runtime.checkpoint_replay),
                           "catalog": runtime.checkpoints is not None,
                           "cfg_keys": [k for k in cfg if "replay" in k or "reselect" in k
                                        or "rebuild" in k]})
    except Exception:
        pass
    runtime.use_kvmem_plan = bool(cfg.get("use_kvmem_plan", False))
    runtime.use_kvmem_host = bool(cfg.get("use_kvmem_host", True))
    try:
        _cls = type(runner)
        _append(log_path, {"kind": "runner_class", "pid": os.getpid(),
                           "module": _cls.__module__, "name": _cls.__qualname__,
                           "has_prepare_pos_seq_lens": hasattr(_cls, "prepare_pos_seq_lens"),
                           "has_sample_tokens": hasattr(_cls, "sample_tokens"),
                           "has_execute_model": hasattr(_cls, "execute_model")})
    except Exception:
        pass
    try:
        import inspect as _insp
        from vllm.model_executor.models import qwen3_next as _q3
        _src = _insp.getsource(_q3.Qwen3NextAttention._project_qkv_gate)
        _append(log_path, {"kind": "gate_source", "pid": os.getpid(),
                           "file": _q3.__file__, "cached": getattr(_q3, "__cached__", None),
                           "has_bridge_attr": hasattr(_q3, "_kvmem_bridge"),
                           "has_capture_call": "kvmem.capture_q" in _src,
                           "tail": _src[-160:]})
    except Exception:
        _append(log_path, {"kind": "gate_source_error", "error": traceback.format_exc()})
    # Disk tier budget (D) and path, straight from --kvmem-disk-mib / --kvmem-disk-path.
    runtime.disk_mib = int(cfg.get("disk_mib", 0) or 0)
    runtime.disk_path = cfg.get("disk_path") or None
    runtime.retrieval_enabled = bool(cfg.get("retrieval", False))
    try:
        # Upstream's budget bands, published for the core scorer (query_score_frame needs them in
        # statistics blocks; the selector's C++ config is not exposed to Python).
        from kvmem_recall import set_default_blocks
        set_default_blocks(sink_tokens=int(cfg.get("sink_tokens", 0) or 0),
                           recent_tokens=int(cfg.get("recent_tokens", 0) or 0),
                           budget_tokens=int(cfg.get("budget_tokens", 0) or 0),
                           block_tokens=64)
    except Exception:
        pass
    try:
        # Allocate the query bridge OUTSIDE any compiled region (required: dynamo cannot trace
        # allocation, and graph capture cannot allocate at all).
        from vllm import kvmem_bridge as _bridge
        hf = getattr(getattr(runner, "vllm_config", None), "model_config", None)
        hf = getattr(hf, "hf_config", None)
        tp = int(getattr(getattr(runner, "vllm_config", None).parallel_config,
                         "tensor_parallel_size", 1)) if getattr(runner, "vllm_config", None) else 1
        # The head counts may live on a nested text config (multimodal models), so fall back to the
        # attention modules themselves, which is also what the compiled graph actually uses.
        def _int_attr(obj, *names):
            for name in names:
                value = getattr(obj, name, None)
                if isinstance(value, int) and value > 0:
                    return value
            return 0

        heads = _int_attr(hf, "num_attention_heads") or _int_attr(
            getattr(hf, "text_config", None), "num_attention_heads")
        kv_heads = _int_attr(hf, "num_key_value_heads") or _int_attr(
            getattr(hf, "text_config", None), "num_key_value_heads") or heads
        # The hf config reports the GLOBAL kv-head count while this rank's key tensor is the TP
        # slice: measured key=(54,512) fp16 for a 4-head/256-dim model at TP=2, i.e. 2 heads per
        # rank. Sizing the capture buffers from the global count made every key fail the width
        # check and the mean-K statistics stayed all-zero (core scores 0.0 on every block).
        try:
            _tp = int(tp or 1)
            if _tp > 1 and int(kv_heads) % _tp == 0:
                _global_kv = int(kv_heads)
                kv_heads = _global_kv // _tp
                _append(log_path, {"kind": "kv_heads_per_rank", "pid": os.getpid(),
                                   "hf_global": _global_kv, "tp": _tp,
                                   "per_rank": int(kv_heads)})
        except Exception:
            pass
        if not heads:
            for _name, _mod in getattr(runner, "model", runner).named_modules():
                if type(_mod).__name__ == "Qwen3NextAttention":
                    heads = int(getattr(_mod, "num_heads", 0) or 0)
                    kv_heads = int(getattr(_mod, "num_kv_heads", kv_heads) or kv_heads)
                    # The TENSOR width is authoritative: with TP>1 `num_kv_heads` reports the GLOBAL
                    # count while the layer's key tensor is the per-rank slice. That mismatch made
                    # capture_k refuse every key and the statistics store filled with zeros
                    # (measured: key=(54,512) fp16 while the buffers were sized 4*256=1024).
                    _global_kv = int(kv_heads)
                    _per_rank = _global_kv
                    try:
                        _kv = int(getattr(_mod, "kv_size", 0) or 0)
                        _hs = int(getattr(_mod, "head_size", 0) or 0)
                        if _kv and _hs and _kv % _hs == 0 and _kv // _hs != _global_kv:
                            _per_rank = _kv // _hs
                        elif _global_kv % max(1, tp) == 0:
                            _per_rank = _global_kv // max(1, tp)
                    except Exception:
                        _per_rank = max(1, _global_kv // max(1, tp))
                    if _per_rank != _global_kv:
                        _append(log_path, {"kind": "kv_heads_corrected", "pid": os.getpid(),
                                           "num_kv_heads_global": _global_kv, "tp": tp,
                                           "per_rank_kv_heads": _per_rank})
                    kv_heads = _per_rank
                    head_dim = int(getattr(_mod, "head_size", head_dim) or head_dim)
                    break
        per_rank = max(1, heads // max(1, tp)) if heads else 0
        layers = int(getattr(hf, "num_hidden_layers", 0) or 64)
        head_dim = int(cfg.get("head_dim", 256))
        # The query buffer must be a FLOAT buffer, NOT the KV cache dtype. With `--kv-cache-dtype
        # fp8` the paged tensors are uint8/fp8, and allocating the query buffer with that dtype
        # made `buf.copy_(query)` truncate every query value in [-1, 1] to 0 - so the whole query
        # summary was a zero vector, every cosine came back 0, the ranking was flat, the compacted
        # view kept arbitrary blocks instead of the ones matching the question, and the generation
        # repeated (measured: q_dtype=torch.uint8, q_absmax=0.0, k_norm=4.35).
        import torch as _torch
        _act_dtype = getattr(getattr(runner, "vllm_config", None), "model_config", None)
        _act_dtype = getattr(_act_dtype, "dtype", None) or _torch.bfloat16
        if getattr(_act_dtype, "is_floating_point", False) is False:
            _act_dtype = _torch.bfloat16
        ok = bool(per_rank) and _bridge.ensure(layers=layers, heads=per_rank,
                                              head_dim=head_dim,
                                              device=getattr(view, "device", "cuda"),
                                              dtype=_act_dtype,
                                              kv_heads=kv_heads)
        # The mean-K statistics store needs the model's own geometry: `n_layer` is the number of
        # FULL-attention layers (the ones the capture op numbers), and `n_embd_k` is
        # kv_heads*head_dim. The MTP layer reports slot -1 and is not part of it.
        try:
            _types = None
            for _holder in (hf, getattr(hf, "text_config", None)):
                if _holder is None:
                    continue
                _types = getattr(_holder, "layer_types", None)
                if _types is None and isinstance(_holder, dict):
                    _types = _holder.get("layer_types")
                if _types:
                    break
            _full = sum(1 for _t in list(_types or []) if "full" in str(_t))
            runtime.statistics_layers = int(_full or 0)
            runtime.kv_heads = int(kv_heads or 0)
        except Exception:
            pass
        _append(log_path, {"kind": "bridge_ensure", "ok": bool(ok), "pid": os.getpid(),
                           "tokens": int(os.environ.get("KVMEM_CAPTURE_TOKENS", 0) or 0),
                           "layers": layers, "heads_per_rank": per_rank,
                           "kv_heads": kv_heads, "head_dim": head_dim, "tp": tp,
                           "buf_dtype": str(_act_dtype),
                           "statistics_layers": getattr(runtime, "statistics_layers", 0)})
    except Exception:
        _append(log_path, {"kind": "bridge_ensure_error", "error": traceback.format_exc()})
    global _ACTIVE_RUNTIME
    _ACTIVE_RUNTIME = runtime            # the early capture resolves its index through this
    if _CAPTURE is not None:
        runtime.capture = _CAPTURE
    try:
        cap = None
        # Diagnostic: aggregate EVERY attention-like module, not just the first few. A previous
        # run only printed four and wrongly suggested the text attention was absent.
        try:
            root = getattr(runner, "model", runner)
            groups, total = {}, 0
            for name, module in root.named_modules():
                cls = type(module)
                has_gate = hasattr(module, "_project_qkv_gate")
                if not (has_gate or "Attention" in cls.__name__):
                    continue
                total += 1
                marker = bool(getattr(getattr(cls, "_project_qkv_gate", None),
                                      "_kvmem_capture_patched", False))
                key = f"{cls.__module__}.{cls.__qualname__}|gate={has_gate}|patched={marker}"
                entry = groups.setdefault(key, {"count": 0, "example": name})
                entry["count"] += 1
            import sys as _sys
            our_ids, their_ids = [], []
            try:
                from vllm.model_executor.models import qwen3_next as _q3n
                our_ids.append(id(_q3n.Qwen3NextAttention))
            except Exception:
                pass
            for name, module in root.named_modules():
                if type(module).__name__ == "Qwen3NextAttention":
                    their_ids.append(id(type(module)))
                    break
            _append(log_path, {"kind": "capture_inspect", "pid": os.getpid(),
                               "our_class_id": our_ids[:1], "model_class_id": their_ids[:1],
                               "same_class": bool(our_ids and their_ids and our_ids[0] == their_ids[0]),
                               "sys_modules_qwen3_next": [k for k in _sys.modules
                                                          if "qwen3_next" in k],
                               "attention_total": total, "groups": groups,
                               "model_class": f"{type(root).__module__}."
                                              f"{type(root).__qualname__}",
                               "is_compiled": type(root).__name__ not in
                                              ("Qwen3_5ForConditionalGeneration",)})
        except Exception:
            _append(log_path, {"kind": "capture_inspect_error",
                               "error": traceback.format_exc()})
    except Exception:
        _append(log_path, {"kind": "capture_attach_error", "error": traceback.format_exc()})
    runtime.sessions = int(cfg.get("sessions", 4))
    runtime.needle_trace = int(cfg.get("needle_trace", 0))
    for _orch in ():  # placeholder: configured per request when the orchestrator is created
        pass
    # Bisect switches (env-only, for live diagnosis): turn one piece of the view install off at a
    # time to localise a corrupt read without editing code between engine restarts.
    runtime.install_row = bool(cfg.get("install_row", True)) and not os.environ.get("KVMEM_NO_ROW")
    runtime.install_seq = bool(cfg.get("install_seq", True)) and not os.environ.get("KVMEM_NO_SEQ")
    runtime.install_pos = bool(cfg.get("install_pos", True)) and not os.environ.get("KVMEM_NO_POS")

    # Hook 1b: give the FULL-ATTENTION metadata builder the compacted seq_lens. The GDN layers
    # must keep the true length, and they share one tensor, so the split has to happen at the
    # builder. FlashInfer derives its paged-kv indptr at build time, so it must see the
    # compacted value HERE rather than being patched afterwards.
    wrapped = 0
    try:
        for gi, groups in enumerate(getattr(runner, "attn_groups", None) or []):
            for attn_group in groups:
                spec = getattr(attn_group, "kv_cache_spec", None)
                if spec is None or type(spec).__name__ != "FullAttentionSpec":
                    continue
                for builder in getattr(attn_group, "metadata_builders", []) or []:
                    original_build = builder.build
                    if getattr(original_build, "_kvmem_patched", False):
                        continue

                    def build(common_prefix_len=0, common_attn_metadata=None, _orig=original_build,
                              **kwargs):
                        if common_attn_metadata is not None:
                            saved = common_attn_metadata.seq_lens
                            common_attn_metadata.seq_lens = runtime.attention_seq_lens(saved)
                            try:
                                return _orig(common_prefix_len=common_prefix_len,
                                             common_attn_metadata=common_attn_metadata, **kwargs)
                            finally:
                                common_attn_metadata.seq_lens = saved
                        return _orig(common_prefix_len=common_prefix_len,
                                     common_attn_metadata=common_attn_metadata, **kwargs)

                    build._kvmem_patched = True
                    builder.build = build
                    wrapped += 1
    except Exception:
        _append(log_path, {"kind": "sparse_builder_error", "pid": os.getpid(),
                           "error": traceback.format_exc()})

    # Hook 1: install the view right after the engine builds positions/seq_lens.
    original_prepare = mr.prepare_pos_seq_lens
    if not getattr(original_prepare, "_kvmem_patched", False):
        def prepare_pos_seq_lens(idx_mapping, query_start_loc, num_computed_tokens,
                                 positions, seq_lens):
            original_prepare(idx_mapping, query_start_loc, num_computed_tokens, positions,
                             seq_lens)
            try:
                if _CAPTURE is not None:
                    _CAPTURE._positions = positions
                # `query_start_loc` is the scheduler's own layout, and this hook runs for prefill and
                # decode alike (the sparse view already relies on it). Record each request's new
                # tokens here; `execute_model`'s post-block does not see prefill chunks.
                try:
                    rt = state.get("sparse")
                    # `_kvmem_req_ids` is filled by execute_model, which prefill chunks do not
                    # reach; the input batch is populated at prepare time and is in the same order
                    # as query_start_loc / num_scheduled_tokens.
                    batch = getattr(runner, "input_batch", None)
                    reqs = list(getattr(batch, "req_ids", None) or []) or \
                        list(getattr(runner, "_kvmem_req_ids", None) or [])
                    if rt is not None and len(reqs) == query_start_loc.shape[0] - 1:
                        for i, req in enumerate(reqs):
                            n = int(query_start_loc[i + 1] - query_start_loc[i])
                            # `num_computed_tokens` is the scheduler's authoritative frontier; no
                            # accumulation, so a missed step cannot desynchronise our counter.
                            try:
                                base = int(num_computed_tokens[i])
                            except Exception:
                                base = int(num_computed_tokens)
                            rt.advance(req, base + n)
                except Exception:
                    pass
                view.bind_step(idx_mapping, getattr(runner, "_kvmem_req_ids", None) or [],
                               query_start_loc)
                live = list(getattr(runner, "_kvmem_req_ids", []) or [])
                # A request that disappeared from the batch has finished: give its H share back,
                # otherwise a long-lived engine leaks the shared host budget.
                for gone in [r for r in runtime.live_requests() if r not in live]:
                    runtime.release(gone)
                    _append(log_path, {"kind": "released", "req": gone, "pid": os.getpid()})
                for req in live:
                    if view.row_index(req) is None:
                        continue
                    runtime.before_step(req, row_index=view.row_index(req),
                                        position_slots=view.query_position_slots(req))
            except Exception:
                _append(log_path, {"kind": "sparse_step_error", "pid": os.getpid(),
                                   "error": traceback.format_exc()})

        prepare_pos_seq_lens._kvmem_patched = True
        mr.prepare_pos_seq_lens = prepare_pos_seq_lens

    _append(log_path, {"kind": "sparse_attached", "pid": os.getpid(),
                       "builders_wrapped": wrapped,
                       "page_tokens": view.page_tokens,
                       "budget_tokens": sel_cfg.budget_tokens,
                       "reselect_interval": runtime.reselect_interval})
    return runtime


def _attach_sched_probe(log_path):
    """Read-only: dump what the SCHEDULER holds per request (block inventory per cache group).

    This is the ground truth for the allocation-side work: the worker-side view can shrink the
    attention read, but only the scheduler owns and frees blocks, so the inventory (which group
    holds how many blocks, and how many tokens are computed) decides what can be released.
    """
    try:
        from vllm.v1.core.sched.scheduler import Scheduler
    except Exception:
        _append(log_path, {"kind": "sched_attach_error", "error": traceback.format_exc()})
        return
    original = Scheduler.schedule
    if getattr(original, "_kvmem_patched", False):
        return
    state = {"step": 0, "last": -1}

    def schedule(self, *args, **kwargs):
        # QUIESCENT POINT: the previous step's outputs are already processed and no request has
        # pending placeholders, so this is the only place where a reselect (which rewrites the
        # block table) cannot violate the async scheduler's invariants. Token accounting and
        # reselection both run here, per request.
        try:
            rt = globals().get("_ACTIVE_RUNTIME")
            if rt is not None:
                seen = state.setdefault("frontier", {})
                for req in list(getattr(self, "running", []) or []):
                    rid = getattr(req, "request_id", None)
                    if rid is None:
                        continue
                    frontier = int(getattr(req, "num_computed_tokens", 0)) + \
                        len(getattr(req, "output_token_ids", ()) or ())
                    delta = frontier - int(seen.get(rid, 0))
                    if delta > 0:
                        seen[rid] = frontier
                        rt.advance_tokens(rid, delta)
        except Exception:
            _append(log_path, {"kind": "sched_advance_error", "error": traceback.format_exc()})
        out = original(self, *args, **kwargs)
        try:
            state["step"] += 1
            every = int(_config().get("sched_probe_every", 8))
            if state["step"] % every:
                return out
            running = list(getattr(self, "running", []) or [])
            if not running:
                return out
            req = running[0]
            mgr = getattr(self, "kv_cache_manager", None)
            ids = mgr.get_block_ids(req.request_id) if mgr else ()
            counts = [len(group) for group in ids]
            # How many blocks carry real data (the tail may be a partially-filled block).
            computed = getattr(req, "num_computed_tokens", -1)
            record = {"kind": "sched_inventory", "step": state["step"],
                      "req": req.request_id, "groups": counts,
                      "computed_tokens": int(computed),
                      "num_running": len(running),
                      "free_blocks": (int(mgr.block_pool.get_num_free_blocks())
                                      if mgr is not None and hasattr(mgr.block_pool,
                                                                     "get_num_free_blocks")
                                      else -1)}
            if counts:
                record["attention_group"] = int(max(range(len(counts)), key=lambda i: counts[i]))
                record["attention_blocks"] = int(max(counts))
            _append(log_path, record)
        except Exception:
            _append(log_path, {"kind": "sched_probe_error", "error": traceback.format_exc()})
        return out

    schedule._kvmem_patched = True
    Scheduler.schedule = schedule
    _append(log_path, {"kind": "sched_probe_attached", "pid": os.getpid()})


def _attach_free_path(log_path):
    """Free the MIDDLE aligned blocks of long requests back to vLLM's pool.

    Why this is safe: the worker compacts the attention block-table row to the sparse working
    set, so a block outside the sink prefix and outside the recent tail is never read again
    (the window moves forward monotonically). Verified on the live engine: 20 of 39 blocks
    released for a 60K request with the answer unchanged.

    Restore is NOT needed in recency selection for the same reason: a block that slides out of
    the window can never re-enter it. Retrieval selection would need a worker->scheduler channel.
    """
    try:
        from vllm.v1.core.sched.scheduler import Scheduler
    except Exception:
        _append(log_path, {"kind": "free_attach_error", "error": traceback.format_exc()})
        return
    original = Scheduler.schedule
    if getattr(original, "_kvmem_free_patched", False):
        return
    state = {"freed": 0}

    def schedule(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        try:
            cfg = _config()
            if not cfg.get("free_path", False):
                return out
            mgr = getattr(self, "kv_cache_manager", None)
            running = list(getattr(self, "running", []) or [])
            if mgr is None or not running:
                return out
            pool = mgr.block_pool
            free_now = pool.get_num_free_blocks()
            total = getattr(pool, "num_gpu_blocks", None) or getattr(pool, "capacity", None)
            if total is None:
                blocks = getattr(pool, "blocks", None)
                total = len(blocks) if blocks else 0
            # Pressure trigger: only reclaim when the pool is genuinely short.
            low = float(cfg.get("free_low_watermark", 0.15))
            if total and free_now > total * low:
                return out
            groups = mgr.kv_cache_config.kv_cache_groups
            attn = [i for i, g in enumerate(groups)
                    if type(getattr(g, "kv_cache_spec", None)).__name__ == "FullAttentionSpec"]
            if not attn:
                return out
            gi = attn[0]
            sink = int(cfg.get("sink_blocks", 5))
            recent = int(cfg.get("recent_blocks", 2))
            budget = int(cfg.get("free_per_step", 2))
            for req in running:
                manager = mgr.coordinator.single_type_managers[gi]
                blocks = manager.req_to_blocks.get(req.request_id) or []
                # Only fully-written blocks outside [sink, len-recent) may go.
                last_free = len(blocks) - recent
                if last_free <= sink:
                    continue
                before = pool.get_num_free_blocks()
                manager._remove_blocks_in_range(req.request_id, sink, sink + budget)
                after = pool.get_num_free_blocks()
                if after > before:
                    state["freed"] += after - before
                    _append(log_path, {"kind": "freed_blocks", "req": req.request_id,
                                       "freed": after - before, "free_now": after,
                                       "total_released": state["freed"],
                                       "held": len(manager.req_to_blocks[req.request_id]),
                                       "pool_total": total})
                break
        except Exception:
            _append(log_path, {"kind": "free_path_error", "error": traceback.format_exc()})
        return out

    schedule._kvmem_free_patched = True
    Scheduler.schedule = schedule
    _append(log_path, {"kind": "free_path_attached", "pid": os.getpid()})


_CAPTURE = None
_ACTIVE_RUNTIME = None


def _install_capture_early(log_path, gen_reserve):
    """Install the pre-RoPE hook BEFORE the model is built.

    torch.compile traces the language model at engine start-up, so a class patch applied on the
    first request (the natural place, where the runtime finally knows its geometry) is captured
    too late: the compiled graph keeps calling the ORIGINAL method (observed as calls==0 while
    same_class==True). Installing here, at plugin registration, puts the wrapper in the trace.
    """
    global _CAPTURE
    if _CAPTURE is not None:
        return _CAPTURE
    try:
        # register() runs before the plugin's lazy path setup, so make the adapter importable now.
        import os as _os
        import sys as _sys
        _src = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src")
        if _src not in _sys.path:
            _sys.path.insert(0, _src)
        from kvmem_engine_capture import EngineCapture

        def _active_index():
            runtime = _ACTIVE_RUNTIME
            if runtime is None:
                return None
            for req in runtime.live_requests():
                index = runtime.retrieval_index(req)
                if index is not None:
                    return index
            return None

        capture = EngineCapture(_active_index, log=lambda rec: _append(log_path, rec),
                               gen_reserve=int(gen_reserve))
        try:
            from kvmem_engine_capture import install_op_override
            if install_op_override(capture, log=lambda rec: _append(log_path, rec)):
                _append(log_path, {"kind": "capture_op_attached", "pid": os.getpid()})
        except Exception:
            _append(log_path, {"kind": "capture_op_error", "error": traceback.format_exc()})
        # `install()` now only registers the legacy method wrapper, which is disabled (it hid the
        # fork bridge from the graph). The capture object itself is still needed: it owns the
        # inv_freq cache and the index getter used by harvest_from_bridge.
        _CAPTURE = capture
        _append(log_path, {"kind": "capture_attached", "pid": os.getpid(),
                           "phase": "register", "retrieval": True,
                           "legacy_wrapper": bool(capture.install())})
    except Exception:
        _append(log_path, {"kind": "capture_attach_error", "error": traceback.format_exc()})
    return _CAPTURE


def register():
    global _PATCHED
    cfg = _config()
    probe, smoke = cfg.get("probe_log"), cfg.get("smoke_log")
    sparse = bool(cfg.get("sparse", False))
    _boot(f"register enabled={cfg.get('enabled', True)} probe={bool(probe)} "
          f"smoke={bool(smoke)} sparse={sparse} config_keys={sorted(cfg)}")
    # The KVMem settings arrive VIA vLLM's config (--additional-config), which is not visible at
    # plugin-registration time: there is no runner yet. So the hooks are always installed and each
    # step re-reads the config; `_config_from_runner` publishes it the first time a runner shows up.
    # Without this the plugin would decide "sparse=false" before it could ever see the settings.
    if _PATCHED:
        return
    _PATCHED = True
    # Sparse mode rewrites block tables, so a wedge has to be diagnosable in-process: ptrace is
    # usually unavailable on this box, but faulthandler works from inside the process.
    if os.environ.get("KVMEM_WATCHDOG", "1") == "1":
        _install_watchdog(probe or "/tmp/kvmem_sparse.jsonl")
    # Early, before any model exists (see _install_capture_early).
    _install_capture_early(probe or "/tmp/kvmem_sparse.jsonl",
                           int(cfg.get("gen_reserve_tokens", 16384)))
    import importlib
    runners = []
    for module_name in ("vllm.v1.worker.gpu_model_runner", "vllm.v1.worker.gpu.model_runner"):
        try:
            module = importlib.import_module(module_name)
            runners.append((module_name, module.GPUModelRunner))
        except Exception:
            _boot(f"import {module_name} failed: " + traceback.format_exc())
    if not runners:
        return
    # Block channel (Route A). The settings are NOT visible at register() time in EngineCore (no
    # runner exists there), so the hook is installed unconditionally and the socket server starts
    # lazily on the first schedule whose vllm_config carries `block_channel`.
    try:
        from vllm.v1.core.sched.scheduler import Scheduler
        from kvmem_block_channel import BlockChannelScheduler

        def _sched_cfg(scheduler):
            try:
                raw = getattr(getattr(scheduler, "vllm_config", None),
                              "additional_config", None) or {}
                kv = raw.get("kvmem") if isinstance(raw, dict) else None
                return kv if isinstance(kv, dict) else {}
            except Exception:
                return {}

        def _alloc(request_id, count):
            mgr = getattr(_channel_state.get("scheduler"), "kv_cache_manager", None)
            if mgr is None:
                raise RuntimeError("no kv_cache_manager")
            blocks = mgr.block_pool.get_new_blocks(int(count))
            return [int(getattr(b, "block_id", b)) for b in blocks]

        def _replay(request_id, visible_tokens, replay_tokens):
            """KVMem QueryReplay: rewind BOTH books so the engine re-schedules the tail.

            Prefill-time compaction keeps sink+newest only (no query exists yet), so the tokens
            prefilled while the middle was evicted saw an approximate history. Once the query-driven
            reselect has installed the real working set, rewinding `num_computed_tokens` (resident)
            and `num_logical_tokens` (served) by N makes the scheduler prefill those N tokens again -
            against the full B - refreshing their KV and the recurrent state just before sampling.
            """
            scheduler = _channel_state.get("scheduler")
            if scheduler is None or not request_id:
                return {"ok": False, "error": "no scheduler"}
            req = None
            for r in list(getattr(scheduler, "running", []) or []):
                if getattr(r, "request_id", None) == request_id:
                    req = r
                    break
            if req is None:
                req = (getattr(scheduler, "requests", None) or {}).get(request_id)
            if req is None:
                return {"ok": False, "error": "unknown request"}
            visible = int(visible_tokens or 0) or int(getattr(req, "num_computed_tokens", 0))
            # BOTH TP ranks send the same request (each owns its own KV), but rewinding the two books
            # is a single EngineCore operation: applying it twice rewound `num_logical_tokens` again
            # while `num_computed_tokens` had already reached its floor, which changed the offset by
            # N and shifted every position (measured: "ZX" followed by garbage).
            _seen = _replay_state.get(request_id)
            if _seen == visible:
                return {"ok": True, "skipped": "duplicate", "computed": int(req.num_computed_tokens)}
            n = min(int(replay_tokens or 0), visible,
                    int(getattr(req, "num_logical_tokens", 0) or 0))
            if n <= 0:
                return {"ok": False, "error": "nothing to replay"}
            before = (int(req.num_computed_tokens), int(getattr(req, "num_logical_tokens", 0)))
            req.num_computed_tokens = max(0, visible - n)
            req.num_logical_tokens = max(0, int(getattr(req, "num_logical_tokens", 0)) - n)
            # Keep the two books related exactly as the scheduler expects.
            req.num_kvmem_offset = int(req.num_logical_tokens) - int(req.num_computed_tokens)
            entry = {"kind": "replay_applied", "pid": os.getpid(), "req": request_id,
                     "n": int(n), "visible": visible,
                     "before_computed": before[0], "before_logical": before[1],
                     "computed": int(req.num_computed_tokens),
                     "logical": int(req.num_logical_tokens),
                     "offset": int(req.num_kvmem_offset)}
            _replay_state[request_id] = visible
            _append(_channel_state.get("probe_log") or "/tmp/kvmem_sparse.jsonl", entry)
            return {"ok": True, "replay": int(n), "computed": int(req.num_computed_tokens)}

        def _compact(request_id, pages, page_tokens, visible_tokens=0, rebuild=False,
                     block_pages=0):
            """Resident-book reconciliation (Route A / resident model).

            The worker selected a working set and has ALREADY installed it into the block table row
            (the row IS the working set). EngineCore only has to agree on how long the resident KV
            book is: `num_computed_tokens` becomes the compacted length, while the LOGICAL token
            list (`_all_token_ids`) keeps growing untouched. That decoupling is what lets a long
            prompt be prefilled in chunks with a working set of B+R: the engine never thinks the
            request holds more KV than the pool can, so it never allocates past the working area and
            never preempts.
            """
            scheduler = _channel_state.get("scheduler")
            if scheduler is None or not pages or page_tokens <= 0:
                return False
            req = None
            for r in list(getattr(scheduler, "running", []) or []):
                if getattr(r, "request_id", None) == request_id:
                    req = r
                    break
            if req is None:
                req = (getattr(scheduler, "requests", None) or {}).get(request_id)
            if req is None:
                return False
            # The worker's own visible length wins: it is `min(tokens, len(pages)*page)` and can be
            # a few tokens SHORTER than the page product. Using the product here made the engine's
            # `num_computed_tokens` disagree with the installed row by up to a page, shifting every
            # later position and corrupting the context (measured: 80K prompts answered with noise
            # whenever prefill-time compaction ran).
            visible = int(visible_tokens or 0) or int(len(pages)) * int(page_tokens)
            before_computed = int(req.num_computed_tokens)
            applied = _compact_state.get(request_id)
            already = applied == visible
            if already and before_computed <= visible:
                # TP-rank idempotency: the peer rank already applied this compaction. Still report
                # `truncated` so the second rank installs its own view (each rank owns its KV).
                return {"ok": True, "truncated": True, "visible": visible}
            req.num_computed_tokens = max(0, visible)
            # The served count and the resident count differ by exactly this rewind, and nothing
            # else: the scheduler re-derives `num_logical = num_computed + offset` after every
            # schedule, so the offset MUST be updated together with the rewind. Without it the two
            # books disagree after the first compaction and the scheduler over-schedules.
            req.num_kvmem_offset = int(getattr(req, "num_logical_tokens", 0)) \
                - int(req.num_computed_tokens)
            if rebuild and request_id in _rebuilt_requests:
                return {"ok": True, "truncated": True, "rebuild": True, "skipped": "already rebuilt"}
            if rebuild:
                # RECONSTRUCTION (upstream's CheckpointPurpose::Reconstruction): rewrite the
                # request's token book to the COMPACTED sequence and rewind both counters to zero,
                # so the engine prefills the selected pages from scratch. The recurrent state is
                # zeroed by the worker beforehand, which is what removes the approximation that
                # prefill-time compaction leaves in it.
                try:
                    pages = [int(p) for p in pages]
                    ids = list(getattr(req, "_all_token_ids", []) or [])
                    new_ids = []
                    for p in pages:
                        new_ids.extend(ids[p * page_tokens:(p + 1) * page_tokens])
                    if new_ids:
                        new_ids = new_ids[:visible] if visible else new_ids
                        req._all_token_ids[:] = new_ids
                        req.num_prompt_tokens = len(new_ids)
                        req.num_computed_tokens = 0
                        req.num_logical_tokens = 0
                        req.num_kvmem_offset = 0
                        # Drafts and placeholders describe the sequence BEFORE the rewrite; keeping
                        # them makes the next schedule ask for more logits rows than it schedules
                        # (measured: scheduled=4, logits=5 -> assert in prepare_inputs).
                        try:
                            req.spec_token_ids = []
                            # One placeholder = "a token is in flight", which is the state of a
                            # request that has JUST finished prefilling. Setting 0 made the sampler
                            # emit nothing after the rebuild ("no sampled token", 26-minute stall).
                            req.num_output_placeholders = 1
                        except Exception:
                            pass
                        _compact_state[request_id] = visible      # peer message sees "already applied"
                        _append(_channel_state.get("probe_log") or "/tmp/kvmem_sparse.jsonl",
                                {"kind": "compact_rebuilt", "req": request_id,
                                 "pages": len(pages), "tokens": len(new_ids)})
                        _rebuilt_requests.add(request_id)
                        return {"ok": True, "truncated": True, "rebuild": True,
                                "tokens": len(new_ids)}
                except Exception:
                    _append(_channel_state.get("probe_log") or "/tmp/kvmem_sparse.jsonl",
                            {"kind": "compact_rebuild_error", "req": request_id,
                             "error": traceback.format_exc()[-400:]})
            # Adopt the compacted block list in the ENGINE. The paged table is a flat reflection of
            # the request's engine blocks (a block owns `block_pages` consecutive page slots, block i
            # at pages i*block_pages...), so after installing a shorter row the ENGINE must also
            # think the request has exactly those blocks - otherwise the next allocation appends at
            # the OLD block position and leaves a gap of stale page ids that the kernel then attends
            # to (this is the beyond-pool illegal access).
            try:
                if int(block_pages or 0) > 1 and pages:
                    _bp = int(block_pages)
                    _sel = [int(p) for p in pages]
                    _blocks = sorted({p // _bp for p in _sel})
                    _whole = all(len([p for p in _sel if p // _bp == b]) == _bp for b in _blocks)
                    if not _whole:
                        _append(_channel_state.get("probe_log") or "/tmp/kvmem_sparse.jsonl",
                                {"kind": "compact_blocks_partial", "req": request_id,
                                 "pages": len(_sel), "blocks": len(_blocks), "block_pages": _bp})
                    _mgr = None
                    for _m in list(getattr(scheduler.kv_cache_manager.coordinator,
                                           "single_type_managers", ()) or ()):
                        if "FullAttention" in type(_m).__name__:
                            _mgr = _m
                            break
                    _map = getattr(_mgr, "req_to_blocks", None) if _mgr is not None else None
                    if _map is not None:
                        _new = []
                        for _b in _blocks:
                            _obj = None
                            try:
                                _obj = scheduler.kv_cache_manager.block_pool.blocks[_b]
                            except Exception:
                                _obj = None
                            if _obj is not None:
                                _new.append(_obj)
                        _old = list(_map.get(request_id) or ())
                        _freed = [b for b in _old if b not in _new]
                        if _new:
                            _map[request_id] = _new
                        if _freed:
                            scheduler.kv_cache_manager.block_pool.free_blocks(_freed)
                        _append(_channel_state.get("probe_log") or "/tmp/kvmem_sparse.jsonl",
                                {"kind": "compact_blocks_adopted", "req": request_id,
                                 "blocks": len(_new), "freed": len(_freed),
                                 "whole": bool(_whole),
                                 "free_blocks": int(scheduler.kv_cache_manager.block_pool
                                                    .get_num_free_blocks())})
            except Exception:
                _append(_channel_state.get("probe_log") or "/tmp/kvmem_sparse.jsonl",
                        {"kind": "compact_blocks_error", "req": request_id,
                         "error": traceback.format_exc()[-300:]})
            # Pool accounting: a compaction rewinds the resident book but the ENGINE still owns the
            # evicted blocks (vLLM frees blocks only at request end), so every compaction can leak
            # them and the pool fills until the scheduler preempts - which then recomputes against a
            # compacted row. Record the free-block count so the leak is visible.
            try:
                _pool = getattr(getattr(scheduler, "kv_cache_manager", None), "block_pool", None)
                _free = int(_pool.get_num_free_blocks()) if _pool is not None else -1
                _append(_channel_state.get("probe_log") or "/tmp/kvmem_sparse.jsonl",
                        {"kind": "compact_pool", "req": request_id, "visible": int(visible),
                         "free_blocks": _free})
            except Exception:
                pass
            # Pool accounting: a compaction rewinds the resident book but the ENGINE still owns the
            # evicted blocks (vLLM frees blocks only at request end), so every compaction can leak
            # them, the pool fills, and the scheduler preempts - which then recomputes against a
            # compacted row. Record the free-block count so the leak is visible.
            try:
                _pool = getattr(getattr(scheduler, "kv_cache_manager", None), "block_pool", None)
                _free = int(_pool.get_num_free_blocks()) if _pool is not None else -1
                _append(_channel_state.get("probe_log") or "/tmp/kvmem_sparse.jsonl",
                        {"kind": "compact_pool", "req": request_id, "visible": int(visible),
                         "free_blocks": _free})
            except Exception:
                pass
            _compact_state[request_id] = visible
            # Hand the engine's OWN bookkeeping the compacted row. Until now only the worker's block
            # table was rewritten, so the engine kept the pre-compaction block list (and kept
            # allocating new blocks for every chunk on top of it): the pool drained to a handful of
            # free blocks and the scheduler preempted, whose recompute then faced a compacted row.
            _gt = set(int(p) for p in pages)
            try:
                if os.environ.get("KVMEM_COMPACT_FREE", "1") == "1":
                    _pool = scheduler.kv_cache_manager.block_pool
                    # The engine keeps its OWN per-request block list (per KV cache group) inside the
                    # coordinators; `req.block_ids` is not populated in this fork. Rewriting only the
                    # worker's block table left it holding the pre-compaction blocks forever, so
                    # every chunk allocated new ones and the pool drained.
                    _groups = []
                    try:
                        _coord = scheduler.kv_cache_manager.coordinator
                        _groups = list(getattr(_coord, "single_type_managers", ()) or ())
                    except Exception:
                        _groups = []
                    _blocks = getattr(req, "block_ids", None)
                    if not _groups:
                        _append(_channel_state.get("probe_log") or "/tmp/kvmem_sparse.jsonl",
                                {"kind": "compact_free_nogroups", "req": request_id})
                    for _mgr in _groups:
                        _map = getattr(_mgr, "req_to_blocks", None)
                        if not _map:
                            continue
                        _cur = _map.get(request_id)
                        if not _cur:
                            continue
                        _keep = [b for b in _cur if int(getattr(b, "block_id", -1)) in _gt]
                        _gone = [b for b in _cur if int(getattr(b, "block_id", -1)) not in _gt]
                        if _gone:
                            _pool.free_blocks(_gone)
                            _map[request_id] = _keep
                        _append(_channel_state.get("probe_log") or "/tmp/kvmem_sparse.jsonl",
                                {"kind": "compact_freed", "req": request_id,
                                 "group": type(_mgr).__name__, "kept": len(_keep),
                                 "freed": len(_gone),
                                 "free_blocks": int(_pool.get_num_free_blocks())})
                    if _blocks:
                        _kept = [int(p) for p in pages]
                        _kept_set = set(_kept)
                        _old = list(_blocks[0])
                        _drop = [int(b) for b in _old if int(b) not in _kept_set]
                        if _drop:
                            _objs = [_pool.blocks[b] for b in _drop
                                     if 0 <= b < len(getattr(_pool, "blocks", []))]
                            _pool.free_blocks(_objs)
                        _blocks[0] = _kept
                        _append(_channel_state.get("probe_log") or "/tmp/kvmem_sparse.jsonl",
                                {"kind": "compact_freed", "req": request_id,
                                 "kept": len(_kept), "freed": len(_drop),
                                 "free_blocks": int(_pool.get_num_free_blocks())})
            except Exception:
                _append(_channel_state.get("probe_log") or "/tmp/kvmem_sparse.jsonl",
                        {"kind": "compact_free_error", "req": request_id,
                         "error": traceback.format_exc()[-300:]})
            entry = {"kind": "compact_applied", "pid": os.getpid(),
                     "req": request_id, "pages": len(pages),
                     "visible_tokens": visible,
                     "before_computed": before_computed,
                     "after_computed": int(req.num_computed_tokens),
                     "offset": int(getattr(req, "num_kvmem_offset", 0)),
                     "logical_tokens": int(getattr(req, "num_logical_tokens", 0)),
                     "num_tokens": int(req.num_tokens)}
            # The logical book must stay >= the prompt: the sampler derives the output length from
            # it, and we no longer truncate it - a violation here is a bug, not a config.
            if entry["num_tokens"] < len(getattr(req, "prompt_token_ids", ()) or ()):
                entry["WARNING"] = "logical book below the prompt"
            _append(_sched_cfg(scheduler).get("probe_log") or "/tmp/kvmem_sparse.jsonl", entry)
            return {"ok": True, "truncated": True, "visible": visible}

        _channel_state = {"scheduler": None, "server": None}
        # request_id -> visible length of the last applied compaction (TP-rank idempotency).
        _compact_state = {}
        _replay_state = {}       # request_id -> visible length of the last applied replay
        _rebuilt_requests = set()  # requests whose token book was rewritten (reconstruction)
        # request ids already examined for a resume (a waiting request is looked at once).
        _resume_decided = set()

        def _resume_decided_is_enabled(scheduler=None):
            """Session reuse is opt-in and only meaningful with in-request compaction on.

            The config MUST come from the scheduler's own `vllm_config.additional_config`: this runs
            in EngineCore, which has no model runner, so `_config()` (fed from the runner) is empty
            there and the whole resume path would silently never arm.
            """
            cfg = None
            try:
                ac = getattr(getattr(scheduler, "vllm_config", None),
                             "additional_config", None) or {}
                cfg = ac.get("kvmem") if isinstance(ac, dict) else None
            except Exception:
                cfg = None
            if not isinstance(cfg, dict) or not cfg:
                cfg = _config() or {}
            try:
                return bool(cfg.get("session_reuse", False)) and bool(
                    cfg.get("inrequest_compaction", False))
            except Exception:
                return False

        def _session_park_payload(payload):
            """EngineCore side of a park: keep the routing geometry, never the KV."""
            try:
                from kvmem_session_store import get_meta_store
                hashes = [bytes.fromhex(str(h)) for h in payload.get("hashes", [])]
                get_meta_store().park(
                    hashes,
                    covered_tokens=int(payload.get("covered", 0) or 0),
                    page_tokens=int(payload.get("page_tokens", 0) or 0),
                    pages_per_block=int(payload.get("pages_per_block", 1) or 1),
                    capacity_pages=int(payload.get("capacity_pages", 0) or 0),
                    has_state=bool(payload.get("has_state", False)))
                _append(_sched_cfg(_channel_state.get("scheduler")).get("probe_log")
                        or "/tmp/kvmem_sparse.jsonl",
                        {"kind": "session_meta_parked", "pid": os.getpid(),
                         "blocks": len(hashes),
                         "covered": int(payload.get("covered", 0) or 0),
                         "capacity": int(payload.get("capacity_pages", 0) or 0)})
            except Exception:
                _append(_sched_cfg(_channel_state.get("scheduler")).get("probe_log")
                        or "/tmp/kvmem_sparse.jsonl",
                        {"kind": "session_meta_error", "error": traceback.format_exc()})

        def _maybe_resume(scheduler):
            """Point a fresh request at a parked session: only the two books move, nothing else.

            vLLM already reads `request.num_computed_tokens` for a waiting request (prefix caching
            is off here, so every new request takes that branch) and allocates blocks for
            `[0, num_computed + n)`. Setting the resident book to |sel| pages therefore makes the
            engine allocate exactly the pages the worker will fill from RAM, and setting the served
            book to the hit length makes it schedule ONLY the genuinely new tokens.
            """
            if not _resume_decided_is_enabled(scheduler):
                return
            try:
                from kvmem_session_store import block_hash_sequence, get_meta_store
                store = get_meta_store()
                if not len(store):
                    return
                cache = getattr(scheduler, "cache_config", None)
                block_size = int(getattr(cache, "block_size", 0) or 0)
                if block_size <= 0:
                    return
                waiting = list(getattr(scheduler, "waiting", None) or [])
                live = {getattr(r, "request_id", None) for r in waiting}
                for rid in list(_resume_decided):
                    if rid not in live:
                        _resume_decided.discard(rid)
                for request in waiting:
                    rid = getattr(request, "request_id", None)
                    if rid is None or rid in _resume_decided:
                        continue
                    _resume_decided.add(rid)
                    if int(getattr(request, "num_computed_tokens", 0) or 0) > 0:
                        continue
                    ids = list(getattr(request, "prompt_token_ids", None) or ())
                    if len(ids) < block_size:
                        _append(_sched_cfg(scheduler).get("probe_log")
                                or "/tmp/kvmem_sparse.jsonl",
                                {"kind": "resume_skip", "req": rid, "why": "short prompt",
                                 "n": len(ids), "block": block_size, "pid": os.getpid()})
                        continue
                    _hashes = block_hash_sequence(ids, block_size)
                    found = store.lookup(_hashes)
                    if found is None:
                        _append(_sched_cfg(scheduler).get("probe_log")
                                or "/tmp/kvmem_sparse.jsonl",
                                {"kind": "resume_skip", "req": rid, "why": "no prefix match",
                                 "blocks": len(_hashes), "parked": len(store), "pid": os.getpid()})
                        continue
                    meta, blocks = found
                    hit = int(blocks) * block_size
                    pt = int(meta.page_tokens or 0)
                    ppb = max(1, int(meta.pages_per_block or 1))
                    if pt <= 0:
                        continue
                    pages = min(int(meta.capacity_pages or 0), hit // pt)
                    pages = (pages // ppb) * ppb
                    if pages <= 0:
                        continue
                    visible = pages * pt
                    request.num_computed_tokens = visible
                    request.num_logical_tokens = hit
                    request.num_kvmem_offset = hit - visible
                    _append(_sched_cfg(scheduler).get("probe_log")
                            or "/tmp/kvmem_sparse.jsonl",
                            {"kind": "resume_decided", "pid": os.getpid(), "req": rid,
                             "hit": hit, "resident": visible, "pages": pages,
                             "offset": hit - visible,
                             "prompt": int(getattr(request, "num_prompt_tokens", 0) or 0)})
            except Exception:
                _append("/tmp/kvmem_sparse.jsonl",
                        {"kind": "resume_decide_error", "error": traceback.format_exc()})

        original = Scheduler.schedule
        if not getattr(original, "_kvmem_channel_patched", False):
            def schedule(self, *args, **kwargs):
                _channel_state["scheduler"] = self
                if not hasattr(self, "kvmem_no_preempt"):
                    # Preemption rewinds a request to a recompute, which cannot be done safely for a
                    # KVMem view (it faults in the attention kernel). KVMem always waits instead;
                    # KVMEM_NO_PREEMPT=0 opts out. The environment is NOT reliable inside
                    # EngineCore, so the config is the source of truth, with the default being "wait".
                    _scfg0 = _sched_cfg(self)
                    self.kvmem_no_preempt = str(
                        _scfg0.get("no_preempt", "1")).strip().lower() not in ("0", "false", "no")
                _channel_state["probe_log"] = _sched_cfg(self).get("probe_log")
                if _channel_state["server"] is None:
                    scfg = _sched_cfg(self)
                    if scfg.get("block_channel") and scfg.get("inrequest_compaction"):
                        _channel_state["server"] = BlockChannelScheduler(
                            _alloc, path=scfg.get("block_channel_path",
                                                  "/tmp/kvmem_block_channel.sock"),
                            compactor=_compact, session_sink=_session_park_payload,
                            replayer=_replay).start()
                        _append(scfg.get("probe_log") or "/tmp/kvmem_sparse.jsonl",
                                {"kind": "block_channel_attached", "pid": os.getpid()})
                _maybe_resume(self)
                return original(self, *args, **kwargs)
            schedule._kvmem_channel_patched = True
            Scheduler.schedule = schedule
    except Exception:
        _append(cfg.get("probe_log") or "/tmp/kvmem_sparse.jsonl",
                {"kind": "block_channel_error", "error": traceback.format_exc()})
    if bool(cfg.get("free_path", False)) or bool(cfg.get("free_probe", False)):
        _attach_free_path(cfg.get("probe_log") or "/tmp/kvmem_free.jsonl")
    if bool(cfg.get("sched_probe", False)):
        _attach_sched_probe(cfg.get("probe_log") or "/tmp/kvmem_sched.jsonl")
    state = {"adapter_step": None, "probed": False, "attach_failed": False,
             "sparse": None, "sparse_failed": False,
             "sparse_cfg": bool(cfg.get("sparse", False)),
             "trace": bool(cfg.get("trace", False))}
    for module_name, cls in runners:
        _install(cls, module_name, probe, smoke, state)


def _install(cls, module_name, probe, smoke, state):
    original = cls.execute_model

    def execute_model(self, scheduler_output, *args, **kwargs):
        _config_from_runner(self)
        # Cross-request prefix reuse MUST be recorded before anything calls into the sparse runtime:
        # the runtime creates a request's orchestrator on first sight (`_ensure`), and a resumed
        # request can only ADOPT the parked one if the promotion hint is already on the runtime.
        # Recording it later (inside prepare_inputs) was measured: a brand-new orchestrator was
        # opened for the resumed request and the engine then read an empty row.
        _rt0 = state.get("sparse")
        if _rt0 is not None:
            try:
                for _nr in (getattr(scheduler_output, "scheduled_new_reqs", None)
                            or getattr(scheduler_output, "new_reqs", None) or ()):
                    _rid = getattr(_nr, "req_id", None)
                    if _rid is None:
                        continue
                    _ids = getattr(_nr, "prompt_token_ids", None)
                    _rt0.note_prompt_ids(_rid, _ids)
                    _nlg = int(getattr(_nr, "num_logical_tokens", 0) or 0)
                    _npc = int(getattr(_nr, "num_computed_tokens", 0) or 0)
                    if _nlg > _npc:
                        _rt0.note_resume(_rid, _nlg, _npc)
                    if bool(_config().get("session_reuse", False)):
                        _append(smoke or probe or "/tmp/kvmem_sparse.jsonl",
                                {"kind": "note_prompt_ids", "req": str(_rid),
                                 "n": len(_ids or ()), "nlg": _nlg, "npc": _npc,
                                 "pid": os.getpid()})
            except Exception:
                _append(smoke or probe or "/tmp/kvmem_sparse.jsonl",
                        {"kind": "note_new_reqs_error", "error": traceback.format_exc()})
        # The sparse pre-hook runs inside prepare_pos_seq_lens(); it needs this step's request
        # ids, which only the scheduler output knows.
        try:
            self._kvmem_req_ids = list(getattr(scheduler_output, "num_scheduled_tokens", {}) or {})
            _rt = state.get("sparse")
            _counts = getattr(scheduler_output, "num_scheduled_tokens", None) or {}
            if _rt is not None and _counts:
                # The batch size is the authoritative "are we concurrent" signal: the per-request
                # state dict may still hold a single entry while the first request prefills.
                _rt.batch_size = len(_counts)
                # Requests that carry speculative drafts this step cannot be compacted: vLLM places
                # their logits rows at [query_end - (drafts + bonus), query_end), so rewriting the
                # block table under them trips `assert num_scheduled_tokens >= num_logits`.
                _rt.spec_reqs = set(
                    getattr(scheduler_output, "scheduled_spec_decode_tokens", {}) or {})
                # Diagnostic for the `num_logits` assertion: record the exact per-request numbers
                # whenever a request carries drafts but few scheduled tokens.
                _spec = getattr(scheduler_output, "scheduled_spec_decode_tokens", {}) or {}
                for _r, _d in list(_spec.items()):
                    _n = int(_counts.get(_r, 0) or 0)
                    if _n <= len(_d) and _rt is not None:
                        _append(smoke or probe or "/tmp/kvmem_sparse.jsonl",
                                {"kind": "spec_short", "req": str(_r)[:8], "scheduled": _n,
                                 "drafts": len(_d)})
                if _rt is not None and not os.path.exists(HOT_REQUEST):
                    pass
                elif _rt is not None:
                    _hot_apply(self, _rt, probe or smoke or "/tmp/kvmem_sparse.jsonl")
                for _req, _n in list(_counts.items()):
                    # Publishes how many leading rows of this step may enter the key statistics:
                    # a draft-carrying step writes 1 target + K drafts and only the target is
                    # certain to be accepted (eager path; the compiled path is documented as
                    # eager-only for KVMem retrieval).
                    try:
                        from vllm import kvmem_bridge as _kb
                        _kb.KEY_ROWS_LIMIT = 1 if _req in (_rt.spec_reqs or set()) else 0
                    except Exception:
                        pass
                    try:
                        _rs0 = getattr(self, "req_states", None)
                        if _rs0 is not None and getattr(_rt, "stats_core", False):
                            _i0 = _rs0.req_id_to_index.get(_req)
                            if _i0 is not None:
                                _now = int(_rs0.num_computed_tokens_np[_i0])
                                _prev = (state.get("computed_before") or {}).get(_req)
                                # The previous step's accepted count is known HERE (post-step
                                # counters lag under async scheduling), and the bridge still holds
                                # that step's kept rows - so the fold happens lagged by one step,
                                # exactly like the page accounting.
                                # ONLY a decode step keeps rows in the bridge (n <= 8); a prefill
                                # chunk folds into bucket sums instead. Without this guard the
                                # prefill deltas were folded as "decode rows" and written at
                                # positions past the prompt (measured: take=3232 at first=53580).
                                _was_decode = bool((state.get("was_decode") or {}).get(_req))
                                # Fold only when the PREVIOUS step was a decode step: the step that
                                # follows the last prefill chunk carries that chunk's delta, which
                                # the prefill path already wrote - folding it too overshot the
                                # frontier by exactly one chunk (measured: 32823 vs 27730).
                                # A decode step can accept at most 1 + num_spec tokens, so a delta
                                # larger than that is a prefill chunk that the prefill path already
                                # wrote; folding it made the frontier overshoot by exactly one chunk
                                # (measured: +1868 and +5100 before this bound).
                                _delta = (_now - _prev) if _prev is not None else 0
                                if (0 < _delta <= 8 and int(_n) <= 8 and _was_decode):
                                    _rt.harvest_decode_rows(_req, _delta)
                                state.setdefault("computed_before", {})[_req] = _now
                                state.setdefault("was_decode", {})[_req] = int(_n) <= 8
                    except Exception:
                        pass
                    try:
                        _stb = _rt.state(_req)
                        _rsb = getattr(self, "req_states", None)
                        _cb = _lb = -1
                        if _rsb is not None:
                            _ib = _rsb.req_id_to_index.get(_req)
                            if _ib is not None:
                                _cb = int(_rsb.num_computed_tokens_np[_ib])
                                _lb = int(_rsb.num_logical_tokens_np[_ib])
                        _append(smoke or probe or "/tmp/kvmem_sparse.jsonl",
                                {"kind": "prefill_book", "req": str(_req)[:8], "n": int(_n),
                                 "computed": _cb, "logical": _lb,
                                 "row": len(getattr(_stb, "row", ())),
                                 "seq_len": int(getattr(_stb, "seq_len", 0)),
                                 "tokens": int(getattr(_stb, "tokens", 0))})
                    except Exception as _exc:
                        _append(smoke or probe or "/tmp/kvmem_sparse.jsonl",
                                {"kind": "prefill_book_error", "error": str(_exc)[:160]})
                    try:
                        # Mirror exactly what the engine has written (its own cursor), then make the
                        # attention length match. Both are needed: the row is the KV addressing and
                        # the length is what the kernel is told to read.
                        _bt = getattr(self, "block_tables", None)
                        _bpk = list(getattr(_bt, "blocks_per_kv_block", []) or [])
                        _nb = getattr(_bt, "num_blocks", None)
                        if _nb is not None and _bpk:
                            _idx = None
                            try:
                                _idx = getattr(self, "req_states").req_id_to_index.get(_req)
                            except Exception:
                                _idx = None
                            if _idx is not None:
                                for _g in range(len(_bpk)):
                                    if int(_bpk[_g]) > 1:
                                        _rt.sync_row_from_engine(
                                            _req, int(_nb.np[_g, int(_idx)]), int(_n))
                                        break
                    except Exception:
                        pass
                    _rt.advance_tokens(_req, int(_n))
                    # The mean-K statistics are keyed by TRUE positions while the model may run at
                    # compacted ones; publish the offset before the model executes so the capture op
                    # buckets this step correctly (device tensor -> CUDA graph replays it live).
                    if getattr(_rt, "stats_core", False):
                        _rt.publish_position_offset(_req)
                    # Book-consistency probe for the beyond-pool prefill path: after a compaction the
                    # engine schedules the next chunk from its LOGICAL pointer while its RESIDENT
                    # length is the compacted one, and the row must cover both. Log the three numbers
                    # for the first steps so a divergence is visible instead of showing up as a
                    # FlashInfer fault deep in the kernel.
                    try:
                        _cs = getattr(runtime, "_cmp_logged", None)
                        if _cs is None:
                            runtime._cmp_logged = _cs = [0]
                        if _cs[0] < 24:
                            _st2 = _rt.state(_req)
                            _rs = getattr(self, "req_states", None)
                            _comp = _logi = -1
                            if _rs is not None:
                                try:
                                    _idx = int(_rs.req_id_to_index.get(_req, 0)) \
                                        if hasattr(_rs, "req_id_to_index") else 0
                                    _comp = int(_rs.num_computed_tokens_np[_idx])
                                    _logi = int(_rs.num_logical_tokens_np[_idx])
                                except Exception:
                                    pass
                            _cs[0] += 1
                            _append(smoke or probe or "/tmp/kvmem_sparse.jsonl",
                                    {"kind": "book", "req": str(_req)[:8], "n": int(_n),
                                     "computed": _comp, "logical": _logi,
                                     "rt_tokens": int(getattr(_st2, "tokens", 0)),
                                     "rt_seq_len": int(getattr(_st2, "seq_len", 0)),
                                     "row": len(getattr(_st2, "row", ()))})
                    except Exception:
                        pass
                    # Adapter half of the position split: RoPE keeps the TRUE position (the engine's
                    # own `positions`), the WRITE address uses the compacted one. `kvmem_slot_offset`
                    # is a 0-dim device tensor so a CUDA graph replays the live value; 0 (no
                    # compaction yet) makes the subtraction a no-op.
                    try:
                        _st = _rt.state(_req)
                        # The offset must describe the ENGINE's own books, not the runtime's copy:
                        # after an in-prefill compaction the engine's num_computed can be one chunk
                        # AHEAD of the resident book, which made (tokens - seq_len) negative - and a
                        # negative offset shifted every slot index past the row (illegal memory
                        # access). Use num_logical - num_computed from the request state, and never
                        # publish a negative value.
                        _off = None
                        try:
                            _rso = getattr(self, "req_states", None)
                            _io = _rso.req_id_to_index.get(_req) if _rso is not None else None
                            if _io is not None:
                                _lg = int(_rso.num_logical_tokens_np[_io])
                                _cp = int(_rso.num_computed_tokens_np[_io])
                                _off = _lg - _cp
                        except Exception:
                            _off = None
                        if _off is None:
                            _off = int(getattr(_st, "tokens", 0)) - int(getattr(_st, "seq_len", 0))
                        if _off < 0:
                            _off = 0
                        _buf = getattr(self, "_kvmem_slot_offset", None)
                        if _buf is None:
                            import torch as _t
                            _buf = _t.zeros((), dtype=_t.int64, device="cuda")
                            self._kvmem_slot_offset = _buf
                        _buf.fill_(_off)
                        # One offset, two consumers: RoPE adds it (true position) and the slot
                        # mapping subtracts it (compacted write address). Both are device scalars so
                        # CUDA graphs replay the live value.
                        self.kvmem_rope_offset = _buf
                        self.kvmem_slot_offset = _buf
                    except Exception:
                        self.kvmem_slot_offset = None
        except Exception:
            _append(smoke or probe or "/tmp/kvmem_sparse.jsonl",
                    {"kind": "advance_error", "error": traceback.format_exc()})
        except Exception:
            self._kvmem_req_ids = []
        result = original(self, scheduler_output, *args, **kwargs)
        # Profiling/dummy runs are not real steps.
        if kwargs.get("dummy_run") or kwargs.get("is_profile"):
            return result
        # NOTE: a None result is NOT a skipped step. The V2 runner samples in sample_tokens()
        # under async scheduling, but execute_model has already written this step's KV, so the
        # sparse hooks (and the adapter's safe-boundary work) must still run.
        try:
            if probe and not state["probed"]:
                state["probed"] = True
                _probe_runner(self, probe)
            # Decide from the config that arrived via vLLM's additional_config, not from what was
            # known at plugin-registration time.
            if (_config().get("sparse") and state["sparse"] is None
                    and not state["sparse_failed"]):
                state["sparse"] = _attach_sparse(self, probe or smoke or
                                                 "/tmp/kvmem_sparse.jsonl")
                state["sparse_failed"] = state["sparse"] is None
            if _config().get("smoke_log") and state["adapter_step"] is None \
                    and not state["attach_failed"]:
                state["adapter_step"] = _attach_adapter(self, smoke)
                state["attach_failed"] = state["adapter_step"] is None
            if state["adapter_step"] is not None:
                state["adapter_step"](scheduler_output)
            if state.get("sparse") is not None:
                # Hook 2: commit what this step wrote, and reselect when due.
                runtime = state["sparse"]
                scheduled = getattr(scheduler_output, "num_scheduled_tokens", None) or {}
                trace = state.get("trace", False)
                cap = getattr(runtime, "capture", None)
                state["steps"] = state.get("steps", 0) + 1
                for req, n_new in scheduled.items():
                    _console_step({"runtime": runtime, "req": req, "runner": self,
                                   "state": state, "scheduler_output": scheduler_output,
                                   "step": state.get("steps"), "n_new": int(n_new),
                                   "orch": runtime._orch.get(req),
                                   "index": runtime._retrieval.get(req),
                                   "stats": runtime._statistics.get(req)})
                    if cap is not None:
                        cap.note_tokens(int(n_new))
                    # KVMem mean-K statistics (v0.19 design): the model has just written this
                    # step's K, so the bridge's 64-token bucket sums are complete. Draining them
                    # here (a step boundary) is what keeps the device buffer per step; the
                    # context-sized sum lives in the request's kvmem RawKvStore.
                    if getattr(runtime, "stats_core", False):
                        if state.get("stats_probe", 0) < 6:
                            state["stats_probe"] = state.get("stats_probe", 0) + 1
                            try:
                                _st = runtime.state(req)
                                _append(smoke or probe or "/tmp/kvmem_sparse.jsonl",
                                        {"kind": "harvest_attempt", "pid": os.getpid(),
                                         "step": state["steps"], "n_new": int(n_new),
                                         "tokens": _st.tokens, "seq_len": _st.seq_len,
                                         "prefill_done": bool(getattr(_st, "prefill_done", False)),
                                         "drafts": req in (getattr(runtime, "spec_reqs", None) or ()),
                                         "batch": int(getattr(runtime, "batch_size", 1) or 1)})
                            except Exception:
                                pass
                        try:
                            runtime.harvest_key_statistics(req, int(n_new))
                        except Exception:
                            _append(smoke or probe or "/tmp/kvmem_sparse.jsonl",
                                    {"kind": "stats_harvest_error", "req": req,
                                     "error": traceback.format_exc()[-600:]})
                    # Token accounting now lives in runtime.advance_tokens (driven from this
                    # wrapper's entry). Calling after_step here as well double-counted the
                    # frontier: a 37K request reached tokens=98327.
                    if not getattr(runtime, "_advance_owns_accounting", False):
                        st_before = runtime.state(req)
                        runtime.after_step(req, int(n_new), prefill=st_before.row == ())
                    if state["steps"] <= 5 or state["steps"] % 200 == 0:
                        st_now = runtime.state(req)
                        _append(probe or smoke or "/tmp/kvmem_sparse.jsonl",
                                {"kind": "step_after", "pid": os.getpid(),
                                 "step": state["steps"], "n_new": int(n_new),
                                 "tokens": st_now.tokens, "seq_len": st_now.seq_len,
                                 "row": len(st_now.row)})
                        if getattr(runtime, "stats_core", False):
                            _append(probe or smoke or "/tmp/kvmem_sparse.jsonl",
                                    {"kind": "statistics", "pid": os.getpid(),
                                     "req": str(req)[:8], "n_new": int(n_new),
                                     **runtime.statistics_stats(req)})
                    if trace:
                        # Logging must never break the step.
                        try:
                            st = runtime.state(req)
                            req_states = getattr(self, "req_states", None)
                            computed = (-1 if req_states is None else
                                        int(req_states.num_computed_tokens_np[0]))
                            _append(smoke or probe or "/tmp/kvmem_sparse.jsonl",
                                    {"kind": "trace", "req": req, "n_new": int(n_new),
                                     "tokens": st.tokens, "seq_len": st.seq_len,
                                     "row": len(st.row), "computed": computed})
                        except Exception:
                            pass
        except Exception:
            _append(smoke or probe, {"kind": "hook_error", "error": traceback.format_exc()})
            state["adapter_step"] = None
            state["attach_failed"] = True
        return result

    cls.execute_model = execute_model

    # The V2 runner has no `prepare_pos_seq_lens`, and its execute_model post-block does not run for
    # every prefill chunk (measured: 4 calls for a 37K-token request). `sample_tokens` is the method
    # that DOES run once per step, so the token frontier is driven from there.
    if hasattr(cls, "sample_tokens"):
        _original_sample = cls.sample_tokens

        @functools.wraps(_original_sample)
        def sample_tokens(self, scheduler_output, *args, **kwargs):
            try:
                runtime = state.get("sparse")
                counts = getattr(scheduler_output, "num_scheduled_tokens", None) or {}
                if runtime is not None and counts:
                    for req, n_new in list(counts.items()):
                        runtime.advance_tokens(req, int(n_new))
            except Exception:
                _append(smoke or probe or "/tmp/kvmem_sparse.jsonl",
                        {"kind": "advance_error", "error": traceback.format_exc()})
            return _original_sample(self, scheduler_output, *args, **kwargs)

        # DISABLED: the fork does not call this method per step (measured: no call for a 37K
        # request). Kept as documentation; counting happens at execute_model entry instead.
        _boot(f"sample_tokens hook skipped for {module_name}.{cls.__name__}")

    # The V2 runner has no `prepare_pos_seq_lens` METHOD, so the sparse view would never be
    # installed: the engine would truncate its book (compact) while the block table still pointed
    # at the pages written at TRUE positions - measured as gibberish output. `prepare_inputs` runs
    # once per step (prefill and decode) and is where positions/seq_lens are materialised, so the
    # install (and the row/seq_len/position override) belongs here.
    if not hasattr(cls, "prepare_pos_seq_lens") and hasattr(cls, "prepare_inputs"):
        _orig_prepare_inputs = cls.prepare_inputs
        if not getattr(_orig_prepare_inputs, "_kvmem_v2_patched", False):
            @functools.wraps(_orig_prepare_inputs)
            def prepare_inputs(self, scheduler_output, batch_req_state, batch_desc,
                               *args, **kwargs):
                # KVMem: the sparse view must be installed BEFORE `_orig_prepare_inputs` runs.
                # That method gathers the block table rows into the input buffers the attention
                # kernel reads, so a row written afterwards would only take effect one step later -
                # and that step would run the model with the NEW positions against the OLD row.
                # With one compaction per R output tokens that was a single bad step; with a
                # resident prefill (a compaction every ~8K prompt tokens) it accumulates into
                # corrupted KV (measured: an 80K prompt answered with noise while every needle
                # check inside the runtime still passed).
                try:
                    rt = state.get("sparse")
                    if rt is not None:
                        # Cross-request prefix reuse: a NEW request carries its prompt tokens (the
                        # resume key) and, when EngineCore matched a parked session, a served count
                        # ABOVE its resident book -- that pair is the resume signal. Both values
                        # travel in NewRequestData already, so no extra channel is needed.
                        # V2 runner: `scheduled_new_reqs` (NewRequestData list); the
                        # older/simpler name is kept as a fallback.
                        for _nr in (getattr(scheduler_output, "scheduled_new_reqs", None)
                                    or getattr(scheduler_output, "new_reqs", None) or ()):
                            _rid = getattr(_nr, "req_id", None)
                            if _rid is None:
                                continue
                            try:
                                _ids = getattr(_nr, "prompt_token_ids", None)
                                rt.note_prompt_ids(_rid, _ids)
                                if bool(_config().get("session_reuse", False)):
                                    _append(smoke or probe or "/tmp/kvmem_sparse.jsonl",
                                            {"kind": "note_prompt_ids", "req": str(_rid),
                                             "n": len(_ids or ()), "pid": os.getpid()})
                                _nlg = int(getattr(_nr, "num_logical_tokens", 0) or 0)
                                _npc = int(getattr(_nr, "num_computed_tokens", 0) or 0)
                                if _nlg > _npc:
                                    rt.note_resume(_rid, _nlg, _npc)
                            except Exception:
                                pass
                        reqs = list(getattr(batch_req_state, "req_ids", None) or [])
                        counts = getattr(batch_req_state, "num_scheduled_tokens", None)
                        if reqs and counts is not None:
                            import numpy as _np
                            counts_np = _np.asarray(counts, dtype=_np.int64)
                            csl = _np.concatenate(([0], _np.cumsum(counts_np)))
                            idx = getattr(batch_req_state, "idx_mapping_np", None)
                            rt.view.bind_step(idx, reqs, csl)
                            for gone in [r for r in rt.live_requests() if r not in reqs]:
                                rt.release(gone)
                                _append(smoke or probe or "/tmp/kvmem_sparse.jsonl",
                                        {"kind": "released", "req": gone, "pid": os.getpid()})
                            if _env_flag("KVMEM_PREINSTALL_ROW"):
                                # Opt-in: install the row before the gather. With compaction limited
                                # to decode (the shipped default) the row is stable across a step,
                                # and installing here changes when seq_lens/positions are written,
                                # which breaks the CUDA-graph topology check with N>1.
                                for req in reqs:
                                    row = rt.view.row_index(req)
                                    if row is None:
                                        continue
                                    rt.before_step(req, row_index=row,
                                                   position_slots=rt.view.query_position_slots(req),
                                                   rows_only=True)
                except Exception:
                    _append(smoke or probe or "/tmp/kvmem_sparse.jsonl",
                            {"kind": "sparse_pre_install_error", "error": traceback.format_exc()})
                out = _orig_prepare_inputs(self, scheduler_output, batch_req_state,
                                           batch_desc, *args, **kwargs)
                if os.environ.get("KVMEM_DEBUG_POS", "").strip().lower() in (
                        "1", "true", "yes", "on"):
                    # Diagnostic: dump the per-request position window the engine built for this
                    # step together with the two counters it was derived from. A decode step whose
                    # `positions` do not advance by `n` means the served counter is not tracking the
                    # resident one (symptom: the model re-reads its own recent tokens and loops).
                    try:
                        import numpy as _npd
                        rs = getattr(self, "req_states", None)
                        npc_d = getattr(rs, "num_computed_tokens_np", None)
                        nlg_d = getattr(rs, "num_logical_tokens_np", None)
                        ib = getattr(self, "input_buffers", None)
                        pos = getattr(ib, "positions", None)
                        sl = getattr(ib, "seq_lens", None)
                        idx = getattr(batch_req_state, "idx_mapping_np", None)
                        reqs_d = list(getattr(batch_req_state, "req_ids", None) or [])
                        counts_d = getattr(batch_req_state, "num_scheduled_tokens", None)
                        if idx is not None and pos is not None and reqs_d and counts_d is not None:
                            csl = _npd.concatenate(([0], _npd.cumsum(
                                _npd.asarray(counts_d, dtype=_npd.int64))))
                            for _i, _req in enumerate(reqs_d):
                                _r = int(idx[_i])
                                _p0, _p1 = int(csl[_i]), int(csl[_i + 1])
                                _append(smoke or probe or "/tmp/kvmem_sparse.jsonl", {
                                    "kind": "debug_pos", "pid": os.getpid(), "req": _req,
                                    "row": _r, "n": _p1 - _p0,
                                    "positions": [int(v) for v in
                                                  pos[_p0:_p1][:8].tolist()],
                                    "seq_len_buf": (int(sl[_r]) if sl is not None else None),
                                    "npc": (int(npc_d[_r]) if npc_d is not None and _r < len(npc_d)
                                            else None),
                                    "nlg": (int(nlg_d[_r]) if nlg_d is not None and _r < len(nlg_d)
                                            else None),
                                })
                    except Exception:
                        pass
                try:
                    rt = state.get("sparse")
                    if rt is not None:
                        reqs = list(getattr(batch_req_state, "req_ids", None) or [])
                        counts = getattr(batch_req_state, "num_scheduled_tokens", None)
                        if reqs and counts is not None:
                            npc = getattr(getattr(self, "req_states", None),
                                          "num_computed_tokens_np", None)
                            _pl = getattr(getattr(self, "req_states", None), "prompt_len", None)
                            plen = getattr(_pl, "np", None)
                            nlg = getattr(getattr(self, "req_states", None),
                                          "num_logical_tokens_np", None)
                            for req in reqs:
                                row = rt.view.row_index(req)
                                if row is None:
                                    continue
                                if npc is not None and int(row) < len(npc):
                                    # The engine's own count = accepted tokens only. Without this
                                    # the R cadence counts speculative drafts and fires early.
                                    rt.note_engine_computed(req, int(npc[int(row)]))
                                if nlg is not None and int(row) < len(nlg):
                                    # Served count: drives the logical bookkeeping; the resident
                                    # count (`num_computed`) drives positions and seq_lens.
                                    rt.note_engine_logical(req, int(nlg[int(row)]))
                                if plen is not None and int(row) < len(plen):
                                    # The engine's post-tokenizer prompt length (diagnostics).
                                    rt.note_prompt_len(req, int(plen[int(row)]))
                                # Full install AFTER the engine wrote its (optimistic) seq_lens:
                                # the resident length and the compacted positions win.
                                rt.before_step(req, row_index=row,
                                               position_slots=rt.view.query_position_slots(req))
                except Exception:
                    _append(smoke or probe or "/tmp/kvmem_sparse.jsonl",
                            {"kind": "sparse_step_error", "error": traceback.format_exc()})
                return out
            prepare_inputs._kvmem_v2_patched = True
            cls.prepare_inputs = prepare_inputs

    _boot(f"patched {module_name}.{cls.__name__}.execute_model")


_boot("module imported")
