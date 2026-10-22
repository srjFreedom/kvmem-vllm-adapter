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
        if not heads:
            for _name, _mod in getattr(runner, "model", runner).named_modules():
                if type(_mod).__name__ == "Qwen3NextAttention":
                    heads = int(getattr(_mod, "num_heads", 0) or 0)
                    kv_heads = int(getattr(_mod, "num_kv_heads", kv_heads) or kv_heads)
                    head_dim = int(getattr(_mod, "head_size", head_dim) or head_dim)
                    break
        per_rank = max(1, heads // max(1, tp)) if heads else 0
        layers = int(getattr(hf, "num_hidden_layers", 0) or 64)
        head_dim = int(cfg.get("head_dim", 256))
        ok = bool(per_rank) and _bridge.ensure(layers=layers, heads=per_rank,
                                              head_dim=head_dim,
                                              device=getattr(view, "device", "cuda"),
                                              dtype=getattr(getattr(view, "paged_kv", lambda: [None])()[0],
                                                            "dtype", None) or None)
        _append(log_path, {"kind": "bridge_ensure", "ok": bool(ok), "pid": os.getpid(),
                           "tokens": int(os.environ.get("KVMEM_CAPTURE_TOKENS", 0) or 0),
                           "layers": layers, "heads_per_rank": per_rank,
                           "kv_heads": kv_heads, "head_dim": head_dim, "tp": tp})
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

        def _compact(request_id, pages, page_tokens):
            """Three-book reconciliation (Route A): the worker selected a working set;
            truncate the request's `_all_token_ids` to the tokens of those pages (in the
            compacted order the worker will install) and rewind `num_computed_tokens` to the
            visible length. In-place slice assignment keeps the `ConstantList` view in sync,
            so `num_tokens` drops to the visible length and the scheduler sees no gap — it
            will not recompute the spilled tokens. `_output_token_ids` is untouched, so
            `num_output_tokens` (the API's generated count) stays correct."""
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
            all_tokens = req._all_token_ids
            # The scheduler already built THIS step from the pre-compaction book (accounting in the
            # worker is one step behind), so its scheduled tokens are still pending: they will be
            # added to `num_computed_tokens` when the step completes. Set the frontier minus that
            # pending count, otherwise num_computed catches up to num_tokens and the request is
            # never scheduled again (measured: generation stalls at 0 tok/s, GPU idle).
            pending = max(1, len(all_tokens) - int(req.num_computed_tokens))
            visible = []
            # `pages` arrives in COMPACTED order, so the visible-space token range of the k-th
            # page is [k * page_tokens, (k+1) * page_tokens) - NOT [page * page_tokens). The page
            # id is a LOGICAL page (what the engine wrote before compaction); after the first
            # compaction the engine's `_all_token_ids` is visible-space, so using the id as an
            # offset spliced the wrong tokens and corrupted the context from that point on
            # (measured: a count broke ~50 tokens after the first reselect).
            for rank, page in enumerate(pages):
                start = int(rank) * page_tokens
                if start >= len(all_tokens):
                    continue
                visible.extend(all_tokens[start:min(start + page_tokens, len(all_tokens))])
            if not visible or len(visible) >= len(all_tokens):
                # Idempotency across TP ranks: each rank reselects and sends its own compact, so the
                # SECOND rank always arrives after the first one already truncated the book. That
                # rank must still install its own view (each rank has its own KV tensors and block
                # table); reporting a plain no-op there made one rank drop its view and the two
                # ranks diverged - measured as a long-context stall / noise.
                applied = _compact_state.get(request_id)
                already = applied == len(visible) and len(all_tokens) == len(visible)
                _append(_sched_cfg(scheduler).get("probe_log") or "/tmp/kvmem_sparse.jsonl",
                        {"kind": "compact_noop", "pid": os.getpid(), "req": request_id,
                         "visible": len(visible), "all_tokens": len(all_tokens),
                         "computed": int(req.num_computed_tokens), "pages": len(pages),
                         "peer_applied": bool(already)})
                return {"ok": len(visible) == len(all_tokens),
                        "truncated": bool(already),
                        "visible": len(visible)}
            # vLLM requires `_all_token_ids` to COVER the prompt (num_tokens >= num_prompt_tokens):
            # the sampler/stop checker derive the output length from the difference. Truncating
            # below the prompt leaves num_tokens < prompt_len and the request stops being scheduled
            # (measured: an 8305-token prompt compacted to 8192 stalled with the GPU idle). The
            # entry policy is what must keep the prompt <= B; until it always does, refuse here.
            prompt_len = len(getattr(req, "prompt_token_ids", ()) or ())
            if prompt_len and len(visible) < prompt_len:
                _append(_sched_cfg(scheduler).get("probe_log") or "/tmp/kvmem_sparse.jsonl",
                        {"kind": "compact_refused", "pid": os.getpid(), "req": request_id,
                         "visible": len(visible), "prompt_len": prompt_len,
                         "all_tokens": len(all_tokens)})
                return {"ok": False, "truncated": False, "visible": len(visible)}
            req._all_token_ids[:] = visible
            req.num_computed_tokens = max(0, len(visible) - pending)
            _compact_state[request_id] = len(visible)
            _append(_sched_cfg(scheduler).get("probe_log") or "/tmp/kvmem_sparse.jsonl",
                    {"kind": "compact_applied", "pid": os.getpid(),
                     "req": request_id, "pages": len(pages),
                     "visible_tokens": len(visible),
                     "was_tokens": len(all_tokens),
                     # Book reconciliation proof: the scheduler continues with
                     # num_tokens = after_tokens and num_computed = after_computed; a request with
                     # zero new tokens is never scheduled again (the GPU idles, client hangs).
                     "before_computed": int(before_computed),
                     "pending": int(pending),
                     "after_tokens": len(req._all_token_ids),
                     "after_computed": int(req.num_computed_tokens)})
            return {"ok": True, "truncated": True, "visible": len(visible)}

        _channel_state = {"scheduler": None, "server": None}
        # request_id -> visible length of the last applied compaction (TP-rank idempotency).
        _compact_state = {}
        original = Scheduler.schedule
        if not getattr(original, "_kvmem_channel_patched", False):
            def schedule(self, *args, **kwargs):
                _channel_state["scheduler"] = self
                if _channel_state["server"] is None:
                    scfg = _sched_cfg(self)
                    if scfg.get("block_channel") and scfg.get("inrequest_compaction"):
                        _channel_state["server"] = BlockChannelScheduler(
                            _alloc, path=scfg.get("block_channel_path",
                                                  "/tmp/kvmem_block_channel.sock"),
                            compactor=_compact).start()
                        _append(scfg.get("probe_log") or "/tmp/kvmem_sparse.jsonl",
                                {"kind": "block_channel_attached", "pid": os.getpid()})
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
                for _req, _n in list(_counts.items()):
                    _rt.advance_tokens(_req, int(_n))
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
                    if cap is not None:
                        cap.note_tokens(int(n_new))
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
                out = _orig_prepare_inputs(self, scheduler_output, batch_req_state,
                                           batch_desc, *args, **kwargs)
                try:
                    rt = state.get("sparse")
                    if rt is not None:
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
                            npc = getattr(getattr(self, "req_states", None),
                                          "num_computed_tokens_np", None)
                            _pl = getattr(getattr(self, "req_states", None), "prompt_len", None)
                            plen = getattr(_pl, "np", None)
                            for req in reqs:
                                row = rt.view.row_index(req)
                                if row is None:
                                    continue
                                if npc is not None and int(row) < len(npc):
                                    # The engine's own count = accepted tokens only. Without this
                                    # the R cadence counts speculative drafts and fires early.
                                    rt.note_engine_computed(req, int(npc[int(row)]))
                                if plen is not None and int(row) < len(plen):
                                    # The engine's post-tokenizer prompt length: the runtime refuses
                                    # to compact a request whose prompt is above B (the engine would
                                    # refuse every round and the slot churn corrupted the view).
                                    rt.note_prompt_len(req, int(plen[int(row)]))
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
