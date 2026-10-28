"""Feed KVMem's retrieval index from the live vLLM attention layers.

Where: `Qwen3NextAttention._project_qkv_gate(qkv, positions)` returns the *post-RoPE* q/k and is the
only place where both the keys and the positions are in hand. We wrap it, rotate back to the
pre-RoPE frame (`kvmem_pre_rope_capture.unrotate`) and accumulate per-block statistics.

Why not intercept the graph like kvmem's llama adapter: vLLM fuses qk-norm+RoPE into a Triton
kernel, so pre-RoPE tensors never materialise. Inverse rotation is exact (orthogonal per position)
and keeps the fork untouched — the same numbers, obtained one layer higher.

Cost control: work happens only when `capture_needed()` says a reselect is near, and only the
q/k of the current step are touched (never the whole cache).
"""
from __future__ import annotations

import traceback

import torch

from kvmem_pre_rope_capture import capture_needed, rotary_inv_freq, unrotate


class EngineCapture:
    def __init__(self, index_getter, log=None, gen_reserve=16384, layers=None):
        self._index_getter = index_getter     # () -> KvmemRetrievalIndex | None
        self._log = log
        self.gen_reserve = int(gen_reserve)
        self.layers = layers
        self._inv_freq = {}
        self._installed = []
        self._tokens_since_reselect = 0
        self.steps = 0
        self.captured = 0
        self.errors = 0
        self.calls = 0          # times the wrapped attention path ran
        self.wants = 0          # times it decided a capture was due

    # --- control ----------------------------------------------------------------------
    def note_tokens(self, n):
        self._tokens_since_reselect += int(n)

    def reset_tokens(self):
        self._tokens_since_reselect = 0

    def wants_capture(self):
        index = self._index_getter()
        if index is None:
            return False
        return capture_needed(self._tokens_since_reselect, self.gen_reserve)

    # --- the hook ---------------------------------------------------------------------
    def harvest(self, layer, q, k, positions):
        self.calls += 1
        if not self.wants_capture():
            return
        self.wants += 1
        index = self._index_getter()
        if index is None:
            return
        try:
            rotary = getattr(layer, "rotary_emb", None)
            rotary_dim = int(getattr(rotary, "rotary_dim", 0) or 0)
            base = float(getattr(rotary, "base", 1e7) or 1e7)
            if rotary_dim <= 0:
                return
            # The frequencies must live on the same device as q/k (CUDA), not on the CPU.
            key = (rotary_dim, base, str(q.device))
            if key not in self._inv_freq:
                self._inv_freq[key] = rotary_inv_freq(rotary_dim, base, device=q.device)
            inv = self._inv_freq[key]

            lidx = int(getattr(layer, "layer_idx", 0) or 0)
            pos = positions if positions.ndim == 1 else positions.reshape(-1)

            q_pre = unrotate(q.reshape(-1, q.shape[-2], q.shape[-1]), pos, inv, rotary_dim)
            k_pre = unrotate(k.reshape(-1, k.shape[-2], k.shape[-1]), pos, inv, rotary_dim)
            index.set_query(lidx, q_pre)
            for page, chunk in _group_by_page(pos, k_pre, index.page_tokens):
                index.add(page, lidx, chunk)
            self.captured += 1
        except Exception:
            # Rate-limit: a broken capture must not flood the log at every layer of every step.
            self.errors += 1
            if self._log is not None and (self.errors == 1 or self.errors % 5000 == 0):
                self._log({"kind": "capture_error", "count": self.errors,
                           "error": traceback.format_exc()})

    # --- install ----------------------------------------------------------------------
    def install(self):
        try:
            from vllm.model_executor.models import qwen3_next
        except Exception:
            return 0
        target = qwen3_next.Qwen3NextAttention
        # DISABLED: this wrapper replaced `_project_qkv_gate` wholesale and, in doing so, hid the
        # fork's own `torch.ops.kvmem.capture_q(...)` call from the compiled graph (measured:
        # inspect.getsource showed the wrapper, and the op body ran zero times). The fork bridge is
        # now the single capture path, so installing this wrapper would break it again.
        return 0
        original = target._project_qkv_gate
        if getattr(original, "_kvmem_capture_patched", False):
            return 1
        capture = self

        def _project_qkv_gate(self, qkv, positions):
            q, k, v, gate = original(self, qkv, positions)
            capture.harvest(self, q, k, positions)
            return q, k, v, gate

        _project_qkv_gate._kvmem_capture_patched = True
        target._project_qkv_gate = _project_qkv_gate
        self._installed.append((target, original))
        return 1

    def uninstall(self):
        for target, original in self._installed:
            target._project_qkv_gate = original
        self._installed.clear()


def _group_by_page(positions, k_pre, page_tokens):
    """Yield (page, keys) for the token rows of one step, grouping them by logical page."""
    pages = (positions // int(page_tokens)).tolist()
    if not pages:
        return
    start = 0
    for i in range(1, len(pages) + 1):
        if i == len(pages) or pages[i] != pages[start]:
            yield pages[start], k_pre[start:i]
            start = i


# --------------------------------------------------------------------------------------------
# Route B: hook the CUSTOM OP, not the Python module.
#
# Under `torch.compile` + CUDA Graph the model's Python methods are never executed, but a custom
# op (`torch.ops.vllm.unified_attention_with_output`) is an opaque node that dispatches at run time.
# Overriding its CUDA implementation is a documented torch.library extension, so we get the real
# q/k of every step with ZERO compile/eager cost - no AOT cache to drop, no graph to disable.
# --------------------------------------------------------------------------------------------

import re as _re

_LAYER_RX = _re.compile(r"layers\.(\d+)\.")


def layer_index_from_name(layer_name):
    match = _LAYER_RX.search(str(layer_name))
    return int(match.group(1)) if match else None


def install_op_override(capture, log=None):
    """Wrap `vllm::unified_attention_with_output` so every step's q/k reaches `capture`."""
    try:
        import torch
        import vllm.model_executor.layers.attention.attention as attn_mod
    except Exception:
        return False
    original = attn_mod.unified_attention_with_output

    def wrapped(query, key, value, output, layer_name,
                output_scale=None, output_block_scale=None, kv_cache_dummy_dep=None):
        try:
            capture.on_op(layer_name, query, key)
        except Exception:
            capture.errors += 1
            if log is not None and (capture.errors == 1 or capture.errors % 2000 == 0):
                import traceback
                log({"kind": "capture_op_error", "count": capture.errors,
                     "error": traceback.format_exc()})
        return original(query, key, value, output, layer_name,
                        output_scale, output_block_scale, kv_cache_dummy_dep)

    try:
        lib = torch.library.Library("vllm", "IMPL")
        lib.impl("unified_attention_with_output", wrapped, "CUDA")
    except Exception:
        if log is not None:
            import traceback
            log({"kind": "capture_op_install_error", "error": traceback.format_exc()})
        return False
    capture.op_installed = True
    return True


def _capture_on_op(self, layer_name, query, key):
    """Accumulate one step's q/k for the layer the op belongs to."""
    self.calls += 1
    positions = getattr(self, "_positions", None)
    if positions is None:
        return
    if not self.wants_capture():
        return
    self.wants += 1
    index = self._index_getter()
    if index is None:
        return
    layer = layer_index_from_name(layer_name)
    if layer is None:
        return
    import vllm.model_executor.layers.attention.attention as attn_mod
    _md, module, _kv, _ = attn_mod.get_attention_context(layer_name)
    heads = int(getattr(module, "num_heads", 0) or 0)
    kv_heads = int(getattr(module, "num_kv_heads", 0) or 0)
    head_size = int(getattr(module, "head_size", 0) or 0)
    if not (heads and kv_heads and head_size):
        return
    rotary = getattr(module, "rotary_dim", None)  # not on Attention; fall back to the cache below
    rotary_dim = int(rotary or getattr(self, "rotary_dim", 64))
    base = float(getattr(self, "rope_theta", 1e7))
    key_tuple = (rotary_dim, base, str(query.device))
    if key_tuple not in self._inv_freq:
        from kvmem_pre_rope_capture import rotary_inv_freq
        self._inv_freq[key_tuple] = rotary_inv_freq(rotary_dim, base, device=query.device)
    inv = self._inv_freq[key_tuple]

    from kvmem_pre_rope_capture import unrotate
    pos = positions if positions.ndim == 1 else positions.reshape(-1)
    q_all = query.reshape(-1, heads, head_size)
    k_all = key.reshape(-1, kv_heads, head_size)
    n = min(q_all.shape[0], pos.shape[0])
    index.begin_query(layer)
    for start in range(0, n, 512):          # bounded temporaries: never a whole 4K chunk at once
        stop = min(start + 512, n)
        q_pre = unrotate(q_all[start:stop], pos[start:stop], inv, rotary_dim)
        k_pre = unrotate(k_all[start:stop], pos[start:stop], inv, rotary_dim)
        index.add_query(layer, q_pre)
        for page, chunk in _group_by_page(pos[start:stop], k_pre, index.page_tokens):
            index.add(page, layer, chunk)
    self.captured += 1
    return
    index.begin_query(layer)
    index.add_query(layer, q_pre)
    for page, chunk in _group_by_page(pos[:n], k_pre, index.page_tokens):
        index.add(page, layer, chunk)
    self.captured += 1


EngineCapture.on_op = _capture_on_op


def _all_finite(t) -> bool:
    """True when every element of `t` is finite. Diagnostic helper: never raises."""
    try:
        import torch as _t
        return bool(_t.isfinite(t).all())
    except Exception:
        return True


def harvest_from_bridge(capture):
    """Read the fork's query bridge and turn it into per-layer query summaries.

    Runs at reselect time on the host: the bridge already holds the newest query because its copies
    are part of the compiled graph. No per-step Python is involved, so the original (compiled)
    vLLM path stays untouched.
    """
    try:
        from vllm import kvmem_bridge as bridge
    except Exception:
        return 0
    if not getattr(bridge, "ARMED", False):
        return 0
    index = capture._index_getter()
    if index is None:
        # Diagnostic: "fed 0" was indistinguishable from "no index" and from "every layer empty",
        # which hid the fact that the harvest never even started.
        if getattr(capture, "_log", None) is not None:
            capture._log({"kind": "bridge_harvest_no_index",
                          "armed": bool(getattr(bridge, "ARMED", False)),
                          "layers": len(getattr(bridge, "_query", {}) or {})})
        return 0
    from kvmem_pre_rope_capture import rotary_inv_freq, unrotate
    fed = 0
    rows = 0
    dbg = None
    n_layers = int(getattr(index, "layers", 1 << 30) or (1 << 30))
    for layer, (q, pos) in bridge.read_all().items():
        # Every boundary below is guarded on its own: one bad layer must never take down the whole
        # harvest (an out-of-range layer, a shape mismatch or an exception used to abort the loop and
        # leave the request with NO query at all).
        try:
            if q is None or pos is None or q.numel() == 0:
                continue
            layer = int(layer)
            if layer < 0 or layer >= n_layers:
                if getattr(capture, "_log", None) is not None:
                    capture._log({"kind": "bridge_harvest_bad_layer", "layer": layer,
                                  "index_layers": n_layers})
                continue
            rows += int(q.shape[0])
            rotary_dim = int(getattr(capture, "rotary_dim", 64))
            base = float(getattr(capture, "rope_theta", 1e7))
            key = (rotary_dim, base, str(q.device))
            if key not in capture._inv_freq:
                capture._inv_freq[key] = rotary_inv_freq(rotary_dim, base, device=q.device)
            # The bridge inverts RoPE inside its capture op, so its rows are already the pre-RoPE
            # frame PageQueryView means. Rotating them again would score against a turned query
            # (the exact failure this migration removes), so it is skipped when the bridge says so.
            if bool(getattr(bridge, "PRE_ROPE", False)):
                q_pre = q
            else:
                try:
                    q_pre = unrotate(q, pos[:q.shape[0]], capture._inv_freq[key], rotary_dim)
                except Exception:
                # Fall back to the captured query unchanged: the rotation frame is then approximate,
                # but a slightly-wrong direction still ranks far better than no ranking at all.
                    q_pre = q
                    if getattr(capture, "_log", None) is not None:
                        capture._log({"kind": "bridge_harvest_unrotate_fallback", "layer": layer})
            if dbg is None:
                # Recorded BEFORE the non-finite check: a broken payload is exactly the evidence we
                # need, and it must not be swallowed by the filter.
                try:
                    dbg = {"layer": layer, "rows": int(q.shape[0]),
                           "q_dtype": str(q.dtype), "q_dev": str(q.device),
                           "q_absmax": float(q.abs().max()),
                           "q_finite": bool(_all_finite(q)),
                           "pos_absmax": float(pos[:q.shape[0]].abs().max()),
                           "qpre_absmax": float(q_pre.abs().max()),
                           "rotary_dim": int(rotary_dim)}
                except Exception:
                    dbg = None
            if not _all_finite(q_pre):
                # One non-finite layer would make the layer-mean NaN and flatten the whole ranking,
                # so it is dropped here instead of being averaged in.
                if getattr(capture, "_log", None) is not None:
                    capture._log({"kind": "bridge_harvest_skip_nonfinite", "layer": layer})
                continue
            index.begin_query(layer)
            index.add_query(layer, q_pre)
            fed += 1
        except Exception:
            if getattr(capture, "_log", None) is not None:
                import traceback as _tb
                capture._log({"kind": "bridge_harvest_layer_error", "layer": int(layer),
                              "error": _tb.format_exc()[-300:]})
            continue
    if dbg is not None and getattr(capture, "_log", None) is not None:
        rec = {"kind": "bridge_harvest_dbg"}
        rec.update(dbg)
        try:
            rec["cap"] = bridge.capture_diag()
        except Exception:
            pass
        capture._log(rec)
    if fed:
        capture.captured += 1
    elif getattr(capture, "_log", None) is not None:
        # Nothing usable: record how many layers reported data, so "empty buffers" and "no index"
        # stay distinguishable - AND the capture instrumentation, which is the only thing that can
        # say WHEN the payload became non-finite (an empty harvest used to carry no evidence at all).
        rec = {"kind": "bridge_harvest_empty",
               "layers_with_rows": int(rows),
               "layers_read": len(getattr(bridge, "_query", {}) or {})}
        try:
            rec["cap"] = bridge.capture_diag()
        except Exception:
            pass
        capture._log(rec)
    return fed
