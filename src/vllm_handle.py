# The real vLLM VllmKvHandle: wraps one rank's paged-KV shard (the engine's kv_caches
# tensors + block table) and implements the C++ VllmKvHandle contract through the pybind11
# trampoline. D2H/H2D are torch copies into/out of the C++ host-record buffer (passed as an
# address); the block table maps (req, block_index) -> physical_block_id, which indexes the
# per-layer KV tensor.
import atexit
import ctypes
import weakref

import numpy as np
import torch

import vllm_kvmem_ext as ext


class VllmKvHandleVllm:
    """One TP rank's view of the paged KV cache, expressed as the kvmem handle contract.

    `kv_caches` is the engine's list of per-layer KV tensors. For a full-attention layer the
    tensor is the paged KV (indexed by physical block id). The block table (a list of
    StagedWriteTensor, one per KV cache group) maps (req, block_index) -> physical_block_id.
    """

    def __init__(self, kv_caches, block_tables, geometry, session_id, incarnation,
                 catalog, resident, frontier, execution_history, history_revision,
                 pages_per_block=1):
        self._kv_caches = kv_caches          # list[torch.Tensor], one per full-attn layer
        self._block_tables = block_tables    # list[StagedWriteTensor] per group
        self._geom = geometry                # ext.Geometry
        self._catalog = list(catalog)        # list[ext.CatalogBlock]
        self._resident = list(resident)      # list[int] physical block ids
        self._frontier = frontier
        self._execution_history = execution_history
        self._history_revision = history_revision
        # mamba_cache_mode=align pads a logical block to several 16-token pages: the KV tensor's
        # first dim is PAGES, so a logical block is kv[block*pages_per_block:(block+1)*ppb].
        self._pages_per_block = max(1, int(pages_per_block))
        self._scratch = None                 # pinned staging for the async arm
        self._stream = None                  # torch-owned transfer stream (kept alive here)
        self._streams = {}                   # cudaStream_t -> torch.cuda.ExternalStream
        self._stamp = ext.MemoryStamp()
        self._stamp.valid = True
        si = ext.SessionIdentity()
        si.id = session_id
        si.incarnation = incarnation
        self._stamp.session = si
        self._stamp.history_revision = history_revision
        self._stamp.view_version = 0
        self._stamp.frontier = frontier
        self._stamp.execution_history = execution_history

    # --- geometry / catalog / resident / stamp (contract read side) ---
    def geometry(self):
        return self._geom

    def catalog(self):
        return self._catalog

    def resident(self):
        return self._resident

    def stamp(self):
        return self._stamp

    # --- D2H: copy this rank's KV payload for a physical block into the C++ record buffer ---
    # The record buffer holds ALL layers concatenated; each layer writes to its own offset.
    def spill_block(self, block_id, dst_addr, nbytes):
        per_layer = self._per_layer_bytes()
        for layer, kv in enumerate(self._kv_caches):
            off = layer * per_layer
            self._copy_block_to_host(kv, block_id, dst_addr + off, per_layer)

    # --- H2D: copy from the C++ record buffer into this rank's KV payload for a block ---
    def restore_block(self, block_id, src_addr, nbytes):
        per_layer = self._per_layer_bytes()
        for layer, kv in enumerate(self._kv_caches):
            off = layer * per_layer
            self._copy_block_from_host(kv, block_id, src_addr + off, per_layer)

    def _per_layer_bytes(self):
        g = self._geom
        return g.block_tokens * g.num_kv_heads * g.head_dim * g.dtype_bytes * 2

    # --- async extension (the KVMEM_ASYNC_TRANSFER performance arm) ---
    # The pipeline stages each block through a PINNED buffer so the copy is a genuine async
    # D2H/H2D that overlaps the compute stream. The pinned tensor is owned here and kept alive
    # for the handle's lifetime; the C++ side only sees its address.
    def pinned_scratch(self, nbytes):
        if self._scratch is None or self._scratch.numel() < nbytes:
            self._scratch = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
            self._register_scratch_cleanup()
        return self._scratch.data_ptr()

    def transfer_stream(self):
        # Torch owns the stream, so C++ never destroys one torch still references.
        if self._stream is None:
            self._stream = torch.cuda.Stream(priority=-1)
        return self._stream.cuda_stream

    def _register_scratch_cleanup(self):
        # The pinned tensor must be released while the CUDA context is still alive: freeing it
        # during interpreter teardown throws "context is destroyed" from the pinned allocator.
        ref = weakref.ref(self)

        def cleanup():
            handle = ref()
            if handle is None or handle._scratch is None:
                return
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            handle._scratch = None

        self._cleanup = cleanup  # keep the callback referenced
        atexit.register(cleanup)

    def _torch_stream(self, stream):
        # Our own stream is the normal case; anything else is wrapped (and cached) defensively.
        if self._stream is not None and stream == self._stream.cuda_stream:
            return self._stream
        s = self._streams.get(stream)
        if s is None:
            s = torch.cuda.ExternalStream(stream)
            self._streams[stream] = s
        return s

    def _scratch_window(self, addr, per_layer):
        """The pinned staging window [addr, addr + layers*per_layer) inside our scratch."""
        scratch = self._scratch
        total = per_layer * len(self._kv_caches)
        if scratch is None:
            return None, None
        base = addr - scratch.data_ptr()
        if base < 0 or base + total > scratch.numel():
            return None, None
        return scratch, base

    def spill_block_async(self, block_id, dst_addr, stream):
        per_layer = self._per_layer_bytes()
        scratch, base = self._scratch_window(dst_addr, per_layer)
        if scratch is None:
            self.spill_block(block_id, dst_addr, per_layer * len(self._kv_caches))
            return
        s = self._torch_stream(stream)
        with torch.cuda.stream(s):
            for layer, kv in enumerate(self._kv_caches):
                off = base + layer * per_layer
                scratch[off:off + per_layer].copy_(
                    self._block_slice(kv, block_id).reshape(-1), non_blocking=True)

    def restore_block_async(self, block_id, src_addr, stream):
        per_layer = self._per_layer_bytes()
        scratch, base = self._scratch_window(src_addr, per_layer)
        if scratch is None:
            self.restore_block(block_id, src_addr, per_layer * len(self._kv_caches))
            return
        s = self._torch_stream(stream)
        with torch.cuda.stream(s):
            for layer, kv in enumerate(self._kv_caches):
                off = base + layer * per_layer
                target = self._block_slice(kv, block_id)
                src = scratch[off:off + per_layer].view(target.shape)
                target.copy_(src, non_blocking=True)

    # --- publish: expose the prepared view (update resident set + stamp) ---
    def publish_view(self, view):
        self._resident = [vb.block.id for vb in view.blocks]
        self._stamp = view.stamp

    # --- the per-layer copy helpers (the paged KV tensor is 4D (B, H, N, C) in bytes) ---
    def _block_slice(self, kv, block_id):
        a = block_id * self._pages_per_block
        return kv[a:a + self._pages_per_block]

    def _copy_block_to_host(self, kv, block_id, dst_addr, nbytes):
        # The block's K+V payload; copy it into the C++ record at dst_addr.
        block = self._block_slice(kv, block_id).contiguous()
        block_bytes = block.cpu().numpy().tobytes()
        dst_np = np.frombuffer((ctypes.c_uint8 * nbytes).from_address(dst_addr), dtype=np.uint8)
        dst_np[: len(block_bytes)] = np.frombuffer(block_bytes, dtype=np.uint8)

    def _copy_block_from_host(self, kv, block_id, src_addr, nbytes):
        # Copy the C++ record's bytes (at src_addr) into the GPU block.
        src_np = np.frombuffer((ctypes.c_uint8 * nbytes).from_address(src_addr), dtype=np.uint8)
        host = torch.frombuffer(src_np.tobytes(), dtype=torch.uint8)
        target = self._block_slice(kv, block_id)
        target.copy_(host.to(kv.device).view(target.shape))
