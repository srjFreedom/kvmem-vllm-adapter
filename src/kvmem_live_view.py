# LiveRunnerView: adapts the vLLM V2 model runner to the sparse runtime's RunnerView protocol.
#
# Everything the sparse runtime touches lives on the runner:
#   paged_kv           -> the FullAttentionSpec KV tensors of this rank (NOT the Mamba state)
#   block_table_rows   -> runner.block_tables.block_tables[i].gpu  (rows x pages, int32)
#   seq_lens           -> runner.input_buffers.seq_lens
#   positions          -> runner.input_buffers.positions
#
# The per-step binding (which request sits in which batch row, and where its query tokens start)
# is supplied by bind_step(), because the runner rebuilds those every step.
#
# Group ids matter: the block-table list is indexed by KV-cache group, and the Mamba state groups
# have their own numbering, so only the full-attention group may be written.
import torch


class LiveRunnerView:
    def __init__(self, runner, attn_groups, paged_tensors, page_tokens):
        self.runner = runner
        self.attn_groups = sorted(attn_groups)
        self.paged = list(paged_tensors)
        self.page_tokens = int(page_tokens)
        self._groups = None
        self._idx_mapping = None
        self._req_ids = None
        self._query_start_loc = None

    # --- binding -----------------------------------------------------------------------
    def bind_step(self, idx_mapping, req_ids, query_start_loc):
        """Called once per engine step, before the sparse runtime runs."""
        self._idx_mapping = idx_mapping
        self._req_ids = list(req_ids)
        self._query_start_loc = query_start_loc

    def row_index(self, req):
        """The batch row of request `req` this step, or None when it is not scheduled."""
        if self._req_ids is None or req not in self._req_ids:
            return None
        position = self._req_ids.index(req)
        if self._idx_mapping is not None:
            return int(self._idx_mapping[position])
        return position

    def query_position_slots(self, req):
        """positions[] index range for the request's query tokens this step."""
        if self._query_start_loc is None or self._req_ids is None:
            return []
        position = self._req_ids.index(req)
        start = int(self._query_start_loc[position])
        end = int(self._query_start_loc[position + 1])
        return list(range(start, end))

    # --- RunnerView --------------------------------------------------------------------
    def paged_kv(self):
        """EVERY paged full-attention KV tensor: the block table is shared across layers, so a
        logical page spans all of them and must be moved/rotated together."""
        if not self.paged:
            raise RuntimeError("no paged full-attention KV tensor for this rank")
        return list(self.paged)

    def block_table_rows(self):
        if self._groups is None:
            tables = self.runner.block_tables.block_tables
            self._groups = [tables[g].gpu for g in self.attn_groups]
        return self._groups

    def seq_lens(self):
        return self.runner.input_buffers.seq_lens

    def positions(self):
        return self.runner.input_buffers.positions

    def num_blocks(self):
        return int(self.paged[0].shape[0])


def resolve_live_view(runner, probe):
    """Build a LiveRunnerView from a V2 runner, or None when this rank has no paged KV.

    `probe` must expose the same helpers the plugin used for the geometry (the paged attention
    tensors and their cache group ids).
    """
    paged, attn_groups, _config = probe(runner)
    if not paged:
        return None
    first = paged[0][1]
    page_tokens = int(first.shape[2])
    return LiveRunnerView(runner, attn_groups, [t for _, t, _ in paged], page_tokens)
