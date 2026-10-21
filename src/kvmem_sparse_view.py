# The sparse-view contract (KVMem "A mode" on vLLM), as a pure function.
#
# Compaction puts a request's selected pages at the front of its block-table row:
#   row[i] = selected_pages[i]           (i < |sel|)      keys already written, re-RoPE'd
#   row[|sel| + j] = new_pages[j]        (j < ceil(n_new / page_tokens))
# and the logical sequence length becomes |sel| * page_tokens + n_new.
#
# The query positions are overridden to their COMPACTED positions, so the engine writes the new
# tokens' K already rotated at the compacted position and every new query sits after the selected
# keys. Only the previously-written selected keys need an explicit re-RoPE; nothing else does.
#
# This module owns only the arithmetic (row, seq_len, positions). The caller stages the row into
# the block table, writes seq_lens/positions, and re-RoPEs the selected pages.
from dataclasses import dataclass


@dataclass(frozen=True)
class SparseView:
    row_pages: tuple[int, ...]      # block-table row, compacted front-to-back
    seq_len: int                    # tokens the attention kernel must read
    query_positions: tuple[int, ...]  # compacted positions of the NOT-yet-written query tokens


def build_sparse_view(selected_pages, new_pages, page_tokens: int, n_new_tokens: int) -> SparseView:
    """Compute the compacted row, seq_len and query positions.

    selected_pages: ascending original page ids whose keys are already written (the working set).
    new_pages:      page ids that hold the not-yet-written query tokens, in write order.
    """
    selected = tuple(int(p) for p in selected_pages)
    if len(set(selected)) != len(selected):
        raise ValueError("selected pages must be unique")
    if list(selected) != sorted(selected):
        raise ValueError("selected pages must be ascending (sink..recent order)")
    if page_tokens <= 0:
        raise ValueError("page_tokens must be positive")
    if n_new_tokens < 0:
        raise ValueError("n_new_tokens must be non-negative")
    pages_needed = (n_new_tokens + page_tokens - 1) // page_tokens
    if pages_needed > len(new_pages):
        raise ValueError("not enough new pages for the query tokens")

    rows = selected + tuple(int(p) for p in new_pages[:pages_needed])
    if len(set(rows)) != len(rows):
        raise ValueError("a page cannot be both a selected key page and a new query page")

    base = len(selected) * page_tokens
    positions = tuple(base + i for i in range(n_new_tokens))
    seq_len = base + n_new_tokens
    if positions and positions[-1] >= seq_len:
        raise ValueError("query positions must stay below the sequence length")
    return SparseView(row_pages=rows, seq_len=seq_len, query_positions=positions)


def pages_to_re_rope(selected_pages, page_tokens: int):
    """(src_page_slot, dst_page_slot) pairs: the selected page's new slot is its selection rank.

    A page's compacted slot is its index in the ascending selection, so the rotation delta for
    every key in that page is (rank - original_page) * page_tokens.
    """
    return [(int(page), rank) for rank, page in enumerate(selected_pages)]


def rotation_deltas(selected_pages, page_tokens: int):
    """Per selected page: the token delta that must be applied to its stored K."""
    return [(int(page), (rank - int(page)) * page_tokens)
            for rank, page in enumerate(selected_pages)]
