# vLLM's SCHEDULER allocates in ALIGNED blocks (block_size = 1,616 tokens on this model =
# 101 physical pages), while the worker addresses 16-token pages. Any allocation-side work
# (freeing a page back to the pool) therefore has to be expressed in whole aligned blocks:
# you can hand back a block only when EVERY page of it is outside the working set.
from dataclasses import dataclass


@dataclass(frozen=True)
class AlignedBlockPlan:
    freeable: tuple[int, ...]     # aligned block ids fully outside the working set
    kept: tuple[int, ...]         # aligned block ids with at least one selected page
    partial: tuple[int, ...]      # kept ids that are only partially selected (cannot free)
    tokens_per_block: int


def plan_aligned_blocks(selected_pages, first_page: int, pages_per_block: int,
                        page_tokens: int) -> AlignedBlockPlan:
    """Split the pool's aligned blocks into freeable / kept.

    selected_pages: logical page ids (ascending) that must stay resident.
    first_page: the first usable page id (the NULL block occupies `pages_per_block` pages).
    """
    if pages_per_block <= 0:
        raise ValueError("pages_per_block must be positive")
    tokens_per_block = pages_per_block * page_tokens
    selected = set(int(p) for p in selected_pages)
    if not selected:
        return AlignedBlockPlan((), (), (), tokens_per_block)

    blocks = sorted({(p - first_page) // pages_per_block for p in selected})
    kept, partial = [], []
    for block in blocks:
        base = first_page + block * pages_per_block
        span = set(range(base, base + pages_per_block))
        kept.append(block)
        if not span.issubset(selected):
            partial.append(block)
    # Blocks that exist in the pool but have no selected page at all are freeable.
    highest = max(blocks)
    freeable = [b for b in range(0, highest + 1) if b not in blocks]
    return AlignedBlockPlan(tuple(freeable), tuple(kept), tuple(partial), tokens_per_block)


def block_of_page(page: int, first_page: int, pages_per_block: int) -> int:
    return (int(page) - first_page) // pages_per_block


def block_pages(block: int, first_page: int, pages_per_block: int) -> range:
    base = first_page + int(block) * pages_per_block
    return range(base, base + pages_per_block)
