# Verifies the sparse-view contract: the compacted row, the kernel's seq_len, the query positions
# and the per-page rotation deltas must all agree with the intended layout.
import sys

sys.path.insert(0, "src")
from kvmem_sparse_view import build_sparse_view, pages_to_re_rope, rotation_deltas

PAGE = 16
failed = False


def check(cond, msg):
    global failed
    print(("ok   " if cond else "FAIL ") + msg)
    failed |= not cond


def main():
    # A 200K conversation: 12,800 pages, of which B=32768/16 = 2048 are selected:
    # the sink prefix, a scattered middle and the recent tail.
    total_pages = 200 * 1024 // PAGE
    sink = list(range(0, 8192 // PAGE))                      # 512 pages
    middle = list(range(2000, 2000 + 1024))                  # 1024 scattered pages
    recent = list(range(total_pages - 512, total_pages))      # 512 pages
    selected = sorted(sink + middle + recent)
    check(len(selected) == 2048, f"working set is B/16 = 2048 pages (got {len(selected)})")

    view = build_sparse_view(selected, [total_pages, total_pages + 1], PAGE, n_new_tokens=20)
    check(view.row_pages[:len(selected)] == tuple(selected), "the selected pages fill the row front")
    check(view.row_pages[len(selected):] == (total_pages, total_pages + 1),
          "the query pages follow the selected pages")
    check(len(view.row_pages) == len(set(view.row_pages)), "no page appears twice")
    check(view.seq_len == len(selected) * PAGE + 20,
          f"seq_len = |sel|*16 + new (got {view.seq_len})")
    check(view.query_positions == tuple(range(32768, 32788)),
          "query positions continue right after the selected keys")
    check(view.query_positions[-1] < view.seq_len, "all query positions are below seq_len")

    deltas = rotation_deltas(selected, PAGE)
    check(deltas[0] == (0, 0), "the first selected page does not move")
    # Page 2000 becomes slot 512 -> delta = (512 - 2000) * 16 tokens.
    rank_of_2000 = selected.index(2000)
    check(deltas[rank_of_2000] == (2000, (rank_of_2000 - 2000) * PAGE),
          "a scattered middle page gets a negative delta matching its new rank")
    checked_moves = sum(1 for _, d in deltas if d != 0)
    check(checked_moves > 0, f"most pages move ({checked_moves} of {len(deltas)})")

    pairs = pages_to_re_rope(selected, PAGE)
    check(pairs[5] == (selected[5], 5), "the re-RoPE page map targets the compacted slot")

    # A decode step right after a reselect has no new pages yet.
    view0 = build_sparse_view(selected, [], PAGE, n_new_tokens=0)
    check(view0.seq_len == len(selected) * PAGE and view0.query_positions == (),
          "a step with no new tokens still has the compacted seq_len")

    # Guard rails.
    for bad, why in (((1, 1), "duplicate selected page"),
                     ((5, 3), "descending selected pages")):
        try:
            build_sparse_view(bad, [9], PAGE, 1)
            check(False, why + " must be rejected")
        except ValueError:
            check(True, why + " is rejected")
    try:
        build_sparse_view([1, 2], [9], PAGE, PAGE * 2)
        check(False, "too few new pages must be rejected")
    except ValueError:
        check(True, "too few new pages is rejected")
    try:
        build_sparse_view([1, 2], [2], PAGE, 1)
        check(False, "a page reused as key and query must be rejected")
    except ValueError:
        check(True, "a page reused as key and query is rejected")

    print("PASS sparse_view_test" if not failed else "FAILED sparse_view_test")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
