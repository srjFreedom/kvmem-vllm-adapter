# Validates LiveRunnerView against a fake V2 runner: group selection, batch-row mapping via
# idx_mapping, and the query position slots via query_start_loc.
import sys

import torch

sys.path.insert(0, "src")
from kvmem_live_view import LiveRunnerView

PAGE = 16
PAGES = 64
failed = False


def check(cond, msg):
    global failed
    print(("ok   " if cond else "FAIL ") + msg)
    failed |= not cond


class FakeTable:
    def __init__(self, rows, pages):
        self.gpu = torch.zeros(rows, pages, dtype=torch.int32)


class FakeBlockTables:
    def __init__(self, rows, pages, groups):
        # one entry per KV-cache group; only the attention group is used
        self.block_tables = [FakeTable(rows, pages) for _ in range(groups)]


class FakeBuffers:
    def __init__(self, rows, tokens):
        self.seq_lens = torch.zeros(rows, dtype=torch.int32)
        self.positions = torch.zeros(tokens, dtype=torch.int64)


class FakeRunner:
    def __init__(self, rows, tokens, groups):
        self.block_tables = FakeBlockTables(rows, PAGES, groups)
        self.input_buffers = FakeBuffers(rows, tokens)


def main():
    # 4 KV groups; only group 2 is a FullAttentionSpec group (the Mamba groups 0/1/3 are ignored).
    runner = FakeRunner(rows=2, tokens=8, groups=4)
    paged = [torch.zeros(PAGES, 2, PAGE, 512, dtype=torch.uint8)]
    view = LiveRunnerView(runner, attn_groups={2}, paged_tensors=paged, page_tokens=PAGE)

    check(view.num_blocks() == PAGES, "num_blocks comes from the paged tensor")
    check(len(view.block_table_rows()) == 1, "only the attention group's table is exposed")

    # Batch: request "b" is in row 0, "a" in row 1 (idx_mapping maps batch position -> row).
    view.bind_step(idx_mapping=[1, 0], req_ids=["b", "a"], query_start_loc=[0, 3, 5])
    check(view.row_index("b") == 1, "request b maps to row 1 through idx_mapping")
    check(view.row_index("a") == 0, "request a maps to row 0 through idx_mapping")
    check(view.row_index("zzz") is None, "an unscheduled request has no row")
    check(view.query_position_slots("b") == [0, 1, 2], "b's query slots come from query_start_loc")
    check(view.query_position_slots("a") == [3, 4], "a's query slots follow b's")

    # seq_lens/positions are the runner's own buffers.
    view.seq_lens()[0] = 123
    check(int(runner.input_buffers.seq_lens[0]) == 123, "seq_lens writes through to the runner")
    view.positions()[4] = 77
    check(int(runner.input_buffers.positions[4]) == 77, "positions writes through to the runner")

    print("PASS live_view_test" if not failed else "FAILED live_view_test")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
