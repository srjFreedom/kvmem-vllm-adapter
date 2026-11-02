#!/usr/bin/env python3
"""prefill_pressure_test: contracts for bounded in-prefill residency (goal-7833cb61).

1) prefill_pressure_due truth table (the throttle semantics of the trigger).
2) source guards: the pressure branch is gated by the flag; the prompt-end socket-shrink
   branch is skipped while the flag is ON (mutual exclusion, no double compaction path);
   the residency measure falls back to the resident BOOK (dense prefill leaves the view
   row empty, so row-length alone would never fire).
3) the flag defaults OFF everywhere (fail-closed: unset env -> production path).
"""
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kvmem_sparse_runtime import prefill_pressure_due  # noqa: E402

failed = [0]


def check(name, cond):
    print(("  ok  " if cond else "FAIL ") + name)
    if not cond:
        failed[0] += 1


def src(path):
    return io.open(path, encoding="utf-8").read()


ROOT = os.path.join(os.path.dirname(__file__), "..")

# 1) trigger truth table
check("fires when resident exceeds high watermark and no prior reselect",
      prefill_pressure_due(40000, 2500, 2368, 0, 16384) is True)
check("does not fire while resident is under the watermark",
      prefill_pressure_due(40000, 2000, 2368, 0, 16384) is False)
check("throttles: re-fires only after the reserve advances",
      (prefill_pressure_due(40000, 2500, 2368, 38000, 16384) is False
       and prefill_pressure_due(60000, 2500, 2368, 38000, 16384) is True))

# 2) source guards
rt = src(os.path.join(ROOT, "src", "kvmem_sparse_runtime.py"))
check("pressure elif honors the new flag OR the classic switch",
      'and (getattr(self, "prefill_pressure", False) or not self.no_prefill_compact)' in rt)
check("resident book is the fallback residency measure",
      "int(st.tokens) // max(1, int(self.page_tokens))" in rt)
check("prompt-end shrink branch is skipped while pressure is ON",
      "if self.no_prefill_compact and not getattr(self, \"prefill_pressure\", False):" in rt)
wk = src(os.path.join(ROOT, "backends", "vllm", "worker.py"))
check("worker publishes the cfg key only from the env flag",
      '_env_flag("KVMEM_PREFILL_PRESSURE")' in wk
      and 'cfg["prefill_pressure"] = True' in wk)
pl = src(os.path.join(ROOT, "plugin", "vllm_kvmem_plugin.py"))
check("plugin installs the runtime flag from cfg (default False)",
      'runtime.prefill_pressure = bool(cfg.get("prefill_pressure", False))' in pl)
dp = src(os.path.join(ROOT, "deploy", "run-kvmem-vllm.sh"))
check("deploy default is OFF (fail-closed)",
      "KVMEM_PREFILL_PRESSURE=${KVMEM_PREFILL_PRESSURE:-0}" in dp)

# v0.5.x architecture note: CTX is a VIRTUAL ceiling; with no disk tier it must
# self-limit to the host share (spilled pages the rebuild needs cannot be dropped).
_rs = src("src/kvmem_sparse_runtime.py")
check("CTX self-limits to the H share without a disk tier",
      "host_capacity_short" in _rs and "self.ctx_tokens = int(share_tokens)" in _rs)

print(("PASS " if not failed[0] else "FAIL ") + "prefill_pressure_test")
sys.exit(1 if failed[0] else 0)
