#!/usr/bin/env python3
"""Integration-surface probe: what do we depend on, and is it still there?

Run this right after upgrading kvmem or vLLM. It reports every touch point as OK / MISSING /
CHANGED with the file or attribute it looked at, so an upgrade turns into "fix these three lines"
instead of "start the engine and guess".

    scripts/check_integration_surface.py [--vllm-root DIR] [--kvmem-root DIR] [--json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

HOME_DIR = os.path.expanduser("~")
VLLM_DEFAULT = os.environ.get("VLLM_ROOT") or os.path.join(HOME_DIR, "vLLM-2080Ti-Definitive")
KVMEM_DEFAULT = os.environ.get("KVMA_SRC") or os.path.join(HOME_DIR, "src", "kvmem-v018rc1")

results = []


def record(side, name, ok, detail=""):
    results.append({"side": side, "name": name, "ok": bool(ok), "detail": detail})
    print(f"{'ok  ' if ok else 'MISS'} [{side}] {name}" + (f"  ({detail})" if detail and not ok else ""))


def grep_any(root, pattern, exts=(".py", ".hpp", ".h", ".cpp", ".sh")):
    rx = re.compile(pattern)
    for base, _dirs, files in os.walk(root):
        if any(part in base for part in (".git", "build", "__pycache__", "node_modules")):
            continue
        for name in files:
            if not name.endswith(exts):
                continue
            path = os.path.join(base, name)
            try:
                with open(path, "r", errors="ignore") as handle:
                    for lineno, line in enumerate(handle, 1):
                        if rx.search(line):
                            return f"{os.path.relpath(path, root)}:{lineno}"
            except OSError:
                continue
    return None


def check_vllm(root):
    # --- extension points we are ALLOWED to use -------------------------------------------
    record("vllm", "entry-point group vllm.general_plugins",
           bool(grep_any(root, r"vllm\.general_plugins")))
    # The in-process entry policy rides on the endpoint-plugin group: vLLM loads it into the API
    # server process and calls attach_router(app), which is where our middleware is installed.
    record("vllm", "entry-point group vllm.endpoint_plugins",
           bool(grep_any(root, r"vllm\.endpoint_plugins")))
    record("vllm", "attach_endpoint_plugins() in the app builder",
           bool(grep_any(root, r"def attach_endpoint_plugins")))
    record("vllm", "EndpointPlugin.attach_router hook",
           bool(grep_any(root, r"def attach_router")))
    # The entry policy rides on the parsed chat request; this is the method we wrap.
    record("vllm", "OpenAIServingChat.create_chat_completion",
           bool(grep_any(root, r"def create_chat_completion")))
    record("vllm", "VllmConfig.additional_config",
           bool(grep_any(root, r"additional_config:\s*dict")))
    record("vllm", "launcher supports ADDITIONAL_CONFIG_JSON",
           bool(grep_any(root, r"ADDITIONAL_CONFIG_JSON")))
    record("vllm", "full-attention spec builder.build()",
           bool(grep_any(root, r"class FullAttentionSpec")) and
           bool(grep_any(root, r"def build\(")))
    record("vllm", "attention kernel takes seq_lens (unified/flashinfer wrapper)",
           bool(grep_any(root, r"unified_attention")) or bool(grep_any(root, r"flashinfer")))

    # --- internals we PATCH (each one is a line in INTEGRATION-SURFACE.md) ----------------
    # Either runner class may carry a hook; the plugin patches every class it finds.
    for hook in ("execute_model", "prepare_pos_seq_lens"):
        found = []
        for cls in ("vllm/v1/worker/gpu/model_runner.py", "vllm/v1/worker/gpu_model_runner.py"):
            path = os.path.join(root, cls)
            if os.path.exists(path) and hook in open(path, errors="ignore").read():
                found.append(os.path.basename(cls))
        record("vllm", f"runner hook: {hook}", bool(found), ",".join(found) or "not found")
    record("vllm", "GPUModelRunner.vllm_config", bool(grep_any(root, r"self\.vllm_config = ")))
    record("vllm", "Scheduler.schedule", bool(grep_any(root, r"def schedule\(")))
    record("vllm", "block pool get_new_blocks", bool(grep_any(root, r"def get_new_blocks")))
    record("vllm", "block pool get_num_free_blocks", bool(grep_any(root, r"def get_num_free_blocks")))
    record("vllm", "_remove_blocks_in_range (aligned-block release)",
           bool(grep_any(root, r"def _remove_blocks_in_range")))
    record("vllm", "KVCacheManager.req_to_blocks",
           bool(grep_any(root, r"req_to_blocks")))


def check_kvmem(root):
    include = os.path.join(root, "kvmem", "include", "kvmem")
    required = {
        "MemoryBackend/MemoryTransfer/MemorySession": "memory_contract.hpp",
        "plan_working_set": "memory_contract.hpp",
        "WorkingSetPlan/ExecutionView": "memory_contract.hpp",
        "HostKvStorage/HostKvRecord": "host_kv_storage.hpp",
        "KvMemStore": "kvmem_store.hpp",
        "mean_key_retrieval_scores": "mean_key_retrieval.hpp",
        "nvme tier (disk budget)": "nvme_kv_tier.hpp",
        "cli_options (--kvmem-*)": "cli_options.h",
    }
    for label, header in required.items():
        path = os.path.join(include, header)
        record("kvmem", label, os.path.exists(path), header)

    record("kvmem", "prepare_reselect on MemorySession",
           bool(grep_any(include, r"prepare_reselect")))
    record("kvmem", "capabilities.pre_rope_statistics",
           bool(grep_any(include, r"pre_rope_statistics")))
    record("kvmem", "pre-RoPE capture layer (reference implementation)",
           bool(grep_any(os.path.join(root, "src", "adapter"), r"kvmem_capture_register")))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-root", default=VLLM_DEFAULT)
    ap.add_argument("--kvmem-root", default=KVMEM_DEFAULT)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if os.path.isdir(a.vllm_root):
        check_vllm(a.vllm_root)
    else:
        record("vllm", "tree present", False, a.vllm_root)
    if os.path.isdir(a.kvmem_root):
        check_kvmem(a.kvmem_root)
    else:
        record("kvmem", "tree present", False, a.kvmem_root)
    missing = [r for r in results if not r["ok"]]
    if a.json:
        json.dump({"results": results, "missing": len(missing)}, sys.stdout, indent=2)
        sys.stdout.write("\n")
    print(f"\n{len(results) - len(missing)}/{len(results)} touch points present"
          + (f"; {len(missing)} need attention" if missing else ""))
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
