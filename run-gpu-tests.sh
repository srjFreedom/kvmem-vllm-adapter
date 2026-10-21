#!/usr/bin/env bash
# run-gpu-tests.sh — the GPU-mode verification for the vLLM × KVMem adapter.
#
# Runs ONLY when the serving vLLM is stopped (the GPUs must be free). Per the restart discipline,
# stop the serving vLLM first (a delayed background job, sleep 10–20), then run this script.
#
# What it does:
#   1. gpu_real_tensor_test.py — the async arm on real CUDA tensors (byte-for-byte needle-alive
#      + async-vs-sync timing).
#   2. bench_ab_capacity.py (GPU mode) — the sync-vs-async A/B deep-overload + the 256K-in-64K-VRAM
#      capacity numbers.
#
# Usage: ./run-gpu-tests.sh [--iters N] [--num-blocks M]
set -euo pipefail

cd "$(dirname "$0")"
PY=${VLLM_PY:-${VLLM_ROOT:-$HOME/vLLM-2080Ti-Definitive}/.venv/bin/python}

# Refuse to run if the serving vLLM still holds the GPUs (it would OOM or deadlock).
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; then
    echo "ERROR: a compute process still holds the GPUs. Stop the serving vLLM first." >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >&2
    exit 1
fi

echo "=== GPUs free; running the GPU-mode verification ==="
echo "--- 1/2 GPU 真实张量测试（异步臂，字节级埋针判活 + 同步/异步计时） ---"
$PY tests/gpu_real_tensor_test.py
echo "--- 2/2 A/B + 容量基准（GPU 模式） ---"
$PY bench_ab_capacity.py "$@"
echo "=== GPU 模式验证完成 ==="
