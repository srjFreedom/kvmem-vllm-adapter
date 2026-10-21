#!/usr/bin/env bash
# Create the engine's Python environment (torch matching the fork's pin + vLLM itself).
#
# This is the one heavy, network-dependent step: the fork pins torch (2.13.0) and CUDA 13.2, so the
# wheel set is several GB. If your fork documents a different recipe, follow that instead - the rest
# of the flow only needs `$ENGINE/.venv/bin/python` to exist.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ENGINE=${ENGINE:-$ROOT/engines/vllm}
PY_VER=${PY_VER:-3.12}

[ -d "$ENGINE/vllm" ] || { echo "引擎未就绪，先跑: python3 scripts/prepare-backends.py --engine vllm"; exit 1; }
command -v uv >/dev/null || { echo "需要 uv（https://docs.astral.sh/uv/getting-started/installation/）"; exit 1; }

echo "[venv] $ENGINE/.venv (python $PY_VER)"
uv venv --python "$PY_VER" "$ENGINE/.venv"
uv pip install -r "$ENGINE/requirements/common.txt" --python "$ENGINE/.venv/bin/python"
uv pip install -r "$ENGINE/requirements/cuda.txt"  --python "$ENGINE/.venv/bin/python"
uv pip install -e "$ENGINE" --no-build-isolation --python "$ENGINE/.venv/bin/python"

"$ENGINE/.venv/bin/python" - <<'PY'
import torch, vllm
print(f"[venv] torch {torch.__version__} cuda {torch.version.cuda} | vllm {vllm.__version__}")
PY
