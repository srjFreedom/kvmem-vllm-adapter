#!/usr/bin/env bash
# ============================================================================
# 从零把 KVMem + vLLM 集成装起来（幂等，可重复执行）
#
#   ./install.sh                     # 用默认路径
#   VLLM_ROOT=... KVMA_SRC=... ./install.sh
#
# 需要的外部件（不在本仓库里）：
#   1. vLLM SM75 fork（会被打一个小补丁 + 放入 kvmem_bridge.py）
#   2. kvmem 源码（编译扩展要它的头文件和 kvmem 库）
#   3. 模型权重
# ============================================================================
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
VLLM_ROOT=${VLLM_ROOT:-/home/srj/vLLM-2080Ti-Definitive}
KVMA_SRC=${KVMA_SRC:-/home/srj/src/kvmem-v018rc1}
PY=${PY:-"$VLLM_ROOT/.venv/bin/python"}
PATCH="$ROOT/patches/qwen3-next-capture-q.patch"
TARGET="$VLLM_ROOT/vllm/model_executor/models/qwen3_next.py"

say() { printf '\n[install] %s\n' "$*"; }

say "1/5 检查外部件"
[ -d "$VLLM_ROOT/vllm" ] || { echo "缺 vLLM fork: $VLLM_ROOT"; exit 1; }
[ -f "$KVMA_SRC/kvmem/include/kvmem/kvmem_runtime.hpp" ] || { echo "缺 kvmem 源码: $KVMA_SRC"; exit 1; }
[ -x "$PY" ] || { echo "缺 venv python: $PY"; exit 1; }
echo "  vLLM   : $VLLM_ROOT"
echo "  kvmem  : $KVMA_SRC"
echo "  python : $PY"

say "2/5 放入 vLLM 侧的 query bridge（本仓库 vendor/ 下的副本）"
cp "$ROOT/vendor/vllm/kvmem_bridge.py" "$VLLM_ROOT/vllm/kvmem_bridge.py"
echo "  -> $VLLM_ROOT/vllm/kvmem_bridge.py"

say "3/5 打 8 行 capture 补丁（已打过则跳过）"
if grep -q "kvmem_bridge" "$TARGET"; then
  echo "  已有 kvmem 调用，跳过"
else
  git -C "$VLLM_ROOT" apply "$PATCH"
  echo "  已应用 $PATCH"
fi

say "4/5 安装插件包（editable）"
if command -v uv >/dev/null 2>&1; then
  uv pip install -e "$ROOT/plugin" --python "$PY" --no-deps -q
else
  "$PY" -m pip install -e "$ROOT/plugin" --no-deps -q
fi
echo "  -> $(cat "$VLLM_ROOT"/.venv/lib/python3.12/site-packages/vllm_kvmem_plugin-*.dist-info/entry_points.txt | grep -c '=') 个入口点"

say "5/5 编译原生扩展"
if [ -f "$ROOT/build/python/vllm_kvmem_ext.cpython-312-x86_64-linux-gnu.so" ]; then
  echo "  已有构建产物，跳过（要重编先删 build/）"
else
  cmake -S "$ROOT" -B "$ROOT/build" -DKVMEM_BASE="$KVMA_SRC" -DCMAKE_BUILD_TYPE=RelWithDebInfo >/dev/null
  cmake --build "$ROOT/build" -j "$(nproc)" >/dev/null
  echo "  -> build/python/"
fi

say "自检"
PYTHONPATH="$ROOT/build/python:$ROOT/src:$ROOT/plugin" "$PY" - <<'PYCHECK'
import importlib.metadata as md
eps = md.distribution("vllm-kvmem-plugin").entry_points
groups = sorted({ep.group for ep in eps})
print("  entry points:", groups)
from kvmem_entry_policy import EntryPolicy
print("  entry policy :", EntryPolicy().budget, EntryPolicy().gen_reserve)
PYCHECK
"$ROOT/scripts/check_integration_surface.py" --vllm-root "$VLLM_ROOT" --kvmem-root "$KVMA_SRC" 2>/dev/null | tail -1 || true

cat <<'NEXT'

[install] 完成。启动方式：

  cp deploy/kvmem-vllm.service /etc/systemd/system/
  cp deploy/run-current.example.sh /home/srj/run-current.sh   # 按需改参数
  sudo systemctl daemon-reload
  sudo systemctl enable --now kvmem-vllm
  scripts/wait_ready.sh 18200 900 --smoke
NEXT
