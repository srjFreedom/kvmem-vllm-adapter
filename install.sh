#!/usr/bin/env bash
# ============================================================================
# 从零把 KVMem + vLLM 集成装起来（幂等，可重复执行）
#
#   ./install.sh                     # 用默认路径
#   VLLM_ROOT=... KVMA_SRC=... ./install.sh
#
# 需要的外部件（不在本仓库里）：
#   1. vLLM SM75 fork：按 backends/versions.json 拉固定 base_commit，再整体打
#      backends/patches/vllm-kvmem.patch（第三方引擎只用补丁替换，不 vendored）
#   2. kvmem 源码（编译扩展要它的头文件和 kvmem 库）
#   3. 模型权重
#
# 默认不碰任何仓库外的已有 vLLM 目录（它们可能是别人的日常实例）。
# 只有显式 USE_LOCAL_FORK=1 VLLM_ROOT=<自备 fork> 才会改写那个 fork。
# ============================================================================
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
VLLM_ROOT=${VLLM_ROOT:-$ROOT/engines/vllm}
KVMA_SRC=${KVMA_SRC:-$HOME/src/kvmem-v018rc1}
PY=${PY:-"$VLLM_ROOT/.venv/bin/python"}
PATCH="$ROOT/patches/qwen3-next-capture-q.patch"
TARGET="$VLLM_ROOT/vllm/model_executor/models/qwen3_next.py"

say() { printf '\n[install] %s\n' "$*"; }

FETCH_LOCAL=${USE_LOCAL_FORK:-0}
say "1/5 检查外部件"
if [ "$FETCH_LOCAL" = "1" ]; then
  [ -d "$VLLM_ROOT/vllm" ] || { echo "缺 vLLM fork: $VLLM_ROOT（或用 USE_LOCAL_FORK=0 自动拉取）"; exit 1; }
fi
if [ ! -f "$KVMA_SRC/kvmem/include/kvmem/kvmem_runtime.hpp" ]; then
  echo "  kvmem 源码不在 $KVMA_SRC，按 versions.json 拉取固定 tag"
  python3 "$ROOT/scripts/prepare-backends.py" --engine kvmem
  KVMA_SRC="$ROOT/engines/kvmem"
fi
[ -x "$PY" ] || { echo "缺 venv python: $PY"; exit 1; }
if [ "$FETCH_LOCAL" = "1" ]; then
  echo "  vLLM   : $VLLM_ROOT（使用本地 fork）"
else
  echo "  vLLM   : 将从 backends/versions.json 拉取到 engines/vllm"
fi
echo "  kvmem  : $KVMA_SRC"
echo "  python : $PY"

say "2/5 固定引擎版本 + 应用 KVMem 集成补丁（kvmem 同款：子模块 + 补丁）"
# 已经在这台机器上装好的 fork 直接打补丁；否则由 prepare-backends 拉取固定 commit 再打。
if [ "${USE_LOCAL_FORK:-0}" = "1" ] && [ -d "$VLLM_ROOT/.git" ]; then
  if [ "$VLLM_ROOT" = "$ROOT/engines/vllm" ]; then
    echo "USE_LOCAL_FORK=1 需要显式 VLLM_ROOT=<你自己的 fork>；仓库内引擎走补丁流程（USE_LOCAL_FORK=0）" >&2
    exit 1
  fi
  if (cd "$VLLM_ROOT" && git diff --quiet -- vllm/model_executor/models/qwen3_next.py) \
     || ! grep -q kvmem_bridge "$TARGET"; then
    cp "$ROOT/vendor/vllm/kvmem_bridge.py" "$VLLM_ROOT/vllm/kvmem_bridge.py"
    if ! grep -q "kvmem_bridge" "$TARGET"; then
      git -C "$VLLM_ROOT" apply "$PATCH" || { echo "  补丁冲突，请检查 $VLLM_ROOT"; exit 1; }
    fi
  fi
  cp "$ROOT/vendor/vllm/kvmem_bridge.py" "$VLLM_ROOT/vllm/kvmem_bridge.py"
  echo "  本地 fork 已就绪: $VLLM_ROOT"
else
  python3 "$ROOT/scripts/prepare-backends.py" --engine vllm
  VLLM_ROOT="$ROOT/engines/vllm"
  PY="$VLLM_ROOT/.venv/bin/python"
  [ -x "$PY" ] || {
    echo "  该引擎还没有 Python 环境，先跑："
    echo "      ENGINE=$VLLM_ROOT ./scripts/setup-venv.sh"
    echo "  （会按 fork 的 requirements 装 torch/CUDA 依赖，几个 GB；若 fork 有官方配方以其为准）"
    exit 1
  }
fi

say "3/5 安装插件包（editable）"
if command -v uv >/dev/null 2>&1; then
  uv pip install -e "$ROOT/plugin" --python "$PY" --no-deps -q
else
  "$PY" -m pip install -e "$ROOT/plugin" --no-deps -q
fi
echo "  -> $(cat "$VLLM_ROOT"/.venv/lib/python3.12/site-packages/vllm_kvmem_plugin-*.dist-info/entry_points.txt | grep -c '=') 个入口点"

say "4/5 编译原生扩展"
if [ -f "$ROOT/build/python/vllm_kvmem_ext.cpython-312-x86_64-linux-gnu.so" ]; then
  echo "  已有构建产物，跳过（要重编先删 build/）"
else
  cmake -S "$ROOT" -B "$ROOT/build" -DKVMEM_BASE="$KVMA_SRC" -DCMAKE_BUILD_TYPE=RelWithDebInfo >/dev/null
  cmake --build "$ROOT/build" -j "$(nproc)" >/dev/null
  echo "  -> build/python/"
fi

say "5/5 自检"
PYTHONPATH="$ROOT/build/python:$ROOT/src:$ROOT/plugin" "$PY" - <<'PYCHECK'
import importlib.metadata as md
eps = md.distribution("vllm-kvmem-plugin").entry_points
groups = sorted({ep.group for ep in eps})
print("  entry points:", groups)
from kvmem_entry_policy import EntryPolicy
print("  entry policy :", EntryPolicy().budget, EntryPolicy().gen_reserve)
PYCHECK
"$ROOT/scripts/check_integration_surface.py" --vllm-root "$VLLM_ROOT" --kvmem-root "$KVMA_SRC" 2>/dev/null | tail -1 || true

cat <<NEXT

[install] 完成。启动方式（仓库默认参数即可跑，只差 MODEL；模型无内置模板时再加 CHAT_TEMPLATE_FILE）：

  前台：
    MODEL=$HOME/models/<你的模型> $ROOT/start-kvmem-vllm.sh

  服务：
    sudo $ROOT/deploy/install-service.sh
    mkdir -p ~/.config && printf 'MODEL=%s\n' "$HOME/models/<你的模型>" > ~/.config/kvmem-vllm.env
    sudo systemctl daemon-reload && sudo systemctl enable --now kvmem-vllm

  就绪检查：$ROOT/scripts/wait_ready.sh 18200 900 --smoke
NEXT
