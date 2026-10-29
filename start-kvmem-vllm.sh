#!/usr/bin/env bash
# ============================================================================
#  start-kvmem-vllm.sh —— 0.3.0 验证过的配置（常驻模式 + 超池整体重算）
#
#  这一份是"一套站点配置"，不放通用逻辑：通用启动在 deploy/run-current.sh，
#  参数含义见 docs/PARAMETERS.md。下面每个变量都能用同名环境变量覆盖。
#
#  必填：MODEL=<权重目录>            （served 名默认 Qwen3.8-27B-FP8-MTP）
#  常用：PORT=18200  B=36864  R=48000  GPU_UTIL=0.65  N=1
# ============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

: "${MODEL:?MODEL 未设置：指向权重目录，例如 MODEL=\$HOME/models/<your-model>}"

# --- 0.3.0 验收时的配置（改了就重跑 docs/DELIVERY.md 第六节的验收）----------
export GPU_UTIL=${GPU_UTIL:-0.65}
export B=${B:-36864}
export R=${R:-48000}
export N=${N:-1}
export PORT=${PORT:-18200}

# 常驻模式：入口不裁剪 + 池内装不下时在 prefill 期间压缩
export INREQUEST_COMPACTION=${INREQUEST_COMPACTION:-1}
export KVMEM_ENTRY_RESIDENT_PREFILL=${KVMEM_ENTRY_RESIDENT_PREFILL:-1}
export KVMEM_RESIDENT_PREFILL_COMPACT=${KVMEM_RESIDENT_PREFILL_COMPACT:-1}

# 块与池：engine block 16、允许逻辑 prompt 超过物理池
export KVMEM_BLOCK_SIZE=${KVMEM_BLOCK_SIZE:-16}
export KVMEM_SKIP_KV_CHECK=${KVMEM_SKIP_KV_CHECK:-1}
export KVMEM_MAX_MODEL_LEN=${KVMEM_MAX_MODEL_LEN:-200000}

# prompt 结束时用两张卡的平均评分选片，并把工作集整体重算（实验路径）
export KVMEM_COMPACT_REBUILD=${KVMEM_COMPACT_REBUILD:-1}
export KVMEM_PROMPT_END_RESELECT=${KVMEM_PROMPT_END_RESELECT:-1}

# 打分与执行：page-softmax + mean，V2 runner，eager（检索需要每步的 Python 钩子）
export KVMEM_STATS=${KVMEM_STATS:-core}
export KVMEM_FORCE_EAGER=${KVMEM_FORCE_EAGER:-1}
export VLLM_USE_V2_MODEL_RUNNER=${VLLM_USE_V2_MODEL_RUNNER:-1}

# 引擎取仓库内固定并打过补丁的 checkout（install.sh 准备）
export ADAPTER=${ADAPTER:-$HERE}
export VLLM_DIR=${VLLM_DIR:-$HERE/engines/vllm}
export VLLM_ROOT=${VLLM_ROOT:-$VLLM_DIR}
export VLLM_PY=${VLLM_PY:-$VLLM_DIR/.venv/bin/python}
export LOG=${LOG:-$HERE/run-logs/kvmem-vllm.log}
export KV_PIDFILE=${KV_PIDFILE:-$HERE/run-logs/kvmem-vllm.pid}
mkdir -p "$(dirname "$LOG")"

# 前台模式：worker 起完引擎就退出，引擎自己 setsid 常驻（systemd 用 PIDFile 跟踪）。
# 不能用默认的非前台模式：那段逻辑会先杀掉所有 VLLM 进程，systemd 下会连环重启。
export FOREGROUND=1

"$HERE/deploy/run-kvmem-vllm.sh"
rc=$?
[ "$rc" = "0" ] || exit "$rc"

# deploy 脚本在 FOREGROUND 下 exec 了 worker；这里补写引擎 PID 供 Type=forking 使用。
api_pid=$(pgrep -f "vllm.entrypoints.openai.api_server.*--port $PORT" | head -1)
if [ -z "${api_pid:-}" ]; then
  echo "[start-kvmem-vllm] 引擎未就绪：端口 $PORT 上没有 api_server 进程" >&2
  exit 1
fi
printf '%s\n' "$api_pid" > "$KV_PIDFILE"
echo "[start-kvmem-vllm] ready: http://127.0.0.1:$PORT/v1 (pid $api_pid, log $LOG)"
