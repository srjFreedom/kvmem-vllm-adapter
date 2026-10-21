#!/usr/bin/env bash
# ============================================================================
# KVMem + vLLM(SM75) —— 单服务、单端口（kvmem 官方端口 18200）
#
#   * CTX   = kvmem 逻辑上下文（可保留历史总量），与引擎窗口无关
#   * B / R = 入口策略：发多少历史 / 允许生成多少（R 就是 max_tokens 上限）
#   * 引擎窗口 = N x (B + R)，由 worker 自动计算
#   * 入口策略（选 B、max_tokens=R、CTX 检查、H/D 历史块缓存）跑在 vLLM 的 API server 进程内，
#     通过官方 vllm.endpoint_plugins 扩展点加载，不再有旁挂网关。
#
#   覆盖：B= R= SINK= RECENT= SESSIONS= HOST_MIB= N= CTX= PORT= ./本脚本
# ============================================================================
set -uo pipefail
LOG=${LOG:-/home/srj/kvmem.log}
PORT=${PORT:-18200}
B=${B:-36864}
R=${R:-16384}
SINK=${SINK:-8192}
RECENT=${RECENT:-2048}
SESSIONS=${SESSIONS:-4}
HOST_MIB=${HOST_MIB:-49152}
N=${N:-1}
CTX=${CTX:-524288}
# 检索用的 query 捕获窗口（KVMem query_max_tokens 的官方默认是 512）
CAPTURE_TOKENS=${CAPTURE_TOKENS:-512}
MODEL=${MODEL:-/home/srj/models/Coder390/FP8-fp8mtp}
# 对外模型 id：按实际权重命名（FP8 + MTP）。实际加载的是
# /home/srj/models/Coder390/FP8-fp8mtp，不是 llama.cpp 的 Q6_K GGUF，也不是 ninfer 的 .ninfer。
SERVED_NAME=${SERVED_NAME:-Qwen3.8-27B-Coder390-EfficientThink-FP8-MTP}

ADAPTER=${ADAPTER:-/home/srj/vllm-kvmem-adapter}
KV_PIDFILE=${KV_PIDFILE:-/home/srj/kvmem-vllm.pid}
PARAMS_FILE=${PARAMS_FILE:-/home/srj/kvmem-params.env}

# FOREGROUND=1: run under systemd (no daemonising, no self-cleanup - the unit owns the process).
if [ "${FOREGROUND:-0}" != "1" ]; then
  for p in $(ps -eo pid=,args= | awk '/VLLM::|vllm\.entrypoints|launcher\.sh/ && !/awk/ {print $1}'); do
    kill -9 "$p" 2>/dev/null
  done
  for _ in $(seq 1 60); do ss -tln 2>/dev/null | grep -q ":$PORT " || break; sleep 1; done
fi

# 入口策略参数：单一来源，避免 sink/recent 在引擎与入口两边漂移
export VLLM_PLUGINS=${VLLM_PLUGINS:-kvmem,kvmem_entry}
# 入口策略实现方式：middleware = 标准 Starlette 中间件（默认）；hook = 解析后请求上的兜底钩子
export KVMEM_ENTRY_MODE=${KVMEM_ENTRY_MODE:-middleware}
export KVMEM_ENTRY_B="$B"
export KVMEM_ENTRY_R="$R"
export KVMEM_ENTRY_CTX="$CTX"
export KVMEM_ENTRY_SINK="$SINK"
export KVMEM_ENTRY_RECENT="$RECENT"
export KVMEM_ENTRY_SELECT=${KVMEM_ENTRY_SELECT:-retrieval}
export KVMEM_CAPTURE_TOKENS="$CAPTURE_TOKENS"
export KVMEM_ENTRY_HISTORY_KIB=${KVMEM_ENTRY_HISTORY_KIB:-64}
export KVMEM_ENTRY_HISTORY_DISK_MIB=${KVMEM_ENTRY_HISTORY_DISK_MIB:-256}
export KVMEM_ENTRY_HISTORY_DISK_PATH=${KVMEM_ENTRY_HISTORY_DISK_PATH:-/tmp/kvmem_history_tier}

{
  echo "==================== $(date '+%F %T') KVMem+vLLM start ===================="
  echo "CTX(logical)=$CTX  B=$B  R=$R  sink=$SINK  recent=$RECENT  sessions=$SESSIONS  H=${HOST_MIB}MiB  N=$N"
  echo "engine window = N x (B+R) = $(( N * (B + R) ))   port=$PORT   served=$SERVED_NAME"
  echo "entry policy = in-process, mode=$KVMEM_ENTRY_MODE (vllm.endpoint_plugins: kvmem_entry)  select=$KVMEM_ENTRY_SELECT"
  echo "query capture = last $CAPTURE_TOKENS tokens per step (KVMEM_CAPTURE_TOKENS)"
} >> "$LOG"

# 把本次生效的 kvmem 参数落盘（诊断用；可选网关读同一份）
printf 'CTX=%s\nB=%s\nR=%s\nSINK=%s\nRECENT=%s\nSESSIONS=%s\nHOST_MIB=%s\nN=%s\nPORT=%s\nCAPTURE_TOKENS=%s\n' \
  "$CTX" "$B" "$R" "$SINK" "$RECENT" "$SESSIONS" "$HOST_MIB" "$N" "$PORT" "$CAPTURE_TOKENS" > "$PARAMS_FILE"

RUN=(env NO_ASYNC_SCHEDULING=1 MAX_NUM_SEQS="$N" \
  /home/srj/vLLM-2080Ti-Definitive/.venv/bin/python \
  "$ADAPTER/backends/vllm/worker.py" \
    --model "$MODEL" --port "$PORT" --served-name "$SERVED_NAME" \
    --tensor-parallel-size 2 --max-num-seqs "$N" \
    --kvmem-budget "$B" --kvmem-gen-reserve "$R" \
    --kvmem-sink-tokens "$SINK" --kvmem-recent-tokens "$RECENT" \
    --kvmem-sessions "$SESSIONS" --kvmem-host-mib "$HOST_MIB" \
    --kvmem-capture-tokens "$CAPTURE_TOKENS" \
    --kvmem-block-tokens 128 \
    --kvmem-lrfu-lambda 0.5 --kvmem-lrfu-top-frac 0.9 \
    --kvmem-score-aggregation max --kvmem-retrieval)

if [ "${FOREGROUND:-0}" = "1" ]; then
  echo "foreground mode (systemd); log: $LOG"
  exec "${RUN[@]}" >> "$LOG" 2>&1
fi

echo "starting engine on :$PORT (log: $LOG) ..."
setsid "${RUN[@]}" >> "$LOG" 2>&1 < /dev/null &

# 就绪判定：短轮询 + 日志触发 + 失败快退（不在固定 sleep 上硬等）
"$ADAPTER/scripts/wait_ready.sh" "$PORT" "${READY_TIMEOUT:-900}" --log "$LOG"
ready=$?
[ "$ready" = "0" ] || exit "$ready"

api_pid=$(pgrep -f "vllm.entrypoints.openai.api_server.*--port $PORT" | head -1)
if [ -n "${api_pid:-}" ]; then
  printf '%s\n' "$api_pid" > "$KV_PIDFILE" 2>/dev/null || true
  echo "pid $api_pid -> $KV_PIDFILE"
fi
echo "READY http://127.0.0.1:$PORT/v1  (tail -f $LOG)"
