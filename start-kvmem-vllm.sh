#!/usr/bin/env bash
# ============================================================================
# KVMem + vLLM (SM75 fork) —— 0.3.0 组合启动
#
#   结构对齐 start-kvmem-coder390-q6.sh：kvmem 参数在前、是主口径；vLLM 参数在后、
#   只负责"怎么把这块显存和窗口摆下"。每项都可用同名环境变量覆盖：
#       BUDGET=65536 RESERVE=32768 GPU_UTIL=0.9 DRYRUN=1 ./start-kvmem-vllm.sh
#
#   kvmem 侧（主）
#     BUDGET   选中历史预算 B (36K)      RESERVE  追加工作区 R (16K，输出上限)
#     SINK     头部固定 (12K)           RECENT   尾部固定 (1K)
#     CTX      逻辑上下文上限 (256K)     HOST_MIB H 层主机内存预算 (16G)
#     SESSIONS H 池会话份数 (1)          BLOCK    选片网格
#     CAPTURE  每步 query 捕获窗口       DRAFT    MTP 投机 token 数
#     OVER_POOL=1 允许 prompt 超出物理池：prefill 期压缩 + prompt 末整体重算
#   vLLM 侧（辅）
#     GPU_UTIL 显存占比  MAX_BATCHED_TOKENS prefill chunk（取 1616 的整数倍）
#     MAX_MODEL_LEN 引擎窗口  SERVED_NAME 对外 id  CHAT_TEMPLATE_FILE 模板
#     MESSAGE_TYPE / MM_IMAGE_LIMIT 多模态  THINK_BUDGET 思考预算（0=不限）
#
#   引擎用仓库内固定并打过补丁的 checkout（install.sh 准备），不碰仓库外的 fork。
# ============================================================================
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

: "${MODEL:?MODEL 未设置：指向权重目录，例如 MODEL=\$HOME/models/<your-model>}"

# ---- kvmem 预算（主口径；下面是站点默认值） ----
export BUDGET=${BUDGET:-36864}          # B：选中历史 36K
export RESERVE=${RESERVE:-16384}        # R：追加工作区 16K（输出上限）
export SINK=${SINK:-12288}              # sink：头部固定 12K
export RECENT=${RECENT:-1024}           # recent：尾部固定 1K
export CTX=${CTX:-262144}               # 逻辑上下文上限 256K（与引擎窗口无关）
export HOST_MIB=${HOST_MIB:-16384}      # H 层主机预算 16 GiB
export SESSIONS=${SESSIONS:-1}          # H 池会话份数
export BLOCK=${BLOCK:-32}               # 选片网格（16 的整数倍；32 = 两条 needle 全中）
export CAPTURE=${CAPTURE:-512}          # query 捕获窗口（kvmem query_max_tokens）
export DRAFT=${DRAFT:-4}                # MTP4
export CONCURRENCY=${CONCURRENCY:-1}    # 并发请求数 N
export PORT=${PORT:-18200}

# ---- 超池（0.3.0 实验路径，默认开：这是本版本验收过的能力） ----
export OVER_POOL=${OVER_POOL:-1}
# ---- KV 工作区实现细节（一般不用改） ----
export WORKSPACE=${WORKSPACE:-16}       # KVMem 页 / 请求引擎的块大小
export SCORE_AGG=${SCORE_AGG:-mean}     # page-softmax + mean（LRFU 关闭：已验证路径）

# ---- vLLM 侧（辅） ----
export GPU_UTIL=${GPU_UTIL:-0.97}
export MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-4096}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:--1}   # -1 = 按 KV 池自动适配窗口
export SERVED_NAME=${SERVED_NAME:-Qwen3.8-27B-FP8-MTP}
export MESSAGE_TYPE=${MESSAGE_TYPE:-text+image}
export MM_IMAGE_LIMIT=${MM_IMAGE_LIMIT:-64}
export SERVICE_SCOPE=${SERVICE_SCOPE:-lan}
export TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-qwen3_xml}
export THINK_BUDGET=${THINK_BUDGET:-0}  # 0 = 不限制

# ---- 引擎与入口（仓库内固定 checkout） ----
export ADAPTER=${ADAPTER:-$HERE}
export VLLM_DIR=${VLLM_DIR:-$HERE/engines/vllm}
export VLLM_ROOT=${VLLM_ROOT:-$VLLM_DIR}
export VLLM_PY=${VLLM_PY:-$VLLM_DIR/.venv/bin/python}
export LOG=${LOG:-$HERE/run-logs/kvmem-vllm.log}
export KV_PIDFILE=${KV_PIDFILE:-$HERE/run-logs/kvmem-vllm.pid}

[ -d "$VLLM_DIR/vllm" ] || { echo "未找到引擎: $VLLM_DIR（先跑 install.sh / prepare-backends）" >&2; exit 1; }
[ -x "$VLLM_PY" ]       || { echo "未找到引擎 venv: $VLLM_PY" >&2; exit 1; }
[ -f "$MODEL" ] || [ -d "$MODEL" ] || { echo "未找到模型: $MODEL" >&2; exit 1; }

# vLLM 侧 chunk：切成 1616（mamba align 块）的整数倍收益最好（如 6464=4 块）。
# 本版验收用的是 4096，所以默认值不动；调大后请按 docs/DELIVERY.md 第六节重跑验收。
if [ $((MAX_BATCHED_TOKENS % 1616)) -ne 0 ]; then
  echo "note: MAX_BATCHED_TOKENS=$MAX_BATCHED_TOKENS 非 1616 整数倍（6464/8080 可减少切块）" >&2
fi

# ---- kvmem 主口径 -> 适配器环境变量 ----
export B="$BUDGET" R="$RESERVE"
export N="$CONCURRENCY"
export SELECT_BLOCK_TOKENS="$BLOCK"
export CAPTURE_TOKENS="$CAPTURE"
export SPECULATIVE_TOKENS="$DRAFT"
export KVMEM_CAPTURE_TOKENS="$CAPTURE"
export THINKING_BUDGET="$THINK_BUDGET"
[ "$THINK_BUDGET" = "0" ] && unset THINKING_BUDGET

# 常驻工作区：入口不裁剪、池内装不下时在 prefill 期间压缩
export INREQUEST_COMPACTION=${INREQUEST_COMPACTION:-1}
export KVMEM_ENTRY_RESIDENT_PREFILL=${KVMEM_ENTRY_RESIDENT_PREFILL:-1}
export KVMEM_BLOCK_SIZE=${KVMEM_BLOCK_SIZE:-$WORKSPACE}
export KVMEM_STATS=${KVMEM_STATS:-core}
export KVMEM_FORCE_EAGER=${KVMEM_FORCE_EAGER:-1}
export VLLM_USE_V2_MODEL_RUNNER=${VLLM_USE_V2_MODEL_RUNNER:-1}

if [ "$OVER_POOL" = "1" ]; then
  export KVMEM_RESIDENT_PREFILL_COMPACT=${KVMEM_RESIDENT_PREFILL_COMPACT:-1}
  export KVMEM_SKIP_KV_CHECK=${KVMEM_SKIP_KV_CHECK:-1}
  export KVMEM_MAX_MODEL_LEN=${KVMEM_MAX_MODEL_LEN:-$MAX_MODEL_LEN}
  export KVMEM_COMPACT_REBUILD=${KVMEM_COMPACT_REBUILD:-1}
  export KVMEM_PROMPT_END_RESELECT=${KVMEM_PROMPT_END_RESELECT:-1}
fi
[ -n "${CHAT_TEMPLATE_FILE:-}" ] && export CHAT_TEMPLATE_FILE

# 启动器的内置短 smoke 在 -1 自动窗口 + 大 sink 的站点配置下会返回空内容并自清理；
# 这里跳过它，改为本脚本在就绪后自己探 /health。
export SKIP_STARTUP_SMOKE=${SKIP_STARTUP_SMOKE:-1}

# 前台模式：worker 起完引擎就退出，引擎自己 setsid 常驻（systemd 用 PIDFile 跟踪）。
# 不能用默认的非前台模式：那段逻辑会先杀掉所有 VLLM 进程，systemd 下会连环重启。
export FOREGROUND=${FOREGROUND:-1}

echo "kvmem: B=$BUDGET R=$RESERVE sink=$SINK recent=$RECENT CTX=$CTX H=${HOST_MIB}MiB" \
     "sessions=$SESSIONS block=$BLOCK capture=$CAPTURE draft=$DRAFT over_pool=$OVER_POOL"
echo "vllm : util=$GPU_UTIL chunk=$MAX_BATCHED_TOKENS window=$MAX_MODEL_LEN port=$PORT served=$SERVED_NAME"

"$HERE/deploy/run-kvmem-vllm.sh"
rc=$?
[ "$rc" = "0" ] || exit "$rc"

# deploy 脚本在 FOREGROUND 下 exec 了 worker；这里补写引擎 PID 供 Type=forking 使用。
api_pid=$(pgrep -f "vllm.entrypoints.openai.api_server.*--port $PORT" | head -1)
if [ -z "${api_pid:-}" ]; then
  echo "[start-kvmem-vllm] 引擎未就绪：端口 $PORT 上没有 api_server 进程" >&2
  exit 1
fi
# 就绪确认：/health 200 才写 PIDFile，避免 systemd 把半启动的引擎当成可用。
code=""
for _ in $(seq 1 30); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:$PORT/health" || true)
  [ "$code" = "200" ] && break
  sleep 5
done
if [ "$code" != "200" ]; then
  echo "[start-kvmem-vllm] /health=$code（端口 $PORT）；不写 PIDFile" >&2
  exit 1
fi
printf '%s\n' "$api_pid" > "$KV_PIDFILE"
echo "[start-kvmem-vllm] ready: http://127.0.0.1:$PORT/v1 (pid $api_pid, log $LOG)"
