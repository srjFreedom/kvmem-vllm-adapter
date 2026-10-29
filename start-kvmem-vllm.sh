#!/usr/bin/env bash
# ============================================================================
# KVMem + vLLM (SM75 fork) —— 组合启动（仓库侧，不含任何站点参数）
#
#   这里只做两件事：
#     1) 把 kvmem 口径的参数名映射到适配器认识的环境变量；
#     2) 打开 0.3.0 的超池相关开关。
#   所有"值"的默认都在 deploy/run-kvmem-vllm.sh 里（单一来源）；本站点配置写在
#   自己的 start-*.sh 里再 exec 本脚本即可。
#
#   kvmem 口径（不设就用适配器默认）：BUDGET RESERVE SINK RECENT CTX HOST_MIB
#   SESSIONS BLOCK CAPTURE DRAFT CONCURRENCY PORT
#   vLLM 口径：GPU_UTIL MAX_BATCHED_TOKENS MAX_MODEL_LEN SERVED_NAME CHAT_TEMPLATE_FILE …
#   例：BUDGET=65536 GPU_UTIL=0.9 DRYRUN=1 ./start-kvmem-vllm.sh
# ============================================================================
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

: "${MODEL:?MODEL 未设置：指向权重目录，例如 MODEL=\$HOME/models/<your-model>}"

# ---- kvmem 口径 -> 适配器变量（设了才覆盖，值仍由 deploy 脚本兜底） ----
[ -n "${BUDGET:-}" ]      && export B="$BUDGET"
[ -n "${RESERVE:-}" ]     && export R="$RESERVE"
[ -n "${CONCURRENCY:-}" ] && export N="$CONCURRENCY"
[ -n "${BLOCK:-}" ]       && export SELECT_BLOCK_TOKENS="$BLOCK"
[ -n "${CAPTURE:-}" ]     && export CAPTURE_TOKENS="$CAPTURE" KVMEM_CAPTURE_TOKENS="$CAPTURE"
[ -n "${DRAFT:-}" ]       && export SPECULATIVE_TOKENS="$DRAFT"
[ -n "${THINK_BUDGET:-}" ] && export THINKING_BUDGET="$THINK_BUDGET"
# SINK / RECENT / CTX / HOST_MIB / SESSIONS / PORT / GPU_UTIL / MAX_* / SERVED_NAME
# 与适配器同名，直接用环境变量传入，不需要在这里复制一份默认值。

# ---- 0.3.0 能力开关（deploy 脚本里没有默认值，必须在这里开） ----
export KVMEM_STATS=${KVMEM_STATS:-core}
export KVMEM_FORCE_EAGER=${KVMEM_FORCE_EAGER:-1}
export VLLM_USE_V2_MODEL_RUNNER=${VLLM_USE_V2_MODEL_RUNNER:-1}
# 复用：压缩模式会强制关闭 vLLM 前缀缓存（块哈希按逻辑 token、KV 已压缩），所以用 KVMem
# 自己的会话复用：请求结束把 KV+状态停放到 H 池，下一轮命中同一前缀时直接接上。
export KVMEM_SESSION_REUSE=${KVMEM_SESSION_REUSE:-1}

# 超池：允许逻辑 prompt 超过物理池 —— prefill 期压缩 + prompt 末整体重算
export OVER_POOL=${OVER_POOL:-1}
if [ "$OVER_POOL" = "1" ]; then
  export KVMEM_RESIDENT_PREFILL_COMPACT=${KVMEM_RESIDENT_PREFILL_COMPACT:-1}
  export KVMEM_SKIP_KV_CHECK=${KVMEM_SKIP_KV_CHECK:-1}
  export KVMEM_COMPACT_REBUILD=${KVMEM_COMPACT_REBUILD:-1}
  export KVMEM_PROMPT_END_RESELECT=${KVMEM_PROMPT_END_RESELECT:-1}
  [ -n "${MAX_MODEL_LEN:-}" ] && export KVMEM_MAX_MODEL_LEN="$MAX_MODEL_LEN"
fi

# ---- 引擎与入口（仓库内固定并打过补丁的 checkout） ----
export ADAPTER=${ADAPTER:-$HERE}
export VLLM_DIR=${VLLM_DIR:-$HERE/engines/vllm}
export VLLM_ROOT=${VLLM_ROOT:-$VLLM_DIR}
export VLLM_PY=${VLLM_PY:-$VLLM_DIR/.venv/bin/python}
export LOG=${LOG:-$HERE/run-logs/kvmem-vllm.log}
export KV_PIDFILE=${KV_PIDFILE:-$HERE/run-logs/kvmem-vllm.pid}
PORT=${PORT:-18200}

[ -d "$VLLM_DIR/vllm" ] || { echo "未找到引擎: $VLLM_DIR（先跑 install.sh / prepare-backends）" >&2; exit 1; }
[ -x "$VLLM_PY" ]       || { echo "未找到引擎 venv: $VLLM_PY" >&2; exit 1; }
[ -f "$MODEL" ] || [ -d "$MODEL" ] || { echo "未找到模型: $MODEL" >&2; exit 1; }

# 启动器的内置短 smoke 在自动窗口 + 大 sink 的配置下会返回空内容并自清理；
# 跳过它，改由本脚本在就绪后自己探 /health。
export SKIP_STARTUP_SMOKE=${SKIP_STARTUP_SMOKE:-1}

# 前台模式：worker 起完引擎就退出，引擎自己 setsid 常驻（systemd 用 PIDFile 跟踪）。
# 不能用默认的非前台模式：那段逻辑会先杀掉所有 VLLM 进程，systemd 下会连环重启。
export FOREGROUND=${FOREGROUND:-1}

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
