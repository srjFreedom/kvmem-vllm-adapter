#!/bin/bash
# ============================================================================
#  引擎启动器（唯一一份）—— 把 worker.py 算好的参数交给 vLLM fork 的 launcher.sh
#
#  单并发 / 多并发共用这一个脚本：并发数 N 由 worker.py 用 --set MAX_NUM_SEQS 传进来，
#  引擎窗口 N×(B+R) 也用 --set MAX_MODEL_LEN 传进来；这里的默认值只是兜底。
#  要调档位，就在环境变量或 ~/.config/kvmem-vllm.env 里改，不用改本文件。
# ============================================================================
_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# Default to the engine this repository pins and patches (scripts/prepare-backends.py);
# point VLLM_DIR at another checkout only if you maintain one yourself.
cd "${VLLM_DIR:-$_REPO/engines/vllm}" || exit 1

# 非 YaRN 档下 launcher 原样透传（合并逻辑只在 ENABLE_YARN=1 时执行）
if [ -n "${HF_OVERRIDES_FILE:-}" ]; then export HF_OVERRIDES_JSON="$(cat "$HF_OVERRIDES_FILE")"; fi
export VLLM_LOG_STATS_INTERVAL=${VLLM_LOG_STATS_INTERVAL:-3}   # 周期速度统计间隔(s)
export PATH="$HOME/.local/bin:$PATH"
# 前缀缓存：默认开启。仅当启用 --kvmem-inrequest-compaction（会改写块表）时必须关掉。
export DISABLE_PREFIX_CACHING=${DISABLE_PREFIX_CACHING:-0}
export KVMEM_PLUGIN_CONFIG=${KVMEM_PLUGIN_CONFIG:-/tmp/kvmem_plugin.json}
export KV_CONCURRENCY_EFFECTIVE_LEN=${KV_CONCURRENCY_EFFECTIVE_LEN:-32768}
# 思考 token 上限：未设置 = 无上限（默认）。设了才会限制推理长度。
if [ -n "${THINKING_BUDGET:-}" ]; then export REASONING_BUDGET="$THINKING_BUDGET"; fi
# SM75 + MTP + CUDA graph 的 fork 内策略（v0.4.0 生产值，实测组合）：
#   nosync = 投机解码图内无额外流同步（launcher 的 --mode fast 会推导成 safe，那是调试档）；
#   MAMBA_SPEC_FULL_CUDAGRAPH=1 = GDN/Mamba 投机全图快路径（本仓库已验证的解码路径）。
# 显式导出，克隆后无需任何 /etc/default 即可复现生产行为；要改在环境里覆盖即可。
export VLLM_SM75_SPEC_SYNC_MODE=${VLLM_SM75_SPEC_SYNC_MODE:-nosync}
export VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=${VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH:-1}

# GPU_UTIL=0.972：编译图模式的静态缓冲要吃掉约 2.5K token 的池；0.97 时池=198,561 <
# MAX_MODEL_LEN=200,000，200K 请求会 OOM（实测 2026-11-02 夜间服务死亡）。曲线：
# 0.93->152K, 0.95->175K, 0.97->198.6K, 0.972->201.4K tokens。
# 注意：这些注释必须留在 ARGS 上方，绝不能插进参数表中间。
#  [教训] 反斜杠续行的参数表中间插 # 注释会静默截断命令，后面所有 --set 全部失效。
ARGS=(
  --model-dir "${MODEL:?MODEL not set: point it at your weights directory}"
  --profile "${PROFILE:-2x2080Ti/qwen27b/w8a16/mtp4-fp8kv-1x184K-text-image.env}"
  --mode fast --gpu-devices 0,1 --tp-size 2 --pp-size 1
  --set MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
  --set GPU_UTIL=${GPU_UTIL:-0.972}
  --set MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-4096}
  --set MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
  --set SERVED_NAME=${SERVED_NAME:-KVMEM}
  --set MESSAGE_TYPE=${MESSAGE_TYPE:-text+image}
  --set MM_IMAGE_LIMIT=${MM_IMAGE_LIMIT:-64}
  --set SPECULATIVE_TOKENS=${SPECULATIVE_TOKENS:-4}
  --set SERVICE_SCOPE=${SERVICE_SCOPE:-lan}
  --set DISABLE_LOG_STATS=0
  --set PER_REQUEST_SPEC_DECODE_METRICS=detailed
  --set ENABLE_AUTO_TOOL_CHOICE=1
  --set TOOL_CALL_PARSER=qwen3_xml
)
# 只有显式给了模板才透传（模型自带 chat template 时不需要）
[ -n "${CHAT_TEMPLATE_FILE:-}" ] && ARGS+=(--set "CHAT_TEMPLATE_FILE=${CHAT_TEMPLATE_FILE}")
exec ./launcher.sh "${ARGS[@]}" "$@"
