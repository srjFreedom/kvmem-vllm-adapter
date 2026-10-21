#!/usr/bin/env bash
# ============================================================================
#  KVMem + vLLM —— 站点配置模板
#
#  用法：复制成 ~/start-<你的配置名>.sh，改下面的参数，
#        再把 systemd 的 ExecStart 指向它（见 deploy/kvmem-vllm.service）。
#
#    cp deploy/start-example.sh ~/start-kvmem-vllm.sh
#    vi ~/start-kvmem-vllm.sh
#    DRYRUN=1 ~/start-kvmem-vllm.sh        # 只打印参数与命令，不启动
#    sudo systemctl daemon-reload && sudo systemctl restart kvmem-vllm
#
#  写法与 start-kvmem-coder390-q6.sh / start-kvmem-ninfer-22g.sh 一致：
#   export VAR=${VAR:-默认}  → 环境变量可覆盖；本文件只放参数，启动逻辑在 run-current.sh。
# ============================================================================
set -uo pipefail

# ---------------- 参数区（改这里）--------------------------------------------
export B=${B:-36864}                              # 历史预算（36K）
export R=${R:-16384}                              # 生成上限（16K）
export CTX=${CTX:-524288}                         # kvmem 逻辑上下文上限
export SINK=${SINK:-8192}
export RECENT=${RECENT:-2048}
export SESSIONS=${SESSIONS:-4}
export HOST_MIB=${HOST_MIB:-49152}
export N=${N:-1}                                  # 并发数；窗口 = N x (B + R)
export PORT=${PORT:-18200}
export CAPTURE_TOKENS=${CAPTURE_TOKENS:-512}

# ---- 引擎 / 硬件 ----
export GPU_UTIL=${GPU_UTIL:-0.97}                 # 显存占用比例，OOM 先降这个
export MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-4096}
export SPECULATIVE_TOKENS=${SPECULATIVE_TOKENS:-4}
export NO_ASYNC_SCHEDULING=${NO_ASYNC_SCHEDULING:-1}  # 1=同步（N>1 默认）；入口选片下可设 0（异步），实测同速
export MESSAGE_TYPE=${MESSAGE_TYPE:-text+image}   # text | text+image
export MM_IMAGE_LIMIT=${MM_IMAGE_LIMIT:-64}
export SERVICE_SCOPE=${SERVICE_SCOPE:-lan}        # lan | local
export THINKING_BUDGET=${THINKING_BUDGET:-}       # 空 = 思考 token 无上限

# ---- 权重（任意 vLLM 支持的模型；换模型时这几项要一起改）----
export MODEL=${MODEL:-/home/srj/models/Coder390/FP8-fp8mtp}
export CHAT_TEMPLATE_FILE=${CHAT_TEMPLATE_FILE:-/home/srj/models/Coder390/FP8/chat_template.jinja}
# PROFILE = 引擎档位（.env：量化/KV dtype/上下文/投机档），最终被 vLLM fork 的
# launcher.sh 消费。worker.py 自己不解析它，但会 exec run-kvmem-sparse.sh(N=1)
# 或 run-kvmem-4way.sh(N>1)，两者把 profile 喂给 launcher.sh。
#   N=1：sparse 读 ${PROFILE:-…}，改这个环境变量即生效。
#   N>1：4way 把同一路径写死，改 PROFILE 无效，要直接改 run-kvmem-4way.sh。
export PROFILE=${PROFILE:-2x2080Ti/qwen27b/w8a16/mtp4-fp8kv-1x184K-text-image.env}
export SERVED_NAME=${SERVED_NAME:-Qwen3.8-27B-Coder390-EfficientThink-FP8-MTP}

# ---- 入口策略 ----
export KVMEM_ENTRY_MODE=${KVMEM_ENTRY_MODE:-middleware}     # middleware | hook
export KVMEM_ENTRY_SELECT=${KVMEM_ENTRY_SELECT:-retrieval}  # retrieval | recency
export DISABLE_PREFIX_CACHING=${DISABLE_PREFIX_CACHING:-0}
export KVMEM_UI_DIR=${KVMEM_UI_DIR:-/home/srj/local-model-console}
# ---------------------------------------------------------------------------

if [ "${DRYRUN:-0}" = "1" ]; then
  echo "DRYRUN: 以下参数将交给 run-current.sh"
  env | grep -E '^(B|R|CTX|SINK|RECENT|SESSIONS|HOST_MIB|N|PORT|CAPTURE_TOKENS|GPU_UTIL|MAX_BATCHED_TOKENS|SPECULATIVE_TOKENS|MESSAGE_TYPE|MM_IMAGE_LIMIT|SERVICE_SCOPE|THINKING_BUDGET|MODEL|CHAT_TEMPLATE_FILE|PROFILE|SERVED_NAME|KVMEM_[A-Z_]*|DISABLE_PREFIX_CACHING)=' | sort
  exit 0
fi

exec /home/srj/run-current.sh
