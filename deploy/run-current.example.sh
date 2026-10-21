#!/usr/bin/env bash
# ============================================================================
#  KVMem + vLLM —— 当前生效配置（改这里，然后 restart 服务即可）
#
#    sudo systemctl restart kvmem-vllm
#    tail -f /home/srj/kvmem.log
#
#  只改「参数区」；下面的实现逻辑在 run-kvmem-vllm.sh 里，一般不用动。
# ============================================================================

# ---------------- 参数区（改这里）--------------------------------------------
B=36864                 # 历史预算：每次发给引擎的历史 token 上限（36K）
R=16384                 # 生成上限：max_tokens 会被夹到 R（16K）
CTX=524288              # kvmem 逻辑上下文上限（512K），与引擎窗口无关
SINK=8192               # 永远保留的开头 token 数
RECENT=2048             # 永远保留的结尾 token 数
SESSIONS=4              # H 池按并发会话数平分的份数
HOST_MIB=49152          # H 层（主机内存）预算 MiB
N=1                     # 并发请求数（默认 1）；引擎窗口 = N x (B + R)，自动算
PORT=18200              # kvmem 官方端口
CAPTURE_TOKENS=512      # 每步为检索捕获的 query token 数（官方 query_max_tokens 默认 512）

# ---- 引擎/硬件（换卡、换机器时需要调）----
GPU_UTIL=0.97           # vLLM 显存占用比例；22GB 卡用 0.97，小卡适当降低，OOM 时先降这个
MAX_BATCHED_TOKENS=4096 # 单步 prefill 的 token 上限；调大可减 chunk 数，但吃显存
SPECULATIVE_TOKENS=4    # MTP 投机 token 数；0 = 关闭投机（显存/速度之间取舍）
MESSAGE_TYPE=text+image # text | text+image，是否启用视觉塔
MM_IMAGE_LIMIT=64       # 每个 prompt 允许的图片数
SERVICE_SCOPE=lan       # lan = 监听 0.0.0.0；local = 只监听 127.0.0.1
THINKING_BUDGET=        # 思考 token 上限；空 = 无上限（默认），填数字才限制推理长度

# ---- 权重（改这里换模型）----
MODEL=/home/srj/models/Coder390/FP8-fp8mtp
# 换模型时通常还要一起改：chat template、launcher profile（量化/KV/上下文档）、SERVED_NAME
CHAT_TEMPLATE_FILE=/home/srj/models/Coder390/FP8/chat_template.jinja
PROFILE=2x2080Ti/qwen27b/w8a16/mtp4-fp8kv-1x184K-text-image.env
SERVED_NAME=Qwen3.8-27B-Coder390-EfficientThink-FP8-MTP

KVMEM_UI_DIR=/home/srj/local-model-console  # 同端口挂载自写 UI（空字符串 = 不挂）

KVMEM_ENTRY_MODE=middleware    # middleware = 标准 Starlette body 改写；hook = 兜底
KVMEM_ENTRY_SELECT=retrieval   # retrieval | recency
DISABLE_PREFIX_CACHING=0       # 1 = 关前缀缓存（仅 in-request compaction 时需要）
# ---------------------------------------------------------------------------

export B R CTX SINK RECENT SESSIONS HOST_MIB N PORT CAPTURE_TOKENS
export GPU_UTIL MAX_BATCHED_TOKENS SPECULATIVE_TOKENS MESSAGE_TYPE MM_IMAGE_LIMIT SERVICE_SCOPE
export MODEL SERVED_NAME CHAT_TEMPLATE_FILE PROFILE
export KVMEM_ENTRY_MODE KVMEM_ENTRY_SELECT DISABLE_PREFIX_CACHING KVMEM_UI_DIR

exec /home/srj/run-kvmem-vllm.sh
