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
B=65536                 # 历史预算：每次发给引擎的历史 token 上限
R=32768                 # 生成上限：max_tokens 会被夹到 R（也是驻留 B+R 的一半）
CTX=524288              # kvmem 逻辑上下文上限（512K），与引擎窗口无关
SINK=8192               # 永远保留的开头 token 数
RECENT=2048             # 永远保留的结尾 token 数
SESSIONS=4              # H 池按并发会话数平分的份数
HOST_MIB=49152          # H 层（主机内存）预算 MiB
N=2                     # 并发请求数；引擎窗口 = N x (B + R)，自动算
PORT=18200              # kvmem 官方端口
CAPTURE_TOKENS=512      # 每步为检索捕获的 query token 数（官方 query_max_tokens 默认 512）

MODEL=/home/srj/models/Coder390/FP8-fp8mtp
SERVED_NAME=Qwen3.8-27B-Coder390-EfficientThink-FP8-MTP

KVMEM_UI_DIR=/home/srj/local-model-console  # 同端口挂载自写 UI（空字符串 = 不挂）

KVMEM_ENTRY_MODE=middleware    # middleware = 标准 Starlette body 改写；hook = 兜底
KVMEM_ENTRY_SELECT=retrieval   # retrieval | recency
DISABLE_PREFIX_CACHING=0       # 1 = 关前缀缓存（仅 in-request compaction 时需要）
# ---------------------------------------------------------------------------

export B R CTX SINK RECENT SESSIONS HOST_MIB N PORT CAPTURE_TOKENS
export MODEL SERVED_NAME
export KVMEM_ENTRY_MODE KVMEM_ENTRY_SELECT DISABLE_PREFIX_CACHING KVMEM_UI_DIR

exec /home/srj/run-kvmem-vllm.sh
