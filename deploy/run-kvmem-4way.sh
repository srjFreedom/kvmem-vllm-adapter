#!/bin/bash
# Coder390 FP8 + MTP 已量化成 FP8(fp8mtp 包) + MTP K=4(投机) + fp8 KV + 图像 + 工具 + 1x200K(204800)
#
# 2026-10-17 新建(用户定档 0.97 / 200K / 测完停留不回退): run-coder390-fp8-mtp4-fix-1x180K 的 fp8mtp 变体。
#   动机: BF16 MTP 档降 util 就装不下大上下文(0.96 时 KV 3.38GiB < 184320 需 3.48GiB);
#   换 fp8mtp 省下的 ~0.19 GiB/卡 用来换更大的上下文和/或更低的 util。
#
# ⚠️ 先纠正一个易误读的数: fp8mtp 省的**不是** 1 GiB/卡, 只有 ~0.19 GiB/卡。
#   232K 档(text-only)看着少 0.7~0.94 GiB, 主要差在视觉塔: vision-bf16.safetensors = 0.86 GiB,
#   text-only 档根本不加载它。实际账: MTP 模块 0.791 -> 0.396 GiB, 共省 ~0.4 GiB / 每卡 ~0.19 GiB。
#   另: 裸缓冲垫 = 21.48 x (1 - util), 与权重无关; 省权重只换来"更大 KV 池"。
#
# 与 run-coder390-fp8-mtp4-fix-1x180K-img-tools.sh 的差异(只三处):
#   1) --model-dir  .../FP8        -> .../FP8-fp8mtp
#      (符号链接农场: config/chat_template/tokenizer/text-*/vision-*/preprocessor 全软链回 FP8,
#       唯一实体差异是 mtp-fp8.safetensors 405MB 替换 mtp-bf16.safetensors 810MB。原 FP8 目录零改动)
#   2) HF_OVERRIDES  .../coder390-fp8-mtp-exclude.json -> .../coder390-fp8-mtp-noexclude.json
#      (293 -> 259 条, 去掉 34 条 mtp 排除项。必须: fp8mtp 的 MTP 权重是 FP8, 要让 MTP 按 FP8 构建;
#       反之用 exclude.json 会强制 MTP 按 BF16 构建 -> 与 FP8 权重错配 -> 静默错装/命中率为 0)
#   3) MAX_MODEL_LEN 184320 -> 204800, GPU_UTIL 0.97 (用户定档)
#
# 实测口径校准(每卡 21.48 GiB, fp8 KV ~20,297 B/token, 检查还会再扣每个 KV group 的 null block ~0.02GiB):
#   184320 需 3.74GB(=3.48GiB) | 204800 需 4.16GB(=3.87GiB)
#   BF16 MTP 实测: 0.96 -> KV 3.38GiB(起不来) | 0.97 -> 3.59GiB -> 189,863 tok(1.03x) | 0.98 -> 3.81GiB(1.09x)
#   fp8mtp 每卡轻 ~0.19GiB -> 同样 util 下 KV 多 ~0.19GiB。
#   ⚠️ 预判(待实测): 0.97 + fp8mtp 约 3.71~3.78GiB, 对 204800 需 3.87GiB, 可能差 0.1~0.16GiB 而起不来。
#   若日志报 "To serve at least one request ... (X GiB KV cache is needed ... estimated maximum model length is N)":
#   ✅ 2026-10-17 实测(用户定档 0.97/200K): 0.97 + 204800 **起不来**, 差 0.06 GiB ——
#     run-logs/vllm-Coder390-FP8-20261017-105926.log:
#       ValueError: max seq len (204800), (3.80 GiB KV cache is needed, available 3.74 GiB),
#       estimated maximum model length is 200384
#     => fp8mtp 的真实收益得到确认: 同 util 0.97 下上限从 BF16 MTP 的 ~189,863 涨到 200,384
#        (+10.5K token ≈ 0.19 GiB/卡), 但 204800 还差 4,416 token。
#   最终定 0.98 (保 200K): 预计 KV ~3.95GiB > 需 3.80GiB, 约 1.04x, 垫子回 0.43GiB。
#   备选(一条改动就能换): 保 0.97 垫子 0.64GiB -> MAX_MODEL_LEN 降 196608(192K, 上限 200384 留 3.8K 余量)
#     或 194560(190K, 留 5.8K); 不建议 199680(195K), 只离上限 704 token, 抖一点就起不来。
#   ⚠️ 与 232K 档口径不同(那是 text-only 不加载视觉塔, 0.96 就有 4.47GiB), 不要照搬。
#
# 两个必须点(缺一即崩或降级, 沿用 FP8 档的教训):
#  1) QUANTIZATION 必须清空: fp8mtp 的 config.json 是 fp8 系, profile 里的 compressed-tensors 会传错。
#  2) 内置 chat_template 为空, 必须显式指定 CHAT_TEMPLATE_FILE。
#
# 注意: 这些注释必须留在 exec 上方, 绝不能插进下面的参数表!
#  [教训] 反斜杠续行的参数表中间插 # 注释会静默截断命令, 后面所有 --set 全部失效。
cd "${VLLM_DIR:-$HOME/vLLM-2080Ti-Definitive}"

# 非 YaRN 档下 launcher 原样透传（合并逻辑只在 ENABLE_YARN=1 时执行）
if [ -n "${HF_OVERRIDES_FILE:-}" ]; then export HF_OVERRIDES_JSON="$(cat "$HF_OVERRIDES_FILE")"; fi
export VLLM_LOG_STATS_INTERVAL=3   # 周期速度统计 3s 一输出(nvfp4 同款, 缺了退回默认 10s)

export PATH="$HOME/.local/bin:$PATH"
# 前缀缓存：默认开启（仅 --kvmem-inrequest-compaction 时必须关）
export DISABLE_PREFIX_CACHING=${DISABLE_PREFIX_CACHING:-0}
export KVMEM_PLUGIN_CONFIG=/tmp/kvmem_plugin.json
export KV_CONCURRENCY_EFFECTIVE_LEN="${KV_CONCURRENCY_EFFECTIVE_LEN:-32768}"
exec ./launcher.sh \
  --model-dir "${MODEL:-$HOME/models/Coder390/FP8-fp8mtp}" \
  --profile "${PROFILE:-2x2080Ti/qwen27b/w8a16/mtp4-fp8kv-1x184K-text-image.env}" \
  --mode fast --gpu-devices 0,1 --tp-size 2 --pp-size 1 \
  --set MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768} \
  --set GPU_UTIL=${GPU_UTIL:-0.97} \
  --set MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-4096} \
  --set MAX_NUM_SEQS=${MAX_NUM_SEQS:-4} \
  --set SERVED_NAME=${SERVED_NAME:-KVMEM-SPARSE} \
  --set CHAT_TEMPLATE_FILE="${CHAT_TEMPLATE_FILE:-$HOME/models/Coder390/FP8/chat_template.jinja}" \
  --set MESSAGE_TYPE=${MESSAGE_TYPE:-text+image} \
  --set MM_IMAGE_LIMIT=${MM_IMAGE_LIMIT:-64} \
  --set SPECULATIVE_TOKENS=${SPECULATIVE_TOKENS:-4} \
  --set SERVICE_SCOPE=${SERVICE_SCOPE:-lan} \
  --set DISABLE_LOG_STATS=0 \
  --set PER_REQUEST_SPEC_DECODE_METRICS=detailed \
  --set ENABLE_AUTO_TOOL_CHOICE=1 \
  --set TOOL_CALL_PARSER=qwen3_xml "$@"
