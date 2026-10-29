# 超池（prompt > 物理池）现状与后续

**2026-10-29 新验收（实验，不代表通用支持）**：TP2、MTP4、同步 V2、单文本请求，
`GPU_UTIL=0.65` 时物理 KV 池 122,816 token，原始 prompt 196,598 token、needle 居中
（约第 98,266 token）。27 次 prefill 压缩后最终重算 33,654 token，HTTP 200，
`usage.prompt_tokens=196598`，回答 `ZX-4417-QJ7`；
`/tmp/kvmem-reconstruction-offset.metadata.json` 和 probe kind
`compact_applied` / `compact_rebuild_queued` / `reconstruction_started`（TP0、TP1）为证据。
**限制**：重建前尾页 `needle=false`、`kv_len_clamped` 仍在；重建后的请求禁用投机，
会话复用、多模态、并发、不同题型未验收。

**这份是压缩版**：只留结论、证据与下一步。60 轮的逐轮记录在
[`archive/BEYOND-POOL-PLAN-log.md`](archive/BEYOND-POOL-PLAN-log.md)（要查"当时为什么这么改"看那里）。

## 1. 能用的配置（回归基线）

| 场景 | 关键参数 | 实测 |
|---|---|---|
| 池内、答案正确 | `GPU_UTIL=0.95 B=200000 INREQUEST_COMPACTION=0` | 196,673 token prompt → **`ZX-4417-QJ7`** |
| 超池、整体重算正确（实验） | `GPU_UTIL=0.65` `B=36864` `R=48000` `KVMEM_ENTRY_RESIDENT_PREFILL=1` `KVMEM_RESIDENT_PREFILL_COMPACT=1` `KVMEM_SKIP_KV_CHECK=1` `KVMEM_MAX_MODEL_LEN=200000` `KVMEM_COMPACT_REBUILD=1` `KVMEM_PROMPT_END_RESELECT=1`，同步 TP2/MTP4 | 196,598 > 122,816；27 次压缩，重算 33,654，**`ZX-4417-QJ7`** |
| 超池、通道跑通（旧基线） | `GPU_UTIL=0.65`(池 122,816) `B=36864` `KVMEM_ENTRY_RESIDENT_PREFILL=1` `KVMEM_SKIP_KV_CHECK=1` `KVMEM_MAX_MODEL_LEN=200000` | 152,820 token prompt → **HTTP 200**（9~25 次压缩） |
| 超池、少压缩 | 同上 + `R=48000` | 压缩 25 → **9 次**，但答案仍乱码 |

旧版**未重建**的超池请求答案随压缩次数退化：`0 次=正确` / `1 次=ZXZXZX-4417-QJ7` / `9 次=乱码` / `25 次=只输出 0`。

## 2. 超池的因果链（每条都已修或已定性）

1. **入口裁剪**：`KVMEM_ENTRY_RESIDENT_PREFILL=1` 必须设，否则入口在 API server 里按文本启发式裁 prompt
   （CPU 空转、GPU 0%），引擎根本看不到完整 prompt。
2. **启动不变式**：vLLM 强制 `max_model_len ≤ 池` → `KVMEM_SKIP_KV_CHECK=1` 关掉
   （`kv_cache_utils._check_enough_kv_cache_memory` 的内层，两处调用都覆盖）。
3. **准入闸门**：`scheduler_reserve_full_isl=True` 要求整段 prompt 放进池 → KVMem 模式下强制 `False`
   （必须写在 `__init__` 那条赋值**之后**，否则被覆盖）。
4. **写地址偏移**：取 scheduler 输出的 served `num_logical` − resident `num_computed`，不能用 V2
   的 CPU `num_computed_tokens_np`（它为 sampler 镜像逻辑长度），否则压缩后 offset 错误。
5. **行/游标**：块表行按**引擎写入游标**镜像；写入游标在压缩后重置；行长度按常驻口径算。
6. **块回收**：设 `KVMEM_BLOCK_SIZE=16` 请求小块；本混合模型实际仍对齐成 1,616-token
   attention 块（101 个 16-token 页）。压缩只按引擎持有的完整原生块重排、释放，不能用逻辑页号释放。
7. **压缩静默失效**：`reselect` 里 `raw` 未初始化 → `UnboundLocalError` 被吞掉，压缩 0 次（已修，0→32 次）。
8. **Mamba 状态块**：`mamba_block_size` 必须等于 `max_model_len`（每序列一份状态），用小值会让分配器索要
   `cdiv(seq,1616)` 个状态块 → `exceeds row capacity (10 > 5)`。解耦已被实测否决。

## 3. 超池正确性的两个剩余障碍

### 3.1 保留页的 KV 带"被淘汰历史"的近似

- needle **能选中**（page-softmax 排名第一，5.32 vs 次名 1.77；`retrieval_check.mid_selected=True`）；
- 尾部重放（QueryReplay，8K）能重算尾窗，但其余 ~28K 保留页仍是旧历史下算出的；
- 结论：必须**按压缩顺序整体重算**（下面第 4 节），调参无效。

### 3.2 块池上限 ≈ 90K，限制 B 的上限

断点实测（`GPU_UTIL=0.65`）：池 76 块，三个 Mamba 组每请求固定占 15 块（1 状态 + 4 投机），
注意力只剩 ~58 块 ≈ 93K token，再减一个 chunk → **真实可常驻 ≈ 90K**。
所以 B=90K/100K/110K 都在 `seq_len=86,016`、`free_blocks=9` 停住（不是 watermark 的问题，是 Mamba 组
无法在只剩 9 块时再分配）。

## 4. 整体重算（reconstruction）—— 实验路径已跑通

完整 prompt 进入引擎并按 B 压缩；最后一个 prefill forward 结束后，TP rank 共享
核心 page-softmax 分数的均值，选择原始**逻辑页**，从原始 prompt token 列表取出
33,654 token。EngineCore 在旧采样对外发布前将请求释放并重新入队；V2 把选中 token
重建为全新输入，两个 worker 各自清零自己持有的 Mamba 状态与投机 scratch。
重建后的请求不再投机。普通 prefill 压缩不跨 TP 同步评分。

| 部分 | 位置 |
|---|---|
| 原始逻辑页取词、物理 attention 块验主/回收 | `src/kvmem_reconstruction.py`、`plugin/vllm_kvmem_plugin.py` |
| 引擎安全边界：旧采样不发布、完整块释放、fresh admission | `engines/vllm/vllm/v1/core/sched/scheduler.py` |
| 两 rank worker 的 token 表、Mamba 状态新建 | `engines/vllm/vllm/v1/worker/gpu/model_runner.py` |
| prompt-end 跨 TP 评分均值（只在同步重算） | `src/kvmem_tp_scores.py` |
| 写地址用 scheduler served-resident，块表在 staged writes 后镜像 | `src/kvmem_position_books.py`、`plugin/vllm_kvmem_plugin.py` |

**未解决**：压缩期间的尾页 `needle=false` 和 `kv_len_clamped`；因此最终重算必须丢弃
旧 KV。暂不支持并发、多模态、跨请求复用；本次只证明一个 TP2/MTP4 超池文本请求的答案。

## 5. 复现命令

```bash
# 独立测试端口，确认无请求后再启动；只改本仓库与其 engines/vllm
FOREGROUND=1 SKIP_STARTUP_SMOKE=1 KVMEM_ENTRY_RESIDENT_PREFILL=1 GPU_UTIL=0.65 \
KVMEM_SKIP_KV_CHECK=1 KVMEM_MAX_MODEL_LEN=200000 B=36864 R=48000 \
INREQUEST_COMPACTION=1 KVMEM_RESIDENT_PREFILL_COMPACT=1 KVMEM_BLOCK_SIZE=16 \
KVMEM_COMPACT_REBUILD=1 KVMEM_PROMPT_END_RESELECT=1 KVMEM_STATS=core \
KVMEM_FORCE_EAGER=1 VLLM_USE_V2_MODEL_RUNNER=1 \
MODEL=/home/srj/models/Coder390/NVFP4/W4A16-fp8mtp \
SERVED_NAME=Qwen3.5-Coder-NVFP4-MTP ADAPTER=$PWD VLLM_DIR=$PWD/engines/vllm \
RUNTIME_ROOT=$PWD/engines/vllm VLLM_PY=$PWD/engines/vllm/.venv/bin/python \
PORT=18200 ./deploy/run-kvmem-vllm.sh

PYTHONPATH=$PWD/engines/vllm engines/vllm/.venv/bin/python \
  scripts/reconstruction_needle.py \
  --tokenizer /home/srj/models/Coder390/NVFP4/W4A16-fp8mtp \
  --pool-tokens 122816 --output /tmp/kvmem-reconstruction-offset
```

`SKIP_STARTUP_SMOKE=1` 只用于该实验：启动器的短 smoke 曾返回空内容导致自清理，
正式验收以脚本打印的实际 `usage.prompt_tokens`、答案和探针为准。调试遵守 `AGENTS.md`。
