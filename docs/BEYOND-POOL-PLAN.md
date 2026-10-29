# 超池（prompt > 物理池）现状与后续

**这份是压缩版**：只留结论、证据与下一步。60 轮的逐轮记录在
[`archive/BEYOND-POOL-PLAN-log.md`](archive/BEYOND-POOL-PLAN-log.md)（要查"当时为什么这么改"看那里）。

## 1. 能用的配置（回归基线）

| 场景 | 关键参数 | 实测 |
|---|---|---|
| 池内、答案正确 | `GPU_UTIL=0.95 B=200000 INREQUEST_COMPACTION=0` | 196,673 token prompt → **`ZX-4417-QJ7`** |
| 超池、通道跑通 | `GPU_UTIL=0.65`(池 122,816) `B=36864` `KVMEM_ENTRY_RESIDENT_PREFILL=1` `KVMEM_SKIP_KV_CHECK=1` `KVMEM_MAX_MODEL_LEN=200000` | 152,820 token prompt → **HTTP 200**（9~25 次压缩） |
| 超池、少压缩 | 同上 + `R=48000` | 压缩 25 → **9 次**，但答案仍乱码 |

超池请求的答案随压缩次数退化：`0 次=正确` / `1 次=ZXZXZX-4417-QJ7` / `9 次=乱码` / `25 次=只输出 0`。

## 2. 超池的因果链（每条都已修或已定性）

1. **入口裁剪**：`KVMEM_ENTRY_RESIDENT_PREFILL=1` 必须设，否则入口在 API server 里按文本启发式裁 prompt
   （CPU 空转、GPU 0%），引擎根本看不到完整 prompt。
2. **启动不变式**：vLLM 强制 `max_model_len ≤ 池` → `KVMEM_SKIP_KV_CHECK=1` 关掉
   （`kv_cache_utils._check_enough_kv_cache_memory` 的内层，两处调用都覆盖）。
3. **准入闸门**：`scheduler_reserve_full_isl=True` 要求整段 prompt 放进池 → KVMem 模式下强制 `False`
   （必须写在 `__init__` 那条赋值**之后**，否则被覆盖）。
4. **负偏移越界**：位置偏移必须取引擎的 `num_logical - num_computed`（用运行时账本会变负 → 槽位反向偏移）。
5. **行/游标**：块表行按**引擎写入游标**镜像；写入游标在压缩后重置；行长度按常驻口径算。
6. **块回收**：`KVMEM_BLOCK_SIZE=16` 让引擎块==KVMem 页，否则 1616-token 块永远整块不释放 → `assert block`。
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

## 4. 整体重算（reconstruction）—— 正在做的事

目标：prompt 结束时把**选中的 36,864 token 按压缩顺序整体重新 prefill 一遍**（位置 0..N-1 重烘），
让 KV 与循环状态都与最终工作集一致。

已就位：

| 部分 | 位置 |
|---|---|
| 引擎侧重建：按 `pages` 重建 `_all_token_ids`、清零两本账、清 drafts/placeholders、标记 `kvmem_rebuilt`（只做一次） | `plugin/vllm_kvmem_plugin.py::_compact(rebuild=True)` |
| 运行时侧：重建成功后两本账归零、`row=选中页(位置序)`、`prefill_done=False`、`spare=None` | `src/kvmem_sparse_runtime.py`（`_reselect` 内 `reconstruction_started`） |
| 循环状态清零 | `runtime.reset_state` |
| 重建请求禁止投机（否则 `assert num_scheduled >= num_logits`） | `engines/vllm .../scheduler.py::update_draft_token_ids` |

待办（按顺序）：

1. **确认开关真的生效**：启动后用 console 读 `runtime.compact_rebuild`（两次运行传了同样的 env，
   一次 `compact_rebuilt=1`、一次 0，说明有进程读不到；已加 worker env 兜底但未验证）。
2. **核对重建输入**：`_compact` 用 `req._all_token_ids[p*page_tokens:(p+1)*page_tokens]` 取 token，
   而页是**压缩后**的物理页 —— 重建时必须按"页 → 原始逻辑位置"的真映射取 token，否则拿去重算的是错的内容。
   这是最容易出错的一环，先在 47K（1 次压缩）上验证答案是否回到 `ZX-4417-QJ7`。
3. 通过后上 152K 超池。

## 5. 复现命令

```bash
# 通道（超池跑通）
KVMEM_ENTRY_RESIDENT_PREFILL=1 GPU_UTIL=0.65 KVMEM_SKIP_KV_CHECK=1 KVMEM_MAX_MODEL_LEN=200000 \
B=36864 R=48000 INREQUEST_COMPACTION=1 KVMEM_RESIDENT_PREFILL_COMPACT=1 \
MODEL=/home/srj/models/Coder390/NVFP4/W4A16-fp8mtp SERVED_NAME=Qwen3.5-Coder-NVFP4-MTP \
PORT=18200 ./deploy/run-kvmem-vllm.sh

# 整体重算（实验）
KVMEM_COMPACT_REBUILD=1 KVMEM_PROMPT_END_RESELECT=1 ...   # 其余同上；先确认
                                                          # console: runtime.compact_rebuild
```

调试一律用断点/console/热重载（见 `AGENTS.md`），不要加打印重启。
