# 超池 prefill：还缺的那一层接口

日期：2026-10-29　状态：阻塞点已定位，方案明确

## 已经做到的

| 能力 | 状态 | 证据 |
|---|---|---|
| mean-K 统计（RawKvStore，64-token 桶，设备侧求和） | ✅ | 53,422/62,347/88,256 token prompt：`frontier` 精确等于 prompt 长度 |
| page-softmax 打分（sm75 CUDA，逐层 ≤64MiB） | ✅ | 828 块、sum=512、dev 1.16 MiB、无 error |
| 位置拆分（RoPE 用真实位置、写地址用压缩位置） | ✅ 代码就位 | `compute_slot_mappings(slot_positions=...)` + `kvmem_rope_offset` |
| 压缩 = 纯拷贝（不再重烘 K） | ✅ 代码就位 | `restore_page` byte-identical |
| 88K prompt 常驻（不压缩）答对 needle | ✅ | 88,256 token，答案 `ZX-4417`，引擎存活 |

## 阻塞点（本机实测）

在 **prefill 还在进行时**压缩块表 → `BatchPrefillWithPagedKVCache` 报 illegal memory access。

```
tvm.error.InternalError: Check failed: (status == cudaSuccess) is false:
  BatchPrefillWithPagedKVCache failed with error an illegal memory access was encountered
```

原因不是打分、不是位置相位、也不是传输出错：**引擎的 token 账本仍然是原始长度**。我们改写行、把 `num_computed_tokens` 拨回去，但 EngineCore 手里的 `_all_token_ids` 仍是「原始 88K 个 token」，于是 Prefill 内核按原始长度要页，行里却没有那么多页 → 越界。

压缩后的 decode 之所以能跑一阵，是因为 decode 每步只有 1 个 token、且我们通过 block channel 同步截断了 `_all_token_ids`（三本账）；prefill 的 chunk 是 4096 个 token，引擎按「已算 4096·k」继续排，账本和行立刻分叉。

## 需要的接口（仓库里已有设计稿：DESIGN-compacted-sequence.md）

核心是让**引擎自己知道序列被压缩了**，而不是只改行：

1. `RequestState.compacted_tokens`（None = 未压缩）
2. `kv_cache_manager.allocate_slots()` / `free()` 以它为块下标基准
3. `gpu/model_runner.prepare_inputs()` 用有效位置算 logits 行与 `num_scheduled_tokens` 的关系（那条 `assert num_scheduled_tokens >= num_logits` 就是在这里炸的）
4. **EngineCore 把 `_all_token_ids` 截成压缩后的 token 列表**（这是关键，现有 channel 只传了页列表和长度，没传「保留哪几段」）
5. block channel 的 compact 消息扩展为 `(req_id, compacted_len, kept_ranges_in_pages, page_list)`

只做 1–3 不够：K（页）对了、位置对了，但**token 账本仍长**，prefill 还会按长账本要页。所以 4 是必须的。

## 顺序

1. channel 消息带上 kept page ranges；EngineCore 据此重建 `_all_token_ids`（sink 段 + 选中中段 + recent 段 + 新 chunk）
2. `RequestState.compacted_tokens` + scheduler/kv_cache_manager/model_runner 三处用它
3. 打开 `KVMEM_RESIDENT_PREFILL_COMPACT=1`，用 200K prompt（> 177K 池）验收：
   - 答案正确（needle 居中）
   - `mid_selected = true`
   - `compact_applied` 三本账不变式成立
   - 第二轮同 prompt 命中 RAM

## 已知可用配置（回归基线）

```
B=36864 或 131072，compaction 关（或 B 足以容纳 prompt）
KVMEM_STATS=core, page-softmax, LRFU 关, MTP 开
→ 88K prompt、needle 居中，答对 ZX-4417，引擎存活
```
