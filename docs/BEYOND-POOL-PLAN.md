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

## 2026-10-29 第二轮探测结果（58K prompt，B=36864，允许 prefill 期压缩）

压缩不再崩引擎（纯拷贝 + 位置拆分生效），**重复压缩账本自洽**：

```
compact_applied: 49152 -> 36864 (offset 12288)
compact_applied: 46371 -> 36864
compact_applied: 36865 -> 36864
pages=2304 x 16 token = 36864 = visible_tokens   ✔ 三本账一致
```

但答案是截断的 `ZX`，原因在**选择质量**，不在账本：

```
reselect: prefill_pressure=true, selected=2304 pages, mandatory=640
retrieval_check: mid_block=458, mid_selected=false, end_selected=true
scores: 来自旧索引（core 分数此刻还不可用：prefill 早期没有 query）
```

结论（与上游设计一致）：**prefill 期只能做 sink+newest 的压力选择**，因为问题还没进窗口，中间段没有 query 可比。needle 落在被丢弃的区间，所以答案被截断。上游对此的做法是 prefill 后做一次 **QueryReplay**（把最后 N 个 token 当作 query 重放一遍去给中间段打分），再装 B 视图。

所以超池路径的完整顺序应当是：

1. prefill 分 chunk；每 chunk 的 KV 立即落 RAM；
2. prefill 期 resident 只保留 sink + 最新（pressure）——**这一步的近似是设计内允许的**；
3. prefill 结束 → **QueryReplay**：用 query 窗口给中段打分（page-softmax）→ 装 B 视图 → 解码；
4. 生成超出 R → 追加 RAM → 重新检索。

下一步实现 QueryReplay（第 3 步），它是"中间内容不丢"的关键。

## QueryReplay 的实现钩子（下一轮直接照这个改）

目标：prefill 结束后，用 query 窗口给中段打分并装 B 视图，然后把**最后 N 个 token 重算一遍**，让
采样点看到完整的 B（而不是只看过被削掉中段的近似历史）。

现有可复用的机制（已读源码确认）：

| 需要 | 位置 | 说明 |
|---|---|---|
| 逻辑指针 `num_logical_tokens` | `vllm/v1/core/sched/scheduler.py`（调度基准、ctx 上限都用它） | 由 `num_computed + num_kvmem_offset` 推出，重放时要把两者一起回退 |
| 常住指针 `num_computed_tokens` | 同上 + `_compact`（plugin:1212）改写 | 压缩已在回退它，重放只是回退更多 |
| 三本账通道 | `src/kvmem_block_channel.py` 的 `compact` 消息 | 新增一种 `replay` 消息：`(req, visible, replay_tokens)` |
| 打分器 | `kvmem_recall.KvmemRecall`（page-softmax，已验证） | prefill 结束即可用（query=最后 512 token） |

步骤：

1. `compact(replay_tokens=N)`：EngineCore 侧把 `num_computed_tokens = visible - N`，
   `num_kvmem_offset = num_logical - num_computed`，并把 `num_logical_tokens` 也回退 N；
2. scheduler 下一轮会把这 N 个 token 当成新的 prefill chunk 调度（token ids 从逻辑列表取），
   写入位置由 `positions - offset` 算出，仍落在同一批行槽位；
3. 这 N 个 token 的前向此时看到的是**已装好的 B 视图**（sink+中段+recent），KV 与 GDN 状态被刷新；
4. 之后正常解码；R 用满后再走一次同样的流程。

注意：重放只修复"采样点之前的 N 个 token"，之前的近似误差仍留在更早的 KV 里——这是上游
`QueryReplay` 的语义，不是缺陷；要彻底无近似必须让 prefill 全程 resident（即窗口 ≥ prompt）。

## QueryReplay 第一版实测（58K prompt，B=36864）

链路已通：`query_replay` → `replay_applied`（n=2048，computed 36864→34816，logical 58660→56612），
两个 TP rank 重复下发已用 (req, visible) 去重。

结果：答案从完全截断（`ZX`）变成 **`ZX_tokens =!.`（找到了 ZX，但后续乱码）**。

原因（混合模型的根本限制）：**GDN 循环状态不能被"重放"**。重算这 2048 个 token 会再次把它们
施加到循环状态上（同一批 token 被计两次），而状态里原本已经包含它们。

正确做法（上游 `CheckpointPurpose::QueryReplay` + StateImage 的语义）：
1. 在 prefill 经过 `visible - N` 这个边界时，**快照一次循环状态**（runtime 已有
   `capture_state(req)` / `restore_state(req, snap)` 钩子，是给 session park 用的）；
2. 重放前先 `restore_state` 回到该边界；
3. 再回退两本账、重算 N 个 token —— 此时状态从头正确重算，KV 也在新视图下刷新。

没有第 1、2 步，重放只会把循环状态算两遍。这是下一步（第 3 轮）的工作。

## QueryReplay 第 3 次迭代：快照+恢复已生效，但采样点仍然错位（时间盒到这里）

本轮已做：
- 循环状态钩子改为**有循环组就接线**（原来只在 session reuse 分支里，导致 replay 静默把 GDN 状态算两遍）；
- prefill 最后一个 chunk 之前**快照一次**，记录 `replay_at`；replay 长度 = `visible - replay_at`（两 rank 一致）；
- 重放前 `restore_state`，实测日志 `replay_state_restored` 出现。

结果仍然不收敛：`ZX_tokens =`（找到 ZX 但随后像在续写源码）。判断：
**chunk 粒度的状态快照不足以支撑精确重放**——快照点与理想边界（visible-N）相差最多一个 chunk，
而引擎的重放在此之后按"少算 N 个 token"重排，位置/状态/KV 三者对不齐。

需要的能力（记录备查）：按**任意 token 位置**切分 prefill 并快照状态（上游靠 prefix checkpoint 做到），
即 `prefix_checkpoint.hpp` 里 `CheckpointPurpose::QueryReplay` 那一套。这属于 P0.5 级工作量。

因此本轮决定：**把 QueryReplay 记为"部分实现、未收敛"**，先推进其余可交付项（decode 期统计与接受窗口、
checkpoint 目录、清理旧打分旁路、文档与交付说明），超池路径继续留在计划里。
