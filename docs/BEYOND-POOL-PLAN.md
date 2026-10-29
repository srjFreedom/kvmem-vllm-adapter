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

## 2026-10-29 第 7 轮：关键进展 + 超池边界确认

### 打通：B 内压缩 + page-softmax 检索能捞回中间内容

修掉两个接线错误后（① 打分管线被嵌在 legacy 分支里，core 分数算完被丢掉；② 绑定要求每层行数严格
等于 rows，核心允许 `layer_rows[l] <= rows`），58K prompt（B=36864，prefill 末压缩）实测：

```
retrieval_check: mid_selected = True
top_blocks: [[600,0.0885],[601,0.0885],[1220,0.0826],...]      ← page-softmax 排序进入选择器
answer: "ZX_tokens\n\nThe code is ZX-4417.\n\nThe code"        ← needle 被捞回来了
```

对照（B=131072、不压缩、88,256 token）：`answer = ZX-4417`，`core_scores` 704 块 / sum=480 /
top=[149,392,393,391,197]。**说明统计→打分→选择→解码整条链在不压缩时完全正确。**

### 仍然阻塞：prefill 进行中的压缩（>池的唯一路径）

111K prompt（< 池 177K，但 > B，因此触发 prefill 期压缩）：
`CUDA error: an illegal memory access`（引擎存活，请求 500）。

与第 3 轮定位一致：prefill 期间引擎的 `_all_token_ids` 仍是原始长度，块表却已被压缩 →
`BatchPrefillWithPagedKVCache` 越界。

### 结论

| 场景 | 状态 |
|---|---|
| prompt ≤ B | ✅ 正确（不压缩，或 prefill 末压缩后检索） |
| B < prompt ≤ 池（常驻 prefill，末压缩） | ✅ 58K 实测：mid_selected=True，答案含 ZX-4417 |
| prompt > 池（需 prefill 期压缩） | ❌ FlashInfer 越界，需 compacted-sequence 接口 |

下一步就是那个接口：channel compact 消息带 kept page ranges → EngineCore 重建 `_all_token_ids` →
`RequestState.compacted_tokens` 三处使用。这是唯一挡住"prompt 不受显存限制"的东西。

## 第 8 轮：行滞后修好，越界仍有第二个来源

修：块表行原来在 `after_step`（模型跑完之后）才扩展，所以每个跨页边界的 prefill chunk 都在
"行比 seq_len 短一截"的状态下执行（实测 `row=4864 pages` 而本步要写到 81920 → 需要 5120）。
现在 `ensure_row_for_chunk()` 在**步进之前**按引擎自己的页映射预留本 chunk 需要的页，实测
`row` 精确长到 `need`、`row_short_pre_step = 0`。

仍未通过：111K prompt（> B，prefill 期压缩）依旧 `illegal memory access`，而此时三条账是：

```
engine num_computed = 81920     （已推进）
runtime tokens/seq_len = 77824  （常驻账）
row = 5120 pages                （覆盖到 81920）
```

即**引擎的 computed 已经跑在常驻账前面**，attention metadata 的长度与行的实际内容仍不一致。
下一步要查：`seq_lens` / `seq_lens_cpu_upper_bound` 与 `num_kvmem_offset` 在这条路径上的更新次序
（压缩发生在 chunk 之间，引擎已经为下一个 chunk 记账）。

## 第 9 轮：prefill 期压缩不再崩溃；剩下的障碍是循环状态

修掉"负偏移"后（offset 必须取**引擎自己**的 `num_logical - num_computed`；用运行时账本会在压缩后
变成负数，把槽位索引整体反向偏移 → 越界），**95,356 token prompt（B=36864，压缩发生在 prefill
期间）第一次返回 HTTP 200，引擎存活**。

但答案是 `ductductduct...`（复读），而选择端是对的：

```
reselect: selected=2304 pages, visible=36860
retrieval_check: top_blocks=[[256,0.3607],[257,0.3607],...]   ← 有排序
compact_applied: 42114 -> 36864, offset 58498
replay_snapshot: at=90112 (logical) vs visible=36864 (resident)   ← 坐标混用，replay 长度无效
```

结论：**prefill 期压缩使 GDN 循环状态带上近似误差，而它是整请求一份、永不重算命中的**。注意力视图
可以靠检索修好，状态不行。这正是上游 `prefix_checkpoint` 里 `CheckpointPurpose::Reconstruction` /
`QueryReplay` 存在的原因：**在 prefill 过程中周期性地保存状态镜像，最后从某个有效检查点重算尾部**。

因此超池的收官路径 = objective ③ 的"checkpoint 目录"：
1. 每次压缩边界存一份**循环状态镜像**（`capture_state` 已有）；
2. prefill 结束时挑一个"视图已正确"的检查点，恢复状态后重算剩余 token（replay）；
3. 之前 `replay` 的坐标混用（logical vs resident）必须先统一到 **resident** 口径。

优先做 (3)+(1)：先把 replay 长度改成 resident 口径并禁用无效 replay，再决定是否需要多检查点。

## 第 11 轮：重建路径跑到了"第三块拼图"上

`KVMEM_COMPACT_REBUILD=1` 已实现并被触发：

```
compact_rebuilt: pages=2304 tokens=36864      ← 引擎的 token 账本被改写成压缩后的序列
```

随后撞上仓库设计稿里早就写明的那个断言：

```
model_runner.py:1360  assert (num_scheduled_tokens_np >= num_logits).all()
```

即 `DESIGN-compacted-sequence.md` 的第三块拼图：**model_runner 必须按压缩后的账本计算 logits 行**。
前两块（channel 带页列表、EngineCore 重建 `_all_token_ids`）本轮已完成，第三块是
`prepare_inputs` 里 `num_scheduled_tokens` / `num_logits` 的口径统一。

在这之上一旦完成，超池路径的最后一块就补齐了：状态从零重算 + token 账本与视图一致。
该开关默认关闭，不影响现有可用路径。

## 第 12 轮：重建路径的收敛条件写清楚了

现象：95K prompt 触发多次 `compact_rebuilt`，随后 `assert num_scheduled_tokens_np >= num_logits`。

已定位的收敛条件（下一次实现按这三条来）：

1. **每个请求只重建一次**，且必须在**所有 rank 的 compact 消息合并之后**再执行——现在两个 TP rank
   各发一次，EngineCore 会各跑一次重建（实测 5 次）；
2. **重建只能在 prefill 边界**（最后一次 chunk 之后），不能在 prefill 进行中，否则引擎正要按旧账本
   排下一个 chunk；
3. **拼图③必须先做**：`prepare_inputs` 里 logits 行数要按压缩后的账本算，否则重建后第一步就断言。

开关 `KVMEM_COMPACT_REBUILD` 默认关闭，现有可用路径不受影响。

## 第 17 轮：行内重复页 id（一个真 BUG）与池容量的更正

**行有效性探针**（新增）在首几个 prefill 就报：

```
len=256, uniq=102, dups=154   ← 同一物理页被多个位置引用
```

来源是我第 9 轮加的 `ensure_row_for_chunk`：它按**页映射（逻辑页）**造行，而行的索引是**常驻 rank**，
两者错位后就出现重复页。重复页本身不一定崩，但会让注意力读到同一页的多份并污染写入。

改为**默认关闭**（`runtime.row_grow_pre_step`），恢复"引擎自己分配 + after-step 镜像"。修后：
`row_invalid = 0`，58K 压缩用例重新 `mid_selected=True`。

**池容量更正**：实测 `num_blocks = 12,726`，×16 = **203,616 token**。所以 201K 的 prompt 其实**仍在池内**，
真正"超池"验收需要 >203K（约 880K 字符以上）。之前按 177K 估算是偏保守的。

201K prompt（7 次压缩）在去掉重复页后仍以 `BatchPrefillWithPagedKVCache` 崩溃，引擎退出 →
下一轮继续追"行内 page id 是否指向已归还的页"。

## 第 22 轮：粒度问题测清楚了 —— 页(16) vs 引擎块(1616)

从运行中的 worker 直接读出（console）：

```
page_tokens            = 16          ← KVMem 页 / 内核块表的一列
view.num_blocks()      = 12726       ← 内核块表的列数（12,726 × 16 = 203,616 token）
cache_config.block_size= 1616        ← 引擎分配器的块
max_model_len          = 177,760
FullAttentionManager 里本请求的块数 = 31   （31 × 1616 ≈ 50K token）
```

也就是说**存在两套完全不同的空间**：

| 空间 | 单位 | 谁在用 |
|---|---|---|
| 内核块表（`runner.block_tables.gpu`） | **16 token/列** | FlashInfer 的 `slot_mapping` / KV 索引 |
| 引擎分配器（`FullAttentionManager.req_to_blocks`） | **1616 token/块** | 调度器的分配/释放/抢占 |

由此得到两条结论：

1. 我上一轮"按页 id 去回收引擎块"的做法**在两个不可比的空间里做减法**——所以 `freed=0`
   （运气好，没有误释放）。这个方向要撤回，不能这样回收。
2. **超池崩溃的根因是这个粒度不匹配**：我们按 16-token 页做压缩选择，而引擎按 1616-token 块
   记账与释放。只要调度器释放/抢占一个 1616-token 块，我们行里保留的 16-token 页就会有一批被
   抽走 → 内核读到已释放显存 → `BatchPrefillWithPagedKVCache: illegal memory access`。
   这也解释了为什么"行长度对、写地址对、id 在池内、无重复"却仍然崩。

修法（下一轮）：**压缩选择必须在引擎块粒度（1616 token）上做**，即选择/淘汰以引擎块为单位
（页只是块内偏移），这样释放一个块绝不会带走仍被行引用的页。这需要把 selector 的块网格从 16
页改成 1616 引擎块（或在两者之间做严格的对齐映射）。

## 第 29~31 轮：超池 prefill 打通，答案正确性卡在"重建会停摆"

### 已打通（有实测）

| 环节 | 证据 |
|---|---|
| 201,915 token prompt（池 ~179,182） | `prompt_tokens=201915`，HTTP 200 |
| prefill 期压缩 | 38 次压缩，`logical/tokens` 涨到 201,915，常驻稳定压回 36,864 |
| 压缩请求不再崩 | MTP 下也能返回 200（压缩请求禁止投机解码） |
| 检索能选中 needle | 147K prompt：needle 块得分 **5.32**，次名 **1.77**，Top-1 命中 |

### 卡点：把"选对的工作集"装上会破坏状态

- 压力选择（prefill 期）丢掉中段 → 答案 `7`（错）；
- prefill 末用 query 重排 → **排名正确**，但装上后答案仍是 `7`（错）；
- 再做"重建"（把压缩后的 36,864 token 用零状态重新 prefill 一遍）：
  - 能触发（`compact_rebuilt`）、能重新 prefill，
  - 但请求**停摆**（GPU 0%、日志 `no sampled token`），25 分钟后才返回，答案仍 `7`。

### 结论（下一阶段从这里继续）

超池的**通道**已经完整：分块 prefill、逐块压缩、地址重映射、位置分离、引擎内 page-softmax 检索、
压缩请求不投机、200K prompt 返回 200。**缺的是"状态一致的重建"**：

1. 重建后引擎进入"只 decode 不采样"的停摆，需要查 `num_output_placeholders` / `num_logits` /
   draft 状态在 token 账本被改写后的组合（本轮已清空 spec/drafts，仍停摆）；
2. 或改走"多检查点"路线：prefill 时每 N 个 token 保存一份循环状态，末尾恢复到某个"视图已正确"
   的检查点后重算尾部（上游 `prefix_checkpoint` 的 Reconstruction 语义）。

两个开关都默认关闭：`KVMEM_PROMPT_END_RESELECT=1`、`KVMEM_COMPACT_REBUILD=1`。

## 第 35 轮：QueryReplay 已实现并验证，得到"超池正确性"的决定性否证

实现并跑通（开关：`KVMEM_PROMPT_END_RESELECT=1 KVMEM_CHECKPOINT_REPLAY=1 KVMEM_REPLAY_WINDOW=8192`）：

```
query_reselect (prompt_complete)  →  选出 needle 块（排名第一，5.41 vs 次名 1.81）
retrieval_check: mid_selected = TRUE        ← 修正了 ±2 块的估算偏差后，确认 needle 在工作集里
tail_replay: n=8192 → replay_applied: computed 36864→28672, logical 回退 8192
http=200，引擎存活，无停摆
answer = "7"                                 ← 仍然错
```

**这三条同时成立，就是这一轮最重要的结论**：

1. 检索选对了（needle 页在工作集内）；
2. 尾部 8,192 token 已按新视图重算；
3. 答案仍错。

所以错误不在"尾部的视图一致性"，而在**保留的中段页本身的 KV 是在中段已被淘汰的状态下算出来的**——
它们带着近似。要修，必须把选中的 36,864 个 token **按压缩后的顺序整体重新 prefill 一遍**（这就是
上游 `Reconstruction`），而这件事必须让引擎的 token 账本与行一致（compacted-sequence 接口）。
token 账本当前一旦被改写，采样簿记就失效（`no sampled token` 停摆）。

结论（对 objective ② 的最终判断）：

| 部分 | 状态 |
|---|---|
| 分块 prefill / 逐块压缩 / 地址重映射 / 位置分离 | ✅ |
| 引擎内 page-softmax 检索 | ✅（needle 排名第一、且确实入选） |
| 200K prompt 不崩、返回 200 | ✅（201,915 token） |
| QueryReplay（尾部重算） | ✅ 机制可用 |
| **超池的答案正确性** | ❌ 需要"按压缩顺序整体重算"，即 compacted-sequence 接口 |

这是架构性结论，不是调参能解决的：混合模型（GDN）里，被保留下来的页的 KV 与循环状态都压缩了
"被淘汰的历史"，近似不可消除，除非能重算——而重算需要序列身份可变。

## 第 36 轮：压缩次数与"可用性"的定量关系

| prompt | 压缩次数 | HTTP | 答案 |
|---|---|---|---|
| 88,256（B=131072，不压缩） | 0 | 200 | `ZX-4417` 干净 |
| 147,367（B=36864，needle 在 1/4） | ~24 | 200 | `7`（错，needle 已入选） |
| **152,820（needle 在末尾，必被 recent 保留）** | ~6 | 200 | `ZX question...`（**乱码**，16 token 全跑满） |
| 236,896（needle 在末尾） | 45 | 500 | prefill 到 236,827 后最后一个 chunk 越界 |

关键：**即使 needle 在末尾（recent 窗口里、每次压缩都保留），压缩过的请求仍然输出乱码**。

这说明近似不是"针丢没丢"的问题，而是**保留页的 KV 与循环状态共同偏离**：
- 保留页的 KV 是在"中段被淘汰"的状态下算出来的；
- 压缩次数越多，偏离越大（1 次≈部分可读，6 次≈乱码，24 次≈完全错）。

与第 35 轮的否证一致：尾部重放（QueryReplay）只能修最后 8K，修不了其余 28K 保留页的近似。
唯一解是**把选中的 36,864 token 按压缩顺序整体重算**（compacted-sequence 接口）。

## 第 42 轮：停摆的确切现场（留给下一阶段的唯一线索）

重建之后引擎停摆，日志里的那一行是全部信息：

```
kvmem: prompt served but no sampled token
  req=... logical=36864 computed=36864 tokens=36865 prompt=36864 scheduled=4096 placeholders=1
```

解读：
- `computed == prompt == logical == 36864` → 重新 prefill 已经完成；
- `tokens = 36865` → `num_tokens` 只比 prompt 多 1（我们留的那个 in-flight placeholder）；
- **`scheduled = 4096`** → 调度器却给这一步排了 4,096 个 token（等于一个满 chunk），
  而请求已经没有更多 token 可算 → worker 不产生 logits → 采样器无 token 可出 → 停摆。

也就是说，问题不在采样器写法，而在**重建之后调度器对"这一步要算多少 token"的判断**：
`is_prefill_chunk` / `num_output_placeholders` / `num_tokens_with_spec` 三者的组合让它以为还要走一个
满 chunk 的 prefill。下一阶段要从这里入手（在 `schedule()` 里为"刚完成重建"的请求打印/校正
`num_new_tokens`）。

## 第 46~47 轮：超池最后一个成因被证实 —— vLLM 的不变式本身

修掉 `raw` 未初始化（第 46 轮）之后，同一配置的压缩次数从 0 变成 32，随后失败点变成了：

```
Block table write for request 0, group 3 exceeds row capacity (6673 > 6464)
```

6,464 页 = 103,424 token = 引擎窗口；而池是 7,676 页 = 122,816 token。也就是说**块表行是按
`max_model_len` 开的，而 `max_model_len` 比池还小**（vLLM 的 auto-fit 会扣掉保留量）。

试着把窗口开到 200,000（`KVMEM_MAX_MODEL_LEN=200000`），vLLM 直接拒绝启动：

```
ValueError: To serve at least one request with the model's max seq len (200000),
            (3.64 GiB KV cache is needed, which is larger than the available KV cache memory (1.97 GiB))
```

**结论（这是超池的最终成因，比"近似"更根本）**：
vLLM 在启动时就强制 `max_model_len ≤ KV 池`。而 KVMem 的前提恰恰相反——**逻辑窗口 > 物理池**。
适配器现在只能在这个不变式之内工作（prompt ≤ 池）。要打破它，必须实现 compacted-sequence 接口
（token 账本 + 位置 + 采样），单纯"请求内压缩"不够——这也是为什么所有超池实验最终都停在
"行宽/窗口"或"改写账本后采样失效"这两处。

## 第 51 轮（续）：用断点定位"窗口>池时请求不进调度"——根因是 Mamba 块随 max_model_len 放大

用调试设施（把 `kvmem_dbg` 接进 EngineCore 的 `Scheduler.schedule`，断点落 `/tmp/kvmem_dbg.bp`），
一次命中就把调度器局部变量全部落盘：

```
# 断点 @ scheduler.py:899 (allocate_slots)
request.num_prompt_tokens   = 245,556
request.num_computed_tokens = 0
num_new_tokens              = 4,096
free_blocks                 = 75
groups = [('MambaManager', 200000, 5), ('MambaManager', 200000, 5),
          ('MambaManager', 200000, 5), ('FullAttentionManager', 1616, 1)]
# 换成小请求（16 token）后：
groups = [('MambaManager', 200000, 5), ('MambaManager', 200000, 5),
          ('MambaManager', 200000, 5), ('FullAttentionManager', 1616, 1)]
free_blocks = 59
```

关键：**Mamba 组的 `block_size = max_model_len`（200,000）**，且每个请求每组已占 **5 个块**。
所以：
- 窗口 200K 时，一个请求的 Mamba 状态块按 200K token 的尺度计，3 组 × 5 块就把 75 块的池吃掉一半；
- 245K 的 prompt 还需要 152 个注意力块（245K/1616），池里根本没有 → `allocate_slots` 永远失败 →
  请求停在 `waiting`（我们关了抢占，所以是"等"而不是"崩"）。

**结论**：`KVMEM_SKIP_KV_CHECK` 只能跳过启动检查；在混合模型（3 个 Mamba 组）里，Mamba 块的尺寸
随 `max_model_len` 放大，因此"窗口 > 池"仍然不可行。要真正超越物理池，必须让 Mamba 状态块与
`max_model_len` 解耦（固定为状态本身的大小），这属于 fork 层的结构性改动。

这也解释了为什么之前所有"窗口>池"的尝试都停在调度：不是注意力 KV 不够，而是 **Mamba 状态块按窗口
放大把池吃掉了**。

## 下一步工程项：解耦 Mamba 状态块与 max_model_len（第 52 轮方案）

### 现状（源码已核对）

- `MambaSpec.block_size` 来自 `vllm_config.cache_config.mamba_block_size`；
- 该字段**未设置时等于 `max_model_len`**（`vllm/config/vllm.py:3167` 用 `!= max_model_len` 判断"是否显式设置"）；
- `--mamba-block-size` 存在（`arg_utils.py:1309`），但 `validate_mamba_block_size()` 规定
  **只有开启 `--enable-prefix-caching` 才能设置**；而本适配器做请求内压缩时前缀缓存必须关闭 →
  于是窗口抬高到 200K 时，Mamba 块也按 200K 计。
- 后果（断点实测，第 51 节）：每个请求每组持 5 个块，`block_size=max_model_len`；
  再加上 `need_mamba_block_aligned_split` 会按 `mamba_block_size` 对齐 chunk，
  窗口 > 池时请求要么分不到块、要么 `num_new_tokens` 被对齐成 0，永久停在 `waiting`。

### 要做的两处改动

1. **放开校验**：`vllm/config/vllm.py::validate_mamba_block_size` 在
   `KVMEM_MAMBA_BLOCK_TOKENS` 显式存在时跳过该 ValueError（附注释说明这是 KVMem 的用法）；
2. **显式给值**：适配器侧（`backends/vllm/worker.py`）在 KVMem 模式下传
   `--mamba-block-size <KVMEM_MAMBA_BLOCK_TOKENS>`（建议与注意力块对齐，如 1616 或 4096），
   让 Mamba 块尺寸不再随 `max_model_len` 变化。

### 验收

- 启动：`GPU_UTIL=0.65`（池 ~122K）、`KVMEM_SKIP_KV_CHECK=1`、`KVMEM_MAX_MODEL_LEN=200000`、
  `KVMEM_MAMBA_BLOCK_TOKENS=1616`；
- 断点 `scheduler.py:899` 应看到 `MambaManager(block_size=1616)`，且 245K prompt 能进入
  `allocate_slots`（而不是停在 waiting）；
- 再跑 200K/245K prompt 看是否正常进入压缩与解码。

## 第 52 轮：Mamba 块解耦被实测否决

改动（`KVMEM_MAMBA_BLOCK_TOKENS=1616`，窗口 200,000）后断点确认块尺寸确实变了：

```
groups = [('MambaManager', 1616, 5), ('MambaManager', 1616, 5), ('MambaManager', 1616, 5),
          ('FullAttentionManager', 1616, 1)]
```

但大 prompt 一发就得到明确的引擎错误：

```
Block table write for request 0, group 0 exceeds row capacity (10 > 5)
```

原因：本 fork 在 `mamba_cache_mode="none"` 下，**每个请求只有一个状态块**，块表行也只按这个尺寸开
（5 列 = 1 状态 + 4 投机槽）。把块尺寸改成 1616 后，分配器按 `cdiv(seq_len, 1616)` 索要状态块
（10 个），直接越过 5 列的容量。

所以 **`mamba_block_size` 必须等于 `max_model_len`**：状态是"每序列一份"，块表按此开。
这不是参数问题，而是这套混合模型状态管理的结构性约束。已回退（`b5c4f34`），并把原因写进提交信息。

**对超池的含义**：窗口 > 池在这套架构里不可行，除非把 Mamba 状态改成"与窗口无关的每序列一份、
块尺寸恒为 1"的布局（fork 级重构：`MambaManager` 的分配数学 + 块表宽度）。

## 第 53 轮：超池配置的两个必要开关 + 新的停点

### 关键发现：`KVMEM_ENTRY_RESIDENT_PREFILL=1` 是必需的

前几轮所有"请求不进调度"的实验里，我都没设这个变量。**不设它，入口会按文本启发式把 prompt 裁到 B**
（在 API server 里做 token 估算，1.3M 字符的 prompt 会长时间占 CPU，GPU 0%），引擎根本看不到完整
prompt。设上之后：

```
warm http=200
196,673-token prompt → compact_applied = 1 → row_synced = 6 → 然后 GPU 掉到 0%（停住）
```

即：**请求这次真的进了引擎，并且发生了 1 次压缩**，然后停住。

### 剩下的停点

停在第 1 次压缩之后。下一步用断点（把 hits 设大，专门抓这个大请求）看：
- 是调度器没再被调用（`waiting` 里卡着），还是
- worker 在等 EngineCore 的 block channel 响应（同步调度 + 压缩的死锁）。

### 已确认的环境组合（超池）

```
KVMEM_ENTRY_RESIDENT_PREFILL=1      # 必需：入口不裁，交给引擎压缩
KVMEM_SKIP_KV_CHECK=1               # 必需：窗口可以大于池
KVMEM_MAX_MODEL_LEN=200000          # 窗口 ≥ prompt
GPU_UTIL=0.65                       # 池 ~122K < prompt（真正的超池）
B=36864 INREQUEST_COMPACTION=1 KVMEM_RESIDENT_PREFILL_COMPACT=1
# 不要设 KVMEM_MAMBA_BLOCK_TOKENS（第 52 轮已证否）
```

## 第 54 轮：测试样本尺寸纠正 + 窗口/池分离

断点读到的真实数字：`/tmp/nv200real.json` 实际是 **369,931 token**（1.36M 字符 ÷ 3.68），
不是我之前估的 196,673。它比窗口 200,000 大 → 请求被卡在 `waiting`（`num_prompt_tokens=369931`,
`max_model_len=200000`），这正是"看起来卡在调度"的真正原因之一。

把窗口抬到 400,000（`KVMEM_SKIP_KV_CHECK=1` 允许）后重跑，请求仍不前进（120 秒内 0 次压缩、GPU 0%），
说明还有第二道门槛，需要下一轮用断点（把 hits 调大或先用 `remove()` 清点）继续定位。

**经验**：以后所有 prompt 的 token 数必须用引擎报的 `prompt_tokens` 为准（字符数不可靠，
同一份文本在不同 tokenizer 下可以差 2 倍），并且测试前先确认 `prompt_tokens < max_model_len`。

## 第 55 轮：把"分不到块"精确到 allocate_slots 的拒绝

断点（`scheduler.py:1371`，等待队列里的分配点）一次命中给出全部数字：

```
num_new_tokens = 4096            # 请求的 chunk 计算正确
new_blocks     = None            # 分配失败
free_blocks    = 75              # 池里还有 75 块
每组需要的块   = Mamba 5 + 5 + 5 + FullAttention 3 = 18   << 75
```

**需要 18 块、池里有 75 块，但 `allocate_slots` 仍然返回 None** → 拒绝来自 `kvcache_manager` 内部的
预留/水位检查，而不是块不够。`kv_cache_manager.py` 里有 6 处 `return None`，本轮只验证到 521/574/580
三处都不触发，下一轮把其余几处全部挂上断点（一次命中即可定位是哪个条件）。

结论：超池请求停在等待队列的**根因是分配器的预留检查**，与压缩、Mamba 块、窗口都不再是同一个层面
的问题——这是一个很小、很具体的检查条件。

## 第 56 轮：超池请求第一次完整跑通（HTTP 200）

**真正的准入闸门是 `scheduler_reserve_full_isl`**（默认 True）：等待中的请求被要求"整段 prompt 必须
放进池子"。KVMem 的前提正相反，所以在 `kvmem_resident` 模式下强制关掉它。

（注意：第一次改的时候插在了 `__init__` 里被后面那行赋值覆盖，断点仍显示 True；改成插在赋值之后才生效。）

实测（GPU_UTIL=0.65 → 池 122,816；窗口 200,000；B=36,864；prompt 152,820 token = **超池 ~24%**）：

```
http            = 200
prompt_tokens   = 152,820
completion      = 2
25 次压缩、80 次行同步
answer          = "0"        ← 错的（近似，见第 42 节）
```

这是**超池请求第一次完整走完 prefill + 压缩 + 解码并返回**。正确答案仍受"保留页近似"限制：
25 次压缩后模型输出退化。

下一步（下轮）：把压缩次数降下来（B 更大 / 压缩间隔更宽），看能否把答案拉回正确。

## 第 57 轮：超池"能跑"与"能答对"的分界（本轮定稿数据）

### 已打通（可复现）

| 配置 | prompt | 池 | 压缩次数 | HTTP | 答案 |
|---|---|---|---|---|---|
| 池内常驻（GPU_UTIL=0.95, B=200000） | 196,673 | ~520K | 0 | 200 | **`ZX-4417-QJ7` ✅** |
| 超池（0.65 → 122,816; B=36,864, R=16K） | 152,820 | 122,816 | 25 | 200 | `0` ❌ |
| 超池（同上, R=48,000 放宽压缩间隔） | 152,820 | 122,816 | 9 | 200 | 乱码 ❌ |

### 结论

- **超池的通道完全打通**：准入（`scheduler_reserve_full_isl` 关闭）、分块 prefill、逐 chunk 压缩、
  地址重映射、行同步、解码返回，全链路可跑；
- **答案正确性取决于压缩次数**：0 次 → 精确；9 次 → 退化；25 次 → 只剩一个 token；
- 放宽压缩间隔（R=48,000）能把 25 次降到 9 次，但仍不足以恢复正确性；
- 根本原因仍是第 42 节的"保留页近似"，需要"按压缩顺序整体重算"（compacted-sequence 接口）才能消除。

### 调参无效的下一个证据

B=100,000（接近池容量）时请求**卡住**（0 次压缩、GPU 从 100% 掉到 0、48 次行同步后停），说明大 B 下压缩
触发与池容量之间还有未处理的条件；B=36,864 是当前唯一稳定的超池配置。

## 第 58 轮：B 接近池容量时会静默停住（新证据）

| B | prompt | 池 | 结果 |
|---|---|---|---|
| 36,864 | 152,820 | 122,816 | ✅ 跑完（25 次压缩，答案错） |
| 110,000 | 131,039 | 122,816 | ❌ 停在 resident ≈ 86K（0 次压缩、无报错、GPU 掉 0） |
| 100,000 | 131,039 | 122,816 | ❌ 同上 |

停在 86K 时池里还有 ~36K 余量，**不是分配失败、也没有任何异常**，只是调度不再推进
（最后一条 `row_synced`: row=5656 页 / seq_len=86016 / tokens=81920）。

也就是说：压缩触发阈值（`budget_pages + 64`）与 B 绑得太紧时，请求会在"还没到阈值、池也没满"的空档里
停住。B=36,864 因为阈值低（~37K），压缩在早期就发生，反而稳定。

下一阶段要查的是**这个空档里调度器为什么不再调度**（断点挂 `schedule`，看 `num_new_tokens` 与
`request_token_budget`），以及阈值是否应该改成"池余量驱动"而不是"B 驱动"。
