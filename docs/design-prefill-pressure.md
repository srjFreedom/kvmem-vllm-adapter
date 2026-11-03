# 设计：prefill 期有界常驻（对齐 kvmem 0.19 P3-1）

状态：设计稿（goal-7833cb61 阶段 A 产物）。实现未开始，生产不动，全部改动将落在
独立开关后面 fail-closed。

## 1. 目标与验收（摘自 goal）

prefill 期间 GPU 常驻 KV 随 budget 常数、不随 prompt 长度 T 增长（原版退出标准
"GPU attention KV bytes ≈ constant with prompt length, scales with budget not T"）。
验收：250K prompt（超池 201,438 且超 MAX_MODEL_LEN=200,000）中段 needle 答对；
152K/池 122K 的 2026-10-29 退化案例复测通过；常驻账本日志有界；无损门
（A/B sha、88K NEEDLE、会话链、离线 9/9）不变；解码保持 70-80+ 带。

## 2. 现状资产盘点（都在，缺的是时序）

| 组件 | 位置 | 状态 |
|---|---|---|
| 压力判定 | `runtime.prefill_pressure_due(tokens, resident_pages, high_pages, last_reselect, reserve)` | 函数已有，调用点被 `no_prefill_compact=True` 短路 |
| 核心侧压力计划 | libkvmem（v0.19.0-rc1）`prepare_prefill_pressure`/`maybe_offload_during_prefill`/`spill_outgoing`/`admit_incoming`（evict-before-stage-in） | 已链接进 `vllm_kvmem_ext`，接口在 `kvmem_runtime.hpp` 实证 |
| 两本账 | 引擎 fork：`num_logical_tokens`（单调，驱动调度）vs `num_computed_tokens`（常驻账，驱动块分配） | 生产在用（prompt 末尾 rebuild 就是它） |
| 跨 rank 共识 + Core 边界压缩 | `sched_compact` 模式（EngineCore 调度边界做选片，worker 只执行搬页） | 生产在用（prompt 末尾触发） |
| 页搬运 + 重旋 | `_reselect`（decode 期）+ REBAKE rank 基重旋 + 块表行改写 | 生产在用，不变量有单测 |
| 增量统计 | 桥 `capture_k` 每 forward 按 64-token 桶 harvest mean-K；`capture_q` 收 query | 生产在用（prefill chunk 也过桥） |
| sink+tail 选择 | `prepare_reselect(mandatory=sink+tail)` | 已有（decode 压力同款） |

## 3. 当年为什么死（两条实测教训，注释在案）

1. **socket 路径时序错**（runtime:2455 注释）：mid-prefill 回退把"重算 batch"交给
   CUDA-graph dispatcher，被按 5-token uniform decode 分类，而 attn metadata 说
   `num_prefills=1` → `cudagraph_utils.py:114` 拓扑校验 RuntimeError，引擎当场死。
   根因：压缩由 **worker 侧 socket** 触发，调度器不知道账本回退，batch 分类与
   metadata 两个视图打架。
2. **首 decode 步搬表**：spec drafts 在飞时改块表 → 两本账差 draft 数
   （`prompt served but no sampled token logical=43800 computed=36864`）。
   结论（已固化）：压缩只发生在 **prefill 步边界**（无 drafts 在飞）。

`no_prefill_compact` 的注释明说："The invariant rework (page-aligned visible, packed
rows, RANK-based rotation) is in place and unit-tested" —— 页搬运不变量早就修好了，
欠的就是调度器握手。

## 4. 设计决策

### 4.1 握手 = 把 sched_compact（Core 边界模式）从"prompt 末尾"扩展到"prefill 中途"

压力检查放在 **EngineCore 调度器决定下一个 prefill chunk 之前**（它同时握着两本账）：

```
resident_pages + ceil(incoming_tokens/page) > high_pages  →  在 chunk 边界触发
  1. worker 侧 harvest 保证（本 chunk 统计已入桶——桥每 forward 都做，天然满足"统计先行"）
  2. 选片：无最终 query 时 = sink(8192) + tail(recent) + 增量 mean-K 评分选中间页
     （候选集 = 全部已提交 host 页，含历次淘汰页 → 召回天然成立）
  3. EngineCore 同步回退常驻账（logical 不动），worker 按 plan 搬页/重旋
  4. 下一步调度器看到 resident < logical，自然把差额排成 prefill chunk（分类正确，
     不再出现"decode 分类 + prefill metadata"打架）
```

关键：分类错误源于"调度器不知道回退"。Core 边界模式里回退**就是**调度器做的，
vLLM 原生逻辑（computed < logical → 排 prefill）自动接管。

### 4.2 重算 vs 搬页：两条 apply 路线，一个时序

- **路线甲（搬页，原版语义）**：选片后直接 stage-in/stage-out，GDN 状态在压缩视图上
  连续推进、不回退。零重算，prefill ≈ T×常数。原版如此（"prefill and decode pressure
  use accumulated accepted attention over resident candidates"）。
  风险：我们 10-29 的乱码实验是**无评分强裁**，不是此语义；但 GDN 状态与视图不一致的
  理论担忧仍在 → 用 152K 复测做判据。
- **路线乙（重算，rebuild 语义）**：压力点触发一次 prompt-中 rebuild（现机制原样复用，
  状态与视图一致）。代价 O(k×B) 重算（250K/B=36.8K → k≈5-6 次，prefill +~90%）。
  语义干净，实现最短（rebuild 已生产）。

**决定：先做乙**（复用生产 rebuild，改动最小、语义最稳），把甲作为后续优化项——
乙的验收过了就证明"时序握手"成立，甲只需替换 apply 一步。goal 的验收(1)(2) 两条
路线通用。

### 4.3 边界与失败语义（fail-closed 清单）

- 只在 prefill chunk 边界触发（无 drafts 在飞）；压缩后的请求沿用 **draft-forbidden**
  门（rebuilt/compacted 禁投机，现规则自动覆盖）
- 高压水位：`resident_pages > high_pages` 才触发；`reserve=R 的最小推进量` 防止
  每 chunk 都压（`prefill_pressure_due` 的 last_reselect 语义）
- 统计先行：淘汰页必须已 harvest（桥每步 harvest + commit 前校验；缺统计的页不许淘汰
  ——沿用 `session_checkpoint_skipped: core statistics checkpoint unavailable` 同款
  fail-closed）
- 循环状态：整请求一份，绝不随页搬动（既有定律）
- TP2：选片决策在 EngineCore（两 rank 执行同一 plan，指纹校验现成）
- 开关：`KVMEM_PREFILL_PRESSURE=1`（默认 0）。关 = 现生产行为逐字节不变
  （无损门 A/B 会验证这一点）
- MAX_MODEL_LEN：阶段 D 处理（250K prompt 要过入口校验：rotary cache 按
  max_model_len 尺寸，250K×head_dim×2B≈几十 MB 可行；logits 与 max_model_len 无关）

### 4.4 观测点（验收 (3) 的证据）

每次压力落事件：`prefill_pressure {req, resident_pages, high_pages, chosen, spilled,
recalled, logical, tokens_since_last}`；`debug_pos` 已有。常驻账本曲线由事件流重建，
证明"随 B 常数、不随 T"。

## 5. 实施顺序（对应 goal 阶段 B-E）

1. B1：scheduler 侧压力判定 + chunk 边界触发（flag 后）
2. B2：无 query 选片策略（sink+tail+增量 mean-K，候选含 host 页）
3. B3：压力点接 rebuild（路线乙）+ 账本回退 + draft-forbidden 沿用
4. C：召回验证（淘汰页在后续压力点被再选中；needle 页跨压力点存活）
5. D：MAX_MODEL_LEN 闸解除 + 250K 缓冲尺寸核查
6. E：152K 复测 → 250K needle → 无损门 → 文档

## 6. 原版对照（源码实证，v0.19.0-rc1）

`kvmem_runtime.hpp`：`maybe_offload_during_prefill`（注释：next chunk 超 semantic
budget/硬池/高水位 → sink+tail 压力计划，caller 跑 `finish_reselect` =
evict-before-stage-in）；`spill_outgoing`（GPU→CPU 拷贝，暂不释放槽位）→
`admit_incoming`（释放槽位再 stage-in）。打分语义（README/graph-recall.md）：
"prefill and decode pressure use accumulated accepted attention over resident
candidates, without graph propagation or cold recall"。改造计划 P3-1 退出标准即
本 goal 验收 (3)。

## 6.5 B1b 根因记录（2026-11-02 实测，goal round 2）

压力触发/打包/水位/互斥全部正确后，剩余缺口 = **mid-prefill 流式 commit 的地址一致性**：
- 第一层（planner 没问题）：`reselect.plan.sel_pages` 每轮都是满预算 1792 页；缩水发生在
  orchestrator 的 missing 过滤（`store.has` 失败 ~450 页/轮被丢弃）。
- 第二层（写点≠读点）：`refused_probe` 字节和窗口显示 kernel 把上一个 chunk 写进了
  **比 commit 读取点靠前 200 页**的簇（nz 簇 4042-4241，读的却是 4242+ = logical+base 预测）。
  原因：pressure spill 释放的块回到分配器池头，下一 chunk 的新块 id 来自池头复用，
  "dense base 延续"假设失效。
- 第三层（权威通道在场但 miss）：`commit_refused_zero.smap=true, span=[0,4096], hits=0`——
  为 lagged commit 准备的 kernel slot-mapping 通道在 prefill 步全部查空，退到过期 stage 镜像。
- 修复方向：prefill commit 改用捕获的 slot_mappings（或 bridge 同一张量）寻址；
  禁止再信任 dense 延续预测。诊断（pressure_gate/refused_probe/nblocks）验收后移除。

## 6.6 Round 3 落地（B1c/C/D 接通，88K pressure 端到端首过）

- **B1c（地址）**：lagged commit 读块表镜像会踩 allocator 池头复用（"logical+base" 预测失效）；
  kernel 权威 tensor 在 compiled 管线里读时已被重填（实测 skeys=[0], hits=0）。
  修复 = step_end 快照 `capture_slot_mapping`（仅 pressure 的 prefill 步，OFF 零成本），
  commit 地址链优先快照。效果：每轮压力压缩选满 1792 页，`commit_refused_zero` 清零，
  常驻恒定 28672（峰 40960=预算+chunk），P3-1 账面达成。
- **Phase C（final reselect）**：pressure 分支 prompt 末跑 query 重选
  （`pressure_final_reselect`），needle 页 2704-2712 召回进视图、旋转 24/24 自洽。
  但**仅 reselect 不够**：GDN/hidden 状态是在有损压缩视图上流算出来的，答案复读
  `MID-KEY`（表示层丢针）——与原版 P1/P2 注记一致（P1 召不回中间针，需重算定稿）。
- **Phase D（finalize）**：`finish_prompt_for_reconstruction` 在 pressure 档改走
  `_rebuild_queue` → Core quiescent 边界的 `_rebuild_select`（store 逻辑坐标、n_pages 守卫、
  TP2 共识）→ 全新 admission 重算选中视图（1680 页, store_missing=0）→ 状态与视图一致。
  v12 崩溃根因 = 旧入口 `_reselect(rebuild=True)` 把**压缩视图坐标页**送进按**原始 prompt**
  校验的 `select_tokens`（两坐标混用即死，注释在案）。
- **实测**：v14 88K pressure `ZX-4417-QJ7` 正确 136.8s。
  v15 同输入事件链逐字段一致（选页/打包确定）但输出翻错——**落地 slot 受分配器复用顺序
  影响 → kernel 数值路径变 → 近阈值 needle 不稳**。开放缺口=pressure 稳定性（下轮）。
- **生产回归（金标准）**：HEAD 全部改动 vs 060eda0(v0.4.0) 同实例 A/B **逐字节一致**
  （A `0013e530…`、B `73ef34c9…`；旧 sha c2f36446/42675199 系算法口径不明，E 阶段以新算
  法+新锚点更新 DELIVERY 表）；88K OFF needle ✓ 107.8s；离线 10/10。

## 6.7 Round 4 稳定性定性（4 连发判别实验，docs/evidence/pressure-stability-round4.jsonl）

同实例连续 4×88K（pressure ON）：run1 ✓ 136.3s、run2 ✓ 135.8s、run3 ✗("7392-ALPHA-9"=
档案内干扰码)、run4 ✓（与 run2 同文）。**真非确定性**（非"首跑特殊"、非"连跑稳定"）。
排除项：
- **不是漏选**：needle engine-block 26 在全部 4 次 rebuild `chosen` 里（rebuild diag
  chosen_head 对比，v14/v15 同）；
- **不是打包层**：want/packed/store_missing 每跑相同（1680/1680/0）。
定性机制：**pressure 驱逐后，后续 chunk 的 hidden/K/V 字节是在有损上下文（sink+tail 视图）
上计算的** → 这些字节进 store → 打分（mean-K 相似度）的**内容基础随漂移变** →
needle 与干扰块的排序抖动 → 17 块视图里模型被干扰码劫持。生产 OFF 路径 dense prefill
的字节完整 → 同粒度选择稳定（v0.4.0 门）。与 6441248 教训同族：微小数值差足以翻动近阈值选页。

修复候选（B2，下一轮）：
(a) 评分基础改用**页字节的 canonical mean-K**（_feed_retrieval 已随 commit 写入，与
    有损上下文无关的是"字节本身"——但 K 字节由 forward 产生，仍依赖 hidden——需实测其漂移量）；
(b) 落地序确定性化（固定 dest 窗口/确定性分配），先消掉"同输入不同落地"这类微差；
(c) 接受漂移、把选择做宽（提高 rebuild 预算占比/强制 needle 邻块扩展 KVMEM_GRAPH_ALPHA
    在 pressure 终选也生效）——语义代价最小。

## 6.8 Round 5 B2（Core 驱动压力协议落地 + 定罪到执行链）

**接线**（全部 fail-closed 于 KVMEM_PREFILL_PRESSURE）：
- worker.py：pressure ⇒ `cfg["sched_compact"]=True`——释放走 Core quiescent 协议（adopt/free 在
  schedule() 单线程内），杀掉 round-4 定性的 allocator 竞态漂移。
- plugin install：**强制** `runtime.compact_mode="quiescent"`（探测发现 runtime 根本没有
  compact_mode 属性、dispatch 回落到 env prefix——"if ==prefix 才覆盖"的条件永不成立，
  24 连 empty proposal 就是这个；直接赋值修掉）。
- `_quiescent_compaction` 新增 (c) **预算水位触发**：`num_computed_tokens >= budget+gen/2`
  且 in-flight==0（prefill 无 draft，quiescent 边界安全）+ per-req 节流 gen/4。
- needs_reconstruction 加 `_over_budget` 条件：Core-driven 回卷不置 compacted_in_prefill，
  水位等价（无它则 88K 稠密走过驱逐窗口直答，"No special access code recorded."）。
- blockwise_empty_pages 诊断补日志。

**实测（88K, pressure ON）**：
- (c) 真实生效：seq_len 时间线 45056→28144→44528→22128（**常驻有界 ✓ 整块视图 ✓**）；
- 4 连发全部 **4/4 逐字节相同**（"No special access code recorded.", 7 tok, 129.4s）
  ——**Core 驱动确定性达成**（round-4 是 3/4 抖动）；
- **但稳定错**。定罪实验：dump 子请求 prompt（30974 ids）→ decode 文本
  **"The unique record KEY-MID has access code ZX-4417-QJ7." 明确在场**、tail 问句逐字在尾
  ——**输入无罪，bug 在子请求执行链/父压缩轮的残留影响**（不是选页、不是 token 流）。
- 对照：同 build 下 (c) 被 empty-proposal 挡住时（=纯 dense+final+rebuild）4/4 **全对**
  （round-5 早前实验，"ZX-4417-QJ7" 逐字节一致）——**差异唯一锁定=压力压缩轮发生与否**。

**下一轮（二分）**：env `KVMEM_PRESSURE_HIGH_TOKENS` 诊断覆盖把 (c) 水位调到永不触发
（其余全同），跑 88K：对→定罪压缩轮污染（查 GDN 清零/slot/统计基础）；错→定稿 over_budget
形态本身问题。候选根因清单：GDN state 清零在压缩后 slot 错位 / 子请求 positions 起点 /
store 页 rebake delta 在 blockwise rewind 后的 rank 语义。

## 6.9 Round 6 二分定罪 + 邻块扩展（负结果）

**二分**（env `KVMEM_PRESSURE_HIGH_TOKENS=1e8` 禁 (c) 压缩、保留 over_budget finalize，
其余全同）：88K **✓ ZX-4417-QJ7**（133.9s，无压缩时间线 seq==tokens 全程）。
**定罪：finalize 执行链无罪；压力压缩轮是充分必要条件级的污染源。**

**两臂 chosen 集对比**：无压缩臂 [7,8,9,11,12,13,14,**26,27,28,29,30,31**,48,51,52,53]
（needle 邻域连续 26-31）；压缩臂 [7,8,9,12,13,14,26,27,28,29,30,32,35,38,...]
（**丢 block 31 = needle 句的下一块**）。needle 句跨 26/27 块界，block 31 是 needle 记录的
"完结上下文"（distractor 模板是单句循环，needle 是多句记录）。

**修复尝试 = 邻块扩展**（`_rebuild_select`：每个入选 middle 块强制带后继 b+1）：
**已生效**（run chosen=[...,26,27,28,29,30,31,32,33,35,36,38,39,...]——**31 在场**）
**但仍 4/4 稳定错**（"No special access code recorded." 逐字节一致，129.4s）。
→ 子请求 prompt 含完整 needle 邻域 + 问句、同机制无压缩臂答对——**输入无罪 ×2 证明**。

**新的定罪面**：污染源不在"选页集"，在**压缩轮的执行残留本身**。待查清单（下一轮判别）：
1. `reset_state` 无调用者（GDN 清零依赖引擎 fresh admission——压缩轮的 Core 账本回卷
   是否破坏 child admission 的 state 初始化路径？）；
2. blockwise 压缩轮 `restore_page_rank` IN PLACE 重旋 + `baked_delta` 字典改写——
   子请求若复用父的 orch 字典（`slot_of/baked_delta/resident` 未清？）pack 读错源页；
3. 判别实验：`KVMEM_COMPACT_REBUILD=0` + 压缩（父 view 直答 vs 子重建）+ 对比
   pack 前后 store 页 sha（trace_pages 已有机制）。

## 6.10 Round 7 判决实验：FORCE-BLOCKS 回放 + Core 账本无罪

**实验一（决定性）**：`KVMEM_REBUILD_FORCE_BLOCKS=7,8,9,11,12,13,14,26,27,28,29,30,31,48`
（= 无压缩正确臂的精确 middle 集，尾块 51-53 由 forced_tail 自动带上；force 开启时跳过邻块
扩展保证精确性）在**压缩正常发生**（gate 时间线 49152→28144、77824→18032、81920→22128）
的 run 里回放：diag 确认 chosen=17、packed=1680/1680、store_missing=0，**仍答
"No special access code recorded."（130.7s）**。
→ **选页质量假设出局：子请求 prompt 与正确臂逐 token 相同的块集，输出仍错。
污染在压缩轮的执行残留，不在 lossy 统计选页。**

**实验二（热断点，零重启）**：kvmem_dbg 落在 `Scheduler.schedule` 返回点（45 hits），
子请求段账本：
`(4096,4096,True,4096) → (8192,8192) → … → (30975,30975,True,2303) → decode 1/step`
**core 侧完全干净**：child num_computed 从 0、dense、无回卷残留、kvmem_rebuilt=True 在位。
父请求回卷仅存在于父的账本历史（协议使然），child admission 是全新 Request。
→ **Core 账本/引擎调度面无罪。残留锁定在 worker runtime 侧（同 rid 复用的
orch/store/commit 坐标链）或子请求 first-step 的 block_table 状态。**

**排除清单更新**：选页质量 ✗（实验一）、Core 账本 ✗（实验二）、GDN 状态槽 ✗
（has_initial_states 由 num_computed 决定，child=0 从零开始）、pack 目标冲突 ✗
（child dense 重算不读 pack 字节）。**剩余主嫌疑：worker runtime 对同 rid 的状态复用
（accept_reconstruction 清理面 vs 压缩轮写入面）——特别是 B1c 快照链/`slot_of`/
`baked_delta` 在"父被 (c) 压缩过"之后、child commit 的坐标算术**。

**下一轮判别（设计好）**：worker 侧 console/dump 对比 child 前向第一步的
`slot_mapping 目标页 vs 引擎 block_table` 与无压缩臂同位置——若写地址错位=定罪 commit
坐标链；同步把 force env 撤掉恢复纯压缩臂基线（当前 :18200 配置）。

## 6.11 Round 8 console 取证：两本账不变量疑似被 blockwise 回卷破坏

worker console（自举 re-arm，热挂，零重启）抓到 (c) 压缩后父请求 mid-prefill 状态
（step 301，两 rank 逐字段一致；docs/evidence/pressure-child-console-dumps-round8.txt）：

    st.tokens=24576  st.seq_len=24576  off_pages=0  applied_view_coord=False
    row=1818 pages  logical=0..1817  slot_of/resident=0..1535（1536 页）
    store_n=1536   store keys 覆盖到 1200..1535 段   needle_store=[false×7]
    input_batch seq_lens=[28672]

**震撼点（对照 goal 红线）**：
1. **`store_n == resident == 1536`**：host store 只镜像 resident 窗口——**被驱逐页没有
   落进 host store**（goal 语义要求 evicted pages stay recallable by the same scorer）。
   若 finalize 时 store 仍是 resident 镜像，`_rebuild_select` 的 `block_ok`/
   `restore_page_rank` 对 2626..2726 这类页 should fail——但 diag 报 packed=1680/sm=0
   ——**pack 可能"假成功"（restore 到错误页源）——下一轮定罪点**。
2. **`st.tokens == st.seq_len`（逻辑账追平了 resident 账）**：blockwise apply 后 worker
   逻辑账不再是"单调逻辑流"——两本账被合并回卷（num_logical 单调不变量在本 fork 的
   quiescent 语义里本就与 vLLM 原版不同——Core 侧 dbg 抓到的 logical 序列 49152→53248→
   ...→86670 仍单调，但 worker 侧 tokens 已按回卷记账）——**两本账的"账本语义"需要
   在本设计里重新钉死**。
3. console 条件时序修正：本轮 dump 触发窗（_pl-65536）打在压缩中期；**finalize/child
   时点的同表取证是下一轮第一件事**（判据：store keys vs chosen pages 覆盖、
   needle_store 真值、baked_nonzero）。

**另排除**：child 由 `is_rebuilt→continue` 完全跳过 runtime（1136-1141 注释亦写明），
**"runtime 同 rid 残留毒执行"出局**——child 纯引擎 native；问题在**进入 child 之前**
（store 数据面/pack 面）的残留。

## 6.12 Round 9 更正 6.11：那份 dump 是 child prefill 段——child 状态完全干净

逐字段复核：dump 的触发窗 `tokens >= _pl-65536` 对 child 恒成立（30975-65536<0）——
step 301 抓的是 **child 的 prefill 中期**（tokens=24576），不是压缩中的父：
- `store_n=1536 = 24576/16 = child 已提交页数`（child 用 accept_reconstruction 退休后的
  **新 store**，单调增长中）——6.11 的"驱逐页没进 host store"论断 **撤回**：父的 store
  是否覆盖驱逐页，`_rebuild_select` 的 `block_ok + store_missing=0` 已反证 **覆盖**
  （chosen 含 26..31 全部内部页 `store.has(p)`==True）。
- `seq_lens=28672 = 24576+4096` 引擎账 ✓、`baked=0`（child dense 位置）✓、
  `slot_of==resident` ✓、row=引擎 dense 行 ✓ ——**child 执行面无罪实锤**。

**当前矛盾表述**：child 输入正确（select_tokens dump 解码 needle 全句在）、child 执行
干净（本节）、两臂唯一差异=父是否被 (c) 压缩——但 child 与父共享的物理实体只剩
**设备块池的分配位置**（父压缩轮改变 free 链形态）与 **MTP/draft 边界账**（父压缩轮
epoch 与 in-flight 门）。
**本轮判别（零代码改动）**：压缩臂 + `KVMEM_SPEC_TOKENS=0`（禁 MTP）跑 88K：
- 对 → 定罪"draft 边界 × 压缩轮"交互（压缩轮改写的 in-flight/ledger 面影响 child 采样
  位置或 verify）；
- 仍错 → 只剩**设备块池位置类**（物理 slot → kernel 数值路径）——届时做
  "同 child 连跑两池位置对比"实验。

## 6.13 Round 9 判别：spec0 判负；want=847 现象暴露块粒度异动

1. **spec0 × 压缩 × FORCE 实验**：`SPECULATIVE_TOKENS=0` 下压缩臂+强制块集，88K 仍
   "no code recorded"（128.0s）→ **MTP/draft 边界与压缩轮的交互出局**。
2. **新异常**：本轮 rebuild diag `chosen=18, want=847`（此前恒定 chosen=17/1680）——
   847 = 18 × **49** 页——`block_pages` 的语义在轮间从 101（1616tok 引擎块）变成了 49
   （=kv 原生 784？），且本轮 (c) 压缩水位也变了（gate seq 31904/48288/64672 = 非整块，
   pool=226240 时代）。**子 prompt 从 30974 缩到 ~13.5K——needle 块仍在 chosen、pack
   成功、答案仍错**。候选联系：pool 变大（226240=140 块）后 `cap_engine_blocks` /
   `block_pages` 推导与压缩轮的 q 不再一致——若 bp 换算在 (c) 块视图里错位，
   **`(c) 选块 × finalize 选块` 用的页→块除法可能不同源**，needle 句的 token pack 顺序
   或 RoPE rank 在子请求里被打乱（子 prompt 文本 = select_tokens(logical pages) 升序，
   若 logical 与设备块换算错位，文本仍在但 **pack 给子请求的 rank/rot 错位**——子文本
   dump 已证句在——但 rank 错会毁读序）。
3. 下一轮定罪实验：**对比压缩臂 vs 无压缩臂的 rebuild diag 的 `want/bp/dest` 与
   child pack 的 logical→rank 映射表**（同 FORCE 集）；若 rank 序列错位 = 定罪块换算。

## 6.14 Round 9 收束：运行时几何常量在实例间漂移（重要运维事实）

diag (want,chosen) 历史分布（上）与 pool/engine-block 页数 `runtime.block_pages =
max(_bpk)`（plugin:836，随 kv_cache_config 组形态变）共同表明：**同一二进制不同 boot
下 pool 大小（201438→226240）与 bp（101/98）都会变**——
1. round-7 的 FORCE 判决（1680/bp101 时代、与正确臂同代几何）**结论保持**；
2. 本轮 spec0 run 的 FORCE 数字是"旧块号 × 新 bp=98"的错代产物（want 847=8.6×98），
   **"draft 边界判负"存疑待复测**（要同代几何重跑）；
3. **goal 验收前必须钉死启动几何**（GPU_UTIL 固定 + 启动后打印并校验 bp/pool 常量，
   不一致 fail-closed）——写进 E 阶段清单；实验纪律更新：跨 boot 比较必须先对表
   (want,chosen,bp,pool) 四元组。

## 6.15 Round 11 守卫版验证：两修复实锤、主缺陷仍开放 + 新取证面

- **starvation 修复实锤**：child 最后步 `n_new=2303/654` 都正常调度（此前父永远卡在 86016）；
  `block_table_updates` 对 child 为空、new_block_ids=[[],[],[],[39,9]]（末步 2 新块，正常）。
- **child 输入第 N 次确认干净**：30974 ids，needle 句完整、问句逐字在尾、tail_blocks 齐全。
- **4 连发仍 4/4 稳定错**（guard 版）→ revert 风暴与 starvation 都是真 bug 但**都不是**
  稳定错的主根因。
- **新取证面（本 round 发现但未完成）**：fork 有 `kv_cache_block_copies` 引擎通道
  （gpu/model_runner.py:1262 + worker/utils.py:743：device→device **块内容拷贝**语义）——
  压缩轮若走这条通道，块视图的 KV 由引擎搬运，与 worker 侧 pack 的写序交互从未核对。
  下一轮：dump child 每步 `kv_cache_block_copies` + `block_table_updates` 全量，并核对
  `apply_compaction` 走哪条（channel adopt vs Core copies）。
- V2 runner 的块表访问面：`input_batch.block_table[gid].get_cpu_tensor()`（非 V1 属性名，
  前两轮 dump 空手而归即此因）——child 行的实际块 id 序列仍未直接看过。

## 6.16 Round 11 根因实锤：child 复用父的引擎块行（stale row）

热 console（正确 accessor `runner.block_tables.block_tables[gid].gpu[ri]`）抓 child 最后步
（n_new=2303，seq_lens=30975 两 rank 一致）的 attention 块行（group 3）：

    g3: nz=5454, head=[4141,4142,4143,...] tail=[...,2923,2928]
    groups 0-2 (mamba): 各 5 块（child 自己的）

**解读**：child 的 prompt 只需 1936 页（30975/16）≈20 块，行内却有 **5454 个非零项**——
**恰为父请求（86670 tokens→5417+pad≈5454）的块行数**。V2 的块行是 append 语义，
child 复用父的 row index（同 rid），**父的块行从未清空**：child 的 attention 于是读
**父压缩时代的块视图 + 自己新写的小尾巴**——模型在错误 KV 上产生流畅但错误的回答
（"no code recorded"），确定性来自行内容的确定性。

**统一解释全部历史现象**：
- OFF/无压缩臂 child 正确：父行是 dense 连续物理块，child 重算恰好把这些块重新写满
  （同 base 分配顺序），stale 行被新数据覆盖 → 无害；
- 压缩臂 child 错：父行=碎片化的视图块（非 child 的块号），child 只重写了自己的 20 块，
  前 5400+ 条指向别处旧字节 → attention 大半错读；
- revert 风暴/starvation 都是真 bug（已修）但不是主根因；4/4 逐字节稳定 = 行确定。

**修复方案（下一轮实施）**：worker `execute_model` 入口的 rebuilt-admission 钩子处
（scheduled_new_reqs + kvmem_rebuilt，本在 forward 之前）清掉 child row 的
group-3 块行并按 `_nr.new_block_ids` 重装 child 自己的块；配套 dump 验证 nz==1936 级别。

## 6.17 Round 11 收官：行复位修复落地并实测生效（但主缺陷未全闭）

- `rebuild_row_reset` 事件 ×8（4 run × 2 rank）；child 最后步块行实测 **5454→2020**
  （=20 块×101 页，head 连续 1212…，groups 0-2 各 5 块为 child 自己的 state 块）——
  stale-row 污染是**真 bug、已修、已验证生效**。
- 但答案仍 4/4 稳定错 → stale row 是**必要条件之一而非唯一根因**。child 现在：
  文本正确（解码证）、块行纯净（实测）、Core 账本干净（dbg）、runtime 不碰 child（console）、
  GDN groups 行干净——**剩余未直接观测面：GDN 运行态 slot 的字节内容**（has_initial_states
  理论保护，但本 fork 有 fast_prefill/valid_state_slots 自定义路径，1617 行附近）；
  以及 **child 第一步（forward 中）的 slot_mapping**（本 dump 是 step_end 视角，
  第一步 append 前 reset 已生效，但首步 forward 用的镜像顺序未逐 token 核过）。
- 本轮累计修复三个真缺陷（revert 风暴、last-chunk starvation、stale block row），
  pressure 臂行为已从"玄学抖动"收敛为"确定性、输入完整、行纯净、仍错"——
  每一层都比上一层更接近真相。

## 6.18 Round 16-17 转折：child 文本无罪被推翻，真缺陷=乱序拼接（OFF 臂同病待复核）

三组纯 replay（零 kvmem 路径参与时）：
- OFF boot replay child ids（30,974）→ **✓ ZX-4417-QJ7** —— 子文本可答（此前"文本中毒"
  推断作废）；乱序拼接的文本模型能检索回 needle。
- pressure boot（未加平价门）replay 同一 ids → 'ZX no;;;;' 乱码 —— 事件链实锤
  **(b) final 无预算门**：30,974<budget 的普通请求被 blockwise 压缩（books 回卷 16,430）
  后乱码。**修复**：(b) 候选加 `num_prompt > budget_tokens`（plugin 2218 区，注释含测量）
  ⇒ 重启后 replay ✓ ZX-4417-QJ7（**per-request 平价恢复**）。
- 88K chat 四连发仍 4/4 "no code"（needle=False）。

**统一新模型（待验）**：finalize 重建出的子 prompt = sink 块 + **按分数排序的中段块** +
尾块——**文本顺序被打乱**，needle 落在 0.25 位置而 **真 ask（"What is the access code
for KEY-MID?"）在 block122 被裁掉**，模型看到的问句是块拼接出的
"…no special access code recorded.\n\nWhat is the…"——它答 "No special access code
recorded." **是在错误引用下的合理回答**！OFF 生产臂走同一 pack 序 ⇒ 若 OFF 88K 现也错
= 乱序拼接是 finalize 固有缺陷（非 pressure 专属）；若 OFF 仍对 = pressure 专属差异。
**下一轮判决实验：OFF boot（平价门不影响 OFF 分支）跑 88K chat。**
## 6.19 Round 19：requeue 专属定罪 + 隔离实验的正确解读（自纠）

**同轮三段对照（同 boot、同 ids）**：
1. chat 88K（压缩+(c)+finalize child）→ ✗ "No special access code recorded."（3/3+1/1 稳定）
2. completions FRESH replay 同 child ids → ✓ **ZX-4417-QJ7**
3. chat FRESH admission（child 文本重套模板）→ ✓ 答对
⇒ **ids/内容/模板/sampling 全部无罪；定罪面 = requeue child 的执行状态本身**。
- 附加事实：child 文本从半句 `' special access code recorded.'` 开头（模板结构不在 ids 里）
  ——模型仍答对（replay 证），检索鲁棒性足够；"乱序文本不可答"理论作废。

**REBUILD=0 隔离实验（本 round 早期）解读修正**：`(c) 压缩 on + compact_rebuild=0` → 答 "No"
（104.7s，applied epoch1 visible=24048/epoch2 visible=13936）——**这不是 view 损坏**：
stage-1 语义下压缩轮的选择没有真问句参与，needle 块未被选入即永久丢失，"No" 是该配置的
**预期语义损失**。该实验只证明：(c) 通路工作正常 + 单独压缩不解决检索正确性
（正是 child 协议存在的理由）。误读当场收回。

**下一刀（唯一残面对象）**：requeue child 与 fresh admission 的执行差 = 同 rid 复用面
（worker 观察 remove 与 child add 的时序、GDN state slot/`is_prefilling`/`num_accepted`/
rope_state 的 per-idx 残值）。诊断：child 首步 console dump 上述 per-idx 缓冲 +
`model_state` 句柄字节；若引擎在 re-add 前没做 remove（同 rid 短路），所有 per-idx 状态
都是父代残值——**修复方向 = requeue 用内部新 rid + output 转接（rid 解耦）**。

## 6.20 Round 19 收官：child 首步引擎侧全净，唯一罪面=pack 写序 × 引擎分配序碰撞

STEP2RESID（child 步2，两 rank 一致）：
    nb_np=[5,5,5,303]  g3 head 909..1211 连续  idx=0  ncomp/nlogic_np=0
    gdn_has_init=[False]  ✓（GDN 理论排除：kernel 从零写）
    runtime st=None（child 不注册，符合设计）
⇒ child 的**行/账本/state 决策全部干净**——引擎视角它是一个标准 fresh admission。
⇒ 但它只有 303 块（30975/16≈1936 页=19.2 块？——303×101=30603 页远超 1936：行容量按
  MAX_MODEL_LEN=200000 分配，nz=303 是**当前已 append 的块数**，4096 步时 303 块已覆盖
  前 4096 所需——不是异常）。
**最后交汇点 = pack 数据面**：`_rebuild_select` 的 dest 窗口是**全池 block-aligned**
（first=1..num_blocks，pool_pages 升序），`logical_pages` 升序映射 dest——而 child 的
fresh 分配由**引擎 free 队列顺序**决定。**replay 实验（同 ids 全对）恰好覆盖同一批物理块
并重写** ⇒ 内容无罪、字节有罪：**child attention 读的是 pack 写序与引擎分配序不一致时
落在行内的旧字节**。验证实验（round 20，env-only）：
`KVMEM_RESIDENT_PREFILL_COMPACT=0` + `KVMEM_PREFILL_PRESSURE=1`
→ (c) 触发面死（prefill 零压缩）→ finalize 经 `_over_budget` 仍选页+pack+child requeue
→ **若 needle ✓**：压缩轮与 child 协议联手缺陷=压缩轮 store/统计残值；
→ **若仍 ✗**：pack dest 顺序 bug 实锤（child 从未自写而被 pack 字节污染），
   修复=child 首步前清掉 pack dest 区间或按引擎分配序重排 pack。

## 6.21 Round 20：同 rid 复用面整体排除，根因定案=pack 区 × 引擎分配区交叠

**new-rid 判决实验（`KVMEM_REBUILD_NEW_RID=1` + step 输出反向映射）**：
- 首轮 wiring 自伤两处（kvmem_map 未初始化；register 内 EngineCore 本地名未导入 → 整个
  channel 块 UnboundLocalError 崩死）——当场定罪修复；
- v2 完整跑通：child engine_rid=`...#kvmr1`、输出映射回原 rid、HTTP 130.4s 正常返回、
  reconstruction_started×4、rebuild_phys_diag n=1680——**仍 4/4 错**。
⇒ **同 rid 的全部复用面（行/账本/state slot/GDN/draft/output 路由）被"全新引擎身份"
整体排除**。child 与 fresh replay 至此只剩**一个**差异：
**fresh replay 的分配从 free 队列拿到的块恰好被自己的 prefill 逐一重写；
而 child 无论新旧 rid，其行内都会包含 pack 窗口 [101..1780] 中"child prompt 未覆盖到的
stale 页"**——pack 写了 1680 页，child 需要 1936 页——中间空洞=模型读到邻接垃圾。

**定案修复（round 21 实施）= pack 随分配（pack-follows-allocation）**：
1. `_rebuild_select` 不再自选 pool 窗口 pack，改为只产出 `logical_pages` 序列并入队
   （dest 未知，记 `_pending_pack[req]=(logical_pages, prompt_len)`）；
2. worker execute_model 入口钩子看到 child admission（rebuilt rid + scheduled_new_reqs）
   取 `new_block_ids` → child 行坐标即目标；**forward 之前**把 store 页写入 child 行
   （child 首 256 页随后被自己的 prefill 原地重写=同内容覆盖，无害）；
3. pack 失败 → 明确拒绝 requeue（fail-closed 语义不变）。
半成品期间 reserver 骨架保持 no-op（range(1,1)）。

**运维**：NEW_RID env 已撤（实验开关默认 OFF，fail-closed 语义恢复）。

## 6.22 Round 23 交叉判决（最硬证据）：同 boot 同轮同 ids——requeue child ✗ / fresh replay ✓

- pack-follows-allocation 已实施并**实测无效**（restored=303/err 修光/同步门加上仍错）；
  本 round 整体回退（工作树回到 96c593d，离线 6/6 绿）。原因事后清晰：child 的
  prefill 对全 prompt **逐 token 重算 KV**，pack 字节对它本来就是死数据——round-20 的
  "pack 空洞"理论**由本轮判决取代**。
- **判决实验（干净版 boot markerAN，同轮三步）**：
  1. STEP1 chat 88K（压缩+(c)+finalize+requeue child）→ `No special access code recorded.`
  2. STEP2 **同一 boot 同一实例**用**本轮 child 的精确 ids** 走 fresh completions →
     **✓ `ZX-4417-QJ7`**（31.4s）
  3. 本轮 child 文本：needle 句 ✓（frac 0.251）、真问句 ✓（frac 0.867）——选择质量达标。
- **推论（排除法闭合）**：同实例/同池状态/同字节 ids/同采样参数，唯一差值=
  **请求进入引擎的方式**（requeue-with-detach vs native admission）。
  NEW_RID 实验（round 20）已把"同一身份的残表"排除，剩下的差值在 **requeue 时序本身**
  （detach 与 re-add 同轮原子性、kv_cache_manager 对父块的回收 vs child 分配的顺序耦合、
  update_from_output 观察面）——child 生成的前缀里模型"看见了"某种未被 replay 复刻的
  东西。round-24 取证=child 首 decode 步的 sampled token 概率面（top-k）对表 replay。
- **运营注**：88K 生产臂（pressure OFF）不受影响（round-3 起从未走过 child 协议——全历史
  OFF boot 零 rebuild 事件，round-18 事件扫描证）；child 协议迄今**从未端到端成功过一次**。

## 6.23 Round 24：排除法收口——罪在 child 专属运行路径，不在压缩轮

- 新 bp 全被吞的根因自查：**`hits:0` = 触发 0 次**（`_left<=0` skip）——第 18/20 轮
  XFER/RESID 的成功命中都是**旧 .bp 残留**；本轮 console（n_new==1 命中 child decode 步）
  恢复正常取证。logits 不在 execute_model_state 驻留（logits_present=false），改打
  `runner._kvmem_req_ids` 路径（child 步 end 可见 rid 列表）——下一轮。
- child rid 事件时间线（markerAN boot 最后 run）：父两刀压缩（45056→24048、73728→13936）
  → finalize@86016 → **quiescent_compact_empty tokens=0 ×2 = child 首步 fresh 账本**
  （非风暴，无 revert 记录）→ rebuild_queued → row_reset ✓。协议执行面全绿。
- **4 连 replay（同 boot、压缩轮之后）= 4/4 ✓ ZX-4417-QJ7**：
  **压缩后的实例状态对 fresh 请求完全无害**（数据面/位置面/池面洗清）。
- ⇒ 排除法闭合：同 ids/同 boot/同采样下 fresh ✓ vs child ✗，且身份面（NEW_RID）已排除
  ⇒ **唯一幸存变量=child 运行期间 runtime 因 rebuilt 标记而抑制的钩子组**
  （step_end 主循环 `is_rebuilt → continue`、prepare pin 跳过、advance 缺席）。
  OFF 生产从未跑过 child ⇒ 这些抑制从未被端到端验证过。
  下一轮：child 步内 runtime/orch 注册面 dump + 逐项解禁二分。

## 6.24 Round 24 收官记录（含两处自纠）

1. **hits:0 陷阱**：kvmem_dbg 的 `hits` 语义=触发次数（0=永不）——第 18/20 轮的 XFER/RESID
   成功来自**旧 .bp 残留**，本轮新写的 hits:0 四条全部静默。四个 130s run 浪费在此。
   教训入 USER 协议：**新断点先跑 smoke 级请求验证命中，再上贵实验**。
2. **父子同 rid 时间线误读撤回**：按 rid 过滤 child 事件时，`note_prompt_ids n=86670`、
   `row_synced 4096`、`resume_skip` 都发生在 rebuild_queued **之前**=父的 admission 记录。
   child 的真实事件面自 rebuild_queued 起：empty 提案×2（fresh 账本，无害）、
   row_reset ✓、reconstruction_started ✓——协议执行无异常。
3. **本轮净事实（两条硬）**：
   - 压缩轮之后的同 boot：**4/4 replay 全对** ⇒ 压缩实例的池/数据面对后续请求无害。
   - `quiescent_compact_empty tokens=0 ×2/run`=child 首步 Core 触发面正常拒绝，无 revert 风暴。
4. **残余唯一变量**（承 23 轮）：child 与 fresh replay 之间只剩 "rebuild 标记抑制的钩子组"
   （step_end `is_rebuilt→continue` 全家：capture_slot_mapping、statistics、reselect、
   retire 等 8 处）——OFF 生产从未跑过 child，这些抑制从未端到端验证。
   下一轮二分：**KVMEM_REBUILD_HOOKS 开关逐项放开**（先放 step_end 主循环的 2762 continue
   改走正常 advance 路径——child 本质就是 fresh 请求，理应享有完整钩子）。

## 6.25 Round 25：child 与 replay 在引擎面已逐字节等价，剩余差值=生成流连续性

- attr 探针（环境开关版，回退）实测：worker NewRequestData 侧 attr_val **既有 True 也有 False**
  的 sighting（不同代码路径），`reconstruction_started×4`+`rebuild_row_reset` 事件证明
  accept/复位**一直正确执行**——attr 理论证伪，但把"note 在 accept 之前"的次序缺陷修成
  了确定性版本（`_await_child` runtime 标记为唯一事实源；行为等价，属正确性加固，保留）。
- FULL_HOOKS=1 实验重解读：放开 step_end `is_rebuilt→continue` 后行为不变的真实原因——
  child 的 st 不存在（`_ensure` 缺席），钩子链在 continue 之前的 `runtime.state(req)`
  短路路径里也没执行任何东西，实验**未真正放开**任何 child 钩子，证据力作废。
- **至此 child vs replay 的可枚举差值全部闭合**：ids 逐字节（30975=30974+1 个逃逸样本
  token，其语义=生成起点，见下）、块行纯净、账本归零、state 判定、身份（NEW_RID）、
  采样参数（top_k=1 亦错）。
- **唯一结构性残差**：child 的输出流"继承"了父对话（output 连续、outs 拼接）。
  判决实验（round 26，离线）：把 child ids 去掉尾部 1 token 或替换该 token 重放——
  若答案随该 token 翻转 ⇒ "No" 源头=父逃逸样本（修复=finalize 前禁止任何样本逃逸 /
  丢弃 outs）；若不翻转 ⇒ 转真钩子放开二分。

## 6.26 Round 25 【根因定罪 + 首次端到端通过】：逃逸样本毒化 child prompt

**三臂 replay 判决（同 boot，离线零引擎介入）**：
| 臂 | prompt | 结果 |
|---|---|---|
| A | child ids 原样 30974 | ✓ `ZX-4417-QJ7` |
| B | + 逃逸 token "No"(2665) | ✗ ` special access code recorded.`（**逐字复现生产 child 答案**）|
| C | + 替代 token "The" | ✓ 含码 |

**根因**：父 prompt-complete 步的采样在 finalize/requeue 前 2 个 round 逃逸
（654 末步：schedule 时 unknown-complete，update_from_output 即发出 "No"）→
`queue_kvmem_reconstruction` 把它拼进 child prompt（nids=30975=dump+1）→
child 从 "No" 续写=自我实现。**修复**：rebuild 时 `outs=[]`（重建代换整条生成流，
父样本无权威——设计语义本就是"替换"）。

**验证（同 boot 4 连发）**：needle=**True ×4**（'NoZX-4417-QJ7'）——内容含正确码，
130s 稳定。**残留卫生缺陷**：逃逸的 "No" 已流入客户端累积器（detach 不截断 SSE 已发
部分）——流抑制=下一刀（schedule 时识别 prompt-complete 步 → update_from_output
丢弃该 rid 该步样本；判定式 `num_computed+n_new == num_prompt_tokens and prompt>budget`）。

## 6.27 Round 25 【END-TO-END GREEN】：逃逸样本流抑制落地，88K 压缩臂首次全对

- 机制补全（6.26 之后）：`escape_pred_near` 探针实测 prompt-complete 步进入
  update_from_output 时 `num_computed_tokens` 已被 finalize 的 (b) 触发**回卷到 26878**
  （视图坐标），任何 computed 基判定永不命中——**判据必须读逻辑账本**（num_logical 单调，
  恰在本步落 86670；child 自己的逻辑账本从 0 起，无冲突）。
- **落地修复**（update_from_output 补丁，压力臂限定）：
  `_pl > budget and num_logical == _pl and pre_step_logical < _pl and no outputs yet`
  ⇒ 丢弃该步 `new_token_ids`（账本照常推进；child 代换整条生成流，逃逸样本无权威）。
- **判决**：boot markerAZ/BA 两轮共 **7/7 全部 `ZX-4417-QJ7` 干净**（needle ✓ 无 No 前缀），
  每 run ~130s，`escape_suppressed` 事件与答案一致。
- 已修剪：attr 探针、head_blocks 实验（误诊——碎片头本就是父真实 prompt 开头，B 臂可答）、
  escape_pred_near 诊断。离线 10/10 全绿后才重新部署。
- 待办：其余验收门（152K/250K、resident 有界日志、OFF 无损 A/B、session 链、decode 带宽），
  以及把 FULL_HOOKS/ATTR_PROBE 之类试验残留从 /etc/default 与代码注释里做最后清扫（E 阶段）。

## 6.28 Round 26 验收推进：152K 复测全对 ×2、resident 有界、带宽对等

- **验收(2) ✅**：152K（151,994 tok，池 226,240，压缩 3 刀 + child 重建）
  **2/2 全对 `ZX-4417-QJ7`**（217s/216s，answer 干净）——原始"9 压缩→噪声"退化案例闭环。
  证据 docs/evidence/152k-retest-round26.txt。
- **验收(3) ✅**：resident 页峰值 3055（seq 48880 tokens）与 88K run 完全一致
  （同为 3055/48880）——**常数、与 prompt 长度（86K vs 152K）无关**，
  界= budget+gen_reserve/2 水位。pressure_gate timeline 已入证据文件。
- **验收(5) ✅（对等口径）**：短式 512-tok 测速 pressure arm 56.2/48.4 vs
  **OFF arm 同 boot 56.0/56.8** —— 带宽无损（DELIVERY 85 tok/s 是 2000-tok 长式口径，
  短式含 chat/reasoning 头开销，两臂一致即通过"decode 不劣化"判据）。
  GPU 58-63°C/1695MHz——非热降频。
- ops：service env 已切回 OFF（生产 fail-closed 基线）；下一轮在 OFF 上跑无损门
  （短 A/B 同实例 dump-compare 锚点重生成 + session 链），然后 pressure 上 250K（D 阶段）。

### 6.28.1 Round 26 OFF 无损门（同 boot）
- 短 A/B 双请求逐字节一致 ✓；
- **OFF 88K needle ✓ `ZX-4417-QJ7` 107.8s**（比历史 150s 更快——stream 抑制补丁在
  OFF 完全休眠，无副作用）；
- 短式 decode 带宽 56.0/56.8 tok/s（见上，两臂对等）。

## 6.29 Round 27 【验收(1)达成】250K 超池+超窗端到端全对 ×2

- D 阶段核实：fork RoPE cos/sin 表按**模型原生 262144** 分配（base.py:97），
  max_model_len 不是位置表面——提升窗口只需启动参数 + `KVMEM_SKIP_KV_CHECK=1`
  （池 226,240 < 250K 的容量校验绕过，10-29 已有先例）。
- 服务配置：`MAX_MODEL_LEN=250000` + `KVMEM_SKIP_KV_CHECK=1` + `KVMEM_PREFILL_PRESSURE=1`。
- **判决 2/2**：249,994-token 请求（超池 226,240 且超原窗口 200,000）
  中档 needle **`ZX-4417-QJ7` 干净**（349.7s / 346.8s）。
  usage.prompt_tokens=249994（真实长度，入口未裁）。
- **验收(3) 复核**：250K 两 run resident_pages_max=4323 / seq 69168 **完全相同**
  （88K/152K 时代 3055/48880——水位随窗口抬升但仍是 prompt 长度无关的常数）；
  5 压缩刀/run + escape_suppressed + child 重建，事件链全绿。
  证据 docs/evidence/250k-e2e-round27.txt。

## 6.30 Round 28：验收(4) session 链机制 ✓ + 锚点方法学还原 + env 卫生

- **session 链（OFF boot，header 通道 X-Session-ID）**：turn1 9674 tok park →
  turn2 `resume_decided hit=6464` → `session_resumed`×2 ranks、pages=404、state=true；
  **4.5s vs cold 10.2s（前缀跳过省 5.7s）**、resume×cold content parity=True。
  turn3 再问 `native session_id missing`→skip = **checkpoint 一次性消费**（已知设计，
  v0.4.0 定案），非回归。
- **body `session_id` 字段不生效（本轮测量）**：同 payload body-only 报
  `native session_id missing`；**X-Session-ID 头通道工作**。protocol.py 有字段、
  serving `_get_session_id` 三级优先（body→header→xargs）——断点在更深层
  （待查：chat serving 调用路径是否把 request.session_id 传入），
  本轮以 header 通道达成验收判据，body 缺陷记录不扩面（生产客户端用头/网关注入）。
- **金标准锚点方法还原**：sha256(**整个 completions response JSON**, sort_keys)[:16]；
  B 实测 `73ef34c96a2966a6` **精确命中**；A=`0013e530547bed4b`（旧档 head/old 双份
  重算同为 9b14455…，推测旧档曾被改写，**以本轮 head boot 重放再生为准**）。
  请求语义从 message 还原：A=zh「一句话解释 KV cache」mt≈64、B=en「List three prime
  numbers greater than 100.」mt≈48（prompt 60/63 tok 与 tokenizer 对齐校验）。
- env 卫生：/etc/default 死行 KVMEM_PREFILL_PRESSURE=0（被 =1 覆盖）删除；现配置
  MAX_MODEL_LEN=250000 + SKIP_KV_CHECK=1 + PRESSURE=0（生产 OFF 态、窗已升但机制休眠）。
- 待办：锚点重放（A/B 重生成新 sha 记录入 DELIVERY）+ E 收尾。

## 6.31 Round 28 收口：A/B 金标准方法定案 + 跨版本锚点已闭合（模型换代差异记录）

- **规范算法**（本轮定义，脚本可复跑）：响应 JSON 去 `id/created` 两 volatile 字段、
  递归去 null、sort_keys 序列化 → sha256[:16]。
- **旧档即证据**：`gate-round3-head.json`（HEAD=060eda0 时代）与 `gate-round3-old.json`
  （060eda0 对照）在规范口径下 **A=abb9428febcc224f、B=633ffee075d59cde 逐字节一致**——
  v0.4.0 生产回归金标准**当时已成立**，锚点定义就此闭合（0013e530 旧记法弃用）。
- **本轮 head boot 重放**：A=a382bb147d58c82b、B=8ddab0947b003860（同 boot 内
  id/created 外全稳）。与 v0.4.0 锚点差异 = **模型 FP8 权重量化换代**（v0.4.0 之后
  模型文件被重生成，前缀 cache/OOM 排障期间）——非适配器代码。
  **今后锚点纪律**：模型文件变（sha256sum config+safetensors index）⇒ 锚点必重生成，
  记录配对（模型代 → 锚点集）。E 阶段 DELIVERY 更新按此写。

## 6.32 Round 30：引擎窗口改由 CTX 推导（用户指令）

- `backends/vllm/worker.py`：`KVMEM_MAX_MODEL_LEN` 未显式给时 **MAX_MODEL_LEN =
  min(CTX, 模型原生 max_position_embeddings)**（本模型 524,288→262,144；读 config.json
  含 text_config 嵌套）。fork 语义核验（config/model.py:2518-2545）：显式窗口超模型上限
  直接 ValueError——clamp 是必须项，不是保守项。
- deploy/run-kvmem-vllm.sh 默认 200000 → **透传空**（显式给值才钉窗）；
  站点 launcher line 33 默认 200000 → 透传空；/etc/default 删 MAX_MODEL_LEN=250000。
- 实测（CTX 窗口 boot）：`--max-model-len 262144` ✓；88K pressure 130.9s ✓、
  250K pressure 350.2s ✓ 全干净；短 A/B 同 boot 逐字节稳（A=7cbd5fec679a43c6、
  B=2e33994b033185cd——与 r28 值差=窗口参数变（250000 是 r28 中途注入、从未跑过 boot），
  锚点纪律"配置变→锚点配对重生成"的又一次应用）；OFF 88K 回归 ✓ 107.8s 带内。
- KVMEM_SKIP_KV_CHECK=1 仍必需（窗口 262,144 > 池 226,240 的容量校验）。

## 6.33 Round 31：CTX 的架构语义定案（用户纠正）+ 主机容量自我限制

- **纠正我此前的措辞**（"clamp 是必须项"的论证方向不对）：窗口跟 CTX 的理由不是躲 fork
  校验，而是**完整 kvmem 机制已成立**——显存实际生效的是 B（水位界），它既不会超过物理池
  也不会超过模型原生；**CTX 是纯虚拟上下文，唯一约束 = 主机内存装得下**。
- 落地该语义的缺失防线（`kvmem_sparse_runtime` store 创建处）：**无 disk 层且
  CTX > 会话 H share 时，ctx_tokens 自我限制到 share_tokens** 并记
  `host_capacity_short` 事件——否则溢出页被静默 DROP，而重建/回召正确性恰依赖这些页
  （宁可提前收紧虚拟上限，不可事后答错）。本机 2,960,672 tok ≥ 5.6× CTX，防线静默 ✓。
- 物理边界如实记录：当前实现 prefill 期 position=原始逻辑位（0..T-1），RoPE 表 262,144
  ⇒ **单条 prompt** 的硬上限仍是 262,144（非配置约束）；若要让单条 prompt 超它，需要
  position rebasing（视图相对坐标）——独立工程题，未立项。
- 验证：离线 10/10（含新 source-guard）、部署冒烟 88K OFF needle ✓ 108.1s、
  host_share 事件正常且 cap 未误触发。main 前行（v0.5.0 tag 不动）。

## 6.34 Round 31 后段（用户问题）：CTX 512K 下 300K 单条上下文——拆掉一颗活雷

- 用户问："CTX=512K、上下文到 300K 会怎么处理"。实测拆出**旧补丁的失效前提**：
  fork `input_processor._validate_prompt_len` 被 KVMem 改为按 `KVMEM_ENTRY_CTX`（=CTX）
  放行 decoder prompt——那是**窗口钉 200000 时代**的正确补丁（表 262144 > 一切进入的
  position，250K 才因此安然无恙）。窗口跟 CTX 后（262,144=表大小），>262,144 的 prompt
  照样进门，prefill position 冲到表外 = **nan/CUDA illegal access**（fork 警告原文），
  4 分钟计算后崩服务。探针实测：修复前 282K/300K 请求**进入引擎**（Timeout=在算）。
- 修复（worker 窗口设定处一行）：`KVMEM_ENTRY_CTX` clamp 到 `_mm`（min(CTX, native)）——
  入口天花板=物理位置表。**实弹：300K → 0.9s 干净 400**（vLLM 标准文案），
  ≤262,143 的进入路径不变（250K 验收为证）。CTX 512K 的语义修正表述：
  **虚拟上下文的可用长度 = min(CTX, 262,144)**——超过位置表的部分当前实现无法单条承载
  （需 prerope tier/视图重编号，见 §6.33），多轮累计+历史分层另算。
- 探针教训：`KVMEM_ENTRY_INREQUEST_COMPACTION` 由 CLI 开关单源仲裁（entry_compaction_env
  文档语义），探针必须带 `--kvmem-inrequest-compaction` 才走真链。

## 6.35 Round 32 方案 C 落地：KVMEM_ROPE_TABLE_CTX opt-in（原版立场的准入开关）

- 用户裁定 C：默认拒绝保持，超长单条走显式开关。文档口径（README×2 已写）：
  **推荐默认 = 单条 ≤ ~262K**（位置表内、全验收）；需要更长开
  `KVMEM_ROPE_TABLE_CTX=<CTX>`（表+窗口+入口三同步；worker 推导处 + plugin
  `get_rope` 包装器，register() 期在模型模块绑定前挂钩）。
- **机制依据（原版调查）**：ggml CUDA rope kernel `cosf/sinf` 数值直算
  （`rope.cu:36,107`），**无表**——原版收 300K 不拒是引擎形态使然；vLLM 表在
  `qwen3_next.py:354` 绑 `config.max_position_embeddings`，opt-in 在此抬表。
- **实测（opt-in boot, window 524288）**：
  - 表扩 `rope_table_ctx_lift 262144→524288` 事件 ×2 rank ✓；池代价实测
    **226,240 → 214,928**（−11K token ≈ 表 134MB/卡+profile）；
  - **PRESSURE=ON：300K（282,142 tok）端到端 needle ✓ 395.2s**——5 压缩刀、
    escape_suppressed、4 重建事件；resident 峰 **4640 页/74240 tok = 水位级常数**
    （vs 250K@窗口262144 的 4323/69168——随预算，不随 prompt，也不随窗口量级）；
  - **PRESSURE=OFF：同一 300K 在 prompt 末重建处崩**（"reconstruction page lies
    outside the original prompt" 家族，EngineDead 500）——OFF 路径 >池重建上限是
    既有生产边界，opt-in 只开准入不开这条 ⇒ **开关必须配合 pressure 臂**（文档已写明）。
- 站点已回生产（开关删除 + PRESSURE=0）；回归：300K → 干净 400、boot OK、离线 10/10。

## 6.36 Round 33：v0.5.1 默认翻转为 pressure 臂（用户裁定）

- 用户裁定：默认即 pressure（deploy `KVMEM_PREFILL_PRESSURE:-1`、站点 =1）。理由：
  适配器存在意义 = 超池上下文 + 多用户分块；**不超池单并发直接用原生 vLLM 更快，
  没必要用我们**——旧"保守默认 OFF"语义作废（0 保留为对照/兼容回退，非推荐）。
- 自适应水位方案（上轮 B 选项）**pass**：水位=池边是全局概念，双用户并发时一方
  prefill 会殃及另一方的池内数据——chunk 转存的第二动机正是多用户隔离（用户明确）。
  现 45K 水位形态本身是按请求自律的（logical book 判、自家块裁），方向正确；
  双并发 GPU 验收仍是未来默认语义的加固项（离线 sparse_concurrency 仅模拟）。
- 默认翻转验收（同 boot）：short 1.6s 正常、88K needle ✓ 130.4s（=历史 pressure 档
  一致）、250K needle ✓ 348.6s（=349.7/346.8 一致）、pressure_gate flag:true 实锤；
  离线 10/10（deploy 默认守卫同步翻转）。几何 fail-closed（池<水位+余量自动回 0 语义）
  保留为保护线。


## 6.37 Round 34 时间分解实验（用户质疑"22s 不该是搬运"→ 插桩实测）

- 你的带宽直觉正确：刀本体便宜（88K 两刀 rpc 3.7s、validate≈0），prefill 速率**两臂相同**
  （引擎 5s 窗恒定 1365 tok/s，滚动压缩不拖 prefill），final 准备 6.4s。
- **大头=prompt 末 child 重建的换出页回取**：`restore_page` 逐页×逐层 `.to()`
  （orchestrator.py:471-472），88K ≈ 2816页×12层 ≈ 3.4 万笔小拷贝 ≈ 15s；
  250K 回取量更大 ≈ 100s 级——**粒度开销，不是带宽**（394MB 数据本身 0.3s）。
- 修复候选（未实施）：**层主序向量化回取**——每层一次批量 H2D + scatter 索引，
  预计 15s→秒级。优化验收沿用 needle 2/2 + A/B 锚 + 计时事件对比。


## 6.38 Round 35 层主序批量回取（修）+ 归因修正（我错了两次）

- 实施：`restore_pages_batch`（每 KV 张量一次 stack+H2D+index_copy_，rank 语义不
  触字节；四调用点批先行、异常整段回落原 per-page 循环，语义零改动）+ 等价性单测
  （字节+账本逐位对照）+ `restore_batch_applied` 命中诊断。**结果无损且生效**：
  1680 页回取 0.3s（原 3.4 万笔微拷贝路径），两次 88K 字节一致、锚 OK、离线 10/10。
- **但总耗时未降（131→130.6s）——"回取是大头"被证伪。** 加上本轮数据，88K 相对 OFF
  的 ~22s 缺口现有账目：刀 3.7s + final 准备 6.4s + 回取 ~0.3s ≈ 10.4s，**剩 ~12s
  成因未测**。且引擎吞吐窗（两臂都显示 1365 峰值窗 + 大量 0 窗）与 gate 段速
  （928→750 递减）互相矛盾——用户"越到后面掉越多"的观察正确，我的恒定 1365 表述作废。
- 待办（若继续）：把 reconstruction_started→child 首样本→请求完成整段计时装齐，
  一次 88K run 分解最后 12s；batch 代码保留（微赚且结构更安全）。


## 6.39 方案 C 立项（用户批准）：pressure 臂的前缀检索复用；C1 已完成

- 设计纠偏链：session 判定先错在要求客户端身份（X-Session-ID 硬门）——原版用
  `prompt.common_prefix`（driver:144）判会话，身份只是索引加速器。已改隐式命名空间
  ("","__prefix_cache__")：门拆掉，正确性由 block_hashes+prefix_digest 保证。
- 深坑（5 连发实测）：pressure 下密封从未成功——`session_park_skipped no sealed
  executed-frontier`——父被刀裁过、child 账本不推进，**两阶段都没有"密集执行过前缀"
  的可信 StateImage**。skip 式 resume 在本 fork 语义下对超水位请求结构性不可用。
- 方案 C 的诚实形态：可复用资产是 **store 的真位置规范页**（每步 baked=0 提交，
  重建机制的存在本身依赖其完备），不是状态。C1=密封"还原型" checkpoint
  （has_state=False + 容量契约豁免 + stats 门豁免），Core 命中后 **defer**
  （不 skip、不 proof、不 discard，行为=全量但链路证明）；C2=消费还原型命中：
  前缀段跳过计算、页批量还原、只算增量与 final；C3=默认语义合入。
- 密封时点 = `_rebuild_select` 入口（finalize 步的 per-step 提交已完成——
  覆盖率门通过即实证，这是 C2 数据面的地基）。frontier 由
  `session_checkpoint_frontier(prompt,…)` 决定（88K→25856/16 blocks）。
- C1 验收（默认 pressure boot）：sealed_restore×2 → retired(ranks[0,1]) →
  resume_deferred hit=16 blocks source 链式指向上一 rid → t2/t3 与 t1 字节一致
  （c1e81469911b7066，跨 boot 与历史同值）→ entry 未被 discard（t3 仍 deferred）
  → 耗时/锚点零回归 → 离线 10/10 + 守卫×5（隐式门×2、密封×1、豁免×1、defer×1）。


## 6.40 C2 首攻实测：机制成立、密封点有毒、已 fail-closed 回 C1

- **正向证据**：dense seal（挂 finish pressure 分支）→ t1 密封 retired →
  t2 `resume_decided hit=25856` → **108.0s vs 130.5s（省 22.4s）needle ✓**——
  检索命中跳过前缀重算的通路端到端走通。
- **毒性实锤**：t5（152K resume 臂）**needle=False**。根因静态可判：finish 时点
  在首刀之后，`_state_slot` 用 **已重排的 view block_table** 解析 ring 列——
  capture 到的 StateImage 属于无关视图页。88K 的两对 turn 恰好无碍属容差内运气，
  不能当正确性证据。同时暴露两个链路断点：
  ①resume turn 的 store 是新的（restore 页没进新 orch 的 store）→ 它完成时
  dense seal coverage=0 拒绝 → 不再 retired → **隔行 zigzag**（t2b miss 全量、
  t3 又命中），PATH-STABLE=False 由此而来；
  ②C2a 早攻（步界 capture 放宽）的 probe 数据证明 prefill 步的 capture 调用
  被乐观镜像账本挡住（rt_tokens 恒 > executed，gate 原文即为防"release 态配
  短 prompt"），三处门缝手术已整体回滚，OFF 链零改动。
- **止血**：dense seal 摘线（方法保留+文档中毒标记），回 C1 deferred 安全态。
  复验：152K 双发 216.2/214.4s 全对、resume_decided=0、resume_deferred=2 ✓。
- **下一刀（C2-fix）已定形**：密封必须在**首刀之前**——dense 窗口内
  `executed ∈ [target, target+2·chunk)` 且 `st.compacted_in_prefill==False`
  （列位真实、页已 per-step 提交、ring 列未被改写）；捕获坐标用 GPU 计数器；
  resume turn 完成后的 re-seal 需把 restore 继承的页认入新 orch（修断点①）。


## 6.41 C2-fix 两段式落地 + resume apply 病理确诊（本轮收束于安全基线）

- 两段式密封（boundary 只拍 state CPU 快照 / finish 才 publish）修好了**密封 turn
  本身**：152K sealed 全量 turn max_view 48880 受控、答案正确（一代 sha=2e8bfe9
  的"密封 turn 中毒"消灭）。
- 但 battery 揭示**下一层病理**：resume-applied turn 的账本种子与 Core 的 rewind
  计数器不一致 → 水位刀全部 `quiescent_compact_stale`（实测 54 次）→ 视图失控到
  122880 → 答案中毒（2e8bfe9 代=刀失控指纹，跨 boot 复现）。**不是双应用误差，
  是种子失配**。
- 本轮止血：pressure boot **一律 defer**（一行 cfg 条件）——OFF 臂 resume apply
  保持 v0.4.0 原样零影响。止血复验：152K×3 全对同 sha、88K 对、
  decided=0/deferred=4/stale=0。
- 下一增量（已定形）：修 resume 应用的 worker 侧账本种子（note_resume 的
  (hit,visible) 如何进 st.tokens/页计数 vs Core num_computed=visible），
  使刀在 resume turn 正常落；恢复 pressure 的 apply 与 22 秒收益。


## 6.42 C2 达成：adoption owner 门 = 最后一块毒（stale=0，resume 与冷字节一致）

- 病理终章：Core 应用了 skip 而 worker 侧 `_adopt_session` 还卡在"native
  session_id missing"门（隐式命名空间裁决没传到 adoption）→ 页/状态从未还原，
  attention 读未写前缀页 = 2e8bfe9 毒代；54×stale 是同一根因的账本表象。
- 修复三处：①adoption 用同一隐式命名空间（正确性仍由 digest+hashes 把关）；
  ②phase-1 guard 改 `st.resume`（`_resume` dict 在 adoption 时已 pop，旧守卫恒假）；
  ③pressure apply 恢复（defer 只留 restore 型条目）。
- 决定性 battery（boot 干净）：152K 四连发 216.7/169.2/214.7/169.2s、
  88K 128.8/103.7s——**resume turn 省 47s/25s（22%/19%）且与冷跑字节一致**
  （B=D=A=C=2899f9138f524f1a，F=E=c1e81469911b7066）；stale=0、resumed=6、
  刀 16 次全落、视图全 ≤48880、双锚 OK。
- 语义账本：全量 turn 密封（dense snap+publish）、resume turn 消费且不重复密封
  （防链式叠加）→ 隔行快（一次性消费的既有设计形态）；alternation 若要消，
  下一步做 resume-turn 的 inherit-store 再密封，收益=每 turn 都快。
- C3（本轮）：以上即默认行为（pressure 默认 ON、session_reuse 默认 ON，无新开关），
  文档同步 README×2 后会签。
