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
