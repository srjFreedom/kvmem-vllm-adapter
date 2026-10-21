# 稀疏选入（A 模式）设计笔记 — 实测结论，不是推测

## 1. vLLM 的注意力没有 per-key 位置

`unified_attention(q, k, v, out, cu_seqlens_q, seqused_k, ..., block_table, ...)` 只接受
`block_table` + `seqused_k`：块表第 j 槽 = 逻辑位置 `j*16 + offset`。kvmem 自己后端能"保留原
RoPE 位置"，是因为算子是自研的。所以在 vLLM 上做稀疏视图只有**压缩 + 重打 RoPE**一条路。

## 2. RoPE 参数（从引擎实测，不是从配置推）

| 项 | 值 | 来源 |
|---|---|---|
| head_dim | 256 | config |
| rotary_dim | 64 | `partial_rotary_factor=0.25` |
| theta | 1e7 | `rope_parameters.rope_theta` |
| 配对 | NeoX split-half | 逐对解出角度 = `inv_freq[i]*delta`，与 split-half 完全吻合 |
| M-RoPE interleaved | 纯文本下等价恒等 | 三段位置相同，`apply_interleaved_rope` 不变 |

**只转 K、只转前 64/256 维**，所以重旋转成本是整块 K 的 1/2（K 占内容一半）× 1/4（rotary 占比）。

## 3. fp32 cos/sin 表在长位置早就丢精度

引擎自己的 `cos_sin_cache` 与 float64 真值对比：

| position | cos_err | sin_err |
|---|---|---|
| 1,234 | 3.5e-05 | 1.0e-05 |
| 200,000 | **3.7e-03** | 2.7e-03 |

所以重旋转必须用 **float64** 内部计算，否则是给引擎再叠一层误差。大位置处与引擎不逐位相等是
引擎自身精度，不是实现错误（测试里已按此判定）。

## 4. fp8 量化的两个性质（决定了压缩策略）

1. **旋转与 scale 可交换**：旋转只在头内做，是线性的，所以 per-tensor 或 per-token-head 的
   正 scale 都不影响——可以直接旋转 fp8 字节，不需要知道 scale。我们的缓存 `(8686, 2, 16, 512)`
   = `K(256)|V(256)` 无内联 scale 填充，即 per-tensor。
2. **每次重量化都有代价**：e4m3 只有 3 位尾数。实测随机量级 0.5 的 K：
   - 从 canonical 旋转一次：相对误差 **0.056**
   - 在已旋转过的页上再旋转（链式）：**0.071**

因此规则是：**设备上的 K 永远从 canonical 副本派生**（canonical 存主机池，就是溢出的那份），
绝不"转已经转过的设备页"。这样误差不累积，但每次重选仍会多一个量化台阶——注意力看到的是
≈2× 的 fp8 K 噪声。这正是"稀疏 vs 全量质量对照"要量的东西。

## 5. 备选：内核原生 prefix+window（零拷贝、零精度损失）

`unified_attention` 自带 `mm_prefix_range`（PrefixLM 前缀始终可见）和
`rswa_prefix_lens`/`rswa_window`（前缀全局 + 生成段滑动窗）。这条路 = **sink + recent，不搬
任何数据、不重打 RoPE、不重量化**，但**选不了中间块**（没有检索）。如果以后需要一个"廉价
模式"，这就是；要检索式选入（kvmem 的核心），必须走压缩。

## 6. 重选成本（每 rank，按 B=32768/R=16384）

| 项 | 量 |
|---|---|
| 重选页数 | 2048 页（B/16） |
| K 重旋转字节 | 2048 × 16 × 2 头 × 64 维 × 1 B ≈ **4.2 MB**（只 rotary 段） |
| 若从主机还原 canonical | 2048 × 272 KiB ≈ **557 MiB H2D**，PCIe3 x16 约 47 ms |
| 摊到 64 步 | ≈ 0.7 ms/步 |

## 7. 已固化为测试的结论

- `tests/sparse_rope_test.py`：与引擎 `MRotaryEmbedding` 逐位对齐（中等位置 5e-6），大位置按
  引擎自身精度判定，往返可逆，非 rotary 通道不动。
- `tests/sparse_rope_fp8_test.py`：单跳 0.056、链式 0.071（证明不能链式）、V 字节不动。
- `tests/sparse_selector_test.cpp`：B=32768 → 2048 页，sink/recent 必留，中间按分数选。

## 8. 稀疏视图契约（`src/kvmem_sparse_view.py`，已单测）

压缩后该请求的块表行 = `[选中页(升序) | 新页]`，逻辑长度 `seq_len = |sel|*16 + n_new`。

关键简化：**只要把该请求的 `positions` 覆盖成压缩后位置**，引擎写入新 token 的 K 时就按压缩位
置打好了 RoPE，查询也落在压缩位置上。所以**只有"上一轮已写入的选中键"需要重打**，新键天然自洽。
重打量 = `(新槽位 - 原页号) * 16` token。

实测（200K 会话、B=32768、sink 8192 + recent 8192 + 中间 1024 页）：**2048 页里 1536 页会移动**
（sink 前缀 512 页原地不动）。所以每次重选：

| 项 | 量 | 备注 |
|---|---|---|
| 需重打 RoPE 的页 | 1536 页 | 只转 rotary 段（64/256 维），≈3.1 MB |
| 需从 canonical 还原的页 | 1536 页 × 272 KiB ≈ **408 MiB H2D** | PCIe3 x16 约 34 ms |
| 摊到 64 步 | ≈ 0.5 ms/步 | |

`tests/sparse_view_test.py` 覆盖：行拼接、seq_len、位置连续性、无重复页、每页位移量、
以及四类非法输入（重复页/降序/新页不足/页兼作键与查询）全部拒绝。

## 9. 编排器（`src/kvmem_sparse_orchestrator.py`）与离线端到端验证

四条不变量固化为代码：

1. 主机页库保存**每个已写页在真实位置的 canonical 字节**；
2. 设备页 = canonical 经**恰好一次**旋转 `(slot - page) * 16`；
3. 引擎在**压缩后位置**写入新 token（因为 `positions` 被覆盖），落盘时反向旋转一次转回 canonical；
4. 重选**从不**旋转已带旋转的设备页，一律从 canonical 重建 —— 所以 fp8 重量化误差不累积。

`tests/sparse_orchestrator_test.py`（离线，模拟 200K 会话 + 3 轮 decode 重选）：

```
ok  canonical pool holds all 12800 pages
ok  working set shrinks to B/16 pages (got 2048)
ok  the rest is evicted (evicted=10752)
ok  decode step 7/15/23: resident tracks the selection
ok  canonical bytes are byte-identical to what the engine wrote
ok  every resident page is one rotation of canonical (bad=[])
ok  resident 2048 <= budget 2048
ok  the sink prefix stays resident
ok  a repeated reselect writes identical bytes (no extra hop)
PASS sparse_orchestrator_test
```

"重复重选写入相同字节"这一条很关键：它证明**同一次选择重复执行不会多叠一个量化台阶**。

## 10. 接线层（`src/kvmem_sparse_runtime.py`）— 逻辑页 vs 物理槽

接线时踩到一个必须写死的区分：

- **选择器与主机页库按"逻辑页号"工作**（真实 token 页 = position // 16），因为选入是按对话历史算的；
- **块表里必须写"物理槽位"**，kernel 读的就是槽位号。

压缩后逻辑页 i 被放到槽位 i，所以行 = `[0, 1, ..., |sel|-1, spare]`，逻辑页另存一份对齐表用于
把新写入的 token 归属回历史页。两者混用会静默读错数据——这条现在有测试守着。

另外两个接线细节：

1. **prefill 期间不做稀疏选入**。选择器按 token 前沿工作，会选到"最后一个只写了一半的选入块"
   扩展出的尚未写入的页；kvmem 自己也用独立的 `pick_prefill_pressure_blocks` 处理这个阶段。
   所以运行时空闲时才在 decode 阶段触发重选，并过滤 `page*16 >= tokens` 的未写页。
2. **`seq_len` 是压缩后的长度**（≈32.7K），不是真实历史长度（204.8K）。

`tests/sparse_runtime_test.py`（离线，fake runner 模拟引擎写入）：

```
ok  prefill tracked 204800 tokens
ok  working set shrinks to 2048 pages
ok  the row is the selected pages plus one spare
ok  resident stays within budget
ok  needle exact on every resident page
ok  store holds the canonical history (12802 pages)
ok  compacted seq_len stays near B (got 32672 for 204824 tokens)     ← 6.27x
ok  the compacted view never exceeds the true history
ok  row matches the tensor / seq_len matches the tensor
ok  reselect happened on schedule (4 times)
PASS sparse_runtime_test
```

## 11. 真 runner 适配（`src/kvmem_live_view.py`）与插件两个 hook 点

`LiveRunnerView` 把 V2 runner 接到运行时协议上：

| 运行时需要 | 真 runner 上 |
|---|---|
| 分页 KV | `kv_caches` 里 `FullAttentionSpec` 组（17 个），**不含** 48 个 Mamba 状态块 |
| 块表行 | `runner.block_tables.block_tables[g].gpu`（只取注意力组） |
| `seq_lens` | `runner.input_buffers.seq_lens` |
| `positions` | `runner.input_buffers.positions` |
| 请求→行号 | `idx_mapping`（批次位置→行号） |
| 请求→位置槽 | `query_start_loc[i]:query_start_loc[i+1]` |

插件里的两个 hook 点：

1. `prepare_pos_seq_lens`（引擎刚算完 positions/seq_lens 之后）→ 装稀疏视图；
2. `execute_model` 返回后 → 提交本步写入的页 + 到点重选。

两者都是"包一层"实现，不改 vLLM 源码。`tests/live_view_test.py` 用假 V2 runner 验证了组选择、
`idx_mapping` 行映射、`query_start_loc` 位置槽与写透。

## 12. 现在的状态

离线部分全部完成并有测试：选入、视图契约、重打 RoPE（含 fp8）、编排器、接线层、真 runner 适配、
插件挂载（`"sparse": true` 即启用，默认关闭）。**唯一未做的是活引擎验证**，需要独占两张卡。

## 13. 并发安全：槽位必须按请求互斥（一个会毁掉 4 并发的缺陷）

代码复查时发现：编排器把选中页放进槽位 `0..|sel|-1`，**每个请求都从 0 开始**。单请求测试全绿，
但两个并发请求会互相覆盖对方的 KV——而且是静默覆盖，只在输出质量上体现。

修正为**共享池 + 槽位所有权**模型（贴近真实：vLLM 的页表本来就是共享池）：

- 运行时维护 `slot_owner`（槽位→请求）与每个请求的 `owned_slots`；
- 每次重选，请求先归还自己的槽位，再从空闲池取 `budget_pages + 1` 个**不与他人重叠**的槽位；
- 编排器接受显式槽位列表，不再假设从 0 开始；
- prefill 页也从同一池里分配（真机上就是引擎分配器）。

`tests/sparse_concurrency_test.py`（两个并发 64K 请求，交错 decode 后多重选）：

```
ok  the two requests use disjoint device slots (shared=[])
ok  each working set is the budget plus a spare (A=2049, B=2049)
ok  slots stay disjoint after reselects
ok  req-a / req-b: needle exact on 2042 pages (bad=[])
ok  req-a / req-b: resident within budget
ok  each request keeps its own canonical page store
PASS sparse_concurrency_test
```

## 14. 活引擎首轮结果（V1 通过）与两点关键核实

真引擎（TP=2、Coder390 FP8、204800 上下文、关前缀缓存、稀疏开启）上一条 ~127K token 的请求：

| 指标 | 实测 |
|---|---|
| 重选次数 | 354 次，**0 错误** |
| `resident`（常驻页） | 恒在 2041–2048（= B/16 上限） |
| `seq_len`（kernel 实际读） | 恒在 32,656–32,768 |
| 对话真实长度 | 涨到 143,440 token |
| 压缩比 | ≈ 4.4x（目标 4.2x） |
| prefill 压力选择 | 正常（把 4,096 页压到 2,048）|

**核实 1：`slot_mapping` 不需要单独接管。** 它由 `_compute_slot_mappings_kernel` 按
`positions` 计算（`block_indices = positions // kernel_block_size`），而我们已经把 `positions`
覆盖成压缩后位置，所以**新 token 的写入槽位自动与压缩视图一致**。这是之前最大的未知项。

**核实 2：token 计数没有虚高。** trace 显示每步 `tokens` 与引擎自己的
`req_states.num_computed_tokens_np` 相差 ≤5（投机解码的草稿数），且两个 TP rank 各写一条
对称记录（不是 hook 重复执行）。

## 15. 上卡过程中修掉的插件缺陷

1. **异步调度下 hook 被跳过**：V2 runner 在 `sample_tokens()` 取样，`execute_model` 返回
   `None`；但 KV 写入已发生，hook 必须照跑（原来 `if result is None: return` 是错的）。
2. **压力重选后 prefill 断裂**：行变非空被误判成 decode，跨页不追加槽位 → 改为行增长无条件，
   prefill 由"本步调度 token 数 ≥ 一页"推断。
3. **trace 自身的两个 NameError**（`log_path`/`runner` 写错作用域）→ 日志代码现在独立 try，
   **永不阻断推理**。
4. **GPU_UTIL=0.94 放不下 204800**（KV 3.09 < 3.72 GiB）→ 必须 0.98，与现网一致。

## 16. 关键障碍：`seq_lens` 与 GDN 状态寻址共享（活引擎 A/B 实测）

**决定性 A/B**（同一引擎、同一提示词、同一次启动方式，只切 `sparse`）：

| 配置 | 回答（题目：item 100 / 600 / 1190 的编码） |
|---|---|
| `sparse: false`（基线） | `ZQ00100 ZQ00600 ZQ01190`，finish=stop —— **全对** |
| `sparse: true` | 无意义 token 串 —— **崩坏** |

同时 V1（常驻 2048 页 / seq_len 32.7K）与 V4（4 页埋针全过）都是通过的，说明
问题不在页搬移与旋转本身，而在**元数据**。

**根因**：这个模型 64 层里 48 层是 GDN（线性注意力），它们的递归状态块索引来自**共用的**
`common_attn_metadata.seq_lens`：

- `mamba_attn.py:441` — `(seq_lens + mamba_block_size - 1) // mamba_block_size` 决定状态块号；
- `mamba_attn.py:492` — `has_prior_state = seq_lens_cpu > query_lens_cpu`。

而 `build_attn_metadata()` 对**所有 KV group 只传一份 `seq_lens`**。我们把注意力视图压到
32.7K，等于同时改了 48 层 GDN 的状态寻址 → 状态读错块 → 输出崩坏。**这不是搬 KV 的问题，
是混合模型里"注意力的 token 视图"和"线性注意力的状态索引"不能共用同一个长度的约束。**

## 17. 下一步修法（已定，未实施）

不能事后改元数据：FlashInfer 的 paged-kv indptr / last_page_len 是在 **build 时**由 seq_lens
推导的。所以要**按 group 分别喂 seq_lens**：

- 找到 `attn_groups` 里 `FullAttentionSpec` 那一组的 metadata builder 实例；
- 包一层它的 `build()`，把传入的 `seq_lens` 参数替换成**压缩后**的张量；
- Mamba 组的 builder 保持原样，继续拿真实 `seq_lens`。

这样只影响注意力组的 kernel 视图，GDN 的状态索引保持正确。落地后重跑这组 A/B：
基线三码全对，稀疏版至少要保住 sink（item 100）与 recent（item 1190）。

## 18. 元数据二分结论：破坏源是「块表行写入」，且页号起点不是 0

**二分实验**（同一提示词、同一引擎、逐项开关写入）：

| 配置 | 输出 |
|---|---|
| `install_root: false`（全不写） | `ZQ00100 ZQQ00601 ZQ0110` —— 基本正确 ✅ |
| 行+seq 开、pos 关 | 乱码 ❌ |
| 行关、seq+pos 开 | `ZQ0000000000110ZZQ01¹Q01200`（token 真实，finish=stop）✅ |

**结论**：破坏来自我们改写**块表行**，不是 seq_lens、不是 positions、不是旋转/搬页。

**为什么**：引擎自己的行是 `[101, 102, 103, ...]` —— 真实页号从 **101** 开始。原因是
`mamba_cache_mode=align` 下 vLLM 的 NULL 块占满一个**对齐单元**（1616 token / 16 = 101 页），
所以可用页从 101 起。我们的槽位池从 0 开始分配 → 整行错位 101 页 → 注意力读保留区 → token 汤。

把 `first_page` 改成 101 之后**仍然乱码**，说明还有第二层不一致：我们的页号是"逻辑页"，
而引擎的页号是它自己分配器给的物理页；两者即使都从 101 起，**顺序与数量也不保证一致**
（实测引擎该请求有 1818 个非零页，我们选中 1815）。

**正确的做法（下一轮实施）**：不要再自建池。重选时**直接取引擎自己那一行的页号**作为候选集：
`engine_pages = row[:n]`，在其中选出工作集，并把它们**重排到该请求自己页区间的头部**
（`row[:|sel|] = selected`）。这样：页号天然有效、不与其他请求冲突、也不需要 `first_page`。

## 19. 之前的 V1/V4 为什么没抓到

- V1 只统计页数与 seq_len（与页号无关）；
- V4 埋针只验证"设备页 = 我们页库的一跳旋转"（自比对，自洽即通过）。

两者都对**页号映射**不敏感，所以只有"稀疏 vs 全量"的**输出对照**能暴露它。这条已写进验证清单：
任何改动后都必须跑一次 A/B 输出对照，不能只看页数和埋针。

## 20. 两个真根因（修好后稀疏输出与基线一致）

### 根因 A：页号必须取自引擎自己的行，不能自建池
引擎的行是 `[104, 105, 106, ...]`（`mamba_cache_mode=align` 下 NULL 块占一个对齐单元 = 101 页，
所以首个可用页不是 0）。我们曾自建从 0 或从 101 开始的池，两次都错——因为引擎页号的**顺序与数量**
也不保证与我们一致（实测该请求 1919 个非零页，我们选中 1833）。

**修法**：`page_map[logical] = 引擎行[logical]`，重选时只在这张表里做重排，不再自建池。

### 根因 B：一个逻辑页跨 17 个分页注意力层，而运行时只接了第 0 层
块表是所有注意力层共享的，所以压缩后 17 层的 K 都必须同步搬移 + 重打 RoPE。此前只处理
`paged_kv()[0]`，另外 16 层的 K 停在原位置 → **必然乱码**。

**修法**：`RunnerView.paged_kv()` 返回全部 17 个张量；页库每条记录是 17 个张量的列表；
旋转/校验逐层进行。

**验证**（29K token、B=32768，恒等视图）：
```
sparse: false → ZQ00100 ZQ00600 ZQ01190   (finish=stop)
sparse: true  → ZQ00100 ZQ00600 ZQ01190   (finish=stop)   ← 完全一致
```

### 附带修掉：页库曾把 GPU 克隆当主机副本
`store.put(kv[slot].clone())` 在 GPU 上持有 2000 页 × 17 层 → 39K token 的请求直接把引擎 OOM。
现在 `put` 一律 `.to("cpu")`，还原时逐层搬回设备。

## 21. 长上下文实测（V1 在 64K / 185K 上都成立）与一个性能瓶颈

**V1 在长上下文上确认**（真引擎，B=32768）：

| 对话长度 | 常驻页 | seq_len | 重选次数 | 错误 |
|---|---|---|---|---|
| 64K token | 2,044–2,048 | 32,704–32,768 | 数十次 | 0 |
| 185K token | 2,048 | 32,768 | 数十次 | 0 |

即 185K token 的对话在显存里只占 32,768 token —— **5.6x 压缩**，且全程无错误。

**新发现的性能瓶颈**：`commit_written_page` 每步都对 17 层各做一次**同步 D2H**（272 KiB），
强制每步一次 GPU 同步；加上重选时 2,044 页 × 17 层 = 570 MB 的 H2D 与逐页小 kernel 旋转，
decode 被拖到每 token 数秒级。

**优化方向（下一轮）**：
1. 不要每步提交：页在重选前仍驻留设备，重选时再一次性读取（批量 D2H + 异步）；
2. 旋转改为**整批**（把选中页堆成一个张量做一次 delta_rotate，而不是逐页 17 次调用）；
3. 重选间隔从 64 再放宽，或按 token 数触发。

这两项都是纯性能改动，不改变语义。

## 22. 性能优化与一次真实检索验证

### 批量旋转（34,000 次算子调用 → 17 次）
原来每页每层做一次小旋转：2,044 页 × 17 层 = 约 3.5 万次 kernel 调用。改为**按层堆叠后一次
批量旋转**（每页 delta 不同，用逐页角度向量广播），每层一次 kernel。

### 顺带修掉一个真正确性 bug
`_dequant` 原来直接对**非连续切片** `out[..., :head_dim]` 调用 `.view(fp8)`。torch 的 dtype-view
要求最后一维连续，否则**静默错误重解释字节**。恒等视图（delta=0）不做旋转，所以之前一直没暴露。
现在先 `.contiguous()`。

另外：`delta == 0` 的页不再重新量化——否则会白加一个 fp8 台阶，破坏与页库的字节一致性。

### 实测
| 用例 | 结果 |
|---|---|
| 29K token（恒等视图） | `ZQ00100 ZQ00600 ZQ01190`，与基线一致；<45 s |
| 64K token（有逐出） | **125 s 完成**（优化前 >4 分钟未完成） |
| 随机密钥检索（不可推导） | `GSS5H3 GQZHLN 9M6V5C` —— **三段全对** |

随机密钥那一项很重要：编码由记录号推导的设计（`ZQ0NNNN`）测不出检索丢失，换成每记录随机
6 字符后才有意义。

**还没测到的**：把上下文拉到显著超过 B（>100K）时，中间段应当真的读不到。64K 这一档实测
kept 区（sink 8K + 最新 23.5K）仍覆盖了全部三个探针，所以"中间被丢弃"的行为尚未被观测到。

## 23. 关键发现：混合模型下"丢中段"不会真的遗忘（GDN 状态是免费的长时记忆）

**实验**：同一份 ~99K token 的随机密钥语料（每记录 6 字符随机 key，无法从记录号推导），
B=32768，稀疏开启（常驻恒为 2,048 页 = 32,768 token）。保留区 = sink 8K（记录 1–312）+ 最新
23.5K（记录约 3,580–4,700），记录 2000 位于约 42K token，**本应落在被丢弃的中段**。

| 提问 | 正确答案 | 稀疏版回答 |
|---|---|---|
| 记录 100（sink） | `GSS5H3` | `GSS5H3` ✅ |
| 记录 2000（**应被丢弃的中段**） | `SC5T57` | `SC5T57` ✅ |
| 记录 9000（**根本不存在**） | — | **未作答**（没有编造）✅ |

**结论**：模型既没有靠猜（对不存在的记录拒绝作答），也确实答出了"已被丢弃"的中段。原因只能
是这个模型 64 层里有 **48 层 GDN（线性注意力）**：它们在 prefill 时把整段历史压进了递归状态，
所以**丢掉注意力 KV 并不会抹掉这些信息**。

这解释了两件事：
1. 用户此前的判断"实际影响非常小"在这个架构上成立得比预期更强——历史仍有 GDN 状态托底；
2. 显存收益是真实的（我们丢的正是注意力 KV），但不能用"遗忘"来解释质量差异，也不该拿
   "中间被丢弃"来预期模型会答错。

**对后续的影响**：
- 质量对照必须用**更细的探针**（例如需要逐字复述中段原文、或依赖中段的具体措辞），否则
  GDN 状态会把差异掩盖掉；
- 反过来说，稀疏化的**质量风险**在这个模型上明显低于纯注意力模型。

## 24. V2（4 并发）被 vLLM 的容量记账卡住 —— 这是设计约束，不是实现 bug

尝试 4 并发时，引擎在**启动阶段**就拒绝服务：

```
Reduce MAX_MODEL_LEN or MAX_NUM_SEQS, or increase available KV capacity.
```

原因：未修改的 vLLM 在初始化时按 **`max_num_seqs × max_model_len`** 估算所需 KV 块数，
并据此校验池容量。实测该池 12,019 页（192,304 token），于是：

| 配置 | 需求 | 结果 |
|---|---|---|
| 4 × 204,800 | 819,200 token | 拒绝 |
| 4 × 49,152 | 196,608 token | 拒绝（略大于 192,304）|
| 4 × 45,056 | 180,224 token | 通过校验，但进程在启动中被 Killed（主机侧）|

**含义**：KVMem 的稀疏驻留只改变**运行时实际占用**，不改变 vLLM 的**准入估算**。因此"4 条并发
200K 会话"不能靠启参数实现，必须让引擎的容量记账知道稀疏驻留的存在——即调度器的块预算要按
`B` 而不是 `max_model_len` 来算。

这是后续要做的**第二处插桩**（第一处是注意力元数据）：
- 位置：`v1/core/kv_cache_utils.py::_check_enough_kv_cache_memory` 与调度器的块预算初始化；
- 做法：当稀疏模式开启时，把每请求的 KV 需求按 `min(max_model_len, B + R)` 计入，并让
  `block_pool` 的可用块数按实际池报告。

在完成这处插桩前，V2 用"多条中等长度请求并发"验证机制（页面互斥、独立工作集）是可行的，
但不能声称"4×200K 已跑通"。

## 25. 重要更正：当前实现是「读侧稀疏」，不是「分配侧节省」

核对真引擎日志后必须更正一条口径。29K token 的请求在重选时：

| 指标 | 值 | 含义 |
|---|---|---|
| 引擎自己的块表行 | **1918–1919** 个非零页 | 引擎为该请求**每个 token** 都持有页 |
| 我们的稀疏视图 | 1833–2048 页 | 注意力实际读取的页 |

也就是说：

- ✅ **注意力读取被压到 ≤ 32,768 token**（V1 量到的就是这个）——省的是 **kernel 读取量与计算**；
- ❌ **没有把页还给 vLLM 的池子**——引擎仍为整段历史持有页，池容量**没有任何增加**。

因此 "单请求常驻从 200K 降到 48K / 池支持 4 并发 200K" 这个目标**尚未达成**，当前只完成了读侧。

## 26. 要达成分配侧节省，必须改在调度器侧

worker 侧无法归还页：页的所有权在 `block_pool` + `req_to_blocks`，只有调度器能释放与重分配。所以：

1. 逐出决策放到 EngineCore 的调度器（TP=2 下调度器只有一份，天然锁步）；
2. 落选页 → 通知 worker 把 KV 溢到主机 → 调度器 `block_pool.free(page)` 并把这些块从该请求的
   `req_to_blocks` 移除；
3. 还原 → 从池里取新页 → worker 写入并重打 RoPE → 调度器把新页登记回 `req_to_blocks`；
4. 同步改写块表行（worker 侧已有能力）。

这同时解决了 V2 的容量记账问题：调度器按实际持有块数计算，池自然容纳更多长请求。

**在此之前**，任何"显存省了多少 / 支持多少并发"的结论都不成立；能成立的是：注意力读取量下降、
质量在 GDN 托底下影响很小、埋针字节一致。

## 27. 调度器侧地面真相（只读探针实测）

在 EngineCore 里包装 `Scheduler.schedule`（只读、不改状态），对运行中的请求打印块清单：

```json
{"groups": [5, 5, 5, 39], "computed_tokens": 62838,
 "attention_group": 3, "attention_blocks": 39, "free_blocks": 96}
```

三条关键事实：

1. **调度器的分配单位是对齐块（1,616 token = 101 页），不是 16 token 的页。**
   39 块 × 1,616 = 63,024 ≈ computed 62,838 ✓。所以调度器侧能释放的最小粒度是 **1,616 token**。
   这直接影响选入粒度：要真正还页，选择必须按 1,616 token 对齐，或者保留跨界的部分块。

2. **Mamba（GDN）组各占 5 块，与注意力组分开记账** —— 我们的释放只能动注意力组（group 3），
   这也印证了"GDN 状态不参与稀疏"的处理是对的。

3. **`free_blocks` 是现成的压力信号**（本次 96 空闲）。比我们自设的水位线更可靠：它反映调度器
   真实的可用块数，可以直接作为"何时逐出"的触发条件。

## 28. 对 vLLM 吞吐/并发的直接含义

池 = 119–132 个对齐块（1,616 token）。一条 200K 请求需要约 127 块，几乎吃满整个池——这正是
"最大并发 1.04x"的来源。要让 4 条 200K 并存，必须让每条只**持有**约 B/1,616 + R/1,616 ≈ 30 块
（32,768/1,616 ≈ 20 块），其余靠主机池 + 调度器释放。

## 29. 对齐块规划（`src/kvmem_aligned_blocks.py`，已单测）

调度器以 1,616-token 对齐块分配/释放，worker 以 16-token 页寻址。分配侧要还页，必须先回答
"哪些整块可以还"。规则：**只有当一个对齐块的 101 页全部不在工作集内时**，才能整块归还。

`plan_aligned_blocks()` 把选中的页映射成 `{freeable, kept, partial}`，单测覆盖：整块选中不可还、
半选块标记 partial、页→块映射与 NULL 块偏移一致。

**算出来的并发上限**（这是 V2 的硬账）：

| 项 | 值 |
|---|---|
| 对齐块 = | 1,616 token（101 页） |
| B = 32,768 token | **21 个对齐块** |
| 池 | 119–132 个对齐块 |
| 4 条并发各持 21 块 | **84 块 ≤ 119** ✅ 装得下 |

也就是说：**只要调度器真的释放未选中的块，4 条 B=32K 的并发在工作集层面是可行的**。这与
当前"一条 200K 请求要 127 块、几乎吃满整池"形成对比，正是目标要的效果。

**粒度代价**：还页只能以 101 页（1,616 token）为单位，所以选入边界附近的块会"半选"而无法归还；
B=32,768 → 21 块里至少有一块是半选的垃圾，实际可还 20 块。

## 30. 分配侧释放：真引擎实测通过（这是目标里"把页还给池子"的那一半）

在 EngineCore 里对**运行中请求**调用 `_remove_blocks_in_range(req, i, i+1)`，逐块释放"中段对齐块"
（sink 5 块与 recent 2 块之外），实测：

| 释放上限 | 实际释放 | 请求仍持有 | 空闲块变化 | 答案 |
|---|---|---|---|---|
| 8 | 8 块 | 18–26 | 122 → 117 | `ZQ00100 ZQ01250 ZQ02490` ✅ |
| 40 | **20 块** | 39 | 升到 116 | `ZQ00100 ZQ01250 ZQ02490` ✅ |

零错误。含义：

1. **块真的回到池子**（`get_num_free_blocks` 上升），这是此前一直缺的"分配侧节省"；
2. **输出仍然正确**——因为 worker 侧把注意力限制在压缩后的行里，被释放的块根本不在行内；
3. 一条 60K 请求释放 20/39 块 = **约 51% 的 KV 显存还给了池子**。

### 为什么 recency 模式不需要"还原"通道

recency 选入永远保留 sink + 最新窗口，窗口单调前移：一个块一旦滑出窗口就**永远不会再被选中**，
所以"释放"是终态，不需要 worker→scheduler 的取回通道。

**检索模式（mean-K）则不同**：被驻泊的块可能重新变相关，那时才需要取回通道（仍是缺口）。

## 31. 由此得到的最小可用配置

- 选入：recency（sink 8192 + 最新 ~23.5K）
- 释放：调度器逐出滑出窗口的对齐块
- 结果：**真实显存回收 + 输出正确**，且无需新 RPC

这已经能支撑"多条长请求并发"：每条只持有 B 对应的 ~21 块，其余还给池子。

## 32. V2（4 并发）实测通过

引擎以 `MAX_NUM_SEQS=4` / `MAX_MODEL_LEN=32768` 启动，准入检查通过：

```
KV concurrency check: OK (132892 tokens >= 4 x 32768 = 131072)
```

4 条并发请求（各约 29K token、随机密钥、不可推导），**全部完成且答案逐字正确**：

| 请求 | 期望 | 实得 |
|---|---|---|
| req0 | `9P6HQ8 TXRQD8` | `9P6HQ8 TXRQD8` ✅ |
| req1 | `8YXPN4 Y2SZ75` | `8YXPN4 Y2SZ75` ✅ |
| req2 | `9MG4D2 GN73PZ` | `9MG4D2 GN73PZ` ✅ |
| req3 | `FDQ64D R2CDBP` | `FDQ64D R2CDBP` ✅ |

引擎全程健康（health 200），释放路径也在工作（`freed: 2, free_now: 51, pool_total: 146`）。

### 两条必须注明的限制

1. **规模是 4 × 29K，不是 4 × 200K**：launcher 的准入公式要求
   `max_num_seqs × max_model_len ≤ 池 token 数`，而池容量本身**随 max_model_len 收缩**
   （实测 204800→192K、45056→153K、36864→142K、32768→133K），所以 4 条 200K 需要该公式
   认识稀疏驻留（即按 `B+R` 而不是 `max_model_len` 计入）。
2. **本轮没有触发重选**（29K < B=32768，无需溢出），所以验证的是"并发 + 释放路径"，
   而不是"并发下的溢出重选"。下一轮用 4 × 45K 以上复跑才能覆盖后者 —— 前提是先把准入
   公式改成按 `B+R` 计。

## 33. 并发下的稀疏重选：功能正确，但性能是瓶颈

4 条并发 40K 请求（超过 B=32768，必然触发溢出）：

- **0 错误**、156 次重选全部完成（此前一次批量旋转有 device 类型不匹配的 bug：`inv_freq` 在 CPU、
  页在 CUDA；已修）；
- 但 10 分钟仍未跑完 4 条 —— **重选开销随"页数 × 层数"线性增长**：每次重选要搬 2,048 页 ×
  17 层 × 16 KiB ≈ **570 MB H2D + 旋转**，4 条请求各自触发，直接把 decode 拖垮。

### 结论

功能上打通了"并发 + 溢出重选"，但**当前的重选成本不适合生产**。要让它可用，必须做其中至少一项：

1. **降低重选频率**（按 token 增量触发，而不是每 N 步）；
2. **增量旋转**：只处理自上轮以来槽位发生变化的页（sink 前缀不移动，最近窗内的页只在小范围内挪动）；
3. **异步搬运**：把 H2D/旋转放到独立流，与计算重叠（我们已有 `KVMEM_ASYNC_TRANSFER` 的骨架可复用）。

## 34. 4×40K 崩溃点的精确机制（`assert num_output_placeholders >= 0`）

崩溃位置：`vllm/v1/core/sched/async_scheduler.py:62-63`

```python
if not is_stale:
    request.num_output_placeholders -= len(new_token_ids)
    assert request.num_output_placeholders >= 0
```

记账模型（`async_scheduler.py:39`）：

```
每调度一步： placeholders += num_sampled_tokens_per_step + len(上一轮的 draft tokens)
收到输出时： placeholders -= len(new_token_ids)          # 包括被接受的 draft + 1 个 bonus
```

也就是说：**worker 每步返回的 token 数必须 ≤ 调度时预扣的占位数**。断言触发意味着
worker 某步返回的 token 数**超过了**预扣数。两个可能来源：

1. **投机解码（MTP K=4）的 draft 计数不一致**：`cur_num_spec_tokens` 取自
   `spec_decode_tokens[req]`（上一轮 worker 报的 draft 数），若我们覆盖 `seq_lens`/`positions`
   后 worker 某步多产出/少产出 draft，两边就会漂移；
2. **异步调度的陈旧投递**：`is_stale` 分支本该跳过扣减，若某条输出被误判为非陈旧，也会走减扣。

### 为什么暂时无法定论

隔离实验（`MTP_K=0` 关掉投机解码后复跑 4×40K）**还没做**——用户要求先搁置并发。这是唯一能把
"投机解码记账"和"异步调度陈旧投递"两个假设分开的手段：若关掉 MTP 后不再崩，就是 (1)。

### 复跑步骤（已就绪）

1. `run-kvmem-sparse.sh` 里加 `export MTP_K=0`；
2. `--set MAX_NUM_SEQS=4`、`--set MAX_MODEL_LEN=204800`、`KV_CONCURRENCY_EFFECTIVE_LEN=40960`；
3. 发 4 条 ~40K 请求（`/tmp/v2_req{0..3}.json` 已生成），跑到 6 万 token 以上观察是否再触发断言。

## 35. 结构性结论：滑动窗压缩=每次重选必然整窗重打 RoPE

试过把重选改用**设备间搬运**（HBM ~600 GB/s）替代主机重读（PCIe ~12 GB/s）。测试直接否掉了它：

```
窗口前移 6 页（96 token）后重选：
  selected = 2046 页
  device_moves = 0
  host_rebuild = 1534 页
```

原因：压缩后每页的旋转量是 `(名次 - 逻辑页) x 16`，窗口一移动，**几乎所有页的名次都变**，
旋转量随之改变。而"在已旋转的页上再旋转"会给 fp8 叠加台阶（每次 ~6% 相对误差，49 次重选后
累积不可接受），所以旋转过的页**只能从页库重建**。

也就是说：

| 手段 | 是否可行 |
|---|---|
| 设备间搬运（不重旋转） | ❌ 名次变了，旋转量必须跟着变 |
| 链式旋转（复用设备上的旧旋转） | ❌ fp8 误差累积 |
| 不压缩、用 null 块占位 | ❌ FlashInfer 无 R-SWA 掩码 |
| **降低重选频率** | ✅ 唯一有效杠杆（已做：按 4,096 新 token 触发） |
| 缩小工作集 B | ✅ 直接减半成本 |

**结论**：重选成本 `= 选中页数 x 层数 x 16 KiB`，这是滑动窗压缩的固有代价；能调的只有频率和 B。
若要连这个都省掉，就必须换到"不压缩、靠掩码"的方案（即需要支持 R-SWA 的后端）。

代码里 `plan_moves/order_moves` 保留（逻辑正确、单测覆盖），但实际会退化为"全部重建"，
`tests/device_move_test.py` 就是把这条结构性事实固化成断言。

## 36. 还原数据通路（`restore_page`，已单测）

目标要求"选中但缺失的页从主机池溢回"。数据面已实现并测透：

```python
def restore_page(self, page, slot):
    needed = (slot - page) * page_tokens
    step   = needed - store_delta[page]
    kv[slot] = rotate(store[page], step)     # 恰好一跳，绝不旋转"已旋转的页"
```

单测覆盖四点：
1. 被逐出的中段页还原后与"一跳重建"字节一致；
2. 从未驻留过的页还原路径相同；
3. 还原到相同旋转量时**零跳**（字节不变，不加 fp8 台阶）；
4. 连续还原 3 次**不累积**误差（页库只读，永远从它派生）。

**控制面仍然缺**：向调度器要一个空闲块需要 worker→scheduler 通道（recency 模式下不需要，
因为滑出窗口的页永远不会再被选中；检索模式才需要）。

### 主机层改由 KVMem 管理（`KvmemHostStore`，已单测）

绑定了 kvmem 的 `HostKvStorage`/`HostKvRecord`，页库可以改用它：一个逻辑页 = 一条记录
（278,528 B），**字节往返一致**，容量按 H 记账，超限时淘汰最旧记录而不是让 RSS 无限增长
（`tests/host_store_test.py` 7 项全过）。

**已通用化并设为默认**（`use_kvmem_host=True`）：记录布局改为从 payload 自身发现
（每层 shape/dtype/nbytes），因此不同层几何形状共用同一个 store；两条 store 都补上
`len()` 与 `pages` 视图，调用方不再访问内部字段。

活引擎验证（33K 请求，规划器 + kvmem 主机层同时开启）：重选 8 次、**埋针 8/8 全过**、0 错误。

## 37. 还原需求其实已被 reselect 覆盖

`reselect()` 本身就是"把所有选中页从页库重建"，所以"选中但缺失"这个情形在重选时自动解决。
`restore_page` 是把这件事**显式化并单独可测**，用于"重选之外、单页按需回填"的场合。

## 38. 收敛第一步：kvmem 自己的规划器与手写视图等价（已验证）

方向确定：**kvmem 主办、vLLM 当后端**（见 `DIRECTION.md`、`MULTIGPU-CONCURRENCY.md`）。
切换前必须先证明"换成 kvmem 的规划器"不会改变语义，于是绑定并对照：

```python
# 新绑定（src/vllm_kvmem_ext.cpp）
plan_working_set(descriptor, snapshot, selected, mandatory, budget) -> WorkingSetPlan
MemorySession.prepare_reselect(expected, policy, mandatory, budget) -> WorkingSetPlan
```

对照结果（`tests/kvmem_view_test.py`，同一份 24 块语料、预算 1024）：

| 项 | 手写视图 | kvmem 规划器 |
|---|---|---|
| 保留块 | 8 | 8 ✅ |
| 可见 token | 1024 | 1024 ✅ |
| 压缩地址 | 从 0 递增 | 从 0 递增 ✅ |

**结论**：把视图来源换成 kvmem 的 `plan_working_set` / `prepare_reselect` 是**语义等价**的，
不会改变已经过活引擎验证的行为。可以安全切换。

### 踩到的两个坑（后续接 kvmem 契约时必看）

1. `def_readwrite` 对 `std::vector<...>` 成员返回的是**副本**：`snap.blocks.append(x)` 不会写回 C++，
   必须整表赋值 `snap.blocks = [...]`；
2. `plan_working_set` 的校验很严：descriptor 需要完整 identity + 合法 payload layout（每层必须有
   Key 和 Value 两个 plane）+ `native_payload/sparse_view/original_rope_positions` 三个能力位；
   snapshot 需要 `stamp.valid` 且 session/incarnation/history_revision/execution_history 全非零，
   每个 block 的 `session`/`content_version` 非零、`original_start` 必须与 frontier 首尾相接，
   冷块还要有当前主机副本（`host_version == content_version`）。

## 39. 规划器接进活路径后出现回归（已回退，问题定位中）

按收敛计划把视图来源换成 kvmem 的 `plan_working_set` 后，**离线 0 个坏埋针，活引擎 682/684 失败**：

```
{"kind": "needle", "pages": 4, "ok": false, "detail": {"0": false, "1": false, "2302": true, "2303": true}}
reselect: 628 次，0 错误，resident 2,043，seq_len 32,688，tokens 116,518
```

离线复现（同一份代码、同一份配置、只有引擎页号不同）**埋针全过**，说明问题出在活路径特有的
"引擎页号 → 槽位"这一段，而不是规划器的视图本身（视图等价性已由 `tests/kvmem_view_test.py` 证明）。

最可疑的机制：规划器返回的页列表与选择器的页列表**在块粒度上一致、但顺序/成员可能有细微差别**，
于是某次重选里 page X 被写到 slot 0，而 `slot_of[0]` 仍是 0、`baked_delta[0]` 没变，`plan_moves`
判定 page 0 "不需要搬"，结果 slot 0 里其实是 page X 的字节 → 埋针失败（page 0/1 失败正好符合）。

**处置**：`use_kvmem_plan` 默认 **False**（配置里也关掉），活路径退回已验证的手写视图；
规划器、适配层、等价性测试全部保留。

### 复现两次后的收敛结论（第二轮：加了槽位唯一性防护后仍失败）

```
needle: 646 records | bad: 644 | first bad detail: {'0': False, '1': False, '2302': True, '2303': True}
reselect: 646 | errors: 0 | selected 2046 | seq_len 32,736 | tokens 116,563
```

- **加"槽位必须非零且唯一"的防护后行为不变** → 不是引擎页号重复/为空；
- 失败固定是**最低的两个常驻页**（0、1），最高的两页正常；
- 离线用同一份代码、同一份配置（含 prefill pressure 路径、多轮重选）**永远是 0 坏针**。

因此差异只剩下活路径特有的那一层：**"引擎在压缩位置上写、我们在引擎页号上记"**。怀疑点是
`commit_written_page(logical, page_map[logical])` 读的槽位与引擎真正写入的槽位（由我们覆盖的
`positions` 决定，即压缩名次）不是同一个。这条需要一次带 trace 的活引擎单请求来确认——
它同时也能解释为什么只有最早写入、之后不再被重写的那几个页会挂。

**下一步（留给下一轮）**：在 `commit_written_page` 前后打印 `logical / engine_page /
engine_written_slot / store_delta / baked_delta`，用一条 4K token 的小请求跑 2~3 次重选即可定位。

## 40. 规划器回归的真正根因：我自己加的"reserve 触发"退化了

上一节把失败归到"引擎写位置 vs 我们记的页号"。**错了**。真正的根因是触发条件：

```python
# 错的（我写的）
if self.total_tokens > self.last_visible_tokens + self.planner.gen_reserve:   # 32K+16K=48K
```
上下文一旦超过 48K，这个条件**永远为真** → 每一步都重选一次。那轮 99K 实测里 646 次重选、
644/646 坏针，就是这个原因（而且极慢）。

```python
# 对的（kvmem 语义：R 是"两次重选之间允许新增的 token 数"）
if self.total_tokens - self._tokens_at_reselect >= self.planner.gen_reserve:
```

修复后实测（同一条 99K 请求、规划器开启）：

| 指标 | 值 |
|---|---|
| 上下文 | 114,688 token |
| 重选次数 | **40**（每约 16K token 一次，符合 R=16,384） |
| 埋针 | **40 条全过、0 坏**（页 0/1/7166/7167 都 True） |
| 错误 | 0 |

打点数据（修复后）：
```
page 0: engine_page 101, slot_of 101, store_delta 1,616, baked 0,   needle True
page 1: engine_page 102, slot_of 102, store_delta 1,616, baked 0,   needle True
page 7166: slot_of 2,147, store_delta -51,968, baked -81,920, needle True
```

**教训**：把一个"看起来等价"的条件从"相对上次"改成"相对固定量"，会让触发变成每步一次；
而每步重选既慢又会让槽位在测量窗口内变化，看起来像数据错乱。诊断时应该先怀疑**触发频率**，
再怀疑数据路径。

## 41. 收敛后的两项验证（规划器驱动 + R 节拍）

### 单请求 99K（`use_kvmem_plan=True`，cadence 由 R 驱动）

| 指标 | 值 |
|---|---|
| 上下文 | 114,688–117,6xx token |
| 常驻 | **2,047–2,048 页 = 32,752–32,768 token**（约 3.5x 压缩） |
| 重选 | 40–46 次（≈ 每 16K token，符合 R=16,384） |
| 埋针 | **全过**（打点：页 0/1 store_delta 1,616、baked 0；页 7166/7167 store_delta −51,968、baked −81,920，均 True） |
| 错误 | 0 |
| 答案 | decode 阶段被重选成本拖慢（结构性代价，见第 35 节），未在超时前返回 |

### 4 并发（`run-kvmem-4way.sh`，:8002，MAX_NUM_SEQS=4、MAX_MODEL_LEN=32768）

准入：`KV concurrency check: OK (132892 tokens >= 4 x 32768 = 131072)`

| 请求 | 结果 |
|---|---|
| req0–req3（各约 27.5K token） | **4/4 答案逐字正确**，124 s，0 错误 |

注：首次发送时用了 32.7K token 的提示词，被 `max_model_len=32768` 以 400 拒绝——4 并发的
提示词长度必须留出输出余量（这就是"准入按 max_model_len 记账"的直接体现）。

## 42. 共享 H 与会话生命周期（④ 收口）

KVMem 的契约是"B/R 每请求、H 共享"。vLLM 一个 EngineCore 服务 C 条请求，所以 H 必须显式分账：

- `SessionBudgetPool`：每会话 `H // sessions`，只可借**未分配余额**，超发抛 `BudgetError` 并说明
  还差多少（8 项测试）；
- `runtime.release(req)`：请求结束释放状态并把份额还给池子；`live_requests()` 供 worker 侧扫尾；
- worker 每步比较批次，消失的请求自动 `release`（runtime 在 worker 进程，所以这个钩子必须放
  worker，不能放 EngineCore——我第一版放错了地方）。

活引擎冒烟（4K 请求）：答案正确、**4 条 `released` 记录**、0 错误。

## 43. 释放闭环可复现（`scripts/free_loop_check.sh`）

一条命令即可复跑"分配侧释放"：

```
$ scripts/free_loop_check.sh 8002
free_path=True low_watermark=0.75 per_step=2 sink_blocks=5 recent_blocks=2
4 concurrent requests fired; waiting for completion...
  req0..req3: OK
answers correct: 4/4
freed_blocks records: 2  blocks released: 4
last: freed=2 free_now=55 held=8 pool_total=146
PASS free_loop_check
```

要点：
- 调度器在池压到水位以下时，把运行中请求的**中段对齐块**还给池子（`_remove_blocks_in_range`）；
- 答案仍然全对——因为 worker 只读压缩后的视图，被释放的块不在其中；
- `free_low_watermark` 每步从配置热读，**调水位不需要重启引擎**（0.35 时几乎不触发，0.75 时会释放）。

**还原那一半仍未做**：recency 选入下滑出窗口的页永不再被选中，所以释放是终态；只有换成
mean-K 检索才需要 worker→scheduler 的取块通道（`ARCHITECTURE-VERDICT.md` 有两条路径）。

## 44. 还原控制面（worker → EngineCore 取块通道）

还原需要"向调度器要一个空闲块"，而 vLLM 的 `scheduler_output` 是单向的，所以走侧通道：

- `src/kvmem_block_channel.py`：`BlockChannelScheduler`（EngineCore 侧，unix socket，分配器由插件
  接到 `block_pool.get_new_blocks`）+ `BlockChannel`（worker 侧，一问一答）；
- 分配器抛错会**回报给 worker 而不是拖垮调度器**，失败后仍能继续服务（单测覆盖）；
- `restore_page(page, slot=None, channel=..., request_id=...)`：没有槽位时通过通道取块再还原，
  校验与"直接给槽位"那条路径一致。

**调用者仍不存在**：recency 选入下滑出窗口的页永不再被选中，所以没有"要回来"的场景；这条通道
是为 mean-K 检索模式准备的（那时被驻泊的页可能重新变相关）。开关 `block_channel`（默认关）。

## 45. 99K 的端到端答案：未捕获（decode 停住，已定位症状）

最终配置（规划器 + kvmem 主机层 + R 节拍）下跑 99K 请求：

```
常驻 2,048 页 / seq_len 32,768        ✅
埋针 40 条 trace，0 坏                ✅
重选 40→44 次，0 错误                 ✅
```

**但答案始终没返回**，症状明确：

```
reselect#43: tokens 118,788  selected 2,041  seq_len 32,656
reselect#44: tokens 122,888  selected 2,041  seq_len 32,656
（此后 token 计数停在 122,888，重选仍在同一 frontier 上触发）
```

即：**token 计数在 122,888 处停住，而重选继续在同一 frontier 上触发**。这说明

1. 不是数据错误（埋针全过、0 错误）；
2. 是**处理停住 + 重选重复触发**：R 触发条件读的 `_tokens_at_reselect` 与实际推进不同步，
   或者 engine 侧该请求的 step 没有继续（GPU 仍在忙，但请求无输出）。

**离线复现结论**：模拟"117K frontier 上每步 +1 token 连续 40 步"，R 触发**只在该窗口开头
触发 1 次**，之后 `_tokens_at_reselect` 正确前进——离线逻辑是对的。

**因此活引擎上的重复触发另有原因**。两条尾记录都是
`{'tokens': 122888, 'prefill_pressure': False}`：

- 同一个 frontier 上触发了两次；
- 而压力路径可以排除（`prefill_pressure: False`，且 resident 2,041 < 水位 2,112）。

剩下的可能：运行时的 `st.tokens` 与 orchestrator 的 `total_tokens` 在**重选后被下一次
`note_tokens` 回退**（例如引擎侧 `num_computed_tokens` 没有前进、而我们的计数又按旧值重算），
于是 `total - _tokens_at_reselect` 反复跨过 R。定位方法：在 `_reselect` 前后同时打印
`st.tokens / orch.total_tokens / orch._tokens_at_reselect / 引擎的 num_computed_tokens`，
一条 4K 请求跑 2 次重选即可看出是谁在回退。

这是唯一未闭合项，且**不影响已完成的四项机制**（埋针全过、0 错误、常驻正确）。

## 46. ① 启动形态对齐 kvmem：配置走 vLLM 官方通道（已活引擎验证）

**问题**：原先用 `/tmp/kvmem_plugin.json` + 环境变量把 kvmem 参数送进 worker。vLLM 会过滤环境
变量，所以当时只能落文件——但这是一条旁路，和 kvmem-llama "argv 直达 worker" 的形态不一致。

**正解**：vLLM 的 `VllmConfig.additional_config` 是官方字段，**随配置序列化到每个 TP worker**
（fork 自己就用它传 `gdn_prefill_backend`）。于是：

```
--kvmem-*  →  backends/vllm/worker.py 组装  →  ADDITIONAL_CONFIG_JSON
           →  --additional-config '{"kvmem":{...}}'
           →  VllmConfig.additional_config  →  每个 worker
           →  插件 _config_from_runner(runner) 读回
```

**踩到的接线顺序问题**：`register()` 在插件加载时执行，那时**还没有 runner**，所以
`additional_config` 不可见——原来 `register()` 会根据（空的）配置决定"是否挂钩子"，结果
sparse=False、钩子全不挂。修法：**钩子无条件安装**，每步用 `_config_from_runner()` 拿到的实时
配置来决定是否 attach（`_config().get("sparse")`）。

**活引擎证据（4K 请求，`/tmp/kvmem_plugin.json` 已删除）**：

```
{"kind": "config_source", "from": "additional_config", "pid": 318598, "keys": [...]}
{"kind": "config_source", "from": "additional_config", "pid": 318622, "keys": [...]}
{"kind": "sparse_attached", "pid": 318622, "builders_wrapped": 1, "page_tokens": 16,
 "budget_tokens": 32768, "reselect_interval": 1000000}
answer: 'YL4P63 WN9TXB'  → 与期望一致
```

两个 TP worker 都拿到配置、都挂上了 sparse；没有任何环境变量或旁路文件。

**打包**：`backends/vllm/worker.py`（对应 `backends/llamacpp`/`backends/ninfer`），
`--dry-run` 可审计输出；`backends/vllm/README.md` 写清与另两个后端的对应关系。

## 47. ③ mean-K 检索：数据通路已通（选片已能按内容而非 recency）

kvmem 的检索要的是**位置无关（pre-RoPE）的统计量**，而 vLLM 的注意力层只暴露 post-RoPE 的 q/k
（`Qwen3NextAttention._project_qkv_gate` 的 docstring 明写 "post-norm, post-RoPE"，且有一条把
norm+rope 融进 Triton kernel 的路径，pre-RoPE 值根本不会出现）。

**做法：用位置反旋转还原**（不改 fork、不动 kernel 开关）。旋转对每个 (head, position) 是正交的，
用该 token 自己的位置做逆旋转即可精确回到 pre-RoPE 空间：

```
tests/pre_rope_capture_test.py::inverse rotation recovers the pre-RoPE tensor (max err 1.19e-07)
```

三块拼起来：

| 模块 | 作用 | 测试 |
|---|---|---|
| `src/kvmem_pre_rope_capture.py` | 逆旋转、Q 按 head 求和、每块余弦打分、**按 R 的 lookahead 决定何时采集** | 6 项 |
| `src/kvmem_retrieval_index.py` | 每块（跨层）的 pre-RoPE K 均值累积 + 当前 query + 余弦打分 | 7 项 |
| `reselect` 接线 | 有 query 时 `selector.set_block_scores(...)`，kvmem 据此排序 | `retrieval_mode_test` |

**行为差异已可观测**（`tests/retrieval_mode_test.py`）：

```
ok  retrieval keeps a different set than recency
ok  the content-relevant block is kept
ok  recency does not specifically keep it
```

即：同样一份页、同样的预算，**检索模式按内容保留，recency 按位置保留**——不是"退化"了。
`MemoryDescriptor.capabilities.pre_rope_statistics` 只在确实有采集时声明，避免 kvmem 拒单。

**还差最后一段**：把采集真正挂到引擎上（`Qwen3NextAttention.forward` → 喂 index）。数据面与
选择面都已就绪。

## 48. ③ 引擎侧接线：离线全通，活引擎尚缺一条证据

**已实现并离线验证**（`tests/engine_capture_test.py`）：

```
ok  the hook harvested one step
ok  both pages landed in block 0
ok  the query landed on the layer the attention layer reported
ok  the block statistic matches the pre-RoPE mean (err 1.49e-08)
```

钩子包 `Qwen3NextAttention._project_qkv_gate`，逆旋转回 pre-RoPE，按页分组喂索引。

**活引擎状态**：`--kvmem-retrieval` 已能经 `--additional-config` 到达两个 TP worker
（`config_source` 记录里 `sparse` 与 `retrieval` 都在），但**没有看到 `capture_attached`
与带 `scores` 的 reselect**；同一轮里 `sparse_attached` 也消失了（此前那版有），说明
`execute_model` 里"是否 attach"的分支这次没走进去，而请求本身仍答对（走了普通引擎路径）。

**已修（真 bug）**：我把 `cap.note_tokens()` 插在了 `runtime` 尚未绑定的位置，触发 NameError，
把整块 hook 打断——所以那一轮连 `sparse_attached` 都没有。移到 `runtime = state["sparse"]` 之后。

修复后（同一份 28K 请求）：

```
kinds: {'config_source': 2, 'capture_attached': 2, 'sparse_attached': 2,
        'engine_row': 6, 'needle': 6, 'reselect': 6}
reselects: 6 | with scores: 0
```

`capture_attached` 与 `sparse_attached` 都到位（每个 TP worker 一条），但**重选里仍没有
`scores`** —— 说明 `harvest()` 没被调用到（索引里没有 query，重选就跳过打分）。

**计数器读数（决定性）**：

```
capture: {'calls': 0, 'wants': 0, 'captured': 0}   # 6 次重选，全部为 0
```

`calls == 0` ⇒ **我们包装的 `_project_qkv_gate` 从未被调用**，不是"采集时机"问题。
但源码里 `Qwen3NextAttention.forward` 明明写着 `q, k, v, gate = self._project_qkv_gate(qkv, positions)`。

可能原因（按概率）：
1. worker 里的模型是 **torch.compile 产物**，编译发生在我们的补丁之前（插件在 `register()` 里
   patch 类方法，但图捕获可能更早，或者编译把方法内联了）；
2. 实际实例来自**另一个模块副本**（fork 里有编译/优化副本，`qwen3_5.py` 的 import 与我们 patch
   的类不是同一个对象）；
3. `MTP/draft` 路径与主模型路径分离，我们挂的那份没跑。

**模型实例诊断（决定性）**：在 worker 里扫 `runner.model.named_modules()`：

```
pid 328720 | patched_count: 0
    Qwen2_5_VisionAttention | vllm.model_executor.models.qwen2_5_vl | patched: False
    MMEncoderAttention      | ...layers.attention.mm_encoder_attention | patched: False
```

**只看到视觉注意力模块，文本侧的 `Qwen3NextAttention` 实例根本不在 `named_modules()` 里**——
说明语言模型被 `torch.compile` 包成了不透明的可调用对象（`support_torch_compile`），
原来的子模块结构在 `runner.model` 这层已经不可见了。这就解释了 `calls == 0`：我们 patch 的
类虽然存在，但跑的是编译后的函数，Python 层的方法补丁不会被调用。

**全量聚合后的真相比上一轮更精确**（上一轮只打了 4 个匹配，结论不成立）：

```
attention_total: 134        model: Qwen3_5ForConditionalGeneration   is_compiled: False
x 27  Qwen2_5_VisionAttention                gate=False patched=False
x 27  MMEncoderAttention                      gate=False patched=False
x 48  QwenGatedDeltaNetAttention (GDN)        gate=False patched=False
x 16  Qwen3NextAttention                      gate=True  patched=False   <- 文本注意力在这里
x 16  Attention                               gate=False patched=False
```

文本注意力**存在**（16 层，正是 full_attention 层数），`_project_qkv_gate` 也在（gate=True），
但 **`patched=False`** —— 而 `capture_attached` 说明 `install()` 返回了真值。

结论：**我们 patch 的类对象和模型实例用的类对象不是同一个**，尽管二者都叫
`vllm.model_executor.models.qwen3_next.Qwen3NextAttention`。最可能的原因是 vLLM 的**懒加载
`__init__`**（`vllm.model_executor.models.__getattr__` 按需导入）导致插件那次 `from ... import
qwen3_next` 拿到的是另一个模块实例。

**下一步（明确）**：在诊断里同时记录
`id(qwen3_next.Qwen3NextAttention)`（插件的）与 `id(type(model_attention))`（模型的），
以及 `[k for k in sys.modules if "qwen3_next" in k]`。若两个 id 不同，就改用
**`sys.modules` 里的那个模块**来 patch（或改成给每个实例设置 bound method，最稳）。

## 49. ③ 的真正障碍：AOT 编译产物绕过了 Python 钩子（已确认）

前几轮分别排除了：类身份不同（`same_class: True`）、补丁时序（把安装提前到 `register()`，
日志显示 `capture_attached phase=register`）、错误的模块（`sys.modules` 里只有一个
`vllm.model_executor.models.qwen3_next`）。

**决定性线索**在启动日志里：

```
Directly load AOT compilation from path /home/srj/.cache/vllm/...
```

fork 支持 `VLLM_USE_AOT_COMPILE`（`vllm/envs.py:138`），并把编译产物缓存到磁盘。加载该产物后，
模型前向**根本不经过 Python 的 `_project_qkv_gate`**，所以 `calls == 0` 与"补丁打得对"可以同时
成立。这解释了全部现象，也不需要再猜。

## 三条出路（按"是否得不偿失"排序）

| 方案 | 做法 | 代价 |
|---|---|---|
| A. 关 AOT/编译 | `VLLM_USE_AOT_COMPILE=0` + 可能的 `enforce_eager` | 需要实测吞吐损失；若小（<10%）可只在检索模式下降级 |
| B. 从 KV cache 反推 K + 另取 Q | K 在 cache 里（post-RoPE），用页号推原始位置逆旋转；Q 需另找来源 | 不牺牲性能，但 Q 的来源仍待解决（logits/MTP 侧） |
| C. 关采集窗口内的编译 | 仅在"采集需要"的那几步禁用编译 | 动态切换编译状态，风险高 |

**下一步**：先量 A（`VLLM_USE_AOT_COMPILE=0` 跑同一请求，看 tokens/s 与首 token 延迟），
再决定是"检索模式下关编译"还是走 B。目标里写的是"如果不是得不偿失"，所以这一步必须有数据。

## 50. 关掉 AOT 后钩子终于被调用，但只调用一次——CUDA Graph 回放也绕过 Python

`VLLM_USE_AOT_COMPILE=0` 重启后，同一份 28K 请求：

```
reselect#5: tokens 40961 | capture {'calls': 17, 'wants': 0, 'captured': 0}
reselect#6: tokens 45061 | capture {'calls': 17, 'wants': 0, 'captured': 0}
```

`calls: 17` = **17 = 16 个 full-attention 层 + 1**，而且**跨两次重选都不再增长**。
这说明注意力前向的 Python 路径只走了**一步**（首步 eager），之后全部走 **CUDA Graph 回放**
（配置里 `cudagraph_mode: FULL_AND_PIECEWISE`）——回放同样不经过 Python 方法。

于是采集的障碍是两层叠加：
1. **AOT 编译产物**（关掉后 `calls` 从 0 变成 17）；
2. **CUDA Graph 回放**（关掉 AOT 后仍只有 1 步走 Python）。

## 取舍的量化开始（部分已完成）

同一 2K 请求（1,817 prompt）：

| 配置 | 用时 | 说明 |
|---|---|---|
| AOT 开（基线） | 5 s（64 completion） | 首次请求含 warmup |
| AOT 关 | 1 s（6 completion） | completion 数不同，**不可直接比** |

需要重测：固定 completion 数、各跑 3 次。更重要的是测 `enforce_eager`（禁用 CUDA Graph）
的代价——那才是"采集能不能在活引擎上成立"的决定性数字。

**下一步**：`VLLM_USE_AOT_COMPILE=0` + `enforce_eager` 跑同一请求 3 次取中位数，与基线比较；
若损失可接受（例如 <15%），则"检索模式用 eager、recency 模式保持编译/图"是合理配置；
若损失过大，则走 B（从 KV cache 反推 K，Q 另找来源）。

## 51. 代价量化（同一 2K 请求、3 次取中位数、固定 64 completion）

| 配置 | 用时 | 相对 |
|---|---|---|
| 基线（AOT 编译 + CUDA Graph） | **1.59 s**（1591/1592/1681） | 1.00x |
| eager（`VLLM_USE_AOT_COMPILE=0` + `ENFORCE_EAGER=1`） | **2.20 s**（2202/2204/2256） | **1.38x（+38% 延迟）** |

两个附带结论：

1. 基线引擎上 `capture.calls = 0` —— AOT 产物**完全**不经过 Python，不是"部分绕过"；
2. 之前 `calls = 17`（只有一步）说明**即使关掉 AOT，CUDA Graph 回放仍然绕过 Python**，
   所以"只在 prefill 期间采集、decode 保持图捕获"这条零成本路走不通。

也就是说：**检索模式要采集，就必须放弃 AOT + CUDA Graph，代价约 +38% 延迟**。这已经是可以
拿去做决策的数字，不再是猜测。

## 52. ③ 完成：mean-K 检索在原始编译路径上跑通（实测数据）

排查顺序与每个阶段的硬证据：

| 阶段 | 证据 | 说明 |
|---|---|---|
| AOT 产物绕过 Python | `capture.calls: 0` | 需禁用缓存或走图内算子 |
| CUDA Graph 回放绕过 Python | 关 AOT 后 `calls: 17`（仅一步） | 图里只有录好的 kernel |
| 自定义算子 + 图外分配 | 离线 `op_calls 1→2` | `torch.compile(fullgraph=True)` **不会**删掉 custom op |
| V2 runner 无 `prepare_pos_seq_lens` | `has_prepare_pos_seq_lens: false` | 计数改用 `execute_model` 入口（实测每请求 44 次） |
| 旧 `EngineCapture` 包装替换了模型方法 | `inspect.getsource` 显示包装体、`capture_call: False` | **根因**：bridge 算子被挡在图外 |
| 停用包装后 | `op_calls: 128`、`wrote: 120` | 算子执行、缓冲写入成功 |
| 设备对齐 | 索引设备被 CPU 页字节定住 | 查询/评分统一到索引设备 |
| 分数表长 | `one score per selection block is required` | 补齐到 `ceil(total_tokens/block_tokens)` |

**最终读数**（37K 请求，原始编译路径）：

```
bridge_harvest: {'op_calls': 128, 'armed': True, 'counts': {0: 1}}
reselect.scores: {'blocks': 391, 'total_blocks': 417, 'top': 0.1312}
```

性能：长生成 3.38–3.45 s（≈87 tok/s），与基线 3.44–3.53 s 一致 —— **检索零性能代价**。
