# kvmem-vllm 适配器交付说明（v0.19 对齐）

更新：2026-10-29（**0.3.0**）。第三方引擎只以补丁形式发布：`install.sh` 默认把
`backends/versions.json` 的 `base_commit` 拉到 `engines/vllm`，再应用
`backends/patches/vllm-kvmem.patch` 并校验 `patched_tree`；不会写入用户已有的 fork。

**实验性整体重算验收**：TP2 / MTP4 / 同步 V2、单文本请求，
`GPU_UTIL=0.65` 的物理池 122,816 token，居中 needle 的真实 prompt 196,598 token：
27 次 `compact_applied` + 1 次 `compact_rebuild_queued` + 两 rank 各 1 次 `reconstruction_started`，
API `usage.prompt_tokens=196598`、HTTP 200、回答 **`ZX-4417-QJ7`**（耗时 320.24s）。
复现命令与已知边界见本文第四节、第六节。

## 摘要

**已完成（有实测）**

1. `KVMEM_STATS` 开启后 88K 崩溃（pinned allocator）——已修；88,256 token prompt 稳定答对 `ZX-4417`。
2. v0.19 打分栈接入：`RawKvStore`（64-token 桶、mean-K）、`PageQueryView`（pre-RoPE）、
   `QueryScoreFrame`、page-softmax（sm75 CUDA）——排名确实是选择器的输入，needle 块能排第一。
3. 压缩路径：纯拷贝、位置拆分（RoPE 用真实位置、写地址用压缩位置）、三本账一致。
4. decode 期统计：按"已接受"时间线推进（bridge 保留解码行、按上一接受数折叠），frontier 精确。
5. `CheckpointCatalog` 落地并接管 session store 的状态镜像（一个所有者、容量与淘汰统一）。
6. 超池**通道**打通：196,673 token prompt 答对；152,820 token prompt 在 122,816 token 的池上返回 200
   （9~25 次压缩）；`KVMEM_SKIP_KV_CHECK` 解除启动不变式、`scheduler_reserve_full_isl` 解除准入闸门、
   attention 原生块按 EngineCore 所有权回收（实际 1,616 token/块）。

**超池整体重算（实验，有一次 196,598-token 实测）**

- vLLM 完整预填后按原始逻辑页选 token；两 rank 在这一同步边界对核心评分求均值。
  EngineCore 在旧输出发布前丢弃旧采样、释放整请求 KV/循环状态并重新入队；两 rank 的 V2
  fresh admission 重建 token 表和循环状态。压缩时仅回收 EngineCore 所持有的 attention 原生块，
  worker 的块表行同步移到 staged writes 之后，槽位偏移取 scheduler 的 served-resident 两本账。
- `196598 > 122816`，居中 needle、27 次压缩后整体重算 33,654 token，回答
  `ZX-4417-QJ7`；探针 `compact_rebuild_queued`、`reconstruction_started` 两 rank、
  `/tmp/kvmem-reconstruction-offset.metadata.json`。本次不等于多请求、会话复用或多模态验收。
- **已知缺口**：重算前的尾页 `needle` 字节检查仍有 false（本次 54 条），attention
  长度仍出现 `kv_len_clamped`；不能宣称中间压缩视图精确。重建请求的 MTP 投机被禁用，
  尽管全局配置仍为 MTP4（1+4 状态槽）。当前只确认最终答案一次正确。
- graph recall / AttentionFeedback（`graph_recall.hpp`）未接入（需注意力质量采样）。

## 一、这一版是什么

把 vLLM（SM75 fork）接成 KVMem 的后端，并让**打分/统计/检索语义**用 kvmem v0.19 核心自己的实现。

| 关注点 | 做法 | 依据 |
|---|---|---|
| mean-K 统计 | 每请求一个 `kvmem::RawKvStore`（`n_layer=16`、`n_embd_k=1024/rank`、`block_tokens=64`），`capture_k` 设备侧按 64-token 桶累加 pre-RoPE K | `memory_statistics.cpp` |
| Query | `capture_q` 设备上反旋转，产出 **pre-RoPE** 行；`PageQueryView` 缺席层为 `layer_rows=0` | `PageQueryView` |
| 打分 | **page-softmax**（sm75 CUDA，逐层 ≤64MiB），mean 聚合，LRFU **关** | `resolve_query_scoring()` |
| 压缩 | **只改地址**：物理页重排、槽位重算，K 字节原样拷贝、不重烘 RoPE | 设计稿 |
| 位置 | RoPE 用真实位置、`slot_mapping` 用压缩位置 | 本适配器 |
| 旧 Python 索引 | **默认不参与排序**（仅审计开关下读取） | 本适配器 |

## 二、已验证的配置（可照抄）

| 场景 | 参数 | 结果 |
|---|---|---|
| 池内正确性基线 | `GPU_UTIL=0.95 B=200000 INREQUEST_COMPACTION=0`（NVFP4） | 88,256 → `ZX-4417`；196,673 → `ZX-4417-QJ7` |
| 压缩可读性 | `B=36864` + `KVMEM_RESIDENT_PREFILL_COMPACT=1`，47K prompt | 1 次压缩 → `ZXZXZX-4417-QJ7` |
| 超池通道（旧基线） | `GPU_UTIL=0.65`（池 122,816）+ `KVMEM_SKIP_KV_CHECK=1` + `KVMEM_MAX_MODEL_LEN=200000` + `KVMEM_ENTRY_RESIDENT_PREFILL=1` | 152,820 → HTTP 200，答案错误 |
| 超池整体重算（实验） | 同上 + `KVMEM_COMPACT_REBUILD=1` + `KVMEM_PROMPT_END_RESELECT=1` + `KVMEM_RESIDENT_PREFILL_COMPACT=1`、TP2/MTP4、同步 V2 | 196,598 / 池 122,816 → 27 次压缩、最终重算 33,654 token、HTTP 200 → `ZX-4417-QJ7` |
| 会话复用（待验） | 默认关闭；显式 opt-in 仅支持 MTP=0、块对齐的完整驻留前缀以及双 TP 退休确认 | 尚无超池 cold/restore token-ID 相等验收；早期两轮 HTTP 200 不证明恢复正确 |

FP8/TP2、MTP=0 的 169,998-token 实验（物理池 161,832）仍未通过冷跑。首次把压力阈值改到物理池附近后，压缩在 145,088 和 149,184 连续发生并以 CUDA illegal access 退出；加入 R=16,384 的最小间隔后，压缩改在 145,088 与 161,472 发生，第二次压缩后的 forward 仍发生 CUDA illegal access。进一步核对发现两次 worker 前沿均比 EngineCore 当时的逻辑前沿落后 8,192 token。新增的双 TP quiescent/frontier/物理块校验仅是只读契约，**尚未接入调度边界或 GPU 搬页**，不能当作修复。此前缀 checkpoint 因 runtime 页映射缺口未能封存；原生 dense block-table 回查及 StateImage 绑定的修复只经离线测试。没有 warm token-ID 对照，不得宣称复用可用。

压缩视图现在在安装 `seq_lens`/positions 之前检查 `resident seq_len + 本步 query ≤ 已发布页容量`，不足即抛 `SparseViewCapacityError` 并由 worker hook 向上传递（不再静默钳制后继续 forward）；会话恢复同时保留引擎本步已同步的调度页，避免误拒合法恢复。`sparse_runtime_test` 与 `sparse_concurrency_test` 的 needle 失败经“移除该保护后逐字复现”确认为**既有偏差**，与本检查无关，尚未修复。

2026-10-30 的账本修正与再次实测：`after_step` 曾用“上一步输出的 logical 再减 n_new”作提交基数，使已执行 frontier 永久落后一个 chunk；`sync_row_from_engine` 又把 `seq_len` 设成 `tokens + n_tokens`（已含本步 query），与 `before_step` 的 `+n` 重复计数。两者已改为单一约定（`seq_len` = 本步之前已写入长度；提交追加本步）。169,998-token 冷跑因此推进到 **147,456 token 并完成第一次压力压缩**（worker 147,456 vs Core 151,552，剩 4,096 滞后），随后在下一步被容量保护明确拒绝：压缩后行只有 1,274 页（40,768 token），而下一步需要 40,960（36,864 + 4,096），差 6 页。这是**压缩后未为下一 chunk 保留 spare 块**的缺口，而非 CUDA fault；未修，且没有完整冷跑或 warm token-ID 对照。

worker 侧已加入 `runtime.quiescent_compact(req, frontier)`：在 Core 主线程调用时先提交已完成 chunk、拒绝过期 frontier，再选片/搬页并返回 proposal，全程不走 BlockChannel（`send=False`）；`Worker.kvmem_quiescent_compact` RPC 已安装但**尚无 Core 侧触发与提交**，默认行为不变。离线回归见 `tests/quiescent_compact_test.py`。

## 三、超池的两个硬约束（实测）

1. **块池上限 ≈ 90K**：池 76 块，三个 Mamba 组每请求固定占 15 块（1 状态 + 4 投机）→ 注意力只剩 ~58 块。
   所以 B 超过 ~90K 会在 `seq_len=86,016`、`free_blocks=9` 停住。
   （`mamba_block_size` 不能与窗口解耦：状态是每序列一份，改小会让分配器索要 `cdiv(seq,1616)` 个状态块
   → `exceeds row capacity (10 > 5)`。）
2. **正确性随压缩次数退化**：needle 能选中（排名第一、`mid_selected=True`），但保留页 KV 与循环状态
   仍是旧历史下算出的；尾部重放（QueryReplay 8K）只能修尾窗。

## 四、运行方式

```bash
MODEL=$HOME/models/<your-model> KVMEM_STATS=core KVMEM_FORCE_EAGER=1 \
  ./deploy/run-kvmem-vllm.sh        # 默认常驻模式；B/R/PORT 等见 run-kvmem-vllm.sh 头部
```

启动日志检查：`Spec decode: mtp/4`、`lrfu=False score=mean`、`[kvmem] device KV pool = … tokens`、
`[kvmem] attention ceiling = … tokens`。全部旋钮见 `deploy/run-kvmem-vllm.sh` 头部注释与 `PARAMETERS.md`。

## 五、调试设施（长期保留，见 `AGENTS.md`）

| 工具 | 用法 |
|---|---|
| 断点 | 写 `/tmp/kvmem_dbg.bp`（`file`/`func`/`line`/`hits`/`expr`）→ 命中落整帧到 `/tmp/kvmem_dbg.out`；**已接进 EngineCore 与入口中间件**，bp 文件改动自动重载 |
| 实时执行 | 写 `/tmp/kvmem_console.py` + `touch /tmp/kvmem_console.arm`，引擎下一步在 worker 内执行 |
| 热重载 | `touch /tmp/kvmem_reload.request` → 下一个 prompt 边界 reload + rebind |
| 探针 | `/tmp/kvmem_sparse.jsonl`（`device_pool`/`pool_ceiling`/`reselect`/`compact_applied`/`reconstruction_started` …） |

## 六、整体重算的验收与限制

`runtime.compact_rebuild=True` 经 worker console 实测，实际加载路径为本仓库 `engines/vllm`；
所传页号是**原始逻辑页**，经 `select_tokens(req.prompt_token_ids, selected, 16)` 提取，
物理槽号单独传给 EngineCore 回收块。`BlockChannel` 保留确认字典，不再把拒绝转换成 truthy。

本次端到端验证 **196,598-token / 122,816-token 池**的答案正确，不推广到其他 prompt、
跨会话状态或并发；后续需修复压缩期间 `needle=false` 与 `kv_len_clamped`，并以多组超池
needle、MTP 接受长度及跨请求回归扩展验收。所有实验开关默认关闭。

2026-10-30 的 Core 边界压缩实验（`GPU_UTIL=0.90 + KVMEM_SCHED_COMPACT=1`，复用仍关闭）：worker 不再走 socket 压缩（`compact_sent:false`），Core 在 `EngineCore.step` 输出后调用 `kvmem_quiescent_compact`，双 rank proposal 通过校验并**成功提交**：`sched_compact_applied` epoch=1、visible=36,864、offset=86,016、freed=55 blocks。冷跑从 147,456 推进到 **163,840 token**（resident 77,824），随后仍发生 CUDA illegal access（此时未再触发第二次边界压缩）。因此边界协议本身可工作，但后续连续 prefill 仍有地址/相位问题，需继续定位；未完成冷跑，无 warm 对照。

2026-10-30 第三次定位（`CUDA_LAUNCH_BLOCKING=1` 同步报错 + `KVMEM_NO_POS` 对照）：崩溃的真正原因是**压缩后 positions 被覆盖成压缩位置**。fork 在 `prepare_pos_seq_lens` 之后加 `kvmem_rope_offset`（RoPE 用真实位置），`prepare_attn` 再减 `kvmem_slot_offset`（写地址用压缩位置）；而 `before_step` 的 `_install` 在压缩后仍把 query positions 写成压缩值，`offset>0` 时槽位变成负值 → 越界/非法访问。代码已加入 `_positions_must_stay_true`（logical > resident 时不再写 positions），并有回归 `position_books_test`。用 `GPU_UTIL=0.90 + KVMEM_SCHED_COMPACT=1 + KVMEM_NO_POS=1` 的对照运行：**HTTP 200、169,998 prompt、64 completion、无 CUDA fault**，仅 1 次边界压缩（visible 36,864、offset 86,016、freed 55 blocks），token ids 已留存。

但这次回答是重复的 `No access ...`，不是正确 needle：边界压缩由 Core adopt 时回收了 worker 行尾部仍引用的块（行/游标不一致），下一 chunk 写入被回收/错位的页。该缺口尚未修，故不能宣称超池路径可用；无 warm 对照。

2026-10-30 第四次实测（强制多次边界压缩，`KVMEM_SCHED_COMPACT_RESERVE_BLOCKS=64`）：169,998-token 冷跑完成 **5 次** Core 边界压缩（epoch 1–5，visible 均 36,864，offset 24,576 → 122,880，每次 adopt 24 块、freed 16），**HTTP 200、无 CUDA illegal access**，prompt=169,998、completion=64、token ids 已留存。这满足“第二次及后续压力压缩后不再 fault”的判据；但回答仍是重复片段（`Based access access…`），说明压缩后的 KV/选片内容仍不正确——**不得**据此宣称超池压缩可用，也没有 warm 对照。

2026-10-30 重复问题定位（用户需求：池按用户份额划分，单人先 ~50K，压缩须控制在本人份额内）：重复的直接原因之一是**查询从未采集**——`KvmemRetrievalIndex.begin_query/add_query` 对模型步内创建的 inference tensor 做原地更新，在 `InferenceMode` 外抛 `RuntimeError: Inplace update to inference tensor outside InferenceMode`，16 层全部失败（日志 `bridge_harvest_layer_error` ×160），压缩选片因此使用零查询。已把原地更新包进 `torch.inference_mode()` 并加回归（`retrieval_index_test`）。实测修复后 `bridge_harvest_layer_error` 消失、`q_norm=2.82`、分数有效；但回答仍是重复片段，下一个嫌疑是**相位约定冲突**：orchestrator 压缩时把 K 重烘到压缩位置，而 fork 的 RoPE 用真实位置（`kvmem_rope_offset`），两者只能取其一（纯拷贝 + 真实相位，或偏移置 0 + 压缩相位）。该结论尚待实测，未声称重复已解决。

2026-10-30 重复问题的对照实验（服务留在 `GPU_UTIL=0.90 + KVMEM_SCHED_COMPACT=1 + reserve 64` 便于继续测试）：
① 查询采集修复后 `q_norm=2.82`、16 层有效、无 `bridge_harvest_layer_error`；
② orchestrator 改为**纯拷贝不重烘**（`KVMEM_REBAKE=1` 可回旧行为）后，`sparse_runtime_test` 与 `sparse_concurrency_test` 的 needle **全部通过**（此前 11272–11275 / 2568–2570 恒失败）；
③ 边界协议跑完 5 次压缩、HTTP 200、无 CUDA fault；但回答仍是重复片段，且与相位/位置配置无关（NO_POS、纯拷贝、偏移 0 三种变体 token ids 完全一致）；
④ **对照基线**：同一 169,998 prompt 在池 258K（不压缩）下正确回答 `ZX-4417-QJ7`，证明 prompt/入口/needle 正常，问题在压缩后的视图内容/选片；
⑤ 旧 socket 压缩路径在同池下仍 500（CUDA fault），边界路径是唯一能完成的。
下一步：核对压缩后写入选片视图的最后若干 chunk 的落页与 `seq_len` 覆盖（写入是否进了被回收页/错位槽），以及选片是否真的包含了 needle 附近内容。

2026-10-30 压缩路径定位（本轮，服务留在可测试配置 `GPU_UTIL=0.90 + KVMEM_SCHED_COMPACT=1 + reserve 64 + KVMEM_COMPACT_REBUILD=0`）：
1. 站点脚本默认 `KVMEM_COMPACT_REBUILD=1`，它**禁用** prompt 结束的 query 重排；且边界压缩从未设置 `compacted_in_prefill`，重建也从未触发（探针无任何 rebuild 记录）。这就是“中间 needle 永远不被重排”的 wiring 原因。
2. 改为 `KVMEM_COMPACT_REBUILD=0` 后，Core 侧新增“prompt 完成时最后一次 query 压缩”触发（`sched_compact_applied.final=true`），worker 侧 prompt-end reselect 在 sched 模式下禁用，避免走旧 socket 路径（实测该路径会让下一步差 1 页：38,945 需要 vs 38,944 覆盖）。
3. **needle 位置**：`ZX-4417-QJ7` 在 token 84,979、逻辑页 2,655（prompt 正中间；问题在 169,965）。
4. **选片证据**（每次压缩选择 1,152 页，sink 384 页）：86,016 处 needle 页在 recent 窗口内 → 选中；110,592 / 135,168 / 159,744 / 170,016（final，真实问题查询）→ **均未选中** page 2,655。因此重复的根因是**检索打分没有把 needle 块排进前 1,152**，不是协议/相位。
下一步：锐化 query（只取最后一段问题 token，而不是整个 chunk 的均值）并/或改块级打分（如按 query token 取 max），验收标准是 final reselect 的 `selected_pages` 包含 2655。

2026-10-30 检索链路的最后一段（live console，无需重启）：needle 页 2655 的 K 统计**存在**（`sums_n=5315`、`has_needle_sums=true`），core 评分为 0.0004、**排名约 2,080/5,315**，进不了前 1,152；`last_score_stats.diag.q_norm = 0.0`——**core 侧的查询为空**，打分因此不是内容驱动（top10 恰好是 needle 邻近块）。这解释了为什么 query 窗口/冻结/相位修正都不改变回答：选片从未真正按问题打分。
下一步：修复 bridge → core 统计/RawKvStore 的 query 交接（使 reselect 时 q_norm > 0），验收标准仍是 final reselect 的 `selected_pages` 含 2655 且回答为 `ZX-4417-QJ7`。官方默认配置已核对（page-softmax、分组聚合=平均、sink 12288、recent 1024、问题评分上限 512、回放上限 2048、图传播 0.2、LRFU 关），query 上限已改回 512。

2026-10-30 查询交接修复与其局限：已在 harvest 时把问题预填阶段的 query 行存进请求 index（`_q_rows`，冻结后不覆盖），并让 `core_selection_scores` 在 bridge 为空或查询已冻结时优先使用它。实测：预填压缩阶段 core `q_norm≈2.7`（bridge 有效），但 needle 页 2655 仍只在 recent 窗口（86,016）被选中；**最终 reselect（170,016）仍报 `q_norm=0.0`**，说明该处 `core_selection_scores` 没有拿到冻结行（`orch.retrieval_index` 与 harvest 使用的 index 可能不是同一对象，或 `_q_rows` 未写入）。下一步用 console 确认：`orch.retrieval_index is rt._retrieval[req]`、`len(index._q_rows)`、`_query_frozen`、以及 170,016 处 `core_selection_scores()` 的返回是否为 None。验收标准不变：final reselect 选中 2655 且回答 `ZX-4417-QJ7`。

2026-10-30 检索排序的最终定位：冻结问题 query 已生效（`idx_is_orch=true`、`q_rows=16`、needle 得分从 0.0004 升到 0.1447），但 needle 页 2655 排名仍约 2,628/5,315（≈中位）。原因指向**统计桶(64) 与选片块(32) 的粒度错位**：`align_scores` 把选片块 j 映射到统计桶 j//2；needle 那行只有 19 token（"The unique record KEY-MID has access code ZX-4417-QJ7."，token 84,979 起），**横跨统计桶 1327/1328**，均值 K 被劈开，两个桶都拿不到完整模式。实测 top10 恰好包含选片块 2656（=统计桶 1328，装着 code 后半段），而 2655（桶 1327）落选——模型只可能看到半行。
可选修法（待定，二选一或组合）：①把统计桶与选片块统一（统计 32 或选片 64），消除跨界劈裂；②对入选块做邻域扩张（对应官方“图传播强度 0.2”的图传播，我们尚未实现）。验收标准不变：final reselect 选中 2655，回答恢复 `ZX-4417-QJ7`。

2026-10-30 组合方案实现（用户指定：块统一 64 + 邻域扩张，对应官方图传播 0.2）：
① `SELECT_BLOCK_TOKENS` 默认改为 **64**（launcher 与 worker CLI），与统计桶 64 对齐，`align_scores` 不再做 64→32 重采样；
② 新增 `kvmem_recall.propagate_neighbor_scores(scores, alpha)`：一维邻域平滑（每个有限块加上 `alpha × 相邻有限块均值`），`KVMEM_GRAPH_ALPHA` 默认 0.2，接入 `core_selection_scores → set_block_scores` 之前；单测见 `retrieval_index_test`（峰值保留、邻居继承、非有限值不参与、alpha=0 为 no-op）。
实测（169,998 / 池 161,832）：块=64 + 软传播 0.2 → HTTP 200、无 fault，但 needle 页 2655/2656 仍只在 recent 窗口（86,016）被选中；needle 得分约 0.14，而 top 分约 26（差约 200×），0.2 的软传播无法跨过。
尝试“硬扩张”（把最高分块的 ±1 邻居加入必选集）时暴露了既有的 **spare 页缺口**：压缩后行 1,274 页（40,768）< 下一步需要 40,960，被 `SparseViewCapacityError` 拒绝；该扩张已回退。
下一步（按依赖顺序）：先修 spare 预留（压缩时保留 `visible + max_num_batched_tokens` 的整块并按块 adopt），再启用硬邻域扩张（`KVMEM_GRAPH_TOP`），最后复测 needle 选中与回答。

2026-10-30 spare 预留假设**被证伪并回退**：上一轮把硬邻域扩张触发的 `SparseViewCapacityError: needs 40960 but 1274 published` 归因于"压缩后未给下一个 chunk 预留页"，实现了 `adopted_row_pages` helper 与 `prefill_chunk_tokens` 发布。但用最近一次成功运行（软传播、无硬扩张）的 `row_synced` 时间线核对：每次 `sched_compact_applied` 之后行都从 1323 自然增长到 1421→1568→…，证明 **Core 的下一次 schedule 会重新分配块并由 sync 镜像回来**，adopted row 无需预留下一个 chunk。据此删除了 helper 与调用点改动，回到干净基线（`_reselect` 的 keep_pages 恢复为仅覆盖 selected 的块对齐值）。硬扩张那次的 1274 页差异属于 mandatory 扩大改变 selected/adopt 集合的副作用，不是 spare 缺口——该方向尚未解决，needle 仍未在 recent 窗口外被选中。
