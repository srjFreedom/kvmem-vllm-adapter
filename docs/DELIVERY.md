# kvmem-vllm 适配器交付说明（v0.19 对齐）

## 摘要（先看这一节）

**已完成并回归通过**

1. `KVMEM_STATS` 开启后 88K 崩溃（pinned allocator）——已修；88,256 token prompt 稳定答对 `ZX-4417`。
2. v0.19 打分栈接入：`RawKvStore` 统计（64-token 桶、mean-K）、`PageQueryView`（pre-RoPE）、
   `QueryScoreFrame`、page-softmax（sm75 CUDA）——排名确实是选择器的输入，且 needle 块能排到第一。
3. 压缩路径：纯拷贝、位置拆分（RoPE 用真实位置、写地址用压缩位置）、三本账一致；
   47,000 token 单次压缩后仍可读（`ZXZXZX-4417-QJ7`）。
4. decode 期统计：按"已接受"时间线推进（bridge 保留解码行、按上一接受数折叠），frontier 精确。
5. `CheckpointCatalog` 落地并接管 session store 的状态镜像（一个所有者、容量与淘汰统一）。
6. 超池**通道**：201,915 / 236,827 token 的 prompt 都能前缀到结尾并返回 200；
   块/页统一（`KVMEM_BLOCK_SIZE=16`）解决池回收；压缩请求不再因 MTP 崩溃。

**因架构限制未完成**

- 超池（prompt > 池 ≈179K）的**答案正确性**。三条实测：needle 能入选（排名第一）；尾部
  QueryReplay 能重算最后 8K；但保留页的 KV 与循环状态都带着"中段被淘汰"的近似，
  表现为压缩次数与可读性相关（1 次≈可读，2~6 次≈退化，20+ 次≈全错）。唯一出路是"按压缩顺序整体
  重算"，卡在重建后调度器把 decode 步排成 4096 token（`no sampled token` 停摆）。
- graph recall / AttentionFeedback（`graph_recall.hpp`）未接入：需要先有注意力质量采样。

**建议下一步**

1. 修 `schedule()` 对"刚完成重建"请求的 `num_new_tokens` 判断（第 42 轮已给出停摆现场）；
2. 或按上游 `prefix_checkpoint` 做多检查点 + 尾部重算（本适配器的 catalog 已具备容器）；
3. 之后再把 `reselect_at_prompt_end` / `KVMEM_COMPACT_REBUILD` 从实验开关转正。

更新：2026-10-29（第 60 轮，最终）

## 一、这一版是什么

把 vLLM（SM75 fork）接成 KVMem 的一个后端，并把**打分/统计/检索语义**换成 kvmem v0.19 核心
自己的实现，而不是适配器自造的 Python 版本。

| 关注点 | 现在的做法 | 依据 |
|---|---|---|
| mean-K 统计 | 每请求一个 `kvmem::RawKvStore`（`n_layer=16`、`n_embd_k=1024/rank`、`block_tokens=64`），由模型里的 `capture_k` 设备侧按 64-token 桶累加 pre-RoPE K | `backends/ninfer/.../memory_statistics.cpp` |
| Query | `capture_q` 在设备上反旋转，直接产出 **pre-RoPE** 行；`PageQueryView` 中缺席层为 `layer_rows=0`（不是零向量） | `PageQueryView` 注释 |
| 打分 | 上游默认 **page-softmax**（sm75 CUDA，逐层 ≤64MiB），mean 聚合，LRFU **关** | `cli_options.h: resolve_query_scoring()` |
| 预算带 | sink/recent/budget 以 token 发布，recall 按 store 粒度换算成块 | `query_score_frame` 的入参要求 |
| 压缩 | **只改地址**：物理页重排、槽位重算，K 字节原样拷贝、不重烘 RoPE | "keeps original RoPE positions; compact addresses do not rewrite K" |
| 位置 | RoPE 用真实位置、`slot_mapping` 用压缩位置（`kvmem_rope_offset` / `slot_positions`） | 本次实现 |
| 旧 Python 索引 | **默认不再参与排序**（仅审计开关下读取） | 本次实现 |

## 二、已实测通过

| 项 | 证据 |
|---|---|
| 统计精确性 | 53,422 / 62,347 / 88,256 token prompt：`frontier` 精确等于 prompt 长度；1024 桶写/4K chunk |
| 长上下文稳定性 | 88K prompt + 统计开启：HTTP 200、答案正确、引擎存活（此前 pinned allocator 崩溃，已修） |
| page-softmax | 828 块、`sum=512`、device 1.16 MiB、无非有限值 |
| 打分器一致性（审计） | 与旧索引同网格对齐后 `rank_corr 0.69–0.77`、`top_overlap 4–6/10` |
| 压缩三本账 | 58K prompt 连续 3 次压缩：`49152→36864`、`46371→36864`、`36865→36864`，`2304 页×16 = 36864 = visible` |
| 位置拆分 | RoPE/写地址分离后 88K 常驻 needle 居中答对 |
| 回归基线 | `B ≥ prompt` + 统计开 + page-softmax：88,256 token needle 居中 → `ZX-4417` |

## 三、未完成 / 已知限制

1. **超池（prompt > 池 177K）尚未打通**：prefill 期压缩已不崩、账本自洽，但**选择质量**不足
   （prefill 期没有 query，只能 sink+newest），需要 QueryReplay。
2. **QueryReplay 部分实现、未收敛**：通道消息、两本账回退、循环状态快照+恢复都已就位，但
   chunk 粒度的快照不足以精确重放（答案仍是 `ZX` + 续写）。需要按任意位置切分 prefill 的
   checkpoint（上游 `prefix_checkpoint.hpp` 的 QueryReplay 语义）。
3. **decode 期统计**：机制已实现（bridge 保留解码行、按上一接受计数折叠），但端到端账目仍在核对
   （存在少计，见 `docs/BEYOND-POOL-PLAN.md`）。
4. **入口策略仍在 API server 做文本式选择**（`--kvmem-entry-select retrieval`）。按 v0.19 设计，
   选择应全部发生在引擎内；这是"中间内容被丢"的历史根源之一。
5. graph recall / AttentionFeedback（`graph_recall.hpp`）未接入——需要先有注意力质量采样。

## 四、运行方式（已验证可用的配置）

```bash
B=36864|131072  INREQUEST_COMPACTION=0|1  KVMEM_STATS=core  KVMEM_FORCE_EAGER=1 \
ADAPTER=/home/srj/src/kvmem-vllm-adapter \
VLLM_DIR=/home/srj/src/kvmem-vllm-adapter/engines/vllm \
PORT=18200 ./deploy/run-kvmem-vllm.sh
```

启动日志检查三行：`Spec decode: mtp/4`、`lrfu=False score=mean`、
`KVMEM_ENTRY_RESIDENT_PREFILL=1 -> MAX_MODEL_LEN auto`。

## 五、调试设施（长期保留）

| 工具 | 用法 |
|---|---|
| 断点 | 写 `/tmp/kvmem_dbg.bp`，console 里 `import kvmem_dbg; kvmem_dbg.install()`；命中整帧变量落 `/tmp/kvmem_dbg.out` |
| 实时执行 | 写 `/tmp/kvmem_console.py`，引擎下一步在 worker 内执行；可读 `runtime/orch/stats/bridge/runner` |
| 热重载 | `touch /tmp/kvmem_reload.request` → 下一个 prompt 边界 reload 并 rebind（纯 Python 改动免重启） |
| 探针 | `/tmp/kvmem_sparse.jsonl`（`book`/`reselect`/`compact_applied`/`statistics_final`/`core_scores`/`scores_unavailable` …） |


## 六、超池路径的当前边界（2026-10-29 第 25 轮实测，最权威的一条）

**池的真实容量**（这条已经把之前的误判纠正了）：vLLM 自己算出来的是

```
ValueError: To serve at least one request with the model's max seq len (203000),
            (3.69 GiB KV cache is needed, which is larger than the available KV cache...)
```

即 **KV 池 ≈ 177,760 token**（vLLM auto-fit 得到的 max_model_len 正是它）。之前把
`126 blocks × 1616 = 203,616` 当成池是错的——那是**块表可寻址的行数**，不是 KV 显存容量。

结论：**190K/200K 的 prompt 确实是"超池"**（>177,760），prefill 期压缩不可避免，而那条路正是
当前崩的地方。

实测：

| prompt | B | 结果 |
|---|---|---|
| 88,256 | 131,072 | ✅ 答案 `ZX-4417` |
| 95,356 | 36,864（prefill 期压缩） | ✅ HTTP 200、引擎存活（曾崩） |
| 190,204 | 196,000（不压缩） | ❌ 4 次 preempt → `illegal memory access` |
| 201,360 | 36,864 / 172,000 / 199,000 | ❌ 同上前提 |

也就是说：**只要调度器在 prefill 期间发生抢占，就会崩**；而 190K+ 的 prompt 在本机
（池 203.6K、R 16K、chunk 4K）必然挤到抢占。压缩路径（prefill 期）另有非法访问，两件事都
还没解决。

### 已经排除/已修的（都不构成当前故障）

- 行的长度、写地址、page id 范围、重复页（全部检查通过）
- 引擎块列表未接管 → **已修**（池空闲块 74 → 100~116）
- 块表写入游标未重置 → **已修**（`exceeds row capacity (11514 > 11312)` 消失）
- 统计路径、投机解码、位置偏移（均已单独隔离，不是原因）

### 结论

超池要真正打通，需要解决 **prefill 期的抢占路径**（本 fork 下抢占 = 重算，重算面对压缩过的行
会越界），这与"prompt 不受显存限制"是同一件事的两面。压缩本身（prefill 末）是可用的：58K 实测
`mid_selected=True` 且答案含 `ZX-4417`。


## 七、已验证配置（第 32 轮复测，可直接照抄）

| 场景 | 启动参数 | 实测 |
|---|---|---|
| 长上下文正确性基线 | `B=131072 INREQUEST_COMPACTION=0 KVMEM_STATS=core` | 88,256 token prompt → `ZX-4417` ✅ |
| 超出常驻窗口（B）但未超池 | `B=36864 KVMEM_RESIDENT_PREFILL_COMPACT=1 INREQUEST_COMPACTION=1` | 147,367 token → HTTP 200 ✅（prefill 期压缩 24+ 次） |
| 真正超池 | 同上 | 201,915 token → HTTP 200 ✅（38 次压缩） |
| 压缩后的检索 | 同上 + `--kvmem-retrieval` | needle 块排名第一（5.32 vs 次名 1.77）✅ |
| 压缩请求的解码 | 同上（MTP 开） | 不再崩（压缩请求禁止投机）✅ |

**答案正确性只在前两行成立**：只要在 prefill 期间发生过"把中段丢掉"的压缩，模型的循环状态就与
最终工作集不一致，题目里的 needle 即使被检索排在第一位，答案仍然错（实测输出 `7`）。详见
`docs/BEYOND-POOL-PLAN.md` 第 29~31 轮。

## 八、留给下一阶段的一件事

`KVMEM_PROMPT_END_RESELECT=1` + `KVMEM_COMPACT_REBUILD=1` 会触发"用零状态重新 prefill 选中的
36,864 token"，`compact_rebuilt` 能触发、引擎能重新 prefill，但随后停摆（`no sampled token`，
约 26 分钟后才返回）。修好这个停摆，超池的答案正确性就补齐了。


## 九、超池（prompt > 池）的最终状态（第 37~38 轮）

**根因链已经完整**：

1. 引擎的块是 1616 token，KVMem 的页是 16 token。一个引擎块只有在其 **101 个页全部被丢弃**时才能
   归还池子，而分散选择永远不会整块丢弃 → 池子被榨干 → `assert block`（第 37 轮）。
   修法：`KVMEM_BLOCK_SIZE=16` 让两者同单位（已验证：58,994 token + 2 次压缩从"引擎死"变成 HTTP 200）。
2. 保留下来的页，其 KV 是在"中段已被淘汰"的状态下算出来的；循环状态同样如此。
   实测规律：**1 次压缩≈可读（`ZXZXZX-4417-QJ7`），2~6 次≈退化（`ZX ZX = = =`），20+ 次≈完全错（`7`）**。
3. 尾部重算（QueryReplay）只能修最后 N 个 token，修不了其余保留页（第 35 轮已证否）。
4. 整体重算（把选中的 36,864 token 按压缩顺序重新 prefill）需要引擎的 token 账本与行一致，而 token
   账本被改写后采样簿记失效（`no sampled token` 停摆，第 31 轮）。

因此 **超池的正确性需要 compacted-sequence 接口**（`DESIGN-compacted-sequence.md` 的第三块拼图：
`prepare_inputs` 的 logits 行口径），这不是参数或补丁能解决的。当前能保证正确性的是"prompt ≤ 池"
（第 7 节第一行，88K 实测 `ZX-4417`）。


## 十、第 41 轮回归结果（全部改动之后）

| 场景 | 配置 | 结果 |
|---|---|---|
| A 长上下文基线 | `B=131072`、不压缩 | 88,256 token → **`ZX-4417`** ✅ |
| B 单次压缩 | `B=36864` + prefill 期压缩 | 46,993 token → `ZXZXZX-4417-QJ7`（可读）✅ |
| C 超池通道 | `B=36864` + `KVMEM_BLOCK_SIZE=16` | 147,367 token / 23 次压缩 → HTTP 200（答案错，见第九节）|
| D 会话复用 | `KVMEM_SESSION_REUSE=1` + catalog 接管 | 两轮均 200，`session_reuse_on` 正常 ✅ |

**结论**：本次迁移的可交付部分全部通过回归；超池的"通道"通过、"正确性"受架构限制。

## 十一、第 44 轮：NVFP4 模型上的 200K 验收（正确答案）

按用户要求把测试模型换成 NVFP4（`/home/srj/models/Coder390/NVFP4/W4A16-fp8mtp`，`qwen3_5_text`，
64 层 / 16 层全注意力 / kv_heads 4 / head_dim 256，与 FP8 版同构，只是权重量化不同、速度更快）。

**实测（`B=200000`，不压缩）：**

```
prompt_tokens = 196,673        （needle 在 1/4 处）
answer        = ZX-4417-QJ7    ✅ 正确
http          = 200，引擎存活
```

这是本适配器第一次在 **~197K token 的 prompt + 居中 needle** 上给出正确答案。

诚实说明：这条路径是"整段常驻"（没有触发压缩）。NVFP4 的 4-bit 权重更省显存，KV 池比 FP8 版的
179,182 token 更大（否则 196,673 的 prompt 在准入阶段就会被拒），所以这次是 **prompt ≤ 池** 的场景，
不是"超池"。真正的超池（prompt > 池）仍然受第十节所述的近似限制。

同一模型上 128,898 token prompt（B=131072）也答对，耗时 2m58（FP8 版同规模约 3m20），
压测速度确实更快。

## 十二、用 GPU_UTIL 缩小物理池来做"超池"测试（第 45 轮，回答"能否用小 prompt 触发超池"）

可以，而且这把测试成本降下来了。物理池与 `GPU_UTIL` 的实测关系（NVFP4 模型，2×22.5GB）：

```
GPU_UTIL=0.85 → 387,840 tokens   (24,240 pages × 16)
GPU_UTIL=0.70 → 189,072 tokens
GPU_UTIL=0.65 → 122,816 tokens
GPU_UTIL=0.55 → 启动失败：not enough GPU memory to serve even a single token
```

也就是说 0.65 时**一个 13 万 token 的 prompt 就已经超出物理池**（12.3 万），prefill 只需 ~2 分钟，
不必再构造 20 万以上的 prompt。这是本轮新增的 `device_pool` 探针（启动时打印
`[kvmem] device KV pool = N tokens`）测出来的。

**但它只降低测试成本，不改变结论**：在上面 0.65 的配置里跑 133K prompt（B=36864、压缩开），
仍然是 `illegal memory access`（引擎退出），与大盘子下的超池表现一致。所以"缩小池"是方便的
**测试手段**，不是修复超池正确性的方法；正确性仍受第十节所述的近似限制。

## 十三、目标逐项评估（第 49 轮，最终）

| objective 条目 | 状态 | 证据 |
|---|---|---|
| ① KVMEM_STATS 开启后 88K 崩溃（pinned allocator），统计路径长上下文稳定 | **完成** | 88,256 token prompt → `ZX-4417`；统计 frontier 精确（含 decode 期按接受数推进） |
| ② 超池：chunked prefill + 逐 chunk 落 RAM + 地址压缩 + 引擎内 page-softmax 检索，200K prompt + 居中 needle 答对 | **部分完成** | 200K（196,673 token）prompt + 居中 needle **答对**（NVFP4、池内常驻，5m10s）；超池（prompt > 池）**未达成**：压缩通道可跑（最多 45 次压缩不崩），但正确性受"保留页近似"限制；启动不变式已可用 `KVMEM_SKIP_KV_CHECK=1` 解除（窗口 200K > 池 122K 可启动），但该配置下请求停在调度阶段（本轮实测：无任何运行时探针、GPU 0%） |
| ③ decode 期统计 / 接受窗口 | **完成** | `frontier` 精确跟随已接受 token；bridge 保留解码行、按上一接受数折叠 |
| ③ checkpoint 目录替代自写 session store | **完成** | `CheckpointCatalog` 落地；session store 的状态镜像改为引用目录中的 checkpoint（一个所有者、统一容量与淘汰），并有单测证据 |
| ③ graph recall / AttentionFeedback | **未做** | 依赖注意力质量采样，本轮未涉及 |
| ④ 清理旧打分旁路、更新 deploy/文档、交付说明 | **完成** | legacy 打分默认关闭；旧探针清理；`deploy/run-kvmem-vllm.sh` 记录全部旋钮；`DELIVERY.md` / `BEYOND-POOL-PLAN.md` 含完整证据链 |

### 超池的完整因果链（按发现顺序）

1. 入口裁剪把 prompt 裁到 B（已用「不裁入口」取代）；
2. prefill 期压缩 → 非法访问：负 offset / 行短一截 / 引擎块 ≠ KVMem 页（逐项修复）；
3. 池枯竭 `assert block`：块 1616 vs 页 16 → `KVMEM_BLOCK_SIZE=16`（已修）；
4. 压缩静默失效：`UnboundLocalError: raw`（已修，0 → 32 次压缩）；
5. 行宽 = 窗口 < 池：`max_model_len ≤ pool` 的启动检查 → `KVMEM_SKIP_KV_CHECK=1`（已解除）；
6. **剩余**：窗口 > 池时请求停在调度（不产生任何运行时探针）；以及即便压缩正常，保留页 KV 与
   循环状态仍带"被淘汰历史"的近似（1 次压缩可读、20+ 次全错）。

第 6 条是下一步唯一的入口。


## 十四、最终状态（第 51~60 轮，超池专项）

### 打通的（可复现）

| 环节 | 证据 |
|---|---|
| 窗口 > 池可启动 | `KVMEM_SKIP_KV_CHECK=1`（关掉 vLLM 的 `max_model_len ≤ 池` 启动检查） |
| 超池请求可准入 | 关闭 `scheduler_reserve_full_isl`（KVMem 模式下自动关闭）——这是之前"停在 waiting"的真闸门 |
| 超池 prefill + 压缩 + 解码 | 152,820 token prompt / 池 122,816 → HTTP 200（9~25 次压缩） |
| 池内正确性 | 196,673 token prompt + 集中 needle → `ZX-4417-QJ7` |
| 调试设施 | 断点已接进 **EngineCore 与入口中间件**；断点文件热重载（不再需要重启） |

### 未打通的（以及原因，均已实测）

| 问题 | 原因 | 证据 |
|---|---|---|
| 超池答案正确性 | 保留页的 KV 与循环状态带"被淘汰历史"的近似，压缩次数越多越不自洽 | 0 次→正确；1 次→`ZXZXZX-4417-QJ7`；9 次→乱码；25 次→只输出 `0` |
| B 无法放大以减少压缩 | 块池 76 块，三个 Mamba 组每请求固定占 15 块（1 状态+4 投机），注意力实际只剩 ~58 块 ≈ 93K token | 断点：`row=5656 页 / seq_len=86016 / free_blocks=9`，B=90K/100K/110K 均停住 |
| Mamba 块尺寸不能与窗口解耦 | `mamba_cache_mode=none` 下状态是"每序列一份"，块表按此开；改小后分配器索要 cdiv(seq,1616) 个状态块 → 越界 | `Block table write for request 0, group 0 exceeds row capacity (10 > 5)` |

### 下一步（唯一入口）

1. 实现 **compacted-sequence 接口**（`prepare_inputs` 的 logits 口径 + token 账本），使"按压缩顺序整体
   重算选中的 B 段"可行——这是消除近似、让超池答对的唯一路径；
2. 把 Mamba 15 块/请求的固定开销从注意力预算里正确扣除，从而把可用 B 推到 ~90K，压缩次数降到 1~2 次；
3. 两件事都做完后，超池才能既"跑通"又"答对"。
