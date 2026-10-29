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
| 会话复用 | `KVMEM_SESSION_REUSE=1`（catalog 接管状态镜像） | 两轮 200，`session_reuse_on` 正常 |

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
