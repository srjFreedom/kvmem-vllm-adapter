# kvmem-vllm 适配器交付说明（v0.19 对齐）

更新：2026-10-29。**压缩版**：结论 + 证据 + 下一步。逐轮过程见
[`archive/BEYOND-POOL-PLAN-log.md`](archive/BEYOND-POOL-PLAN-log.md)，超池机制图见
[`BEYOND-POOL-PLAN.md`](BEYOND-POOL-PLAN.md)。

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
   `KVMEM_BLOCK_SIZE=16` 解决块回收。

**未完成**

- **超池答案正确性**：保留页的 KV 与循环状态带"被淘汰历史"的近似，输出随压缩次数退化
  （0 次=正确 / 1 次=`ZXZXZX-4417-QJ7` / 9 次=乱码 / 25 次=`0`）。唯一出路是"按压缩顺序整体重算"，
  已实现骨架，卡在两处待验（见文末）。
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
| 超池通道 | `GPU_UTIL=0.65`（池 122,816）+ `KVMEM_SKIP_KV_CHECK=1` + `KVMEM_MAX_MODEL_LEN=200000` + `KVMEM_ENTRY_RESIDENT_PREFILL=1` | 152,820 → HTTP 200 |
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
B=36864|131072  INREQUEST_COMPACTION=0|1  KVMEM_STATS=core  KVMEM_FORCE_EAGER=1 \
ADAPTER=/home/srj/src/kvmem-vllm-adapter \
VLLM_DIR=/home/srj/src/kvmem-vllm-adapter/engines/vllm \
PORT=18200 ./deploy/run-kvmem-vllm.sh
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

## 六、留给下一步的两件事（按顺序）

1. **确认重建开关生效**：启动后用 console 读 `runtime.compact_rebuild`（两次相同 env 一次生效一次没有，
   说明有进程读不到；已加 worker env 兜底但未验证）。
2. **核对重建的 token 映射**：`_compact(rebuild=True)` 现在用
   `_all_token_ids[p*page_tokens : (p+1)*page_tokens]` 取 token，而 `p` 是**压缩后**的物理页 ——
   必须按"页 → 原始逻辑位置"的真映射取，否则重算的是错内容。先在 47K（1 次压缩）验证答案回到
   `ZX-4417-QJ7`，再上 152K 超池。
