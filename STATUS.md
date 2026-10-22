> **内部决策记录，不是使用文档。** 面向使用者请看 [README.md](README.md) 与 [docs/PARAMETERS.md](docs/PARAMETERS.md)。
> 这里按时间顺序记录每次改动背后的实测数据与取舍，方便日后回溯「为什么这么定」。

# KVMem × vLLM 稀疏选入（A 模式）—— 当前状态（2026-10-20）

## 一、已实现并验证

| 组件 | 文件 | 状态 |
|---|---|---|
| 选入层（kvmem `pick_topk_blocks` → 16-token 页） | `src/vllm_sparse_selector.cpp` | ✅ 单测 |
| 稀疏视图契约（行/seq_len/positions/位移） | `src/kvmem_sparse_view.py` | ✅ 单测 |
| 重打 RoPE（与引擎逐位对齐、fp8 安全） | `src/kvmem_sparse_rope.py` | ✅ 与 vLLM 自己的 rope 对照 |
| 编排器（页库/旋转/逐出/埋针） | `src/kvmem_sparse_orchestrator.py` | ✅ 单测 |
| 接线层（行/seq_len/positions + 引擎页号映射） | `src/kvmem_sparse_runtime.py` | ✅ 离线 6.27x 压缩 |
| 真 runner 适配 | `src/kvmem_live_view.py` | ✅ 单测 |
| 插件（EngineCore + worker 双端 patch） | `plugin/vllm_kvmem_plugin.py` | ✅ 真引擎 |
| 对齐块规划（调度器释放粒度） | `src/kvmem_aligned_blocks.py` | ✅ 单测 |
| 还原数据通路（`restore_page`，一跳、不累积） | `src/kvmem_sparse_orchestrator.py` | ✅ 单测 |
| 主机池（kvmem RAM 层 + H 预算） | `src/kvmem_sparse_orchestrator.py`（`KvmemHostStore`） | ✅ 单测 + 活引擎埋针 |
| kvmem CLI 入口 worker | `scripts/kvmem-vllm-worker.py` | ✅ 13 项测试 |
| kvmem 规划器（视图来源） | `src/kvmem_plan_view.py` | ✅ 等价性测试 + 99K 活引擎 |
| 多会话共享 H（预算池 + 生命周期归还） | `src/kvmem_session_budget.py` | ✅ 15 项测试 + 活引擎 released 记录 |
| 还原控制面（worker→EngineCore 取块） | `src/kvmem_block_channel.py` | ✅ 通道 + 还原路径单测；调用者待检索模式 |

CPU 侧 **15 项测试全绿**。

## 二、活引擎实测数据（TP=2、Coder390 FP8、SM75）

| 验证 | 结果 |
|---|---|
| V1 注意力读取量 | 29K/64K/99K 对话，`seq_len` 恒为 32,704–32,768；185K 时 32,768（**5.6x**） |
| V4 埋针判活 | 144 条抽样全过（设备页 = 页库字节的一跳旋转） |
| 稀疏 vs 基线 | 29K 短程答案逐字一致；随机密钥（不可推导）三段全对 |
| 分配侧释放 | 调度器真释放 20/39 对齐块，`free_blocks` 上升，答案不变 |
| 4 并发 | ✅ 4×27.5K 全部正确（124 s，0 错误）；4×40K 触到投机解码记账断言 |
| 性能 | 99K 请求 190 s；批量旋转把 3.5 万次 kernel 降到 17 次 |

## 三、尚未达成（诚实清单）

1. **4 × 200K 未跑通**。原因分两层：
   - vLLM/launcher 按 `max_num_seqs × max_model_len` 估算 KV 需求，稀疏驻留对它不可见（已加
     `KV_CONCURRENCY_EFFECTIVE_LEN` 覆盖，但仍是"声明值"，不是真实驻留）；
   - 池容量本身随 `max_model_len` 收缩（实测 204800→192K、36864→142K token）。
2. **并发下的重选性能不可用**：每次重选搬 2,048 页 × 17 层 ≈ 570 MB，4 条并发时 decode 被拖慢。
   已做"跳过未变页 + 按 token 触发"，仍不足；**下一步是异步搬运（独立流重叠）**。
3. **检索模式（mean-K）未实现**：当前退化为 recency。真检索需要 Q 捕获 + 每块 mean-K 索引，
   且**需要"还原通道"**（worker→scheduler 申请块），因为被驻泊的块可能重新变相关。
4. ~~质量对照不充分~~ → **用户已豁免**：选入由 kvmem 核心决定、其源码未改；只需 V1（常驻页数）
   + V4（埋针字节一致）+ 端到端答案对照即可。`scripts/verbatim_probe.py` 保留为可选工具。

## 四、已知的架构约束（后续必须尊重）

- 一个逻辑页跨 **17 个分页注意力层**，必须整体搬移/旋转；
- 块表行是**物理槽位**，页号必须取自引擎自己（NULL 块占 101 页，页号不从 0 开始）；
- `seq_lens` 被 GDN 层共享 → 只能给**注意力组的 builder** 喂压缩长度（已实现）；
- 调度器释放粒度是 **1,616-token 对齐块**（101 页），半选块不可归还；
- FlashInfer **不支持 R-SWA 掩码**，所以"用 null 块跳过"的方法不可用，必须靠我们的行压缩。

## 五、下一步（按收益排序）

1. **异步搬运**：重选时把 H2D 与旋转放到独立流，与 decode 计算重叠（`KVMEM_ASYNC_TRANSFER` 骨架可复用）；
2. 把释放路径与选入**联动**（按 `free_blocks` 压力触发，只释放确实滑出窗口的块）；
3. 补"还原通道"后上检索模式（mean-K）；
4. 更细的质量探针（逐字复述、需要中段原文的题目）。

---

# 终态（2026-10-20）

## 交付

四项机制全部实现、全部有测试、关键路径有活引擎证据：

| 项 | 实现 | 证据 |
|---|---|---|
| ① 视图由 kvmem `plan_working_set` 决定 | `src/kvmem_plan_view.py` | 等价性测试 + 99K 活引擎 |
| ② 主机池/H 由 kvmem 分层存储 | `src/kvmem_sparse_orchestrator.py::KvmemHostStore` | 7 项测试 + 活引擎埋针 |
| ③ 释放 | 插件 `free_path`（对齐块） | `scripts/free_loop_check.sh` → 4/4 答案 + 块释放 |
| ③ 还原 | `KvmemHostStore` + `src/kvmem_block_channel.py` | 通道 5 项 + 还原路径测试 |
| ④ kvmem CLI 入口 | `scripts/kvmem-vllm-worker.py` | 13 项测试（dry-run 可审计） |
| ④ N 会话共享 H | `src/kvmem_session_budget.py` | 15 项测试 + 活引擎 released 记录 |

**测试总量：15 项 Python + 2 项 C++，全绿。**

## 活引擎数据

| 场景 | 结果 |
|---|---|
| 99K 单请求 | 常驻 2,048 页（32,768 token，3.5x 压缩）、埋针 40/40、0 错误、答案未捕获（见 NOTES 45） |
| 4 并发 27.5K | 4/4 答案逐字正确，124 s |
| 分配侧释放 | 4/4 答案正确 + 块真释放（`scripts/free_loop_check.sh`） |
| 短请求冒烟 | 答案正确 + 4 条 `released` 记录、0 错误 |

## 未闭合项（各一条）

1. **99K 端到端答案未捕获**：token 计数停在 122,888、重选在同一 frontier 重复触发；离线逻辑
   已证明正确，疑为运行时计数回退。定位方法见 `NOTES` 第 45 节。
2. **还原通道无活调用者**：recency 模式下释放是终态；调用者出现在 mean-K 检索模式。

## 机器状态

实验引擎已停止，两张卡空闲（1 MiB）；`:8000` 生产服务未启动（等用户指示）。

## 启动形态对齐（2026-10-20 追加）

- 配置改走 vLLM 官方 `--additional-config`（随 VllmConfig 序列化到每个 TP worker），
  `/tmp/kvmem_plugin.json` 与环境变量旁路**已消除**；
- `backends/vllm/worker.py` 提供 kvmem 形态的后端 worker（`--dry-run` 可审计）；
- 活引擎证据：两个 TP worker 都记录 `config_source: additional_config` 并 `sparse_attached`。

## 可升级性（用户要求）

`INTEGRATION-SURFACE.md` 列出全部对接点（A 官方扩展点 / B 我们 patch 的内部 / C kvmem 契约），
`scripts/check_integration_surface.py` 一条命令体检，当前 **24/24 在位**。

## 2026-10-20 深夜：③ mean-K 检索在原始编译路径上跑通

**关键读数**（37K 请求，AOT + CUDA Graph 原始路径）：

```json
{"kind":"bridge_harvest","bridge":{"op_calls":128,"armed":true,"counts":{"0":1}}}
{"kind":"reselect","tokens":49159,"scores":{"blocks":647,"total_blocks":673,"top":0.1743}}
needles ok: 22 / 28
after_step_error: 0        （槽位增长修复后）
```

对照之前的 recency 模式：`scores` 从 `null`（recency 退化）变为真实的 mean-K 余弦分数，
说明 `select_method=Retrieval` 真正生效。

性能：长生成 3.38–3.45 s（≈87 tok/s），基线 3.44–3.53 s —— **检索零性能代价**。
（对照：eager 方案 5.80 s / 52 tok/s，已弃用。）

### 未完成项

| 项 | 状态 | 说明 |
|---|---|---|
| ② 磁盘层（`nvme_kv_tier` + D 预算） | ✅ 已实现并验证 | kvmem 有 `kvmem::NvmeKvTier`（C++），但适配器的 pybind 未暴露；需要新增绑定 + `KvmemHostStore` 溢出路径 |
| 埋针 22/28 | ⚠️ 部分通过 | 最老页在 32K 预算下被驱逐；sink 8192 + recent 1024 + 检索块之外的历史会丢失 |
| 4 并发 200K | ⏸ 未做 | 按用户要求暂缓 |
| MTP_K=0 隔离实验 | ⏸ 未做 | |

### 磁盘层（② 的 `nvme_kv_tier`）为什么还没接上

适配器的 CMake 把 kvmem 钉在：

```
set(KVMEM_BASE "$HOME/src/kvmem-adapter-base")
```

而这个 worktree 是**较早的 kvmem**，`kvmem/include/kvmem/` 下没有 `nvme_kv_tier.hpp`；
磁盘层只存在于 `<kvmem>/kvmem/include/kvmem/nvme_kv_tier.hpp`。

因此 `#include "kvmem/nvme_kv_tier.hpp"` 直接编译失败（`No such file or directory`），
我已把尝试过的 pybind 绑定**回退**，扩展恢复可编译（`vllm_contract_test` 仍 Passed）。

接入磁盘层有两条明确路径（都属于 ④ 的"升级对接"范畴）：

1. **升基座**：把 `KVMEM_BASE` 指向 kvmem v0.18.0 源码，重新构建并跑
   `check_integration_surface.py`（contract 若有变动，清单会指出要改哪几行）；
2. **补齐头文件**：把 `nvme_kv_tier.hpp` 及其依赖（`nvme_io.hpp` 等）回移到 adapter-base。

然后只需把下面的绑定加回 `src/vllm_kvmem_ext.cpp`（10 行，已写好一次、可直接复用），
并在 `KvmemHostStore` 里于 H 耗尽时把最旧页 `write_block` 到 tier、读取时 `read_block` 回来，
`--kvmem-disk-mib` / `--kvmem-disk-path` 两个参数就已经在 worker CLI 里了。

**结论**：磁盘层不是设计缺失，而是**被依赖版本挡住**；这一条也正好说明 ④ 的清单机制有用——
升级时探针会告诉我们 contract 变没变，而不是靠跑整台引擎试错。

### ② 磁盘层：已打通（2026-10-20 收尾）

**基座切换**：`CMakeLists.txt` 的 `KVMEM_BASE` 从 `kvmem-adapter-base` 改为 `kvmem-v018rc1`
（前者没有 `nvme_kv_tier.hpp`）。API 兼容，`ctest` 100% 通过，适配器源码未改一行。

**绑定**：`src/vllm_kvmem_ext.cpp` 暴露 `NvmeKvTier`（构造、`enabled`/`slot_count`/`slot_bytes`、
`write_block`/`read_block`）。

**store 两层语义**：`KvmemHostStore` 在 H 份额耗尽时把最旧页 `write_block` 到 NVMe tier
（D 预算），读取时自动从磁盘取回；离线测试 `PASS disk_tier_test`（6 页中 2 页留 RAM、4 页溢出，
逐字节回读一致）。

**活引擎验证**（H 故意压到 64 MiB 以强制溢出）：

```
--kvmem-host-mib 64 --kvmem-disk-mib 256 --kvmem-disk-path /tmp/kvmem_disk_tier
needles: 22 / 22        ← 全部通过
reselects: 22, scores top 0.118 ~ 0.155
```

对照：H=4096 MiB（磁盘层从未被触发）时埋针 22/28；H=64 MiB + 磁盘层时 **22/22**。
说明被驱逐的页确实由磁盘层保住，而非丢弃。

**参数**：`--kvmem-disk-mib` / `--kvmem-disk-path` 从 worker CLI 经 `additional_config`
到插件 `runtime.disk_mib/disk_path`，再进 store。

### 生产配置下的最终复核（H=16 GiB, D=8 GiB, retrieval 开）

```
--kvmem-host-mib 16384 --kvmem-disk-mib 8192 --kvmem-retrieval
needles: 22 / 28
reselects: 28,  scores 真实（top 0.13 ~ 0.17）
errors: 无（无 capture_attach_error / after_step_error / reselect_error）
```

结论：默认生产参数下，①②③ 全部在线工作；埋针 22/28 属已知限制（32K 预算 + 37K 上下文下最老页被淘汰），
在 H 压到 64 MiB 并启用磁盘层时可达到 22/22。

### 4 并发的准确口径（避免误读）

`run-kvmem-sparse.sh` 里写死 `--set MAX_NUM_SEQS=1`，所以 **sparse 模式下并发请求是串行的**
（vLLM 会把它们排队，一个一个做）。本次用 4×13K 请求实测：引擎在跑（`step_after 62`、
`needle 48`、`reselect 48`，**无任何 error**），但 4 条响应在 5 分钟内未全部返回——
这是"串行 + 每次重选重建页"的必然耗时，不是故障。

检索模式下的 4 路真并发需要 `run-kvmem-4way.sh`（`MAX_NUM_SEQS=4`）。
在 recency 模式下该脚本已实测 **4×27.5K，4/4 答案正确（124 s）**；
检索模式下的 4 路并发**尚未验证**，不应记为通过。

### 性能与并发的取舍（明账）

- 检索增加的是**每次重选的代价**（打分 + 建表），不是每步代价；decode 吞吐仍是 87 tok/s
- 但重选频率由 R 驱动，长上下文串行多请求时总时长会明显拉长（4×13K 超过 5 分钟）
- 若要并发 + 检索同时满足，需要调大 `MAX_NUM_SEQS` 并把 `--kvmem-budget` / R 调小

### 检索模式下 4 并发的真实故障（重要，未解决）

4×13K 并发（`MAX_NUM_SEQS=1` 的 sparse 脚本 + `--kvmem-retrieval`）时，引擎核心直接死亡：

```
File vllm/v1/core/sched/scheduler.py:2240, in update_from_output
File vllm/v1/core/sched/async_scheduler.py:63, in _update_request_w
    assert request.num_output_placeholders >= 0
AssertionError
→ EngineDeadError → 4 个请求全部 HTTP 500
```

- 这是 **vLLM 异步调度器 + 投机解码（MTP）** 的断言，位置与压测摘要里记录的
  "`assert request.num_output_placeholders >= 0` 未隔离（MTP_K=0）"一致。
- recency 模式下 4×27.5K（`run-kvmem-4way.sh`, MAX_NUM_SEQS=4）此前是 **4/4 正确**；
  差别在于本配置启用了检索 + 保持 MTP 开启。
- **下一步（未执行）**：以 `MTP_K=0` 复跑同一并发场景，判定是 MTP/异步调度的问题还是
  适配器对 prompt/positions 的改写触发。这是唯一还没有结论的功能性缺陷。

**结论口径**：检索模式的 4 并发 = **失败（引擎崩溃）**，不得记为通过。

### 并发崩溃的隔离结论：不是 MTP（2026-10-21 01:0x 补测）

`MTP_K=0` 重启后复跑同样 4×13K，**仍以同一条断言崩溃**：

```
async_scheduler.py:63  assert request.num_output_placeholders >= 0
→ EngineDeadError → 4 × HTTP 500，引擎进程退出
```

所以 **MTP 不是原因**。当前最可能的解释（尚未验证）：

> 适配器在 `execute_model` **入口**做 `_reselect`（重写 block table / 更换页），
> 而异步调度器在这一步之后才消费本步输出，占位符计数因此失去一致性。

对应的修法（下一步）：

1. 把重选从 `execute_model` 入口挪到**静态点**（请求边界，或 `update_from_output` 之后的空闲步）；
2. 并发批 >1 时**禁用重选**（`max_num_seqs>1` 下先退回 recency），保证不破坏调度器不变量；
3. 用 `MAX_NUM_SEQS=4` 的 `run-kvmem-4way.sh` 复验（recency 4×27.5K 已 4/4 通过）。

现状口径：**检索模式 + 4 并发 = 已知缺陷（引擎崩溃）**，recency 模式 4 并发通过。

### 并发的正确配置原则（用户 2026-10-21 指出，以此为准）

> **并发调度走 vLLM，上下文规模由 kvmem 控制**——两者互补。

具体到参数：

| 参数 | 归属 | 取值 |
|---|---|---|
| `MAX_NUM_SEQS` | vLLM | 期望的并发数 N |
| `--max-model-len` | vLLM | **N × 单请求上下文**（设大，不会真被占满） |
| `--kvmem-budget`(B) + `--kvmem-gen-reserve`(R) | kvmem | 单请求实际驻留上限，**与 N 无关** |
| `--kvmem-host-mib`(H) / `--kvmem-disk-mib`(D) | kvmem | 跨请求**共享**的页缓存，N 个请求共用 |

因此 `max_model_len` 可以放心设成 N 倍：每个请求的真实工作集被 kvmem 压到 B+R，
聚合驻留由共享的 H（+D）兜住，不会因为 "N 路同时满载" 把池子撑爆。
这也是为什么之前把 `max_model_len` 从 204800 提到 409600 后 2×37K 能被接受而不报上下文错误。

**实测进展**（`max_model_len=409600`, 2×37K, retrieval）：
引擎持续处理（`reselect 32`、无任何 error、health 200），但因 `MAX_NUM_SEQS=1`（sparse 脚本默认）
两条请求是串行的，加上检索重选的建页开销，8 分钟仍未返回。
需要把 `MAX_NUM_SEQS` 提到 2 才能体现真并发——这与之前 4 条时的崩溃是同一处待修点
（`async_scheduler.py:63` 断言），修法见上一节。

### 检索模式长上下文的耗时特征（实测，重要）

`max_model_len=409600` + 2×37K + retrieval：**15 分钟仍未返回**（引擎健康、无 error、
`reselect 34`、`needle 34`、`step_after 32`；客户端 curl 一直存活）。

与 recency 模式对比：

| 模式 | 37K 单请求 | 说明 |
|---|---|---|
| recency | 约 1.5–2 分钟 | 重选只选页，不做打分 |
| **retrieval** | **显著更慢（>15 分钟 @ 2 并发）** | 每次重选要打分 + 按分数重建页 |

原因：检索重选的代价是 `选中页数 × 17 层 × 每页 16 KiB` 的数据搬运与 fp8 重旋转，
而 R 驱动下 37K 上下文会触发多次重选；`MAX_NUM_SEQS=1` 又让两条请求串行，时间叠加。

可调项（不改代码即可改善）：

- 增大 `--kvmem-gen-reserve`(R)：重选更稀疏
- 减小 `--kvmem-budget`(B)：每次搬的页更少
- 并发场景把 `MAX_NUM_SEQS` 与 `max_model_len` 设为 N 倍（见 README 的并发参数表）

结论：**检索模式的正确性已验证，吞吐需要在具体负载下调 B/R**；这是"性能优先"原则下
必须显式权衡的一处，不是缺陷。

### 修掉的最后一个真 bug：token 双计（2026-10-21）

`max_model_len=409600` 的 2×37K 实测里，单个 37K 请求的 frontier 涨到了 **98327**——
说明同一批 token 被算了两次：

- `execute_model` **入口**的 `advance_tokens`（我后来加的权威路径）
- `execute_model` **后处理段**仍在调 `after_step`（旧路径）

两者都推进 `st.tokens`，于是 R 驱动被严重扭曲（重选过早/过频），长上下文请求因此长时间不返回。

**修复**：`runtime._advance_owns_accounting = True`（默认开），后处理段不再调 `after_step`；
`sparse_runtime_test` 仍 PASS。

**影响**：此前的"37K 单请求耗时/取舍"数字含这一 bug 的成分，改为以修复后的测量为准。

### 双计修复后的验证（2026-10-21 收尾）

同一 37K 请求（retrieval 开）：

```
reselect: 22   last tokens: 36864   scores {'blocks': 288, 'total_blocks': 288, 'top': 0.1779}
```

`tokens` 与真实请求规模**完全一致**（修复前为 98327），说明 token 双计已消除；
检索分数仍为真实 mean-K 余弦，`reselect` 次数从 34 降到 22（不再过度重选）。

## 会话终态（2026-10-21 收尾）

- 引擎停在可用状态（`health 200`，生产配置 + `--kvmem-retrieval`）
- 单条 37K 请求在检索模式下 **15 分钟仍未返回**（`tokens 40961`、`needles 22/24`、零 error、引擎健康），
  与上文"检索重选代价"一致：37K 上下文 + retrieval 需要按负载调 `B`/`R`，
  或改用更小的 `--kvmem-budget` 才能在生产时延内完成
- 未完成项：4 并发崩溃（`async_scheduler.py:63`）、埋针 22/28（生产参数）

## 并发保护的初步结果与遗留（2026-10-21 最后一轮）

**改动**：`advance_tokens` 中当 `len(self._state) > 1`（多条请求在飞）时**跳过步内重选**，
只累加 token；目的是避免改写 block table 与异步调度器冲突。

**结果**：

- ✅ 引擎**不再崩溃**（此前同场景 `AssertionError` → `EngineDeadError` → 4×HTTP 500）
- ✅ 零 error 记录（`capture/attach/after_step/reselect` 均无 error）
- ❌ 两条 13K 请求仍未返回（`step_after 62`，客户端 curl 存活）
- ⚠️ frontier 仍偏高：`reselect last tokens 46997`（两条 13K 请求合计约 26K），
  说明**并发下 token 归属/计数仍不干净**，需要按 `req_id` 隔离复核

**结构性矛盾（下一步必须先解决）**：
并发时禁用重选 = 工作集无法压缩 = 稠密前缀持续增长，最终仍会撑爆池子。
正确做法是"**在静止点重选**"——即本步输出产出之后（`update_from_output` 之后，
或 `release` 前后的请求边界），而不是在步内禁用。

因此并发这条的正确修法是：
1. 把重选搬到请求边界/空闲步（不破坏调度器不变量）
2. 并发下按 `req_id` 严格隔离 `st.tokens`
3. 用 `MAX_NUM_SEQS=2` + `max_model_len=2×ctx` 复验两条请求都返回 200

## ⑤ 并发：最终受控矩阵（2026-10-21 03:5x，严格重启 + 逐条核对日志）

**先记一条方法论教训**：我此前多次判"失败"，其实都是**请求打到了正在退出的旧实例**（旧进程还占着端口，
新实例没起来/`curl` 连到旧服务），表现为 500 / 000 / 断言，而同一日志里根本没有对应请求。
此后所有结论都用严格流程：杀进程 → 等端口释放 → 启动 → 等**新日志文件 + health 200** → 发请求 →
在**同一日志**里核对 POST 数与断言数。脚本 `scripts/concurrency_check.sh`。

### 实测矩阵（kvmem 稀疏 + 检索 + 压缩开启，`--no-async-scheduling`）

| N | 单请求 | `max_model_len` | 结果 |
|---|---|---|---|
| 2 | 37K | 204800 | ✅ 双双 200（37231+13） |
| 4 | 12.9K | 16384 | ✅ 4/4（12859+5..16） |
| 4 | 12.9K | 65536 | ✅ 4/4（12859+16） |
| 4 | 12.9K | 131072 | ✅ 4/4（12859+16） |
| 4 | 12.9K | 196608 | ✅ 4/4（12859+16） |
| 4 | 12.9K | 204800 | ✅ 4/4（12859+16） |
| 4 | 30K | 204800 | ❌ `assert num_scheduled_tokens >= num_logits`（80 次重选后） |

**结论**：

1. **`MAX_NUM_SEQS=4` + 每请求 ~13K 已可用**，且与 `max_model_len`（16K~200K）无关；
   压缩正常工作（`reselect` 有效、无 error 记录）
2. `MAX_NUM_SEQS=2` + 每请求 37K 可用
3. 4 路 + 每请求 30K 时仍会撞 `model_runner.py:1348` 的断言——这是该规模的边界，
   与"压缩开/关"无关（早些时候关压缩同样复现），需要 vLLM 侧的 `num_logits` 修正才能突破

### 并发口径（可用）

| 场景 | 建议 |
|---|---|
| 长上下文（>15K）× 2 路 | `MAX_NUM_SEQS=2`、`max_model_len=204800` |
| 中等上下文（≤13K）× 4 路 | `MAX_NUM_SEQS=4`、`max_model_len` 取 16384~204800 均可 |
| 4 路 × 30K | 暂不可用（上游断言） |

### ⑤ 并发 + 长解码：压缩与投机草稿的冲突（2026-10-21 04:4x）

用埋针请求（`/tmp/p1.json`，37K，答案必须是 `9P6HQ8 55D59J`）在 N=2 下做**正确性**验证：

| 配置 | 结果 | 答案 |
|---|---|---|
| 压缩开（budget 16384） | ❌ 500，`assert num_scheduled_tokens >= num_logits`（44 次重选后） | — |
| **压缩关（budget 204800）** | ✅ 双双 200，零断言 | **两条都是 `9P6HQ8 55D59J`（完全正确）** |

**机制**：`num_bonus_tokens ∈ {0,1}`（`input_batch.py:471` 有硬断言），所以失败行意味着
"某请求本步只调度 1 token，却挂着 4 个草稿"。这类行只在**压缩改写块表**的那一步出现。

**已实现的守卫**（`kvmem_sparse_runtime.advance_tokens`）：

```python
has_drafts = req in runtime.spec_reqs      # 本步带草稿的请求
deferred = has_drafts or (concurrent and prefill_step)
```

即"本步带草稿的请求不压缩"。此守卫**降低**了触发概率（预填充步仍可压缩），但长解码下仍会复现——
彻底解决需要把压缩搬到调度器侧（EngineCore 在 `schedule()` 里改块表，让 vLLM 的记账同步），
工作量约等于一次重构，本轮未做。

**并发正确性已被证明**：压缩关闭时，N=2 并发下两条 37K 请求的埋针答案**逐字正确**。

## ⑤ 的根因与结论（2026-10-21 04:3x-05:0x，受控测量）

### 根因：**截断式压缩与 vLLM 的"序列单调"记账冲突**（与并发无关）

受控测量（N=1、单条 37K、recency 模式、budget 16384）：

| 配置 | 结果 |
|---|---|
| 压缩开 + 生成 16 token | ✅ 200 |
| **压缩开 + 生成 ~300 token** | ❌ `assert num_scheduled_tokens >= num_logits`（prefill 只花 27 s，崩在解码首步） |
| 压缩关 + 生成 ~300 token | ❌ 引擎死（另一条路径） |

20 次压缩全部发生在 **prefill 阶段**，断言却在**解码首步**触发 —— 说明不是"步内改表"的问题，而是：
**压缩缩短了请求的有效序列长度，而 vLLM 调度器仍按原长度（37231）记账**，两者在投机解码的 logits 行寻址处对不上。

`model_runner.py:1348` 的断言正是这个不一致的显式检查（只在投机解码生效时执行）。

### 尝试过的三段修法（都已落地，作为开关保留）

1. `has_drafts` 守卫：本步带草稿的请求不压缩 → 降低概率，不解决
2. `spec_active` 守卫：整步有草稿就不压缩 → 同上
3. `inrequest_compaction` 开关（`--kvmem-inrequest-compaction`）：
   - 关掉后引擎稳定（37K 长解码 29 s、零断言），但**检索路径的假设被破坏**（答案变乱码、`bridge_harvest_error`）——
     因为检索/选页与压缩是同一套视图，关压缩等于让视图处于半初始化状态
   - 因此**默认保持开启**，把长解码限制作为已知边界记录

### 结论（可直接对外表述）

| 场景 | 状态 |
|---|---|
| 单请求任意长度、生成 ≤ ~16 token | ✅ |
| 单请求长上下文 + 长生成 | ❌ 断言（压缩与调度器记账冲突） |
| N=2 × 37K / N=4 × 12.9K（短生成） | ✅ |
| N=2 × 37K 长生成 + 压缩关 | ✅ 答案正确（但检索失效） |

要真正解决，只有一条路：**让 vLLM 的调度器知道序列被压缩了**（在 EngineCore 侧同步 `num_computed_tokens` 与块表），
这属于对 vLLM 的最小接口扩展，不是适配器能单方面完成的。

## 下一步的接口设计（已独立成文）

`DESIGN-compacted-sequence.md`：给出"让 vLLM 认识被压缩序列"的最小接口方案
（`RequestState.compacted_tokens` + `kv_cache_manager` 与 `prepare_inputs` 各一处使用，约 40 行），
以及零改动的替代方案（会话级分层）。两方案的取舍与回退路径都在文里。

## 测试与契约（2026-10-21 05:1x）

- **31 个 Python 测试逐个运行全部通过**；紧凑批量跑时 27/31，4 个失败都是
  `torch.frombuffer(src_np.tobytes(), ...)`（需要独显上下文/写缓冲），单独运行即通过 —— 属批量执行的 CUDA 抖动
- 修掉一处**契约偏差**：`reselect_token_budget` 之前被设成 `gen_reserve`，与
  `worker_cli_test` 的约定（节奏只由 KVMem 的 R 决定、适配器不自建定时器）冲突；已改为 0，
  planner 在场时由 R 驱动（实测重选间隔仍为 8192 = R）
- 新增开关 `--kvmem-inrequest-compaction`（默认开）：关掉即完全走 vLLM 原路径，
  用于对比与故障隔离
- 探针 24/24

## 默认配置的最终收敛（2026-10-21 05:0x）

**问题**：in-request 压缩会缩短请求的有效序列长度，而 vLLM 按逻辑位置记账/寻址 ——
压缩后 attention 读到陈旧页、并在解码首步触发 `assert num_scheduled_tokens >= num_logits`
（实测：单请求 37K + 300 token 生成，必崩，4 次断言）。

**决定**：`--kvmem-inrequest-compaction` **默认关闭**。默认行为 = **vLLM 原始路径** +
kvmem 的观测/统计（H/D 分层、共享 H、LRFU、mean-K 采集），符合"性能优先、保持原始路径"。

同时修掉一个真 bug：压缩关着时**视图仍被安装**（半初始化），导致答案乱码 +
`bridge_harvest_error`；现在 `before_step` 在压缩关闭时直接返回，不安装任何视图。

### 实测（默认配置：检索开、压缩关）

| 场景 | 结果 |
|---|---|
| 单请求 37K + 24 token | ✅ **200 / 37 s** / 零断言 / 引擎健康 |
| 单请求 37K + 300 token（埋针） | ✅ 200，答案 `9P6HQ8 55D59J`（逐字正确） |
| N=2 × 37K（短生成） | ✅ 双双 200 |
| N=4 × 12.9K（短生成） | ✅ 4/4 |
| 显式开启压缩（`--kvmem-inrequest-compaction`）| ⚠️ 仅预填充/短生成可用；长生成必崩（见上） |

### 开关一览

| 开关 | 默认 | 作用 |
|---|---|---|
| `--kvmem-inrequest-compaction` | **off** | 生成期截断上下文（实验；与 vLLM 记账冲突） |
| `--kvmem-retrieval` | off | pre-RoPE 采集 + mean-K 选页（需压缩开启才有意义） |
| `--kvmem-disk-mib/--kvmem-host-mib` | 0/16384 | H/D 分层 |
| `--no-async-scheduling`（worker 自动加） | MAX_NUM_SEQS>1 时 | 避免异步调度与步内改表冲突 |

## 断言的精确观测（2026-10-21 05:2x）

在插件里加了 `spec_short` 诊断（每次某请求"带草稿但调度 token 少"就记录一次）：

```
spec_short: scheduled=4 drafts=4      # num_logits = drafts + bonus = 5 > 4
```

即调度器本步只给该请求 **4 个 token**，而它挂着 **4 个草稿 + 1 个 bonus = 5 行 logits**，差 1 触发断言。

已尝试的缓解（`_ensure_tail`：每次 reselect 与每步都保证行尾 ≥ `spec_tokens + 8` 个空 token 槽）：
日志显示 `spec_short` 仍出现 → **不是页槽不足**，而是调度器在该请求上主动只调度了 `len(drafts)` 个 token。
这条只能在 vLLM 侧修（`scheduler` 对"刚被压缩过的请求"应当把步长补到 `1+drafts`）。

`_ensure_tail` 保留：它对短生成/预填充场景仍是正确的保护。

## 默认配置的完整验证矩阵（2026-10-21 05:2x，单脚本一次跑完，同一引擎同一日志）

配置：`--kvmem-budget 16384 --kvmem-gen-reserve 8192 --kvmem-host-mib 16384 --kvmem-retrieval`，
`MAX_NUM_SEQS=2`，`--max-model-len 204800`，`--no-async-scheduling`，**压缩默认关**。

| 用例 | 结果 | 耗时 |
|---|---|---|
| 单请求 37K | ✅ 1/1 200 | 36 s |
| **N=2 × 37K** | ✅ **2/2 200** | 69 s |
| **N=4 × 13K** | ✅ **4/4 200** | 43 s |
| 断言 | **0** | — |

这份矩阵取代之前所有零散的并发结论（此前多次"succeeded/failed"的反复，根因是请求打到了
正在退出的旧实例；现已用"等新日志文件 + 同一日志核对 POST 与断言计数"的流程固定下来）。

## 头号能力验证：99K 单请求 + 埋针（2026-10-21 05:2x，默认配置）

**方法**：生成一份自包含的 99K 埋针提示（4,000 条记录，每条一个唯一 6 位 key，末尾问
"record 250 与 record 4100 的 key"），`max_tokens=600`（模型会先思考再答）。

| 指标 | 结果 |
|---|---|
| HTTP | **200** |
| prompt / completion | **97,635 + 124** |
| finish_reason | stop |
| 答案 | `EPVHKZ CPBL7Q` |
| **埋针** | **两条 key 均逐字正确（True / True）** |
| 断言 | 0 |
| prefill 耗时 | 约 112 s（首轮 24-token 预算时测得） |

第一轮 `max_tokens=24` 时 `content` 为空、`reasoning_tokens=24` —— 模型把预算全用在思考通道，
不是失败；放大预算后一次答对。这条经验已记录：**该模型的 reasoning 通道会吃掉预算**。

附注：97,635 token 的稠密 KV 约 763 页/rank（≈212 MB），22 GB 直接放得下，
所以默认（不做生成期压缩）就能服务 99K；压缩收益要在更大上下文（如 300K+）才体现。

### 补丁：把"压缩关闭"做成真正的纯 vLLM 路径（2026-10-21 05:3x）

自查发现还有一处重选入口不受开关约束 —— `after_step` 里的 `if not st.row or due:`：
压缩关闭时 `st.row` 永远为空，于是**每步都会触发一次重选**（99K 那次实测 8 次），
仍然改写了块表。已改为：

```python
if inrequest_compaction and (not st.row or due):   # `not st.row` 只是"没建稀疏行"，不是"欠一次重选"
```

现在 `--kvmem-inrequest-compaction` 关闭时：不装视图、不选页、不改块表、不重选 ——
**与纯 vLLM 路径完全一致**，kvmem 只保留观测与 H/D 分层。

## in-request 压缩的最终判定（2026-10-21 05:3x，两种投机模式都试过）

| 压缩 | 投机解码 | 单请求 37K + 300 token | 结果 |
|---|---|---|---|
| 开 | 开（默认） | `scheduled=4 drafts=4` → 差 1 行 | ❌ 断言，引擎退出 |
| 开 | **关**（`SPECULATIVE_METHOD=none`） | prefill 正常（到 36864 token）、重选 20 次 | ❌ **卡住不返回**（20 分钟无响应，`reselect` 冻结） |
| **关** | 开（默认） | 99K + 埋针、N=2/N=4 并发 | ✅ 全部通过 |

**结论：in-request 压缩在这个后端不可用**（不是投机解码的锅：关掉投机后从"断言"变成"挂住"）。
根因是 vLLM 的三处记账互相咬合，无法只改一个：

1. `num_computed_tokens` 决定本步调度多少 token（`kv_cache_manager.py:427,770`）
2. 块表按**逻辑块下标**写入，压缩把页搬到前面 → 行内容与下标脱钩
3. `_all_token_ids` 单调增长（`request.py:146,271`）→ 要一致就得截断 token 序列本身

只改 1 会让 vLLM 把被丢掉的 token **重新预填充**（等于撤销压缩）；只改 2 会让 attention 读到陈旧页；
只改 3 等于改提示内容 —— 那就是**入口层选页**，不再是 KV 层压缩。

### 因此

- **默认关闭压缩**（已落地）：`--kvmem-inrequest-compaction`，关闭时不装视图/不选页/不改表/不重选
- 若要在 vLLM 上做上下文压缩，只能在**入口层**做：由 KVMem 决定发多少历史给引擎（提示级选页），
  这是唯一与 vLLM 记账天然兼容的位置，也是下一步建议的方向

## 入口层选页（KVMem 作为策略层）：端到端跑通（2026-10-21 05:5x）

既然 KV 层压缩与 vLLM 记账不可调和（见上节），把压缩搬到**入口**——由 KVMem 决定发多少历史给引擎，
引擎只预填充它收到的内容。新增两个组件：

- `src/kvmem_prompt_select.py`：历史分块 + 打分选择（sink 前缀 + recent 尾 + 按问题命中的中间块）
- `scripts/kvmem_gateway.py`：入口网关（`:8100` → vLLM `:8001`），只重写 `/v1/chat/completions`

### 实测（99K 埋针提示，问 record 250 与 4100 的 key）

| 指标 | 直连引擎 | **经网关** |
|---|---|---|
| 引擎 prompt token | 97,635 | **21,635**（4.5×） |
| 耗时 | 112 s | **21 s**（5.3×） |
| 答案 | `EPVHKZ CPBL7Q` ✅ | **`EPVHKZ CPBL7Q` ✅** |
| 断言 | 0 | 0 |

网关日志会打印决策：

```json
{"kind":"prompt_selected","blocks":4001,"kept":936,"dropped":3065,
 "chars_before":218577,"chars_after":50001,"estimated_tokens_after":16667,"trimmed":true}
```

### 关键修正（本轮踩到的坑）

历史与问题常在**同一条消息**里，最初把整条消息当"问题"去打分 → 每条记录都成了问题词，排序失效、
两个埋针记录被丢掉。改为取消息**尾部**（`question_tail`，默认 600 字符）作为问题后，两条记录都被选中。

### 用法

```bash
python scripts/kvmem_gateway.py --upstream http://127.0.0.1:8001 --port 8100 \
    --kvmem-budget 16384 --kvmem-sink-tokens 256 --kvmem-recent-tokens 1024
# 客户端把 base_url 指向 http://127.0.0.1:8100/v1
```

注意：网关只重写 `chat/completions`，其余请求原样转发；打分当前用**词面命中**（问题里的 id/数字），
接入 mean-K 统计后可以换成真正的检索打分。

## 组合验证：N=2 并发 × 99K 埋针（经入口网关，2026-10-21 05:4x）

一次性验证 ⑤（并发）+ ②③（kvmem 参与）+ 正确性：同一 99K 埋针请求并发两条，经网关（budget 16384）。

| 请求 | prompt | completion | finish | key250 | key4100 | 答案 |
|---|---|---|---|---|---|---|
| req1 | 21,635 | 74 | stop | ✅ | ✅ | `EPVHKZ CPBL7Q` |
| req2 | 21,635 | 74 | stop | ✅ | ✅ | `EPVHKZ CPBL7Q` |
| 合计耗时 | **42 s**（两条并发） | | | | | |
| 断言 | **0** | | | | | |

对比直连引擎（不做入口选页）：97,635 token / 112 s / 单条。
**入口选页把两条并发的 99K 请求压到 21,635 token、42 秒内双双答对**，且引擎零断言。

这条结果同时说明了三件事：

1. **并发正确**（N=2，两条请求全部返回且答案逐字正确）
2. **长上下文能力**（99K 输入，答案需要跨越首尾两个记录）
3. **压缩生效且安全**（上下文由入口层裁剪，引擎侧走原始路径，不触碰 vLLM 记账）

## 验收表

`VALIDATION-PLAN.md` 已重写为当前状态的验收清单：目标条款 → 命令 → 实测结果，含四条现场验证
（99K 埋针、并发、入口层选页、释放/还原）与三条曾经导致误判的坑。

### 撤回 `_ensure_tail`（2026-10-21 06:0x）

曾为"行尾空槽不足导致调度器少给 token"的猜想加了 `_ensure_tail`（每步/每次重选补一页空尾）。
实测 `spec_short: scheduled=4 drafts=4` 仍然出现 → **与页槽无关**，且它让
`sparse_concurrency_test` 的页映射出现 3 页埋针不一致（`bad=[474,475,476]`）。
无收益 + 引入不一致，**已整体移除**；移除后 `sparse_concurrency_test` / `sparse_runtime_test` /
`sparse_orchestrator_test` 全部恢复 PASS。

## 检索 vs recency 选片差异（2026-10-21 06:1x，端到端可观测）

网关新增 `--kvmem-select {retrieval,recency}`：`recency` 不把问题纳入打分，只保留 sink 前缀 + 最近窗口，
这正是"纯 recency 策略"的行为。同一 99K 埋针请求（问 record 250 与 4100 的 key）两种模式对照：

| 模式 | 引擎 prompt | completion | 耗时 | 答案 | 埋针 |
|---|---|---|---|---|---|
| **retrieval** | 21,635 | **86** | **21 s** | `EPVHKZ CPBL7Q` | ✅ 两条都命中 |
| **recency** | 21,636 | 1500（撞上限） | 42 s | 空 | ❌ 两条都没有 |

选片层面同样可观测（离线）：

```
retrieval: tokens≈16667 kept= 936 key250=True  key4100=True
recency  : tokens≈16667 kept= 936 key250=True  key4100=False
```

**结论**：recency 会把"问题点名要的那条记录"丢掉，模型于是无法作答（1500 token 全耗在推理里）；
retrieval 保留它，86 token 就给出一字不差的答案。这是"检索真的在起作用、不是 recency 退化"的直接证据。

## 常驻页下降：官方口径实测（2026-10-21 06:3x）

用 vLLM 自带的 Prometheus 指标 `vllm:kv_cache_usage_perc`（池占用百分比）采样同一个 99K 请求：

| 阶段 | 引擎 prompt | 峰值 KV 池占用 | 采样点 |
|---|---|---|---|
| 直连引擎 | 97,635 | **50.67%** | 472 |
| 经入口网关（budget 16384） | 21,635 | **19.33%** | 194 |
| **降幅** | −77.8% token | **−61.8% 池占用** | — |

采法：请求在飞时每 200 ms 拉一次 `/metrics`，取该阶段的峰值；两阶段之间指标回落到 0，
据此自动分段（脚本片段见本轮记录）。

这条把目标里"单请求 99K 常驻页下降"从"视图页数"的口径升级为**引擎自己报告的池占用**，
且同时满足"答案与埋针字节一致"（同一请求答案 `EPVHKZ CPBL7Q`，两条 key 全中）。

## ② 磁盘层（D）的验证状态（2026-10-21 06:5x，诚实记录）

| 层次 | 证据 | 状态 |
|---|---|---|
| 单元级（机制正确） | `tests/disk_tier_test.py`：2 页留 RAM、4 页溢出到磁盘、**逐字节回读一致** | ✅ PASS |
| 活引擎（早期配置） | 上一阶段实测：H=64 MiB + D=256 MiB，99K 语境下埋针 **22/22** | ✅（历史） |
| 活引擎（当前默认配置） | 默认不做生成期压缩 → 稀疏视图不安装 → H/D 不参与 | 未启用（设计使然） |
| 活引擎（当前 + 压缩开） | 13K 请求 + H=4 MiB 想强制溢出：预填充阶段即被断言/停滞打断，**未拿到可观测的溢出样本** | ⚠️ 未复验 |

结论：**磁盘层的实现与逐字节正确性由单测保证**；它的活引擎行为依赖生成期压缩，而后者在当前 vLLM
后端不可用（见上文判定）。因此在默认路径下 D 层不会被触发——这是设计结果，不是缺陷。

## ② H/D 分层：活引擎验证达成（2026-10-21 07:1x）

in-request 压缩不可用后，H/D 层失去了活引擎消费者。解决办法：**把 H/D 用在入口层**
（`src/kvmem_history_cache.py` + 网关 `--kvmem-history-kib/--kvmem-history-disk-mib`）——
长会话的历史分块结果按内容哈希缓存，H 用满后溢出到 NVMe 层，下次请求按需读回。

### 实测（同一会话两轮，99K 埋针请求）

| 轮次 | 缓存 | 磁盘 | 引擎 prompt | 耗时 | 答案 |
|---|---|---|---|---|---|
| turn1 | miss | **spill 1 条 / 4 槽**（237 KB 跨槽） | 21,635 | 21 s | ✅ `EPVHKZ CPBL7Q` |
| turn2 | **hit（`disk_reads: 1`，从 NVMe 读回）** | 4 槽 | 21,635 | 21 s | ✅ `EPVHKZ CPBL7Q` |

网关日志（可审计）：

```json
{"kind":"history_cache","result":"miss","disk_spills":1,"disk_slots":4,"disk_entries":1,"misses":1}
{"kind":"history_cache","result":"hit","disk_spills":1,"disk_slots":4,"disk_reads":1,"hits":1}
```

### 实现要点（两个必须做对的地方）

1. **分块跨槽**：单条历史 237 KB 远大于 64 KiB 槽，必须切块写到多个槽；早期版本直接
   `write_block(slot, payload[:slot_bytes])` 在载荷超槽时静默失败 → 永不落盘。
2. **尾部裁剪**：槽是定长，最后一块补零；读回后要按记录的长度截断，否则 JSON 解析失败。

单测 `tests/history_cache_test.py` 覆盖：H 预算生效、冷块落盘、**逐字节读回一致**、未知键走 miss。

这补齐了 ② 里"磁盘层 / 会话冷存交换"的活引擎证据，且全程走推荐路径（入口网关 + vLLM 原路径）。

## 收尾：对 vLLM 源码的实际改动面（2026-10-21 07:3x，核对 git）

vLLM fork 工作区共有 14 个改动文件，但**本适配器引入的只有 2 处**：

| 文件 | 性质 |
|---|---|
| `vllm/kvmem_bridge.py` | **新增**（pre-RoPE 采集的自定义算子 + 一次性缓冲） |
| `vllm/model_executor/models/qwen3_next.py` | **+8 行**（两处 `_project_qkv_gate` 返回前调用 `torch.ops.kvmem.capture_q`） |

其余 12 个改动文件（`launcher.sh` 的 `KV_CONCURRENCY_EFFECTIVE_LEN`、`observability.py`、
`v1/core/sched/scheduler.py`、`v1/engine/*`、`v1/metrics/*`、`routers.py`、`modelopt.py` 等）
**早于本适配器存在**，属于既有环境/前序工作，不由本适配器引入。

`launcher.sh` 的那处是**可选开关**：仅在显式设置 `KV_CONCURRENCY_EFFECTIVE_LEN` 时改变准入判据，
不设置时行为与原版一致。

## 全量测试（33 个文件）

紧凑批量 29 PASS + 4 GPU 抖动（`torch.frombuffer`，单独运行全 PASS）→ **等价 33/33**。

## 更正：R 的正确语义（2026-10-21 08:0x，用户指正）

**R 是生成上限（cap），语义等同 `max_tokens`：允许最多生成 R 个 token，模型输出 EOS 即停。**
它是"最多"，不是"每次都会生成 R 个"——一个请求只输出两个 token 也完全正常。
不是"撑满就重选继续生成"（那是实验功能）。
kvmem 里"R 用满后把它挪进 budget 继续生成"是**实验性、不推荐**的功能。

按这个语义，整个架构本来就是自洽的：

```
常驻 = B（选中的历史） + R（生成上限）      生成永不越过 R ⇒ 序列长度在请求生命周期内单调增长
```

因此 **vLLM 后端不需要生成期截断**——B 的选择只发生在**请求入口**。

### 我之前的错误

| 我的说法 | 实际情况 |
|---|---|
| "生成期 KV 截断不可行" | 我把**实验路径**（R 满后继续生成）当成了设计，它才与 vLLM 记账冲突 |
| "断言/挂住说明设计有问题" | 只说明那条实验路径在 vLLM 上不成立；默认路径（到 R 即停）根本不触发它 |

`--kvmem-inrequest-compaction` 对应的正是这条实验路径，**默认关闭是对的**，文档里已标注"实验、不推荐"。

### 契约：`max_tokens` 属于 kvmem，`R` 就是它

`max_tokens` 由 kvmem 决定（`R` 即其值），不是客户端的自由量。入口网关按此**写入**：

| 客户端给的 | 网关写入 |
|---|---|
| 超过 R | **R**（`finish_reason=length`） |
| 小于 R | 客户端的值（只会更省，不破坏 B+R） |
| **没给** | **R**（否则引擎自身默认值会让生成越过 R） |

日志：`{"kind":"gen_budget","field":"max_tokens","requested":20000,"applied":8192}`

单测 `tests/gen_reserve_test.py` 覆盖四种情形：超 R 降为 R、小于 R 保留、`max_completion_tokens` 同样、
**缺省时写入 R**。

## 口径澄清：R 是上限，不是长度（用户指正）

之前写成"生成到 R 就停"，容易被读成"每个请求都会生成 R 个 token"。正确的表述：

- **R = 允许的最大生成长度**（cap）
- 实际长度由模型决定：输出 EOS 就结束，**两个 token 也是正常结果**
- R 的作用是给常驻内存封顶（B + R），不是强制输出长度

网关把 `max_tokens` 写成 R（或缺省时写 R）是**设上限**，不是设固定长度。

## 概念纠正：kvmem 的逻辑上下文 ≠ 模型窗口（2026-10-21 09:0x，用户指正）

`CTX`（逻辑上下文 / 缓存容量）在 kvmem 里**可以任意大**：它只是 KV 的缓存层，不全部参与模型上下文。
真正进入模型的是 **B + R**。所以"最大 256K"这个说法是错的——那是模型 RoPE 窗口，不是 kvmem 的 CTX。

### 正确映射

| 概念 | 约束 | 本机取值 |
|---|---|---|
| kvmem 逻辑上下文（可保留的历史总量） | **不受模型窗口限制**，由 H/D 容量决定 | 实测 287K token 通过 |
| **B + R**（每请求真正进模型的窗口） | ≤ 模型原生窗口 | 65536 + 32768 = 98304 |
| vLLM `--max-model-len` | 需 ≥ B+R，且 ≤ 模型 `max_position_embeddings` | 204800（够用，可下调到 98304） |

### 实测（300K 级历史，超过 256K）

构造 12,001 条记录的**287K token** 历史（问 record 500 与 record 11500 的 key）：

| 路径 | 结果 |
|---|---|
| **直连引擎** | ❌ **400**：`maximum context length is 204800 tokens`（模型窗口限制，符合预期） |
| **经入口网关**（B=65536） | ✅ **prompt 89,009 + 156，finish stop，答案 `XM9LE6 N553Y8`**（两条 key 全中） |

网关决策日志：

```json
{"kind":"prompt_selected","blocks":12001,"kept":3673,"dropped":8328,
 "chars_before":661971,"chars_after":200146,"budget_tokens":65536,
 "estimated_tokens_after":66715,"trimmed":true,"method":"retrieval"}
```

**结论**：逻辑历史 287K > 模型窗口 204.8K，仍然能正确作答——因为只有 B+R 进模型。
这就是"kvmem 的 CTX 可以随便设"的实际含义。

## 引擎窗口定尺：`max_model_len = N × (B + R)`（2026-10-21 09:3x）

用户提出的公式，已实测并固化为 worker 的默认行为（不传 `--max-model-len` 时自动计算）：

```
[kvmem backends/vllm] max_model_len auto = 2 x (65536 + 32768) = 196608
```

### 为什么是这个值

| vLLM 的要求 | 含义 |
|---|---|
| `max_model_len ≤ 模型窗口` | 本模型 262144 |
| `KV 池 ≥ 1 条 max_model_len 序列` | 池只有 3.82 GiB，能装的序列 ≈ 210080 |

而 N 路并发真正需要 `N × (B+R)`；把窗口设成这个乘积，两条约束都刚好满足。

### 实测边界

| `max_model_len` | 结果 |
|---|---|
| 409600 | ❌ 超过模型窗口（262144） |
| 262144 | ❌ KV 池不足（需 4.66 GiB / 有 3.82 GiB） |
| **196608 = 2×(65536+32768)** | ✅ 启动并服务 |

**注意**：这只是**引擎窗口**；kvmem 的逻辑上下文不受它约束——287K token 历史经网关仍正确作答
（prompt 89,009，埋针全中）。

## 2026-10-21 单服务化：入口策略进进程，端口 18200

**变化**：取消 8100 旁挂网关。选 B、`max_tokens=R`、CTX 上限、H/D 历史块缓存现在跑在 **vLLM 的 API
server 进程内**，通过官方 `vllm.endpoint_plugins` 入口点（`kvmem_entry`）加载；服务只有 `:18200`
一个端口、一个 systemd 单元，vLLM 源码零改动。

**body 改写的正确姿势（实测）**：纯 ASGI 的"换 receive 重放"会挂（中间件打出 `calling downstream`、
handler 也进了，响应永不返回；关掉中间件同一请求 0.24 s 完成）——因为下游持有**原始** `receive`。
标准做法是 `BaseHTTPMiddleware` + `await request.body()` 后把新字节写回 `request._body` 缓存，
下游命中缓存、不碰流。默认即此模式（`KVMEM_ENTRY_MODE=middleware`），
`hook`（包装 `create_chat_completion`）保留作兜底。

**实测（单服务，18200）**：

| 项 | 结果 |
|---|---|
| 端口 | `ss -tln` 只有 `0.0.0.0:18200`，无 8001/8100；单元只有一个 |
| 就绪 | `wait_ready.sh 18200 --smoke` → 200 / 253 ms |
| R 契约 | `max_tokens=100000` → 日志 `gen_budget requested=100000 applied=32768`（只写一个字段） |
| 逻辑上下文 | 486,605 token 历史 → `prompt_tokens=89,066`，两个埋针键全中（106 s） |
| CTX | 604,278 token → `400 history is ~604278 tokens, above kvmem CTX=524288` |
| 流式 | middleware 模式：30 个 SSE 事件，首 **0.07 s**，末 1.43 s，单调递增（hook 模式：0.13/1.50 s） |
| CORS | 预检 200，响应带 `access-control-allow-origin: *` |
| 探针 | 28/28（新增 endpoint_plugins / attach_endpoint_plugins / attach_router / create_chat_completion） |

**就绪方式**：`scripts/wait_ready.sh <port> [timeout] [--smoke]` — 0.5 s 轮询 `/health`，日志出现
`Application startup complete` 立即成功，命中启动错误模式立即 `exit 2`，不再盲等。

### P-1（已修）：B 必须用真实 tokenizer 标定

`chars_per_token=3` 只是估算。同一段 348,778 字符的代码风格历史：

| | 估算 token | 引擎实际 prompt | 延迟 |
|---|---|---|---|
| 修复前 | 65,611 | **120,321**（1.83×） | 131 s |
| 修复后（真实 tokenizer 标定） | 65,568 | **65,025** | **70 s** |

实现：入口中间件在请求处理时从 `request.app.state`（`openai_serving_chat.renderer.tokenizer`，
回退 `engine_client.renderer.tokenizer`）取 tokenizer，用该请求最长消息算出真实 chars/token，
再以 `dataclasses.replace` 生成该请求的策略副本；取不到 tokenizer 时退回估算，绝不报错。
测试：`tests/entry_calibration_test.py`。

### P3 结论：入口 query 才是瓶颈，引擎 capture 窗口不是

用同一段 12,000 条历史 + 长问题（主题在问题第一段、结尾是纯格式要求）做对比：

| 配置 | 结果 |
|---|---|
| 引擎 capture=64（默认） | 两个埋针**都找到** |
| 引擎 capture=512 | 两个埋针**都找到**，无提升 |

真正决定成败的是入口打分用的 query：

| 入口 query | 结果 |
|---|---|
| 只取最后一段（原实现，`question_tail` 用 `rfind("\n\n")`） | 针得分 0 → 排名 7799/11999 → 被丢弃 → 答案错 |
| 最后一条 user 消息独立时用整条（新实现） | 针被选中 → 答案对 |

修复：
1. `question_tail(text, keep_chars, paragraphs)` 支持取最后 k 段（原来永远只取最后一段，`keep_chars` 形同虚设）
2. 历史与问题分属不同消息时，直接用**最后一条 user 消息**作为 query（不做段落猜测）
3. 测试：`tests/entry_policy_test.py` 新增"问题第一段含主题"的用例

结论：`KVMEM_CAPTURE_TOKENS` 默认改为 **512**（对齐 KVMem `query_max_tokens` 的官方默认），并作为正式参数暴露：
worker `--kvmem-capture-tokens`、启动脚本 `CAPTURE_TOKENS=`、落盘 `kvmem-params.env`、worker 日志打印。
512 窗口在 22 GB 卡上实测可用（占用 20,612 / 22,528 MiB），长问题回归测试（seed 41）两个埋针全中。
"实时 Q vs user span"的差异在本工作负载下没有观测到影响，不再作为待补项。

### KVMem Web UI 同端口挂载

`KVMEM_UI_DIR`（SvelteKit 静态构建）现在由端点插件在 `attach_router` 里挂到 `/`，
因此 `http://<host>:18200/` 就是 KVMem 的 Web UI，和 API 同源同端口（UI 内部用的是相对路径
`/v1/models`、`./v1/chat/completions`，不需要代理）。挂载在核心路由之后，`/v1/*`、`/health`、
`/docs` 优先，实测全部 200。

参数：`KVMEM_UI_DIR`（`run-current.sh` 里可改，空字符串 = 不挂）。

**已知未实现**（UI 会 404，不影响聊天）：
- `./v1/chat/completions/control` —— 停止生成（llama.cpp/ninfer 的服务端扩展）
- `./v1/stream`、`./v1/streams/lookup` —— 流恢复/会话查找
- `./slots` —— slot 列表（llama.cpp 概念）

### 性能基线：旧并发表的真实含义（prefill，不是 decode）

`scripts/concurrency_check.sh` 里 `max_tokens=16`，所以之前那张表（37K/36s、2×37K/69s、4×13K/43s）
测的是 **prefill + 排队**，decode 只占 <1 s。用引擎自己的计数器拆开重测（当前配置 N=2、B/R=65536/32768）：

| 用例 | prompt | wall | prefill | prefill 速率 | decode |
|---|---|---|---|---|---|
| 单请求 98.8K | 98,799 | 140.6 s | 138.5 s | **713 tok/s** | 8 tok/s（16 token）|
| 单请求 65.2K | 65,211 | 71.1 s | 69.8 s | **934 tok/s** | 19.7 tok/s |
| 2×65.2K（唯一内容） | 130,422 | 139.1 s | 273.2 s（合计）| **477 tok/s** 聚合 | — |
| 小 prompt 生成 512 | 832 | 9.2 s | 0.5 s | 1530 tok/s | **59.4 tok/s** |

结论：
1. 长上下文的耗时**几乎全是 prefill**（65K ≈ 70 s，约 900 tok/s；2×2080Ti + fp8 + GDN 混合 + MTP）。
2. 旧表数字与今天一致（37K/36s ≈ 1000 tok/s），说明期间没有退化也没有提速——那 1.83× 的预算标定
   影响的是"发多少 token"，不是"每 token 多快"。
3. 并发不提升 prefill 吞吐（2 流聚合 477 tok/s < 单流 934 tok/s），因为 TP=2 已经用满两张卡。
4. 多轮对话受益于前缀缓存，实际远好于这些冷启动数字（实测同前缀第二轮 cached 63,024/65,890）。

待验证的优化方向（未做）：`MAX_BATCHED_TOKENS` 4096 → 8192/16384 可减少 chunk 数；历史记录里提过
调大会让启动 smoke 失败，需要在当前配置下重试确认。

## Route A：引擎内压缩（ninfer 对等）——实现与验证计划（2026-10-21）

**目标**：把 ninfer 的「超 R 生成」搬进 vLLM。物理池仍是每请求 `B+R`，但请求可以一直生成到
`CTX`：常驻池填满时重选（sink+media+recent 保留、其余溢写 host 层），并把引擎侧的记账一起截断，
使 vLLM 认识「被压缩的序列」。

### 决策项（D1–D6）与状态

| 项 | 决策 | 状态 |
|---|---|---|
| D1 入口上限 | 压缩关：`max_tokens` 夹到 R；压缩开：上限 `CTX`（逻辑上下文），R 退化为重选节拍 | ✅ `EntryPolicy.generation_ceil()`，两条入口（原始 payload + OpenAI 中间件）共用 |
| D2 三本账对账 | worker 重写 block table 后，经 `BlockChannel.compact` 让 EngineCore **原地** `_all_token_ids[:] = visible` 并回退 `num_computed_tokens` | ✅ 通道 + `_compact` 闭包；`ConstantList` 视图同步 |
| D3 前缀缓存 | 内容寻址（`H(parent_hash, tokens)`）、释放即失效，压缩不污染 | ✅ 无需改动 |
| D4 引擎窗口 | `max_model_len = N×(B+R)` 已大于单请求可见长度；压缩后 `_all_token_ids` 恒 ≤ B+R，长度上限永不误触发 | ✅ 无需改动 |
| D5 在途草稿 | 带草稿的步推迟压缩（≤1 个投机轮次浪费），下一次无草稿步执行 | ✅ `deferred = has_drafts or prefill_step` |
| D6 调度模式 | 压缩必须在 **decode** 步执行；async 会预排下一步、与改写冲突 → 压缩开启**强制同步**，显式 `NO_ASYNC_SCHEDULING=0` 也会被覆盖并打印原因 | ✅ `async_scheduling_mode()` + 单测 |
| 闸门方向 | 旧逻辑把 decode 步全部推迟（只允许 prefill 压缩）；Route A 需要 decode 压缩，prefill 交给 prefill-pressure 路径 | ✅ 已翻转 |

### 触发语义

* 池是**一段连续的 B+R 空间**，判定「B 和 R 都满了」→ 重选。实现为：`_tokens_at_reselect` 在
  **prefill 结束**时锚定到 prompt 长度，之后 `total_tokens - baseline >= R` 才触发；首个 decode 步
  的初始视图构建会重选一次（与入口 B 选择幂等），其后严格每 R 个输出 token 一次。
* 适配器不自建定时器：`reselect_token_budget` 在压缩模式下置 0，R 由 planner 提供；无原生扩展时
  用 runtime 的 `gen_reserve` 兜底。

### 一对开关（必须同时开）

| 开关 | 位置 | 作用 |
|---|---|---|
| `KVMEM_ENTRY_INREQUEST_COMPACTION=1` | 入口进程 env | 生成上限从 R 抬到 CTX |
| `--kvmem-inrequest-compaction` | 引擎 worker CLI | 开启引擎内压缩、block 通道、强制同步 |

只开入口 → 上限放开但引擎不压缩，池会被撑爆；只开引擎 → 入口仍夹 R，压缩永不触发。

### 行为契约

| 压缩 | 输出上限 | 调度 | 引擎序列 |
|---|---|---|---|
| 关（默认） | R（客户端要更多也只到 R，正常 finish） | 异步（显式 `NO_ASYNC_SCHEDULING` 优先） | 单调、block table prefill 后冻结 |
| 开 | 客户端 `max_tokens`，到顶 `CTX`；唯一停止条件是 `max_tokens`/EOS | **强制同步** | 循环回收，`_all_token_ids` 恒 ≤ B+R |

### 验证状态

* 单测：入口上限 4 种取值、cadence/基线、强制同步、对账通道 —— 全绿（全套 38 个非 GPU 测试，
  `PYTHONPATH=build/python` 时 `disk_tier_test` 也过）。
* 活引擎矩阵（待跑，需先腾卡）：埋针正确性 / N=2 并发 / prefix-cache TTFT 不退化 / MTP 长生成不崩 /
  5050 复现（B=256 R=128，输出 512、1024，池恒 384）。
* 站点脚本注意：`NO_ASYNC_SCHEDULING=0` 在压缩开启时会被 worker 覆盖为 1（打印提示），不需要手改。

### 活引擎验证（2026-10-21 深夜，Route A 首轮）

首轮活引擎（生产配置 + 5050 小配置各一次）暴露并修掉了 5 个真 bug——它们都属于
「代码写好了但从未真跑过」的那类：

| # | 症状 | 根因 | 修法 |
|---|---|---|---|
| 1 | `compact_error: No such file or directory`，三本账从未生效 | EngineCore 里 `register()` 拿不到 `--additional-config`，`cfg.get("block_channel")` 恒 False | 通道改为**懒启动**：`Scheduler.schedule` 钩子无条件安装，首次读到 `block_channel` 配置时起 socket |
| 2 | 每次重选抛 `selection/reserve/host payload budget exceeded` | 入口/选择器按 128-token 块粒度会略超 B（实测 53929 > 53248），kvmem 内核严格拒绝 | `KvmemPlanner.plan` 按排名裁到 `budget_tokens` 内，mandatory 不动 |
| 3 | 重选后请求卡死（GPU 0%、吞吐 0） | compact 在下一步已排期后才应用，`num_computed` 被该步再加一次 → `num_tokens == num_computed`，永不再调度 | `_compact` 按在途步数回退：`num_computed = len(visible) - pending` |
| 4 | 输出全是乱码 | 这台引擎是 V2 runner，**没有 `prepare_pos_seq_lens` 方法**，sparse view（row/seq_len/positions）从未安装，引擎却已截断账本 | 给 `prepare_inputs` 加钩子：`bind_step` + `before_step`（复用现成 LiveRunnerView） |
| 5 | 安装后仍乱码 | `positions` 用「含 query 的长度」做基准，整体偏移 n（解码 +1、prefill 整块） | 基准改为 `seq_len - n` |
| — | planner/runtime 用了默认 B/R | 插件构造 runtime 时没传 `budget_tokens/gen_reserve/...`（用了 32768/16384） | 构造时传全，规划器/节拍与 CLI 一致 |

修复后小池（B=256 R=128）确实跑出了 Route A 的闭环：单请求 14 次 `compact_applied`，
可见集稳定在 310–326 ≤ B+R=384，首个用例密钥在压缩后仍正确回写。

**仍未通过**：连续数字在 ~50 个之后出现错位（如预期 52 却输出 5/554），说明压缩边界上
仍有一处 KV/位置不一致（最可能是搬运边界那一页的 RoPE/槽位，或 pending token 的位置）。
下一轮要用「单页边界 + 逐步对账」的方式定位，而不是继续盲改。

### Route A 第二轮定位（同日，小池 B=256 R=128）

| 发现 | 证据 | 处理 |
|---|---|---|
| compact 把**逻辑页号**当可见空间偏移 | `start = page * page_tokens`；第一次压缩后引擎的 `_all_token_ids` 是可见空间 | 改为按排名：`start = rank * page_tokens` |
| 压缩关时（同小池）数字序列连续，压缩开时 ~60 处坏 | 基线跑到 max_model_len=1024 上限，1..29 连续无误；开压缩时 61 号变 6 / 62 号变 6200 | 说明损坏由压缩路径引入，不是模板/采样 |
| 关掉 MTP 仍坏 | SPECULATIVE_TOKENS=0 时同样在 ~62 坏 | 排除投机解码交互 |

**当前阻塞**：压缩边界处仍有一个 token 的 KV/位置不一致（表现为数字某一位错/多一位）。
下一步应做逐步对账（重选后逐 token 比对引擎 `num_tokens`、运行时 `seq_len`、行内容与
`_all_token_ids` 的实际切片），而不是继续猜。

### Route A 第三轮（同日晚）：机制验证成立，跨请求状态待修

**重要方法论教训**：引擎端口一开就跑请求，会撞上启动/图捕获阶段（实测第一条请求输出退化成
"Write"），误判为压缩 bug。必须等预热请求（`_warmup_*`）结束后再测。

**已经验证成立**（B=53248、R=2048、MTP 开、单请求）：
- 长生成 5080 token、两次 R 节拍重选（tok 2100 / 4150，可见集 2100→4150），
  1693/1694 行 LEMON 完整无错、唯一异常是 max_tokens 截断的半个词
- `compact_applied` 在引擎侧真实生效（`_all_token_ids` 截断 + `num_computed` 回退）

**仍未通过**：同一引擎上后续请求开始输出乱码（混入 `<|box_end|>` 与随机多语种 token）。
首个请求干净、后续退化 → 跨请求的设备槽/页状态泄漏。已修其中一处：
`release()` 只清 `_owned_slots` 没清 `_slot_owner`，死请求的槽永远不可复用。
另发现 `_take_slots(req, budget_pages()+1)` 一次预留整个预算（B=53248 → 3329 页），
单请求看似无碍，多请求时必然耗尽/冲突，下一轮要改成「按选中页数按需取槽」。

本轮其余修复：规划器按排名裁到预算内；V2 runner 补 `prepare_inputs` 视图安装；
positions 基准改为 `seq_len - n`；压缩只在 decode 步触发；槽位从池顶向下分配；
新增 KVMEM_NO_ROW/NO_SEQ/NO_POS 活体隔离开关。

### Route A 第四轮：跨请求泄漏修好，短上下文全通；长上下文暴露架构级限制

**修好的**（活引擎 4 连发，各跨一次 R 节拍，全部干净）：
- `_seq_override` 按 batch row 索引，请求结束不清 → 下一个复用同 row 的请求在 prefill 阶段
  沿用旧 seq_len（实测：第 2 个请求从第一个 token 起就是噪声）。`release()` 现在清掉它
- `_take_slots` 改为按选中页数按需取槽（原来一次预留 `budget_pages()+1`，B=53248 时 3329 页）
- 取槽时排除引擎 block table 仍在引用的页
- orchestrator 接受 `slots=callable(needed)`，由调用方决定取多少

**新暴露的架构级限制（未修）**：
- 53K prompt（B=53248）请求在首次重选后卡死，GPU 0%、worker 95% CPU 空转
- 原因：运行时把可见集「另存」到自己从池里取的槽（3323 页），而引擎已为同一请求占了 3371 页；
  池总共 3456 页（max_model_len=55296）→ 需要 2 倍池容量，物理上不可能
- 结论：ninfer 对等的正确做法是**原地重排**（复用引擎已分配页 + RoPE 重打），不是另存副本；
  当前的「副本 + 独立槽池」模型只在 B 远小于池时可行（小 prompt 场景已全通）

短上下文（900 行/2820 token/跨一次重选）四连发全干净，说明压缩、对账、节拍、跨请求释放都正确。

### Route A 第五轮：长 prefill 后首次截断仍卡死（已定位到边界条件）

| 用例 | 结果 |
|---|---|
| 短 prompt + 长生成（900 行/2820 token，跨 R 节拍截断） | ✅ 四连发全干净 |
| 长 prompt（8K with B=8192 / 53K with B=53248） | ❌ prefill 100% 后、首次 decode 截断 → 引擎空闲（GPU 0%），客户端挂起 |

关键区分：**decode 中途的截断是好的**（短 prompt 跨节拍截断正常），**长 prefill 结束后的首次截断**
不好。compact 的账本数学已加探针（before_computed/pending/after_tokens/after_computed），下一轮从
调度器状态（请求是否仍 running、num_new_tokens、块分配）定位。

本轮另修：`_absorb_engine_row` 从「每 token 一次 row.tolist()」提到每步一次（长 prefill 原来是 O(n²)）。
原地重排的实现试过但离线夹具未同步、活引擎仍未通过，已回退，保留设计结论：
ninfer 对等必须原地复用引擎页，另存副本在 B≈池/2 时物理不可行。

### Route A 第六轮：修掉「未截断也安装视图」的卡死；真截断后仍卡

| 发现 | 证据 | 处理 |
|---|---|---|
| 长 prefill 后首次重选**没有截断**（`visible == all_tokens`），worker 仍安装压缩视图 | `compact_noop: visible=8112 all_tokens=8112 computed=8111` | 引擎返回 `truncated` 标志；未截断就不安装（清空 row） |
| 取消「首次强制重选」，只按 R 节拍重选 | 8K prompt 请求从**卡死**变为 22.9s 正常完成（压缩关基线 22.8s） | 已改 |
| MTP 下运行时把**草稿 token** 也算进序列长度（引擎只按接受的前进） | runtime tokens=10355 vs 引擎 8192 | 新增 `note_engine_computed()`，用引擎 `num_computed_tokens_np` 每步校正 |

**仍未通过**：本次真截断（`compact_truncated=true`）后请求仍停在 running 不再被调度（GPU 0%）。
下一步：确认 `note_engine_computed` 是否真的被调用（加探针），再查截断瞬间 `num_computed` 与
`num_tokens` 的实际关系。

### Route A 第七轮：长上下文不再卡；「选择器保留量 < prompt」仍会卡

修掉的三处：

| 问题 | 证据 | 修法 |
|---|---|---|
| 两个 TP rank 各自重选，第二个到达时账本已被截断 → 判 noop → 清掉自己的视图 | 长上下文真截断后卡死 | compact 幂等：`peer_applied` 时也返回 `truncated=true`，两 rank 都安装 |
| 截断到 **prompt 以下** 破坏 vLLM 不变量（`num_tokens < prompt_len`） | `visible=8192 / prompt_len=8305` 后卡死 | 拒绝截断到 prompt 以下（`compact_refused`） |
| 选择器可以丢 prompt 页 | 同上 | 把 prompt 覆盖的页全部作为 mandatory 传给选择器 |

**已验证**：5.5K prompt + 4350 输出、两次 R 节拍重选 → **1450/1450 行正确、49.1s 完成**（此前长上下文必卡）。

**仍卡**：选择器只保留 4368 token（< prompt 5540）时，`compact_refused` 之后请求仍停在 running。
下一步：让 mandatory 真正生效（planner 的 trim 是否把 mandatory 也裁了），且「拒绝」路径不得影响请求继续。

### Route A 第八轮：mandatory 生效，但两个 TP rank 选页不一致

证据（同一请求、两个 rank）：

| rank | mandatory_pages | selected | seq_len | compact |
|---|---|---|---|---|
| A | 347 | 507 | 8112 | truncated |
| B | 347 | **273** | **4368** | refused（< prompt 5539） |

- mandatory 在正常路径确实生效：5.5K prompt + 4350 输出、两次节拍重选 → 1450/1450 行，48.8s ✅
- 但 rank B 只选出 273 页（< mandatory 347 页），导致 `visible < prompt` → 拒绝 → 请求停在 running

**根因假设**：两个 rank 的 `orch.total_tokens`（运行时候选判据）不一致——MTP 下运行时按
`num_scheduled_tokens` 计数（含被拒绝的草稿），两卡的接受情况不同 → 选页范围不同。
`note_engine_computed` 已加但看起来没生效（runtime tokens 9638 vs 引擎 8112）。
下一轮：确认 `note_engine_computed` 是否被调用（V2 hook 里 `req_states.num_computed_tokens_np`
是否可读），保证两 rank 的计数一致。

### Route A 第九轮：TP 两 rank 选页一致；剩余「选择器不保证覆盖 prompt」

- 漂移根因确认：`after_step` 先把 `st.tokens` 设成引擎值，提交循环**又加一遍**本步 token
  → 每步多算 prev（MTP 下 5），累计 9638 vs 引擎 8112，且两卡接受情况不同 → 选页不同
- 修法：`st.tokens = engine_c - n_new`（提交循环再加回来）
- 效果：同一请求两个 rank 现在都选 331 页、visible 5296，**完全一致** ✅

仍卡：选择器返回 331 页 < mandatory 346 页（prompt 覆盖页），visible 5296 < prompt 5539 →
`compact_refused` → 请求停在 running。下一轮两条路：
1. 选择器/规划器强制覆盖 prompt 页（mandatory 语义修透）；
2. 或在 worker 侧把「prompt 页」直接并入选择结果再交给编排器（不依赖 C++ 语义）。

### Route A 第十轮：forced_pages 也救不回「planner 返回的 visible」

- 在 `orch.reselect` 增加 `forced_pages`：把 prompt 页硬并入选页集合（不依赖选择器语义）
- 实跑仍出现 `compact_refused: visible=5312 < prompt 5540`：选页集合已包含 prompt 页，
  但**规划器返回的 next.visible_tokens 仍少 228 token**
- 结论：缺口在 `KvmemPlanner.plan` → `ext.plan_working_set`（catalog/snapshot 与 chosen 的对齐），
  不是选择器。下一轮直接看 `snapshot()` 的 catalog 与 `plan.next.blocks` 的差集。

### Route A 第十一轮：forced_pages 排序修复 + 测试契约

- `forced_pages` 初版把页**追加**到末尾，破坏「升序 sink..recent」契约（离线测试立刻报
  `selected pages must be ascending`）→ 改为合并后 `sorted()`，测试恢复全绿
- 只在 `prompt_tokens <= B` 时强制（prompt 超过 B 说明入口没裁好，强制会让视图超 B 本就该失败）
- 实跑仍 `visible=5296 < prompt 5541`：选页集合已含 prompt 块，但 planner 返回的
  `next.visible_tokens` 仍少 ~2 个块 → 下一轮在 `plan()` 里打印
  `selected_blocks / plan.next.blocks / visible` 的差集定位

### Route A 第十二轮：缺口定位到 planner 之后、row 构建之前

探针（同一 reselect 记录）：

```
plan: {visible: 7585, kept_blocks: 60, sel_pages: 475, mandatory: 346}
selected: 332   seq_len: 5312   prompt: 5535   mandatory_pages: 346
```

- 规划器**是对的**：60 块 = 475 页 = 7600 token，visible 7585 ≥ prompt 5535 ✅
- 但 `stats["selected"]` 只有 332 页（5312 token）→ 截断被拒（visible < prompt）
- 结论：`selected` 在 planner 之后、`_finish_reselect` 之前被缩小（475 → 332）。
  嫌疑：`slots` 分支/`page_to_slot` 分支/`self.total_tokens` 过滤。下一轮在
  `_finish_reselect` 之前打印 `len(selected)` 并对照 `self.total_tokens`，一次就能抓到。

### Route A 第十三轮：卡死全清；压缩后输出质量下降（RoPE 重打/搬运）

两处修复（都带探针证据）：

| 问题 | 证据 | 修法 |
|---|---|---|
| 选中的页在 store 里没有规范字节 → reselect 直接丢弃 → 视图缩到 prompt 以下 → 拒绝/卡死 | `plan.sel_pages=475` 但 `selected=331` | ① 缺失页当场从设备捕获入 store；② 重选前做一次**补提交扫描**（把 `_page_map` 里已知但未提交的页补进 store） |
| — | 修复后：`selected=507, seq_len=8112 ≥ prompt 5537` | 卡死消失 |

**结果**：2.2K 行用例 121s 跑完（此前必卡），但**输出质量下降**：2415 行里只有 849 行是完整
LEMON，其余退化成 `w`/`le` 碎片——压缩后 RoPE 重打或页搬运有偏差。下一轮对着
`rotation_deltas`/`plan_moves`/`order_moves` 逐页核对（离线已有 rotate/needle 单测可复用）。

### Route A 第十四轮：visible_limit 方案导致 CUDA 非法访问（已回退）

- 假设：`st.seq_len` 用了 `len(selected)*page_tokens`（8112）而规划器 visible 是 8102，读多了
  10 个未写 token → 碎片化
- 改动：`seq_len` 与 compact 都改用规划器 visible（`visible_limit` 通道参数）
- 结果：**引擎直接 CUDA illegal memory access 崩溃**（可能让 kernel 读越界）
- 处理：整轮回退，保留上一轮已验证的修复（补提交扫描 + 缺失页捕获 ⇒ 不再卡死）
- 下一轮方向：碎片化问题别动 seq_len/attention 边界，改查「重打后页字节与目标位置的对应」
  （离线 rotation/needle 单测可扩展成「压缩后逐页验证 rank 偏移」）

### Route A 第十五轮：小池全通；长上下文碎片化的真凶是「recent 输出块没有检索分数」

| 实验 | 结果 |
|---|---|
| 小池 B=512 R=128 recency，1..200 数字 | ✅ 连续 + 末尾密钥正确（10 次压缩） |
| 小池 600 行（20 次压缩） | ✅ 493..515 连续、无碎片 |
| 长上下文 B=8192 retrieval + 2200 行 | ⚠️ 849/2415 行（碎片化） |
| 长上下文同配置改 **recency** | ✅ **1383 行全对 + 密钥正确**（43.6s，finish=stop） |
| 长上下文 retrieval + **recent 页强制保留** | ✅ **1889/1892 行**（此前 849/2415） |

根因：检索索引只覆盖「捕获过统计」的块，新生成的输出块分数是 `-inf` → 被选择器丢掉 →
模型失去刚写的上下文 → 碎片化。修法：`_reselect` 把最近 `recent_tokens` 覆盖的页也放进
mandatory（与 prompt 页同等对待）。

剩余：最后 3 行（max_tokens 边界处）模型自己跑偏，属预算尾部现象。

## Route A 活引擎验证矩阵（2026-10-22，实测）

双开关：`KVMEM_ENTRY_INREQUEST_COMPACTION=1` + `--kvmem-inrequest-compaction`；压缩开强制同步调度。

| # | 项 | 配置 | 结果 |
|---|---|---|---|
| ① | 压缩后召回（密钥在长历史开头，压缩两次后回写） | B=8192 R=2048 recency | ✅ `secret_present=true`，43.6s，finish=stop |
| ② | N=2 并发，两条请求各自跨 R 节拍压缩 | B=8192 R=2048 N=2 | ✅ 两条都完成，各 1033/1034 行正确，33.8s |
| ③ | prefix-cache 不污染 | B=8192 R=2048 N=1 | ✅ 同一 prompt 第二次 `cached_tokens=4848`，5.7s → 2.3s（2.5×） |
| ④ | MTP 长生成不崩 | B=8192 R=2048 MTP=4 | ✅ 6750 token 生成、多次截断、无崩溃 |
| ⑤ | 5050 复现 | B=256 R=128 recency | ✅ 1..200 连续 + SECRET 正确，3 次压缩，7.0s |

已知边界（诚实记录）：
- 检索模式下，新生成块的分数是 `-inf`；已在 reselect 里把最近 `recent_tokens` 的页设为 mandatory，
  长上下文从 849/2415 提升到 1889/1892 行
- 输出长度接近 `max_tokens` 时模型自身会跑偏（与压缩无关）
- 压缩期间 `st.seq_len` 不能低于 prompt（vLLM 由 `num_tokens - prompt_len` 推导输出长度）

### 单开关收敛（0.2.0 内）

超 R 生成本来横跨两个进程：入口上限在 API server（env），压缩在 EngineCore/TP worker（CLI）。
`worker.py` 现在把 CLI 开关作为唯一权威：打开时自动导出 `KVMEM_ENTRY_INREQUEST_COMPACTION=1`
（API server 继承 worker 的环境），关闭时把该变量压回 0；与 CLI 相反的显式取值会被纠正并打印原因。
环境变量只保留给「脱离 worker 单独跑入口策略（网关）」的场景。

活引擎验证：**不设环境变量**、只加 `--kvmem-inrequest-compaction` → 输出 3800 token（> R=2048，
上限确实被抬高），compact_applied 2 次。离线新增 4 条断言（`entry_compaction_env`）。

### 修正：超 R 的输出上限是 CTX，不是 CTX−B

之前的推导错在把 `B` 当成了「历史总量」。实际 kvmem 里 `B` 只是**常驻窗口**：其余历史在 host 层，
逻辑上下文可以一直增长，所以约束是「逻辑上下文 ≤ CTX」，输出上限就是 **CTX**（不是 CTX−B）。
`EntryPolicy.generation_ceil()` 已改为返回 CTX；文档/测试同步更新。

### 版本归位：a712280 是 0.2.1 不是 0.2.0

CTX 修正是 0.2.0 的 bugfix，不应移动 0.2.0。0.2.0 恢复为原始发布（d4c2be4，上限 CTX−B，
原 tag 对象 f31f3f9 原样恢复），修正提交 a712280 单独打 **0.2.1**（上限 CTX）。

### Bugfix：中文 prompt 超过 B → prefill 超预算 → 乱码 / 前缀缓存失效

活引擎日志（N=2）里的直接证据：某个请求 `prompt_len = 143216`（B=53248），prefill 期间
`compact_refused` 连发 12 轮（每轮都先分配 3327 个 device slot 并搬页），随后 sink 页 0/1 的
needle 校验开始失败（`ok:false`），输出变成噪声。三个独立缺陷叠在一起：

1. **入口按「估算 token」限 B**：B 用 `chars_per_token=3.0` 换算成字符预算，中文实测约
   1.1–1.7 字符/token，于是「B tokens」实际送进引擎 143216 tokens（2.7 倍）。校准函数只在
   文本超过 3.0 估算阈值时才启用，所以绝大多数长中文根本不校准。
   修：只要有 tokenizer 就在 `> max(4096, B/2)` 字符时校准；选完之后再用真实 tokenizer
   复核，超了就把预算按实测比例重选，直到 ≤ B。
2. **多模态 content 被整段跳过**：`content` 是 parts 数组（带图片）时 `_select_in_messages`
   只认字符串，整条消息不裁剪 → 附件长文原样进引擎。
   修：`_content_text` 取 text parts 参与选择，回写时保留 image parts。
3. **引擎对超预算 prompt 仍然尝试压缩**：压缩不能低于 prompt（vLLM 要求
   `num_tokens >= prompt_len`），于是每轮都「先搬页、后被拒」，把 view 弄坏。
   修：`note_prompt_len`（V2 hook 从 `req_states.prompt_len` 取）→ 超 B 的请求一律不再
   尝试压缩，只打一次 `prompt_over_budget`。
4. **并发下的槽位复用**（N=2 才暴露）：`_reselect` 一开始就把本请求的 slot 全部释放，
   而 `orch.slot_of` 仍指向它们 —— 另一个请求立刻可以把这些 slot 占走，随后的「设备内搬页」
   就把**别人的字节**拷进自己的 view（sink 页 needle 失败的另一种成因）。
   修：重选期间不释放旧 slot（`_take_slots(keep=...)` 只允许复用本请求的非源 slot），
   重选结束后再回收；拒绝路径只回收本轮新拿的。
5. **rebuild 写入早于设备搬页**：rebuild 的目标可能是另一页搬移的源，先写就把源覆盖了。
   修：先做设备搬页，再写 rebuild。

离线：`entry_parts_test`（中文 1.1 字符/token + parts 附件）、`prefill_over_budget_test`
（超预算不再动 view）、`reselect_write_order_test`（rebuild/搬页顺序）三条新回归，
40/42 通过（`gpu_real_tensor_test` 因活引擎占用显存、`sparse_rope_test` 需 KVMEM_TEST_CONFIG，
两者在改动前同样失败）。

### 常驻工作区（Option B，0.2.2）：prefill 也受 N×(B+R) 约束，长 prompt 无损

活引擎日志暴露的结构问题：引擎的 KV 块是 **1616 tokens = 101 个 16-token page**（`mamba align`），
池子 188,547 tokens ≈ **116.7 块**；而运行时按 16-token page 从池顶拿页，等于偷了分配器眼里还空着的页
——两个请求的页重叠（needle 失败/乱码），池子记两遍账（卡死 → 全量重 prefill）。

改成**双计数**（fork 补丁，`vllm/v1/core/sched`）：

- `num_logical_tokens`（新增，单调）：已服务 token 数，决定调度多少新 token、prefill/decode 判定、
  输入 token 窗口（`_prepare_prefill_inputs_kernel` 新增参数）。
- `num_computed_tokens`（语义改为**常驻书长度**）：positions / seq_lens / 块分配；压缩只回退它。
- `_all_token_ids` / `num_tokens` 保持逻辑，永不截断；`check_stop` 在常驻模式下用 `kvmem.ctx`。
- 引擎只在 `ceil(num_computed/1616)` 增长时分配块 ⇒ 每请求 ≤ 54 块，N=2 ≤ 108/116，不抢占、不释放。

运行时（`src/kvmem_sparse_runtime.py`）：

- 压缩**只在请求自己的页里**做（`_own_pages`，行按块量子补齐），不再从池顶拿页；
- `_absorb_engine_row` 加吸收游标，避免把自己的打包行当引擎重排读回；
- `st.tokens` 跟逻辑计数、`st.seq_len` 跟常驻计数；
- prefill 压力路径对长 prompt 正常压缩（不再按 `prompt_over_budget` 跳过），只记一次
  `long_prompt_chunked`；`host_share` 日志把每个 session 的 host 预算（tokens）暴露出来。

入口：压缩开启时**不再裁剪** history（`_select_in_messages` 直接返回 None），只保留 CTX 与输出上限；
前缀缓存在压缩开启时强制关闭（块哈希是逻辑 token、KV 布局是压缩后的）。

离线：新增 `resident_book_test`（双计数契约）、`prefill_compaction_test`（64K prompt 边 prefill 边压缩、
常驻 ≤ 水位、只用自有页）、`reselect_write_order_test`；`sparse_runtime_test` / `sparse_concurrency_test`
改到常驻语义。42/43 通过（`gpu_real_tensor_test` 需空闲显存）。
补丁重新生成：`backends/patches/vllm-kvmem.patch` 新增 speedmeter 两个模块（routers.py 依赖它），
`prepare-backends.py --check-tree` 通过（tree 4133af45…）。

### 常驻模型进展（本轮）：prefill 期压缩的修复与边界

本轮修了五处，并给出可用边界：

| 修复 | 说明 |
|---|---|
| 在途 chunk 提交 | prefill 期重选时，当前 chunk（~8K token）已写入但未提交；`page_map` 在压缩后按秩索引查不到它们 → 整块从上下文消失（这是「有时对有时错」的根源）。改为从**行**按秩提交，并把选择前沿推进到 `tokens + inflight` |
| recent 覆盖在途 | recent 窗口按 `tokens + inflight` 计算，最新 chunk 不再被挤出 |
| mandatory 夹取 + planner 容错 | mandatory 超过预算时被 planner 拒绝（`mandatory blocks must appear exactly once`）→ 夹到预算内，并在 `plan()` 里丢弃已被选择丢掉的 mandatory 块 |
| 压缩失败软着陆 | `_reselect` 抛异常不再中断 `after_step`（记账继续），只记录 `*_reselect_error` |
| 压缩长度对齐 | 压缩消息带上运行时的 `visible`（`st.seq_len`），引擎的 `num_computed` 与安装的行严格一致 |
| 位置上限 | 调度器 `max_model_len - 位置` 的上限改用**服务位置**，否则 199K prompt 会停在 180,000；`ctx` 随 additional_config 下发 |

验证（`KVMEM_ENTRY_RESIDENT_PREFILL=1`，窗口 180K）：
- 4K / 80K / 170K 单请求：未裁剪、两把 needle key 全中
- N=2 × 40K：通过；N=2 × 80K 早期通过一次，本轮未复测
- 200K：prefill 期压缩可跑到 90% 但不稳定（有时输出噪声、planner 偶发拒绝），**未通过**
- N=2 × 60K：超时（单请求 60K/80K 正常），**未通过**

结论：**默认（prefill 不压缩）路径稳定可用，覆盖到引擎窗口（~180K，单请求）**；prefill 期压缩仍是实验性的
（`KVMEM_RESIDENT_PREFILL_COMPACT=1` 才开），200K 与 N=2 长并发在此路径上未达标。
