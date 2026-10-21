# 方向核对：kvmem 主办、vLLM 当后端（对照 kvmem-llama 的实现）

## 1. kvmem-llama 的结构（读源码确认）

```
kvmem 核心（策略 + 分层存储）
  kvmem/include/kvmem/memory_contract.hpp     <- 后端必须实现的契约
      MemoryBackend{ descriptor snapshot stamp prepare invalidate }
      MemoryTransfer{ start_spill poll start_restore publish abort }
      MemorySession(backend){ prepare / prepare_reselect / advance / publish / abort }
  kvmem/src/host/*                            <- 主机池、SSD 层、pinned 缓冲、LRFU、mean-k 检索

worker（每个后端一个）：
  tools/llama-kvmem-server.cpp                <- llama.cpp worker，KVMem 策略库直接链在 worker 内
  ninfer worker（ninfer-serve）
启动器：
  scripts/windows/start-backend.ps1 -Backend llamacpp|ninfer
  参数以 kvmem 的 CLI 为准（--kvmem-budget / --kvmem-host-mib / ... 共 15 个）
```

关键点：**入口与策略在 kvmem 侧，引擎是"被嵌入的 worker"**；KV 的溢出/还原由 kvmem 的
`MemorySession` 驱动 `MemoryBackend`，主机预算 H 由 kvmem 管。

## 2. 本仓库已有的、方向正确的部分

| 组件 | 对应关系 |
|---|---|
| `src/vllm_backend.cpp`（VllmMemoryBackend） | 实现 `MemoryBackend`：descriptor/snapshot/stamp/prepare/invalidate ✅ |
| `src/vllm_backend.cpp`（VllmMemoryTransfer） | 实现 `MemoryTransfer`：start_spill/start_restore/publish/abort ✅ |
| `src/vllm_integration.py`（KvmemAdapter） | **已经**通过 `MemorySession.prepare(stamp, selected, mandatory, budget)` 驱动 ✅ |
| `src/policy_store.py` | 提供 `KvMemStore` 风格的策略（确定性、TP 锁步）✅ |
| pybind 导出 | MemoryDescriptor/BlockDescriptor/ViewBlock/ExecutionView/Geometry ✅ |

也就是说：**"kvmem 主办、vLLM 当后端"这条路，前几阶段已经建好了骨架。**

## 3. 我后来偏离的地方（只有一处）

`src/kvmem_sparse_runtime.py` 是**第二套 driver**：

- 自己计时（`reselect_interval` / `reselect_token_budget`）；
- 自己调 `VllmSparseSelector` 选片；
- 自己用 `HostPageStore` 存主机副本（绕过 kvmem 的 `host_kv_storage` / H 预算）。

它绕过了 kvmem 的 `MemorySession`，所以严格说"kvmem 主办"这条不成立——这是我加的旁路。

## 4. 收敛方案（不是大改，是换驱动）

| 要做 | 替换成 | 规模 |
|---|---|---|
| 重选触发 + 选片 | `session.prepare_reselect(stamp, policy_store, mandatory, budget)` | 接线，小 |
| 主机页库 | kvmem 的 `host_kv_storage` / `pinned_kv_tier`（H 预算由 kvmem 执行） | 接线，中 |
| 入口 | 新增 worker：解析 kvmem CLI → 启动 vLLM → pybind 绑定 backend | 新封装，中 |
| **物理层** | **全部保留，零改动** | — |

物理层就是这些（都是一次次踩坑验证出来的，任何 vLLM 后端都绕不过）：

1. 一个逻辑页 = 17 个分页注意力层（整体搬移/旋转）；
2. 重打 RoPE 的 delta = (压缩名次 − 逻辑页) × 16，fp8 只做**一跳**、绝不链式；
3. 页面字节要**与 vLLM 自己的 rope 逐位一致**才不漂移；
4. 块表行改写 + `seq_len`/`positions` 覆盖 + `slot_mapping` 由 positions 派生；
5. `seq_lens` 只能喂注意力组，GDN 组必须保持真实长度；
6. 页号必须取自引擎自己的行（NULL 块占 101 页，不从 0 开始）；
7. 调度器释放粒度是 1,616-token 对齐块（101 页）；
8. 滑动窗压缩下每次重选必须整窗重打（已证明搬运/链式都不可行）。

## 5. 结论

- **方向是正确的，而且不需要推翻重来**：骨架在，物理层在；
- 需要的是把**第二套 driver 退回到 kvmem 的 `MemorySession`**，并把主机池换成 kvmem 的层；
- 入口按 `start-backend.ps1 -Backend vllm` 的形态补一个 worker。

风险提示：`prepare_reselect` 走的是 kvmem 自己的 `KvMemStore`（含 LRFU/mean-k），它按
**1,161-token 块**度量；而调度器的对齐块是 **1,616 token**，两者粒度不一致需要在适配层换算
（把 kvmem 的块映射到页区间，再判断哪些对齐块可整体释放）。
