# 两条路：读侧已通，分配侧有两条实现路径

## 现状（已实测）

- ✅ 注意力读取量：99K 对话压到 ≤32,768 token（V1/V4/检索探针全过）
- ❌ 页没有还给池子：引擎仍为每个 token 持有页（29K 请求实测 1918 个非零页）

目标里的"引擎池支持 4 条并发 200K"取决于**分配侧**，因此还需要选一条路。

## 路 B：用 vLLM 自带的 R-SWA 机制（短、稳、真省显存）

vLLM 已有完整的"丢弃窗口外块"设施：

| 组件 | 作用 |
|---|---|
| `remove_skipped_blocks()` | 把不再需要的块**释放**并把槽位替换成 `null_block`（位置索引不变） |
| `get_num_skipped_tokens()` | 每种注意力类型自己定义"窗口外有多少 token"；R-SWA 版处理中间 gap |
| `unified_attention(..., rswa_prefix_lens, rswa_window)` | kernel 原生按"前缀全局 + 滑动窗"掩码 |

做法：开启 R-SWA 语义（sink = prefix，recent = window），由 `remove_skipped_blocks` 释放 gap 块。

- 优点：**真释放显存**、无重打 RoPE、无块表压缩、无 positions 覆盖，全部走既有机制；
- 限制：选入只能是"前缀 + 最近窗口"，**没有检索**（kvmem 的核心价值用不上）；
- 工作量：小（主要是把 `rswa_prefix_lens`/`rswa_window` 接上并确认 hybrid 模型下 Mamba 组不受影响）。

## 路 A：调度器侧自由选入（我们现在这条的完整版）

worker 侧无法还页（所有权在 `block_pool` + `req_to_blocks`），必须：

1. 逐出决策放 EngineCore 调度器（TP=2 下只有一份，天然锁步）；
2. 落选页 → worker 溢到主机 → 调度器 `block_pool.free_blocks()` 并从 `req_to_blocks` 移除；
3. 还原 → `get_new_blocks()` 取新页 → worker 写入并重打 RoPE → 登记回 `req_to_blocks`；
4. 同步改写块表行（worker 能力已有）。

- 优点：保留 kvmem 的检索式选入（相关性 → 显存）；
- 风险点（都必须处理）：
  - `remove_skipped_blocks` 用的是 `null_block` 占位，位置索引不变；而我们压缩行会让**长度变短**，
    这两套语义冲突，需要统一；
  - 前缀缓存按块哈希记账，被移出的块会被回收/复用；
  - 预抢占（preemption）与我们移除块逻辑的交互；
  - MTP/投机解码下每步的 draft 块。
- 工作量：大（调度器 + worker 双端改动），且需要重跑全部验证。

## 建议

先做**路 B**：它是唯一能在短周期内真正拿到"4 并发长请求"的路径，且复用引擎既有机制、风险最低。
路 A 保留为下一步增强（把 B 的"窗口"换成"检索选出的集合"）。

## 更正：路 B 在本机不是"短路径"（实测后端不支持 R-SWA）

查证结果推翻了上面"路 B 工作量小"的判断：

| 事实 | 位置 |
|---|---|
| 本机实际后端是 **FlashInfer** | 启动日志 "Using FLASHINFER attention backend out of potential backends: [...]" |
| FlashInfer **完全不支持 R-SWA** | `vllm/v1/attention/backends/flashinfer.py` 里 `rswa` 出现 **0** 次 |
| 支持 R-SWA 的只有 FlashAttention / Triton | `flash_attn.py` 41 处、`triton_attn.py` 有完整字段 |
| 这个模型没声明 rswa 窗口 | `rswa_window` 来自模型架构配置，qwen3_5/next **没有任何声明** |
| SM75 上 FlashAttention-2 不可用 | FA2 要求 SM80+；本机是 Turing |

所以要走路 B，必须同时：①换到 Triton attention 后端（SM75 可用）；②给模型配置注入 `rswa_window`。
这已经不是"接线"，而是**换引擎后端 + 改模型配置**，风险（MTP、融合 kernel、图捕获）都要重新验证。

**结论修正**：
- 路 B 在本机**不是短路径**；
- 路 A（调度器侧释放）才是唯一能在当前栈上拿到真实显存收益的路径；
- 若你愿意换 Triton 后端，B 可以作为"零重打 RoPE"的对照方案，但它给不了检索。
