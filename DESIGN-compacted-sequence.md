# 让 vLLM 认识"被压缩的序列"：最小接口设计

> 背景：KVMem 的 in-request 压缩会缩短请求的有效序列长度；vLLM 的调度器/块管理器
> 按**逻辑位置**记账和寻址，两者冲突时触发 `model_runner.py:1348`
> 的 `assert num_scheduled_tokens >= num_logits`。本文给出最小、可回退的接口扩展方案。

## 1. 冲突的准确位置（已实测确认）

| 层 | 行为 | 冲突点 |
|---|---|---|
| `Scheduler.schedule()` | 按 `num_computed_tokens` 决定本步调度多少 token | 它认为序列有 37231 token |
| `KVCacheManager.allocate_slots()` | 按**逻辑块下标**（`num_computed_tokens // 128`）写入 block table 行 | 压缩把页搬到 0..127，逻辑下标却到 290 |
| `GPUModelRunner.prepare_inputs()` | 用 `num_scheduled_tokens` 与 `draft+bonus` 计算 logits 行 | 行满/越界时 `num_scheduled < num_logits` → 断言 |
| 我们的 `_reselect()` | 把选中的页写进行的前 N 项，`st.seq_len` 改小 | 引擎随后仍按旧下标写/读 |

实测（N=1、37K、budget 16384）：压缩 20 次全部在 prefill，断言在解码首步触发。

## 2. 方案 A（推荐）：给 `RequestState` 加一个"可见长度"

### 2.1 vLLM 侧改动（3 个文件、约 40 行）

1. **`vllm/v1/request.py`**
   ```python
   # 由外部组件（KVMem 适配器）设置；None 表示未压缩
   self.compacted_tokens: int | None = None
   ```
2. **`vllm/v1/core/kv_cache_manager.py`**
   - 在 `allocate_slots()` 里用 `effective = req.compacted_tokens or req.num_computed_tokens`
     作为块下标基准；
   - `free()` 时按同一基准回收。
3. **`vllm/v1/worker/gpu/model_runner.py`**
   - `prepare_inputs()` 里把 `num_scheduled_tokens_np` 与 logits 行计算都改用
     `effective_positions`（压缩后的位置映射），断言因此恒成立。

### 2.2 适配器侧改动（约 60 行）

- `_reselect()` 完成后，通过**已有的 block channel**（现在是 EngineCore→worker 的单向分配请求，
  扩成双向）上报 `(req_id, compacted_tokens, new_page_list)`；
- EngineCore 在 `schedule()` 前应用：设置 `req.compacted_tokens`、写入块表行。

### 2.3 风险与回退

- 全部行为挂在 `compacted_tokens is not None` 上：**不压缩时零影响**（默认 None）
- 若中途异常，`compacted_tokens=None` 即回到原始路径
- 需要重新验证：99K 常驻页下降、埋针 22/22、释放/还原闭环、N=2/N=4 并发

## 3. 方案 B（零改动）：把压缩降级为"会话级"

- 生成期间完全不截断（走 vLLM 原路径）
- kvmem 只负责：H/D 分层、跨会话共享、**会话恢复时的检索选页与重建**
- 代价：单请求内工作集不再随生成收敛，长上下文靠显存硬扛
- 收益：零 vLLM 改动、性能最好（prefill 37K 实测 27 s）

## 4. 建议

| 目标 | 选 |
|---|---|
| 要"99K 上下文在 22 GB 上跑、且生成期间驻留下降" | **方案 A** |
| 要"现在就能稳定、性能优先" | **方案 B**（已可用，见 STATUS 的能力矩阵） |

方案 A 的接口面很小（一个字段 + 三处使用），符合"只加对外接口、不深度定制源码"的约束；
但它改变了 vLLM 的块寻址语义，必须配一组回归测试才能上生产。

## 5. 复核后的更正（2026-10-21 05:3x）：方案 A 比预估的大

重读 `kv_cache_manager.py` 与 `request.py` 后，必须修正第 2 节的估算：

| 事实 | 位置 | 后果 |
|---|---|---|
| 块下标由 `num_computed_tokens` 派生（`cdiv(num_computed_tokens, block_size)`） | `kv_cache_manager.py:427,770` | 只改一个 `compacted_tokens` 字段不够：调度器会认为"还差 37231-24576 个 token 没算"，从而**重新预填充**被丢掉的上下文 |
| 请求的 token id 列表是单调增长的 `_all_token_ids`（`append_output_token_ids`） | `request.py:146,271` | 要保持一致，必须**同时截断 token id 列表**，否则 logits/采样与 KV 内容错位 |
| 块表行由调度器按逻辑下标写入，worker 只是拷贝 | `get_block_ids` + worker 的 `block_table` | 压缩后行内容与逻辑下标脱钩，attention 读到陈旧页（实测：答案变乱码） |

**结论**：真正自洽的方案需要在 vLLM 里引入"上下文重写"语义（截断 `_all_token_ids` + 回退
`num_computed_tokens` + 重排块表），这已经属于**改动 vLLM 核心数据模型**，不符合
"只加对外接口 / 不深度定制源码"的约束。

因此推荐 **方案 B（会话级）**，并把 in-request 压缩保留为实验开关：

| 能力 | 方案 B（默认） | 方案 A（需改核心） |
|---|---|---|
| H/D 分层、跨会话共享、LRFU | ✅ | ✅ |
| 生成期工作集随生成收敛 | ❌（靠显存硬扛） | ✅ |
| 长上下文 + 长生成 | ✅（实测 37K prefill 27 s） | ✅ |
| vLLM 源码改动 | **零** | 核心数据模型，违反约束 |
| 性能 | 最好（走原路径） | 压缩有开销 |

**实测支撑**：压缩关掉后单请求 37K 长解码 **200 / 29 s / 零断言**；N=2、N=4 并发均返回 200；
但"视图装着、压缩关着"是半初始化状态（答案乱码 + `bridge_harvest_error`），说明二者必须同开同关。
