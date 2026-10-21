# 交接计划：KVMem × vLLM 收敛的剩余三项

## 已完成（有活引擎证据）

| 项 | 证据 |
|---|---|
| ① 视图由 kvmem `plan_working_set` 决定 | 99K 上下文 → 常驻 2,047–2,048 页、重选 40–46 次、埋针 40/42 全过、0 错误 |
| ④ kvmem CLI 入口 | `scripts/kvmem-vllm-worker.py`，13 项测试（参数透传/校验/dry-run） |
| 4 并发 | 4×27.5K 全部答案正确，124 s，0 错误 |
| 释放路径 | 调度器真释放 20/39 对齐块，`free_blocks` 上升，答案不变 |
| 还原数据面 | `restore_page` 一跳、不累积，4 项单测 |

## ~~剩余 1：`use_kvmem_host` 通用化~~（已完成）

**已完成**：记录布局从 payload 自身发现（每层 shape/dtype/nbytes），`use_kvmem_host` 默认 True；
活引擎 33K 请求：重选 8 次、埋针 8/8、0 错误。

位置：`src/kvmem_sparse_orchestrator.py::KvmemHostStore`

现状：`__init__` 收一个 `tensor_shape`，`put/get` 用它切分；不同用例层形状不一致时会
`shape '[16, 512]' is invalid`。

改法：
1. `put` 时记录 `[(shape, dtype, nbytes), ...]` 而不是单一 shape；
2. `get` 按记录的清单逐层 `reshape`；
3. 去掉 `record_bytes` 的强校验（每条记录按实际字节分配，H 按实际字节记账）；
4. 把 `use_kvmem_host` 默认改 `True`，`tests/sparse_runtime_test.py` 与
   `sparse_concurrency_test.py` 应保持绿。

验证：`tests/host_store_test.py` + `sparse_runtime_test` + 一轮活引擎 29K（答案 + 埋针）。

## 还原控制面：通道已实现并单测（调用者待检索模式）

释放闭环：`scripts/free_loop_check.sh <port>`（4 并发 → 4/4 答案正确 + 块真释放）。

**背景**：recency 选入下"滑出窗口的页永不再被选中"，所以释放是终态、不需要还原。
只有换成 mean-K 检索才需要把驻泊页取回。

**已完成**：`src/kvmem_block_channel.py`（socket 通道，失败上报不拖垮调度器）+
`restore_page(..., slot=None, channel=...)`；`tests/block_channel_test.py` 与
`restore_path_test.py` 覆盖。开关 `block_channel`（默认关）。

位置：`plugin/vllm_kvmem_plugin.py`（两侧都加载它）+ `src/kvmem_sparse_runtime.py`

两条路（见 `ARCHITECTURE-VERDICT.md`）：
- 干净：两侧 patch，worker 在 `ModelRunnerOutput` 附带请求字段；
- 务实：unix socket / 共享文件轮询。

最小闭环验证：释放 1 个对齐块 → `free_blocks` +1 → 申请取回 → 写回 KV → 埋针一致。

## ~~剩余 3：EngineCore 内 N 会话共享 H~~（已完成大半）

**已完成**：`src/kvmem_session_budget.py`（H 按会话均分、只可借未分配余额、超发报错）、
`runtime.release()` + worker 侧扫尾（请求消失即归还份额），15 项测试 + 活引擎 `released` 记录。

**剩余**：把预算池的 open/close 与 EngineCore 的请求生命周期在**多卡**下实测一遍。

要点：
1. EngineCore 里为每个活动请求建一个 kvmem 会话（`MemorySession`），共享一个 H 池；
2. 预算分配：按 `sessions` 均分或先到先得，**超发必须报错而不是静默**；
3. TP 各 rank 执行**同一条 plan**（已有确定性策略 + 指纹校验，见 `src/policy_store.py`）；
4. 释放/还原按对齐块（1,616 token = 101 页）操作，见 `src/kvmem_aligned_blocks.py`。

## 已知的坑（别再踩）

1. **重选触发必须相对"上次重选"计**，不能相对固定量——否则长上下文时每步都重选（既慢又会让
   埋针观测失败）。见 NOTES 第 40 节。
2. `plan_working_set` 的校验很严：完整 identity、每层 Key+Value plane、三个能力位、
   stamp 各 revision 非零、catalog 首尾相接、冷块要有当前主机副本。
3. pybind 的 `def_readwrite` 对 `std::vector` 返回副本，容器必须整表赋值。
4. 引擎页号不从 0 开始（NULL 块占 101 页）；一个逻辑页跨 17 个注意力层。
5. 每次重选必须整窗重打 RoPE（设备搬运/链式旋转都不可行，见 NOTES 第 35 节）。
6. 改代码才重启引擎；否则用 `scripts/ensure_engine.sh` 判断复用。

## 未闭合项：99K 端到端答案

症状与定位方法见 `NOTES` 第 45 节：token 计数停在 122,888、重选在同一 frontier 重复触发、
埋针全过且 0 错误。先确认是"计数不同步"还是"请求真的卡住"。

## 复现命令

```bash
# 单请求（kvmem CLI 入口）
.venv/bin/python scripts/kvmem-vllm-worker.py --model <path> --kvmem-budget 32768 \
  --kvmem-gen-reserve 16384 --kvmem-sink-tokens 8192 --kvmem-recent-tokens 1024 \
  --kvmem-host-mib 16384 --kvmem-sessions 4 --no-kvmem-lrfu --max-model-len 204800

# 4 并发
bash deploy/run-kvmem-engine.sh --port 8002

# 全套 CPU 回归
for t in tests/*_test.py; do .venv/bin/python "$t" | tail -1; done
```

## 升级后的自检顺序（vLLM 或 kvmem 换版本时）

1. `python scripts/check_integration_surface.py` —— 24 个触点；缺哪个就知道改哪几行
2. `cd build && cmake .. && make -j4 && ctest` —— contract 是否仍成立
3. `python tests/retrieval_mode_test.py`、`tests/host_store_test.py`、`tests/disk_tier_test.py`
   （磁盘层：`PYTHONPATH=build/python:src python /tmp/disk_tier_test.py`）
4. 起引擎后看 `/tmp/kvmem_sparse.jsonl`：
   - `{"kind":"config_source","source":"additional_config"}` —— 配置通道正常
   - `{"kind":"bridge_ensure","ok":true}` —— 查询桥接缓冲已分配
   - `{"kind":"reselect","scores":{...}}` —— 检索出分（不是 `error`）
   - 没有 `capture_attach_error` / `after_step_error` / `reselect_error`

### 最容易踩的三个坑（已在 INTEGRATION-SURFACE.md 附录列全）

- 编译区里不能调 builtin（`open`/`print`）→ `Failed to trace builtin operator`
- 采集算子内不能 host 同步（`.item()`）→ `CUDA error ... during capture`
- 不要在插件里替换 `_project_qkv_gate` → 会把 fork 里的采集算子挡在图外（`op_calls == 0`）
