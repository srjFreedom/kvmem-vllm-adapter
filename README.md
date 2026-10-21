# vLLM (TP=2) × KVMem MemoryBackend Adapter

An out-of-tree graft that bolts KVMem's decoupled memory-swap onto vLLM (the weicj SM75 fork) to
extend context beyond the GPU KV budget. vLLM stays the engine (a black box); KVMem lives in a
separate black box that handles vLLM's KV. The trade is a little performance for the union of
both strengths.

## Architecture

- **Per-rank self-contained KVMem units.** Each TP rank owns its own host pool + deterministic
  policy store + `MemoryBackend`/`MemoryTransfer`/`MemorySession`. Cross-rank zero sharing/IPC;
  lockstep by determinism (the policy is a pure function of the conversation state, so both ranks
  make identical decisions).
- **Sync/async switch.** `KVMEM_ASYNC_TRANSFER=1` selects the async arm: each block is staged
  through a pinned buffer and copied on a torch-owned transfer stream, so the D2H/H2D overlaps
  compute; `poll()` harvests at step boundaries and moves the pinned bytes into the retained
  host record. Unset/0 keeps the blocking arm. The KV tensor is not in the TP allreduce, so a
  KV-only transfer stream is deadlock-safe.
- **Real per-rank Python handle.** `VllmKvHandleVllm` wraps the engine's `kv_caches` (4D
  `(B,H,N,C)` per layer) + block table; spills/restores blocks via per-layer torch copies.

## Build

```bash
cd /home/srj/vllm-kvmem-adapter
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j$(nproc)
```

The extension lands at `build/python/vllm_kvmem_ext.cpython-312-x86_64-linux-gnu.so`.

## CPU-side tests (no GPU required)

```bash
PY=/home/srj/vLLM-2080Ti-Definitive/.venv/bin/python
./build/vllm_contract_test                          # C++ contract (sync arm)
$PY tests/vllm_trampoline_test.py                    # pybind11 boundary
$PY tests/vllm_real_handle_test.py                   # real handle copy logic
$PY tests/policy_store_test.py                       # deterministic policy
$PY tests/tp2_lockstep_test.py                       # TP=2 lockstep
$PY tests/correctness_test.py                        # 埋针判活 byte-identical
$PY tests/integration_hook_test.py                   # factory hook -> policy -> publish
$PY tests/sparse_rope_test.py                        # delta re-RoPE vs vLLM's own RoPE
$PY tests/sparse_rope_fp8_test.py                    # fp8 re-quantization cost
$PY tests/sparse_view_test.py                        # sparse-view contract
$PY tests/sparse_orchestrator_test.py                # 200K sparse end-to-end (offline)
$PY tests/sparse_runtime_test.py                     # live wiring contract (offline)
$PY tests/live_view_test.py                          # V2 runner adapter (offline)
```

All seven must print PASS.

## GPU-mode tests (requires the serving vLLM stopped)

Stop the serving vLLM first (a delayed background job, `sleep 10–20`), then:

```bash
cd /home/srj/vllm-kvmem-adapter
./run-gpu-tests.sh            # real-tensor async arm + A/B + capacity
```

The script refuses to run (exit 1) if any compute process still holds the GPUs.

## Live-engine mount (validated)

The out-of-tree plugin `plugin/vllm_kvmem_plugin.py` is installed into the vLLM venv under the
`vllm.general_plugins` entry point, so vLLM's `load_general_plugins()` loads it in the engine core
and in every TP worker. Configuration is read from a FILE (`/tmp/kvmem_plugin.json`, see
`scripts/kvmem_plugin.example.json`) because vLLM spawns those processes with a filtered
environment. With no config file the plugin is a no-op.

Grounding the real engine overturned three assumptions:

1. **The fork runs the V2 model runner** (`vllm/v1/worker/gpu/model_runner.py`), not the V1
   `gpu_model_runner.py`, so both classes must be patched, and block tables hang off
   `runner.block_tables` (entries expose `.gpu`) rather than `runner.input_batch`.
2. **`kv_caches` mixes the GDN state and the attention KV** (65 tensors vs 17 paged attention
   tensors). Only the `FullAttentionSpec` groups are spillable token payload; feeding a Mamba
   state tensor into the handle fails with a shape mismatch.
3. **`mamba_cache_mode=align` allocates attention in 1616-token blocks, but the block table
   stores PAGE ids** in `[0, pages)` (8686 pages here). So the KVMem block unit is one 16-token
   page, not the aligned block.

Measured geometry on the live engine: 17 paged layers, tensor `(8686, 2, 16, 512)` uint8,
2 KV heads, head_dim 256, 278,528 bytes per block.

Live needle result (both TP ranks, real KV, real block evicted to the host pool and restored):

```
rank pid=165810: evicted=[1616] restored_blocks=102 bytes_identical=True -> OK
rank pid=165809: evicted=[1616] restored_blocks=102 bytes_identical=True -> OK
```

Evidence: `smoke_live_engine.json`; checks: `scripts/live_needle_check.py`.

## Files

| Path | Role |
|---|---|
| `src/vllm_backend.cpp` / `include/vllm_kvmem/vllm_backend.hpp` | `VllmMemoryBackend` + sync/async transfer selection |
| `src/async_page_pipeline.cpp` | `VllmAsyncTransfer` (CUDA transfer stream MemoryTransfer) |
| `src/vllm_handle.py` | `VllmKvHandleVllm` (real 5D paged KV, per-layer D2H/H2D) |
| `src/vllm_integration.py` | factory hook + `on_execute_model_done` (capture-boundary skip) |
| `src/policy_store.py` | deterministic policy (pure function of conversation state) |
| `tests/integration_hook_test.py` | factory hook end-to-end (fake runner, CPU) |
| `src/vllm_kvmem_ext.cpp` | pybind11 binding |
| `bench_ab_capacity.py` | sync-vs-async A/B + 256K-in-64K-VRAM capacity |
| `run-gpu-tests.sh` | one-command GPU-mode verification (with GPU guard) |
| `plugin/vllm_kvmem_plugin.py` | out-of-tree vLLM plugin (probe + adapter hook, both runners) |
| `scripts/kvmem_plugin.example.json` | plugin config template |
| `scripts/live_needle_check.py` | asserts the live-engine needle result |
| `src/kvmem_sparse_*.py`, `src/vllm_sparse_selector.cpp` | the sparse-selection (A mode) path |
| `tests/` | the six CPU-side tests + the GPU-mode real-tensor test |

## 快速开始（kvmem 入口，vLLM 后端）

```bash
# 与 kvmem-llamacpp / kvmem-ninfer 完全一致的启动方式，只换 -Backend
python vllm-kvmem-adapter/backends/vllm/worker.py \
  -Backend vllm -Model /path/to/model -Port 8001 \
  -KvMemBudget 32768 -KvMemGenReserve 16384 \
  -KvMemSinkTokens 8192 -KvMemRecentTokens 1024 \
  -KvMemSessions 4 -KvMemHostMiB 16384 \
  -KvMemDiskMiB 8192 -KvMemDiskPath /var/tmp/kvmem-tier \
  -KvMemBlockTokens 128 -KvMemRetrieval
```

`--kvmem-*` 参数会序列化成 vLLM 官方 `--additional-config`（随配置下发到每个 TP worker），
**没有 `/tmp/kvmem_plugin.json`、没有环境变量旁路**。

### 两种选片模式

| 模式 | 开关 | 选片依据 | 性能 |
|---|---|---|---|
| recency（默认） | 无 | sink + recent + LRFU | 原始编译路径，≈87 tok/s |
| **retrieval** | `--kvmem-retrieval` | **mean-K 余弦**（sink/recent 仍强制保留） | **同一路径，≈87 tok/s** |

retrieval 模式在 `reselect` 记录里会带真实分数：

```json
{"kind":"reselect","tokens":49159,
 "scores":{"blocks":647,"total_blocks":673,"top":0.1743}}
```

recency 模式同一字段为 `null`——这就是两种模式选片差异的可观测证据。

### 三层内存

```
H  （--kvmem-host-mib）  HostKvStorage，按会话均分（--kvmem-sessions）
D  （--kvmem-disk-mib）  nvme_kv_tier，H 耗尽时溢出，读取自动取回
显存                      vLLM 自己管理，budget_tokens 控制常驻页
```

实测（H 故意压到 64 MiB 以强制溢出）：`needles 22/22`，而 H=4096 MiB（磁盘层未触发）时 `22/28`。

### 并发：vLLM 调度，kvmem 控制上下文

| 参数 | 归属 | 取值 |
|---|---|---|
| `MAX_NUM_SEQS` | vLLM | 期望并发数 N |
| `--max-model-len` | vLLM | **N × 单请求上下文**（放心设大） |
| `--kvmem-budget` + `--kvmem-gen-reserve` | kvmem | 单请求真实驻留上限（与 N 无关） |
| `--kvmem-host-mib` / `--kvmem-disk-mib` | kvmem | 全部请求**共享**的页缓存 |

2 并发示例：

```bash
python vllm-kvmem-adapter/backends/vllm/worker.py -Backend vllm -Model /path/to/model -Port 8001 \
  -KvMemBudget 32768 -KvMemGenReserve 16384 -KvMemSinkTokens 8192 -KvMemRecentTokens 1024 \
  -KvMemSessions 4 -KvMemHostMiB 16384 -KvMemDiskMiB 8192 -KvMemDiskPath /var/tmp/kvmem-tier \
  -KvMemRetrieval --max-model-len 409600        # = 2 x 单请求上下文
# 并在启动脚本里放开 MAX_NUM_SEQS=2
```

每个请求的实际工作集被 kvmem 压到 B+R，聚合驻留由共享的 H(+D) 兜住，
所以 `max_model_len` 设成 N 倍也不会真的占满显存。

### 并发与调度的对应关系（重要）

| `MAX_NUM_SEQS` | 调度器 | decode（882+300） | 说明 |
|---|---|---|---|
| **1（默认）** | async（默认） | **3.38–3.45 s ≈ 87 tok/s** | 单请求最快路径 |
| ≥2 | **强制 `--no-async-scheduling`** | 4.1–4.5 s ≈ 72 tok/s | 并发必须同步调度 |

原因：kvmem 会给被压缩的请求**重写 block table**；vLLM 的异步调度在当前步输出尚未结算前就规划下一步，
改写会触发 `assert request.num_output_placeholders >= 0`（`async_scheduler.py:63`）并让引擎退出。
同步调度让每一步自洽，正是 kvmem 需要的不变量。worker 在 `MAX_NUM_SEQS>1` 时**自动**关掉异步调度。

实测（`MAX_NUM_SEQS=2` 与 `=4`，`--max-model-len 409600`，retrieval 开）：

```
2 并发：两条都返回 200（8544 + 16 token，约 20 s）
4 并发：四条都返回 200（8544 + 14/15/15/7 token，约 35 s）
```

### 默认配置验证矩阵（2026-10-21，单脚本一次跑完）

| 用例 | 结果 |
|---|---|
| 单请求 37K | ✅ 200 / 36 s |
| N=2 × 37K | ✅ 2/2 / 69 s |
| N=4 × 13K | ✅ 4/4 / 43 s |
| 断言 | 0 |

默认**不做生成期压缩**（`--kvmem-inrequest-compaction` 关闭），走 vLLM 原始路径；
`MAX_NUM_SEQS>1` 时 worker 自动加 `--no-async-scheduling`。

### 已验证的头号能力（2026-10-21）

单请求 **97,635 token** 上下文 + 埋针提问 → **HTTP 200**，答案 `EPVHKZ CPBL7Q`（两条 key 逐字正确），
零断言。默认配置即可服务 99K（稠密 KV ≈ 212 MB/rank）；生成期压缩是更大上下文才需要的可选手段。

### 入口层选页（推荐的长上下文压缩方式）

KV 层压缩与 vLLM 的记账冲突（生成期截断会导致断言/挂住），所以压缩在**入口**做：

```bash
python scripts/kvmem_gateway.py --upstream http://127.0.0.1:8001 --port 8100 \
    --kvmem-budget 16384 --kvmem-sink-tokens 256 --kvmem-recent-tokens 1024
```

实测（99K 埋针提示，问 record 250 / 4100 的 key）：

| 指标 | 直连 | 经网关 |
|---|---|---|
| prompt token | 97,635 | **21,635** |
| 耗时 | 112 s | **21 s** |
| 答案 | `EPVHKZ CPBL7Q` | **`EPVHKZ CPBL7Q`**（一致） |

### 组合验证：N=2 并发 × 99K 埋针（经网关）

| | prompt | 耗时 | 答案 |
|---|---|---|---|
| req1 | 21,635 | \ | `EPVHKZ CPBL7Q` ✅ |
| req2 | 21,635 | 42 s（合计） | `EPVHKZ CPBL7Q` ✅ |
| 直连（无选页） | 97,635 | 112 s | 同上 |

### R 的语义（重要）

`--kvmem-gen-reserve R` 就是 kvmem 给请求的 **`max_tokens` cap**：最多生成 R 个 token，
模型输出 EOS 即停（所以短回答不会被拖长）。
"R 满后挪进 budget 继续生成"是 kvmem 的实验功能，不推荐，本后端默认关闭。

| 客户端给的 `max_tokens` | 网关写入 |
|---|---|
| 超过 R / 没给 | **R** |
| 小于 R | 客户端的值 |

```bash
python scripts/kvmem_gateway.py --upstream http://127.0.0.1:8001 --port 8100 \
    --kvmem-budget 16384 --kvmem-gen-reserve 8192
# 客户端要 20000 → 实际 8192（finish_reason=length）；没给 → 8192
```

常驻上限 = **B + R**；生成不越过 R，序列长度单调，所以 vLLM 不需要任何生成期截断。

### 逻辑上下文 vs 模型窗口（重要）

kvmem 的 `CTX`（可保留历史总量）**可以任意大**，它只是 KV 的缓存层；真正进模型的是 **B + R**。

| 概念 | 约束 |
|---|---|
| 逻辑上下文（KVMem） | 不受模型窗口限制（H/D 容量决定） |
| B + R（每请求进模型） | ≤ 模型原生窗口 |
| vLLM `--max-model-len` | ≥ B+R，≤ 模型 `max_position_embeddings` |

实测：**287K token 的历史**（12,001 条记录）直连引擎被 400 拒绝（超 204,800 窗口），
经入口网关（B=65536）→ `prompt 89,009`，答案 `XM9LE6 N553Y8`，两条埋针全中。

### 引擎窗口怎么设：`max_model_len = N × (B + R)`

vLLM 要求 KV 池装得下**一条完整的 `max_model_len` 序列**；N 路并发真正需要的是 `N × (B+R)`。
把窗口设成这个乘积，两个要求正好相等——不多占，也够用。worker 现在**自动**这么做：

```bash
# 不传 --max-model-len 时：
[kvmem backends/vllm] max_model_len auto = 2 x (65536 + 32768) = 196608
```

本机实测边界：`409600` 被模型窗口拒绝（>262144）；`262144` 因 KV 池不足启动失败
（需 4.66 GiB / 有 3.82 GiB，vLLM 自算上限 ≈ 210080）；**`196608` 正常启动**。

逻辑上下文不受这个窗口约束：287K token 的历史经网关仍能正确作答（只有 B+R 进模型）。

### 单服务 / 单端口（kvmem 官方 18200）

入口策略（选 B、`max_tokens=R`、CTX 上限、H/D 历史块缓存）跑在 **vLLM 的 API server 进程内**，
通过官方 `vllm.endpoint_plugins` 扩展点加载（策略作用在解析后的 `ChatCompletionRequest` 上），**没有旁挂网关**，vLLM 源码零改动，流式/CORS 保持原生。

```bash
sudo systemctl start kvmem-vllm          # 一条命令，一个端口
scripts/wait_ready.sh 18200 900 --smoke  # 就绪检查：短轮询 + 日志触发 + 失败快退
```

- 端点：`http://<ip>:18200/v1`（模型 id 见 `/v1/models`）
- 日志：`tail -f /home/srj/kvmem.log`（入口策略与引擎同一份日志，`entry_rewrite` 可见）
- 旧网关 `scripts/kvmem_gateway.py` 保留为 A/B 工具（默认不启用，systemd 单元已归档）

| 参数 | 默认 | 含义 |
|---|---|---|
| `--kvmem-capture-tokens` / `KVMEM_CAPTURE_TOKENS` | **512** | 每步为检索捕获的 query 行数（对齐 KVMem `query_max_tokens` 的 512 默认）；每 rank 约 100–200 MiB |
