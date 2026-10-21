# 升级时的对接面（跑一条命令就知道要改哪几行）

```bash
.venv/bin/python scripts/check_integration_surface.py      # 24/24 全部在位
```

升级 kvmem 或 vLLM 之后先跑它：缺哪项、在哪个文件、对应哪条机制，一目了然。

## A. 我们只"使用"的 vLLM 官方扩展点（优先，改动风险最低）

| 依赖 | 用途 | 变了会怎样 |
|---|---|---|
| entry point group `vllm.general_plugins` | 插件装载入口（EngineCore + 每个 worker） | 插件不加载：日志里没有 `kvmem_boot` |
| `VllmConfig.additional_config` | kvmem 参数随配置分发到每个 TP worker | 参数到不了 worker：`config_source` 缺失 |
| `launcher.sh` 的 `ADDITIONAL_CONFIG_JSON` | 把 `{"kvmem":{...}}` 变成 `--additional-config` | worker 里的 `_config()` 为空 → 稀疏不生效 |
| `FullAttentionSpec.build(...)` | 只给注意力组喂压缩后的 `seq_lens`（GDN 组保持真实长度） | 输出变乱码（GDN 状态索引错位） |
| 注意力 kernel 读 `seq_lens` | 压缩视图的长度来源 | 同上 |

## B. 我们"monkey-patch"的 vLLM 内部（必须登记，升级重点看这里）

| 依赖 | 位置 | 用途 | 缺了会怎样 |
|---|---|---|---|
| `GPUModelRunner.execute_model` | `v1/worker/gpu/model_runner.py`（V2）或 `v1/worker/gpu_model_runner.py`（V1） | 每步的稀疏钩子 | 完全不起作用，退化为普通引擎 |
| `GPUModelRunner.prepare_pos_seq_lens` | 同上 | 覆盖 positions/seq_lens（压缩视图落点） | 行改写后注意力读错位置 |
| `GPUModelRunner.vllm_config` | 同上 | 读 `additional_config["kvmem"]` | 配置为空 |
| `Scheduler.schedule` | `v1/core/sched/scheduler.py` | 分配侧释放（对齐块）/ 块通道 / 挂探针 | 释放与还原失效 |
| `block_pool.get_new_blocks` / `get_num_free_blocks` | `v1/core/block_pool.py` | 取回块 / 池压力信号 | 还原与压力触发失效 |
| `_remove_blocks_in_range` | `v1/core/single_type_kv_cache_manager.py` | 把中段对齐块还给池子 | 释放失效 |
| `KVCacheManager.req_to_blocks` | `v1/core/kv_cache_manager.py` | 逐请求的块清单 | 释放判定失效 |

## C. 我们"链接"的 kvmem 契约（跨版本稳定性最高）

| 依赖 | 头文件 | 用途 |
|---|---|---|
| `MemoryBackend` / `MemoryTransfer` / `MemorySession` | `memory_contract.hpp` | 后端契约（spill/restore/view） |
| `plan_working_set` / `WorkingSetPlan` / `ExecutionView` | `memory_contract.hpp` | 工作集与压缩视图 |
| `MemorySession::prepare_reselect` | `memory_contract.hpp` | 让 kvmem 自己选片 |
| `capabilities.pre_rope_statistics` | `memory_contract.hpp` | 检索模式的能力位（第 ③ 项要打开） |
| `HostKvStorage` / `HostKvRecord` | `host_kv_storage.hpp` | RAM 分层与 H 预算 |
| `KvMemStore`（`pick_topk_blocks`） | `kvmem_store.hpp` | 选片算法 |
| `mean_key_retrieval_scores` | `mean_key_retrieval.hpp` | mean-K 检索打分（第 ③ 项） |
| `nvme_kv_tier` | `nvme_kv_tier.hpp` | 磁盘层（第 ② 项） |
| `--kvmem-*` 定义 | `cli_options.h` | 参数面 |
| `kvmem_capture_*`（参考实现） | `src/adapter/llama-kvmem-capture.h` | pre-RoPE 采集的做法（第 ③ 项照着写） |

## D. 升级后的标准动作

1. `check_integration_surface.py` → 看缺哪项；
2. 缺的是 A 类：改 `plugin/vllm_kvmem_plugin.py` 里对应的读取（通常 1–3 行）；
3. 缺的是 B 类：改对应的 patch 函数；若 vLLM 把该函数改名/挪位置，插件里的 `runners` 列表加一行即可；
4. 缺的是 C 类：改 `src/*.cpp` 里的绑定（编一次 `vllm_kvmem_ext`）；
5. 全绿后跑 `tests/` 与 `scripts/free_loop_check.sh` 做回归。

## 附录：pre-RoPE 采集（③）在 SM75 fork 上的六条硬约束

这一节全部来自实测失败，**每条都对应一个已验证的报错**，升级 vLLM 后按顺序自检。

| # | 约束 | 违反后的症状 |
|---|---|---|
| 1 | 采集必须在**模型内**（`qwen3_next.py` 的 `_project_qkv_gate`），不能在插件里替换该方法 | 包装函数把源码里的算子调用挡在图外 → `op_calls == 0` |
| 2 | 编译区（dynamo fullgraph 追踪范围）**不得调用 builtin**（`open`/`print` 等） | `RuntimeError: Failed to trace builtin operator` |
| 3 | 采集算子**不得做 host 同步**（`.item()`、打印张量值） | `CUDA error: operation failed due to a previous error during capture` |
| 4 | 缓冲必须**图外分配、每进程只分配一次** | 重分配后 kernel 仍写旧指针 → 查询静默消失 |
| 5 | 算子必须声明 `mutates_args` 或返回张量 | inductor 判为纯函数并删除 → `op_calls == 0` |
| 6 | 每步推进的钩子只能是 **`execute_model` 入口**（V2 runner 无 `prepare_pos_seq_lens`，也不按步调 `sample_tokens`） | 计数恒为 0、重选永不触发 |

另有三条数据形状约束：

- `q` 传入是**扁平** `[tokens, heads*head_dim]`，缓冲是 `[tokens, heads, dim]`，需 reshape
- `positions` 是 **M-RoPE `[3, tokens]`**，取第 0 行
- `set_block_scores` 要求**每个 selection block 一个分数**（`store_.block_count()`），未评分块填 `-inf`

验证成功的判据（当前状态）：

```
bridge_harvest: {'op_calls': 128, 'armed': True, 'counts': {0: 1}}
reselect.scores: {'blocks': 391, 'total_blocks': 417, 'top': 0.1312}
```

即：查询由 fork 内的 `torch.ops.kvmem.capture_q` 采到，K 由页字节在 host 侧逆旋转得到，
两者在 `KvmemRetrievalIndex` 里做 mean-K 余弦，交 `set_block_scores` 参与选片。

## 入口策略：官方端点插件扩展点（2026-10-21）

入口压缩（选 B、`max_tokens=R`、CTX 上限、H/D 历史块缓存）跑在 **vLLM 的 API server 进程内**，
不再有旁挂网关。它只依赖下面这些**官方扩展点**，vLLM 源码零改动：

| 接触点 | 位置 | 症状（若升级后失效） |
|---|---|---|
| `vllm.endpoint_plugins` 入口点组 | `vllm/plugins/__init__.py` | 入口策略完全不生效：`kvmem.log` 里没有 `kvmem entry policy` 那行 |
| `load_endpoint_plugins()` 的 gating | 同上：仅当 `VLLM_PLUGINS` 点名才加载 | 同上（最容易踩：`VLLM_PLUGINS` 漏了 `kvmem_entry`） |
| `EndpointPlugin` 协议（`name` / `required_tasks` / `attach_router` / `init_state`） | `vllm/plugins/endpoint_plugins/interface.py` | 导入即报错，引擎起不来 |
| `attach_endpoint_plugins(app, tasks)` 在 `build_app()` 内调用 | `vllm/entrypoints/launchers/app.py` | 插件不被调用：策略不生效 |
| `OpenAIServingChat.create_chat_completion(request, raw_request)` | `vllm/entrypoints/openai/chat_completion/serving.py` | 包装点消失：策略不再作用于解析后的请求 |
| 调用顺序：`attach_endpoint_plugins` 先、`init_entrypoints_middleware`（CORS）后 | 同上 | 顺序反转会让 CORS 落到我们内层，预检可能被我们拦到 |

### 改 body 的正确姿势（实测对比）

| 做法 | 结果 |
|---|---|
| 纯 ASGI：`await receive()` 读完，再给下游一个**新的 receive** 重放 | **挂**：中间件打出 `calling downstream`、handler 也进了，响应永不返回（关掉中间件同一请求 0.24 s 完成） |
| 标准 Starlette：`BaseHTTPMiddleware` + `await request.body()` 后把新字节写回 `request._body` | **可用**：handler 解析到改写后的 body，流式 0.07 s 首事件、CORS 正常 |

原因：ASGI `receive` 是一次性流，下游若持有**原来的** `receive`（Starlette 的 `_CachedRequest`
就是这样），换流后它仍去读已抽干的流 → 永远 await。写回 `_body` 缓存则让下游命中缓存，根本不碰流。

两种模式都保留：`KVMEM_ENTRY_MODE=middleware`（默认，标准路径）/ `hook`（解析后请求上的兜底钩子，
包装 `OpenAIServingChat.create_chat_completion`）。

### 为什么不用 shadow 路由

`attach_router` 允许注册同路径路由来覆盖核心 handler（"routes attached here can shadow core
routes"），但那要重新实现整个 chat handler。我们只在请求进入核心路由前改 body，**响应侧完全不碰**，
因此流式（SSE）、CORS、错误路径全部保持 vLLM 原生行为。

### vLLM 侧改动登记表

| 改动 | 行数 | 开关 | 说明 |
|---|---|---|---|
| `vllm/kvmem_bridge.py`（新增） | 新文件 | `KVMEM_CAPTURE=1`（检索时） | pre-RoPE 采集自定义算子 |
| `vllm/model_executor/models/qwen3_next.py` | **8 行** | 同上（无条件调用，算子内部按开关决定是否写入） | 在 `_project_qkv_gate` 返回前调用 `torch.ops.kvmem.capture_q` |
| 入口策略（默认 middleware 模式） | **0 行** | `VLLM_PLUGINS` 是否含 `kvmem_entry` + `KVMEM_ENTRY_MODE` | 官方端点插件 + Starlette 标准 body 改写（`request._body` 缓存） |
| 入口策略（hook 兜底模式） | **0 行**（插件内 monkey-patch 不算 vLLM 源码） | `KVMEM_ENTRY_MODE=hook` | 包装 `OpenAIServingChat.create_chat_completion` |
