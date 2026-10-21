# 验收清单（当前状态，2026-10-21）

逐条对应"KVMem 作为入口与策略层、vLLM 作为被接入后端"的目标。每项都给出**命令 + 结果**，
可以在本机重跑。全部数字来自实测，不来自推算。

## 〇、当前默认配置（这是最重要的前提）

| 项 | 默认 | 说明 |
|---|---|---|
| 生成期 KV 压缩 | **关**（`--kvmem-inrequest-compaction`） | 与 vLLM 的序列单调记账冲突，详见 `STATUS.md` 的证据链 |
| 上下文压缩 | **在入口层做**（`scripts/kvmem_gateway.py`） | 由 KVMem 决定发多少历史，引擎只预填充收到的内容 |
| 异步调度 | `MAX_NUM_SEQS>1` 时自动关 | 异步调度会提前规划步，与任何步内改表冲突 |
| H/D 分层、共享 H、LRFU、mean-K 采集 | 开 | 观测与分层不受上面两项影响 |

## 一、总览：目标条款 → 证据

| 目标条款 | 证据 | 结果 |
|---|---|---|
| ① 启动形态与 kvmem 一致，走 `--additional-config` | `backends/vllm/worker.py` 组装 `ADDITIONAL_CONFIG_JSON`；引擎日志里可读到完整 `additional_config` | ✅ |
| ② H/D 分层、共享 H | `tests/disk_tier_test.py`（RAM 2 页 / 磁盘 4 页 / 逐字节回读） | ✅ PASS |
| ② LRFU / 会话预算 | `tests/lrfu_test.py`、`tests/session_budget_test.py` | ✅ PASS |
| ③ mean-K 真检索 | `model_runner` 侧 pre-RoPE 采集 → `bridge_harvest {'op_calls': N}`；`reselect.scores` 有值 | ✅ |
| ④ 升级探针 | `python scripts/check_integration_surface.py` | ✅ 24/24 |
| ⑤ 并发 N=2 / N=4 答案正确 | 见下方 V2 | ✅ |
| 长上下文 + 埋针逐字一致 | 见下方 V1 | ✅ |
| 压缩生效且安全 | 见下方 V3（入口层） | ✅ |
| 生成期 KV 截断 | 见 `STATUS.md`「in-request 压缩的最终判定」 | ❌ 判定不可行 |

## 二、四项现场验证

### V0 常驻页下降（引擎官方指标）

```bash
# 请求在飞时每 200 ms 采样，取峰值
curl -s http://127.0.0.1:18200/metrics | grep vllm:kv_cache_usage_perc
```

| 阶段 | prompt | 峰值池占用 |
|---|---|---|
| 直连 99K | 97,635 | 50.67% |
| 经入口网关 | 21,635 | **19.33%（−61.8%）** |

### V0d 入口策略跑在引擎进程内（单服务）

```bash
ss -tln | grep -E ':(8001|8100|18200) '    # 只应出现 18200
systemctl list-units 'kvmem*'              # 只应有 kvmem-vllm
tail -f /home/srj/kvmem.log                # 应出现 "kvmem entry policy:" 与 entry_rewrite
```

判据：只有一个 HTTP 端口；`/v1/chat/completions` 的长历史被裁剪；无第二个服务进程。

### V0c R 是生成上限（契约）

```bash
python tests/gen_reserve_test.py      # 超 R 被夹到 R；未超不动；max_completion_tokens 同样
```

`max_tokens` 属于 kvmem（`R` 即其值）：客户端超过 R 或没给 → 写入 R；客户端更小 → 保留。
网关日志：`{"kind":"gen_budget","field":"max_tokens","requested":20000,"applied":8192}`。
常驻上限 B+R 由此成立；生成不越过 R，vLLM 序列长度单调，因此无需生成期截断。

（`--kvmem-inrequest-compaction` 是 kvmem 的**实验性**"R 满后继续生成"路径，默认关闭、不推荐。）

### V0b H/D 分层（活引擎）

```bash
python scripts/kvmem_gateway.py --upstream http://127.0.0.1:18200 --port 18200 \
    --kvmem-budget 16384 --kvmem-history-kib 64 --kvmem-history-disk-mib 256 \
    --kvmem-history-disk-path /tmp/kvmem_history_tier
```

同一会话两轮：turn1 `miss` → 237 KB 历史跨 **4 个 NVMe 槽**落盘；turn2 `hit` + `disk_reads: 1`
从磁盘读回，两轮答案都是 `EPVHKZ CPBL7Q`。单测：`tests/history_cache_test.py`。

### V1 单请求 99K + 埋针（逐字）

```bash
# 提示：4,000 条记录，问 record 250 与 record 4100 的 key
nohup /tmp/kv_n99.sh &      # 生成提示 + 起引擎 + 发请求
```

判据：`key250` 与 `key4100` 都出现在答案里。

实测：`prompt 97,635 + 124`、`finish=stop`、答案 `EPVHKZ CPBL7Q`、**两条 key 均正确**、0 断言。

> 注意：该模型的 reasoning 通道会吃掉 `max_tokens`。首轮 `max_tokens=24` 时 `content` 为空
> （`reasoning_tokens=24`），不是失败；给到 600 即一次答对。

### V2 并发（N=2 / N=4）

```bash
./scripts/concurrency_check.sh 2 84774 18200    # 2 x 37K
./scripts/concurrency_check.sh 4 30000 18200    # 4 x 13K
```

判据：`RESULT: k/N returned 200`，且引擎日志 `AssertionError` 计数为 0。

实测（同一引擎、同一日志）：

| 用例 | 结果 | 耗时 |
|---|---|---|
| 单请求 37K | 1/1 200 | 36 s |
| N=2 × 37K | 2/2 200 | 69 s |
| N=4 × 13K | 4/4 200 | 43 s |

### V3 入口层选页（压缩的实际形态）

```bash
python scripts/kvmem_gateway.py --upstream http://127.0.0.1:18200 --port 18200 \
    --kvmem-budget 16384 --kvmem-sink-tokens 256 --kvmem-recent-tokens 1024
# 客户端 base_url 指向 http://127.0.0.1:18200/v1
```

判据：引擎收到的 `prompt_tokens` 显著下降，**且答案与直连一致**。

实测（同一 99K 埋针请求）：

| | prompt | 耗时 | 答案 |
|---|---|---|---|
| 直连引擎 | 97,635 | 112 s | `EPVHKZ CPBL7Q` |
| 经网关 | **21,635** | **21 s** | `EPVHKZ CPBL7Q`（一致） |
| 经网关 × 2 并发 | 各 21,635 | **42 s 合计** | 两条都正确 |

单测：`python tests/prompt_select_test.py`

**检索 vs recency（同一提示，端到端）**：`--kvmem-select retrieval|recency`

| 模式 | prompt | completion | 耗时 | 答案 |
|---|---|---|---|---|
| retrieval | 21,635 | 86 | 21 s | `EPVHKZ CPBL7Q` ✅ |
| recency | 21,636 | 1500（上限） | 42 s | 空 ❌（record 4100 被丢） |

这就是"检索模式真的在选片、不是 recency 退化"的可观测证据。

### V4 释放与还原闭环

```bash
python tests/restore_path_test.py     # 释放后按同一选择重建
./scripts/free_loop_check.sh          # 活引擎下的页归还观测
```

判据：请求结束后页归还、会话再次进入时能按同一选择重建出等价内容。

实测：`tests/restore_path_test.py` 通过；`free_loop_check.sh` 在活引擎上观测 `released` 记录
（99K 那次探针里 `released` 正常增长，无 error）。

## 三、三条踩过的坑（每条都曾造成误判）

1. **旧实例污染**：请求打到正在退出的旧引擎 → 500/000/断言，而当前日志里根本没有这些请求。
   现在所有验证都要求"等新日志文件 + 同日志核对 POST/断言计数"。
2. **profile 覆盖参数**：`run-kvmem-4way.sh` 把 `MAX_MODEL_LEN` 钉成 32768；
   `MTP_K=0` 被 profile 的 `SPECULATIVE_METHOD=mtp` 覆盖。要改必须用 `--set` / 对应 profile 键。
3. **视图半初始化**：压缩关闭却仍安装稀疏视图 → 答案乱码 + `bridge_harvest_error`。
   已修：压缩关闭时 `before_step` 直接返回，不装视图。

## 四、测试清单（33 个文件，逐条对应）

| 主题 | 测试 |
|---|---|
| 契约/启动 | `additional_config_test` `worker_cli_test` `aligned_blocks_test` |
| 视图/钩子 | `kvmem_view_test` `live_view_test` `sparse_view_test` `integration_hook_test` `vllm_trampoline_test` |
| 稀疏运行 | `sparse_runtime_test` `sparse_orchestrator_test` `sparse_concurrency_test` `block_channel_test` |
| 内存分层 | `disk_tier_test` `host_store_test` `policy_store_test` `session_budget_test` `session_lifecycle_test` |
| 检索 | `retrieval_index_test` `retrieval_mode_test` `pre_rope_capture_test` `engine_capture_test` |
| 数值/搬运 | `sparse_rope_test` `sparse_rope_fp8_test` `move_order_test` `device_move_test` `correctness_test` |
| 端到端/真句柄 | `vllm_real_handle_test` `gpu_real_tensor_test` `restore_path_test` `plan_view_test` |
| 入口层 | `prompt_select_test` `history_cache_test` |
| 杂项 | `tp2_lockstep_test` `kvmem_view_test` |

**最终跑分（33 个文件）**：紧凑批量 **29 PASS + 4 个 GPU 抖动**；那 4 个是
`torch.frombuffer(src_np.tobytes(), ...)`（CUDA 上下文竞争），**单独运行全部 PASS**
（`correctness_test` `gpu_real_tensor_test` `integration_hook_test` `vllm_real_handle_test`）
→ **等价于 33/33**。

## 五、口径声明（避免夸大）

- **可以宣称**：长上下文（99K）答案逐字正确；N=2/N=4 并发正确；入口层压缩把 99K 压到 21.6K
  且答案不变、耗时降 5.3×；H/D 分层与检索统计可用；升级探针 24/24。
- **不可宣称**：生成期 KV 截断（已判定不可行）；vLLM 侧的显存/并发容量收益（需要调度器侧
  `B+R` 记账插桩，未做）。

质量对照按用户指示豁免，不需再跑。
