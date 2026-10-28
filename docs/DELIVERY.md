# kvmem-vllm 适配器交付说明（v0.19 对齐）

更新：2026-10-29

## 一、这一版是什么

把 vLLM（SM75 fork）接成 KVMem 的一个后端，并把**打分/统计/检索语义**换成 kvmem v0.19 核心
自己的实现，而不是适配器自造的 Python 版本。

| 关注点 | 现在的做法 | 依据 |
|---|---|---|
| mean-K 统计 | 每请求一个 `kvmem::RawKvStore`（`n_layer=16`、`n_embd_k=1024/rank`、`block_tokens=64`），由模型里的 `capture_k` 设备侧按 64-token 桶累加 pre-RoPE K | `backends/ninfer/.../memory_statistics.cpp` |
| Query | `capture_q` 在设备上反旋转，直接产出 **pre-RoPE** 行；`PageQueryView` 中缺席层为 `layer_rows=0`（不是零向量） | `PageQueryView` 注释 |
| 打分 | 上游默认 **page-softmax**（sm75 CUDA，逐层 ≤64MiB），mean 聚合，LRFU **关** | `cli_options.h: resolve_query_scoring()` |
| 预算带 | sink/recent/budget 以 token 发布，recall 按 store 粒度换算成块 | `query_score_frame` 的入参要求 |
| 压缩 | **只改地址**：物理页重排、槽位重算，K 字节原样拷贝、不重烘 RoPE | "keeps original RoPE positions; compact addresses do not rewrite K" |
| 位置 | RoPE 用真实位置、`slot_mapping` 用压缩位置（`kvmem_rope_offset` / `slot_positions`） | 本次实现 |
| 旧 Python 索引 | **默认不再参与排序**（仅审计开关下读取） | 本次实现 |

## 二、已实测通过

| 项 | 证据 |
|---|---|
| 统计精确性 | 53,422 / 62,347 / 88,256 token prompt：`frontier` 精确等于 prompt 长度；1024 桶写/4K chunk |
| 长上下文稳定性 | 88K prompt + 统计开启：HTTP 200、答案正确、引擎存活（此前 pinned allocator 崩溃，已修） |
| page-softmax | 828 块、`sum=512`、device 1.16 MiB、无非有限值 |
| 打分器一致性（审计） | 与旧索引同网格对齐后 `rank_corr 0.69–0.77`、`top_overlap 4–6/10` |
| 压缩三本账 | 58K prompt 连续 3 次压缩：`49152→36864`、`46371→36864`、`36865→36864`，`2304 页×16 = 36864 = visible` |
| 位置拆分 | RoPE/写地址分离后 88K 常驻 needle 居中答对 |
| 回归基线 | `B ≥ prompt` + 统计开 + page-softmax：88,256 token needle 居中 → `ZX-4417` |

## 三、未完成 / 已知限制

1. **超池（prompt > 池 177K）尚未打通**：prefill 期压缩已不崩、账本自洽，但**选择质量**不足
   （prefill 期没有 query，只能 sink+newest），需要 QueryReplay。
2. **QueryReplay 部分实现、未收敛**：通道消息、两本账回退、循环状态快照+恢复都已就位，但
   chunk 粒度的快照不足以精确重放（答案仍是 `ZX` + 续写）。需要按任意位置切分 prefill 的
   checkpoint（上游 `prefix_checkpoint.hpp` 的 QueryReplay 语义）。
3. **decode 期统计**：机制已实现（bridge 保留解码行、按上一接受计数折叠），但端到端账目仍在核对
   （存在少计，见 `docs/BEYOND-POOL-PLAN.md`）。
4. **入口策略仍在 API server 做文本式选择**（`--kvmem-entry-select retrieval`）。按 v0.19 设计，
   选择应全部发生在引擎内；这是"中间内容被丢"的历史根源之一。
5. graph recall / AttentionFeedback（`graph_recall.hpp`）未接入——需要先有注意力质量采样。

## 四、运行方式（已验证可用的配置）

```bash
B=36864|131072  INREQUEST_COMPACTION=0|1  KVMEM_STATS=core  KVMEM_FORCE_EAGER=1 \
ADAPTER=/home/srj/src/kvmem-vllm-adapter \
VLLM_DIR=/home/srj/src/kvmem-vllm-adapter/engines/vllm \
PORT=18200 ./deploy/run-kvmem-vllm.sh
```

启动日志检查三行：`Spec decode: mtp/4`、`lrfu=False score=mean`、
`KVMEM_ENTRY_RESIDENT_PREFILL=1 -> MAX_MODEL_LEN auto`。

## 五、调试设施（长期保留）

| 工具 | 用法 |
|---|---|
| 断点 | 写 `/tmp/kvmem_dbg.bp`，console 里 `import kvmem_dbg; kvmem_dbg.install()`；命中整帧变量落 `/tmp/kvmem_dbg.out` |
| 实时执行 | 写 `/tmp/kvmem_console.py`，引擎下一步在 worker 内执行；可读 `runtime/orch/stats/bridge/runner` |
| 热重载 | `touch /tmp/kvmem_reload.request` → 下一个 prompt 边界 reload 并 rebind（纯 Python 改动免重启） |
| 探针 | `/tmp/kvmem_sparse.jsonl`（`book`/`reselect`/`compact_applied`/`statistics_final`/`core_scores`/`scores_unavailable` …） |
