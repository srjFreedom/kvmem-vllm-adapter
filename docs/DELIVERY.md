# kvmem-vllm 适配器交付说明（v0.4.0）

更新：2026-11-02（**0.4.0**）。第三方引擎只以补丁形式发布：`install.sh` 默认把
`backends/versions.json` 的 `base_commit` 拉到 `engines/vllm`，再应用
`backends/patches/vllm-kvmem.patch` 并校验 `patched_tree`；不会写入用户已有的 fork。

> 本文只记录**当前有效**的形态与实测。历史演进（超池通道的 152K/196K 实验、账本修正、
> 被迫 eager 时代）见 git log 与 `docs/design-mtp-draft-ledger.md`（工程账本，含各
> Amendment 的失败证据链）。

## 摘要

**v0.4.0 = 编译 + CUDA graph + MTP4 + 会话复用的生产形态**，单实例端到端验收全绿：

1. **CUDA-graph 兼容（本版主题）**：打分桥（`capture_q`/`capture_k`）在流捕获期零分配——
   eager-first 命名静态缓冲、全原地写、host probe 移出捕获区；与 eager 数值**逐位一致**
   （离线对拍 max diff 0.0，`tests/bridge_capture_test.py` 13 项守卫）。解码
   **46 → 85.0 tok/s**（2,000-token 倒数，spec=4，参考带 70–80 之上）。
2. **超池正确性默认路径**：入口不裁剪（resident prefill）+ prompt 末尾整体重算
   （`KVMEM_COMPACT_REBUILD=1`，默认开）。88K prompt / 常驻窗 36,864 → needle
   `ZX-4417-QJ7` 答对；3,000 token 生成双跨 1,616 容量边界存活。
3. **会话复用**：`X-Session-ID` → 结束轮在 checkpoint 前沿停放（前缀 KV FP8 原字节 +
   循环状态 + 块哈希链进主机内存），下一轮命中恢复、跳过前缀 prefill（85K 档案
   143 s → 110 s）。停放/恢复/重放事件链日志可查（`retired frontier` →
   `resume_decided hit` → `session_resumed`×2 ranks）。
4. **池几何**：`GPU_UTIL=0.972` → KV 池 201,438 tokens（graph 静态缓冲比 eager 吃约
   2.5K；0.97 = 198,561 < `MAX_MODEL_LEN=200000`，实测 OOM 过一次；曲线
   0.93→152K / 0.95→175K / 0.97→198.6K / 0.972→201.4K）。
5. **无损门（每轮复测）**：短请求 A/B 输出与 eager/spec0 基线逐字节一致
   （sha `c2f3644600da41bb` / `426751992d476adc`）；冷×2 逐字节一致；恢复路径输出
   跨重启、跨 eager/graph、跨桥重写稳定（sha `6f56aadf80e4d4b4`）。

## 一、克隆即可运行

`deploy/run-kvmem-vllm.sh` 的缺省值 = 本验收的生产配置，**无 /etc/default 依赖**：

```
B=36864 R=16384 N=1 CTX=524288 GPU_UTIL=0.972 SELECT_BLOCK_TOKENS=64
SPECULATIVE_TOKENS=4 KVMEM_SESSION_REUSE=1 KVMEM_MAX_MODEL_LEN=200000
KVMEM_COMPACT_REBUILD=1 KVMEM_REBAKE=1 KVMEM_COMPACT_MODE=prefix
KVMEM_BLOCKWISE=1 VLLM_SM75_SPEC_SYNC_MODE=nosync
VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=1
（KVMEM_FORCE_EAGER 不设 = 编译路径；INREQUEST_COMPACTION=1 默认 = 常驻入口 + 请求内压缩）
```

```bash
MODEL=$HOME/models/<FP8+MTP 权重目录> ./deploy/run-kvmem-vllm.sh
scripts/wait_ready.sh 18200 900 --smoke
```

安装三步（打补丁、装插件、建原生扩展）与 README「Quick start」一致。

## 二、验收矩阵（v0.4.0，同一生产实例）

| 项 | 判据 | 结果 |
|---|---|---|
| 编译启动 | CUDA graph capture 成功、无 resize/capture 告警 | ✓（MTP graph policy nosync ×2 ranks）|
| 速度 | 2,000-tok 倒数 ≥65 tok/s | **85.0**（3,000-tok 84.7；冷启首样本 75.5 亦在带内）|
| 无损 A | sha == `c2f3644600da41bb` | ✓ 逐字节 |
| 无损 B | sha == `426751992d476adc` | ✓ 逐字节 |
| 超池 NEEDLE | 88K prompt → `ZX-4417-QJ7` | ✓ 141.7 s |
| 存活 | 3,000 tok 生成跨两次 1,616 边界（b3789d7 两级容量政策） | ✓ 35.4 s |
| 会话复用 | park→retire→resume 链 + 省 prefill + needle 双路一致 | ✓ hit 32,320，省 34 s |
| 确定性 | 冷×2 一致；恢复输出跨实例/编译形态稳定 | ✓（每路径确定） |
| 离线套件 | 核心 9 项全绿（含桥守卫 `bridge_capture`） | ✓ 9/9 |

**恢复 vs 冷跑的字节一致不是契约**（gate7 实测：恢复轮重放停放时的选页
`[7,8,9,10,11,12…]`，冷轮按当前查询重选 `[7,8,9,12,13,14,16…]`，同页数不同集合——
按请求检索语义决定的差异；论证链见 `docs/design-mtp-draft-ledger.md` Amendment 5/5b）。

## 三、已知边界（保持原样，勿当回归）

- 首轮 needle 可 miss：压缩打分器没选中 needle 页时（打分质量，backlog；与投机无关）
- 会话 checkpoint **一次性消费**（恢复轮不再原位停放）；store 按 `SESSIONS` 条 LRU，
  第三条停放挤掉最早
- 压缩强制**同步调度**（块表中途改写）；async 解禁需另行设计，当前按决策保留禁令
  （85 tok/s 已超参考带，无收益支撑该风险）
- 重建/压缩后的请求**禁投机**（draft 账本与改写视图不同步时的 fail-closed）
- 跨重启的 turn1 输出非确定（批形变化影响采样路径）——拼进 turn2 的历史要用同实例的 turn1
- 单条长文本端到端验证；**并发 × 压缩 × 复用未验证**
- `sparse_runtime/concurrency` 两测试在 200K 模拟后段的个别 needle 页为**既有偏差**
  （2026-10-30 移除容量保护逐字复现确认），修复前不算新回归
- UI 依赖 llama.cpp/ninfer 扩展端点的功能（`/v1/chat/completions/control`、`/v1/stream`、
  `/slots`）返回 404
- 三级磁盘/会话冷存（`nvme-gb`、`session-cache-dir`）未实现

## 四、调试设施（长期保留，见 `AGENTS.md`）

事件流 `/tmp/kvmem_sparse.jsonl`（`session_checkpoint_retired`、`resume_decided`、
`session_resumed`、`rebuild_select_diag`、`debug_pos` 等），引擎日志
`engines/vllm/run-logs/`，入口决策与启动摘要 `~/kvmem.log`（`session_reuse_on` 一行
带 block_size 与 ranks）。超池测试构造器：`scripts/reconstruction_needle.py`
（fail-closed：拒绝构造池内数据的重复验证）。

启动日志核对行：`Spec decode: mtp/4`、`MTP graph policy: nosync`、
`GPU KV cache size: 201,438 tokens`、`session_reuse_on`、
`retrieval=on -> original compiled path + query bridge`。
