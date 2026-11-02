# KVMem × vLLM — SM75 双卡高性能、单服务集成

[English](README.md) | **中文**

**把 kvmem 的策略层跑在 vLLM 进程内，面向两张 Turing（SM75）显卡的高性能版本。**
不是"引擎前面挂个网关"，而是入口策略直接跑在 vLLM 的 API 进程里；引擎本身为两张 22 GB
消费级卡做过调优——MTP 投机解码、FP8 KV 缓存、前缀缓存，以及按并发数 N 定尺寸的 KV 池。

在 2× RTX 2080 Ti 上换来的东西（以下全部为 v0.4.0 生产实例实测）：

- **CUDA graph + MTP 投机，满速。** 2,000 token 倒数解码 **84–85 tok/s**（`spec=4`）——
  落在原版参考带（70–80）之上，是被迫 eager 旧适配器路径（46）的 ~1.8 倍。打分桥已做到
  图可捕获（静态缓冲、原地算子、与 eager 数值逐位一致）。
- **超池上下文，默认开启。** 常驻 prefill + 请求内压缩是缺省路径：88K prompt 在 36,864
  常驻窗口上仍答对档案中段 needle；KV 池 `GPU_UTIL=0.972` 下装 201,438 token。
- **会话复用。** 带 `X-Session-ID` 的请求结束后把前缀 KV + 循环状态存入主机内存；
  下一轮从前沿恢复，85K 档案省 ~34 s prefill（needle 两路答案一致）。
- **一个服务、一个端口（18200）。** OpenAI 兼容 API、`/metrics`、`/docs` 和 Web UI 同源——
  没有旁挂网关，没有第二个要养的进程。
- **kvmem 仍是策略层。** 发多少历史（B）、哪些页常驻（用真实 pre-RoPE Q 检索、mean-K 打分、
  LRFU 衰减）、哪些进 H/D 分层，都由它决定；vLLM 继续当引擎。

```
客户端 ──▶ :18200 ┬─ /v1/*        OpenAI 兼容 API
                   ├─ /metrics     引擎指标
                   ├─ /docs        OpenAPI
                   └─ /            静态 UI（可换）
```

## 性能

实测环境：**2× RTX 2080 Ti 22 GB**（NVLink 桥接，NCCL 自动使用），TP=2，`N=1`，
`B=36,864 + R=16,384`，MTP `spec=4`，FP8 KV，CUDA graph 开启（未设 `KVMEM_FORCE_EAGER`），
`VLLM_SM75_SPEC_SYNC_MODE=nosync`，`temperature=0`，与 graph 前构建同实例对照：

| 负载 | v0.3.x（被迫 eager） | **v0.4.0（编译 + graph）** |
|---|---|---|
| 2,000 token 倒数解码 | 46.0 tok/s | **85.0 tok/s** |
| 3,000 token 存活跑（两次跨 1,616 块边界） | 2,000 tok 43.5 s | **3,000 tok 35.4 s**（84.7 tok/s） |
| 85,675 token 档案，首轮冷跑 | ~150 s | **143 s** |
| 同档案第二轮会话恢复 | 115.7 s | **110.0 s**（比冷重跑省 34 s） |

测试钉死的验收门：短请求无损（乘法表/随机两位数输出跨 spec0/spec4 逐字节一致，
sha `c2f36446`/`42675199`）、88K 档案中段 needle `ZX-4417-QJ7`、每路径确定性
（冷×2 逐字节一致；恢复输出跨重启、跨 eager/graph 稳定——恢复 vs 冷跑的字节一致
为何不是本设计属性，见"已知限制"）。

## 现在的架构

**引擎侧（vLLM 进程内）**

| 组件 | 作用 |
|---|---|
| `plugin/vllm_kvmem_plugin.py` | `vllm.general_plugins` 入口：探针、runner 钩子、稀疏运行时接线；所有进程加载 |
| `src/kvmem_sparse_runtime.py` + `kvmem_sparse_orchestrator.py` | 按页的 H（主机内存）/D（磁盘）分层、LRFU 淘汰、mean-K 检索 |
| `src/vllm_kvmem_ext.cpp` | 原生扩展（链接 kvmem 库），暴露 `VllmMemoryBackend` / `NvmeKvTier` / 检索索引 |

**入口侧（同一进程，官方 `vllm.endpoint_plugins` 扩展点）**

| 组件 | 作用 |
|---|---|
| `plugin/kvmem_entry_middleware.py` → `KvmemEntryPlugin` | `attach_router`：装中间件 + 挂 UI |
| 中间件（默认 `mode=middleware`）| 标准 Starlette body 改写（`request._body` 缓存）：按 B 选历史、`max_tokens` 夹到 R、CTX 上限、H/D 历史块缓存 |
| `mode=hook`（兜底）| 包装 `OpenAIServingChat.create_chat_completion`，在已解析请求上应用同一策略 |
| `src/kvmem_entry_policy.py` | 两个模式共用的唯一策略实现 |

响应侧完全不干预，所以**流式、CORS、错误处理都是 vLLM 原生行为**。

### 为什么选片放在入口

vLLM 用逻辑位置索引 KV 块，且假定序列长度单调增长：中途截断会让调度器失步（实测
`num_scheduled_tokens >= num_logits` 断言，关掉投机后变成挂起）。所以 kvmem 的"只保留 B"
必须在**请求进入引擎之前**决定；引擎内则用 kvmem 自己的按页策略（`prepare_reselect`）。

### 检索 query 怎么来

`torch.ops.kvmem.capture_q` 在注意力层捕获**真实的前一步 pre-RoPE Q**（默认最近 512 个 token，
`CAPTURE_TOKENS`），kvmem 用它和页内 mean-K 做打分；LRFU 负责历史衰减。参考后端
（llama.cpp / ninfer）用的是"最后一条 user 消息的 token span"，两者都在用模型内部表示，
差别只是取哪些 token。

## 快速开始

依赖三样外部件，仓库本身不含（`backends/versions.json` 固定版本）：

| 外部件 | 固定版本 | 获取方式 |
|---|---|---|
| vLLM SM75 fork | `93c9013e375b` + 集成补丁 | `install.sh` 把 `base_commit` 拉到 `engines/vllm`，再用 `backends/patches/vllm-kvmem.patch` 整体替换（校验 `patched_tree`）；默认不改动任何已有 fork，除非显式 `USE_LOCAL_FORK=1` |
| kvmem 源码 | tag `v0.19.0-rc1` | 同上（编译原生扩展要头文件与库）|
| 模型权重 | **vLLM 能跑的任何模型**（本部署用的是 FP8 + MTP 导出）| 自己放，改 `MODEL` |

```bash
git clone https://github.com/srjFreedom/kvmem-vllm-adapter
cd kvmem-vllm-adapter

USE_LOCAL_FORK=0 ./install.sh            # 拉引擎 → 打补丁 → 装插件 → 编扩展 → 自检
ENGINE=$PWD/engines/vllm ./scripts/setup-venv.sh   # torch / CUDA（几个 GB，已有 venv 可跳过）

sudo ./deploy/install-service.sh
sudo systemctl daemon-reload && sudo systemctl enable --now kvmem-vllm
scripts/wait_ready.sh 18200 900 --smoke
```

已有环境时只需一步（跳过拉取与 venv）：

```bash
VLLM_ROOT=/path/to/fork KVMA_SRC=/path/to/kvmem PY=/path/to/venv/bin/python ./install.sh
```

**不可避免的三步**（属于本仓库的代码，任何环境都要做）：打补丁、装插件、编原生扩展。

### 支持哪些模型

能加载什么由 vLLM 决定（架构、量化、chat template）；KVMem 层在此之上只多两条要求：

- **按页分层**（H/D 溢出、LRFU）作用在 full-attention 的 KV 组上，所以混合架构（注意力 + GDN/Mamba）
  没问题——默认模型就是这种。
- **检索打分**依赖 pre-RoPE query 捕获，钩子落在 `vllm/model_executor/models/qwen3_next.py`
  （属于集成补丁）。换别的架构需要在它的注意力层里加同样两行；没有钩子时入口选片、R 上限、
  历史缓存仍然工作，只是 mean-K 页打分不可用。

所以：vLLM 支持的模型都能跑，FP8 + MTP 只是这里实测过的配置。

## 配置

两层，仓库默认值就是可用的起点——**不需要站点脚本**：

```
start-kvmem-vllm.sh 0.4.0 生产默认配置（kvmem 预算为主、vLLM 窗口为辅）；systemd 跑它
run-current.sh      通用启动入口，不含任何站点参数
   │ exec
run-kvmem-vllm.sh   实现（每个参数都是 ${VAR:-default}）
```

默认：`B=36864 R=16384 N=1 GPU_UTIL=0.972`（并带 `KVMEM_SESSION_REUSE=1`、
`KVMEM_MAX_MODEL_LEN=200000`、`VLLM_SM75_SPEC_SYNC_MODE=nosync`、MTP `spec=4`——
v0.4.0 端到端验证过的组合），profile 用仓库自带的
`2x2080Ti/qwen27b/w8a16/mtp4-fp8kv-1x184K-text-image.env`。唯一必须提供的是 `MODEL`（权重目录）；只有模型没有内置 chat template 时才需要给 `CHAT_TEMPLATE_FILE`；其余参数用环境变量覆盖，或写进 `~/.config/kvmem-vllm.env`
（单元存在则加载）：

```bash
CONCURRENCY=2 BUDGET=53248 GPU_UTIL=0.98 ./start-kvmem-vllm.sh
```

完整参数说明见 **[docs/PARAMETERS.md](docs/PARAMETERS.md)**。

关键约束：**引擎窗口 = `N × (B + R)` 必须装进 KV 池**（本机 `GPU_UTIL=0.972` 下 201,438 token；
CUDA-graph 缓冲比 eager 多吃约 2.5K token，0.97 会把池压到 `MAX_MODEL_LEN=200000` 以下——实测 OOM）。
默认 `N=1, B+R=53,248` 余量很大；`N=2` → 106,496，`N=3` → 159,744，`N=4` → 212,992 需要把 B 降到约 26K。

`THINKING_BUDGET` 默认为空 = **思考 token 无上限**；填数字才会限制推理长度。

有界常驻压缩（实验开关，默认关）：`KVMEM_PREFILL_PRESSURE=1` 让 prefill 期间按水位
（B+R/2）滚动压缩——sink+尾块+评分选块、未选页回写主机，prompt 完成时用主机存储重建
选中视图并重排队。实测 250K token（超池 226,240 且超原窗 200,000）端到端 2/2 答对
`ZX-4417-QJ7`（详见 docs/design-prefill-pressure.md §6.25-6.29）。该臂启动即
fail-closed：池顶不足水位+余量时自动禁用并回落生产语义（`pressure_disabled_geometry`
事件）。超长 prompt 另需 `MAX_MODEL_LEN` 高于 prompt 且 `KVMEM_SKIP_KV_CHECK=1`
（RoPE 位置表本身按模型原生 262,144 分配，不是瓶颈）。

改完 `sudo systemctl restart kvmem-vllm` 生效（无热改），`kvmem-params.env` 记录本次生效值。

## 运维

```bash
sudo systemctl start|stop|restart kvmem-vllm
tail -f ~/kvmem.log          # 引擎 + 入口策略同一份日志
cat ~/kvmem-params.env       # 本次生效参数
curl -s localhost:18200/health       # 健康
curl -s localhost:18200/metrics      # vllm:* 指标（prefill/decode/KV 使用率）
```

入口策略可在日志里看到：`kvmem entry: {"kind":"entry_rewrite",...}`（含 `gen_budget`、
`prompt_selected` 的块数与字符数）。

## 契约检查（不是性能基准）

| 项目 | 结果 |
|---|---|
| 逻辑上下文 | 486,605 token 历史 → `prompt_tokens=89,066`，两个埋针全中（106 s，B=65536 时测得）|
| 生成上限 | compaction 默认开 → 上限是 `CTX`（经典路径才回落 R 并裁剪入口）|
| CTX | 747,614 token → `400 history is ~747614 tokens, above kvmem CTX=524288` |
| 流式 | 30 个 SSE 事件，首 0.07 s、末 1.43 s，时刻单调 |
| 端口 | 只有 `0.0.0.0:18200`，无旁挂进程 |
| 会话复用 | park `retired frontier=32320` → resume `hit=32320` → prefill 从 32320+4096 起，省 34 s，needle 两路一致 |
| 图捕获 | 编译实例启动无 capture 违规（离线守卫 `tests/bridge_capture_test.py`，数值逐位对拍） |

这些行钉的是策略契约（多少历史进入引擎、R 上限、CTX 拒绝、SSE 行为）；吞吐数字见[性能](#性能)。

## 测试与探针

```bash
PY=$PWD/engines/vllm/.venv/bin/python
for t in tests/*_test.py; do PYTHONPATH="build/python:src:plugin" $PY "$t"; done   # 59 个文件
# 核心契约 9 项必须全绿：quiescent_compact position_books store_semantics commit_guard
# sched_gate sparse_runtime sparse_concurrency session_clip bridge_capture
python3 scripts/check_integration_surface.py        # 28/28 个 vLLM/kvmem 接触点
./run-gpu-tests.sh                                  # GPU 模式（需先停服务）
```

## 仓库结构

```
backends/vllm/worker.py     vLLM 后端的启动包装（窗口 N×(B+R)、参数转发）
plugin/                     general 插件 + endpoint 插件（入口策略、UI 挂载）
src/                        入口策略、文本选片、H/D 缓存、稀疏运行时、原生扩展源码
scripts/                    worker、wait_ready、prepare-backends、setup-venv、探针
tests/                      59 个测试（核心契约 9 项全非 GPU）
deploy/                     systemd 单元、通用入口、启动脚本
backends/versions.json      引擎固定版本 + 补丁 sha256 + patched_tree
backends/patches/           vLLM 侧全部改动（17 文件 / 612 行）
docs/PARAMETERS.md          每个参数的含义、约束与影响
INTEGRATION-SURFACE.md      vLLM 接触点清单与源码改动登记
```

## 已知限制

- **入口文本选片只是经典路径的代偿**；默认的常驻入口把完整 prompt 送进引擎，选页由引擎内
  mean-K 打分器决定
- **in-request compaction 是默认路径**，比朴素路径更精细：强制关闭前缀缓存、强制同步调度
  （压缩在请求中途改写块表；async 禁令是正确性决策，只有当吞吐再次掉出参考带才值得重评）；
  新生成块没有检索分数（recent 尾部强制保留）；带投机 draft 的步不做压缩
- **恢复 vs 冷跑的字节一致不是本设计的契约。** 每条路径各自确定（冷×2 逐字节一致；恢复输出
  跨重启、跨 eager/graph 稳定），needle 两路答案一致；但恢复轮重放的是停放时的选页，冷轮按
  当前查询重新选页——检索语义决定了两个物理页集合本就不同（实测同页数下 chosen 头部不同）。
  存的前缀 KV 是 FP8 原字节回放，冷跑会重新量化；贪心采样可能在 gap 处翻转
- **会话 checkpoint 一次性消费**（恢复后的轮不会原位再停放；第三轮重新 prefill）；
  store 按 `SESSIONS` 条 LRU 淘汰，第三条停放会挤掉最早的
- **首轮 needle miss** 仍可能发生：压缩打分器没选中 needle 页时（与投机无关，属打分质量，待办）
- 单条长文本请求端到端验证过；**并发 × 压缩 × 复用未验证**
- 未实现参考后端的三级磁盘/会话冷存（`nvme-gb`、`session-cache-dir`）与 NVMe 页级存储
- UI 的"停止生成"等用法依赖 llama.cpp/ninfer 的服务端扩展接口（`/v1/chat/completions/control`、
  `/v1/stream`、`/slots`），vLLM 没有，会 404

## 版本固定与集成面

引擎源码不 vendored：`scripts/prepare-backends.py` 按 `backends/versions.json` 拉取固定
commit/tag，应用 `backends/patches/vllm-kvmem.patch`，再用 `patched_tree` 校验结果与产线一致
（kvmem 官方对 llama.cpp / ninfer 用的也是这套「固定版本 + 整份集成补丁」布局）。

vLLM 侧的接触点、每处改动的作用与行数见 [INTEGRATION-SURFACE.md](INTEGRATION-SURFACE.md)。

## 许可

本仓库为 Apache-2.0，见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。

注意：它只是**集成层**，不包含引擎源码。
- 被补丁的 vLLM 文件仍属上游 Apache-2.0（补丁流程保留其 LICENSE/NOTICE）
- kvmem 以固定 tag 拉取并链接，不 vendored、不改许可；其上游仓库目前没有顶层 LICENSE 文件，
  若要分发包含它的二进制，请先与上游确认
- 模型权重不在本仓库内，另有其许可
