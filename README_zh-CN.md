# KVMem × vLLM (SM75, TP=2) — 单服务集成

[English](README.md) | **中文**

把 [kvmem](https://github.com/kvmem/kvmem) 的策略层接到 vLLM（weicj 的 SM75 fork）上：
**一个服务、一个端口（18200）**，OpenAI 兼容 API、指标和 Web UI 同源；vLLM 仍是引擎，
kvmem 决定"每次发多少历史、KV 留在哪一层"。

```
客户端 ──▶ :18200 ┬─ /v1/*        OpenAI 兼容 API
                   ├─ /metrics     引擎指标
                   ├─ /docs        OpenAPI
                   └─ /            静态 UI（可换）
```

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
| vLLM SM75 fork | `93c9013e375b` + 集成补丁 | `install.sh` 拉取并打补丁（校验 `patched_tree`）|
| kvmem 源码 | tag `v0.18.0` | 同上（编译原生扩展要头文件与库）|
| 模型权重 | **vLLM 能跑的任何模型**（本部署用的是 FP8 + MTP 导出）| 自己放，改 `MODEL` |

```bash
git clone https://github.com/srjFreedom/kvmem-vllm-adapter
cd kvmem-vllm-adapter

USE_LOCAL_FORK=0 ./install.sh            # 拉引擎 → 打补丁 → 装插件 → 编扩展 → 自检
ENGINE=$PWD/engines/vllm ./scripts/setup-venv.sh   # torch / CUDA（几个 GB，已有 venv 可跳过）

cp deploy/kvmem-vllm.service /etc/systemd/system/
cp deploy/run-current.example.sh /home/srj/run-current.sh   # 参数都在这里改
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

所有运行时参数集中在 `run-current.sh`，完整说明见 **[docs/PARAMETERS.md](docs/PARAMETERS.md)**。

```bash
B=65536 R=32768 CTX=524288 SINK=8192 RECENT=2048 N=2 PORT=18200
CAPTURE_TOKENS=512
MODEL=/home/srj/models/Coder390/FP8-fp8mtp
CHAT_TEMPLATE_FILE=.../chat_template.jinja
PROFILE=2x2080Ti/qwen27b/w8a16/mtp4-fp8kv-1x184K-text-image.env
KVMEM_UI_DIR=/home/srj/local-model-console
```

关键约束：**引擎窗口 = `N × (B + R)` 必须装进 KV 池**（本机 210,651 token）。
当前 `N=2, B+R=98,304` → 窗口 196,608，刚好；`N=4` 需要把 B 降到约 26K。

改完 `sudo systemctl restart kvmem-vllm` 生效（无热改），`kvmem-params.env` 记录本次生效值。

## 运维

```bash
sudo systemctl start|stop|restart kvmem-vllm
tail -f /home/srj/kvmem.log          # 引擎 + 入口策略同一份日志
cat /home/srj/kvmem-params.env       # 本次生效参数
curl -s localhost:18200/health       # 健康
curl -s localhost:18200/metrics      # vllm:* 指标（prefill/decode/KV 使用率）
```

入口策略可在日志里看到：`kvmem entry: {"kind":"entry_rewrite",...}`（含 `gen_budget`、
`prompt_selected` 的块数与字符数）。

## 实测（当前配置：N=2，B/R=65536/32768，CAPTURE=512）

| 项目 | 结果 |
|---|---|
| 逻辑上下文 | 486,605 token 历史 → `prompt_tokens=89,066`，两个埋针全中（106 s）|
| R 契约 | `max_tokens=100000` → `applied=32768`（只写 `max_completion_tokens`）|
| CTX | 747,614 token → `400 history is ~747614 tokens, above kvmem CTX=524288` |
| 流式 | 30 个 SSE 事件，首 0.07 s、末 1.43 s，时刻单调 |
| 端口 | 只有 `0.0.0.0:18200`，无旁挂进程 |
| prefill | 65.2K prompt：70 s ≈ **934 tok/s**；98.8K：713 tok/s；2 路并发聚合 477 tok/s |
| decode | 小 prompt 生成 512 token ≈ **59 tok/s** |
| 前缀复用 | 同前缀第二轮 `cached=63,024/65,890` |

长上下文耗时**几乎全是 prefill**；并发不提升吞吐（TP=2 已用满两张卡）。

## 测试与探针

```bash
PY=/home/srj/vLLM-2080Ti-Definitive/.venv/bin/python
for t in tests/*_test.py; do PYTHONPATH="build/python:src:plugin" $PY "$t"; done   # 39 个，全 PASS
python3 scripts/check_integration_surface.py        # 28/28 个 vLLM/kvmem 接触点
./run-gpu-tests.sh                                  # GPU 模式（需先停服务）
```

## 仓库结构

```
backends/vllm/worker.py     vLLM 后端的启动包装（窗口 N×(B+R)、参数转发）
plugin/                     general 插件 + endpoint 插件（入口策略、UI 挂载）
src/                        入口策略、文本选片、H/D 缓存、稀疏运行时、原生扩展源码
scripts/                    worker、wait_ready、prepare-backends、setup-venv、探针
tests/                      39 个测试（非 GPU 全通过）
deploy/                     systemd 单元、run-current 模板、启动脚本
backends/versions.json      引擎固定版本 + 补丁 sha256 + patched_tree
backends/patches/           vLLM 侧全部改动（17 文件 / 612 行）
docs/PARAMETERS.md          每个参数的含义、约束与影响
STATUS.md                   每次决策背后的实测数据
INTEGRATION-SURFACE.md      vLLM 接触点清单与源码改动登记
```

## 已知限制

- **入口文本选片是代偿路径**（vLLM 不能中途截断 KV），不是 kvmem 的策略本身；历史与问题挤在
  同一条消息时仍用"末尾段落"启发式，问题很长且主题在前段时可能丢块
- **不支持 in-request compaction**（R 满后继续到 budget）：vLLM 单调长度假设，实测断言/挂起；
  上游也是实验性默认关
- `SESSIONS` 的语义是**把 H 平分给活请求**；请求结束即释放，不等同于参考后端的"保留 N 个非活跃历史"
- 未实现参考后端的三级磁盘/会话冷存（`nvme-gb`、`session-cache-dir`）与 NVMe 页级存储
- UI 的"停止生成"等用法依赖 llama.cpp/ninfer 的服务端扩展接口（`/v1/chat/completions/control`、
  `/v1/stream`、`/slots`），vLLM 没有，会 404

## 版本固定与集成面

引擎源码不 vendored：`scripts/prepare-backends.py` 按 `backends/versions.json` 拉取固定
commit/tag，应用 `backends/patches/vllm-kvmem.patch`，再用 `patched_tree` 校验结果与产线一致
（kvmem 官方对 llama.cpp / ninfer 用的也是这套「固定版本 + 整份集成补丁」布局）。

vLLM 侧的接触点、每处改动的作用与行数见 [INTEGRATION-SURFACE.md](INTEGRATION-SURFACE.md)。
