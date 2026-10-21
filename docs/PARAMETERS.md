# 参数说明

所有**运行时**参数写在站点配置脚本 `start-<名字>.sh` 里（模板：`deploy/start-example.sh`），
它 `exec` 通用入口 `run-current.sh`，后者再 `exec` 实现 `run-kvmem-vllm.sh`。
systemd 的 `ExecStart` 指向哪个 start 脚本就用哪套配置。改完 `sudo systemctl restart kvmem-vllm` 生效
（没有热改参数）。`kvmem-params.env` 会记录本次实际生效值。

**安装期**参数只在装机器时用一次，见文末。

---

## 一、运行时参数

### 预算与生成

| 参数 | 默认 | 含义 | 改大会怎样 / 约束 |
|---|---|---|---|
| `B` | `36864`（36K）| 每次请求最多发给引擎的**历史** token 数（kvmem 的 budget）| 越大越吃上下文窗口和 prefill 时间；必须 ≥ `SINK+RECENT` 否则选片无处可裁（启动会告警）|
| `R` | `16384`（16K）| 生成上限，同时是**驻留预留**：`max_tokens` 会被夹到 `R`，缺省则写入 `R` | 越大显存/上下文占用越多；`N×(B+R)` 必须装进 KV 池 |
| `CTX` | `524288` | kvmem 的**逻辑上下文上限**（512K），与引擎窗口无关 | 超限直接 `400`。它只约束"历史能有多长"，不影响显存 |
| `SINK` | `8192` | 永远保留的开头 token 数 | 指令/角色设定放开头时调大 |
| `RECENT` | `2048` | 永远保留的结尾 token 数 | 最近对话更重要时调大 |
| `N` | `1` | 并发请求数（默认 1，先保证能跑）| 引擎窗口 = `N×(B+R)`，因此**翻倍并发 = 翻倍窗口 = 翻倍 KV 需求** |

**关键约束（来自实测）**：窗口 `N×(B+R)` 必须 ≤ KV 池。本机（2×2080Ti、fp8 KV、util 0.98）
池子 ≈ **210,651 token**、可用 3.93 GiB。所以：

仓库默认 `B+R = 36,864+16,384 = 53,248`，约束 `N×(B+R) ≤ 210,651`：

```
N=1: 窗口  53,248  → 默认，余量 4.0×
N=2: 窗口 106,496  → 通过（余量 2.0×）
N=3: 窗口 159,744  → 通过（余量 1.3×）
N=4: 窗口 212,992  → 略超 → 需把 B 降到 ~26K
```

启动失败时日志会直接告诉你估算上限：
`... 3.80 GiB KV cache is needed, available 3.74 GiB, estimated maximum model length is NNNNN`。

### 端口与入口策略

| 参数 | 默认 | 含义 |
|---|---|---|
| `PORT` | `18200` | kvmem 官方端口；API + UI + `/metrics` 全在这一个端口 |
| `CAPTURE_TOKENS` | `512` | 每步为检索捕获的 query token 数（对齐 KVMem `query_max_tokens` 默认 512）。**代价**：每 rank 约 `64 层 × N × 12 heads × 256 dim × dtype`，512 ≈ 200 MiB（bf16）/ 100 MiB（fp8）|
| `KVMEM_ENTRY_MODE` | `middleware` | 入口策略实现：`middleware`（标准 Starlette body 改写，默认）/ `hook`（解析后请求上的兜底钩子）|
| `KVMEM_ENTRY_SELECT` | `retrieval` | 选片打分方式：`retrieval`（按问题检索）/ `recency`（只留 sink+最近，不检索）|
| `DISABLE_PREFIX_CACHING` | `0` | `1` 关闭 vLLM 前缀缓存。**仅当**启用 `--kvmem-inrequest-compaction` 时必须关 |
| `KVMEM_UI_DIR` | `/home/srj/local-model-console` | 同端口挂载的静态 UI；空字符串 = 不挂 |

### 内存分层

| 参数 | 默认 | 含义 |
|---|---|---|
| `HOST_MIB` | `49152` | H 层（主机内存）KV 预算，48 GiB。会话间按 `SESSIONS` 平分 |
| `SESSIONS` | `4` | H 池的**并发会话**份数（注意：语义是把 H 平分给活请求，不是"保留 N 个非活跃历史"）|

### 引擎与硬件（换卡、换机器时调）

| 参数 | 默认 | 含义 | 什么时候改 |
|---|---|---|---|
| `GPU_UTIL` | `0.97` | vLLM 请求的显存占用比例（`--gpu-memory-utilization`）| 22 GB 卡用 0.97；显存更小的卡降低（0.90~0.95），OOM 时**先降这个** |
| `MAX_BATCHED_TOKENS` | `4096` | 单步 prefill 的 token 上限（chunk 大小）| 调大可减少 chunk 数、提升 prefill 并行度，但吃显存；调大后必须重跑启动 smoke |
| `SPECULATIVE_TOKENS` | `4` | MTP 投机 token 数 | `0` = 关闭投机：省显存/更稳，但 decode 变慢 |
| `NO_ASYNC_SCHEDULING` | `0` | `0` = 异步调度（默认），`1` = 强制同步 | 入口选片下引擎无块表改写，异步调度安全；两种模式实测**同速**。开 `inrequest-compaction` 时 worker 自动写回 `1`（显式设置仍优先） |
| `MESSAGE_TYPE` | `text+image` | 是否加载视觉塔 | 纯文本模型/只跑文本时设 `text`（省 ~0.9 GiB 权重）|
| `MM_IMAGE_LIMIT` | `64` | 每 prompt 允许图片数 | 与 `MESSAGE_TYPE` 配套 |
| `SERVICE_SCOPE` | `lan` | 监听范围 | `lan` = `0.0.0.0`；`local` = 只监听 `127.0.0.1` |
| `THINKING_BUDGET` | 空 | 思考 token 上限（`REASONING_BUDGET`）| **默认空 = 无上限**；填数字才限制推理长度 |

这些参数在 `run-kvmem-sparse.sh` 里都有 `${VAR:-默认}` 兜底，所以既可以写进 `run-current.sh`，
也可以临时 `GPU_UTIL=0.9 ./run-kvmem-vllm.sh` 试。

### 模型

| 参数 | 默认 | 含义 |
|---|---|---|
| `MODEL` | `/home/srj/models/Coder390/FP8-fp8mtp` | 权重目录（需含 `config.json`、tokenizer、safetensors）。**不限于 FP8**：vLLM 支持的模型都能用，本部署的 FP8+MTP 只是实测配置 |
| `CHAT_TEMPLATE_FILE` | `.../FP8/chat_template.jinja` | 聊天模板；该模型内置模板为空，必须显式给 |
| `PROFILE` | `2x2080Ti/qwen27b/w8a16/mtp4-fp8kv-1x184K-text-image.env` | 引擎档位 .env（量化/KV dtype/上下文/投机），经 worker.py exec 的 `run-kvmem-sparse.sh`(N=1)/`run-kvmem-4way.sh`(N>1) 喂给 launcher.sh。**N=1 改环境变量即生效；N>1 时 4way 脚本写死同一路径，要改脚本** |
| `SERVED_NAME` | `Qwen3.8-27B-Coder390-EfficientThink-FP8-MTP` | 对外模型 id（客户端用它）|

换模型的最短清单：`MODEL` + `CHAT_TEMPLATE_FILE` + `PROFILE` + `SERVED_NAME`。

- 量化不再是 FP8 时，`PROFILE` 里的量化/KV dtype 也要跟着换（或直接改 `--set` 项）。N>1 时 4way 脚本写死了 profile 路径，要改脚本
- **检索打分**需要 pre-RoPE query 钩子，目前只加在 `qwen3_next.py`（Qwen3.5/3.8 架构）；
  换架构要加同样两行。没有钩子时其他能力（入口选片、R 上限、H/D 缓存、前缀缓存）不受影响

---

## 二、安装期参数（装机器时一次）

| 变量 | 默认 | 用途 |
|---|---|---|
| `USE_LOCAL_FORK` | `1` | `1` 用本机已有的 vLLM fork；`0` 按 `backends/versions.json` 拉固定 commit |
| `VLLM_ROOT` | `/home/srj/vLLM-2080Ti-Definitive` | vLLM fork 路径（`USE_LOCAL_FORK=1` 时）|
| `KVMA_SRC` | `/home/srj/src/kvmem-v018rc1` | kvmem 源码（编译原生扩展要头文件+库）；不存在会自动拉 `v0.18.0` |
| `PY` | `$VLLM_ROOT/.venv/bin/python` | 用哪个 Python 环境；**已有可用 venv 时指过来就不用建** |
| `GPU_UTIL` / `MAX_BATCHED_TOKENS` / … | 见上文"引擎与硬件" | 同样可在这层用环境变量覆盖 |
| `ENGINE` / `PY_VER` | 由脚本推导 / `3.12` | `setup-venv.sh` 的目标引擎与 Python 版本 |

```bash
# 已有 vLLM fork + venv + kvmem 源码 + 权重：只需这一步
VLLM_ROOT=/path/to/fork KVMA_SRC=/path/to/kvmem PY=/path/to/venv/bin/python ./install.sh

# 全新机器
USE_LOCAL_FORK=0 ./install.sh            # 拉引擎 + 打补丁 + 装插件 + 编扩展
ENGINE=$PWD/engines/vllm ./scripts/setup-venv.sh   # torch / CUDA（几个 GB）
# 然后放权重，改 MODEL，装 systemd 单元
```

**哪些步骤是"我们的代码"、任何环境都免不了**：打补丁、装插件、编原生扩展。
**哪些是"环境"、别人有就不用**：vLLM fork 本身、venv（torch/CUDA）、kvmem 源码、权重。

---

## 三、派生值（只读，供排查）

`kvmem-params.env` 记录本次生效的 `B/R/CTX/SINK/RECENT/SESSIONS/HOST_MIB/N/PORT/CAPTURE_TOKENS`；
引擎日志启动时会打印：

```
engine window = N x (B+R) = 196608
max_model_len auto = 2 x (65536 + 32768) = 196608
KVMEM_CAPTURE_TOKENS=512
```
