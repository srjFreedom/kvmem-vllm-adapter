# 参数说明

所有**运行时**参数集中在 `/home/srj/run-current.sh`（仓库里是 `deploy/run-current.example.sh`）。
改完 `sudo systemctl restart kvmem-vllm` 生效（没有热改参数）。`kvmem-params.env` 会记录本次实际生效值。

**安装期**参数只在装机器时用一次，见文末。

---

## 一、运行时参数

### 预算与生成

| 参数 | 默认 | 含义 | 改大会怎样 / 约束 |
|---|---|---|---|
| `B` | `65536` | 每次请求最多发给引擎的**历史** token 数（kvmem 的 budget）| 越大越吃上下文窗口和 prefill 时间；必须 ≥ `SINK+RECENT` 否则选片无处可裁（启动会告警）|
| `R` | `32768` | 生成上限，同时是**驻留预留**：`max_tokens` 会被夹到 `R`，缺省则写入 `R` | 越大显存/上下文占用越多；`N×(B+R)` 必须装进 KV 池 |
| `CTX` | `524288` | kvmem 的**逻辑上下文上限**（512K），与引擎窗口无关 | 超限直接 `400`。它只约束"历史能有多长"，不影响显存 |
| `SINK` | `8192` | 永远保留的开头 token 数 | 指令/角色设定放开头时调大 |
| `RECENT` | `2048` | 永远保留的结尾 token 数 | 最近对话更重要时调大 |
| `N` | `2` | 并发请求数 | 引擎窗口 = `N×(B+R)`，因此**翻倍并发 = 翻倍窗口 = 翻倍 KV 需求** |

**关键约束（来自实测）**：窗口 `N×(B+R)` 必须 ≤ KV 池。本机（2×2080Ti、fp8 KV、util 0.98）
池子 ≈ **210,651 token**、可用 3.93 GiB。所以：

```
N=1: B+R ≤ 210651      当前 32768+32768 = 65536  → 2× 余量
N=2: B+R ≤ 105325      当前 65536+32768 = 98304  → 刚好（窗口 196,608）
N=4: B+R ≤  52662      会装不下 → 必须减小 B/R 或降 N
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

### 模型

| 参数 | 默认 | 含义 |
|---|---|---|
| `MODEL` | `/home/srj/models/Coder390/FP8-fp8mtp` | 权重目录（需含 `config.json`、tokenizer、safetensors）|
| `CHAT_TEMPLATE_FILE` | `.../FP8/chat_template.jinja` | 聊天模板；该模型内置模板为空，必须显式给 |
| `PROFILE` | `2x2080Ti/qwen27b/w8a16/mtp4-fp8kv-1x184K-text-image.env` | launcher 档位：量化方式、KV dtype、上下文、视觉、投机档。**换量化必须同时换 profile** |
| `SERVED_NAME` | `Qwen3.8-27B-Coder390-EfficientThink-FP8-MTP` | 对外模型 id（客户端用它）|

换模型的最短清单：`MODEL` + `CHAT_TEMPLATE_FILE` + `PROFILE` + `SERVED_NAME`。

---

## 二、安装期参数（装机器时一次）

| 变量 | 默认 | 用途 |
|---|---|---|
| `USE_LOCAL_FORK` | `1` | `1` 用本机已有的 vLLM fork；`0` 按 `backends/versions.json` 拉固定 commit |
| `VLLM_ROOT` | `/home/srj/vLLM-2080Ti-Definitive` | vLLM fork 路径（`USE_LOCAL_FORK=1` 时）|
| `KVMA_SRC` | `/home/srj/src/kvmem-v018rc1` | kvmem 源码（编译原生扩展要头文件+库）；不存在会自动拉 `v0.18.0` |
| `PY` | `$VLLM_ROOT/.venv/bin/python` | 用哪个 Python 环境；**已有可用 venv 时指过来就不用建** |
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
