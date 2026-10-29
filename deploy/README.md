# deploy/ 约定

两层，仓库自带的默认值就是可用的起点——**不需要站点脚本**，clone 下来直接跑：

```
run-current.sh         ← 通用启动入口，不含任何站点参数（systemd 指向它）
        │ exec
        ▼
run-kvmem-vllm.sh      ← 实现：拼参数、调 worker.py（默认值都在这里的 ${VAR:-default}）
```

默认值：`B=36864 R=16384 N=1 GPU_UTIL=0.97`，profile 用仓库自带的
`2x2080Ti/qwen27b/w8a16/mtp4-fp8kv-1x184K-text-image.env`。唯一必须由你提供的是 `MODEL`（权重目录）；模型没有内置 chat template 时再加 `CHAT_TEMPLATE_FILE`：

```bash
MODEL=$HOME/models/<你的模型> ./deploy/run-current.sh
```

覆盖参数有两种方式，都不需要改仓库文件：

1. **环境变量**（临时/一次性）：`N=2 B=53248 GPU_UTIL=0.98 ./deploy/run-current.sh`
2. **`~/.config/kvmem-vllm.env`**（systemd 用法）：单元里有 `EnvironmentFile=-%h/.config/kvmem-vllm.env`，
   文件存在就加载，不存在就忽略。内容就是 `KEY=VALUE` 一行一个。

```bash
sudo ./deploy/install-service.sh
mkdir -p ~/.config && printf 'MODEL=%s\n' "$HOME/models/<你的模型>" > ~/.config/kvmem-vllm.env
sudo systemctl daemon-reload && sudo systemctl enable --now kvmem-vllm
```

要在一台机器上并存多套配置（fp8 / text-only / 不同并发档），各自写一个 env 文件，用
`ExecStart=%h/vllm-kvmem-adapter/deploy/run-current.sh` + `EnvironmentFile=` 指过去即可——
这属于使用者的习惯，不是仓库的要求。

## 与本机其它几套的对照

| 单元 | Type | 启动脚本 | 日志 |
|---|---|---|---|
| `kvmem.service`（llama.cpp）| `simple` | `start-kvmem-coder390-q6.sh`：`export VAR=${VAR:-默认}` + `exec "$BIN"`（前台）| `StandardOutput=append:~/kvmem.log` |
| `kvmem-ninfer.service` | `simple` | `start-kvmem-ninfer-22g.sh`，同套路 | `append:~/kvmem-ninfer.log` |
| `vllm.service`（旧）| `oneshot` | `start-vllm.sh`：先清理残留进程，再调 `run-current.sh` | — |
| **`kvmem-vllm.service.in`（本仓库）** | **`forking` + `PIDFile`** | `start-kvmem-vllm.sh`（0.3.0 配置） | 仓库内 `run-logs/kvmem-vllm.log` |

三处刻意的差异：

1. **`Type=forking`**：vLLM 的 `launcher.sh` 会把引擎 `setsid` 出去再退出，`Type=simple` 会让单元立刻
   变成 inactive（`vllm.service` 用 `oneshot` 绕开了这点，代价是 `is-active` 永远不反映真实状态）。
2. **日志由脚本写**：`run-kvmem-vllm.sh` 同时输出启动横幅、worker 与 launcher 的输出到 `kvmem.log`，
   再加上 launcher 自己的 `run-logs/vllm-*.log`，比单纯 `append:` 多一层信息；因此单元里
   `StandardOutput=null` 避免重复。
3. **`DRYRUN=1`** 两层都支持（`run-current.sh` 打印真正的启动命令），和 llama.cpp/ninfer 脚本的习惯一致。
