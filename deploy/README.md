# deploy/ 的三层约定

```
start-<配置名>.sh      ← 站点参数（B/R/N/GPU_UTIL/MODEL/...），每个配置一份
        │ exec
        ▼
run-current.sh         ← 通用启动入口，不含任何站点参数
        │ exec
        ▼
run-kvmem-vllm.sh      ← 实现：拼参数、调 worker.py（默认值都在这里的 ${VAR:-default}）
```

systemd 的 `ExecStart` 指向**哪个 start 脚本**，就启动哪套配置：

```
ExecStart=/home/srj/start-kvmem-vllm.sh
```

换配置 = 复制一份 `start-*.sh`、改参数、改 `ExecStart` 一行，然后：

```bash
sudo systemctl daemon-reload && sudo systemctl restart kvmem-vllm
```

这样通用入口与站点参数分开，同一台机器可以并存多套配置（fp8、text-only、不同并发档），
切换只动 systemd 那一行。

## 与本机其它几套的对照

| 单元 | Type | 启动脚本 | 日志 |
|---|---|---|---|
| `kvmem.service`（llama.cpp）| `simple` | `start-kvmem-coder390-q6.sh`：`export VAR=${VAR:-默认}` + `exec "$BIN"`（前台）| `StandardOutput=append:/home/srj/kvmem.log` |
| `kvmem-ninfer.service` | `simple` | `start-kvmem-ninfer-22g.sh`，同套路 | `append:/home/srj/kvmem-ninfer.log` |
| `vllm.service`（旧）| `oneshot` | `start-vllm.sh`：先清理残留进程，再调 `run-current.sh` | — |
| **`kvmem-vllm.service`（本仓库）** | **`forking` + `PIDFile`** | `start-<配置>.sh` → `run-current.sh` → `run-kvmem-vllm.sh` | 脚本内追加 `/home/srj/kvmem.log` |

三处刻意的差异：

1. **`Type=forking`**：vLLM 的 `launcher.sh` 会把引擎 `setsid` 出去再退出，`Type=simple` 会让单元立刻
   变成 inactive（`vllm.service` 用 `oneshot` 绕开了这点，代价是 `is-active` 永远不反映真实状态）。
2. **日志由脚本写**：`run-kvmem-vllm.sh` 同时输出启动横幅、worker 与 launcher 的输出到 `kvmem.log`，
   再加上 launcher 自己的 `run-logs/vllm-*.log`，比单纯 `append:` 多一层信息；因此单元里
   `StandardOutput=null` 避免重复。
3. **`DRYRUN=1`** 两层都支持（`start-*.sh` 打印参数、`run-current.sh` 打印真正的启动命令），
   和 llama.cpp/ninfer 脚本的习惯一致。
