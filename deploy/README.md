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
