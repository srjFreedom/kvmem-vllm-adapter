# KVMem backend: vllm

对应 `backends/llamacpp` / `backends/ninfer` 的 vLLM 后端 worker。

## 契约

- 入口：`worker.py`，解析 kvmem 的 `--kvmem-*`（与另两个后端同名同义）
- 传递：`--kvmem-*` → `--additional-config '{"kvmem": {...}}'`（vLLM 官方通道，随 VllmConfig
  序列化到每个 TP worker）
- 布局：本目录可直接拷进 kvmem 仓库的 `backends/vllm/`（纯新增，不改任何既有文件）
- 启动链：worker.py exec `deploy/run-kvmem-engine.sh` → 引擎 `launcher.sh`；
  CUDA graph + MTP + 会话复用的生产默认值全部在 deploy/ 脚本里（v0.4.0）

## 启动

```bash
python3 backends/vllm/worker.py --model /path/to/model --port 8000 \
  --kvmem-budget 32768 --kvmem-gen-reserve 16384 \
  --kvmem-sink-tokens 8192 --kvmem-recent-tokens 1024 \
  --kvmem-sessions 4 --kvmem-host-mib 16384 --kvmem-block-tokens 64 \
  --no-kvmem-lrfu --max-model-len 204800 --tensor-parallel-size 2
```

`--dry-run` 打印将要传给 vLLM 的 `additional_config` 与插件配置，用于审计/测试。
