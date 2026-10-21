# KVMem backend: vllm

对应 `backends/llamacpp` / `backends/ninfer` 的 vLLM 后端 worker。

## 契约

- 入口：`worker.py`，解析 kvmem 的 `--kvmem-*`（与另两个后端同名同义）
- 传递：`--kvmem-*` → `--additional-config '{"kvmem": {...}}'`（vLLM 官方通道，随 VllmConfig
  序列化到每个 TP worker）
- 布局：本目录可直接拷进 kvmem 仓库的 `backends/vllm/`（纯新增，不改任何既有文件）

## 启动

```bash
python3 backends/vllm/worker.py --model /path/to/model --port 8000 \
  --kvmem-budget 32768 --kvmem-gen-reserve 16384 \
  --kvmem-sink-tokens 8192 --kvmem-recent-tokens 1024 \
  --kvmem-sessions 4 --kvmem-host-mib 16384 --kvmem-block-tokens 128 \
  --no-kvmem-lrfu --max-model-len 204800 --tensor-parallel-size 2
```

`--dry-run` 打印将要传给 vLLM 的 `additional_config` 与插件配置，用于审计/测试。

## 与 llama/ninfer 的对应关系

| kvmem worker | vllm worker |
|---|---|
| 链接 kvmem 静态库 | `vllm_kvmem_ext` pybind 模块（同一套 `MemoryBackend`/`MemorySession`/`HostKvStorage`） |
| 命令行参数直接可用 | 参数组装进 `--additional-config`，由 vLLM 分发到每个 worker |
| 引擎与策略同进程 | 策略在 EngineCore 决策、各 TP worker 执行同一 plan（确定性 + 指纹校验） |
