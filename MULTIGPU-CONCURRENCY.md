# 多卡 + 并发 ≥ 2：能不能解决？（结论：能，且不是 kvmem 核心的限制）

## 1. llama 后端为什么不行——限制的确切位置

`src/kvmem-v018rc1/tools/llama-kvmem-server.cpp:1288`

```cpp
if (options.parallel > 1 && (!st.kparams.enabled ||
        options.device_names == "none" || mparams.n_gpu_layers == 0 ||
        options.session_disk_bytes ||
        st.kparams.nvme_bytes || st.kparams.raw_k_nvme ||
        device_config.devices.size() > 2 || mparams.split_mode == LLAMA_SPLIT_MODE_TENSOR ||
        (options.threads_http > 0 && options.threads_http < 2 * options.parallel))) {
    fprintf(stderr, "KVMEM_STARTUP_ERROR --parallel %d requires KVMem, one target GPU, ...");
```

这是 **llama worker 的启动守卫**，不是 kvmem 核心的契约。它成立的原因是 llama.cpp 侧的
KVMem 集成建立在「**一份 GPU 工作集、多会话时分复用**」之上
（`tools/llama-kvmem-driver.h:62` 原话："N host KV stores, one GPU working set, time-multiplexed"），
再叠加 llama.cpp 的 layer/tensor split 就管不住了，所以直接禁掉。

## 2. kvmem 核心本身没有这个限制

| 证据 | 位置 |
|---|---|
| 运行时/会话是**每请求一个实例**（block table + CPU/NVMe 层 + 重选节拍） | `kvmem/include/kvmem/kvmem_runtime.hpp:34` |
| H 是**共享预算**，B/R 是**每请求完整逻辑窗口**，不按卡数拆分 | `docs/multi-backend-differences.md:76` |
| 两个后端都开放并发：llama 1..4、ninfer 1..8 | `docs/multi-backend-differences.md:111` |
| ninfer **在多卡上**用 layer pipeline 实现，"各卡只持有自己的层" | 同上第 76 行 |

也就是说：**"多卡 + 并发"在 kvmem 的设计里是被支持的**（ninfer 就是例子），llama 只是没实现。

## 3. 我们已经用 vLLM 跑通过

| 实测 | 结果 |
|---|---|
| TP=2 + 4 并发（4×29K） | **4/4 答案逐字正确**，引擎健康 |
| TP=2 单请求 99K | 常驻恒定 2,048 页、190 s 完成 |
| TP=2 分配侧释放 | 调度器真释放 20/39 对齐块 |

vLLM 天生就是这个模型：TP=2 每层切两半，每个 rank 为**同一个逻辑块**持有自己的分片；
连续批处理让 C 个请求共享一个池。这正好对应 kvmem 的"B/R 每请求、H 共享"。

## 4. 为什么在我们的架构里比 llama 更容易

| 维度 | llama worker | 我们的 vLLM 后端 |
|---|---|---|
| 决策位置 | 单进程 worker 内 | **EngineCore 一个调度器**，天然只有一份决策 |
| 多卡一致性 | 每个 worker 各自决策，需要额外同步 | TP 各 rank 执行**同一条** plan（我们已有确定性策略 + 指纹校验） |
| 并发 | `--parallel` 守卫单卡 | vLLM 连续批处理，池按块数记账 |
| H 预算 | 每次启动一个值 | **每 rank 一个值**（与目标里的 H=16384MiB/rank 一致） |

## 5. 真正需要解决的（都在我们这侧，可控）

1. **C 个会话共享 H**：kvmem 的 `KvMemRuntime` 是每会话一个实例，所以 EngineCore 里要有 C 个
   实例 + 一个薄预算分配器（按 `sessions` 均分或先到先得），否则 H 会被超发；
2. **选入粒度换算**：kvmem 按它自己的块（llama 1,161 token）度量，vLLM 调度器按 1,616-token
   对齐块释放，需要在适配层把 kvmem 的块映射到页区间再判断整块可还；
3. **vLLM 侧两个已知坑**（与 kvmem 无关）：准入公式按 `max_model_len` 估算、投机解码的
   `num_output_placeholders` 断言。

## 结论

**能解决，而且不需要改 kvmem 核心。** 需要的是把会话管理从"每进程一个"改成"EngineCore 里
N 个实例 + 共享 H"，以及上面第 1、2 条的适配层工作。llama 的那条守卫不该被继承——它是
llama.cpp 实现的产物。
