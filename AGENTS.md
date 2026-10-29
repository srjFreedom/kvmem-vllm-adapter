# 项目规范（必须遵守）

本文件是**强制**约定，任何在本仓库工作的 agent 都必须遵守。

## 1. 调试：先断点/热重载，不要改代码加打印、不要重启

引擎启动一次要 ~80 秒，重启会打断用户正在观察的实例。**调查运行时行为的第一手段是下面三件设施**，
它们都**不需要改代码、不需要重启**：

| 设施 | 位置 | 用法 |
|---|---|---|
| 断点 | `src/kvmem_dbg.py`（`/tmp/kvmem_dbg.bp` → `/tmp/kvmem_dbg.out`） | 写一行 JSON 断点（`file`/`func`/`line`/`hits`/`expr`），命中时把**整个栈帧 + 指定表达式**落盘；`hits` 用完自动卸载。worker 侧从 console 里 `import kvmem_dbg; kvmem_dbg.install()`；**EngineCore 侧已接线**（`Scheduler.schedule` 包装器在 `/tmp/kvmem_dbg.bp` 存在时自动 install） |
| 实时执行 | `/tmp/kvmem_console.py` + `/tmp/kvmem_console.arm` | 引擎下一步在 worker 内执行，作用域里直接有 `runtime`/`orch`/`bridge`/`runner`，结果写 `/tmp/kvmem_console.out`；snippet 执行后自动删除（一次性） |
| 热重载 | `touch /tmp/kvmem_reload.request` | 下一个 prompt 边界 reload 模块并 rebind 类，纯 Python 改动免重启 |
| 调度器侧 | 同上断点 + `KVMEM_SKIP_KV_CHECK` 等开关 | 调度器在 EngineCore，断点已接线；读 `self.waiting`/`kv_cache_manager`/`single_type_managers` |

**允许加打印/记日志的情形**（仅这两种）：
1. 需要**长期留证**的探针（例如 `device_pool`、`compact_applied`、`retrieval_check`）——必须是命名
   `kind`，在提交信息里写明为什么需要长期保留；
2. 需要**跨重启**保留的证据（例如重启前那一瞬间的状态）。

**禁止**：为一个猜测变量改代码 → 重启 → 看日志 → 再猜。一次断点命中能拿到整帧变量。
**重启前**：确认无请求在跑；需要频繁换配置时优先换端口/另起实例。

## 2. 不要改动用户日常实例

用户既有的 vLLM 安装（仓库外的任何 checkout）是日常实例，**绝不修改**。
所有改动只落在本仓库与其 `engines/vllm`；不要在仓库里写死任何人的主目录绝对路径。

## 3. 结论要有实测

凡是"已修复 / 已打通"的结论，必须给出一条可复现的实测（命令 + 关键数字 + 日志 kind）。
否定结论同样要有实测（"为什么不行"比"试过了"有用）。

## 4. 实验开关默认关闭

任何会改变语义的实验（重建、尾部重放、块大小、跳过检查）都必须是**显式开关**且默认关闭，
并在 `docs/PARAMETERS.md` 或 launcher 注释里写明作用与已知问题。
