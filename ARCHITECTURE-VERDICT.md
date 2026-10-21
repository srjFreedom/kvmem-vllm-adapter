# 分配侧可行性：已定位到"最后一块缺失部件"

## 已核实的既有能力（都在 vLLM 里，不需要 fork）

`_remove_blocks_in_range()`（`single_type_kv_cache_manager.py:654`）：

```python
for i in range(last_block - 1, first_block - 1, -1):
    freed.append(blocks[i])
    blocks[i] = self._null_block      # 槽位保留，长度不变
if freed:
    self.block_pool.free_blocks(freed) # 块真的回到池子
```

也就是说：**释放**是原生支持的——选一段不再需要的对齐块区间，块回池、位置槽保留为 null_block。

## 但释放之后还有两个缺口

### 缺口 1：注意力仍会读 null 页
FlashInfer 没有 R-SWA/掩码支持（`flashinfer.py` 里 `rswa` 出现 0 次）。null 页是零值，
FlashInfer 会把它当正常 token 计算（score 0 → softmax 分母被稀释），所以**光释放会让输出错**。

**已解决**：我们的 worker 侧压缩会**重写块表行**并覆盖 `seq_len`/`positions`，注意力只读压缩后的
页；被释放的块不在压缩行里，所以 null 页不会被读到。这块已经实现并验证过（99K/64K/29K 实测）。

### 缺口 2：还原时需要"向调度器要块"，而这是 worker→scheduler 方向
分配权在 EngineCore（`block_pool`），worker 只有只读的行。当前 `scheduler_output` 是**单向**
（调度器 → worker），没有 worker 反向申请块的通道。

这是唯一缺失的部件。两种做法：

| 方案 | 做法 | 代价 |
|---|---|---|
| A（干净） | 在插件里同时 patch 两侧：worker 在 `ModelRunnerOutput` 上附带"请求块"字段，EngineCore 读它、分配、下一步 stage 下来 | 需要两侧都加字段（都在插件里，不改 vLLM 源码），但依赖 fork 的 RPC 结构 |
| B（务实） | 用进程间通道（unix socket / 共享文件 + 轮询）做 worker→EngineCore 的块申请 | 简单、无需改 RPC；但要处理时序与失败路径 |

## 结论

分配侧不是"不可行"，而是**还差一条 worker→scheduler 的块申请通道**。在补齐它之前：

- 释放本身可用（`free_blocks` 已验证语义）；
- 注意力正确性依赖我们的压缩（已实现、已实测）；
- 因此补上通道后，"释放未选中块 → 池里多出块 → 其他请求可用"就能闭环。

## 建议顺序

1. 先用**方案 B**（socket/文件）做最小闭环验证：释放 1 个对齐块 → `free_blocks` 计数 +1 →
   申请并取回 1 块 → 写回 KV → 校验埋针；
2. 闭环通了再考虑换成方案 A（更干净、无轮询）。
