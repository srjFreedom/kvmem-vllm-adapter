# 方案 D：全量 token 流 + 压缩 attention 地址空间（完整版，用户拍板 2026-11-03）

## 0. 动机（v0.5.2 用户实测的两大症状）

1. 二次请求命中后仍要重算 ~47K（skip 深度被 StateImage 只能在首刀前拍死锁在 ~32K）；
2. prompt 完成后有 child 全量重算 30480（"原版没有这一步"——正确）。

两个症状同根：**v0.5.x 的刀压缩了 token 流本身**（rewind `_all_token_ids`/num_computed），
GDN 线性状态因此与全序列失配，一切（child 重算、浅 skip、快照窗口）都是这个失配的补偿。
原版 ninfer 按层分离执行程序（`set_linear_state_slots`/`linear_attention` 独立管线），
token 流对线性层永远全量，换出只发生在 attention 的地址空间——所以它没有 child、
reuse 深度 = `observed prefix/cursor`（全量）。

## 1. 不变量（D 的全部语义）

- **D1 token 轴永不压缩**：`num_computed_tokens`、`_all_token_ids`、GDN 输入流按真实
  长度推进；positions 恒为真位置（≤262,144 位置表内，入口防线已有）。
- **D2 attention 轴是窗口**：每请求 block_table 行 = **≤B+R 页的检索视图**，行内按
  真位置**升序**排列（sink 头部、评分中段升序、recent 尾部）；写地址=滚动游标进
  尾部槽位。行内容完全由 adapter 拥有（fork 既有固定容量 row 设施）。
- **D3 写映射走 slot_positions**：fork `compute_slot_mappings(positions, slot_positions)`
  的双位置签名就是为此预留——`positions` 喂 RoPE（真位置），`slot_positions` 喂写
  地址（视图滚动槽）。D 模式 adapter 必须逐 token 提供 slot_positions（真位置稀疏，
  行索引推不出来）。
- **D4 刀 = 纯页重排**：水位过时，选择（sink+recent+评分）→ 淘汰页 bytes 已在 store
  （per-step 提交 baked=0 规范真位置）→ 保留/新选页按真位置升序**重排入行**
  （restore_pages_batch 纯拷贝）→ 释放页归还池。**不截 token、不 rewind、无 child。**
- **D5 正确性论证（不改 kernel）**：视图行真位置升序 + 全部 ≤ 当前 chunk 起点 ⇒
  kernel 的索引式 causal 与真位置 causal **对前缀视图等价**；RoPE 相对相位由 K 写入
  时的真位置烘焙决定（baked=0 canonical），评分数学正确。chunk 内 causal 与采样
  路径不涉及视图坐标。
- **D6 状态永远真**：GDN 连续执行 ⇒ prompt 末的状态就是 observed prefix 的状态 ⇒
  resume 深度 = 全前缀（state 归档 + 视图页归档 + token 尾部），**命中轮只算增量**。
  child 机制、QueryReplay、escape 抑制、双账本 rewind——**整体删除**。

## 2. 手术点

| # | 位置 | 改动 |
|---|---|---|
| S1 | fork `gpu/model_runner.py:1607` | compute_slot_mappings 增传 `slot_positions=`（来自 adapter 注入的 per-request 写游标表；无注入时 None=现行为） |
| S2 | fork `v1/core/sched/scheduler.py` + kv_cache_utils | D 模式下页配额 = 窗口常数（`--reserve` 已有 off 通道），按视图容量而非逻辑长度 |
| S3 | adapter runtime | 写游标管理（滚动槽分配/复用）、slot_positions 张量供给、行=升序视图 |
| S4 | adapter 刀 | blockwise 刀改 D 语义：选择→重排→归还，删 rewind/channel truncate/`_all_token_ids` 截断 |
| S5 | 删除清单 | child rebuild、QueryReplay、escape_suppressed、双账本（num_logical offset 体系退化为恒等）、shallow seal/两段快照补丁、OFF 整体重算路径——**不留兼容** |
| S6 | resume/archive | StateImage 在 prompt 末即真；entry = 视图页集 + cursor + 尾部 token ⇒ skip 全前缀（原版 observed prefix 语义） |

## 3. 验收（goal 定义）

- 88K/152K/250K needle 全对；**152K 二次请求 ≈ 只算增量（目标 <15s，非 169s）**；
- 命中轮与冷跑字节一致；A/B 锚重生成（执行模型换了，锚换代一起出）；
- 离线全套 + 双租户并发初步（水位=每请求份额自律的初验）；
- 文档：README/DELIVERY 双臂合一语义重写，design-plan-d 收尾。

## 4. 分轮

R1 fork 通道（S1/S2 + adapter S3 游标）→ identity-D 短请求与现行字节一致；
R2 刀 D 化（S4）+ 旧机器删除（S5）→ 超池 needle 全对；
R3 resume 深度（S6）+ 全 battery；R4 文档/发布。
