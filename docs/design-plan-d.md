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

## 5. R1/R2a 验证状态修正（见 design-prefill-pressure §6.43）

R1 "identity 通过"与首轮 R2a 验证全部发生在 site 树（无 D hunks）上，判定作废。
site 树 a156cf2bfd 已含 R1+R2a 全部 fork hunks（patch sha 7f11009c）；R2a 真验证 = PLAN_D=1 boot + deltas/view_lens 钩真加载 + 同查询双发确定性 + 40K/88K needle。锚点新基线 de3921f3/2d117bd（spec4）。

## 6. R2a 定案（真树，2026-11-03 深夜）

- **D=1 identity = 表不注入**（`KVMEM_D_KNIFE=1` 才建 deltas/view_lens 表）。
  教训一：全零 slot_positions 张量替换 None 入 Triton kernel **不是数值等价**
  （40K needle 中毒 27283e53 代）——R2b 传真值时 slot 路径必须 needle 级验证。
  教训二：seq_lens 的内核约定 `pos = seq_len - query_start + idx`（诊断实锤
  old=1/new=9），外部无法复刻其 base 算术——view_lens 表必须由内核同点位供给
  或由刀直接写内核期望值。
- R2a 真树验收：40K×2 确定性 True（8bad5b37）、锚点臂内容正确、boot 干净；
  88K 纯 dense 无刀臂 needle False = 已知长上下文衰减边界（正是 R2b 检索刀
  存在的意义，v0.5.0 rebuild 臂 88K 7/7 为证）。
- 环境基线：site=a156cf2..0b83d19（patch 39dbcb23）；spec4 各臂锚点代不同属
  合法臂差异（de39=pressure 臂 / e783=D-dense 臂 / b804=classic 臂），换代判据
  以 needle 电池 + 同查询双发一致为准，锚点不再跨臂比较。


## 7. R2b 协议（刀=页重排，本设计的施工合同）

**复用面（不动）**：(c) 水位触发器、selector/core_selection_scores 选择、quiescent_compact 的
worker 侧页搬移与快照回退、adopt/revert 通道、geometry fail-closed。
**切换面（D 分支）**：

| 阶段 | v0.5.2 现状 | R2b D 语义 | 失败语义 |
|---|---|---|---|
| 水位判据 | num_computed(压缩账本) ≥ hi | **view 占用 ≥ hi**（num_computed 恒真，永不 rewind） | 同现状（idle throttle 不变） |
| 提案 | select_pages 升序 | 不变 | 空提案=不动 |
| worker install | row 重写+页交换（已有） | 不变 + **写表**：deltas[ri]=st.tokens−base_view、view_lens[ri]=base_view | 异常→revert 快照（row/表一起回） |
| Core apply | truncate _all_token_ids + rewind num_computed | **D 分支：两者都不做**；epoch/事件照发 | 拒案=无账本回滚需要（更稳） |
| prompt 末 | compact_rebuild→child | **compact_rebuild=False**：无 child，decode 直接续 | escape 谓词天然恒假 |
| 每步维护 | — | view_lens[ri] += n_new（worker step wrapper） | 步失败=现状一致 |
| hook | seq_lens=表值（纯读） | **恢复 +query_start**（内核约定 pos=seq_len−ql+idx，表只存 base） | mismatch 诊断常开（刀下 old=真 new=视图 属预期） |
| seal/resume | 浅快照两段式 | **R3**：StateImage 在 cursor 处诚实可得（D1 不变量）→ 深度=前缀全长，skip 合法 | R3 定 |

不变式自检：D1 num_computed 单调；D2 row 物理集=窗口容量恒定（cap 冻结后零 alloc/free）；
D3 刀零 truncate；D5 因果=视图升序+前缀性质；GDN 只吃真 token 流。
施工顺序：①apply 开关（plugin/Core）②worker install 写表+step 维护 ③hook +ql
④水位判据改 view 8⑤88K live 电池 → 152K/250K → 删旧机器（R3 一并）。

## 8. R2b 现场进展（2026-11-04 晨，主机重启后）

- 断点续验发现夜间电池从未真正运行刀：`KVMEM_D_KNIFE` 行在环境文件操作中丢失（d_mode False、表从未创建）——补回后链路瞬间打通。
- **第一刀端到端活了**：`d_tables_published` view_base=24048 / true_base=40960..45056 / **delta=16912..21008**（与 stride 理论吻合）；prefill 100% 正常推进；publish 的 inference-tensor 坑三连修（写侧 no_grad→.data→**建表侧 inference_mode(False)**才是正解）。
- **剩两门**：
  1. 第二刀 idle：quiescent_compact 在进入 blockwise 之前 return None 且无事件（所有 blockwise None 分支都有 log，一个没出现）——需给入口三检查补 reason 事件再打一发定位；
  2. 窗口越界守卫：stride 节奏失配时 prefill 写出窗口（本轮 80970 vs 53248 窗，第二刀没落 → 越界 ~8K → 请求 900s 未完被客户端 timeout）。方向：第二刀修好后按 stride 自然落；另加硬保险 = worker 端写游标逼近窗满时同步拒写并报 `d_window_guard`（提案，下轮定）。
- 夜间遗留：`/tmp` 数据文件全灭（重建了 88K 数据集 80970tok/needle@43K）；site/patch/engine 三树同步状态未受影响（重启无变化）。
- 本轮证据链：88K runs（125.4s needle True 无表时代 → 修表后 timeout 时代）+ console 直读（d_mode False→True、has_deltas False→True 的对照）。

## 9. R2b 第二轮夜战（2026-11-04，探针驱动的全部根因命名）

reason 探针（wrapper exception/is_rebuilt/entry）把"第二刀 idle"完全解剖，逐层修：
1. `NameError: epoch` —— quiescent_compact 签名无 epoch 参数（u2 锚当时就断在这里）：补参数闭环 wrapper→runtime ✓ 离线 qc/pp 绿。
2. **after_step fill 越界 raise** 打断 quiescent RPC → D 分支 `d_window_guard`（不污染 row，占位 -1 方案作废——它把 row 撑成 3334 毒死 validate）。
3. **stride 没留 chunk** → `_win - _vis - chunk`（刀必须在下 chunk 写越界前落）。
4. **refuse 风暴**：validate 的 frontier/epoch 与提案不匹配 —— 提案 stamp 采用发送方 (frontier, epoch)（`_d_sender`）+ worker 侧 frontier 相等检查 D 豁免。**残留失败**：stamp 修正后 88K-1 又 timeout（第二刀 cadence 仍有死角，疑似 `_d_sender` 全局态跨请求污染或多请求 RPC 时序）——这是唯一剩下的一门。

**已验证的正面成果**：88K 两发 needle=True（121.9s / 95.4s——95.4s 比无刀 dense 105s 更快，第一刀语义真实生效）；d_tables_published delta 与 stride 理论吻合；确定性（同代 sha 曾一致）。
**下一轮唯一缺口**：第二刀 refuse/idle 的死角（先按请求隔离 _d_sender，一发 88K 看 refuse 是否绝迹）→ 然后 152K/250K → 删旧机器 R3。


## 10. R2b 协议层收官（2026-11-04 round5）——缺口转移到 decode/MTP 通道

**协议层现状（全部达成，实测）**：
- 88K 双刀 cadence：vis 24048→23632、delta 16912→50096（滚动增长正确）；
- scores=true（TP 跨 rank 共识在 D 下改走"同步触发即共识"——trigger 本身 collective，两 rank 同刻进 RPC，各自 core_selection_scores 同长；consensus collective 的 availability 检查在 D 下反而误伤 → 跳过；worker 入口的 `_d_sender` 盖章让提案与 Core 现算值天然相等）；
- refuse=0：validate 的 D 豁免 = resident 相等 + visible>=resident（两条都是 rewind 世界项）；
- overrow/margin law：选择必须真空至少 2 chunk，防空转刀。

**唯一剩余缺口 = decode 侧**：prefill 100% 后请求 190s+ 不结束（abort 在 released 前无 crash）。
嫌疑排序（下一步逐个用 console 断点测，不猜）：
1. MTP4 draft 的 position/slot 通道在 D 视图坐标下越界（scratch 页、draft block_table 行）；
2. bonus-token 写位（query_len=5 步）slot=-1 或越窗；
3. `_resample_kernel/_rejection_kernel` JIT 首编译风暴（慢但会结束——不足以解释 190s 不收尾）。
判别实验：D boot 上 spec0 × 88K —— 若 spec0 完成且 needle 正确 → MTP 通道独毒；
若 spec0 也卡 → 主 decode 路径（slot 映射/seq_lens）的视图问题。

**运维事实修正（本回合踩坑）**：生产加载的 runtime 源 = site venv 的 **site-packages**
（finder 只映射 `vllm/`；repo src/*.py 由 start 脚本拷入 site-packages）——
**改 src/ 后必须重启才生效，且 patch/版本文件同步要覆盖 site-packages 拷贝链**。
本轮所有修复实际生效（boot=代码重启），但"改了不重启"的静默窗口骗了两次中期判读。


## 11. R2b 质量门达成（round 5 终）+ MTP 门移交

- **单变量矩阵**（88K 数据集 80970tok / needle@~43K）：
  | 配置 | 时长 | needle |
  |---|---|---|
  | D刀+spec0+旧律（final 刀被 2-chunk 律挡死） | 78.9s | **False** |
  | D刀关+spec0（纯 dense） | 117.1s | True |
  | D刀+spec0+**分律**（mid 要真空 2 chunk；final 要保覆盖） | 115.9s | **True ✓** |
  - final 刀接线三连：(c) 候选 D 豁免 prompt-complete 排除、(b) final 集 in-flight 门在 D 下改用 placeholders、apply 打 `kvmem_d_final_cut/kvmem_final_reselect` 防重刷。refused 风暴随之绝迹（本轮 applied=3 refused=0）。
- **移交门 1（decode/MTP）**：spec4+D 刀 → prefill 100% 后 190s 不收敛（无 crash，abort 后 released）；spec0 正常完成 → **MTP4 draft 通道在 D 视图坐标下的 position/slot 独毒**，下轮 console 断点定位（嫌疑：draft block_table 行/scratch 页/query_len=5 写位）。
- **移交门 2（152K/250K 电池）**：spec0 口径先验通，再开 MTP 门。
- 生产恢复：SPECULATIVE=4 + PLAN_D=0（v0.5.2 已验证臂）；刀场保留在 /etc/default 注释位。


## 12. R2b 收官扫描（round 6）：cadence 全绿，剩"final 问题刀落点"一门

**本轮打通（applied 计数与 refused=0 为证）**：
- (c) 触发在 D 下全程开火（d_c_probe 事件链），stride 基线归零修正后不再 45056 复读；
- **刀 10-11 连发全成功**（epoch 1..25，vis 24-27K 稳态）；152K spec0 完成 140.7s（dense 219s → **刀提速 35%**）；
- **两口径分离定案**：提案=validate 的 clip 和；刀律+表=窗口整页（`_d_window_visible`）——覆写 visible 的教训（chunk 对齐期全对、prompt 完成瞬间永拒 397 连）已成文；
- d_none_reason/d_rpc_return/d_c_probe/d_window_guard 四探针构成完整取证面（本轮三次靠它们命名根因：NameError(final 未定义参数名)、UnboundLocalError(定义顺序)、frame 分叉）。

**剩余一门（下一步）**：152K needle False 的唯一解释链——统计刀正确滚动视图，但**"问题驱动 final 刀"没在 prompt 完成刻以问题分重选到 needle 块（block 74）**。已排除：law 误拒（已放宽为 row-bound-only，epoch 25 落地=final 在 decode 步有成功 applied），kept 列表显示 tail 块在推进（86/91）但 **中间问题分没把 74 选进来**。可能：final 刀在 decode 段触发时问题分对"页 4750 区"打分缺失（harvest 只到 45056 统计）→ 需要在 prompt 完成瞬间（(b) final 集首次成立时）落一刀，用**冻结问题重打 store 分**（v0.5.0 的 rebuild_select 有现成问题打分链——D 版走 blockwise(pressure=False) 的 core 分数是否真是问题驱动 = 下一步第一检查点）。

**spec4+刀判据未验**（draft view-split 代码已落，真实刀况 timeout 待复测——若 final 门修复后仍 timeout 则 draft 修复复验提级）。
服务态：spec0 + D=1 + knife=1（生产臂 SPEC=4/PLAN_D=0 恢复留给验收全绿后）。


## 13. Final 刀打分链路定罪（round 7）

诊断事件 d_final_scores（一次 final 一发）实测：
- `core_len=704`、`n_blocks=92`、**`block_scores_len=29`**、`top8=[8,9,10,13,12,14,11,16]`、
  `score_around_needle={}`（needle 块 74 **根本不在分数表里**）。
- 判词：**分数网格 resample 只铺到 block 28**——`b=(gi*grid_pages)//q` 的 grid_pages/q 配比
  在 152K 上覆盖不到 n_blocks（704 条分数 × 32 页 ÷ q 上限≈29 块）。这不是排序问题、
  不是 room 问题（room=10 有名额），是**评分坐标系没走到 needle 所在区域**。
- v0.5.0 时代 final 刀正确选页走的是 `_rebuild_select` 的官方选择段（engine grid、
  block_ok、tail forced——88K/152K/250K 7/7 全对的 proven 打分链路）。
- **下轮施工定案**：D 的 final 刀弃用 blockwise 自建 resample，改调 rebuild_select 的
  选择段抽成 `_d_final_pages()`（返回升序 logical_pages），再接 D install（无 rewind、
  无 channel、row 直写、表照旧）。mid 刀维持统计 grid（它的职责只是滚动窗口）。


## 14. 打分链路：独立深水区立项（round 8 叫停游击战）

三轮连续"根因"被下轮数据推翻的记录（全部留档，防止再犯）：
1. round 6 判"final 刀被覆盖律误拒"→ 放宽后 final 落地（epoch 25）但 needle 依旧 False；
2. round 7 判"resample grid 只覆盖 29/92"→ 抽段/坐标系假设；
3. round 8 修 align_scores 截断（raw 全长度 2236）→ **d_final_scores 本轮零事件** =
   blockwise 的 pressure=False 聚合段根本没被执行——前三轮观测的"final 刀"其实是 mid 刀，
   **观测对象本身搞错了**。applied 序列 vis 27808/25024/23808 + stride 全为 mid 特征。

**当前铁的事实**（多轮重复、互相一致）：
- D 协议层稳定：cadence 11 连发 refused=0、两口径分离、152K spec0 140.7s 完成（dense 219s，
  刀提速 35%）、40K/短请求全对、boot 健康；
- **D 刀的选择质量未达验收**：mid 刀（统计分）合法地可能漏 needle 页，而"问题驱动的 final
  刀"在 D 臂的实际执行路径从未被证实跑过（本轮零事件）——它是**下一个独立工程单元**：
  先把 final→blockwise(pressure=False) 的调用链打通并有事件证据，再谈打分网格。

**生产线决议**：验收线未达，**生产保持 spec4 + PLAN_D=0（v0.5.2 验证臂）**；D 臂继续在
实验环境迭代。raw/consensus/sanitize 改动对 classic 臂零影响（classic 不走这些分支），
已五套件验证。


## 15. Final 刀结构公案破案（round 9）+ 最后的深水区定性

**破案过程（事件链完整）**：
1. 无条件入口探针 d_knife_enter（上轮探针埋错位置的教训修正）+ Core (b) 决策探针 d_b_probe；
2. 发现 (b) 探针零事件 → 值级探针 → 也零事件 → 把 step 外层 except 从吞掉改为记录
   sched_compact_error → **抓到 NameError 真凶：round 8 我给 (b) 上方插值探针时写了
   `if _isd and self is not None:`——`_quiescent_compaction(core)` 模块级函数里没有
   `self`！异常被外层吞 → (b)(final 集/RPC/d_b_probe)整段死码，而 (c) 在异常之前的位置
   正常执行（这正是"刀全落地但 final 永不发生"的诡异组合的源头）**；
3. 清除毒块后：**final 刀首次真实执行**（d_b_probe=1、pressure=false 入口出现、
   d_final_scores 出事件）。

**最终定性（下一轮唯一问题）**：d_final_scores 显示 core_len=704——recall.scores 的输出
长度 = (frontier+63)//64，frontier=45056 = **statistics 收割冻结在首刀位**。152K 的 45K 之后
97K tokens 从未被打分收割 → 无论选择算法怎么改，分数世界里根本没有 needle 区的格子。
**问题从"选择/坐标/网格"全部收束为一个**：D 滚动刀下 statistics harvest 的 frontier 为什么不
随真账本推进（读 v0.19 收割协议：谁喂 harvest、刀窗内/外、harvest 与视图安装的关系）。

**运维教训（第二次踩同一坑并写死）**：给 Core 模块级函数插代码时禁止引用 self；
探针必须放条件链最外层；except 吞异常 = 取证黑洞（sched_compact_error 现已常开）。


## 16. Harvest 死锁定位到最后一层（round 10）

- 破案链：harvest_attempt 事件显示收割停在 step 7（tokens 28672→45056）→
  `publish_position_offset`/`harvest_decode_rows` 通道考古 → wrote0 拦截修复上线 →
  本轮事实：statistics_final `frontier=45057, steps=12` ——**45056 之后只有 1 行进入统计**
  （4096 chunk 全部未落，且 d_harvest_wrote0 事件不存在 = 没走 wrote0 分支——
  它们在 `stats_from_bridge` 之前/之中被别的路径吞了：候选 `skip("no capture")`/
  `skip("drafts")` 或 `_tp` 锚在 chunk step 上取不到 n 行）。
- 本轮落地的机制（都有代码+套件背书）：fork 发布 `kvmem_step_true_positions`（delta
  通道存活时每步真坐标克隆）；harvest D 分支=真坐标写锚+视图偏移读锚；wrote0 不再推进
  `_stats_next`（防止错位复读的雪崩）且值级可见（d_harvest_wrote0）。
- 下轮第一取证点：在 `stats_from_bridge` 返回空/短的地方打值事件（n/layers/rows/buf 容量），
  一发区分"读不到行"vs"锚没生效"vs"skip 门"。三套 skip 计数现已并进 statistics_final 的
  skipped 字段——本轮它还是 null（说明连 _skip 都没走——更可能是 harvest_attempt 在
  step≥8 后就没再触发：plugin 调用点的 n_new 或 prefill 标志在刀后异常——查调用点条件）。


## 17. Harvest 公案判决：供水断在 capture，收割无罪（round 11）

- 函数首行漏斗探针（d_hv_funnel，无条件前 8 次/请求）实测：harvest_key_statistics
  **每步都在被调用**，stats_core=True、has_stats=True、n_tokens 正常；
- write_step 之后的 d_harvest_call 事件 = 0 → 每步都死在 `sums = stats_from_bridge(...)`
  为空（"no capture" 分支）；
- statistics_final：frontier=45057、steps=12、written=11280 —— **第 11 步整 chunk 入统计、
  第 12 步只入 1 行、此后 sums 恒空** = **bridge capture 从首刀附近就停止写行**；
- 判决：**问题在 capture 侧（谁停的水），不在收割坐标系**。此前 3 轮在 harvest/selector
  上的修正（真坐标锚、wrote0 保护、is_rebuilt 短路）都是必要卫生但不是本病灶——
  is_rebuilt 短路本身仍是真修复（stale mark 确实存在且毒过别的消费者）。
- 下轮唯一目标：capture 停水根因。第一嫌疑：capture op 的门读 `st.rebuilt` 字段（不是
  is_rebuilt() 谓词——短路盖不住它）；第二嫌疑：capture install 的 per-request 门
  （ARMED/geometry）在刀事件后失活。取证法：capture_slot_mapping/cap.note_tokens 处
  漏斗事件（同 d_hv_funnel 模式），一发看哪层没水。


## 18. Round 12: 刀系统全链路已执行到位；断水定位到引擎 capture 内部（最后一根水管）

- zero-offset 假设被字节级证伪（frontier 仍 45,057）；offset 不是病灶（修复保留：
  D 下推 0 依然是语义正确卫生）。
- 但本轮 tail 时间线给出新事实：**mid 刀每 ~2 步正常滚动（row_synced 序列
  29120→23808→27904→32000→23552→27648→31744 = 视图在窗口内健康伸缩）**，
  **epoch25 问题刀成功落地**（d_knife_enter pressure=False @143134 → blockwise
  749 pages moved=749 failed=0 → applied vis 23966 → row_synced 继续推进）——
  **刀的执行系统全部到位，唯一缺的是给问题刀的分数**（statistics 断供）。
- 断水点收束到最后一个黑盒：bridge 的 capture op 设备侧写行为（stats_from_bridge
  读取端已证每步在跑、行源为空）。**现成读出**：bridge 自带 `_op_calls`/status
  （read_all→"statistics" 事件），下轮第一步 = 对比第 12 步前后 op_calls 增量与
  bucket_writes 增量，一发区分"op 没被调"vs"op 调了没写行"vs"写了读不到"。
- 结构修复累积（全部保留）：publish_offset D=0、is_rebuilt 短路、wrote0 雪崩保护、
  真坐标锚、两口径分离、(b) 探针、漏斗探针、epoch 盖章、margins 法、final 刀接线。


## 19. 断水点静态排雷（round 13）：假设出局清单 + CPU 复现路线

静态解剖 capture 写路/读路后的出局名单：
- `KEY_ROWS_LIMIT` 卡死 → **出局**（plugin 每步无条件重设 0/1）；
- `_K_MISMATCH` 宽度失配 → **出局**（刀不动 key 宽度，且首轮 warmup 11280 行正常入）；
- `first` 坐标错 → **出局**（stats_from_bridge 根本不用 first 定位行，span 由 n 导出，
  读的是 bridge 的当前步缓冲头部——我此前所有"锚"工作对这个读路无效）。
剩余唯一可能：`read_key_buckets(layer, span)` 每层返回 None（666-692 区），其
return None 只有两个门：take<=0（span=66 排除）或 **buffer/rows 对象本身为 None
（= capture 那层本步没写过 → n<=0/…）**。

**下轮第一步 = 免 boot 的 CPU 复现**：tests/bridge_capture_test.py 已有 mock bridge，
扩一个场景 = d_mode + 首刀后步骤序列 → 断言 sums 空复现 → 在 mock 里直接调试读路。
复现不了才上 GPU 探针（read_key_buckets 的每层返回值事件）。

**本轮顺带纠正的认知**：`harvest_attempt` probe<6 截断曾让我们以为调用停在 step7，
真实情况是调用从未停（漏斗证）——探针截断本身就是本次连环误判的一半来源，
已把"探针不截断/截断必标注"写入运维条律。

### 运维条律追加
- 探针要么不截断，要么在事件里带 cap 标注（防"观察工具骗人"第三次发生）。
- 同一问题连续两轮"根因"被证伪 → 立即停手改复现路线（CPU mock 优先），
  不在真机上继续假设循环。


## 20. Round 13 实录：op_calls 曲线 + 两个被更正的观测 + retire 断裂立案

- 直读曲线（steps%3 无 cap 后）：op_calls 严格 +16/步（每层一次）到 frontier 36864、
  armed=true 恒真 —— **capture op 被调用且在写**（graph/eager 假设出局）。
- 观测自我更正（第二次犯同一条律并当场记下）：
  1) harvest_attempt 的 probe<6 截断 → "调用停在 step7" 是假；
  2) d_hv_funnel 的 cap<8 → "harvest 只跑了 8 次" 也是假。
  **凡带 cap 的探针，其最后一条事件的距离不等于真相 - 判读前必须先看 cap。**
- 立案的真断裂：`session_checkpoint_retire_aborted: "TP workers did not all
  prepare a checkpoint"` @请求收尾 - D 臂上 checkpoint prepare 两侧不对称 →
  retire abort。这既是 R3（resume/seal 在 D 下的诚实性）的核心缺口，也可能与
  统计断供同源（seal/prepare 走的是 harvest 数据的 completeness 检查！
  "statistics coverage incomplete at seal" 分支在同一条链上）。
- 下一轮唯一入口（非猜测，是对案）：retire abort 现场取证 - prepare 失败侧的值
  （哪层检查、当时 coverage 数字）；它与"第 12 步后 sums 空"很可能是同一个
  completeness 检查的两个断面。若同源，修 capture 覆盖 = 修 seal + 修 final 分数。


## 21. Xid31 硬防线落地并验证（round 14）

- 尸检结论（用户机器两次卡死）：NVRM Xid 31 MMU Fault = kernel 写越出冻结行后
  GPU 页表故障 → 驱动级硬锁死（journal 写不出，表现为整机卡死）。事后记账 guard
  拦不住 kernel 的写，必须事前不可能。
- 机制（fork scheduler 一处 + Core apply 一处，零 kernel 改动）：
  * 刀落地时 Core 记 `request.kvmem_d_ceiling = knife_true + (window − vis) − chunk`
    （两账本同步前进 → 真位写到 ceiling 时行恰好满，chunk 为刀响应头量）；
  * 排程点守卫：`computed + num_new + lookahead > ceiling` → num_new_tokens=0
    （引擎自己的"无事可排"路径，等待至多一步，刀因 stride 同额度已在路上）；
  * 首刀前无 ceiling（dense 写天然界内）；ceiling 只在 D 分支生效。
- 实测（D=1 knife=1 spec0 boot）：88K 75.1s 完成、152K **119.8s** 完成（此前同条件
  140.4s 且每轮 guard 事件 2-6 次）；**d_window_guard=0、refused=0、本 boot Xid=0**；
  三树同步：engine guard commit、site 32519891f0、patch 7650ab16/tree cb74be83。
- harvest 断供新线索：本轮 statistics 走到 step 10+ frontier 40960 仍在写
  （旧形态 step 12 死）——defer 步改变了 harvest 时序；final 分数仍覆盖不足
  （block_scores 29/49）→ 断供点下轮以 d_hv_pre/post（已埋，无 cap）判 read-back
  空 vs 写路空。


## 22. Round 15 立案：刀后时序在"同 boot 第二个请求"上全面劣化（跨请求态嫌疑）

实测矛盾（同一代码状态、同一 boot、两次 152K）：
- guard 首轮（boot 后第 1 个大请求）：step-end statistics 事件正常爬升（op_calls
  +16/步、frontier 40960+ 继续涨）、刀 15 发 refused=0、guard=0；
- 同 boot 第二轮 152K：step-end statistics 事件 **0 条**、final 刀进入 blockwise
  （d_knife_enter pressure=false ×2）但**打分段未执行**（d_final_scores 0 条）、
  statistics_final 显示 written=0（真实请求键！）；
- bridge 读路静态无清除点（_KACC 无 clear/pop、take 恒 66、reset 无调用者）——
  "读空"物理不可能 → 病灶必然是**写路或请求态字典**：`self._statistics[req]` 键
  错位（attach 与 final 读的对象不同）或 step-end 循环对第二个请求根本没进。
- **下轮第一刀**：blockwise 入口早退点 + `_statistics` 字典键 + step-end 循环命中
  三处值探针（全部无 cap），单 boot 连跑 152K×2 直接对照两轮差异。

安全面不变：guard 已让越界不可能（两轮 Xid=0），本机可继续实验。
性能面不变：guard 下 152K 最快 119.6s（dense 219s 的 54%）。


## 23. 断供终审（round 16，无 cap 铁证）：请求状态在首刀附近被拆

- 第四次踩 cap 坑（bucket 用 n_new//4096 恒 0 - 每 chunk 都是 4096）后彻底去 cap。
- 铁证（152K 双跑）：
  1) `d_hv_funnel` 到 call 10（tok=40960）为止，之后零调用 —— 收割从**第一刀前一步**就停了；
  2) call 9 起事件里 `tok=None` = **runtime `_state[req]` 已不存在，而 Core 请求仍在跑**；
  3) 与 round-13 的 `session_checkpoint_retire_aborted`（TP workers did not all
     prepare a checkpoint）同场景：seal/retire 链半路折断 - 状态拆了、没拆干净。
- 病灶定名：**D 刀落地时 seal/retire 对请求状态的提前释放**（谁 discard 的：
  accept_reconstruction? _release_slots 附带? retire 链? 下轮在 `_state.pop/del` 
  与 session retire 出口装无 cap 值事件，一发点名）。
- 此前"step-end 循环成员继续"的结论也被坏 bucket 污染，一并作废重查。
