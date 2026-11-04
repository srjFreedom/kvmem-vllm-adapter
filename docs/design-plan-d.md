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


## 24. Round 17 双证伪与路标

- 证伪一（我的 sweep 误杀论）：加了 8 步 grace 后 `d_sweep_spared=0` 且 funnel 照旧
  停在 call 10 —— sweep 根本没做过误杀（round-16 的"disposer 定名"过早，撤回）。
- 新线索（比 disposer 更基础）：`released` 事件仍是**旧格式**（无 after_absent 字段）
  ——**worker 进程执行的 plugin 代码 ≠ repo 磁盘文件**！loader 鉴定是下轮第一步：
  `python -c "import vllm_kvmem_plugin as m; print(m.__file__)"` 在 worker 上下文
  （console 探针打印 module __file__ + d_mode 实际值 + sweep 函数源码指纹）。
  **若 worker 加载的是 site-packages 的旧拷贝，则过去几轮所有"plugin 修改"对
  该进程可能从未生效**——多个"修了没用"的现象将获得统一解释。
- funnel/harvest 的 call 停滞机制本身未定名；但先解决 loader 真相等最低层问题。


## 25. Round 18：sweep 误杀论成立并被治；新病灶=我自己的 0-token defer

- 统一 release 政策落地（runtime.sweep_release + seen_alive，plugin 三处出口全部
  改走单一政策；site A/B + retirement 合法路保留终端授权）——"旧格式 released"
  之谜=三处平行 sweep 代码分叉（不是 loader）。
- 判决性数据：`d_hv_funnel call=57`（贯穿整个请求生命周期，首刀不再是收割坟墓）
  + skipped 字典首次可见：**"wrote 0 (frontier 31199 first 31199 n 0)"** ——
  **新病灶定名：round-14 的 ceiling guard 用 num_new_tokens=0 做 defer，
  而 vLLM 的 num_scheduled_tokens 收了 0-token 条目，下游（harvest/commit/
  book）没有为 n=0 的步准备语义；请求时间 119→208s。**
- 修法（下轮第一件事，非假设）：defer 用"该请求本轮完全不排"（loop 里 skip，
  scheduled dict 不出现 0 条目）——engine 原生语义即支持某请求跳过一轮。
  复验判据：wrote-0(n=0) 噪声消失 + 152K 回到 ~120s + funnel 继续贯穿 +
  statistics frontier 越过首刀位持续增长。


## 26. Round 19：guard 撤除与倒退链的诚实记录（下轮 git bisect 定标）

同码同 boot 对照（全部 152K spec0，D=1 knife=1）：
| 代码状态 | 时长 | knives | 异常 |
|---|---|---|---|
| round-14 guard（0-entry 版）+ 无 sweep_release | **119.8s** | 15 | 当时判"guard 致 208s"存疑 |
| round-16 grace sweep + guard(0-entry) | 208.5s | 0(no_row×100) | blockwise_no_row 新病灶 |
| round-19 guard(native skip) | 208.5s | 0 | 与 0-entry 同症状 → guard 形态无罪 |
| round-19 guard 撤除 | 221.9s(spec4×2) | **0** | pressure_gate=0、funnel 狂转 476 |

**结论修正**：208s 与 guard 形态无关；**119.8s 是 round-14 代码态（guard 0-entry、无
sweep 统一政策）的产物**——倒退发生在 round-16/18 的 sweep/grace/seen_alive 系列改动
里（`blockwise_no_row=100` = worker 状态在 RPC 时刻 row 为空 = **before_step/absorb
链被 sweep 改动饿死**，或 funnel 476 次里 after_step 主链被跳过：pressure_gate=0）。

**下轮第一动作（方法定死，不再假设）**：git bisect 三个已知状态文件组——
`033b7eb`（119.8s 态）与 HEAD 之间，文件集锁定
plugin/vllm_kvmem_plugin.py + src/kvmem_sparse_runtime.py；每态单跑 152K 计时 +
数 knives/no_row；两三次 boot 之间必收敛。
guard 已撤（其保护在 cadence 正确时代本就未用到；cadence 修复后按 round-14 判据重加）。


## 27. Round 20：收割死亡定形=请求被"合法 retirement"释放（Core 视其 finished）

- import 回归治愈（无 `_np` 丢失后：no_row=0、knives=11、gate=16、refused=0、140.4s）。
- d_hv_result：44 条=11 步×2pid×2，**第 11 步后函数零进入**；funnel 尾部 `tok=None`
  而 `has_stats=True`（_state 没了、_statistics 还在）；`released` 事件为**旧格式**
  =出自 2860 的 `retire_finished_worker_requests`（唯一合法 retirement：
  `req in scheduler_output.finished_req_ids`）。
- **判决**：收割不是被谁杀了 - 是 Core **在第 11-12 步（首刀窗口）把未完成的请求
  标成 finished**，retirement 依规释放了 state。此后引擎还在跑该请求（prefill 继续、
  刀继续落、140s 完成响应）但 runtime 账本已注销 → 统计冻结、final 无分、needle False。
- 下轮唯一问题（新问题域）：**为什么 finished_req_ids 含一个 prefill 没走完的请求**——
  嫌疑：preempt（D 的 cap 让 allocate 的 demand 失真→资源判定误 preempt？）、
  max-model-len/len 检查、或 `_finishing_reconstruction` 残留标志。取证：
  scheduler_output.finished_req_ids 到达 retirement 时的值事件（带请求当前
  computed/prompt 进度对照）+ 引擎日志 preempt/abort grep。
- 若=preempt：D 的 cap（R2a 临时 book 改写）需与 scheduler 的资源判定和解——
  这条线索把"刀 cadence"与"假 finished"统一（也解释 round-14 15 knives 后统计仍冻结）。


## 28. Round 21：假 finished 论证伪；死亡收束到"首刀后 step_end 不再含该请求"

- 到达探针（d_retire_arrival）实测：真实请求的 retirement 只在**正常终点**发生
  （tokens=143140 ≥ prompt=143134，unfinished=false，still_scheduled=false）——
  round-20 的"Core 假 finished"判决**错误**，撤回。合法 retirement 无罪。
- 新的收束（全部无 cap 探针互证）：
  * d_hv_result：22 条（11×2pid），sums_layers=16 全健康，first 止于 40960；
  * funnel 止于 call 10-11；**没有 stats_rows_error、没有 sparse_step_error、
    skipped=None** ——harvest 不是死在内部任何门，是**根本不再被调用**；
  * pressure_gate 仍 16 次（它们走 quiescent RPC 的 pending flush 路径，不经
    step_end）→ **首刀之后，worker 的 step_end scheduled 循环不再包含该请求**。
- **下轮唯一取证**（一处无 cap 事件）：step_end 循环入口打印
  {pid, steps, sorted(scheduled keys)}——一发对比首刀前后 scheduled 内容；
  若请求真不在 scheduled 里：查 scheduler→worker 的 num_scheduled_tokens 构造
  （D 的 cap/defer/epoch 交互），这是收割断供的真正上游。
- 已排除名单（累计）：capture 写停/读停、align 截断（已修 raw）、坐标锚、
  sweep 误杀（grace 后照旧）、假 finished、_np import（已修）。


## 29. Round 22：死亡再上移一层 - step_end 包装器自身约在 step 12-14 停跑

- 本轮探针（snapshot 含 nn_vals；funnel 含 d_mode；warmup 过滤 + cap 120）实测：
  * d_step_snapshot：step 11-14 `n=1 nn=[4096]` —— 请求持续被排、n_new 恒 4096，
    **scheduled 内容与数值两罪均证伪**（round-21 的"循环不再含请求"错，
    round-22 的"n_new=0 静默门"也错 - 双双撤回）；
  * 但每 pid 的 snapshot 条数停在 ~12：step_end 包装器**自身**不再被调用；
  * funnel 的 call 计数同样停在 10-11（两事件互证同一时刻）；
  * pressure_gate 仍 16 次：RPC pending-flush 路径活着（它不依赖 step_end）。
- 死亡时刻 ≈ step 12-14 = 第二、三刀窗口；请求照常 140.6s 完成（引擎主链路无恙）。
- **下轮取证（同层再上移）**：execute_model 入口第一行无条件事件（每步一条，含
  `is_profile/dummy_run` 标志）：若 execute_model 活着而 step_end 死 - wrapper 内部
  的提前 return 链（`KVMEM_EXEC_POST`/`dummy_run`/`is_profile`）是凶手候补；
  若 execute_model 也停在同步数 - CUDA graph replay 路径绕过了包装器（decode 步走
  捕获图，不再进 Python execute_model），harvest 需要挪进 graph 外唯一必经点
  （quiescent boundary 的 pending flush 已经在这么活）。
- 累计证伪清单：capture写停 / align截断 / sweep误杀 / 假finished / scheduled缺席 /
  n_new=0 - 六案皆埋于证据。


## 30. Round 23 案卷：execute_model 全程活着而 step_end 只跑 12 步——矛盾定案

- d_em_entry（包装器第一行）：42 条/pid，真实请求 1227ca 到最后一步都在，
  dummy=False prof=False 全程；
- sparse_step_error=0（大 try 无异常）、引擎日志 0 错误、retirement 只在终点；
- d_step_snapshot（step_end 循环头）：12 条/pid，step 14 后不再出现；
- **静态已读尽**：em entry 与 snapshot 之间只剩四行代码 + 两个证伪的门
  （EXEC_POST/dummy）。代码与观测矛盾。
- RPC 路径补收割（d_rpc_harvest）written=0 也定案：bridge 缓冲只存最后一步的
  行，且刀后步的 harvest 若在 step_end 死路外补收，锚已错位 - 补救路线关闭。
- **下轮唯一动作**：snapshot 前再放一个无条件事件（d_preclear：steps、
  state['sparse'] is None、n_keys），一发闭环"最后四行"之谜；若 d_preclear
  同样停在 12，则凶手在 result=original() 内（它没返回，worker 线程阻塞后
  由别的路径继续执行引擎步 - 需要查 async 采样线程与 execute_model 的关系）。
- 案卷补充：D 臂 spec0 收割死亡史 = 与 spec4 同症状（MTP 通道假设作废）；
  生产臂 v0.5.2 harvest 无恙（无刀）。问题被精确隔离在"D 刀 + step_end"。


## 32. Round 24-25 案卷：我亲手埋的"裸 return"才是头号元凶；新前沿=state 在 ~step 8 蒸发

- **头号案件（已破已修）**：round-21 我用 s.find 移动错位的 d_pre_after 探针时，
  把 `if os.environ.get("KVMEM_EXEC_POST", "1") == "0":` 条件行整个吃掉，留下
  **裸的 `return result`** —— step_end 段自此 EVERY 步无条件返回。round 21-24 的
  "step_end 死亡"不是引擎之谜，是我对文件的一次毁伤。修法=条件行复原（本轮），
  验证=snapshots 从 11 → 116/pid 级（funnel 232 条，stats 写到 31200 后靠
  decode-drain 续命）。**教训（第六条取证条律）：用 s.find 锚定"多行语句中间行"
  的编辑必须先 grep 该文本出现次数；移动探针=删旧+插新两次独立编辑，不是一行替换。**
- **修复 step_end 后的新前沿**：stats 冻结点从 45057（刀1）提前到 31200（~step 8），
  funnel 分布显示 tok=None×82：state 在中段消失而调用继续。seal 豁免（本轮第二修，
  证据支持弱）没有改变冻结点 → 我的"seal 清零 _stats_next"假设**未被证实**，记录
  待查。真凶候选：grace=8 的 sweep_release——请求若在任意 8 连续步缺席
  （async 双阶段里 `_kvmem_req_ids` 的某阶段可能不含 prefill 请求）就会被中程释放。
  **下轮唯一取证**：release() 入口无条件栈事件（前 6 次/pid，含调用方栈），一发
  点名 31200 时刻是谁按下的按钮；若是 sweep，则 grace 的缺席判定要改用
  "scheduler 报告过该请求的步数"而非"连续缺席计数"。
- 本轮其余落档：EXEC_POST 复原；seal D 豁免（保留：ledger 连续性论证独立成立，
  即便非本案元凶）；decode 桶路径早退（消除 n=0 雪崩的正当性来自 decode 行走
  harvest_decode_rows 专职通道）；RPC 补读删除（与 step_end 竞争同一缓冲）。
- 预算：13 boots。纪律：停。交还。


## 34. Round 25 案卷：三处机械修正落地；final 刀疑点收窄到两点

已修（本轮）：
- decode 桶路径早退**删**（stats 冻结时它把 decode n=1 的正常入桶也拦了；
  原 clamp 分支健在，语义回归 v0.5.2 行为）；
- 复核确认 margin 律现场文本**已经是正确形态**（row-bound only for final）——
  本窗 overrow×10 是 MID 刀被 margin 律正当拒绝（空转刀），不是 final 被害；
- death-watch 三处全装（rethrow=0 事件 → worker-respawn 假设**证伪**）；
- 上轮"state 蒸发 31200"被本窗数据推翻：主请求 funnel 到 call 57 tok 31212
  一路健康——那是把**前一请求（88K×6 混跑窗口）的终盘**误读成主请求。
  **教训（第七条取证条律）：跨请求窗口必须按 rid 分组后再读时间线。**

final 刀未落的剩余疑点（两点，下轮一发分辨）：
1. raw=True 是否真在 runtime 生效（recall.scores 调用点的 selection_blocks
   截断行为——d_final_scores 若再现且 core_len 仍 704 = raw 没生效）；
2. (b) 在 88K 短窗口只 fire 一次（d_b_probe=1）而那一发的 pressure 标记与
   进入段是否吻合（enter 事件按 rid+pressure 分组统计）。
探针方案：enter 事件已有的话，加 recall.scores 入口的 blocks 参数值事件
（每请求首次），一发同时读 1+2。

预算纪律：本轮 6 次 boot，停止。产线恢复 spec4/PLAN_D=0。


## 35. Round 26：旧机器链锁死 + 终末问题视图锁落地；问题收窄到"视图→decode 读"最后一环

本轮定案成果：
1. **stage-2 rebuild 链在 D 臂彻底锁死**（_rebuild_select 函数级守卫 + finish 驱动
   出口丢弃队列项）：rebuild_select_diag/compact_rebuild_queued/reconstruction_started
   三事件全归零（此前每轮 6-8 条 - v0.5.2 机器在 D 臂的幽灵执行被消灭，
   目标"删除旧机器"完成此项）。
2. **raw 分数覆盖证实生效**：core_len 704→1178、block_scores_len 29→**49/49 全覆盖**，
   且 **needle 块 39 打进 top8 rank3** - 问题打分正确命中。
3. **终末问题视图锁（TQL）**：问题刀落地后 MID 刀拒绝对该请求再压缩
   （worker 端 d_question_cut 即时标记 - Core 端 final_cut 要晚一个 RPC 回复，
   实测拦不住紧随的下一步）。双发同 sha 确定性成立，applied 序列含问题刀。
4. 期间两次"我以为写了"（cw1/fl3 锚败未写盘）都被诚实事件数据当场拆穿 - 教训：
   每处编辑后 grep 目标文本计数是唯一可信的落盘证明。

**剩余唯一问题（精确）**：问题视图已安装、锁定、确定性稳定，但 needle 仍 False。
病灶候选只剩两枚：(a) 安装后 view_lens/deltas 表的值与视图内容不一致（decode 读错行）；
(b) 问题刀 install 的页集与 kept 列表不一致（restore 未真正搬到 needle 页）。
下轮第一诊断：问题刀落地时快照 st.logical 是否含 needle block 的页 + 表值三元组
(view_base/delta/needle_page)，一发分辨 (a)/(b)。

## 36. Round 27 案卷：刀无罪（needle 页恒在视图内），病灶=读路径表值

- coin 三轮修正后终判：`needle_page=1264 in view=1, logical_len=869, visible=27808` - **页坐标核查（40471//16），非块坐标**（我先 39*64 的块算术差点冤枉了刀）。刀安装、锁定、覆盖全部正确。
- **D 读路径唯一剩余嫌疑**：问题刀落地后 MID 刀持续 republish 表值（delta 随流增大到 17248+），若 decode 的视图读表在**某一步后不再 republish 或被 flush 顺序错位**，attention 就按错 delta 取行。病灶=表值时间线与视图安装的一致性 - audit 探针因缩进毁损被还原未遂，下轮第一步在 remobilize 与 republish 两处（以 12/16 空格正确缩进）装 slot 解析审计（needle 真位→slot 应得视图位置 vs delta 公式实际给出），一发定最后案。
- 本轮 boot 13 次，严重超预算；停止。


## 37. Round 28 案终：设备内容全程真实；时序定罪——问题刀晚于答案落锤

- needle_trace 直通（KVMEM_TRACE_PAGES）后内容审计铁证：
  * pre-knife：`compacted_slot=null`（needle 页根本不在视图内 - MID 选择按统计不含它，合法）；
  * question knife 安装后：`device_sha == store_sha` 全程一致 - **读路径无罪、内容无罪、delta 无罪**。
- 唯一病灶=时序：问题刀在 prompt 完成步 (b) 落地，但 `prompt 完成 → 第一个 decode 步`的
  窗口里答案已按"无 needle 页视图"生成（max_tokens=16 全部在问题视图安装前 commit）。
- 修法（R2b 最后闸门，下轮施工）：问题刀提前到"问题被冻结的瞬间"安装 - 即 prompt 最后一
  chunk 完成后、第一次采样之前（Core 端：scheduled_new_reqs 的最后一 chunk 步之后立即触发
  (b)，或 worker 端 prefill_done 首个 decode 步之前置于采样钩子内）。安装点提前一步，
  答案的第一 token 就出生在带 needle 页的问题视图里。
- 至此方案 D 的每个部件都有实证无罪/有罪判决：刀✓锁✓覆盖✓表✓内容✓ - 只剩这个安装时机。


## 38. Round 28 终节：push 恢复 + 最后闸门的物理陈述（R3 设计裁决点）

- GitHub push 恢复（21 commits 一次补上，origin/main=7c8fb2c）。
- 时序闸门的完整物理：(b) 挂在 Core.step 末尾 = `result=original_step` 之后，而**第一个
  采样 token 在 prompt 完成步的 execute_model 内部已产生**——问题刀在其后安装救不回它；
  且**问题向量本身来自最后 chunk 的 forward harvest**（q_rows），最后 chunk 之前不存在
  "已冻结的问题"——**纯安装提前无解**。
- v0.5.2 靠 child rebuild（重 prefill 末段）跨过此闸；D 禁 rewind/child，等价能力需要
  "末段视图重写 + GDN 状态回置"——GDN 单槽性质使其无法在不动 token 流的前提下重算末段
  （重 forward 会覆盖 state——除非有 D6 StateImage 引擎钩子）。
- **裁决候选（R3 议题，非本轮 hack）**：
  A. 采样器让路：prompt 完成步**不采样**（scheduler 把完成步排为 forward-only，
     采样推迟一步；问题刀在步尾安装 → 第一 token 出生在问题视图）。改动在排程语义，
     不动 GDN/state——**D 不变量完整**，vLLM 有 forward-only 步先例（chunked prefill 步
     本就不采样）。
  B. 双答案窗口：接受首 token 出生统计视图，从第二 token 起问题视图（needle 若是
     长思维链答案，16-token 窗口里可能已足够 - 需实测）。
  C. StateImage 引擎钩子（D6 完整版）：末段重算 + 状态快照回置 - 改动最深，收益最大
     （顺带把 R3 resume 深度一起解决）。
- 倾向序：**A 优先**（最小合法）、C 并入 R3 设计、B 作对照实验。


## 39. Round 29：step 间缝不存在（nosync），候选 A 真形态=worker 内 pre-sampler 刀

- 装 step 级 seam（forward→sample 之间下刀）：d_seam_knife 零事件 + 完成步返回
  executed=True - **本 fork nosync 调度：EngineCore.step 同步包含采样**，Core 层
  不存在 forward/sample 消息间隙。88K 80s 完成（seam 层无副作用，保留作 async 模式保险）。
- **候选 A 真形态（下轮施工）**：间隙在 worker 内部 - GPUModelRunner.execute_model 的
  forward 完成后、sampler 调用前。且 D 刀的安装**本来就全是 worker 侧动作**
  （选择/restore/row 写/表发布皆不碰 Core 账本）- **worker 本地 pre-sampler 刀**合法：
  完成步 forward 后（问题向量此刻刚被 harvest），runner 钩子调 runtime 的
  `d_local_question_knife(req)`：确定性选择（raw 分数、两 rank 同输入同结果 - consensus
  在 D 下已跳过）+ remobilize 安装 + 表发布，然后才 sampler - **第一 token 出生在问题视图**。
  风险登记：TP 两 rank 需同轮执行（确定性输入成立）；与 (c)/(b) 的幂等（final_cut 标记已有）。
- 本轮预算：4 boots 止，seam 保险层代码保留（D 门内无害）。


## 40. Round 30：seam 刀全链运转；候选 A 判死——首 token logits 物理先于刀

- slots 血案（`__slots__` AttributeError 杀光两臂 2 轮）修复入 `__slots__` +
  try/except fallback 双保险；离线 6/6。
- 本 boot 刀链全绿：`d_seam_local_knife after:True`（错误 0）、blockwise_reselect×2、
  锁生效、答案从复读机变 "user is is"（token2+ 确实活在了问题视图里）。
- **候选 A 的物理判决（定案）**：完成步的 logits 由 forward 在装刀**之前**算出——
  pre-sampler 刀救不了第一个 token；16-token 思维链被错首词带偏，needle 到不了输出。
  v0.5.2 靠 child rebuild 重 prefill 整个 prompt 才有正确首 token。
  **D 不变量（永不 rewind/无 child）下唯一出路 = D6 StateImage 重评原语**：
  末段 forward 重放 + GDN 状态在 cursor 处快照/回置（token 流与账本不动，
  重放是幂等再计算）——这正是 R3 resume 深度的同一个机制，一并施工。
- seam 刀保留（token2+ 的正确视图 + 锁的终末语义，都是 R3 重评的地基）。
