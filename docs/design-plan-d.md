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
