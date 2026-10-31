# MTP draft-ledger 协议设计（KVMem resident × 投机解码）

状态：**设计稿，未实现**。日期 2026-10-31。仓库 /home/srj/src/kvmem-vllm-adapter，
引擎文件 `engines/vllm/vllm/v1/core/sched/scheduler.py`（行号以 HEAD 001317a 为准）。
触发事件：MTP/4 下短请求首个 draft 步崩 `assert num_scheduled >= num_logits`
（实测 scheduled=1 vs logits=5，三次复现，见 /tmp/kvmem_assert.jsonl）。

---

## 1. 账本清单（谁写、谁读）

| 账本 | 所在 | 写入者 | 语义 |
|---|---|---|---|
| `num_computed_tokens` (Core) | Request | `_update_after_schedule`:1937 += 全行；`update_from_output`:2457 -= rejected | **行账**：已进过 forward 的行数（含被拒草稿，拒后返还） |
| `num_logical_tokens` (KVMem) | Request | 同上 :1938 += 全行；:1944 **无条件** = computed+offset；:2465 -= rejected | 设计意图 = **served 账**（已采样为真实 token 的前缀） |
| `num_kvmem_offset` | Request | 压缩 recompute 时置 | = logical − computed 的**唯一合法来源** |
| `request.spec_token_ids` | Request | `update_draft_token_ids`:2909（drafter 回执） | 待发草稿池；:1030 被消费置空 |
| `scheduled_spec_decode_tokens` (dict) | SchedulerOutput | 运行循环 :1043 / 准入 pad :1535 | **本步实际占用的草稿行**，即 logits 的草稿来源 |
| `num_scheduled_tokens[req]` | SchedulerOutput | 运行循环 :1015 | 本步 query 行数 = num_new_tokens |
| worker `draft_tokens` book | ReqStates(states.py:88) | add :146 清零 / 每步重写 | **与 draft 字典无关**；spec_short(1,4) 时它非空而 logits 仍取 1，未被采用 |

runner 契约（model_runner.py:1370-1375）：
`logits 行 = 草稿行数 + bonus(1)`，且 logits 行放在 `[query_end − logits, query_end)`，
所以必须 **scheduled ≥ logits**。

---

## 2. 根因（四重，全部实测）

R1 **无条件相等毁掉 sampled 语义**：`num_logical_tokens = num_computed_tokens + offset`
在 offset=0 的普通 decode 请求上恒立 → logical 被草稿行推进（虚增最多 K），
又随 rejection 回退（而 sampled 账本里草稿从未进入 → 双重扣减）。
boundary gate（1030 上方：`logical <= prompt` 才禁草稿）与 prefill_frontier_step
（chunk 预算）全部被虚增账本骗过。

R2 **两套账本被强行合一**：行计数（要含草稿、拒后要退）与 served 计数
（草稿未 served，永远只由 sampled ack 推进）在**有无草稿在飞**时相差
`0..num_spec_tokens`。1944 注释（"唯一合法差异是压缩 offset"）正是 R1 的出处——
它对**无投机**世界成立，MTP 引入第二种合法差异。

R3 **pad 与 draft gate 打架**（waiting 路径，准入循环 1276-1343）：gate 刚把
`spec_token_ids` 清空、证明"本步一行草稿都不该有"，pad 却仍把 `num_new_tokens==1`
补成 `1+num_spec`，经 1526 断言 → 1531 `[-1]*num_spec` 写出**幽灵草稿字典**，
runner 按草稿数加 bonus 得 logits=5，scheduled 被别处削成 1。
已提交的 001317a 只守卫了运行循环侧，**waiting 侧同源缺口仍在**（行号上移 1286）。

R4 **准入公式丢草稿行**（1272）：非 resident 分支 `num_tokens − computed` 不含
spec 行（上游用 `num_tokens_with_spec`）；resident 分支换用 logical 后又与
`num_new + computed − num_tokens`（:1034）的**行制**草稿计数口径互斥——
草稿在字典与不在字典的两种步各错一次。

---

## 3. 目标不变量（协议核心）

**I1 双账本分离**
```
computed = logical + D + E
  D = 在飞草稿行（≥0，≤ num_spec_tokens）
  E = 压缩 offset（≥0）
```
任何一步二者不得混淆：D 由"草稿字典写入/回执"驱动，E 由"压缩 recompute"驱动，
二者都不许改 logical。

**I2 logical 推进律（served 账）**
scheduled 步：`logical += (num_scheduled − 本步草稿行数)`。
拒绝回滚：**不动 logical**（草稿从未进入 sampled 账）。
唯一例外：压缩 offset 置位时 `logical = computed + E`。

**I3 行计数一律用行制账本**
`num_new_tokens = num_tokens_with_spec + placeholders − computed`
（resident 下**不得**替换为 logical——行账本才是 forward 行数）。

**I4 草稿与压缩请求互斥**（已存在，保留）
`offset > 0 || rebuilt` → `spec_token_ids = []`。

**I5 draft gate 只读 logical**（83437a1 语义不变）：
`logical ≤ prompt` 的步禁止草稿，**任何路径不得补出草稿行**。

**I6 草稿行必须与 query 行同源同数**
字典写了 K 行草稿，则 num_scheduled ≥ K+1；反之 gate 清草稿后字典必空。

---

## 4. 改动点（文件:行 → 改成什么）

全部集中在 `engines/vllm/vllm/v1/core/sched/scheduler.py`。每处独立可回滚。

### C1 `_update_after_schedule`（1937-1944）：双账本分离
```python
# 删 :1938 的 `num_logical_tokens += num_scheduled_token`（无条件行制递增）
# 删 :1944 无条件等式 `logical = computed + offset`
# 改为：
_logical_inc = num_scheduled_token - len(
    scheduler_output.scheduled_spec_decode_tokens.get(req_id) or ())
if _logical_inc > 0:
    request.num_logical_tokens += _logical_inc          # sampled 账本只进 sampled 行
if self.kvmem_resident and request.num_kvmem_offset:   # 仅压缩请求保持旧等式
    request.num_logical_tokens = (request.num_computed_tokens
                                  + request.num_kvmem_offset)
```
理由：草稿行推进 computed（行制、可回退），永不推进 logical。压缩请求禁投机（I4），
故等式只在 offset≠0 时有意义且此时 D 恒 0，I1 闭合。

### C2 rejection 回滚（2464-2466）：不动 logical
```python
if request.num_logical_tokens > 0 and not self.kvmem_resident:
    ...  # 原逻辑保留给非 resident 引擎（无 KVMem 账本世界）
```
resident 下草稿从未进 sampled 账本，双重扣减会开永久缺口。

### C3 waiting 路径 pad（1276-1300 区间）：把 001317a 守卫搬过来
在 `if (... num_new_tokens == 1 ...)` 的与链尾加
```python
and not (self.kvmem_resident and request.num_logical_tokens
         <= request.num_prompt_tokens)
```
**并加防回归守卫**：1535 `[-1]*num_spec` 写入前断言
`not (kvmem_resident and request not in scheduled_spec_decode_tokens)` 之类——
幽灵字典从此不可能（见 C6）。

### C4 准入公式（1272）：行制 + spec
resident：`num_new_tokens = request.num_tokens_with_spec - request.num_computed_tokens`
（行账本 computed，**不是** logical；prefill chunk 无草稿时 with_spec≡num_tokens，
与旧行为一致）。非 resident 分支维持 fork 原样。
草稿字典计数（:1034 已用行制 computed）随之自洽：
`num_scheduled_spec = new + computed − tokens − placeholders` 对 resident 同样成立。

### C5 running 主循环公式（826-829）：删 logical 替换
上游本就 `with_spec + placeholders − computed`；KVMem 换 logical 是为压缩账本
（注释所述"防重排已 served 的 token"）。**改为**：压缩/重建请求因 I4 无草稿，
行账本差异恰是 offset，`computed` 直接用即可（KV 长度语义），无需 logical——
logical 只留给 I5 的 gate。826 恢复上游式。
（若实测压缩请求 prefill chunk 重排问题复发，则保留 resident 分支，但草稿请求
走非 resident 式——按 C6 验证决定。）

### C6 统一断言（临时诊断，验证后撤）
pad 写点和 dict 写点各加一行：把 (req, num_new, len(dict), computed, logical, offset)
追加 /tmp/kvmem_sparse.jsonl，kind=`draft_ledger`——用于离线回归比对，不是新协议。

### 不动的部分
worker ReqStates `draft_tokens` book（R3 证明它不驱动 logits）；
runner 计数式；83437a1 gate 本身；`_reserve_prefill_lookahead`（multi-module MTP
的 chunk 末端留白，lookahead=num_spec 时只作用 prefill 步，与 decode 草稿不冲突；
若本模型 `use_multi_module_mtp()` 为 False 则 lookahead=1 近 no-op——验证轮确认）。

---

## 5. 失败语义

- 任一门不满足 → 该步**清草稿**（字典空、scheduled 不含 draft 行），而非崩溃、
  而非静默少一行。
- I5/I6 破坏 = 协议 bug，必须 `draft_ledger` 记录现场；
  普通超预算 = 正常等待，不记。
- C2 与 C1 顺序耦合：先 C1 后 C2（否则 resident 下 logical 被无条件递增+双扣，
  缺口更大）。

## 6. 验证计划（每轮一个假设）

1. **离线**：新 `tests/draft_ledger_test.py`（CPU 模拟 Request/SchedulerOutput，
   无 GPU）：prefill 完成→首个 decode 步（gate 未开、pad 被跳）→第二 decode 步
   （dict=K, logits=K+1, scheduled≥K+1）→拒绝回滚不扣 logical→
   压缩 offset 与草稿互斥。这是唯一能便宜跑的门。
2. **GPU 短请求**（82 token 级）：HTTP 200 + 输出正确 + tok/s 对比 spec=0 基线。
3. **GPU 超池**（88K）：压缩/rebuild/resume 全路径 NEEDLE + 字节一致（回归）。
4. 撤除 C6 诊断，打 tag（仅在全部通过后）。

## 7. 已知边界（本设计不解决）

- session checkpoint 封存的 draft StateImage（所以 `KVMEM_SESSION_REUSE=1`
  与 spec>0 的 SystemExit 闸门**保留不动**，直到该协议单独设计）。
- enforce_eager：本站点必须 eager（V2 Model Runner 路径），
  `spec_short` 类 cudagraph 形状问题不会自动消失。
- 多模块 MTP 的 StateImage 快照时序（decode 步草稿被拒后循环状态的回滚语义）
  ——若短请求验证输出漂移，此项升为下一个设计题。

## Amendment (2026-11-01, after GPU evidence - SUPERSEDES the closed-form section)

1. FINAL LEDGER (implemented, engine e3ddb28): the MILESTONE books are the correct
   draft ledger once the ordering/pad fixes are in: `logical += scheduled rows` at
   schedule time, `logical -= rejected` at output time. The closed loop equals the
   acknowledged count. The closed form `logical = num_tokens` is WRONG for the
   beyond-pool path: during a long prefill it jumps to the whole prompt while the
   worker executed book tracks committed rows, so the quiescent equality protocol
   (want == have) never matches and the rebuild never fires. Evidence: 88K/MTP with
   closed form = HTTP200 + 3 tokens, `quiescent_compact_stale want=86686 have=4096`;
   the CONTROL at SPECULATIVE_TOKENS=0 fails IDENTICALLY - the ledger change itself
   was the regression. With the restored loop: 88K/MTP NEEDLE correct, 151s, zero
   stale, reconstruction fires.

2. OPEN DEFECT (MTP losslessness): at temperature 0, greedy spec decode must be
   LOSSLESS, and it is on short requests (multiplication table byte-identical to
   the milestone outputs across repeats), but the 88K beyond-pool output diverges
   from spec=0 in the reasoning prefix (content answer still correct). Suspect:
   the MTP drafter re-attends the COMPACTED row (88K > the 60,480-token row) while
   the draft state is never rewound on rejection (no mamba/state restore exists in
   the speculators); prefill lookahead slots the drafter also reads the compacted
   table. This is design question 2 of this doc, now measurable on short prompts
   (acceptance vs GDN state contamination) - fix belongs in the drafter state, not
   the scheduler ledger.

3. CONFIG LAW unchanged and still hard-gated: session-reuse + spec>0 must not
   start (the checkpoint seals GDN state; the draft contamination above makes the
   sealed state non-comparable between spec modes). Lift only after (2) is fixed.

## Amendment 2 (2026-11-01, round 9): the losslessness defect is FIXED at the view layer

Root cause (measured, bisection rounds 5-9): `VllmSparseRuntime.before_step` published
`step_end = st.seq_len + len(position_slots)` as the attention book for EVERY request with a
row - compacted or not. Under speculation the lagged runtime book (driven by the SCHEDULED
rows of the previous step) already covers the committing step, so the formula double-counted
the 1+K draft rows and the override SHRANK the attention book below the engine's real rows.
Greedy MTP diverged from spec0 from the FIRST draft step, on short prompts too. The upstream
Definitive fork (same weights, MTP4) was byte-equal with spec0, and the no-plugin / hooks-off
KVMem builds were lossless - the isolation matrix is recorded in commit 43f0689.

Fix (commit 54a6900): the seq/positions override is REWIND-ONLY, gated on
`applied_view_coord` (set exactly at the real apply points: `book_rewound` on EngineCore
rewind, the stage-2 rebuild, the prefix-mode apply). Un-compacted requests keep the engine's
own books and still get the block-table row published.

Verification (GPU, spec=4, resident, full hooks, greedy temp0):
- short A: c2f3644600da41bb == spec0 baseline; speed 38.5 tok/s (~2.9x over spec0)
- short B: 426751992d476adc == spec0 baseline
- 88K beyond-pool: NEEDLE ZX-4417-QJ7 True, 70 tok in 150.5s, no 3-token stall, content
  byte-equal with the spec0 run
- offline suite: all 7 tests green (new regression UncompactedKeepsEngineBooks)

Still open: MTP x session-reuse composition (seal-time draft gate + StateImage-at-accepted
prefix), diagnostics cleanup, tag stage4-mtp.

## Amendment 3 (2026-11-01, round 10): MTP x session-reuse composition experiment

Lifted both refusals (worker SystemExit + seal gate spec>0 term) and ran the composition
end-to-end (spec=4, reuse=1, 88K prompt via `X-Session-ID`):
- prefill-boundary StateImage capture SUCCEEDS under MTP: `session_checkpoint_captured`
  logged at the frontier; the scheduler's boundary gate keeps the entire prefill
  draft-free, so the books match exactly (the old "StateImage law" premise - draft rows
  advancing the GDN main slot - is disproved by the fork's own spec_state_indices
  scratch-slot isolation, and by Definitive MTP4 being byte-equal with spec0).
- turn1 NEEDLE correct (ZX-4417-QJ7), turn2 KEY-LATE answer correct.
- BUT: the terminal PARK fails the TP barrier - `prepare_session_checkpoint` yields
  non-dict records on at least one rank -> "TP workers did not all prepare a
  checkpoint" -> nothing is parked and turn2 silently re-prefills (150.9s = full
  prefill, resume never hit).

Both refusals are therefore RESTORED fail-closed, with the evidence written into the
messages (worker.py) and this amendment. The remaining gap for stage4-mtp is exactly
one protocol: make the terminal park TP-safe under spec>0 (prepare/readiness semantics
for a request whose final step carried drafts). Until then: production combo remains
spec=0 x reuse=1 (milestone stage3-session-reuse), and MTP runs without reuse
(validated this milestone: short requests byte-equal + ~2.9x, 88K beyond-pool NEEDLE
correct, offline suite green).
