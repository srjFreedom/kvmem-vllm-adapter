# KVMem × vLLM — dual-GPU SM75, high-throughput, single service

**English** | [中文](README_zh-CN.md)

**kvmem's policy layer running *inside* vLLM, on two Turing (SM75) cards.** This is the
high-performance dual-GPU build: instead of a gateway in front of the engine, kvmem's entry policy
runs in the vLLM API process, and the engine itself is tuned for two consumer 22 GB cards —
MTP speculative decoding, FP8 KV cache, prefix caching and a KV pool sized for N concurrent
long-context requests.

What that buys you on 2× RTX 2080 Ti (all numbers measured on the v0.4.0 production instance):

- **CUDA graphs + MTP speculation, at full speed.** Decode **84–85 tok/s** on a 2,000-token
  countdown with `spec=4` — inside the upstream reference band (70–80) and ~1.8× the old
  eager-only adapter path (46). The KVMem scoring bridge is graph-capturable (static buffers,
  in-place ops, bit-exact with the eager numerics).
- **Beyond-pool context, by default.** Resident prefill + in-request compaction are ON out of the
  box: an 88K prompt on a 36,864-token resident window answers a mid-archive needle correctly;
  the pool itself holds 201,438 tokens at `GPU_UTIL=0.972`.
- **Session reuse.** With `X-Session-ID`, a finished conversation parks its prefix KV + recurrent
  state into host RAM; the next turn resumes from the frontier and skips ~34 s of prefill on the
  85K archive (needle answered identically through either path).
- **One service, one port (18200).** OpenAI-compatible API, `/metrics`, `/docs` and the web UI on
  the same origin — no sidecar gateway, no second process to keep alive.
- **kvmem stays the policy layer.** It decides how much history is sent (B), which pages stay
  resident (retrieval over the real pre-RoPE Q, mean-K scoring, LRFU), and what lives in the H/D
  tiers; vLLM remains the engine.

```
client ──▶ :18200 ┬─ /v1/*        OpenAI-compatible API
                  ├─ /metrics     engine metrics
                  ├─ /docs        OpenAPI
                  └─ /            static UI (swappable)
```

## Performance

Measured on **2× RTX 2080 Ti 22 GB** (NVLink bridge, NCCL uses it automatically), TP=2, `N=1`,
`B=36,864 + R=16,384`, MTP `spec=4`, FP8 KV, CUDA graphs on (`KVMEM_FORCE_EAGER` unset),
`VLLM_SM75_SPEC_SYNC_MODE=nosync`, `temperature=0`, same-instance A/B against the pre-graph build:

| workload | v0.3.x (forced eager) | **v0.4.0 (compiled + graphs)** |
|---|---|---|
| 2,000-token countdown decode | 46.0 tok/s | **85.0 tok/s** |
| 3,000-token survival run (two 1,616-block boundary crossings) | 43.5 s/2,000 | **35.4 s/3,000** (84.7 tok/s) |
| 85,675-token archive, cold first turn | ~150 s | **143 s** |
| same archive, second turn via session resume | 115.7 s | **110.0 s** (−34 s vs cold re-run) |

Acceptance gates pinned by the test suite: short-request losslessness (multiplication-table and
"random two-digit" outputs byte-identical across spec0/spec4, shas `c2f36446`/`42675199`),
88K mid-archive needle `ZX-4417-QJ7`, and per-path determinism (two cold runs byte-equal;
resume output byte-stable across boots and eager/graph switches — see Known limitations for why
resume-vs-cold byte equality is not a property of this design).

## Architecture

**Engine side (inside the vLLM processes)**

| Component | Role |
|---|---|
| `plugin/vllm_kvmem_plugin.py` | `vllm.general_plugins` entry: probe, runner hooks, sparse-runtime wiring; loaded in every process |
| `src/kvmem_sparse_runtime.py` + `kvmem_sparse_orchestrator.py` | per-page H (host RAM) / D (disk) tiering, LRFU eviction, mean-K retrieval |
| `src/vllm_kvmem_ext.cpp` | native extension (links the kvmem library): `VllmMemoryBackend`, `NvmeKvTier`, retrieval index |

**Entry side (same process, official `vllm.endpoint_plugins` extension point)**

| Component | Role |
|---|---|
| `plugin/kvmem_entry_middleware.py` → `KvmemEntryPlugin` | `attach_router`: installs the middleware and mounts the UI |
| middleware (default `mode=middleware`) | standard Starlette body rewrite through the `request._body` cache: select history to B, cap `max_tokens` at **R** (or **`CTX`** with in-request compaction on), enforce CTX, H/D history-block cache |
| `mode=hook` (fallback) | wraps `OpenAIServingChat.create_chat_completion` and applies the same policy to the parsed request |
| `src/kvmem_entry_policy.py` | the single policy implementation both modes share |

The response path is never touched, so **streaming, CORS and error handling are vLLM's own**.

### Why selection happens at the entry

vLLM indexes KV blocks by logical position and assumes a monotonically growing sequence: truncating
mid-request naively desynchronises its scheduler (measured: the `num_scheduled_tokens >= num_logits`
assertion, then a hang once speculative decoding is off). By default kvmem therefore decides "keep only
B" **before the request reaches the engine**, and the engine's sequence only grows.

### Super-R generation (opt-in, `--kvmem-inrequest-compaction`)

With that flag the engine runs a **resident-book** model: VRAM per request is `B + R` no matter how
long the conversation is. The request's two counters are split (fork patch, `vllm/v1/core/sched`):

* `num_logical_tokens` — monotone served count (prompt + generated). Drives scheduling.
* `num_computed_tokens` — the **resident KV book**. Compaction rewinds it while the logical book
  (`_all_token_ids`) keeps growing, so positions, `seq_lens` and block allocation stay inside the
  working area.

Because the engine never thinks the request holds more KV than the pool can, it never allocates past
`B + R` and never preempts a long context:

- **Prefill is chunked and compacted too.** A prompt longer than B is prefilled in chunks; the
  runtime reselects as the working set grows, spills the pages that are no longer needed to the host
  tier, and compacts **inside the request's own engine blocks** (never by taking fresh pages from the
  shared pool). No prompt token is dropped — the entry does not trim when the flag is on.
- **Decode uses the same working set**, refreshed every R output tokens.
- EngineCore only agrees on the resident length (`num_computed_tokens` = compacted length) over a
  worker → EngineCore channel; the row itself is written by the runtime in the request's own pages,
  padded to the engine's block quantum so its next block append stays contiguous.

- **Default (flag absent): the generation ceiling is R** and the entry selects history down to B.
  `max_tokens` is clamped to R and a request stops there; the sequence is monotone and the pool is
  never recycled.
- **With the flag: the ceiling is `CTX`**, the entry stops trimming, and one switch drives
  everything: the entry ceiling, the engine-side compaction, the block channel and synchronous
  scheduling. Prefix caching must stay off with compaction on (the block hashes hash logical tokens
  while the KV layout is compacted), so the worker forces it off and says so.
- Reaching `CTX` is an ordinary `max_tokens` stop (`finish_reason=length`); the response text is
  never truncated, because compaction only rewrites the engine's token book, not the output tokens.

### Where the retrieval query comes from

`torch.ops.kvmem.capture_q` captures the **real pre-RoPE Q** of the current step inside the attention
layer (last 512 tokens by default, `CAPTURE_TOKENS`); kvmem scores pages against it with mean-K, and
LRFU decays history. The reference backends (llama.cpp / ninfer) use the token span of the last user
message instead — both rely on model-internal representations, they only differ in which tokens they
aggregate.

## Quick start

Three external pieces are required and are not vendored (`backends/versions.json` pins them):

| Piece | Pinned version | How it is obtained |
|---|---|---|
| vLLM SM75 fork | `93c9013e375b` + integration patch | `install.sh` fetches `base_commit` into `engines/vllm` and replaces it with `backends/patches/vllm-kvmem.patch` (verifies `patched_tree`); existing forks are never modified unless you pass `USE_LOCAL_FORK=1` |
| kvmem sources | tag `v0.19.0-rc1` | same (the native extension needs its headers and library) |
| Model weights | **any model vLLM can serve** (this deployment runs an FP8 + MTP export) | you provide them; point `MODEL` at the directory |

```bash
git clone https://github.com/srjFreedom/kvmem-vllm-adapter
cd kvmem-vllm-adapter

USE_LOCAL_FORK=0 ./install.sh            # fetch engines → patch → plugin → extension → self-check
ENGINE=$PWD/engines/vllm ./scripts/setup-venv.sh   # torch / CUDA (several GB; skip if you have a venv)

# Run it in the foreground with the repository defaults (default profile included):
MODEL=$HOME/models/<your-model> ./start-kvmem-vllm.sh

# Or as a service (the unit's ExecStart points at start-kvmem-vllm.sh):
sudo ./deploy/install-service.sh
mkdir -p ~/.config && printf 'MODEL=%s\n' "$HOME/models/<your-model>" > ~/.config/kvmem-vllm.env
sudo systemctl daemon-reload && sudo systemctl enable --now kvmem-vllm
scripts/wait_ready.sh 18200 900 --smoke
```

Already have the environment? One command (skips fetching and the venv):

```bash
VLLM_ROOT=/path/to/fork KVMA_SRC=/path/to/kvmem PY=/path/to/venv/bin/python ./install.sh
```

**The three steps that are never skippable** (they are this repository's own code): apply the patch,
install the plugin, build the native extension.

### Which models work

vLLM decides what can be loaded (architecture, quantization, chat template); the KVMem layer adds
two requirements on top:

* **Per-page tiering** (H/D spill, LRFU) works on the full-attention KV groups, so hybrid models
  (attention + GDN/Mamba) are fine — that is what the default model is.
* **Retrieval scoring** needs the pre-RoPE query hook, which lives in
  `vllm/model_executor/models/qwen3_next.py` (part of the integration patch). Another architecture
  needs the same two-line call inside its attention layer; without it the entry policy, R ceiling
  and history cache still work, only mean-K page scoring is unavailable.

So: any vLLM-supported model runs, and FP8 + MTP is just the configuration measured here.

## Configuration

Two layers, and the defaults are a working start — no site script required:

```
start-kvmem-vllm.sh 0.3.0 configuration (kvmem budgets + vLLM window); systemd runs it
run-current.sh      generic entry, no site parameters
   │ exec
run-kvmem-vllm.sh   implementation (every knob is ${VAR:-default})
```

Defaults: `B=36864 R=16384 N=1 GPU_UTIL=0.972` (with `KVMEM_SESSION_REUSE=1`,
the engine window follows CTX (clamped at the model-native 262,144), `VLLM_SM75_SPEC_SYNC_MODE=nosync` and MTP `spec=4` - the
combination validated end-to-end in v0.4.0) with the shipped
`2x2080Ti/qwen27b/w8a16/mtp4-fp8kv-1x184K-text-image.env` profile. The only required input is
`MODEL` (your weights directory); set `CHAT_TEMPLATE_FILE` only when the model ships no built-in
chat template. Override anything else with environment variables, or put
`KEY=VALUE` lines in `~/.config/kvmem-vllm.env` (the unit loads it when present):

```bash
CONCURRENCY=2 BUDGET=53248 GPU_UTIL=0.98 ./start-kvmem-vllm.sh
```

Full parameter reference: **[docs/PARAMETERS.md](docs/PARAMETERS.md)** (Chinese).

```bash
B=36864 R=16384 CTX=524288 SINK=8192 RECENT=2048 N=1 PORT=18200
CAPTURE_TOKENS=512 GPU_UTIL=0.97 MAX_BATCHED_TOKENS=4096 THINKING_BUDGET=
PROFILE=2x2080Ti/qwen27b/w8a16/mtp4-fp8kv-1x184K-text-image.env
MODEL=$HOME/models/<your-model>          # required
CHAT_TEMPLATE_FILE=                      # only if the model has no built-in template
KVMEM_UI_DIR=                            # optional static UI mounted at /
```

Generation ceiling: **`CTX` by default** - `run-kvmem-vllm.sh` ships with
`INREQUEST_COMPACTION=1` + `KVMEM_ENTRY_RESIDENT_PREFILL=1` (the validated beyond-pool path: the
entry does not trim, prompts longer than B are chunk-prefilled and compacted inside the request,
prefix caching is force-disabled, scheduling stays synchronous because compaction rewrites the
block table mid-request). `max_tokens` is clamped to `CTX`, not R. Set
`INREQUEST_COMPACTION=0` only to debug the classic entry-trim path (ceiling then reverts to R).

Single-prompt length tiers (v0.5.x): the RECOMMENDED default caps one prompt at about
262K tokens (inside the model position table, fully qualified). For genuinely longer
prompts - the original kvmem engine has NO position table (ggml computes RoPE
numerically) and admits 300K+ on this same model - opt in with
`KVMEM_ROPE_TABLE_CTX=<CTX>`: it lifts the RoPE table, engine window and entry
admission together. The default arm since v0.5.1 already satisfies this: a 282,142-token prompt
answered correctly end-to-end (395s) on the pressure arm, while the OFF arm dies at the
prompt-end rebuild past the pool (a known OFF-path boundary). Honest costs: the bigger
table takes ~134 MB per rank (KV pool 226,240 -> 214,928 measured) and positions beyond
the trained range are RoPE extrapolation - quality is the model's, exactly as in the
original. Remove the switch to return to the clean 0.9s HTTP 400 for oversized prompts.

v0.5.1: the pressure arm IS the default (beyond-pool context and multi-user chunking are what this adapter exists for). To fall back to the legacy prompt-end path (comparison/compat): `KVMEM_PREFILL_PRESSURE=0 ./start-kvmem-vllm.sh`.

Session reuse rides the same switch (`KVMEM_SESSION_REUSE=1` default): pass
`X-Session-ID: <id>` on chat requests and a finished conversation parks its prefix at the
checkpoint frontier; the next turn with the same id resumes from it.

Bounded in-prefill residency (**default ON since v0.5.1**): `KVMEM_PREFILL_PRESSURE`
makes prefill itself roll a watermark compaction (sink+tail+scored blocks, unselected pages
spilled to host) and rebuilds the selected view from host storage at prompt completion -
verified 2/2 at 250K tokens, beyond both the physical pool (226,240) and the historical
200,000 window (`ZX-4417-QJ7`, see docs/design-prefill-pressure.md §6.25-6.29). The arm is
fail-closed at boot: a pool too small for watermark+margin disables it automatically
(`pressure_disabled_geometry` in the sparse log) and the request falls back to the end-of-prompt rebuild
above. Long prompts additionally need `MAX_MODEL_LEN` above the prompt and
`KVMEM_SKIP_KV_CHECK=1` (the RoPE table itself sizes at the model's native 262,144).

The constraint that bites: **engine window = `N × (B + R)` must fit the KV pool** (201,438 tokens on
this box at `GPU_UTIL=0.972`; CUDA-graph buffers cost ~2.5K tokens versus eager, and 0.97 drops the
pool below `MAX_MODEL_LEN=200000` - measured OOM). Defaults (`N=1, B+R=53,248`) leave a large margin; `N=2` → 106,496, `N=3` → 159,744,
`N=4` → 212,992 which needs a smaller B (≈26K) or R.

`THINKING_BUDGET` is empty by default, i.e. **no cap on reasoning tokens**; set a number only if you
want to bound it.

Changes take effect on `sudo systemctl restart kvmem-vllm` (no hot reload); `kvmem-params.env`
records what is actually in force.

## Operations

```bash
sudo systemctl start|stop|restart kvmem-vllm
tail -f ~/kvmem.log          # engine and entry policy in one log
cat ~/kvmem-params.env       # effective parameters
curl -s localhost:18200/health       # health
curl -s localhost:18200/metrics      # vllm:* metrics (prefill/decode/KV usage)
```

Entry decisions are logged as `kvmem entry: {"kind":"entry_rewrite",...}` (with `gen_budget` and the
block/character counts of `prompt_selected`).

## Contract checks (not benchmarks)

| Item | Result |
|---|---|
| Logical context | 486,605-token history → `prompt_tokens=89,066`, both needles found (106 s; measured with B=65536) |
| Generation ceiling | compaction default-on → ceiling is `CTX` (the classic path falls back to R with entry trimming) |
| CTX | 747,614 tokens → `400 history is ~747614 tokens, above kvmem CTX=524288` |
| Streaming | 30 SSE events, first at 0.07 s, last at 1.43 s, monotonic |
| Ports | only `0.0.0.0:18200`; no sidecar process |
| Session reuse | park `retired frontier=32320` → resume `hit=32320` → prefill resumes at 32320+4096, −34 s wall clock, needle identical through both paths |
| Graph capture | compiled boot with CUDA graphs: no capture-invalidating ops in the bridge (offline guard `tests/bridge_capture_test.py`, bit-exact numerics) |

These rows pin the policy contract (how much history reaches the engine, the R clamp, the CTX
rejection, SSE behaviour); throughput lives in [Performance](#performance).

## Tests and probe

```bash
PY=$PWD/engines/vllm/.venv/bin/python
for t in tests/*_test.py; do PYTHONPATH="build/python:src:plugin" $PY "$t"; done   # 59 files
# the core contract suite (9) must stay green: quiescent_compact position_books store_semantics
# commit_guard sched_gate sparse_runtime sparse_concurrency session_clip bridge_capture
python3 scripts/check_integration_surface.py        # 28/28 vLLM/kvmem touch points
./run-gpu-tests.sh                                  # GPU mode (stop the service first)
```

## Repository layout

```
backends/vllm/worker.py     launcher wrapper (window N×(B+R), parameter plumbing)
plugin/                     general plugin + endpoint plugin (entry policy, UI mount)
src/                        entry policy, text selection, H/D cache, sparse runtime, native ext
scripts/                    worker, wait_ready, prepare-backends, setup-venv, probe
tests/                      39 tests (all non-GPU ones pass)
deploy/                     systemd unit, generic entry, launch scripts
backends/versions.json      pinned engine versions + patch sha256 + patched_tree
backends/patches/           every vLLM-side change (17 files / 612 lines)
docs/PARAMETERS.md          each knob, its constraint and its effect
INTEGRATION-SURFACE.md      touch-point list and source-change registry
```

## Known limitations

- **Entry-side text selection is a stopgap for the classic path only**; with the default resident
  entry the whole prompt reaches the engine and page choice is made by the in-engine mean-K scorer
- **In-request compaction is the default path** and is more delicate than the plain one: it forces
  prefix caching off and synchronous scheduling (compaction rewrites the block table mid-request;
  the async-scheduler ban is a correctness decision, re-evaluate only if throughput ever drops
  below the reference band again), retrieval has no score for freshly generated blocks (the recent
  tail is force-kept), and a step carrying speculative drafts is never compacted mid-request
- **Resume-vs-cold byte equality is not a contract of this design.** Each path is deterministic
  (cold×2 byte-equal; resume output stable across reboots and eager/graph switches) and both answer
  the needle, but the restored turn replays the parked page selection while a cold turn re-scores
  pages for its own query - different physical context sets, by retrieval design (measured:
  chosen-page heads differ at identical total page counts). The stored prefix KV is FP8
  round-tripped verbatim; a cold prefill re-quantises it. Greedy sampling can flip wherever those
  logits gaps cross a decision boundary
- **Session checkpoints are one-shot** (a resumed turn does not re-park the same frontier; a third
  turn re-prefills), and the store keeps `SESSIONS` records with LRU eviction - a third parked
  session evicts the oldest
- **First-turn needle misses** remain possible when the compaction scorer does not select the
  needle page (independent of speculation; scorer quality work, backlog)
- A single long text request is validated end-to-end; **concurrency with compaction + reuse is not**
- No tiered disk / session cold store (`nvme-gb`, `session-cache-dir`) and no NVMe page tier
- UI features that rely on llama.cpp/ninfer server extensions (`/v1/chat/completions/control`,
  `/v1/stream`, `/slots`) return 404 here

## Pinning and integration surface

Engine sources are not vendored: `scripts/prepare-backends.py` fetches the pinned commit/tag from
`backends/versions.json`, applies `backends/patches/vllm-kvmem.patch`, then verifies the resulting
`patched_tree` so a fresh clone reproduces the deployed sources byte for byte — the same layout kvmem
uses for its llama.cpp and ninfer backends.

Every vLLM touch point, what each change does and how many lines it costs:
[INTEGRATION-SURFACE.md](INTEGRATION-SURFACE.md).

## License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

Note that this is an **integration layer** and contains no engine sources:
- the patched vLLM files stay under the upstream Apache-2.0 license (the patch flow preserves their
  LICENSE/NOTICE)
- kvmem is fetched at a pinned tag and linked, not vendored and not relicensed; its upstream
  repository does not currently ship a top-level LICENSE file, so confirm with its maintainers
  before redistributing binaries that contain it
- model weights are not distributed here and keep their own terms
