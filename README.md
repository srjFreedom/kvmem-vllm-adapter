# KVMem × vLLM — dual-GPU SM75, high-throughput, single service

**English** | [中文](README_zh-CN.md)

**kvmem's policy layer running *inside* vLLM, on two Turing (SM75) cards.** This is the
high-performance dual-GPU build: instead of a gateway in front of the engine, kvmem's entry policy
runs in the vLLM API process, and the engine itself is tuned for two consumer 22 GB cards —
MTP speculative decoding, FP8 KV cache, prefix caching and a KV pool sized for N concurrent
long-context requests.

What that buys you on 2× RTX 2080 Ti:

- **Real concurrency, not a queue.** Engine window = `N × (B + R)`; at `N=2` two code-generation
  streams keep **70–73 tok/s each** (single stream: 74.9) for **≈136 tok/s aggregate**.
- **Long context that fits.** A **172K-token** window on 44 GB of VRAM at `GPU_UTIL=0.97`
  (B = 52K history + 32K generation per request, FP8 KV).
- **Fast prefill.** ~1.3K tok/s at 5K tokens, still ~1K tok/s at the full 52K budget.
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

Measured on **2× RTX 2080 Ti 22 GB**, TP=2, `N=2`, `B=53,248`, `R=32,768` (window 172,032),
FP8 KV, MTP `K=4`, `GPU_UTIL=0.97`, prefix caching on; `temperature=0`.

**Prefill** (single request, cold prompt):

| prompt tokens | TTFT | prefill |
|---|---|---|
| 5,035 | 3.9 s | **1,295 tok/s** |
| 19,838 | 17.5 s | **1,136 tok/s** |
| 51,723 | 52.5 s | **985 tok/s** |

Larger prompts are pointless by design: the entry policy trims history to B before the engine sees
it, so 52K is the ceiling that matters.

**Decode** (code generation):

| output | streams | throughput |
|---|---|---|
| 343 tok (binary search + tests) | 1 | **74.9 tok/s** |
| 768 tok (TOML-subset parser) | 1 | **59.7 tok/s** |
| 2,048 tok (TOML-subset parser) | 1 | **58.3 tok/s** |
| 2 × 512 tok | 2 | **70–73 tok/s each**, ≈**136 tok/s** aggregate |

Two things explain the spread, and both are properties of MTP speculative decoding, not noise:
acceptance is higher on dense repetitive code (74.9) than on prose-heavy answers (58–60), and very
long outputs amortise the faster opening phase. With two concurrent streams each request keeps
~95 % of its single-stream speed — the second card is doing real work rather than waiting.

Prefix cache: re-sending the same 57K-token prompt costs 2.1 s TTFT instead of 58.6 s
(≈20.7K tok/s effective).

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

With that flag the engine-side compaction is enabled as well: a request may generate past R up to
**`CTX`**. B is only the resident window - everything else in the history lives on the host tier, so
the logical context is what CTX bounds. When the resident pool fills (B + R) the runtime reselects, spills the pages
that are no longer needed to the host tier, and tells EngineCore to truncate the request's token book
in place (`_all_token_ids` → visible set, `num_computed_tokens` rewound) over a worker → EngineCore
channel, so the scheduler sees no gap and the sequence keeps growing while the physical pool stays
`B + R`.

- **Default (flag absent): the generation ceiling is R.** `max_tokens` is clamped to R and a request
  stops there; the sequence is monotone and the pool is never recycled.
- **With the flag: the ceiling is `CTX`.** One switch drives everything: the entry ceiling, the
  engine-side compaction, the block channel and synchronous scheduling. The prompt stays immutable
  (vLLM derives the output length from `num_tokens − prompt_len`), so only generated tokens are
  compacted and the most recent `RECENT` tokens are always kept.
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
| vLLM SM75 fork | `93c9013e375b` + integration patch | `install.sh` fetches it and applies the patch (verifies `patched_tree`) |
| kvmem sources | tag `v0.18.0` | same (the native extension needs its headers and library) |
| Model weights | **any model vLLM can serve** (this deployment runs an FP8 + MTP export) | you provide them; point `MODEL` at the directory |

```bash
git clone https://github.com/srjFreedom/kvmem-vllm-adapter
cd kvmem-vllm-adapter

USE_LOCAL_FORK=0 ./install.sh            # fetch engines → patch → plugin → extension → self-check
ENGINE=$PWD/engines/vllm ./scripts/setup-venv.sh   # torch / CUDA (several GB; skip if you have a venv)

# Run it in the foreground with the repository defaults (default profile included):
MODEL=$HOME/models/<your-model> ./deploy/run-current.sh

# Or as a service (the unit's ExecStart points at this repository's run-current.sh):
cp deploy/kvmem-vllm.service /etc/systemd/system/
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
run-current.sh      generic entry, no site parameters (systemd points here)
   │ exec
run-kvmem-vllm.sh   implementation (every knob is ${VAR:-default})
```

Defaults: `B=36864 R=16384 N=1 GPU_UTIL=0.97` with the shipped
`2x2080Ti/qwen27b/w8a16/mtp4-fp8kv-1x184K-text-image.env` profile. The only required input is
`MODEL` (your weights directory); set `CHAT_TEMPLATE_FILE` only when the model ships no built-in
chat template. Override anything else with environment variables, or put
`KEY=VALUE` lines in `~/.config/kvmem-vllm.env` (the unit loads it when present):

```bash
N=2 B=53248 GPU_UTIL=0.98 ./deploy/run-current.sh
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

Generation ceiling: **R by default** (`max_tokens` is clamped to R). Pass
`--kvmem-inrequest-compaction` to the worker to enable super-R generation, which raises the ceiling to
`CTX`, compacts the live context when the resident pool fills and forces synchronous scheduling — one flag
for the whole feature, no environment variable to set:

```bash
./backends/vllm/worker.py --model "$MODEL" --port 18200 ... --kvmem-inrequest-compaction
```

The constraint that bites: **engine window = `N × (B + R)` must fit the KV pool** (210,651 tokens on
this box). Defaults (`N=1, B+R=53,248`) leave a large margin; `N=2` → 106,496, `N=3` → 159,744,
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
| R contract | `max_tokens=100000` → `applied=32768` (only `max_completion_tokens` written) |
| CTX | 747,614 tokens → `400 history is ~747614 tokens, above kvmem CTX=524288` |
| Streaming | 30 SSE events, first at 0.07 s, last at 1.43 s, monotonic |
| Ports | only `0.0.0.0:18200`; no sidecar process |
| Prefix reuse | second identical prefix: `cached=63,024/65,890` |

These rows pin the policy contract (how much history reaches the engine, the R clamp, the CTX
rejection, SSE behaviour); throughput lives in [Performance](#performance).

## Tests and probe

```bash
PY=$HOME/vLLM-2080Ti-Definitive/.venv/bin/python
for t in tests/*_test.py; do PYTHONPATH="build/python:src:plugin" $PY "$t"; done   # 39, all PASS
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
STATUS.md                   internal decision log with the measurements behind each choice
INTEGRATION-SURFACE.md      touch-point list and source-change registry
```

## Known limitations

- **Entry-side text selection is a stopgap** (vLLM cannot truncate KV mid-request), not kvmem's own
  policy; when history and question share one message the trailing-paragraph heuristic is used, so a
  very long question whose topic sits in an earlier paragraph may lose blocks
- **In-request compaction is opt-in** (`--kvmem-inrequest-compaction`) and, by design, more delicate
  than the default path: the prompt must fit in B, retrieval has no score for freshly generated blocks
  (the runtime keeps the recent tail mandatory), and compaction is not combined with speculative
  drafts on the same step
- `SESSIONS` splits the H pool across **live** requests; a finished request releases its share, which
  is not the reference backends' "retain N inactive histories"
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
