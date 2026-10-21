# KVMem × vLLM (SM75, TP=2) — single-service integration

**English** | [中文](README_zh-CN.md)

Plugs [kvmem](https://github.com/kvmem/kvmem)'s policy layer into vLLM (the weicj SM75 fork):
**one service, one port (18200)** serving the OpenAI-compatible API, metrics and a web UI from the
same origin. vLLM stays the engine; kvmem decides how much history is sent and which KV stays
resident.

```
client ──▶ :18200 ┬─ /v1/*        OpenAI-compatible API
                  ├─ /metrics     engine metrics
                  ├─ /docs        OpenAPI
                  └─ /            static UI (swappable)
```

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
| middleware (default `mode=middleware`) | standard Starlette body rewrite through the `request._body` cache: select history to B, clamp `max_tokens` to R, enforce CTX, H/D history-block cache |
| `mode=hook` (fallback) | wraps `OpenAIServingChat.create_chat_completion` and applies the same policy to the parsed request |
| `src/kvmem_entry_policy.py` | the single policy implementation both modes share |

The response path is never touched, so **streaming, CORS and error handling are vLLM's own**.

### Why selection happens at the entry

vLLM indexes KV blocks by logical position and assumes a monotonically growing sequence: truncating
mid-request desynchronises its scheduler (measured: the `num_scheduled_tokens >= num_logits` assertion,
then a hang once speculative decoding is off). kvmem's "keep only B" therefore has to be decided
**before the request reaches the engine**; inside the engine, kvmem's own per-page policy
(`prepare_reselect`) does the rest.

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

## Measured (current config: N=2, B/R=65536/32768, CAPTURE=512)

| Item | Result |
|---|---|
| Logical context | 486,605-token history → `prompt_tokens=89,066`, both needles found (106 s; measured with B=65536) |
| R contract | `max_tokens=100000` → `applied=32768` (only `max_completion_tokens` written) |
| CTX | 747,614 tokens → `400 history is ~747614 tokens, above kvmem CTX=524288` |
| Streaming | 30 SSE events, first at 0.07 s, last at 1.43 s, monotonic |
| Ports | only `0.0.0.0:18200`; no sidecar process |
| Prefill | 65.2K prompt: 70 s ≈ **934 tok/s**; 98.8K: 713 tok/s; two concurrent streams: 477 tok/s aggregate |
| Decode | 512 tokens from a small prompt ≈ **59 tok/s** |
| Prefix reuse | second identical prefix: `cached=63,024/65,890` |

Long-context wall time is almost entirely prefill; concurrency does not add throughput (TP=2 already
uses both GPUs).

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
- **No in-request compaction** (continue past R into the budget): vLLM's monotonic-length assumption
  makes it assert or hang; upstream also has it experimental and off by default
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
