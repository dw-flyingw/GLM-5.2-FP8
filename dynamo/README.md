# GLM-5.2-FP8 on NVIDIA Dynamo (SGLang backend, 8× H200)

Serves `zai-org/GLM-5.2-FP8` on `sprocket` using **NVIDIA Dynamo** as the
serving/orchestration layer, OpenAI-compatible on host port `:8000`. This is the
project's serving path (the earlier plain-vLLM container was removed).

## Why SGLang, not vLLM

GLM-5.2 is `GlmMoeDsaForCausalLM`: MoE + MLA + **DeepSeek-style Sparse Attention (DSA)**
with `head_size=704`. The vLLM path needs **vLLM ≥ 0.23.0** for that.

Verified on sprocket (2026-06-27): **no published Dynamo `vllm-runtime` image, nor the
`ai-dynamo` PyPI wheel, ships vLLM ≥ 0.23.0.**
`1.2.1`→0.20.1, `1.3.0-dev.1`→0.22.0 (its sparse-MLA backends cap at `head_size=576`),
`kimi-k2.6-dev`→0.21.0. So we use Dynamo's **SGLang** backend — NVIDIA's own GLM-5
Dynamo recipe (`ai-dynamo/dynamo/recipes/glm-5-nvfp4`) is SGLang too.

The image `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.2.1-cuda13` bundles **SGLang 0.5.11**,
which registers `GlmMoeDsaForCausalLM` and ships the `nsa` (Native Sparse Attention)
backend with the DSA indexer + MTP support. It auto-configures NSA for this arch and has
a non-Blackwell (Hopper) code path.

## Why aggregated only (no disaggregation on one node)

The weights are **~756 GB**. A full copy needs ~6 H200s (143 GB each). Disaggregated
serving runs **separate** prefill and decode workers, each holding a **full** model copy —
that's ~12 GPUs, impossible on a single 8-GPU node (TP4 = 189 GB/GPU > 143 GB). NVIDIA's
own GLM-5 recipe is therefore 5 nodes / 20 GPUs. **Disaggregation here needs ≥ 2 nodes.**

On one node the only fit is **aggregated, tensor-parallel over all 8 GPUs** — what this
stack does. Dynamo still adds: an OpenAI frontend + runtime/observability, KV-aware
routing (kicks in once you scale to multiple replicas/nodes), request migration, and a
clean path to multi-node disaggregation later.

## Layout

- `docker-compose.yml` — etcd + NATS + Dynamo frontend + one SGLang worker (TP=8). Host
  networking; tunables via `${...}` env (see below).
- `serve.sh` / `stop.sh` — bring the stack up / down.
- `bench.sh` — load-test any OpenAI endpoint (Dynamo or vLLM) with aiperf/genai-perf.

## Usage

```bash
cd dynamo
./serve.sh                       # start (detached)
docker compose logs -f worker    # watch model load (several minutes, 756 GB)
curl http://localhost:8000/v1/models
./stop.sh
```

Tunables (env): `PORT`, `MAX_MODEL_LEN` (→ sglang `--context-length`, default 262144),
`MEM_FRACTION` (default 0.85), `TP_SIZE` (default 8), `PAGE_SIZE` (default 64, NSA),
`HF_CACHE` (default `/data/huggingface`), `MODEL`, `SERVED_NAME`, `DYNAMO_IMAGE`.

## Serving config (mirrors the model card / NVIDIA recipe)

- `--tp-size 8`, `--kv-cache-dtype fp8_e4m3`
- `--attention-backend nsa` (DSA sparse attention), `--page-size 64`
- `--dyn-tool-call-parser glm47`, `--dyn-reasoning-parser glm45` (Dynamo frontend parsers)

### Performance levers (add after the baseline is up)

- **MTP / EAGLE speculative decoding** (GLM-5.2 has 1 MTP layer). Append to the worker
  command in `docker-compose.yml`:
  `--speculative-algorithm EAGLE --speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3`
  (mirrors the recipe; verify it composes with NSA on Hopper).
- **KV-aware routing** — add `--router-mode kv` to the frontend and a `--kv-events-config`
  to the worker; only a win with ≥ 2 workers.
- **MoE scaling** — `--ep-size` / dp-attention for expert parallelism across the 8 GPUs.

## Benchmark

```bash
./bench.sh http://localhost:8000 32 200   # concurrency 32, 200 prompts
```

Record TTFT / ITL / throughput here once measured.

## Status / verification checklist

- [x] vLLM path blocked (no Dynamo image ≥ vLLM 0.23.0) — SGLang chosen
- [x] Model requires **SGLang ≥ 0.5.13.post1** (model card). Stock images too old
      (1.2.1→0.5.11, 1.3.0-dev.1→0.5.12.post1) → custom image (see `Dockerfile`):
      Dynamo dev base + `sglang==0.5.13.post1` + `sglang-kernel==0.4.3`.
- [x] Model fits aggregated TP=8 (~94 GB/GPU weights); disagg needs ≥ 2 nodes
- [x] Aggregated stack boots, worker loads weights, NSA selected on H200, registers
- [x] OpenAI smoke tests pass: `/v1/models` lists `glm-5.2-fp8`; chat completion
      returns a clean answer with `reasoning_content` (glm45 reasoning parser working)
- [ ] Tool-call (`glm47`) exercised with a real tool schema
- [ ] Benchmark vs vLLM recorded

### Build gotchas (this environment)

- **Corporate proxy:** Docker bridge can't reach pypi; the build needs
  `--network=host` + `--build-arg HTTP(S)_PROXY` (handled by `serve.sh`).
- **Frontend needs the HF cache:** on discovery the frontend loads the model
  config/tokenizer (not weights) and will otherwise try huggingface.co and fail —
  the compose mounts the HF cache + sets `HF_HUB_OFFLINE=1` for the frontend too.
- **First start is slow:** SGLang runs a DeepGEMM JIT pre-compile (~10–20 min) +
  CUDA-graph capture. During this phase GPU0 drives compilation (oscillates 0↔100%)
  while GPUs 1–7 spin-wait at a barrier (100% util but ~126 W). The JIT kernels are
  persisted to the `dynamo-jit-cache` volume (`/home/dynamo/.cache`), so **only the
  first start pays this cost** — later restarts on the same image/GPU reuse the cache
  and come up fast. (Removing the volume or changing the SGLang/GPU arch invalidates
  it and triggers one more recompile.)
