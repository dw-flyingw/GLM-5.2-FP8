# GLM-5.2-FP8 on NVIDIA Dynamo (SGLang, 8× H200)

Serves [`zai-org/GLM-5.2-FP8`](https://huggingface.co/zai-org/GLM-5.2-FP8) on a single node
across all 8 H200 GPUs via **NVIDIA Dynamo** with the **SGLang** backend, exposing an
OpenAI-compatible API on `:8000`.

> Previously this repo served the model with a plain vLLM container (`serve.sh`). That
> path was removed in favor of Dynamo/SGLang — see `dynamo/` for the full setup and the
> `dynamo/README.md` for the why (vLLM ≥ 0.23.0 isn't available in any Dynamo runtime
> yet; SGLang 0.5.13.post1 is the supported engine for GLM-5.2's sparse attention).

## Model

- Architecture: `GlmMoeDsaForCausalLM` — MoE (256 routed + 1 shared experts, 8/tok),
  MLA attention, **DeepSeek-style Sparse Attention (DSA)** with an indexer
  (`index_topk=2048`), FP8 block-quant (128×128, e4m3), 78 layers, 1 MTP layer, 1M
  max context. ~756 GB of weights (≈94 GB/GPU at TP=8).
- Weights live in a shared Hugging Face cache (point `HF_CACHE` at it); they are not re-downloaded.

## Serve

```bash
cd dynamo
./serve.sh                       # build image if needed + start the stack (detached)
docker compose logs -f worker    # watch startup (first boot is slow; see below)
./stop.sh                        # stop + remove the stack
```

The stack = etcd + NATS + Dynamo frontend + one SGLang worker (aggregated, TP=8,
auto-selected DSA attention backend (`flashmla_kv` on Hopper+fp8), fp8 KV cache,
**MTP/EAGLE speculative decoding on** — ~2× single-stream decode). Context is served at **512K** (`--context-length 524288`); the
model's 1M max isn't servable on one node (the KV pool tops out at ~540K tokens
alongside the weights). Disaggregated prefill/decode is **not possible on a single
node** for this model (a full copy per worker exceeds 8 GPUs) — it needs ≥ 2 nodes.
Details, tunables, and benchmarks: [`dynamo/README.md`](dynamo/README.md).

> First start runs a DeepGEMM JIT pre-compile + CUDA-graph capture (~10–20 min). It's
> cacheable with `python3 -m sglang.compile_deep_gemm` (same args).

## Test

```bash
curl http://localhost:8000/v1/models
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "glm-5.2-fp8",
  "messages": [{"role": "user", "content": "Hello!"}]
}'
```

Or use the bundled streaming CLI: `./chat.py` (stdlib only; reads/streams the
`reasoning_content` from the glm45 reasoning parser).
