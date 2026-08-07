# Context Window: Why 1M Isn't Enabled

The GLM-5.2-FP8 model supports a **1M token** max context natively, but this
deployment serves **512K** (`--context-length 524288`).

## Why not 1M on a single node

- Hardware: 8× H200 on a single node, TP=8.
- The KV cache pool tops out at **~540K tokens** alongside the ~756 GB of FP8
  weights (≈94 GB/GPU). There isn't enough GPU memory left to hold a full 1M
  KV pool.
- Disaggregated prefill/decode — the path that would let us exceed single-node
  KV limits — **is not possible on a single node** for this model, since a full
  weights copy per worker exceeds 8 GPUs.

## What it would take

- ≥ **2 nodes** so the stack can disaggregate prefill and decode (separate
  workers hold weights/KV on different nodes), unblocking the 1M context.

## Current settings

| Setting | Value |
|---|---|
| Model max context | 1,048,576 (1M) |
| Served context (`--context-length`) | 524,288 (512K) |
| KV pool ceiling (observed) | ~540K tokens |
| Backend | Dynamo + SGLang 0.5.13.post1 |
| Attention | DSA, `flashmla_kv` on Hopper+fp8, `index_topk=2048` |
| Speculative decoding | MTP/EAGLE on |

References: `README.md`, `dynamo/README.md`, `chat.py` (`CONTEXT_WINDOW = 524_288`).
