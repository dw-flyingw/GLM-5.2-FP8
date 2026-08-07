#!/usr/bin/env python3
"""Tiny streaming benchmark for the GLM-5.2-FP8 Dynamo (SGLang) OpenAI endpoint.

Stdlib only (no aiperf/genai-perf/tokenizer needed — those aren't in the runtime
image and an offline environment can't fetch a corpus for sglang.bench_serving). Measures
TTFT, inter-token latency (ITL), end-to-end latency and decode throughput by
streaming /v1/chat/completions with usage accounting. Designed to compare MTP
speculative decoding OFF vs ON: run it before and after enabling the EAGLE flags.

Usage:
    ./bench_stream.py --concurrency 1  --num 16 --max-tokens 256
    ./bench_stream.py --concurrency 32 --num 128 --max-tokens 256
    BASE_URL=http://localhost:8000 ./bench_stream.py --tag mtp-off
"""
import argparse
import json
import os
import statistics as stats
import sys
import threading
import time
import urllib.request

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")
MODEL = os.environ.get("MODEL", "glm-5.2-fp8")

# A fixed, deterministic prompt so OFF vs ON see identical work. Asks for a
# sized output so decode (where MTP helps) dominates over prefill.
PROMPT = (
    "You are benchmarking a language model server. Write a detailed, continuous "
    "technical explanation of how tensor parallelism, MoE expert routing, and "
    "speculative decoding interact on an 8-GPU inference server. Keep writing "
    "until you are asked to stop; do not use bullet lists."
)


def one_request(max_tokens, no_think):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if no_think:
        # GLM honours an explicit no-think hint; keeps output to plain content.
        body["messages"].insert(0, {"role": "system", "content": "/nothink Reply directly."})
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    t0 = time.perf_counter()
    ttft = None
    last = t0
    itls = []
    completion_tokens = None
    chunk_count = 0
    with urllib.request.urlopen(req) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            usage = obj.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens")
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            piece = delta.get("content") or delta.get("reasoning_content")
            if piece:
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - t0
                else:
                    itls.append(now - last)
                last = now
                chunk_count += 1
    total = time.perf_counter() - t0
    out_tok = completion_tokens if completion_tokens else chunk_count
    return {
        "ttft": ttft if ttft is not None else total,
        "total": total,
        "out_tok": out_tok,
        "itls": itls,
        "decode_tps": (out_tok - 1) / (total - (ttft or 0)) if total > (ttft or 0) and out_tok > 1 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--num", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    results = []
    lock = threading.Lock()
    sem = threading.Semaphore(args.concurrency)
    errors = [0]

    def worker():
        with sem:
            try:
                r = one_request(args.max_tokens, args.no_think)
                with lock:
                    results.append(r)
            except Exception as e:  # noqa: BLE001
                with lock:
                    errors[0] += 1
                sys.stderr.write(f"req error: {e}\n")

    wall0 = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(args.num)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - wall0

    if not results:
        print("no successful requests")
        sys.exit(1)

    ttfts = sorted(r["ttft"] for r in results)
    decode_tps = [r["decode_tps"] for r in results if r["decode_tps"] > 0]
    all_itls = [x for r in results for x in r["itls"]]
    tot_out = sum(r["out_tok"] for r in results)

    def pct(xs, p):
        if not xs:
            return 0.0
        i = min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1))))
        return sorted(xs)[i]

    print(f"\n=== bench {args.tag or ''}  conc={args.concurrency} num={args.num} "
          f"max_tokens={args.max_tokens} ===")
    print(f"requests ok/err     : {len(results)}/{errors[0]}")
    print(f"wall time           : {wall:.2f} s")
    print(f"TTFT  mean/p50/p99  : {stats.mean(ttfts)*1000:.0f} / {pct(ttfts,50)*1000:.0f} "
          f"/ {pct(ttfts,99)*1000:.0f} ms")
    if all_itls:
        print(f"ITL   mean/p50/p99  : {stats.mean(all_itls)*1000:.1f} / {pct(all_itls,50)*1000:.1f} "
              f"/ {pct(all_itls,99)*1000:.1f} ms")
    if decode_tps:
        print(f"per-req decode tok/s: mean {stats.mean(decode_tps):.1f}  "
              f"(min {min(decode_tps):.1f}, max {max(decode_tps):.1f})")
    print(f"output tokens total : {tot_out}")
    print(f"system output tok/s : {tot_out / wall:.1f}")


if __name__ == "__main__":
    main()
