#!/usr/bin/env bash
#
# Benchmark the running OpenAI endpoint (Dynamo on :8000) with NVIDIA's load
# generator. Reports TTFT / ITL / throughput.
#
# Usage:  ./bench.sh [URL] [CONCURRENCY] [NUM_PROMPTS]
#   URL          default http://localhost:8000
#   CONCURRENCY  default 32
#   NUM_PROMPTS  default 200
#

set -euo pipefail
URL="${1:-http://localhost:8000}"
CONC="${2:-32}"
NUM="${3:-200}"
MODEL="${SERVED_NAME:-glm-5.2-fp8}"
IMAGE="${DYNAMO_IMAGE:-glm52-dynamo-sglang:0.5.13post1}"

# aiperf (newer) if present in the image, else genai-perf.
docker run --rm --network host "${IMAGE}" bash -lc "
set -e
if command -v aiperf >/dev/null 2>&1; then
  aiperf profile --model ${MODEL} --url ${URL} --endpoint-type chat \
    --streaming --concurrency ${CONC} --request-count ${NUM} \
    --synthetic-input-tokens-mean 1024 --output-tokens-mean 256
else
  genai-perf profile -m ${MODEL} --url ${URL} --endpoint-type chat \
    --streaming --concurrency ${CONC} --request-count ${NUM} \
    --synthetic-input-tokens-mean 1024 --output-tokens-mean 256
fi
"
