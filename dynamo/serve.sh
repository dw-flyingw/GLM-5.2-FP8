#!/usr/bin/env bash
#
# Serve GLM-5.2-FP8 via NVIDIA Dynamo (SGLang backend) on 8x H200, aggregated TP=8.
# Brings up etcd + NATS + Dynamo frontend + one SGLang worker (full model, all GPUs).
#
# Serves on host port 8000 (OpenAI-compatible). Set PORT= to change.
#
# Tunables (env): PORT, MAX_MODEL_LEN, MEM_FRACTION, TP_SIZE, PAGE_SIZE,
#   HF_CACHE, MODEL, SERVED_NAME, DYNAMO_IMAGE.  See docker-compose.yml.

set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${DYNAMO_IMAGE:-glm52-dynamo-sglang:0.5.13post1}"

# Build the custom SGLang-0.5.13.post1 image if it's not present. --network=host is
# required: Docker's default bridge network can't reach pypi.org behind a proxy.
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "Image ${IMAGE} not found; building (needs host networking + proxy for pip) ..."
  docker build --network=host \
    --build-arg HTTP_PROXY="${HTTP_PROXY:-${http_proxy:-}}" \
    --build-arg HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy:-}}" \
    --build-arg NO_PROXY="${NO_PROXY:-${no_proxy:-localhost,127.0.0.1}}" \
    -t "${IMAGE}" -f Dockerfile .
fi

echo "Starting Dynamo (SGLang) stack for GLM-5.2-FP8 ..."
docker compose up -d

cat <<EOF

Up. The model load + NSA/MTP warmup takes several minutes (756 GB of weights).

  Follow worker startup:   docker compose -f $(pwd)/docker-compose.yml logs -f worker
  Frontend logs:           docker compose -f $(pwd)/docker-compose.yml logs -f frontend
  Stop everything:         ./stop.sh

Once the worker registers, test:
  curl http://localhost:${PORT:-8000}/v1/models
EOF
