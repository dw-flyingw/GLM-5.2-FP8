#!/usr/bin/env bash
# Stop and remove the Dynamo (SGLang) stack. Keeps the etcd data volume.
# Pass --volumes to also drop the etcd volume.
set -euo pipefail
cd "$(dirname "$0")"
docker compose down "$@"
echo "Dynamo stack stopped."
