#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${ZK_VNE_IMAGE:-zk-vne-zk:local}"
USER_ID="$(id -u)"
GROUP_ID="$(id -g)"
RUN_ID="${TOPOLOGY_RUN_ID:-topology_$(date -u +%Y%m%dT%H%M%SZ)}"
PROFILE="${TOPOLOGY_PROFILE:-vn3_sn32_h3}"
REPETITIONS="${TOPOLOGY_REPETITIONS:-5}"
WARMUPS="${TOPOLOGY_WARMUPS:-1}"
NODE_HEAP_MB="${ZKP_NODE_HEAP_MB:-16384}"
RUN_DIR="artifacts/benchmarks/zk_overhead/${RUN_ID}"

if ! docker info >/dev/null 2>&1; then
  echo "Docker is required, but this user cannot access the Docker daemon." >&2
  exit 1
fi

if [ "${ZKP_SKIP_BUILD:-0}" != "1" ]; then
  docker build -t "$IMAGE" "$ROOT"
fi

docker run --rm \
  --network none \
  --user "${USER_ID}:${GROUP_ID}" \
  --env "NODE_OPTIONS=--max-old-space-size=${NODE_HEAP_MB}" \
  --env "MPLCONFIGDIR=/tmp/matplotlib" \
  --env "XDG_CACHE_HOME=/tmp/cache" \
  -v "$ROOT:/workspace" \
  -w /workspace \
  "$IMAGE" \
  /opt/zk-vne-venv/bin/python benchmarks/zk_overhead/topology_run.py \
    --run-dir "$RUN_DIR" \
    --profile "$PROFILE" \
    --repetitions "$REPETITIONS" \
    --warmups "$WARMUPS"

echo "Topology timing results: ${RUN_DIR}/results.json"
