#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${ZK_VNE_IMAGE:-zk-vne-zk:local}"
USER_ID="$(id -u)"
GROUP_ID="$(id -g)"
RUN_ID="${ZKP_RUN_ID:-zkp_$(date -u +%Y%m%dT%H%M%SZ)}"
PROFILES="${ZKP_PROFILES:-vn3_sn6_h3}"
REPETITIONS="${ZKP_REPETITIONS:-3}"
WARMUPS="${ZKP_WARMUPS:-1}"
NODE_HEAP_MB="${ZKP_NODE_HEAP_MB:-16384}"
RUN_DIR="artifacts/benchmarks/zk_overhead/${RUN_ID}"

case "$NODE_HEAP_MB" in
  ''|*[!0-9]*)
    echo "ZKP_NODE_HEAP_MB must be an integer." >&2
    exit 2
    ;;
esac
if [ "$NODE_HEAP_MB" -lt 1024 ]; then
  echo "ZKP_NODE_HEAP_MB must be at least 1024." >&2
  exit 2
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker is required, but this user cannot access the Docker daemon." >&2
  exit 1
fi

if [ "${ZKP_SKIP_BUILD:-0}" != "1" ]; then
  docker build -t "$IMAGE" "$ROOT"
fi

read -r -a PROFILE_ARGS <<< "$PROFILES"
docker run --rm \
  --network none \
  --user "${USER_ID}:${GROUP_ID}" \
  --env "NODE_OPTIONS=--max-old-space-size=${NODE_HEAP_MB}" \
  -v "$ROOT:/workspace" \
  -w /workspace \
  "$IMAGE" \
  /opt/zk-vne-venv/bin/python benchmarks/zk_overhead/run.py \
    --run-dir "$RUN_DIR" \
    --repetitions "$REPETITIONS" \
    --warmups "$WARMUPS" \
    --profiles "${PROFILE_ARGS[@]}"

echo "ZKP overhead results: ${RUN_DIR}/results.json"
echo "Incremental journal: ${RUN_DIR}/journal.jsonl"
