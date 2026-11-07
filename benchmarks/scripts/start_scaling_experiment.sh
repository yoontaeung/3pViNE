#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${ZK_VNE_IMAGE:-zk-vne-zk:local}"
USER_ID="$(id -u)"
GROUP_ID="$(id -g)"
RUN_ID="${SCALING_RUN_ID:-scaling_v3_$(date -u +%Y%m%dT%H%M%SZ)}"
REPETITIONS="${SCALING_REPETITIONS:-3}"
WARMUPS="${SCALING_WARMUPS:-1}"
CASE_TIMEOUT_SECONDS="${SCALING_CASE_TIMEOUT_SECONDS:-18000}"
MAX_MEASUREMENT_PTAU_POWER="${SCALING_MAX_MEASUREMENT_PTAU_POWER:-23}"
NODE_HEAP_MB="${ZKP_NODE_HEAP_MB:-65536}"
PROFILE_OVERRIDE="${SCALING_PROFILES:-}"
PREFLIGHT_OVERRIDE="${SCALING_PREFLIGHT_PROFILES:-}"
RUN_DIR="artifacts/benchmarks/zk_overhead/${RUN_ID}"
CONTROL_DIR="$ROOT/artifacts/benchmarks/zk_overhead/background"
CONTAINER_NAME="zk-vne-${RUN_ID//[^a-zA-Z0-9_.-]/-}"

EXTRA_ARGS=()
if [ -n "$PROFILE_OVERRIDE" ]; then
  read -r -a PROFILE_ARGS <<< "$PROFILE_OVERRIDE"
  EXTRA_ARGS+=(--profiles "${PROFILE_ARGS[@]}")
fi
if [ -n "$PREFLIGHT_OVERRIDE" ]; then
  read -r -a PREFLIGHT_ARGS <<< "$PREFLIGHT_OVERRIDE"
  EXTRA_ARGS+=(--preflight-profiles "${PREFLIGHT_ARGS[@]}")
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker is required, but this user cannot access the Docker daemon." >&2
  exit 1
fi
if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "Container already exists: $CONTAINER_NAME" >&2
  exit 1
fi
if [ "${ZKP_SKIP_BUILD:-0}" != "1" ]; then
  docker build -t "$IMAGE" "$ROOT"
fi

mkdir -p "$CONTROL_DIR"
CONTAINER_ID="$(
  docker run --detach \
    --name "$CONTAINER_NAME" \
    --network none \
    --user "${USER_ID}:${GROUP_ID}" \
    --env "NODE_OPTIONS=--max-old-space-size=${NODE_HEAP_MB}" \
    --env "MPLCONFIGDIR=/tmp/matplotlib" \
    --env "XDG_CACHE_HOME=/tmp/cache" \
    -v "$ROOT:/workspace" \
    -w /workspace \
    "$IMAGE" \
    /opt/zk-vne-venv/bin/python benchmarks/zk_overhead/scaling_experiment.py \
      --run-dir "$RUN_DIR" \
      --repetitions "$REPETITIONS" \
      --warmups "$WARMUPS" \
      --case-timeout-seconds "$CASE_TIMEOUT_SECONDS" \
      --max-measurement-ptau-power "$MAX_MEASUREMENT_PTAU_POWER" \
      "${EXTRA_ARGS[@]}"
)"

printf '%s\n' "$CONTAINER_ID" > "$CONTROL_DIR/${RUN_ID}.container_id"
printf '%s\n' "$CONTAINER_NAME" > "$CONTROL_DIR/${RUN_ID}.container_name"
sleep 1
STATUS="$(docker inspect "$CONTAINER_NAME" --format '{{.State.Status}}')"
if [ "$STATUS" != "running" ]; then
  echo "Background scaling experiment failed to remain running." >&2
  docker logs "$CONTAINER_NAME" >&2 || true
  exit 1
fi

echo "Background scaling experiment started."
echo "Run ID: $RUN_ID"
echo "Container: $CONTAINER_NAME"
echo "Container ID: $CONTAINER_ID"
echo "Per-case timeout: ${CASE_TIMEOUT_SECONDS}s"
echo "Maximum measured Powers of Tau exponent: ${MAX_MEASUREMENT_PTAU_POWER}"
echo "Profiles: ${PROFILE_OVERRIDE:-default}"
echo "Preflight profiles: ${PREFLIGHT_OVERRIDE:-default}"
echo "Progress: docker logs $CONTAINER_NAME"
echo "Status: $RUN_DIR/status.csv"
echo "Results: $RUN_DIR"
