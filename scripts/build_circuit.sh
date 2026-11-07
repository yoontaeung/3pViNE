#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p build/zk
CIRCUIT_NAME="${CIRCUIT_NAME:-resource_feasibility}"
CIRCUIT_SOURCE="${CIRCUIT_SOURCE:-circuits/${CIRCUIT_NAME}.circom}"

CIRCOMLIB_PATH="${CIRCOMLIB_PATH:-/opt/zk-vne-js/node_modules/circomlib/circuits}"
if [ ! -d "$CIRCOMLIB_PATH" ]; then
  CIRCOMLIB_PATH="${NODE_PATH:-node_modules}/circomlib/circuits"
fi

circom "$CIRCUIT_SOURCE" \
  --r1cs \
  --wasm \
  --sym \
  -l "$CIRCOMLIB_PATH" \
  -o build/zk

snarkjs r1cs info "build/zk/${CIRCUIT_NAME}.r1cs"
