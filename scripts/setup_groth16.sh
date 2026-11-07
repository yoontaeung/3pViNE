#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p build/zk
CIRCUIT_NAME="${CIRCUIT_NAME:-resource_feasibility}"
PTAU_POWER="${PTAU_POWER:-18}"
PTAU_PREFIX="pot${PTAU_POWER}"
R1CS_PATH="build/zk/${CIRCUIT_NAME}.r1cs"
R1CS_HASH_PATH="build/zk/${CIRCUIT_NAME}.r1cs.sha256"
CURRENT_R1CS_HASH="$(sha256sum "$R1CS_PATH" | awk '{print $1}')"

if [ -f "build/zk/${CIRCUIT_NAME}_final.zkey" ] \
  && [ -f "build/zk/${CIRCUIT_NAME}_verification_key.json" ] \
  && [ -f "$R1CS_HASH_PATH" ] \
  && [ "$(cat "$R1CS_HASH_PATH")" = "$CURRENT_R1CS_HASH" ]; then
  cp "build/zk/${CIRCUIT_NAME}_verification_key.json" build/zk/verification_key.json
  echo "Groth16 demo setup already exists under build/zk"
  exit 0
fi

if [ ! -f "build/zk/${PTAU_PREFIX}_final.ptau" ]; then
  snarkjs powersoftau new bn128 "$PTAU_POWER" "build/zk/${PTAU_PREFIX}_0000.ptau"
  snarkjs powersoftau contribute \
    "build/zk/${PTAU_PREFIX}_0000.ptau" \
    "build/zk/${PTAU_PREFIX}_0001.ptau" \
    --name="zk-vne demo powers of tau" \
    -e="zk-vne demo entropy"
  snarkjs powersoftau prepare phase2 \
    "build/zk/${PTAU_PREFIX}_0001.ptau" \
    "build/zk/${PTAU_PREFIX}_final.ptau"
fi

snarkjs groth16 setup \
  "build/zk/${CIRCUIT_NAME}.r1cs" \
  "build/zk/${PTAU_PREFIX}_final.ptau" \
  "build/zk/${CIRCUIT_NAME}_0000.zkey"
snarkjs zkey contribute \
  "build/zk/${CIRCUIT_NAME}_0000.zkey" \
  "build/zk/${CIRCUIT_NAME}_final.zkey" \
  --name="zk-vne demo phase2" \
  -e="zk-vne demo phase2 entropy"
snarkjs zkey export verificationkey \
  "build/zk/${CIRCUIT_NAME}_final.zkey" \
  "build/zk/${CIRCUIT_NAME}_verification_key.json"
cp "build/zk/${CIRCUIT_NAME}_verification_key.json" build/zk/verification_key.json
printf '%s\n' "$CURRENT_R1CS_HASH" > "$R1CS_HASH_PATH"
