#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PUBLIC="${1:-build/zk/public.json}"
PROOF="${2:-build/zk/proof.json}"
CIRCUIT_NAME="${CIRCUIT_NAME:-resource_feasibility}"

snarkjs groth16 verify \
  "build/zk/${CIRCUIT_NAME}_verification_key.json" \
  "$PUBLIC" \
  "$PROOF"
