"""Private witness generation and Groth16 proving inside a facility service."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx

from local_solver import LocalEmbedding
from system.api_models import ProofBundle, ProofMode
from verifier import verify_resource_feasibility
from zk_encoding import S_NODE_IDS, build_zk_input, write_zk_input


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class GeneratedClaimProof:
    """A private relation witness plus the public proof material it produced."""

    zk_input: dict[str, Any]
    proof_bundle: ProofBundle | None
    timings_seconds: dict[str, float]


class ClaimProofRuntime:
    """Generate one proof without exposing private witness files to other services."""

    def __init__(
        self,
        *,
        proof_mode: ProofMode,
        work_dir: Path,
        project_root: Path = ROOT,
    ) -> None:
        self.proof_mode = proof_mode
        self.work_dir = work_dir
        self.project_root = project_root

    def generate(
        self,
        *,
        request_id: int,
        facility_id: str,
        proof_index: int,
        substrate: nx.Graph,
        local_vnr: nx.Graph,
        embedding: LocalEmbedding,
        randomness: dict[str, int],
        endpoint_substrate_node: str | None,
    ) -> GeneratedClaimProof:
        timings: dict[str, float] = {}
        endpoint_idx = (
            S_NODE_IDS[endpoint_substrate_node]
            if endpoint_substrate_node is not None
            else None
        )

        started = time.perf_counter()
        zk_input = build_zk_input(
            local_vnr,
            substrate,
            node_mapping=embedding.node_mapping,
            paths=embedding.paths,
            req_id=request_id,
            randomness=randomness,
            endpoint_idx=endpoint_idx,
        )
        timings["encoding"] = time.perf_counter() - started

        started = time.perf_counter()
        accepted, errors = verify_resource_feasibility(zk_input)
        timings["reference_validation"] = time.perf_counter() - started
        if not accepted:
            raise RuntimeError(
                f"request {request_id} {facility_id} produced an invalid witness: {errors}"
            )

        claim_dir = self.work_dir / f"request_{request_id:06d}_{proof_index:02d}"
        private_dir = claim_dir / "private"
        public_dir = claim_dir / "public"
        private_dir.mkdir(parents=True, exist_ok=True)
        public_dir.mkdir(parents=True, exist_ok=True)
        input_path = private_dir / "input.json"
        witness_path = private_dir / "witness.wtns"
        proof_path = public_dir / "proof.json"
        public_path = public_dir / "public.json"

        started = time.perf_counter()
        write_zk_input(zk_input, input_path)
        timings["input_write"] = time.perf_counter() - started

        if self.proof_mode == "reference":
            timings["witness"] = 0.0
            timings["prove"] = 0.0
            return GeneratedClaimProof(
                zk_input=zk_input,
                proof_bundle=None,
                timings_seconds=timings,
            )

        self._require_proving_artifacts()
        started = time.perf_counter()
        self._run(
            [
                "node",
                "build/zk/resource_feasibility_js/generate_witness.js",
                "build/zk/resource_feasibility_js/resource_feasibility.wasm",
                str(input_path),
                str(witness_path),
            ]
        )
        timings["witness"] = time.perf_counter() - started

        started = time.perf_counter()
        self._run(
            [
                "snarkjs",
                "groth16",
                "prove",
                "build/zk/resource_feasibility_final.zkey",
                str(witness_path),
                str(proof_path),
                str(public_path),
            ]
        )
        timings["prove"] = time.perf_counter() - started

        proof = json.loads(proof_path.read_text(encoding="utf-8"))
        public_values = json.loads(public_path.read_text(encoding="utf-8"))
        public_inputs = [str(value) for value in public_values]
        return GeneratedClaimProof(
            zk_input=zk_input,
            proof_bundle=ProofBundle(proof=proof, public_inputs=public_inputs),
            timings_seconds=timings,
        )

    def _require_proving_artifacts(self) -> None:
        required = (
            self.project_root / "build/zk/resource_feasibility_js/resource_feasibility.wasm",
            self.project_root / "build/zk/resource_feasibility_final.zkey",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(f"missing proving artifacts: {missing}")

    def _run(self, args: list[str]) -> None:
        subprocess.run(args, check=True, cwd=self.project_root)
