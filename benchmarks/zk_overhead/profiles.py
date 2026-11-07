"""Circuit-size profiles and generated Circom entrypoints."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CircuitProfile:
    vn: int
    sn: int
    hops: int

    def __post_init__(self) -> None:
        if self.vn < 1:
            raise ValueError("VN must be at least 1")
        if self.sn < 1:
            raise ValueError("SN must be at least 1")
        if self.hops < 1:
            raise ValueError("H must be at least 1")

    @property
    def pair_count(self) -> int:
        return self.vn * (self.vn - 1) // 2

    @property
    def sn_bits(self) -> int:
        return max(1, math.ceil(math.log2(self.sn)))

    @property
    def profile_id(self) -> str:
        return f"vn{self.vn}_sn{self.sn}_h{self.hops}"

    @property
    def dominant_term(self) -> int:
        return self.pair_count * self.hops * self.sn * self.sn

    def as_dict(self) -> dict[str, int | str]:
        return {
            "profile_id": self.profile_id,
            "vn": self.vn,
            "sn": self.sn,
            "sn_bits": self.sn_bits,
            "pair_count": self.pair_count,
            "hops": self.hops,
            "dominant_term": self.dominant_term,
        }


def parse_profile(value: str) -> CircuitProfile:
    normalized = value.strip().lower()
    match = re.fullmatch(r"vn(\d+)_sn(\d+)_h(\d+)", normalized)
    if match:
        return CircuitProfile(*(int(item) for item in match.groups()))
    parts = normalized.split(",")
    if len(parts) == 3:
        return CircuitProfile(*(int(item) for item in parts))
    raise ValueError(f"invalid profile {value!r}; use vn3_sn6_h3 or 3,6,3")


def generate_circuit_entrypoint(
    profile: CircuitProfile,
    *,
    base_circuit: Path,
    output_path: Path,
) -> Path:
    """Copy the active relation and replace only its main instantiation."""
    source = base_circuit.read_text(encoding="utf-8")
    replacement = (
        f"ResourceFeasibility({profile.vn}, {profile.sn}, {profile.sn_bits}, "
        f"{profile.pair_count}, {profile.hops});"
    )
    updated, count = re.subn(
        r"ResourceFeasibility\(\d+,\s*\d+,\s*\d+,\s*\d+,\s*\d+\);\s*$",
        replacement + "\n",
        source,
        count=1,
    )
    if count != 1:
        raise RuntimeError("could not locate the ResourceFeasibility main instantiation")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(updated, encoding="utf-8")
    return output_path
