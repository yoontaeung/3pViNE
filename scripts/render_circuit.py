#!/usr/bin/env python3
"""Render one fixed-size ResourceFeasibility circuit entrypoint."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path


def render(source: Path, output: Path, *, vn: int, sn: int, hops: int) -> None:
    if min(vn, sn, hops) < 1:
        raise ValueError("circuit bounds must be positive")
    pair_count = vn * (vn - 1) // 2
    sn_bits = max(1, math.ceil(math.log2(sn)))
    replacement = f"ResourceFeasibility({vn}, {sn}, {sn_bits}, {pair_count}, {hops});"
    rendered, count = re.subn(
        r"ResourceFeasibility\(\d+,\s*\d+,\s*\d+,\s*\d+,\s*\d+\);\s*$",
        replacement + "\n",
        source.read_text(encoding="utf-8"),
        count=1,
    )
    if count != 1:
        raise RuntimeError("could not find the circuit's main ResourceFeasibility instance")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--vn", type=int, required=True)
    parser.add_argument("--sn", type=int, required=True)
    parser.add_argument("--hops", type=int, required=True)
    args = parser.parse_args()
    render(args.source, args.output, vn=args.vn, sn=args.sn, hops=args.hops)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
