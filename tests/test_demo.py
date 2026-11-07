"""Smoke tests for the small resource-feasibility relation."""

from __future__ import annotations

import unittest

from graph_data import build_toy_substrate, build_toy_vnr
from verifier import verify_resource_feasibility
from zk_encoding import build_zk_input


class DemoTests(unittest.TestCase):
    def test_valid_resource_feasibility_witness_is_accepted(self) -> None:
        data = build_zk_input(build_toy_vnr(), build_toy_substrate(), compute_commitments=False)

        accepted, errors = verify_resource_feasibility(data)

        self.assertTrue(accepted, errors)

    def test_tampered_cpu_after_state_is_rejected(self) -> None:
        data = build_zk_input(build_toy_vnr(), build_toy_substrate(), compute_commitments=False)
        data["sCpuAfter"][0] = 10

        accepted, errors = verify_resource_feasibility(data)

        self.assertFalse(accepted)
        self.assertTrue(any("CPU after-state" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
