"""Tests for durable benchmark result helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from benchmarks.common.experiment_log import git_metadata, source_manifest


ROOT = Path(__file__).resolve().parents[2]


class SourceManifestTests(unittest.TestCase):
    def test_git_metadata_has_revision_and_cleanliness_fields(self) -> None:
        metadata = git_metadata(ROOT)

        self.assertTrue(metadata["revision"])
        self.assertIsInstance(metadata["dirty"], bool)
        self.assertIsInstance(metadata["status"], list)

    def test_manifest_is_stable_and_changes_with_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.txt"
            second = root / "second.txt"
            first.write_text("alpha", encoding="utf-8")
            second.write_text("beta", encoding="utf-8")

            original = source_manifest(root, [second, first])
            reordered = source_manifest(root, [first, second])
            self.assertEqual(original, reordered)

            second.write_text("changed", encoding="utf-8")
            changed = source_manifest(root, [first, second])
            self.assertNotEqual(original["sha256"], changed["sha256"])


if __name__ == "__main__":
    unittest.main()
