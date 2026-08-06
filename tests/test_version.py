from __future__ import annotations

import re
import unittest
from pathlib import Path

from mr_memory.version import EXTRACTOR_VERSION, PLUGIN_VERSION


class VersionConsistencyTests(unittest.TestCase):
    def test_public_version_surfaces_are_consistent(self) -> None:
        root = Path(__file__).resolve().parents[1]
        metadata = (root / "metadata.yaml").read_text(encoding="utf-8")
        readme = (root / "README.md").read_text(encoding="utf-8")
        match = re.search(r"^version:\s*([^\s]+)$", metadata, re.MULTILINE)
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.group(1), PLUGIN_VERSION)
        self.assertIn(PLUGIN_VERSION, readme)
        self.assertEqual(EXTRACTOR_VERSION, f"mr-memory-{PLUGIN_VERSION}")


if __name__ == "__main__":
    unittest.main()
