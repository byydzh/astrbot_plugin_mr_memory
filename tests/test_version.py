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
        changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
        development_log = (root / "docs" / "DEVELOPMENT_LOG.md").read_text(
            encoding="utf-8"
        )
        match = re.search(r"^version:\s*([^\s]+)$", metadata, re.MULTILINE)
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.group(1), PLUGIN_VERSION)
        self.assertIn(PLUGIN_VERSION, readme)
        changelog_match = re.search(r"(?m)^## ([0-9]+\.[0-9]+\.[0-9]+)$", changelog)
        development_match = re.search(
            r"(?m)^## .* / ([0-9]+\.[0-9]+\.[0-9]+)$",
            development_log,
        )
        self.assertIsNotNone(changelog_match)
        self.assertIsNotNone(development_match)
        assert changelog_match is not None
        assert development_match is not None
        self.assertEqual(changelog_match.group(1), PLUGIN_VERSION)
        self.assertEqual(development_match.group(1), PLUGIN_VERSION)
        self.assertEqual(EXTRACTOR_VERSION, f"mr-memory-{PLUGIN_VERSION}")

    def test_runtime_manifest_covers_layered_modules(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest = (root / "docs" / "RUNTIME_FILES.md").read_text(encoding="utf-8")
        exact_paths = {
            "__init__.py",
            "main.py",
            "metadata.yaml",
            "_conf_schema.json",
            "requirements.txt",
            "mr_memory/snapshot.py",
            "mr_memory/routing.py",
            "mr_memory/reader.py",
            "mr_memory/orchestrator.py",
            "mr_memory/certificate.py",
            "mr_memory/surface.py",
            "mr_memory/singleflight.py",
            "mr_memory/evidence_closure.py",
            "mr_memory/storage.py",
            "mr_memory/service.py",
        }
        for relative_path in sorted(exact_paths):
            with self.subTest(path=relative_path):
                self.assertIn(f"`{relative_path}`", manifest)
                self.assertTrue((root / relative_path).is_file())

        required_globs = {
            "mr_memory/*.py": "mr_memory/*.py",
            "pages/console/*": "pages/console/*",
            ".astrbot-plugin/i18n/*.json": ".astrbot-plugin/i18n/*.json",
        }
        for documented_rule, glob_pattern in required_globs.items():
            with self.subTest(rule=documented_rule):
                self.assertIn(f"`{documented_rule}`", manifest)
                self.assertTrue(any(root.glob(glob_pattern)))


if __name__ == "__main__":
    unittest.main()
