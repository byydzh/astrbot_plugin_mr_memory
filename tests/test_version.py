from __future__ import annotations

import unittest
from pathlib import Path


class PluginMetadataTests(unittest.TestCase):
    def test_plugin_has_no_runtime_version_mechanism(self) -> None:
        root = Path(__file__).resolve().parents[1]
        metadata = (root / "metadata.yaml").read_text(encoding="utf-8")
        main_source = (root / "main.py").read_text(encoding="utf-8")
        console_script = (root / "pages" / "console" / "script.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("version: unversioned", metadata)
        self.assertFalse((root / "mr_memory" / "version.py").exists())
        self.assertNotIn("PLUGIN_VERSION", main_source)
        self.assertNotIn("EXTRACTOR_VERSION", main_source)
        self.assertNotIn("overview?.version", console_script)

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
