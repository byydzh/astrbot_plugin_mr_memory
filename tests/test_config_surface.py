from __future__ import annotations

import json
import unittest
from pathlib import Path


class ConfigSurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = json.loads(
            (Path.cwd() / "_conf_schema.json").read_text(encoding="utf-8")
        )

    def test_essential_model_and_trigger_controls_are_visible(self) -> None:
        essential = {
            "allowed_umos",
            "subconscious_provider_id",
            "distillation_thinking_mode",
            "runtime_wake_mode",
            "consult_tool_enabled",
            "feedback_window_hours",
            "feedback_debounce_seconds",
            "embedding_backend",
            "embedding_model_name",
            "auto_distillation_enabled",
            "auto_distillation_min_pending",
            "maintenance_interval_minutes",
            "private_daily_token_budget",
            "feedback_daily_token_budget",
        }
        for key in essential:
            with self.subTest(key=key):
                self.assertIn(key, self.schema)
                self.assertFalse(self.schema[key].get("invisible", False))

    def test_one_off_history_budget_is_not_a_runtime_setting(self) -> None:
        self.assertNotIn("history_backfill_daily_token_budget", self.schema)
        self.assertNotIn("history_budget_reserve_tokens", self.schema)

    def test_online_distillation_defaults_to_150_messages_or_daily(self) -> None:
        self.assertEqual(self.schema["auto_distillation_min_pending"]["default"], 150)
        self.assertEqual(self.schema["maintenance_interval_minutes"]["default"], 1440)
        self.assertEqual(self.schema["maintenance_interval_seconds"]["default"], 86400)

    def test_every_request_contract_requires_the_private_semantic_model(self) -> None:
        wake_hint = str(self.schema["runtime_wake_mode"]["hint"])
        budget_hint = str(self.schema["private_daily_token_budget"]["hint"])
        main_source = (Path.cwd() / "main.py").read_text(encoding="utf-8")
        self.assertIn("独立记忆模型完整判断", wake_hint)
        self.assertIn("每次回答前的记忆判断", budget_hint)
        self.assertIn("self._run_fast_reconstruction_with_ledger(", main_source)
        self.assertNotIn("materialize_reconstruction_packet(", main_source)

    def test_umo_guidance_uses_platform_instance_id(self) -> None:
        hint = str(self.schema["allowed_umos"]["hint"])
        self.assertIn("平台实例 ID", hint)
        self.assertIn("留空即处理所有群", hint)
        self.assertNotIn("例如 aiocqhttp:", hint)

    def test_console_defers_app_until_astrbot_bridge_is_injected(self) -> None:
        html = (
            Path.cwd() / "pages" / "console" / "index.html"
        ).read_text(encoding="utf-8")
        self.assertIn('<script type="module" src="./script.js"></script>', html)

        script = (
            Path.cwd() / "pages" / "console" / "script.js"
        ).read_text(encoding="utf-8")
        self.assertIn("async function waitForPluginBridge", script)
        self.assertIn("await waitForPluginBridge()", script)

    def test_recent_calls_open_an_evidence_graph_detail(self) -> None:
        html = (Path.cwd() / "pages" / "console" / "index.html").read_text(
            encoding="utf-8"
        )
        script = (Path.cwd() / "pages" / "console" / "script.js").read_text(
            encoding="utf-8"
        )
        web_api = (Path.cwd() / "mr_memory" / "web_api.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('id="run-detail-dialog"', html)
        self.assertIn("function openRunDetail", script)
        self.assertIn("graph.exact_memory_brief", script)
        self.assertIn('"scopes/<scope_id>/runs/<run_id>"', web_api)


if __name__ == "__main__":
    unittest.main()
