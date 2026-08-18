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

    def test_layered_runtime_exposes_host_owned_routing(self) -> None:
        wake_hint = str(self.schema["runtime_wake_mode"]["hint"])
        budget_hint = str(self.schema["private_daily_token_budget"]["hint"])
        main_source = (Path.cwd() / "main.py").read_text(encoding="utf-8")
        self.assertIn("low_latency", wake_hint)
        self.assertIn("balanced", wake_hint)
        self.assertIn("research", wake_hint)
        self.assertIn("manual_only", wake_hint)
        self.assertTrue(budget_hint)
        self.assertIn("self._execute_layered_reconstruction(", main_source)
        self.assertIn("RoutePolicy(", main_source)
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
        self.assertIn(
            '<script type="module" src="./script.js?v=0.18.0-trace-label"></script>',
            html,
        )

        script = (
            Path.cwd() / "pages" / "console" / "script.js"
        ).read_text(encoding="utf-8")
        self.assertIn("async function waitForPluginBridge", script)
        self.assertIn("await waitForPluginBridge()", script)

    def test_recent_calls_open_a_provenance_trace_not_a_memory_graph(self) -> None:
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
        self.assertIn("本次调用追溯链路", html)
        self.assertIn("输入证据 → 处理步骤 → 输出或反馈结果", html)
        self.assertIn("重置链路视图", html)
        self.assertIn("调用证据与处理追溯链路", html)
        self.assertNotIn("本次记忆子图", html)
        self.assertNotIn("调用记忆子图", html)
        self.assertIn("function openRunDetail", script)
        self.assertIn("function runDetailTraceSummary", script)
        self.assertIn('run: "处理运行"', script)
        self.assertIn("detail?.graph?.exact_memory_brief", script)
        self.assertIn('"包含本次反馈处理结果"', script)
        self.assertIn('"仅显示已落盘的证据与结果"', script)
        self.assertNotIn("旧记录仅还原证据", script)
        self.assertIn('"这次反馈学习处理了什么"', script)
        self.assertIn('"这次回答前实际读取了什么证据"', script)
        self.assertNotIn('"这次反馈修改了什么记忆"', script)
        self.assertIn('"scopes/<scope_id>/runs/<run_id>"', web_api)
        self.assertIn("结果与证据追溯链路", web_api)
        self.assertNotIn("结果与证据子图", web_api)


if __name__ == "__main__":
    unittest.main()
