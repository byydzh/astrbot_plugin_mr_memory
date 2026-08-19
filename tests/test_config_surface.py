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
            '<script type="module" src="./script.js?v=0.18.1-conservative-effects"></script>',
            html,
        )

        script = (
            Path.cwd() / "pages" / "console" / "script.js"
        ).read_text(encoding="utf-8")
        self.assertIn("async function waitForPluginBridge", script)
        self.assertIn("await waitForPluginBridge()", script)

    def test_recent_calls_show_persisted_memory_effects_and_fold_the_ledger(self) -> None:
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
        self.assertIn("本次实际读取或改变的记忆", html)
        self.assertIn('aria-label="本次调用实际读取或改变的持久记忆关系图"', html)
        for access, label in (
            ("read", "读取"),
            ("write", "写入"),
            ("upsert", "写入或更新"),
            ("modify", "修改"),
            ("context", "结构端点"),
        ):
            self.assertIn(f'<span data-access="{access}">{label}</span>', html)
        self.assertIn('<details class="run-detail-ledger">', html)
        self.assertIn("调用处理账本", html)
        self.assertIn('<details class="run-detail-result">', html)
        self.assertNotIn('<details class="run-detail-ledger" open', html)
        self.assertNotIn('<details class="run-detail-result" open', html)
        self.assertIn("function openRunDetail", script)
        self.assertIn("function renderRunDetailMemoryEffects", script)
        self.assertIn("function renderRunDetailLedger", script)
        self.assertIn("const value = detail?.memory_effects", script)
        self.assertIn("memoryEffectsEmptyState", script)
        self.assertIn("effects?.empty_reason", script)
        self.assertIn("copy: reason", script)
        self.assertIn("UNAVAILABLE_LEGACY", script)
        self.assertIn('return "未记录"', script)
        self.assertIn("provided[access] === null", script)
        self.assertIn('"记忆节点 / 连接"', script)
        self.assertIn("effects?.identity_exact", script)
        self.assertIn('return "账本确认身份"', script)
        self.assertIn("effects?.payload_as_of", script)
        self.assertIn('return "当前状态解析"', script)
        self.assertIn('RECORDED: "已记录"', script)
        self.assertIn('PARTIAL: "部分记录"', script)
        self.assertIn("memoryEffectsAreTruncated", script)
        self.assertIn('return metric.value > 0 ? `至少 ${formatNumber(metric.value)}` : "未知"', script)
        self.assertNotIn("精确记录", script)
        self.assertIn('read: "读取"', script)
        self.assertIn('write: "写入"', script)
        self.assertIn('upsert: "写入或更新"', script)
        self.assertIn('modify: "修改"', script)
        self.assertIn('context: "结构端点"', script)
        self.assertNotIn("function renderRunDetailGraph", script)
        self.assertNotIn("state.runDetail?.graph || {}", script)
        self.assertIn("function selectRunDetailMemoryEdge", script)
        self.assertIn('["陈述", edge.statement]', script)
        self.assertIn('["认知状态", edge.epistemic_state]', script)
        self.assertIn('["不确定性", edge.uncertainty]', script)
        self.assertIn('["效用", edge.utility]', script)
        self.assertIn('"这次反馈实际改了哪些记忆"', script)
        self.assertIn('"这次回答实际激活了哪些记忆"', script)
        self.assertIn('"scopes/<scope_id>/runs/<run_id>"', web_api)
        self.assertIn("实际激活或改变的记忆子图与处理账本", web_api)


if __name__ == "__main__":
    unittest.main()
