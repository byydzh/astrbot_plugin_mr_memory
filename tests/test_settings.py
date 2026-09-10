import copy
import unittest

from mr_memory.settings import normalize_settings


class SettingsTests(unittest.TestCase):
    def test_learning_window_defaults_and_user_choice(self):
        defaults = normalize_settings({})
        self.assertTrue(defaults["learning_window_enabled"])
        self.assertEqual(defaults["learning_window_start"], "00:00")
        self.assertEqual(defaults["learning_window_end"], "06:00")
        chosen = normalize_settings({"learning_window_enabled": False,
                                     "learning_window_start": "22:30", "learning_window_end": "5:30"})
        self.assertFalse(chosen["learning_window_enabled"])
        self.assertEqual(chosen["learning_window_start"], "22:30")
        self.assertEqual(chosen["learning_window_end"], "05:30")
        with self.assertRaisesRegex(ValueError, "后台工作时段"):
            normalize_settings({"learning_window_start": "25:00"})

    def test_original_values_win_and_keep_original_time_units(self):
        original = {
            "capture_enabled": True,
            "local_serving_enabled": False,
            "local_serving_timeout_seconds": 170,
            "memory_timeout_seconds": 20,
            "max_loop_steps": 9,
            "memory_max_turns": 4,
            "distillation_max_messages": 420,
            "background_batch_size": 60,
            "distillation_max_output_tokens": 90000,
            "maintenance_llm_timeout_seconds": 1800,
            "distillation_thinking_mode": "enabled",
            "private_daily_token_budget": 0,
            "background_daily_tokens": 500000,
            "feedback_daily_token_budget": 120000,
            "maintenance_interval_minutes": 90,
            "maintenance_interval_seconds": 86400,
            "feedback_window_hours": 2,
            "feedback_window_seconds": 21600,
            "embedding_enabled": False,
            "embedding_batch_size": 4,
            "embedding_model_name": "microsoft/harrier-oss-v1-270m",
            "allowed_umos": ["demo:GroupMessage:1"],
        }
        untouched = copy.deepcopy(original)
        result = normalize_settings(original)
        for key in ("capture_enabled", "local_serving_enabled", "local_serving_timeout_seconds",
                    "max_loop_steps", "distillation_max_messages", "distillation_max_output_tokens",
                    "maintenance_llm_timeout_seconds", "distillation_thinking_mode",
                    "private_daily_token_budget", "feedback_daily_token_budget", "embedding_enabled",
                    "embedding_batch_size"):
            self.assertEqual(result[key], original[key])
        self.assertEqual(result["maintenance_interval_seconds"], 5400)
        self.assertEqual(result["feedback_window_seconds"], 7200)
        self.assertEqual(result["embedding_query_prompt_name"], "web_search_query")
        result["allowed_umos"].append("demo:GroupMessage:2")
        self.assertEqual(original, untouched)
        self.assertNotIn("memory_timeout_seconds", result)

    def test_reduced_names_migrate_only_when_original_field_is_missing(self):
        result = normalize_settings({
            "memory_timeout_seconds": 25,
            "memory_max_turns": 5,
            "background_min_messages": 35,
            "background_batch_size": 65,
            "background_timeout_seconds": 95,
            "background_max_tokens": 4100,
            "background_daily_tokens": 230000,
            "background_interval_seconds": 45,
            "recent_messages": 30,
        })
        self.assertEqual({key: result[key] for key in (
            "local_serving_timeout_seconds", "max_loop_steps", "auto_distillation_min_pending",
            "distillation_max_messages", "maintenance_llm_timeout_seconds",
            "distillation_max_output_tokens", "private_daily_token_budget", "recent_messages",
        )}, {
            "local_serving_timeout_seconds": 25, "max_loop_steps": 5,
            "auto_distillation_min_pending": 35, "distillation_max_messages": 65,
            "maintenance_llm_timeout_seconds": 95, "distillation_max_output_tokens": 4100,
            "private_daily_token_budget": 230000, "recent_messages": 30,
        })
        self.assertEqual(result["background_interval_seconds"], 45)
        self.assertEqual(result["maintenance_interval_seconds"], 86400)
        self.assertEqual(result["memory_max_tokens"], 8192)
        self.assertEqual(result["feedback_daily_token_budget"], 500000)
        self.assertEqual(result["feedback_window_seconds"], 21600)
        self.assertEqual(result["feedback_debounce_seconds"], 15)
        self.assertEqual(result["distillation_thinking_mode"], "enabled")


if __name__ == "__main__":
    unittest.main()
