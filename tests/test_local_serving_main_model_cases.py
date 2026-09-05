from __future__ import annotations

import json
import shutil
import unittest
import uuid
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.local_serving_main_model_cases import (
    CASE_KEYS,
    build_messages,
    load_case,
    run,
)


def completion(answer: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=answer, tool_calls=[]))]
    )


class LocalServingMainModelCaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path.cwd() / ".dev" / "test-tmp" / uuid.uuid4().hex
        self.acceptance = self.root / "acceptance"
        self.output = self.root / "output"
        self.config = self.root / "config.json"
        self.acceptance.mkdir(parents=True)
        self.config.write_text("{}\n", encoding="utf-8")
        for case_key in CASE_KEYS:
            (self.acceptance / f"{case_key}.private.json").write_text(
                json.dumps(
                    {
                        "case_id": f"case-{case_key}",
                        "query": f"问题 {case_key}",
                        "envelope": {
                            "schema_version": "mr-local-serving.v1",
                            "operational_status": "COMPLETED",
                            "constraints": ["保留不确定性"],
                            "source_records": [{"id": "s1", "text": "证据"}],
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def args(self, *, execute: bool = True) -> Namespace:
        return Namespace(
            acceptance_dir=self.acceptance,
            config=self.config,
            main_provider_id="openai/gemini-3.5-flash",
            output_dir=self.output,
            max_output_tokens=1600,
            deadline_seconds=120.0,
            execute=execute,
        )

    def test_messages_preserve_query_and_full_local_envelope(self) -> None:
        case = load_case(self.acceptance, "case-a")
        messages = build_messages(case)
        self.assertEqual(len(messages), 2)
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload["current_message"], "问题 case-a")
        self.assertIn("<mr_memory_local_evidence>", payload["extra_user_content"])
        self.assertIn("保留不确定性", payload["extra_user_content"])

    def test_run_is_exactly_three_no_tool_no_retry_calls_with_raw_answers(self) -> None:
        def fake_call(**kwargs):
            self.assertIsNone(kwargs["tools"])
            self.assertEqual(kwargs["thinking_mode"], "disabled")
            self.assertFalse(kwargs["json_object"])
            request_id = f"{kwargs['run_id']}:{kwargs['phase']}:0"
            ledger = Path(kwargs["ledger_path"])
            with ledger.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps({"event": "attempted", "request_id": request_id})
                    + "\n"
                )
                handle.write(
                    json.dumps(
                        {
                            "event": "completed",
                            "request_id": request_id,
                            "usage_present": True,
                            "input_other": 100,
                            "input_cached": 0,
                            "input": 100,
                            "output": 20,
                            "total": 120,
                            "elapsed_ms": 250.0,
                        }
                    )
                    + "\n"
                )
            kwargs["budget"].reserve_call()
            return completion(f"原始回答-{kwargs['run_id']}")

        with (
            patch(
                "scripts.local_serving_main_model_cases._provider_config",
                return_value=(object(), "gemini-model", {}),
            ),
            patch(
                "scripts.local_serving_main_model_cases._provider_fingerprint",
                return_value={"provider_source_id": "source-main", "max_retries": 0},
            ),
            patch(
                "scripts.local_serving_main_model_cases._pilot_completion",
                side_effect=fake_call,
            ) as provider,
        ):
            summary = run(self.args())

        self.assertEqual(provider.call_count, 3)
        self.assertEqual(summary["main_provider_calls"], 3)
        self.assertEqual(summary["memory_provider_calls"], 0)
        self.assertEqual(summary["usage"]["provider_tokens_measured_lower_bound"], 360)
        self.assertEqual(summary["quality_status"], "PENDING_USER_REVIEW")
        self.assertEqual(len(summary["actual_answers"]), 3)
        self.assertTrue(
            all(item["actual_answer"].startswith("原始回答-") for item in summary["actual_answers"])
        )

    def test_billable_execution_requires_explicit_flag(self) -> None:
        with self.assertRaisesRegex(ValueError, "--execute"):
            run(self.args(execute=False))
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
