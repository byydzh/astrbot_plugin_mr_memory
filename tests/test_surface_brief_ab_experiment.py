from __future__ import annotations

import json
import shutil
import unittest
import uuid
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.surface_brief_ab_experiment import (
    DEFAULT_MAIN_PROVIDER_ID,
    SURFACE_SYSTEM_PROMPT,
    build_parser,
    build_surface_messages,
    generate_command,
    load_case,
    parse_judge_response,
    score_command,
)


def _completion(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    reasoning_content="",
                    tool_calls=[],
                )
            )
        ]
    )


class SurfaceBriefAbExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path.cwd() / ".dev" / f"surface-ab-test-{uuid.uuid4().hex}"
        self.root.mkdir(parents=True)
        self.case_path = self.root / "case.json"
        self.control_path = self.root / "control.json"
        self.memory_path = self.root / "memory.json"
        self.config_path = self.root / "config.json"
        self.output_dir = self.root / "output"
        self.case = {
            "schema_version": "surface.brief.case.v1",
            "case_id": "case-726",
            "query": "他这算口嫌体正直吗？",
            "recent_context": [
                {"sender": "甲", "content": "这个游戏我还是买了"},
                {"sender": "乙", "content": "你之前不是说玩吐了吗"},
            ],
            "cutoff_at": 200,
        }
        self.brief = {
            "claims": [
                {
                    "statement": "同一账号先说玩吐，后来又买了相关游戏。",
                    "source_keys": ["source-a", "source-b"],
                    "confidence": 0.8,
                }
            ],
            "conflicts": [],
            "unresolved": [
                {
                    "statement": "不能把购买动机说成当事人的原话。",
                    "source_keys": ["source-b"],
                }
            ],
        }
        values = {
            self.case_path: self.case,
            self.control_path: {
                "schema_version": "surface.brief.arm.v1",
                "case_id": "case-726",
                "arm_id": "control",
                "memory_brief": None,
            },
            self.memory_path: {
                "schema_version": "surface.brief.arm.v1",
                "case_id": "case-726",
                "arm_id": "eccr",
                "memory_brief": self.brief,
            },
            self.config_path: {},
        }
        for path, value in values.items():
            path.write_text(
                json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _generate_args(self) -> Namespace:
        return Namespace(
            case=str(self.case_path),
            arm=[str(self.control_path), str(self.memory_path)],
            config=str(self.config_path),
            main_provider_id=DEFAULT_MAIN_PROVIDER_ID,
            output_dir=str(self.output_dir),
            repetitions=1,
            max_provider_calls=2,
            soft_token_limit=0,
            thinking_mode="disabled",
            max_output_tokens=1200,
            deadline_seconds=120.0,
            resume=False,
        )

    def _generate(self) -> tuple[dict, object]:
        with (
            patch(
                "scripts.surface_brief_ab_experiment._provider_config",
                return_value=(object(), "main-model", {}),
            ),
            patch(
                "scripts.surface_brief_ab_experiment._provider_fingerprint",
                return_value={"provider_source_id": "source-main"},
            ),
            patch(
                "scripts.surface_brief_ab_experiment._pilot_completion",
                side_effect=[
                    _completion("控制回答"),
                    _completion("带记忆且保留疑虑的回答"),
                ],
            ) as completion,
        ):
            summary = generate_command(self._generate_args())
        return summary, completion

    def test_surface_prompt_treats_brief_as_evidence_and_keeps_context_untrusted(
        self,
    ) -> None:
        messages = build_surface_messages(self.case, self.brief)
        self.assertEqual(len(messages), 2)
        self.assertIn("fallible evidence", SURFACE_SYSTEM_PROMPT)
        self.assertIn("conflict or unresolved uncertainty", SURFACE_SYSTEM_PROMPT)
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload["current_message"], self.case["query"])
        self.assertEqual(payload["recent_context"], self.case["recent_context"])
        self.assertEqual(payload["memory_brief_evidence"], self.brief)
        self.assertNotIn("arm_id", payload)

    def test_case_rejects_embedded_gold_fields(self) -> None:
        value = dict(self.case)
        value["rubric"] = {"required_semantics": ["leak"]}
        self.case_path.write_text(
            json.dumps(value, ensure_ascii=False), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "evaluation-only gold"):
            load_case(self.case_path)

    def test_generate_calls_real_provider_adapter_without_tools_and_writes_private_results(
        self,
    ) -> None:
        summary, completion = self._generate()
        self.assertEqual(summary["completed_runs"], 2)
        self.assertEqual(summary["evaluation_status"], "NOT_SCORED_GOLD_NOT_LOADED")
        self.assertEqual(completion.call_count, 2)
        for call in completion.call_args_list:
            kwargs = call.kwargs
            self.assertIsNone(kwargs["tools"])
            self.assertFalse(kwargs["json_object"])
            self.assertEqual(kwargs["thinking_mode"], "disabled")
            self.assertEqual(kwargs["max_output_tokens"], 1200)
        private = json.loads(
            (self.output_dir / "private_results.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(private["results"]), 2)
        self.assertTrue(all(item["tools_sent"] is False for item in private["results"]))
        manifest_text = (self.output_dir / "manifest.json").read_text(encoding="utf-8")
        self.assertNotIn("gold_sha256", manifest_text)
        self.assertNotIn("rubric", manifest_text)

    def test_gold_is_loaded_only_by_separate_blinded_score_stage(self) -> None:
        self._generate()
        gold_path = self.root / "gold.json"
        gold_path.write_text(
            json.dumps(
                {
                    "schema_version": "surface.brief.gold.v1",
                    "case_id": "case-726",
                    "rubric": {
                        "required_semantics": ["指出前后行为反差"],
                        "required_uncertainty": ["购买动机不是逐字事实"],
                        "forbidden_conclusions": ["客观人格诊断"],
                        "style_constraints": ["自然简洁"],
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        captured: dict[str, object] = {}

        def judge(**kwargs):
            captured.update(kwargs)
            payload = json.loads(kwargs["messages"][1]["content"])
            answer_ids = [item["answer_id"] for item in payload["answers"]]
            scores = [
                {
                    "answer_id": answer_id,
                    "groundedness": 4,
                    "query_answering": 4,
                    "uncertainty_preservation": 4,
                    "hallucination_control": 5,
                    "naturalness": 4,
                    "overall_score": 84,
                    "fatal_errors": [],
                    "brief_reason": "答案有依据且保留疑虑。",
                }
                for answer_id in answer_ids
            ]
            return _completion(
                json.dumps(
                    {
                        "scores": scores,
                        "ranking": answer_ids,
                        "comparison_note": "盲评完成。",
                    },
                    ensure_ascii=False,
                )
            )

        args = Namespace(
            generation_dir=str(self.output_dir),
            gold=str(gold_path),
            config=str(self.config_path),
            main_provider_id=DEFAULT_MAIN_PROVIDER_ID,
            max_provider_calls=1,
            soft_token_limit=0,
            thinking_mode="disabled",
            max_output_tokens=6000,
            deadline_seconds=120.0,
            resume=False,
        )
        with (
            patch(
                "scripts.surface_brief_ab_experiment._provider_config",
                return_value=(object(), "main-model", {}),
            ),
            patch(
                "scripts.surface_brief_ab_experiment._provider_fingerprint",
                return_value={"provider_source_id": "source-main"},
            ),
            patch(
                "scripts.surface_brief_ab_experiment._pilot_completion",
                side_effect=judge,
            ),
        ):
            summary = score_command(args)
        self.assertEqual(summary["status"], "COMPLETED")
        self.assertTrue(summary["gold_loaded_after_generation"])
        self.assertEqual(
            {item["arm_id"] for item in summary["arms"]}, {"control", "eccr"}
        )
        self.assertIsNone(captured["tools"])
        self.assertTrue(captured["json_object"])
        judge_payload = json.loads(captured["messages"][1]["content"])
        self.assertIn("post_generation_gold", judge_payload)
        self.assertNotIn("arm_id", json.dumps(judge_payload["answers"]))
        self.assertTrue((self.output_dir / "score" / "result.private.json").is_file())

    def test_score_rejects_gold_reusing_case_file(self) -> None:
        self._generate()
        args = Namespace(
            generation_dir=str(self.output_dir),
            gold=str(self.case_path),
            config=str(self.config_path),
            main_provider_id=DEFAULT_MAIN_PROVIDER_ID,
            max_provider_calls=1,
            soft_token_limit=0,
            thinking_mode="disabled",
            max_output_tokens=6000,
            deadline_seconds=120.0,
            resume=False,
        )
        with self.assertRaisesRegex(ValueError, "physically separate"):
            score_command(args)

    def test_judge_parser_requires_exact_ids_and_bounded_scores(self) -> None:
        valid = {
            "scores": [
                {
                    "answer_id": "answer-01",
                    "groundedness": 5,
                    "query_answering": 4,
                    "uncertainty_preservation": 3,
                    "hallucination_control": 5,
                    "naturalness": 4,
                    "overall_score": 84,
                    "fatal_errors": [],
                    "brief_reason": "合格。",
                }
            ],
            "ranking": ["answer-01"],
            "comparison_note": "只有一个答案。",
        }
        parsed = parse_judge_response(
            json.dumps(valid, ensure_ascii=False), answer_ids={"answer-01"}
        )
        self.assertEqual(parsed["scores"][0]["composite_score"], 84.0)
        invalid = json.loads(json.dumps(valid))
        invalid["scores"][0]["groundedness"] = 6
        with self.assertRaisesRegex(ValueError, "integer 0..5"):
            parse_judge_response(json.dumps(invalid), answer_ids={"answer-01"})
        fractional = json.loads(json.dumps(valid))
        fractional["scores"][0]["groundedness"] = 4.5
        with self.assertRaisesRegex(ValueError, "integer 0..5"):
            parse_judge_response(json.dumps(fractional), answer_ids={"answer-01"})

    def test_judge_parser_accepts_fenced_json_with_leading_text(self) -> None:
        valid = {
            "scores": [
                {
                    "answer_id": "answer-01",
                    "groundedness": 5,
                    "query_answering": 4,
                    "uncertainty_preservation": 3,
                    "hallucination_control": 5,
                    "naturalness": 4,
                    "overall_score": 84,
                    "fatal_errors": [],
                    "brief_reason": "合格。",
                }
            ],
            "ranking": ["answer-01"],
            "comparison_note": "只有一个答案。",
        }
        parsed = parse_judge_response(
            "评分如下：\n```json\n"
            + json.dumps(valid, ensure_ascii=False)
            + "\n```",
            answer_ids={"answer-01"},
        )
        self.assertEqual(parsed["ranking"], ["answer-01"])

    def test_score_rejects_incomplete_declared_arm_matrix(self) -> None:
        self._generate()
        private_path = self.output_dir / "private_results.json"
        private = json.loads(private_path.read_text(encoding="utf-8"))
        private["results"].pop()
        private_path.write_text(
            json.dumps(private, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        gold_path = self.root / "gold.json"
        gold_path.write_text(
            json.dumps(
                {
                    "schema_version": "surface.brief.gold.v1",
                    "case_id": "case-726",
                    "rubric": {"required_semantics": ["指出反差"]},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        args = Namespace(
            generation_dir=str(self.output_dir),
            gold=str(gold_path),
            config=str(self.config_path),
            main_provider_id=DEFAULT_MAIN_PROVIDER_ID,
            max_provider_calls=1,
            soft_token_limit=0,
            thinking_mode="disabled",
            max_output_tokens=6000,
            deadline_seconds=120.0,
            resume=False,
        )
        with self.assertRaisesRegex(ValueError, "declared arm matrix"):
            score_command(args)

    def test_cli_defaults_to_main_provider_non_thinking_and_bounded_output(
        self,
    ) -> None:
        args = build_parser().parse_args(
            [
                "generate",
                "--case",
                str(self.case_path),
                "--arm",
                str(self.control_path),
                "--config",
                str(self.config_path),
                "--output-dir",
                str(self.output_dir),
            ]
        )
        self.assertEqual(args.main_provider_id, DEFAULT_MAIN_PROVIDER_ID)
        self.assertEqual(args.thinking_mode, "disabled")
        self.assertEqual(args.max_output_tokens, 1200)


if __name__ == "__main__":
    unittest.main()
