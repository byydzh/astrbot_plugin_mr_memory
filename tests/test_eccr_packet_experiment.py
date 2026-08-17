from __future__ import annotations

import copy
import json
import shutil
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mr_memory.usage import TokenUsageRecord
from scripts.eccr_packet_experiment import (
    DIAGNOSTIC_LAYER,
    _run_deterministic,
    _run_eccr,
    _run_eccr_audit,
    _run_one_pass,
    build_eccr_messages,
    build_one_pass_messages,
    build_summary,
    load_case_bundle,
    run_case_arm,
    score_result,
)
from scripts.masked_ab_experiment import PilotBudget, _usage_ledger_audit


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


class EccrPacketExperimentTests(unittest.TestCase):
    umo = "shadow:GroupMessage:group-a"

    def setUp(self) -> None:
        self.root = Path.cwd() / ".dev" / f"eccr-packet-test-{uuid.uuid4().hex}"
        self.case_dir = self.root / "case-a"
        self.case_dir.mkdir(parents=True)
        self.case = {
            "schema_version": "eccr.packet.case.v1",
            "case_id": "case-a",
            "layer": DIAGNOSTIC_LAYER,
            "query": "甲以前的昵称和现在的昵称是不是同一个账号？",
            "umo": self.umo,
            "cutoff_at": 200,
            "authorized_participant_keys": ["user-1"],
            "host_subject_bindings": [
                {
                    "reference": "甲",
                    "participant_key": "user-1",
                    "mode": "HOST",
                    "candidate_participant_keys": [],
                    "source_keys": ["source-1"],
                    "valid_at": 100,
                }
            ],
        }
        self.packet = {
            "candidates": {"feedback_hypotheses": [], "associations": []},
            "expanded_episodes": [
                {
                    "source_key": "source-1",
                    "umo": self.umo,
                    "sent_at": 100,
                    "sender_id": "user-1",
                    "sender_participant_key": "user-1",
                    "sender_name": "旧昵称",
                    "plain_text": "证据文本",
                }
            ],
        }
        self.gold = {
            "schema_version": "eccr.packet.gold.v1",
            "case_id": "case-a",
            "evidence_groups": {
                "identity": {"required_any": ["source-1"], "support": []}
            },
            "identity": {
                "expected_participant_keys": ["user-1"],
                "expected_binding_modes": ["HOST"],
            },
            "required_semantics": ["SECRET_GOLD_SENTINEL"],
            "required_uncertainty": [],
            "forbidden_conclusions": [],
        }
        self._write_inputs()

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _write_inputs(self) -> None:
        for name, value in (
            ("case.json", self.case),
            ("evidence_packet.json", self.packet),
            ("gold.json", self.gold),
        ):
            (self.case_dir / name).write_text(
                json.dumps(value, ensure_ascii=False), encoding="utf-8"
            )

    def _snapshot_identity_inputs(
        self,
    ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        case = copy.deepcopy(self.case)
        case["cutoff_at"] = 200
        case["host_subject_bindings"] = [
            {
                "reference": reference,
                "participant_key": "user-1",
                "mode": "HOST",
                "candidate_participant_keys": [],
                "source_keys": [source_key],
                "valid_at": valid_at,
            }
            for reference, source_key, valid_at in (
                ("旧名", "source-1", 100),
                ("现名", "source-2", 140),
                ("未来名", "source-3", 150),
            )
        ]
        messages = [
            {
                "source_key": "source-1",
                "scope_token": self.umo,
                "sent_at": 100,
                "sender_participant_key": "user-1",
                "display_name": "旧名",
                "plain_text": "旧名时期",
            },
            {
                "source_key": "source-2",
                "scope_token": self.umo,
                "sent_at": 140,
                "sender_participant_key": "user-1",
                "display_name": "现名",
                "plain_text": "现名时期",
            },
            {
                "source_key": "source-3",
                "scope_token": self.umo,
                "sent_at": 150,
                "sender_participant_key": "user-1",
                "display_name": "未来名",
                "plain_text": "恰好位于 cutoff，不应可见",
            },
            {
                "source_key": "source-x",
                "scope_token": "shadow:GroupMessage:group-b",
                "sent_at": 130,
                "sender_participant_key": "user-2",
                "display_name": "现名",
                "plain_text": "其他群同名干扰",
            },
        ]
        packet = {
            "candidates": {"feedback_hypotheses": [], "associations": []},
            "snapshots": [
                {
                    "cutoff_id": "cut-1",
                    "cutoff_at": 150,
                    "query_scope_token": self.umo,
                    "messages": messages,
                }
            ],
        }
        gold = {
            "schema_version": "eccr.packet.gold.v1",
            "case_id": "case-a",
            "evidence_groups": {
                "identity": {
                    "required_any": ["source-1", "source-2"],
                    "support": [],
                }
            },
            "identity": {
                "expected_participant_keys": ["user-1"],
                "expected_binding_modes": ["HOST"],
            },
            "snapshots": [
                {
                    "cutoff_id": "cut-1",
                    "cutoff_at": 150,
                    "target_scope_token": self.umo,
                    "target_actor_token": "user-1",
                    "visible_aliases_same_actor": ["旧名", "现名"],
                    "expected_latest_display_name": "现名",
                    "post_cutoff_aliases_forbidden": ["未来名"],
                    "cross_scope_actor_tokens_forbidden": ["user-2"],
                }
            ],
            "required_semantics": [],
            "required_uncertainty": [],
            "forbidden_conclusions": [],
        }
        return case, packet, gold

    @staticmethod
    def _eccr_response(*, terminal: bool = True) -> str:
        return json.dumps(
            {
                "contract": {
                    "contract_id": "model-placeholder",
                    "scope_sha256": "0" * 64,
                    "query_sha256": "0" * 64,
                    "cutoff_at": 1,
                    "revision_vector": {
                        "message": "x",
                        "graph": "x",
                        "identity": "x",
                        "relation": "x",
                        "feedback": "x",
                        "protocol": "x",
                    },
                    "step_index": 0,
                    "subjects": [
                        {
                            "reference": "甲",
                            "participant_key": "user-1",
                            "mode": "HOST",
                            "candidate_participant_keys": [],
                            "source_keys": ["source-1"],
                            "valid_at": 100,
                        }
                    ],
                    "obligations": [
                        {
                            "id": "identity",
                            "kind": "identity",
                            "question": "甲是否绑定到 user-1？",
                            "critical": True,
                            "status": "SUPPORTED" if terminal else "OPEN",
                            "support_keys": ["source-1"] if terminal else [],
                            "counter_keys": [],
                            "last_changed_step": 0,
                        }
                    ],
                    "interpretations": [],
                    "uncertainties": [],
                    "guarded_claims": [],
                    "visited_source_keys": ["source-1"],
                    "selected_edge_ids": [],
                    "selected_hypothesis_ids": [],
                    "tried_action_signatures": [],
                    "exhausted_discriminators": [],
                    "frontier_discriminators": [],
                },
                "actions": [],
                "memory_brief": (
                    {
                        "claims": [
                            {
                                "statement": "甲绑定到同一个稳定账号。",
                                "source_keys": ["source-1"],
                                "confidence": 0.99,
                            }
                        ],
                        "conflicts": [],
                        "unresolved": [],
                    }
                    if terminal
                    else None
                ),
                "terminal": terminal,
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _eccr_audit_responses() -> tuple[str, str]:
        base_contract = {
            "contract_id": "model-placeholder",
            "scope_sha256": "0" * 64,
            "query_sha256": "0" * 64,
            "cutoff_at": 1,
            "revision_vector": {
                "message": "x",
                "graph": "x",
                "identity": "x",
                "relation": "x",
                "feedback": "x",
                "protocol": "x",
            },
            "step_index": 0,
            "subjects": [
                {
                    "reference": "甲",
                    "participant_key": "user-1",
                    "mode": "HOST",
                    "candidate_participant_keys": [],
                    "source_keys": ["source-1"],
                    "valid_at": 100,
                }
            ],
            "obligations": [
                {
                    "id": "identity",
                    "kind": "identity",
                    "question": "甲是否绑定到 user-1？",
                    "critical": True,
                    "status": "OPEN",
                    "support_keys": ["source-1"],
                    "counter_keys": [],
                    "last_changed_step": 0,
                }
            ],
            "interpretations": [
                {
                    "id": "same-account",
                    "statement": "两个称呼指向同一稳定账号。",
                    "status": "CANDIDATE",
                    "support_keys": ["source-1"],
                    "counter_keys": [],
                    "uncertainty": "仍需检查同名反例。",
                },
                {
                    "id": "same-name-decoy",
                    "statement": "相同称呼也可能来自不同账号。",
                    "status": "CANDIDATE",
                    "support_keys": [],
                    "counter_keys": [],
                    "uncertainty": "需要审计未选证据。",
                },
            ],
            "uncertainties": [
                {
                    "id": "homonym-risk",
                    "statement": "同名消息可能造成错误绑定。",
                    "status": "OPEN",
                    "source_keys": ["source-1"],
                }
            ],
            "guarded_claims": ["不得仅凭昵称字符串合并账号。"],
            "visited_source_keys": ["source-1"],
            "selected_edge_ids": [],
            "selected_hypothesis_ids": [],
            "tried_action_signatures": [],
            "exhausted_discriminators": [],
            "frontier_discriminators": [],
        }
        compile_response = {
            "contract": copy.deepcopy(base_contract),
            "actions": [],
            "memory_brief": None,
            "terminal": False,
        }
        reviewed_contract = copy.deepcopy(base_contract)
        reviewed_contract["obligations"][0].update(
            {
                "status": "CONTESTED",
                "counter_keys": ["source-2"],
                "last_changed_step": 1,
            }
        )
        reviewed_contract["interpretations"][0].update(
            {"status": "CONTESTED", "counter_keys": ["source-2"]}
        )
        reviewed_contract["interpretations"][1].update(
            {"status": "SUPPORTED", "support_keys": ["source-2"]}
        )
        reviewed_contract["uncertainties"][0].update(
            {"status": "PRESERVED", "source_keys": ["source-1", "source-2"]}
        )
        reviewed_contract["visited_source_keys"] = ["source-1", "source-2"]
        review_response = {
            "contract": reviewed_contract,
            "actions": [],
            "memory_brief": {
                "claims": [],
                "conflicts": [
                    {
                        "statement": "固定 packet 同时含稳定账号线索和同名反例。",
                        "source_keys": ["source-1", "source-2"],
                    }
                ],
                "unresolved": [
                    {
                        "statement": "不能仅凭昵称字符串确定账号同一性。",
                        "source_keys": ["source-1", "source-2"],
                    }
                ],
            },
            "terminal": True,
        }
        return (
            json.dumps(compile_response, ensure_ascii=False),
            json.dumps(review_response, ensure_ascii=False),
        )

    def _audit_packet(self) -> dict[str, object]:
        packet = copy.deepcopy(self.packet)
        packet["chronological_neighborhoods"] = [
            {
                "neighborhood_id": "n-1",
                "messages": [
                    {
                        "source_key": "source-2",
                        "umo": self.umo,
                        "sent_at": 120,
                        "sender_id": "user-2",
                        "sender_participant_key": "user-2",
                        "sender_name": "同名干扰",
                        "plain_text": "覆盖审计反例",
                    }
                ],
            }
        ]
        return packet

    def test_gold_is_physically_separate_and_never_enters_model_messages(self) -> None:
        bundle = load_case_bundle(self.case_dir)
        one_pass = json.dumps(
            build_one_pass_messages(bundle.case, bundle.packet), ensure_ascii=False
        )
        eccr = json.dumps(
            build_eccr_messages(bundle.case, bundle.packet)[0], ensure_ascii=False
        )
        self.assertNotIn("SECRET_GOLD_SENTINEL", one_pass)
        self.assertNotIn("SECRET_GOLD_SENTINEL", eccr)
        self.assertNotEqual(
            bundle.hashes["gold_sha256"], bundle.hashes["evidence_packet_sha256"]
        )

    def test_case_json_rejects_gold_fields(self) -> None:
        self.case["required_semantics"] = ["leak"]
        self._write_inputs()
        with self.assertRaisesRegex(ValueError, "evaluation-only gold"):
            load_case_bundle(self.case_dir)

    def test_packet_rejects_future_evidence(self) -> None:
        self.packet["expanded_episodes"][0]["sent_at"] = 200
        self._write_inputs()
        with self.assertRaisesRegex(ValueError, "strictly before cutoff"):
            load_case_bundle(self.case_dir)

    def test_one_pass_is_exactly_one_no_tool_provider_call(self) -> None:
        response = json.dumps(
            {
                "decision": "brief",
                "memory_brief": {
                    "claims": [
                        {
                            "statement": "证据支持稳定账号绑定。",
                            "source_keys": ["source-1"],
                            "confidence": 0.9,
                        }
                    ],
                    "conflicts": [],
                    "unresolved": [],
                },
                "activate_hypotheses": [],
                "activate_edges": [],
                "escalation_question": "",
            },
            ensure_ascii=False,
        )
        with patch(
            "scripts.eccr_packet_experiment._pilot_completion",
            return_value=_completion(response),
        ) as mocked:
            result = _run_one_pass(
                case=self.case,
                packet=self.packet,
                client=object(),
                provider_id="deepseek/test",
                model="deepseek-v4-flash",
                provider_extra_body={},
                max_output_tokens=32000,
                thinking_mode="enabled",
                deadline_seconds=30,
                ledger_path=self.root / "unused.jsonl",
                budget=PilotBudget(max_calls=1, soft_token_limit=0),
                run_id="case-a:one-pass:rep-001",
                repetition=1,
            )
        self.assertEqual(mocked.call_count, 1)
        self.assertIsNone(mocked.call_args.kwargs["tools"])
        self.assertEqual(mocked.call_args.kwargs["arm"], "one-pass")
        self.assertEqual(result["model_calls"], 1)
        self.assertFalse(result["retrieval_available"])
        self.assertEqual(result["brief"]["claims"][0]["source_keys"], ["source-1"])

    def test_eccr_closes_one_contract_in_one_no_tool_call(self) -> None:
        with patch(
            "scripts.eccr_packet_experiment._pilot_completion",
            return_value=_completion(self._eccr_response()),
        ) as mocked:
            result = _run_eccr(
                case=self.case,
                packet=self.packet,
                client=object(),
                provider_id="deepseek/test",
                model="deepseek-v4-flash",
                provider_extra_body={},
                max_output_tokens=32000,
                thinking_mode="enabled",
                deadline_seconds=30,
                ledger_path=self.root / "unused.jsonl",
                budget=PilotBudget(max_calls=1, soft_token_limit=0),
                run_id="case-a:eccr:rep-001",
                repetition=1,
            )
        self.assertEqual(mocked.call_count, 1)
        self.assertIsNone(mocked.call_args.kwargs["tools"])
        self.assertEqual(mocked.call_args.kwargs["arm"], "eccr")
        self.assertTrue(result["terminal"])
        self.assertEqual(result["retrieval_rounds"], 0)
        self.assertEqual(result["subjects"][0]["participant_key"], "user-1")
        self.assertNotEqual(result["contract"]["scope_sha256"], "0" * 64)

    def test_eccr_host_adds_cited_sources_to_visited_before_strict_parse(self) -> None:
        response = json.loads(self._eccr_response())
        response["contract"]["visited_source_keys"] = []
        with patch(
            "scripts.eccr_packet_experiment._pilot_completion",
            return_value=_completion(json.dumps(response, ensure_ascii=False)),
        ):
            result = _run_eccr(
                case=self.case,
                packet=self.packet,
                client=object(),
                provider_id="deepseek/test",
                model="deepseek-v4-flash",
                provider_extra_body={},
                max_output_tokens=32000,
                thinking_mode="enabled",
                deadline_seconds=30,
                ledger_path=self.root / "unused.jsonl",
                budget=PilotBudget(max_calls=1, soft_token_limit=0),
                run_id="case-a:eccr:rep-001",
                repetition=1,
            )
        self.assertEqual(result["visited_source_keys"], ["source-1"])

    def test_eccr_rejects_subject_binding_that_contradicts_host_truth(self) -> None:
        self.case["authorized_participant_keys"].append("user-2")
        response = json.loads(self._eccr_response())
        response["contract"]["subjects"][0]["participant_key"] = "user-2"
        with patch(
            "scripts.eccr_packet_experiment._pilot_completion",
            return_value=_completion(json.dumps(response, ensure_ascii=False)),
        ):
            with self.assertRaisesRegex(ValueError, "contradicts host binding"):
                _run_eccr(
                    case=self.case,
                    packet=self.packet,
                    client=object(),
                    provider_id="deepseek/test",
                    model="deepseek-v4-flash",
                    provider_extra_body={},
                    max_output_tokens=32000,
                    thinking_mode="enabled",
                    deadline_seconds=30,
                    ledger_path=self.root / "unused.jsonl",
                    budget=PilotBudget(max_calls=1, soft_token_limit=0),
                    run_id="case-a:eccr:rep-001",
                    repetition=1,
                )

    def test_eccr_qualified_close_requires_visible_unresolved_brief(self) -> None:
        response = json.loads(self._eccr_response())
        obligation = response["contract"]["obligations"][0]
        obligation["status"] = "EXHAUSTED"
        obligation["support_keys"] = []
        response["memory_brief"] = None
        with patch(
            "scripts.eccr_packet_experiment._pilot_completion",
            return_value=_completion(json.dumps(response, ensure_ascii=False)),
        ):
            with self.assertRaisesRegex(
                ValueError, "surface every|explicit unresolved brief"
            ):
                _run_eccr(
                    case=self.case,
                    packet=self.packet,
                    client=object(),
                    provider_id="deepseek/test",
                    model="deepseek-v4-flash",
                    provider_extra_body={},
                    max_output_tokens=32000,
                    thinking_mode="enabled",
                    deadline_seconds=30,
                    ledger_path=self.root / "unused.jsonl",
                    budget=PilotBudget(max_calls=1, soft_token_limit=0),
                    run_id="case-a:eccr:rep-001",
                    repetition=1,
                )

    def test_eccr_rejects_nonterminal_first_call_instead_of_silently_looping(
        self,
    ) -> None:
        with patch(
            "scripts.eccr_packet_experiment._pilot_completion",
            return_value=_completion(self._eccr_response(terminal=False)),
        ):
            with self.assertRaisesRegex(ValueError, "must close in its only"):
                _run_eccr(
                    case=self.case,
                    packet=self.packet,
                    client=object(),
                    provider_id="deepseek/test",
                    model="deepseek-v4-flash",
                    provider_extra_body={},
                    max_output_tokens=32000,
                    thinking_mode="enabled",
                    deadline_seconds=30,
                    ledger_path=self.root / "unused.jsonl",
                    budget=PilotBudget(max_calls=1, soft_token_limit=0),
                    run_id="case-a:eccr:rep-001",
                    repetition=1,
                )

    def test_eccr_audit_is_exactly_two_no_tool_provider_calls(self) -> None:
        compile_response, review_response = self._eccr_audit_responses()
        packet = self._audit_packet()
        with patch(
            "scripts.eccr_packet_experiment._pilot_completion",
            side_effect=[
                _completion(compile_response),
                _completion(review_response),
            ],
        ) as mocked:
            result = _run_eccr_audit(
                case=self.case,
                packet=packet,
                client=object(),
                provider_id="deepseek/test",
                model="deepseek-v4-flash",
                provider_extra_body={},
                max_output_tokens=32000,
                thinking_mode="enabled",
                deadline_seconds=30,
                ledger_path=self.root / "unused.jsonl",
                budget=PilotBudget(max_calls=2, soft_token_limit=0),
                run_id="case-a:eccr-audit:rep-001",
                repetition=1,
            )

        self.assertEqual(mocked.call_count, 2)
        calls = mocked.call_args_list
        self.assertEqual([item.kwargs["call_index"] for item in calls], [0, 1])
        self.assertEqual(
            [item.kwargs["phase"] for item in calls],
            [
                "oracle_packet_audit_compile",
                "oracle_packet_counterexample_coverage_audit",
            ],
        )
        self.assertTrue(all(item.kwargs["tools"] is None for item in calls))
        self.assertTrue(all(item.kwargs["arm"] == "eccr-audit" for item in calls))
        self.assertEqual(result["model_calls"], 2)
        self.assertTrue(result["terminal"])
        self.assertFalse(result["retrieval_available"])
        self.assertEqual(len(result["rounds"]), 2)
        self.assertFalse(result["rounds"][0]["terminal"])
        self.assertTrue(result["rounds"][1]["terminal"])
        self.assertIsNone(result["rounds"][0]["brief"])
        self.assertIsNotNone(result["rounds"][1]["brief"])
        self.assertEqual(len(result["model_input_sha256_by_round"]), 2)
        self.assertTrue(
            all(len(value) == 64 for value in result["model_input_sha256_by_round"])
        )
        self.assertEqual(result["first_round_selected_source_keys"], ["source-1"])
        self.assertEqual(result["first_round_unselected_source_keys"], ["source-2"])

        prompts = json.dumps(
            [item.kwargs["messages"] for item in calls], ensure_ascii=False
        )
        self.assertNotIn("SECRET_GOLD_SENTINEL", prompts)
        self.assertIn("chronological_neighborhoods", prompts)
        self.assertIn("n-1", prompts)
        review_prompt = json.loads(calls[1].kwargs["messages"][1]["content"])
        source_status = review_prompt[
            "audit_marked_packet_preserving_original_neighborhoods_and_snapshots"
        ]["chronological_neighborhoods"][0]["messages"][0]["_eccr_audit"][
            "source_selection"
        ][
            0
        ]
        self.assertFalse(source_status["first_round_selected"])

    def test_eccr_audit_second_call_cannot_drop_obligations_or_uncertainties(
        self,
    ) -> None:
        original_compile, original_review = self._eccr_audit_responses()
        packet = self._audit_packet()
        variants: list[tuple[str, str, str]] = []

        compile_with_extra = json.loads(original_compile)
        compile_with_extra["contract"]["obligations"].append(
            {
                "id": "counterexample-coverage",
                "kind": "coverage",
                "question": "是否检查了未选证据？",
                "critical": False,
                "status": "OPEN",
                "support_keys": [],
                "counter_keys": [],
                "last_changed_step": 0,
            }
        )
        variants.append(
            (
                "obligation set",
                json.dumps(compile_with_extra, ensure_ascii=False),
                original_review,
            )
        )

        review_without_uncertainty = json.loads(original_review)
        review_without_uncertainty["contract"]["uncertainties"] = []
        variants.append(
            (
                "uncertainty set",
                original_compile,
                json.dumps(review_without_uncertainty, ensure_ascii=False),
            )
        )

        for expected, compile_response, review_response in variants:
            with self.subTest(expected=expected):
                with patch(
                    "scripts.eccr_packet_experiment._pilot_completion",
                    side_effect=[
                        _completion(compile_response),
                        _completion(review_response),
                    ],
                ):
                    with self.assertRaisesRegex(ValueError, expected):
                        _run_eccr_audit(
                            case=self.case,
                            packet=packet,
                            client=object(),
                            provider_id="deepseek/test",
                            model="deepseek-v4-flash",
                            provider_extra_body={},
                            max_output_tokens=32000,
                            thinking_mode="enabled",
                            deadline_seconds=30,
                            ledger_path=self.root / "unused.jsonl",
                            budget=PilotBudget(max_calls=2, soft_token_limit=0),
                            run_id="case-a:eccr-audit:rep-001",
                            repetition=1,
                        )

    def test_eccr_audit_restores_only_omitted_previous_evidence(self) -> None:
        compile_response, review_response = self._eccr_audit_responses()
        review = json.loads(review_response)
        contract = review["contract"]
        contract["subjects"][0]["source_keys"] = []
        contract["obligations"][0]["support_keys"] = []
        contract["interpretations"][0]["support_keys"] = []
        contract["uncertainties"][0]["source_keys"] = ["source-2"]

        with patch(
            "scripts.eccr_packet_experiment._pilot_completion",
            side_effect=[
                _completion(compile_response),
                _completion(json.dumps(review, ensure_ascii=False)),
            ],
        ):
            result = _run_eccr_audit(
                case=self.case,
                packet=self._audit_packet(),
                client=object(),
                provider_id="deepseek/test",
                model="deepseek-v4-flash",
                provider_extra_body={},
                max_output_tokens=32000,
                thinking_mode="enabled",
                deadline_seconds=30,
                ledger_path=self.root / "unused.jsonl",
                budget=PilotBudget(max_calls=2, soft_token_limit=0),
                run_id="case-a:eccr-audit:rep-001",
                repetition=1,
            )

        normalized = result["contract"]
        self.assertEqual(normalized["subjects"][0]["source_keys"], ["source-1"])
        self.assertEqual(normalized["obligations"][0]["support_keys"], ["source-1"])
        self.assertEqual(normalized["interpretations"][0]["support_keys"], ["source-1"])
        self.assertEqual(
            normalized["uncertainties"][0]["source_keys"],
            ["source-2", "source-1"],
        )
        restored = {
            (item["entity_type"], item["entity_id"], item["field"])
            for item in result["normalization_audit"]
        }
        self.assertEqual(
            restored,
            {
                ("subject", "甲", "source_keys"),
                ("obligation", "identity", "support_keys"),
                ("interpretation", "same-account", "support_keys"),
                ("uncertainty", "homonym-risk", "source_keys"),
            },
        )

    def test_eccr_audit_normalization_does_not_hide_unsafe_mutations(self) -> None:
        compile_response, review_response = self._eccr_audit_responses()
        packet = self._audit_packet()
        variants: list[tuple[str, dict[str, object]]] = []

        forged = json.loads(review_response)
        forged["contract"]["uncertainties"][0]["source_keys"] = ["source-forged"]
        variants.append(("source-forged|allow", forged))

        invalid_transition = json.loads(review_response)
        invalid_transition["contract"]["obligations"][0]["last_changed_step"] = 0
        variants.append(("transition step", invalid_transition))

        for expected, review in variants:
            with self.subTest(expected=expected):
                with patch(
                    "scripts.eccr_packet_experiment._pilot_completion",
                    side_effect=[
                        _completion(compile_response),
                        _completion(json.dumps(review, ensure_ascii=False)),
                    ],
                ):
                    with self.assertRaisesRegex(ValueError, expected):
                        _run_eccr_audit(
                            case=self.case,
                            packet=packet,
                            client=object(),
                            provider_id="deepseek/test",
                            model="deepseek-v4-flash",
                            provider_extra_body={},
                            max_output_tokens=32000,
                            thinking_mode="enabled",
                            deadline_seconds=30,
                            ledger_path=self.root / "unused.jsonl",
                            budget=PilotBudget(max_calls=2, soft_token_limit=0),
                            run_id="case-a:eccr-audit:rep-001",
                            repetition=1,
                        )

    def test_eccr_audit_parse_failure_keeps_both_visible_completions(self) -> None:
        compile_response, _ = self._eccr_audit_responses()
        compile_completion = _completion(compile_response)
        compile_completion.choices[0].message.reasoning_content = "SECRET_COMPILE_COT"
        invalid_review = "{not valid json"
        review_completion = _completion(invalid_review)
        review_completion.choices[0].message.reasoning_content = "SECRET_REVIEW_COT"
        checkpoint = self.root / "audit-checkpoint.private.json"

        with patch(
            "scripts.eccr_packet_experiment._pilot_completion",
            side_effect=[compile_completion, review_completion],
        ):
            with self.assertRaises(ValueError):
                _run_eccr_audit(
                    case=self.case,
                    packet=self._audit_packet(),
                    client=object(),
                    provider_id="deepseek/test",
                    model="deepseek-v4-flash",
                    provider_extra_body={},
                    max_output_tokens=32000,
                    thinking_mode="enabled",
                    deadline_seconds=30,
                    ledger_path=self.root / "unused.jsonl",
                    budget=PilotBudget(max_calls=2, soft_token_limit=0),
                    run_id="case-a:eccr-audit:rep-001",
                    repetition=1,
                    private_checkpoint_path=checkpoint,
                )

        private = json.loads(checkpoint.read_text(encoding="utf-8"))
        self.assertEqual(private["status"], "REVIEW_VALIDATION_FAILED")
        self.assertEqual(len(private["rounds"]), 2)
        self.assertEqual(
            private["rounds"][0]["provider_visible_completion_content"],
            compile_response,
        )
        self.assertEqual(
            private["rounds"][1]["provider_visible_completion_content"],
            invalid_review,
        )
        self.assertIn("parsed_turn", private["rounds"][0])
        serialized = json.dumps(private, ensure_ascii=False)
        self.assertNotIn("SECRET_COMPILE_COT", serialized)
        self.assertNotIn("SECRET_REVIEW_COT", serialized)
        self.assertNotIn("reasoning_content", serialized)

    def test_eccr_audit_second_provider_failure_keeps_compile_draft(self) -> None:
        compile_response, _ = self._eccr_audit_responses()
        checkpoint = self.root / "audit-checkpoint.private.json"
        with patch(
            "scripts.eccr_packet_experiment._pilot_completion",
            side_effect=[_completion(compile_response), TimeoutError("review timeout")],
        ):
            with self.assertRaises(TimeoutError):
                _run_eccr_audit(
                    case=self.case,
                    packet=self._audit_packet(),
                    client=object(),
                    provider_id="deepseek/test",
                    model="deepseek-v4-flash",
                    provider_extra_body={},
                    max_output_tokens=32000,
                    thinking_mode="enabled",
                    deadline_seconds=30,
                    ledger_path=self.root / "unused.jsonl",
                    budget=PilotBudget(max_calls=2, soft_token_limit=0),
                    run_id="case-a:eccr-audit:rep-001",
                    repetition=1,
                    private_checkpoint_path=checkpoint,
                )

        private = json.loads(checkpoint.read_text(encoding="utf-8"))
        self.assertEqual(private["status"], "REVIEW_CALL_FAILED")
        self.assertEqual(len(private["rounds"]), 1)
        self.assertIn("parsed_turn", private["rounds"][0])
        self.assertEqual(private["error"]["type"], "TimeoutError")

    def test_eccr_audit_two_attempts_are_durable_in_usage_ledger(self) -> None:
        self.packet = self._audit_packet()
        self._write_inputs()
        bundle = load_case_bundle(self.case_dir)
        compile_response, review_response = self._eccr_audit_responses()
        completions = [
            _completion(compile_response),
            _completion(review_response),
        ]
        for completion in completions:
            completion.usage = SimpleNamespace()
        ledger = self.root / "output" / "usage.jsonl"
        with patch(
            "scripts.masked_ab_experiment._chat_completion",
            side_effect=[
                (completions[0], TokenUsageRecord(input_other=10, output=2), 11.0),
                (completions[1], TokenUsageRecord(input_other=20, output=4), 17.0),
            ],
        ):
            result = run_case_arm(
                bundle=bundle,
                arm="eccr-audit",
                repetition=1,
                output_dir=self.root / "output",
                ledger_path=ledger,
                budget=PilotBudget(max_calls=2, soft_token_limit=0),
                client=object(),
                provider_id="deepseek/test",
                model="deepseek-v4-flash",
                provider_extra_body={},
                provider_fingerprint={"provider_id": "deepseek/test"},
                max_output_tokens=32000,
                thinking_mode="enabled",
                deadline_seconds=30,
            )

        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(result["usage"]["calls"], 2)
        self.assertEqual(result["usage"]["total_measured_lower_bound"], 36)
        self.assertEqual(len(result["result"]["rounds"]), 2)
        checkpoint = (
            self.root
            / "output"
            / "runs"
            / "case-a"
            / "eccr-audit"
            / "rep-001"
            / "audit-checkpoint.private.json"
        )
        private = json.loads(checkpoint.read_text(encoding="utf-8"))
        self.assertEqual(private["status"], "COMPLETED")
        self.assertEqual(len(private["rounds"]), 2)
        audit = _usage_ledger_audit(ledger)
        self.assertEqual(audit["attempted_calls"], 2)
        self.assertEqual(audit["completed_calls"], 2)
        self.assertEqual(audit["failed_calls"], 0)
        self.assertTrue(audit["usage_complete"])
        summary = build_summary([result], ledger)
        self.assertEqual(summary["arms"]["eccr-audit"]["provider_calls"], 2)
        self.assertEqual(summary["results"][0]["provider_calls"], 2)

    def test_eccr_audit_fails_when_second_call_is_not_terminal(self) -> None:
        compile_response, review_response = self._eccr_audit_responses()
        review = json.loads(review_response)
        review["terminal"] = False
        with patch(
            "scripts.eccr_packet_experiment._pilot_completion",
            side_effect=[
                _completion(compile_response),
                _completion(json.dumps(review, ensure_ascii=False)),
            ],
        ):
            with self.assertRaisesRegex(ValueError, "call 2 must terminate"):
                _run_eccr_audit(
                    case=self.case,
                    packet=self._audit_packet(),
                    client=object(),
                    provider_id="deepseek/test",
                    model="deepseek-v4-flash",
                    provider_extra_body={},
                    max_output_tokens=32000,
                    thinking_mode="enabled",
                    deadline_seconds=30,
                    ledger_path=self.root / "unused.jsonl",
                    budget=PilotBudget(max_calls=2, soft_token_limit=0),
                    run_id="case-a:eccr-audit:rep-001",
                    repetition=1,
                )

    def test_eccr_audit_fails_when_first_call_is_terminal(self) -> None:
        compile_response, review_response = self._eccr_audit_responses()
        compile_turn = json.loads(compile_response)
        compile_turn["terminal"] = True
        with patch(
            "scripts.eccr_packet_experiment._pilot_completion",
            side_effect=[
                _completion(json.dumps(compile_turn, ensure_ascii=False)),
                _completion(review_response),
            ],
        ) as mocked:
            with self.assertRaisesRegex(
                ValueError,
                "call 1 must be nonterminal|critical obligation open",
            ):
                _run_eccr_audit(
                    case=self.case,
                    packet=self._audit_packet(),
                    client=object(),
                    provider_id="deepseek/test",
                    model="deepseek-v4-flash",
                    provider_extra_body={},
                    max_output_tokens=32000,
                    thinking_mode="enabled",
                    deadline_seconds=30,
                    ledger_path=self.root / "unused.jsonl",
                    budget=PilotBudget(max_calls=2, soft_token_limit=0),
                    run_id="case-a:eccr-audit:rep-001",
                    repetition=1,
                )
        self.assertEqual(mocked.call_count, 1)

    def test_deterministic_identity_arm_is_zero_call(self) -> None:
        result = _run_deterministic(case=self.case, packet=self.packet)
        self.assertEqual(result["model_calls"], 0)
        self.assertEqual(result["subjects"][0]["participant_key"], "user-1")
        self.assertEqual(result["response_source"], "host_deterministic_binding")

    def test_deterministic_snapshot_score_covers_identity_invariants(self) -> None:
        case, packet, gold = self._snapshot_identity_inputs()
        result = _run_deterministic(case=case, packet=packet)
        snapshot = result["identity_snapshots"][0]
        self.assertEqual(snapshot["visible_aliases"], ["旧名", "现名"])
        self.assertEqual(snapshot["latest_display_name"], "现名")
        self.assertEqual(snapshot["selected_source_keys"], ["source-1", "source-2"])

        score = score_result(result, gold)
        identity = score["identity_snapshots"]
        self.assertTrue(identity["applicable"])
        self.assertEqual(identity["snapshot_count"], 1)
        self.assertEqual(identity["passed_snapshots"], 1)
        self.assertEqual(identity["snapshot_pass_rate"], 1.0)
        self.assertTrue(identity["all_passed"])
        checks = identity["snapshots"][0]
        self.assertTrue(checks["same_actor_exact_match"])
        self.assertTrue(checks["latest_name_exact_match"])
        self.assertTrue(checks["post_cutoff_aliases_excluded"])
        self.assertTrue(checks["cross_scope_participants_excluded"])
        self.assertTrue(checks["passed"])
        self.assertTrue(score["identity_contract"]["participant_exact_match"])
        self.assertIsNone(score["required_group_recall"])
        self.assertFalse(score["brief_evidence_recall_applicable"])

    def test_snapshot_identity_gates_fail_independently(self) -> None:
        case, packet, gold = self._snapshot_identity_inputs()
        baseline = _run_deterministic(case=case, packet=packet)
        variants: list[tuple[str, dict[str, object]]] = []

        wrong_actor = copy.deepcopy(baseline)
        wrong_actor["identity_snapshots"][0]["visible_alias_bindings"][1][
            "participant_keys"
        ] = ["user-2"]
        variants.append(("same_actor_exact_match", wrong_actor))

        wrong_latest = copy.deepcopy(baseline)
        wrong_latest["identity_snapshots"][0]["latest_display_name"] = "未来名"
        variants.append(("latest_name_exact_match", wrong_latest))

        leaked_alias = copy.deepcopy(baseline)
        leaked_alias["identity_snapshots"][0]["filtered_host_bindings"].append(
            {"reference": "未来名", "source_keys": ["source-3"]}
        )
        variants.append(("post_cutoff_aliases_excluded", leaked_alias))

        crossed_scope = copy.deepcopy(baseline)
        crossed_scope["identity_snapshots"][0]["cross_scope_selected_source_keys"] = [
            "source-x"
        ]
        variants.append(("cross_scope_participants_excluded", crossed_scope))

        fields = {
            "same_actor_exact_match",
            "latest_name_exact_match",
            "post_cutoff_aliases_excluded",
            "cross_scope_participants_excluded",
        }
        for failed_field, result in variants:
            with self.subTest(failed_field=failed_field):
                score = score_result(result, gold)["identity_snapshots"]
                checks = score["snapshots"][0]
                self.assertFalse(checks[failed_field])
                for passing_field in fields - {failed_field}:
                    self.assertTrue(checks[passing_field])
                self.assertFalse(checks["passed"])
                self.assertEqual(score["passed_snapshots"], 0)
                self.assertFalse(score["all_passed"])

    def test_deterministic_identity_binding_cannot_be_future_dated(self) -> None:
        self.case["host_subject_bindings"][0]["valid_at"] = 200
        with self.assertRaisesRegex(ValueError, "strictly before cutoff"):
            _run_deterministic(case=self.case, packet=self.packet)

    def test_failed_real_attempt_is_durable_and_private_result_is_written(self) -> None:
        bundle = load_case_bundle(self.case_dir)
        ledger = self.root / "output" / "usage.jsonl"
        with patch(
            "scripts.masked_ab_experiment._chat_completion",
            side_effect=TimeoutError("provider timeout"),
        ):
            result = run_case_arm(
                bundle=bundle,
                arm="one-pass",
                repetition=1,
                output_dir=self.root / "output",
                ledger_path=ledger,
                budget=PilotBudget(max_calls=1, soft_token_limit=0),
                client=object(),
                provider_id="deepseek/test",
                model="deepseek-v4-flash",
                provider_extra_body={},
                provider_fingerprint={"provider_id": "deepseek/test"},
                max_output_tokens=32000,
                thinking_mode="enabled",
                deadline_seconds=30,
            )
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["error_type"], "TimeoutError")
        audit = _usage_ledger_audit(ledger)
        self.assertEqual(audit["attempted_calls"], 1)
        self.assertEqual(audit["failed_calls"], 1)
        private_result = (
            self.root
            / "output"
            / "runs"
            / "case-a"
            / "one-pass"
            / "rep-001"
            / "result.private.json"
        )
        self.assertTrue(private_result.is_file())

    def test_summary_cannot_be_misread_as_end_to_end_retrieval(self) -> None:
        summary = build_summary([], self.root / "missing-ledger.jsonl")
        self.assertEqual(summary["diagnostic_layer"], DIAGNOSTIC_LAYER)
        self.assertFalse(summary["end_to_end_retrieval_evaluated"])
        self.assertFalse(summary["candidate_generation_evaluated"])
        self.assertIn("must not be reported", summary["interpretation"])

    def test_summary_exposes_identity_exact_and_snapshot_pass(self) -> None:
        rows = []
        for index, passed in enumerate((True, False), start=1):
            rows.append(
                {
                    "run_id": f"identity-{index}",
                    "case_id": "identity-case",
                    "arm": "deterministic",
                    "status": "COMPLETED",
                    "result": {"decision": "host_identity_binding"},
                    "usage": {"calls": 0, "total_measured_lower_bound": 0},
                    "evaluation": {
                        "required_group_recall": None,
                        "identity_contract": {
                            "participant_exact_match": passed,
                        },
                        "identity_snapshots": {
                            "applicable": True,
                            "snapshot_count": 1,
                            "passed_snapshots": int(passed),
                            "all_passed": passed,
                        },
                    },
                }
            )
        summary = build_summary(rows, self.root / "missing-ledger.jsonl")
        arm = summary["arms"]["deterministic"]
        self.assertIsNone(arm["mean_required_group_recall"])
        self.assertEqual(arm["identity_exact_evaluated_runs"], 2)
        self.assertEqual(arm["identity_exact_passed_runs"], 1)
        self.assertEqual(arm["identity_exact_pass_rate"], 0.5)
        self.assertEqual(arm["identity_snapshots_evaluated"], 2)
        self.assertEqual(arm["identity_snapshots_passed"], 1)
        self.assertEqual(arm["identity_snapshot_pass_rate"], 0.5)
        self.assertEqual(arm["identity_snapshot_all_passed_runs"], 1)
        self.assertTrue(summary["results"][0]["identity_exact_match"])
        self.assertTrue(summary["results"][0]["identity_snapshot_pass"])
        self.assertFalse(summary["results"][1]["identity_exact_match"])
        self.assertFalse(summary["results"][1]["identity_snapshot_pass"])


if __name__ == "__main__":
    unittest.main()
