from __future__ import annotations

import copy
import hashlib
import json
import unittest
from collections.abc import Callable

from mr_memory.evidence_closure import RetrievalAction
from mr_memory.orchestrator import (
    ECCR_TOOL_ACTION_CATALOG,
    EccrLimits,
    EccrOrchestrator,
    validate_tool_action_arguments,
)


class EccrOrchestratorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.scope = hashlib.sha256(b"scope").hexdigest()
        self.query = hashlib.sha256(b"query").hexdigest()
        self.host = {
            "scope_sha256": self.scope,
            "query_sha256": self.query,
            "cutoff_at": 100,
            "revision_vector": {
                "message": "1",
                "graph": "1",
                "identity": "1",
                "relation": "1",
                "feedback": "1",
                "protocol": "eccr-runtime-v2",
            },
        }
        self.packet = {
            "participants": [
                {"canonical_key": "p2", "display_name": "另一位群友"}
            ],
            "graph_metadata": {"canonical_key": "not-a-participant"},
            "messages": [
                {
                    "source_key": "s1",
                    "participant_key": "p1",
                    "sent_at": 80,
                    "plain_text": "旧证据",
                },
                {
                    "source_key": "s2",
                    "participant_key": "p1",
                    "sent_at": 90,
                    "plain_text": "未选反例",
                },
            ]
        }

    def contract(self, step: int) -> dict[str, object]:
        return {
            "contract_id": "c1",
            **self.host,
            "step_index": step,
            "subjects": [{
                "reference": "甲", "participant_key": "p1", "mode": "HOST",
                "candidate_participant_keys": [], "source_keys": ["s1"], "valid_at": 80,
            }],
            "obligations": [{
                "id": "o1", "kind": "semantic", "question": "是否成立", "critical": True,
                "status": "OPEN", "support_keys": [], "counter_keys": [],
                "last_changed_step": 0,
            }],
            "interpretations": [{
                "id": "i1", "statement": "解释一", "status": "CANDIDATE",
                "support_keys": [], "counter_keys": [], "uncertainty": "待证",
                "origin": "COMPILE", "discriminates_interpretation_ids": [],
            }],
            "uncertainties": [{
                "id": "u1", "statement": "不可升级", "status": "OPEN", "source_keys": [],
                "origin": "COMPILE", "discriminates_interpretation_ids": [],
            }],
            "guarded_claims": ["不可升级"], "visited_source_keys": ["s1"],
            "selected_edge_ids": [], "selected_hypothesis_ids": [],
            "tried_action_signatures": [], "exhausted_discriminators": [],
            "frontier_discriminators": ["检查反例"],
        }

    async def run_contract_pair(
        self,
        first_contract: dict[str, object],
        second_contract: dict[str, object],
    ):
        responses = [
            {
                "contract": first_contract,
                "actions": [],
                "memory_brief": None,
                "terminal": False,
            },
            {
                "contract": second_contract,
                "actions": [],
                "memory_brief": None,
                "terminal": False,
            },
        ]

        async def complete(
            _system: str,
            _prompt: str,
            index: int,
            _phase: str,
        ):
            return responses[index]

        async def execute(_action: RetrievalAction):
            self.fail("no action expected")

        return await EccrOrchestrator(
            limits=EccrLimits(
                max_model_calls=2,
                max_retrieval_rounds=0,
                audit_discovery=True,
            )
        ).run(
            query="问题",
            host_contract_fields=self.host,
            evidence_packet=self.packet,
            complete=complete,
            execute_action=execute,
            allowed_tool_names=set(),
        )

    async def test_three_phase_audit_discovery_is_bounded(self) -> None:
        responses: list[dict[str, object]] = []
        prompts: list[dict[str, object]] = []
        first = self.contract(0)
        responses.append({"contract": first, "actions": [], "memory_brief": None, "terminal": False})
        second = copy.deepcopy(first)
        second["step_index"] = 1
        responses.append({"contract": second, "actions": [], "memory_brief": None, "terminal": False})
        third = copy.deepcopy(second)
        third["step_index"] = 2
        third["visited_source_keys"] = ["s1", "s2"]
        third["obligations"][0].update({"status": "CONTESTED", "support_keys": ["s1"], "counter_keys": ["s2"], "last_changed_step": 2})
        third["interpretations"].append({
            "id": "i2", "statement": "反例解释", "status": "CONTESTED",
            "support_keys": ["s2"], "counter_keys": ["s1"],
            "uncertainty": "仅候选", "origin": "AUDIT_DISCOVERY",
            "discriminates_interpretation_ids": ["i1"],
        })
        third["uncertainties"][0].update({"status": "PRESERVED", "source_keys": ["s1", "s2"]})
        responses.append({
            "contract": third, "actions": [],
            "memory_brief": {"claims": [], "conflicts": [{"statement": "两种解释竞争", "source_keys": ["s1", "s2"]}], "unresolved": [{"statement": "仍不可升级", "source_keys": ["s1", "s2"]}]},
            "terminal": True,
        })

        async def complete(_system: str, prompt: str, index: int, _phase: str):
            prompts.append(json.loads(prompt))
            return responses[index]

        async def execute(_action):
            self.fail("no action expected")

        result = await EccrOrchestrator(
            limits=EccrLimits(max_model_calls=3, max_retrieval_rounds=0)
        ).run(
            query="问题", host_contract_fields=self.host,
            evidence_packet=self.packet, complete=complete,
            execute_action=execute, allowed_tool_names=set(),
        )
        self.assertEqual(result.model_calls, 3)
        self.assertEqual(result.status, "CERTIFIED")
        self.assertEqual(result.trace[-1].phase, "AUDIT_DISCOVERY")
        self.assertEqual(prompts[1]["current_visible_source_keys"], ["s1"])
        self.assertEqual(
            prompts[2]["current_visible_source_keys"],
            ["s1", "s2"],
        )

    async def test_dsv4_fenced_json_uses_expanded_action_contract(self) -> None:
        action_raw = {
            "obligation_id": "o1",
            "tool_name": "mr_query_event_context",
            "arguments": {"event_id": 7, "limit": 20},
            "discriminator": "展开事件七的原始证据",
            "expected_delta": "找到支持或反驳解释一的消息",
        }
        signature = RetrievalAction.from_value(
            action_raw,
            field="action",
        ).signature()
        first = {
            "contract": self.contract(0),
            "actions": [action_raw],
            "memory_brief": None,
            "terminal": False,
        }
        second_contract = copy.deepcopy(first["contract"])
        second_contract["step_index"] = 1
        second_contract["visited_source_keys"] = ["s1", "s3"]
        second_contract["tried_action_signatures"] = [signature]
        second_contract["obligations"][0].update(
            {
                "status": "SUPPORTED",
                "support_keys": ["s3"],
                "last_changed_step": 1,
            }
        )
        second_contract["interpretations"][0].update(
            {"status": "SUPPORTED", "support_keys": ["s3"], "uncertainty": ""}
        )
        second_contract["uncertainties"][0].update(
            {"status": "RESOLVED", "source_keys": ["s3"]}
        )
        second = {
            "contract": second_contract,
            "actions": [],
            "memory_brief": {
                "claims": [
                    {
                        "statement": "事件七的原始消息支持解释一。",
                        "source_keys": ["s3"],
                        "confidence": 0.78,
                    }
                ],
                "conflicts": [],
                "unresolved": [],
            },
            "terminal": True,
        }
        prompts: list[dict[str, object]] = []
        calls: list[tuple[int, str]] = []
        executed: list[RetrievalAction] = []

        async def complete(_system: str, prompt: str, index: int, phase: str):
            prompts.append(json.loads(prompt))
            calls.append((index, phase))
            response = first if index == 0 else second
            return "```json\n" + json.dumps(response, ensure_ascii=False) + "\n```"

        async def execute(action: RetrievalAction):
            executed.append(action)
            return {
                "messages": [
                    {
                        "source_key": "s3",
                        "participant_key": "p1",
                        "plain_text": "新取回的支持证据",
                    }
                ]
            }

        result = await EccrOrchestrator(
            limits=EccrLimits(
                max_model_calls=2,
                max_retrieval_rounds=1,
                audit_discovery=False,
            )
        ).run(
            query="问题",
            host_contract_fields=self.host,
            evidence_packet=self.packet,
            complete=complete,
            execute_action=execute,
            allowed_tool_names={"mr_query_event_context"},
        )

        self.assertEqual(calls, [(0, "COMPILE"), (1, "DISCRIMINATE")])
        self.assertEqual(result.status, "CERTIFIED")
        self.assertEqual(result.model_calls, 2)
        self.assertEqual(len(executed), 1)
        self.assertEqual(executed[0].arguments, {"event_id": 7, "limit": 20})
        first_prompt = prompts[0]
        self.assertEqual(first_prompt["protocol"], "eccr-runtime-v2")
        self.assertEqual(
            first_prompt["host_contract_fields"]["scope_sha256"],
            self.scope,
        )
        self.assertIn("p2", first_prompt["authorized_participant_keys"])
        self.assertNotIn(
            "not-a-participant",
            first_prompt["authorized_participant_keys"],
        )
        schema = first_prompt["output_schema"]
        self.assertEqual(
            schema["properties"]["contract"]["properties"]["scope_sha256"]["const"],
            self.scope,
        )
        argument_schema = first_prompt["action_catalog"][
            "mr_query_event_context"
        ]["arguments"]
        self.assertEqual(argument_schema["required"], ["event_id"])
        self.assertFalse(argument_schema["additionalProperties"])
        self.assertEqual(prompts[1]["action_catalog"], {})
        self.assertEqual(
            first_prompt["current_visible_source_keys"],
            ["s1", "s2"],
        )
        self.assertEqual(
            prompts[1]["current_visible_source_keys"],
            ["s1", "s3"],
        )
        self.assertNotIn("s2", prompts[1]["current_visible_source_keys"])

    async def test_discriminate_cannot_attach_unserialized_initial_source(
        self,
    ) -> None:
        first_contract = self.contract(0)
        second_contract = copy.deepcopy(first_contract)
        second_contract["step_index"] = 1
        second_contract["visited_source_keys"] = ["s1", "s2"]
        second_contract["obligations"][0].update(
            {
                "status": "REFUTED",
                "counter_keys": ["s2"],
                "last_changed_step": 1,
            }
        )
        second_contract["interpretations"][0].update(
            {"status": "REFUTED", "counter_keys": ["s2"]}
        )
        second_contract["uncertainties"][0].update(
            {"status": "RESOLVED", "source_keys": ["s2"]}
        )
        responses = [
            {
                "contract": first_contract,
                "actions": [],
                "memory_brief": None,
                "terminal": False,
            },
            {
                "contract": second_contract,
                "actions": [],
                "memory_brief": {
                    "claims": [
                        {
                            "statement": "未交付的反例被错误使用。",
                            "source_keys": ["s2"],
                            "confidence": 0.8,
                        }
                    ],
                    "conflicts": [],
                    "unresolved": [],
                },
                "terminal": True,
            },
        ]
        prompts: list[dict[str, object]] = []

        async def complete(_system: str, prompt: str, index: int, _phase: str):
            prompts.append(json.loads(prompt))
            return responses[0] if index == 0 else responses[1]

        async def execute(_action: RetrievalAction):
            self.fail("no action expected")

        with self.assertRaisesRegex(ValueError, "not visible in this call"):
            await EccrOrchestrator(
                limits=EccrLimits(
                    max_model_calls=3,
                    max_retrieval_rounds=0,
                    audit_discovery=True,
                )
            ).run(
                query="问题",
                host_contract_fields=self.host,
                evidence_packet=self.packet,
                complete=complete,
                execute_action=execute,
                allowed_tool_names=set(),
            )

        self.assertEqual(prompts[1]["phase"], "DISCRIMINATE")
        self.assertEqual(prompts[1]["authorized_source_keys"], ["s1", "s2"])
        self.assertEqual(prompts[1]["current_visible_source_keys"], ["s1"])

    async def test_forged_visited_source_does_not_expand_current_visibility(
        self,
    ) -> None:
        first_contract = self.contract(0)
        forged_contract = copy.deepcopy(first_contract)
        forged_contract["step_index"] = 1
        forged_contract["visited_source_keys"] = ["s1", "s2"]
        responses = [
            {
                "contract": first_contract,
                "actions": [],
                "memory_brief": None,
                "terminal": False,
            },
            {
                "contract": forged_contract,
                "actions": [],
                "memory_brief": None,
                "terminal": False,
            },
        ]

        async def complete(_system: str, _prompt: str, index: int, _phase: str):
            return responses[0] if index == 0 else responses[1]

        async def execute(_action: RetrievalAction):
            self.fail("no action expected")

        with self.assertRaisesRegex(
            ValueError,
            "contract.visited_source_keys attached evidence not visible",
        ):
            await EccrOrchestrator(
                limits=EccrLimits(
                    max_model_calls=3,
                    max_retrieval_rounds=0,
                    audit_discovery=True,
                )
            ).run(
                query="问题",
                host_contract_fields=self.host,
                evidence_packet=self.packet,
                complete=complete,
                execute_action=execute,
                allowed_tool_names=set(),
            )

    async def test_audit_discovery_cannot_ground_new_hypothesis_only_on_old_selected(
        self,
    ) -> None:
        first_contract = self.contract(0)
        audit_contract = copy.deepcopy(first_contract)
        audit_contract["step_index"] = 1
        audit_contract["visited_source_keys"] = ["s1", "s2"]
        audit_contract["interpretations"].append(
            {
                "id": "i-old-only",
                "statement": "只复用了旧已选证据的新解释。",
                "status": "CANDIDATE",
                "support_keys": ["s1"],
                "counter_keys": [],
                "uncertainty": "未被本轮新反例支撑。",
                "origin": "AUDIT_DISCOVERY",
                "discriminates_interpretation_ids": ["i1"],
            }
        )
        responses = [
            {
                "contract": first_contract,
                "actions": [],
                "memory_brief": None,
                "terminal": False,
            },
            {
                "contract": audit_contract,
                "actions": [],
                "memory_brief": None,
                "terminal": False,
            },
        ]

        async def complete(_system: str, _prompt: str, index: int, _phase: str):
            return responses[index]

        async def execute(_action: RetrievalAction):
            self.fail("no action expected")

        with self.assertRaisesRegex(
            ValueError,
            "newly visited evidence visible in its discovery call",
        ):
            await EccrOrchestrator(
                limits=EccrLimits(
                    max_model_calls=2,
                    max_retrieval_rounds=0,
                    audit_discovery=True,
                )
            ).run(
                query="问题",
                host_contract_fields=self.host,
                evidence_packet=self.packet,
                complete=complete,
                execute_action=execute,
                allowed_tool_names=set(),
            )

    def test_action_arguments_are_host_validated_before_execution(self) -> None:
        def action(tool_name: str, arguments: dict[str, object]) -> RetrievalAction:
            return RetrievalAction.from_value(
                {
                    "obligation_id": "o1",
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "discriminator": "区分竞争解释",
                    "expected_delta": "获得新的可引用证据",
                },
                field="action",
            )

        with self.assertRaisesRegex(ValueError, "must be an integer"):
            validate_tool_action_arguments(
                action("mr_query_event_context", {"event_id": "7"}),
                action_catalog=ECCR_TOOL_ACTION_CATALOG,
            )
        with self.assertRaisesRegex(ValueError, "unknown fields: surprise"):
            validate_tool_action_arguments(
                action("mr_query_event_context", {"event_id": 7, "surprise": True}),
                action_catalog=ECCR_TOOL_ACTION_CATALOG,
            )
        with self.assertRaisesRegex(ValueError, "at least one of"):
            validate_tool_action_arguments(
                action("mr_query_associations", {}),
                action_catalog=ECCR_TOOL_ACTION_CATALOG,
            )

    async def test_protocol_repair_consumes_existing_model_call_budget(self) -> None:
        valid = {
            "contract": self.contract(0),
            "actions": [],
            "memory_brief": None,
            "terminal": False,
        }
        invalid = copy.deepcopy(valid)
        invalid["actions"] = [
            {
                "obligation_id": "o1",
                "tool_name": "mr_query_event_context",
                "arguments": {"event_id": "7"},
                "discriminator": "展开事件七",
                "expected_delta": "获得原始消息",
            }
        ]
        calls: list[tuple[int, str, dict[str, object]]] = []

        async def complete(_system: str, prompt: str, index: int, phase: str):
            calls.append((index, phase, json.loads(prompt)))
            if index == 0:
                return "```json\n" + json.dumps(invalid, ensure_ascii=False) + "\n```"
            return "```json\n" + json.dumps(valid, ensure_ascii=False) + "\n```"

        async def execute(_action: RetrievalAction):
            self.fail("a final-budget repair cannot schedule retrieval")

        result = await EccrOrchestrator(
            limits=EccrLimits(
                max_model_calls=2,
                max_retrieval_rounds=1,
                audit_discovery=False,
            )
        ).run(
            query="问题",
            host_contract_fields=self.host,
            evidence_packet=self.packet,
            complete=complete,
            execute_action=execute,
            allowed_tool_names={"mr_query_event_context"},
        )

        self.assertEqual([item[:2] for item in calls], [(0, "COMPILE"), (1, "COMPILE")])
        self.assertEqual(result.model_calls, 2)
        self.assertEqual(len(result.trace), 1)
        self.assertEqual(result.trace[0].call_index, 1)
        self.assertEqual(result.status, "PARTIAL")
        self.assertEqual(result.stop_reason, "BUDGET_EXHAUSTED")
        self.assertTrue(result.repair_attempted)
        self.assertFalse(result.degraded)
        self.assertEqual(len(result.protocol_failures), 1)
        self.assertEqual(result.protocol_failures[0].attempt, 0)
        repair_payload = calls[1][2]
        self.assertEqual(repair_payload["protocol_repair"]["attempt"], 1)
        self.assertEqual(repair_payload["action_catalog"], {})
        self.assertEqual(
            repair_payload["output_schema"]["properties"]["actions"]["maxItems"],
            0,
        )

    async def test_protocol_repair_is_attempted_only_once(self) -> None:
        calls: list[int] = []

        async def complete(_system: str, _prompt: str, index: int, _phase: str):
            calls.append(index)
            return "still not json"

        async def execute(_action: RetrievalAction):
            self.fail("no action expected")

        with self.assertRaisesRegex(ValueError, "exactly one JSON object"):
            await EccrOrchestrator(
                limits=EccrLimits(
                    max_model_calls=3,
                    max_retrieval_rounds=1,
                    audit_discovery=False,
                )
            ).run(
                query="问题",
                host_contract_fields=self.host,
                evidence_packet=self.packet,
                complete=complete,
                execute_action=execute,
                allowed_tool_names={"mr_query_event_context"},
            )
        self.assertEqual(calls, [0, 1])

    async def test_later_protocol_failure_repairs_once_then_returns_last_valid_turn(
        self,
    ) -> None:
        first_contract = self.contract(0)
        first = {
            "contract": first_contract,
            "actions": [],
            "memory_brief": {
                "claims": [],
                "conflicts": [],
                "unresolved": [
                    {"statement": "证据尚未闭合。", "source_keys": ["s1"]}
                ],
            },
            "terminal": False,
        }
        invalid = copy.deepcopy(first)
        invalid["contract"]["step_index"] = 1
        invalid["contract"]["interpretations"][0]["statement"] = "被模型改写的解释"
        calls: list[int] = []

        async def complete(_system: str, _prompt: str, index: int, _phase: str):
            calls.append(index)
            return first if index == 0 else invalid

        async def execute(_action: RetrievalAction):
            self.fail("no action expected")

        result = await EccrOrchestrator(
            limits=EccrLimits(
                max_model_calls=3,
                max_retrieval_rounds=0,
                audit_discovery=True,
            )
        ).run(
            query="问题",
            host_contract_fields=self.host,
            evidence_packet=self.packet,
            complete=complete,
            execute_action=execute,
            allowed_tool_names=set(),
        )

        self.assertEqual(calls, [0, 1, 2])
        self.assertEqual(result.status, "PARTIAL")
        self.assertEqual(result.stop_reason, "PROTOCOL_DEGRADED")
        self.assertTrue(result.degraded)
        self.assertTrue(result.repair_attempted)
        self.assertEqual(result.model_calls, 3)
        self.assertEqual(len(result.trace), 1)
        self.assertEqual(result.final_turn.contract.as_dict(), first_contract)
        self.assertEqual(len(result.protocol_failures), 2)
        self.assertEqual(
            [item.attempt for item in result.protocol_failures],
            [0, 1],
        )
        self.assertEqual(
            [item.call_index for item in result.protocol_failures],
            [1, 2],
        )
        for failure in result.protocol_failures:
            self.assertEqual(failure.error_type, "ValueError")
            self.assertRegex(failure.response_sha256, r"^[0-9a-f]{64}$")
            self.assertGreater(failure.response_chars, 0)
            self.assertLessEqual(len(failure.message), 1000)

    async def test_later_protocol_failure_without_repair_budget_degrades(
        self,
    ) -> None:
        first_contract = self.contract(0)
        first = {
            "contract": first_contract,
            "actions": [],
            "memory_brief": {
                "claims": [],
                "conflicts": [],
                "unresolved": [
                    {"statement": "证据尚未闭合。", "source_keys": ["s1"]}
                ],
            },
            "terminal": False,
        }
        invalid = copy.deepcopy(first)
        invalid["contract"]["step_index"] = 1
        invalid["contract"]["scope_sha256"] = hashlib.sha256(b"other").hexdigest()

        async def complete(_system: str, _prompt: str, index: int, _phase: str):
            return first if index == 0 else invalid

        async def execute(_action: RetrievalAction):
            self.fail("no action expected")

        result = await EccrOrchestrator(
            limits=EccrLimits(
                max_model_calls=2,
                max_retrieval_rounds=0,
                audit_discovery=True,
            )
        ).run(
            query="问题",
            host_contract_fields=self.host,
            evidence_packet=self.packet,
            complete=complete,
            execute_action=execute,
            allowed_tool_names=set(),
        )
        self.assertEqual(result.stop_reason, "PROTOCOL_DEGRADED")
        self.assertFalse(result.repair_attempted)
        self.assertTrue(result.degraded)
        self.assertEqual(result.model_calls, 2)
        self.assertEqual(len(result.protocol_failures), 1)
        self.assertEqual(len(result.trace), 1)

    async def test_later_invalid_turn_without_previous_brief_still_fails(self) -> None:
        first = {
            "contract": self.contract(0),
            "actions": [],
            "memory_brief": None,
            "terminal": False,
        }
        invalid = copy.deepcopy(first)
        invalid["contract"]["step_index"] = 1
        invalid["contract"]["contract_id"] = "changed-contract"

        async def complete(_system: str, _prompt: str, index: int, _phase: str):
            return first if index == 0 else invalid

        async def execute(_action: RetrievalAction):
            self.fail("no action expected")

        with self.assertRaisesRegex(ValueError, "contract_id"):
            await EccrOrchestrator(
                limits=EccrLimits(
                    max_model_calls=3,
                    max_retrieval_rounds=0,
                    audit_discovery=True,
                )
            ).run(
                query="问题",
                host_contract_fields=self.host,
                evidence_packet=self.packet,
                complete=complete,
                execute_action=execute,
                allowed_tool_names=set(),
            )

    async def test_provider_failure_after_valid_turn_never_uses_protocol_fallback(
        self,
    ) -> None:
        first = {
            "contract": self.contract(0),
            "actions": [],
            "memory_brief": {
                "claims": [],
                "conflicts": [],
                "unresolved": [
                    {"statement": "证据尚未闭合。", "source_keys": ["s1"]}
                ],
            },
            "terminal": False,
        }

        async def complete(_system: str, _prompt: str, index: int, _phase: str):
            if index == 0:
                return first
            raise RuntimeError("provider transport failed")

        async def execute(_action: RetrievalAction):
            self.fail("no action expected")

        with self.assertRaisesRegex(RuntimeError, "provider transport failed"):
            await EccrOrchestrator(
                limits=EccrLimits(
                    max_model_calls=2,
                    max_retrieval_rounds=0,
                    audit_discovery=True,
                )
            ).run(
                query="问题",
                host_contract_fields=self.host,
                evidence_packet=self.packet,
                complete=complete,
                execute_action=execute,
                allowed_tool_names=set(),
            )

    async def test_resolved_identity_mutation_only_returns_previous_turn(self) -> None:
        first_contract = self.contract(0)
        first = {
            "contract": first_contract,
            "actions": [],
            "memory_brief": {
                "claims": [],
                "conflicts": [],
                "unresolved": [
                    {"statement": "证据尚未闭合。", "source_keys": ["s1"]}
                ],
            },
            "terminal": False,
        }
        rebound = copy.deepcopy(first)
        rebound["contract"]["step_index"] = 1
        rebound["contract"]["subjects"][0]["participant_key"] = "p2"

        async def complete(_system: str, _prompt: str, index: int, _phase: str):
            return first if index == 0 else rebound

        async def execute(_action: RetrievalAction):
            self.fail("no action expected")

        result = await EccrOrchestrator(
            limits=EccrLimits(
                max_model_calls=2,
                max_retrieval_rounds=0,
                audit_discovery=True,
            )
        ).run(
            query="问题",
            host_contract_fields=self.host,
            evidence_packet=self.packet,
            complete=complete,
            execute_action=execute,
            allowed_tool_names=set(),
        )
        self.assertEqual(result.stop_reason, "PROTOCOL_DEGRADED")
        self.assertEqual(
            result.final_turn.contract.subjects[0].participant_key,
            "p1",
        )
        self.assertEqual(len(result.trace), 1)

    async def test_invalid_audit_after_terminal_turn_degrades_exact_terminal_turn(
        self,
    ) -> None:
        first_contract = self.contract(0)
        first_contract["visited_source_keys"] = ["s1", "s2"]
        first = {
            "contract": first_contract,
            "actions": [],
            "memory_brief": None,
            "terminal": False,
        }
        terminal_contract = copy.deepcopy(first_contract)
        terminal_contract["step_index"] = 1
        terminal_contract["visited_source_keys"] = ["s1", "s2"]
        terminal_contract["obligations"][0].update(
            {
                "status": "CONTESTED",
                "support_keys": ["s1"],
                "counter_keys": ["s2"],
                "last_changed_step": 1,
            }
        )
        terminal_contract["interpretations"][0].update(
            {
                "status": "CONTESTED",
                "support_keys": ["s1"],
                "counter_keys": ["s2"],
            }
        )
        terminal_contract["uncertainties"][0].update(
            {"status": "PRESERVED", "source_keys": ["s1", "s2"]}
        )
        terminal = {
            "contract": terminal_contract,
            "actions": [],
            "memory_brief": {
                "claims": [],
                "conflicts": [
                    {"statement": "两种解释竞争。", "source_keys": ["s1", "s2"]}
                ],
                "unresolved": [
                    {"statement": "仍不可升级。", "source_keys": ["s1", "s2"]}
                ],
            },
            "terminal": True,
        }
        invalid_audit = copy.deepcopy(terminal)
        invalid_audit["contract"]["step_index"] = 2
        invalid_audit["contract"]["uncertainties"][0]["statement"] = "改写后的疑虑"

        async def complete(_system: str, _prompt: str, index: int, _phase: str):
            return [first, terminal, invalid_audit][index]

        async def execute(_action: RetrievalAction):
            self.fail("no action expected")

        result = await EccrOrchestrator(
            limits=EccrLimits(
                max_model_calls=3,
                max_retrieval_rounds=0,
                audit_discovery=True,
            )
        ).run(
            query="问题",
            host_contract_fields=self.host,
            evidence_packet=self.packet,
            complete=complete,
            execute_action=execute,
            allowed_tool_names=set(),
        )
        self.assertEqual(result.status, "PARTIAL")
        self.assertEqual(result.stop_reason, "PROTOCOL_DEGRADED")
        self.assertTrue(result.final_turn.terminal)
        self.assertEqual(result.final_turn.contract.as_dict(), terminal_contract)
        self.assertEqual(len(result.trace), 2)

    async def test_later_guarded_claim_mutations_are_host_canonicalized(self) -> None:
        variants = {
            "addition": {
                "value": ["不可升级", "模型新增的安全约束"],
                "mutation_kinds": ["ADD"],
                "added": ["模型新增的安全约束"],
                "omitted": [],
            },
            "deletion": {
                "value": [],
                "mutation_kinds": ["REMOVE"],
                "added": [],
                "omitted": ["不可升级"],
            },
            "rewrite": {
                "value": ["模型改写后的安全约束"],
                "mutation_kinds": ["ADD", "REMOVE", "REWRITE"],
                "added": ["模型改写后的安全约束"],
                "omitted": ["不可升级"],
            },
        }

        for name, variant in variants.items():
            with self.subTest(name=name):
                first_contract = self.contract(0)
                second_contract = copy.deepcopy(first_contract)
                second_contract["step_index"] = 1
                second_contract["guarded_claims"] = variant["value"]
                responses = [
                    {
                        "contract": first_contract,
                        "actions": [],
                        "memory_brief": None,
                        "terminal": False,
                    },
                    {
                        "contract": second_contract,
                        "actions": [],
                        "memory_brief": None,
                        "terminal": False,
                    },
                ]

                async def complete(
                    _system: str,
                    _prompt: str,
                    index: int,
                    _phase: str,
                ):
                    return responses[index]

                async def execute(_action: RetrievalAction):
                    self.fail("no action expected")

                result = await EccrOrchestrator(
                    limits=EccrLimits(
                        max_model_calls=2,
                        max_retrieval_rounds=0,
                        audit_discovery=True,
                    )
                ).run(
                    query="问题",
                    host_contract_fields=self.host,
                    evidence_packet=self.packet,
                    complete=complete,
                    execute_action=execute,
                    allowed_tool_names=set(),
                )

                self.assertEqual(
                    result.final_turn.contract.guarded_claims,
                    ("不可升级",),
                )
                self.assertEqual(result.trace[0].normalization_audit, ())
                self.assertEqual(len(result.trace[1].normalization_audit), 1)
                audit = result.trace[1].normalization_audit[0]
                self.assertEqual(
                    audit["normalization"],
                    "preserve_previous_guarded_claims",
                )
                self.assertEqual(audit["model_value"], variant["value"])
                self.assertEqual(audit["canonical_value"], ["不可升级"])
                self.assertEqual(
                    audit["mutation_kinds"], variant["mutation_kinds"]
                )
                self.assertEqual(audit["added_by_model"], variant["added"])
                self.assertEqual(audit["omitted_by_model"], variant["omitted"])

    async def test_guarded_claim_canonicalization_does_not_hide_other_mutation(self) -> None:
        first_contract = self.contract(0)
        second_contract = copy.deepcopy(first_contract)
        second_contract["step_index"] = 1
        second_contract["guarded_claims"] = ["模型新增的安全约束"]
        second_contract["contract_id"] = "changed-contract"
        responses = [
            {
                "contract": first_contract,
                "actions": [],
                "memory_brief": None,
                "terminal": False,
            },
            {
                "contract": second_contract,
                "actions": [],
                "memory_brief": None,
                "terminal": False,
            },
        ]

        async def complete(
            _system: str,
            _prompt: str,
            index: int,
            _phase: str,
        ):
            return responses[index]

        async def execute(_action: RetrievalAction):
            self.fail("no action expected")

        with self.assertRaisesRegex(
            ValueError,
            "changed immutable contract_id",
        ):
            await EccrOrchestrator(
                limits=EccrLimits(
                    max_model_calls=2,
                    max_retrieval_rounds=0,
                    audit_discovery=True,
                )
            ).run(
                query="问题",
                host_contract_fields=self.host,
                evidence_packet=self.packet,
                complete=complete,
                execute_action=execute,
                allowed_tool_names=set(),
            )

    async def test_previous_evidence_sets_are_restored_without_losing_additions(
        self,
    ) -> None:
        first = self.contract(0)
        first["subjects"][0]["source_keys"] = ["s1"]
        first["obligations"][0]["support_keys"] = ["s1"]
        first["obligations"][0]["counter_keys"] = ["s1"]
        first["interpretations"][0]["support_keys"] = ["s1"]
        first["interpretations"][0]["counter_keys"] = ["s1"]
        first["uncertainties"][0]["source_keys"] = ["s1"]

        second = copy.deepcopy(first)
        second["step_index"] = 1
        second["visited_source_keys"] = ["s2"]
        second["subjects"][0]["source_keys"] = ["s2"]
        second["obligations"][0]["support_keys"] = ["s2"]
        second["obligations"][0]["counter_keys"] = ["s2"]
        second["interpretations"][0]["support_keys"] = ["s2"]
        second["interpretations"][0]["counter_keys"] = ["s2"]
        second["uncertainties"][0]["source_keys"] = ["s2"]
        second["interpretations"].append(
            {
                "id": "i2",
                "statement": "审计发现的新竞争解释",
                "status": "CANDIDATE",
                "support_keys": ["s2"],
                "counter_keys": [],
                "uncertainty": "只能维持为候选",
                "origin": "AUDIT_DISCOVERY",
                "discriminates_interpretation_ids": ["i1"],
            }
        )
        second["uncertainties"].append(
            {
                "id": "u2",
                "statement": "新证据仍不能排除另一解释",
                "status": "OPEN",
                "source_keys": ["s2"],
                "origin": "AUDIT_DISCOVERY",
                "discriminates_interpretation_ids": ["i1"],
            }
        )

        result = await self.run_contract_pair(first, second)
        contract = result.final_turn.contract
        self.assertEqual(contract.visited_source_keys, ("s2", "s1"))
        self.assertEqual(contract.tried_action_signatures, ())
        self.assertEqual(contract.subjects[0].source_keys, ("s2", "s1"))
        self.assertEqual(contract.obligations[0].support_keys, ("s2", "s1"))
        self.assertEqual(contract.obligations[0].counter_keys, ("s2", "s1"))
        self.assertEqual(contract.interpretations[0].support_keys, ("s2", "s1"))
        self.assertEqual(contract.interpretations[0].counter_keys, ("s2", "s1"))
        self.assertEqual(contract.uncertainties[0].source_keys, ("s2", "s1"))
        self.assertEqual(contract.interpretations[1].interpretation_id, "i2")
        self.assertEqual(contract.interpretations[1].support_keys, ("s2",))
        self.assertEqual(contract.uncertainties[1].constraint_id, "u2")
        self.assertEqual(contract.uncertainties[1].source_keys, ("s2",))

        audit = result.trace[1].normalization_audit
        restored_top_fields = {
            item["field"]
            for item in audit
            if item["normalization"] == "monotonic_previous_set_union"
        }
        self.assertEqual(
            restored_top_fields,
            {"contract.visited_source_keys"},
        )
        restored_entity_fields = {
            (item["entity_type"], item["entity_id"], item["field"])
            for item in audit
            if item["normalization"] == "monotonic_previous_evidence_union"
        }
        self.assertEqual(
            restored_entity_fields,
            {
                ("subject", "甲", "source_keys"),
                ("obligation", "o1", "support_keys"),
                ("obligation", "o1", "counter_keys"),
                ("interpretation", "i1", "support_keys"),
                ("interpretation", "i1", "counter_keys"),
                ("uncertainty", "u1", "source_keys"),
            },
        )
        self.assertFalse(
            any(item.get("entity_id") in {"i2", "u2"} for item in audit)
        )

    async def test_tried_signatures_are_canonicalized_to_host_execution(
        self,
    ) -> None:
        fake_first = "a" * 64
        fake_second = "b" * 64
        action_raw = {
            "obligation_id": "o1",
            "tool_name": "mr_query_event_context",
            "arguments": {"event_id": 7, "limit": 20},
            "discriminator": "展开事件七的原始证据",
            "expected_delta": "找到支持或反驳解释一的消息",
        }
        action_signature = RetrievalAction.from_value(
            action_raw,
            field="action",
        ).signature()
        first = self.contract(0)
        first["tried_action_signatures"] = [fake_first]
        second = copy.deepcopy(first)
        second["step_index"] = 1
        second["tried_action_signatures"] = [fake_second]
        responses = [
            {
                "contract": first,
                "actions": [action_raw],
                "memory_brief": None,
                "terminal": False,
            },
            {
                "contract": second,
                "actions": [],
                "memory_brief": None,
                "terminal": False,
            },
        ]
        executed: list[RetrievalAction] = []

        async def complete(
            _system: str,
            _prompt: str,
            index: int,
            _phase: str,
        ):
            return responses[index]

        async def execute(action: RetrievalAction):
            executed.append(action)
            return {"messages": []}

        result = await EccrOrchestrator(
            limits=EccrLimits(
                max_model_calls=2,
                max_retrieval_rounds=1,
                audit_discovery=True,
            )
        ).run(
            query="问题",
            host_contract_fields=self.host,
            evidence_packet=self.packet,
            complete=complete,
            execute_action=execute,
            allowed_tool_names={"mr_query_event_context"},
        )

        self.assertEqual(len(executed), 1)
        self.assertEqual(
            result.final_turn.contract.tried_action_signatures,
            (action_signature,),
        )
        first_audit = result.trace[0].normalization_audit[0]
        self.assertEqual(
            first_audit["normalization"],
            "canonicalize_host_tried_action_signatures",
        )
        self.assertEqual(
            first_audit["removed_unexecuted_action_signatures"],
            [fake_first],
        )
        second_audit = result.trace[1].normalization_audit[0]
        self.assertEqual(second_audit["canonical_value"], [action_signature])
        self.assertEqual(
            second_audit["restored_action_signatures"],
            [action_signature],
        )
        self.assertEqual(
            second_audit["removed_unexecuted_action_signatures"],
            [fake_second],
        )

    async def test_normalization_runs_only_after_schema_validation(self) -> None:
        first = self.contract(0)
        second = copy.deepcopy(first)
        second["step_index"] = 1
        second["subjects"][0]["source_keys"] = "s1"

        with self.assertRaisesRegex(ValueError, "must be an array"):
            await self.run_contract_pair(first, second)

    async def test_monotonic_normalization_does_not_hide_semantic_rewrites(
        self,
    ) -> None:
        variants: dict[
            str,
            tuple[Callable[[dict[str, object]], object], str],
        ] = {
            "obligation-question": (
                lambda contract: contract["obligations"][0].update(
                    {"question": "模型改写后的问题"}
                ),
                "obligation o1 changed its definition",
            ),
            "interpretation-statement": (
                lambda contract: contract["interpretations"][0].update(
                    {"statement": "模型用同一 ID 换了命题", "support_keys": []}
                ),
                "rewrote interpretation i1",
            ),
            "uncertainty-statement": (
                lambda contract: contract["uncertainties"][0].update(
                    {"statement": "模型用同一 ID 换了约束", "source_keys": []}
                ),
                "rewrote uncertainty u1",
            ),
            "obligation-kind": (
                lambda contract: contract["obligations"][0].update(
                    {"kind": "identity"}
                ),
                "obligation o1 changed its definition",
            ),
            "obligation-critical": (
                lambda contract: contract["obligations"][0].update(
                    {"critical": False}
                ),
                "critical obligation",
            ),
            "host-identity": (
                lambda contract: contract["subjects"][0].update(
                    {"participant_key": "p2", "mode": "UNIQUE_ALIAS"}
                ),
                "attempted to rewrite host identity",
            ),
        }
        for name, (mutate, error) in variants.items():
            with self.subTest(name=name):
                first = self.contract(0)
                first["interpretations"][0]["support_keys"] = ["s1"]
                first["uncertainties"][0]["source_keys"] = ["s1"]
                second = copy.deepcopy(first)
                second["step_index"] = 1
                mutate(second)
                with self.assertRaisesRegex(ValueError, error):
                    await self.run_contract_pair(first, second)

    async def test_evidence_union_does_not_hide_ungrounded_status_drift(
        self,
    ) -> None:
        variants: dict[
            str,
            tuple[Callable[[dict[str, object]], object], str],
        ] = {
            "obligation": (
                lambda contract: contract["obligations"][0].update(
                    {
                        "status": "SUPPORTED",
                        "support_keys": [],
                        "last_changed_step": 1,
                    }
                ),
                "obligation o1 changed status without new evidence",
            ),
            "interpretation": (
                lambda contract: contract["interpretations"][0].update(
                    {"status": "SUPPORTED", "support_keys": []}
                ),
                "interpretation i1 changed without new evidence",
            ),
            # Desensitized reproduction of v3: a later turn regresses a
            # PRESERVED constraint to OPEN while omitting one old source.
            "uncertainty-v3-shape": (
                lambda contract: contract["uncertainties"][0].update(
                    {"status": "OPEN", "source_keys": ["s1"]}
                ),
                "uncertainty u1 changed without new evidence",
            ),
        }
        for name, (mutate, error) in variants.items():
            with self.subTest(name=name):
                first = self.contract(0)
                first["obligations"][0]["support_keys"] = ["s1"]
                first["interpretations"][0]["support_keys"] = ["s1"]
                first["uncertainties"][0].update(
                    {"status": "PRESERVED", "source_keys": ["s1", "s2"]}
                )
                first["visited_source_keys"] = ["s1", "s2"]
                second = copy.deepcopy(first)
                second["step_index"] = 1
                mutate(second)
                with self.assertRaisesRegex(ValueError, error):
                    await self.run_contract_pair(first, second)

    async def test_deleted_contract_rows_remain_hard_failures(self) -> None:
        variants: dict[
            str,
            tuple[Callable[[dict[str, object]], object], str],
        ] = {
            "subject": (
                lambda contract: contract.update({"subjects": []}),
                "changed the subject-reference set",
            ),
            "obligation": (
                lambda contract: contract["obligations"].pop(),
                "changed the obligation set",
            ),
            "interpretation": (
                lambda contract: contract.update({"interpretations": []}),
                "removed an interpretation",
            ),
            "uncertainty": (
                lambda contract: contract.update({"uncertainties": []}),
                "removed an uncertainty",
            ),
        }
        for name, (mutate, error) in variants.items():
            with self.subTest(name=name):
                first = self.contract(0)
                first["obligations"].append(
                    {
                        "id": "o2",
                        "kind": "semantic",
                        "question": "第二个必须保留的问题",
                        "critical": True,
                        "status": "OPEN",
                        "support_keys": [],
                        "counter_keys": [],
                        "last_changed_step": 0,
                    }
                )
                second = copy.deepcopy(first)
                second["step_index"] = 1
                mutate(second)
                with self.assertRaisesRegex(ValueError, error):
                    await self.run_contract_pair(first, second)


if __name__ == "__main__":
    unittest.main()
