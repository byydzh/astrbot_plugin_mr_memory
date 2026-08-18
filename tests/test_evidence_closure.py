from __future__ import annotations

import copy
import hashlib
import unittest

from mr_memory.evidence_closure import (
    BudgetState,
    GainVector,
    compile_or_update_contract,
    evidence_gain,
    parse_contract_turn,
    should_stop,
    validate_actions,
)


class EvidenceClosureTests(unittest.TestCase):
    scope_hash = hashlib.sha256(b"scope-a").hexdigest()
    query_hash = hashlib.sha256(b"query-a").hexdigest()
    allowed_tools = {
        "mr_query_topic_events",
        "mr_query_event_context",
    }

    @classmethod
    def contract(
        cls,
        *,
        step: int = 0,
        obligation_status: str = "OPEN",
        support_keys: list[str] | None = None,
        counter_keys: list[str] | None = None,
        last_changed_step: int = 0,
        subject: dict[str, object] | None = None,
        visited: list[str] | None = None,
        tried: list[str] | None = None,
        exhausted: list[str] | None = None,
        frontier: list[str] | None = None,
        with_uncertainty: bool = True,
    ) -> dict[str, object]:
        support = support_keys or []
        counter = counter_keys or []
        interpretation_status = (
            "SUPPORTED" if obligation_status == "SUPPORTED" else "CANDIDATE"
        )
        return {
            "contract_id": "contract-a",
            "scope_sha256": cls.scope_hash,
            "query_sha256": cls.query_hash,
            "cutoff_at": 123456,
            "revision_vector": {
                "message": "m1",
                "graph": "g1",
                "identity": "i1",
                "relation": "r1",
                "feedback": "f1",
                "protocol": "eccr-v1",
            },
            "step_index": step,
            "subjects": [
                subject
                or {
                    "reference": "群友甲",
                    "participant_key": "participant:a",
                    "mode": "HOST",
                    "candidate_participant_keys": [],
                    "source_keys": [],
                    "valid_at": 123400,
                }
            ],
            "obligations": [
                {
                    "id": "o-main",
                    "kind": "semantic",
                    "question": "早期表态与后续行为是否形成带保留的反差？",
                    "critical": True,
                    "status": obligation_status,
                    "support_keys": support,
                    "counter_keys": counter,
                    "last_changed_step": last_changed_step,
                }
            ],
            "interpretations": [
                {
                    "id": "reading-a",
                    "statement": "存在带保留的群内调侃释义。",
                    "status": interpretation_status,
                    "support_keys": support if interpretation_status == "SUPPORTED" else [],
                    "counter_keys": [],
                    "uncertainty": "不得升级为逐字动机或购买事实。",
                }
            ],
            "uncertainties": (
                [
                    {
                        "id": "u-overclaim",
                        "statement": "不得把行为概括升级成当事人的逐字动机。",
                        "status": "PRESERVED" if support else "OPEN",
                        "source_keys": support,
                    }
                ]
                if with_uncertainty
                else []
            ),
            "guarded_claims": ["不能声称有逐字预购或明确动机。"],
            "visited_source_keys": visited or [],
            "selected_edge_ids": [],
            "selected_hypothesis_ids": [],
            "tried_action_signatures": tried or [],
            "exhausted_discriminators": exhausted or [],
            "frontier_discriminators": frontier or ["查找后续实际参与证据"],
        }

    @staticmethod
    def action(
        *,
        tool: str = "mr_query_topic_events",
        discriminator: str = "查找后续实际参与证据",
    ) -> dict[str, object]:
        return {
            "obligation_id": "o-main",
            "tool_name": tool,
            "arguments": {"topic": "相关游戏", "limit": 12},
            "discriminator": discriminator,
            "expected_delta": "找到或排除后续实际购买/游玩证据",
        }

    @classmethod
    def turn(
        cls,
        contract: dict[str, object],
        *,
        actions: list[dict[str, object]] | None = None,
        brief: dict[str, object] | None = None,
        terminal: bool = False,
    ) -> dict[str, object]:
        return {
            "contract": contract,
            "actions": actions or [],
            "memory_brief": brief,
            "terminal": terminal,
        }

    def parse_initial(self):
        return parse_contract_turn(
            self.turn(self.contract(), actions=[self.action()]),
            allowed_source_keys={"source-1", "source-2"},
            allowed_participant_keys={"participant:a", "participant:b"},
            allowed_tool_names=self.allowed_tools,
        )

    def test_resolved_subject_canonicalizes_a_redundant_single_candidate(self) -> None:
        contract = self.contract()
        contract["subjects"][0]["candidate_participant_keys"] = ["participant:a"]
        parsed = parse_contract_turn(
            self.turn(contract, actions=[self.action()]),
            allowed_source_keys={"source-1"},
            allowed_participant_keys={"participant:a", "participant:b"},
            allowed_tool_names=self.allowed_tools,
        )
        self.assertEqual(parsed.contract.subjects[0].participant_key, "participant:a")
        self.assertEqual(parsed.contract.subjects[0].candidate_participant_keys, ())

        unsafe = self.contract()
        unsafe["subjects"][0]["candidate_participant_keys"] = [
            "participant:a",
            "participant:b",
        ]
        with self.assertRaisesRegex(ValueError, "exactly one participant_key"):
            parse_contract_turn(
                self.turn(unsafe, actions=[self.action()]),
                allowed_source_keys={"source-1"},
                allowed_participant_keys={"participant:a", "participant:b"},
                allowed_tool_names=self.allowed_tools,
            )

    def test_status_direction_must_match_each_items_evidence_sides(self) -> None:
        invalid_obligations = (
            ("SUPPORTED", ["source-1"], ["source-2"], "retain counterevidence"),
            ("REFUTED", ["source-1"], ["source-2"], "retain supporting evidence"),
            ("CONTESTED", ["source-1"], [], "both evidence sides"),
        )
        for status, support, counter, message in invalid_obligations:
            with self.subTest(kind="obligation", status=status):
                contract = self.contract(
                    obligation_status=status,
                    support_keys=support,
                    counter_keys=counter,
                    visited=["source-1", "source-2"],
                )
                with self.assertRaisesRegex(ValueError, message):
                    parse_contract_turn(
                        self.turn(contract),
                        allowed_source_keys={"source-1", "source-2"},
                        allowed_participant_keys={"participant:a"},
                        allowed_tool_names=self.allowed_tools,
                    )

        invalid_interpretations = (
            ("SUPPORTED", ["source-1"], ["source-2"], "retain counterevidence"),
            ("REFUTED", ["source-1"], ["source-2"], "retain supporting evidence"),
            ("CONTESTED", ["source-1"], [], "both evidence sides"),
        )
        for status, support, counter, message in invalid_interpretations:
            with self.subTest(kind="interpretation", status=status):
                contract = self.contract(visited=["source-1", "source-2"])
                contract["interpretations"][0].update(
                    {
                        "status": status,
                        "support_keys": support,
                        "counter_keys": counter,
                    }
                )
                with self.assertRaisesRegex(ValueError, message):
                    parse_contract_turn(
                        self.turn(contract),
                        allowed_source_keys={"source-1", "source-2"},
                        allowed_participant_keys={"participant:a"},
                        allowed_tool_names=self.allowed_tools,
                    )

    def test_resolved_subject_modes_freeze_identity_mode_and_valid_time(self) -> None:
        for mode in ("HOST", "STRUCTURED_REF", "UNIQUE_ALIAS"):
            with self.subTest(mode=mode):
                initial_contract = self.contract()
                initial_contract["subjects"][0]["mode"] = mode
                initial = parse_contract_turn(
                    self.turn(initial_contract),
                    allowed_source_keys={"source-identity"},
                    allowed_participant_keys={"participant:a", "participant:b"},
                    allowed_tool_names=self.allowed_tools,
                )

                augmented = self.contract(step=1, visited=["source-identity"])
                augmented["subjects"][0].update(
                    {"mode": mode, "source_keys": ["source-identity"]}
                )
                parsed = parse_contract_turn(
                    self.turn(augmented),
                    allowed_source_keys={"source-identity"},
                    allowed_participant_keys={"participant:a", "participant:b"},
                    allowed_tool_names=self.allowed_tools,
                    previous=initial.contract,
                )
                self.assertEqual(parsed.contract.subjects[0].mode, mode)

                mutations = (
                    {"participant_key": "participant:b"},
                    {"mode": "HOST" if mode != "HOST" else "UNIQUE_ALIAS"},
                    {"valid_at": 123401},
                )
                for mutation in mutations:
                    rewritten = copy.deepcopy(augmented)
                    rewritten["subjects"][0].update(mutation)
                    with self.assertRaisesRegex(ValueError, "resolved binding"):
                        parse_contract_turn(
                            self.turn(rewritten),
                            allowed_source_keys={"source-identity"},
                            allowed_participant_keys={
                                "participant:a",
                                "participant:b",
                            },
                            allowed_tool_names=self.allowed_tools,
                            previous=initial.contract,
                        )

    def test_ambiguous_candidate_changes_require_new_binding_evidence(self) -> None:
        subject = {
            "reference": "群内称呼",
            "participant_key": "",
            "mode": "AMBIGUOUS",
            "candidate_participant_keys": ["participant:a", "participant:b"],
            "source_keys": ["source-old"],
            "valid_at": 123400,
        }
        initial_contract = self.contract(
            subject=subject,
            visited=["source-old"],
        )
        initial = parse_contract_turn(
            self.turn(initial_contract),
            allowed_source_keys={"source-old", "source-new"},
            allowed_participant_keys={
                "participant:a",
                "participant:b",
                "participant:c",
            },
            allowed_tool_names=self.allowed_tools,
        )
        changed = copy.deepcopy(initial_contract)
        changed["step_index"] = 1
        changed["subjects"][0]["candidate_participant_keys"] = [
            "participant:b",
            "participant:c",
        ]
        with self.assertRaisesRegex(ValueError, "without host evidence"):
            parse_contract_turn(
                self.turn(changed),
                allowed_source_keys={"source-old", "source-new"},
                allowed_participant_keys={
                    "participant:a",
                    "participant:b",
                    "participant:c",
                },
                allowed_tool_names=self.allowed_tools,
                previous=initial.contract,
            )

        changed["subjects"][0]["source_keys"] = ["source-old", "source-new"]
        changed["visited_source_keys"] = ["source-old", "source-new"]
        parsed = parse_contract_turn(
            self.turn(changed),
            allowed_source_keys={"source-old", "source-new"},
            allowed_participant_keys={
                "participant:a",
                "participant:b",
                "participant:c",
            },
            allowed_tool_names=self.allowed_tools,
            previous=initial.contract,
        )
        self.assertEqual(
            parsed.contract.subjects[0].candidate_participant_keys,
            ("participant:b", "participant:c"),
        )

    def test_unchanged_obligation_cannot_drift_last_changed_step(self) -> None:
        initial = self.parse_initial()
        drifted = self.contract(step=1, last_changed_step=1)
        with self.assertRaisesRegex(ValueError, "without a status transition"):
            parse_contract_turn(
                self.turn(drifted),
                allowed_source_keys={"source-1", "source-2"},
                allowed_participant_keys={"participant:a", "participant:b"},
                allowed_tool_names=self.allowed_tools,
                previous=initial.contract,
            )

    def test_grounded_transition_closes_with_auditable_uncertainty(self) -> None:
        initial = self.parse_initial()
        signature = initial.actions[0].signature()
        final_contract = self.contract(
            step=1,
            obligation_status="SUPPORTED",
            support_keys=["source-1"],
            last_changed_step=1,
            visited=["source-1"],
            tried=[signature],
            frontier=[],
        )
        final = compile_or_update_contract(
            self.turn(
                final_contract,
                brief={
                    "claims": [
                        {
                            "statement": "证据支持带保留地概括为前后反差。",
                            "source_keys": ["source-1"],
                            "confidence": 0.72,
                        }
                    ],
                    "conflicts": [],
                    "unresolved": [
                        {
                            "statement": "不能据此断言逐字动机或明确预购。",
                            "source_keys": ["source-1"],
                        }
                    ],
                },
                terminal=True,
            ),
            allowed_source_keys={"source-1", "source-2"},
            allowed_participant_keys={"participant:a", "participant:b"},
            allowed_tool_names=self.allowed_tools,
            previous=initial.contract,
            tried_action_signatures={signature},
        )
        self.assertTrue(final.terminal)
        self.assertEqual(final.contract.obligations[0].status, "SUPPORTED")

    def test_frontier_progress_cannot_reclassify_already_selected_evidence(self) -> None:
        initial_contract = self.contract(
            support_keys=["source-1"],
            visited=["source-1"],
        )
        initial_contract["interpretations"][0]["support_keys"] = ["source-1"]
        initial = parse_contract_turn(
            self.turn(initial_contract, actions=[self.action()]),
            allowed_source_keys={"source-1"},
            allowed_participant_keys={"participant:a"},
            allowed_tool_names=self.allowed_tools,
        )
        signature = initial.actions[0].signature()
        final_contract = self.contract(
            step=1,
            obligation_status="SUPPORTED",
            support_keys=["source-1"],
            last_changed_step=1,
            visited=["source-1"],
            tried=[signature],
            frontier=[],
        )
        with self.assertRaisesRegex(ValueError, "without new evidence"):
            compile_or_update_contract(
                self.turn(final_contract, terminal=True),
                allowed_source_keys={"source-1"},
                allowed_participant_keys={"participant:a"},
                allowed_tool_names=self.allowed_tools,
                previous=initial.contract,
                tried_action_signatures={signature},
            )

    def test_terminal_turn_cannot_silently_drop_preserved_uncertainty(self) -> None:
        initial = self.parse_initial()
        signature = initial.actions[0].signature()
        final_contract = self.contract(
            step=1,
            obligation_status="SUPPORTED",
            support_keys=["source-1"],
            last_changed_step=1,
            visited=["source-1"],
            tried=[signature],
            frontier=[],
        )
        with self.assertRaisesRegex(ValueError, "explicit unresolved brief"):
            parse_contract_turn(
                self.turn(
                    final_contract,
                    brief={
                        "claims": [
                            {
                                "statement": "有前后反差。",
                                "source_keys": ["source-1"],
                                "confidence": 0.7,
                            }
                        ],
                        "conflicts": [],
                        "unresolved": [],
                    },
                    terminal=True,
                ),
                allowed_source_keys={"source-1"},
                allowed_participant_keys={"participant:a"},
                allowed_tool_names=self.allowed_tools,
                previous=initial.contract,
                tried_action_signatures={signature},
            )

    def test_allowlists_and_host_scope_boundary_are_strict(self) -> None:
        ungrounded = self.contract(
            obligation_status="SUPPORTED",
            support_keys=["invented-source"],
            visited=["invented-source"],
        )
        with self.assertRaisesRegex(ValueError, "delivered allowlist"):
            parse_contract_turn(
                self.turn(ungrounded),
                allowed_source_keys={"source-1"},
                allowed_participant_keys={"participant:a"},
                allowed_tool_names=self.allowed_tools,
            )

        bad_identity = self.contract()
        bad_identity["subjects"][0]["participant_key"] = "participant:invented"
        with self.assertRaisesRegex(ValueError, "host-authorized"):
            parse_contract_turn(
                self.turn(bad_identity, actions=[self.action()]),
                allowed_source_keys=set(),
                allowed_participant_keys={"participant:a"},
                allowed_tool_names=self.allowed_tools,
            )

        scoped_action = self.action()
        scoped_action["arguments"] = {
            "topic": "相关游戏",
            "umo": "model-selected-tenant",
        }
        with self.assertRaisesRegex(ValueError, "cannot choose scope"):
            parse_contract_turn(
                self.turn(self.contract(), actions=[scoped_action]),
                allowed_source_keys=set(),
                allowed_participant_keys={"participant:a"},
                allowed_tool_names=self.allowed_tools,
            )

    def test_contract_requires_at_least_one_critical_obligation(self) -> None:
        no_critical = self.contract()
        no_critical["obligations"][0]["critical"] = False
        with self.assertRaisesRegex(ValueError, "at least one critical obligation"):
            parse_contract_turn(
                self.turn(no_critical),
                allowed_source_keys=set(),
                allowed_participant_keys={"participant:a"},
                allowed_tool_names=self.allowed_tools,
            )

    def test_repeated_or_exhausted_action_is_rejected(self) -> None:
        initial = self.parse_initial()
        signature = initial.actions[0].signature()
        with self.assertRaisesRegex(ValueError, "previously tried"):
            validate_actions(
                initial,
                allowed_tool_names=self.allowed_tools,
                tried_signatures={signature},
            )

        exhausted_contract = self.contract(
            exhausted=["查找后续实际参与证据"]
        )
        with self.assertRaisesRegex(ValueError, "exhausted discriminator"):
            parse_contract_turn(
                self.turn(exhausted_contract, actions=[self.action()]),
                allowed_source_keys=set(),
                allowed_participant_keys={"participant:a"},
                allowed_tool_names=self.allowed_tools,
            )

    def test_global_frontier_does_not_ground_exhaustion_without_bound_evidence(
        self,
    ) -> None:
        initial_contract = self.contract(
            visited=["source-context"],
            with_uncertainty=False,
        )
        initial = parse_contract_turn(
            self.turn(
                initial_contract,
                actions=[self.action()],
            ),
            allowed_source_keys={"source-context"},
            allowed_participant_keys={"participant:a"},
            allowed_tool_names=self.allowed_tools,
        )
        signature = initial.actions[0].signature()
        exhausted = self.contract(
            step=1,
            obligation_status="EXHAUSTED",
            last_changed_step=1,
            tried=[signature],
            exhausted=["查找后续实际参与证据"],
            frontier=[],
            visited=["source-context"],
            with_uncertainty=False,
        )
        with self.assertRaisesRegex(ValueError, "bound to that obligation"):
            parse_contract_turn(
                self.turn(exhausted, terminal=True),
                allowed_source_keys={"source-context"},
                allowed_participant_keys={"participant:a"},
                allowed_tool_names=self.allowed_tools,
                previous=initial.contract,
                tried_action_signatures={signature},
            )

        grounded = copy.deepcopy(exhausted)
        grounded["obligations"][0]["support_keys"] = ["source-exhaustion"]
        grounded["visited_source_keys"] = ["source-context", "source-exhaustion"]
        final = parse_contract_turn(
            self.turn(
                grounded,
                brief={
                    "claims": [],
                    "conflicts": [],
                    "unresolved": [
                        {
                            "statement": "新返回证据仍不足以闭合目标语义。",
                            "source_keys": ["source-exhaustion"],
                        }
                    ],
                },
                terminal=True,
            ),
            allowed_source_keys={"source-context", "source-exhaustion"},
            allowed_participant_keys={"participant:a"},
            allowed_tool_names=self.allowed_tools,
            previous=initial.contract,
            tried_action_signatures={signature},
        )
        self.assertTrue(final.terminal)
        self.assertEqual(final.contract.obligations[0].status, "EXHAUSTED")

        ungrounded = copy.deepcopy(exhausted)
        ungrounded["tried_action_signatures"] = []
        ungrounded["exhausted_discriminators"] = []
        with self.assertRaisesRegex(ValueError, "bound to that obligation"):
            parse_contract_turn(
                self.turn(ungrounded, terminal=True),
                allowed_source_keys={"source-context"},
                allowed_participant_keys={"participant:a"},
                allowed_tool_names=self.allowed_tools,
                previous=initial.contract,
            )

    def test_global_frontier_cannot_mark_unrelated_interpretation_unresolved(
        self,
    ) -> None:
        initial_contract = self.contract(visited=["source-old"])
        initial_contract["interpretations"][0]["support_keys"] = ["source-old"]
        initial = parse_contract_turn(
            self.turn(initial_contract, actions=[self.action()]),
            allowed_source_keys={"source-old", "source-new"},
            allowed_participant_keys={"participant:a"},
            allowed_tool_names=self.allowed_tools,
        )
        signature = initial.actions[0].signature()

        unrelated = copy.deepcopy(initial_contract)
        unrelated["step_index"] = 1
        unrelated["tried_action_signatures"] = [signature]
        unrelated["exhausted_discriminators"] = [
            "查找后续实际参与证据"
        ]
        unrelated["frontier_discriminators"] = []
        unrelated["interpretations"][0]["status"] = "UNRESOLVED"
        with self.assertRaisesRegex(ValueError, "evidence bound to it"):
            parse_contract_turn(
                self.turn(unrelated),
                allowed_source_keys={"source-old", "source-new"},
                allowed_participant_keys={"participant:a"},
                allowed_tool_names=self.allowed_tools,
                previous=initial.contract,
                tried_action_signatures={signature},
            )

        grounded = copy.deepcopy(unrelated)
        grounded["visited_source_keys"] = ["source-old", "source-new"]
        grounded["interpretations"][0]["support_keys"] = [
            "source-old",
            "source-new",
        ]
        parsed = parse_contract_turn(
            self.turn(grounded),
            allowed_source_keys={"source-old", "source-new"},
            allowed_participant_keys={"participant:a"},
            allowed_tool_names=self.allowed_tools,
            previous=initial.contract,
            tried_action_signatures={signature},
        )
        self.assertEqual(
            parsed.contract.interpretations[0].status,
            "UNRESOLVED",
        )

    def test_identity_ambiguity_is_a_safe_terminal_state(self) -> None:
        unbound = {
            "reference": "新昵称",
            "participant_key": "",
            "mode": "UNBOUND",
            "candidate_participant_keys": [],
            "source_keys": [],
            "valid_at": 123400,
        }
        initial_contract = self.contract(
            subject=unbound,
            visited=["source-identity"],
            with_uncertainty=False,
        )
        initial_contract["obligations"][0]["kind"] = "identity"
        initial = parse_contract_turn(
            self.turn(initial_contract),
            allowed_source_keys={"source-identity"},
            allowed_participant_keys={"participant:a", "participant:b"},
            allowed_tool_names=self.allowed_tools,
        )

        ambiguous = copy.deepcopy(initial_contract)
        ambiguous["step_index"] = 1
        ambiguous["subjects"][0].update(
            {
                "mode": "AMBIGUOUS",
                "candidate_participant_keys": [
                    "participant:a",
                    "participant:b",
                ],
                "source_keys": ["source-identity"],
            }
        )
        ambiguous["obligations"][0].update(
            {
                "status": "AMBIGUOUS",
                "counter_keys": ["source-identity"],
                "last_changed_step": 1,
            }
        )
        final = parse_contract_turn(
            self.turn(
                ambiguous,
                brief={
                    "claims": [],
                    "conflicts": [],
                    "unresolved": [
                        {
                            "statement": "该新昵称仍可能指向两个不同账号。",
                            "source_keys": ["source-identity"],
                        }
                    ],
                },
                terminal=True,
            ),
            allowed_source_keys={"source-identity"},
            allowed_participant_keys={"participant:a", "participant:b"},
            allowed_tool_names=self.allowed_tools,
            previous=initial.contract,
        )
        stop = should_stop(
            final.contract,
            actions=(),
            budget=BudgetState(2, 3, 1, 2),
        )
        self.assertEqual(stop.reason, "SAFETY_ABSTAIN")
        self.assertTrue(stop.force_unresolved)

    def test_terminal_unbound_subject_requires_and_keeps_qualification(self) -> None:
        unbound = {
            "reference": "未识别称呼",
            "participant_key": "",
            "mode": "UNBOUND",
            "candidate_participant_keys": [],
            "source_keys": [],
            "valid_at": 123400,
        }
        contract = self.contract(
            obligation_status="EXHAUSTED",
            subject=unbound,
            visited=["source-context"],
            with_uncertainty=False,
        )
        contract["obligations"][0]["last_changed_step"] = 0
        with self.assertRaisesRegex(ValueError, "explicit unresolved brief"):
            parse_contract_turn(
                self.turn(contract, terminal=True),
                allowed_source_keys={"source-context"},
                allowed_participant_keys={"participant:a"},
                allowed_tool_names=self.allowed_tools,
            )

        final = parse_contract_turn(
            self.turn(
                contract,
                brief={
                    "claims": [],
                    "conflicts": [],
                    "unresolved": [
                        {
                            "statement": "称呼无法安全绑定到群内账号。",
                            "source_keys": ["source-context"],
                        }
                    ],
                },
                terminal=True,
            ),
            allowed_source_keys={"source-context"},
            allowed_participant_keys={"participant:a"},
            allowed_tool_names=self.allowed_tools,
        )
        self.assertTrue(final.terminal)
        self.assertEqual(final.contract.subjects[0].mode, "UNBOUND")

    def test_empty_frontier_can_preserve_uncertainty_and_mark_reading_unresolved(
        self,
    ) -> None:
        initial_contract = self.contract(visited=["source-context"])
        initial_contract["interpretations"][0]["support_keys"] = [
            "source-context"
        ]
        initial_contract["uncertainties"][0]["source_keys"] = [
            "source-context"
        ]
        initial = parse_contract_turn(
            self.turn(initial_contract, actions=[self.action()]),
            allowed_source_keys={"source-context"},
            allowed_participant_keys={"participant:a"},
            allowed_tool_names=self.allowed_tools,
        )
        signature = initial.actions[0].signature()

        exhausted = copy.deepcopy(initial_contract)
        exhausted["step_index"] = 1
        exhausted["tried_action_signatures"] = [signature]
        exhausted["exhausted_discriminators"] = [
            "查找后续实际参与证据"
        ]
        exhausted["frontier_discriminators"] = []
        exhausted["visited_source_keys"] = ["source-context", "source-exhaustion"]
        exhausted["obligations"][0].update(
            {
                "status": "EXHAUSTED",
                "support_keys": ["source-exhaustion"],
                "last_changed_step": 1,
            }
        )
        exhausted["interpretations"][0].update(
            {
                "status": "UNRESOLVED",
                "support_keys": ["source-context", "source-exhaustion"],
            }
        )
        exhausted["uncertainties"][0].update(
            {
                "status": "PRESERVED",
                "source_keys": ["source-context", "source-exhaustion"],
            }
        )
        final = parse_contract_turn(
            self.turn(
                exhausted,
                brief={
                    "claims": [],
                    "conflicts": [],
                    "unresolved": [
                        {
                            "statement": "空结果不能消除原有语义疑虑。",
                            "source_keys": ["source-context", "source-exhaustion"],
                        }
                    ],
                },
                terminal=True,
            ),
            allowed_source_keys={"source-context", "source-exhaustion"},
            allowed_participant_keys={"participant:a"},
            allowed_tool_names=self.allowed_tools,
            previous=initial.contract,
            tried_action_signatures={signature},
        )
        self.assertEqual(final.contract.obligations[0].status, "EXHAUSTED")
        self.assertEqual(final.contract.interpretations[0].status, "UNRESOLVED")
        self.assertEqual(final.contract.uncertainties[0].status, "PRESERVED")

    def test_host_identity_cannot_be_downgraded_or_reassigned(self) -> None:
        initial = self.parse_initial()
        rewritten = self.contract(step=1)
        rewritten["subjects"][0].update(
            {
                "participant_key": "",
                "mode": "AMBIGUOUS",
                "candidate_participant_keys": [
                    "participant:a",
                    "participant:b",
                ],
            }
        )
        with self.assertRaisesRegex(ValueError, "rewrite host identity"):
            parse_contract_turn(
                self.turn(rewritten),
                allowed_source_keys=set(),
                allowed_participant_keys={"participant:a", "participant:b"},
                allowed_tool_names=self.allowed_tools,
                previous=initial.contract,
            )

    def test_zero_source_graph_bridge_counts_as_progress(self) -> None:
        initial = self.parse_initial()
        signature = initial.actions[0].signature()
        next_contract = self.contract(
            step=1,
            tried=[signature],
            frontier=["查找后续实际参与证据", "沿事件节点展开原始消息"],
        )
        next_turn = parse_contract_turn(
            self.turn(
                next_contract,
                actions=[
                    self.action(
                        tool="mr_query_event_context",
                        discriminator="沿事件节点展开原始消息",
                    )
                ],
            ),
            allowed_source_keys={"source-1", "source-2"},
            allowed_participant_keys={"participant:a"},
            allowed_tool_names=self.allowed_tools,
            previous=initial.contract,
            tried_action_signatures={signature},
        )
        gain = evidence_gain(
            initial.contract,
            next_turn.contract,
            delivered_graph_anchors={"cue:later-play", "event:42"},
            result_hashes={hashlib.sha256(b"bridge-result").hexdigest()},
        )
        self.assertTrue(gain.has_progress)
        self.assertFalse(gain.has_semantic_progress)
        decision = should_stop(
            next_turn.contract,
            actions=next_turn.actions,
            budget=BudgetState(2, 3, 1, 2),
            gain=gain,
            consecutive_no_progress_rounds=2,
        )
        self.assertEqual(decision.reason, "CONTINUE")

    def test_saturation_and_budget_stops_remain_distinct(self) -> None:
        turn = self.parse_initial()
        saturated = should_stop(
            turn.contract,
            actions=turn.actions,
            budget=BudgetState(1, 3, 1, 3),
            gain=GainVector(),
            consecutive_no_progress_rounds=2,
        )
        self.assertEqual(saturated.reason, "SATURATED")
        exhausted = should_stop(
            turn.contract,
            actions=turn.actions,
            budget=BudgetState(3, 3, 1, 2),
            gain=GainVector(new_graph_anchors=("cue:1",)),
        )
        self.assertEqual(exhausted.reason, "BUDGET_EXHAUSTED")
        self.assertTrue(exhausted.force_unresolved)

    def test_audit_discovery_can_add_only_grounded_unpromoted_hypothesis(self) -> None:
        initial = self.parse_initial()
        discovered = self.contract(step=1, visited=["source-audit"])
        discovered["interpretations"].append(
            {
                "id": "reading-audit",
                "statement": "新证据提示侮辱性讳称与后续二次戏仿的竞争解释。",
                "status": "CANDIDATE",
                "support_keys": ["source-audit"],
                "counter_keys": [],
                "uncertainty": "邻接并非显式共指，只能保留为候选。",
                "origin": "AUDIT_DISCOVERY",
                "discriminates_interpretation_ids": ["reading-a"],
            }
        )
        discovered["uncertainties"].append(
            {
                "id": "u-audit-link",
                "statement": "尚未证明两段玩笑由同一隐义连接。",
                "status": "OPEN",
                "source_keys": ["source-audit"],
                "origin": "AUDIT_DISCOVERY",
                "discriminates_interpretation_ids": ["reading-a"],
            }
        )
        turn = parse_contract_turn(
            self.turn(discovered),
            allowed_source_keys={"source-audit"},
            allowed_participant_keys={"participant:a"},
            allowed_tool_names=self.allowed_tools,
            previous=initial.contract,
        )
        self.assertEqual(turn.contract.interpretations[-1].origin, "AUDIT_DISCOVERY")

        promoted = copy.deepcopy(discovered)
        promoted["interpretations"][-1]["status"] = "SUPPORTED"
        with self.assertRaisesRegex(ValueError, "cannot be promoted"):
            parse_contract_turn(
                self.turn(promoted),
                allowed_source_keys={"source-audit"},
                allowed_participant_keys={"participant:a"},
                allowed_tool_names=self.allowed_tools,
                previous=initial.contract,
            )

    def test_audit_discovery_must_cite_new_source_and_discriminate_existing(self) -> None:
        initial_contract = self.contract(visited=["source-old"])
        initial = parse_contract_turn(
            self.turn(initial_contract),
            allowed_source_keys={"source-old"},
            allowed_participant_keys={"participant:a"},
            allowed_tool_names=self.allowed_tools,
        )
        invalid = self.contract(step=1, visited=["source-old"])
        invalid["interpretations"].append(
            {
                "id": "reading-audit",
                "statement": "没有新增来源的候选。",
                "status": "CANDIDATE",
                "support_keys": ["source-old"],
                "counter_keys": [],
                "uncertainty": "仍不确定。",
                "origin": "AUDIT_DISCOVERY",
                "discriminates_interpretation_ids": ["reading-a"],
            }
        )
        with self.assertRaisesRegex(ValueError, "newly visited evidence"):
            parse_contract_turn(
                self.turn(invalid),
                allowed_source_keys={"source-old"},
                allowed_participant_keys={"participant:a"},
                allowed_tool_names=self.allowed_tools,
                previous=initial.contract,
            )


if __name__ == "__main__":
    unittest.main()
