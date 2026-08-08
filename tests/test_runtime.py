from __future__ import annotations

import json
import unittest

from mr_memory.runtime import (
    feedback_packet_edge_ids,
    feedback_packet_evidence,
    parse_feedback_batch_plan,
    parse_reconstruction_plan,
    parse_structured_response,
    reconstruction_packet_allowlist,
    structured_response_candidates,
)


class RuntimePlanTests(unittest.TestCase):
    @staticmethod
    def none_plan() -> dict[str, object]:
        return {
            "decision": "none",
            "memory_brief": {"claims": [], "conflicts": [], "unresolved": []},
            "activate_hypotheses": [],
            "activate_edges": [],
            "escalation_question": "",
        }

    def test_terminal_json_is_recovered_without_accepting_scratch_objects(self) -> None:
        value = (
            'provider note {"diagnostic":true}\n'
            + json.dumps(self.none_plan(), ensure_ascii=False)
        )
        plan = parse_reconstruction_plan(value, allowed_source_keys=set())
        self.assertEqual(plan.decision, "none")

        reasoning = (
            "分析完成。\n```json\n"
            + json.dumps(self.none_plan(), ensure_ascii=False)
            + "\n```"
        )
        parsed, source = parse_structured_response(
            completion_text="",
            reasoning_content=reasoning,
            parser=lambda item: parse_reconstruction_plan(
                item,
                allowed_source_keys=set(),
            ),
        )
        self.assertEqual(parsed.decision, "none")
        self.assertEqual(source, "reasoning_terminal")

        self.assertEqual(
            structured_response_candidates(
                "",
                json.dumps(self.none_plan()) + "\n仍在推理，尚未形成最终回答",
            ),
            (),
        )
        with self.assertRaisesRegex(ValueError, "terminal JSON"):
            parse_structured_response(
                completion_text="",
                reasoning_content=(
                    json.dumps(self.none_plan()) + "\n仍在推理，尚未形成最终回答"
                ),
                parser=lambda item: parse_reconstruction_plan(
                    item,
                    allowed_source_keys=set(),
                ),
            )

    def test_one_pass_reconstruction_is_grounded_and_bounded(self) -> None:
        plan = parse_reconstruction_plan(
            json.dumps(
                {
                    "decision": "brief",
                    "memory_brief": {
                        "claims": [
                            {
                                "statement": "群内把这个称为好女孩。",
                                "source_keys": ["source-1"],
                                "confidence": 0.7,
                            }
                        ],
                        "conflicts": [],
                        "unresolved": [],
                    },
                    "activate_hypotheses": [{"id": 7, "relevance": 0.82}],
                    "activate_edges": [{"id": 9, "relevance": 0.74}],
                    "escalation_question": "",
                },
                ensure_ascii=False,
            ),
            allowed_source_keys={"source-1"},
            allowed_hypothesis_ids={7},
            allowed_edge_ids={9},
        )
        self.assertEqual(plan.decision, "brief")
        self.assertEqual(plan.hypothesis_activations, ((7, 0.82),))
        self.assertEqual(plan.edge_activations, ((9, 0.74),))
        self.assertIsNotNone(plan.brief)

        with self.assertRaises(ValueError):
            parse_reconstruction_plan(
                {
                    "decision": "brief",
                    "memory_brief": {
                        "claims": [
                            {
                                "statement": "无来源结论",
                                "source_keys": ["invented"],
                                "confidence": 0.9,
                            }
                        ],
                        "conflicts": [],
                        "unresolved": [],
                    },
                },
                allowed_source_keys={"source-1"},
            )

    def test_non_brief_reconstruction_cannot_activate_memory_paths(self) -> None:
        for decision, extra in (
            ("none", {}),
            ("escalate", {"escalation_question": "需要沿哪条边继续检索？"}),
        ):
            with self.subTest(decision=decision), self.assertRaises(ValueError):
                parse_reconstruction_plan(
                    {
                        "decision": decision,
                        "memory_brief": {
                            "claims": [],
                            "conflicts": [],
                            "unresolved": [],
                        },
                        "activate_hypotheses": [{"id": 7, "relevance": 0.8}],
                        "activate_edges": [],
                        **extra,
                    },
                    allowed_source_keys=set(),
                    allowed_hypothesis_ids={7},
                )

        plan = parse_reconstruction_plan(
            {
                "decision": "none",
                "memory_brief": {
                    "claims": [],
                    "conflicts": [],
                    "unresolved": [],
                },
                "activate_hypotheses": [],
                "activate_edges": [],
            },
            allowed_source_keys=set(),
        )
        self.assertEqual(plan.decision, "none")

    def test_feedback_batch_requires_one_plan_per_proposal(self) -> None:
        value = {
            "plans": [
                {
                    "proposal_id": 1,
                    "decision": {
                        "target_trace_id": "trace-1",
                        "mutation": "upsert",
                        "feedback_valence": -1.0,
                        "confidence": 0.4,
                        "scope_type": "sender",
                        "scope_key": "user-a",
                        "aspect": "image_density",
                        "statement": "群友认为生成图元素太密集。",
                        "prospective_cue": "后续生图减少同时出现的元素。",
                        "trigger_cues": ["生图", "画图"],
                        "activation_mode": "semantic",
                        "target_hypothesis_id": None,
                    },
                    "graph_mutations": [],
                },
                {
                    "proposal_id": 2,
                    "decision": {
                        "target_trace_id": "",
                        "mutation": "ignore",
                        "feedback_valence": 0,
                        "confidence": 0.2,
                        "scope_type": "sender",
                        "scope_key": "",
                        "aspect": "",
                        "statement": "",
                        "prospective_cue": "",
                        "trigger_cues": [],
                        "activation_mode": "always",
                        "target_hypothesis_id": None,
                    },
                    "graph_mutations": [],
                },
            ]
        }
        plans = parse_feedback_batch_plan(
            value,
            proposal_evidence={1: {"feedback-1"}, 2: {"feedback-2"}},
        )
        self.assertEqual([plan.proposal_id for plan in plans], [1, 2])
        self.assertEqual(plans[0].decision.confidence, 0.4)

        with self.assertRaises(ValueError):
            parse_feedback_batch_plan(
                {"plans": value["plans"][:1]},
                proposal_evidence={1: {"feedback-1"}, 2: {"feedback-2"}},
            )

    def test_allowlists_follow_only_the_delivered_packet(self) -> None:
        sources, hypotheses, edges = reconstruction_packet_allowlist(
            {
                "candidates": {
                    "feedback_hypotheses": [{"id": 7, "source_key": "s-1"}],
                    "associations": [{"id": 9, "source_keys": ["s-2"]}],
                },
                "expanded_episodes": [{"source_key": "s-3"}],
            }
        )
        self.assertEqual(sources, {"s-1", "s-2", "s-3"})
        self.assertEqual(hypotheses, {7})
        self.assertEqual(edges, {9})

        evidence = feedback_packet_evidence(
            {
                "items": [
                    {"proposal_id": 4, "evidence": {"source_key": "f-1"}},
                    {"proposal_id": 5, "evidence": {"source_keys": ["f-2"]}},
                ]
            }
        )
        self.assertEqual(evidence, {4: {"f-1"}, 5: {"f-2"}})
        edge_ids = feedback_packet_edge_ids(
            {
                "items": [
                    {
                        "proposal_id": 4,
                        "evidence": {
                            "activated_plastic_edges": [
                                {"edge_id": 12, "relation": "means"}
                            ]
                        },
                    }
                ]
            }
        )
        self.assertEqual(edge_ids, {4: {12}})

    def test_feedback_packet_allowlists_accept_runtime_nested_proposal(self) -> None:
        packet = {
            "items": [
                {
                    "proposal": {"id": 12, "surface_score": 0.8},
                    "evidence": {
                        "feedback": {"source_key": "feedback:12"},
                        "activated_plastic_edges": [
                            {"edge_id": 37, "relation": "corrects"}
                        ],
                    },
                }
            ]
        }

        self.assertEqual(
            feedback_packet_evidence(packet),
            {12: {"feedback:12"}},
        )
        self.assertEqual(feedback_packet_edge_ids(packet), {12: {37}})


if __name__ == "__main__":
    unittest.main()
