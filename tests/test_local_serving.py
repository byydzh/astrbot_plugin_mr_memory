from __future__ import annotations

import json
import unittest

from mr_memory.brief import EvidenceBrief, EvidenceClaim, EvidenceQualification
from mr_memory.local_serving import (
    LOCAL_SERVING_SCHEMA_VERSION,
    LocalServingEnvelopeError,
    _participant_history,
    _participant_history_alias_view,
    _query_identity,
    compile_local_serving_envelope,
)
from mr_memory.runtime import MaterializedReconstruction
from mr_memory.runtime import materialize_reconstruction_packet


class LocalServingEnvelopeTests(unittest.TestCase):
    @staticmethod
    def materialized() -> MaterializedReconstruction:
        return MaterializedReconstruction(
            brief=EvidenceBrief(
                claims=(
                    EvidenceClaim("甲后来明确说已经买了。", ("source-buy",), 0.83),
                ),
                conflicts=(
                    EvidenceQualification(
                        "甲此前说自己不喜欢这个作品。", ("source-dislike",)
                    ),
                ),
                unresolved=(
                    EvidenceQualification(
                        "“口嫌体正直”只是群友调侃，不能升级为人格事实。",
                        ("source-joke",),
                    ),
                ),
            ),
            source_keys=("source-buy", "source-dislike", "source-joke"),
            edge_ids=(9,),
            hypothesis_ids=(),
        )

    @staticmethod
    def packet() -> dict[str, object]:
        return {
            "request_identity_context": {
                "authority": "current_platform_event",
                "sender": {
                    "participant_key": 'participant:["bot","requester"]',
                    "platform_id": "bot",
                    "account_id": "requester",
                    "display_name": "提问者",
                    "binding_basis": "message_sender",
                    "same_account_as_sender": True,
                },
                "mentions": [],
                "reply_target": None,
            },
            "query_alias_resolution": {
                "query": "/chat 回忆甲前后态度",
                "ambiguous": False,
                "participants": [
                    {
                        "canonical_key": 'participant:["bot","account-a"]',
                        "platform_id": "bot",
                        "account_id": "account-a",
                        "current_display_name": "甲",
                        "matched_aliases": ["甲"],
                        "matched_alias_observations": [
                            {
                                "alias": "甲",
                                "source_key": "source-dislike",
                                "sent_at": 100,
                                "source_kind": "observed",
                            }
                        ],
                    }
                ],
                "ambiguous_aliases": [],
            },
            "semantic_evidence": [
                {
                    "memory": {
                        "person_cue": "甲",
                        "aspect_tag": "购买状态",
                        "content": "甲后来明确说已经买了。",
                        "epistemic_status": "ASSERTED",
                        "status": "ACTIVE",
                        "semantic_subject": {
                            "canonical_key": "participant:synthetic-semantic-subject",
                            "account_id": "synthetic-semantic-subject",
                        },
                    },
                    "evidence": [
                        {
                            "source_key": "source-buy",
                            "sent_at": 300,
                            "sender_id": "account-a",
                            "sender_name": "甲",
                            "role": "USER",
                            "plain_text": "我买了",
                            "evidence_role": "SUPPORT",
                        }
                    ],
                }
            ],
            "expanded_episodes": [
                {
                    "summary": "甲先说不喜欢，后来又买了。",
                    "messages": [
                        {
                            "source_key": "source-dislike",
                            "sent_at": 100,
                            "sender_id": "account-a",
                            "sender_name": "甲",
                            "role": "USER",
                            "plain_text": "我不喜欢",
                        },
                        {
                            "source_key": "source-joke",
                            "sent_at": 301,
                            "sender_id": "account-b",
                            "sender_name": "乙",
                            "role": "USER",
                            "plain_text": "口嫌体正直是吧（笑）",
                        },
                    ],
                }
            ],
            "candidates": {
                "associations": [
                    {
                        "id": 9,
                        "score": 0.72,
                        "source_label": "甲",
                        "relation_name": "态度发生变化",
                        "target_label": "作品",
                        "statement": "只可描述前后发言张力。",
                        "epistemic_state": "CONTESTED",
                        "epistemic_confidence": 0.61,
                        "source_keys": ["source-dislike", "source-buy"],
                    }
                ]
            },
            "participant_activity": [],
            "reply_context": None,
        }

    def test_source_backed_memory_and_graph_are_injected_as_candidates(self) -> None:
        result = compile_local_serving_envelope(
            self.packet(),
            self.materialized(),
            request_kind="MEMORY_QUERY",
            max_chars=3000,
        )

        self.assertTrue(result.usable)
        self.assertLessEqual(len(result.json_text), 3000)
        self.assertEqual(
            set(result.source_keys),
            {"source-buy", "source-dislike", "source-joke"},
        )
        value = json.loads(result.json_text)
        self.assertEqual(value["schema_version"], LOCAL_SERVING_SCHEMA_VERSION)
        self.assertEqual(value["retrieval"]["memory_provider_calls"], 0)
        reasoning = value["identity"]["person_reasoning_candidates"]
        self.assertEqual(
            reasoning["parser"]["references"],
            [],
        )
        self.assertEqual(
            reasoning["semantic"][0]["subject_candidate"],
            "甲",
        )
        self.assertEqual(
            reasoning["semantic"][0]["predicate"],
            "购买状态",
        )
        self.assertEqual(
            reasoning["semantic"][0]["epistemic_state"],
            "ASSERTED",
        )
        hypothesis_source_ids = reasoning["semantic"][0]["source_ids"]
        self.assertEqual(len(hypothesis_source_ids), 1)
        source_records = {item["id"]: item for item in value["source_records"]}
        self.assertEqual(source_records[hypothesis_source_ids[0]]["text"], "我买了")
        self.assertNotIn("participant_key", reasoning["semantic"][0])
        self.assertNotIn("synthetic-semantic-subject", reasoning["semantic"][0].values())
        self.assertIn("identity anchors", value["identity"]["rules"][0])
        self.assertIn("not identity verdicts", value["identity"]["rules"][1])
        self.assertIn("never a canonical merge", value["identity"]["rules"][2])

    def test_identity_coverage_and_participant_history_are_visible(self) -> None:
        packet = self.packet()
        packet["query_alias_resolution"]["mentions"] = [
            {"alias": "甲", "status": "RESOLVED", "participant_keys": ['participant:["bot","account-a"]']},
            {"alias": "幽灵", "status": "UNRESOLVED", "participant_keys": []},
        ]
        packet["participant_history"] = [
            {
                "participant": packet["query_alias_resolution"]["participants"][0],
                "status": "SOURCE_BACKED",
                "messages": [
                    {
                        "source_key": "source-dislike",
                        "sent_at": 100,
                        "sender_id": "account-a",
                        "sender_name": "甲",
                        "role": "USER",
                        "plain_text": "我不喜欢",
                    }
                ],
            }
        ]

        result = compile_local_serving_envelope(
            packet,
            self.materialized(),
            request_kind="CHAT",
            max_chars=4000,
        )
        value = json.loads(result.json_text)
        parser_evidence = value["identity"]["person_reasoning_candidates"][
            "parser"
        ]
        self.assertEqual(
            [item["parser_signal"] for item in parser_evidence["references"]],
            ["UNIQUE_ALIAS_CANDIDATE", "NO_ALIAS_CANDIDATE"],
        )
        self.assertEqual(
            parser_evidence["references"][0]["candidate_participant_keys"],
            ['participant:["bot","account-a"]'],
        )
        self.assertEqual(value["participant_history"][0]["status"], "SOURCE_BACKED")
        self.assertEqual(value["participant_history"][0]["messages"][0]["source_id"], "s1")

        unresolved = compile_local_serving_envelope(
            {
                "query_alias_resolution": {
                    "participants": [],
                    "ambiguous": False,
                    "ambiguous_aliases": [],
                    "mentions": [
                        {"alias": "幽灵甲", "status": "UNRESOLVED"},
                        {"alias": "幽灵乙", "status": "UNRESOLVED"},
                    ],
                },
                "participant_history": [],
                "participant_activity": [],
                "candidates": {},
            },
            MaterializedReconstruction(None, (), (), ()),
            request_kind="MEMORY_QUERY",
            max_chars=3000,
        )
        unresolved_value = json.loads(unresolved.json_text)
        self.assertFalse(unresolved.usable)
        self.assertEqual(unresolved.semantic_status, "NO_LOCAL_EVIDENCE")
        unresolved_references = unresolved_value["identity"][
            "person_reasoning_candidates"
        ]["parser"]["references"]
        self.assertEqual(
            [item["reference"] for item in unresolved_references],
            ["幽灵甲", "幽灵乙"],
        )
        self.assertEqual(
            {item["parser_signal"] for item in unresolved_references},
            {"NO_ALIAS_CANDIDATE"},
        )

    def test_history_budget_never_claims_that_existing_history_does_not_exist(self) -> None:
        packet = {
            "participant_history": [
                {
                    "participant_key": "participant-1",
                    "status": "SOURCE_BACKED",
                    "messages": [
                        {"source_key": f"history-{index}", "plain_text": f"消息{index}"}
                        for index in range(4)
                    ],
                }
            ]
        }
        compact = _participant_history(packet, message_limit=1, alias_limit=1)
        self.assertEqual(compact[0]["status"], "SOURCE_BACKED")
        self.assertEqual(compact[0]["source_count_total"], 4)
        self.assertTrue(compact[0]["messages_truncated"])

        omitted = _participant_history_alias_view(compact, {})
        self.assertEqual(omitted[0]["status"], "HISTORY_OMITTED_BY_BUDGET")
        self.assertEqual(omitted[0]["source_count_total"], 4)
        self.assertTrue(omitted[0]["messages_truncated"])

        no_history = _participant_history_alias_view(
            [
                {
                    "participant_key": "participant-2",
                    "status": "NO_HISTORY",
                    "messages": [],
                    "source_count_total": 0,
                    "messages_truncated": False,
                }
            ],
            {},
        )
        self.assertEqual(no_history[0]["status"], "NO_HISTORY")
        self.assertFalse(no_history[0]["messages_truncated"])

    def test_serving_profiles_cannot_truncate_or_rewrite_explicit_identity(self) -> None:
        participants = []
        mentions = []
        for index in range(12):
            key = f"participant-{index}"
            participants.append(
                {
                    "canonical_key": key,
                    "account_id": f"account-{index}",
                    "current_display_name": f"成员{index}",
                    "matched_aliases": [f"成员{index}"],
                    "matched_alias_observations": [
                        {
                            "alias": f"成员{index}",
                            "source_key": f"identity-{index}",
                            "sent_at": 100 + index,
                            "source_kind": "observed",
                        }
                    ],
                }
            )
            mentions.append(
                {
                    "alias": f"成员{index}",
                    "status": "RESOLVED",
                    "participant_keys": [key],
                }
            )
        resolution = {
            "participants": participants,
            "ambiguous": False,
            "ambiguous_aliases": [],
            "mentions": mentions,
        }
        compact = _query_identity(
            resolution,
            alias_limit=1,
            participant_limit=3,
            ambiguous_limit=2,
            candidates_per_alias=2,
        )
        self.assertEqual(len(compact["participants"]), 12)
        self.assertEqual(len(compact["mentions"]), 12)
        self.assertFalse(compact.get("participants_truncated", False))

        packet = {
            "query_alias_resolution": resolution,
            "participant_history": [],
            "participant_activity": [],
            "candidates": {},
        }
        try:
            envelope = compile_local_serving_envelope(
                packet,
                MaterializedReconstruction(None, (), (), ()),
                request_kind="MEMORY_QUERY",
                max_chars=3000,
            )
        except LocalServingEnvelopeError:
            return
        value = json.loads(envelope.json_text)
        visible = value["identity"]["person_reasoning_candidates"][
            "parser"
        ]
        self.assertEqual(len(visible["participants"]), 12)
        self.assertEqual(
            [item["parser_signal"] for item in visible["references"]],
            ["UNIQUE_ALIAS_CANDIDATE"] * 12,
        )
        self.assertTrue(
            all(item["candidate_participant_keys"] for item in visible["references"])
        )
        self.assertEqual(value["retrieval"]["memory_provider_tokens"], 0)
        self.assertEqual(
            value["retrieval"]["main_model_incremental_cost"],
            "UNKNOWN_NOT_MEASURED",
        )
        source_ids = {item["id"] for item in value["source_records"]}
        self.assertTrue(source_ids)
        self.assertTrue(
            all(
                source_id in source_ids
                for section in value["memory_brief"].values()
                for item in section
                for source_id in item["source_ids"]
            )
        )
        source_text = " ".join(
            str(item.get("text") or "") for item in value["source_records"]
        )
        self.assertIn("我买了", source_text)
        self.assertIn("我不喜欢", source_text)
        self.assertNotIn("source-buy", result.json_text)
        self.assertEqual(value["graph_connections"][0]["edge_id"], 9)

    def test_partial_source_budget_keeps_row_and_reports_truncation(self) -> None:
        source_keys = tuple(f"source-{index}" for index in range(10))
        packet = {
            "query_alias_resolution": {
                "ambiguous": False,
                "participants": [],
                "ambiguous_aliases": [],
            },
            "expanded_episodes": [
                {
                    "summary": "十条来源共同支持的候选",
                    "messages": [
                        {
                            "source_key": source_key,
                            "sent_at": index + 1,
                            "sender_id": "a",
                            "sender_name": "甲",
                            "role": "USER",
                            "plain_text": f"证据 {index}",
                        }
                        for index, source_key in enumerate(source_keys)
                    ],
                }
            ],
            "candidates": {},
            "participant_activity": [],
            "reply_context": None,
        }
        materialized = MaterializedReconstruction(
            EvidenceBrief(
                claims=(EvidenceClaim("候选结论", source_keys, 0.7),),
                conflicts=(),
                unresolved=(),
            ),
            source_keys,
            (),
            (),
        )

        result = compile_local_serving_envelope(
            packet,
            materialized,
            request_kind="MEMORY_QUERY",
            max_chars=16000,
        )

        value = json.loads(result.json_text)
        row = value["memory_brief"]["claims"][0]
        self.assertEqual(row["source_count_total"], 10)
        self.assertEqual(len(row["source_ids"]), 8)
        self.assertTrue(row["sources_truncated"])
        self.assertTrue(result.truncated)
        self.assertTrue(value["retrieval"]["truncated"])

    def test_missing_raw_source_never_creates_placeholder_evidence(self) -> None:
        packet = {
            "query_alias_resolution": {
                "ambiguous": False,
                "participants": [],
                "ambiguous_aliases": [],
            },
            "candidates": {},
            "participant_activity": [],
            "reply_context": None,
        }
        materialized = MaterializedReconstruction(
            EvidenceBrief(
                claims=(EvidenceClaim("不能无来源呈现", ("missing-source",), 0.7),),
                conflicts=(),
                unresolved=(),
            ),
            ("missing-source",),
            (),
            (),
        )

        result = compile_local_serving_envelope(
            packet,
            materialized,
            request_kind="MEMORY_QUERY",
            max_chars=3000,
        )

        value = json.loads(result.json_text)
        self.assertFalse(result.usable)
        self.assertEqual(value["source_records"], [])
        self.assertEqual(value["memory_brief"]["claims"], [])
        self.assertEqual(value["retrieval"]["unavailable_source_count"], 1)
        self.assertTrue(value["retrieval"]["truncated"])

    def test_tight_budget_preserves_conflict_and_unresolved_before_claims(self) -> None:
        rows = [
            {
                "source_key": f"s-{index}",
                "sent_at": index + 1,
                "sender_id": "a",
                "sender_name": "甲",
                "role": "USER",
                "plain_text": ("原始证据" + str(index)) * 40,
            }
            for index in range(8)
        ]
        packet = {
            "query_alias_resolution": {
                "ambiguous": False,
                "participants": [],
                "ambiguous_aliases": [],
            },
            "expanded_episodes": [{"summary": "压力样本", "messages": rows}],
            "candidates": {},
            "participant_activity": [],
            "reply_context": None,
        }
        materialized = MaterializedReconstruction(
            EvidenceBrief(
                claims=tuple(
                    EvidenceClaim(f"普通候选 {index} " + "甲" * 240, (f"s-{index}",), 0.7)
                    for index in range(6)
                ),
                conflicts=(
                    EvidenceQualification("存在明确冲突 " + "乙" * 240, ("s-6",)),
                ),
                unresolved=(
                    EvidenceQualification("仍然无法确定 " + "丙" * 240, ("s-7",)),
                ),
            ),
            tuple(f"s-{index}" for index in range(8)),
            (),
            (),
        )

        result = compile_local_serving_envelope(
            packet,
            materialized,
            request_kind="MEMORY_QUERY",
            max_chars=3000,
        )

        value = json.loads(result.json_text)
        self.assertTrue(value["memory_brief"]["conflicts"])
        self.assertTrue(value["memory_brief"]["unresolved"])
        self.assertTrue(result.truncated)

    def test_presented_feedback_pattern_keeps_exact_id_and_source(self) -> None:
        packet = {
            "query_alias_resolution": {
                "ambiguous": False,
                "participants": [],
                "ambiguous_aliases": [],
            },
            "feedback_hypothesis_evidence": [
                {
                    "hypothesis": {
                        "id": 42,
                        "statement": "提问代码时先给最小复现",
                        "prospective_cue": "涉及代码问题时先给最小复现",
                        "activation_mode": "semantic",
                        "trigger_cues": ["代码"],
                        "aspect": "response_style",
                        "evidence_confidence": 0.73,
                    },
                    "evidence": [
                        {
                            "source_key": "feedback-source",
                            "sent_at": 10,
                            "sender_id": "u1",
                            "sender_name": "甲",
                            "role": "USER",
                            "plain_text": "代码问题先给个最小复现会更好",
                        }
                    ],
                }
            ],
            "candidates": {},
            "participant_activity": [],
            "reply_context": None,
        }
        materialized = materialize_reconstruction_packet(
            packet,
            query="这个代码为什么报错",
        )

        result = compile_local_serving_envelope(
            packet,
            materialized,
            request_kind="CHAT",
            max_chars=3000,
        )

        value = json.loads(result.json_text)
        self.assertEqual(result.hypothesis_ids, (42,))
        self.assertEqual(value["learned_patterns"][0]["hypothesis_id"], 42)
        self.assertEqual(value["learned_patterns"][0]["source_ids"], ["s1"])
        self.assertEqual(value["source_records"][0]["id"], "s1")

    def test_ambiguous_alias_keeps_accounts_separate(self) -> None:
        packet = self.packet()
        packet["query_alias_resolution"] = {
            "query": "/chat 合成昵称-X 是谁",
            "ambiguous": True,
            "participants": [],
            "ambiguous_aliases": [
                {
                    "alias": "合成昵称-X",
                    "candidate_participants": [
                        {
                            "canonical_key": 'participant:["bot","100"]',
                            "account_id": "100",
                            "current_display_name": "合成成员-X1",
                            "matched_aliases": ["合成昵称-X"],
                            "matched_alias_observations": [
                                {
                                    "alias": "合成昵称-X",
                                    "source_key": "source-dislike",
                                    "sent_at": 100,
                                    "source_kind": "observed",
                                }
                            ],
                        },
                        {
                            "canonical_key": 'participant:["bot","200"]',
                            "account_id": "200",
                            "current_display_name": "合成成员-X2",
                            "matched_aliases": ["合成昵称-X"],
                            "matched_alias_observations": [
                                {
                                    "alias": "合成昵称-X",
                                    "source_key": "source-joke",
                                    "sent_at": 301,
                                    "source_kind": "observed",
                                }
                            ],
                        },
                    ],
                }
            ],
        }
        empty = MaterializedReconstruction(None, (), (), ())

        result = compile_local_serving_envelope(
            packet,
            empty,
            request_kind="MEMORY_QUERY",
            max_chars=3000,
        )

        self.assertTrue(result.usable)
        self.assertEqual(result.semantic_status, "EVIDENCE_AVAILABLE")
        value = json.loads(result.json_text)
        candidates = value["identity"]["person_reasoning_candidates"][
            "parser"
        ]["ambiguous_references"][0]["candidates"]
        self.assertEqual([item["account_id"] for item in candidates], ["100", "200"])
        self.assertNotEqual(candidates[0]["canonical_key"], candidates[1]["canonical_key"])
        candidate_source_ids = {
            candidate["alias_observations"][0]["source_id"]
            for candidate in candidates
        }
        self.assertEqual(len(candidate_source_ids), 2)
        self.assertTrue(
            candidate_source_ids.issubset(
                {item["id"] for item in value["source_records"]}
            )
        )

    def test_unbacked_ambiguous_alias_is_not_injectable_identity_evidence(self) -> None:
        packet = {
            "query_alias_resolution": {
                "query": "/chat 合成昵称-Y 是谁",
                "ambiguous": True,
                "participants": [],
                "ambiguous_aliases": [
                    {
                        "alias": "合成昵称-Y",
                        "candidate_participants": [
                            {
                                "canonical_key": 'participant:["bot","100"]',
                                "account_id": "100",
                                "current_display_name": "合成成员-Y1",
                            },
                            {
                                "canonical_key": 'participant:["bot","200"]',
                                "account_id": "200",
                                "current_display_name": "合成成员-Y2",
                            },
                        ],
                    }
                ],
            },
            "participant_activity": [],
            "reply_context": None,
            "candidates": {},
        }

        result = compile_local_serving_envelope(
            packet,
            MaterializedReconstruction(None, (), (), ()),
            request_kind="MEMORY_QUERY",
            max_chars=3000,
        )

        self.assertFalse(result.usable)
        self.assertEqual(result.semantic_status, "NO_LOCAL_EVIDENCE")
        value = json.loads(result.json_text)
        self.assertEqual(
            value["identity"]["person_reasoning_candidates"]["parser"][
                "ambiguous_references"
            ],
            [],
        )


    def test_empty_local_packet_is_not_reported_as_semantic_absence(self) -> None:
        packet = {
            "request_identity_context": {
                "sender": {
                    "participant_key": 'participant:["bot","requester"]',
                    "account_id": "requester",
                }
            },
            "query_alias_resolution": {
                "ambiguous": False,
                "participants": [],
                "ambiguous_aliases": [],
            },
            "participant_activity": [],
            "reply_context": None,
            "candidates": {},
        }

        result = compile_local_serving_envelope(
            packet,
            MaterializedReconstruction(None, (), (), ()),
            request_kind="CHAT",
            max_chars=3000,
        )

        self.assertFalse(result.usable)
        self.assertEqual(result.semantic_status, "NO_LOCAL_EVIDENCE")
        self.assertNotEqual(result.semantic_status, "SEMANTIC_NONE")
        self.assertEqual(result.source_keys, ())

    def test_trace_identifiers_are_exactly_the_items_in_the_compact_envelope(
        self,
    ) -> None:
        packet = self.packet()
        packet["feedback_hypothesis_evidence"] = [
            {
                "hypothesis": {
                    "id": 77,
                    "prospective_cue": "回答时保留不确定性。",
                    "aspect": "response_style",
                    "activation_mode": "always",
                    "trigger_cues": [],
                    "evidence_confidence": 0.8,
                },
                "evidence": [
                    {
                        "source_key": "source-joke",
                        "sent_at": 301,
                        "sender_id": "account-b",
                        "sender_name": "乙",
                        "role": "USER",
                        "plain_text": "不要把群友调侃写成确定事实",
                    }
                ],
            }
        ]
        base = self.materialized()
        materialized = MaterializedReconstruction(
            brief=base.brief,
            source_keys=base.source_keys,
            edge_ids=(9, 404),
            hypothesis_ids=(77, 808),
        )

        result = compile_local_serving_envelope(
            packet,
            materialized,
            request_kind="MEMORY_QUERY",
            max_chars=3000,
        )

        value = json.loads(result.json_text)
        visible_edge_ids = tuple(
            item["edge_id"] for item in value["graph_connections"]
        )
        visible_hypothesis_ids = tuple(
            item["hypothesis_id"] for item in value["learned_patterns"]
        )
        self.assertEqual(result.edge_ids, visible_edge_ids)
        self.assertEqual(result.hypothesis_ids, visible_hypothesis_ids)
        self.assertNotIn(404, result.edge_ids)
        self.assertNotIn(808, result.hypothesis_ids)


if __name__ == "__main__":
    unittest.main()
