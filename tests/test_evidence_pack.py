from __future__ import annotations

from copy import deepcopy
import json
import unittest

from mr_memory.evidence_pack import (
    EVIDENCE_ATOM_PACK_FORMAT,
    compile_evidence_atom_pack,
    hydrate_evidence_atom_pack,
    participant_speaker_source_bindings,
    participant_source_bindings,
)


def _message(source_key: str, text: str, sent_at: int) -> dict[str, object]:
    return {
        "source_key": source_key,
        "sent_at": sent_at,
        "sender_id": f"account-{source_key}",
        "sender_name": f"member-{source_key}",
        "role": "user",
        "plain_text": text,
        "components": [{"type": "plain", "text": text}],
    }


def _all_source_references(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "source_key" or (
                key != "request_source_key" and key.endswith("_source_key")
            ):
                if str(item or "").strip():
                    found.add(str(item))
            elif (key == "source_keys" or key.endswith("_source_keys")) and isinstance(
                item, list
            ):
                found.update(str(source) for source in item if str(source))
            else:
                found.update(_all_source_references(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_all_source_references(item))
    return found


class EvidenceAtomPackTests(unittest.TestCase):
    def test_hydration_only_overlays_selected_sources_and_rebuilds_speakers(
        self,
    ) -> None:
        pack = {
            "format": EVIDENCE_ATOM_PACK_FORMAT,
            "sources": [
                {
                    "source_key": "selected-one",
                    "sender_id": "stale-account",
                    "sender_name": "stale name",
                    "sender_participant_key": "stale-participant",
                    "plain_text": "stale raw payload",
                    "components": [{"type": "plain", "text": "stale raw payload"}],
                    "component_types": ["stale-component"],
                    "message_row_id": 999,
                    "source_message_id": 998,
                    "payload_status": "FULL",
                    "origins": ["lexical"],
                },
                {
                    "source_key": "selected-two",
                    "sender_id": "retained-account",
                    "sender_name": "retained name",
                    "sender_participant_key": "retained-participant",
                    "plain_text": "retained raw payload",
                    "payload_status": "FULL",
                    "origins": ["recent"],
                },
            ],
            "retrieval_coverage": {"upstream_marker": True},
            "participant_source_keys": {
                "stale-participant": ["selected-one"]
            },
            "participant_speaker_source_keys": {
                "stale-participant": ["selected-one"]
            },
        }
        original = deepcopy(pack)

        hydrated = hydrate_evidence_atom_pack(
            pack,
            [
                {
                    "source_key": "selected-one",
                    "sent_at": 101,
                    "sender_id": "authoritative-account",
                    "sender_name": "authoritative name",
                    "sender_participant_key": "authoritative-participant",
                    "role": "user",
                    "plain_text": "authoritative raw payload",
                    "components": [
                        {"type": "plain", "text": "authoritative raw payload"}
                    ],
                    "revision_no": 3,
                },
                {
                    "source_key": "selected-two",
                    "sent_at": 102,
                    "sender_id": "authoritative-second-account",
                    "sender_name": "authoritative second name",
                    "sender_participant_key": "authoritative-second-participant",
                    "role": "user",
                    "plain_text": "authoritative second raw payload",
                    "components": [
                        {
                            "type": "plain",
                            "text": "authoritative second raw payload",
                        }
                    ],
                    "revision_no": 2,
                },
            ],
        )

        self.assertEqual(pack, original)
        self.assertEqual(
            [source["source_key"] for source in hydrated["sources"]],
            ["selected-one", "selected-two"],
        )
        selected_one, selected_two = hydrated["sources"]
        self.assertEqual(selected_one["plain_text"], "authoritative raw payload")
        self.assertEqual(selected_one["sender_id"], "authoritative-account")
        self.assertEqual(selected_one["sender_name"], "authoritative name")
        self.assertEqual(
            selected_one["sender_participant_key"],
            "authoritative-participant",
        )
        self.assertEqual(selected_one["revision_no"], 3)
        self.assertNotIn("component_types", selected_one)
        self.assertNotIn("message_row_id", selected_one)
        self.assertNotIn("source_message_id", selected_one)
        self.assertTrue(selected_one["snapshot_hydrated"])
        self.assertEqual(
            selected_two["plain_text"],
            "authoritative second raw payload",
        )
        self.assertEqual(
            selected_two["sender_participant_key"],
            "authoritative-second-participant",
        )
        self.assertTrue(selected_two["snapshot_hydrated"])
        self.assertEqual(
            hydrated["participant_speaker_source_keys"],
            {
                "authoritative-participant": ["selected-one"],
                "authoritative-second-participant": ["selected-two"],
            },
        )
        self.assertNotIn(
            "stale-participant",
            hydrated["participant_speaker_source_keys"],
        )
        self.assertEqual(
            hydrated["retrieval_coverage"],
            {
                "upstream_marker": True,
                "source_hydration_requested": 2,
                "source_hydration_returned": 2,
                "source_hydration_missing": 0,
            },
        )

    def test_hydration_fails_closed_for_missing_or_duplicate_selected_source(
        self,
    ) -> None:
        pack = {
            "format": EVIDENCE_ATOM_PACK_FORMAT,
            "sources": [
                {"source_key": "selected-one"},
                {"source_key": "selected-two"},
            ],
        }
        selected_one = {
            "source_key": "selected-one",
            "sender_participant_key": "participant-one",
            "plain_text": "authoritative one",
        }

        with self.subTest("missing"):
            with self.assertRaisesRegex(ValueError, "hydration is incomplete"):
                hydrate_evidence_atom_pack(pack, [selected_one])
        with self.subTest("duplicate"):
            with self.assertRaisesRegex(ValueError, "duplicate source"):
                hydrate_evidence_atom_pack(pack, [selected_one, dict(selected_one)])
        with self.subTest("extra"):
            with self.assertRaisesRegex(ValueError, "unselected source"):
                hydrate_evidence_atom_pack(
                    pack,
                    [
                        selected_one,
                        {
                            "source_key": "not-selected",
                            "plain_text": "must be rejected",
                        },
                    ],
                )
        with self.subTest("blank"):
            with self.assertRaisesRegex(ValueError, "blank source key"):
                hydrate_evidence_atom_pack(pack, [{"source_key": ""}])
        with self.subTest("non-object"):
            with self.assertRaisesRegex(ValueError, "must be objects"):
                hydrate_evidence_atom_pack(pack, [selected_one, "invalid"])

    def test_compiler_always_emits_source_scoped_participant_bindings(self) -> None:
        packet = {
            "person_reference_candidates": {
                "references": [
                    {
                        "candidate_participants": [
                            {
                                "participant_key": "synthetic-participant-a",
                                "alias_observations": [
                                    _message("synthetic-source-a", "alias a", 1)
                                ],
                            },
                            {
                                "participant_key": "synthetic-participant-b",
                                "alias_observations": [
                                    _message("synthetic-source-b", "alias b", 2)
                                ],
                            },
                        ]
                    }
                ]
            }
        }

        compact = compile_evidence_atom_pack(packet, max_sources=4)

        self.assertEqual(
            compact["participant_source_keys"],
            {
                "synthetic-participant-a": ["synthetic-source-a"],
                "synthetic-participant-b": ["synthetic-source-b"],
            },
        )

    def test_participant_binding_helper_does_not_cross_candidate_siblings(
        self,
    ) -> None:
        bindings = participant_source_bindings(
            {
                "references": [
                    {
                        "candidate_participants": [
                            {
                                "participant_key": "synthetic-participant-a",
                                "alias_observations": [
                                    {"source_key": "synthetic-source-a"}
                                ],
                            },
                            {
                                "participant_key": "synthetic-participant-b",
                                "alias_observations": [
                                    {"source_key": "synthetic-source-b"}
                                ],
                            },
                        ]
                    }
                ]
            }
        )

        self.assertEqual(
            bindings,
            {
                "synthetic-participant-a": ["synthetic-source-a"],
                "synthetic-participant-b": ["synthetic-source-b"],
            },
        )

    def test_subject_reference_does_not_become_identity_or_speaker_evidence(
        self,
    ) -> None:
        bindings = participant_source_bindings(
            {
                "source_key": "speaker-source",
                "sender_participant_key": "speaker-participant",
                "subject_participant_key": "mentioned-participant",
            }
        )

        self.assertEqual(
            bindings,
            {"speaker-participant": ["speaker-source"]},
        )

    def test_structured_mention_never_becomes_direct_speaker_evidence(self) -> None:
        packet = {
            "source_key": "synthetic-source",
            "sender_participant_key": "synthetic-speaker",
            "mentions": [
                {"participant_key": "synthetic-mentioned-participant"}
            ],
        }

        self.assertEqual(
            participant_source_bindings(packet),
            {"synthetic-speaker": ["synthetic-source"]},
        )
        self.assertEqual(
            participant_speaker_source_bindings(packet),
            {"synthetic-speaker": ["synthetic-source"]},
        )

    def test_semantic_subject_source_never_becomes_identity_or_speaker_evidence(
        self,
    ) -> None:
        packet = {
            "semantic_evidence": [
                {
                    "participant_key": "synthetic-subject",
                    "candidate_basis": "semantic_subject_binding",
                    "source_keys": ["third-party-source"],
                }
            ],
            "sources": [
                {
                    "source_key": "third-party-source",
                    "sender_participant_key": "synthetic-speaker",
                }
            ],
        }

        self.assertEqual(
            participant_source_bindings(packet),
            {"synthetic-speaker": ["third-party-source"]},
        )
        self.assertEqual(
            participant_speaker_source_bindings(packet),
            {"synthetic-speaker": ["third-party-source"]},
        )

    def test_participant_message_relation_cannot_contradict_catalog_sender(
        self,
    ) -> None:
        packet = {
            "sources": [
                {
                    "source_key": "synthetic-source",
                    "sender_participant_key": "synthetic-speaker",
                }
            ],
            "participant_history": [
                {
                    "participant_key": "different-participant",
                    "messages": [{"source_key": "synthetic-source"}],
                }
            ],
        }

        with self.assertRaisesRegex(ValueError, "authoritative message sender"):
            participant_source_bindings(packet)

    def test_third_party_mention_can_support_identity_but_not_speaker(self) -> None:
        packet = {
            "sources": [
                {
                    "source_key": "synthetic-mention-source",
                    "sender_participant_key": "synthetic-speaker",
                }
            ],
            "person_reference_candidates": {
                "references": [
                    {
                        "candidate_participants": [
                            {
                                "participant_key": "synthetic-mentioned",
                                "alias_observations": [
                                    {
                                        "source_key": "synthetic-mention-source",
                                        "relation": "MENTIONED",
                                    }
                                ],
                            }
                        ]
                    }
                ]
            },
        }

        self.assertEqual(
            participant_source_bindings(packet),
            {
                "synthetic-mentioned": ["synthetic-mention-source"],
                "synthetic-speaker": ["synthetic-mention-source"],
            },
        )
        self.assertEqual(
            participant_speaker_source_bindings(packet),
            {"synthetic-speaker": ["synthetic-mention-source"]},
        )

    def test_compiler_rebuilds_stale_participant_bindings_from_retained_sources(
        self,
    ) -> None:
        packet = {
            "participant_source_keys": {
                "synthetic-participant-a": ["stale-dropped-source"]
            },
            "participant_history": [
                {
                    "participant_key": "synthetic-participant-a",
                    "messages": [
                        _message("retained-source", "retained observation", 1)
                    ],
                }
            ],
        }

        compact = compile_evidence_atom_pack(packet, max_sources=1)

        self.assertEqual(
            compact["participant_source_keys"],
            {"synthetic-participant-a": ["retained-source"]},
        )
        self.assertEqual(compact["participant_speaker_source_keys"], {})
        self.assertNotIn("stale-dropped-source", json.dumps(compact))

    def test_one_raw_payload_is_shared_by_all_views_without_mutating_input(self) -> None:
        shared = _message("shared-source", "unique raw payload marker", 10)
        packet = {
            "host_notice": "synthetic evidence only",
            "expanded_episodes": [
                {"id": 1, "summary": "episode", "messages": [dict(shared)]}
            ],
            "participant_history": [
                {
                    "participant_key": "participant-one",
                    "messages": [dict(shared)],
                }
            ],
            "recent_context": [dict(shared)],
            "retrieval_coverage": {"semantic_none_allowed": True},
        }
        original = deepcopy(packet)

        compact = compile_evidence_atom_pack(packet, max_sources=8)

        self.assertEqual(packet, original)
        self.assertEqual(compact["source_count"], 1)
        self.assertEqual(
            json.dumps(compact, ensure_ascii=False).count("unique raw payload marker"),
            1,
        )
        source = compact["sources"][0]
        self.assertEqual(source["source_key"], "shared-source")
        self.assertNotIn("source_id", source)
        self.assertIn("episode", source["origins"])
        self.assertIn("history", source["origins"])
        self.assertIn("recent", source["origins"])
        self.assertNotIn("plain_text", compact["recent_context"][0])
        self.assertNotIn(
            "plain_text",
            compact["participant_history"][0]["messages"][0],
        )

    def test_strict_cap_keeps_reply_person_and_semantic_minimum(self) -> None:
        reply = _message("reply-source", "quoted anchor", 1)
        alias_one = _message("alias-one-source", "first observation", 2)
        alias_two = _message("alias-two-source", "second observation", 3)
        packet = {
            "reply_context": reply,
            "person_reference_candidates": {
                "host_decision": "NONE",
                "references": [
                    {
                        "reference": "synthetic-reference",
                        "candidate_participants": [
                            {
                                "participant_key": "candidate-one",
                                "alias_observations": [
                                    {
                                        **alias_one,
                                        "alias": "synthetic-reference",
                                        "relation": "SPEAKER",
                                    }
                                ],
                            },
                            {
                                "participant_key": "candidate-two",
                                "alias_observations": [
                                    {
                                        **alias_two,
                                        "alias": "synthetic-reference",
                                        "relation": "SPEAKER",
                                    }
                                ],
                            },
                        ],
                    }
                ],
            },
            "semantic_evidence": [
                {
                    "memory": {"id": 1, "content": "derived claim"},
                    "evidence": [_message("semantic-source", "support", 4)],
                }
            ],
            "retrieval_coverage": {"semantic_none_allowed": True},
        }

        compact = compile_evidence_atom_pack(packet, max_sources=3)

        selected = [source["source_key"] for source in compact["sources"]]
        self.assertEqual(
            selected,
            ["reply-source", "alias-one-source", "semantic-source"],
        )
        self.assertEqual(len(selected), 3)
        references = compact["person_reference_candidates"]["references"]
        self.assertEqual(len(references[0]["candidate_participants"]), 1)
        self.assertEqual(len(compact["semantic_evidence"]), 1)
        self.assertTrue(compact["retrieval_coverage"]["truncated"])
        self.assertEqual(compact["retrieval_coverage"]["dropped"], 1)
        self.assertFalse(compact["retrieval_coverage"]["semantic_none_allowed"])
        self.assertFalse(
            compact["retrieval_coverage"]["person_candidates_complete"]
        )

    def test_derived_views_never_reference_a_dropped_source(self) -> None:
        packet = {
            "reply_context": _message("reply-source", "reply", 1),
            "semantic_evidence": [
                {
                    "memory": {"id": 1, "content": "kept claim"},
                    "evidence": [_message("semantic-source", "support", 2)],
                }
            ],
            "feedback_hypothesis_evidence": [
                {
                    "hypothesis": {"id": 2, "statement": "dropped hypothesis"},
                    "evidence": [_message("feedback-source", "feedback", 3)],
                }
            ],
            "expanded_episodes": [
                {
                    "id": 3,
                    "messages": [
                        {
                            **_message("episode-source", "episode", 4),
                            "reply_to_source_key": "unselected-reply-target",
                        }
                    ],
                }
            ],
        }

        compact = compile_evidence_atom_pack(packet, max_sources=2)
        selected = {source["source_key"] for source in compact["sources"]}
        derived = {
            key: value
            for key, value in compact.items()
            if key not in {"sources", "retrieval_coverage"}
        }

        self.assertEqual(selected, {"reply-source", "semantic-source"})
        self.assertTrue(_all_source_references(derived).issubset(selected))
        self.assertEqual(len(compact["semantic_evidence"]), 1)
        self.assertEqual(compact["feedback_hypothesis_evidence"], [])
        self.assertEqual(compact["expanded_episodes"], [])

    def test_activity_mode_moves_activity_ahead_of_lexical_and_episode_sources(
        self,
    ) -> None:
        packet = {
            "participant_activity": [
                {
                    "participant_key": "participant-one",
                    "found": True,
                    "messages": [_message("activity-source", "activity", 3)],
                }
            ],
            "lexical_messages": [_message("lexical-source", "lexical", 2)],
            "expanded_episodes": [
                {"id": 1, "messages": [_message("episode-source", "episode", 1)]}
            ],
        }

        activity_pack = compile_evidence_atom_pack(
            packet,
            max_sources=1,
            activity_mode=True,
        )
        ordinary_pack = compile_evidence_atom_pack(
            packet,
            max_sources=1,
            activity_mode=False,
        )

        self.assertEqual(activity_pack["sources"][0]["source_key"], "activity-source")
        self.assertEqual(ordinary_pack["sources"][0]["source_key"], "lexical-source")

    def test_many_alias_candidates_do_not_starve_question_evidence(self) -> None:
        candidates = [
            {
                "participant_key": f"candidate-{index}",
                "alias_observations": [
                    _message(f"alias-{index}", f"alias observation {index}", index)
                ],
            }
            for index in range(6)
        ]
        packet = {
            "person_reference_candidates": {
                "references": [
                    {
                        "reference": "common synthetic alias",
                        "candidate_participants": candidates,
                    }
                ]
            },
            "semantic_evidence": [
                {
                    "memory": {"id": 1},
                    "evidence": [_message("semantic-source", "semantic", 20)],
                }
            ],
            "lexical_messages": [_message("lexical-source", "lexical", 21)],
        }

        compact = compile_evidence_atom_pack(packet, max_sources=5)
        selected = {source["source_key"] for source in compact["sources"]}

        self.assertIn("semantic-source", selected)
        self.assertIn("lexical-source", selected)
        coverage = compact["retrieval_coverage"]
        self.assertEqual(coverage["person_candidate_groups_total"], 6)
        self.assertLess(coverage["person_candidate_groups_covered"], 6)
        self.assertFalse(coverage["person_candidates_complete"])

    def test_alias_and_lexical_volume_do_not_starve_episode_or_recent(self) -> None:
        candidates = [
            {
                "participant_key": f"candidate-{index}",
                "alias_observations": [
                    _message(f"alias-{index}", f"alias {index}", index)
                ],
            }
            for index in range(5)
        ]
        packet = {
            "person_reference_candidates": {
                "references": [
                    {
                        "reference": "synthetic alias",
                        "candidate_participants": candidates,
                    }
                ]
            },
            "lexical_messages": [
                _message("lexical-one", "lexical one", 10),
                _message("lexical-two", "lexical two", 11),
            ],
            "expanded_episodes": [
                {
                    "id": 1,
                    "messages": [_message("episode-source", "episode", 12)],
                }
            ],
            "recent_context": [_message("recent-source", "recent", 13)],
            "retrieval_coverage": {"semantic_none_allowed": True},
        }

        compact = compile_evidence_atom_pack(packet, max_sources=4)
        selected = {source["source_key"] for source in compact["sources"]}

        self.assertEqual(len(selected), 4)
        expected_layers = {
            "alias-0",
            "lexical-one",
            "episode-source",
            "recent-source",
        }
        self.assertTrue(expected_layers.issubset(selected))
        coverage = compact["retrieval_coverage"]
        self.assertTrue(coverage["truncated"])
        self.assertFalse(coverage["semantic_none_allowed"])
        self.assertFalse(coverage["person_candidates_complete"])
        self.assertEqual(coverage["strata_selected_sources"]["episode"], 1)
        self.assertEqual(coverage["strata_selected_sources"]["recent"], 1)
        self.assertGreater(
            coverage["strata_available_sources"]["person"],
            coverage["strata_selected_sources"]["person"],
        )

    def test_shared_source_wins_by_covering_two_uncovered_layers(self) -> None:
        shared = _message("shared-context", "shared episode and recent", 20)
        packet = {
            "semantic_evidence": [
                {
                    "memory": {"id": 1},
                    "evidence": [_message("semantic-source", "semantic", 18)],
                }
            ],
            "lexical_messages": [_message("lexical-source", "lexical", 19)],
            "expanded_episodes": [{"id": 2, "messages": [dict(shared)]}],
            "recent_context": [dict(shared)],
            "retrieval_coverage": {"semantic_none_allowed": True},
        }

        first = compile_evidence_atom_pack(packet, max_sources=2)
        second = compile_evidence_atom_pack(packet, max_sources=2)

        self.assertEqual(first, second)
        self.assertEqual(
            [source["source_key"] for source in first["sources"]],
            ["semantic-source", "shared-context"],
        )
        shared_source = next(
            source
            for source in first["sources"]
            if source["source_key"] == "shared-context"
        )
        self.assertIn("episode", shared_source["origins"])
        self.assertIn("recent", shared_source["origins"])
        selected = {source["source_key"] for source in first["sources"]}
        self.assertNotIn("lexical-source", selected)
        self.assertTrue(first["retrieval_coverage"]["truncated"])
        self.assertFalse(first["retrieval_coverage"]["semantic_none_allowed"])

    def test_projected_activity_recomputes_source_backed_statistics(self) -> None:
        packet = {
            "participant_activity": [
                {
                    "participant_key": "participant-one",
                    "found": True,
                    "message_count": 3,
                    "statistics_basis": "returned_source_messages_only",
                    "sampling_method": "synthetic-sampling",
                    "hour_histogram": {"01": 1, "02": 2},
                    "messages": [
                        {**_message("activity-one", "one", 1), "local_hour": 1},
                        {**_message("activity-two", "two", 2), "local_hour": 2},
                        {**_message("activity-three", "three", 3), "local_hour": 2},
                    ],
                    "messages_truncated": False,
                }
            ],
            "lexical_messages": [_message("lexical", "lexical", 4)],
        }

        compact = compile_evidence_atom_pack(
            packet,
            max_sources=2,
            activity_mode=True,
        )

        activity = compact["participant_activity"][0]
        self.assertEqual(activity["message_count"], 1)
        self.assertEqual(sum(activity["hour_histogram"].values()), 1)
        self.assertTrue(activity["messages_truncated"])
        self.assertEqual(
            activity["statistics_basis"],
            "atom_pack_selected_source_messages_only",
        )

    def test_projected_person_candidate_counts_match_retained_array(self) -> None:
        packet = {
            "person_reference_candidates": {
                "references": [
                    {
                        "reference": "synthetic alias",
                        "candidate_count_total": 3,
                        "candidate_count_returned": 3,
                        "truncated": False,
                        "candidate_participants": [
                            {
                                "participant_key": f"candidate-{index}",
                                "source_count_total": 1,
                                "alias_observations": [
                                    _message(f"alias-{index}", "alias", index)
                                ],
                            }
                            for index in range(3)
                        ],
                    }
                ],
                "coverage": {
                    "matched_reference_count_total": 1,
                    "matched_reference_count_returned": 1,
                    "candidate_count_total": 3,
                    "candidate_count_returned": 3,
                    "distinct_candidate_count_total": 3,
                    "distinct_candidate_count_returned": 3,
                    "truncated": False,
                },
            },
            "semantic_evidence": [
                {
                    "memory": {"id": 1},
                    "evidence": [_message("semantic", "semantic", 10)],
                }
            ],
        }

        compact = compile_evidence_atom_pack(packet, max_sources=2)

        person = compact["person_reference_candidates"]
        reference = person["references"][0]
        self.assertEqual(reference["candidate_count_total"], 3)
        self.assertEqual(reference["candidate_count_returned"], 1)
        self.assertTrue(reference["truncated"])
        self.assertEqual(person["coverage"]["candidate_count_returned"], 1)
        self.assertTrue(person["coverage"]["truncated"])

    def test_conflicting_payload_for_selected_source_fails_closed(self) -> None:
        packet = {
            "lexical_messages": [_message("shared", "first payload", 10)],
            "recent_context": [_message("shared", "conflicting payload", 10)],
        }

        with self.assertRaisesRegex(ValueError, "conflicting payloads"):
            compile_evidence_atom_pack(packet, max_sources=4)


if __name__ == "__main__":
    unittest.main()
