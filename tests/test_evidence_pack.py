from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
import json
import unittest

from mr_memory.activity_statistics import activity_aggregate_id
from mr_memory.evidence_pack import (
    EVIDENCE_ATOM_PACK_FORMAT,
    compile_evidence_atom_pack,
    hydrate_evidence_atom_pack,
    participant_speaker_source_bindings,
    participant_source_bindings,
)
from mr_memory.identity import canonical_participant_key


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
    def test_stored_derivation_survives_raw_cap_without_claiming_raw_sources(self) -> None:
        descriptor = {
            "derivation_id": "stored:" + "d" * 64,
            "kind": "semantic", "owner_id": "13",
            "text": {"content": "合成项目经历了三次独立复核。"},
            "dependency_keys": ["synthetic-check-1", "synthetic-check-2", "synthetic-check-3"],
            "source_fingerprints": {
                f"synthetic-check-{index}": {"message_id": index}
                for index in range(1, 4)
            },
        }
        compact = compile_evidence_atom_pack({
            "lexical_messages": [_message("synthetic-current", "当前问题原句", 9)],
            "stored_derivations": [descriptor],
            "candidates": {"semantic_memories": [{
                "id": 13, "content": descriptor["text"]["content"],
                "source_keys": ["synthetic-current"],
            }]},
        }, max_sources=1)
        self.assertEqual(compact["stored_derivations"], [descriptor])
        self.assertEqual([item["source_key"] for item in compact["sources"]],
                         ["synthetic-current"])
        self.assertEqual(_all_source_references(compact), {"synthetic-current"})
        self.assertNotIn("content", compact["candidates"]["semantic_memories"][0])
        self.assertEqual(compact["candidates"]["semantic_memories"][0]["stored_derivation_id"],
                         descriptor["derivation_id"])
        hydrated = hydrate_evidence_atom_pack(compact, [
            _message("synthetic-current", "当前问题原句", 9),
        ])
        self.assertEqual(hydrated["stored_derivations"], [descriptor])

    def test_truncated_derived_sources_withhold_all_duplicate_assertion_views(self) -> None:
        summary = "合成整理员完成了复核和归档。"
        anchor = _message("review-question", "有人问：“合成整理员完成了复核和归档。”是否属实？", 1)
        evidence = [anchor, _message("review-confirmation", "我刚完成复核。", 2),
                    _message("archive-confirmation", "归档也已完成。", 3)]
        memory = {"id": 1, "content": summary, "aspect_tag": "review_and_archive_complete",
                  "epistemic_status": "ASSERTED", "confidence": 0.8,
                  "source_keys": ["review-question"]}
        complete = {"id": 2, "content": "有人提出了复核问题。", "source_keys": ["review-question"]}
        aggregate = {
            "schema_version": "mr-memory.activity-window.v1", "authority": "HOST_SQLITE_SNAPSHOT",
            "basis": "all_snapshot_visible_direct_speaker_messages",
            "scope": {"umo": "synthetic:GroupMessage:closure", "participant_key": "participant-one",
                      "start_sent_at": 10, "end_sent_at_exclusive": 20, "message_upper_bound": 3},
            "timezone": "Asia/Shanghai", "source_count": 0,
            "hour_histogram": {f"{hour:02d}": 0 for hour in range(24)}, "daily": [],
            "source_revision_sha256": "a" * 64,
        }
        aggregate["aggregate_id"] = activity_aggregate_id(aggregate)
        packet = {
            "reply_context": [_message("reply", "当前回复锚点", 4)],
            "semantic_evidence": [{"memory": deepcopy(memory), "evidence": evidence},
                                  {"memory": deepcopy(complete), "evidence": [anchor]}],
            "candidates": {"semantic_memories": [deepcopy(memory), deepcopy(complete)]},
            "participant_activity": [{"participant_key": "participant-one", "messages": [],
                                      "window_statistics": aggregate}],
        }
        original = deepcopy(packet)
        compact = compile_evidence_atom_pack(packet, max_sources=2)
        self.assertEqual({source["source_key"] for source in compact["sources"]}, {"reply", "review-question"})
        views = [compact["candidates"]["semantic_memories"][0], compact["semantic_evidence"][0]["memory"]]
        for view in views:
            self.assertNotIn("content", view)
            self.assertNotIn("aspect_tag", view)
            self.assertEqual(view["source_closure"], {
                "basis": "input_packet_evidence_sources_only", "available_source_count": 3,
                "selected_source_count": 1, "missing_source_count": 2, "truncated": True,
                "state": "TRUNCATED_REQUIRES_SOURCE_RETRIEVAL",
            })
            self.assertEqual({item["reason"] for item in view["reader_withheld_fields"]}, {"SOURCE_CLOSURE_TRUNCATED"})
        self.assertEqual(compact["semantic_evidence"][0]["source_closure"], views[0]["source_closure"])
        self.assertEqual(compact["candidates"]["semantic_memories"][1]["content"], complete["content"])
        self.assertEqual(compact["semantic_evidence"][1]["memory"]["content"], complete["content"])
        self.assertEqual(compact["participant_activity"][0]["window_statistics"], aggregate)
        raw = next(source for source in compact["sources"] if source["source_key"] == "review-question")
        self.assertEqual(raw["plain_text"], anchor["plain_text"])
        self.assertIn(summary, raw["plain_text"])
        self.assertEqual(_all_source_references(compact), {"reply", "review-question"})
        self.assertEqual(compact["retrieval_coverage"]["derived_source_closure"]["units_truncated"], 1)
        self.assertFalse(compact["retrieval_coverage"]["semantic_none_allowed"])
        self.assertEqual(packet, original)

    def test_definition_next_context_uses_last_slot_after_literal_and_layer_coverage(self) -> None:
        ordinary = _message("ordinary-anchor", "Velora 是一个被提到的代号", 100)
        definition = _message("definition-anchor", "Quneth 是啥？", 200)
        groups = []
        for label, anchor, timestamp in (("ordinary", ordinary, 100), ("definition", definition, 200)):
            groups.append({
                "anchor_source_key": anchor["source_key"],
                "context_relation": "chronological_neighbor_not_reply",
                "messages": [{**_message(f"{label}-before", "前一条上下文", timestamp - 1), "relative_seconds": -1},
                             {**anchor, "relative_seconds": 0},
                             {**_message(f"{label}-after", "后一条上下文", timestamp + 1), "relative_seconds": 1}],
            })
        packet = {
            "reply_context": [_message("reply", "当前回复源", 1)],
            "semantic_evidence": [{"evidence": [_message(f"semantic-{i}", "独立语义来源", 10 + i)]}
                                  for i in range(5)],
            "lexical_messages": [ordinary, definition], "lexical_context": groups,
            "expanded_episodes": [{"messages": [_message("episode-start", "事件开头", 20),
                                                   _message("episode-end", "事件末尾", 30)]}],
            "participant_history": [{"messages": [_message("history", "历史层来源", 40)]}],
            "recent_context": [_message("recent", "最近层来源", 50)],
        }
        original = deepcopy(packet)
        compact = compile_evidence_atom_pack(packet, max_sources=12, query="Velora Quneth")
        selected = {source["source_key"] for source in compact["sources"]}
        self.assertEqual(len(selected), 12)
        self.assertTrue({"ordinary-anchor", "definition-anchor", "definition-after", "reply",
                         "episode-start", "history", "recent"}.issubset(selected))
        self.assertTrue({f"semantic-{i}" for i in range(5)}.issubset(selected))
        self.assertTrue({"ordinary-before", "ordinary-after", "definition-before"}.isdisjoint(selected))
        self.assertEqual(compact["retrieval_coverage"]["query_facets_missing"], [])
        definition_group = compact["lexical_context"][1]
        self.assertEqual(definition_group["missing_source_count"], 1)
        self.assertTrue(definition_group["context_truncated"])
        self.assertEqual(definition_group["context_relation"], "chronological_neighbor_not_reply")
        self.assertEqual(_all_source_references(compact), selected)
        self.assertEqual(packet, original)

    def test_bot_or_negated_fragment_cannot_cover_human_definition_question(self) -> None:
        bot = {**_message("old-bot", "Velora 更不是什么神秘物件。关于Velora是什么，已有解释。", 1),
               "role": "BOT"}
        negated = _message("negated-human", "Velora 更不是什么陌生代号。", 2)
        question = _message("human-question", "Velora 是啥？", 3)
        packet = {"semantic_evidence": [{"evidence": [bot]}],
                  "lexical_messages": [bot, negated, question]}
        original = deepcopy(packet)
        compact = compile_evidence_atom_pack(packet, max_sources=2, query="Velora")
        catalog = {source["source_key"]: source for source in compact["sources"]}
        self.assertEqual(set(catalog), {"old-bot", "human-question"})
        self.assertEqual(catalog["old-bot"]["plain_text"], bot["plain_text"])
        self.assertEqual(compact["retrieval_coverage"]["lexical_evidence_facets"]["definition_questions"],
                         {"total": 1, "covered": 1, "missing": []})
        self.assertEqual(packet, original)

    def test_same_literal_keeps_structured_candidates_and_definition_under_cap(self) -> None:
        def raw(key, text, sent_at):
            message = _message(key, text, sent_at)
            message["sender_participant_key"] = canonical_participant_key("synthetic", message["sender_id"])
            return message

        def addressed(key, target, sent_at):
            message = raw(key, "Velora 请看一下", sent_at)
            message["mentions"] = [{"account_id": target,
                                    "participant_key": canonical_participant_key("synthetic", target)}]
            message["components"] = [{"type": "mention", "account_id": target}]
            return message

        casual = raw("casual-literal", "Velora 今天被随口提到", 10)
        first = addressed("candidate-a", "target-a", 20)
        second = addressed("candidate-b", "target-b", 30)
        definition = raw("definition", "Velora 是什么？", 40)
        following = raw("definition-next", "这是一份合成项目代号说明。", 41)
        unverified = raw("unverified-subject", "Velora 另一条闲聊", 50)
        unverified["subject_participant_key"] = canonical_participant_key("synthetic", "target-c")
        unverified["mentions"] = [{"account_id": "target-c",
                                  "participant_key": canonical_participant_key("synthetic", "target-c")}]
        packet = {
            "reply_context": [raw("reply", "当前回复锚点", 100)],
            "semantic_evidence": [
                {"evidence": [casual]}, {"evidence": [raw("semantic-other", "独立语义来源", 11)]},
            ],
            "lexical_messages": [casual, first, second, definition, unverified],
            "lexical_context": [{"anchor_source_key": "definition",
                                 "context_relation": "chronological_neighbor_not_reply",
                                 "messages": [{**definition, "relative_seconds": 0},
                                              {**following, "relative_seconds": 1}]}],
            "expanded_episodes": [{"messages": [raw(f"episode-{i}", "噪声事件原文", 200 + i)
                                                  for i in range(20)]}],
            "participant_history": [{"messages": [raw("history", "历史层来源", 300)]}],
            "recent_context": [raw("recent", "最近层来源", 400)],
        }
        original = deepcopy(packet)
        compact = compile_evidence_atom_pack(packet, max_sources=12, query="Velora")
        selected = {source["source_key"] for source in compact["sources"]}
        self.assertEqual(len(selected), 12)
        self.assertTrue({"casual-literal", "candidate-a", "candidate-b", "definition",
                         "definition-next", "reply", "history", "recent"}.issubset(selected))
        facets = compact["retrieval_coverage"]["lexical_evidence_facets"]
        self.assertEqual(facets["structured_candidates"], {"total": 2, "covered": 2, "missing": []})
        self.assertEqual(facets["definition_questions"], {"total": 1, "covered": 1, "missing": []})
        group = compact["lexical_context"][0]
        self.assertFalse(group["context_truncated"])
        self.assertEqual(group["context_relation"], "chronological_neighbor_not_reply")
        self.assertEqual(_all_source_references(compact), selected)
        self.assertNotIn(canonical_participant_key("synthetic", "target-c"), compact["participant_source_keys"])
        for target, source in (("target-a", "candidate-a"), ("target-b", "candidate-b")):
            key = canonical_participant_key("synthetic", target)
            self.assertIn(source, compact["participant_source_keys"][key])
            self.assertNotIn(key, compact["participant_speaker_source_keys"])
        self.assertEqual(packet, original)

    def test_lexical_anchors_keep_bounded_neighbors_before_episode_fill(self) -> None:
        first = _message("literal-first", "Velora is mentioned here", 100)
        second = _message("literal-second", "Quneth is mentioned here", 200)
        before = _message("first-before", "earlier speaker's context", 99)
        after = _message("first-after", "another speaker's clarification", 101)
        packet = {
            "reply_context": [_message("current-reply", "current quoted anchor", 300)],
            "lexical_messages": [_message("irrelevant-hit", "unrelated result", 1), first, second],
            "lexical_context": [
                {"anchor_source_key": first["source_key"],
                 "context_relation": "chronological_neighbor_not_reply",
                 "messages": [{**before, "relative_seconds": -1}, {**first, "relative_seconds": 0},
                              {**after, "relative_seconds": 1}]},
                {"anchor_source_key": second["source_key"],
                 "context_relation": "chronological_neighbor_not_reply",
                 "messages": [{**_message("second-before", "prior detail", 199), "relative_seconds": -1},
                              {**second, "relative_seconds": 0},
                              {**_message("second-after", "later detail", 201), "relative_seconds": 1}]},
            ],
            "expanded_episodes": [{"id": 1, "messages": [
                _message("episode-start", "episode begins", 10),
                _message("episode-end", "episode ends", 20),
            ]}],
        }
        original = deepcopy(packet)
        compact = compile_evidence_atom_pack(packet, max_sources=6, query="Velora Quneth")
        selected = {source["source_key"] for source in compact["sources"]}
        self.assertEqual(selected, {"current-reply", "literal-first", "literal-second",
                                    "first-before", "first-after", "episode-start"})
        self.assertEqual(len(selected), 6)
        self.assertNotIn("irrelevant-hit", selected)
        groups = compact["lexical_context"]
        self.assertTrue(all(group["anchor_selected"] for group in groups))
        self.assertFalse(groups[0]["context_truncated"])
        self.assertTrue(groups[1]["context_truncated"])
        self.assertEqual(groups[1]["missing_source_count"], 2)
        self.assertEqual([message["relative_seconds"] for message in groups[0]["messages"]], [-1, 0, 1])
        self.assertEqual(groups[0]["context_relation"], "chronological_neighbor_not_reply")
        self.assertEqual(_all_source_references(compact), selected)
        catalog = {source["source_key"]: source for source in compact["sources"]}
        self.assertEqual(catalog["first-after"]["sender_id"], after["sender_id"])
        self.assertEqual(catalog["first-after"]["plain_text"], after["plain_text"])
        self.assertNotIn("reply_to_source_key", catalog["first-after"])
        coverage = compact["retrieval_coverage"]
        self.assertEqual(coverage["lexical_context_groups_fully_retained"], 1)
        self.assertEqual(coverage["lexical_context_groups_truncated"], 1)
        self.assertEqual(coverage["lexical_query_terms_dropped"], [])
        self.assertEqual(packet, original)

    def test_host_time_and_quoted_author_are_explicit_without_rebinding_source(self) -> None:
        sent_at = int(datetime(2025, 1, 2, 18, 30, tzinfo=UTC).timestamp())
        source = {
            **_message("reply-message", "current author's question", sent_at),
            "sender_participant_key": "participant-author",
            "components": [{
                "type": "reply", "sender_id": "quoted-account",
                "plain_text": "quoted speaker's statement", "sent_at": sent_at - 60,
            }],
        }
        compact = compile_evidence_atom_pack({"lexical_messages": [source]}, max_sources=1)
        hydrated = hydrate_evidence_atom_pack(compact, [source])
        catalog = hydrated["sources"][0]
        self.assertEqual(catalog["local_datetime"], "2025-01-03T02:30:00+08:00")
        self.assertEqual(catalog["timezone"], "Asia/Shanghai")
        self.assertEqual(catalog["plain_text"], "current author's question")
        quote = catalog["components"][0]
        self.assertEqual(quote["content_scope"], "QUOTED_MESSAGE")
        self.assertEqual(quote["timestamp_basis"], "ADAPTER_REPORTED_QUOTE_METADATA")
        self.assertEqual(quote["sender_id"], "quoted-account")
        self.assertEqual(quote["local_datetime"], "2025-01-03T02:29:00+08:00")
        self.assertEqual(hydrated["participant_speaker_source_keys"], {
            "participant-author": ["reply-message"],
        })

    def test_source_cap_retains_rare_literal_query_terms_before_episode_fill(self) -> None:
        for terms, query in (
            (("霜鹿", "锦舟", "铜塔"), "霜鹿、锦舟与铜塔有什么关联？"),
            (("Velora", "Nembit", "Quorix"), "Velora, Nembit and Quorix relationships"),
        ):
            with self.subTest(query_language="CJK" if not terms[0].isascii() else "ASCII"):
                lexical = [
                    _message(f"common-{index}", terms[0], index)
                    for index in range(10)
                ] + [
                    _message("rare-second", terms[1], 11),
                    _message("rare-third", terms[2], 12),
                ]
                packet = {
                    "lexical_messages": lexical,
                    "semantic_evidence": [
                        {"memory": {"id": index}, "evidence": [
                            _message(f"semantic-{index}", "other subject", 20 + index)
                        ]}
                        for index in range(2)
                    ],
                    "expanded_episodes": [
                        {"id": index, "messages": [
                            _message(f"episode-{index}-{offset}", "other discussion", 30 + offset)
                            for offset in range(8)
                        ]}
                        for index in range(6)
                    ],
                    "participant_history": [{"messages": [_message("history", "history", 50)]}],
                    "recent_context": [_message("recent", "recent", 60)],
                }
                original = deepcopy(packet)
                compact = compile_evidence_atom_pack(packet, max_sources=12, query=query)
                selected = {source["source_key"] for source in compact["sources"]}
                self.assertEqual(len(selected), 12)
                self.assertTrue({"rare-second", "rare-third"}.issubset(selected))
                for term in terms:
                    self.assertTrue(any(term in source["plain_text"] for source in compact["sources"]))
                coverage = compact["retrieval_coverage"]
                self.assertEqual(coverage["lexical_query_terms_dropped"], [])
                for layer in ("semantic", "lexical", "episode", "history", "recent"):
                    self.assertGreaterEqual(coverage["strata_selected_sources"][layer], 1)
                self.assertTrue({"episode-0-0", "episode-0-7"}.issubset(selected))
                self.assertFalse(coverage["semantic_none_allowed"])
                self.assertEqual(packet, original)

    def test_literal_query_coverage_distinguishes_unmatched_and_evicted_sources(self) -> None:
        compact = compile_evidence_atom_pack(
            {"reply_context": _message("reply", "direct context", 1),
             "lexical_messages": [_message("known", "Velora", 2)]},
            max_sources=1,
            query="Velora Nembit",
        )
        coverage = compact["retrieval_coverage"]
        self.assertEqual([source["source_key"] for source in compact["sources"]], ["reply"])
        self.assertEqual(coverage["lexical_query_terms_total"], 2)
        self.assertEqual(coverage["lexical_query_terms_available"], 1)
        self.assertEqual(coverage["lexical_query_terms_covered"], 0)
        self.assertEqual(coverage["lexical_query_terms_dropped"], ["lexical_query:000"])
        self.assertEqual(coverage["lexical_query_terms_unmatched"], ["lexical_query:001"])

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

    def test_verified_raw_mention_and_catalog_reply_support_participation_only(self) -> None:
        first = canonical_participant_key("synthetic", "account-a")
        second = canonical_participant_key("synthetic", "account-b")
        own = {"source_key": "source-own", "sender_participant_key": first,
               "sender_id": "account-a", "plain_text": "材料已归档。"}
        mentioned = {"source_key": "source-mention", "sender_participant_key": second,
                     "sender_id": "account-b", "plain_text": "请查收材料。",
                     "mentions": [{"account_id": "account-a", "participant_key": first}],
                     "components": [{"type": "mention", "account_id": "account-a"}]}
        replied = {"source_key": "source-reply", "sender_participant_key": second,
                   "sender_id": "account-b", "plain_text": "收到。", "reply_to_source_key": "source-own"}
        mismatched = {**deepcopy(mentioned), "source_key": "source-mismatch"}
        mismatched["mentions"][0]["participant_key"] = canonical_participant_key("other-platform", "account-a")
        packet = {"sources": [own, mentioned, replied, mismatched]}
        before = deepcopy(packet)
        self.assertEqual(participant_source_bindings(packet), {
            first: ["source-mention", "source-own", "source-reply"],
            second: ["source-mention", "source-mismatch", "source-reply"],
        })
        self.assertEqual(participant_speaker_source_bindings(packet), {
            first: ["source-own"], second: ["source-mention", "source-mismatch", "source-reply"],
        })
        self.assertEqual(packet, before)

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

    def test_activity_source_cap_preserves_time_spread_among_competing_strata(
        self,
    ) -> None:
        packet = {
            "person_reference_candidates": {
                "references": [
                    {
                        "reference": "synthetic member",
                        "candidate_participants": [
                            {
                                "participant_key": "participant-person",
                                "alias_observations": [
                                    _message("person-source", "alias", 1)
                                ],
                            }
                        ],
                    }
                ]
            },
            "semantic_evidence": [
                {
                    "memory": {"id": 1},
                    "evidence": [_message("semantic-source", "semantic", 2)],
                }
            ],
            "participant_activity": [
                {
                    "participant_key": "participant-activity",
                    "found": True,
                    "messages": [
                        _message(
                            f"activity-{index}",
                            f"activity sample {index}",
                            10 + index,
                        )
                        for index in range(12)
                    ],
                }
            ],
            "lexical_messages": [_message("lexical-source", "lexical", 30)],
            "expanded_episodes": [
                {
                    "id": 2,
                    "messages": [
                        _message("episode-anchor", "episode opens", 40),
                        _message("episode-middle", "episode continues", 41),
                        _message("episode-closure", "episode closes", 42),
                    ],
                }
            ],
            "participant_history": [
                {
                    "participant_key": "participant-history",
                    "messages": [_message("history-source", "history", 50)],
                }
            ],
            "recent_context": [_message("recent-source", "recent", 60)],
        }

        compact = compile_evidence_atom_pack(
            packet,
            max_sources=12,
            activity_mode=True,
        )

        selected_activity_indices = {
            int(str(source["source_key"]).rpartition("-")[2])
            for source in compact["sources"]
            if str(source["source_key"]).startswith("activity-")
        }
        self.assertEqual(len(selected_activity_indices), 5)
        self.assertIn(0, selected_activity_indices)
        self.assertIn(11, selected_activity_indices)
        self.assertTrue(
            any(4 <= index <= 7 for index in selected_activity_indices)
        )
        self.assertTrue(
            any(8 <= index <= 10 for index in selected_activity_indices)
        )
        self.assertNotEqual(selected_activity_indices, set(range(5)))

    def test_many_query_facets_cannot_starve_present_retrieval_strata(
        self,
    ) -> None:
        packet = {
            "person_reference_candidates": {
                "references": [
                    {
                        "reference": f"synthetic member {index}",
                        "candidate_participants": [
                            {
                                "participant_key": f"participant-{index}",
                                "alias_observations": [
                                    _message(
                                        f"person-{index}",
                                        f"alias {index}",
                                        index,
                                    )
                                ],
                            }
                        ],
                    }
                    for index in range(20)
                ]
            },
            "semantic_evidence": [
                {
                    "memory": {"id": 1},
                    "evidence": [_message("semantic-source", "semantic", 30)],
                }
            ],
            "lexical_messages": [_message("lexical-source", "lexical", 31)],
            "expanded_episodes": [
                {
                    "id": 2,
                    "messages": [
                        _message("episode-anchor", "episode opens", 32),
                        _message("episode-closure", "episode closes", 33),
                    ],
                }
            ],
            "recent_context": [_message("recent-source", "recent", 34)],
        }

        compact = compile_evidence_atom_pack(packet, max_sources=12)

        selected = {source["source_key"] for source in compact["sources"]}
        self.assertTrue(
            {
                "semantic-source",
                "lexical-source",
                "episode-anchor",
                "recent-source",
            }.issubset(selected)
        )
        coverage = compact["retrieval_coverage"]
        for stratum in ("person", "semantic", "lexical", "episode", "recent"):
            self.assertGreaterEqual(
                coverage["strata_selected_sources"][stratum],
                1,
            )
        self.assertGreater(len(coverage["query_facets_missing"]), 0)

    def test_activity_requires_an_integer_time_axis(self) -> None:
        invalid_activity = _message("activity-source", "activity", 1)
        invalid_activity["sent_at"] = "not-an-integer"

        with self.assertRaisesRegex(
            ValueError,
            "activity message sent_at must be an integer",
        ):
            compile_evidence_atom_pack(
                {
                    "participant_activity": [
                        {
                            "participant_key": "participant-activity",
                            "messages": [invalid_activity],
                        }
                    ]
                },
                max_sources=1,
                activity_mode=True,
            )

    def test_episode_boundaries_follow_list_position_not_message_time(self) -> None:
        compact = compile_evidence_atom_pack(
            {
                "expanded_episodes": [
                    {
                        "id": 1,
                        "messages": [
                            _message("list-first", "first", 30),
                            _message("time-first", "middle", 10),
                            _message("list-last", "last", 20),
                        ],
                    }
                ]
            },
            max_sources=2,
        )

        self.assertEqual(
            {source["source_key"] for source in compact["sources"]},
            {"list-first", "list-last"},
        )

    def test_plural_source_projection_strips_and_deduplicates_in_order(
        self,
    ) -> None:
        compact = compile_evidence_atom_pack(
            {
                "candidates": [
                    {
                        "kind": "synthetic",
                        "source_keys": [
                            " source-b ",
                            "source-a",
                            "source-b",
                            "dropped-source",
                        ],
                    }
                ]
            },
            max_sources=2,
        )

        self.assertEqual(
            compact["candidates"][0]["source_keys"],
            ["source-b", "source-a"],
        )

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

    def test_source_cap_covers_explicit_query_facets_and_episode_boundaries(
        self,
    ) -> None:
        packet = {
            "person_reference_candidates": {
                "references": [
                    {
                        "reference": reference,
                        "candidate_participants": [
                            {
                                "participant_key": f"participant-{reference}",
                                "alias_observations": [
                                    _message(
                                        f"person-{reference}",
                                        f"observed alias {reference}",
                                        index,
                                    )
                                ],
                            }
                        ],
                    }
                    for index, reference in enumerate(
                        ("alpha", "beta", "gamma"),
                        start=1,
                    )
                ]
            },
            "participant_activity": [
                {
                    "participant_key": "participant-activity",
                    "found": True,
                    "messages": [
                        _message("activity-start", "first sample", 10),
                        _message("activity-middle", "middle sample", 11),
                        _message("activity-end", "last sample", 12),
                    ],
                }
            ],
            "semantic_evidence": [
                {
                    "memory": {"id": 7, "content": "entity fact"},
                    "evidence": [_message("semantic-entity", "support", 20)],
                }
            ],
            "expanded_episodes": [
                {
                    "id": 9,
                    "messages": [
                        _message("episode-anchor", "episode opens", 30),
                        _message("episode-middle", "episode continues", 31),
                        _message("episode-closure", "episode closes", 32),
                    ],
                }
            ],
        }

        compact = compile_evidence_atom_pack(
            packet,
            max_sources=8,
            activity_mode=True,
        )

        selected = {source["source_key"] for source in compact["sources"]}
        self.assertEqual(
            selected,
            {
                "person-alpha",
                "person-beta",
                "person-gamma",
                "activity-start",
                "activity-end",
                "semantic-entity",
                "episode-anchor",
                "episode-closure",
            },
        )
        coverage = compact["retrieval_coverage"]
        self.assertEqual(coverage["query_facets_total"], 6)
        self.assertEqual(coverage["query_facets_covered"], 6)
        self.assertEqual(coverage["query_facets_missing"], [])
        self.assertEqual(coverage["continuity_facets_total"], 2)
        self.assertEqual(coverage["continuity_facets_covered"], 2)
        self.assertEqual(coverage["continuity_facets_missing"], [])

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

    def test_complete_activity_distribution_is_not_mixed_with_sample_zeros(self) -> None:
        first_at = int(datetime(2025, 1, 1, 17, tzinfo=UTC).timestamp())
        last_at = first_at + 4 * 3600
        histogram = {f"{hour:02d}": 0 for hour in range(24)}
        histogram.update({"01": 1, "05": 2})
        aggregate = {
            "schema_version": "mr-memory.activity-window.v1",
            "authority": "HOST_SQLITE_SNAPSHOT",
            "basis": "all_snapshot_visible_direct_speaker_messages",
            "scope": {"umo": "synthetic:GroupMessage:activity", "participant_key": "participant-one",
                      "start_sent_at": first_at - 3600, "end_sent_at_exclusive": last_at + 3600,
                      "message_upper_bound": 3},
            "timezone": "Asia/Shanghai", "source_count": 3,
            "hour_histogram": histogram,
            "daily": [{"local_date": "2025-01-02", "source_count": 3,
                       "first_sent_at": first_at, "last_sent_at": last_at,
                       "first_local_datetime": "2025-01-02T01:00:00+08:00",
                       "last_local_datetime": "2025-01-02T05:00:00+08:00"}],
            "source_revision_sha256": "a" * 64,
        }
        aggregate["aggregate_id"] = activity_aggregate_id(aggregate)
        source = {**_message("sample-at-one", "合成发言样本", first_at), "local_hour": 1}
        packet = {"participant_activity": [{
            "participant_key": "participant-one", "found": True, "message_count": 1,
            "sample_count": 1, "hour_histogram": {"01": 1, "05": 0},
            "sampling_method": "synthetic-one-message-sample", "messages_truncated": True,
            "messages": [source], "window_statistics": aggregate,
        }]}
        original = deepcopy(packet)
        compact = compile_evidence_atom_pack(packet, max_sources=1, activity_mode=True)
        activity = compact["participant_activity"][0]
        self.assertNotIn("message_count", activity)
        self.assertNotIn("hour_histogram", activity)
        self.assertEqual(activity["sample_count"], 1)
        self.assertEqual(activity["statistics_basis"], "host_window_statistics")
        self.assertTrue(activity["messages_truncated"])
        self.assertIn("synthetic-one-message-sample", activity["sampling_method"])
        self.assertEqual(activity["window_statistics"], aggregate)
        self.assertEqual(activity["window_statistics"]["hour_histogram"]["05"], 2)
        self.assertEqual(compact["sources"][0]["plain_text"], source["plain_text"])
        self.assertEqual(packet, original)

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
