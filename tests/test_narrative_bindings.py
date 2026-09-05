from __future__ import annotations

import copy
import json
import unittest
import uuid
from dataclasses import replace
from pathlib import Path

from mr_memory.distillation import (
    build_distillation_prompt, build_distillation_prompt_aliases,
    parse_distillation_response, persist_distillation,
)
from mr_memory.models import NormalizedMessage
from mr_memory.narrative_bindings import (
    build_narrative_bindings, canonical_narrative_fingerprint_text,
    compose_narrative_summary, matching_narrative_bindings, participant_alias_tokens,
)
from mr_memory.reader import build_l2_reader_prompt
from mr_memory.storage import MemoryStorage
from tests.test_certificate_v2 import _snapshot


class NarrativeBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        test_root = Path.cwd() / ".dev" / "test-tmp"
        test_root.mkdir(parents=True, exist_ok=True)
        self.database_path = test_root / f"{uuid.uuid4().hex}.db"
        self.storage = MemoryStorage(self.database_path)
        self.umo = "synthetic:GroupMessage:narrative"
        for index, name in enumerate(("合成整理员", "合成记录员"), start=1):
            text = "p0是文件中的原始标记。" if index == 1 else "材料已归档。"
            self.storage.upsert_message(NormalizedMessage(
                platform="synthetic", platform_id="synthetic", umo=self.umo,
                group_id="narrative", message_id=str(index), sender_id=f"account-{index}",
                sender_name=name, sent_at=100 * index, plain_text=text,
                content=[{"type": "text", "text": text}],
            ))
        self.messages = sorted(self.storage.search_messages(umo=self.umo, limit=2), key=lambda m: m.id)

    def tearDown(self) -> None:
        self.storage.close()
        for suffix in ("", "-wal", "-shm"):
            Path(f"{self.database_path}{suffix}").unlink(missing_ok=True)

    def _reader(self, candidates):
        packet = {"candidates": candidates, "sources": [
            {"source_key": m.source_key, "sender_participant_key": m.sender_participant_key,
             "plain_text": m.plain_text} for m in self.messages
        ]}
        return build_l2_reader_prompt(
            query="纸鹤计划进展如何", evidence_packet=packet,
            snapshot=replace(_snapshot(), umo=self.umo),
            allowed_source_keys={m.source_key for m in self.messages},
            allowed_participant_keys={m.sender_participant_key for m in self.messages},
            participant_source_keys={m.sender_participant_key: {m.source_key} for m in self.messages},
            pack_read_complete=True, allow_l3=False,
        )

    def test_generated_narrative_round_trips_without_binding_a_copied_quote(self) -> None:
        context = self.storage.distillation_identity_context(
            umo=self.umo, source_keys=[m.source_key for m in self.messages],
        )
        context["active_claims"].append({"id": 9999, "content": "p9曾讨论旧材料。"})
        original_context = copy.deepcopy(context)
        aliases = build_distillation_prompt_aliases(self.messages, identity_context=context)
        self.assertNotIn("p0", aliases.alias_to_participant)
        prompt = json.loads(build_distillation_prompt(self.messages, identity_context=context, aliases=aliases))
        self.assertEqual(prompt["messages"][0]["text"], self.messages[0].plain_text)
        self.assertNotIn("content", prompt["identity_context"]["active_claims"][0])
        self.assertEqual(prompt["identity_context"]["active_claims"][0]["reader_withheld_fields"][0]["reason"],
                         "UNMAPPED_HISTORICAL_PARTICIPANT_TOKEN")
        self.assertEqual(context, original_context)
        participant = self.messages[1].sender_participant_key
        local_id = aliases.participant_to_alias[participant]
        summary = f'{local_id}完成归档；原文为“{self.messages[0].plain_text}”。'
        response = {"episodes": [{
            "source_keys": list(aliases.alias_to_source), "title": "材料归档", "summary": summary,
            "tag": "归档", "cues": ["材料"],
        }], "claims": [{
            "subject": {"participant_key": local_id, "unresolved_text": ""},
            "claim_type": "BEHAVIOR", "predicate": "archive_status", "object": summary,
            "epistemic_status": "ASSERTED", "operation": "ASSERT", "target_claim_ids": [],
            "confidence": 0.8, "evidence": [{"source_key": aliases.source_to_alias[self.messages[1].source_key],
                                            "role": "SUPPORT", "confidence": 0.8}],
        }], "topics": [], "associations": [], "ignored_source_keys": []}
        batch = parse_distillation_response(json.dumps(response, ensure_ascii=False), self.messages,
                                            identity_context=context, aliases=aliases)
        persisted = persist_distillation(self.storage, batch, extractor_version="synthetic-narrative")
        candidates = self.storage.expand_seed_candidates(umo=self.umo, matches=[
            {"owner_type": "episode", "owner_key": str(persisted.episode_ids[0]), "score": 1.0},
            {"owner_type": "semantic", "owner_key": str(persisted.semantic_ids[0]), "score": 1.0},
        ])
        for field, item in (("summary", candidates["episodes"][0]), ("content", candidates["semantic_memories"][0])):
            self.assertEqual(item[field], summary)
            references = item["narrative_bindings"]["fields"][field]["references"]
            self.assertEqual([r["token"] for r in references], [local_id])
            self.assertEqual(references[0]["participant_key"], participant)
        request = self._reader(candidates)
        delivered = json.loads(request.user_prompt)["evidence_packet"]["candidates"]["episodes"][0]
        self.assertEqual(delivered["summary"], summary)
        self.assertEqual(delivered["reader_text_identity_scope"], "EXPLICIT_POSITIONS_ONLY")
        reference = delivered["narrative_bindings"]["fields"]["summary"]["references"][0]
        self.assertEqual(reference["participant_key"], request.participant_key_to_alias[participant])
        self.assertNotEqual(reference["participant_key"], local_id)

    def test_topic_keeps_same_token_bound_to_different_people_across_batches(self) -> None:
        summary = "p1已完成材料整理。"
        episode_ids = []
        for index, message in enumerate(self.messages):
            episode_ids.append(self.storage.store_episode(
                umo=self.umo, started_at=message.sent_at, ended_at=message.sent_at,
                title="整理记录", summary=summary, source_keys=[message.source_key], keywords=[],
                stable_key=f"synthetic-episode-{index}",
                narrative_aliases={"p1": message.sender_participant_key},
            ))
        topic = self.storage.store_topic(umo=self.umo, name="材料整理进度", summary="提案摘要",
                                         event_ids=episode_ids, narrative_aliases={})
        candidate = self.storage.expand_seed_candidates(umo=self.umo, matches=[
            {"owner_type": "topic", "owner_key": str(topic), "score": 1.0},
        ])["topics"][0]
        self.assertEqual(candidate["summary"], summary + "；" + summary)
        references = candidate["narrative_bindings"]["fields"]["summary"]["references"]
        self.assertEqual([r["start"] for r in references], [0, len(summary) + 1])
        self.assertEqual([r["participant_key"] for r in references], [m.sender_participant_key for m in self.messages])
        request = self._reader({"topics": [candidate]})
        delivered = json.loads(request.user_prompt)["evidence_packet"]["candidates"]["topics"][0]
        projected = delivered["narrative_bindings"]["fields"]["summary"]["references"]
        self.assertEqual([r["participant_key"] for r in projected],
                         [request.participant_key_to_alias[m.sender_participant_key] for m in self.messages])
        self.assertEqual(len({r["participant_key"] for r in projected}), 2)
        self.assertEqual(delivered["summary"], candidate["summary"])
        first = canonical_narrative_fingerprint_text(summary, {"p1": self.messages[0].sender_participant_key})
        second = canonical_narrative_fingerprint_text(summary, {"p1": self.messages[1].sender_participant_key})
        self.assertNotEqual(first, second)
        cropped, metadata = compose_narrative_summary([(candidate["summary"], candidate["narrative_bindings"])], max_chars=len(summary))
        self.assertEqual(cropped, summary)
        self.assertEqual(metadata["fields"]["summary"]["references"][0]["participant_key"], self.messages[1].sender_participant_key)

    def test_snake_case_alias_round_trips_without_binding_original_field_names(self) -> None:
        original_text = "源字段foo_p1；ap2与p3b只是名称。"
        self.storage.upsert_message(NormalizedMessage(
            platform="synthetic", platform_id="synthetic", umo=self.umo, group_id="narrative",
            message_id="1", sender_id="account-1", sender_name="合成整理员", sent_at=100,
            plain_text=original_text, content=[{"type": "text", "text": original_text}],
        ))
        self.messages = sorted(self.storage.search_messages(umo=self.umo, limit=2), key=lambda m: m.id)
        self.assertEqual(participant_alias_tokens(original_text), {"p1"})
        context = self.storage.distillation_identity_context(
            umo=self.umo, source_keys=[m.source_key for m in self.messages],
        )
        aliases = build_distillation_prompt_aliases(self.messages, identity_context=context)
        self.assertNotIn("p1", aliases.alias_to_participant)
        participant = self.messages[1].sender_participant_key
        local_id = aliases.participant_to_alias[participant]
        predicate = f"record_for_{local_id}"
        content = "已整理材料，原字段为foo_p1。"
        response = {"episodes": [], "claims": [{
            "subject": {"participant_key": local_id}, "claim_type": "BEHAVIOR",
            "predicate": predicate, "object": content, "epistemic_status": "ASSERTED",
            "operation": "ASSERT", "target_claim_ids": [], "confidence": 0.8,
            "evidence": [{"source_key": aliases.source_to_alias[self.messages[1].source_key],
                          "role": "SUPPORT", "confidence": 0.8}],
        }], "topics": [], "associations": [],
            "ignored_source_keys": [aliases.source_to_alias[self.messages[0].source_key]]}
        batch = parse_distillation_response(json.dumps(response), self.messages,
                                            identity_context=context, aliases=aliases)
        persisted = persist_distillation(self.storage, batch, extractor_version="synthetic-snake-case")
        candidates = self.storage.expand_seed_candidates(umo=self.umo, matches=[
            {"owner_type": "semantic", "owner_key": str(persisted.semantic_ids[0]), "score": 1.0},
        ])
        item = candidates["semantic_memories"][0]
        self.assertEqual(item["aspect_tag"], predicate)
        self.assertEqual(item["content"], content)
        fields = item["narrative_bindings"]["fields"]
        self.assertEqual(set(fields), {"aspect_tag"})
        reference = fields["aspect_tag"]["references"][0]
        self.assertEqual(reference["participant_key"], participant)
        self.assertEqual(predicate[reference["start"]:reference["end"]], local_id)
        first = canonical_narrative_fingerprint_text(predicate, {local_id: participant})
        second = canonical_narrative_fingerprint_text(predicate, {local_id: self.messages[0].sender_participant_key})
        self.assertNotEqual(first, second)
        request = self._reader(candidates)
        self.assertNotIn("p1", request.participant_key_to_alias.values())
        self.assertNotIn(local_id, request.participant_key_to_alias.values())
        packet = json.loads(request.user_prompt)["evidence_packet"]
        delivered = packet["candidates"]["semantic_memories"][0]
        projected = delivered["narrative_bindings"]["fields"]["aspect_tag"]["references"][0]
        self.assertEqual(projected["participant_key"], request.participant_key_to_alias[participant])
        self.assertEqual(delivered["aspect_tag"], predicate)
        self.assertEqual(packet["sources"][0]["plain_text"], original_text)

    def test_legacy_and_stale_narratives_keep_unknown_identity(self) -> None:
        original_text = "p1是源文献中的编号，不能改写。"
        self.storage.upsert_message(NormalizedMessage(
            platform="synthetic", platform_id="synthetic", umo=self.umo, group_id="narrative",
            message_id="1", sender_id="account-1", sender_name="合成整理员", sent_at=100,
            plain_text=original_text, content=[{"type": "text", "text": original_text}],
        ))
        self.messages = sorted(self.storage.search_messages(umo=self.umo, limit=2), key=lambda m: m.id)
        message = self.messages[0]
        summary = "p1提到了待处理材料。"
        episode = self.storage.store_episode(
            umo=self.umo, started_at=100, ended_at=100, title="旧整理记录", summary=summary,
            source_keys=[message.source_key], keywords=[], stable_key="synthetic-legacy",
        )
        candidates = self.storage.expand_seed_candidates(umo=self.umo, matches=[
            {"owner_type": "episode", "owner_key": str(episode), "score": 1.0},
        ])
        request = self._reader(candidates)
        delivered = json.loads(request.user_prompt)["evidence_packet"]["candidates"]["episodes"][0]
        self.assertNotIn("summary", delivered)
        self.assertEqual(delivered["reader_withheld_fields"][0]["reason"], "UNMAPPED_HISTORICAL_PARTICIPANT_TOKEN")
        self.assertEqual(candidates["episodes"][0]["summary"], summary)
        self.assertEqual(json.loads(request.user_prompt)["evidence_packet"]["sources"][0]["plain_text"], original_text)
        self.assertEqual(delivered["reader_text_identity_scope"], "HISTORICAL_UNMAPPED")
        self.assertNotIn("narrative_bindings", delivered)
        metadata = build_narrative_bindings({"summary": summary}, {"p1": message.sender_participant_key})
        self.assertEqual(matching_narrative_bindings({"summary": summary + "后来改动。"}, metadata)["fields"], {})
        broken = copy.deepcopy(metadata)
        broken["fields"]["summary"]["references"][0]["start"] = 1
        self.assertEqual(matching_narrative_bindings({"summary": summary}, broken)["fields"], {})


if __name__ == "__main__":
    unittest.main()
