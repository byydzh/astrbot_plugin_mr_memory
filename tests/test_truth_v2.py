from __future__ import annotations

import json
import unittest
import uuid
from pathlib import Path

from mr_memory.brief import parse_evidence_brief, render_evidence_brief
from mr_memory.distillation import parse_distillation_response, persist_distillation
from mr_memory.feedback import feedback_surface_score
from mr_memory.identity import (
    build_request_identity_context,
    canonical_participant_key,
    sanitize_components,
)
from mr_memory.models import NormalizedMessage
from mr_memory.storage import MemoryStorage


class TruthLayerV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        root = Path.cwd() / ".dev" / "test-tmp"
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / f"{uuid.uuid4().hex}.db"
        self.storage = MemoryStorage(self.path)
        self.umo = "shadow:GroupMessage:group-v2"
        self.platform_id = "shadow"

    def tearDown(self) -> None:
        self.storage.close()
        for suffix in ("", "-wal", "-shm"):
            Path(f"{self.path}{suffix}").unlink(missing_ok=True)

    def test_current_request_keeps_sender_mention_and_reply_accounts_distinct(
        self,
    ) -> None:
        context = build_request_identity_context(
            platform_id=self.platform_id,
            sender_id="1006039062",
            sender_name="空奏列車",
            content=[
                {
                    "type": "mention",
                    "account_id": "1094354810",
                    "display_name": "🍹想喝气泡水🥤",
                },
                {
                    "type": "reply",
                    "message_id": "quoted-message",
                    "sender_id": "1094354810",
                    "sender_name": "🍹想喝气泡水🥤",
                },
                {"type": "text", "text": "她是谁，是不是我"},
            ],
        )

        sender = context["sender"]
        mentioned = context["mentions"][0]
        reply_target = context["reply_target"]
        self.assertEqual(sender["account_id"], "1006039062")
        self.assertEqual(mentioned["account_id"], "1094354810")
        self.assertFalse(mentioned["same_account_as_sender"])
        self.assertIsNotNone(reply_target)
        self.assertFalse(reply_target["same_account_as_sender"])
        self.assertEqual(
            mentioned["participant_key"],
            canonical_participant_key(self.platform_id, "1094354810"),
        )

    def message(
        self,
        message_id: str,
        sender_id: str,
        sender_name: str,
        text: str,
        sent_at: int,
        *,
        content: list[dict[str, object]] | None = None,
        role: str = "USER",
    ) -> NormalizedMessage:
        return NormalizedMessage(
            platform="aiocqhttp",
            platform_id=self.platform_id,
            umo=self.umo,
            group_id="group-v2",
            message_id=message_id,
            sender_id=sender_id,
            sender_name=sender_name,
            sent_at=sent_at,
            plain_text=text,
            content=content or [{"type": "text", "text": text}],
            role=role,  # type: ignore[arg-type]
        )

    def test_same_account_keeps_identity_across_nickname_changes(self) -> None:
        self.storage.upsert_message(self.message("1", "10001", "老王", "早", 100))
        self.storage.upsert_message(
            self.message("2", "10001", "王哥的新昵称", "晚", 200)
        )
        participants = self.storage.list_participants(umo=self.umo)
        self.assertEqual(len(participants), 1)
        self.assertEqual(participants[0]["account_id"], "10001")
        aliases = {item["alias"] for item in participants[0]["aliases"]}
        self.assertEqual(aliases, {"老王", "王哥的新昵称"})
        old = self.storage.resolve_participants(umo=self.umo, reference="老王")
        new = self.storage.resolve_participants(
            umo=self.umo, reference="王哥的新昵称"
        )
        self.assertEqual(
            old["participants"][0]["canonical_key"],
            new["participants"][0]["canonical_key"],
        )

    def test_older_import_cannot_roll_back_current_display_name(self) -> None:
        self.storage.upsert_message(
            self.message("new", "10001", "当前群名片", "较新", 200)
        )
        self.storage.upsert_message(
            self.message("old", "10001", "历史群名片", "补录旧消息", 100)
        )
        participant = self.storage.resolve_participants(
            umo=self.umo, reference="10001"
        )["participants"][0]
        self.assertEqual(participant["current_display_name"], "当前群名片")
        self.assertEqual(participant["first_seen_at"], 100)
        self.assertEqual(participant["last_seen_at"], 200)
        self.assertEqual(
            {item["alias"] for item in participant["aliases"]},
            {"当前群名片", "历史群名片"},
        )

    def test_mention_alias_does_not_replace_observed_display_name(self) -> None:
        self.storage.upsert_message(
            self.message("1", "10001", "本人群名片", "在", 100)
        )
        self.storage.upsert_message(
            self.message(
                "2",
                "20002",
                "旁观者",
                "@玩笑称呼",
                110,
                content=[
                    {
                        "type": "mention",
                        "account_id": "10001",
                        "display_name": "玩笑称呼",
                    }
                ],
            )
        )
        participant = self.storage.resolve_participants(
            umo=self.umo, reference="10001"
        )["participants"][0]
        self.assertEqual(participant["current_display_name"], "本人群名片")
        self.assertEqual(
            {item["alias"] for item in participant["aliases"]},
            {"本人群名片", "玩笑称呼"},
        )

    def test_mention_only_alias_is_not_presented_as_observed_group_card(self) -> None:
        self.storage.upsert_message(
            self.message(
                "1",
                "20002",
                "旁观者",
                "@临时称呼",
                100,
                content=[
                    {
                        "type": "mention",
                        "account_id": "10001",
                        "display_name": "临时称呼",
                    }
                ],
            )
        )
        participant = self.storage.resolve_participants(
            umo=self.umo, reference="10001"
        )["participants"][0]
        self.assertEqual(participant["current_display_name"], "")
        self.assertEqual(participant["aliases"][0]["source_kind"], "mention")

    def test_duplicate_alias_stays_ambiguous_and_cannot_bind_claim(self) -> None:
        self.storage.upsert_message(self.message("1", "10001", "小明", "在", 100))
        self.storage.upsert_message(self.message("2", "10002", "小明", "也在", 110))
        source = self.message("3", "10003", "旁观者", "小明喜欢红色", 120)
        self.storage.upsert_message(source)
        messages = self.storage.search_messages(umo=self.umo, limit=20)
        context = self.storage.distillation_identity_context(
            umo=self.umo,
            source_keys=[item.source_key for item in messages],
        )
        resolved = self.storage.resolve_participants(umo=self.umo, reference="小明")
        self.assertTrue(resolved["ambiguous"])
        invented_choice = resolved["participants"][0]["canonical_key"]
        value = {
            "episodes": [],
            "claims": [
                {
                    "subject": {
                        "participant_key": invented_choice,
                        "unresolved_text": "",
                    },
                    "claim_type": "PREFERENCE",
                    "predicate": "颜色偏好",
                    "object": "喜欢红色",
                    "epistemic_status": "ASSERTED",
                    "operation": "ASSERT",
                    "target_claim_ids": [],
                    "confidence": 0.8,
                    "evidence": [
                        {
                            "source_key": source.resolved_source_key(),
                            "role": "SUPPORT",
                            "span": "小明喜欢红色",
                            "confidence": 0.8,
                        }
                    ],
                }
            ],
            "topics": [],
        }
        with self.assertRaisesRegex(ValueError, "lacks deterministic"):
            parse_distillation_response(
                json.dumps(value, ensure_ascii=False),
                messages,
                identity_context=context,
                target_source_keys=[source.resolved_source_key()],
            )

    def test_ambiguous_alias_never_combines_two_people_claims(self) -> None:
        first = self.message("1", "10001", "小明", "我喜欢红色", 100)
        second = self.message("2", "10002", "小明", "我喜欢蓝色", 110)
        self.storage.upsert_message(first)
        self.storage.upsert_message(second)
        for message, color in ((first, "红色"), (second, "蓝色")):
            participant = self.storage.resolve_participants(
                umo=self.umo, reference=message.sender_id
            )["participants"][0]
            self.storage.store_semantic_claim(
                umo=self.umo,
                stable_key=f"preference-{message.sender_id}",
                subject_participant_key=participant["canonical_key"],
                subject_text="",
                claim_type="PREFERENCE",
                aspect="颜色",
                content=color,
                epistemic_status="ASSERTED",
                operation="ASSERT",
                target_claim_ids=[],
                evidence=[
                    {
                        "source_key": message.resolved_source_key(),
                        "role": "SUPPORT",
                        "span": f"喜欢{color}",
                        "confidence": 0.9,
                    }
                ],
                confidence=0.9,
            )
        result = self.storage.query_personal_information(
            umo=self.umo, person="小明"
        )
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0]["identity_ambiguous"])
        self.assertEqual(len(result[0]["candidate_participants"]), 2)
        self.assertNotIn("aspect_tag", result[0])

    def test_mention_binds_claim_subject_separately_from_speaker(self) -> None:
        mentioned = self.message("1", "20002", "阿乙", "冒泡", 90)
        self.storage.upsert_message(mentioned)
        source = self.message(
            "2",
            "10001",
            "阿甲",
            "阿乙喜欢低密度构图",
            100,
            content=[
                {"type": "mention", "account_id": "20002", "display_name": "阿乙"},
                {"type": "text", "text": "喜欢低密度构图"},
            ],
        )
        self.storage.upsert_message(source)
        messages = self.storage.search_messages(umo=self.umo, limit=20)
        context = self.storage.distillation_identity_context(
            umo=self.umo,
            source_keys=[item.source_key for item in messages],
        )
        subject = self.storage.resolve_participants(
            umo=self.umo, reference="20002"
        )["participants"][0]["canonical_key"]
        value = {
            "episodes": [],
            "claims": [
                {
                    "subject": {"participant_key": subject, "unresolved_text": ""},
                    "claim_type": "PREFERENCE",
                    "predicate": "图像构图密度",
                    "object": "喜欢低密度构图",
                    "epistemic_status": "ASSERTED",
                    "operation": "ASSERT",
                    "target_claim_ids": [],
                    "confidence": 0.9,
                    "evidence": [
                        {
                            "source_key": source.resolved_source_key(),
                            "role": "SUPPORT",
                            "span": "阿乙喜欢低密度构图",
                            "confidence": 0.9,
                        }
                    ],
                }
            ],
            "topics": [],
        }
        batch = parse_distillation_response(
            json.dumps(value, ensure_ascii=False),
            messages,
            identity_context=context,
            target_source_keys=[source.resolved_source_key()],
        )
        persisted = persist_distillation(
            self.storage, batch, extractor_version="truth-v2-test"
        )
        self.assertEqual(len(persisted.semantic_ids), 1)
        row = self.storage._connection.execute(
            "SELECT subject_participant_id FROM semantic_memories WHERE id = ?",
            (persisted.semantic_ids[0],),
        ).fetchone()
        subject_id = self.storage.resolve_participants(
            umo=self.umo, reference="20002"
        )["participants"][0]["id"]
        speaker_id = self.storage.resolve_participants(
            umo=self.umo, reference="10001"
        )["participants"][0]["id"]
        self.assertEqual(row["subject_participant_id"], subject_id)
        self.assertNotEqual(row["subject_participant_id"], speaker_id)

    def test_incremental_checkpoint_and_edit_requeue(self) -> None:
        for index in range(1, 6):
            self.storage.upsert_message(
                self.message(str(index), "10001", "甲", f"消息 {index}", index * 100)
            )
        first = self.storage.next_distillation_batch(
            umo=self.umo, limit=2, overlap=0
        )
        assert first is not None
        self.assertEqual(first.target_source_keys[-1].rsplit("|", 1)[-1], "2")
        self.storage.finish_distillation_batch(work_item=first)
        second = self.storage.next_distillation_batch(
            umo=self.umo, limit=2, overlap=1
        )
        assert second is not None
        self.assertEqual(
            [key.rsplit("|", 1)[-1] for key in second.target_source_keys],
            ["3", "4"],
        )
        self.assertEqual(second.messages[0].message_id, "2")
        self.storage.finish_distillation_batch(work_item=second)
        self.storage.upsert_message(
            self.message("1", "10001", "甲", "消息 1 已编辑", 100)
        )
        edited = self.storage.next_distillation_batch(
            umo=self.umo, limit=1, overlap=0
        )
        assert edited is not None
        self.assertEqual(edited.target_source_keys[0].rsplit("|", 1)[-1], "1")

    def test_interrupted_checkpoint_is_recovered_on_reopen(self) -> None:
        source = self.message("1", "10001", "甲", "等待整理", 100)
        self.storage.upsert_message(source)
        claimed = self.storage.next_distillation_batch(
            umo=self.umo, limit=1, overlap=0
        )
        assert claimed is not None
        self.storage.close()
        self.storage = MemoryStorage(self.path)
        recovered = self.storage._connection.execute(
            "SELECT status, attempts, last_error FROM message_processing"
        ).fetchone()
        self.assertEqual(recovered["status"], "FAILED")
        self.assertEqual(recovered["attempts"], 1)
        self.assertIn("interrupted", recovered["last_error"])
        retry = self.storage.next_distillation_batch(
            umo=self.umo, limit=1, overlap=0
        )
        assert retry is not None
        self.assertEqual(retry.target_source_keys, claimed.target_source_keys)
        self.storage.finish_distillation_batch(work_item=retry)

    def test_terminal_distillation_messages_retry_only_when_explicit(self) -> None:
        source = self.message("1", "10001", "甲", "等待修复后重试", 100)
        self.storage.upsert_message(source)
        for _ in range(3):
            work_item = self.storage.next_distillation_batch(
                umo=self.umo,
                limit=1,
                overlap=0,
            )
            assert work_item is not None
            self.storage.finish_distillation_batch(
                work_item=work_item,
                error="provider failure",
            )

        self.assertEqual(self.storage.pending_distillation_count(umo=self.umo), 0)
        self.assertIsNone(
            self.storage.next_distillation_batch(
                umo=self.umo,
                limit=1,
                overlap=0,
            )
        )
        self.assertEqual(
            self.storage.retry_terminal_distillation_failures(umo=self.umo),
            1,
        )
        self.assertEqual(self.storage.pending_distillation_count(umo=self.umo), 1)
        retried = self.storage.next_distillation_batch(
            umo=self.umo,
            limit=1,
            overlap=0,
        )
        assert retried is not None
        state = self.storage._connection.execute(
            "SELECT status, attempts FROM message_processing"
        ).fetchone()
        self.assertEqual(state["status"], "PROCESSING")
        self.assertEqual(state["attempts"], 1)

    def test_host_time_ignores_model_timestamp(self) -> None:
        source = self.message("1", "10001", "甲", "真实时间", 1234)
        self.storage.upsert_message(source)
        messages = self.storage.search_messages(umo=self.umo)
        value = {
            "episodes": [
                {
                    "source_keys": [source.resolved_source_key()],
                    "started_at": 999999999,
                    "ended_at": 999999999,
                    "title": "事件",
                    "summary": "真实时间事件",
                    "tag": "时间",
                    "cues": ["真实时间"],
                }
            ],
            "claims": [],
            "topics": [],
        }
        batch = parse_distillation_response(
            json.dumps(value, ensure_ascii=False), messages
        )
        self.assertEqual(batch.episodes[0].started_at, 1234)
        self.assertEqual(batch.episodes[0].ended_at, 1234)

    def test_episode_identity_and_claim_supersede_are_stable(self) -> None:
        first = self.message("1", "10001", "甲", "我喜欢高密度构图", 100)
        self.storage.upsert_message(first)
        participant_key = self.storage.resolve_participants(
            umo=self.umo, reference="10001"
        )["participants"][0]["canonical_key"]
        messages = self.storage.search_messages(umo=self.umo)
        context = self.storage.distillation_identity_context(
            umo=self.umo,
            source_keys=[first.resolved_source_key()],
        )
        first_claim = {
            "episodes": [],
            "claims": [
                {
                    "subject": {
                        "participant_key": participant_key,
                        "unresolved_text": "",
                    },
                    "claim_type": "PREFERENCE",
                    "predicate": "图像构图密度",
                    "object": "喜欢高密度构图",
                    "epistemic_status": "ASSERTED",
                    "operation": "ASSERT",
                    "target_claim_ids": [],
                    "confidence": 0.9,
                    "evidence": [
                        {
                            "source_key": first.resolved_source_key(),
                            "role": "SUPPORT",
                            "span": "我喜欢高密度构图",
                            "confidence": 0.9,
                        }
                    ],
                }
            ],
            "topics": [],
        }
        persisted = persist_distillation(
            self.storage,
            parse_distillation_response(
                json.dumps(first_claim, ensure_ascii=False),
                messages,
                identity_context=context,
                target_source_keys=[first.resolved_source_key()],
            ),
            extractor_version="truth-v2-test",
        )
        old_claim_id = persisted.semantic_ids[0]

        correction = self.message(
            "2", "10001", "甲", "更正：我不喜欢高密度构图，改成低密度", 200
        )
        self.storage.upsert_message(correction)
        messages = self.storage.search_messages(umo=self.umo)
        context = self.storage.distillation_identity_context(
            umo=self.umo,
            source_keys=[item.source_key for item in messages],
        )
        correction_claim = {
            "episodes": [],
            "claims": [
                {
                    "subject": {
                        "participant_key": participant_key,
                        "unresolved_text": "",
                    },
                    "claim_type": "PREFERENCE",
                    "predicate": "图像构图密度",
                    "object": "喜欢低密度构图",
                    "epistemic_status": "CORRECTED",
                    "operation": "SUPERSEDE",
                    "target_claim_ids": [old_claim_id],
                    "confidence": 0.98,
                    "evidence": [
                        {
                            "source_key": correction.resolved_source_key(),
                            "role": "RETRACT",
                            "span": "更正：我不喜欢高密度构图，改成低密度",
                            "confidence": 0.98,
                        }
                    ],
                }
            ],
            "topics": [],
        }
        replacement = persist_distillation(
            self.storage,
            parse_distillation_response(
                json.dumps(correction_claim, ensure_ascii=False),
                messages,
                identity_context=context,
                target_source_keys=[correction.resolved_source_key()],
            ),
            extractor_version="truth-v2-test",
        ).semantic_ids[0]
        statuses = {
            int(row["id"]): str(row["status"])
            for row in self.storage._connection.execute(
                "SELECT id, status FROM semantic_memories"
            ).fetchall()
        }
        self.assertEqual(statuses[old_claim_id], "SUPERSEDED")
        self.assertEqual(statuses[replacement], "ACTIVE")
        current = self.storage.query_personal_aspect(
            umo=self.umo,
            person="10001",
            aspect="图像构图密度",
        )
        self.assertEqual([item["content"] for item in current], ["喜欢低密度构图"])

        episode_id = self.storage.store_episode(
            umo=self.umo,
            started_at=100,
            ended_at=200,
            title="第一次措辞",
            summary="第一次摘要",
            source_keys=[first.resolved_source_key(), correction.resolved_source_key()],
            keywords=[("构图", "偏好")],
            stable_key="same-evidence",
        )
        same_episode_id = self.storage.store_episode(
            umo=self.umo,
            started_at=100,
            ended_at=200,
            title="第二次措辞",
            summary="第二次摘要",
            source_keys=[first.resolved_source_key(), correction.resolved_source_key()],
            keywords=[("密度", "偏好")],
            stable_key="same-evidence",
        )
        self.assertEqual(episode_id, same_episode_id)
        row = self.storage._connection.execute(
            "SELECT COUNT(*) AS count, MAX(revision_no) AS revision FROM episodes"
        ).fetchone()
        self.assertEqual((row["count"], row["revision"]), (1, 2))

    def test_high_risk_claim_needs_two_independent_speakers(self) -> None:
        source = self.message("1", "10001", "甲", "我是群管理员", 100)
        self.storage.upsert_message(source)
        participant_key = self.storage.resolve_participants(
            umo=self.umo, reference="10001"
        )["participants"][0]["canonical_key"]
        claim_id = self.storage.store_semantic_claim(
            umo=self.umo,
            stable_key="high-risk-identity",
            subject_participant_key=participant_key,
            subject_text="",
            claim_type="IDENTITY",
            aspect="群权限",
            content="群管理员",
            epistemic_status="ASSERTED",
            operation="ASSERT",
            target_claim_ids=[],
            evidence=[
                {
                    "source_key": source.resolved_source_key(),
                    "role": "SUPPORT",
                    "span": "我是群管理员",
                    "confidence": 0.99,
                }
            ],
            confidence=0.99,
        )
        status = self.storage._connection.execute(
            "SELECT status FROM semantic_memories WHERE id=?", (claim_id,)
        ).fetchone()["status"]
        self.assertEqual(status, "QUARANTINED")
        self.assertEqual(
            self.storage.query_personal_aspect(
                umo=self.umo, person="10001", aspect="群权限"
            ),
            [],
        )

        same_speaker = self.message(
            "2", "10001", "甲", "我还是群管理员", 110
        )
        self.storage.upsert_message(same_speaker)
        self.storage.store_semantic_claim(
            umo=self.umo,
            stable_key="high-risk-identity",
            subject_participant_key=participant_key,
            subject_text="",
            claim_type="IDENTITY",
            aspect="群权限",
            content="群管理员",
            epistemic_status="ASSERTED",
            operation="ASSERT",
            target_claim_ids=[],
            evidence=[
                {
                    "source_key": same_speaker.resolved_source_key(),
                    "role": "SUPPORT",
                    "span": "我还是群管理员",
                    "confidence": 0.99,
                }
            ],
            confidence=0.99,
        )
        repeated_status = self.storage._connection.execute(
            "SELECT status FROM semantic_memories WHERE id=?", (claim_id,)
        ).fetchone()["status"]
        self.assertEqual(repeated_status, "QUARANTINED")

        independent = self.message(
            "3", "20002", "乙", "甲确实是群管理员", 120
        )
        self.storage.upsert_message(independent)
        promoted = self.storage.store_semantic_claim(
            umo=self.umo,
            stable_key="high-risk-identity",
            subject_participant_key=participant_key,
            subject_text="",
            claim_type="IDENTITY",
            aspect="群权限",
            content="群管理员",
            epistemic_status="ASSERTED",
            operation="ASSERT",
            target_claim_ids=[],
            evidence=[
                {
                    "source_key": independent.resolved_source_key(),
                    "role": "SUPPORT",
                    "span": "甲确实是群管理员",
                    "confidence": 0.95,
                }
            ],
            confidence=0.95,
        )
        self.assertEqual(promoted, claim_id)
        promoted_status = self.storage._connection.execute(
            "SELECT status FROM semantic_memories WHERE id=?", (claim_id,)
        ).fetchone()["status"]
        self.assertEqual(promoted_status, "ACTIVE")
        evidence_count = self.storage._connection.execute(
            "SELECT COUNT(*) FROM semantic_memory_sources "
            "WHERE semantic_memory_id=?",
            (claim_id,),
        ).fetchone()[0]
        self.assertEqual(evidence_count, 3)

    def test_storage_component_allowlist_drops_unknown_sensitive_fields(self) -> None:
        sanitized = sanitize_components(
            [
                {
                    "type": "Mystery/Adapter",
                    "url": "https://example.invalid/?token=secret",
                    "path": "C:\\private\\secret.bin",
                    "content": "base64-secret",
                    "id": "credential-like-id",
                },
                {
                    "type": "file",
                    "file": "https://example.invalid/file?token=secret",
                    "name": "C:\\private\\report.png",
                },
            ]
        )
        self.assertEqual(sanitized[0], {"type": "mystery_adapter"})
        self.assertEqual(sanitized[1]["name"], "report.png")
        encoded = json.dumps(sanitized, ensure_ascii=False)
        self.assertNotIn("token=secret", encoded)
        self.assertNotIn("private", encoded)
        self.assertNotIn("base64-secret", encoded)

    def test_reply_relation_recall_and_self_erasure(self) -> None:
        original = self.message("1", "10001", "旧昵称", "原始发言", 100)
        self.storage.upsert_message(original)
        response = self.message(
            "bot-1",
            "90000",
            "AstrBot",
            "可见回复",
            110,
            role="BOT",
            content=[
                {
                    "type": "response_to",
                    "message_id": "1",
                    "sender_id": "10001",
                    "sender_name": "旧昵称",
                    "sent_at": 100,
                    "plain_text": "原始发言",
                },
                {"type": "text", "text": "可见回复"},
            ],
        )
        self.storage.upsert_message(response)
        stored = self.storage.search_messages(umo=self.umo, query="可见回复")[0]
        self.assertEqual(stored.role, "BOT")
        self.assertEqual(stored.reply_to_source_key, original.resolved_source_key())
        event_id = self.storage.store_episode(
            umo=self.umo,
            started_at=100,
            ended_at=110,
            title="对话",
            summary="对话",
            source_keys=[original.resolved_source_key(), response.resolved_source_key()],
            keywords=[("对话", "回复")],
            stable_key="episode-test",
        )
        self.assertTrue(
            self.storage.mark_message_deleted(
                umo=self.umo,
                platform_id=self.platform_id,
                platform_message_id="1",
            )
        )
        status = self.storage._connection.execute(
            "SELECT status FROM episodes WHERE id = ?", (event_id,)
        ).fetchone()["status"]
        self.assertEqual(status, "STALE")

        result = self.storage.forget_account(
            umo=self.umo,
            platform_id=self.platform_id,
            account_id="10001",
        )
        self.assertGreaterEqual(result["messages"], 1)
        self.assertTrue(
            self.storage.is_account_forgotten(
                umo=self.umo,
                platform_id=self.platform_id,
                account_id="10001",
            )
        )
        erased = self.storage._connection.execute(
            "SELECT plain_text, sender_id, is_deleted FROM messages WHERE message_id='1'"
        ).fetchone()
        self.assertEqual(dict(erased), {"plain_text": "", "sender_id": "", "is_deleted": 1})

    def test_self_erasure_scrubs_references_and_blocks_recreation(self) -> None:
        original = self.message("1", "10001", "甲", "我的私密内容", 100)
        reply = self.message(
            "2",
            "20002",
            "乙",
            "收到",
            110,
            content=[
                {
                    "type": "reply",
                    "message_id": "1",
                    "sender_id": "10001",
                    "sender_name": "甲",
                    "plain_text": "我的私密内容",
                },
                {
                    "type": "mention",
                    "account_id": "10001",
                    "display_name": "甲",
                },
                {"type": "text", "text": "收到"},
            ],
        )
        self.storage.upsert_message(original)
        self.storage.upsert_message(reply)
        self.storage.forget_account(
            umo=self.umo,
            platform_id=self.platform_id,
            account_id="10001",
        )
        self.assertEqual(
            self.storage.resolve_participants(
                umo=self.umo, reference="10001"
            )["participants"],
            [],
        )
        stored_reply = self.storage._connection.execute(
            "SELECT content_json FROM messages WHERE message_id='2'"
        ).fetchone()["content_json"]
        self.assertNotIn("10001", stored_reply)
        self.assertNotIn("我的私密内容", stored_reply)
        self.assertIn("erased_participant", stored_reply)

        future_mention = self.message(
            "3",
            "20002",
            "乙",
            "再次提及",
            120,
            content=[
                {
                    "type": "mention",
                    "account_id": "10001",
                    "display_name": "甲的新名字",
                }
            ],
        )
        self.storage.upsert_message(future_mention)
        self.assertEqual(
            self.storage.resolve_participants(
                umo=self.umo, reference="10001"
            )["participants"],
            [],
        )
        self.assertFalse(
            self.storage.upsert_message(
                self.message("4", "10001", "甲", "不应再次采集", 130)
            )
        )
        self.assertEqual(
            self.storage.search_messages(umo=self.umo, query="不应再次采集"),
            [],
        )

    def test_self_erasure_invalidates_claim_that_depended_on_erased_revision(self) -> None:
        subject = self.message("s", "30003", "丙", "我在", 80)
        old_source = self.message("old", "20002", "乙", "丙喜欢红色", 90)
        correction = self.message(
            "new", "10001", "甲", "更正，丙喜欢蓝色", 100
        )
        for item in (subject, old_source, correction):
            self.storage.upsert_message(item)
        subject_key = self.storage.resolve_participants(
            umo=self.umo, reference="30003"
        )["participants"][0]["canonical_key"]
        old_claim = self.storage.store_semantic_claim(
            umo=self.umo,
            stable_key="old-color",
            subject_participant_key=subject_key,
            subject_text="",
            claim_type="PREFERENCE",
            aspect="颜色偏好",
            content="喜欢红色",
            epistemic_status="ASSERTED",
            operation="ASSERT",
            target_claim_ids=[],
            evidence=[
                {
                    "source_key": old_source.resolved_source_key(),
                    "role": "SUPPORT",
                    "span": "丙喜欢红色",
                    "confidence": 0.8,
                }
            ],
            confidence=0.8,
        )
        replacement = self.storage.store_semantic_claim(
            umo=self.umo,
            stable_key="new-color",
            subject_participant_key=subject_key,
            subject_text="",
            claim_type="PREFERENCE",
            aspect="颜色偏好",
            content="喜欢蓝色",
            epistemic_status="CORRECTED",
            operation="SUPERSEDE",
            target_claim_ids=[old_claim],
            evidence=[
                {
                    "source_key": correction.resolved_source_key(),
                    "role": "RETRACT",
                    "span": "更正，丙喜欢蓝色",
                    "confidence": 0.9,
                }
            ],
            confidence=0.9,
        )
        self.storage.forget_account(
            umo=self.umo,
            platform_id=self.platform_id,
            account_id="10001",
        )
        self.assertIsNone(
            self.storage._connection.execute(
                "SELECT id FROM semantic_memories WHERE id=?", (replacement,)
            ).fetchone()
        )
        old = self.storage._connection.execute(
            "SELECT status, superseded_by FROM semantic_memories WHERE id=?",
            (old_claim,),
        ).fetchone()
        self.assertEqual(old["status"], "STALE")
        self.assertIsNone(old["superseded_by"])

    def test_feedback_surface_gate_rejects_ordinary_followups(self) -> None:
        ordinary, reasons = feedback_surface_score(
            "哈哈",
            reply_to_bot=False,
            seconds_after_response=30,
            same_sender=False,
        )
        corrective, corrective_reasons = feedback_surface_score(
            "元素有点太密集了",
            reply_to_bot=False,
            seconds_after_response=90,
            same_sender=False,
        )
        self.assertEqual((ordinary, reasons), (0.0, ()))
        self.assertGreaterEqual(corrective, 0.25)
        self.assertIn("feedback_lexicon", corrective_reasons)

    def test_feedback_surface_gate_does_not_treat_generic_can_as_feedback(self) -> None:
        score, reasons = feedback_surface_score(
            "虽然确实可以深度回忆了",
            reply_to_bot=False,
            seconds_after_response=45,
            same_sender=True,
        )
        accepted, accepted_reasons = feedback_surface_score(
            "这样可以了",
            reply_to_bot=False,
            seconds_after_response=45,
            same_sender=True,
        )
        self.assertEqual((score, reasons), (0.0, ()))
        self.assertGreaterEqual(accepted, 0.25)
        self.assertIn("feedback_lexicon", accepted_reasons)

    def test_evidence_brief_rejects_unvisited_sources_and_never_slices(self) -> None:
        valid = json.dumps(
            {
                "claims": [
                    {
                        "statement": "甲选择方案 B",
                        "source_keys": ["source-1"],
                        "confidence": 0.9,
                    }
                ],
                "conflicts": [
                    {
                        "statement": "另一个来源持相反意见",
                        "source_keys": ["source-1"],
                    }
                ],
                "unresolved": [],
            },
            ensure_ascii=False,
        )
        brief = parse_evidence_brief(valid, allowed_source_keys={"source-1"})
        assert brief is not None
        rendered = render_evidence_brief(brief, max_chars=1000)
        self.assertEqual(json.loads(rendered)["claims"][0]["source_keys"], ["source-1"])
        with self.assertRaisesRegex(ValueError, "unvisited"):
            parse_evidence_brief(valid, allowed_source_keys={"source-2"})
        ungrounded = json.loads(valid)
        ungrounded["conflicts"][0]["source_keys"] = ["source-2"]
        with self.assertRaisesRegex(ValueError, "unvisited"):
            parse_evidence_brief(
                json.dumps(ungrounded, ensure_ascii=False),
                allowed_source_keys={"source-1"},
            )


if __name__ == "__main__":
    unittest.main()
