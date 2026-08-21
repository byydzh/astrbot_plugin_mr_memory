from __future__ import annotations

import asyncio
import sqlite3
import time
import unittest
import uuid
from pathlib import Path

from mr_memory.models import NormalizedMessage
from mr_memory.service import MemoryService
from mr_memory.snapshot import RequestSnapshot
from mr_memory.storage import MemoryStorage


class LayeredStorageTests(unittest.TestCase):
    UMO = "shadow:GroupMessage:layered"

    def setUp(self) -> None:
        test_root = Path.cwd() / ".dev" / "test-tmp"
        test_root.mkdir(parents=True, exist_ok=True)
        self.database_path = test_root / f"{uuid.uuid4().hex}.db"
        self.storage = MemoryStorage(self.database_path)

    def tearDown(self) -> None:
        self.storage.close()
        for suffix in ("", "-wal", "-shm"):
            Path(f"{self.database_path}{suffix}").unlink(missing_ok=True)

    @classmethod
    def message(
        cls,
        message_id: str,
        text: str,
        *,
        sent_at: int,
        sender_id: str = "user-a",
        content: list[dict[str, object]] | None = None,
    ) -> NormalizedMessage:
        return NormalizedMessage(
            platform="aiocqhttp",
            platform_id="shadow",
            umo=cls.UMO,
            group_id="layered",
            message_id=message_id,
            sender_id=sender_id,
            sender_name=sender_id,
            sent_at=sent_at,
            plain_text=text,
            content=content or [{"type": "plain", "text": text}],
        )

    def capture(
        self,
        *,
        request: NormalizedMessage,
        cutoff_at: int,
    ) -> dict[str, object]:
        return self.storage.capture_request_snapshot(
            umo=self.UMO,
            cutoff_at=cutoff_at,
            query=request.plain_text,
            context={"request": request.message_id},
            request_source_key=request.resolved_source_key(),
            sender_participant_key=f"shadow:{request.sender_id}",
        )

    def test_schema_16_contains_layered_and_long_graph_tables(self) -> None:
        version = self.storage._connection.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()
        self.assertEqual(version["value"], "16")
        expected = {
            "revision_heads",
            "request_snapshots",
            "evidence_pack_cache",
            "memory_certificates",
            "certificate_dependencies",
            "reconstruction_jobs",
            "invalidation_events",
            "derived_claim_revisions",
            "derived_edge_revisions",
            "derived_edge_evidence_groups",
            "behavior_policy_revisions",
            "mutation_proposals",
        }
        actual = {
            str(row["name"])
            for row in self.storage._connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        self.assertTrue(expected <= actual)

        with self.storage._connection:
            edge = self.storage._connection.execute(
                """
                INSERT INTO derived_edge_revisions(
                    umo, stable_key, revision_no, source_key, relation_key,
                    target_key, source_group_hash
                ) VALUES (?, 'edge:x', 1, 'a', 'rel', 'b', 'group-x')
                """,
                (self.UMO,),
            )
            values = (int(edge.lastrowid), self.UMO, "group-x")
            self.storage._connection.execute(
                """
                INSERT INTO derived_edge_evidence_groups(
                    edge_revision_id, umo, source_group_hash
                ) VALUES (?, ?, ?)
                """,
                values,
            )
            with self.assertRaises(sqlite3.IntegrityError):
                self.storage._connection.execute(
                    """
                    INSERT INTO derived_edge_evidence_groups(
                        edge_revision_id, umo, source_group_hash
                    ) VALUES (?, ?, ?)
                    """,
                    values,
                )

    def test_revision_heads_follow_message_identity_graph_and_deletion_writes(self) -> None:
        original = self.message("one", "内容", sent_at=100)
        self.storage.upsert_message(original)
        heads = self.storage.revision_vector(umo=self.UMO)["data"]
        self.assertEqual(heads["message"], 1)
        self.assertEqual(heads.get("identity", 0), 0)

        renamed = NormalizedMessage(
            platform=original.platform,
            platform_id=original.platform_id,
            umo=original.umo,
            group_id=original.group_id,
            message_id=original.message_id,
            sender_id=original.sender_id,
            sender_name="新昵称",
            sent_at=original.sent_at,
            plain_text=original.plain_text,
            content=original.content,
        )
        self.storage.upsert_message(renamed)
        heads = self.storage.revision_vector(umo=self.UMO)["data"]
        self.assertEqual(heads["message"], 1)
        self.assertEqual(heads["identity"], 1)

        second = self.message("two", "第二条", sent_at=110)
        self.storage.upsert_message(second)
        heads = self.storage.revision_vector(umo=self.UMO)["data"]
        self.assertEqual(heads["identity"], 1)
        self.storage.store_episode(
            umo=self.UMO,
            started_at=100,
            ended_at=110,
            title="事件",
            summary="两条证据",
            source_keys=[original.resolved_source_key(), second.resolved_source_key()],
            keywords=[("事件", "测试")],
        )
        heads = self.storage.revision_vector(umo=self.UMO)["data"]
        self.assertEqual(heads["graph"], 1)
        self.storage.mark_message_deleted(
            umo=self.UMO,
            platform_id=second.platform_id,
            platform_message_id=second.message_id,
            deleted_at=120,
        )
        heads = self.storage.revision_vector(umo=self.UMO)["data"]
        self.assertEqual(heads["deletion"], 1)
        self.assertEqual(heads["message"], 3)

    def test_future_append_does_not_stale_identity_but_history_edit_does(self) -> None:
        target = self.message(
            "target-old",
            "旧消息",
            sent_at=100,
            sender_id="target",
        )
        request = self.message(
            "request",
            "/chat @target 她是谁，是不是我",
            sent_at=110,
            sender_id="speaker",
            content=[
                {"type": "mention", "account_id": "target", "display_name": "目标"},
                {"type": "plain", "text": "她是谁，是不是我"},
            ],
        )
        self.storage.upsert_message(target)
        self.storage.upsert_message(request)
        snapshot = self.capture(request=request, cutoff_at=111)

        self.storage.upsert_message(
            self.message(
                "target-future",
                "[图片]",
                sent_at=112,
                sender_id="target",
                content=[{"type": "image", "reference_sha256": "a" * 64}],
            )
        )
        self.assertEqual(
            self.storage.revision_vector(umo=self.UMO)["data"].get("identity", 0),
            int(snapshot["data_revision"]["identity"]),
        )

        edited_target = NormalizedMessage(
            platform=target.platform,
            platform_id=target.platform_id,
            umo=target.umo,
            group_id=target.group_id,
            message_id=target.message_id,
            sender_id=target.sender_id,
            sender_name="目标的新名",
            sent_at=target.sent_at,
            plain_text=target.plain_text,
            content=target.content,
        )
        self.storage.upsert_message(edited_target)
        self.assertNotEqual(
            self.storage.revision_vector(umo=self.UMO)["data"]["identity"],
            int(snapshot["data_revision"]["identity"]),
        )

    def test_snapshot_excludes_current_source_with_same_second_timestamp(self) -> None:
        previous = self.message("previous", "上一条", sent_at=200)
        request = self.message("request", "当前问题", sent_at=200)
        later = self.message("later", "同秒但稍后入库", sent_at=200)
        for message in (previous, request, later):
            self.storage.upsert_message(message)

        snapshot_value = self.capture(request=request, cutoff_at=201)
        snapshot = RequestSnapshot.from_value(
            {
                key: snapshot_value[key]
                for key in (
                    "snapshot_id",
                    "umo",
                    "scope_sha256",
                    "cutoff_at",
                    "message_upper_bound",
                    "request_source_key",
                    "sender_participant_key",
                    "reply_source_key",
                    "query_sha256",
                    "context_sha256",
                    "data_revision",
                    "inference_revision",
                    "captured_at",
                )
            }
        )
        ids = {
            str(row["source_key"]): int(row["id"])
            for row in self.storage._connection.execute(
                "SELECT id, source_key FROM messages"
            ).fetchall()
        }
        self.assertEqual(
            snapshot.message_upper_bound,
            ids[previous.resolved_source_key()],
        )
        self.assertTrue(
            snapshot.allows_evidence(
                umo=self.UMO,
                sent_at=previous.sent_at,
                message_row_id=ids[previous.resolved_source_key()],
                source_key=previous.resolved_source_key(),
            )
        )
        self.assertFalse(
            snapshot.allows_evidence(
                umo=self.UMO,
                sent_at=request.sent_at,
                message_row_id=ids[request.resolved_source_key()],
                source_key=request.resolved_source_key(),
            )
        )
        audit = self.storage.audit_snapshot_sources(
            snapshot_id=snapshot.snapshot_id,
            umo=self.UMO,
            source_keys=(
                previous.resolved_source_key(),
                request.resolved_source_key(),
                later.resolved_source_key(),
            ),
        )
        self.assertFalse(audit["valid"])
        self.assertEqual(audit["accepted_source_keys"], [previous.resolved_source_key()])
        reasons = {item["reason"] for item in audit["violations"]}
        self.assertEqual(
            reasons,
            {"CURRENT_REQUEST_SOURCE", "AFTER_MESSAGE_UPPER_BOUND"},
        )

    def test_packet_and_media_reads_obey_cutoff_and_message_upper_bound(self) -> None:
        fingerprint = "a" * 64
        previous_one = self.message(
            "previous-1",
            "第一张旧图",
            sent_at=200,
            content=[{"type": "image", "reference_sha256": fingerprint}],
        )
        previous_two = self.message(
            "previous-2",
            "第二张旧图",
            sent_at=200,
            sender_id="user-b",
            content=[{"type": "image", "reference_sha256": fingerprint}],
        )
        request = self.message(
            "request",
            "当前这张图不能成为证据",
            sent_at=200,
            sender_id="user-c",
            content=[{"type": "image", "reference_sha256": fingerprint}],
        )
        for message in (previous_one, previous_two, request):
            self.storage.upsert_message(message)
        snapshot = self.capture(request=request, cutoff_at=201)
        bound = int(snapshot["message_upper_bound"])
        self.assertEqual(
            len(
                self.storage.resolve_participants(
                    umo=self.UMO,
                    reference="user-a",
                    before_sent_at=201,
                    message_upper_bound=bound,
                )["participants"]
            ),
            1,
        )
        self.assertEqual(
            self.storage.resolve_participants(
                umo=self.UMO,
                reference="user-c",
                before_sent_at=201,
                message_upper_bound=bound,
            )["participants"],
            [],
        )

        old_episode = self.storage.store_episode(
            umo=self.UMO,
            started_at=200,
            ended_at=200,
            title="旧证据",
            summary="仅由请求前消息组成",
            source_keys=[
                previous_one.resolved_source_key(),
                previous_two.resolved_source_key(),
            ],
            keywords=[("旧图", "反馈")],
        )
        mixed_episode = self.storage.store_episode(
            umo=self.UMO,
            started_at=200,
            ended_at=200,
            title="污染证据",
            summary="包含当前请求，不应暴露",
            source_keys=[request.resolved_source_key()],
            keywords=[("当前", "反馈")],
        )
        old_memory = self.storage.store_semantic_memory(
            umo=self.UMO,
            person="user-a",
            aspect="偏好",
            content="喜欢旧图",
            source_key=previous_one.resolved_source_key(),
            confidence=0.8,
        )
        current_memory = self.storage.store_semantic_memory(
            umo=self.UMO,
            person="user-c",
            aspect="偏好",
            content="由当前请求生成，不应暴露",
            source_key=request.resolved_source_key(),
            confidence=0.8,
        )

        packet = self.storage.reconstruction_evidence_packet(
            umo=self.UMO,
            candidates={
                "episodes": [{"id": old_episode}, {"id": mixed_episode}],
                "semantic_memories": [
                    {"id": old_memory, "content": "喜欢旧图"},
                    {"id": current_memory, "content": "当前请求污染"},
                ],
            },
            before_sent_at=201,
            message_upper_bound=bound,
        )
        self.assertEqual(
            [item["id"] for item in packet["expanded_episodes"]],
            [old_episode],
        )
        encoded_packet = str(packet)
        self.assertIn(previous_one.resolved_source_key(), encoded_packet)
        self.assertNotIn(request.resolved_source_key(), encoded_packet)
        self.assertNotIn("当前请求污染", encoded_packet)

        patterns = self.storage.query_media_patterns(
            umo=self.UMO,
            fingerprints=[fingerprint],
            min_observations=2,
            before_sent_at=201,
            message_upper_bound=bound,
        )
        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0]["observation_count"], 2)
        encoded_patterns = str(patterns)
        self.assertNotIn(request.resolved_source_key(), encoded_patterns)
        self.assertNotIn("当前这张图不能成为证据", encoded_patterns)

    def test_cache_certificate_invalidation_and_job_lifecycle(self) -> None:
        previous = self.message("previous", "证据", sent_at=100)
        request = self.message("request", "问题", sent_at=110)
        for message in (previous, request):
            self.storage.upsert_message(message)
        snapshot = self.capture(request=request, cutoff_at=111)
        snapshot_id = str(snapshot["snapshot_id"])
        packet = self.storage.put_evidence_pack_cache(
            cache_key="packet:one",
            umo=self.UMO,
            snapshot_id=snapshot_id,
            packet={"sources": [previous.resolved_source_key()]},
            source_keys=[previous.resolved_source_key()],
        )
        packet_hash = str(packet["packet_hash"])
        certificate = self.storage.put_memory_certificate(
            certificate_key="certificate:one",
            umo=self.UMO,
            snapshot_id=snapshot_id,
            packet_hash=packet_hash,
            certificate_status="CERTIFIED",
            certificate={"answer": "证据支持"},
            dependencies=[{"type": "claim", "key": "claim:a", "revision": 1}],
        )
        self.assertEqual(certificate["dependencies"][0]["dependency_revision"], 1)
        invalidated = self.storage.invalidate_cached_memory(
            umo=self.UMO,
            dependency_type="claim",
            dependency_key="claim:a",
            revision=2,
        )
        self.assertEqual(invalidated["certificates"], 1)
        self.assertIsNone(
            self.storage.get_memory_certificate(
                umo=self.UMO,
                certificate_key="certificate:one",
            )
        )
        self.assertIsNotNone(
            self.storage.get_evidence_pack_cache(
                umo=self.UMO,
                cache_key="packet:one",
            )
        )

        job = self.storage.enqueue_reconstruction_job(
            job_key="reconstruct:one",
            umo=self.UMO,
            snapshot_id=snapshot_id,
            cache_key="packet:one",
            requested_level="L2",
            contract={"round": 0},
            available_at=0,
        )
        duplicate = self.storage.enqueue_reconstruction_job(
            job_key="reconstruct:one",
            umo=self.UMO,
            snapshot_id=snapshot_id,
            available_at=0,
        )
        self.assertEqual(job["job_id"], duplicate["job_id"])
        claimed = self.storage.claim_reconstruction_job(
            job_id=str(job["job_id"]),
            umo=self.UMO,
            now=10,
            lease_seconds=10,
        )
        self.assertIsNotNone(claimed)
        assert claimed is not None
        self.assertEqual(claimed["status"], "RUNNING")
        self.assertEqual(claimed["lease_until"], 40)
        completed = self.storage.finish_reconstruction_job(
            job_id=str(job["job_id"]),
            umo=self.UMO,
            status="COMPLETED",
            last_result_hash="b" * 64,
        )
        self.assertEqual(completed["status"], "COMPLETED")
        self.assertIsNotNone(completed["finished_at"])

    def test_online_budget_includes_reader_and_all_eccr_phases(self) -> None:
        for index, (phase, tokens) in enumerate(
            (
                ("certificate_reader", 11),
                ("eccr_route", 13),
                ("eccr_round_2", 17),
                ("feedback_maintenance", 19),
            )
        ):
            run_id = f"budget-{index}"
            self.storage.start_experiment(
                run_id=run_id,
                umo=self.UMO,
                experiment_type="budget-test",
            )
            self.storage.record_llm_usage(
                run_id=run_id,
                phase=phase,
                input_other=tokens,
            )
        self.assertEqual(
            self.storage.private_token_usage_since(
                umo=self.UMO,
                since=0,
                budget_class="online",
            ),
            41,
        )
        reset = self.storage.reset_token_budget(
            umo=self.UMO,
            budget_class="online",
            reason="wildcard-test",
        )
        self.assertEqual(int(reset["usage_event_id"]), 3)
        self.assertEqual(
            self.storage.private_token_usage_since(
                umo=self.UMO,
                since=0,
                budget_class="online",
            ),
            0,
        )

    def test_snapshot_bound_hides_future_identity_sources_and_derived_events(
        self,
    ) -> None:
        old = self.message(
            "old", "旧证据", sent_at=100, sender_id="person-a"
        )
        request = self.message(
            "request", "当前请求", sent_at=200, sender_id="requester"
        )
        future = NormalizedMessage(
            platform="aiocqhttp",
            platform_id="shadow",
            umo=self.UMO,
            group_id="layered",
            message_id="future",
            sender_id="person-a",
            sender_name="未来昵称",
            sent_at=200,
            plain_text="未来证据",
            content=[{"type": "plain", "text": "未来证据"}],
        )
        old = NormalizedMessage(
            platform=old.platform,
            platform_id=old.platform_id,
            umo=old.umo,
            group_id=old.group_id,
            message_id=old.message_id,
            sender_id=old.sender_id,
            sender_name="旧昵称",
            sent_at=old.sent_at,
            plain_text=old.plain_text,
            content=old.content,
        )
        for message in (old, request, future):
            self.storage.upsert_message(message)
        snapshot = self.capture(request=request, cutoff_at=201)
        bound = int(snapshot["message_upper_bound"])

        participant_key = self.storage.resolve_participants(
            umo=self.UMO, reference="person-a"
        )["participants"][0]["canonical_key"]
        old_memory = self.storage.store_semantic_claim(
            umo=self.UMO,
            stable_key="old-aspect",
            subject_participant_key=str(participant_key),
            subject_text="",
            claim_type="PREFERENCE",
            aspect="旧方面",
            content="只由旧证据支持",
            epistemic_status="ASSERTED",
            operation="ASSERT",
            target_claim_ids=[],
            evidence=[
                {
                    "source_key": old.resolved_source_key(),
                    "role": "SUPPORT",
                    "span": "旧证据",
                    "confidence": 0.8,
                }
            ],
            confidence=0.8,
        )
        future_memory = self.storage.store_semantic_claim(
            umo=self.UMO,
            stable_key="future-aspect",
            subject_participant_key=str(participant_key),
            subject_text="",
            claim_type="PREFERENCE",
            aspect="未来方面",
            content="不能泄漏",
            epistemic_status="ASSERTED",
            operation="ASSERT",
            target_claim_ids=[],
            evidence=[
                {
                    "source_key": future.resolved_source_key(),
                    "role": "SUPPORT",
                    "span": "未来证据",
                    "confidence": 0.8,
                }
            ],
            confidence=0.8,
        )
        self.assertGreater(future_memory, old_memory)

        old_episode = self.storage.store_episode(
            umo=self.UMO,
            started_at=100,
            ended_at=100,
            title="旧事件",
            summary="旧摘要",
            source_keys=[old.resolved_source_key()],
            keywords=[("人物", "变化")],
        )
        mixed_episode = self.storage.store_episode(
            umo=self.UMO,
            started_at=100,
            ended_at=200,
            title="混合事件",
            summary="包含未来证据",
            source_keys=[old.resolved_source_key(), future.resolved_source_key()],
            keywords=[("人物", "变化")],
        )
        topic_id = self.storage.store_topic(
            umo=self.UMO,
            name="昵称变化",
            summary="主题",
            event_ids=[old_episode, mixed_episode],
        )

        bounded_identity = self.storage.resolve_participants(
            umo=self.UMO,
            reference="person-a",
            before_sent_at=201,
            message_upper_bound=bound,
        )["participants"][0]
        self.assertEqual(bounded_identity["current_display_name"], "旧昵称")
        self.assertEqual(
            [item["alias"] for item in bounded_identity["aliases"]],
            ["旧昵称"],
        )
        self.assertEqual(
            self.storage.resolve_participants(
                umo=self.UMO,
                reference="未来昵称",
                before_sent_at=201,
                message_upper_bound=bound,
            )["participants"],
            [],
        )
        participant_id = int(bounded_identity["id"])
        expanded = self.storage.expand_seed_candidates(
            umo=self.UMO,
            matches=[
                {
                    "owner_type": "participant",
                    "owner_key": str(participant_id),
                    "score": 0.9,
                },
                {
                    "owner_type": "topic",
                    "owner_key": str(topic_id),
                    "score": 0.8,
                },
            ],
            before_sent_at=201,
            message_upper_bound=bound,
        )
        self.assertEqual(
            expanded["participants"][0]["current_display_name"], "旧昵称"
        )
        self.assertEqual(
            [item["alias"] for item in expanded["participants"][0]["aliases"]],
            ["旧昵称"],
        )
        self.assertEqual(expanded["topics"][0]["summary"], "")
        aspects = self.storage.query_personal_information(
            umo=self.UMO,
            person="person-a",
            before_sent_at=201,
            message_upper_bound=bound,
        )
        self.assertEqual([item["aspect_tag"] for item in aspects], ["旧方面"])
        detail = self.storage.query_personal_aspect(
            umo=self.UMO,
            person="person-a",
            aspect="旧方面",
            before_sent_at=201,
            message_upper_bound=bound,
        )
        self.assertEqual(detail[0]["source_keys"], [old.resolved_source_key()])
        self.assertEqual(
            self.storage.query_personal_aspect(
                umo=self.UMO,
                person="person-a",
                aspect="未来方面",
                before_sent_at=201,
                message_upper_bound=bound,
            ),
            [],
        )
        self.assertEqual(
            [
                item["id"]
                for item in self.storage.query_tag_events(
                    umo=self.UMO,
                    cue="人物",
                    tag="变化",
                    before_sent_at=201,
                    message_upper_bound=bound,
                )
            ],
            [old_episode],
        )
        self.assertIsNone(
            self.storage.query_conversation_time(
                umo=self.UMO,
                event_id=mixed_episode,
                before_sent_at=201,
                message_upper_bound=bound,
            )
        )
        self.assertEqual(
            self.storage.query_event_keywords(
                umo=self.UMO,
                event_id=mixed_episode,
                before_sent_at=201,
                message_upper_bound=bound,
            ),
            [],
        )
        self.assertEqual(
            [
                item["id"]
                for item in self.storage.query_topic_events(
                    umo=self.UMO,
                    topic="昵称变化",
                    before_sent_at=201,
                    message_upper_bound=bound,
                )
            ],
            [old_episode],
        )

    def test_reply_source_is_resolved_inside_snapshot_boundary(self) -> None:
        original = self.message("original", "原话", sent_at=100)
        current = self.message(
            "current",
            "回复",
            sent_at=110,
            sender_id="bot",
            content=[
                {
                    "type": "response_to",
                    "message_id": "original",
                    "sender_id": "user-a",
                    "sender_name": "user-a",
                    "sent_at": 100,
                    "plain_text": "原话",
                },
                {"type": "text", "text": "回复"},
            ],
        )
        for message in (original, current):
            self.storage.upsert_message(message)
        original_id = self.storage._connection.execute(
            "SELECT id FROM messages WHERE source_key=?",
            (original.resolved_source_key(),),
        ).fetchone()["id"]
        self.assertEqual(
            self.storage.reply_source_for_message(
                umo=self.UMO,
                source_key=current.resolved_source_key(),
                before_sent_at=111,
                message_upper_bound=int(original_id),
            ),
            original.resolved_source_key(),
        )
        self.assertEqual(
            self.storage.reply_source_for_message(
                umo=self.UMO,
                source_key=current.resolved_source_key(),
                before_sent_at=100,
                message_upper_bound=int(original_id),
            ),
            "",
        )
        visible = self.storage.message_for_source(
            umo=self.UMO,
            source_key=original.resolved_source_key(),
            before_sent_at=111,
            message_upper_bound=int(original_id),
        )
        self.assertIsNotNone(visible)
        assert visible is not None
        self.assertEqual(visible["plain_text"], "原话")
        self.assertEqual(visible["source_key"], original.resolved_source_key())
        self.assertIsNone(
            self.storage.message_for_source(
                umo=self.UMO,
                source_key=current.resolved_source_key(),
                before_sent_at=111,
                message_upper_bound=int(original_id),
            )
        )
        self.assertIsNone(
            self.storage.message_for_source(
                umo=self.UMO,
                source_key=original.resolved_source_key(),
                before_sent_at=100,
                message_upper_bound=int(original_id),
            )
        )

    def test_layered_runtime_recovery_and_cleanup(self) -> None:
        now = int(time.time())
        valid = self.storage.capture_request_snapshot(
            umo=self.UMO,
            cutoff_at=now - 100,
            query="valid",
            expires_at=now + 1000,
        )
        expiring = self.storage.capture_request_snapshot(
            umo=self.UMO,
            cutoff_at=now - 100,
            query="expired",
            expires_at=now + 100,
        )
        packet = self.storage.put_evidence_pack_cache(
            cache_key="expiring-packet",
            umo=self.UMO,
            snapshot_id=str(expiring["snapshot_id"]),
            packet={"sources": []},
            expires_at=now + 1000,
        )
        self.storage.put_memory_certificate(
            certificate_key="expiring-certificate",
            umo=self.UMO,
            snapshot_id=str(expiring["snapshot_id"]),
            packet_hash=str(packet["packet_hash"]),
            certificate_status="CERTIFIED",
            certificate={"answer": "temporary"},
            expires_at=now + 1000,
        )
        valid_job = self.storage.enqueue_reconstruction_job(
            job_key="valid-job",
            umo=self.UMO,
            snapshot_id=str(valid["snapshot_id"]),
            available_at=0,
        )
        retry_job = self.storage.enqueue_reconstruction_job(
            job_key="retry-job",
            umo=self.UMO,
            snapshot_id=str(valid["snapshot_id"]),
            available_at=0,
        )
        self.storage.update_reconstruction_job(
            job_id=str(retry_job["job_id"]),
            umo=self.UMO,
            status="RETRY",
            last_error="provider retry would have been orphaned",
        )
        expired_job = self.storage.enqueue_reconstruction_job(
            job_key="expired-job",
            umo=self.UMO,
            snapshot_id=str(expiring["snapshot_id"]),
            available_at=0,
        )
        self.assertIsNotNone(
            self.storage.claim_reconstruction_job(
                job_id=str(valid_job["job_id"]), umo=self.UMO, now=now
            )
        )
        recovered = self.storage.recover_layered_runtime(
            umo=self.UMO,
            now=now + 200,
        )
        self.assertEqual(
            recovered,
            {
                "expired_snapshots": 1,
                "interrupted_jobs": 3,
            },
        )
        self.assertEqual(
            self.storage.reconstruction_job(
                job_id=str(valid_job["job_id"]), umo=self.UMO
            )["status"],
            "STALE_RESTART",
        )
        self.assertEqual(
            self.storage.reconstruction_job(
                job_id=str(retry_job["job_id"]), umo=self.UMO
            )["status"],
            "STALE_RESTART",
        )
        self.assertEqual(
            self.storage.reconstruction_job(
                job_id=str(expired_job["job_id"]), umo=self.UMO
            )["status"],
            "STALE_RESTART",
        )
        self.assertIsNone(
            self.storage.get_evidence_pack_cache(
                cache_key="expiring-packet", umo=self.UMO, now=now + 200
            )
        )
        self.assertIsNone(
            self.storage.get_memory_certificate(
                certificate_key="expiring-certificate",
                umo=self.UMO,
                now=now + 200,
            )
        )
        cleaned = self.storage.cleanup_layered_runtime(
            umo=self.UMO,
            now=now + 300,
            terminal_retention_seconds=0,
        )
        self.assertEqual(cleaned["evidence_packs"], 1)
        self.assertEqual(cleaned["certificates"], 1)
        self.assertEqual(cleaned["terminal_jobs"], 3)
        self.assertEqual(cleaned["snapshots"], 1)
        self.assertIsNotNone(
            self.storage.request_snapshot(
                snapshot_id=str(valid["snapshot_id"]), umo=self.UMO
            )
        )

    def test_synthetic_schema_15_migrates_to_16_without_data_loss(self) -> None:
        database_path = self.database_path.with_name(f"{uuid.uuid4().hex}.db")
        storage = MemoryStorage(database_path)
        try:
            message = self.message("schema-15", "保留的数据", sent_at=100)
            storage.upsert_message(message)
        finally:
            storage.close()
        connection = sqlite3.connect(database_path)
        try:
            connection.execute("PRAGMA foreign_keys=OFF")
            for table in (
                "certificate_dependencies",
                "derived_edge_evidence_groups",
                "evidence_pack_cache",
                "memory_certificates",
                "reconstruction_jobs",
                "invalidation_events",
                "derived_claim_revisions",
                "derived_edge_revisions",
                "behavior_policy_revisions",
                "mutation_proposals",
                "request_snapshots",
                "revision_heads",
            ):
                connection.execute(f"DROP TABLE IF EXISTS {table}")
            connection.execute(
                "UPDATE schema_meta SET value='15' WHERE key='schema_version'"
            )
            connection.commit()
        finally:
            connection.close()

        migrated = MemoryStorage(database_path)
        try:
            version = migrated._connection.execute(
                "SELECT value FROM schema_meta WHERE key='schema_version'"
            ).fetchone()["value"]
            self.assertEqual(version, "16")
            self.assertEqual(
                migrated._connection.execute(
                    "SELECT COUNT(*) AS count FROM messages"
                ).fetchone()["count"],
                1,
            )
            self.assertEqual(
                migrated._connection.execute("PRAGMA quick_check").fetchone()[0],
                "ok",
            )
            tables = {
                row["name"]
                for row in migrated._connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            self.assertIn("request_snapshots", tables)
            self.assertIn("memory_certificates", tables)
            self.assertIn("derived_edge_evidence_groups", tables)
        finally:
            migrated.close()
            for suffix in ("", "-wal", "-shm"):
                Path(f"{database_path}{suffix}").unlink(missing_ok=True)

    def test_service_exposes_snapshot_and_cache_facade(self) -> None:
        request = self.message("request", "异步接口", sent_at=100)
        self.storage.upsert_message(request)
        service = MemoryService(self.storage)

        async def exercise() -> None:
            snapshot = await service.capture_request_snapshot(
                umo=self.UMO,
                cutoff_at=101,
                query=request.plain_text,
                request_source_key=request.resolved_source_key(),
            )
            loaded = await service.request_snapshot(
                umo=self.UMO,
                snapshot_id=str(snapshot["snapshot_id"]),
            )
            self.assertEqual(loaded["snapshot_id"], snapshot["snapshot_id"])
            audit = await service.audit_snapshot_sources(
                umo=self.UMO,
                snapshot_id=str(snapshot["snapshot_id"]),
                source_keys=[request.resolved_source_key()],
            )
            self.assertFalse(audit["valid"])

        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
