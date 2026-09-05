from __future__ import annotations

import asyncio
import sqlite3
import time
import unittest
import uuid
from pathlib import Path

import numpy as np

from mr_memory.embedding import encode_vector, normalize_vector
from mr_memory.models import NormalizedMessage
from mr_memory.retrieval_terms import fts_recall_terms, recall_coverage_terms, short_recall_terms
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

    def legacy_embedding_search(
        self,
        *,
        model: str,
        query_vector: list[float],
        owner_types: tuple[str, ...],
        limit: int,
        min_score: float = -1.0,
        before_sent_at: int | None = None,
        message_upper_bound: int | None = None,
    ) -> list[dict[str, object]]:
        """Reference the former fetchall/stable-sort exact search contract."""

        normalized_query = normalize_vector(query_vector)
        placeholders = ",".join("?" for _ in owner_types)
        with self.storage._lock:
            rows = self.storage._connection.execute(
                f"""
                SELECT owner_type, owner_key, dimensions, vector
                FROM memory_embeddings
                WHERE umo = ? AND model = ?
                  AND owner_type IN ({placeholders})
                """,
                (self.UMO, model, *owner_types),
            ).fetchall()
            visible_keys = self.storage._visible_memory_owner_keys_locked(
                umo=self.UMO,
                owner_types=owner_types,
                before_sent_at=before_sent_at,
                message_upper_bound=message_upper_bound,
            )
        query_array = np.asarray(normalized_query, dtype=np.float64)
        scored: list[dict[str, object]] = []
        for row in rows:
            owner_type = str(row["owner_type"])
            owner_key = str(row["owner_key"])
            lookup_key = owner_key.casefold() if owner_type == "cue" else owner_key
            if lookup_key not in visible_keys.get(owner_type, set()):
                continue
            dimensions = int(row["dimensions"])
            if dimensions != len(normalized_query):
                continue
            vector_blob = bytes(row["vector"])
            expected_bytes = dimensions * 4
            if dimensions <= 0 or len(vector_blob) != expected_bytes:
                raise ValueError(
                    "invalid embedding blob: "
                    f"dimensions={dimensions}, bytes={len(vector_blob)}"
                )
            stored = np.frombuffer(vector_blob, dtype="<f4", count=dimensions)
            score = float(np.dot(query_array, stored))
            if score < float(min_score):
                continue
            scored.append(
                {
                    "owner_type": owner_type,
                    "owner_key": owner_key,
                    "score": round(score, 6),
                }
            )
        safe_limit = max(1, min(100, int(limit)))
        scored.sort(key=lambda item: float(item["score"]), reverse=True)
        grouped = {owner_type: [] for owner_type in owner_types}
        for item in scored:
            grouped[str(item["owner_type"])].append(item)
        nonempty = [owner_type for owner_type in owner_types if grouped[owner_type]]
        if not nonempty:
            return []
        quota = max(1, safe_limit // len(nonempty))
        selected: list[dict[str, object]] = []
        selected_keys: set[tuple[str, str]] = set()
        for owner_type in nonempty:
            for item in grouped[owner_type][:quota]:
                selected.append(item)
                selected_keys.add((str(item["owner_type"]), str(item["owner_key"])))
        if len(selected) < safe_limit:
            for item in scored:
                key = (str(item["owner_type"]), str(item["owner_key"]))
                if key in selected_keys:
                    continue
                selected.append(item)
                selected_keys.add(key)
                if len(selected) >= safe_limit:
                    break
        selected.sort(key=lambda item: float(item["score"]), reverse=True)
        return selected[:safe_limit]

    def test_long_stream_is_bijective_and_snapshot_bounded(self) -> None:
        messages = [
            self.message(
                f"stream-{index:04d}",
                f"synthetic stream payload {index}",
                sent_at=1000 + index,
                sender_id=f"synthetic-account-{index % 7}",
            )
            for index in range(501)
        ]
        expected_keys = [message.resolved_source_key() for message in messages]
        for index, message in enumerate(messages):
            outcome = self.storage.upsert_message_with_outcome(message)
            self.assertEqual(outcome.status, "INSERTED")
            if index % 37 == 0:
                duplicate = self.storage.upsert_message_with_outcome(message)
                self.assertEqual(duplicate.status, "UNCHANGED")

        table_counts = {
            "messages": self.storage._connection.execute(
                "SELECT COUNT(*) FROM messages WHERE umo=?", (self.UMO,)
            ).fetchone()[0],
            "messages_fts": self.storage._connection.execute(
                "SELECT COUNT(*) FROM messages_fts"
            ).fetchone()[0],
            "message_processing": self.storage._connection.execute(
                """
                SELECT COUNT(*) FROM message_processing AS p
                JOIN messages AS m ON m.id=p.message_id WHERE m.umo=?
                """,
                (self.UMO,),
            ).fetchone()[0],
            "speaker_links": self.storage._connection.execute(
                """
                SELECT COUNT(*) FROM message_participants AS mp
                JOIN messages AS m ON m.id=mp.message_id
                WHERE m.umo=? AND mp.relation='SPEAKER'
                """,
                (self.UMO,),
            ).fetchone()[0],
            "alias_observations": self.storage._connection.execute(
                "SELECT COUNT(*) FROM participant_alias_observations WHERE umo=?",
                (self.UMO,),
            ).fetchone()[0],
        }
        self.assertEqual(set(table_counts.values()), {501}, table_counts)
        alias_count = self.storage._connection.execute(
            """
            SELECT COALESCE(SUM(a.observation_count), 0)
            FROM participant_aliases AS a
            JOIN participants AS p ON p.id=a.participant_id
            WHERE p.umo=? AND a.source_kind='observed'
            """,
            (self.UMO,),
        ).fetchone()[0]
        self.assertEqual(int(alias_count), 501)
        stored_keys = [
            str(row["source_key"])
            for row in self.storage._connection.execute(
                "SELECT source_key FROM messages WHERE umo=? ORDER BY sent_at,id",
                (self.UMO,),
            ).fetchall()
        ]
        self.assertEqual(stored_keys, expected_keys)

        first_batch = self.storage.next_distillation_batch(
            umo=self.UMO, limit=500, overlap=0
        )
        self.assertIsNotNone(first_batch)
        assert first_batch is not None
        self.assertEqual(len(first_batch.target_source_keys), 500)
        self.storage.finish_distillation_batch(work_item=first_batch)
        second_batch = self.storage.next_distillation_batch(
            umo=self.UMO, limit=500, overlap=8
        )
        self.assertIsNotNone(second_batch)
        assert second_batch is not None
        self.assertEqual(len(second_batch.target_source_keys), 1)
        first_targets = set(first_batch.target_source_keys)
        second_targets = set(second_batch.target_source_keys)
        self.assertFalse(first_targets & second_targets)
        self.assertEqual(first_targets | second_targets, set(expected_keys))
        overlap_keys = {
            message.source_key
            for message in second_batch.messages
            if message.source_key not in second_targets
        }
        self.assertTrue(overlap_keys)
        self.assertTrue(overlap_keys.issubset(first_targets))
        self.storage.finish_distillation_batch(work_item=second_batch)

        stream_upper_bound = int(
            self.storage._connection.execute(
                "SELECT MAX(id) FROM messages WHERE umo=?", (self.UMO,)
            ).fetchone()[0]
        )
        recent = self.storage.query_recent_context(
            umo=self.UMO,
            before_sent_at=2000,
            message_upper_bound=stream_upper_bound,
            limit=64,
        )
        recent_keys = [str(item["source_key"]) for item in recent]
        self.assertEqual(recent_keys, expected_keys[-64:])
        self.assertEqual(len(recent_keys), len(set(recent_keys)))

        request = self.message(
            "stream-request",
            "synthetic current request",
            sent_at=2000,
            sender_id="synthetic-requester",
        )
        self.storage.upsert_message(request)
        snapshot = self.capture(request=request, cutoff_at=2001)
        future = self.message(
            "stream-future",
            "future payload",
            sent_at=3000,
            sender_id="synthetic-future",
        )
        late_old = self.message(
            "stream-late-old",
            "late inserted old payload",
            sent_at=900,
            sender_id="synthetic-late",
        )
        self.storage.upsert_message(future)
        self.storage.upsert_message(late_old)
        upper_bound = int(snapshot["message_upper_bound"])
        self.assertEqual(
            self.storage.count_snapshot_messages(
                umo=self.UMO,
                before_sent_at=int(snapshot["cutoff_at"]),
                message_upper_bound=upper_bound,
                exclude_source_key=request.resolved_source_key(),
            ),
            501,
        )
        visible = self.storage.messages_for_sources(
            umo=self.UMO,
            source_keys=(
                expected_keys[0],
                request.resolved_source_key(),
                expected_keys[250],
                future.resolved_source_key(),
                expected_keys[-1],
                late_old.resolved_source_key(),
            ),
            before_sent_at=int(snapshot["cutoff_at"]),
            message_upper_bound=upper_bound,
        )
        self.assertEqual(
            [item["source_key"] for item in visible],
            [expected_keys[0], expected_keys[250], expected_keys[-1]],
        )

    def test_chunked_embedding_search_preserves_exact_rank_and_snapshot(self) -> None:
        old_a = self.message("vector-old-a", "old a", sent_at=100, sender_id="a")
        old_b = self.message("vector-old-b", "old b", sent_at=110, sender_id="b")
        request = self.message(
            "vector-request",
            "current request",
            sent_at=200,
            sender_id="requester",
        )
        for message in (old_a, old_b, request):
            self.storage.upsert_message(message)
        snapshot = self.capture(request=request, cutoff_at=201)
        future = self.message(
            "vector-future",
            "future",
            sent_at=300,
            sender_id="future",
        )
        self.storage.upsert_message(future)

        old_episode_a = self.storage.store_episode(
            umo=self.UMO,
            started_at=100,
            ended_at=100,
            title="old episode a",
            summary="visible",
            source_keys=[old_a.resolved_source_key()],
            keywords=[("old-a", "synthetic")],
        )
        old_episode_b = self.storage.store_episode(
            umo=self.UMO,
            started_at=110,
            ended_at=110,
            title="old episode b",
            summary="visible",
            source_keys=[old_b.resolved_source_key()],
            keywords=[("old-b", "synthetic")],
        )
        future_episode = self.storage.store_episode(
            umo=self.UMO,
            started_at=300,
            ended_at=300,
            title="future episode",
            summary="hidden",
            source_keys=[future.resolved_source_key()],
            keywords=[("future", "synthetic")],
        )
        participant_ids = {
            str(row["account_id"]): int(row["id"])
            for row in self.storage._connection.execute(
                "SELECT id, account_id FROM participants WHERE umo=?",
                (self.UMO,),
            ).fetchall()
        }
        model = "synthetic/exact-vector"
        embeddings = (
            ("participant", str(participant_ids["a"]), [1.0, 0.0]),
            ("participant", str(participant_ids["b"]), [0.8, 0.6]),
            ("participant", str(participant_ids["future"]), [1.0, 0.0]),
            ("episode", str(old_episode_a), [1.0, 0.0]),
            ("episode", str(old_episode_b), [0.6, 0.8]),
            ("episode", str(future_episode), [1.0, 0.0]),
        )
        for owner_type, owner_key, vector in embeddings:
            self.storage.upsert_memory_embedding(
                umo=self.UMO,
                owner_type=owner_type,
                owner_key=owner_key,
                model=model,
                vector=vector,
            )
        search = {
            "model": model,
            "query_vector": [1.0, 0.0],
            "owner_types": ("participant", "episode"),
            "limit": 3,
            "before_sent_at": int(snapshot["cutoff_at"]),
            "message_upper_bound": int(snapshot["message_upper_bound"]),
        }
        expected = self.legacy_embedding_search(**search)
        actual = self.storage.search_memory_embeddings(umo=self.UMO, **search)
        self.assertEqual(actual, expected)
        self.assertNotIn(
            ("participant", str(participant_ids["future"])),
            {(item["owner_type"], item["owner_key"]) for item in actual},
        )
        self.assertNotIn(
            ("episode", str(future_episode)),
            {(item["owner_type"], item["owner_key"]) for item in actual},
        )

    def test_chunked_embedding_search_retains_only_bounded_top_candidates(
        self,
    ) -> None:
        row_count = 1537
        model = "synthetic/chunked-vector"
        with self.storage._connection:
            self.storage._connection.executemany(
                """
                INSERT INTO participants(
                    umo, platform_id, account_id, canonical_key,
                    current_display_name, first_seen_at, last_seen_at
                ) VALUES (?, 'shadow', ?, ?, ?, 100, 100)
                """,
                [
                    (
                        self.UMO,
                        f"bulk-{index:04d}",
                        f"shadow:bulk-{index:04d}",
                        f"bulk-{index:04d}",
                    )
                    for index in range(row_count)
                ],
            )
            participants = self.storage._connection.execute(
                "SELECT id FROM participants WHERE umo=? ORDER BY id",
                (self.UMO,),
            ).fetchall()
            self.storage._connection.executemany(
                """
                INSERT INTO memory_embeddings(
                    umo, owner_type, owner_key, model, dimensions, vector
                ) VALUES (?, 'participant', ?, ?, 2, ?)
                """,
                [
                    (
                        self.UMO,
                        str(row["id"]),
                        model,
                        encode_vector(
                            [float(row_count - index), float(index + 1)]
                        ),
                    )
                    for index, row in enumerate(participants)
                ],
            )

        result = self.storage.search_memory_embeddings(
            umo=self.UMO,
            model=model,
            query_vector=[1.0, 0.0],
            owner_types=("participant",),
            limit=17,
        )
        self.assertEqual(len(result), 17)
        self.assertEqual(
            [float(item["score"]) for item in result],
            sorted((float(item["score"]) for item in result), reverse=True),
        )
        stats = self.storage._last_embedding_search_stats
        self.assertEqual(stats["scanned_rows"], row_count)
        self.assertEqual(stats["scored_rows"], row_count)
        self.assertEqual(stats["max_chunk_rows"], 512)
        self.assertLess(stats["max_chunk_rows"], row_count)
        self.assertLessEqual(
            stats["max_retained_candidates"],
            stats["retained_candidate_bound"],
        )
        self.assertLess(stats["max_retained_candidates"], row_count)

        with self.storage._connection:
            self.storage._connection.execute(
                """
                UPDATE memory_embeddings SET vector=?
                WHERE umo=? AND model=? AND owner_type='participant'
                  AND owner_key=?
                """,
                (b"broken", self.UMO, model, str(participants[0]["id"])),
            )
        with self.assertRaisesRegex(ValueError, "invalid embedding blob"):
            self.storage.search_memory_embeddings(
                umo=self.UMO,
                model=model,
                query_vector=[1.0, 0.0],
                owner_types=("participant",),
                limit=17,
            )

    def test_current_schema_contains_layered_and_search_index_tables(self) -> None:
        version = self.storage._connection.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()
        self.assertEqual(version["value"], "18")
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
            "participant_alias_observations",
        }
        actual = {
            str(row["name"])
            for row in self.storage._connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        self.assertTrue(expected <= actual)
        expected_indexes = {
            "idx_messages_umo_participant_time": (
                "umo",
                "sender_participant_id",
                "sent_at",
                "id",
            ),
            "idx_message_participants_participant": (
                "participant_id",
                "relation",
                "message_id",
                "position",
            ),
        }
        for index_name, expected_columns in expected_indexes.items():
            columns = tuple(
                str(row["name"])
                for row in self.storage._connection.execute(
                    f"PRAGMA index_info({index_name})"
                ).fetchall()
            )
            self.assertEqual(columns, expected_columns)

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
        audited_sql: list[str] = []
        self.storage._connection.set_trace_callback(audited_sql.append)
        try:
            audit = self.storage.audit_snapshot_sources(
                snapshot_id=snapshot.snapshot_id,
                umo=self.UMO,
                source_keys=(
                    previous.resolved_source_key(),
                    request.resolved_source_key(),
                    later.resolved_source_key(),
                ),
            )
        finally:
            self.storage._connection.set_trace_callback(None)
        message_reads = [
            statement
            for statement in audited_sql
            if "FROM messages WHERE source_key IN" in " ".join(statement.split())
        ]
        self.assertEqual(len(message_reads), 1)
        self.assertFalse(audit["valid"])
        self.assertEqual(audit["accepted_source_keys"], [previous.resolved_source_key()])
        reasons = {item["reason"] for item in audit["violations"]}
        self.assertEqual(
            reasons,
            {"CURRENT_REQUEST_SOURCE", "AFTER_MESSAGE_UPPER_BOUND"},
        )
        frozen_fingerprint = audit["source_fingerprints"][
            previous.resolved_source_key()
        ]
        edited_previous = self.message(
            "previous",
            "上一条已编辑",
            sent_at=200,
        )
        self.storage.upsert_message(edited_previous)
        repeated_audit = self.storage.audit_snapshot_sources(
            snapshot_id=snapshot.snapshot_id,
            umo=self.UMO,
            source_keys=(previous.resolved_source_key(),),
            fail_closed=True,
        )
        self.assertNotEqual(
            repeated_audit["source_fingerprints"][previous.resolved_source_key()],
            frozen_fingerprint,
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

    def test_person_reference_candidates_are_source_bound_and_non_decisive(
        self,
    ) -> None:
        def synthetic_sender(
            message_id: str,
            account_id: str,
            alias: str = "Shared Synthetic Alias",
        ) -> NormalizedMessage:
            return NormalizedMessage(
                platform="aiocqhttp",
                platform_id="shadow",
                umo=self.UMO,
                group_id="layered",
                message_id=message_id,
                sender_id=account_id,
                sender_name=alias,
                sent_at=100,
                plain_text="synthetic observation",
                content=[{"type": "plain", "text": "synthetic observation"}],
            )

        first = synthetic_sender("candidate-first", "synthetic-account-a")
        second = synthetic_sender("candidate-second", "synthetic-account-b")
        third = synthetic_sender(
            "candidate-third",
            "synthetic-account-c",
            "Second Synthetic Alias",
        )
        request = self.message(
            "candidate-request",
            (
                "Please compare Shared Synthetic Alias and Second Synthetic "
                "Alias without deciding identity."
            ),
            sent_at=200,
            sender_id="synthetic-requester",
        )
        future = NormalizedMessage(
            platform="aiocqhttp",
            platform_id="shadow",
            umo=self.UMO,
            group_id="layered",
            message_id="candidate-future",
            sender_id="synthetic-account-future",
            sender_name="Shared Synthetic Alias",
            sent_at=200,
            plain_text="future synthetic observation",
            content=[
                {"type": "plain", "text": "future synthetic observation"}
            ],
        )
        for message in (first, second, third, request, future):
            self.storage.upsert_message(message)
        self.storage.bind_participant_alias(
            umo=self.UMO,
            platform_id="shadow",
            account_id="synthetic-account-admin-only",
            alias="Shared Synthetic Alias",
            at=100,
        )
        snapshot = self.capture(request=request, cutoff_at=201)

        result = self.storage.query_person_reference_candidates(
            umo=self.UMO,
            text=request.plain_text,
            before_sent_at=201,
            message_upper_bound=int(snapshot["message_upper_bound"]),
            limit=1,
        )

        self.assertEqual(result["host_decision"], "NONE")
        self.assertEqual(
            result["coverage"],
            {
                "matched_reference_count_total": 2,
                "matched_reference_count_returned": 1,
                "candidate_count_total": 3,
                "candidate_count_returned": 2,
                "distinct_candidate_count_total": 3,
                "distinct_candidate_count_returned": 2,
                "truncated": True,
            },
        )
        self.assertEqual(len(result["references"]), 1)
        reference = result["references"][0]
        self.assertEqual(reference["reference"], "shared synthetic alias")
        candidates = reference["candidate_participants"]
        self.assertEqual(
            {candidate["account_id"] for candidate in candidates},
            {"synthetic-account-a", "synthetic-account-b"},
        )
        self.assertEqual(
            {
                observation["source_key"]
                for candidate in candidates
                for observation in candidate["alias_observations"]
            },
            {first.resolved_source_key(), second.resolved_source_key()},
        )
        self.assertTrue(
            all(
                observation["relation"] == "SPEAKER"
                for candidate in candidates
                for observation in candidate["alias_observations"]
            )
        )
        serialized = repr(result).casefold()
        for forbidden_decision in ("resolved", "unique", "ambiguous"):
            self.assertNotIn(forbidden_decision, serialized)

    def test_person_reference_observations_are_indexed_bounded_and_synchronized(
        self,
    ) -> None:
        alias = "Bounded Synthetic Alias"
        account_id = "bounded-synthetic-account"
        messages = [
            NormalizedMessage(
                platform="aiocqhttp",
                platform_id="shadow",
                umo=self.UMO,
                group_id="layered",
                message_id=f"bounded-{index}",
                sender_id=account_id,
                sender_name=alias,
                sent_at=100 + index,
                plain_text=f"observation {index}",
                content=[{"type": "plain", "text": f"observation {index}"}],
            )
            for index in range(12)
        ]
        request = self.message(
            "bounded-request",
            f"Recall {alias}",
            sent_at=200,
            sender_id="bounded-requester",
        )
        for message in (*messages, request):
            self.storage.upsert_message(message)
        snapshot = self.capture(request=request, cutoff_at=201)
        result = self.storage.query_person_reference_candidates(
            umo=self.UMO,
            text=request.plain_text,
            before_sent_at=201,
            message_upper_bound=int(snapshot["message_upper_bound"]),
        )
        candidate = result["references"][0]["candidate_participants"][0]
        self.assertEqual(candidate["source_count_total"], 12)
        self.assertEqual(len(candidate["alias_observations"]), 8)
        self.assertTrue(candidate["observations_truncated"])

        participant_id = self.storage._connection.execute(
            """
            SELECT id FROM participants
            WHERE umo=? AND platform_id='shadow' AND account_id=?
            """,
            (self.UMO, account_id),
        ).fetchone()["id"]
        plan = " ".join(
            str(row["detail"])
            for row in self.storage._connection.execute(
                """
                EXPLAIN QUERY PLAN
                SELECT observation.alias, message.source_key
                FROM participant_alias_observations AS observation
                     INDEXED BY idx_alias_observations_snapshot
                JOIN messages AS message ON message.id=observation.message_id
                WHERE observation.umo=?
                  AND observation.participant_id=?
                  AND observation.normalized_alias=?
                  AND observation.sent_at<?
                  AND observation.message_id<=?
                  AND message.is_deleted=0
                ORDER BY observation.sent_at DESC,
                         observation.message_id DESC,
                         observation.position
                LIMIT 8
                """,
                (
                    self.UMO,
                    int(participant_id),
                    alias.casefold(),
                    201,
                    int(snapshot["message_upper_bound"]),
                ),
            ).fetchall()
        )
        self.assertIn("idx_alias_observations_snapshot", plan)
        self.assertIn("SEARCH message USING INTEGER PRIMARY KEY", plan)

        edited = NormalizedMessage(
            platform=messages[-1].platform,
            platform_id=messages[-1].platform_id,
            umo=messages[-1].umo,
            group_id=messages[-1].group_id,
            message_id=messages[-1].message_id,
            sender_id=messages[-1].sender_id,
            sender_name="Edited Synthetic Alias",
            sent_at=messages[-1].sent_at,
            plain_text=messages[-1].plain_text,
            content=messages[-1].content,
            role=messages[-1].role,
            source_key=messages[-1].source_key,
        )
        self.storage.upsert_message(edited)
        old_count = self.storage._connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM participant_alias_observations
            WHERE participant_id=? AND normalized_alias=?
            """,
            (int(participant_id), alias.casefold()),
        ).fetchone()["count"]
        self.assertEqual(old_count, 11)
        self.assertTrue(
            self.storage.mark_message_deleted(
                umo=self.UMO,
                platform_id="shadow",
                platform_message_id=edited.message_id,
                deleted_at=250,
            )
        )
        self.assertEqual(
            self.storage._connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM participant_alias_observations
                WHERE message_id=(SELECT id FROM messages WHERE source_key=?)
                """,
                (edited.resolved_source_key(),),
            ).fetchone()["count"],
            0,
        )
        self.storage.forget_account(
            umo=self.UMO,
            platform_id="shadow",
            account_id=account_id,
            requested_at=260,
        )
        self.assertEqual(
            self.storage._connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM participant_alias_observations WHERE participant_id=?
                """,
                (int(participant_id),),
            ).fetchone()["count"],
            0,
        )

    def test_semantic_seed_exposes_source_bound_subject_candidate(self) -> None:
        old = NormalizedMessage(
            platform="aiocqhttp",
            platform_id="shadow",
            umo=self.UMO,
            group_id="layered",
            message_id="subject-old",
            sender_id="synthetic-account",
            sender_name="Synthetic Subject Old",
            sent_at=100,
            plain_text="Synthetic support mentions NarrativeLabel.",
            content=[
                {
                    "type": "plain",
                    "text": "Synthetic support mentions NarrativeLabel.",
                }
            ],
        )
        request = self.message(
            "subject-request", "Current request", sent_at=200, sender_id="requester"
        )
        future = NormalizedMessage(
            platform="aiocqhttp",
            platform_id="shadow",
            umo=self.UMO,
            group_id="layered",
            message_id="subject-future",
            sender_id="synthetic-account",
            sender_name="Synthetic Subject Future",
            sent_at=300,
            plain_text="Synthetic future support.",
            content=[{"type": "plain", "text": "Synthetic future support."}],
        )
        for message in (old, request, future):
            self.storage.upsert_message(message)
        snapshot = self.capture(request=request, cutoff_at=201)
        bound = int(snapshot["message_upper_bound"])
        participant_key = str(
            self.storage.resolve_participants(
                umo=self.UMO,
                reference="synthetic-account",
            )["participants"][0]["canonical_key"]
        )
        visible_memory_id = self.storage.store_semantic_claim(
            umo=self.UMO,
            stable_key="synthetic-visible-subject",
            subject_participant_key=participant_key,
            subject_text="",
            claim_type="PREFERENCE",
            aspect="synthetic-aspect",
            content="NarrativeLabel is prose, not an identity binding.",
            epistemic_status="ASSERTED",
            operation="ASSERT",
            target_claim_ids=[],
            evidence=[
                {
                    "source_key": old.resolved_source_key(),
                    "role": "SUPPORT",
                    "span": "NarrativeLabel",
                    "confidence": 0.8,
                }
            ],
            confidence=0.8,
        )
        mixed_memory_id = self.storage.store_semantic_claim(
            umo=self.UMO,
            stable_key="synthetic-mixed-subject",
            subject_participant_key=participant_key,
            subject_text="",
            claim_type="PREFERENCE",
            aspect="synthetic-mixed-aspect",
            content="A mixed-source synthetic claim.",
            epistemic_status="ASSERTED",
            operation="ASSERT",
            target_claim_ids=[],
            evidence=[
                {
                    "source_key": old.resolved_source_key(),
                    "role": "SUPPORT",
                    "span": "Synthetic support",
                    "confidence": 0.8,
                },
                {
                    "source_key": future.resolved_source_key(),
                    "role": "SUPPORT",
                    "span": "Synthetic future support",
                    "confidence": 0.8,
                },
            ],
            confidence=0.8,
        )

        expanded = self.storage.expand_seed_candidates(
            umo=self.UMO,
            matches=[
                {
                    "owner_type": "semantic",
                    "owner_key": str(visible_memory_id),
                    "score": 0.9,
                },
                {
                    "owner_type": "semantic",
                    "owner_key": str(mixed_memory_id),
                    "score": 0.8,
                },
            ],
            before_sent_at=201,
            message_upper_bound=bound,
        )

        self.assertEqual(
            [item["id"] for item in expanded["semantic_memories"]],
            [visible_memory_id],
        )
        semantic = expanded["semantic_memories"][0]
        self.assertEqual(semantic["subject_participant_key"], participant_key)
        self.assertEqual(semantic["subject_display_name"], "Synthetic Subject Old")
        self.assertEqual(semantic["person_cue"], "Synthetic Subject Old")
        self.assertEqual(semantic["subject_source_keys"], [old.resolved_source_key()])
        participant = expanded["participants"][0]
        self.assertEqual(participant["participant_key"], participant_key)
        self.assertEqual(participant["canonical_key"], participant_key)
        self.assertEqual(
            participant["candidate_basis"], "semantic_subject_binding"
        )
        self.assertEqual(participant["source_keys"], [old.resolved_source_key()])

    def test_recent_context_is_ordered_bounded_and_excludes_current_request(
        self,
    ) -> None:
        first = self.message(
            "recent-first",
            "first",
            sent_at=100,
            sender_id="a",
            content=[
                {"type": "plain", "text": "first"},
                {
                    "type": "mention",
                    "account_id": "mentioned-account",
                    "display_name": "Mentioned Synthetic",
                },
            ],
        )
        deleted = self.message(
            "recent-deleted", "deleted", sent_at=150, sender_id="deleted"
        )
        second = self.message(
            "recent-second",
            "second",
            sent_at=180,
            sender_id="b",
            content=[
                {
                    "type": "response_to",
                    "message_id": first.message_id,
                    "sender_id": first.sender_id,
                    "sender_name": first.sender_name,
                    "sent_at": first.sent_at,
                    "plain_text": first.plain_text,
                },
                {"type": "plain", "text": "second"},
            ],
        )
        third = self.message("recent-third", "third", sent_at=200, sender_id="c")
        fourth = self.message("recent-fourth", "fourth", sent_at=250, sender_id="d")
        request = self.message(
            "recent-request",
            "current request",
            sent_at=300,
            sender_id="requester",
        )
        future = self.message("recent-future", "future", sent_at=400, sender_id="c")
        for message in (first, deleted, second, third, fourth, request, future):
            self.storage.upsert_message(message)
        self.assertTrue(
            self.storage.mark_message_deleted(
                umo=self.UMO,
                platform_id=deleted.platform_id,
                platform_message_id=deleted.message_id,
                deleted_at=290,
            )
        )
        snapshot = self.capture(request=request, cutoff_at=301)
        upper_bound = int(snapshot["message_upper_bound"])

        self.assertEqual(
            self.storage.count_snapshot_messages(
                umo=self.UMO,
                before_sent_at=301,
                message_upper_bound=upper_bound,
                exclude_source_key=request.resolved_source_key(),
            ),
            4,
        )

        traced: list[str] = []
        self.storage._connection.set_trace_callback(traced.append)
        try:
            recent = self.storage.query_recent_context(
                umo=self.UMO,
                before_sent_at=301,
                message_upper_bound=upper_bound,
                exclude_source_key=request.resolved_source_key(),
                exclude_source_keys=(
                    fourth.resolved_source_key(),
                    third.resolved_source_key(),
                ),
                limit=2,
            )
        finally:
            self.storage._connection.set_trace_callback(None)

        self.assertEqual(
            [item["source_key"] for item in recent],
            [first.resolved_source_key(), second.resolved_source_key()],
        )
        self.assertNotIn(
            request.resolved_source_key(),
            {item["source_key"] for item in recent},
        )
        self.assertNotIn(
            future.resolved_source_key(),
            {item["source_key"] for item in recent},
        )
        self.assertEqual(
            recent[0]["mentions"][0]["account_id"], "mentioned-account"
        )
        self.assertEqual(
            recent[1]["reply_to_source_key"], first.resolved_source_key()
        )
        hydration_selects = [
            statement
            for statement in traced
            if any(
                marker in statement
                for marker in (
                    "SELECT id, canonical_key FROM participants",
                    "FROM message_relations\n            WHERE source_message_id IN",
                    "FROM message_participants AS mp\n"
                    "            JOIN participants AS p",
                )
            )
        ]
        self.assertEqual(len(hydration_selects), 3)

    def test_message_search_honors_snapshot_for_fts_and_short_substring(self) -> None:
        visible = self.message(
            "search-visible", "needle xy visible", sent_at=100, sender_id="search-a"
        )
        request = self.message(
            "search-request", "search now", sent_at=200, sender_id="requester"
        )
        for message in (visible, request):
            self.storage.upsert_message(message)
        snapshot = self.capture(request=request, cutoff_at=201)
        late_backfill = self.message(
            "search-late-backfill",
            "needle xy late",
            sent_at=90,
            sender_id="search-b",
        )
        self.storage.upsert_message(late_backfill)

        traced: list[str] = []
        self.storage._connection.set_trace_callback(traced.append)
        try:
            fts_rows = self.storage.search_messages(
                umo=self.UMO,
                query="needle",
                before_sent_at=201,
                message_upper_bound=int(snapshot["message_upper_bound"]),
            )
        finally:
            self.storage._connection.set_trace_callback(None)
        self.assertEqual(
            [message.source_key for message in fts_rows],
            [visible.resolved_source_key()],
        )
        self.assertTrue(
            any("ORDER BY rank" in statement for statement in traced)
        )
        short_rows = self.storage.search_messages(
            umo=self.UMO,
            query="xy",
            before_sent_at=201,
            message_upper_bound=int(snapshot["message_upper_bound"]),
        )
        self.assertEqual(
            [message.source_key for message in short_rows],
            [visible.resolved_source_key()],
        )

        recall = self.message(
            "search-recall",
            "作品态度后来发生变化",
            sent_at=110,
            sender_id="search-c",
        )
        self.storage.upsert_message(recall)
        recall_rows = self.storage.search_messages(
            umo=self.UMO,
            query="请回忆这个成员最近的作品态度",
            before_sent_at=201,
            match_mode="recall",
            exclude_source_key=request.resolved_source_key(),
        )
        self.assertIn(
            recall.resolved_source_key(),
            {message.source_key for message in recall_rows},
        )
        excluded = self.storage.search_messages(
            umo=self.UMO,
            query="search now",
            before_sent_at=201,
            message_upper_bound=int(snapshot["message_upper_bound"]),
            exclude_source_key=request.resolved_source_key(),
        )
        self.assertNotIn(
            request.resolved_source_key(),
            {message.source_key for message in excluded},
        )

    def test_recall_search_preserves_bm25_order(self) -> None:
        strongest = self.message(
            "rank-strong",
            "作品态度发生变化",
            sent_at=100,
        )
        weaker = self.message(
            "rank-weak",
            "作品态度",
            sent_at=200,
        )
        for message in (strongest, weaker):
            self.storage.upsert_message(message)

        rows = self.storage.search_messages(
            umo=self.UMO,
            query="作品态度发生变化",
            match_mode="recall",
        )

        self.assertEqual(rows[0].source_key, strongest.resolved_source_key())

    def test_recall_search_keeps_two_character_cjk_entities(self) -> None:
        entity_message = self.message(
            "short-cjk-entity",
            "小禾今天值班",
            sent_at=100,
        )
        question_noise = self.message(
            "short-cjk-noise",
            "这是谁留下的",
            sent_at=200,
        )
        for message in (entity_message, question_noise):
            self.storage.upsert_message(message)

        rows = self.storage.search_messages(
            umo=self.UMO,
            query="小禾是谁",
            match_mode="recall",
            limit=4,
        )

        self.assertIn(
            entity_message.resolved_source_key(),
            {message.source_key for message in rows},
        )

    def test_recall_keeps_rare_mixed_script_term_amid_common_cjk_suffix(self) -> None:
        rare = [
            self.message(f"mixed-rare-{index}", "Q馆长留下记录", sent_at=100 + index)
            for index in range(3)
        ]
        common = [
            self.message(f"common-suffix-{index}", "馆长正在值班", sent_at=200 + index)
            for index in range(120)
        ]
        for message in (*rare, *common):
            self.storage.upsert_message(message)
        query = "Q馆长是谁"
        rows = self.storage.search_messages(
            umo=self.UMO, query=query, match_mode="recall", limit=12,
            before_sent_at=500,
        )
        self.assertEqual(len(rows), 12)
        self.assertTrue(
            {message.resolved_source_key() for message in rare}.issubset(
                message.source_key for message in rows
            )
        )
        self.assertIn("q馆长", fts_recall_terms(query))
        self.assertIn("q馆长", recall_coverage_terms(query))
        self.assertLessEqual(len(fts_recall_terms(query)), 32)
        self.assertLessEqual(len(short_recall_terms(query)), 16)

    def test_recall_does_not_lose_an_old_two_character_entity_to_common_bigrams(
        self,
    ) -> None:
        limit = 4
        entity_message = self.message(
            "short-entity-before-common-noise",
            "小禾今天值班",
            sent_at=100,
        )
        newer_common_bigram_messages = [
            self.message(
                f"common-bigram-noise-{index}",
                f"这是谁留下的第 {index} 条记录",
                sent_at=200 + index,
            )
            for index in range(limit)
        ]
        for message in (entity_message, *newer_common_bigram_messages):
            self.storage.upsert_message(message)

        rows = self.storage.search_messages(
            umo=self.UMO,
            query="小禾是谁",
            match_mode="recall",
            limit=limit,
        )

        self.assertIn(
            entity_message.resolved_source_key(),
            {message.source_key for message in rows},
        )

    def test_recall_term_coverage_is_monotone_and_prefers_lower_frequency_terms(
        self,
    ) -> None:
        limit = 12
        entity_term = "小禾"
        entity_messages = [
            self.message(
                f"coverage-entity-{index}",
                entity_term,
                sent_at=100 + index,
            )
            for index in range(5)
        ]
        common_terms = ("今天", "是谁", "留下", "消息")
        common_messages = [
            self.message(
                f"coverage-common-{term_index}-{index}",
                term,
                sent_at=200 + term_index * 10 + index,
            )
            for term_index, term in enumerate(common_terms)
            for index in range(4)
        ]
        for message in (*entity_messages, *common_messages):
            self.storage.upsert_message(message)

        with self.subTest("all_terms_fit_the_result_budget"):
            rows = self.storage.search_messages(
                umo=self.UMO,
                query=" ".join((entity_term, *common_terms)),
                match_mode="recall",
                limit=limit,
            )
            returned = {message.source_key for message in rows}
            self.assertTrue(
                returned.intersection(
                    message.resolved_source_key() for message in entity_messages
                )
            )

        rare_terms = (
            "甲乙",
            "丙丁",
            "戊己",
            "庚辛",
            "壬癸",
            "子丑",
            "寅卯",
            "辰巳",
            "午未",
            "申酉",
            "戌亥",
            "天地",
        )
        common_over_budget_term = "玄黄"
        rare_messages = [
            self.message(
                f"coverage-rare-{index}",
                term,
                sent_at=400 + index,
            )
            for index, term in enumerate(rare_terms)
        ]
        over_budget_common_messages = [
            self.message(
                f"coverage-over-budget-common-{index}",
                common_over_budget_term,
                sent_at=500 + index,
            )
            for index in range(4)
        ]
        for message in (*rare_messages, *over_budget_common_messages):
            self.storage.upsert_message(message)

        with self.subTest("more_terms_than_budget_prefers_lower_document_frequency"):
            rows = self.storage.search_messages(
                umo=self.UMO,
                query=" ".join((*rare_terms, common_over_budget_term)),
                match_mode="recall",
                limit=limit,
            )
            returned = {message.source_key for message in rows}
            self.assertEqual(
                returned,
                {message.resolved_source_key() for message in rare_messages},
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

    def test_synthetic_schema_15_migrates_to_current_without_data_loss(self) -> None:
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
            connection.execute(
                "DELETE FROM schema_meta WHERE key='alias_observations_v17'"
            )
            connection.execute("DROP TABLE participant_alias_observations")
            connection.commit()
        finally:
            connection.close()

        migrated = MemoryStorage(database_path)
        try:
            version = migrated._connection.execute(
                "SELECT value FROM schema_meta WHERE key='schema_version'"
            ).fetchone()["value"]
            self.assertEqual(version, "18")
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
            self.assertIn("participant_alias_observations", tables)
            self.assertEqual(
                migrated._connection.execute(
                    "SELECT COUNT(*) AS count FROM participant_alias_observations"
                ).fetchone()["count"],
                1,
            )
        finally:
            migrated.close()
            for suffix in ("", "-wal", "-shm"):
                Path(f"{database_path}{suffix}").unlink(missing_ok=True)

    def test_schema_16_alias_backfill_crosses_the_500_message_batch(self) -> None:
        database_path = self.database_path.with_name(f"{uuid.uuid4().hex}.db")
        storage = MemoryStorage(database_path)
        try:
            for index in range(501):
                storage.upsert_message(
                    self.message(
                        f"schema-16-{index}",
                        f"合成消息 {index}",
                        sent_at=index + 1,
                    )
                )
        finally:
            storage.close()

        connection = sqlite3.connect(database_path)
        try:
            connection.execute("DELETE FROM participant_alias_observations")
            connection.execute(
                "DELETE FROM schema_meta WHERE key='alias_observations_v17'"
            )
            connection.execute(
                "UPDATE schema_meta SET value='16' WHERE key='schema_version'"
            )
            connection.commit()
        finally:
            connection.close()

        migrated = MemoryStorage(database_path)
        try:
            version = migrated._connection.execute(
                "SELECT value FROM schema_meta WHERE key='schema_version'"
            ).fetchone()["value"]
            marker = migrated._connection.execute(
                "SELECT value FROM schema_meta WHERE key='alias_observations_v17'"
            ).fetchone()["value"]
            observations = migrated._connection.execute(
                """
                SELECT COUNT(*) AS count,
                       COUNT(DISTINCT message_id) AS message_count
                FROM participant_alias_observations
                """
            ).fetchone()

            self.assertEqual(version, "18")
            self.assertEqual(marker, "completed")
            self.assertEqual(observations["count"], 501)
            self.assertEqual(observations["message_count"], 501)
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


class SelectedSourceHydrationStorageTests(unittest.TestCase):
    UMO = "synthetic:GroupMessage:selected-source-hydration"

    @classmethod
    def message(
        cls,
        message_id: str,
        text: str,
        *,
        sent_at: int,
        sender_id: str,
    ) -> NormalizedMessage:
        return NormalizedMessage(
            platform="synthetic",
            platform_id="selected-source-hydration",
            umo=cls.UMO,
            group_id="selected-source-hydration",
            message_id=message_id,
            sender_id=sender_id,
            sender_name=f"name-{sender_id}",
            sent_at=sent_at,
            plain_text=text,
            content=[{"type": "plain", "text": text}],
        )

    def test_messages_for_sources_preserves_order_and_snapshot_visibility(
        self,
    ) -> None:
        storage = MemoryStorage(":memory:")
        try:
            first = self.message(
                "visible-first",
                "first visible payload",
                sent_at=100,
                sender_id="first",
            )
            after_cutoff = self.message(
                "after-cutoff",
                "must be excluded by cutoff",
                sent_at=300,
                sender_id="future",
            )
            second = self.message(
                "visible-second",
                "second visible payload",
                sent_at=110,
                sender_id="second",
            )
            deleted = self.message(
                "deleted",
                "must be excluded as deleted",
                sent_at=120,
                sender_id="deleted",
            )
            request = self.message(
                "current-request",
                "must be excluded as the current request",
                sent_at=200,
                sender_id="requester",
            )
            for message in (first, after_cutoff, deleted, second, request):
                storage.upsert_message(message)
            self.assertTrue(
                storage.mark_message_deleted(
                    umo=self.UMO,
                    platform_id=deleted.platform_id,
                    platform_message_id=deleted.message_id,
                    deleted_at=190,
                )
            )
            snapshot = storage.capture_request_snapshot(
                umo=self.UMO,
                cutoff_at=201,
                query=request.plain_text,
                request_source_key=request.resolved_source_key(),
            )
            late_backfill = self.message(
                "late-backfill",
                "old timestamp inserted after the snapshot",
                sent_at=90,
                sender_id="late",
            )
            storage.upsert_message(late_backfill)
            row_ids = {
                str(row["source_key"]): int(row["id"])
                for row in storage._connection.execute(
                    "SELECT id, source_key FROM messages WHERE umo=?",
                    (self.UMO,),
                ).fetchall()
            }
            upper_bound = int(snapshot["message_upper_bound"])
            self.assertLessEqual(
                row_ids[after_cutoff.resolved_source_key()],
                upper_bound,
            )
            self.assertLessEqual(
                row_ids[deleted.resolved_source_key()],
                upper_bound,
            )
            self.assertGreater(
                row_ids[request.resolved_source_key()],
                upper_bound,
            )
            self.assertGreater(
                row_ids[late_backfill.resolved_source_key()],
                upper_bound,
            )

            records = storage.messages_for_sources(
                umo=self.UMO,
                source_keys=(
                    second.resolved_source_key(),
                    request.resolved_source_key(),
                    after_cutoff.resolved_source_key(),
                    first.resolved_source_key(),
                    deleted.resolved_source_key(),
                    late_backfill.resolved_source_key(),
                    second.resolved_source_key(),
                ),
                before_sent_at=int(snapshot["cutoff_at"]),
                message_upper_bound=upper_bound,
            )

            self.assertEqual(
                [record["source_key"] for record in records],
                [second.resolved_source_key(), first.resolved_source_key()],
            )
            self.assertEqual(
                [record["plain_text"] for record in records],
                ["second visible payload", "first visible payload"],
            )
        finally:
            storage.close()


if __name__ == "__main__":
    unittest.main()
