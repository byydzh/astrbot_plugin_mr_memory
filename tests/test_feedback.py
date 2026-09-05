from __future__ import annotations

import json
import unittest
import uuid
from pathlib import Path

from mr_memory.feedback import FeedbackDecision, parse_feedback_decision
from mr_memory.models import NormalizedMessage
from mr_memory.plasticity import parse_graph_mutation
from mr_memory.storage import FeedbackEvidenceUnavailableError, MemoryStorage


class FeedbackMemoryTests(unittest.TestCase):
    umo = "shadow:GroupMessage:group-a"

    def setUp(self) -> None:
        test_root = Path.cwd() / ".dev" / "test-tmp"
        test_root.mkdir(parents=True, exist_ok=True)
        self.database_path = test_root / f"{uuid.uuid4().hex}.db"
        self.storage = MemoryStorage(self.database_path)
        self.storage.bind_scope(
            umo=self.umo,
            platform_id="shadow",
            group_id="group-a",
        )

    def tearDown(self) -> None:
        self.storage.close()
        for suffix in ("", "-wal", "-shm"):
            Path(f"{self.database_path}{suffix}").unlink(missing_ok=True)

    def message(
        self,
        message_id: str,
        text: str,
        *,
        sender_id: str = "user-a",
        sent_at: int,
        content: list[dict[str, object]] | None = None,
    ) -> NormalizedMessage:
        return NormalizedMessage(
            platform="aiocqhttp",
            platform_id="shadow",
            umo=self.umo,
            group_id="group-a",
            message_id=message_id,
            sender_id=sender_id,
            sender_name=sender_id,
            sent_at=sent_at,
            plain_text=text,
            content=content or [{"type": "plain", "text": text}],
        )

    def open_trace(
        self,
        trace_id: str,
        *,
        request_id: str,
        query: str,
        sender_id: str,
        sent_at: int,
        response: str,
    ) -> None:
        request = self.message(
            request_id,
            query,
            sender_id=sender_id,
            sent_at=sent_at,
        )
        self.storage.upsert_message(request)
        self.storage.start_interaction_trace(
            trace_id=trace_id,
            umo=self.umo,
            sender_id=sender_id,
            request_source_key=request.resolved_source_key(),
            request_sent_at=sent_at,
            query=query,
        )
        self.storage.finish_interaction_trace(
            trace_id=trace_id,
            umo=self.umo,
            response_text=response,
            response_at=sent_at + 1,
        )

    def learn(
        self,
        *,
        trace_id: str,
        feedback_id: str,
        feedback_text: str,
        feedback_at: int,
        cue: str,
        triggers: tuple[str, ...] = (),
        sender_id: str = "user-a",
    ) -> dict[str, object]:
        feedback = self.message(
            feedback_id,
            feedback_text,
            sender_id=sender_id,
            sent_at=feedback_at,
        )
        self.storage.upsert_message(feedback)
        proposal_id = self.storage.enqueue_feedback_candidate(
            umo=self.umo,
            feedback_source_key=feedback.resolved_source_key(),
        )
        self.assertIsNotNone(proposal_id)
        return self.storage.apply_feedback_decision(
            umo=self.umo,
            proposal_id=int(proposal_id),
            decision=FeedbackDecision(
                target_trace_id=trace_id,
                mutation="upsert",
                feedback_valence=-1.0,
                confidence=0.9,
                scope_type="sender",
                scope_key=sender_id,
                aspect="response_style",
                statement=feedback_text,
                prospective_cue=cue,
                trigger_cues=triggers,
                activation_mode="semantic" if triggers else "always",
            ),
        )

    def downgrade_feedback_proposals_to_legacy_schema(self) -> None:
        """Recreate only the pre-revision proposal table for migration tests."""

        with self.storage._connection:
            self.storage._connection.execute(
                "DELETE FROM schema_meta "
                "WHERE key='feedback_proposal_revision_binding'"
            )
            self.storage._connection.execute(
                "DROP INDEX IF EXISTS idx_feedback_proposals_pending"
            )
            self.storage._connection.execute(
                "DROP INDEX IF EXISTS idx_feedback_proposals_current"
            )
            self.storage._connection.execute(
                "ALTER TABLE feedback_proposals "
                "RENAME TO feedback_proposals_bound"
            )
            self.storage._connection.execute(
                """
                CREATE TABLE feedback_proposals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    umo TEXT NOT NULL,
                    feedback_source_key TEXT NOT NULL,
                    feedback_sent_at INTEGER NOT NULL,
                    candidate_trace_ids_json TEXT NOT NULL DEFAULT '[]',
                    surface_score REAL NOT NULL DEFAULT 0,
                    candidate_reason TEXT NOT NULL DEFAULT '',
                    decision_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    decided_at TEXT,
                    UNIQUE (umo, feedback_source_key)
                )
                """
            )
            self.storage._connection.execute(
                """
                INSERT INTO feedback_proposals(
                    id, umo, feedback_source_key, feedback_sent_at,
                    candidate_trace_ids_json, surface_score, candidate_reason,
                    decision_json, status, error, created_at, decided_at
                )
                SELECT id, umo, feedback_source_key, feedback_sent_at,
                       candidate_trace_ids_json, surface_score,
                       candidate_reason, decision_json, status, error,
                       created_at, decided_at
                FROM feedback_proposals_bound
                """
            )
            self.storage._connection.execute(
                "DROP TABLE feedback_proposals_bound"
            )

    def test_legacy_revision_one_proposal_migration_binds_exact_source(self) -> None:
        self.open_trace(
            "trace-legacy-revision-one",
            request_id="request-legacy-revision-one",
            query="继续",
            sender_id="user-a",
            sent_at=50,
            response="还需要继续吗？",
        )
        feedback = self.message(
            "feedback-legacy-revision-one",
            "不要反问",
            sender_id="user-a",
            sent_at=60,
        )
        self.storage.upsert_message(feedback)
        proposal_id = self.storage.enqueue_feedback_candidate(
            umo=self.umo,
            feedback_source_key=feedback.resolved_source_key(),
        )
        message = self.storage._connection.execute(
            "SELECT id, content_sha256 FROM messages WHERE source_key=?",
            (feedback.resolved_source_key(),),
        ).fetchone()
        self.downgrade_feedback_proposals_to_legacy_schema()

        self.storage.close()
        self.storage = MemoryStorage(self.database_path)
        proposal = self.storage._connection.execute(
            """
            SELECT feedback_message_id, feedback_revision_no,
                   feedback_content_sha256, is_current, invalidation_reason
            FROM feedback_proposals WHERE id=?
            """,
            (int(proposal_id or 0),),
        ).fetchone()
        self.assertEqual(int(proposal["feedback_message_id"]), int(message["id"]))
        self.assertEqual(int(proposal["feedback_revision_no"]), 1)
        self.assertEqual(
            str(proposal["feedback_content_sha256"]),
            str(message["content_sha256"]),
        )
        self.assertEqual(int(proposal["is_current"]), 1)
        self.assertEqual(str(proposal["invalidation_reason"]), "")
        edited = self.message(
            "feedback-legacy-revision-one",
            "谢谢，这样可以了。",
            sender_id="user-a",
            sent_at=60,
        )
        self.storage.upsert_message_with_outcome(edited)
        next_proposal_id = self.storage.enqueue_feedback_candidate(
            umo=self.umo,
            feedback_source_key=edited.resolved_source_key(),
        )
        self.assertIsNotNone(next_proposal_id)
        self.assertGreater(int(next_proposal_id or 0), int(proposal_id or 0))
        migrated_rows = self.storage._connection.execute(
            """
            SELECT id, feedback_revision_no, is_current
            FROM feedback_proposals WHERE feedback_source_key=? ORDER BY id
            """,
            (edited.resolved_source_key(),),
        ).fetchall()
        self.assertEqual(
            [
                (
                    int(row["id"]),
                    int(row["feedback_revision_no"]),
                    int(row["is_current"]),
                )
                for row in migrated_rows
            ],
            [
                (int(proposal_id or 0), 1, 0),
                (int(next_proposal_id or 0), 2, 1),
            ],
        )

    def test_terminal_feedback_edit_creates_a_new_revision_proposal(self) -> None:
        self.open_trace(
            "trace-edit-terminal",
            request_id="request-edit-terminal",
            query="继续",
            sender_id="user-a",
            sent_at=100,
            response="要不要继续？",
        )
        original = self.message(
            "feedback-edit-terminal",
            "不要反问",
            sender_id="user-a",
            sent_at=110,
        )
        self.storage.upsert_message(original)
        old_proposal_id = self.storage.enqueue_feedback_candidate(
            umo=self.umo,
            feedback_source_key=original.resolved_source_key(),
        )
        committed = self.storage.apply_feedback_decision(
            umo=self.umo,
            proposal_id=int(old_proposal_id or 0),
            decision=FeedbackDecision(
                target_trace_id="trace-edit-terminal",
                mutation="upsert",
                feedback_valence=-1.0,
                confidence=0.9,
                scope_type="sender",
                scope_key="user-a",
                aspect="style",
                statement="不要反问",
                prospective_cue="直接完成回答。",
                trigger_cues=(),
                activation_mode="always",
            ),
        )
        hypothesis_id = int(committed["hypothesis_id"])
        graph_mutation = parse_graph_mutation(
            {
                "operation": "upsert_edge",
                "evidence_source_keys": [original.resolved_source_key()],
                "confidence": 0.85,
                "utility_delta": 0.5,
                "statement": "回答应直接完成。",
                "source": {"kind": "behavior", "label": "回答"},
                "relation": {
                    "key": "prefers_response_style",
                    "name": "偏好回答风格",
                    "description": "回答方式偏好",
                    "source_kinds": ["behavior"],
                    "target_kinds": ["preference"],
                },
                "target": {"kind": "preference", "label": "直接完成"},
            }
        )
        graph_result = self.storage.apply_graph_mutation(
            umo=self.umo,
            feedback_proposal_id=int(old_proposal_id or 0),
            mutation=graph_mutation,
        )
        graph_mutation_id = int(graph_result["mutation_id"])
        graph_edge_id = int(graph_result["target_id"])

        edited = self.message(
            "feedback-edit-terminal",
            "谢谢，这样可以了。",
            sender_id="user-a",
            sent_at=110,
        )
        outcome = self.storage.upsert_message_with_outcome(edited)
        self.assertTrue(outcome.needs_processing)
        new_proposal_id = self.storage.enqueue_feedback_candidate(
            umo=self.umo,
            feedback_source_key=edited.resolved_source_key(),
        )

        self.assertIsNotNone(new_proposal_id)
        self.assertNotEqual(old_proposal_id, new_proposal_id)
        proposals = self.storage._connection.execute(
            """
            SELECT id, feedback_revision_no, is_current, invalidation_reason
            FROM feedback_proposals
            WHERE umo=? AND feedback_source_key=? ORDER BY id
            """,
            (self.umo, edited.resolved_source_key()),
        ).fetchall()
        self.assertEqual(
            [
                (
                    int(row["id"]),
                    int(row["feedback_revision_no"]),
                    int(row["is_current"]),
                )
                for row in proposals
            ],
            [
                (int(old_proposal_id or 0), 1, 0),
                (int(new_proposal_id or 0), 2, 1),
            ],
        )
        self.assertTrue(str(proposals[0]["invalidation_reason"]))
        pending = self.storage.pending_feedback_proposals(umo=self.umo, limit=10)
        self.assertEqual([int(item["id"]) for item in pending], [new_proposal_id])
        hypothesis = self.storage._connection.execute(
            """
            SELECT status, invalidation_reason FROM feedback_hypotheses
            WHERE id=?
            """,
            (hypothesis_id,),
        ).fetchone()
        self.assertEqual(str(hypothesis["status"]), "DORMANT")
        self.assertTrue(str(hypothesis["invalidation_reason"]))
        self.assertEqual(
            self.storage._connection.execute(
                "SELECT COUNT(*) FROM hypothesis_evidence WHERE hypothesis_id=?",
                (hypothesis_id,),
            ).fetchone()[0],
            0,
        )
        self.assertEqual(
            self.storage._connection.execute(
                "SELECT COUNT(*) FROM feedback_links WHERE feedback_source_key=?",
                (edited.resolved_source_key(),),
            ).fetchone()[0],
            0,
        )
        graph_receipt = self.storage._connection.execute(
            """
            SELECT status, error, feedback_proposal_id FROM graph_mutations
            WHERE id=?
            """,
            (graph_mutation_id,),
        ).fetchone()
        self.assertEqual(str(graph_receipt["status"]), "INVALIDATED")
        self.assertTrue(str(graph_receipt["error"]))
        self.assertEqual(
            int(graph_receipt["feedback_proposal_id"]),
            int(old_proposal_id or 0),
        )
        self.assertEqual(
            self.storage._connection.execute(
                "SELECT status FROM plastic_edges WHERE id=?",
                (graph_edge_id,),
            ).fetchone()[0],
            "DORMANT",
        )
        self.assertEqual(
            self.storage._connection.execute(
                "SELECT COUNT(*) FROM plastic_edge_evidence WHERE edge_id=?",
                (graph_edge_id,),
            ).fetchone()[0],
            0,
        )
        self.storage.compact_plastic_graph(
            umo=self.umo,
            now=2_000_000_000,
        )
        still_invalidated = self.storage._connection.execute(
            "SELECT status, invalidation_reason FROM plastic_edges WHERE id=?",
            (graph_edge_id,),
        ).fetchone()
        self.assertEqual(str(still_invalidated["status"]), "DORMANT")
        self.assertTrue(str(still_invalidated["invalidation_reason"]))
        self.storage.apply_feedback_decision(
            umo=self.umo,
            proposal_id=int(new_proposal_id or 0),
            decision=FeedbackDecision(
                target_trace_id="trace-edit-terminal",
                mutation="upsert",
                feedback_valence=1.0,
                confidence=0.9,
                scope_type="sender",
                scope_key="user-a",
                aspect="style",
                statement="直接完成回答很好",
                prospective_cue="直接完成回答。",
                trigger_cues=(),
                activation_mode="always",
            ),
        )
        rebuilt_graph = self.storage.apply_graph_mutation(
            umo=self.umo,
            feedback_proposal_id=int(new_proposal_id or 0),
            mutation=graph_mutation,
        )
        self.assertNotEqual(
            int(rebuilt_graph["mutation_id"]),
            graph_mutation_id,
        )
        rebuilt_edge = self.storage._connection.execute(
            """
            SELECT status, utility, support_count, invalidation_reason
            FROM plastic_edges WHERE id=?
            """,
            (graph_edge_id,),
        ).fetchone()
        self.assertEqual(str(rebuilt_edge["status"]), "ACTIVE")
        self.assertEqual(float(rebuilt_edge["utility"]), 0.5)
        self.assertEqual(int(rebuilt_edge["support_count"]), 1)
        self.assertEqual(str(rebuilt_edge["invalidation_reason"]), "")

    def test_stale_inflight_feedback_decision_is_rejected_after_edit(self) -> None:
        self.open_trace(
            "trace-stale-inflight",
            request_id="request-stale-inflight",
            query="继续",
            sender_id="user-a",
            sent_at=200,
            response="还要继续吗？",
        )
        original = self.message(
            "feedback-stale-inflight",
            "不要反问",
            sender_id="user-a",
            sent_at=210,
        )
        self.storage.upsert_message(original)
        old_proposal_id = self.storage.enqueue_feedback_candidate(
            umo=self.umo,
            feedback_source_key=original.resolved_source_key(),
        )
        self.storage.inspect_feedback_proposal(
            umo=self.umo,
            proposal_id=int(old_proposal_id or 0),
        )

        edited = self.message(
            "feedback-stale-inflight",
            "不错，这样可以了。",
            sender_id="user-a",
            sent_at=210,
        )
        self.storage.upsert_message_with_outcome(edited)
        new_proposal_id = self.storage.enqueue_feedback_candidate(
            umo=self.umo,
            feedback_source_key=edited.resolved_source_key(),
        )
        self.assertNotEqual(old_proposal_id, new_proposal_id)
        with self.assertRaisesRegex(
            FeedbackEvidenceUnavailableError,
            "source revision is stale",
        ):
            self.storage.inspect_feedback_proposal(
                umo=self.umo,
                proposal_id=int(old_proposal_id or 0),
            )
        with self.assertRaisesRegex(ValueError, "source revision is stale"):
            self.storage.apply_feedback_decision(
                umo=self.umo,
                proposal_id=int(old_proposal_id or 0),
                decision=FeedbackDecision(
                    target_trace_id="trace-stale-inflight",
                    mutation="upsert",
                    feedback_valence=-1.0,
                    confidence=0.9,
                    scope_type="sender",
                    scope_key="user-a",
                    aspect="style",
                    statement="不要反问",
                    prospective_cue="直接完成回答。",
                    trigger_cues=(),
                    activation_mode="always",
                ),
            )
        with self.assertRaisesRegex(
            FeedbackEvidenceUnavailableError,
            "source revision is stale",
        ):
            self.storage.reject_feedback_proposal(
                umo=self.umo,
                proposal_id=int(old_proposal_id or 0),
                error="stale worker result",
            )
        self.assertEqual(
            self.storage._connection.execute(
                "SELECT COUNT(*) FROM feedback_hypotheses"
            ).fetchone()[0],
            0,
        )
        self.assertEqual(
            self.storage._connection.execute(
                "SELECT COUNT(*) FROM feedback_links"
            ).fetchone()[0],
            0,
        )

    def test_edit_below_surface_gate_retracts_without_replacement(self) -> None:
        self.open_trace(
            "trace-edit-no-replacement",
            request_id="request-edit-no-replacement",
            query="继续",
            sender_id="user-a",
            sent_at=300,
            response="要不要继续？",
        )
        original = self.message(
            "feedback-edit-no-replacement",
            "不要反问",
            sender_id="user-a",
            sent_at=310,
        )
        self.storage.upsert_message(original)
        old_proposal_id = self.storage.enqueue_feedback_candidate(
            umo=self.umo,
            feedback_source_key=original.resolved_source_key(),
        )
        committed = self.storage.apply_feedback_decision(
            umo=self.umo,
            proposal_id=int(old_proposal_id or 0),
            decision=FeedbackDecision(
                target_trace_id="trace-edit-no-replacement",
                mutation="upsert",
                feedback_valence=-1.0,
                confidence=0.9,
                scope_type="sender",
                scope_key="user-a",
                aspect="style",
                statement="不要反问",
                prospective_cue="直接完成回答。",
                trigger_cues=(),
                activation_mode="always",
            ),
        )
        edited = self.message(
            "feedback-edit-no-replacement",
            "今天阳光很好",
            sender_id="user-a",
            sent_at=310,
        )
        self.storage.upsert_message_with_outcome(edited)
        self.assertIsNone(
            self.storage.enqueue_feedback_candidate(
                umo=self.umo,
                feedback_source_key=edited.resolved_source_key(),
            )
        )
        self.assertEqual(
            self.storage._connection.execute(
                "SELECT is_current FROM feedback_proposals WHERE id=?",
                (int(old_proposal_id or 0),),
            ).fetchone()[0],
            0,
        )
        self.assertEqual(
            self.storage._connection.execute(
                "SELECT status FROM feedback_hypotheses WHERE id=?",
                (int(committed["hypothesis_id"]),),
            ).fetchone()[0],
            "DORMANT",
        )
        self.assertEqual(
            self.storage.pending_feedback_proposals(umo=self.umo, limit=10),
            [],
        )

    def test_same_feedback_revision_enqueue_is_exactly_once(self) -> None:
        self.open_trace(
            "trace-same-revision",
            request_id="request-same-revision",
            query="继续",
            sender_id="user-a",
            sent_at=400,
            response="还需要继续吗？",
        )
        feedback = self.message(
            "feedback-same-revision",
            "不要反问",
            sender_id="user-a",
            sent_at=410,
        )
        self.storage.upsert_message(feedback)
        first = self.storage.enqueue_feedback_candidate(
            umo=self.umo,
            feedback_source_key=feedback.resolved_source_key(),
        )
        second = self.storage.enqueue_feedback_candidate(
            umo=self.umo,
            feedback_source_key=feedback.resolved_source_key(),
        )
        self.assertEqual(first, second)
        rows = self.storage._connection.execute(
            """
            SELECT id, feedback_revision_no, is_current
            FROM feedback_proposals WHERE umo=? AND feedback_source_key=?
            """,
            (self.umo, feedback.resolved_source_key()),
        ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(int(rows[0]["feedback_revision_no"]), 1)
        self.assertEqual(int(rows[0]["is_current"]), 1)

    def test_pending_feedback_rejects_revision_binding_drift(self) -> None:
        self.open_trace(
            "trace-pending-binding",
            request_id="request-pending-binding",
            query="继续",
            sender_id="user-a",
            sent_at=450,
            response="还需要继续吗？",
        )
        feedback = self.message(
            "feedback-pending-binding",
            "不要反问",
            sender_id="user-a",
            sent_at=460,
        )
        self.storage.upsert_message(feedback)
        self.assertIsNotNone(
            self.storage.enqueue_feedback_candidate(
                umo=self.umo,
                feedback_source_key=feedback.resolved_source_key(),
            )
        )
        with self.storage._connection:
            self.storage._connection.execute(
                "UPDATE messages SET revision_no=revision_no+1 "
                "WHERE source_key=?",
                (feedback.resolved_source_key(),),
            )
        with self.assertRaisesRegex(
            FeedbackEvidenceUnavailableError,
            "source revision is stale",
        ):
            self.storage.pending_feedback_proposals(umo=self.umo, limit=10)

    def test_legacy_deleted_revision_one_proposal_migration_fails_closed(
        self,
    ) -> None:
        self.open_trace(
            "trace-legacy-deleted",
            request_id="request-legacy-deleted",
            query="继续",
            sender_id="user-a",
            sent_at=470,
            response="还需要继续吗？",
        )
        feedback = self.message(
            "feedback-legacy-deleted",
            "不要反问",
            sender_id="user-a",
            sent_at=480,
        )
        self.storage.upsert_message(feedback)
        proposal_id = self.storage.enqueue_feedback_candidate(
            umo=self.umo,
            feedback_source_key=feedback.resolved_source_key(),
        )
        committed = self.storage.apply_feedback_decision(
            umo=self.umo,
            proposal_id=int(proposal_id or 0),
            decision=FeedbackDecision(
                target_trace_id="trace-legacy-deleted",
                mutation="upsert",
                feedback_valence=-1.0,
                confidence=0.9,
                scope_type="sender",
                scope_key="user-a",
                aspect="style",
                statement="不要反问",
                prospective_cue="直接完成回答。",
                trigger_cues=(),
                activation_mode="always",
            ),
        )
        hypothesis_id = int(committed["hypothesis_id"])
        with self.storage._connection:
            # Reproduce an old database whose deleted head never advanced its
            # revision counter.  Migration must not infer that revision one is
            # still live merely because the number and hash look bindable.
            self.storage._connection.execute(
                """
                UPDATE messages SET is_deleted=1, deleted_at=490
                WHERE source_key=?
                """,
                (feedback.resolved_source_key(),),
            )
        self.downgrade_feedback_proposals_to_legacy_schema()

        self.storage.close()
        self.storage = MemoryStorage(self.database_path)
        proposal = self.storage._connection.execute(
            """
            SELECT feedback_message_id, feedback_revision_no,
                   feedback_content_sha256, is_current, invalidation_reason
            FROM feedback_proposals WHERE id=?
            """,
            (int(proposal_id or 0),),
        ).fetchone()
        self.assertGreater(int(proposal["feedback_message_id"]), 0)
        self.assertEqual(int(proposal["feedback_revision_no"]), 0)
        self.assertEqual(str(proposal["feedback_content_sha256"]), "")
        self.assertEqual(int(proposal["is_current"]), 0)
        self.assertIn("legacy_unbound_revision", proposal["invalidation_reason"])
        hypothesis = self.storage._connection.execute(
            """
            SELECT status, invalidation_reason FROM feedback_hypotheses
            WHERE id=?
            """,
            (hypothesis_id,),
        ).fetchone()
        self.assertEqual(str(hypothesis["status"]), "DORMANT")
        self.assertIn("legacy_unbound_revision", hypothesis["invalidation_reason"])
        self.assertEqual(
            self.storage._connection.execute(
                "SELECT COUNT(*) FROM hypothesis_evidence WHERE hypothesis_id=?",
                (hypothesis_id,),
            ).fetchone()[0],
            0,
        )

        self.storage.close()
        self.storage = MemoryStorage(self.database_path)
        rebound = self.storage._connection.execute(
            """
            SELECT feedback_revision_no, feedback_content_sha256,
                   is_current, invalidation_reason
            FROM feedback_proposals WHERE id=?
            """,
            (int(proposal_id or 0),),
        ).fetchone()
        self.assertEqual(int(rebound["feedback_revision_no"]), 0)
        self.assertEqual(str(rebound["feedback_content_sha256"]), "")
        self.assertEqual(int(rebound["is_current"]), 0)
        self.assertIn("legacy_unbound_revision", rebound["invalidation_reason"])

    def test_legacy_revised_feedback_proposal_migration_fails_closed(self) -> None:
        self.open_trace(
            "trace-legacy-revision",
            request_id="request-legacy-revision",
            query="继续",
            sender_id="user-a",
            sent_at=500,
            response="还需要继续吗？",
        )
        feedback = self.message(
            "feedback-legacy-revision",
            "不要反问",
            sender_id="user-a",
            sent_at=510,
        )
        self.storage.upsert_message(feedback)
        proposal_id = self.storage.enqueue_feedback_candidate(
            umo=self.umo,
            feedback_source_key=feedback.resolved_source_key(),
        )
        committed = self.storage.apply_feedback_decision(
            umo=self.umo,
            proposal_id=int(proposal_id or 0),
            decision=FeedbackDecision(
                target_trace_id="trace-legacy-revision",
                mutation="upsert",
                feedback_valence=-1.0,
                confidence=0.9,
                scope_type="sender",
                scope_key="user-a",
                aspect="style",
                statement="不要反问",
                prospective_cue="直接完成回答。",
                trigger_cues=(),
                activation_mode="always",
            ),
        )
        hypothesis_id = int(committed["hypothesis_id"])
        with self.storage._connection:
            self.storage._connection.execute(
                "UPDATE messages SET revision_no=2 WHERE source_key=?",
                (feedback.resolved_source_key(),),
            )
        self.downgrade_feedback_proposals_to_legacy_schema()

        self.storage.close()
        self.storage = MemoryStorage(self.database_path)
        proposal = self.storage._connection.execute(
            """
            SELECT feedback_revision_no, feedback_content_sha256,
                   is_current, invalidation_reason
            FROM feedback_proposals WHERE id=?
            """,
            (int(proposal_id or 0),),
        ).fetchone()
        self.assertEqual(int(proposal["feedback_revision_no"]), 0)
        self.assertEqual(str(proposal["feedback_content_sha256"]), "")
        self.assertEqual(int(proposal["is_current"]), 0)
        self.assertIn("legacy_unbound_revision", proposal["invalidation_reason"])
        hypothesis = self.storage._connection.execute(
            """
            SELECT status, invalidation_reason FROM feedback_hypotheses
            WHERE id=?
            """,
            (hypothesis_id,),
        ).fetchone()
        self.assertEqual(str(hypothesis["status"]), "DORMANT")
        self.assertIn("legacy_unbound_revision", hypothesis["invalidation_reason"])
        self.assertEqual(
            self.storage._connection.execute(
                "SELECT COUNT(*) FROM hypothesis_evidence WHERE hypothesis_id=?",
                (hypothesis_id,),
            ).fetchone()[0],
            0,
        )
        unique_indexes = []
        for index in self.storage._connection.execute(
            "PRAGMA index_list(feedback_proposals)"
        ).fetchall():
            if not int(index["unique"] or 0):
                continue
            name = str(index["name"]).replace('"', '""')
            unique_indexes.append(
                (
                    tuple(
                        str(row["name"])
                        for row in self.storage._connection.execute(
                            f'PRAGMA index_info("{name}")'
                        ).fetchall()
                    ),
                    int(index["partial"] or 0),
                    str(index["origin"] or ""),
                )
            )
        self.assertIn(
            (
                ("umo", "feedback_source_key", "feedback_revision_no"),
                0,
                "u",
            ),
            unique_indexes,
        )
        self.assertIn(
            (("umo", "feedback_source_key"), 1, "c"),
            unique_indexes,
        )
        self.assertNotIn(
            (("umo", "feedback_source_key"), 0, "u"),
            unique_indexes,
        )
        self.assertIsNone(
            self.storage._connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='feedback_proposals_legacy'"
            ).fetchone()
        )
        self.assertEqual(
            self.storage._connection.execute("PRAGMA foreign_key_check").fetchall(),
            [],
        )
        self.storage.close()
        self.storage = MemoryStorage(self.database_path)
        rebound = self.storage._connection.execute(
            """
            SELECT feedback_revision_no, feedback_content_sha256,
                   is_current, invalidation_reason
            FROM feedback_proposals WHERE id=?
            """,
            (int(proposal_id or 0),),
        ).fetchone()
        self.assertEqual(int(rebound["feedback_revision_no"]), 0)
        self.assertEqual(str(rebound["feedback_content_sha256"]), "")
        self.assertEqual(int(rebound["is_current"]), 0)
        self.assertIn("legacy_unbound_revision", rebound["invalidation_reason"])

    def test_decision_parser_rejects_unbounded_or_ambiguous_mutations(self) -> None:
        parsed = parse_feedback_decision(
            json.dumps(
                {
                    "target_trace_id": "t-1",
                    "mutation": "upsert",
                    "feedback_valence": -1,
                    "confidence": 0.8,
                    "scope_type": "sender",
                    "scope_key": "user-a",
                    "aspect": "style",
                    "statement": "不要询问是否继续",
                    "prospective_cue": "有后续内容时直接给出。",
                    "trigger_cues": [],
                    "activation_mode": "always",
                },
                ensure_ascii=False,
            )
        )
        self.assertEqual(parsed.mutation, "upsert")
        with self.assertRaises(ValueError):
            parse_feedback_decision({**parsed.as_dict(), "confidence": 1.5})
        with self.assertRaises(ValueError):
            parse_feedback_decision(
                {**parsed.as_dict(), "prospective_cue": "x" * 501}
            )
        with self.assertRaises(ValueError):
            parse_feedback_decision(
                {
                    **parsed.as_dict(),
                    "activation_mode": "semantic",
                    "trigger_cues": [],
                }
            )
        with self.assertRaises(ValueError):
            parse_feedback_decision(
                {
                    **parsed.as_dict(),
                    "activation_mode": "always",
                    "trigger_cues": ["画图"],
                }
            )

    def test_commit_threshold_and_inactive_search_keep_sender_boundary(self) -> None:
        self.open_trace(
            "trace-threshold",
            request_id="req-threshold",
            query="继续",
            sender_id="user-a",
            sent_at=100,
            response="要不要继续？",
        )
        feedback = self.message(
            "feedback-threshold",
            "不要反问",
            sender_id="user-a",
            sent_at=110,
        )
        self.storage.upsert_message(feedback)
        proposal_id = self.storage.enqueue_feedback_candidate(
            umo=self.umo,
            feedback_source_key=feedback.resolved_source_key(),
        )
        weak = FeedbackDecision(
            target_trace_id="trace-threshold",
            mutation="upsert",
            feedback_valence=-1.0,
            confidence=0.4,
            scope_type="sender",
            scope_key="user-a",
            aspect="style",
            statement="不要反问",
            prospective_cue="直接完成回答。",
            trigger_cues=(),
            activation_mode="always",
        )
        provisional = self.storage.apply_feedback_decision(
            umo=self.umo,
            proposal_id=int(proposal_id),
            decision=weak,
            min_commit_score=0.65,
        )
        self.assertEqual(provisional["status"], "COMMITTED")
        self.assertEqual(provisional["hypothesis_status"], "PROVISIONAL")
        self.assertEqual(
            self.storage.search_feedback_hypotheses(
                umo=self.umo,
                sender_id="user-a",
                query="任意",
                at=120,
            ),
            [],
            "weak evidence is retained but must not affect behavior yet",
        )
        retained = self.storage.search_feedback_hypotheses(
            umo=self.umo,
            sender_id="user-a",
            query="任意",
            at=120,
            include_inactive=True,
        )
        self.assertEqual(retained[0]["status"], "PROVISIONAL")
        self.assertEqual(
            self.storage.search_feedback_hypotheses(
                umo=self.umo,
                sender_id="user-b",
                query="任意",
                at=120,
                include_inactive=True,
            ),
            [],
        )

    def test_repeated_weak_feedback_promotes_a_provisional_hypothesis(self) -> None:
        for index, base_at in enumerate((100, 200), start=1):
            trace_id = f"trace-provisional-{index}"
            self.open_trace(
                trace_id,
                request_id=f"req-provisional-{index}",
                query="继续",
                sender_id="user-a",
                sent_at=base_at,
                response="要不要继续？",
            )
            feedback = self.message(
                f"feedback-provisional-{index}",
                "不要反问",
                sender_id="user-a",
                sent_at=base_at + 10,
            )
            self.storage.upsert_message(feedback)
            proposal_id = self.storage.enqueue_feedback_candidate(
                umo=self.umo,
                feedback_source_key=feedback.resolved_source_key(),
            )
            result = self.storage.apply_feedback_decision(
                umo=self.umo,
                proposal_id=int(proposal_id),
                decision=FeedbackDecision(
                    target_trace_id=trace_id,
                    mutation="upsert",
                    feedback_valence=-1.0,
                    confidence=0.4,
                    scope_type="sender",
                    scope_key="user-a",
                    aspect="style",
                    statement="不要反问",
                    prospective_cue="直接完成回答。",
                    trigger_cues=(),
                    activation_mode="always",
                ),
                min_commit_score=0.65,
            )
        self.assertEqual(result["hypothesis_status"], "ACTIVE")
        self.assertEqual(
            len(
                self.storage.search_feedback_hypotheses(
                    umo=self.umo,
                    sender_id="user-a",
                    query="任意",
                    at=220,
                )
            ),
            1,
        )

    def test_feedback_is_strictly_masked_and_sender_scoped(self) -> None:
        self.open_trace(
            "trace-1",
            request_id="req-1",
            query="继续解释",
            sender_id="user-a",
            sent_at=100,
            response="如果你愿意，我可以继续。",
        )
        result = self.learn(
            trace_id="trace-1",
            feedback_id="feedback-1",
            feedback_text="不要问我要不要，直接发后面的内容",
            feedback_at=110,
            cue="不要征询是否继续；有相关后续内容就直接给出。",
        )
        self.assertEqual(result["status"], "COMMITTED")

        self.assertEqual(
            self.storage.activate_feedback_hypotheses(
                umo=self.umo,
                sender_id="user-a",
                query="解释一下",
                at=110,
            ),
            [],
            "the feedback itself must not leak through an equal cutoff",
        )
        active = self.storage.activate_feedback_hypotheses(
            umo=self.umo,
            sender_id="user-a",
            query="解释一下",
            at=111,
        )
        self.assertEqual(len(active), 1)
        self.assertIn("直接给出", str(active[0]["prospective_cue"]))
        self.assertEqual(
            self.storage.activate_feedback_hypotheses(
                umo=self.umo,
                sender_id="user-b",
                query="解释一下",
                at=111,
            ),
            [],
        )

        graph = self.storage.interaction_trace_graph(
            umo=self.umo, trace_id="trace-1"
        )
        self.assertIsNotNone(graph)
        node_types = {node["node_type"] for node in graph["nodes"]}  # type: ignore[index]
        self.assertTrue({"request", "response", "feedback", "hypothesis"} <= node_types)
        dashboard = self.storage.dashboard_graph(umo=self.umo)
        dashboard_types = {node["type"] for node in dashboard["nodes"]}  # type: ignore[index]
        self.assertTrue({"action", "feedback", "hypothesis"} <= dashboard_types)

    def test_later_negative_feedback_changes_utility_not_evidence_confidence(self) -> None:
        self.open_trace(
            "trace-1",
            request_id="req-1",
            query="继续解释",
            sender_id="user-a",
            sent_at=100,
            response="要不要我继续？",
        )
        first = self.learn(
            trace_id="trace-1",
            feedback_id="feedback-1",
            feedback_text="不要反问",
            feedback_at=110,
            cue="直接完成回答，不要反问是否继续。",
        )
        hypothesis_id = int(first["hypothesis_id"])
        before = self.storage.search_feedback_hypotheses(
            umo=self.umo,
            sender_id="user-a",
            query="任意问题",
            at=199,
        )[0]

        self.open_trace(
            "trace-2",
            request_id="req-2",
            query="任意问题",
            sender_id="user-a",
            sent_at=200,
            response="预占位",
        )
        self.storage.activate_feedback_hypotheses(
            umo=self.umo,
            sender_id="user-a",
            query="任意问题",
            at=200,
            trace_id="trace-2",
        )
        self.storage.finish_interaction_trace(
            trace_id="trace-2",
            umo=self.umo,
            response_text="如果你愿意，我可以继续。",
            response_at=201,
        )
        second = self.learn(
            trace_id="trace-2",
            feedback_id="feedback-2",
            feedback_text="还是在反问",
            feedback_at=210,
            cue="避免条件式征询，直接给完整结果。",
        )
        self.assertLess(float(second["backward_credit"]), 0)
        after = self.storage.search_feedback_hypotheses(
            umo=self.umo,
            sender_id="user-a",
            query="任意问题",
            at=211,
            include_inactive=True,
            limit=20,
        )
        original = next(row for row in after if int(row["id"]) == hypothesis_id)
        self.assertLess(float(original["utility"]), float(before["utility"]))
        self.assertEqual(
            float(original["evidence_confidence"]),
            float(before["evidence_confidence"]),
        )

    def test_task_cues_prevent_irrelevant_activation(self) -> None:
        self.open_trace(
            "trace-image",
            request_id="req-image",
            query="帮我画图",
            sender_id="artist",
            sent_at=100,
            response="我先介绍绘图方案。",
        )
        self.learn(
            trace_id="trace-image",
            feedback_id="feedback-image",
            feedback_text="让你直接画",
            feedback_at=110,
            cue="收到明确生图请求时直接调用绘图工具。",
            triggers=("画", "生图", "图片"),
            sender_id="artist",
        )
        self.assertEqual(
            self.storage.activate_feedback_hypotheses(
                umo=self.umo,
                sender_id="artist",
                query="BMI怎么算",
                at=120,
            ),
            [],
        )
        self.assertEqual(
            len(
                self.storage.activate_feedback_hypotheses(
                    umo=self.umo,
                    sender_id="artist",
                    query="画一张图片",
                    at=120,
                )
            ),
            1,
        )

    def test_private_agent_can_bridge_paraphrases_with_a_scoped_activation(self) -> None:
        self.open_trace(
            "trace-source",
            request_id="req-source",
            query="原神和黑神话谁才是国产游戏之光",
            sender_id="chooser",
            sent_at=100,
            response="两个都算，只是方向不同。",
        )
        learned = self.learn(
            trace_id="trace-source",
            feedback_id="feedback-source",
            feedback_text="不要端水，只能选一个",
            feedback_at=110,
            cue="遇到二选一请求时明确选择一个。",
            triggers=("选一个", "谁才是"),
            sender_id="chooser",
        )
        hypothesis_id = int(learned["hypothesis_id"])
        self.open_trace(
            "trace-target",
            request_id="req-target",
            query="今晚吃西餐还是日料",
            sender_id="chooser",
            sent_at=200,
            response="待生成",
        )
        self.assertEqual(
            self.storage.activate_feedback_hypotheses(
                umo=self.umo,
                sender_id="chooser",
                query="今晚吃西餐还是日料",
                at=200,
            ),
            [],
            "surface cues alone cannot bridge this paraphrase",
        )
        candidates = self.storage.feedback_hypothesis_candidates(
            umo=self.umo,
            sender_id="chooser",
            at=200,
        )
        self.assertEqual([int(row["id"]) for row in candidates], [hypothesis_id])
        activated = self.storage.activate_feedback_hypotheses(
            umo=self.umo,
            sender_id="chooser",
            query="今晚吃西餐还是日料",
            at=200,
            trace_id="trace-target",
            selected=[{"id": hypothesis_id, "activation_score": 0.88}],
            activation_method="subconscious_agent",
        )
        self.assertEqual(len(activated), 1)
        row = self.storage._connection.execute(
            """
            SELECT activation_method FROM hypothesis_activations
            WHERE trace_id = ? AND hypothesis_id = ?
            """,
            ("trace-target", hypothesis_id),
        ).fetchone()
        self.assertEqual(row["activation_method"], "subconscious_agent")
        self.assertEqual(
            self.storage.feedback_hypothesis_candidates(
                umo=self.umo,
                sender_id="someone-else",
                at=200,
            ),
            [],
        )

    def test_snapshot_excludes_a_hypothesis_revised_by_future_evidence(self) -> None:
        self.open_trace(
            "trace-historical-1",
            request_id="req-historical-1",
            query="继续",
            sender_id="user-a",
            sent_at=100,
            response="要不要继续？",
        )
        first = self.learn(
            trace_id="trace-historical-1",
            feedback_id="feedback-historical-1",
            feedback_text="不要反问",
            feedback_at=110,
            cue="直接完成回答，不要反问是否继续。",
            sender_id="user-a",
        )
        hypothesis_id = int(first["hypothesis_id"])
        frozen_upper_bound = int(
            self.storage._connection.execute(
                "SELECT MAX(id) FROM messages WHERE umo = ?",
                (self.umo,),
            ).fetchone()[0]
        )
        before_revision = self.storage.feedback_hypothesis_candidates(
            umo=self.umo,
            sender_id="user-a",
            at=150,
            message_upper_bound=frozen_upper_bound,
        )
        self.assertEqual(
            [int(row["id"]) for row in before_revision],
            [hypothesis_id],
        )

        self.open_trace(
            "trace-historical-2",
            request_id="req-historical-2",
            query="继续",
            sender_id="user-a",
            sent_at=200,
            response="还是要不要继续？",
        )
        second = self.learn(
            trace_id="trace-historical-2",
            feedback_id="feedback-historical-2",
            feedback_text="不要反问",
            feedback_at=210,
            cue="直接完成回答，不要反问是否继续。",
            sender_id="user-a",
        )
        self.assertEqual(int(second["hypothesis_id"]), hypothesis_id)
        self.assertEqual(
            self.storage.feedback_hypothesis_candidates(
                umo=self.umo,
                sender_id="user-a",
                at=150,
                message_upper_bound=frozen_upper_bound,
            ),
            [],
            "a mutable head revised after the snapshot must not time-travel",
        )
        self.assertEqual(
            self.storage.feedback_hypothesis_candidates(
                umo=self.umo,
                sender_id="user-a",
                at=250,
                message_upper_bound=frozen_upper_bound,
            ),
            [],
            "the row upper bound must also reject late evidence with old time",
        )
        current = self.storage.feedback_hypothesis_candidates(
            umo=self.umo,
            sender_id="user-a",
            at=250,
        )
        self.assertEqual([int(row["id"]) for row in current], [hypothesis_id])

    def test_inspection_redacts_media_urls_and_merge_is_reversible(self) -> None:
        self.open_trace(
            "trace-1",
            request_id="req-1",
            query="画图",
            sender_id="user-a",
            sent_at=100,
            response="已完成",
        )
        self.storage.record_trace_node(
            trace_id="trace-1",
            umo=self.umo,
            node_key="tool:0:call",
            node_type="tool_call",
            content={
                "tool": "image_generation",
                "argument_keys": ["prompt"],
                "arguments_sha256": "deadbeef",
            },
        )
        self.storage.record_trace_edge(
            trace_id="trace-1",
            umo=self.umo,
            source_key="request",
            target_key="tool:0:call",
            relation="CALLS",
        )
        feedback = self.message(
            "feedback-1",
            "元素太密了",
            sender_id="user-a",
            sent_at=110,
            content=[
                {"type": "reply", "reply_id": "bot-1"},
                {"type": "image", "url": "https://secret.example/token"},
            ],
        )
        self.storage.upsert_message(feedback)
        proposal_id = self.storage.enqueue_feedback_candidate(
            umo=self.umo,
            feedback_source_key=feedback.resolved_source_key(),
        )
        evidence = self.storage.inspect_feedback_proposal(
            umo=self.umo, proposal_id=int(proposal_id)
        )
        rendered = json.dumps(evidence, ensure_ascii=False)
        self.assertNotIn("secret.example", rendered)
        self.assertIn("bot-1", rendered)
        self.assertIn("image_generation", rendered)

        first = self.storage.apply_feedback_decision(
            umo=self.umo,
            proposal_id=int(proposal_id),
            decision=FeedbackDecision(
                target_trace_id="trace-1",
                mutation="upsert",
                feedback_valence=-1,
                confidence=0.9,
                scope_type="sender",
                scope_key="user-a",
                aspect="image_density",
                statement="元素太密",
                prospective_cue="减少画面元素密度。",
                trigger_cues=("画",),
                activation_mode="semantic",
            ),
        )
        first_id = int(first["hypothesis_id"])

        self.open_trace(
            "trace-2",
            request_id="req-2",
            query="画图",
            sender_id="user-a",
            sent_at=200,
            response="又画了一张",
        )
        second = self.learn(
            trace_id="trace-2",
            feedback_id="feedback-2",
            feedback_text="留白再多点",
            feedback_at=210,
            cue="增加构图留白。",
            triggers=("画",),
        )
        second_id = int(second["hypothesis_id"])
        self.storage.merge_feedback_hypotheses(
            umo=self.umo, source_id=second_id, target_id=first_id
        )
        with self.assertRaises(ValueError):
            self.storage.merge_feedback_hypotheses(
                umo=self.umo, source_id=first_id, target_id=second_id
            )
        merged = self.storage.search_feedback_hypotheses(
            umo=self.umo,
            sender_id="user-a",
            query="画图",
            at=220,
            include_inactive=True,
            limit=20,
        )
        self.assertEqual(
            next(row for row in merged if int(row["id"]) == second_id)["status"],
            "MERGED",
        )
        self.storage.unmerge_feedback_hypothesis(
            umo=self.umo, source_id=second_id
        )
        active = self.storage.activate_feedback_hypotheses(
            umo=self.umo,
            sender_id="user-a",
            query="画图",
            at=220,
            limit=10,
        )
        self.assertEqual({int(row["id"]) for row in active}, {first_id, second_id})

    def test_compaction_bounds_active_view_without_deleting_evidence(self) -> None:
        learned_ids: list[int] = []
        for index in range(3):
            sent_at = 100 + index * 20
            trace_id = f"trace-{index}"
            self.open_trace(
                trace_id,
                request_id=f"req-{index}",
                query="任务",
                sender_id="user-a",
                sent_at=sent_at,
                response="结果",
            )
            learned = self.learn(
                trace_id=trace_id,
                feedback_id=f"feedback-{index}",
                feedback_text=f"修正 {index}",
                feedback_at=sent_at + 10,
                cue=f"规则 {index}",
            )
            learned_ids.append(int(learned["hypothesis_id"]))
        result = self.storage.compact_feedback_memory(
            umo=self.umo,
            now=1000,
            max_active_hypotheses=1,
        )
        self.assertEqual(result["dormant_by_budget"], 2)
        all_rows = self.storage.search_feedback_hypotheses(
            umo=self.umo,
            sender_id="user-a",
            query="任务",
            at=1001,
            include_inactive=True,
            limit=20,
        )
        self.assertEqual({int(row["id"]) for row in all_rows}, set(learned_ids))
        self.assertEqual(sum(row["status"] == "ACTIVE" for row in all_rows), 1)


if __name__ == "__main__":
    unittest.main()
