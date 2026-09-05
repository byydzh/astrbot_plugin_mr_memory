from __future__ import annotations

import json
import unittest
import uuid
from pathlib import Path

from mr_memory.models import NormalizedMessage
from mr_memory.storage import MemoryStorage


class FeedbackSchedulingTests(unittest.TestCase):
    umo = "synthetic:GroupMessage:feedback-scheduling"

    def setUp(self) -> None:
        directory = Path.cwd() / ".dev" / "test-tmp"
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{uuid.uuid4().hex}.db"
        self.storage = MemoryStorage(self.path)

    def tearDown(self) -> None:
        self.storage.close()
        for suffix in ("", "-wal", "-shm"):
            Path(f"{self.path}{suffix}").unlink(missing_ok=True)

    def message(self, index: int, text: str = "synthetic feedback") -> NormalizedMessage:
        return NormalizedMessage(
            platform="aiocqhttp", platform_id="synthetic", umo=self.umo,
            group_id="feedback-scheduling", message_id=f"feedback-{index}",
            sender_id="synthetic-account", sender_name="Synthetic Participant",
            sent_at=100 + index, plain_text=text, content=[],
        )

    def add_proposal(self, index: int) -> int:
        message = self.message(index)
        self.storage.upsert_message(message)
        with self.storage._connection:
            cursor = self.storage._connection.execute(
                """INSERT INTO feedback_proposals(
                       umo,feedback_source_key,feedback_message_id,
                       feedback_revision_no,feedback_content_sha256,feedback_sent_at
                   ) SELECT umo,source_key,id,revision_no,content_sha256,sent_at
                     FROM messages WHERE source_key=?""",
                (message.resolved_source_key(),),
            )
        return int(cursor.lastrowid)

    def job(self, job_id: int) -> dict:
        row = dict(self.storage._connection.execute(
            "SELECT * FROM maintenance_jobs WHERE id=?", (job_id,),
        ).fetchone())
        row["payload"] = json.loads(row["payload_json"])
        return row

    def test_failed_batch_only_consumes_attempted_proposals_and_later_work_runs(self) -> None:
        first, second = self.add_proposal(1), self.add_proposal(2)
        legacy = self.storage.enqueue_maintenance_job(
            umo=self.umo, job_type="feedback", dedupe_key="feedback:batch", available_at=100,
        )
        self.storage.claim_maintenance_job(umo=self.umo, job_id=legacy, now=100)
        self.storage.fail_maintenance_job(umo=self.umo, job_id=legacy, error="legacy error", now=101)
        job_id = self.storage.enqueue_pending_feedback_job(umo=self.umo, limit=2, available_at=102)
        self.assertNotEqual(job_id, legacy)
        frozen = self.job(job_id)["payload"]["proposal_snapshots"]
        third = self.add_proposal(3)
        self.assertEqual(self.storage.enqueue_pending_feedback_job(umo=self.umo, limit=3), job_id)
        self.assertEqual(self.job(job_id)["payload"]["proposal_snapshots"], frozen)
        self.assertEqual([row["id"] for row in self.storage.pending_feedback_proposals(
            umo=self.umo, limit=20, snapshots=frozen,
        )], [first, second])
        self.storage.claim_maintenance_job(umo=self.umo, job_id=job_id, now=103)
        with self.storage._connection:
            self.storage._connection.execute("UPDATE feedback_proposals SET error='earlier diagnostic' WHERE id=?", (first,))
        self.assertEqual(self.storage.fail_feedback_proposals(
            umo=self.umo, snapshots=frozen[:1], error="synthetic provider failure",
        ), 1)
        self.storage.fail_maintenance_job(umo=self.umo, job_id=job_id, error="synthetic provider failure", now=104)
        next_job = self.storage.enqueue_pending_feedback_job(umo=self.umo, limit=2, available_at=105)
        self.assertNotIn(next_job, (None, legacy, job_id))
        self.assertEqual([item["id"] for item in self.job(next_job)["payload"]["proposal_snapshots"]], [second, third])
        status = self.storage.feedback_proposal_status(umo=self.umo, proposal_id=first)
        self.assertEqual(status["status"], "FAILED")
        self.assertEqual(status["error"], "earlier diagnostic\nsynthetic provider failure")
        self.assertEqual(self.job(legacy)["last_error"], "legacy error")
        self.assertEqual(self.job(job_id)["status"], "FAILED")
        self.assertEqual(self.storage.fail_feedback_proposals(umo=self.umo, snapshots=frozen[:1], error="second failure"), 0)
        self.assertEqual(self.storage.runtime_health_summary(umo=self.umo)["feedback_queue"]["proposal_status"]["failed"], 1)

    def test_frozen_batch_excludes_revisions_commits_and_new_arrivals(self) -> None:
        revised, committed, pending = (self.add_proposal(index) for index in (1, 2, 3))
        job_id = self.storage.enqueue_pending_feedback_job(umo=self.umo, limit=3)
        frozen = self.job(job_id)["payload"]["proposal_snapshots"]
        self.storage.upsert_message(self.message(1, "synthetic revised feedback"))
        with self.storage._connection:
            self.storage._connection.execute("UPDATE feedback_proposals SET status='COMMITTED' WHERE id=?", (committed,))
        arrival = self.add_proposal(4)
        selected = self.storage.pending_feedback_proposals(umo=self.umo, limit=20, snapshots=frozen)
        self.assertEqual([row["id"] for row in selected], [pending])
        self.assertEqual(self.storage.fail_feedback_proposals(umo=self.umo, snapshots=frozen, error="synthetic response error"), 1)
        rows = {row["id"]: dict(row) for row in self.storage._connection.execute("SELECT id,status,is_current FROM feedback_proposals")}
        self.assertEqual(rows[committed]["status"], "COMMITTED")
        self.assertFalse(rows[revised]["is_current"])
        self.assertNotEqual(rows[revised]["status"], "FAILED")
        self.assertEqual(rows[pending]["status"], "FAILED")
        self.assertEqual(rows[arrival]["status"], "PENDING")

    def test_budget_deferral_retains_frozen_batch_and_deadline(self) -> None:
        self.add_proposal(1)
        job_id = self.storage.enqueue_pending_feedback_job(umo=self.umo, limit=1, available_at=100)
        frozen = self.job(job_id)["payload"]
        self.storage.claim_maintenance_job(umo=self.umo, job_id=job_id, now=100)
        self.storage.defer_maintenance_job(umo=self.umo, job_id=job_id, available_at=2000, reason="budget_exhausted:feedback")
        self.add_proposal(2)
        self.assertEqual(self.storage.enqueue_pending_feedback_job(umo=self.umo, limit=2, available_at=101), job_id)
        self.assertEqual(self.job(job_id)["available_at"], 2000)
        self.assertEqual(self.job(job_id)["payload"], frozen)

    def test_failure_before_attempt_pauses_only_frozen_batch(self) -> None:
        paused = self.add_proposal(1)
        failed_job = self.storage.enqueue_pending_feedback_job(umo=self.umo, limit=6, available_at=100)
        self.storage.claim_maintenance_job(umo=self.umo, job_id=failed_job, now=100)
        self.storage.fail_maintenance_job(umo=self.umo, job_id=failed_job, error="synthetic preparation failure", now=101)
        self.assertIsNone(self.storage.enqueue_pending_feedback_job(umo=self.umo, limit=6))
        later = self.add_proposal(2)
        next_job = self.storage.enqueue_pending_feedback_job(umo=self.umo, limit=6)
        self.assertNotIn(next_job, (None, failed_job))
        self.assertEqual([item["id"] for item in self.job(next_job)["payload"]["proposal_snapshots"]], [later])
        self.assertEqual(self.storage.feedback_proposal_status(umo=self.umo, proposal_id=paused)["status"], "PENDING")
        self.assertEqual(self.job(failed_job)["last_error"], "synthetic preparation failure")


if __name__ == "__main__":
    unittest.main()
