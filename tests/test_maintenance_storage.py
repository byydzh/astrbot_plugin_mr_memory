from __future__ import annotations

import unittest
import uuid
from pathlib import Path
from unittest import mock

from mr_memory.models import NormalizedMessage
from mr_memory.storage import MemoryStorage


class PendingDistillationSchedulingTests(unittest.TestCase):
    umo = "synthetic:GroupMessage:maintenance-recovery"

    def setUp(self) -> None:
        test_root = Path.cwd() / ".dev" / "test-tmp"
        test_root.mkdir(parents=True, exist_ok=True)
        self.database_path = test_root / f"{uuid.uuid4().hex}.db"
        self.storage = MemoryStorage(self.database_path)

    def tearDown(self) -> None:
        self.storage.close()
        for suffix in ("", "-wal", "-shm"):
            Path(f"{self.database_path}{suffix}").unlink(missing_ok=True)

    def add_message(self, index: int) -> str:
        message = NormalizedMessage(
            platform="aiocqhttp",
            platform_id="synthetic",
            umo=self.umo,
            group_id="maintenance-recovery",
            message_id=f"synthetic-message-{index}",
            sender_id="synthetic-account",
            sender_name="SyntheticParticipant",
            sent_at=100 + index,
            plain_text=f"synthetic message content {index}",
            content=[],
        )
        self.storage.upsert_message(message)
        return message.resolved_source_key()

    def test_distinct_pending_batch_survives_failed_jobs_without_retrying_failed_rows(self) -> None:
        sources = [self.add_message(index) for index in range(3)]
        legacy_id = self.storage.enqueue_maintenance_job(
            umo=self.umo, job_type="distill", dedupe_key="distill:pending", available_at=100
        )
        self.storage.claim_maintenance_job(umo=self.umo, job_id=legacy_id, now=100)
        self.storage.fail_maintenance_job(
            umo=self.umo, job_id=legacy_id, error="synthetic legacy failure", now=100
        )

        first_id = self.storage.enqueue_pending_distillation_job(
            umo=self.umo, limit=2, available_at=101
        )
        self.assertIsNotNone(first_id)
        self.assertNotEqual(first_id, legacy_id)
        self.storage.claim_maintenance_job(umo=self.umo, job_id=first_id, now=101)
        failed_batch = self.storage.next_distillation_batch(
            umo=self.umo, limit=2, overlap=0, processing_class="LIVE"
        )
        self.assertEqual(failed_batch.target_source_keys, tuple(sources[:2]))
        self.storage.finish_distillation_batch(work_item=failed_batch, error="synthetic invalid response")
        self.storage.fail_maintenance_job(
            umo=self.umo, job_id=first_id, error="synthetic invalid response", now=102
        )

        next_id = self.storage.enqueue_pending_distillation_job(
            umo=self.umo, limit=2, available_at=103
        )
        self.assertNotIn(next_id, (None, legacy_id, first_id))
        self.storage.claim_maintenance_job(umo=self.umo, job_id=next_id, now=103)
        next_batch = self.storage.next_distillation_batch(
            umo=self.umo, limit=2, overlap=0, processing_class="LIVE"
        )
        self.assertEqual(next_batch.target_source_keys, (sources[2],))
        self.storage.finish_distillation_batch(work_item=next_batch)
        self.storage.finish_maintenance_job(umo=self.umo, job_id=next_id)
        self.assertIsNone(self.storage.enqueue_pending_distillation_job(umo=self.umo, limit=2))
        failures = self.storage._connection.execute(
            "SELECT id, last_error FROM maintenance_jobs WHERE status='FAILED' ORDER BY id"
        ).fetchall()
        self.assertEqual([row["id"] for row in failures], [legacy_id, first_id])
        self.assertEqual(failures[1]["last_error"], "synthetic invalid response")
        self.assertEqual(
            self.storage._connection.execute(
                "SELECT COUNT(*) FROM message_processing WHERE status='FAILED'"
            ).fetchone()[0],
            2,
        )

    def test_active_job_absorbs_arrivals_but_same_failed_work_is_not_requeued(self) -> None:
        self.add_message(0)
        job_id = self.storage.enqueue_pending_distillation_job(
            umo=self.umo, limit=1, available_at=200
        )
        self.add_message(1)
        self.assertEqual(
            self.storage.enqueue_pending_distillation_job(umo=self.umo, limit=1, available_at=100),
            job_id,
        )
        self.storage.claim_maintenance_job(umo=self.umo, job_id=job_id, now=100)
        self.assertEqual(
            self.storage.enqueue_pending_distillation_job(umo=self.umo, limit=1, available_at=101),
            job_id,
        )
        self.storage.fail_maintenance_job(
            umo=self.umo, job_id=job_id, error="synthetic provider unavailable before batch claim", now=102
        )
        self.assertEqual(
            self.storage.enqueue_pending_distillation_job(umo=self.umo, limit=1, available_at=103),
            job_id,
        )
        self.assertEqual(self.storage.pending_maintenance_jobs(umo=self.umo, now=103), [])
        self.assertEqual(self.storage.pending_distillation_count(umo=self.umo, processing_class="LIVE"), 2)

    def test_budget_deferral_keeps_deadline_and_cannot_reopen_failure(self) -> None:
        self.add_message(0)
        job_id = self.storage.enqueue_pending_distillation_job(
            umo=self.umo, limit=1, available_at=100
        )
        claimed = self.storage.claim_maintenance_job(umo=self.umo, job_id=job_id, now=100)
        self.assertTrue(self.storage.defer_maintenance_job(
            umo=self.umo, job_id=job_id, available_at=2000, reason="budget_exhausted:online"
        ))
        self.assertEqual(
            self.storage.enqueue_pending_distillation_job(umo=self.umo, limit=1, available_at=150),
            job_id,
        )
        self.storage.enqueue_maintenance_job(
            umo=self.umo,
            job_type="distill",
            dedupe_key=claimed["dedupe_key"],
            payload=claimed["payload"],
            available_at=150,
        )
        deferred = self.storage._connection.execute(
            "SELECT * FROM maintenance_jobs WHERE id=?", (job_id,)
        ).fetchone()
        self.assertEqual(deferred["available_at"], 2000)
        self.assertEqual(deferred["attempts"], 1)
        self.assertEqual(deferred["payload_json"], claimed["payload_json"])
        self.assertIsNone(deferred["lease_until"])
        with mock.patch("mr_memory.storage.time.time", return_value=1999):
            self.assertFalse(self.storage.maintenance_job_ready(umo=self.umo, job_id=job_id))
        with mock.patch("mr_memory.storage.time.time", return_value=2000):
            self.assertTrue(self.storage.maintenance_job_ready(umo=self.umo, job_id=job_id))
        self.storage.claim_maintenance_job(umo=self.umo, job_id=job_id, now=2000)
        self.storage.fail_maintenance_job(
            umo=self.umo, job_id=job_id, error="synthetic genuine failure", now=2001
        )
        self.assertFalse(self.storage.defer_maintenance_job(
            umo=self.umo, job_id=job_id, available_at=3000, reason="budget_exhausted:online"
        ))
        failed = self.storage._connection.execute(
            "SELECT status,last_error FROM maintenance_jobs WHERE id=?", (job_id,)
        ).fetchone()
        self.assertEqual(failed["status"], "FAILED")
        self.assertEqual(failed["last_error"], "synthetic genuine failure")

    def test_owned_startup_recovers_interrupted_batch_without_reprocessing_commit(self) -> None:
        for index in range(3):
            self.add_message(index)
        committed = self.storage.next_distillation_batch(umo=self.umo, limit=1, overlap=0)
        self.storage.finish_distillation_batch(work_item=committed)
        job_id = self.storage.enqueue_pending_distillation_job(
            umo=self.umo, limit=2, available_at=100
        )
        self.storage.claim_maintenance_job(umo=self.umo, job_id=job_id, now=100)
        interrupted = self.storage.next_distillation_batch(umo=self.umo, limit=2, overlap=0)
        recovered = self.storage.recover_distillation_runtime(
            umo=self.umo, batch_limit=2, now=101
        )
        self.assertEqual(recovered["requeued_sources"], 2)
        self.assertEqual(recovered["recovery_jobs"], 1)
        self.assertEqual(self.storage.pending_distillation_count(umo=self.umo), 2)
        statuses = dict(self.storage._connection.execute(
            "SELECT batch_key,status FROM distillation_batches"
        ).fetchall())
        self.assertEqual(statuses[committed.batch_key], "COMPLETED")
        self.assertEqual(statuses[interrupted.batch_key], "FAILED")
        repeated = self.storage.recover_distillation_runtime(umo=self.umo, batch_limit=2)
        self.assertEqual(repeated["requeued_sources"], 0)
        self.assertEqual(repeated["recovery_jobs"], 0)
        resumed = self.storage.pending_maintenance_jobs(umo=self.umo, now=101)[0]
        self.storage.claim_maintenance_job(umo=self.umo, job_id=resumed["id"], now=101)
        cancelled = self.storage.next_distillation_batch(umo=self.umo, limit=2, overlap=0)
        self.storage.finish_distillation_batch(work_item=cancelled, error="CancelledError")
        self.storage.fail_maintenance_job(
            umo=self.umo, job_id=resumed["id"], error="CancelledError", now=102
        )
        interrupted_again = self.storage.recover_distillation_runtime(
            umo=self.umo, batch_limit=2, now=103
        )
        self.assertEqual(interrupted_again["requeued_sources"], 2)
        self.assertEqual(interrupted_again["recovery_jobs"], 1)

    def test_preclaim_failure_has_one_audited_recovery_without_retry_chain(self) -> None:
        self.add_message(0)
        job_id = self.storage.enqueue_pending_distillation_job(
            umo=self.umo, limit=1, available_at=100
        )
        self.storage.claim_maintenance_job(umo=self.umo, job_id=job_id, now=100)
        self.storage.fail_maintenance_job(
            umo=self.umo, job_id=job_id, error="provider unavailable", now=100
        )
        recovered = self.storage.recover_distillation_runtime(
            umo=self.umo, batch_limit=1, now=101
        )
        self.assertEqual(recovered["recovery_jobs"], 1)
        jobs = self.storage.pending_maintenance_jobs(umo=self.umo, now=101)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["payload"]["recovery_of_job_id"], job_id)
        old = self.storage._connection.execute(
            "SELECT status,attempts,last_error FROM maintenance_jobs WHERE id=?", (job_id,)
        ).fetchone()
        self.assertEqual(tuple(old), ("FAILED", 1, "provider unavailable"))
        self.storage.claim_maintenance_job(umo=self.umo, job_id=jobs[0]["id"], now=101)
        self.storage.fail_maintenance_job(
            umo=self.umo, job_id=jobs[0]["id"], error="still unavailable", now=102
        )
        self.assertEqual(self.storage.recover_distillation_runtime(
            umo=self.umo, batch_limit=1, now=103
        )["recovery_jobs"], 0)
        self.assertEqual(self.storage.pending_maintenance_jobs(umo=self.umo, now=103), [])

    def test_cancelled_batch_requeues_but_genuine_failure_stays_failed(self) -> None:
        for index in range(2):
            self.add_message(index)
        cancelled = self.storage.next_distillation_batch(umo=self.umo, limit=1, overlap=0)
        self.storage.finish_distillation_batch(work_item=cancelled, error="CancelledError")
        failed = self.storage.next_distillation_batch(umo=self.umo, limit=1, overlap=0)
        self.storage.finish_distillation_batch(work_item=failed, error="invalid provider response")
        recovered = self.storage.recover_distillation_runtime(umo=self.umo, batch_limit=1)
        self.assertEqual(recovered["requeued_sources"], 1)
        self.assertEqual(self.storage.pending_distillation_count(umo=self.umo), 1)
        self.assertEqual(self.storage._connection.execute(
            "SELECT COUNT(*) FROM message_processing WHERE status='FAILED'"
        ).fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main()
