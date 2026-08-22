from __future__ import annotations

import unittest
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from mr_memory.feedback import FeedbackDecision
from mr_memory.models import NormalizedMessage
from mr_memory.plasticity import parse_graph_mutation
from mr_memory.storage import MemoryStorage


class MemoryStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        test_root = Path.cwd() / ".dev" / "test-tmp"
        test_root.mkdir(parents=True, exist_ok=True)
        self.database_path = test_root / f"{uuid.uuid4().hex}.db"
        self.storage = MemoryStorage(self.database_path)

    def tearDown(self) -> None:
        self.storage.close()
        for suffix in ("", "-wal", "-shm"):
            Path(f"{self.database_path}{suffix}").unlink(missing_ok=True)

    @staticmethod
    def message(
        message_id: str,
        text: str,
        *,
        umo: str = "shadow:GroupMessage:group-a",
        sender_id: str = "user-a",
        sender_name: str = "",
        sent_at: int = 100,
    ) -> NormalizedMessage:
        return NormalizedMessage(
            platform="aiocqhttp",
            platform_id="shadow",
            umo=umo,
            group_id=umo.rsplit(":", 1)[-1],
            message_id=message_id,
            sender_id=sender_id,
            sender_name=sender_name or sender_id,
            sent_at=sent_at,
            plain_text=text,
            content=[{"type": "plain", "text": text}],
        )

    def test_upsert_is_idempotent_and_updates_fts(self) -> None:
        self.assertTrue(self.storage.upsert_message(self.message("1", "方案 A")))
        self.assertFalse(self.storage.upsert_message(self.message("1", "方案 B")))
        self.assertEqual(self.storage.count_messages(), 1)
        self.assertEqual(
            [
                item.plain_text
                for item in self.storage.search_messages(
                    umo="shadow:GroupMessage:group-a", query="方案 B"
                )
            ],
            ["方案 B"],
        )
        self.assertEqual(
            self.storage.search_messages(
                umo="shadow:GroupMessage:group-a", query="方案 A"
            ),
            [],
        )

    def test_search_is_scoped_to_umo(self) -> None:
        self.storage.upsert_message(self.message("1", "秘密决策"))
        self.storage.upsert_message(
            self.message(
                "1",
                "秘密决策",
                umo="shadow:GroupMessage:group-b",
                sender_id="user-b",
            )
        )
        results = self.storage.search_messages(
            umo="shadow:GroupMessage:group-a",
            query="秘密决策",
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].umo, "shadow:GroupMessage:group-a")

    def test_recent_results_are_returned_chronologically(self) -> None:
        self.storage.upsert_message(self.message("1", "早", sent_at=100))
        self.storage.upsert_message(self.message("2", "晚", sent_at=200))
        results = self.storage.search_messages(
            umo="shadow:GroupMessage:group-a",
            limit=2,
        )
        self.assertEqual([item.plain_text for item in results], ["早", "晚"])

    def test_query_alias_and_recent_activity_are_snapshot_bounded(self) -> None:
        umo = "shadow:GroupMessage:group-a"
        local_timezone = ZoneInfo("Asia/Shanghai")

        def epoch(day: int, hour: int, minute: int = 0) -> int:
            return int(
                datetime(2026, 8, day, hour, minute, tzinfo=local_timezone).timestamp()
            )

        cutoff = epoch(22, 16)
        first = self.message(
            "mllop-1",
            "凌晨发言",
            umo=umo,
            sender_id="account-mllop",
            sender_name="mllop",
            sent_at=epoch(22, 1, 15),
        )
        second = self.message(
            "mllop-2",
            "深夜发言",
            umo=umo,
            sender_id="account-mllop",
            sender_name="mllop新名",
            sent_at=epoch(21, 23, 45),
        )
        too_old = self.message(
            "mllop-old",
            "窗口之外",
            umo=umo,
            sender_id="account-mllop",
            sender_name="mllop",
            sent_at=epoch(15, 15, 59),
        )
        for message in (first, second, too_old):
            self.storage.upsert_message(message)
        message_upper_bound = int(
            self.storage._connection.execute(
                "SELECT MAX(id) FROM messages WHERE umo=?",
                (umo,),
            ).fetchone()[0]
        )

        # This row arrives after the frozen snapshot even though its platform
        # timestamp lies inside the seven-day window.
        self.storage.upsert_message(
            self.message(
                "mllop-late-arrival",
                "快照之后才写入",
                umo=umo,
                sender_id="account-mllop",
                sender_name="未来昵称",
                sent_at=epoch(22, 2),
            )
        )

        resolved = self.storage.resolve_query_participants(
            umo=umo,
            query="/chat 通过最近几天mllop的发言时间预测何时醒",
            before_sent_at=cutoff,
            message_upper_bound=message_upper_bound,
        )
        self.assertFalse(resolved["ambiguous"])
        self.assertEqual(len(resolved["participants"]), 1)
        participant = resolved["participants"][0]
        self.assertEqual(participant["account_id"], "account-mllop")
        self.assertEqual(participant["matched_aliases"], ["mllop"])
        self.assertEqual(
            participant["matched_alias_observations"],
            [
                {
                    "alias": "mllop",
                    "source_key": first.resolved_source_key(),
                    "sent_at": first.sent_at,
                    "source_kind": "observed",
                }
            ],
        )
        self.assertEqual(
            self.storage.resolve_query_participants(
                umo=umo,
                query="未来昵称什么时候出现",
                before_sent_at=cutoff,
                message_upper_bound=message_upper_bound,
            )["participants"],
            [],
        )

        activity = self.storage.query_participant_activity(
            umo=umo,
            participant_key=str(participant["canonical_key"]),
            before_sent_at=cutoff,
            message_upper_bound=message_upper_bound,
            days=7,
        )
        self.assertTrue(activity["found"])
        self.assertEqual(activity["timezone"], "Asia/Shanghai")
        self.assertEqual(activity["message_count"], 2)
        self.assertEqual(activity["hour_histogram"]["01"], 1)
        self.assertEqual(activity["hour_histogram"]["23"], 1)
        self.assertEqual(
            [item["source_key"] for item in activity["messages"]],
            [second.resolved_source_key(), first.resolved_source_key()],
        )
        self.assertTrue(
            all(
                set(item) == {
                    "source_key",
                    "sent_at",
                    "local_datetime",
                    "local_hour",
                }
                for item in activity["messages"]
            )
        )
        self.assertEqual(activity["statistics_basis"], "returned_source_messages_only")
        self.assertEqual(
            activity["sampling_method"],
            "daily_boundaries_plus_message_order_quantiles",
        )

    def test_query_alias_reports_ambiguous_accounts_without_guessing(self) -> None:
        umo = "shadow:GroupMessage:group-a"
        for account_id in ("account-one", "account-two"):
            self.storage.upsert_message(
                self.message(
                    account_id,
                    "同名成员发言",
                    umo=umo,
                    sender_id=account_id,
                    sender_name="mllop",
                    sent_at=100,
                )
            )
        message_upper_bound = int(
            self.storage._connection.execute(
                "SELECT MAX(id) FROM messages WHERE umo=?",
                (umo,),
            ).fetchone()[0]
        )
        result = self.storage.resolve_query_participants(
            umo=umo,
            query="mllop最近什么时候发言",
            before_sent_at=200,
            message_upper_bound=message_upper_bound,
        )
        self.assertTrue(result["ambiguous"])
        self.assertEqual(result["participants"], [])
        self.assertEqual(len(result["ambiguous_aliases"]), 1)
        candidates = result["ambiguous_aliases"][0]["candidate_participants"]
        self.assertEqual(
            {item["account_id"] for item in candidates},
            {"account-one", "account-two"},
        )

    def test_query_alias_keeps_independent_short_alias_occurrence_ambiguous(self) -> None:
        umo = "shadow:GroupMessage:group-a"
        records = (
            ("d", "account-d", "D老师"),
            ("one", "account-one", "老师"),
            ("two", "account-two", "老师"),
        )
        for message_id, account_id, alias in records:
            self.storage.upsert_message(
                self.message(
                    message_id,
                    "身份观察",
                    umo=umo,
                    sender_id=account_id,
                    sender_name=alias,
                    sent_at=100,
                )
            )
        message_upper_bound = int(
            self.storage._connection.execute(
                "SELECT MAX(id) FROM messages WHERE umo=?",
                (umo,),
            ).fetchone()[0]
        )

        result = self.storage.resolve_query_participants(
            umo=umo,
            query="D老师和老师最近谁活跃",
            before_sent_at=200,
            message_upper_bound=message_upper_bound,
        )

        self.assertTrue(result["ambiguous"])
        self.assertEqual(
            {item["account_id"] for item in result["participants"]},
            {"account-d"},
        )
        self.assertEqual(len(result["ambiguous_aliases"]), 1)
        self.assertEqual(
            {
                item["account_id"]
                for item in result["ambiguous_aliases"][0][
                    "candidate_participants"
                ]
            },
            {"account-one", "account-two"},
        )

    def test_activity_target_does_not_bind_common_words_used_as_aliases(self) -> None:
        umo = "shadow:GroupMessage:group-a"
        for message_id, account_id, alias in (
            ("common", "account-common", "最近"),
            ("when", "account-when", "什么时候"),
            ("what-time", "account-what-time", "何时"),
            ("target", "account-mllop", "mllop"),
        ):
            self.storage.upsert_message(
                self.message(
                    message_id,
                    "身份观察",
                    umo=umo,
                    sender_id=account_id,
                    sender_name=alias,
                    sent_at=100,
                )
            )
        message_upper_bound = int(
            self.storage._connection.execute(
                "SELECT MAX(id) FROM messages WHERE umo=?",
                (umo,),
            ).fetchone()[0]
        )

        result = self.storage.resolve_query_participants(
            umo=umo,
            query="/chat 通过最近几天mllop的发言时间来预测它什么时候醒",
            before_sent_at=200,
            message_upper_bound=message_upper_bound,
        )

        self.assertFalse(result["ambiguous"])
        self.assertEqual(
            {item["account_id"] for item in result["participants"]},
            {"account-mllop"},
        )

    def test_activity_statistics_use_only_bounded_daily_spanning_sources(self) -> None:
        umo = "shadow:GroupMessage:group-a"
        local_timezone = ZoneInfo("Asia/Shanghai")

        def epoch(day: int, hour: int) -> int:
            return int(
                datetime(2026, 8, day, hour, tzinfo=local_timezone).timestamp()
            )

        expected_boundaries: set[str] = set()
        for day in (20, 21):
            for hour in range(20):
                message = self.message(
                    f"activity-{day}-{hour}",
                    "活动采样",
                    umo=umo,
                    sender_id="account-active",
                    sender_name="活跃成员",
                    sent_at=epoch(day, hour),
                )
                self.storage.upsert_message(message)
                if hour in {0, 19}:
                    expected_boundaries.add(message.resolved_source_key())
        message_upper_bound = int(
            self.storage._connection.execute(
                "SELECT MAX(id) FROM messages WHERE umo=?",
                (umo,),
            ).fetchone()[0]
        )
        resolved = self.storage.resolve_participants(
            umo=umo,
            reference="account-active",
            before_sent_at=epoch(22, 0),
            message_upper_bound=message_upper_bound,
        )
        participant_key = str(resolved["participants"][0]["canonical_key"])

        activity = self.storage.query_participant_activity(
            umo=umo,
            participant_key=participant_key,
            before_sent_at=epoch(22, 0),
            message_upper_bound=message_upper_bound,
            days=7,
            limit=4,
        )

        returned_sources = {
            item["source_key"] for item in activity["messages"]
        }
        self.assertEqual(returned_sources, expected_boundaries)
        self.assertEqual(activity["message_count"], 4)
        self.assertEqual(sum(activity["hour_histogram"].values()), 4)
        self.assertTrue(activity["messages_truncated"])
        self.assertEqual(activity["statistics_basis"], "returned_source_messages_only")

    def test_timestamp_correction_is_a_revision_and_requeues_distillation(self) -> None:
        umo = "shadow:GroupMessage:group-a"
        original = self.message("time", "同一条消息", sent_at=100)
        self.storage.upsert_message(original)
        work_item = self.storage.next_distillation_batch(
            umo=umo,
            limit=1,
            overlap=0,
        )
        assert work_item is not None
        self.storage.finish_distillation_batch(work_item=work_item)

        self.storage.upsert_message(self.message("time", "同一条消息", sent_at=125))
        stored = self.storage.search_messages(umo=umo, limit=1)[0]
        self.assertEqual(stored.sent_at, 125)
        self.assertEqual(stored.revision_no, 2)
        revision = self.storage._connection.execute(
            """
            SELECT revision_kind, sent_at FROM message_revisions
            WHERE message_id=? ORDER BY revision_no DESC LIMIT 1
            """,
            (stored.id,),
        ).fetchone()
        self.assertEqual(revision["revision_kind"], "TIMESTAMP_CORRECTED")
        self.assertEqual(revision["sent_at"], 100)
        self.assertEqual(
            self.storage.pending_distillation_count(
                umo=umo,
                processing_class="LIVE",
            ),
            1,
        )

    def test_future_maintenance_deadline_survives_reopen(self) -> None:
        umo = "shadow:GroupMessage:group-a"
        self.storage.bind_scope(
            umo=umo,
            platform_id="shadow",
            group_id="group-a",
        )
        job_id = self.storage.enqueue_maintenance_job(
            umo=umo,
            job_type="distill",
            dedupe_key="distill:pending",
            available_at=2**31,
        )
        self.storage.close()
        self.storage = MemoryStorage(self.database_path)
        jobs = self.storage.pending_maintenance_jobs(
            umo=umo,
            job_type="distill",
            include_future=True,
        )
        self.assertEqual([int(job["id"]) for job in jobs], [job_id])
        self.assertEqual(int(jobs[0]["available_at"]), 2**31)

    def test_legacy_completed_maintenance_job_is_requeued(self) -> None:
        umo = "shadow:GroupMessage:group-a"
        self.storage.bind_scope(
            umo=umo,
            platform_id="shadow",
            group_id="group-a",
        )
        job_id = self.storage.enqueue_maintenance_job(
            umo=umo,
            job_type="feedback",
            dedupe_key="feedback:batch",
            available_at=100,
        )
        self.assertIsNotNone(
            self.storage.claim_maintenance_job(umo=umo, job_id=job_id, now=100)
        )
        with self.storage._connection:
            self.storage._connection.execute(
                "UPDATE maintenance_jobs SET status='COMPLETED' WHERE id=?",
                (job_id,),
            )
            self.storage._connection.execute(
                "DELETE FROM schema_meta WHERE key='maintenance_terminal_v15'"
            )
        self.storage.close()
        self.storage = MemoryStorage(self.database_path)
        migrated = self.storage._connection.execute(
            "SELECT status FROM maintenance_jobs WHERE id=?",
            (job_id,),
        ).fetchone()
        self.assertEqual(migrated["status"], "DONE")

        self.assertEqual(
            self.storage.enqueue_maintenance_job(
                umo=umo,
                job_type="feedback",
                dedupe_key="feedback:batch",
                available_at=101,
            ),
            job_id,
        )
        self.assertIsNotNone(
            self.storage.claim_maintenance_job(umo=umo, job_id=job_id, now=101)
        )
        self.storage.finish_maintenance_job(
            umo=umo,
            job_id=job_id,
            status="completed",
        )
        finished = self.storage._connection.execute(
            "SELECT status FROM maintenance_jobs WHERE id=?",
            (job_id,),
        ).fetchone()
        self.assertEqual(finished["status"], "DONE")

    def test_dashboard_does_not_report_cancelled_housekeeping_as_error(self) -> None:
        umo = "shadow:GroupMessage:group-a"
        self.storage.bind_scope(
            umo=umo,
            platform_id="shadow",
            group_id="group-a",
        )
        cancelled_id = self.storage.enqueue_maintenance_job(
            umo=umo,
            job_type="feedback",
            dedupe_key="feedback:old",
            available_at=100,
        )
        failed_id = self.storage.enqueue_maintenance_job(
            umo=umo,
            job_type="feedback",
            dedupe_key="feedback:failed",
            available_at=100,
        )
        with self.storage._connection:
            self.storage._connection.execute(
                """
                UPDATE maintenance_jobs
                SET status='CANCELLED', last_error='superseded'
                WHERE id=?
                """,
                (cancelled_id,),
            )
            self.storage._connection.execute(
                """
                UPDATE maintenance_jobs
                SET status='FAILED', last_error='ValueError'
                WHERE id=?
                """,
                (failed_id,),
            )

        errors = self.storage.dashboard_summary(umo=umo)[
            "recent_maintenance_errors"
        ]
        self.assertEqual([item["last_error"] for item in errors], ["ValueError"])

    def test_runtime_health_uses_terminal_status_and_wall_clock_latency(self) -> None:
        umo = "shadow:GroupMessage:group-a"
        for run_id in ("complete", "timeout", "running"):
            self.storage.start_experiment(
                run_id=run_id,
                umo=umo,
                experiment_type="runtime_reconstruction",
            )
        self.storage.record_llm_usage(
            run_id="complete",
            phase="reconstruction",
            input_other=100,
            output=20,
            elapsed_ms=750,
        )
        self.storage.finish_experiment(
            run_id="complete",
            status="completed",
            result={"no_relevant_memory": False, "path": "fast"},
        )
        self.storage.finish_experiment(
            run_id="timeout",
            status="failed",
            result={"error_type": "TimeoutError", "path": "fast"},
        )
        with self.storage._connection:
            self.storage._connection.execute(
                """
                UPDATE experiment_runs
                SET started_at='2026-01-01 00:00:00',
                    finished_at='2026-01-01 00:00:01'
                WHERE run_id='complete'
                """
            )
            self.storage._connection.execute(
                """
                UPDATE experiment_runs
                SET started_at='2026-01-01 00:01:00',
                    finished_at='2026-01-01 00:02:30'
                WHERE run_id='timeout'
                """
            )
            self.storage._connection.execute(
                """
                UPDATE experiment_runs
                SET started_at='2026-01-01 00:03:00'
                WHERE run_id='running'
                """
            )

        health = self.storage.runtime_health_summary(umo=umo, since=0)
        reconstruction = health["reconstruction"]
        self.assertEqual(reconstruction["calls"], 3)
        self.assertEqual(reconstruction["completed"], 1)
        self.assertEqual(reconstruction["running"], 1)
        self.assertEqual(reconstruction["failed"], 1)
        self.assertEqual(reconstruction["timeouts"], 1)
        self.assertAlmostEqual(reconstruction["p50_elapsed_ms"], 45500, delta=1)
        recent = {item["run_id"]: item for item in health["recent"]}
        self.assertAlmostEqual(recent["timeout"]["elapsed_ms"], 90000, delta=1)
        self.assertEqual(recent["timeout"]["outcome"], "failed")
        self.assertEqual(recent["running"]["outcome"], "running")

    def test_repeated_images_use_bounded_hash_metadata_not_media_bytes(self) -> None:
        umo = "shadow:GroupMessage:group-a"
        fingerprint = "a" * 64
        messages = (
            self.message("before", "这个表情又来了", sent_at=90),
            NormalizedMessage(
                platform="aiocqhttp",
                platform_id="shadow",
                umo=umo,
                group_id="group-a",
                message_id="image-1",
                sender_id="user-a",
                sender_name="甲",
                sent_at=100,
                plain_text="[图片]",
                content=[
                    {
                        "type": "image",
                        "name": "meme.jpg",
                        "reference_sha256": fingerprint,
                        "url": "https://secret.invalid/signed?token=do-not-store",
                        "bytes": "do-not-store",
                    }
                ],
            ),
            NormalizedMessage(
                platform="aiocqhttp",
                platform_id="shadow",
                umo=umo,
                group_id="group-a",
                message_id="image-2",
                sender_id="user-b",
                sender_name="乙",
                sent_at=110,
                plain_text="你又发这个",
                content=[{"type": "image", "reference_sha256": fingerprint}],
            ),
        )
        for message in messages:
            self.storage.upsert_message(message)

        patterns = self.storage.query_media_patterns(
            umo=umo,
            fingerprints=[fingerprint],
        )
        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0]["observation_count"], 2)
        self.assertEqual(patterns[0]["unique_sender_count"], 2)
        encoded = str(patterns[0])
        self.assertNotIn("secret.invalid", encoded)
        self.assertNotIn("do-not-store", encoded)
        self.assertIn("这个表情又来了", encoded)

        self.storage.mark_message_deleted(
            umo=umo,
            platform_id="shadow",
            platform_message_id="image-2",
            deleted_at=120,
        )
        self.assertEqual(
            self.storage.query_media_patterns(
                umo=umo,
                fingerprints=[fingerprint],
            ),
            [],
        )

    def test_historical_cutoff_is_strict(self) -> None:
        past = self.message("past", "太空采矿玩吐了", sent_at=100)
        future = self.message("future", "后来又买了采矿游戏首发", sent_at=200)
        self.storage.upsert_message(past)
        self.storage.upsert_message(future)
        past_event = self.storage.store_episode(
            umo=past.umo,
            started_at=100,
            ended_at=100,
            title="过去观点",
            summary="群友表示太空采矿玩吐了。",
            source_keys=[past.resolved_source_key()],
            keywords=[("太空采矿", "游戏偏好")],
        )
        future_event = self.storage.store_episode(
            umo=future.umo,
            started_at=200,
            ended_at=200,
            title="未来购买",
            summary="群友后来购买了首发。",
            source_keys=[future.resolved_source_key()],
            keywords=[("太空采矿", "游戏偏好")],
        )

        self.assertEqual(
            [
                item.message_id
                for item in self.storage.search_messages(
                    umo=past.umo, before_sent_at=200
                )
            ],
            ["past"],
        )
        events = self.storage.query_tag_events(
            umo=past.umo,
            cue="太空采矿",
            tag="游戏偏好",
            before_sent_at=200,
        )
        self.assertEqual([item["id"] for item in events], [past_event])
        self.assertIsNone(
            self.storage.query_conversation_time(
                umo=past.umo,
                event_id=future_event,
                before_sent_at=200,
            )
        )

    def test_experiment_ledger_sums_usage_and_hashes_results(self) -> None:
        run_id = "masked-test"
        umo = "shadow:GroupMessage:group-a"
        self.storage.start_experiment(
            run_id=run_id,
            umo=umo,
            experiment_type="masked_ab",
            cutoff_at=200,
            query_sha256="abc",
            metadata={"provider_stat_id": 7},
        )
        self.storage.record_llm_usage(
            run_id=run_id,
            phase="construction",
            call_index=0,
            input_other=100,
            input_cached=20,
            output=30,
            elapsed_ms=15,
        )
        self.storage.record_reconstruction_step(
            run_id=run_id,
            step_index=0,
            tool_name="query_event_context",
            arguments={"event_id": 1},
            evidence_keys=["source-1"],
            result_text="evidence",
        )
        self.storage.finish_experiment(
            run_id=run_id,
            status="completed",
            result={"winner": "memory"},
        )
        report = self.storage.experiment_report(run_id=run_id)
        self.assertIsNotNone(report)
        assert report is not None
        self.assertEqual(report["usage"][0]["total"], 150)
        self.assertEqual(report["steps"][0]["evidence_keys"], ["source-1"])
        self.assertEqual(report["run"]["result"], {"winner": "memory"})
        self.storage.start_experiment(
            run_id="foreign-run",
            umo="shadow:GroupMessage:group-b",
            experiment_type="runtime_reconstruction",
        )
        self.storage.record_llm_usage(
            run_id="foreign-run",
            phase="reconstruction",
            input_other=999,
        )
        recent = self.storage.recent_experiments(umo=umo)
        self.assertEqual([item["run_id"] for item in recent], [run_id])
        self.assertEqual(recent[0]["total"], 150)

    def test_experiment_detail_restores_the_exact_memory_provenance_graph(self) -> None:
        umo = "shadow:GroupMessage:group-a"
        request = self.message("detail-request", "群内怎么称呼这个？", sent_at=100)
        evidence = self.message("detail-evidence", "大家是在用反话。", sent_at=90)
        self.storage.upsert_message(evidence)
        self.storage.upsert_message(request)
        trace_id = "trace-detail"
        run_id = "runtime-reconstruction-detail"
        self.storage.start_interaction_trace(
            trace_id=trace_id,
            umo=umo,
            sender_id=request.sender_id,
            request_source_key=request.resolved_source_key(),
            request_sent_at=request.sent_at,
            query=request.plain_text,
        )
        brief = {
            "claims": [
                {
                    "statement": "该称呼在这段对话里是反话。",
                    "source_keys": [evidence.resolved_source_key()],
                    "confidence": 0.72,
                }
            ],
            "conflicts": [],
            "unresolved": [],
        }
        self.storage.start_experiment(
            run_id=run_id,
            umo=umo,
            experiment_type="runtime_reconstruction",
            metadata={"trace_id": trace_id, "path": "materialized_local"},
        )
        self.storage.record_reconstruction_step(
            run_id=run_id,
            step_index=0,
            tool_name="materialized_working_memory",
            evidence_keys=[evidence.resolved_source_key()],
        )
        self.storage.record_memory_brief_trace(
            trace_id=trace_id,
            umo=umo,
            run_id=run_id,
            memory_brief=brief,
            source_keys=[evidence.resolved_source_key()],
            path="materialized_local",
        )
        self.storage.finish_experiment(
            run_id=run_id,
            status="completed",
            result={
                "path": "materialized_local",
                "memory_brief": brief,
                "visited_source_keys": [evidence.resolved_source_key()],
                "trace_id": trace_id,
                "no_relevant_memory": False,
            },
        )

        detail = self.storage.experiment_detail(run_id=run_id, umo=umo)
        self.assertIsNotNone(detail)
        assert detail is not None
        self.assertTrue(detail["graph"]["exact_memory_brief"])
        self.assertEqual(detail["graph"]["warnings"], [])
        nodes = detail["graph"]["nodes"]
        node_types = {item["type"] for item in nodes}
        self.assertIn("memory_claim", node_types)
        self.assertIn("memory_brief", node_types)
        self.assertIn("evidence", node_types)
        relations = {item["relation"] for item in detail["graph"]["edges"]}
        self.assertIn("SUPPORTS", relations)
        self.assertIn("SUPPORTS_RECALL", relations)
        with self.assertRaisesRegex(ValueError, "group boundary"):
            self.storage.experiment_detail(
                run_id=run_id,
                umo="shadow:GroupMessage:group-b",
            )

    def test_experiment_detail_projects_only_verified_memory_activations(self) -> None:
        umo = "shadow:GroupMessage:group-a"
        request = self.message("memory-request", "这次应该怎么回答？", sent_at=100)
        feedback = self.message(
            "memory-feedback",
            "下次直接给结论，不要反问。",
            sent_at=110,
        )
        graph_evidence = self.message(
            "memory-edge-evidence",
            "群内把这个称呼当作反话。",
            sent_at=90,
        )
        for message in (request, feedback, graph_evidence):
            self.storage.upsert_message(message)

        source_trace_id = "memory-source-trace"
        self.storage.start_interaction_trace(
            trace_id=source_trace_id,
            umo=umo,
            sender_id=request.sender_id,
            request_source_key=request.resolved_source_key(),
            request_sent_at=request.sent_at,
            query=request.plain_text,
        )
        self.storage.finish_interaction_trace(
            trace_id=source_trace_id,
            umo=umo,
            response_text="你可以直接给结论。",
            response_at=101,
        )
        proposal_id = self.storage.enqueue_feedback_candidate(
            umo=umo,
            feedback_source_key=feedback.resolved_source_key(),
        )
        learned = self.storage.apply_feedback_decision(
            umo=umo,
            proposal_id=int(proposal_id or 0),
            decision=FeedbackDecision(
                target_trace_id=source_trace_id,
                mutation="upsert",
                feedback_valence=1.0,
                confidence=0.92,
                scope_type="sender",
                scope_key=request.sender_id,
                aspect="answer_style",
                statement="用户希望回答直接给结论。",
                prospective_cue="直接给出结论，不要用反问拖延。",
                trigger_cues=(),
                activation_mode="always",
            ),
        )
        hypothesis_id = int(learned["hypothesis_id"])

        edge_result = self.storage.apply_graph_mutation(
            umo=umo,
            mutation=parse_graph_mutation(
                {
                    "operation": "upsert_edge",
                    "evidence_source_keys": [
                        graph_evidence.resolved_source_key()
                    ],
                    "confidence": 0.87,
                    "utility_delta": 0.6,
                    "statement": "该称呼在这段群聊里是反话。",
                    "source": {
                        "kind": "symbol",
                        "label": "该称呼",
                        "description": "群内表达",
                    },
                    "relation": {
                        "key": "local_pragmatic_sense",
                        "name": "群内语用",
                        "description": "表达在本群上下文中的非字面含义",
                        "source_kinds": ["symbol"],
                        "target_kinds": ["concept"],
                    },
                    "target": {
                        "kind": "concept",
                        "label": "反话",
                        "description": "非字面解释",
                    },
                }
            ),
        )
        edge_id = int(edge_result["target_id"])

        target_trace_id = "memory-target-trace"
        self.storage.start_interaction_trace(
            trace_id=target_trace_id,
            umo=umo,
            sender_id=request.sender_id,
            request_source_key=request.resolved_source_key(),
            request_sent_at=120,
            query="现在怎么回答？",
        )
        self.storage.activate_feedback_hypotheses(
            umo=umo,
            sender_id=request.sender_id,
            query="现在怎么回答？",
            at=120,
            trace_id=target_trace_id,
            selected=[{"id": hypothesis_id, "activation_score": 0.88}],
            activation_method="layered_certificate_surface",
        )
        self.storage.activate_plastic_edges(
            umo=umo,
            edge_ids=[edge_id],
            at=120,
            trace_id=target_trace_id,
            relevance=0.77,
        )
        self.storage.finish_interaction_trace(
            trace_id=target_trace_id,
            umo=umo,
            response_text="直接回答。",
            response_at=121,
        )
        run_id = "layered-memory-effects"
        self.storage.start_experiment(
            run_id=run_id,
            umo=umo,
            experiment_type="runtime_layered_reconstruction",
            metadata={"trace_id": target_trace_id},
        )
        self.storage.finish_experiment(
            run_id=run_id,
            status="completed",
            result={
                "trace_id": target_trace_id,
                "selected_edge_ids": [edge_id],
                "selected_hypothesis_ids": [hypothesis_id],
            },
        )

        detail = self.storage.experiment_detail(run_id=run_id, umo=umo)
        assert detail is not None
        effects = detail["memory_effects"]
        self.assertEqual(effects["state"], "RECORDED")
        self.assertFalse(effects["exact"])
        self.assertTrue(effects["identity_exact"])
        self.assertEqual(effects["payload_as_of"], "current_resolution")
        self.assertEqual(effects["counts"]["nodes"], 3)
        self.assertEqual(effects["counts"]["edges"], 1)
        self.assertEqual(
            {node["type"] for node in effects["nodes"]},
            {"hypothesis", "plastic"},
        )
        self.assertFalse(
            {"run", "request", "response", "evidence", "feedback_proposal"}
            & {node["type"] for node in effects["nodes"]}
        )
        hypothesis = next(
            node for node in effects["nodes"] if node["type"] == "hypothesis"
        )
        self.assertEqual(hypothesis["access"], ["read"])
        self.assertAlmostEqual(hypothesis["activation"]["score"], 0.88)
        self.assertEqual(effects["edges"][0]["access"], ["read"])
        self.assertAlmostEqual(
            effects["edges"][0]["activation"]["score"],
            0.77,
        )
        self.assertTrue(
            all(
                node["access"] == ["context"]
                for node in effects["nodes"]
                if node["type"] == "plastic"
            )
        )
        self.assertEqual(
            len({node["id"] for node in effects["nodes"]}),
            len(effects["nodes"]),
        )
        self.assertEqual(
            len({edge["id"] for edge in effects["edges"]}),
            len(effects["edges"]),
        )

        # Corrupt current display metadata must not turn a historical-detail
        # read into a 500; identity still comes from the activation ledger.
        with self.storage._lock, self.storage._connection:
            self.storage._connection.execute(
                "UPDATE feedback_hypotheses SET trigger_cues_json='{' WHERE id=?",
                (hypothesis_id,),
            )
        corrupt_detail = self.storage.experiment_detail(run_id=run_id, umo=umo)
        assert corrupt_detail is not None
        corrupt_hypothesis = next(
            node
            for node in corrupt_detail["memory_effects"]["nodes"]
            if node["type"] == "hypothesis"
        )
        self.assertEqual(corrupt_hypothesis["trigger_cues"], [])

    def test_experiment_detail_does_not_promote_selected_memory_to_activation(self) -> None:
        umo = "shadow:GroupMessage:group-a"
        request = self.message("selected-request", "给个短答案", sent_at=100)
        feedback = self.message(
            "selected-feedback", "下次直接给结论，不要反问。", sent_at=110
        )
        graph_evidence = self.message(
            "selected-edge-evidence", "这个代号表示测试环境", sent_at=90
        )
        for message in (request, feedback, graph_evidence):
            self.storage.upsert_message(message)
        source_trace_id = "selected-source-trace"
        self.storage.start_interaction_trace(
            trace_id=source_trace_id,
            umo=umo,
            sender_id=request.sender_id,
            request_source_key=request.resolved_source_key(),
            request_sent_at=request.sent_at,
            query=request.plain_text,
        )
        self.storage.finish_interaction_trace(
            trace_id=source_trace_id,
            umo=umo,
            response_text="短答案。",
            response_at=101,
        )
        proposal_id = self.storage.enqueue_feedback_candidate(
            umo=umo,
            feedback_source_key=feedback.resolved_source_key(),
        )
        self.assertIsNotNone(proposal_id)
        learned = self.storage.apply_feedback_decision(
            umo=umo,
            proposal_id=int(proposal_id or 0),
            decision=FeedbackDecision(
                target_trace_id=source_trace_id,
                mutation="upsert",
                feedback_valence=1.0,
                confidence=0.9,
                scope_type="sender",
                scope_key=request.sender_id,
                aspect="brevity",
                statement="用户希望答案简短。",
                prospective_cue="回答时保持简短。",
                trigger_cues=(),
                activation_mode="always",
            ),
        )
        hypothesis_id = int(learned["hypothesis_id"])
        edge_result = self.storage.apply_graph_mutation(
            umo=umo,
            mutation=parse_graph_mutation(
                {
                    "operation": "upsert_edge",
                    "evidence_source_keys": [
                        graph_evidence.resolved_source_key()
                    ],
                    "confidence": 0.8,
                    "utility_delta": 0.4,
                    "statement": "该代号表示测试环境。",
                    "source": {"kind": "symbol", "label": "该代号"},
                    "relation": {
                        "key": "denotes_environment",
                        "name": "表示环境",
                        "description": "符号表示的运行环境",
                        "source_kinds": ["symbol"],
                        "target_kinds": ["concept"],
                    },
                    "target": {"kind": "concept", "label": "测试环境"},
                }
            ),
        )
        edge_id = int(edge_result["target_id"])
        run_id = "selected-without-activation"
        self.storage.start_experiment(
            run_id=run_id,
            umo=umo,
            experiment_type="runtime_layered_reconstruction",
        )
        self.storage.finish_experiment(
            run_id=run_id,
            status="completed",
            result={
                "selected_edge_ids": [edge_id],
                "presented_edge_ids": [edge_id],
                "activated_edge_ids": [edge_id],
                "selected_hypothesis_ids": [hypothesis_id],
                "presented_hypothesis_ids": [hypothesis_id],
            },
        )
        detail = self.storage.experiment_detail(run_id=run_id, umo=umo)
        assert detail is not None
        effects = detail["memory_effects"]
        self.assertEqual(effects["nodes"], [])
        self.assertEqual(effects["edges"], [])
        self.assertEqual(effects["state"], "UNAVAILABLE_LEGACY")
        self.assertEqual(
            effects["selected_not_activated"],
            {"edge_ids": [edge_id], "hypothesis_ids": [hypothesis_id]},
        )

    def test_ignored_feedback_has_no_memory_graph(self) -> None:
        umo = "shadow:GroupMessage:group-a"
        request = self.message("ignored-request", "先给一个方案", sent_at=100)
        feedback = self.message(
            "ignored-feedback", "不是这样，下次不要反问，直接给方案。", sent_at=110
        )
        self.storage.upsert_message(request)
        self.storage.upsert_message(feedback)
        trace_id = "ignored-feedback-trace"
        self.storage.start_interaction_trace(
            trace_id=trace_id,
            umo=umo,
            sender_id=request.sender_id,
            request_source_key=request.resolved_source_key(),
            request_sent_at=request.sent_at,
            query=request.plain_text,
        )
        self.storage.finish_interaction_trace(
            trace_id=trace_id,
            umo=umo,
            response_text="这是方案。",
            response_at=101,
        )
        proposal_id = self.storage.enqueue_feedback_candidate(
            umo=umo,
            feedback_source_key=feedback.resolved_source_key(),
        )
        self.assertIsNotNone(proposal_id)
        ignored = self.storage.apply_feedback_decision(
            umo=umo,
            proposal_id=int(proposal_id or 0),
            decision=FeedbackDecision(
                target_trace_id="",
                mutation="ignore",
                feedback_valence=0.0,
                confidence=0.0,
                scope_type="sender",
                scope_key="",
                aspect="",
                statement="",
                prospective_cue="",
                trigger_cues=(),
                activation_mode="always",
            ),
        )
        run_id = "ignored-feedback-memory-effects"
        self.storage.start_experiment(
            run_id=run_id,
            umo=umo,
            experiment_type="runtime_feedback_maintenance",
        )
        self.storage.finish_experiment(
            run_id=run_id,
            status="completed",
            result={
                "outcomes": [
                    {
                        "proposal_id": int(proposal_id or 0),
                        "proposal_status": ignored["status"],
                        "commit_score": 0.0,
                        "trace_id": "",
                        "hypothesis_id": 0,
                        "graph_mutation_results": [],
                    }
                ]
            },
        )
        detail = self.storage.experiment_detail(run_id=run_id, umo=umo)
        assert detail is not None
        effects = detail["memory_effects"]
        self.assertEqual(effects["state"], "NOT_APPLICABLE")
        self.assertTrue(effects["exact"])
        self.assertEqual(effects["nodes"], [])
        self.assertEqual(effects["edges"], [])
        self.assertEqual(effects["counts"]["nodes"], 0)
        self.assertIn("忽略", effects["empty_reason"])

    def test_unverified_empty_results_are_not_exact_zero(self) -> None:
        umo = "shadow:GroupMessage:group-a"
        self.storage.start_experiment(
            run_id="unverified-ignore",
            umo=umo,
            experiment_type="runtime_feedback_maintenance",
        )
        self.storage.finish_experiment(
            run_id="unverified-ignore",
            status="completed",
            result={
                "outcomes": [
                    {"proposal_id": 94, "proposal_status": "IGNORED"}
                ]
            },
        )
        ignored_detail = self.storage.experiment_detail(
            run_id="unverified-ignore", umo=umo
        )
        assert ignored_detail is not None
        ignored_effects = ignored_detail["memory_effects"]
        self.assertEqual(ignored_effects["state"], "INCOMPLETE_CAPTURE")
        self.assertFalse(ignored_effects["exact"])
        self.assertTrue(
            all(value is None for value in ignored_effects["counts"].values())
        )

        self.storage.start_experiment(
            run_id="unverified-no-relevant",
            umo=umo,
            experiment_type="runtime_reconstruction",
        )
        self.storage.finish_experiment(
            run_id="unverified-no-relevant",
            status="completed",
            result={"no_relevant_memory": True},
        )
        no_relevant_detail = self.storage.experiment_detail(
            run_id="unverified-no-relevant", umo=umo
        )
        assert no_relevant_detail is not None
        no_relevant_effects = no_relevant_detail["memory_effects"]
        self.assertEqual(no_relevant_effects["state"], "INCOMPLETE_CAPTURE")
        self.assertFalse(no_relevant_effects["identity_exact"])
        self.assertTrue(
            all(value is None for value in no_relevant_effects["counts"].values())
        )

    def test_modern_capture_can_verify_no_activation(self) -> None:
        umo = "shadow:GroupMessage:group-a"
        request = self.message("zero-request", "没有相关记忆吗", sent_at=100)
        self.storage.upsert_message(request)
        trace_id = "zero-capture-trace"
        run_id = "zero-capture-run"
        self.storage.start_interaction_trace(
            trace_id=trace_id,
            umo=umo,
            sender_id=request.sender_id,
            request_source_key=request.resolved_source_key(),
            request_sent_at=request.sent_at,
            query=request.plain_text,
        )
        self.storage.finish_interaction_trace(
            trace_id=trace_id,
            umo=umo,
            response_text="没有。",
            response_at=101,
        )
        self.storage.start_experiment(
            run_id=run_id,
            umo=umo,
            experiment_type="runtime_reconstruction",
            metadata={"trace_id": trace_id},
        )
        self.storage.record_memory_brief_trace(
            trace_id=trace_id,
            umo=umo,
            run_id=run_id,
            memory_brief=None,
            presented_edge_ids=(),
            presented_hypothesis_ids=(),
        )
        self.storage.finish_experiment(
            run_id=run_id,
            status="completed",
            result={
                "trace_id": trace_id,
                "no_relevant_memory": True,
                "presented_edge_ids": [],
                "presented_hypothesis_ids": [],
            },
        )
        detail = self.storage.experiment_detail(run_id=run_id, umo=umo)
        assert detail is not None
        effects = detail["memory_effects"]
        self.assertEqual(effects["state"], "NO_ACTIVATION")
        self.assertTrue(effects["exact"])
        self.assertTrue(effects["identity_exact"])
        self.assertTrue(all(value == 0 for value in effects["counts"].values()))

    def test_experiment_trace_association_is_scope_checked_and_not_union_merged(
        self,
    ) -> None:
        umo = "shadow:GroupMessage:group-a"
        for trace_id in ("association-a", "association-b"):
            self.storage.start_interaction_trace(
                trace_id=trace_id,
                umo=umo,
                sender_id="user-a",
                request_source_key="",
                request_sent_at=100,
                query=trace_id,
            )
            self.storage.finish_interaction_trace(
                trace_id=trace_id,
                umo=umo,
                response_text="ok",
                response_at=101,
            )
        self.storage.start_experiment(
            run_id="trace-mismatch-run",
            umo=umo,
            experiment_type="runtime_layered_reconstruction",
            metadata={"trace_id": "association-a"},
        )
        self.storage.finish_experiment(
            run_id="trace-mismatch-run",
            status="completed",
            result={"trace_id": "association-b"},
        )
        mismatch = self.storage.experiment_detail(
            run_id="trace-mismatch-run", umo=umo
        )
        assert mismatch is not None
        mismatch_effects = mismatch["memory_effects"]
        self.assertEqual(mismatch_effects["state"], "INCOMPLETE_CAPTURE")
        self.assertIn("run_trace_id:mismatch", mismatch_effects["integrity_errors"])
        self.assertEqual(mismatch_effects["nodes"], [])

        foreign_umo = "shadow:GroupMessage:group-b"
        self.storage.start_interaction_trace(
            trace_id="foreign-trace",
            umo=foreign_umo,
            sender_id="user-b",
            request_source_key="",
            request_sent_at=100,
            query="foreign",
        )
        self.storage.start_experiment(
            run_id="foreign-trace-run",
            umo=umo,
            experiment_type="runtime_layered_reconstruction",
            metadata={"trace_id": "foreign-trace"},
        )
        self.storage.finish_experiment(
            run_id="foreign-trace-run",
            status="completed",
            result={},
        )
        foreign = self.storage.experiment_detail(
            run_id="foreign-trace-run", umo=umo
        )
        assert foreign is not None
        foreign_effects = foreign["memory_effects"]
        self.assertEqual(foreign_effects["state"], "INCOMPLETE_CAPTURE")
        self.assertTrue(
            any(
                "outside_scope" in item
                for item in foreign_effects["integrity_errors"]
            )
        )

    def test_memory_effect_id_caps_and_types_fail_closed(self) -> None:
        umo = "shadow:GroupMessage:group-a"
        run_id = "bounded-memory-effect-ids"
        self.storage.start_experiment(
            run_id=run_id,
            umo=umo,
            experiment_type="runtime_layered_reconstruction",
        )
        self.storage.finish_experiment(
            run_id=run_id,
            status="completed",
            result={
                "selected_edge_ids": [*range(1, 100), True, 1.5, 1 << 80],
            },
        )
        detail = self.storage.experiment_detail(run_id=run_id, umo=umo)
        assert detail is not None
        effects = detail["memory_effects"]
        self.assertTrue(effects["truncated"])
        self.assertEqual(effects["state"], "INCOMPLETE_CAPTURE")
        self.assertLessEqual(len(effects["selected_not_activated"]["edge_ids"]), 80)
        self.assertTrue(all(value is None for value in effects["counts"].values()))

    def test_experiment_detail_recovers_legacy_feedback_provenance(self) -> None:
        umo = "shadow:GroupMessage:group-a"
        request = self.message("legacy-request", "帮我画一张海报", sent_at=100)
        feedback = self.message(
            "legacy-feedback",
            "元素太密了，下次少放一点。",
            sent_at=110,
        )
        self.storage.upsert_message(request)
        trace_id = "legacy-feedback-trace"
        self.storage.start_interaction_trace(
            trace_id=trace_id,
            umo=umo,
            sender_id=request.sender_id,
            request_source_key=request.resolved_source_key(),
            request_sent_at=request.sent_at,
            query=request.plain_text,
        )
        self.storage.finish_interaction_trace(
            trace_id=trace_id,
            umo=umo,
            response_text="已生成元素丰富的海报。",
            response_at=101,
        )
        self.storage.upsert_message(feedback)
        proposal_id = self.storage.enqueue_feedback_candidate(
            umo=umo,
            feedback_source_key=feedback.resolved_source_key(),
        )
        self.assertIsNotNone(proposal_id)
        learned = self.storage.apply_feedback_decision(
            umo=umo,
            proposal_id=int(proposal_id or 0),
            decision=FeedbackDecision(
                target_trace_id=trace_id,
                mutation="upsert",
                feedback_valence=-1.0,
                confidence=0.92,
                scope_type="sender",
                scope_key=request.sender_id,
                aspect="image_composition",
                statement="用户认为画面元素过于密集。",
                prospective_cue="再次生图时减少元素数量并留出空间。",
                trigger_cues=("生图", "海报"),
                activation_mode="semantic",
            ),
        )
        graph_mutation = parse_graph_mutation(
            {
                "operation": "upsert_edge",
                "evidence_source_keys": [feedback.resolved_source_key()],
                "confidence": 0.85,
                "utility_delta": 0.5,
                "statement": "生图回答应减少画面元素。",
                "source": {"kind": "behavior", "label": "生图回答"},
                "relation": {
                    "key": "prefers_composition",
                    "name": "偏好构图",
                    "description": "任务对应的构图偏好",
                    "source_kinds": ["behavior"],
                    "target_kinds": ["preference"],
                },
                "target": {"kind": "preference", "label": "减少元素"},
            }
        )
        graph_result = self.storage.apply_graph_mutation(
            umo=umo,
            mutation=graph_mutation,
            feedback_proposal_id=int(proposal_id or 0),
        )
        run_id = "legacy-feedback-run"
        self.storage.start_experiment(
            run_id=run_id,
            umo=umo,
            experiment_type="runtime_feedback_maintenance",
        )
        # The hypothesis identity is recovered from durable feedback evidence;
        # the graph effect is accepted only after its mutation receipt matches.
        self.storage.finish_experiment(
            run_id=run_id,
            status="completed",
            result={
                "outcomes": [
                    {
                        "proposal_id": int(proposal_id or 0),
                        "proposal_status": learned["status"],
                        "commit_score": learned["commit_score"],
                        "graph_mutation_results": [
                            {**graph_result, "proposal": graph_mutation.as_dict()}
                        ],
                    }
                ],
                "path": "one_pass_feedback_learning",
            },
        )

        detail = self.storage.experiment_detail(run_id=run_id, umo=umo)
        self.assertIsNotNone(detail)
        assert detail is not None
        nodes = detail["graph"]["nodes"]
        node_types = {item["type"] for item in nodes}
        self.assertTrue(
            {"request", "response", "evidence", "feedback_proposal", "hypothesis"}
            <= node_types
        )
        relations = {item["relation"] for item in detail["graph"]["edges"]}
        self.assertIn("RECEIVES_FEEDBACK", relations)
        self.assertIn("EVALUATED_AS", relations)
        self.assertIn("MATERIALIZED", relations)
        proposal_node = next(
            item for item in nodes if item["type"] == "feedback_proposal"
        )
        self.assertEqual(
            proposal_node["content"]["decision"]["target_trace_id"],
            trace_id,
        )
        effects = detail["memory_effects"]
        written = next(
            item for item in effects["nodes"] if item["type"] == "hypothesis"
        )
        self.assertEqual(written["access"], ["upsert"])
        upserted_edge = next(
            edge for edge in effects["edges"] if edge["type"] == "plastic_relation"
        )
        self.assertEqual(upserted_edge["access"], ["upsert"])
        self.assertTrue(effects["identity_exact"])
        self.assertFalse(effects["exact"])

        relation_revision = parse_graph_mutation(
            {
                "operation": "revise_relation",
                "evidence_source_keys": [feedback.resolved_source_key()],
                "confidence": 0.9,
                "utility_delta": 0.0,
                "relation": {
                    "key": "prefers_composition",
                    "name": "偏好留白构图",
                    "description": "任务对应的留白构图偏好",
                    "source_kinds": ["behavior"],
                    "target_kinds": ["preference"],
                },
            }
        )
        revision_result = self.storage.apply_graph_mutation(
            umo=umo,
            mutation=relation_revision,
            feedback_proposal_id=int(proposal_id or 0),
        )
        revision_run_id = "relation-revision-effects"
        self.storage.start_experiment(
            run_id=revision_run_id,
            umo=umo,
            experiment_type="runtime_feedback_maintenance",
        )
        self.storage.finish_experiment(
            run_id=revision_run_id,
            status="completed",
            result={
                "outcomes": [
                    {
                        "proposal_id": int(proposal_id or 0),
                        "proposal_status": learned["status"],
                        "trace_id": trace_id,
                        "hypothesis_id": learned["hypothesis_id"],
                        "graph_mutation_results": [
                            {
                                **revision_result,
                                "proposal": relation_revision.as_dict(),
                            }
                        ],
                    }
                ]
            },
        )
        revision_detail = self.storage.experiment_detail(
            run_id=revision_run_id, umo=umo
        )
        assert revision_detail is not None
        revision_effects = revision_detail["memory_effects"]
        self.assertEqual(revision_effects["state"], "PARTIAL")
        self.assertEqual(revision_effects["edges"], [])
        self.assertTrue(revision_effects["unsupported_refs"])

        other_trace_id = "mismatched-feedback-trace"
        self.storage.start_interaction_trace(
            trace_id=other_trace_id,
            umo=umo,
            sender_id=request.sender_id,
            request_source_key="",
            request_sent_at=120,
            query="other",
        )
        mismatch_run_id = "feedback-outcome-trace-mismatch"
        self.storage.start_experiment(
            run_id=mismatch_run_id,
            umo=umo,
            experiment_type="runtime_feedback_maintenance",
        )
        self.storage.finish_experiment(
            run_id=mismatch_run_id,
            status="completed",
            result={
                "outcomes": [
                    {
                        "proposal_id": int(proposal_id or 0),
                        "proposal_status": learned["status"],
                        "trace_id": other_trace_id,
                        "hypothesis_id": learned["hypothesis_id"],
                    }
                ]
            },
        )
        mismatch_detail = self.storage.experiment_detail(
            run_id=mismatch_run_id, umo=umo
        )
        assert mismatch_detail is not None
        mismatch_effects = mismatch_detail["memory_effects"]
        self.assertEqual(mismatch_effects["state"], "INCOMPLETE_CAPTURE")
        self.assertEqual(mismatch_effects["nodes"], [])
        self.assertTrue(
            any(
                item.endswith("trace_id:mismatch")
                for item in mismatch_effects["integrity_errors"]
            )
        )

    def test_committed_memory_effect_receipts_reject_tampered_run_outcomes(
        self,
    ) -> None:
        umo = "shadow:GroupMessage:group-a"

        def committed_feedback(
            prefix: str, *, request_at: int, feedback_at: int
        ) -> tuple[str, int, dict[str, object], NormalizedMessage]:
            request = self.message(
                f"{prefix}-request", "请直接给方案", sent_at=request_at
            )
            feedback = self.message(
                f"{prefix}-feedback",
                "不是这样，下次不要反问，直接给结论。",
                sent_at=feedback_at,
            )
            self.storage.upsert_message(request)
            self.storage.upsert_message(feedback)
            trace_id = f"{prefix}-trace"
            self.storage.start_interaction_trace(
                trace_id=trace_id,
                umo=umo,
                sender_id=request.sender_id,
                request_source_key=request.resolved_source_key(),
                request_sent_at=request.sent_at,
                query=request.plain_text,
            )
            self.storage.finish_interaction_trace(
                trace_id=trace_id,
                umo=umo,
                response_text="这是方案。",
                response_at=request_at + 1,
            )
            proposal_id = self.storage.enqueue_feedback_candidate(
                umo=umo,
                feedback_source_key=feedback.resolved_source_key(),
            )
            self.assertIsNotNone(proposal_id)
            learned = self.storage.apply_feedback_decision(
                umo=umo,
                proposal_id=int(proposal_id or 0),
                decision=FeedbackDecision(
                    target_trace_id=trace_id,
                    mutation="upsert",
                    feedback_valence=-1.0,
                    confidence=0.92,
                    scope_type="sender",
                    scope_key=request.sender_id,
                    aspect=f"{prefix}_answer_style",
                    statement="用户要求直接给出结论。",
                    prospective_cue=f"{prefix} 场景直接给出结论。",
                    trigger_cues=(),
                    activation_mode="always",
                ),
            )
            return trace_id, int(proposal_id or 0), learned, feedback

        trace_id, proposal_id, learned, feedback = committed_feedback(
            "receipt-primary", request_at=100, feedback_at=110
        )
        decoy_trace_id, _, decoy_learned, decoy_feedback = committed_feedback(
            "receipt-decoy", request_at=200, feedback_at=210
        )

        mutation = parse_graph_mutation(
            {
                "operation": "upsert_edge",
                "evidence_source_keys": [feedback.resolved_source_key()],
                "confidence": 0.88,
                "utility_delta": 0.5,
                "statement": "回答应直接给出结论。",
                "source": {"kind": "behavior", "label": "回答方案"},
                "relation": {
                    "key": "prefers_directness",
                    "name": "偏好直接",
                    "description": "回答方式的直接程度偏好",
                    "source_kinds": ["behavior"],
                    "target_kinds": ["preference"],
                },
                "target": {"kind": "preference", "label": "直接结论"},
            }
        )
        mutation_result = self.storage.apply_graph_mutation(
            umo=umo,
            mutation=mutation,
            feedback_proposal_id=proposal_id,
        )
        decoy_mutation = parse_graph_mutation(
            {
                "operation": "upsert_edge",
                "evidence_source_keys": [decoy_feedback.resolved_source_key()],
                "confidence": 0.86,
                "utility_delta": 0.4,
                "statement": "回答应保留推导过程。",
                "source": {"kind": "behavior", "label": "解释方案"},
                "relation": {
                    "key": "prefers_explanation",
                    "name": "偏好解释",
                    "description": "回答方式的解释程度偏好",
                    "source_kinds": ["behavior"],
                    "target_kinds": ["preference"],
                },
                "target": {"kind": "preference", "label": "保留推导"},
            }
        )
        decoy_mutation_result = self.storage.apply_graph_mutation(
            umo=umo,
            mutation=decoy_mutation,
        )
        decoy_hypothesis_id = int(decoy_learned["hypothesis_id"])
        decoy_edge_id = int(decoy_mutation_result["target_id"])
        base_outcome = {
            "proposal_id": proposal_id,
            "proposal_status": learned["status"],
            "trace_id": trace_id,
            "hypothesis_id": learned["hypothesis_id"],
            "graph_mutation_results": [
                {**mutation_result, "proposal": mutation.as_dict()}
            ],
        }
        variants = (
            (
                "proposal-status",
                {**base_outcome, "proposal_status": "IGNORED"},
                "proposal_status:mismatch",
                "all",
            ),
            (
                "trace-id",
                {**base_outcome, "trace_id": decoy_trace_id},
                "trace_id:mismatch",
                "all",
            ),
            (
                "hypothesis-id",
                {**base_outcome, "hypothesis_id": decoy_hypothesis_id},
                "hypothesis_id:mismatch",
                "hypothesis",
            ),
            (
                "mutation-target",
                {
                    **base_outcome,
                    "graph_mutation_results": [
                        {
                            **mutation_result,
                            "target_id": decoy_edge_id,
                            "proposal": mutation.as_dict(),
                        }
                    ],
                },
                "receipt_mismatch",
                "edge",
            ),
        )
        for suffix, outcome, expected_error, fake_kind in variants:
            with self.subTest(suffix=suffix):
                run_id = f"tampered-committed-{suffix}"
                self.storage.start_experiment(
                    run_id=run_id,
                    umo=umo,
                    experiment_type="runtime_feedback_maintenance",
                )
                self.storage.finish_experiment(
                    run_id=run_id,
                    status="completed",
                    result={"outcomes": [outcome]},
                )
                detail = self.storage.experiment_detail(run_id=run_id, umo=umo)
                assert detail is not None
                effects = detail["memory_effects"]
                self.assertIn(effects["state"], {"PARTIAL", "INCOMPLETE_CAPTURE"})
                self.assertFalse(effects["exact"])
                self.assertTrue(
                    any(
                        expected_error in item
                        for item in effects["integrity_errors"]
                    )
                )
                if fake_kind == "all":
                    self.assertEqual(effects["nodes"], [])
                    self.assertEqual(effects["edges"], [])
                elif fake_kind == "hypothesis":
                    self.assertNotIn(
                        f"hypothesis:{decoy_hypothesis_id}",
                        {node["id"] for node in effects["nodes"]},
                    )
                else:
                    self.assertNotIn(
                        f"plastic_edge:{decoy_edge_id}",
                        {edge["id"] for edge in effects["edges"]},
                    )

    def test_pending_and_rejected_feedback_are_not_exact_zero(self) -> None:
        umo = "shadow:GroupMessage:group-a"
        request = self.message("nonterminal-request", "给一个方案", sent_at=100)
        self.storage.upsert_message(request)
        trace_id = "nonterminal-trace"
        self.storage.start_interaction_trace(
            trace_id=trace_id,
            umo=umo,
            sender_id=request.sender_id,
            request_source_key=request.resolved_source_key(),
            request_sent_at=request.sent_at,
            query=request.plain_text,
        )
        self.storage.finish_interaction_trace(
            trace_id=trace_id,
            umo=umo,
            response_text="这是方案。",
            response_at=101,
        )
        proposals: list[tuple[int, str]] = []
        for suffix, sent_at in (("pending", 110), ("rejected", 120)):
            feedback = self.message(
                f"{suffix}-feedback",
                "不是这样，下次不要反问，直接给方案。",
                sent_at=sent_at,
            )
            self.storage.upsert_message(feedback)
            proposal_id = self.storage.enqueue_feedback_candidate(
                umo=umo,
                feedback_source_key=feedback.resolved_source_key(),
            )
            self.assertIsNotNone(proposal_id)
            proposals.append((int(proposal_id or 0), suffix.upper()))
        self.storage.reject_feedback_proposal(
            umo=umo,
            proposal_id=proposals[1][0],
            error="fixture rejection",
        )
        statuses = ((proposals[0][0], "PENDING"), (proposals[1][0], "REJECTED"))
        for proposal_id, status in statuses:
            with self.subTest(status=status):
                run_id = f"completed-{status.casefold()}-feedback"
                self.storage.start_experiment(
                    run_id=run_id,
                    umo=umo,
                    experiment_type="runtime_feedback_maintenance",
                )
                self.storage.finish_experiment(
                    run_id=run_id,
                    status="completed",
                    result={
                        "outcomes": [
                            {
                                "proposal_id": proposal_id,
                                "proposal_status": status,
                            }
                        ]
                    },
                )
                detail = self.storage.experiment_detail(run_id=run_id, umo=umo)
                assert detail is not None
                effects = detail["memory_effects"]
                self.assertNotIn(effects["state"], {"NO_ACTIVATION", "NOT_APPLICABLE"})
                self.assertFalse(effects["exact"])
                self.assertFalse(effects["identity_exact"])
                self.assertTrue(
                    all(value is None for value in effects["counts"].values())
                )

    def test_interrupted_experiment_is_closed_on_reopen(self) -> None:
        umo = "shadow:GroupMessage:group-a"
        self.storage.start_experiment(
            run_id="interrupted-run",
            umo=umo,
            experiment_type="runtime_reconstruction",
        )
        self.storage.close()
        self.storage = MemoryStorage(self.database_path)
        report = self.storage.experiment_report(run_id="interrupted-run")
        assert report is not None
        self.assertEqual(report["run"]["status"], "FAILED")
        self.assertEqual(
            report["run"]["result"]["error_type"],
            "InterruptedError",
        )

    def test_layered_reconstruction_is_visible_in_runtime_health(self) -> None:
        """The dashboard must count the reconstruction path used in production."""

        umo = "shadow:GroupMessage:group-a"
        for index in range(12):
            self.storage.upsert_message(
                self.message(
                    f"d-teacher-{index}",
                    f"D老师参与过的历史消息 {index}",
                    umo=umo,
                    sender_id="account-d",
                    sent_at=100 + index,
                )
            )
        self.storage.bind_participant_alias(
            umo=umo,
            platform_id="shadow",
            account_id="account-d",
            alias="d老师",
            at=120,
        )
        resolved = self.storage.resolve_participants(
            umo=umo,
            reference="d老师",
        )
        self.assertEqual(len(resolved["participants"]), 1)

        run_id = "runtime-layered-person-lookup"
        self.storage.start_experiment(
            run_id=run_id,
            umo=umo,
            experiment_type="runtime_layered_reconstruction",
            metadata={"path": "layered", "route": "L2"},
        )
        self.storage.record_llm_usage(
            run_id=run_id,
            phase="reconstruction",
            input_other=120,
            output=30,
            elapsed_ms=250,
        )
        self.storage.finish_experiment(
            run_id=run_id,
            status="completed",
            result={
                "operational_status": "COMPLETED",
                "semantic_status": "CERTIFIED",
                "path": "layered",
                "no_relevant_memory": False,
            },
        )
        failed_run_id = "runtime-layered-person-lookup-failed"
        self.storage.start_experiment(
            run_id=failed_run_id,
            umo=umo,
            experiment_type="runtime_layered_reconstruction",
            metadata={"path": "layered", "route": "L2"},
        )
        self.storage.finish_experiment(
            run_id=failed_run_id,
            status="failed",
            result={
                "operational_status": "FAILED",
                "semantic_status": "UNKNOWN",
                "path": "layered",
                "error_type": "TimeoutError",
                "error_detail": "reader deadline exceeded",
            },
        )

        health = self.storage.runtime_health_summary(umo=umo, since=0)

        self.assertEqual(health["reconstruction"]["calls"], 2)
        self.assertEqual(health["reconstruction"]["completed"], 1)
        self.assertEqual(health["reconstruction"]["failed"], 1)
        self.assertEqual(health["reconstruction"]["timeouts"], 1)
        self.assertEqual(health["reconstruction"]["tokens"], 150)
        recent = {item["run_id"]: item for item in health["recent"]}
        self.assertEqual(recent[run_id]["phase"], "reconstruction")
        self.assertEqual(recent[failed_run_id]["phase"], "reconstruction")
        self.assertEqual(recent[failed_run_id]["status"], "FAILED")
        self.assertEqual(recent[failed_run_id]["outcome"], "failed")
        self.assertEqual(recent[failed_run_id]["error_type"], "TimeoutError")

    def test_online_and_history_budgets_are_physically_separate(self) -> None:
        umo = "shadow:GroupMessage:group-a"
        for run_id, phase, tokens in (
            ("online-run", "reconstruction", 90),
            ("history-run", "history_construction", 240),
        ):
            self.storage.start_experiment(
                run_id=run_id,
                umo=umo,
                experiment_type=run_id,
            )
            self.storage.record_llm_usage(
                run_id=run_id,
                phase=phase,
                input_other=tokens,
            )
        self.assertEqual(
            self.storage.private_token_usage_since(
                umo=umo, since=0, budget_class="online"
            ),
            90,
        )
        self.assertEqual(
            self.storage.private_token_usage_since(
                umo=umo, since=0, budget_class="backfill"
            ),
            240,
        )

    def test_online_budget_reset_keeps_raw_ledger_and_uses_event_watermark(
        self,
    ) -> None:
        umo = "shadow:GroupMessage:group-a"
        self.storage.start_experiment(
            run_id="before-reset",
            umo=umo,
            experiment_type="runtime_reconstruction",
        )
        self.storage.record_llm_usage(
            run_id="before-reset",
            phase="reconstruction",
            input_other=100,
        )
        reset = self.storage.reset_token_budget(
            umo=umo,
            budget_class="online",
            reason="test",
        )
        self.assertGreater(int(reset["usage_event_id"]), 0)
        self.assertEqual(
            self.storage.private_token_usage_since(
                umo=umo,
                since=0,
                budget_class="online",
            ),
            0,
        )
        self.assertEqual(
            self.storage.private_token_usage_since(
                umo=umo,
                since=0,
                budget_class="online",
                apply_resets=False,
            ),
            100,
        )

        self.storage.start_experiment(
            run_id="after-reset",
            umo=umo,
            experiment_type="runtime_reconstruction",
        )
        self.storage.record_llm_usage(
            run_id="after-reset",
            phase="reconstruction",
            input_other=40,
        )
        self.assertEqual(
            self.storage.private_token_usage_since(
                umo=umo,
                since=0,
                budget_class="online",
            ),
            40,
        )

    def test_budget_reset_resumes_only_matching_wait_jobs(self) -> None:
        umo = "shadow:GroupMessage:group-a"
        job_ids = {}
        for budget_class, job_type in (("online", "feedback"), ("backfill", "distill")):
            job_id = self.storage.enqueue_maintenance_job(
                umo=umo,
                job_type=job_type,
                dedupe_key=f"{job_type}:test",
                payload={},
            )
            claimed = self.storage.claim_maintenance_job(
                umo=umo,
                job_id=job_id,
            )
            self.assertIsNotNone(claimed)
            self.storage.defer_maintenance_job_for_budget(
                umo=umo,
                job_id=job_id,
                available_at=2**31,
                budget_class=budget_class,
            )
            job_ids[budget_class] = job_id

        self.storage.reset_token_budget(
            umo=umo,
            budget_class="online",
            reason="test",
        )
        rows = self.storage._connection.execute(
            "SELECT id, status FROM maintenance_jobs ORDER BY id"
        ).fetchall()
        statuses = {int(row["id"]): str(row["status"]) for row in rows}
        self.assertEqual(statuses[job_ids["online"]], "PENDING")
        self.assertEqual(statuses[job_ids["backfill"]], "BUDGET_WAIT")

    def test_live_distillation_preempts_backfill_without_mixing_batches(self) -> None:
        umo = "shadow:GroupMessage:group-a"
        self.storage.upsert_message(
            self.message("history", "旧消息", umo=umo, sent_at=100),
            processing_class="BACKFILL",
        )
        self.storage.upsert_message(
            self.message("live", "新消息", umo=umo, sent_at=200),
            processing_class="LIVE",
        )
        live = self.storage.next_distillation_batch(umo=umo, limit=10, overlap=0)
        assert live is not None
        self.assertEqual(live.processing_class, "LIVE")
        self.assertEqual(live.target_source_keys[0].rsplit("|", 1)[-1], "live")
        self.storage.finish_distillation_batch(work_item=live)
        history = self.storage.next_distillation_batch(umo=umo, limit=10, overlap=0)
        assert history is not None
        self.assertEqual(history.processing_class, "BACKFILL")
        self.assertEqual(history.target_source_keys[0].rsplit("|", 1)[-1], "history")

    def test_terminal_retry_can_be_limited_to_live_messages(self) -> None:
        umo = "shadow:GroupMessage:group-a"
        self.storage.upsert_message(
            self.message("history", "旧消息", umo=umo, sent_at=100),
            processing_class="BACKFILL",
        )
        self.storage.upsert_message(
            self.message("live", "新消息", umo=umo, sent_at=200),
            processing_class="LIVE",
        )
        for processing_class in ("LIVE", "BACKFILL"):
            for _ in range(3):
                work_item = self.storage.next_distillation_batch(
                    umo=umo,
                    limit=1,
                    overlap=0,
                    processing_class=processing_class,
                )
                assert work_item is not None
                self.storage.finish_distillation_batch(
                    work_item=work_item,
                    error="provider failure",
                )

        self.assertEqual(
            self.storage.retry_terminal_distillation_failures(
                umo=umo,
                processing_class="LIVE",
            ),
            1,
        )
        rows = self.storage._connection.execute("""
            SELECT processing_class, status
            FROM message_processing
            ORDER BY processing_class
            """).fetchall()
        self.assertEqual(
            {str(row["processing_class"]): str(row["status"]) for row in rows},
            {"BACKFILL": "FAILED", "LIVE": "PENDING"},
        )

    def test_live_provenance_wins_over_idempotent_history_sync(self) -> None:
        message = self.message("shared", "同一平台消息")
        self.storage.upsert_message(message, processing_class="LIVE")
        self.storage.upsert_message(message, processing_class="BACKFILL")
        row = self.storage._connection.execute(
            "SELECT processing_class, ingestion_source FROM message_processing"
        ).fetchone()
        self.assertEqual(row["processing_class"], "LIVE")
        self.assertEqual(row["ingestion_source"], "adapter_live")

        second = self.message("history-first", "稍后被实时观察")
        self.storage.upsert_message(second, processing_class="BACKFILL")
        self.storage.upsert_message(second, processing_class="LIVE")
        row = self.storage._connection.execute(
            """
            SELECT p.processing_class, p.ingestion_source
            FROM message_processing AS p
            JOIN messages AS m ON m.id=p.message_id
            WHERE m.message_id=?
            """,
            ("history-first",),
        ).fetchone()
        self.assertEqual(row["processing_class"], "LIVE")
        self.assertEqual(row["ingestion_source"], "adapter_live")

    def test_existing_data_is_not_reclassified_by_a_deployment_specific_migration(
        self,
    ) -> None:
        umo = "shadow:GroupMessage:group-a"
        self.storage.upsert_message(self.message("legacy", "旧记录", umo=umo))
        self.storage.start_experiment(
            run_id="legacy-construction",
            umo=umo,
            experiment_type="runtime_construction",
        )
        self.storage.record_llm_usage(
            run_id="legacy-construction",
            phase="construction",
            input_other=321,
        )
        self.storage.close()
        self.storage = MemoryStorage(self.database_path)
        processing = self.storage._connection.execute(
            "SELECT processing_class FROM message_processing"
        ).fetchone()
        usage = self.storage._connection.execute(
            "SELECT phase FROM llm_usage_events WHERE run_id=?",
            ("legacy-construction",),
        ).fetchone()
        self.assertEqual(processing["processing_class"], "LIVE")
        self.assertEqual(usage["phase"], "construction")

    def test_v13_cancels_legacy_online_budget_feedback_jobs(self) -> None:
        umo = "shadow:GroupMessage:group-a"
        self.storage.bind_scope(
            umo=umo,
            platform_id="shadow",
            group_id="group-a",
        )
        job_id = self.storage.enqueue_maintenance_job(
            umo=umo,
            job_type="feedback",
            dedupe_key="feedback:legacy-proposal",
            payload={"proposal_id": 1},
        )
        self.assertIsNotNone(self.storage.claim_maintenance_job(umo=umo, job_id=job_id))
        self.storage.defer_maintenance_job_for_budget(
            umo=umo,
            job_id=job_id,
            available_at=2**31,
            budget_class="online",
        )
        with self.storage._connection:
            self.storage._connection.execute(
                "DELETE FROM schema_meta WHERE key='feedback_budget_v13'"
            )
        self.storage.close()
        self.storage = MemoryStorage(self.database_path)
        row = self.storage._connection.execute(
            "SELECT status, last_error FROM maintenance_jobs WHERE id=?",
            (job_id,),
        ).fetchone()
        self.assertEqual(row["status"], "CANCELLED")
        self.assertEqual(row["last_error"], "superseded_by_feedback_batch_v13")

    def test_physical_database_can_only_bind_to_one_group_scope(self) -> None:
        self.storage.bind_scope(
            umo="shadow:GroupMessage:group-a",
            platform_id="shadow",
            group_id="group-a",
        )
        self.storage.bind_scope(
            umo="shadow:GroupMessage:group-a",
            platform_id="shadow",
            group_id="group-a",
        )
        self.assertEqual(
            self.storage.get_scope_identity(),
            {
                "umo": "shadow:GroupMessage:group-a",
                "platform_id": "shadow",
                "group_id": "group-a",
            },
        )
        with self.assertRaisesRegex(ValueError, "another group scope"):
            self.storage.bind_scope(
                umo="shadow:GroupMessage:group-b",
                platform_id="shadow",
                group_id="group-b",
            )

    def test_paper_traversal_mappings_are_scoped_and_composable(self) -> None:
        source = self.message("event-1", "小林决定采用方案 B", sent_at=200)
        self.storage.upsert_message(source)
        self.assertEqual(self.storage.count_graph_units(umo=source.umo), 0)
        event_id = self.storage.store_episode(
            umo=source.umo,
            started_at=190,
            ended_at=210,
            title="方案评审",
            summary="小林最终采用方案 B。",
            source_keys=[source.resolved_source_key()],
            keywords=[("小林", "方案选择"), ("方案 B", "方案选择")],
            extractor_version="test",
        )
        self.storage.store_semantic_memory(
            umo=source.umo,
            person="小林",
            aspect="工作偏好",
            content="倾向选择可回滚的方案。",
            source_key=source.resolved_source_key(),
            confidence=0.9,
            extractor_version="test",
        )
        self.storage.store_topic(
            umo=source.umo,
            name="项目决策",
            summary="项目中的关键选择。",
            event_ids=[event_id],
            extractor_version="test",
        )
        self.assertEqual(self.storage.count_graph_units(umo=source.umo), 3)

        other_umo = "shadow:GroupMessage:group-b"
        foreign_source = self.message(
            "foreign-1",
            "另一个群的私有内容",
            umo=other_umo,
            sender_id="user-b",
            sent_at=205,
        )
        self.storage.upsert_message(foreign_source)
        foreign_event_id = self.storage.store_episode(
            umo=other_umo,
            started_at=200,
            ended_at=210,
            title="外群事件",
            summary="不应跨群返回。",
            source_keys=[foreign_source.resolved_source_key()],
            keywords=[("小林", "方案选择")],
            extractor_version="test",
        )
        foreign_message_id = self.storage.search_messages(
            umo=other_umo, query="私有内容"
        )[0].id
        topic_id = self.storage._connection.execute(
            "SELECT id FROM topics WHERE umo = ? AND name = ?",
            (source.umo, "项目决策"),
        ).fetchone()[0]
        with self.storage._connection:
            # Simulate corrupt links that bypass the normal scoped write API.
            self.storage._connection.execute(
                """
                INSERT INTO episode_messages(episode_id, message_id, position)
                VALUES (?, ?, ?)
                """,
                (event_id, foreign_message_id, 99),
            )
            self.storage._connection.execute(
                """
                INSERT INTO topic_episodes(topic_id, episode_id)
                VALUES (?, ?)
                """,
                (topic_id, foreign_event_id),
            )

        self.assertEqual(
            [
                item["id"]
                for item in self.storage.query_tag_events(
                    umo=source.umo, cue="小林", tag="方案选择"
                )
            ],
            [event_id],
        )

        events = self.storage.query_tag_events(
            umo=source.umo, cue="小林", tag="方案选择"
        )
        self.assertEqual([item["id"] for item in events], [event_id])
        self.assertEqual(
            self.storage.query_conversation_time(umo=source.umo, event_id=event_id),
            {"id": event_id, "started_at": 190, "ended_at": 210},
        )
        self.assertEqual(
            len(self.storage.query_event_keywords(umo=source.umo, event_id=event_id)),
            2,
        )
        context = self.storage.query_event_context(umo=source.umo, event_id=event_id)
        self.assertEqual([item["plain_text"] for item in context], [source.plain_text])
        self.assertEqual(
            self.storage.query_personal_information(umo=source.umo, person="小林")[0][
                "aspect_tag"
            ],
            "工作偏好",
        )
        self.assertEqual(
            self.storage.query_personal_aspect(
                umo=source.umo, person="小林", aspect="工作偏好"
            )[0]["content"],
            "倾向选择可回滚的方案。",
        )
        self.assertEqual(
            self.storage.query_topic_events(umo=source.umo, topic="项目决策")[0]["id"],
            event_id,
        )

        self.assertEqual(
            [
                item["id"]
                for item in self.storage.query_tag_events(
                    umo=other_umo, cue="小林", tag="方案选择"
                )
            ],
            [foreign_event_id],
        )
        self.assertIsNone(
            self.storage.query_conversation_time(umo=other_umo, event_id=event_id)
        )

        dashboard = self.storage.dashboard_graph(umo=source.umo, limit=20)
        labels = {str(node["label"]) for node in dashboard["nodes"]}
        self.assertIn("方案评审", labels)
        self.assertIn("项目决策", labels)
        self.assertIn("工作偏好", labels)
        self.assertNotIn("外群事件", labels)
        self.assertNotIn("另一个群的私有内容", labels)
        self.assertGreaterEqual(int(dashboard["metrics"]["node_count"]), 3)
        self.assertGreaterEqual(int(dashboard["metrics"]["edge_count"]), 2)
        self.assertIn("average_clustering", dashboard["metrics"])
        self.assertIn("max_core", dashboard["metrics"])
        self.assertTrue(dashboard["degree_histogram"])
        self.assertTrue(dashboard["top_nodes"])

        searched_dashboard = self.storage.dashboard_graph(
            umo=source.umo,
            limit=20,
            query="方案评审",
            depth=2,
        )
        self.assertEqual(searched_dashboard["mode"], "neighborhood")
        self.assertEqual(searched_dashboard["focus_node_id"], f"episode:{event_id}")
        self.assertEqual(searched_dashboard["matches"][0]["label"], "方案评审")
        self.assertIn(
            f"episode:{event_id}",
            {str(node["id"]) for node in searched_dashboard["nodes"]},
        )
        self.assertTrue(
            all(
                int(node.get("distance", 0)) <= 2
                for node in searched_dashboard["nodes"]
            )
        )
        self.assertFalse(searched_dashboard["truncated"])

        connected_dashboard = self.storage.dashboard_graph(
            umo=source.umo,
            limit=20,
            min_degree=1,
            structure_scope="connected",
        )
        self.assertTrue(connected_dashboard["nodes"])
        self.assertTrue(
            all(int(node["degree"]) >= 1 for node in connected_dashboard["nodes"])
        )

        source_edge = dashboard["edges"][0]
        path_dashboard = self.storage.dashboard_graph(
            umo=source.umo,
            limit=20,
            path_source=str(source_edge["source"]),
            path_target=str(source_edge["target"]),
        )
        self.assertEqual(path_dashboard["mode"], "path")
        self.assertTrue(path_dashboard["path"]["found"])
        self.assertEqual(path_dashboard["path"]["length"], 1)
        limited_dashboard = self.storage.dashboard_graph(umo=source.umo, limit=3)
        limited_ids = {str(node["id"]) for node in limited_dashboard["nodes"]}
        self.assertLessEqual(len(limited_ids), 3)
        self.assertTrue(limited_dashboard["truncated"])
        self.assertTrue(
            all(
                str(edge["source"]) in limited_ids
                and str(edge["target"]) in limited_ids
                for edge in limited_dashboard["edges"]
            )
        )


if __name__ == "__main__":
    unittest.main()
