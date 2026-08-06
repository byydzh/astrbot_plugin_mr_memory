from __future__ import annotations

import unittest
import uuid
from pathlib import Path

from mr_memory.models import NormalizedMessage
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
        sent_at: int = 100,
    ) -> NormalizedMessage:
        return NormalizedMessage(
            platform="aiocqhttp",
            platform_id="shadow",
            umo=umo,
            group_id=umo.rsplit(":", 1)[-1],
            message_id=message_id,
            sender_id=sender_id,
            sender_name=sender_id,
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
