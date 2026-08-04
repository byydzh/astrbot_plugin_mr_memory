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
            [item.plain_text for item in self.storage.search_messages(
                umo="shadow:GroupMessage:group-a", query="方案 B"
            )],
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
            [item.message_id for item in self.storage.search_messages(
                umo=past.umo, before_sent_at=200
            )],
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
            self.storage.query_conversation_time(
                umo=source.umo, event_id=event_id
            ),
            {"id": event_id, "started_at": 190, "ended_at": 210},
        )
        self.assertEqual(
            len(self.storage.query_event_keywords(
                umo=source.umo, event_id=event_id
            )),
            2,
        )
        context = self.storage.query_event_context(
            umo=source.umo, event_id=event_id
        )
        self.assertEqual([item["plain_text"] for item in context], [source.plain_text])
        self.assertEqual(
            self.storage.query_personal_information(
                umo=source.umo, person="小林"
            )[0]["aspect_tag"],
            "工作偏好",
        )
        self.assertEqual(
            self.storage.query_personal_aspect(
                umo=source.umo, person="小林", aspect="工作偏好"
            )[0]["content"],
            "倾向选择可回滚的方案。",
        )
        self.assertEqual(
            self.storage.query_topic_events(
                umo=source.umo, topic="项目决策"
            )[0]["id"],
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
            self.storage.query_conversation_time(
                umo=other_umo, event_id=event_id
            )
        )

        dashboard = self.storage.dashboard_graph(umo=source.umo, limit=20)
        labels = {str(node["label"]) for node in dashboard["nodes"]}
        self.assertIn("方案评审", labels)
        self.assertIn("项目决策", labels)
        self.assertIn("工作偏好", labels)
        self.assertNotIn("外群事件", labels)
        self.assertNotIn("另一个群的私有内容", labels)
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
