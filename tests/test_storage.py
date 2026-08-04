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


if __name__ == "__main__":
    unittest.main()
