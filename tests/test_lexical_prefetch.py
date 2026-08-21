from __future__ import annotations

import unittest
import uuid
from pathlib import Path

from mr_memory.models import NormalizedMessage
from mr_memory.storage import MemoryStorage


class LexicalPrefetchTests(unittest.TestCase):
    def setUp(self) -> None:
        root = Path.cwd() / ".dev" / "test-tmp"
        root.mkdir(parents=True, exist_ok=True)
        self.database_path = root / f"{uuid.uuid4().hex}.db"
        self.storage = MemoryStorage(self.database_path)
        self.umo = "shadow:GroupMessage:group-a"

    def tearDown(self) -> None:
        self.storage.close()
        for suffix in ("", "-wal", "-shm"):
            Path(f"{self.database_path}{suffix}").unlink(missing_ok=True)

    def message(self, message_id: str, text: str, sent_at: int) -> NormalizedMessage:
        return NormalizedMessage(
            platform="aiocqhttp",
            platform_id="shadow",
            umo=self.umo,
            group_id="group-a",
            message_id=message_id,
            sender_id="member-a",
            sender_name="群友甲",
            sent_at=sent_at,
            plain_text=text,
            content=[{"type": "plain", "text": text}],
        )

    def test_exact_query_cue_seeds_source_bounded_episode_without_embedding(
        self,
    ) -> None:
        question = self.message("question", "d老师是啥", 100)
        answer = self.message("answer", "DeepSeek", 101)
        future = self.message("future", "未来才出现的解释", 300)
        for message in (question, answer, future):
            self.storage.upsert_message(message)
        visible_episode = self.storage.store_episode(
            umo=self.umo,
            started_at=100,
            ended_at=101,
            title="D老师指代",
            summary="群聊把 D老师解释为 DeepSeek。",
            source_keys=[
                question.resolved_source_key(),
                answer.resolved_source_key(),
            ],
            keywords=[
                ("D老师", "AI工具"),
                *(("/chat", f"命令污染-{index:02d}") for index in range(24)),
            ],
        )
        self.storage.store_episode(
            umo=self.umo,
            started_at=300,
            ended_at=300,
            title="未来解释",
            summary="不可见。",
            source_keys=[future.resolved_source_key()],
            keywords=[("D老师", "未来")],
        )

        matches = self.storage.query_matching_cues(
            umo=self.umo,
            query="/chat 群里有d老师吗，请把他找出来",
            before_sent_at=200,
            message_upper_bound=2,
        )

        self.assertEqual(
            matches,
            [
                {
                    "cue": "D老师",
                    "episode_count": 1,
                    "tags": [{"tag": "AI工具", "episode_count": 1}],
                }
            ],
        )
        packet = self.storage.reconstruction_evidence_packet(
            umo=self.umo,
            candidates={"cues": matches},
            before_sent_at=200,
            message_upper_bound=2,
        )
        self.assertEqual(
            [item["id"] for item in packet["expanded_episodes"]],
            [visible_episode],
        )
        self.assertEqual(
            [
                item["plain_text"]
                for item in packet["expanded_episodes"][0]["messages"]
            ],
            ["d老师是啥", "DeepSeek"],
        )

    def test_unrelated_query_does_not_create_a_lexical_seed(self) -> None:
        message = self.message("one", "D老师已经落后了", 100)
        self.storage.upsert_message(message)
        self.storage.store_episode(
            umo=self.umo,
            started_at=100,
            ended_at=100,
            title="D老师",
            summary="旧对话。",
            source_keys=[message.resolved_source_key()],
            keywords=[("D老师", "AI工具")],
        )
        self.assertEqual(
            self.storage.query_matching_cues(
                umo=self.umo,
                query="今天吃什么",
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
