from __future__ import annotations

import unittest
import uuid
from pathlib import Path

from mr_memory.models import NormalizedMessage
from mr_memory.retrieval_terms import literal_recall_query, recall_coverage_terms, short_recall_terms
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
        question = self.message("question", "q工位是啥", 100)
        answer = self.message("answer", "合成线圈检测台", 101)
        future = self.message("future", "未来才出现的解释", 300)
        for message in (question, answer, future):
            self.storage.upsert_message(message)
        visible_episode = self.storage.store_episode(
            umo=self.umo,
            started_at=100,
            ended_at=101,
            title="Q工位指代",
            summary="本合成对话中，Q工位是线圈检测台的代号。",
            source_keys=[
                question.resolved_source_key(),
                answer.resolved_source_key(),
            ],
            keywords=[
                ("Q工位", "合成设备"),
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
            keywords=[("Q工位", "未来")],
        )

        matches = self.storage.query_matching_cues(
            umo=self.umo,
            query="/chat 群里有q工位吗，请把它找出来",
            before_sent_at=200,
            message_upper_bound=2,
        )

        self.assertEqual(
            matches,
            [
                {
                    "cue": "Q工位",
                    "episode_count": 1,
                    "tags": [{"tag": "合成设备", "episode_count": 1}],
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
            ["q工位是啥", "合成线圈检测台"],
        )

    def test_unrelated_query_does_not_create_a_lexical_seed(self) -> None:
        message = self.message("one", "合成的Q工位已结束线圈检测", 100)
        self.storage.upsert_message(message)
        self.storage.store_episode(
            umo=self.umo,
            started_at=100,
            ended_at=100,
            title="Q工位",
            summary="旧对话。",
            source_keys=[message.resolved_source_key()],
            keywords=[("Q工位", "合成设备")],
        )
        self.assertEqual(
            self.storage.query_matching_cues(
                umo=self.umo,
                query="今天吃什么",
            ),
            [],
        )

    def test_request_framing_cannot_evict_literal_targets(self) -> None:
        targets = [self.message(f"target-{index}", f"今天见到{name}。", 100 + index)
                   for index, name in enumerate(("春岚", "秋岚", "夏岚"))]
        noise = [self.message("noise-one", "分辨一下显示器的颜色", 200),
                 self.message("noise-two", "是几个人去搬桌子", 201),
                 self.message("noise-three", "请把他找出来", 202)]
        for message in (*targets, *noise):
            self.storage.upsert_message(message)
        query = "分辨一下春岚，秋岚和夏岚是几个人"
        selected = self.storage.search_messages(umo=self.umo, query=query,
                                               match_mode="recall", limit=3,
                                               before_sent_at=300, message_upper_bound=6)
        self.assertEqual({message.source_key for message in selected},
                         {message.resolved_source_key() for message in targets})
        self.assertFalse({"分辨", "一下", "几个"}.intersection(recall_coverage_terms(query)))
        self.assertEqual(literal_recall_query("群里有Q老师吗，请把他找出来"), "Q老师")
        self.assertEqual(literal_recall_query("查一下分辨率"), "分辨率")
        self.assertEqual(literal_recall_query("《分辨一下》是几个人"), "《分辨一下》是几个人")
        self.assertNotIn("老师", recall_coverage_terms("群里有Q老师吗，请把他找出来"))
        self.assertIn("q老师", recall_coverage_terms("群里有Q老师吗，请把他找出来"))
        self.assertIn("老师", short_recall_terms("Q老师和老师"))


if __name__ == "__main__":
    unittest.main()
