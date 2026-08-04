from __future__ import annotations

import asyncio
import json
import unittest
import uuid
from pathlib import Path

from mr_memory.distillation import parse_distillation_response
from mr_memory.embedding import HashEmbeddingBackend
from mr_memory.models import NormalizedMessage
from mr_memory.service import MemoryService
from mr_memory.storage import MemoryStorage


class DistillationPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        test_root = Path.cwd() / ".dev" / "test-tmp"
        test_root.mkdir(parents=True, exist_ok=True)
        self.database_path = test_root / f"{uuid.uuid4().hex}.db"
        self.storage = MemoryStorage(self.database_path)
        self.service = MemoryService(self.storage)
        self.umo = "shadow:GroupMessage:group-a"
        for message_id, sender, sent_at, text in (
            ("1", "甲", 100, "先考虑方案 A，价格比较低。"),
            ("2", "乙", 200, "安全审查发现方案 A 的权限隔离有问题。"),
            ("3", "丙", 300, "提议改用方案 B，它已经通过审查。"),
            ("4", "甲", 400, "最终决定使用方案 B，方案 A 不再推进。"),
        ):
            self.storage.upsert_message(
                NormalizedMessage(
                    platform="aiocqhttp",
                    platform_id="shadow",
                    umo=self.umo,
                    group_id="group-a",
                    message_id=message_id,
                    sender_id=sender,
                    sender_name=sender,
                    sent_at=sent_at,
                    plain_text=text,
                    content=[{"type": "plain", "text": text}],
                )
            )

    def tearDown(self) -> None:
        self.storage.close()
        for suffix in ("", "-wal", "-shm"):
            Path(f"{self.database_path}{suffix}").unlink(missing_ok=True)

    def _messages(self):
        return self.storage.search_messages(umo=self.umo, limit=20)

    def _response(self) -> str:
        keys = [message.source_key for message in self._messages()]
        return json.dumps(
            {
                "episodes": [
                    {
                        "source_keys": keys[:2],
                        "started_at": 100,
                        "ended_at": 200,
                        "title": "方案 A 评审",
                        "summary": "群聊先考虑方案 A，随后发现权限隔离问题。",
                        "tag": "方案评审",
                        "cues": ["方案 A", "权限隔离", "安全审查"],
                    },
                    {
                        "source_keys": keys[2:],
                        "started_at": 300,
                        "ended_at": 400,
                        "title": "改用方案 B",
                        "summary": "群聊最终决定采用已通过审查的方案 B。",
                        "tag": "方案选择",
                        "cues": ["方案 B", "最终决定", "通过审查"],
                    },
                ],
                "semantic_memories": [],
                "topics": [
                    {
                        "name": "项目方案决策",
                        "summary": "从方案 A 转向方案 B 的评审过程。",
                        "episode_indices": [0, 1],
                    }
                ],
            },
            ensure_ascii=False,
        )

    def test_reproduces_construction_embedding_seed_and_graph_traversal(self) -> None:
        messages = self._messages()
        batch = parse_distillation_response(self._response(), messages)
        backend = HashEmbeddingBackend(dimensions=128)
        persisted, indexed = asyncio.run(
            self.service.apply_distillation(
                batch,
                extractor_version="test-reproduction",
                embedding_backend=backend,
            )
        )

        self.assertEqual(len(persisted.episode_ids), 2)
        self.assertEqual(len(persisted.topic_ids), 1)
        self.assertGreaterEqual(indexed, 9)

        repeated, repeated_indexed = asyncio.run(
            self.service.apply_distillation(
                batch,
                extractor_version="test-reproduction",
                embedding_backend=backend,
            )
        )
        self.assertEqual(repeated.episode_ids, persisted.episode_ids)
        self.assertEqual(repeated.topic_ids, persisted.topic_ids)
        self.assertEqual(repeated_indexed, indexed)
        self.assertEqual(self.storage.count_graph_units(umo=self.umo), 3)

        foreign_vector = asyncio.run(backend.embed_query("最后为什么选方案 B？"))
        self.storage.upsert_memory_embedding(
            umo="shadow:GroupMessage:group-b",
            owner_type="cue",
            owner_key="外群秘密",
            model=backend.model_id,
            vector=foreign_vector,
        )

        candidates = asyncio.run(
            self.service.initialize_candidates(
                umo=self.umo,
                query="最后为什么选方案 B？",
                embedding_backend=backend,
                limit=12,
            )
        )
        cue_candidates = {item["cue"]: item for item in candidates["cues"]}
        self.assertIn("方案 B", cue_candidates)
        self.assertNotIn("外群秘密", cue_candidates)
        self.assertIn(
            "方案选择",
            {tag["tag"] for tag in cue_candidates["方案 B"]["tags"]},
        )

        events = self.storage.query_tag_events(
            umo=self.umo,
            cue="方案 B",
            tag="方案选择",
        )
        self.assertEqual([item["title"] for item in events], ["改用方案 B"])
        context = self.storage.query_event_context(
            umo=self.umo,
            event_id=int(events[0]["id"]),
        )
        self.assertEqual(len(context), 2)
        self.assertIn("最终决定", context[-1]["plain_text"])

    def test_rejects_invented_source_ids_before_writing_graph(self) -> None:
        value = json.loads(self._response())
        value["episodes"][0]["source_keys"] = ["invented"]
        with self.assertRaisesRegex(ValueError, "invented source keys"):
            parse_distillation_response(
                json.dumps(value, ensure_ascii=False), self._messages()
            )
        self.assertEqual(self.storage.count_graph_units(umo=self.umo), 0)


if __name__ == "__main__":
    unittest.main()
