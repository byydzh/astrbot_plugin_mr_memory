from __future__ import annotations

import asyncio
import json
import unittest
import uuid
from pathlib import Path

from mr_memory.distillation import (
    build_distillation_repair_prompt,
    distillation_generation_options,
    parse_distillation_response,
    parse_distillation_response_resilient,
)
from mr_memory.embedding import HashEmbeddingBackend
from mr_memory.models import NormalizedMessage
from mr_memory.service import MemoryService
from mr_memory.storage import MemoryStorage


class DistillationPipelineTests(unittest.TestCase):
    def test_deepseek_v4_structured_extraction_enables_thinking(self) -> None:
        self.assertEqual(
            distillation_generation_options(
                model_name="deepseek-v4-flash",
                max_tokens=8192,
            ),
            {
                "temperature": 0.0,
                "thinking": {"type": "enabled"},
                "response_format": {"type": "json_object"},
                "max_tokens": 8192,
            },
        )

    def test_deepseek_v4_thinking_can_be_explicitly_disabled(self) -> None:
        options = distillation_generation_options(
            model_name="deepseek-v4-flash",
            thinking_mode="disabled",
        )
        self.assertEqual(options["thinking"], {"type": "disabled"})

    def test_other_providers_do_not_receive_deepseek_only_options(self) -> None:
        self.assertEqual(
            distillation_generation_options(model_name="gpt-5.4-mini"),
            {"temperature": 0.0},
        )

    def test_repair_prompt_contains_complete_retry_context(self) -> None:
        prompt = build_distillation_repair_prompt(
            original_prompt='{"target_source_keys":["source-1"]}',
            invalid_output='{"episodes":[]',
            validation_error="invalid JSON",
        )
        self.assertIn("invalid JSON", prompt)
        self.assertIn('"target_source_keys":["source-1"]', prompt)
        self.assertIn('{"episodes":[]', prompt)

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

    def test_requires_explicit_coverage_or_ignore_for_every_target(self) -> None:
        value = json.loads(self._response())
        omitted_keys = value["episodes"][1]["source_keys"]
        value["episodes"] = value["episodes"][:1]
        value["topics"][0]["episode_indices"] = [0]
        with self.assertRaisesRegex(ValueError, "omitted target source keys"):
            parse_distillation_response(
                json.dumps(value, ensure_ascii=False), self._messages()
            )
        value["ignored"] = [
            {"source_key": key, "reason": "本条不形成额外持久记忆"}
            for key in omitted_keys
        ]
        batch = parse_distillation_response(
            json.dumps(value, ensure_ascii=False), self._messages()
        )
        self.assertEqual(
            {item.source_key for item in batch.ignored_sources},
            set(omitted_keys),
        )

    def test_resilient_parser_drops_invalid_units_and_audits_uncovered_raw(self) -> None:
        value = json.loads(self._response())
        omitted_keys = list(value["episodes"][1]["source_keys"])
        value["episodes"][1]["source_keys"] = ["invented"]
        value["topics"].append(
            {
                "name": "空主题",
                "summary": "模型产生的无效可选主题",
                "episode_indices": [],
            }
        )

        batch, actions = parse_distillation_response_resilient(
            json.dumps(value, ensure_ascii=False),
            self._messages(),
        )

        self.assertEqual(len(batch.episodes), 1)
        self.assertEqual(len(batch.topics), 0)
        self.assertEqual(
            {item.source_key for item in batch.ignored_sources},
            set(omitted_keys),
        )
        self.assertTrue(all("raw evidence retained" in item.reason for item in batch.ignored_sources))
        self.assertIn("drop:episodes", actions)
        self.assertIn("host_ignore:uncovered_target", actions)

    def test_construction_persists_an_unresolved_competing_association(self) -> None:
        value = json.loads(self._response())
        source_key = value["episodes"][0]["source_keys"][0]
        value["associations"] = [
            {
                "operation": "upsert_edge",
                "evidence_source_keys": [source_key],
                "confidence": 0.58,
                "utility_delta": 0.1,
                "statement": "方案 A 可能是群内对低价但高风险选择的代称。",
                "epistemic_state": "HYPOTHESIS",
                "uncertainty": "当前证据也可能只是在讨论这一次具体方案。",
                "source": {
                    "kind": "symbol",
                    "label": "方案 A",
                    "description": "群聊中出现的方案称呼",
                },
                "relation": {
                    "key": "possible_local_shorthand_for",
                    "name": "可能是本群代称",
                    "description": "表达可能在本群形成可复用的局部语义",
                    "source_kinds": ["symbol"],
                    "target_kinds": ["concept"],
                },
                "target": {
                    "kind": "concept",
                    "label": "低价但高风险的选择",
                    "description": "尚未被后续群聊确认的候选含义",
                },
            }
        ]
        batch = parse_distillation_response(
            json.dumps(value, ensure_ascii=False), self._messages()
        )
        persisted, indexed = asyncio.run(
            self.service.apply_distillation(
                batch,
                extractor_version="test-uncertainty",
                embedding_backend=HashEmbeddingBackend(dimensions=64),
            )
        )
        self.assertEqual(len(persisted.plastic_edge_ids), 1)
        self.assertGreater(indexed, 0)
        rows = self.storage.query_plastic_associations(
            umo=self.umo,
            query="方案 A",
        )
        self.assertEqual(rows[0]["epistemic_state"], "HYPOTHESIS")
        self.assertIn("具体方案", rows[0]["uncertainty"])


if __name__ == "__main__":
    unittest.main()
