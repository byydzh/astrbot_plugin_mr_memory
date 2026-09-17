import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from mr_memory.store import Store


class ReflectionDirectoryTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / ".pytest_cache"
        scratch.mkdir(exist_ok=True)
        self.scratch = scratch / ("reflection-directory-" + uuid.uuid4().hex)
        self.scratch.mkdir()
        self.store = Store(self.scratch / "group.db", "test:GroupMessage:42")
        self.source = self.store.append_message(dict(
            platform="test", platform_id="test", umo=self.store.umo, group_id="42",
            message_id="1", sender_id="alice", sender_name="小叶", sent_at=1_800_000_001,
            plain_text="我说的是朋友办展览。", role="USER",
            content=[{"type": "text", "text": "我说的是朋友办展览。"}]))

    def tearDown(self):
        self.store.close()
        for file in self.scratch.iterdir():
            file.unlink()
        self.scratch.rmdir()

    def save(self, item):
        keys = [self.source["source_key"]]
        return self.store.save_memories([{**item, "source_keys": keys}], keys,
                                        mark_processed=False)[0]

    def concern(self, *refs):
        return self.store.reflections.save({
            "content": "要区分发言者和被谈论的朋友。", "priority": 9, "status": "waiting",
            "source_ids": [self.source["id"], 999], "memory_refs": list(refs)})

    def assert_directory(self, compact, complete):
        self.assertNotIn("sources", compact)
        for field in ("id", "content", "priority", "status", "source_ids", "source_keys",
                      "missing_source_ids", "memory_refs"):
            self.assertEqual(compact[field], complete[field])
        self.assertEqual(compact["source_keys"], [self.source["source_key"]])
        self.assertEqual(compact["missing_source_ids"], [999])

    def test_memory_directory_keeps_concern_and_addresses_without_loading_originals(self):
        memory = self.save({"kind": "semantic", "content": "小叶提到朋友办展览"})
        concern = self.concern({"kind": "semantic", "id": memory["id"]})
        with patch.object(self.store, "_message", side_effect=AssertionError("directory opened raw text")):
            lexical = self.store.search_memories(kind="semantic")[0]
            semantic = self.store.memory("semantic", memory["id"], include_sources=False)
        self.assert_directory(lexical["reflections"][0], concern)
        self.assertEqual(lexical, semantic)
        for complete in (
            self.store.reflections.get(concern["id"]),
            self.store.memory("semantic", memory["id"])["reflections"][0],
            self.store.memory("semantic", memory["id"], include_history=True)["reflections"][0],
        ):
            self.assertEqual(complete["sources"], concern["sources"])
            self.assertEqual(complete["sources"][0]["sender_id"], "alice")
            self.assertEqual(complete["sources"][0]["plain_text"], self.source["plain_text"])

    def test_learning_directories_open_sources_only_on_explicit_read(self):
        concern = self.concern()
        self.store.reflections.save({"id": concern["id"], "next_review_at": 1})
        complete = self.store.reflections.get(concern["id"])
        with patch.object(self.store, "_message", side_effect=AssertionError("directory opened raw text")):
            for compact in (self.store.reflections.waiting(include_sources=False)[0],
                            self.store.reflections.due(include_sources=False)[0],
                            self.store.reflections.get(concern["id"], include_sources=False)):
                self.assert_directory(compact, complete)
        self.assertEqual(self.store.reflections.get(concern["id"])["sources"], complete["sources"])

    def test_graph_directory_propagates_view_to_both_node_reflections(self):
        edge = self.save({"kind": "association", "source": "小叶", "target": "朋友",
                          "relation": "提到", "statement": "小叶提到朋友办展览"})
        concern = self.concern({"kind": "association", "id": edge["id"]},
                               {"kind": "node", "id": edge["source_node_id"]},
                               {"kind": "node", "id": edge["target_node_id"]})
        with patch.object(self.store, "_message", side_effect=AssertionError("directory opened raw text")):
            directory = self.store.graph()[0]
            node = self.store.memory("node", edge["source_node_id"], include_sources=False)
        for item in (directory, directory["source_node"], directory["target_node"], node):
            self.assert_directory(item["reflections"][0], concern)
        detail = self.store.memory("association", edge["id"], include_history=True)
        for item in (detail, detail["source_node"], detail["target_node"],
                     self.store.memory("node", edge["source_node_id"])):
            self.assertEqual(item["reflections"][0]["sources"], concern["sources"])


if __name__ == "__main__":
    unittest.main()
