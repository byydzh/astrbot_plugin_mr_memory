"""Storage and tool contracts only; scripted responses are not quality evidence."""
import os
import json
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from mr_memory.store import Store


@contextmanager
def local_database_path():
    root = Path(os.environ['ASTRBOT_ROOT'])
    root.mkdir(exist_ok=True)
    path = root / ("graph-migration-" + uuid4().hex + ".db")
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


class MemoryGraphTests(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:", "test:GroupMessage:42")
        self.message = self.store.append_message({
            "platform": "test", "platform_id": "test", "umo": self.store.umo, "group_id": "42",
            "message_id": "1", "sender_id": "10", "sender_name": "甲", "sent_at": 1800000000,
            "plain_text": "一起修改方案", "content": [], "role": "USER"})

    def tearDown(self):
        self.store.close()

    def save(self, **item):
        return self.store.save_memories([item], mark_processed=False)[0]

    def test_related_account_search_preserves_results_and_uses_key_lookups(self):
        episode = self.save(kind="episode", content="共同经历", source_ids=[self.message["id"]])
        pattern = self.save(kind="pattern", content="由经历形成", connections=[{
            "kind": "episode", "id": episode["id"], "relation": "依据", "purpose": "basis"}])
        higher = self.save(kind="perspective", content="继续形成", connections=[{
            "kind": "pattern", "id": pattern["id"], "relation": "依据", "purpose": "basis"}])
        unrelated = self.save(kind="pattern", content="无关事项")
        original_rows = self.store._rows
        checked = []

        def inspect(sql, params=()):
            if sql.startswith("WITH RECURSIVE related"):
                # Compare the complete candidate result on a small fixture;
                # never benchmark the pathological old plan on production.
                self.assertEqual(original_rows(sql, params), original_rows(sql.replace("CROSS JOIN", "JOIN"), params))
                plan = original_rows("EXPLAIN QUERY PLAN " + sql, params)
                self.assertTrue(any("SEARCH m USING INTEGER PRIMARY KEY" in r["detail"] for r in plan), plan)
                self.assertTrue(any("SEARCH e USING INDEX mr_memory_target" in r["detail"]
                                    and r["detail"].count("<expr>=?") == 2 for r in plan), plan)
                checked.append(True)
            return original_rows(sql, params)

        with patch.object(self.store, "_rows", side_effect=inspect):
            found = self.store.search_memories(related_account_id="10", limit=100)
        refs = {(m["kind"], m["id"]) for m in found}
        for memory in (episode, pattern, higher):
            self.assertIn((memory["kind"], memory["id"]), refs)
        self.assertNotIn((unrelated["kind"], unrelated["id"]), refs)
        self.assertEqual(checked, [True])

    def test_recursive_abstractions_navigate_both_ways_and_revise_basis(self):
        episode = self.save(kind="episode", title="一次协作", content="共同修改方案", source_ids=[self.message["id"]],
                            cues=[{"cue": "协作", "aspect": "具体经历"}])
        pattern = self.save(kind="pattern", title="形成的理解", content="形成了一个暂时的协作认识", connections=[
            {"kind": "episode", "id": episode["id"], "relation": "从这次经历形成", "purpose": "basis"}])
        higher = self.save(kind="perspective", title="另一个视角", content="把已有认识放到更广的情境理解", connections=[
            {"kind": "pattern", "id": pattern["id"], "relation": "重新解释", "purpose": "basis"}])
        directory = self.store.navigate(cue="协作", aspect="具体经历")
        self.assertEqual(directory["items"][0]["id"], episode["id"])
        self.assertNotIn("summary", directory["items"][0])
        self.assertEqual(len(self.store.graph(ref={"kind": "pattern", "id": pattern["id"]})), 2)
        self.assertFalse(self.store.memory("pattern", pattern["id"])["basis"][0]["basis_changed"])
        self.save(kind="episode", id=episode["id"], revision_no=1, content="新的解释", reason="后来理解有变")
        revised = self.store.memory("pattern", pattern["id"])
        self.assertTrue(revised["basis"][0]["basis_changed"])
        self.assertEqual(revised["summary"], pattern["summary"])
        self.assertIsNotNone(self.store.memory("perspective", higher["id"]))
        with self.assertRaisesRegex(ValueError, "changed since"):
            self.save(kind="episode", id=episode["id"], revision_no=1, content="迟到的旧写入")
        self.assertEqual(self.store.memory("episode", episode["id"])["summary"], "新的解释")
        self.assertEqual(self.store.memory("episode", episode["id"], include_history=True)["history"][0]["snapshot"]["memory"]["summary"], "共同修改方案")

    def test_topic_creation_and_edit_preserve_constituent_links(self):
        episode = self.save(kind="episode", title="经历", content="一段经历", source_ids=[self.message["id"]])
        topic = self.save(kind="topic", title="共同事项", content="关于这个事项的理解", connections=[
            {"kind": "episode", "id": episode["id"], "relation": "包含经历", "purpose": "basis"}])
        self.save(kind="topic", id=topic["id"], content="事项出现新进展")
        self.assertEqual(len(self.store.graph(ref={"kind": "topic", "id": topic["id"]})), 1)
        self.assertEqual(self.store.memory("episode", episode["id"])["source_ids"], [self.message["id"]])
        self.assertIn("topic", {m["kind"] for m in self.store.search_memories()})

    def test_withdrawn_basis_remains_visible_to_its_dependent_memory(self):
        episode = self.save(kind="episode", content="共同经历", source_ids=[self.message["id"]])
        pattern = self.save(kind="pattern", content="当时的解释", connections=[{
            "kind": "episode", "id": episode["id"], "relation": "由此推想", "purpose": "basis",
            "belief": {"stance": "可能有关"}}])
        edge = pattern["basis"][0]
        self.assertEqual(edge["belief"]["stance"], "可能有关")
        self.save(kind="association", id=edge["id"], action="withdraw", reason="后续交流改变了理解",
                  belief={"stance": "这条推论不成立"})
        updated = self.store.memory("pattern", pattern["id"])
        self.assertEqual(updated["basis"][0]["status"], "RETRACTED")
        self.assertTrue(updated["basis"][0]["basis_changed"])
        self.assertEqual(updated["basis"][0]["change_reason"], "后续交流改变了理解")
        self.assertEqual(self.store.graph(ref={"kind": "pattern", "id": pattern["id"]}), [])
        history = self.store.memory("association", edge["id"], include_history=True)["history"]
        self.assertTrue(any(row["operation"] == "withdraw" for row in history))
        self.assertEqual(self.store.memory("episode", episode["id"])["content"], "共同经历")

    def test_legacy_topic_and_entity_graph_import_once_with_addresses_intact(self):
        with local_database_path() as path:
            first = Store(path, self.store.umo)
            with first.db:
                first.db.executescript("""
                    CREATE TABLE episodes(id INTEGER PRIMARY KEY,umo TEXT,title TEXT,summary TEXT,status TEXT);
                    CREATE TABLE topics(id INTEGER PRIMARY KEY,umo TEXT,name TEXT,summary TEXT);
                    CREATE TABLE topic_episodes(topic_id INTEGER,episode_id INTEGER);
                    CREATE TABLE plastic_nodes(id INTEGER PRIMARY KEY,umo TEXT,label TEXT,description TEXT);
                    CREATE TABLE relation_types(id INTEGER PRIMARY KEY,umo TEXT,canonical_name TEXT);
                    CREATE TABLE plastic_edges(id INTEGER PRIMARY KEY,umo TEXT,source_node_id INTEGER,target_node_id INTEGER,relation_type_id INTEGER,statement TEXT);
                    CREATE TABLE mr_reflections(id INTEGER PRIMARY KEY,umo TEXT,content TEXT,status TEXT,source_ids_json TEXT,memory_refs_json TEXT,created_at REAL,updated_at REAL);
                    DELETE FROM mr_graph_migrations;
                """)
                first.db.execute("INSERT INTO episodes VALUES(8,?,'经历','经过','READY')", (first.umo,))
                first.db.execute("INSERT INTO topics VALUES(3,?,'主题','概括')", (first.umo,))
                first.db.execute("INSERT INTO topic_episodes VALUES(3,8)")
                first.db.executemany("INSERT INTO plastic_nodes VALUES(?,?,?,?)", [(4, first.umo, "甲", "一个人"), (5, first.umo, "事项", "一件事")])
                first.db.execute("INSERT INTO relation_types VALUES(2,?,'参与')", (first.umo,))
                first.db.execute("INSERT INTO plastic_edges VALUES(9,?,4,5,2,'甲参与事项')", (first.umo,))
                first.db.execute("INSERT INTO mr_reflections VALUES(2,?,'继续理解协作方式','waiting','[]',?,100,101)",
                                 (first.umo, json.dumps([{"kind": "episode", "id": 8}])))
            first.close()
            migrated = Store(path, self.store.umo)
            self.assertEqual(migrated.memory("episode", 8)["summary"], "经过")
            self.assertEqual(migrated.search_memories(kind="episode")[0]["id"], 8)
            self.assertEqual(migrated.memory("association", 9)["source_ref"], {"kind": "node", "id": 4})
            self.assertEqual(migrated.graph(ref={"kind": "topic", "id": 3})[0]["target_ref"], {"kind": "episode", "id": 8})
            self.assertEqual(migrated.reflections.get(2)["content"], migrated.memory("reflection", 2)["content"])
            self.assertEqual(migrated.graph(ref={"kind": "reflection", "id": 2})[0]["target_ref"], {"kind": "episode", "id": 8})
            count = migrated.db.execute("SELECT count(*) FROM mr_memory_objects").fetchone()[0]
            migrated.close()
            migrated = Store(path, self.store.umo)
            self.assertEqual(migrated.db.execute("SELECT count(*) FROM mr_memory_objects").fetchone()[0], count)
            migrated.close()


if __name__ == "__main__":
    unittest.main()
