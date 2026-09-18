"""Storage and tool contracts only; scripted responses are not quality evidence."""
import asyncio
import json
import sqlite3
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from mr_memory.agent import MemoryAgent
from mr_memory.store import Store
from mr_memory.learning_writes import LearningWriter
from tests.test_agent import Provider, response


@contextmanager
def local_database_path():
    root = Path(__file__).resolve().parents[1] / ".pytest_cache"
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

    def test_thought_can_become_the_basis_of_another_memory(self):
        episode = self.save(kind="episode", title="共同经历", content="一起修改方案")
        thought = self.store.reflections.save({"content": "这种协作方式还有哪些适用情境", "memory_refs": [
            {"kind": "episode", "id": episode["id"]}]})
        pattern = self.save(kind="pattern", content="从这项思考形成的暂时理解", connections=[
            {"kind": "reflection", "id": thought["id"], "relation": "由这项思考发展", "purpose": "basis"}])
        self.assertEqual(self.store.memory("reflection", thought["id"])["content"], thought["content"])
        self.assertEqual(len(self.store.navigate(ref={"kind": "reflection", "id": thought["id"]})["connections"]), 2)
        self.store.reflections.save({"id": thought["id"], "content": "进一步区分自愿合作与任务分工", "status": "waiting"})
        self.assertTrue(self.store.memory("pattern", pattern["id"])["basis"][0]["basis_changed"])
        self.assertEqual(self.store.reflections.waiting()[0]["content"], self.store.memory("reflection", thought["id"])["content"])

    def test_recall_experience_is_not_ui_read_or_learning_echo(self):
        first = self.save(kind="pattern", title="认识", content="一个认识")
        second = self.save(kind="episode", title="经历", content="另一个经历")
        self.store.memory("pattern", first["id"])
        self.assertEqual(self.store.reconsider()["items"], [])
        for key in ("a", "a", "b"):
            self.store.recall_memory(first, run_key=key)
        self.store.recall_memory(second, run_key="b")
        self.store.recall_memory(first, run_key="learning", purpose="learning")
        attention = next(r for r in self.store.reconsider()["items"] if r["kind"] == "pattern")
        self.assertEqual(attention["recalls"], 2)
        self.assertEqual(attention["co_recalled"][0]["kind"], "episode")
        self.store.recall_memory(first, run_key="c")
        self.store.save_reconsideration([{**attention, "note": "这次重温形成了下一步问题"}])
        pending = next(r for r in self.store.reconsider()["items"] if r["kind"] == "pattern")
        self.assertEqual(pending["recalls"], 1)
        self.assertEqual(self.store.memory("pattern", first["id"])["last_reconsideration"]["note"], "这次重温形成了下一步问题")

    def test_foreground_can_write_then_read_same_graph(self):
        provider = Provider([
            response(calls=[("remember", {"items": [{"kind": "pattern", "title": "新的认识", "content": "这次形成的理解",
                                                        "source_ids": [self.message["id"]]}]}, "save")]),
            response(calls=[("memory", {"kind": "pattern", "id": 1}, "read")]),
            response('{"background":"本轮背景","working_memory":"pattern/1值得继续理解"}')])
        async def run():
            with patch("mr_memory.agent._tool_set", return_value=object()):
                return await MemoryAgent(provider, self.store, max_turns=4).reconstruct(self.message, [self.message], {})
        result = asyncio.run(run())
        self.assertEqual(result.status, "completed", result.detail)
        self.assertEqual(len(result.written), 1)
        self.assertEqual(result.tool_calls[1]["result"]["summary"], "这次形成的理解")
        self.assertEqual(self.store.reconsider()["items"][0]["kind"], "pattern")

    def test_final_background_can_commit_understanding_without_another_model_turn(self):
        provider = Provider([response(json.dumps({"background": "本轮背景", "working_memory": "接着讨论方案",
            "items": [{"kind": "pattern", "title": "协作方式", "content": "共同修改形成的理解", "source_ids": [self.message["id"]]}]}))])
        async def run():
            with patch("mr_memory.agent._tool_set", return_value=object()):
                return await MemoryAgent(provider, self.store).reconstruct(self.message, [self.message], {})
        result = asyncio.run(run())
        self.assertEqual(result.status, "completed", result.detail)
        self.assertEqual(result.background, "本轮背景")
        self.assertEqual(len(provider.requests), 1)
        self.assertEqual(self.store.memory("pattern", result.written[0]["id"])["content"], "共同修改形成的理解")

    def test_foreground_drafts_survive_failed_request_without_repeating_committed_writes(self):
        first = LearningWriter(self.store, {}, foreground=True)
        second = LearningWriter(self.store, {}, foreground=True)
        first.apply({"items": [{"kind": "pattern", "content": "未完成的认识", "source_ids": [999]}]}, "a")
        second.apply({"items": [{"kind": "episode", "content": "另一请求的草稿", "source_ids": [998]}]}, "b")
        self.assertEqual(set(self.store.recall_drafts()), {"a:0", "b:0"})
        resumed = LearningWriter(self.store, {}, foreground=True)
        save = self.store.save_memories
        def interrupted_receipt(*args, **kwargs):
            save(*args, **kwargs)
            raise RuntimeError("Write committed before its detailed receipt failed")
        with patch.object(self.store, "save_memories", side_effect=interrupted_receipt):
            outcome = resumed.apply({"retry": [{"pending_id": "a:0", "changes": {"source_ids": []}}]}, "resume")
        self.assertEqual(len(outcome.written), 1)
        self.assertEqual(set(self.store.recall_drafts()), {"b:0"})
        self.assertEqual(len(self.store.search_memories(kind="pattern")), 1)

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
