import sqlite3
import struct
import unittest
import uuid
from pathlib import Path

from mr_memory.store import Store


class MemoryRevisionTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / ".pytest_cache"
        scratch.mkdir(exist_ok=True)
        self.temp = scratch / ("revision-" + uuid.uuid4().hex)
        self.temp.mkdir()
        self.path = self.temp / "group.db"
        self.scope = "fixture:GroupMessage:7"
        self.store = Store(self.path, self.scope)

    def tearDown(self):
        self.store.close()
        for path in self.temp.iterdir():
            path.unlink()
        self.temp.rmdir()

    def message(self, number, text, sender="alice"):
        return self.store.append_message(dict(platform="fixture", platform_id="fixture", umo=self.scope,
            group_id="7", message_id=str(number), sender_id=sender, sender_name=sender,
            sent_at=1800000000 + number, plain_text=text, content=[{"type": "text", "text": text}], role="USER"))

    def save(self, item, *sources, **kwargs):
        keys = [source["source_key"] for source in sources]
        return self.store.save_memories([{**item, "source_keys": keys}], keys, mark_processed=False, **kwargs)[0]

    def test_person_correction_replaces_primary_evidence_and_stale_derivatives(self):
        old = self.message(1, "机器人以为展览是小叶办的", "bot")
        correct = self.message(2, "我说的是小叶的朋友举办展览")
        memory = self.save({"kind": "semantic", "subject": {"name": "小叶", "account_id": "alice"},
                           "content": "小叶举办展览", "aspect": "举办者"}, old)
        owner = memory["id"]
        concern = self.store.reflections.save({"content": "需要辨清展览举办者", "source_ids": [correct["id"]],
                                              "memory_refs": [{"kind": "semantic", "id": owner}]})
        self.store.put_vector("semantic", str(owner), "fixture", struct.pack("<2f", 1, 0), 2)
        stale = self.store.pending_embeddings("fixture")[0]
        revised = self.save({"kind": "semantic", "id": owner, "subject": {"name": "小叶的朋友"},
                             "content": "小叶说她的朋友举办展览", "reason": "把被谈论者错当作发言者"}, correct, run_id=12)
        self.assertEqual(revised["source_ids"], [correct["id"]])
        self.assertEqual(revised["subject"]["name"], "小叶的朋友")
        self.assertIsNone(revised["subject"].get("account_id"))
        self.assertEqual(revised["title"], "举办者")
        self.assertIsNone(revised["participant_id"])
        history = self.store.memory("semantic", owner, include_history=True)["history"]
        self.assertEqual(revised["reflections"][0]["id"], concern["id"])
        self.assertNotIn("reflections", history[0]["snapshot"]["memory"])
        self.assertEqual(history[0]["snapshot"]["memory"]["source_ids"], [old["id"]])
        self.assertEqual(history[0]["snapshot"]["memory"]["subject"]["account_id"], "alice")
        self.assertEqual((history[0]["reason"], history[0]["run_id"]), ("把被谈论者错当作发言者", 12))
        self.assertEqual(self.store.vector_rows("fixture"), [])
        self.assertFalse(self.store.save_embedding(stale, "fixture", [1, 0]))
        fresh = self.store.pending_embeddings("fixture")[0]
        self.assertIn("朋友", fresh["text"])
        self.assertTrue(self.store.save_embedding(fresh, "fixture", [0, 1]))
        self.assertEqual(len(self.store.vector_rows("fixture")), 1)

    def test_episode_replacement_recomputes_time_and_explicit_cues(self):
        old = self.message(1, "周一的对话")
        new = self.message(20, "另一段对话")
        episode = self.save({"kind": "episode", "title": "一次约定", "summary": "旧约定", "cues": ["旧提示"]}, old)
        revised = self.save({"kind": "episode", "id": episode["id"], "summary": "修正约定"}, new)
        self.assertEqual((revised["started_at"], revised["ended_at"]), (new["sent_at"], new["sent_at"]))
        self.assertEqual(revised["title"], "一次约定")
        self.assertEqual(revised["cues"], [{"cue": "旧提示", "aspect": ""}])
        revised = self.save({"kind": "episode", "id": episode["id"], "cues": [],
                             "started_at": new["sent_at"] - 3, "ended_at": new["sent_at"] + 3}, new)
        self.assertEqual(revised["summary"], "修正约定")
        self.assertEqual(revised["started_at"], new["sent_at"] - 3)
        self.assertEqual(revised["cues"], [])
        self.assertEqual(self.store.memory("episode", episode["id"], include_history=True)["history"][0]["snapshot"]["memory"]["cues"][0]["cue"], "旧提示")

    def test_graph_node_alias_replacement_and_history_update_all_connected_indexes(self):
        old = self.message(1, "小河的朋友去了展览")
        new = self.message(2, "展览参与者不是小河本人")
        edge = self.save({"kind": "association", "source": {"label": "小河", "aliases": ["河河"]},
                          "target": "展览", "relation": "去过", "statement": "朋友去了展览"}, old)
        node = edge["source_node_id"]
        concern = self.store.reflections.save({"content": "要核查小河与朋友的节点归属", "source_ids": [new["id"]],
                                              "memory_refs": [{"kind": "node", "id": node}]})
        self.assertEqual(self.store.memory("node", node)["reflections"][0]["id"], concern["id"])
        observed = next(row for row in self.store.graph(node_id=node) if row["id"] == edge["id"])
        self.assertEqual(observed["source_node"]["reflections"][0]["id"], concern["id"])
        other = self.save({"kind": "association", "source": {"node_id": node}, "target": "画廊",
                           "relation": "参观", "statement": "朋友也参观画廊"}, old)
        for record in (edge, other):
            self.store.put_vector("plastic_edge", str(record["id"]), "fixture", struct.pack("<2f", 1, 0), 2)
        revised = self.save({"kind": "association", "id": edge["id"],
            "source": {"node_id": node, "label": "小河的朋友", "description": "由小河谈论的朋友", "aliases": []},
            "reason": "节点把朋友误写成发言者"}, new)
        self.assertEqual(revised["source_node"]["aliases"], [])
        self.assertEqual(revised["source_ids"], [new["id"]])
        self.assertEqual(self.store.graph(terms=["河河"]), [])
        node_history = self.store.memory("node", node, include_history=True)["history"]
        self.assertEqual(node_history[0]["snapshot"]["memory"]["aliases"], ["河河"])
        self.assertNotIn("reflections", node_history[0]["snapshot"]["memory"])
        edge_history = self.store.memory("association", edge["id"], include_history=True)["history"]
        self.assertEqual(edge_history[0]["snapshot"]["memory"]["source"], "小河")
        self.assertNotIn("reflections", edge_history[0]["snapshot"]["memory"]["source_node"])
        self.assertEqual(revised["source_node"]["reflections"][0]["id"], concern["id"])
        self.assertEqual(self.store.memory("association", other["id"])["source"], "小河的朋友")
        self.assertEqual(self.store.vector_rows("fixture"), [])
        docs = [doc for doc in self.store.pending_embeddings("fixture") if doc["owner_type"] == "association"
                and int(doc["owner_key"]) in {edge["id"], other["id"]}]
        self.assertEqual(len(docs), 2)
        self.assertTrue(all("由小河谈论的朋友" in doc["text"] for doc in docs))
        self.store.reflections.save({"id": concern["id"], "status": "resolved"})
        self.assertNotIn("reflections", self.store.memory("node", node))
        self.assertNotIn("reflections", self.store.graph(node_id=node)[0]["source_node"])

    def test_withdrawal_excludes_every_memory_kind_and_keeps_explanation(self):
        source = self.message(1, "一段尚未查明的说法")
        entries = [self.save({"kind": "semantic", "content": "旧说法"}, source),
                   self.save({"kind": "episode", "summary": "旧经历"}, source),
                   self.save({"kind": "association", "source": "甲", "target": "乙", "relation": "认识", "statement": "旧关系"}, source)]
        topic = self.save({"kind": "topic", "title": "旧主题", "content": "旧总结", "connections": [
            {"kind": "episode", "id": entries[1]["id"], "relation": "来自这段经历", "purpose": "basis"}]}, source)["id"]
        entries.insert(0, {"kind": "topic", "id": topic})
        for entry in entries:
            result = self.save({**entry, "action": "withdraw", "reason": "原文不支持这个断言"})
            self.assertEqual(result["status"], "RETRACTED")
            self.assertIsNone(self.store.memory(entry["kind"], entry["id"]))
            previous = self.store.memory(entry["kind"], entry["id"], include_history=True)
            self.assertEqual(previous["history"][0]["snapshot"]["memory"]["source_ids"], [source["id"]])
            self.assertEqual(previous["history"][0]["operation"], "withdraw")
        withdrawn_refs = {(entry["kind"], entry["id"]) for entry in entries}
        self.assertFalse(withdrawn_refs & {(item["kind"], item["id"]) for item in self.store.search_memories()})
        # Withdrawing a claim does not silently withdraw unrelated endpoint objects.
        self.assertFalse(withdrawn_refs & {(item["owner_type"], int(item["owner_key"]))
                                          for item in self.store.pending_embeddings("fixture")})
        self.assertEqual(self.store.vector_rows("fixture"), [])
        self.store.close()
        self.store = Store(self.path, self.scope)
        self.assertEqual(self.store.memory("topic", topic, include_history=True)["status"], "RETRACTED")

    def test_source_selection_is_explicit_and_failed_batch_rolls_back_revision(self):
        source = self.message(1, "原文")
        saved = self.save({"kind": "semantic", "content": "旧内容"}, source)
        derived = self.store.save_memories([{"kind": "pattern", "content": "尚未指向具体材料的思考"}],
                                          [source["source_key"]], mark_processed=False)[0]
        self.assertEqual(derived["source_ids"], [])
        with self.assertRaisesRegex(ValueError, "unavailable"):
            self.store.save_memories([
                {"kind": "semantic", "id": saved["id"], "content": "应一起回滚", "source_keys": [source["source_key"]]},
                {"kind": "semantic", "content": "不存在的来源", "source_ids": [999]}], [source["source_key"]])
        current = self.store.memory("semantic", saved["id"], include_history=True)
        self.assertEqual((current["summary"], current["history"]), ("旧内容", []))
        self.assertEqual(self.store.save_memories([], [source["source_key"]], mark_processed=True), [])

    def test_forget_removes_linked_reflection_and_revision_copies_but_keeps_unrelated_withdrawal(self):
        forgotten = self.message(1, "要遗忘的原始经历", "alice")
        retained = self.message(2, "另一个人的独立经历", "bob")
        invalidated = self.save({"kind": "semantic", "content": "要遗忘的旧认识"}, forgotten)
        self.save({"kind": "semantic", "id": invalidated["id"], "content": "要遗忘的修订认识"}, forgotten)
        retargeted = self.save({"kind": "episode", "summary": "旧来源中的经历"}, forgotten)
        self.save({"kind": "episode", "id": retargeted["id"], "summary": "现在只涉及独立经历"}, retained)
        independent = self.save({"kind": "semantic", "content": "无关且后来撤回的推测"}, retained)
        self.save({"kind": "semantic", "id": independent["id"], "action": "withdraw", "reason": "没有形成结论"}, retained)
        by_source = self.store.reflections.save({"content": "原话留下的关注", "source_ids": [forgotten["id"]]})
        by_memory = self.store.reflections.save({"content": "旧认识留下的关注", "memory_refs": [
            {"kind": "semantic", "id": invalidated["id"]}]})
        by_request = self.store.reflections.save({"content": "旧请求留下的关注", "request_id": "1"})
        unrelated = self.store.reflections.save({"content": "独立经历待补充", "status": "waiting",
            "source_ids": [retained["id"]], "memory_refs": [{"kind": "semantic", "id": independent["id"]}]})
        run_id = self.store.record_run("foreground", 10, {"request_id": "1", "question": forgotten["plain_text"],
                                                        "background": "缓存过的原请求背景"})
        self.store.forget("alice")
        for anchor in ({"source_ids": [forgotten["id"]]}, {"request_id": "1"}):
            with self.assertRaisesRegex(ValueError, "explicitly forgotten"):
                self.store.reflections.save({"content": "在途模型晚到的旧经历", **anchor})
        self.assertIsNone(self.store.memory("semantic", invalidated["id"]))
        self.assertIsNone(self.store.memory("semantic", invalidated["id"], include_history=True))
        current = self.store.memory("episode", retargeted["id"], include_history=True)
        self.assertEqual(current["summary"], "现在只涉及独立经历")
        self.assertEqual(current["history"], [])
        for reflection in (by_source, by_memory, by_request):
            self.assertIsNone(self.store.reflections.get(reflection["id"]))
        self.assertEqual(self.store.reflections.due(), [])
        self.assertEqual(self.store.reflections.waiting()[0]["id"], unrelated["id"])
        self.assertEqual(self.store.reflections.associated("semantic", independent["id"])[0]["id"], unrelated["id"])
        self.assertEqual(self.store.reflections.interaction(run_id=run_id, detailed=True)["status"], "unavailable")
        withdrawn = self.store.memory("semantic", independent["id"], include_history=True)
        self.assertEqual(withdrawn["status"], "RETRACTED")
        self.assertEqual(withdrawn["history"][0]["snapshot"]["memory"]["summary"], "无关且后来撤回的推测")

    def test_existing_topic_schema_gets_withdrawal_status_without_losing_data(self):
        self.store.close()
        legacy = self.temp / "legacy.db"
        with sqlite3.connect(legacy) as db:
            db.execute("CREATE TABLE topics(id INTEGER PRIMARY KEY,umo TEXT,name TEXT,summary TEXT,extractor_version TEXT DEFAULT '')")
            db.execute("INSERT INTO topics VALUES(1,?,'主题','保留正文','legacy')", (self.scope,))
        db.close()
        self.store = Store(legacy, self.scope)
        self.assertEqual(self.store.memory("topic", 1)["summary"], "保留正文")
        self.save({"kind": "topic", "id": 1, "action": "withdraw", "reason": "这个主题没有可用来源"})
        self.assertIsNone(self.store.memory("topic", 1))
