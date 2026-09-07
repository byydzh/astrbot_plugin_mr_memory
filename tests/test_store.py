import hashlib
import json
import sqlite3
import struct
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from unittest.mock import patch

from mr_memory.store import Store


class StoreContractTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / ".pytest_cache"
        scratch.mkdir(exist_ok=True)
        self.scratch = scratch / ("store-" + uuid.uuid4().hex)
        self.scratch.mkdir()
        self.path = self.scratch / "group.db"
        self.scope = "test:GroupMessage:42"
        self.store = Store(self.path, self.scope)

    def tearDown(self):
        self.store.close()
        for file in self.scratch.iterdir():
            file.unlink()
        self.scratch.rmdir()

    def message(self, number, text, sender="10", role="USER", **extra):
        row = dict(platform="test", platform_id="test", umo=self.scope,
            group_id="42", message_id=str(number), sender_id=sender, sender_name="Tester " + sender,
            sent_at=1_800_000_000 + number, plain_text=text, content=[{"type": "text", "text": text}],
            role=role)
        row.update(extra)
        return self.store.append_message(row)

    def test_archive_keeps_bot_tool_reply_and_continuous_context(self):
        question = self.message(1, "请查青桥")
        bot = self.message(2, "正在检索", sender="99", role="BOT", reply_to="1")
        tool = self.message(3, "", sender="99", role="SYSTEM", reply_to=question["source_key"],
                            content=[{"type": "tool_result", "name": "search", "result": "南门周六开放"}])
        self.assertEqual(bot["reply_to"], question["source_key"])
        self.assertEqual([m["role"] for m in self.store.context(bot["source_key"])], ["USER", "BOT", "SYSTEM"])
        self.assertEqual(len(self.store.context(bot["source_key"], before_time=tool["sent_at"])), 3)
        self.assertEqual([m["id"] for m in self.store.context(message_id=bot["id"], before_time=bot["sent_at"])],
                         [question["id"], bot["id"]])
        self.assertEqual(self.store.context(message_id=tool["id"], before_time=bot["sent_at"]), [])
        self.assertEqual(self.store.search_messages(["青桥"])[0]["id"], question["id"])
        self.assertEqual(self.store.search_messages(["南门"])[0]["id"], tool["id"])
        self.assertEqual(self.store.search_messages(['" OR 1=1 --']), [])
        revised = self.message(1, "请查青桥新站")
        self.assertEqual(revised["id"], question["id"])
        revisions = self.store.db.execute("SELECT plain_text FROM message_revisions WHERE message_id=? ORDER BY revision_no", (question["id"],)).fetchall()
        self.assertEqual([r[0] for r in revisions], ["请查青桥", "请查青桥新站"])
        self.assertEqual(self.store.delete_message("2"), 1)
        self.assertEqual([m["id"] for m in self.store.recent()], [question["id"], tool["id"]])
        self.message(4, "早期系统事件", sender="88", role="SYSTEM")
        generated = self.message(5, "生成中", sender="88", role="SYSTEM",
                                 content=[{"type": "bot_event", "event": "generated"}])
        self.assertEqual(generated["role"], "SYSTEM")
        self.assertEqual(self.store.members(account_ids=["88"])[0]["account_type"], "BOT")
        self.message(6, "已发送", sender="88", role="BOT")
        self.message(7, "提及机器人", content=[{"type": "mention", "account_id": "88", "display_name": "群助手"}])
        member = self.store.members(account_ids=["88"])[0]
        self.assertEqual(member["account_type"], "BOT")
        self.assertIn("群助手", member["aliases"])

    def test_connection_serializes_reader_and_separate_transaction(self):
        entered, release, second_started, second_done = Event(), Event(), Event(), Event()

        def fail_before_commit(*args):
            entered.set()
            if not release.wait(2):
                raise TimeoutError("Test did not release the transaction")
            raise RuntimeError("rollback this message")

        def separate_write_and_read():
            second_started.set()
            self.store.save_working_state({"independent": True})
            rows = self.store.recent()
            second_done.set()
            return rows

        with ThreadPoolExecutor(max_workers=2) as pool, patch.object(self.store, "_revision", fail_before_commit):
            first = pool.submit(self.message, 1, "不可被另一事务提交的消息")
            self.assertTrue(entered.wait(1))
            second = pool.submit(separate_write_and_read)
            try:
                self.assertTrue(second_started.wait(1))
                self.assertFalse(second_done.wait(0.05))
            finally:
                release.set()
            with self.assertRaisesRegex(RuntimeError, "rollback"):
                first.result(timeout=1)
            self.assertEqual(second.result(timeout=1), [])
        self.assertEqual(self.store.load_working_state(), {"independent": True})

    def test_existing_database_scope_and_forgetting_are_preserved(self):
        old = self.message(1, "保留的原文")
        self.store.db.execute("CREATE TABLE old_extra_state(value TEXT)")
        self.store.db.execute("INSERT INTO old_extra_state VALUES('unchanged')")
        self.store.db.commit()
        self.store.close()
        self.store = Store(self.path, self.scope)
        self.assertEqual(self.store.recent()[0]["plain_text"], "保留的原文")
        self.assertEqual(self.store.db.execute("SELECT value FROM old_extra_state").fetchone()[0], "unchanged")
        with self.assertRaisesRegex(ValueError, "another group"):
            Store(self.path, "test:GroupMessage:43")
        self.store.forget("10")
        self.assertEqual(self.message(2, "不能重新导入"), {"ignored": "forgotten_account"})
        self.assertEqual(self.store.members(account_ids=["10"]), [])
        self.assertEqual(self.store.recent(), [])
        digest = hashlib.sha256("\x1f".join((self.scope, "test", "10")).encode()).hexdigest()
        self.assertTrue(self.store.db.execute("SELECT 1 FROM forgotten_accounts WHERE account_hash=?", (digest,)).fetchone())
        self.assertEqual(self.store.db.execute("SELECT plain_text FROM message_revisions WHERE message_id=?", (old["id"],)).fetchone()[0], "")

    def test_learning_sources_graph_and_durable_index_work(self):
        first = self.message(1, "青桥展览周六开幕")
        second = self.message(2, "集合在东门")
        items = [
            {"kind": "episode", "title": "出行讨论", "summary": "讨论展览和集合位置", "cues": ["青桥"]},
            {"kind": "semantic", "content": "展览在周六开幕", "aspect": "日程", "participant_id": first["participant_id"]},
            {"kind": "association", "source": "展览", "target": "东门", "relation": "集合地点", "statement": "去展览时在东门集合"},
        ]
        saved = self.store.save_memories(items, [first["source_key"], second["source_key"]])
        self.assertEqual([m["kind"] for m in saved], ["episode", "semantic", "association"])
        self.assertEqual([m["source_ids"] for m in saved], [[first["id"], second["id"]]] * 3)
        self.assertEqual(saved[0]["source_speakers"], [{"participant_id": first["participant_id"],
                         "account_id": first["sender_id"], "name_at_message": first["sender_name"], "role": "USER"}])
        self.assertEqual(len(self.store.graph(terms=["东门"])), 1)
        self.assertEqual(self.store.memory("semantic", str(saved[1]["id"]))["source_keys"], [first["source_key"], second["source_key"]])
        self.assertEqual(self.store.db.execute("SELECT cue FROM episode_keywords").fetchone()[0], "青桥")
        self.assertEqual(self.store.pending_messages(20), [])
        self.store.db.execute("UPDATE message_processing SET status='FAILED' WHERE message_id=?", (first["id"],))
        self.store.db.commit()
        self.assertEqual(self.store.pending_messages(20)[0]["id"], first["id"])
        docs = self.store.pending_embeddings("fixture-model", 10)
        self.assertEqual(len(docs), 3)
        self.store.close()
        self.store = Store(self.path, self.scope)
        self.assertEqual(len(self.store.pending_embeddings("fixture-model", 10)), 3)
        for doc in docs:
            self.assertTrue(self.store.save_embedding(doc, "fixture-model", [1.0, 0.0]))
        self.assertEqual(self.store.pending_embeddings("fixture-model", 10), [])
        vectors = self.store.vector_rows("fixture-model")
        self.assertEqual(len(vectors), 3)
        self.assertEqual(struct.unpack("<2f", vectors[0]["vector"]), (1.0, 0.0))
        self.assertEqual(self.store.vector_rows("another-model"), [])
        self.store.delete_message("1")
        self.assertEqual(self.store.graph(), [])
        self.assertEqual(self.store.vector_rows("fixture-model"), [])

    def test_author_filter_and_memory_sources_keep_quote_authors(self):
        first = self.message(1, "我以前住五人寝", sender_name="小桥")
        second = self.message(2, "我现在住两人寝", sender="20", sender_name="小桥", reply_to="1",
            content=[{"type": "reply", "sender_id": "10", "sender_name": "小桥", "plain_text": first["plain_text"]},
                     {"type": "text", "text": "我现在住两人寝"}])
        bot = self.message(3, "你们说的是不同时间", sender="99", role="BOT",
            content=[{"type": "mention", "account_id": "10", "display_name": "小桥"},
                     {"type": "text", "text": "你们说的是不同时间"}])
        self.assertEqual([m["id"] for m in self.store.search_messages(sender_id="10")], [first["id"]])
        self.assertEqual([m["id"] for m in self.store.search_messages(related_account_id="10")],
                         [bot["id"], second["id"], first["id"]])
        self.assertEqual(self.store.activity(account_id="10", start_at=first["sent_at"], end_at=bot["sent_at"] + 1)["message_count"], 1)
        saved = self.store.save_memories([
            {"kind": "episode", "summary": "p7说以前五人寝，p14说现在两人寝"},
            {"kind": "semantic", "content": "以前住五人寝", "participant_id": first["participant_id"]},
        ], [first["source_key"], second["source_key"], bot["source_key"]])
        self.assertEqual(saved[1]["subject"], {"name": "小桥", "account_id": "10"})
        self.assertEqual(saved[0]["summary"], "p7说以前五人寝，p14说现在两人寝")
        sources = self.store.memory_sources("episode", saved[0]["id"])
        self.assertEqual(sources, saved[0]["sources"])
        self.assertEqual([m["sender_id"] for m in sources], ["10", "20", "99"])
        self.assertEqual(sources[1]["content"][0]["sender_id"], "10")
        self.assertEqual(saved[0]["source_speakers"][-1]["role"], "BOT")

    def test_memory_account_candidates_use_subject_and_actual_source_relationships(self):
        first = self.message(1, "我以前去过这个展览", sender_name="小桥")
        other = self.message(2, "我还没去过", sender="20", sender_name="小桥")
        reply = self.message(3, "你问的是展览安排", sender="30", reply_to="1")
        mention = self.message(4, "问问小桥", sender="40",
            content=[{"type": "mention", "account_id": "10", "display_name": "小桥"}])
        old, unrelated = self.store.save_memories([
            {"kind": "semantic", "content": "p9是去过展览的那位小桥", "participant_id": first["participant_id"]},
            {"kind": "semantic", "content": "另一位小桥还没去", "participant_id": other["participant_id"]},
        ], [other["source_key"]], mark_processed=False)
        natural = self.store.save_memories([
            {"kind": "semantic", "person": "小桥", "content": "小桥回忆曾去展览"}
        ], [first["source_key"]], mark_processed=False)[0]
        episode = self.store.save_memories([
            {"kind": "episode", "summary": "群友接着小桥的问题讨论展览"}
        ], [reply["source_key"]], mark_processed=False)[0]
        association = self.store.save_memories([
            {"kind": "association", "source": "展览", "target": "讨论", "relation": "话题", "statement": "有人提及小桥讨论展览"}
        ], [mention["source_key"]], mark_processed=False)[0]
        with self.store.db:
            topic_id = self.store.db.execute("INSERT INTO topics(umo,name,summary) VALUES(?,?,?)",
                (self.scope, "展览话题", "有关展览的交流")).lastrowid
            self.store.db.execute("INSERT INTO topic_episodes(topic_id,episode_id) VALUES(?,?)", (topic_id, episode["id"]))
        candidates = self.store.search_memories(related_account_id="10", limit=20)
        addresses = {(item["kind"], item["id"]) for item in candidates}
        self.assertEqual(addresses, {("semantic", old["id"]), ("semantic", natural["id"]), ("episode", episode["id"]), ("topic", topic_id)})
        self.assertNotIn(("semantic", unrelated["id"]), addresses)
        old_result = next(item for item in candidates if item["kind"] == "semantic" and item["id"] == old["id"])
        self.assertEqual(old_result["subject"]["account_id"], "10")
        self.assertNotIn("sources", old_result)
        self.assertEqual(self.store.memory("semantic", old["id"])["sources"][0]["sender_id"], "20")
        self.assertIsNone(natural["participant_id"])
        self.assertEqual([m["id"] for m in self.store.search_memories(kind="association", related_account_id="10")], [association["id"]])
        self.assertEqual(self.store.search_memories(related_account_id="unknown"), [])

    def test_quotes_use_original_record_time_and_author_without_rewriting_archive(self):
        original = self.message(1, "原来的答复", sender="99", sender_name="群助手", role="BOT")
        wrapper_time = original["sent_at"] + 600
        reply = self.message(2, "这是引用", content=[
            {"type": "reply", "message_id": "1", "sent_at": wrapper_time,
             "sender_id": "wrong", "sender_name": "错误作者", "role": "USER", "plain_text": "原来的答复"},
            {"type": "reply", "source_key": original["source_key"], "time": wrapper_time, "plain_text": "原来的答复"},
            {"type": "reply", "message_id": "missing", "sent_at": wrapper_time, "time": wrapper_time,
             "sender_name": "引用中保留的名字", "plain_text": "找不到原记录的引用"},
        ])
        for quote in reply["content"][:2]:
            self.assertEqual(quote["sent_at"], original["sent_at"])
            self.assertEqual((quote["sender_id"], quote["sender_name"], quote["role"]), ("99", "群助手", "BOT"))
            self.assertEqual(quote["time_status"], "original_message")
        unknown = reply["content"][2]
        self.assertNotIn("sent_at", unknown)
        self.assertNotIn("time", unknown)
        self.assertEqual(unknown["time_status"], "original_time_unknown")
        self.assertEqual(unknown["plain_text"], "找不到原记录的引用")
        archived = json.loads(self.store.db.execute("SELECT content_json FROM messages WHERE id=?", (reply["id"],)).fetchone()[0])
        self.assertEqual(archived[0]["sent_at"], wrapper_time)
        self.assertEqual(archived[0]["sender_id"], "wrong")

    def test_graph_reuses_only_explicit_nodes_and_revises_connections(self):
        first = self.message(1, "两个角色碰巧都叫小桥")
        second = self.message(2, "刚才说的是另一位小桥")
        saved = self.store.save_memories([
            {"kind": "association", "source": {"label": "小桥", "description": "作品甲角色", "aliases": ["桥桥"]},
             "target": {"label": "排练室"}, "relation": "去过", "statement": "作品甲的小桥去过排练室"},
            {"kind": "association", "source": {"label": "小桥", "description": "作品乙角色"},
             "target": {"label": "排练室"}, "relation": "去过", "statement": "作品乙的小桥也去过排练室"},
        ], [first["source_key"]], mark_processed=False)
        self.assertNotEqual(saved[0]["source_node_id"], saved[1]["source_node_id"])
        self.assertNotEqual(saved[0]["target_node_id"], saved[1]["target_node_id"])
        self.assertEqual([m["id"] for m in self.store.graph(terms=["桥桥"])], [saved[0]["id"]])
        reused = self.store.save_memories([
            {"kind": "association", "source": {"node_id": saved[0]["source_node_id"], "label": "小桥甲", "aliases": ["桥某"]},
             "target": {"node_id": saved[0]["target_node_id"]}, "relation": "下次要去", "statement": "桥桥下次还想去排练室"}
        ], [second["source_key"]], mark_processed=False)[0]
        self.assertEqual(reused["source_node_id"], saved[0]["source_node_id"])
        self.assertEqual(self.store.memory("association", saved[0]["id"])["source"], "小桥甲")
        self.assertIn("桥某", reused["source_node"]["aliases"])
        revised = self.store.save_memories([
            {"kind": "association", "id": saved[0]["id"], "source": {"node_id": saved[1]["source_node_id"]},
             "statement": "更正：那次去排练室的是作品乙的小桥"}
        ], [second["source_key"]], mark_processed=False)[0]
        self.assertEqual(revised["id"], saved[0]["id"])
        self.assertEqual(revised["source_node_id"], saved[1]["source_node_id"])
        self.assertEqual(revised["target_node_id"], saved[0]["target_node_id"])
        self.assertEqual(revised["source_ids"], [first["id"], second["id"]])
        doc = next(d for d in self.store.pending_embeddings("fixture", 10) if d["owner_key"] == str(revised["id"]))
        self.assertIn("更正", doc["text"])
        self.assertEqual(self.store.memory("association", reused["id"])["source_node"]["description"], "作品甲角色")

    def test_model_revises_existing_memory_and_preserves_all_sources(self):
        first = self.message(1, "机器人说这里周六开门", sender="99", role="BOT")
        correction = self.message(2, "我是说周日才开门")
        saved = self.store.save_memories([
            {"kind": "episode", "title": "开门讨论", "summary": "机器人说周六开门"},
            {"kind": "semantic", "person": "小桥", "subject": "小桥", "content": "小桥周六去"},
        ], [first["source_key"]], mark_processed=False)
        revised = self.store.save_memories([
            {"kind": "episode", "id": saved[0]["id"], "summary": "机器人曾说周六，群友更正周日才开门"},
            {"kind": "semantic", "id": saved[1]["id"], "person": "桥桥", "content": "桥桥解释周日才开门，先前周六是机器人的误解"},
        ], [correction["source_key"]], mark_processed=False)
        self.assertEqual([m["id"] for m in revised], [m["id"] for m in saved])
        self.assertEqual(revised[0]["title"], "开门讨论")
        self.assertEqual(revised[1]["subject"], {"name": "桥桥", "account_id": None})
        self.assertEqual([m["source_ids"] for m in revised], [[first["id"], correction["id"]]] * 2)
        self.assertEqual(len(self.store.pending_messages(10)), 2)
        self.assertEqual(self.store.save_memories([], [first["source_key"], correction["source_key"]]), [])
        self.assertEqual(self.store.pending_messages(10), [])

    def test_existing_topic_revision_preserves_sources_and_vector_owner(self):
        first = self.message(1, "小桥说展览周六开放")
        correction = self.message(2, "我说的是周日开放")
        episode = self.store.save_memories([{"kind": "episode", "summary": "小桥解释展览日程"}], [first["source_key"]])[0]
        with self.store.db:
            topic_id = self.store.db.execute("INSERT INTO topics(umo,name,summary) VALUES(?,?,?)",
                (self.scope, "旧日程", "p3说展览周六开放")).lastrowid
            self.store.db.execute("INSERT INTO topic_episodes(topic_id,episode_id) VALUES(?,?)", (topic_id, episode["id"]))
        self.store.put_vector("topic", str(topic_id), "fixture", struct.pack("<2f", 1.0, 0.0), 2)
        revised = self.store.save_memories([
            {"kind": "topic", "id": topic_id, "title": "展览日程", "summary": "小桥说展览周日开放"}
        ], [correction["source_key"]], mark_processed=False)[0]
        self.assertEqual((revised["id"], revised["title"]), (topic_id, "展览日程"))
        self.assertEqual(revised["source_ids"], [first["id"], correction["id"]])
        self.assertEqual(revised["sources"][0]["plain_text"], first["plain_text"])
        self.assertEqual(self.store.memory("topic", topic_id)["sources"][1]["plain_text"], correction["plain_text"])
        doc = next(d for d in self.store.pending_embeddings("fixture", 10) if d["owner_type"] == "topic")
        self.assertEqual(doc["owner_key"], str(topic_id))
        self.assertIn("小桥说", doc["text"])
        self.assertTrue(self.store.save_embedding(doc, "fixture", [0.0, 1.0]))
        vector = self.store.db.execute("SELECT owner_key,vector FROM memory_embeddings WHERE owner_type='topic'").fetchone()
        self.assertEqual(vector["owner_key"], str(topic_id))
        self.assertEqual(struct.unpack("<2f", vector["vector"]), (0.0, 1.0))
        with self.assertRaisesRegex(ValueError, "existing id"):
            self.store.save_memories([{"kind": "topic", "summary": "不新建topic"}], [first["source_key"]])
        self.store.delete_message(correction["source_key"])
        self.assertIsNone(self.store.memory("topic", topic_id))
        self.assertEqual(self.store.search_memories(kind="topic"), [])
        self.assertFalse(any(row["owner_type"] == "topic" for row in self.store.vector_rows("fixture")))
        self.assertFalse(any(row["owner_type"] == "topic" for row in self.store.pending_embeddings("fixture", 10)))

    def test_transaction_rollback_activity_and_working_state(self):
        first = self.message(1, "测试发言")
        second = self.message(2, "第二条")
        third = self.message(3, "其他人", sender="20")
        with self.assertRaises(ValueError):
            self.store.save_memories([
                {"kind": "semantic", "content": "这次会回滚"},
                {"kind": "association", "content": "缺少图端点"},
            ], [first["source_key"]])
        self.assertEqual(self.store.search_memories(), [])
        self.assertEqual(len(self.store.pending_messages(10)), 3)
        self.assertEqual([m["id"] for m in self.store.pending_messages(2)], [second["id"], third["id"]])
        self.assertEqual([m["id"] for m in self.store.pending_messages(2, newest=False)], [first["id"], second["id"]])
        activity = self.store.activity(first["participant_id"], first["sent_at"], second["sent_at"] + 1)
        self.assertEqual(activity["message_count"], 2)
        self.assertEqual(sum(h["message_count"] for h in activity["hour_counts"]), 2)
        self.assertEqual(activity["last_at"], second["sent_at"])
        self.store.save_working_state({"note": "用户要求简洁", "consolidated_through": 2})
        self.assertEqual(self.store.load_working_state()["consolidated_through"], 2)
        self.store.cache_roster([{"user_id": 10, "role": "owner"}], 123)
        self.assertEqual(self.store.roster()["members"][0]["role"], "owner")


if __name__ == "__main__":
    unittest.main()
