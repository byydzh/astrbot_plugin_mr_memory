import os
import hashlib
import json
import struct
import unittest
import uuid
from pathlib import Path

from mr_memory.store import Store


class StoreContractTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(os.environ['ASTRBOT_ROOT'])
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

    def remember(self, items, sources, **kwargs):
        # These fixtures intentionally assign their listed messages to each item.
        return self.store.save_memories([{**item, "source_keys": list(sources)} for item in items], sources, **kwargs)

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

    def test_media_bytes_are_archived_but_not_searched_or_sent_as_prose(self):
        from mr_memory.agent import _model_view

        encoded = "base64://" + "AAAAtestneedleBBBB" * 20000
        inline = "data:image/png;base64," + "BBBBtestneedleAAAA" * 20000
        content = [{"type": "image", "file": encoded, "url": "https://example.test/diagram.png"},
                   {"type": "tool_result", "result": json.dumps({"image": inline, "text": "海风图示"}, ensure_ascii=False)}]
        picture = self.message(1, "[image]", role="BOT", content=content)
        original = self.store.db.execute("SELECT content_json FROM messages WHERE id=?", (picture["id"],)).fetchone()[0]
        self.assertEqual(json.loads(original), content)
        self.assertEqual(self.store.search_messages(["testneedle"]), [])
        self.assertEqual(self.store.search_messages(["海风"])[0]["id"], picture["id"])
        self.assertIn("https://example.test/diagram.png", json.dumps(picture))
        self.assertLess(len(json.dumps(picture)), 2500)
        self.assertEqual(self.store.messages([picture["id"], 999999, picture["id"]]), [picture])
        self.assertNotIn("testneedle", json.dumps(_model_view({"current": {"content": content}})))
        # The same word remains searchable when someone actually writes it.
        user = self.message(2, "请查 testneedle 的出处")
        self.assertEqual([row["id"] for row in self.store.search_messages(["testneedle"])], [user["id"]])

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
        saved = self.remember(items, [first["source_key"], second["source_key"]])
        self.assertEqual([m["kind"] for m in saved], ["episode", "semantic", "association"])
        self.assertEqual([m["source_ids"] for m in saved], [[first["id"], second["id"]]] * 3)
        self.assertEqual(saved[0]["source_speakers"], [{"participant_id": first["participant_id"],
                         "account_id": first["sender_id"], "name_at_message": first["sender_name"], "role": "USER"}])
        self.assertEqual(len(self.store.graph(terms=["东门"])), 1)
        self.assertEqual(self.store.memory("semantic", str(saved[1]["id"]))["source_keys"], [first["source_key"], second["source_key"]])
        self.assertEqual(self.store.navigate(cue="青桥", aspect="")["items"][0]["id"], saved[0]["id"])
        self.assertEqual(self.store.pending_messages(20), [])
        self.store.db.execute("UPDATE message_processing SET status='FAILED' WHERE message_id=?", (first["id"],))
        self.store.db.commit()
        self.assertEqual(self.store.pending_messages(20)[0]["id"], first["id"])
        docs = self.store.pending_embeddings("fixture-model", 10)
        self.assertEqual(len(docs), 7)  # Three memories, two graph endpoints, and two original messages.
        self.store.close()
        self.store = Store(self.path, self.scope)
        self.assertEqual(len(self.store.pending_embeddings("fixture-model", 10)), 7)
        for doc in docs:
            self.assertTrue(self.store.save_embedding(doc, "fixture-model", [1.0, 0.0]))
        self.assertEqual(self.store.pending_embeddings("fixture-model", 10), [])
        vectors = self.store.vector_rows("fixture-model")
        self.assertEqual(len(vectors), 7)
        self.assertEqual(struct.unpack("<2f", vectors[0]["vector"]), (1.0, 0.0))
        self.assertEqual(self.store.vector_rows("another-model"), [])
        self.store.delete_message("1")
        self.assertEqual(self.store.graph(), [])
        self.assertEqual([(row["owner_type"], row["owner_key"]) for row in self.store.vector_rows("fixture-model")],
                         [("message", str(second["id"]))])

    def test_memory_account_candidates_use_subject_and_actual_source_relationships(self):
        first = self.message(1, "我以前去过这个展览", sender_name="小桥")
        other = self.message(2, "我还没去过", sender="20", sender_name="小桥")
        reply = self.message(3, "你问的是展览安排", sender="30", reply_to="1")
        mention = self.message(4, "问问小桥", sender="40",
            content=[{"type": "mention", "account_id": "10", "display_name": "小桥"}])
        old, unrelated = self.remember([
            {"kind": "semantic", "content": "p9是去过展览的那位小桥", "subject": {"name": "小桥", "account_id": "10"}},
            {"kind": "semantic", "content": "另一位小桥还没去", "subject": {"name": "小桥", "account_id": "20"}},
        ], [other["source_key"]], mark_processed=False)
        natural = self.remember([
            {"kind": "semantic", "person": "小桥", "content": "小桥回忆曾去展览"}
        ], [first["source_key"]], mark_processed=False)[0]
        episode = self.remember([
            {"kind": "episode", "summary": "群友接着小桥的问题讨论展览"}
        ], [reply["source_key"]], mark_processed=False)[0]
        association = self.remember([
            {"kind": "association", "source": "展览", "target": "讨论", "relation": "话题", "statement": "有人提及小桥讨论展览"}
        ], [mention["source_key"]], mark_processed=False)[0]
        topic_id = self.store.save_memories([{"kind": "topic", "title": "展览话题", "content": "有关展览的交流",
            "connections": [{"kind": "episode", "id": episode["id"], "relation": "包含经历", "purpose": "basis"}]}], mark_processed=False)[0]["id"]
        candidates = self.store.search_memories(related_account_id="10", limit=20)
        addresses = {(item["kind"], item["id"]) for item in candidates}
        self.assertTrue({("semantic", old["id"]), ("semantic", natural["id"]), ("episode", episode["id"]), ("topic", topic_id), ("association", association["id"])} <= addresses)
        self.assertNotIn(("semantic", unrelated["id"]), addresses)
        old_result = next(item for item in candidates if item["kind"] == "semantic" and item["id"] == old["id"])
        self.assertEqual(old_result["subject"]["account_id"], "10")
        self.assertNotIn("sources", old_result)
        self.assertEqual(self.store.memory("semantic", old["id"])["sources"][0]["sender_id"], "20")
        self.assertIsNone(natural["participant_id"])
        self.assertEqual([m["id"] for m in self.store.search_memories(kind="association", related_account_id="10")], [association["id"]])
        self.assertEqual(self.store.search_memories(related_account_id="unknown"), [])

    def test_model_revision_replaces_sources_and_keeps_history(self):
        first = self.message(1, "机器人说这里周六开门", sender="99", role="BOT")
        correction = self.message(2, "我是说周日才开门")
        saved = self.remember([
            {"kind": "episode", "title": "开门讨论", "summary": "机器人说周六开门"},
            {"kind": "semantic", "person": "小桥", "subject": "小桥", "content": "小桥周六去"},
        ], [first["source_key"]], mark_processed=False)
        revised = self.remember([
            {"kind": "episode", "id": saved[0]["id"], "summary": "机器人曾说周六，群友更正周日才开门"},
            {"kind": "semantic", "id": saved[1]["id"], "person": "桥桥", "content": "桥桥解释周日才开门，先前周六是机器人的误解"},
        ], [correction["source_key"]], mark_processed=False)
        self.assertEqual([m["id"] for m in revised], [m["id"] for m in saved])
        self.assertEqual(revised[0]["title"], "开门讨论")
        self.assertEqual(revised[1]["subject"], {"name": "桥桥", "account_id": None})
        self.assertEqual([m["source_ids"] for m in revised], [[correction["id"]]] * 2)
        self.assertEqual(self.store.memory("semantic", revised[1]["id"], include_history=True)["history"][0]["snapshot"]["memory"]["source_ids"], [first["id"]])
        self.assertEqual(len(self.store.pending_messages(10)), 2)
        self.assertEqual(self.remember([], [first["source_key"], correction["source_key"]]), [])
        self.assertEqual(self.store.pending_messages(10), [])

    def test_transaction_rollback_activity_and_working_state(self):
        first = self.message(1, "测试发言")
        second = self.message(2, "第二条")
        third = self.message(3, "其他人", sender="20")
        with self.assertRaises(ValueError):
            self.remember([
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
