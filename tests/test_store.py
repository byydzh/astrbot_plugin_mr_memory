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
                         "account_id": first["sender_id"], "name_at_message": first["sender_name"]}])
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
