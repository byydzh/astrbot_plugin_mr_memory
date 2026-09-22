import asyncio
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from mr_memory.history import CONNECT_EVENT, HistoryRecovery, history_message
from mr_memory.store import Store


class HistoryRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = Path(__file__).resolve().parents[1] / ".pytest_cache" / ("history-" + uuid.uuid4().hex)
        self.directory.mkdir(parents=True)
        self.store = Store(self.directory / "group.db", "test:GroupMessage:42")
        self.bot = SimpleNamespace(get_login_info=AsyncMock(return_value={"user_id": 9000}),
                                   get_group_msg_history=AsyncMock(), subscribe=Mock(), unsubscribe=Mock())
        adapter = SimpleNamespace(meta=lambda: SimpleNamespace(name="aiocqhttp"), get_client=lambda: self.bot)
        self.plugin = SimpleNamespace(config={"capture_enabled": True, "history_recovery_enabled": True,
            "allowed_umos": [], "history_recovery_initial_hours": 24}, stores={self.store.umo: self.store},
            context=SimpleNamespace(get_platform_inst=lambda _: adapter))
        self.recovery = HistoryRecovery(self.plugin)
        self.store.save_history_state({"watermark": 110})

    async def asyncTearDown(self):
        self.recovery.close()
        self.store.close()
        for file in self.directory.iterdir():
            file.unlink()
        self.directory.rmdir()

    def raw(self, number, timestamp, user=1001, parts=None):
        return {"group_id": 42, "message_id": number, "time": timestamp,
                "sender": {"user_id": user, "nickname": "Sample", "card": "Card"},
                "message": parts or [{"type": "text", "data": {"text": f"/chat sample {number}"}}]}

    def archive(self, raw):
        return self.store.append_message(history_message(raw, self.store.umo, "9000"))

    async def test_pages_fill_gap_without_replacing_live_messages_or_replaying_commands(self):
        original = self.archive(self.raw(30, 300))
        self.archive(self.raw(50, 500))  # New live traffic cannot advance the recovery cursor.
        self.bot.get_group_msg_history.side_effect = [
            {"messages": [self.raw(30, 300), self.raw(20, 200, 9000)]},
            {"messages": [self.raw(20, 200, 9000), self.raw(10, 100)]},
        ]
        result = await self.recovery.recover(self.store, self.bot)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["total_inserted"], 1)
        self.assertEqual(result["watermark"], 300)
        bot = self.store.db.execute("SELECT role,sender_id FROM messages WHERE message_id='20'").fetchone()
        self.assertEqual(tuple(bot), ("BOT", "9000"))
        self.assertEqual(self.store.db.execute("SELECT revision_no FROM messages WHERE id=?", (original["id"],)).fetchone()[0], 1)
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM mr_runs").fetchone()[0], 0)
        self.assertEqual([c.kwargs["message_seq"] for c in self.bot.get_group_msg_history.await_args_list], ["0", "20"])

    async def test_failure_keeps_cursor_and_resumes_after_reload(self):
        self.bot.get_group_msg_history.side_effect = [
            {"messages": [self.raw(30, 300), self.raw(20, 200)]}, ConnectionError("interrupted")]
        failed = await self.recovery.recover(self.store, self.bot)
        self.assertEqual(failed["status"], "error")
        self.assertEqual(failed["watermark"], 110)
        self.assertEqual(failed["pending"]["cursor"], "20")
        self.archive(self.raw(50, 500))
        self.bot.get_group_msg_history.side_effect = [{"messages": [self.raw(20, 200), self.raw(10, 100)]}]
        restored = HistoryRecovery(self.plugin)
        completed = await restored.recover(self.store, self.bot)
        self.assertEqual(self.bot.get_group_msg_history.await_args.kwargs["message_seq"], "20")
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["total_inserted"], 2)
        self.assertEqual(completed["watermark"], 300)
        restored.close()

    async def test_platform_boundary_is_visible_instead_of_claiming_complete(self):
        self.bot.get_group_msg_history.side_effect = [
            {"messages": [self.raw(30, 300), self.raw(20, 200)]}, {"messages": [self.raw(20, 200)]}]
        result = await self.recovery.recover(self.store, self.bot)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["unavailable"]["until"], 200)
        self.assertEqual(result["total_inserted"], 2)

    async def test_reconnect_rebuilds_native_cursor_without_forgetting_unfilled_range(self):
        self.store.save_history_state({"watermark": 110, "pending": {
            "since": 109, "cursor": "20", "head_at": 300, "pages": 1,
            "inserted": 0, "existing": 0, "ignored": 0}})
        self.bot.get_group_msg_history.side_effect = [
            {"messages": [self.raw(50, 500), self.raw(30, 300)]},
            {"messages": [self.raw(30, 300), self.raw(10, 100)]}]
        result = await self.recovery.recover(self.store, self.bot, rewind=True)
        self.assertEqual([c.kwargs["message_seq"] for c in self.bot.get_group_msg_history.await_args_list], ["0", "30"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["last_result"]["since"], 109)
        self.assertEqual(result["watermark"], 500)

    async def test_quiet_group_does_not_move_checkpoint_back_to_old_history(self):
        self.bot.get_group_msg_history.return_value = {"messages": [self.raw(10, 100)]}
        result = await self.recovery.recover(self.store, self.bot)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["watermark"], 110)
        self.assertEqual(result["total_inserted"], 0)

    async def test_reconnect_subscription_is_removed_on_unload(self):
        self.recovery.request(self.store)
        self.recovery.request(self.store)
        self.bot.subscribe.assert_called_once()
        name, handler = self.bot.subscribe.call_args.args
        self.assertEqual(name, CONNECT_EVENT)
        self.recovery.requested.clear()
        await handler({"self_id": 9000})
        self.assertIn(self.store.umo, self.recovery.requested)
        self.assertIn(self.store.umo, self.recovery.rewind)
        self.recovery.close()
        self.bot.unsubscribe.assert_called_once_with(name, handler)

    async def test_history_preserves_uid_quote_media_and_recalled_record(self):
        self.archive(self.raw(10, 100))
        recalled = self.archive(self.raw(11, 101))
        self.store.delete_message("11")
        reply = self.raw(20, 200, user=1002, parts=[
            {"type": "reply", "data": {"id": 10}}, {"type": "at", "data": {"qq": "1001"}},
            {"type": "image", "data": {"file": "retained-image-id", "url": "https://example.invalid/image"}}])
        counts = self.store.append_history([history_message(reply, self.store.umo, "9000"),
                                          history_message(self.raw(11, 101), self.store.umo, "9000")])
        self.assertEqual(counts, {"inserted": 1, "existing": 1, "ignored": 0})
        row = self.store.db.execute("SELECT sender_id,content_json FROM messages WHERE message_id='20'").fetchone()
        self.assertEqual(row["sender_id"], "1002")
        self.assertIn("retained-image-id", row["content_json"])
        self.assertEqual(self.store.db.execute("SELECT is_deleted FROM messages WHERE id=?", (recalled["id"],)).fetchone()[0], 1)
        relation = self.store.db.execute("SELECT target_platform_message_id FROM message_relations WHERE relation='REPLY_TO'").fetchone()
        self.assertEqual(relation[0], "10")
