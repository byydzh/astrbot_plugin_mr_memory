import os
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from mr_memory.history import HistoryRecovery, history_message
from mr_memory.store import Store


class HistoryRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = Path(os.environ['ASTRBOT_ROOT']) / ("history-" + uuid.uuid4().hex)
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
