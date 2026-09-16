"""Use native AstrBot conversation storage with entirely synthetic histories."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

from astrbot.core.conversation_mgr import ConversationManager
from astrbot.core.db.sqlite import SQLiteDatabase
from astrbot.core.utils.shared_preferences import SharedPreferences

from mr_memory.conversation import ConversationRepair


class ConversationRepairTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = Path(__file__).resolve().parents[1] / ".pytest_cache" / ("conversation-" + uuid.uuid4().hex)
        self.directory.mkdir(parents=True)
        self.db = SQLiteDatabase(str(self.directory / "synthetic.db"))
        await self.db.initialize()
        # Only the native async preference API is needed; no global scheduler or
        # production preferences should be involved in this local DB test.
        preferences = object.__new__(SharedPreferences)
        preferences.db_helper = self.db
        self.preference_patch = patch("astrbot.core.conversation_mgr.sp", preferences)
        self.preference_patch.start()
        self.manager = ConversationManager(self.db)
        self.repair = ConversationRepair(SimpleNamespace(conversation_manager=self.manager))
        self.umo = "synthetic:GroupMessage:42"

    async def asyncTearDown(self):
        self.preference_patch.stop()
        await self.db.engine.dispose()
        for path in self.directory.iterdir():
            path.unlink()
        self.directory.rmdir()

    async def test_preserves_full_history_and_routes_inflight_completion_to_old_conversation(self):
        history = [
            {"role": "user", "content": [
                {"type": "text", "text": "合成图片请求"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,c3ludGhldGlj"}},
            ]},
            {"role": "assistant", "tool_calls": [
                {"id": "synthetic-call", "type": "function", "function": {"name": "draw", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "synthetic-call", "content": "合成工具结果"},
            {"role": "assistant", "content": "合成历史回复"},
        ]
        old_id = await self.manager.new_conversation(
            self.umo, platform_id="synthetic-platform", content=history,
            title="保留的原始标题", persona_id="synthetic-persona")
        await self.manager.update_conversation(self.umo, old_id, token_usage=123)
        old_before = await self.manager.get_conversation(self.umo, old_id)
        other_umo = "synthetic:GroupMessage:43"
        other_id = await self.manager.new_conversation(other_umo, content=history)

        result = await self.repair.reset_main_conversation(self.umo)
        self.assertEqual(result["status"], "cleared")
        self.assertEqual(result["preserved_messages"], 4)
        self.assertEqual(result["history_title"], "保留的原始标题")
        new_id = result["new_conversation_id"]
        old_after = await self.manager.get_conversation(self.umo, old_id)
        self.assertEqual(old_after, old_before)
        fresh = await self.manager.get_conversation(self.umo, new_id)
        self.assertEqual(json.loads(fresh.history), [])
        self.assertEqual(fresh.persona_id, "synthetic-persona")
        self.assertEqual(fresh.platform_id, "synthetic-platform")
        self.assertEqual(await self.manager.get_curr_conversation_id(other_umo), other_id)
        # A new manager has no in-memory selection: its answer comes from the
        # native persisted preference, as after a process reload.
        self.assertEqual(await ConversationManager(self.db).get_curr_conversation_id(self.umo), new_id)

        # AstrBot 4.27.1 persists agent completion with req.conversation.cid.
        await self.manager.update_conversation(
            self.umo, old_before.cid,
            history=history + [{"role": "assistant", "content": "已在运行的旧回复"}])
        self.assertEqual(json.loads((await self.manager.get_conversation(self.umo, new_id)).history), [])
        self.assertEqual(len(json.loads((await self.manager.get_conversation(self.umo, old_id)).history)), 5)
        self.assertEqual(await self.manager.get_curr_conversation_id(self.umo), new_id)

    async def test_empty_missing_and_concurrent_clicks_do_not_create_duplicate_histories(self):
        result = await self.repair.reset_main_conversation(self.umo)
        self.assertEqual(result["status"], "no_conversation")
        self.assertEqual(await self.manager.get_conversations(self.umo), [])
        old_id = await self.manager.new_conversation(self.umo, content=[])
        result = await self.repair.reset_main_conversation(self.umo)
        self.assertEqual(result["status"], "already_empty")
        self.assertEqual(len(await self.manager.get_conversations(self.umo)), 1)
        await self.manager.update_conversation(self.umo, old_id, history=[{"role": "user", "content": "合成内容"}])
        results = await asyncio.gather(
            self.repair.reset_main_conversation(self.umo),
            self.repair.reset_main_conversation(self.umo))
        self.assertEqual([r["status"] for r in results], ["cleared", "already_empty"])
        self.assertEqual(len(await self.manager.get_conversations(self.umo)), 2)


if __name__ == "__main__":
    unittest.main()
