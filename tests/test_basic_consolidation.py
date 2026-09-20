"""Basic ingestion must build a searchable graph with advanced work paused."""
import copy
import asyncio
import json
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from tests import test_main as fixtures


class BasicConsolidationTests(unittest.IsolatedAsyncioTestCase):
    mode_settings = {}
    asyncSetUp = fixtures.MainIntegrationTests.asyncSetUp
    asyncTearDown = fixtures.MainIntegrationTests.asyncTearDown

    async def test_slow_index_does_not_block_consolidation_schedule(self):
        store = await self.plugin.store_for(fixtures.event())
        started, release = asyncio.Event(), asyncio.Event()

        async def index(_store):
            started.set()
            await release.wait()

        async def consolidate(_store):
            self.plugin.stopping = True

        with patch.object(fixtures.plugin_module.asyncio, "sleep", new_callable=AsyncMock), \
                patch.object(self.plugin, "index_pending", side_effect=index), \
                patch.object(self.plugin, "consolidate", side_effect=consolidate) as learn:
            indexing = self.plugin.spawn(self.plugin.maintain(indexing=True))
            try:
                await asyncio.wait_for(started.wait(), 1)
                await asyncio.wait_for(self.plugin.maintain(), 1)
                learn.assert_awaited_once_with(store)
                self.assertFalse(indexing.done())
            finally:
                release.set()
                await indexing
                self.plugin.stopping = False

    async def test_basic_graph_write_and_retrieval_without_resuming_reflection(self):
        store = await self.plugin.store_for(fixtures.event())
        first = self.plugin.message(fixtures.event())
        first.update(message_id="first", plain_text="我把自制桌游带来了。", sent_at=int(time.time()) - 2)
        first = store.append_message(first)
        second = self.plugin.message(fixtures.event())
        second.update(message_id="second", sender_id="1002", sender_name="小乙",
                      plain_text="我们一起试了你的桌游，四个人都能玩。", sent_at=first["sent_at"] + 1)
        second = store.append_message(second)
        old = store.start_learning_task("background", [first["id"]])
        old = store.update_learning_task("background", {"checkpoint": "尚未完成的高级反思",
            "continuation": {"pending_write_errors": {"reflect": "旧写入未完成"},
                "write_state": {"pending_items": {"old": {"item": {"kind": "hypothesis"}}}}}})
        original = copy.deepcopy(old)

        reply = {"items": [
            {"kind": "episode", "handle": "play", "title": "一起试玩桌游",
             "content": "1001带来自制桌游；1002说大家一起试玩，可供四人游玩。",
             "source_ids": [first["id"], second["id"]]},
            {"kind": "semantic", "content": "1002报告这款桌游可以四人游玩。",
             "source_ids": [second["id"]],
             "connections": [{"handle": "play", "relation": "来自这次试玩", "purpose": "basis"}]},
        ], "progress": {"completed_ids": [first["id"], second["id"]], "checkpoint": "本批已完成"}}

        class Provider:
            provider_config = {}
            calls = []

            async def _prepare_chat_payload(self, **kwargs):
                return {"messages": [{"role": "system", "content": kwargs["system_prompt"]}, *kwargs["contexts"]]}, None

            async def _query(self, payload, tools, **kwargs):
                self.calls.append((copy.deepcopy(payload), [tool.name for tool in tools.tools]))
                return SimpleNamespace(completion_text=json.dumps(reply), tools_call_name=[],
                    usage={"input_other": 100, "input_cached": 50, "output": 80})

        provider = Provider()
        self.context.get_provider_by_id.return_value = provider
        blocked = AssertionError("Advanced background work must stay paused")
        with patch.object(store, "workspace", side_effect=blocked), \
                patch.object(store, "reconsider", side_effect=blocked), \
                patch.object(store, "recall_drafts", side_effect=blocked), \
                patch.object(store, "record_cognition", side_effect=blocked), \
                patch.object(store.reflections, "due", side_effect=blocked), \
                patch.object(store.reflections, "waiting", side_effect=blocked), \
                patch.object(self.plugin, "index_pending", new_callable=AsyncMock):
            result = await self.plugin.learn(store, force=True)
            disabled = await self.plugin.learn(store, force=True, feedback=True)
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual(result["written_count"], 2)
        self.assertEqual(store.pending_status()["count"], 0)
        self.assertEqual(disabled["status"], "disabled")
        self.assertEqual(len(provider.calls), 1)
        self.assertNotIn("reflect", provider.calls[0][1])
        self.assertIn("navigate", provider.calls[0][1])
        self.assertIn("<mr_member_directory>", provider.calls[0][0]["messages"][0]["content"])
        self.assertNotIn("尚未完成的高级反思", str(provider.calls))
        semantic = store.search_memories(terms=["四人"], kind="semantic")[0]
        edges = store.graph(ref={"kind": "semantic", "id": semantic["id"]})
        self.assertTrue(any(edge["target_ref"]["kind"] == "episode" for edge in edges), edges)
        self.assertEqual(store.memory("semantic", semantic["id"])["sources"][0]["sender_id"], "1002")

        # A mode toggle keeps real unfinished work, while recognizing material
        # already processed by the basic organizer.
        restored = store.select_learning_mode("background", "advanced")
        self.assertEqual(restored["continuation"], original["continuation"])
        self.assertEqual(restored["checkpoint"], original["checkpoint"])
        self.assertEqual(restored["completed_ids"], [first["id"]])
        store.select_learning_mode("background", "basic")
        self.assertIsNone(store.learning_task("background"))


if __name__ == "__main__":
    unittest.main()
