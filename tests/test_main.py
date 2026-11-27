"""AstrBot request preservation and observable event capture, without model calls."""
from __future__ import annotations

import os
import asyncio
import hashlib
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
import uuid

from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context
from astrbot.core.agent.tool import FunctionTool
from astrbot.core.message.components import At, Image, Plain
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.platform.astrbot_message import AstrBotMessage, Group, MessageMember
from astrbot.core.platform.message_type import MessageType
from astrbot.core.platform.platform_metadata import PlatformMetadata


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from astrbot_plugin_mr_memory import main as plugin_module
from astrbot_plugin_mr_memory.mr_memory.agent import ReconstructionResult
from tests import seed_legacy_scope


def seed_previous_group(plugin):
    umo = event().unified_msg_origin
    seed_legacy_scope(plugin.scope_dir / (hashlib.sha256(umo.encode()).hexdigest() + ".db"), umo)


def event() -> AstrMessageEvent:
    message = AstrBotMessage()
    message.type = MessageType.GROUP_MESSAGE
    message.message_id = "synthetic-request-1"
    message.session_id = "42"
    message.group = Group(group_id="42")
    message.self_id = "9000"
    message.sender = MessageMember(user_id="1001", nickname="合成发送者")
    message.message = [Plain(text="请问"), At(qq="1002", name="合成成员"), Plain(text="之前说了什么？")]
    message.message_str = "请问 @合成成员 之前说了什么？"
    message.timestamp = 1700000000
    message.raw_message = {"time": 1700000000}
    return AstrMessageEvent(
        message_str=message.message_str, message_obj=message,
        platform_meta=PlatformMetadata(name="aiocqhttp", description="Synthetic integration test", id="synthetic"),
        session_id="42",
    )


class MainIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_qq_official_receipt_keeps_role_context_only_for_successful_send(self):
        current = event()
        current.get_platform_name = Mock(return_value="qq_official")
        current._post_send_one = AsyncMock(return_value={"id": "official-receipt"})
        await self.plugin.capture_group_message(current)
        current._mr_output_context = {"producer": "example_rp", "mode": "roleplay",
                                      "speaker": {"id": "actor-a", "name": "角色甲"},
                                      "utterance_id": "t1", "presentation": "bilingual_text"}
        receipt = await current._post_send_one(MessageChain([Plain(text="角色台词")]))
        store = await self.plugin.store_for(current)
        self.assertEqual(receipt["id"], "official-receipt")
        self.assertEqual(store.recent()[-1]["content"][-1]["speaker"]["id"], "actor-a")
        current._mr_memory_original_post.side_effect = RuntimeError("failed upload")
        with self.assertRaises(RuntimeError):
            await current._post_send_one(MessageChain([Plain(text="未发送台词")]))
        self.assertEqual(len(store.recent()), 2)

    async def asyncSetUp(self):
        self.test_root = Path(os.environ['ASTRBOT_ROOT'])
        self.directory = self.test_root / f"test-main-{uuid.uuid4().hex}"
        self.directory.mkdir(parents=True)
        self.embedder = SimpleNamespace(warmup=AsyncMock(), close=AsyncMock())
        self.context = Mock(spec=Context)
        self.context.registered_web_apis = []
        self.patches = [
            patch.object(plugin_module, "get_astrbot_data_path", return_value=str(self.directory)),
            patch.object(plugin_module, "Embedder", return_value=self.embedder),
        ]
        for item in self.patches:
            item.start()
        self.plugin = plugin_module.MrMemoryPlugin(self.context, {"maintenance_interval_seconds": 600,
            "capture_enabled": True, "embedding_enabled": True, "learning_window_enabled": False,
            **getattr(self, "mode_settings", {"advanced_memory_enabled": True,
                "subconscious_enabled": True, "auto_distillation_enabled": True})})

    async def asyncTearDown(self):
        await self.plugin.terminate()
        for item in reversed(self.patches):
            item.stop()
        directory = self.directory.resolve()
        if directory.parent != self.test_root.resolve():
            raise RuntimeError("test cleanup path is outside the test workspace")
        shutil.rmtree(directory)

    async def test_first_group_reply_uses_network_with_configured_encoder(self):
        current = event()
        current.bot = SimpleNamespace(get_group_member_list=AsyncMock(return_value=[]))
        self.embedder.model_id = "three-dimensional-fixture"
        self.embedder.dimensions = 3
        self.embedder.query = AsyncMock(return_value=[1., 0., 0.])
        self.embedder.texts = AsyncMock(side_effect=lambda inputs: [[1., 0., 0.] for _ in inputs])
        await self.plugin.capture_group_message(current)
        store = await self.plugin.store_for(current)
        self.assertTrue(store.network_enabled)
        await self.plugin.index_pending(store)

        async def model_turn(messages, purpose, menu, turn, **kw):
            self.assertEqual(menu.tools[0].name, "explore_memory")
            self.assertIn("E|message:", messages[0]["content"])
            return SimpleNamespace(completion_text="已读到本群原话。", usage={})

        with patch.object(plugin_module.MemoryAgent, "_model_turn", side_effect=model_turn):
            request = ProviderRequest(prompt="刚才群里说了什么？")
            await self.plugin.inject_subconscious_memory(current, request)
        self.assertTrue(any("已读到本群原话。" in part.text for part in request.extra_user_content_parts))

    async def test_hook_delivers_background_without_forking_graph_understanding(self):
        seed_previous_group(self.plugin)
        current = event()
        current.bot = SimpleNamespace(get_group_member_list=AsyncMock(return_value=[]))
        store = await self.plugin.store_for(current)
        correction = "原图属于乙；甲是上一张图，作者在消息17纠正过。"
        saved = store.save_memories([{"kind": "understanding", "content": correction}], mark_processed=False)[0]
        replies = [
            ReconstructionResult(background="本轮理解了作者纠正。", status="completed"),
            ReconstructionResult(background="这轮只调整另一张图的颜色。", status="completed"),
            ReconstructionResult(background="这轮背景可用，但没有返回新笔记。", status="completed"),
        ]
        memory_agent = SimpleNamespace(reconstruct=AsyncMock(side_effect=replies))
        with patch.object(self.plugin, "agent", return_value=memory_agent):
            for _ in replies:
                request = ProviderRequest(prompt="合成问题")
                await self.plugin.inject_subconscious_memory(current, request)
                injected = next(p.text for p in request.extra_user_content_parts if p.text.startswith("<mr_group_context>"))
                self.assertNotIn(correction, injected)
        for call in memory_agent.reconstruct.call_args_list[1:]:
            self.assertNotIn("working_memory", call.args[2])
        self.assertNotIn("working_memory", store.load_working_state())
        self.assertEqual(store.memory("understanding", saved["id"])["content"], correction)

    async def test_user_generation_tools_and_sent_output_survive_store_reopen(self):
        current = event()
        receipt = {"message_id": "synthetic-receipt"}
        delivery = AsyncMock(return_value=receipt)
        current.send = delivery
        await self.plugin.capture_group_message(current)
        observed_send = current.send
        self.plugin.observe_sends(current)
        self.assertIs(current.send, observed_send)
        tool = FunctionTool(name="synthetic_lookup", description="Synthetic lookup", parameters={"type": "object", "properties": {}})
        arguments = {"query": "合成检索词"}
        await self.plugin.capture_tool_call(current, tool, arguments)
        await self.plugin.capture_tool_result(current, tool, arguments, "合成工具返回：找到了原始资料。")
        await self.plugin.capture_response(current, SimpleNamespace(completion_text="模型生成但尚未发送的合成回答"))
        sent = MessageChain([Plain(text="实际发送的合成回答"), Image(file=None, url="https://example.invalid/sent.png")])
        self.assertIs(await current.send(sent), receipt)
        delivery.assert_awaited_once_with(sent)
        self.assertIsNone(current.get_result())

        path = self.plugin.scope_dir / (hashlib.sha256(current.unified_msg_origin.encode()).hexdigest() + ".db")
        restored = plugin_module.Store(path, current.unified_msg_origin)
        try:
            rows = restored.recent(limit=20)
            user = next(row for row in rows if row["role"] == "USER")
            self.assertEqual(user["source_key"], f"{current.get_platform_id()}|{current.unified_msg_origin}|synthetic-request-1")
            self.assertTrue(any(item.get("account_id") == "1002" for item in user["content"]))
            events = {next(item["event"] for item in row["content"] if item["type"] == "bot_event"): row
                      for row in rows if row["role"] != "USER"}
            self.assertEqual(set(events), {"generated", "sent", "tool_call", "tool_result"})
            linked = restored.response_events("synthetic-request-1")
            self.assertEqual([next(p["event"] for p in row["content"] if p["type"] == "bot_event") for row in linked],
                             ["tool_call", "tool_result", "generated", "sent"])
            self.assertEqual(restored.response_events("different-request"), [])
            self.assertEqual(events["generated"]["role"], "SYSTEM")
            self.assertEqual(events["generated"]["plain_text"], "模型生成但尚未发送的合成回答")
            self.assertEqual(events["sent"]["role"], "BOT")
            self.assertIn("实际发送的合成回答", events["sent"]["plain_text"])
            self.assertTrue(any(item.get("url") == "https://example.invalid/sent.png" for item in events["sent"]["content"]))
            call = next(item for item in events["tool_call"]["content"] if item["type"] == "tool_call")
            returned = next(item for item in events["tool_result"]["content"] if item["type"] == "tool_result")
            self.assertEqual(call["arguments"], arguments)
            self.assertEqual(returned["result"], "合成工具返回：找到了原始资料。")
            for row in events.values():
                self.assertEqual(row["reply_to"], user["source_key"])
                self.assertTrue(any(item.get("type") == "response_to" and item.get("message_id") == "synthetic-request-1" for item in row["content"]))
        finally:
            restored.close()
        await self.plugin.terminate()
        self.assertIs(current.send, delivery)
        self.assertFalse(hasattr(current, "_mr_memory_send_observed"))

    async def test_initialize_with_no_database_starts_only_plugin_maintenance(self):
        self.assertEqual(list(self.plugin.scope_dir.glob("*.db")), [])
        await self.plugin.initialize()
        await asyncio.sleep(0)
        self.embedder.warmup.assert_awaited_once()
        self.assertEqual(self.plugin.stores, {})
        self.assertEqual(list(self.plugin.scope_dir.glob("*.db")), [])
        self.assertTrue(self.plugin.tasks)
        self.assertTrue(all(not task.done() for task in self.plugin.tasks))
        self.assertEqual(
            {task.get_coro().__qualname__ for task in self.plugin.tasks},
            {"MrMemoryPlugin.maintain", "HistoryRecovery.maintain"},
        )
        self.context.get_provider_by_id.assert_not_called()

    async def test_hot_reload_waits_for_memory_search_before_closing_its_store(self):
        seed_previous_group(self.plugin)
        current = event()
        current.send = AsyncMock()
        current.bot = SimpleNamespace(get_group_member_list=AsyncMock(return_value=[]))
        store = await self.plugin.store_for(current)
        entered, release = asyncio.Event(), asyncio.Event()

        async def reconstruct(*args):
            entered.set()
            await release.wait()
            self.assertTrue(store.recent(limit=1))
            return ReconstructionResult(background="仍在途的群聊背景", status="completed",
                                   elapsed_ms=12, usage={}, tool_calls=[])

        request = ProviderRequest(prompt="合成问题")
        with patch.object(self.plugin, "agent", return_value=SimpleNamespace(reconstruct=reconstruct)):
            search = asyncio.create_task(self.plugin.inject_subconscious_memory(current, request))
            await entered.wait()
            unloading = asyncio.create_task(self.plugin.terminate())
            await asyncio.sleep(0)
            self.assertFalse(unloading.done())
            release.set()
            await asyncio.gather(search, unloading)
        self.assertIn("仍在途的群聊背景", request.extra_user_content_parts[-1].text)
        self.assertEqual(self.plugin.stores, {})
        await self.plugin.capture_group_message(event())
        self.assertEqual(self.plugin.stores, {})

if __name__ == "__main__":
    unittest.main()
