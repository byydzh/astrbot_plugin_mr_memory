"""AstrBot request preservation and observable event capture, without model calls."""
from __future__ import annotations

import asyncio
import copy
import hashlib
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
import uuid

from astrbot.api import ToolSet
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context
from astrbot.core.agent.message import ImageURLPart, TextPart
from astrbot.core.agent.tool import FunctionTool
from astrbot.core.message.components import At, File, Image, Plain, Reply
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.platform.astrbot_message import AstrBotMessage, Group, MessageMember
from astrbot.core.platform.message_type import MessageType
from astrbot.core.platform.platform_metadata import PlatformMetadata
from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from astrbot_plugin_mr_memory import main as plugin_module
from astrbot_plugin_mr_memory.mr_memory.agent import ReconstructionResult, ConsolidationResult


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
    async def asyncSetUp(self):
        self.test_root = Path(__file__).resolve().parents[1] / ".dev"
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
            "capture_enabled": True, "embedding_enabled": True})

    async def asyncTearDown(self):
        await self.plugin.terminate()
        for item in reversed(self.patches):
            item.stop()
        directory = self.directory.resolve()
        if directory.parent != self.test_root.resolve():
            raise RuntimeError("test cleanup path is outside the test workspace")
        shutil.rmtree(directory)

    async def test_injection_preserves_astrbot_request_when_working_state_write_fails(self):
        current = event()
        current.send = AsyncMock(return_value={"message_id": "synthetic-receipt"})
        current.bot = SimpleNamespace(get_group_member_list=AsyncMock(return_value=[]))
        store = await self.plugin.store_for(current)
        tools = ToolSet(tools=[FunctionTool(name="synthetic_lookup", description="Synthetic lookup", parameters={"type": "object", "properties": {}})])
        text_part = TextPart(text="其他插件提供的上下文")
        image_part = ImageURLPart(image_url=ImageURLPart.ImageURL(url="https://example.invalid/context.png"))
        old_memory = TextPart(text="<mr_group_context>旧的本次背景</mr_group_context>")
        request = ProviderRequest(
            prompt="真正的当前提问", system_prompt="原始人格与系统设定", session_id="existing-session",
            contexts=[{"role": "user", "content": "上轮消息"}, {"role": "assistant", "content": "上轮实际回复"}],
            image_urls=["https://example.invalid/current.png"], audio_urls=["https://example.invalid/current.wav"],
            func_tool=tools, extra_user_content_parts=[text_part, image_part, old_memory], model="original-main-model",
        )
        preserved = {name: copy.deepcopy(getattr(request, name)) for name in
                     ("prompt", "system_prompt", "session_id", "contexts", "image_urls", "audio_urls", "model")}
        result = ReconstructionResult(background="合成成员是本群参与者，之前讨论了图书馆。", status="completed",
                                 elapsed_ms=12, usage={}, tool_calls=[])
        memory_agent = SimpleNamespace(reconstruct=AsyncMock(return_value=result))
        with patch.object(self.plugin, "agent", return_value=memory_agent), \
                patch.object(store, "save_working_state", side_effect=OSError("synthetic disk failure")), \
                patch.object(plugin_module.logger, "exception") as error_log:
            await self.plugin.inject_subconscious_memory(current, request)
        self.assertEqual({name: getattr(request, name) for name in preserved}, preserved)
        self.assertIs(request.func_tool, tools)
        self.assertEqual(len(request.func_tool.tools), 1)
        self.assertIs(request.extra_user_content_parts[0], text_part)
        self.assertIs(request.extra_user_content_parts[1], image_part)
        self.assertEqual(len(request.extra_user_content_parts), 3)
        self.assertIn(result.background, request.extra_user_content_parts[-1].text)
        self.assertNotIn("旧的本次背景", request.extra_user_content_parts[-1].text)
        self.assertEqual(memory_agent.reconstruct.await_args.args[0]["question"], request.prompt)
        error_log.assert_called_once()
        current.bot.get_group_member_list.assert_awaited_once_with(group_id=42, self_id="9000")
        with patch.object(self.plugin, "record_bot_event", new_callable=AsyncMock) as record:
            await current.send(MessageChain([Plain(text="只有请求钩子的发送")]))
            record.assert_awaited_once()

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

    async def test_aiocqhttp_streaming_records_successful_segments_and_preserves_send_failures(self):
        original = event()
        bot = SimpleNamespace(send_group_msg=AsyncMock(
            side_effect=[{"message_id": "501"}, {"message_id": "502"}],
            return_value={"message_id": "synthetic-receipt"}))
        current = AiocqhttpMessageEvent(original.message_str, original.message_obj, original.platform_meta, "42", bot)
        original_transport = current.send_message
        await self.plugin.capture_group_message(current)

        async def stream():
            yield MessageChain([Plain(text="第一段。")])
            yield MessageChain([Plain(text="第二段")])

        with patch("astrbot.core.platform.astr_message_event.Metric.upload", new_callable=AsyncMock):
            await current.send_streaming(stream(), use_fallback=True)
            failure = RuntimeError("synthetic send failure")
            bot.send_group_msg.side_effect = failure
            with self.assertRaises(RuntimeError) as caught:
                await current.send(MessageChain([Plain(text="没有发出去")]))
            self.assertIs(caught.exception, failure)
            store = await self.plugin.store_for(current)
            sent = [row for row in store.recent(limit=20) if row["role"] == "BOT"]
            self.assertEqual(sorted(row["plain_text"] for row in sent), ["第一段。", "第二段"])
            self.assertEqual({row["source_key"].rsplit("|", 1)[-1] for row in sent}, {"501", "502"})
            self.assertTrue(all(row["reply_to"].endswith("|synthetic-request-1") for row in sent))
            quoted = event()
            quoted.message_obj.message_id = "synthetic-request-2"
            quoted.message_obj.message = [Reply(id="502"), Plain(text="你刚才说第二段")]
            quoted.message_obj.message_str = "你刚才说第二段"
            await self.plugin.capture_group_message(quoted)
            followup = next(row for row in store.recent(limit=20) if row["source_key"].endswith("|synthetic-request-2"))
            target = next(row for row in sent if row["plain_text"] == "第二段")
            self.assertEqual(followup["reply_to"], target["source_key"])
            self.assertEqual(followup["relations"][0]["target_participant_id"], target["participant_id"])
            self.assertIsNone(current.get_result())
            self.assertEqual(bot.send_group_msg.await_count, 3)

            bot.send_group_msg.side_effect = None
            with patch.object(self.plugin, "record_bot_event", side_effect=OSError("synthetic log failure")), \
                    patch.object(plugin_module.logger, "exception") as error_log:
                self.assertIsNone(await current.send(MessageChain([Plain(text="成功发送但日志失败")])))
                error_log.assert_called_once()
            self.assertEqual(bot.send_group_msg.await_count, 4)
        self.assertIs(current.bot, bot)
        await self.plugin.terminate()
        self.assertEqual(current.send_message, original_transport)
        self.assertFalse(hasattr(current, "_mr_memory_send_message_wrapper"))

    async def test_aiocqhttp_one_send_keeps_each_physical_message_content(self):
        original = event()
        bot = SimpleNamespace(send_group_msg=AsyncMock(side_effect=[{"message_id": "601"}, {"message_id": "602"}]))
        current = AiocqhttpMessageEvent(original.message_str, original.message_obj, original.platform_meta, "42", bot)
        await self.plugin.capture_group_message(current)
        chain = MessageChain([Plain(text="文件说明"), File(name="synthetic.txt", url="https://example.invalid/synthetic.txt")])
        with patch("astrbot.core.platform.astr_message_event.Metric.upload", new_callable=AsyncMock), \
                patch("astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event.asyncio.sleep", new_callable=AsyncMock):
            await current.send(chain)
        store = await self.plugin.store_for(current)
        sent = {row["source_key"].rsplit("|", 1)[-1]: row for row in store.recent(limit=20) if row["role"] == "BOT"}
        self.assertEqual(set(sent), {"601", "602"})
        self.assertEqual(sent["601"]["plain_text"], "文件说明")
        self.assertEqual(sent["602"]["plain_text"], "[file]")
        self.assertTrue(any(part.get("name") == "synthetic.txt" for part in sent["602"]["content"]))

    async def test_initialize_with_no_database_starts_only_plugin_maintenance(self):
        self.assertEqual(list(self.plugin.scope_dir.glob("*.db")), [])
        await self.plugin.initialize()
        await asyncio.sleep(0)
        self.embedder.warmup.assert_awaited_once()
        self.assertEqual(self.plugin.stores, {})
        self.assertEqual(list(self.plugin.scope_dir.glob("*.db")), [])
        self.assertEqual(len(self.plugin.tasks), 1)
        self.assertFalse(next(iter(self.plugin.tasks)).done())
        self.context.get_provider_by_id.assert_not_called()

    async def test_hot_reload_waits_for_memory_search_before_closing_its_store(self):
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

    async def test_original_controls_drive_provider_and_manual_learning_honors_budget(self):
        self.plugin.config.update(local_serving_timeout_seconds=180, max_loop_steps=6,
            distillation_max_messages=500, maintenance_llm_timeout_seconds=3600,
            distillation_max_output_tokens=384000, distillation_thinking_mode="enabled",
            local_serving_reader_thinking_mode="disabled", private_daily_token_budget=50)
        store = await self.plugin.store_for(event())
        front = self.plugin.agent(store)
        back = self.plugin.agent(store, background=True)
        self.assertEqual((front.timeout_seconds, front.max_turns, front.thinking_mode), (180, 6, "disabled"))
        self.assertEqual((back.timeout_seconds, back.max_output_tokens, back.thinking_mode), (3600, 384000, "enabled"))
        store.reserve_usage("background", 50)
        with patch.object(self.plugin, "agent") as call:
            result = await self.plugin.consolidate(store, force=True)
        self.assertEqual(result["status"], "budget_exhausted")
        call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
