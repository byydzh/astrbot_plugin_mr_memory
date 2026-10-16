import json
import asyncio
import unittest
from types import SimpleNamespace

from mr_memory.agent import _reconstruction_output, _memory_output
from mr_memory.model_json import decode_envelope
from mr_memory.handoff import format_background
from mr_memory.tool_calls import response_calls
from mr_memory.agent import MemoryAgent
from unittest.mock import patch


class ModelEnvelopeTests(unittest.TestCase):
    def test_complete_values_survive_closing_wrappers_and_missing_outer_brace(self):
        obj = {"background": "原话有引号\"与换行\n，保持作者与语气"}
        wire = json.dumps(obj, ensure_ascii=False)
        for text in (wire + "}", wire[:-1], wire + "</tool", wire + "</｜｜DSML｜｜ parameter>\n</｜｜DSML｜｜ calls>", "```json\n" + wire + "\n```"):
            with self.subTest(text=text):
                self.assertEqual(decode_envelope(text)[0], obj)
                self.assertEqual(_reconstruction_output(text)[0], obj["background"])
        self.assertEqual(_memory_output('{"items":[]}\n</｜｜DSML｜｜ calls>')["items"], [])

    def test_incomplete_or_ambiguous_values_do_not_become_success(self):
        for text in ('{"background":"没有说完', '{"background":"a"} {"background":"b"}',
                     '{"items":[1,2', '{"items":[],"progress":', '{"background":"含有"未转义"引号"}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                decode_envelope(text)

    def test_native_arguments_preserve_wire_and_decode_extra_outer_delimiter(self):
        wire = '{"items":[],"finish":true}}'
        response = SimpleNamespace(raw_completion={"choices": [{"message": {"tool_calls": [
            {"id": "write", "function": {"name": "remember", "arguments": wire}}]}}]})
        call = response_calls(response)[0]
        self.assertEqual(call.arguments, {"items": [], "finish": True})
        self.assertTrue(call.format_note)
        self.assertEqual(call.raw_arguments, wire)
        self.assertEqual(json.loads(call.message()["function"]["arguments"]), call.arguments)

    def test_prose_delivery_is_lossless_and_does_not_absorb_memory_writes(self):
        body = '甲说“可能是朋友的画”，不是确认。\n乙（2002）还没回应。'
        for wire in ('{"background": ' + body + '}', body, json.dumps([body], ensure_ascii=False)):
            with self.subTest(wire=wire):
                call = response_calls(SimpleNamespace(tools_call_name=["complete"], tools_call_ids=["done"],
                    tools_call_args=[wire]))[0]
                self.assertFalse(call.error)
                self.assertEqual(format_background(call.arguments["background"]), body)
                self.assertEqual(call.raw_arguments, wire)
        wire = '{"background": 正文, "belief":{"把握":"可能"}, "references":[{"message_id":3}], "items":[{"content":"不能丢掉的写入"}]}'
        call = response_calls(SimpleNamespace(tools_call_name=["complete"], tools_call_ids=["done"],
            tools_call_args=[wire]))[0]
        self.assertEqual(call.arguments, {"background": "正文", "belief": {"把握": "可能"},
            "references": [{"message_id": 3}], "items": [{"content": "不能丢掉的写入"}]})
        for wire in ('{"background": 正文, "items":[{"content":"没有说完',
                     '{"background": "没有说完}', '{"background": 正文} {"items":[]}',
                     '{"background": [{"text":"内容", "belief": "猜测".to_string()}]}'):
            with self.subTest(wire=wire):
                call = response_calls(SimpleNamespace(tools_call_name=["complete"], tools_call_ids=["done"],
                    tools_call_args=[wire]))[0]
                self.assertTrue(call.error)
                self.assertEqual(call.raw_arguments, wire)
        self.assertEqual(_reconstruction_output('{"background": ' + body + '}')[0], body)

    def test_mixed_paragraphs_keep_wording_and_only_model_supplied_belief(self):
        body = format_background(["甲说的是自己的朋友。", {"text": "乙可能误会了。", "belief": {"把握": "尚未确认"}},
                                  {"text": "还没有后续。", "references": "无效地址"}])
        self.assertIn("甲说的是自己的朋友。", body)
        self.assertIn("乙可能误会了。", body)
        self.assertIn('"把握":"尚未确认"', body)
        self.assertNotIn('"stance":"unconfirmed"', body)
        self.assertIn('"status":"invalid_address"', body)

    def test_native_parser_receives_valid_json_without_mutating_shared_provider_or_wire(self):
        wire = '{"background": 作者是甲，乙只是在引用。}'
        completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[
            SimpleNamespace(id="done", function=SimpleNamespace(name="complete", arguments=wire))]))])

        class Provider:
            provider_config = {}
            async def _prepare_chat_payload(self, **kwargs): return {}, None
            async def _parse_openai_completion(self, value, tools):
                arguments = json.loads(value.choices[0].message.tool_calls[0].function.arguments)
                return SimpleNamespace(raw_completion=value, tools_call_args=[arguments])
            async def _query(self, payload, tools, **kwargs):
                return await self._parse_openai_completion(completion, tools)

        provider = Provider()
        response = asyncio.run(MemoryAgent(provider, None)._generate([], "test", object()))
        self.assertEqual(response.tools_call_args, [{"background": "作者是甲，乙只是在引用。"}])
        self.assertIs(response.raw_completion, completion)
        self.assertEqual(completion.choices[0].message.tool_calls[0].function.arguments, wire)
        self.assertNotIn("_parse_openai_completion", provider.__dict__)
        self.assertTrue(response_calls(response)[0].format_note)

    def test_invalid_final_is_repaired_within_existing_round_limit(self):
        class Agent(MemoryAgent):
            async def _model_turn(self, messages, prompt, tools, turn, **kwargs):
                if turn == 2:
                    self.last_request = messages[-1]
                    self.final_options = kwargs
                return SimpleNamespace(completion_text='{"background":"尚未闭合' if turn == 1 else "群友之前约好在西门见。",
                    tools_call_name=[], tools_call_args=[], tools_call_ids=[],
                    raw_completion={"choices": [{"finish_reason": "stop"}]})
        class Store:
            def recall_drafts(self, *args): return {}
            def record_cognition(self, **kwargs): pass
        agent = Agent(None, Store(), max_turns=2)
        with patch("mr_memory.agent._tool_set", return_value=object()):
            result = asyncio.run(agent.reconstruct({"sent_at": 123}, [], {}))
        self.assertEqual(result.status, "completed", result.detail)
        self.assertEqual(result.background, "群友之前约好在西门见。")
        self.assertIsNone(agent.final_options["response_format"])

    def test_basic_complete_delivers_selected_source_with_actual_author(self):
        from mr_memory.store import Store
        from mr_memory.memory_protocol import tool_definitions
        store = Store(":memory:", "test:GroupMessage:42")
        try:
            original = store.append_message({"platform_id": "test", "umo": store.umo, "group_id": "42",
                "message_id": "a", "sender_id": "1001", "sender_name": "甲", "sent_at": 100,
                "plain_text": "这是我朋友的画，不是我画的。", "content": [], "role": "USER"})
            class Agent(MemoryAgent):
                async def _model_turn(self, *args, **kwargs):
                    return SimpleNamespace(completion_text="", tools_call_name=["complete"], tools_call_ids=["done"],
                        tools_call_args=[{"background": "甲正在说明画的作者。",
                            "references": [{"message_id": original["id"]}]}])
            with patch("mr_memory.agent._tool_set", return_value=object()):
                result = asyncio.run(Agent(None, store).reconstruct({"sent_at": 101}, [], {}))
            self.assertEqual(result.status, "completed", result.detail)
            self.assertIn('"sender_id":"1001"', result.background)
            self.assertEqual(result.background.count(original["plain_text"]), 1)
            self.assertIn("references", tool_definitions(basic=True)["complete"]["parameters"]["properties"])
            legacy = format_background([{"text": "甲正在说明画的作者。", "belief": {"scope": "原话中的说明"},
                "references": [{"message_id": original["id"]}]}], store, 101)
            self.assertIn('"sender_id":"1001"', legacy)
            self.assertEqual(legacy.count(original["plain_text"]), 1)
        finally:
            store.close()
