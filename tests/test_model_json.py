import json
import asyncio
import unittest
from types import SimpleNamespace

from mr_memory.agent import _reconstruction_output, _memory_output
from mr_memory.model_json import decode_envelope
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
        self.assertEqual(call.message()["function"]["arguments"], wire)

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
                        tools_call_args=[{"background": [{"text": "甲正在说明画的作者。", "belief": {"scope": "原话中的说明"},
                            "references": [{"message_id": original["id"]}]}]}])
            with patch("mr_memory.agent._tool_set", return_value=object()):
                result = asyncio.run(Agent(None, store).reconstruct({"sent_at": 101}, [], {}))
            self.assertEqual(result.status, "completed", result.detail)
            self.assertIn('"sender_id":"1001"', result.background)
            self.assertEqual(result.background.count(original["plain_text"]), 1)
            self.assertIn("anyOf", tool_definitions(basic=True)["complete"]["parameters"]["properties"]["background"])
        finally:
            store.close()
