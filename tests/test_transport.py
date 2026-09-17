import asyncio
import json
import unittest
from unittest.mock import patch

from mr_memory.agent import MemoryAgent, _json, _learning_json, _expand_message_tables, compact_learning_task
from tests.test_agent import Provider, Store, response


class TransportTests(unittest.IsolatedAsyncioTestCase):
    def test_learning_table_preserves_each_speaker_quote_and_distinct_optional_fields(self):
        first = {"id": 17, "sender_id": "A", "sender_name": "小桥", "role": "USER", "sent_at": 100,
                 "platform_id": "test", "group_id": "42", "plain_text": "是小叶的朋友", "attachments": []}
        second = {**first, "id": 18, "sender_id": "B", "sender_name": "小叶", "sent_at": 101,
                  "plain_text": "对，朋友搬去了北门", "reply_to_message": first}
        encoded = _learning_json({"messages": [first, second]})
        table = json.loads(encoded)["messages"]
        self.assertEqual(table["message_defaults"]["group_id"], "42")
        rows = [dict(zip(table["message_columns"], row)) for row in table["message_rows"]]
        self.assertEqual([(row["id"], row["sender_id"], row["sender_name"]) for row in rows],
                         [(17, "A", "小桥"), (18, "B", "小叶")])
        self.assertEqual(_expand_message_tables(json.loads(encoded)), json.loads(_json({"messages": [first, second]})))
        self.assertIsInstance(json.loads(_json([first, second])), list)  # foreground remains unchanged

    def test_repeated_memory_fields_reference_original_and_preserve_changes_and_reasoning(self):
        memory = {"kind": "semantic", "id": 17, "summary": "约在西门见", "content": "约在西门见",
                  "source_ids": [1], "subject": "周末见面"}
        thought = {"role": "assistant", "content": None, "reasoning_content": "确认地点更改的思考",
                   "tool_calls": [{"id": "read", "type": "function", "function": {
                       "name": "memory", "arguments": '{"kind": "semantic", "id": 17}'}}]}
        task = {"continuation": {"conversation": [
            {"role": "user", "content": json.dumps(memory, ensure_ascii=False)}, thought,
            {"role": "tool", "tool_call_id": "read", "content": json.dumps([memory,
                {**memory, "summary": "改到北门", "content": "改到北门"}], ensure_ascii=False)}],
            "previous_input_tokens": 90, "previous_input_bytes": 300}}
        compact = compact_learning_task(task)["continuation"]
        self.assertEqual(compact["conversation"][1], thought)
        first = json.loads(compact["conversation"][0]["content"])
        later = json.loads(compact["conversation"][2]["content"])
        self.assertEqual(first["summary"], "约在西门见")
        self.assertNotIn("content", first)
        self.assertEqual(later[0]["memory_ref"]["id"], 17)
        self.assertIn("summary", later[0]["memory_ref"]["fields"])
        self.assertEqual(later[1]["summary"], "改到北门")
        self.assertNotIn("summary", later[1]["memory_ref"]["fields"])
        self.assertEqual(compact["input_tokens_per_byte"], 0.3)
        self.assertEqual(compact["previous_input_tokens"], 0)

    def test_quote_does_not_forget_full_message_and_new_fields_still_arrive(self):
        full = {"id": 17, "plain_text": "这里是原话", "sender_id": "A", "sender_name": "小桥",
                "sent_at": 100, "role": "USER", "content": [{"type": "image", "url": "test:picture"}]}
        seen = {}
        json.loads(_json(full, seen_messages=seen))
        quote = {key: value for key, value in full.items() if key != "content"}
        json.loads(_json({"reply_to_message": quote}, seen_messages=seen))
        again = json.loads(_json([full, {**full, "plain_text": "编辑后的原话"}], seen_messages=seen))
        self.assertNotIn("plain_text", again[0])
        self.assertNotIn("content", again[0])
        self.assertEqual(again[0]["sender_id"], "A")
        self.assertEqual(again[0]["sent_at"], 100)
        self.assertEqual(again[1]["plain_text"], "编辑后的原话")
        self.assertEqual(again[1]["source_ref"]["fields"], ["content"])
        expanded = json.loads(_json({**full, "attachments": [{"description": "新获取的图片描述"}]}, seen_messages=seen))
        self.assertEqual(expanded["attachments"], [{"description": "新获取的图片描述"}])

    async def test_final_turn_keeps_tool_table_and_sets_native_tool_choice(self):
        provider = Provider([response(calls=[("search_messages", {}, "read")]), response("已读到的共同经历")])
        tools = object()
        with patch("mr_memory.agent._tool_set", return_value=tools):
            result = await MemoryAgent(provider, Store(), max_turns=2).reconstruct({"sent_at": 300}, [], {})
        self.assertEqual(result.status, "partial")
        self.assertEqual(len(provider.requests), 2)
        self.assertTrue(all(request[1] is tools for request in provider.requests))
        self.assertNotIn("tool_choice", provider.requests[0][0])
        self.assertEqual(provider.requests[1][0]["tool_choice"], "none")
        self.assertEqual(provider.requests[1][0]["messages"][:len(provider.requests[0][0]["messages"])],
                         provider.requests[0][0]["messages"])

    async def test_parallel_reads_reference_prior_conversation_order_not_completion_order(self):
        provider = Provider([response(calls=[("search_messages", {}, "slow"), ("search_messages", {}, "fast")]),
                             response("已理解")])
        agent = MemoryAgent(provider, Store())
        calls = 0

        async def execute(name, arguments, cutoff):
            nonlocal calls
            calls += 1
            if calls == 1:
                await asyncio.sleep(0.01)
            return [{"id": 1, "plain_text": "完整原文", "sender_id": "A", "role": "USER"}]

        with patch.object(agent, "_execute", side_effect=execute), patch("mr_memory.agent._tool_set", return_value=object()):
            await agent.reconstruct({"sent_at": 300}, [], {})
        messages = [row for row in provider.requests[-1][0]["messages"] if row["role"] == "tool"]
        self.assertEqual([row["tool_call_id"] for row in messages], ["slow", "fast"])
        self.assertEqual(json.loads(messages[0]["content"])[0]["plain_text"], "完整原文")
        self.assertEqual(json.loads(messages[1]["content"])[0]["source_ref"]["message_id"], 1)
