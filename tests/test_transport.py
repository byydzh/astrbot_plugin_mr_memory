import asyncio
import json
import unittest
from unittest.mock import patch

from mr_memory.agent import MemoryAgent, _json
from tests.test_agent import Provider, Store, response


class TransportTests(unittest.IsolatedAsyncioTestCase):
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
