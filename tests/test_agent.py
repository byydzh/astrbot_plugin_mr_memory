import asyncio
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mr_memory.agent import MemoryAgent


def response(text="", calls=(), finish="stop"):
    return SimpleNamespace(completion_text=text, tools_call_name=[x[0] for x in calls],
                           tools_call_args=[x[1] for x in calls], tools_call_ids=[x[2] for x in calls],
                           reasoning_content="", usage={"input_other": 10, "output": 4},
                           raw_completion={"choices": [{"finish_reason": finish}]})


class Provider:
    provider_config = {}

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []

    async def _prepare_chat_payload(self, prompt, contexts, system_prompt):
        assert prompt is None
        return {"model": "deepseek-v4-flash", "messages": copy.deepcopy(contexts)}, contexts

    async def _query(self, payload, tools, request_max_retries):
        assert payload["thinking"] == {"type": "disabled"}
        assert payload["max_tokens"] == 1200
        assert request_max_retries == 1
        self.requests.append((copy.deepcopy(payload), tools))
        value = self.replies.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


class Store:
    def memory(self, kind, id):
        assert kind == "episode"
        assert str(id) == "17"
        return {"kind": kind, "id": 17, "summary": "星舟桌游讨论", "source_keys": ["m1"]}

    def search_messages(self, **kwargs):
        return [{"source_key": "m1", "plain_text": "星舟是我们做的桌游", "sent_at": 100}]

    def context(self, source_key, before_time, **kwargs):
        assert source_key == "m1"
        assert before_time == 300
        return [{"source_key": "m1", "plain_text": "星舟是我们做的桌游"},
                {"source_key": "m2", "plain_text": "已经有四人规则了", "role": "USER"}]

    def activity(self, **kwargs):
        return {"message_count": 237, "first_at": 100, "last_at": 200, "timezone": "Asia/Shanghai"}


class AgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_background_thinking_overrides_call_without_mutating_shared_provider(self):
        class ConfiguredProvider(Provider):
            provider_config = {"custom_extra_body": {"thinking": {"type": "disabled"}, "other": "kept"}}

            async def _query(self, payload, tools, request_max_retries):
                self.requests.append((payload, self.provider_config))
                return response('{"items":[]}')
        provider = ConfiguredProvider([])
        result = await MemoryAgent(provider, Store(), thinking_mode="enabled",
            max_output_tokens=384000).consolidate([{"id": 1, "source_key": "synthetic", "plain_text": "合成对话"}], {})
        self.assertEqual(result.status, "completed")
        payload, actual = provider.requests[0]
        self.assertEqual(payload["max_tokens"], 384000)
        self.assertEqual(actual["custom_extra_body"]["thinking"], {"type": "enabled"})
        self.assertEqual(provider.provider_config["custom_extra_body"]["thinking"], {"type": "disabled"})
        self.assertEqual(actual["custom_extra_body"]["other"], "kept")

    async def test_native_search_then_context_preserves_tool_history(self):
        provider = Provider([
            response(calls=[("search_messages", {"terms": ["星舟"]}, "call-1")]),
            response(calls=[("context", {"source_key": "m1"}, "call-2"),
                            ("memory", {"kind": "episode", "id": 17}, "call-3")]),
            response("星舟指群友制作的桌游，已有四人规则。"),
        ])
        class Embedder:
            async def search(self, store, query, limit):
                assert query == "星舟是什么"
                assert limit == 8
                return [{"owner_type": "episode", "owner_key": "17", "score": 0.6}]
        with patch("mr_memory.agent._tool_set", return_value=object()):
            result = await MemoryAgent(provider, Store(), Embedder()).reconstruct({"sent_at": 300, "plain_text": "星舟是什么"}, [], {})
        self.assertEqual(result.status, "completed")
        self.assertIn("桌游", result.background)
        self.assertEqual(len(result.tool_calls), 3)
        initial = json.loads(provider.requests[0][0]["messages"][0]["content"])
        self.assertEqual(initial["initial_search"]["status"], "completed")
        self.assertEqual(initial["seeds"][0]["memory"]["id"], 17)
        history = provider.requests[-1][0]["messages"]
        self.assertEqual([m["tool_call_id"] for m in history if m["role"] == "tool"], ["call-1", "call-2", "call-3"])
        self.assertIn("四人规则", history[-2]["content"])
        self.assertEqual(result.usage, {"input_other": 30, "output": 12})

    async def test_independent_calls_and_turn_limit_remain_partial(self):
        provider = Provider([response(calls=[
            ("search_messages", {"terms": ["星舟"]}, "a"),
            ("activity", {"participant_id": 7, "start_at": 0, "end_at": 900}, "b"),
        ])])
        with patch("mr_memory.agent._tool_set", return_value=object()):
            result = await MemoryAgent(provider, Store(), max_turns=1).reconstruct({"sent_at": 300}, [], {})
        self.assertEqual(result.status, "partial")
        self.assertEqual(len(result.tool_calls), 2)
        self.assertEqual(result.tool_calls[1]["result"]["message_count"], 237)
        self.assertEqual(result.background, "")

    async def test_timeout_is_not_no_memory(self):
        class SlowProvider(Provider):
            async def _query(self, *args, **kwargs):
                await asyncio.Event().wait()
        with patch("mr_memory.agent._tool_set", return_value=object()):
            result = await MemoryAgent(SlowProvider([]), Store(), timeout_seconds=0.01).reconstruct({}, [], {})
        self.assertEqual(result.status, "partial")
        self.assertIn("time budget", result.detail)

    async def test_consolidation_keeps_correction_and_retries_bad_output(self):
        item = {"kind": "semantic", "content": "用户澄清星舟是桌游，机器人此前误以为是电影。",
                "source_ids": [1, 2], "cues": {"bad_optional_shape": "星舟"}}
        provider = Provider([response(json.dumps({"items": [item]}, ensure_ascii=False)), response("{unfinished")])
        messages = [{"id": 1, "source_key": "u1", "plain_text": "星舟是桌游"},
                    {"id": 2, "source_key": "b1", "plain_text": "原来不是电影", "role": "BOT"}]
        agent = MemoryAgent(provider, Store())
        saved = await agent.consolidate(messages, {})
        retry = await agent.consolidate(messages, {})
        self.assertEqual(saved.status, "completed")
        expected = {key: value for key, value in item.items() if key != "source_ids"}
        self.assertEqual(saved.items, [{**expected, "source_keys": ["u1", "b1"], "cues": []}])
        self.assertEqual(retry.status, "retry")
        self.assertEqual(retry.items, [])
        self.assertEqual(provider.requests[0][0]["response_format"], {"type": "json_object"})


if __name__ == "__main__":
    unittest.main()
