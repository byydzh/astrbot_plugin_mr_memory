import asyncio
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mr_memory.agent import MemoryAgent, _model_view


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
    async def read_async(self, method, /, *args, **kwargs):
        return await asyncio.to_thread(method, *args, **kwargs)

    def workspace(self):
        return {"active": []}

    def continuity(self, before):
        return None

    def record_cognition(self, **experience):
        self.experience = experience

    def recall_drafts(self, pending=None, known=()):
        return {}

    def memory(self, kind, id, *, include_sources=True):
        assert kind == "episode"
        assert str(id) == "17"
        return {"kind": kind, "id": 17, "summary": "星舟桌游讨论", "source_keys": ["m1"]}

    def search_messages(self, **kwargs):
        return [{"id": 1, "source_key": "m1", "plain_text": "星舟是我们做的桌游", "sent_at": 100}]

    def search_message_context(self, **kwargs):
        return self.search_messages(**kwargs)

    def context(self, source_key, before_time, **kwargs):
        assert source_key == "m1"
        assert before_time == 300
        return [{"id": 1, "source_key": "m1", "plain_text": "星舟是我们做的桌游", "sent_at": 100},
                {"source_key": "m2", "plain_text": "已经有四人规则了", "role": "USER"}]

    def activity(self, **kwargs):
        return {"message_count": 237, "first_at": 100, "last_at": 200, "timezone": "Asia/Shanghai"}


class AgentTests(unittest.IsolatedAsyncioTestCase):
    def test_missing_query_reports_unknown_field_and_required_parameter_type(self):
        with self.assertRaises(ValueError) as failure:
            MemoryAgent._arguments("semantic_search", {"terms": ["星舟"]})
        detail = str(failure.exception)
        self.assertIn("unknown_fields=['terms']", detail)
        self.assertIn("missing_fields=['query']", detail)
        self.assertIn('"query":{"type":"string"}', detail)

    async def test_native_search_then_context_preserves_tool_history(self):
        provider = Provider([
            response(calls=[("semantic_search", {"query": "星舟桌游", "limit": 8}, "semantic-1"),
                            ("search_messages", {"terms": ["星舟"]}, "call-1")]),
            response(calls=[("context", {"source_key": "m1"}, "call-2"),
                            ("memory", {"kind": "episode", "id": 17}, "call-3")]),
            response("星舟指群友制作的桌游，已有四人规则。"),
        ])
        class Embedder:
            async def search(self, store, query, limit):
                assert query == "星舟桌游"
                assert limit == 8
                return [{"owner_type": "episode", "owner_key": "17", "score": 0.6}]
        with patch("mr_memory.agent._tool_set", return_value=object()):
            result = await MemoryAgent(provider, Store(), Embedder()).reconstruct({"sent_at": 300, "plain_text": "星舟是什么"}, [], {})
        self.assertEqual(result.status, "completed")
        self.assertIn("桌游", result.background)
        self.assertEqual(len(result.tool_calls), 4)
        self.assertEqual(result.tool_calls[0]["result"][0]["memory"]["id"], 17)
        history = provider.requests[-1][0]["messages"]
        self.assertEqual([m["tool_call_id"] for m in history if m["role"] == "tool"], ["semantic-1", "call-1", "call-2", "call-3"])
        original = next(m for m in history if m.get("tool_call_id") == "semantic-1")
        self.assertEqual(json.loads(original["content"])[0]["memory"]["summary"], "星舟桌游讨论")
        reopened_memory = json.loads(next(m for m in history if m.get("tool_call_id") == "call-3")["content"])
        self.assertEqual(reopened_memory["memory_ref"], {"kind": "episode", "id": 17, "fields": ["summary"]})
        reopened = next(m for m in history if m.get("tool_call_id") == "call-2")
        self.assertIn("四人规则", reopened["content"])
        self.assertEqual(json.loads(reopened["content"])[0]["source_ref"]["message_id"], 1)
        self.assertNotIn("星舟是我们做的桌游", reopened["content"])
        self.assertEqual(result.usage, {"input_other": 30, "output": 12})

    async def test_independent_calls_and_turn_limit_remain_partial(self):
        provider = Provider([response(calls=[
            ("search_messages", {"terms": ["星舟"]}, "a"),
            ("activity", {"account_id": "synthetic-7", "start_at": 0, "end_at": 900}, "b"),
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

    def test_revision_transport_retains_unique_record_meaning(self):
        snapshot = {"record": {"id": 4, "content": "修正后的认识", "status": "ACTIVE", "claim_type": "reported",
                               "superseded_by": 9, "source_message_id": 7, "uncertainty": "原版本的不确定性"},
                    "memory": {"id": 4, "summary": "修正后的认识", "status": "ACTIVE", "uncertainty": "后来形成的不确定性"}}
        result = _model_view({"snapshot": snapshot})["snapshot"]
        self.assertEqual(result["record"], {"claim_type": "reported", "superseded_by": 9,
            "source_message_id": 7, "uncertainty": "原版本的不确定性"})
        self.assertEqual(result["memory"], snapshot["memory"])

if __name__ == "__main__":
    unittest.main()
