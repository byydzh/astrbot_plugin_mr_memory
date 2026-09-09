import asyncio
import copy
import json
from pathlib import Path
import shutil
import uuid
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mr_memory.agent import MemoryAgent, _model_view, _json
from mr_memory.store import Store as MemoryStore


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
    def memory(self, kind, id, *, include_sources=True):
        assert kind == "episode"
        assert str(id) == "17"
        return {"kind": kind, "id": 17, "summary": "星舟桌游讨论", "source_keys": ["m1"]}

    def search_messages(self, **kwargs):
        return [{"id": 1, "source_key": "m1", "plain_text": "星舟是我们做的桌游", "sent_at": 100}]

    def context(self, source_key, before_time, **kwargs):
        assert source_key == "m1"
        assert before_time == 300
        return [{"id": 1, "source_key": "m1", "plain_text": "星舟是我们做的桌游", "sent_at": 100},
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
        self.assertIn("四人规则", history[-2]["content"])
        reopened = next(m for m in history if m.get("tool_call_id") == "call-2")
        self.assertIn("星舟是我们做的桌游", reopened["content"])
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
        self.assertEqual(saved.items, [{**expected, "source_keys": ["u1", "b1"]}])
        self.assertEqual(retry.status, "retry")
        self.assertEqual(retry.items, [])
        self.assertNotIn("response_format", provider.requests[0][0])
        self.assertIsNotNone(provider.requests[0][1])

    async def test_native_memory_array_with_closing_dsml_frame_is_preserved(self):
        item = {"kind": "episode", "summary": "大家将见面地点改到西门", "source_ids": [1]}
        text = json.dumps([item], ensure_ascii=False) + "</｜｜DSML｜｜parameter>\n</｜｜DSML｜｜invoke>\n</｜｜DSML｜｜tool_calls>"
        provider = Provider([response(text)])
        result = await MemoryAgent(provider, Store()).consolidate(
            [{"id": 1, "source_key": "u1", "plain_text": "改到西门了"}], {})
        self.assertEqual(result.status, "completed", result.detail)
        self.assertEqual(result.items, [{"kind": "episode", "summary": item["summary"], "source_keys": ["u1"]}])
        self.assertEqual(result.response_text, text)
        provider = Provider([response('已保存修正并更新关注。\n\n{"items": []}')])
        finished = await MemoryAgent(provider, Store()).consolidate(
            [{"id": 1, "source_key": "u1", "plain_text": "改到西门了"}], {})
        self.assertEqual(finished.status, "completed", finished.detail)
        self.assertEqual(finished.items, [])

    async def test_failed_write_cannot_finish_empty_but_successful_retry_can(self):
        item = {"kind": "episode", "summary": "改到西门见", "source_ids": [1]}
        messages = [{"id": 1, "source_key": "u1", "plain_text": "改到西门见"}]

        class WriteStore(Store):
            attempts = 0

            def save_memories(self, items, sources, *, mark_processed, run_id=None):
                self.attempts += 1
                if self.attempts == 1:
                    raise OSError("database write failed")
                return [{"kind": "episode", "id": 1, "summary": items[0]["summary"]}]

        for retry in (False, True):
            with self.subTest(retry=retry):
                replies = [response(calls=[("remember", {"items": [item]}, "save")])]
                if retry:
                    replies.append(response(calls=[("remember", {"items": [item]}, "retry")]))
                replies.append(response('{"items":[]}'))
                result = await MemoryAgent(Provider(replies), WriteStore()).consolidate(messages, {})
                self.assertEqual(result.status, "completed" if retry else "partial")
                if not retry:
                    self.assertIn("database write failed", result.detail)

    def test_model_sees_natural_speakers_and_sources_without_person_codes(self):
        value = {"members": [{"id": 7, "account_id": "10", "name": "小桥"},
                             {"id": 14, "account_id": "20", "name": "小桥"}],
                 "memory": {"kind": "semantic", "id": 7, "subject_participant_id": 14,
                            "subject": {"name": "小桥", "account_id": "20"},
                            "sources": [{"id": 99, "participant_id": 14, "sender_id": "20",
                                         "sender_name": "小桥", "plain_text": "那是我之前说的", "role": "USER"}]}}
        visible = _model_view(value)
        self.assertEqual(visible["members"], [{"account_id": "10", "name": "小桥"}, {"account_id": "20", "name": "小桥"}])
        self.assertEqual(visible["memory"]["id"], 7)
        self.assertNotIn("subject_participant_id", visible["memory"])
        self.assertEqual(visible["memory"]["sources"][0]["sender_id"], "20")
        self.assertNotIn("participant_id", visible["memory"]["sources"][0])
        source = visible["memory"]["sources"][0]
        seen = {}
        first = json.loads(_json([source, source], seen_messages=seen))
        self.assertEqual(first[0], source)
        self.assertEqual(first[1]["message_id"], source["id"])
        revised = {**source, "plain_text": "后来编辑后的原文"}
        self.assertEqual(json.loads(_json(revised, seen_messages=seen)), revised)

    async def test_failed_reflection_is_not_completed_by_an_empty_memory_result(self):
        class Reflections:
            attempts = 0

            def save(self, item):
                self.attempts += 1
                if self.attempts == 1:
                    raise OSError("reflection write failed")
                return {"id": 1, **item}

        for retry in (False, True):
            store = Store()
            store.reflections = Reflections()
            args = {"content": "等待当事人后续说明", "status": "waiting", "next_review_at": None}
            replies = [response(calls=[("reflect", args, "save")])]
            if retry:
                replies.append(response(calls=[("reflect", args, "retry")]))
            replies.append(response('{"items":[]}'))
            result = await MemoryAgent(Provider(replies), store).consolidate([], {"reflections": [{"id": 1}]})
            self.assertEqual(result.status, "completed" if retry else "partial", result.detail)
            if not retry:
                self.assertIn("reflection write failed", result.detail)

    async def test_turn_limit_after_a_native_write_preserves_it_but_is_partial(self):
        class WriteStore(Store):
            def save_memories(self, items, sources, **kwargs):
                return [{"kind": "episode", "id": 1, "summary": items[0]["summary"]}]
        provider = Provider([response(calls=[("remember", {"items": [
            {"kind": "episode", "summary": "改到西门见", "source_ids": [1]}]}, "save")])])
        result = await MemoryAgent(provider, WriteStore(), max_turns=1).consolidate(
            [{"id": 1, "source_key": "u1", "plain_text": "改到西门见"}], {})
        self.assertEqual(result.status, "partial", result.detail)
        self.assertEqual(len(result.written), 1)
        self.assertEqual(len(provider.requests), 1)
        self.assertIsNotNone(provider.requests[0][1])
        self.assertNotIn("response_format", provider.requests[0][0])

    async def test_saving_continues_from_returned_memory_ids_to_reflection_and_completion(self):
        class Reflections:
            def save(self, item):
                self.saved = item
                return {"id": 9, **item}
        class WriteStore(Store):
            reflections = Reflections()
            def save_memories(self, items, sources, **kwargs):
                return [{"kind": "semantic", "id": 37, "summary": items[0]["content"]}]
        class FollowingProvider(Provider):
            async def _query(self, payload, tools, request_max_retries):
                if len(self.requests) == 2:
                    saved = next(m for m in payload["messages"] if m.get("tool_call_id") == "save")
                    record = json.loads(saved["content"])[0]
                    self.replies.insert(0, response(calls=[("reflect", {
                        "content": "已经厘清星舟指的是共同制作的桌游", "status": "resolved",
                        "memory_refs": [{"kind": record["kind"], "id": record["id"]}]}, "reflect")]))
                return await super()._query(payload, tools, request_max_retries)
        first = response(calls=[("search_messages", {"terms": ["星舟"]}, "read")])
        first.usage = {"input_other": 100, "output": 0}
        provider = FollowingProvider([
            first,
            response(calls=[("remember", {"items": [
                {"kind": "semantic", "content": "星舟是共同制作的桌游", "source_ids": [1]}]}, "save")]),
            response('{"items":[]}')])
        store = WriteStore()
        result = await MemoryAgent(provider, store, max_turns=None).consolidate(
            [{"id": 1, "source_key": "u1", "plain_text": "星舟是我们做的桌游"}], {}, token_budget=400)
        self.assertEqual(result.status, "completed", result.detail)
        self.assertEqual(len(result.written), 1)
        self.assertEqual(store.reflections.saved["memory_refs"], [{"kind": "semantic", "id": 37}])
        self.assertEqual(len(provider.requests), 4)
        for request in provider.requests[1:]:
            self.assertEqual({t.name for t in request[1].tools}, {"remember", "reflect"})

    async def test_learning_reads_old_sources_then_revises_and_reuses_real_graph_nodes(self):
        test_root = Path(__file__).resolve().parents[1] / ".dev"
        test_root.mkdir(exist_ok=True)
        directory = test_root / f"test-agent-{uuid.uuid4().hex}"
        directory.mkdir()
        try:
            store = MemoryStore(Path(directory) / "memory.db", "test:GroupMessage:42")
            try:
                def message(mid, sender, text):
                    return store.append_message({"umo": store.umo, "platform_id": "test", "message_id": mid,
                        "sender_id": sender, "sender_name": "小桥", "sent_at": 100, "role": "USER",
                        "plain_text": text, "content": [{"type": "text", "text": text}]})
                old = message("old", "10", "我周日去排练室")
                now = message("new", "20", "另一位小桥周日去，我周六去")
                previous = store.save_memories([{"kind": "semantic", "person": "小桥", "content": "两位小桥都周六去",
                                                "source_keys": [old["source_key"]]}],
                                               [old["source_key"]], mark_processed=False)[0]

                class LearningProvider(Provider):
                    async def _query(self, payload, tools, request_max_retries):
                        self.requests.append((copy.deepcopy(payload), tools))
                        turn = len(self.requests)
                        if turn == 1:
                            return response(calls=[("search_memories", {"terms": ["小桥"]}, "read"),
                                ("memory", {"kind": "semantic", "id": previous["id"]}, "original")])
                        if turn == 2:
                            return response(calls=[("remember", {"items": [
                                {"kind": "semantic", "id": previous["id"], "content": "账号10的小桥周日去，账号20的小桥周六去",
                                 "source_ids": [now["id"]]},
                                {"kind": "association", "source": {"label": "小桥", "description": "账号10的群友"},
                                 "target": {"label": "排练室"}, "relation": "周日去", "statement": "账号10的小桥周日去排练室",
                                 "source_ids": [now["id"]]}]}, "save")])
                        if turn == 3:
                            saved = json.loads(next(m["content"] for m in reversed(payload["messages"]) if m["role"] == "tool"))[1]
                            return response(calls=[("remember", {"items": [
                                {"kind": "semantic", "id": previous["id"], "content": "读回旧原文，确定是两位小桥不同的安排",
                                 "source_ids": [old["id"], now["id"]]},
                                {"kind": "association",
                                "source": {"label": "小桥", "description": "账号20的另一位群友"},
                                "target": {"node_id": saved["target_node_id"]}, "relation": "周六去",
                                "statement": "另一位小桥周六去同一个排练室", "source_ids": [now["id"]]}]}, "reuse")])
                        return response('{"items":[]}')

                provider = LearningProvider([])
                result = await MemoryAgent(provider, store, max_turns=4).consolidate([now], {})
                self.assertEqual(result.status, "completed", result.detail)
                self.assertEqual(result.items, [])
                self.assertEqual(len(result.written), 3)
                self.assertEqual(result.written[0]["id"], previous["id"])
                self.assertIn("读回旧原文", result.written[0]["summary"])
                self.assertEqual(result.written[0]["source_ids"], [old["id"], now["id"]])
                first, second = result.written[1:]
                self.assertEqual(first["target_node_id"], second["target_node_id"])
                self.assertNotEqual(first["source_node_id"], second["source_node_id"])
                self.assertEqual(len(store.pending_messages(10)), 2)
                self.assertEqual(result.usage, {"input_other": 40, "output": 16})
            finally:
                store.close()
        finally:
            if directory.resolve().parent != test_root.resolve():
                raise RuntimeError("Test cleanup path is outside test workspace")
            shutil.rmtree(directory)


if __name__ == "__main__":
    unittest.main()
