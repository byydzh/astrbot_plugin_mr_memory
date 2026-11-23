"""Offline requests through AstrBot and its real SDKs; no credentials or model calls."""
import asyncio
import copy
import json
import unittest
from importlib.util import find_spec
from types import SimpleNamespace
from unittest.mock import Mock, patch

from mr_memory.memory_protocol import _tool_set
from mr_memory.provider_transport import _gemini_finish, assistant_history, finish_reason, generate
from mr_memory.tool_calls import response_calls


class CapabilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_compatible_gateway_names_the_required_cap_once(self):
        class BadRequest(Exception):
            status_code = 400
        class Gateway:
            provider_config = {"api_base": "https://fixture.invalid/v1"}
            async def _prepare_chat_payload(self, **kwargs):
                return {"model": "fixture"}, []
            async def _query(self, payload, tools, **kwargs):
                seen.append(payload)
                if "max_tokens" in payload:
                    raise BadRequest("Unsupported parameter 'max_tokens'. Use 'max_completion_tokens' instead.")
                return SimpleNamespace()
        seen = []
        provider = Gateway()
        for limit in (17, 23):
            await generate(provider, [], "", None, limit=limit, thinking="disabled")
        self.assertEqual([b.get("max_completion_tokens") for b in seen], [None, 17, 23])
        self.assertTrue(all("thinking" not in b for b in seen))

    async def test_overlapping_google_finishes_never_change_other_traffic_or_leak_on_cancel(self):
        class Config(SimpleNamespace):
            def model_copy(self):
                return copy.copy(self)
        class Provider:
            async def _prepare_query_config(self, payloads):
                return Config(tools=["web"], tool_config="auto", response_mime_type=None)
        provider = Provider()
        ready = asyncio.Event()
        first, second, other = {}, {}, {}
        async def wait_in_finish():
            with _gemini_finish(provider, first, True):
                ready.set()
                await asyncio.Event().wait()
        pending = asyncio.create_task(wait_in_finish())
        await ready.wait()
        with _gemini_finish(provider, second, False):
            self.assertEqual((await provider._prepare_query_config(first)).response_mime_type, "application/json")
            self.assertIsNone((await provider._prepare_query_config(second)).tools)
            self.assertEqual((await provider._prepare_query_config(other)).tools, ["web"])
            pending.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await pending
            self.assertIsNone((await provider._prepare_query_config(second)).response_mime_type)
            self.assertEqual((await provider._prepare_query_config(first)).tools, ["web"])
        self.assertNotIn("_prepare_query_config", vars(provider))

    async def test_public_provider_with_explicit_limits_and_unsupported_provider(self):
        class Public:
            async def text_chat(self, *, contexts, system_prompt, func_tool, max_output_tokens, response_format):
                return (func_tool, max_output_tokens, response_format)
        result = await generate(Public(), [], "", object(), limit=22, thinking="disabled", choice="none", json_output=True)
        self.assertEqual(result, (None, 22, {"type": "json_object"}))
        class Opaque:
            async def text_chat(self, **kwargs):
                raise AssertionError("must reject before sending")
        with self.assertRaisesRegex(ValueError, "输出上限"):
            await generate(Opaque(), [], "", None, limit=22, thinking="enabled")


@unittest.skipUnless(find_spec("astrbot"), "requires the AstrBot runtime")
class NativeTransportTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, provider, exchange):
        import httpx
        wire = []
        async def send(client, request, **kwargs):
            body = json.loads(request.content)
            wire.append(body)
            return httpx.Response(200, json=exchange(len(wire), body), request=request)
        conversation = [{"role": "user", "content": "fixture JSON"}]
        options = dict(limit=222, thinking="disabled")
        with patch.object(httpx.AsyncClient, "send", send):
            r = await generate(provider, conversation, "fixture", _tool_set(), **options)
            calls = response_calls(r)
            self.assertEqual([c.name for c in calls], ["search_messages"])
            conversation.extend([assistant_history(r, r.completion_text or None, calls), {
                "role": "tool", "name": calls[0].name, "tool_call_id": calls[0].id, "content": "found"}])
            r = await generate(provider, conversation, "fixture", _tool_set(learning=True), **options)
            self.assertEqual(finish_reason(r), "length")
            r = await generate(provider, conversation, "fixture JSON", _tool_set(learning=True),
                               choice="none", json_output=True, **options)
            self.assertEqual(json.loads(r.completion_text), {"items": []})
        return wire

    async def test_google_native_signed_tool_and_json_finish(self):
        from astrbot.core.provider.sources.gemini_source import ProviderGoogleGenAI
        provider = ProviderGoogleGenAI({"id": "fixture", "key": ["offline"], "model": "gemini-4-carbon",
            "api_base": "https://generativelanguage.googleapis.com",
            "gm_native_search": True, "gm_url_context": True, "gm_thinking_config": {"level": "LOW"}}, {})
        self.addAsyncCleanup(provider.terminate)
        ledger = None
        try:
            from astrbot_plugin_gemini_native_tools.native import attach, detach
        except ModuleNotFoundError as exc:
            if exc.name != "astrbot_plugin_gemini_native_tools":
                raise
        else:
            ledger = Mock()
            bindings = attach(provider, ledger)
            self.addCleanup(detach, provider, bindings)
        original = provider._prepare_query_config
        def exchange(n, body):
            parts = [{"functionCall": {"id": "call-1", "name": "search_messages", "args": {"terms": ["x"]}},
                      "thoughtSignature": "c2lnbmVk"}] if n == 1 else [{"text": '{"items":[]}'}]
            return {"candidates": [{"content": {"role": "model", "parts": parts},
                                    "finishReason": "MAX_TOKENS" if n == 2 else "STOP"}],
                    "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2}}
        wire = await self.exercise(provider, exchange)
        for n, body in enumerate(wire):
            self.assertEqual(body["generationConfig"]["maxOutputTokens"], 222)
            decl = [f for t in body.get("tools", []) for f in t.get("functionDeclarations", [])]
            self.assertEqual(len(decl), [12, 11, 0][n])
            if n < 2:
                self.assertTrue(any("googleSearch" in t for t in body["tools"]))
                self.assertTrue(any("urlContext" in t for t in body["tools"]))
                if ledger is not None:
                    config = body["generationConfig"]["thinkingConfig"]
                    self.assertEqual(config.get("thinkingLevel", config.get("thinking_level")), "LOW", config)
                    self.assertEqual(body["toolConfig"]["functionCallingConfig"]["mode"], "VALIDATED")
            if n == 1:
                self.assertIn("c2lnbmVk", json.dumps(body["contents"]))
                remember = next(f for f in decl if f["name"] == "remember")
                schema = remember.get("parametersJsonSchema", remember.get("parameters_json_schema"))
                self.assertIsNotNone(schema, remember.keys())
                self.assertIn("anyOf", schema["properties"]["items"]["items"])
        self.assertFalse(wire[-1].get("tools"))
        self.assertEqual(wire[-1]["generationConfig"]["responseMimeType"], "application/json")
        self.assertEqual(provider._prepare_query_config, original)
        if ledger is not None:
            self.assertEqual(ledger.write.call_count, 3)

    async def test_anthropic_keeps_signatures_budget_and_free_form_writes(self):
        from astrbot.core.provider.sources.anthropic_source import ProviderAnthropic
        provider = ProviderAnthropic({"id": "fixture", "key": ["offline"], "model": "claude-fixture"}, {})
        self.addAsyncCleanup(provider.client.close)
        def exchange(n, body):
            blocks = [{"type": "thinking", "thinking": "  signed text  ", "signature": "signature"},
                      {"type": "tool_use", "id": "call-1", "name": "search_messages", "input": {"terms": ["x"]}}]
            if n > 1:
                blocks = [{"type": "text", "text": json.dumps({"memory_json": '{"items":[]}'})}]
            return {"id": "msg_fixture", "type": "message", "role": "assistant", "model": "claude-fixture",
                    "content": blocks, "stop_reason": ("tool_use", "max_tokens", "end_turn")[n-1],
                    "usage": {"input_tokens": 3, "output_tokens": 2}}
        wire = await self.exercise(provider, exchange)
        self.assertEqual([len(b.get("tools", [])) for b in wire], [12, 11, 0])
        self.assertTrue(all(b["max_tokens"] == 222 for b in wire))
        self.assertIn("  signed text  ", json.dumps(wire[1]["messages"]))
        self.assertIn("signature", json.dumps(wire[1]["messages"]))
        self.assertEqual(wire[-1]["output_config"]["format"]["type"], "json_schema")

    async def test_openai_responses_history_and_json_finish(self):
        from astrbot.core.provider.sources.openai_responses_source import ProviderOpenAIResponses
        provider = ProviderOpenAIResponses({"id": "fixture", "key": ["offline"], "model": "o-fixture"}, {})
        self.addAsyncCleanup(provider.client.close)
        def exchange(n, body):
            output = [{"type": "reasoning", "id": "rs_fixture", "encrypted_content": "sealed", "summary": []},
                      {"type": "function_call", "id": "fc_fixture", "call_id": "call-1", "name": "search_messages",
                       "arguments": '{"terms":["x"]}', "status": "completed"}] if n == 1 else [
                      {"type": "message", "id": "msg_fixture", "role": "assistant", "status": "completed",
                       "content": [{"type": "output_text", "text": '{"items":[]}', "annotations": []}]}]
            return {"id": "resp_fixture", "object": "response", "created_at": 1, "model": "o-fixture",
                    "status": "incomplete" if n == 2 else "completed", "output": output,
                    "incomplete_details": {"reason": "max_output_tokens"} if n == 2 else None}
        wire = await self.exercise(provider, exchange)
        self.assertTrue(all(b["max_output_tokens"] == 222 and "max_tokens" not in b for b in wire))
        self.assertIn("sealed", json.dumps(wire[1]["input"]))
        self.assertEqual(wire[-1]["tool_choice"], "none")
        self.assertEqual(wire[-1]["text"]["format"]["type"], "json_object")
        self.assertNotIn("response_format", wire[-1])

    async def test_chat_completions_budget_tools_and_json_finish(self):
        from astrbot.core.provider.sources.openai_source import ProviderOpenAIOfficial
        provider = ProviderOpenAIOfficial({"id": "fixture", "key": ["offline"], "model": "gpt-fixture",
            "api_base": "https://api.openai.com/v1", "custom_extra_body": {"max_tokens": 9999}}, {})
        self.addAsyncCleanup(provider.client.close)
        def exchange(n, body):
            msg = {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1", "type": "function",
                   "function": {"name": "search_messages", "arguments": '{"terms":["x"]}'}}]} if n == 1 else {
                   "role": "assistant", "content": '{"items":[]}'}
            return {"id": "fixture", "object": "chat.completion", "created": 1, "model": "gpt-fixture",
                    "choices": [{"index": 0, "message": msg, "finish_reason": ("tool_calls", "length", "stop")[n-1]}]}
        wire = await self.exercise(provider, exchange)
        self.assertTrue(all(b["max_completion_tokens"] == 222 and "max_tokens" not in b and "thinking" not in b for b in wire))
        self.assertEqual(wire[-1]["response_format"], {"type": "json_object"})
        self.assertEqual(wire[-1]["tool_choice"], "none")
