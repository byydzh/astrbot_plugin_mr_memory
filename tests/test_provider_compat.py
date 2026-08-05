from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from mr_memory.provider_compat import generate_with_enforced_options


class _PreparedPayloadProvider:
    def __init__(self) -> None:
        self.payload = None

    async def _prepare_chat_payload(self, **kwargs):
        return {"model": "deepseek-v4-flash", "messages": []}, []

    async def _query(self, payload, tools, *, request_max_retries=None):
        self.payload = payload
        return "prepared-result"


class _StreamingPreparedPayloadProvider(_PreparedPayloadProvider):
    async def _query_stream(self, payload, tools, *, request_max_retries=None):
        self.payload = payload
        yield "partial-result"
        yield "final-result"


class _IncompleteStreamingProvider(_PreparedPayloadProvider):
    async def _query_stream(self, payload, tools, *, request_max_retries=None):
        yield SimpleNamespace(is_chunk=True)


class ProviderCompatibilityTests(unittest.TestCase):
    def test_options_are_injected_after_astrbot_prepares_payload(self) -> None:
        provider = _PreparedPayloadProvider()

        async def fallback(**kwargs):
            self.fail("fallback should not be used")

        result = asyncio.run(
            generate_with_enforced_options(
                provider=provider,
                fallback_generate=fallback,
                chat_provider_id="deepseek/deepseek-v4-flash",
                prompt="data",
                system_prompt="return json",
                options={
                    "thinking": {"type": "disabled"},
                    "response_format": {"type": "json_object"},
                    "max_tokens": 4096,
                },
            )
        )

        self.assertEqual(result, "prepared-result")
        self.assertEqual(provider.payload["thinking"], {"type": "disabled"})
        self.assertEqual(provider.payload["max_tokens"], 4096)

    def test_provider_without_prepared_payload_uses_public_fallback(self) -> None:
        captured = {}

        async def fallback(**kwargs):
            captured.update(kwargs)
            return "fallback-result"

        result = asyncio.run(
            generate_with_enforced_options(
                provider=object(),
                fallback_generate=fallback,
                chat_provider_id="another/provider",
                prompt="data",
                system_prompt="return json",
                options={"temperature": 0.0},
            )
        )

        self.assertEqual(result, "fallback-result")
        self.assertEqual(captured["temperature"], 0.0)
        self.assertEqual(captured["chat_provider_id"], "another/provider")

    def test_streaming_path_returns_the_provider_final_response(self) -> None:
        provider = _StreamingPreparedPayloadProvider()

        async def fallback(**kwargs):
            self.fail("fallback should not be used")

        result = asyncio.run(
            generate_with_enforced_options(
                provider=provider,
                fallback_generate=fallback,
                chat_provider_id="deepseek/deepseek-v4-flash",
                prompt="data",
                system_prompt="return json",
                options={"thinking": {"type": "enabled"}},
                stream=True,
            )
        )

        self.assertEqual(result, "final-result")
        self.assertEqual(provider.payload["thinking"], {"type": "enabled"})

    def test_incomplete_stream_is_not_accepted_as_a_full_response(self) -> None:
        provider = _IncompleteStreamingProvider()

        async def fallback(**kwargs):
            self.fail("fallback should not be used")

        with self.assertRaisesRegex(RuntimeError, "before assembling"):
            asyncio.run(
                generate_with_enforced_options(
                    provider=provider,
                    fallback_generate=fallback,
                    chat_provider_id="deepseek/deepseek-v4-flash",
                    prompt="data",
                    system_prompt="return json",
                    options={"thinking": {"type": "enabled"}},
                    stream=True,
                )
            )


if __name__ == "__main__":
    unittest.main()
