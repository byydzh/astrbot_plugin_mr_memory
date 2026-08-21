from __future__ import annotations

import asyncio
import inspect
import unittest
from types import SimpleNamespace

from mr_memory.provider_compat import (
    ProviderCompatibilityError,
    generate_with_enforced_options,
)


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


class _StreamingOnlyPreparedPayloadProvider:
    def __init__(self) -> None:
        self.payload = None

    async def _prepare_chat_payload(self, **kwargs):
        return {"model": "deepseek-v4-flash", "messages": []}, []

    async def _query_stream(self, payload, tools, *, request_max_retries=None):
        self.payload = payload
        yield "stream-only-final"


class _DriftedPrivateProvider:
    def __init__(self) -> None:
        self.public_calls = 0

    async def _prepare_chat_payload(self, request):
        return request, []

    async def _query(self, request):
        return request

    async def text_chat(self, **kwargs):
        self.public_calls += 1
        return "public-result"


class _PublicOnlyProvider:
    def __init__(self) -> None:
        self.public_calls = 0

    async def text_chat(self, **kwargs):
        self.public_calls += 1
        return "public-result"


class ProviderCompatibilityTests(unittest.TestCase):
    def test_options_are_injected_after_astrbot_prepares_payload(self) -> None:
        provider = _PreparedPayloadProvider()

        result = asyncio.run(
            generate_with_enforced_options(
                provider=provider,
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

    def test_public_fallback_arguments_are_not_part_of_the_contract(self) -> None:
        parameters = inspect.signature(generate_with_enforced_options).parameters
        self.assertNotIn("fallback_generate", parameters)
        self.assertNotIn("chat_provider_id", parameters)

    def test_provider_without_private_api_fails_without_public_fallback(self) -> None:
        provider = _PublicOnlyProvider()

        with self.assertRaisesRegex(
            ProviderCompatibilityError,
            "public fallback is forbidden",
        ):
            asyncio.run(
                generate_with_enforced_options(
                    provider=provider,
                    prompt="data",
                    system_prompt="return json",
                    options={"temperature": 0.0},
                )
            )
        self.assertEqual(provider.public_calls, 0)

    def test_private_signature_drift_fails_without_public_fallback(self) -> None:
        provider = _DriftedPrivateProvider()

        with self.assertRaisesRegex(
            ProviderCompatibilityError,
            "missing or drifted",
        ):
            asyncio.run(
                generate_with_enforced_options(
                    provider=provider,
                    prompt="data",
                    system_prompt="return json",
                    options={"temperature": 0.0},
                )
            )
        self.assertEqual(provider.public_calls, 0)

    def test_streaming_path_returns_the_provider_final_response(self) -> None:
        provider = _StreamingPreparedPayloadProvider()
        progress = []

        result = asyncio.run(
            generate_with_enforced_options(
                provider=provider,
                prompt="data",
                system_prompt="return json",
                options={"thinking": {"type": "enabled"}},
                stream=True,
                on_stream_progress=lambda index, response: progress.append(
                    (index, response)
                ),
            )
        )

        self.assertEqual(result, "final-result")
        self.assertEqual(provider.payload["thinking"], {"type": "enabled"})
        self.assertEqual(
            progress,
            [(1, "partial-result"), (2, "final-result")],
        )

    def test_streaming_provider_does_not_need_unused_non_stream_query(self) -> None:
        provider = _StreamingOnlyPreparedPayloadProvider()

        result = asyncio.run(
            generate_with_enforced_options(
                provider=provider,
                prompt="data",
                system_prompt="return json",
                options={"thinking": {"type": "enabled"}},
                stream=True,
            )
        )

        self.assertEqual(result, "stream-only-final")
        self.assertEqual(provider.payload["thinking"], {"type": "enabled"})

    def test_incomplete_stream_is_not_accepted_as_a_full_response(self) -> None:
        provider = _IncompleteStreamingProvider()

        with self.assertRaisesRegex(RuntimeError, "before assembling"):
            asyncio.run(
                generate_with_enforced_options(
                    provider=provider,
                    prompt="data",
                    system_prompt="return json",
                    options={"thinking": {"type": "enabled"}},
                    stream=True,
                )
            )


if __name__ == "__main__":
    unittest.main()
