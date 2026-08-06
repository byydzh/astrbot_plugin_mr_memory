from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Any


def _supports_private_call(
    value: object,
    *,
    positional: int = 0,
    keywords: tuple[str, ...] = (),
) -> bool:
    if not callable(value):
        return False
    try:
        parameters = tuple(inspect.signature(value).parameters.values())
    except (TypeError, ValueError):
        return False
    has_var_positional = any(
        item.kind is inspect.Parameter.VAR_POSITIONAL for item in parameters
    )
    positional_capacity = sum(
        item.kind
        in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
        for item in parameters
    )
    if not has_var_positional and positional_capacity < positional:
        return False
    names = {item.name for item in parameters}
    has_var_keyword = any(
        item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters
    )
    return has_var_keyword or all(name in names for name in keywords)


async def generate_with_enforced_options(
    *,
    provider: Any,
    fallback_generate: Callable[..., Awaitable[Any]],
    chat_provider_id: str,
    prompt: str,
    system_prompt: str,
    options: Mapping[str, Any],
    stream: bool = False,
    on_stream_progress: Callable[[int, Any], None] | None = None,
) -> Any:
    """Preserve request options on AstrBot 4.27.1 OpenAI providers.

    That provider accepts ``**kwargs`` in ``text_chat`` but drops them while
    preparing its payload. Preparing first and calling its existing query path
    keeps AstrBot's configured client and response parser while making the
    requested structured-output controls effective. Long thinking calls may
    consume its native stream and return only the final reconstructed response.
    """

    prepare = getattr(provider, "_prepare_chat_payload", None)
    query = getattr(provider, "_query", None)
    query_stream = getattr(provider, "_query_stream", None)
    private_api_compatible = _supports_private_call(
        prepare,
        keywords=("prompt", "system_prompt"),
    ) and _supports_private_call(
        query,
        positional=2,
        keywords=("request_max_retries",),
    )
    if stream:
        private_api_compatible = private_api_compatible and _supports_private_call(
            query_stream,
            positional=2,
            keywords=("request_max_retries",),
        )
    if private_api_compatible:
        payload, _ = await prepare(
            prompt=prompt,
            system_prompt=system_prompt,
        )
        payload.update(dict(options))
        if stream:
            final_response = None
            chunk_count = 0
            async for response in query_stream(
                payload,
                None,
                request_max_retries=1,
            ):
                chunk_count += 1
                final_response = response
                if on_stream_progress is not None:
                    on_stream_progress(chunk_count, response)
            if final_response is None:
                raise RuntimeError("provider stream ended without a response")
            if bool(getattr(final_response, "is_chunk", False)):
                raise RuntimeError(
                    "provider stream ended before assembling a final response"
                )
            return final_response
        return await query(payload, None, request_max_retries=1)

    return await fallback_generate(
        chat_provider_id=chat_provider_id,
        prompt=prompt,
        system_prompt=system_prompt,
        **dict(options),
    )
