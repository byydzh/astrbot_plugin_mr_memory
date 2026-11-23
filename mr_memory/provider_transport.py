"""Map one MR turn onto the chat protocol of the configured AstrBot provider."""
from __future__ import annotations

import copy
import inspect
import json
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from urllib.parse import urlsplit
from weakref import WeakKeyDictionary

from .tool_calls import field, parse_completion


def finish_reason(response):
    raw = field(response, "raw_completion")
    choices = field(raw, "choices") or field(raw, "candidates") or []
    reason = field(choices[0], "finish_reason") if choices else field(raw, "stop_reason")
    if field(raw, "status") == "incomplete":
        reason = field(field(raw, "incomplete_details"), "reason") or "incomplete"
    reason = str(getattr(reason, "value", reason) or "").lower()
    if reason in {"max_tokens", "max_output_tokens", "model_context_window_exceeded"}:
        return "length"
    return reason


def assistant_history(response, text, calls=()):
    message = {"role": "assistant", "content": text,
               "reasoning_content": field(response, "reasoning_content") or ""}
    if signature := field(response, "reasoning_signature"):
        message["reasoning_signature"] = signature
    blocks = field(field(response, "raw_completion"), "content")
    if isinstance(blocks, list):
        thoughts = [b.model_dump(mode="json", exclude_none=True) for b in blocks
                    if field(b, "type") in {"thinking", "redacted_thinking"}]
        if thoughts and (len(thoughts) > 1 or thoughts[0]["type"] == "redacted_thinking"):
            message["native_thoughts"] = thoughts
            message.pop("reasoning_content", None)
            message.pop("reasoning_signature", None)
    if calls:
        message["tool_calls"] = [call.message() for call in calls]
    return message


def _contexts(messages, *, anthropic=False):
    # AstrBot's native adapters restore signed thinking from a ThinkPart.
    # Keep MR's persisted text separate so trace views can omit private reasoning.
    result = []
    for message in messages:
        row = dict(message)
        thoughts = row.pop("native_thoughts", None)
        signature = row.pop("reasoning_signature", None)
        if signature or (anthropic and thoughts):
            parts = list(thoughts) if anthropic and thoughts else [
                {"type": "think", "think": row.get("reasoning_content") or "", "encrypted": signature}]
            content = row.get("content")
            if isinstance(content, list):
                parts.extend(content)
            elif content:
                parts.append({"type": "text", "text": content})
            row["content"] = parts
        result.append(row)
    return result


# Gemini's public text_chat drops per-turn options. Keep the bound instance
# and its native-tool/usage hooks. Only the exact payload of a finishing MR
# turn gets a changed config, including when several groups finish concurrently.
_gemini_scopes = {}


@contextmanager
def _gemini_finish(provider, payload, json_output):
    if provider not in _gemini_scopes:
        original = provider._prepare_query_config
        pending = {}
        async def prepare(payloads, *args, **kwargs):
            config = await original(payloads, *args, **kwargs)
            if id(payloads) in pending:
                config = config.model_copy()
                config.tools = config.tool_config = None
                if pending[id(payloads)]:
                    config.response_mime_type = "application/json"
            return config
        previous = vars(provider).get("_prepare_query_config")
        provider._prepare_query_config = prepare
        _gemini_scopes[provider] = pending, prepare, previous
    pending, prepare, previous = _gemini_scopes[provider]
    pending[id(payload)] = json_output
    try:
        yield
    finally:
        del pending[id(payload)]
        if not pending:
            del _gemini_scopes[provider]
            if provider._prepare_query_config is prepare:
                if previous is None:
                    del provider._prepare_query_config
                else:
                    provider._prepare_query_config = previous


async def _gemini(provider, messages, prompt, tools, limit, choice, json_output):
    payload = {"model": provider.get_model(), "messages": [{"role": "system", "content": prompt}, *messages],
               "max_tokens": limit}
    if choice:
        payload["tool_choice"] = choice
    if tools is not None and choice != "none":
        # AstrBot's legacy Google Schema converter narrows type unions and
        # drops open-object constraints. The SDK accepts JSON Schema natively.
        declarations = [{"name": t.name, "description": t.description, "parameters_json_schema": t.parameters}
                        for t in tools.tools]
        tools = copy.copy(tools)
        tools.get_func_desc_google_genai_style = lambda: {"function_declarations": declarations}
    with _gemini_finish(provider, payload, json_output) if choice == "none" else nullcontext():
        return await provider._query(payload, None if choice == "none" else tools, request_max_retries=1)


def _request_provider(provider):
    scoped = copy.copy(provider)
    config = getattr(provider, "provider_config", {})
    extra = dict(config.get("custom_extra_body") or {})
    scoped.provider_config = {**config, "custom_extra_body": extra}
    return scoped, extra


# A compatible gateway may choose either output-limit field. Learn only from
# an explicit pre-generation 400 naming the supported replacement. Provider
# replacement on hot reload starts a fresh cache.
_chat_options = WeakKeyDictionary()


async def _openai(provider, messages, prompt, tools, limit, thinking, choice, json_output):
    scoped, extra = _request_provider(provider)
    payload, _ = await scoped._prepare_chat_payload(prompt=None, contexts=messages, system_prompt=prompt)
    endpoint = scoped.provider_config.get("api_base") or getattr(getattr(scoped, "client", None), "base_url", "")
    host = urlsplit(str(endpoint)).hostname
    responses = callable(getattr(scoped, "_parse_response", None)) and "input" in payload
    model = str(payload.get("model", ""))
    try:
        options = _chat_options.setdefault(provider, {}).setdefault((host, model), {})
    except TypeError:
        # Third-party provider instances may be unhashable.
        options = {}
    cap = "max_output_tokens" if responses else "max_tokens"
    if not responses and (host == "api.openai.com" or "max_completion_tokens" in extra):
        cap = "max_completion_tokens"
    cap = options.get("cap", cap)
    for key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
        extra.pop(key, None)
        payload.pop(key, None)
    payload[cap] = limit
    if not responses and host == "api.deepseek.com":
        extra["thinking"] = payload["thinking"] = {"type": thinking}
    if options.get("omit_thinking") or (host == "api.poe.com" and model.lower().rsplit("/", 1)[-1] == "gemini-4-hydrogen"):
        # This endpoint explicitly rejects even an operator-supplied disabled value.
        extra.pop("thinking", None)
        payload.pop("thinking", None)
    if responses:
        include = extra.pop("include", payload.get("include")) or []
        payload["include"] = list(dict.fromkeys([*include, "reasoning.encrypted_content"]))
    if choice:
        extra.pop("tool_choice", None)
        payload["tool_choice"] = choice
    if choice == "none":
        extra.pop("tools", None)
    if json_output:
        if responses:
            payload["text"] = {**(extra.pop("text", None) or {}), **(payload.get("text") or {}),
                               "format": {"type": "json_object"}}
        else:
            extra.pop("response_format", None)
            payload["response_format"] = {"type": "json_object"}
    if not responses and (parser := getattr(scoped, "_parse_openai_completion", None)):
        async def parse(completion, tools):
            return await parse_completion(parser, completion, tools)
        scoped._parse_openai_completion = parse
    # Some gateways reject an optional thinking setting inherited from the
    # provider. Retry only an explicit rejection before any generation.
    tried_caps = {cap}
    while True:
        try:
            return await scoped._query(dict(payload), tools, request_max_retries=1)
        except Exception as exc:
            if getattr(exc, "status_code", None) != 400:
                raise
            detail = str(exc).lower()
            if ("thinking" in extra or "thinking" in payload) and any(s in detail for s in (
                    "does not support these request fields: thinking", "unsupported parameter: 'thinking'",
                    "unknown parameter: 'thinking'")):
                options["omit_thinking"] = True
                extra.pop("thinking", None)
                payload.pop("thinking", None)
                continue
            alternate = "max_completion_tokens" if cap == "max_tokens" else "max_tokens"
            if (responses or alternate in tried_caps or cap not in detail or
                    not any(f"use {q}{alternate}{q}" in detail for q in ("'", '"', "`", ""))):
                raise
            tried_caps.add(alternate)
            payload[alternate] = payload.pop(cap)
            cap = options["cap"] = alternate


async def _anthropic(provider, messages, prompt, tools, limit, choice, json_output):
    scoped, extra = _request_provider(provider)
    extra.pop("max_tokens", None)
    if choice:
        extra.pop("tool_choice", None)
    if choice == "none":
        extra.pop("tools", None)
        tools = None
    if json_output:
        # Anthropic requires closed objects in structured output. MR's memory
        # representation admits arbitrary keys. Carry the full JSON losslessly.
        prompt += '\n本轮将完整记忆 JSON 对象序列化为字符串，放入 memory_json 字段。'
        fmt = {"type": "json_schema", "schema": {"type": "object", "properties": {
            "memory_json": {"type": "string"}}, "required": ["memory_json"], "additionalProperties": False}}
    system, history = scoped._prepare_payload([{"role": "system", "content": prompt}, *messages])
    payload = {"model": scoped.get_model(), "messages": history, "max_tokens": limit,
               "system": [{"type": "text", "text": system}] if isinstance(system, str) else system}
    if choice and tools is not None:
        payload["tool_choice"] = choice
    if json_output:
        payload["output_config"] = {**extra.pop("output_config", {}), "format": fmt}
    # The host currently discards Message.stop_reason and strips signed
    # thinking whitespace. Retain its exact response without mutating the SDK.
    raw = None
    async def create(**kwargs):
        nonlocal raw
        raw = await provider.client.messages.create(**kwargs)
        return raw
    scoped.client = SimpleNamespace(messages=SimpleNamespace(create=create))
    response = await scoped._query(payload, tools, request_max_retries=1)
    response.raw_completion = raw
    if raw:
        for block in raw.content:
            if block.type == "thinking":
                response.reasoning_content, response.reasoning_signature = block.thinking, block.signature
    if json_output and finish_reason(response) != "length":
        inner = json.loads(response.completion_text)["memory_json"]
        if not isinstance(inner, str):
            raise ValueError("整理输出的 memory_json 必须是字符串")
        # The regular learning validator can repair a malformed inner document
        # and already knows how to retain its original text and usage.
        response.completion_text = inner
    return response


async def generate(provider, messages, prompt, tools, *, limit, thinking, choice=None, json_output=False):
    anthropic = callable(getattr(provider, "_prepare_payload", None)) and callable(getattr(provider, "_apply_thinking_config", None))
    messages = _contexts(messages, anthropic=anthropic)
    if callable(getattr(provider, "_prepare_query_config", None)) and callable(getattr(provider, "_prepare_conversation", None)):
        response = await _gemini(provider, messages, prompt, tools, limit, choice, json_output)
    elif callable(getattr(provider, "_prepare_chat_payload", None)):
        response = await _openai(provider, messages, prompt, tools, limit, thinking, choice, json_output)
    elif anthropic:
        response = await _anthropic(provider, messages, prompt, tools, limit, choice, json_output)
    else:
        chat = getattr(provider, "text_chat", None)
        params = inspect.signature(chat).parameters if chat else {}
        cap = next((key for key in ("max_output_tokens", "max_completion_tokens", "max_tokens") if key in params), None)
        if not cap or "func_tool" not in params or (json_output and "response_format" not in params):
            raise ValueError(f"{type(provider).__name__} 未提供 MR 所需的函数工具、逐轮输出上限或 JSON 收尾接口")
        options = {cap: limit}
        if json_output:
            options["response_format"] = {"type": "json_object"}
        if choice and "tool_choice" in params:
            options["tool_choice"] = choice
        response = await chat(contexts=messages, system_prompt=prompt, func_tool=None if choice == "none" else tools, **options)
    return response
