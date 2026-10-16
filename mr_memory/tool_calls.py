"""Keep the provider's original tool arguments, including unsuccessful writes."""
from __future__ import annotations

from dataclasses import dataclass
import copy
import json
import re

from .model_json import decode_background, decode_envelope


def field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def memory_text_arguments(text: str) -> dict | None:
    """Read one complete native DSML remember envelope returned as final text."""
    tag = r"[｜|]+DSML[｜|]+\s*"
    root = re.fullmatch(rf"<{tag}(calls|tool_calls|function_calls)>\s*(.*?)\s*</{tag}\1>", text.strip(), re.S)
    if not root:
        return None
    call = re.fullmatch(rf'<{tag}invoke\s+name="remember">\s*(.*?)\s*</{tag}invoke>', root[2], re.S)
    if not call:
        raise ValueError("Final DSML must contain one complete remember call")
    parameter = re.compile(rf'\s*<{tag}parameter\s+name="([^"]+)"\s+string="(true|false)">(.*?)</{tag}parameter>', re.S)
    arguments, position = {}, 0
    while call[1][position:].strip():
        match = parameter.match(call[1], position)
        if not match or match[1] in arguments:
            raise ValueError("Incomplete or repeated DSML remember parameter")
        arguments[match[1]] = match[3] if match[2] == "true" else json.loads(match[3])
        position = match.end()
    if not arguments or set(arguments) - {"items", "progress", "retry", "finish", "recollection"}:
        raise ValueError("DSML remember accepts items, progress, retry, finish and recollection")
    return arguments


@dataclass
class ToolCall:
    id: str
    name: str
    raw_arguments: str
    arguments: dict | None
    error: str = ""
    extra_content: object = None
    format_note: str = ""

    def message(self):
        # Keep original wire data in the trace, but replay a repaired call as
        # valid JSON so it cannot poison the following native request.
        arguments = (json.dumps(self.arguments, ensure_ascii=False) if self.format_note
                     and self.arguments is not None else self.raw_arguments)
        value = {"id": self.id, "type": "function", "function": {
            "name": self.name, "arguments": arguments}}
        if self.extra_content is not None:
            value["extra_content"] = self.extra_content
        return value


def response_calls(response) -> list[ToolCall]:
    """AstrBot can replace malformed JSON with {}; prefer the retained wire data."""
    choices = field(field(response, "raw_completion"), "choices", []) or []
    raw = field(field(choices[0], "message"), "tool_calls") if choices else None
    if raw is None:
        names = field(response, "tools_call_name", []) or []
        arguments = field(response, "tools_call_args", []) or []
        ids = field(response, "tools_call_ids", []) or []
        if not (len(names) == len(arguments) == len(ids)):
            raise ValueError("Provider returned inconsistent tool call IDs/arguments")
        extras = field(response, "tools_call_extra_content", {}) or {}
        raw = [{"id": id, "function": {"name": name, "arguments": args},
                "extra_content": extras.get(id)} for id, name, args in zip(ids, names, arguments)]
    calls = []
    for value in raw:
        if isinstance(value, str):
            value = json.loads(value)
        function = field(value, "function")
        name = str(field(function, "name", ""))
        arguments = field(function, "arguments")
        wire = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
        error, format_note = "", ""
        try:
            parsed, format_note = (decode_background(wire) if name == "complete" else decode_envelope(wire))
            if not isinstance(parsed, dict):
                raise ValueError("Tool arguments must be a JSON object")
        except (ValueError, TypeError) as exc:
            parsed = None
            error = f"工具参数未执行：{exc}。上一条调用保留了完整原始参数；请修正 JSON 后重发此调用，已成功的其他调用不用重做。"
        calls.append(ToolCall(str(field(value, "id", "")), name,
                              wire, parsed, error, field(value, "extra_content"), format_note))
    if any(not call.id or not call.name for call in calls) or len({call.id for call in calls}) != len(calls):
        raise ValueError("Provider returned missing or repeated tool call IDs/names")
    return calls


async def parse_completion(parser, completion, tools):
    """Normalize this MR response before AstrBot parses it; retain original wire."""
    calls = response_calls({"raw_completion": completion})
    repaired = {call.id: call for call in calls if call.format_note and call.arguments is not None}
    prepared = completion
    if repaired:
        prepared = copy.deepcopy(completion)
        raw = field(field(field(prepared, "choices")[0], "message"), "tool_calls")
        for index, value in enumerate(raw):
            if isinstance(value, str):
                value = raw[index] = json.loads(value)
            if call := repaired.get(str(field(value, "id", ""))):
                function = field(value, "function")
                arguments = call.message()["function"]["arguments"]
                if isinstance(function, dict):
                    function["arguments"] = arguments
                else:
                    function.arguments = arguments
    response = await parser(prepared, tools)
    response.raw_completion = completion
    return response
