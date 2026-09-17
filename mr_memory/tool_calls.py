"""Keep the provider's original tool arguments, including unsuccessful writes."""
from __future__ import annotations

from dataclasses import dataclass
import json
import re


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
    if not arguments or set(arguments) - {"items", "progress", "retry", "finish"}:
        raise ValueError("DSML remember accepts items, progress, retry and finish")
    return arguments


@dataclass
class ToolCall:
    id: str
    name: str
    raw_arguments: str
    arguments: dict | None
    error: str = ""
    extra_content: object = None

    def message(self):
        value = {"id": self.id, "type": "function", "function": {
            "name": self.name, "arguments": self.raw_arguments}}
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
        arguments = field(function, "arguments")
        wire = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
        error = ""
        try:
            parsed = json.loads(wire)
            if not isinstance(parsed, dict):
                raise ValueError("Tool arguments must be a JSON object")
        except (ValueError, TypeError) as exc:
            parsed = None
            error = f"工具参数未执行：{exc}。上一条调用保留了完整原始参数；请修正 JSON 后重发此调用，已成功的其他调用不用重做。"
        calls.append(ToolCall(str(field(value, "id", "")), str(field(function, "name", "")),
                              wire, parsed, error, field(value, "extra_content")))
    if any(not call.id or not call.name for call in calls) or len({call.id for call in calls}) != len(calls):
        raise ValueError("Provider returned missing or repeated tool call IDs/names")
    return calls
