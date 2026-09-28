"""Decode model envelopes without rewriting their field names or values."""
from __future__ import annotations

import json
import re


_CLOSING_TAGS = re.compile(r"(?:\s*</(?:[｜|]+DSML[｜|]+\s*)?(?:parameter|invoke|calls|tool_calls|function_calls|tool)>\s*)+")


def decode_envelope(text: str):
    """Return (value, format note); ambiguous or incomplete values still fail."""
    value = text.strip()
    if value.startswith("```"):
        value = value[7:] if value.startswith("```json") else value[3:]
        value = value.strip()
        if value.endswith("```"):
            value = value[:-3].rstrip()
    decoder = json.JSONDecoder(strict=False)
    try:
        result, end = decoder.raw_decode(value)
    except json.JSONDecodeError as exc:
        # Only the outside object delimiter may be missing. A cut-off string,
        # array or field is never filled in, even when the provider says stop.
        if value.startswith("{") and exc.pos == len(value):
            try:
                return json.loads(value + "}", strict=False), "补齐最外层对象括号，字段和值未改动"
            except json.JSONDecodeError:
                pass
        raise
    tail = value[end:].strip()
    if tail.startswith("```"):
        tail = tail[3:].strip()
    if not tail:
        return result, ""
    # DSML wrappers sometimes leak through the API's tool/text conversion.
    # Accept closing wrappers only; never ignore a second object or prose.
    if tail == "}" or _CLOSING_TAGS.fullmatch(tail) or tail in {"</tool", "</tool>"}:
        return result, "读取完整 JSON，忽略多余的外层闭合符；原始输出仍保留"
    raise ValueError("Unexpected content after the complete JSON envelope")
