"""Decode model envelopes without rewriting their field names or values."""
from __future__ import annotations

import json
import re


_CLOSING_TAGS = re.compile(r"(?:\s*</(?:[｜|]+DSML[｜|]+\s*)?(?:parameter|invoke|calls|tool_calls|function_calls|tool)>\s*)+")


def _unfence(text: str):
    value = text.strip()
    if value.startswith("```"):
        value = value[7:] if value.startswith("```json") else value[3:]
        value = value.strip()
        if value.endswith("```"):
            value = value[:-3].rstrip()
    return value


def decode_envelope(text: str):
    """Return (value, format note); ambiguous or incomplete values still fail."""
    value = _unfence(text)
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


def decode_background(text: str):
    """Background is prose, not a database write; accept lossless text envelopes."""
    value = _unfence(text)
    try:
        result, note = decode_envelope(value)
    except ValueError:
        # Native tool conversion sometimes leaves the prose field unquoted.
        # Find a fully parseable suffix, retaining every following field. Never
        # fill in an incomplete value or absorb nested JSON into the prose.
        match = re.match(r'\{\s*"background"\s*:\s*([^\s"\[{])', value)
        if match:
            tail = value[match.start(1):]
            boundaries = [m.start() for m in re.finditer(r',\s*"[^"\n]+"\s*:', tail)]
            boundaries.append(tail.rfind("}"))
            for end in boundaries:
                if end < 0:
                    continue
                body = tail[:end].rstrip()
                if not body or re.search(r'[{}]|,\s*"[^"\n]+"\s*:', body):
                    continue
                try:
                    result, _ = decode_envelope('{"background":' + json.dumps(body, ensure_ascii=False) + tail[end:])
                except ValueError:
                    continue
                return result, "读取未加引号的背景正文；正文及其余完整字段未改写"
        if value and value[0] not in '{["':
            return {"background": value}, "读取纯文本背景，正文未改写"
        raise
    if isinstance(result, (str, list)):
        return {"background": result}, note or "读取直接交付的背景内容"
    if isinstance(result, dict):
        if "background" not in result and "text" in result:
            return {"background": result}, note or "读取直接交付的背景段落"
        return result, note
    raise ValueError("Background must contain text or text paragraphs")
