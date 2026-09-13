"""Readable views of archived content; binary media stays in its original record."""
from __future__ import annotations

import re
from typing import Any


_INLINE_MEDIA = re.compile(
    r"(?:base64://|data:[^,\s]*;base64,)[A-Za-z0-9+/\r\n]*={0,2}", re.IGNORECASE
)


def searchable_text(value: str) -> str:
    """Encoded image/audio bytes are not words, including inside tool JSON."""
    return _INLINE_MEDIA.sub("", value)


def text_view(value: Any) -> Any:
    """Preserve prose, structure and media URLs without spelling out media bytes."""
    if isinstance(value, str):
        return _INLINE_MEDIA.sub(
            lambda match: f"[二进制附件编码，{len(match.group())}字符；保留在原消息中，未作为文字提供]", value
        )
    if isinstance(value, list):
        return [text_view(item) for item in value]
    if isinstance(value, dict):
        return {key: text_view(item) for key, item in value.items()}
    return value
