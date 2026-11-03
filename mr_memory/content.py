"""Readable views of archived content; binary media stays in its original record."""
from __future__ import annotations

import json
import re
from typing import Any


_INLINE_MEDIA = re.compile(
    r"(?:base64://|data:[^,\s]*;base64,)[A-Za-z0-9+/\r\n]*={0,2}", re.IGNORECASE
)


def _attachment_summary(data: str) -> str:
    if not data or data.startswith("[二进制附件编码，"):
        return data
    return f"[二进制附件编码，{len(data)}字符；保留在原消息中，未作为文字提供]"


def searchable_text(value: str) -> str:
    """Encoded image/audio bytes are not words, including inside tool JSON."""
    return _INLINE_MEDIA.sub("", text_view(value))


def text_view(value: Any) -> Any:
    """Preserve prose, structure and media URLs without spelling out media bytes."""
    if isinstance(value, str):
        # Tool results can be archived as either MCP objects or JSON strings.
        # Decode only structured JSON; ordinary prose and unchanged JSON keep
        # their original spelling and whitespace.
        if value.lstrip().startswith(("{", "[")):
            try:
                decoded = json.loads(value)
            except (ValueError, RecursionError):
                pass
            else:
                viewed = text_view(decoded)
                if viewed != decoded:
                    return json.dumps(viewed, ensure_ascii=False)
        return _INLINE_MEDIA.sub(
            lambda match: _attachment_summary(match.group()), value
        )
    if isinstance(value, list):
        return [text_view(item) for item in value]
    if isinstance(value, dict):
        # MCP ImageContent/AudioContent carry *bare* base64 in data, while
        # BlobResourceContents uses blob. Neither has a data: URL prefix.
        binary_key = ("data" if value.get("type") in ("image", "audio")
                      else "blob" if "uri" in value and "blob" in value else None)
        return {key: (_attachment_summary(item)
                      if key == binary_key and isinstance(item, str) else text_view(item))
                for key, item in value.items()}
    return value
