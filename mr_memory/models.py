from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal


MessageRole = Literal["USER", "BOT", "SYSTEM"]


@dataclass(frozen=True, slots=True)
class NormalizedMessage:
    """Stable message shape shared by AstrBot ingestion and offline replay."""

    platform: str
    platform_id: str
    umo: str
    group_id: str
    message_id: str
    sender_id: str
    sender_name: str
    sent_at: int
    plain_text: str
    content: list[dict[str, Any]] = field(default_factory=list)
    role: MessageRole = "USER"
    source_key: str = ""

    def resolved_source_key(self) -> str:
        if self.source_key.strip():
            return self.source_key.strip()
        if self.message_id.strip():
            return "|".join(
                (self.platform_id, self.umo, self.message_id.strip())
            )
        fallback = json.dumps(
            {
                "platform_id": self.platform_id,
                "umo": self.umo,
                "sender_id": self.sender_id,
                "sent_at": self.sent_at,
                "plain_text": self.plain_text,
                "content": self.content,
                "role": self.role,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(fallback).hexdigest()}"

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "NormalizedMessage":
        content = value.get("content") or []
        if not isinstance(content, list):
            raise ValueError("content must be a list")
        role = str(value.get("role") or "USER").upper()
        if role not in {"USER", "BOT", "SYSTEM"}:
            raise ValueError(f"unsupported role: {role}")
        return cls(
            platform=str(value.get("platform") or "unknown"),
            platform_id=str(value.get("platform_id") or "default"),
            umo=str(value.get("umo") or ""),
            group_id=str(value.get("group_id") or ""),
            message_id=str(value.get("message_id") or ""),
            sender_id=str(value.get("sender_id") or ""),
            sender_name=str(value.get("sender_name") or ""),
            sent_at=int(value.get("sent_at") or 0),
            plain_text=str(value.get("plain_text") or ""),
            content=content,
            role=role,  # type: ignore[arg-type]
            source_key=str(value.get("source_key") or ""),
        )


@dataclass(frozen=True, slots=True)
class StoredMessage:
    id: int
    source_key: str
    platform: str
    platform_id: str
    umo: str
    group_id: str
    message_id: str
    sender_id: str
    sender_name: str
    sent_at: int
    plain_text: str
    content: list[dict[str, Any]]
    role: MessageRole
