from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable


def normalize_alias(value: object) -> str:
    """Normalize a display name for exact, scope-local identity lookup.

    Aliases are deliberately not reduced to substrings.  Two accounts may use the
    same nickname in a group, and an ambiguous alias must stay ambiguous instead of
    silently merging their memories.
    """

    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    text = text.removeprefix("@").strip()
    return re.sub(r"\s+", " ", text)


def canonical_participant_key(platform_id: str, account_id: str) -> str:
    platform = str(platform_id or "").strip()
    account = str(account_id or "").strip()
    if not platform or not account:
        raise ValueError("participant platform_id and account_id are required")
    # JSON avoids delimiter ambiguity while keeping the value readable in evidence.
    return "participant:" + json.dumps(
        [platform, account], ensure_ascii=False, separators=(",", ":")
    )


def content_fingerprint(
    *,
    sender_id: str,
    role: str,
    plain_text: str,
    content: Iterable[dict[str, Any]],
) -> str:
    payload = json.dumps(
        {
            "sender_id": str(sender_id),
            "role": str(role).upper(),
            "plain_text": str(plain_text),
            "content": list(content),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _component_type(component: dict[str, Any]) -> str:
    raw = component.get("type") or component.get("component_type") or ""
    text = str(getattr(raw, "value", raw) or "").strip().casefold()
    aliases = {
        "at": "mention",
        "mention": "mention",
        "reply": "reply",
        "quote": "reply",
        "plain": "text",
        "plaintext": "text",
    }
    return aliases.get(text, text)


def _safe_component_kind(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    text = re.sub(r"[^a-z0-9_.-]+", "_", text).strip("_.-")
    return text[:80] or "unknown"


def _safe_attachment_name(value: object) -> str:
    """Retain a display filename without leaking a local path or URL query."""

    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    text = re.sub(r"[\x00-\x1f\x7f]", "", text).replace("\\", "/")
    text = text.rsplit("/", 1)[-1].split("?", 1)[0].split("#", 1)[0]
    return text[:300]


def _first(component: dict[str, Any], *names: str) -> Any:
    data = component.get("data")
    for name in names:
        value = component.get(name)
        if value not in (None, "", 0, "0"):
            return value
        if isinstance(data, dict):
            value = data.get(name)
            if value not in (None, "", 0, "0"):
                return value
    return ""


@dataclass(frozen=True, slots=True)
class MentionReference:
    account_id: str
    display_name: str = ""


@dataclass(frozen=True, slots=True)
class ReplyReference:
    message_id: str
    relation: str = "REPLY_TO"
    sender_id: str = ""
    sender_name: str = ""
    sent_at: int = 0
    plain_text: str = ""


def extract_mentions(content: Iterable[dict[str, Any]]) -> tuple[MentionReference, ...]:
    mentions: dict[str, MentionReference] = {}
    for component in content:
        if not isinstance(component, dict) or _component_type(component) != "mention":
            continue
        account_id = str(_first(component, "account_id", "qq", "user_id")).strip()
        if not account_id or account_id.casefold() == "all":
            continue
        display_name = str(
            _first(component, "display_name", "name", "nickname")
        ).strip()
        mentions[account_id] = MentionReference(account_id, display_name)
    return tuple(mentions.values())


def extract_reply(content: Iterable[dict[str, Any]]) -> ReplyReference | None:
    for component in content:
        if not isinstance(component, dict):
            continue
        kind = _component_type(component)
        if kind not in {"reply", "response_to"}:
            continue
        message_id = str(_first(component, "message_id", "id")).strip()
        if not message_id:
            continue
        sent_at_raw = _first(component, "sent_at", "time")
        try:
            sent_at = int(sent_at_raw or 0)
        except (TypeError, ValueError):
            sent_at = 0
        return ReplyReference(
            message_id=message_id,
            relation=("RESPONDS_TO" if kind == "response_to" else "REPLY_TO"),
            sender_id=str(
                _first(component, "sender_id", "account_id", "qq")
            ).strip(),
            sender_name=str(
                _first(component, "sender_name", "sender_nickname", "nickname")
            ).strip(),
            sent_at=sent_at,
            plain_text=str(
                _first(component, "plain_text", "message_str", "text")
            ).strip(),
        )
    return None


def attachment_metadata(content: Iterable[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Return bounded attachment descriptors; never retain blobs or credentials."""

    result: list[dict[str, Any]] = []
    attachment_types = {"image", "file", "video", "record", "audio", "forward"}
    for index, component in enumerate(content):
        if not isinstance(component, dict):
            continue
        kind = _component_type(component)
        if kind not in attachment_types:
            continue
        existing_hash = str(component.get("reference_sha256") or "").strip()
        raw_ref = str(_first(component, "url", "file", "path", "id")).strip()
        # URLs may contain signed query parameters.  Persist only a digest.
        ref_hash = (
            existing_hash
            if re.fullmatch(r"[0-9a-fA-F]{64}", existing_hash)
            else (
                hashlib.sha256(raw_ref.encode("utf-8")).hexdigest()
                if raw_ref
                else ""
            )
        )
        name = _safe_attachment_name(
            _first(component, "name", "file_name", "title")
        )
        result.append(
            {
                "position": index,
                "kind": kind,
                "name": name,
                "reference_sha256": ref_hash,
            }
        )
    return tuple(result)


def sanitize_components(
    content: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Canonicalize components at the storage boundary using a strict allowlist."""

    result: list[dict[str, Any]] = []
    attachment_types = {
        "image", "file", "video", "record", "audio", "forward", "node", "nodes",
    }
    for raw in content:
        if not isinstance(raw, dict):
            continue
        kind = _component_type(raw) or "unknown"
        stored_kind = _safe_component_kind(kind)
        if kind == "text":
            result.append(
                {
                    "type": "text",
                    "text": str(_first(raw, "text", "content"))[:8000],
                }
            )
            continue
        if kind == "mention":
            item: dict[str, Any] = {
                "type": "mention",
                "account_id": str(
                    _first(raw, "account_id", "qq", "user_id")
                )[:300],
                "display_name": str(
                    _first(raw, "display_name", "name", "nickname")
                )[:300],
            }
            if raw.get("erased_participant"):
                item = {"type": "mention", "erased_participant": True}
            result.append(item)
            continue
        if kind in {"reply", "response_to"}:
            sent_at = _first(raw, "sent_at", "time")
            try:
                parsed_sent_at = int(sent_at or 0)
            except (TypeError, ValueError):
                parsed_sent_at = 0
            item = {
                "type": kind,
                "message_id": str(
                    _first(raw, "message_id", "reply_id", "id")
                )[:500],
                "sender_id": str(
                    _first(raw, "sender_id", "account_id", "qq")
                )[:300],
                "sender_name": str(
                    _first(raw, "sender_name", "sender_nickname", "nickname")
                )[:300],
                "sent_at": parsed_sent_at,
                "plain_text": str(
                    _first(raw, "plain_text", "message_str", "text")
                )[:4000],
            }
            if raw.get("erased_participant"):
                item = {
                    "type": kind,
                    "message_id": item["message_id"],
                    "erased_participant": True,
                }
            result.append(item)
            continue
        if kind in attachment_types:
            raw_ref = str(_first(raw, "url", "file", "path", "id"))
            existing_hash = str(raw.get("reference_sha256") or "")
            reference_hash = (
                existing_hash.lower()
                if re.fullmatch(r"[0-9a-fA-F]{64}", existing_hash)
                else (
                    hashlib.sha256(raw_ref.encode("utf-8")).hexdigest()
                    if raw_ref
                    else ""
                )
            )
            result.append(
                {
                    "type": stored_kind,
                    "name": _safe_attachment_name(
                        _first(raw, "name", "file_name", "title")
                    ),
                    "reference_sha256": reference_hash,
                }
            )
            continue
        # Unknown adapter components are not evidence until an explicit parser is
        # added. Keeping arbitrary scalar fields here would reintroduce signed URLs,
        # local paths, credentials, or blobs through a new component type.
        result.append({"type": stored_kind})
    return result
