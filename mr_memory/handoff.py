"""Carry model-selected observations with an interpretation, without judging it."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from .content import text_view


def context_ref(ref):
    if not isinstance(ref, dict):
        raise ValueError("A selected context address must be an object")
    if "message_id" in ref and "kind" not in ref and "id" not in ref:
        return {"kind": "message", "id": int(ref["message_id"])}
    if "kind" not in ref or "id" not in ref:
        raise ValueError("Use message_id or kind/id for selected context")
    return {"kind": str(ref["kind"]), "id": int(ref["id"])}


def selected_context(store, ref, before=None):
    kind, owner = str(ref["kind"]), int(ref["id"])
    address = {"kind": kind, "id": owner}
    if kind == "message":
        rows = store.messages([owner])
        row = rows[0] if rows else None
        if row is not None and before is not None and row["sent_at"] > before:
            row = None
        if row is None:
            return {**address, "status": "unavailable"}
        keys = ("sender_id", "sender_name", "role", "sent_at", "plain_text", "content", "reply_to_message")
        return {**address, **{key: text_view(row[key]) for key in keys if key in row},
                "sent_at_local": datetime.fromtimestamp(row["sent_at"], timezone.utc).astimezone().isoformat(),
                "meaning": "记录中的原始交流；话语内容及其含义仍需结合语境理解。"}
    row = store.memory(kind, owner, include_sources=False)
    if row is None:
        return {**address, "status": "unavailable"}
    keys = ("title", "revision_no", "content", "representation", "belief", "source_ids", "source_speakers", "basis")
    return {**address, **{key: text_view(row[key]) for key in keys if key in row},
            "meaning": "记忆网络中当前的理解，包含模型形成的解释，并非另一份独立观察。"}


def format_background(value, store=None, before=None):
    if isinstance(value, str):
        text = value.strip()
        # Native output may serialize this optional structured value once more.
        # Decode only the advertised interpretation shape; ordinary text stays text.
        try:
            decoded = json.loads(text)
        except ValueError:
            return text
        parts = decoded if isinstance(decoded, list) else [decoded]
        if not parts or not all(isinstance(part, dict) and isinstance(part.get("text"), str)
                                and ("belief" in part or "references" in part) for part in parts):
            return text
        value = parts
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        raise ValueError("background needs text or text/belief/references interpretations")
    parts, selected, unavailable = [], {}, []
    for part in value:
        if not isinstance(part, dict) or not isinstance(part.get("text", ""), str):
            raise ValueError("Each background interpretation needs text")
        refs = part.get("references", [])
        if not isinstance(refs, list):
            raise ValueError("background.references must be a list")
        addresses = []
        for ref in refs:
            try:
                addresses.append(context_ref(ref))
            except (TypeError, ValueError) as exc:
                # An optional address failure must not erase the interpretation
                # or other usable observations. Preserve the failure explicitly.
                failed = {"requested": text_view(ref), "status": "invalid_address", "detail": str(exc)}
                addresses.append(failed)
                unavailable.append(failed)
        refs = addresses
        body = part.get("text", "").strip()
        if body:
            belief = part.get("belief") or {"stance": "unconfirmed"}
            parts.append("当前理解与把握：" + json.dumps(belief, ensure_ascii=False, separators=(",", ":"))
                         + "\n" + body + ("\n对应交流与记忆：" + json.dumps(refs, ensure_ascii=False, separators=(",", ":")) if refs else ""))
        for ref in refs:
            if ref.get("status") == "invalid_address":
                continue
            if store is None:
                raise ValueError("Selected background references need the group store")
            key = (str(ref["kind"]), int(ref["id"]))
            if key not in selected:
                selected[key] = selected_context(store, ref, before)
    if selected or unavailable:
        parts.append("MR 选择同时带回的交流与记忆（相同来源只呈现一次）：\n"
                     + json.dumps([*selected.values(), *unavailable], ensure_ascii=False, separators=(",", ":")))
    return "\n\n".join(parts)
