"""Carry model-selected observations with an interpretation, without judging it."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from .content import text_view


def request_clock(now, sent_at):
    def times(timestamp):
        utc = datetime.fromtimestamp(timestamp, timezone.utc)
        return {"local": utc.astimezone().isoformat(), "utc": utc.isoformat()}
    return {"now": times(now), "message": times(sent_at),
            "meaning": "local 是服务环境当地时间，含 UTC 偏移；utc 是同一时刻的 UTC 时间。"}


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
        keys = ("sender_id", "sender_name", "role", "plain_text", "content", "reply_to_message")
        result = {**address, **{key: text_view(row[key]) for key in keys if key in row},
                  "sent_at_local": datetime.fromtimestamp(row["sent_at"], timezone.utc).astimezone().isoformat()}
        body = result.get('plain_text')
        if body and isinstance(result.get('content'), list):
            result['content'] = [p for p in result['content'] if not (
                isinstance(p, dict) and p.get('type') in ('text', 'plain') and p.get('text') == body)]
            if not result['content']:
                result.pop('content')
        return result
    row = store.memory(kind, owner, include_sources=False)
    if row is None:
        return {**address, "status": "unavailable"}
    # Message IDs and revisions are retrieval bookkeeping, not reply context.
    # The addressed memory and full provenance remain available in the store.
    keys = ("title", "content", "representation", "belief", "source_speakers", "source_contexts", "basis")
    return {**address, **{key: text_view(row[key]) for key in keys if key in row}}


def source_packet(sources, known_people=()):
    """Share author identities while retaining each source's participants/roles."""
    known, canonical, owners = {}, {}, {}
    for person in known_people:
        uid = str(person['uid'])
        names = {person.get('name'), person.get('call'), *person.get('names', []), *person.get('aliases', [])} - {None, ''}
        for account in (uid, *person.get('ids', [])):
            canonical[str(account)] = uid
            known.setdefault(str(account), set()).update(names)
        for name in names:
            owners.setdefault(name, set()).add(uid)
    for source in sources:
        if source.get('kind') == 'message' and source.get('sender_name'):
            uid = str(source['sender_id'])
            owners.setdefault(source['sender_name'], set()).add(canonical.get(uid, uid))
        for speaker in source.get('source_speakers', []):
            uid = str(speaker['account_id'])
            name = speaker.get('name_at_message')
            if name:
                owners.setdefault(name, set()).add(canonical.get(uid, uid))
    people, records = {}, []
    for source in sources:
        record = dict(source)
        speakers = record.pop("source_speakers", [])
        authors = {}
        for speaker in speakers:
            uid = str(speaker["account_id"])
            name = speaker.get("name_at_message")
            if name and name not in known.get(uid, ()):
                names = people.setdefault(uid, [])
                if name not in names:
                    names.append(name)
            role = speaker.get("role", "USER")
            authors.setdefault(role, [])
            # Reuse a readable name only when the supplied directory resolves it
            # to one account. Same-name users keep their explicit UID here.
            author = (name if name and len(owners[name]) == 1 else
                      {"uid": uid, "name": name} if name else {"uid": uid})
            if author not in authors[role]:
                authors[role].append(author)
        if authors:
            record["speakers"] = authors
        records.append(record)
    return {**({"people": people} if people else {}), "sources": records}


def format_background(value, store=None, before=None, *, references=None, belief=None, known_people=()):
    if isinstance(value, str):
        text = value.strip()
        # Native output may serialize this optional structured value once more.
        # Decode only the advertised interpretation shape; ordinary text stays text.
        try:
            decoded = json.loads(text)
        except ValueError:
            decoded = None
        parts = decoded if isinstance(decoded, list) else [decoded]
        value = (parts if parts and all(isinstance(part, str) or isinstance(part, dict)
                 and isinstance(part.get("text"), str) for part in parts) else [text])
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        raise ValueError("background needs text or text/belief/references interpretations")
    if references is not None:
        value = [*value, {"text": "", "references": references}]
    parts, selected, unavailable = [], {}, []
    if belief is not None:
        parts.append("当前理解与把握：" + json.dumps(belief, ensure_ascii=False, separators=(",", ":")))
    for part in value:
        if isinstance(part, str):
            part = {"text": part}
        if not isinstance(part, dict) or not isinstance(part.get("text", ""), str):
            raise ValueError("Each background interpretation needs text")
        refs = part.get("references", [])
        if refs is None:
            refs = []
        elif not isinstance(refs, list):
            refs = [refs]
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
            belief = part.get("belief")
            prefix = ("当前理解与把握：" + json.dumps(belief, ensure_ascii=False, separators=(",", ":"))
                      + "\n") if belief is not None else ""
            parts.append(prefix + body + ("\n对应交流与记忆：" + json.dumps(refs, ensure_ascii=False, separators=(",", ":")) if refs else ""))
        for ref in refs:
            if ref.get("status") == "invalid_address":
                continue
            if store is None:
                raise ValueError("Selected background references need the group store")
            key = (str(ref["kind"]), int(ref["id"]))
            if key not in selected:
                selected[key] = selected_context(store, ref, before)
    if selected or unavailable:
        packet = source_packet([*selected.values(), *unavailable], known_people)
        parts.append("所选来源（message为原话，其余为记忆理解）：\n"
                     + json.dumps(packet if packet.get("people") else packet["sources"], ensure_ascii=False, separators=(",", ":")))
    return "\n\n".join(parts)
