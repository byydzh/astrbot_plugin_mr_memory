"""Host-owned identity references in generated narrative, never raw quotations.

Offsets address the unchanged Python string. A binding records which account a
batch-local token denoted; it does not prove the model's proposition about it.
"""

from __future__ import annotations

import hashlib
import copy
import re
from collections.abc import Iterable, Mapping


NARRATIVE_BINDINGS_SCHEMA = "mr-memory.narrative-bindings.v1"
_ALIAS_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])p[0-9]+(?![A-Za-z0-9])")


def participant_alias_tokens(value: object) -> set[str]:
    """Find labels delimited by prose or underscores, never inside words/numbers.

    Reservation, persisted offsets, and fingerprint copies use this same rule so
    copied input such as ``field_p1`` cannot acquire a new batch identity.
    """
    if isinstance(value, str):
        return set(_ALIAS_TOKEN_RE.findall(value))
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            found.update(participant_alias_tokens(str(key)))
            found.update(participant_alias_tokens(nested))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            found.update(participant_alias_tokens(nested))
    return found


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_narrative_fingerprint_text(
    text: str, alias_to_participant: Mapping[str, str],
) -> str:
    """Normalize a fingerprint copy, leaving persisted text and quotes intact."""
    return _ALIAS_TOKEN_RE.sub(
        lambda match: ("[participant:" + alias_to_participant[match.group()] + "]"
                       if match.group() in alias_to_participant else match.group()),
        text,
    )


def _envelope(fields: dict[str, object]) -> dict[str, object]:
    return {"schema_version": NARRATIVE_BINDINGS_SCHEMA, "fields": fields}


def build_narrative_bindings(
    fields: Mapping[str, str], alias_to_participant: Mapping[str, str],
) -> dict[str, object]:
    """Bind only generated fields explicitly selected by the persistence host.

    The caller must allocate aliases outside tokens already in the extraction
    input. Consequently copied raw quotes cannot acquire a new batch identity.
    """
    bound: dict[str, object] = {}
    for field, text in fields.items():
        references = [
            {"start": match.start(), "end": match.end(), "token": match.group(),
             "participant_key": alias_to_participant[match.group()]}
            for match in _ALIAS_TOKEN_RE.finditer(text)
            if match.group() in alias_to_participant
        ]
        if references:
            bound[field] = {"text_sha256": _text_sha256(text), "references": references}
    return _envelope(bound)


def matching_narrative_bindings(
    fields: Mapping[str, str], value: object,
) -> dict[str, object]:
    """Return only whole fields whose text hash and every reference still match.

    Missing, stale, or malformed metadata has no identity authority. Callers keep
    the narrative and mark its remaining local tokens as unmapped.
    """
    accepted: dict[str, object] = {}
    if not isinstance(value, Mapping) or value.get("schema_version") != NARRATIVE_BINDINGS_SCHEMA:
        return _envelope(accepted)
    raw_fields = value.get("fields")
    if not isinstance(raw_fields, Mapping):
        return _envelope(accepted)
    for field, text in fields.items():
        raw = raw_fields.get(field)
        if not isinstance(raw, Mapping) or raw.get("text_sha256") != _text_sha256(text):
            continue
        references = raw.get("references")
        if not isinstance(references, list) or not references or len(references) > len(text):
            continue
        tokens = {(m.start(), m.end(), m.group()) for m in _ALIAS_TOKEN_RE.finditer(text)}
        checked: list[dict[str, object]] = []
        last_end = -1
        for reference in references:
            if not isinstance(reference, Mapping) or set(reference) != {"start", "end", "token", "participant_key"}:
                break
            start, end = reference["start"], reference["end"]
            token, participant = reference["token"], reference["participant_key"]
            if (type(start) is not int or type(end) is not int or not isinstance(token, str)
                    or not isinstance(participant, str) or not participant or len(participant) > 256
                    or start < last_end or (start, end, token) not in tokens):
                break
            checked.append(dict(reference))
            last_end = end
        else:
            accepted[field] = {"text_sha256": raw["text_sha256"], "references": checked}
    return _envelope(accepted)


def project_narrative_record(value: Mapping[str, object]) -> dict[str, object]:
    """Project one known derived record, keeping raw evidence children intact."""
    item = copy.deepcopy(dict(value))
    item["reader_evidence_basis"] = "STORED_DERIVATION"
    item["reader_text_identity_scope"] = "HISTORICAL_UNMAPPED"
    narrative_fields = {
        "title", "summary", "content", "aspect_tag", "person_cue", "subject_text", "name", "statement", "uncertainty",
        "source_label", "source_description", "target_label", "target_description",
        "relation_name", "relation_description", "description", "canonical_name",
    }
    fields = {key: text for key, text in item.items()
              if key in narrative_fields and isinstance(text, str)}
    bindings = matching_narrative_bindings(fields, item.get("narrative_bindings"))
    if bindings["fields"]:
        item["narrative_bindings"] = bindings
        item["reader_text_identity_scope"] = "EXPLICIT_POSITIONS_ONLY"
    else:
        item.pop("narrative_bindings", None)
    withheld = []
    for field, text in fields.items():
        if field not in bindings["fields"] and participant_alias_tokens(text):
            item.pop(field)
            withheld.append({"field": field, "text_sha256": _text_sha256(text),
                             "reason": "UNMAPPED_HISTORICAL_PARTICIPANT_TOKEN"})
    if withheld:
        item["reader_withheld_fields"] = withheld
    return item


def compose_narrative_summary(
    parts: Iterable[tuple[str, object]], *, separator: str = "；", max_chars: int = 4000,
) -> tuple[str, dict[str, object]]:
    """Carry character bindings across topic concatenation and suffix bounds."""
    if max_chars <= 0:
        raise ValueError("narrative summary max_chars must be positive")
    text_parts: list[str] = []
    references: list[dict[str, object]] = []
    offset = 0
    for text, metadata in parts:
        if not text:
            continue
        if text_parts:
            offset += len(separator)
        matching = matching_narrative_bindings({"summary": text}, metadata)
        for reference in matching["fields"].get("summary", {}).get("references", []):
            references.append({**reference, "start": reference["start"] + offset,
                               "end": reference["end"] + offset})
        text_parts.append(text)
        offset += len(text)
    text = separator.join(text_parts)
    removed = max(0, len(text) - max_chars)
    text = text[removed:]
    references = [{**reference, "start": reference["start"] - removed,
                   "end": reference["end"] - removed}
                  for reference in references if reference["start"] >= removed]
    metadata = _envelope({"summary": {"text_sha256": _text_sha256(text), "references": references}})
    return text, matching_narrative_bindings({"summary": text}, metadata)
