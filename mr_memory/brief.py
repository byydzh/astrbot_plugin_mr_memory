from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class EvidenceClaim:
    statement: str
    source_keys: tuple[str, ...]
    confidence: float


@dataclass(frozen=True, slots=True)
class EvidenceQualification:
    statement: str
    source_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvidenceBrief:
    claims: tuple[EvidenceClaim, ...]
    conflicts: tuple[EvidenceQualification, ...]
    unresolved: tuple[EvidenceQualification, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "claims": [
                {
                    "statement": claim.statement,
                    "source_keys": list(claim.source_keys),
                    "confidence": claim.confidence,
                }
                for claim in self.claims
            ],
            "conflicts": [
                {
                    "statement": item.statement,
                    "source_keys": list(item.source_keys),
                }
                for item in self.conflicts
            ],
            "unresolved": [
                {
                    "statement": item.statement,
                    "source_keys": list(item.source_keys),
                }
                for item in self.unresolved
            ],
        }


def _extract_object(text: str) -> dict[str, Any]:
    stripped = str(text or "").strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("memory brief is not a JSON object")
    try:
        value = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError("memory brief contains invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("memory brief must be an object")
    return value


def _source_keys(
    value: object,
    *,
    field: str,
    allowed: set[str],
) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or len(value) > 32:
        raise ValueError(f"{field} must contain 1..32 source keys")
    result = tuple(dict.fromkeys(str(item or "").strip() for item in value))
    if any(not item or item not in allowed for item in result):
        raise ValueError(f"{field} cites unvisited evidence")
    return result


def _grounded_qualifications(
    value: object,
    field: str,
    *,
    limit: int,
    allowed: set[str],
) -> tuple[EvidenceQualification, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or len(value) > limit:
        raise ValueError(f"{field} must be an array with at most {limit} items")
    result: list[EvidenceQualification] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{field}[{index}] must be an object")
        statement = str(item.get("statement") or "").strip()
        if not statement or len(statement) > 1000:
            raise ValueError(f"{field}[{index}].statement is invalid")
        result.append(
            EvidenceQualification(
                statement=statement,
                source_keys=_source_keys(
                    item.get("source_keys"),
                    field=f"{field}[{index}].source_keys",
                    allowed=allowed,
                ),
            )
        )
    return tuple(result)


def parse_evidence_brief(
    text: str,
    *,
    allowed_source_keys: Iterable[str],
) -> EvidenceBrief | None:
    if str(text or "").strip() == "NO_RELEVANT_MEMORY":
        return None
    allowed = {str(item) for item in allowed_source_keys if str(item)}
    value = _extract_object(text)
    raw_claims = value.get("claims")
    if not isinstance(raw_claims, list) or len(raw_claims) > 32:
        raise ValueError("memory brief claims must be an array of at most 32 items")
    claims: list[EvidenceClaim] = []
    for index, raw in enumerate(raw_claims):
        if not isinstance(raw, dict):
            raise ValueError(f"memory brief claims[{index}] must be an object")
        statement = str(raw.get("statement") or "").strip()
        if not statement or len(statement) > 2000:
            raise ValueError(f"memory brief claims[{index}].statement is invalid")
        source_keys = _source_keys(
            raw.get("source_keys"),
            field=f"memory brief claims[{index}].source_keys",
            allowed=allowed,
        )
        confidence = float(raw.get("confidence", 0.0))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"memory brief claims[{index}].confidence must be 0..1")
        claims.append(EvidenceClaim(statement, source_keys, confidence))
    conflicts = _grounded_qualifications(
        value.get("conflicts", []),
        "conflicts",
        limit=16,
        allowed=allowed,
    )
    unresolved = _grounded_qualifications(
        value.get("unresolved", []),
        "unresolved",
        limit=16,
        allowed=allowed,
    )
    if not claims and not conflicts and not unresolved:
        return None
    return EvidenceBrief(tuple(claims), conflicts, unresolved)


def render_evidence_brief(brief: EvidenceBrief, *, max_chars: int) -> str:
    """Bound JSON structurally so a qualification is never sliced in half."""

    claims = list(brief.claims)
    conflicts = list(brief.conflicts)
    unresolved = list(brief.unresolved)
    bound = max(256, int(max_chars))
    while True:
        value = EvidenceBrief(tuple(claims), tuple(conflicts), tuple(unresolved))
        encoded = json.dumps(
            value.as_dict(), ensure_ascii=False, separators=(",", ":")
        )
        if len(encoded) <= bound:
            return encoded
        if len(claims) > 1:
            claims.pop()
        elif len(conflicts) > 1:
            conflicts.pop()
        elif len(unresolved) > 1:
            unresolved.pop()
        else:
            # One indivisible claim may exceed a pathological low limit. Refuse to
            # inject it instead of truncating away its provenance or uncertainty.
            raise ValueError("memory brief cannot fit without structural truncation")
