from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping

from .certificate import EvidenceAtom, EvidenceCertificateV2


SURFACE_SCHEMA_VERSION = "memory-surface.v1"


class SurfaceCompilationError(ValueError):
    """The mandatory evidence contract cannot fit without losing meaning."""


def _encode(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _payload(
    certificate: EvidenceCertificateV2,
    *,
    optional_atoms: list[EvidenceAtom],
    omitted_optional: int,
) -> dict[str, object]:
    return {
        "schema_version": SURFACE_SCHEMA_VERSION,
        "certificate_sha256": certificate.digest,
        "snapshot_sha256": certificate.scope_snapshot.digest,
        "status": certificate.status,
        "scope": {
            "umo": certificate.scope_snapshot.umo,
            "cutoff_at": certificate.scope_snapshot.cutoff_at,
        },
        "subjects": [item.as_dict() for item in certificate.subjects],
        "evidence": {
            "required": [item.as_dict() for item in certificate.required_atoms],
            "optional": [item.as_dict() for item in optional_atoms],
        },
        "contract": {
            "must_include": list(certificate.must_include),
            "must_not_upgrade": [
                item.as_dict() for item in certificate.must_not_upgrade
            ],
            "conflicts": [item.as_dict() for item in certificate.conflicts],
            "unresolved": [item.as_dict() for item in certificate.unresolved],
            "open_obligations": [
                item.as_dict() for item in certificate.open_obligations
            ],
        },
        "stop_reason": certificate.stop_reason,
        "omitted_optional": int(omitted_optional),
    }


@dataclass(frozen=True, slots=True)
class SurfacePacket:
    text: str
    certificate_sha256: str
    snapshot_sha256: str
    included_required_atom_ids: tuple[str, ...]
    included_optional_atom_ids: tuple[str, ...]
    omitted_optional: int

    def as_dict(self) -> dict[str, Any]:
        value = json.loads(self.text)
        if not isinstance(value, dict):
            raise SurfaceCompilationError("surface packet is not a JSON object")
        return value


def compile_surface_packet(
    certificate: EvidenceCertificateV2,
    *,
    max_chars: int = 12000,
) -> SurfacePacket:
    """Compile a bounded, loss-intolerant packet for the host/main model.

    Attribution, REQUIRED atoms, conflicts, uncertainty and non-upgrade guards
    are indivisible.  Only OPTIONAL atoms may be omitted, and they are admitted
    in certificate order so the result is deterministic.
    """

    bound = int(max_chars)
    if bound <= 0:
        raise ValueError("max_chars must be positive")
    required_ids = set(certificate.must_include)
    optional = [
        atom for atom in certificate.atoms if atom.atom_id not in required_ids
    ]
    included: list[EvidenceAtom] = []
    core = _payload(
        certificate,
        optional_atoms=included,
        omitted_optional=len(optional),
    )
    encoded = _encode(core)
    if len(encoded) > bound:
        raise SurfaceCompilationError(
            "mandatory surface contract exceeds max_chars; refusing truncation"
        )
    for atom in optional:
        candidate = [*included, atom]
        candidate_payload = _payload(
            certificate,
            optional_atoms=candidate,
            omitted_optional=len(optional) - len(candidate),
        )
        candidate_text = _encode(candidate_payload)
        if len(candidate_text) <= bound:
            included = candidate
            encoded = candidate_text
        else:
            break
    return SurfacePacket(
        text=encoded,
        certificate_sha256=certificate.digest,
        snapshot_sha256=certificate.scope_snapshot.digest,
        included_required_atom_ids=certificate.must_include,
        included_optional_atom_ids=tuple(item.atom_id for item in included),
        omitted_optional=len(optional) - len(included),
    )


def validate_surface_packet(
    packet: SurfacePacket,
    certificate: EvidenceCertificateV2,
) -> None:
    """Reject a tampered or lossy packet before it reaches the main model."""

    try:
        raw = json.loads(packet.text)
    except json.JSONDecodeError as exc:
        raise SurfaceCompilationError("surface packet is invalid JSON") from exc
    if not isinstance(raw, dict):
        raise SurfaceCompilationError("surface packet must be a JSON object")
    if _encode(raw) != packet.text:
        raise SurfaceCompilationError("surface packet is not canonically encoded")
    expected_top = {
        "schema_version",
        "certificate_sha256",
        "snapshot_sha256",
        "status",
        "scope",
        "subjects",
        "evidence",
        "contract",
        "stop_reason",
        "omitted_optional",
    }
    if set(raw) != expected_top:
        raise SurfaceCompilationError("surface packet fields are invalid")
    if raw.get("schema_version") != SURFACE_SCHEMA_VERSION:
        raise SurfaceCompilationError("surface schema version is unsupported")
    if raw.get("certificate_sha256") != certificate.digest:
        raise SurfaceCompilationError("surface certificate digest mismatch")
    if raw.get("snapshot_sha256") != certificate.scope_snapshot.digest:
        raise SurfaceCompilationError("surface snapshot digest mismatch")
    if raw.get("status") != certificate.status:
        raise SurfaceCompilationError("surface semantic status mismatch")
    if raw.get("scope") != {
        "umo": certificate.scope_snapshot.umo,
        "cutoff_at": certificate.scope_snapshot.cutoff_at,
    }:
        raise SurfaceCompilationError("surface scope mismatch")
    if raw.get("subjects") != [item.as_dict() for item in certificate.subjects]:
        raise SurfaceCompilationError("surface subject attribution was changed")
    evidence = raw.get("evidence")
    if not isinstance(evidence, dict) or set(evidence) != {"required", "optional"}:
        raise SurfaceCompilationError("surface evidence fields are invalid")
    if evidence.get("required") != [
        item.as_dict() for item in certificate.required_atoms
    ]:
        raise SurfaceCompilationError("surface omitted or changed a required atom")
    optional_by_id = {
        item.atom_id: item
        for item in certificate.atoms
        if item.atom_id not in set(certificate.must_include)
    }
    raw_optional = evidence.get("optional")
    if not isinstance(raw_optional, list):
        raise SurfaceCompilationError("surface optional evidence must be an array")
    included_ids: list[str] = []
    for item in raw_optional:
        if not isinstance(item, dict):
            raise SurfaceCompilationError("surface optional atom is invalid")
        atom_id = str(item.get("id") or "")
        atom = optional_by_id.get(atom_id)
        if atom is None or item != atom.as_dict() or atom_id in included_ids:
            raise SurfaceCompilationError("surface optional atom is not certified")
        included_ids.append(atom_id)
    certificate_optional_order = [
        item.atom_id
        for item in certificate.atoms
        if item.atom_id not in set(certificate.must_include)
    ]
    if included_ids != certificate_optional_order[: len(included_ids)]:
        raise SurfaceCompilationError("surface optional evidence order is invalid")
    contract = raw.get("contract")
    expected_contract = {
        "must_include": list(certificate.must_include),
        "must_not_upgrade": [
            item.as_dict() for item in certificate.must_not_upgrade
        ],
        "conflicts": [item.as_dict() for item in certificate.conflicts],
        "unresolved": [item.as_dict() for item in certificate.unresolved],
        "open_obligations": [
            item.as_dict() for item in certificate.open_obligations
        ],
    }
    if contract != expected_contract:
        raise SurfaceCompilationError("surface evidence contract was changed")
    omitted = len(certificate_optional_order) - len(included_ids)
    if raw.get("omitted_optional") != omitted:
        raise SurfaceCompilationError("surface omitted_optional count is invalid")
    if raw.get("stop_reason") != certificate.stop_reason:
        raise SurfaceCompilationError("surface stop reason mismatch")
    if packet.certificate_sha256 != certificate.digest:
        raise SurfaceCompilationError("packet metadata certificate digest mismatch")
    if packet.snapshot_sha256 != certificate.scope_snapshot.digest:
        raise SurfaceCompilationError("packet metadata snapshot digest mismatch")
    if packet.included_required_atom_ids != certificate.must_include:
        raise SurfaceCompilationError("packet metadata omitted a required atom")
    if packet.included_optional_atom_ids != tuple(included_ids):
        raise SurfaceCompilationError("packet optional metadata mismatch")
    if packet.omitted_optional != omitted:
        raise SurfaceCompilationError("packet omission metadata mismatch")


def _normalized(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[^\w]+", "", text, flags=re.UNICODE)


def _references_by_participant(
    certificate: EvidenceCertificateV2,
) -> dict[str, tuple[str, ...]]:
    references: dict[str, list[str]] = {}
    for subject in certificate.subjects:
        if not subject.participant_key:
            continue
        bucket = references.setdefault(subject.participant_key, [])
        for value in (subject.reference, subject.participant_key):
            normalized = _normalized(value)
            if normalized and normalized not in bucket:
                bucket.append(normalized)
    return {key: tuple(value) for key, value in references.items()}


@dataclass(frozen=True, slots=True)
class SurfaceVerification:
    passed: bool
    required_total: int
    required_matched: int
    missing_required_atom_ids: tuple[str, ...]
    attribution_violations: tuple[str, ...]
    forbidden_upgrades: tuple[str, ...]
    unresolved_total: int
    unresolved_retained: int
    missing_unresolved: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "required_total": self.required_total,
            "required_matched": self.required_matched,
            "missing_required_atom_ids": list(self.missing_required_atom_ids),
            "attribution_violations": list(self.attribution_violations),
            "forbidden_upgrades": list(self.forbidden_upgrades),
            "unresolved_total": self.unresolved_total,
            "unresolved_retained": self.unresolved_retained,
            "missing_unresolved": list(self.missing_unresolved),
        }


def verify_surface_answer(
    answer: str,
    certificate: EvidenceCertificateV2,
) -> SurfaceVerification:
    """A deterministic, conservative shadow verifier for surface answers.

    This is intentionally a lower-bound check rather than another LLM judge:
    a paraphrase may be flagged for review, but unsupported upgrades cannot be
    waved through by a second probabilistic model.
    """

    normalized_answer = _normalized(answer)
    references = _references_by_participant(certificate)
    missing_required: list[str] = []
    attribution_violations: list[str] = []
    for atom in certificate.required_atoms:
        fragments = [atom.statement, *atom.source_spans]
        normalized_fragments = [
            item for item in (_normalized(value) for value in fragments) if len(item) >= 2
        ]
        matched = any(item in normalized_answer for item in normalized_fragments)
        if not matched:
            missing_required.append(atom.atom_id)
            continue
        participant_keys = tuple(
            dict.fromkeys(
                item
                for item in (
                    atom.speaker_participant_key,
                    atom.subject_participant_key,
                )
                if item
            )
        )
        for participant_key in participant_keys:
            aliases = references.get(
                participant_key,
                (_normalized(participant_key),),
            )
            if not any(alias and alias in normalized_answer for alias in aliases):
                attribution_violations.append(
                    f"{atom.atom_id}:{participant_key}"
                )

    forbidden: list[str] = []
    for guard in certificate.must_not_upgrade:
        for phrase in guard.forbidden:
            normalized_phrase = _normalized(phrase)
            if normalized_phrase and normalized_phrase in normalized_answer:
                forbidden.append(phrase)

    missing_unresolved: list[str] = []
    for item in certificate.unresolved:
        normalized_statement = _normalized(item.statement)
        if normalized_statement and normalized_statement not in normalized_answer:
            missing_unresolved.append(item.statement)

    missing_required_tuple = tuple(dict.fromkeys(missing_required))
    attribution_tuple = tuple(dict.fromkeys(attribution_violations))
    forbidden_tuple = tuple(dict.fromkeys(forbidden))
    missing_unresolved_tuple = tuple(dict.fromkeys(missing_unresolved))
    unresolved_total = len(certificate.unresolved)
    unresolved_retained = unresolved_total - len(missing_unresolved_tuple)
    return SurfaceVerification(
        passed=not any(
            (
                missing_required_tuple,
                attribution_tuple,
                forbidden_tuple,
                missing_unresolved_tuple,
            )
        ),
        required_total=len(certificate.must_include),
        required_matched=len(certificate.must_include) - len(missing_required_tuple),
        missing_required_atom_ids=missing_required_tuple,
        attribution_violations=attribution_tuple,
        forbidden_upgrades=forbidden_tuple,
        unresolved_total=unresolved_total,
        unresolved_retained=unresolved_retained,
        missing_unresolved=missing_unresolved_tuple,
    )
