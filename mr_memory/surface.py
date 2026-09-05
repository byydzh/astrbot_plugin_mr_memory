from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from .certificate import EvidenceAtom, EvidenceCertificateV2


SURFACE_SCHEMA_VERSION = "memory-surface.v1"
ANSWER_CONTEXT_SCHEMA_VERSION = "memory-answer-context.v2"


class SurfaceCompilationError(ValueError):
    """The mandatory evidence contract cannot fit without losing meaning."""


_PUBLIC_ATTRIBUTION = {
    "DIRECT_SPEAKER_STATEMENT": "direct_statement",
    "OTHER_SPEAKER_REPORT": "reported_statement",
    "OBSERVER_SUMMARY": "summary",
    "HOST_IDENTITY": "identity",
    "DERIVED_INTERPRETATION": "inference",
    "BEHAVIORAL_FEEDBACK": "feedback",
    "HOST_ACTIVITY_STATISTIC": "host_activity_statistics",
}

_PUBLIC_MEMORY_STATE = {
    "CERTIFIED": "supported",
    "PARTIAL": "qualified",
    "SAFETY_ABSTAIN": "insufficient",
}


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
    result: dict[str, object] = {
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
    if certificate.referents:
        result["referents"] = [item.as_dict() for item in certificate.referents]
    if certificate.aggregates:
        result["aggregates"] = [item.as_dict() for item in certificate.aggregates]
    return result


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
    """Compile a bounded, loss-intolerant internal audit packet.

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
            continue
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
    if certificate.referents:
        expected_top.add("referents")
    if certificate.aggregates:
        expected_top.add("aggregates")
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
    if certificate.referents and raw.get("referents") != [
        item.as_dict() for item in certificate.referents
    ]:
        raise SurfaceCompilationError("surface typed referent attribution was changed")
    if certificate.aggregates and raw.get("aggregates") != [item.as_dict() for item in certificate.aggregates]:
        raise SurfaceCompilationError("surface host activity statistics were changed")
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
    included_id_set = set(included_ids)
    certified_subsequence = [
        atom_id
        for atom_id in certificate_optional_order
        if atom_id in included_id_set
    ]
    if included_ids != certified_subsequence:
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


def _public_entity_labels(
    certificate: EvidenceCertificateV2,
) -> dict[str, str]:
    """Replace durable participant keys with request-local opaque labels."""

    ordered: list[str] = []
    for subject in certificate.subjects:
        if subject.participant_key and subject.participant_key not in ordered:
            ordered.append(subject.participant_key)
        for candidate in subject.candidate_participant_keys:
            if candidate not in ordered:
                ordered.append(candidate)
    # An evidence author need not be the subject of any fact. Keep distinct
    # host-bound authors distinct even when their display names are unavailable.
    for atom in _answer_atoms(certificate):
        for participant in (atom.speaker_participant_key, atom.subject_participant_key):
            if participant and participant not in ordered:
                ordered.append(participant)
    return {
        participant: f"person_{index}"
        for index, participant in enumerate(ordered, start=1)
    }


def _answer_atoms(certificate: EvidenceCertificateV2) -> tuple[EvidenceAtom, ...]:
    required = set(certificate.must_include)
    for condition in (*certificate.must_not_upgrade, *certificate.conflicts, *certificate.unresolved):
        required.update(condition.atom_ids)
    return tuple(atom for atom in certificate.atoms if atom.atom_id in required)


def _answer_referents(certificate: EvidenceCertificateV2):
    atoms = _answer_atoms(certificate)
    used = {atom.subject_referent_id for atom in atoms if atom.subject_referent_id}
    participants = {
        participant for atom in atoms
        for participant in (atom.speaker_participant_key, atom.subject_participant_key)
        if participant
    }
    return tuple(
        item for item in certificate.referents
        if item.referent_id in used or item.participant_key in participants
        or item.reference_mode in {"AMBIGUOUS", "UNBOUND"}
    )


def _iso_time(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace("+00:00", "Z")


def _answer_aggregates(certificate: EvidenceCertificateV2):
    used = {key for atom in _answer_atoms(certificate) for key in atom.aggregate_ids}
    return tuple(item for item in certificate.aggregates if item.aggregate_id in used)


def _aggregate_metadata(certificate: EvidenceCertificateV2) -> tuple[Mapping[str, object], ...]:
    return tuple({key: value for key, value in item.as_dict().items()
                  if key in {"aggregate_id", "source_revision_sha256", "source_count", "scope"}}
                 for item in _answer_aggregates(certificate))


def _answer_derivations(certificate: EvidenceCertificateV2):
    used = {key for item in (*_answer_atoms(certificate), *_answer_referents(certificate))
            for key in item.derivation_ids}
    return tuple(item for item in certificate.derivations if item.derivation_id in used)


def _derivation_metadata(certificate: EvidenceCertificateV2) -> tuple[Mapping[str, object], ...]:
    return tuple({key: value for key, value in item.as_dict().items()
                  if key in {"derivation_id", "kind", "owner_id", "content_revision", "dependency_keys"}}
                 for item in _answer_derivations(certificate))


def _answer_context_payload(
    certificate: EvidenceCertificateV2,
) -> dict[str, object]:
    """Project an internal evidence certificate into a private answer brief.

    The main model needs semantic facts and uncertainty, not audit evidence.
    In particular, raw source excerpts, durable identifiers, hashes, scopes,
    confidence scores and retrieval metadata must stay inside the plugin.
    """

    labels = _public_entity_labels(certificate)
    references: list[dict[str, object]] = []
    referent_labels: dict[str, str] = {}
    typed = _answer_referents(certificate)
    subjects = typed if certificate.referents else certificate.subjects
    known_participants: set[str] = set()
    for index, subject in enumerate(subjects, start=1):
        kind = getattr(subject, "referent_type", "PARTICIPANT")
        entity = labels.get(subject.participant_key, "")
        if not entity:
            entity = f"{kind.casefold()}_{index}"
        if kind == "PARTICIPANT" and subject.participant_key:
            known_participants.add(subject.participant_key)
        referent_id = getattr(subject, "referent_id", "")
        if referent_id:
            referent_labels[referent_id] = entity
        if subject.reference_mode in {"HOST", "STRUCTURED_REF", "UNIQUE_ALIAS", "EVIDENCE_REF"}:
            resolution = "resolved"
        elif subject.reference_mode == "AMBIGUOUS":
            resolution = "ambiguous"
        else:
            resolution = "unresolved"
        reference: dict[str, object] = {
            "mention": subject.reference,
            "entity": entity,
            "type": kind.casefold(),
            "resolution": resolution,
        }
        if subject.candidate_participant_keys:
            reference["candidates"] = [labels[key] for key in subject.candidate_participant_keys]
        if subject.valid_at is not None:
            reference["valid_at"] = _iso_time(subject.valid_at)
        references.append(reference)
    for participant, label in labels.items():
        if participant not in known_participants:
            references.append({
                "mention": "", "entity": label, "type": "participant",
                "resolution": "unlabeled",
            })

    answer_atoms = _answer_atoms(certificate)
    aggregates = _answer_aggregates(certificate)
    aggregate_labels = {item.aggregate_id: f"statistic_{index}" for index, item in enumerate(aggregates, start=1)}
    derivations = _answer_derivations(certificate)
    derivation_labels = {item.derivation_id: f"stored_memory_{index}" for index, item in enumerate(derivations, start=1)}
    fact_labels = {atom.atom_id: f"fact_{index}" for index, atom in enumerate(answer_atoms, start=1)}
    facts = [
        {
            "id": fact_labels[atom.atom_id],
            "statement": (
                "该窗口内观察到的本人发言统计见关联统计表。"
                if atom.attribution == "HOST_ACTIVITY_STATISTIC" else atom.statement
            ),
            "speaker": labels.get(atom.speaker_participant_key, ""),
            "subject": referent_labels.get(
                atom.subject_referent_id, labels.get(atom.subject_participant_key, ""),
            ),
            "attribution": _PUBLIC_ATTRIBUTION[atom.attribution],
            "evidence_roles": list(atom.evidence_roles),
            "stance": atom.stance.casefold(),
            **({"statistics": [aggregate_labels[key] for key in atom.aggregate_ids]} if atom.aggregate_ids else {}),
            **({"basis": "stored_derivation", "memory_sources": [derivation_labels[key] for key in atom.derivation_ids]}
               if atom.derivation_ids else {}),
        }
        for atom in answer_atoms
    ]

    def qualification(item):
        result: dict[str, object] = {"statement": item.statement}
        if item.basis == "PACKET_COVERAGE_GAP":
            result.update({
                "kind": "coverage_gap", "scope": "current_evidence_packet",
                "not_evidence_of_absence": True,
            })
        if item.atom_ids:
            result["facts"] = [fact_labels[key] for key in item.atom_ids]
        return result

    result = {
        "schema_version": ANSWER_CONTEXT_SCHEMA_VERSION,
        "as_of": _iso_time(certificate.scope_snapshot.cutoff_at),
        "memory_state": _PUBLIC_MEMORY_STATE.get(
            certificate.status,
            "insufficient",
        ),
        "referents": references,
        "facts": facts,
        "qualifications": {
            "do_not_upgrade": [
                {
                    "observed": item.observed,
                    "forbidden": list(item.forbidden),
                    "facts": [fact_labels[key] for key in item.atom_ids],
                    "reason": item.reason,
                }
                for item in certificate.must_not_upgrade
            ],
            "conflicts": [qualification(item) for item in certificate.conflicts],
            "unresolved": [qualification(item) for item in certificate.unresolved],
            "open_questions": [
                {"question": item.question, "critical": item.critical}
                for item in certificate.open_obligations
            ],
        },
    }
    if aggregates:
        result["activity_statistics"] = []
        for aggregate in aggregates:
            descriptor = aggregate.as_dict()
            scope = descriptor["scope"]
            result["activity_statistics"].append({
                "id": aggregate_labels[aggregate.aggregate_id],
                "subject": labels[scope["participant_key"]],
                "timezone": descriptor["timezone"],
                "window_start": _iso_time(scope["start_sent_at"]),
                "window_end_exclusive": _iso_time(scope["end_sent_at_exclusive"]),
                "source_count": descriptor["source_count"],
                "hour_histogram": descriptor["hour_histogram"],
                "daily": descriptor["daily"],
                "scope": "all_snapshot_visible_direct_speaker_messages_in_window",
                "interpretation_limit": "首末时间仅为各自然日观察到的本人发言边界，不证明醒来、入睡或发言原文。",
            })
    if derivations:
        result["stored_memories"] = [{
            "id": derivation_labels[item.derivation_id], "kind": item.as_dict()["kind"],
            "source_count": len(item.dependency_keys), "source_roles": list(item.source_roles),
            "basis": "stored_derivation", "raw_evidence_expanded": False,
        } for item in derivations]
    return result


@dataclass(frozen=True, slots=True)
class AnswerContextPacket:
    text: str
    required_fact_count: int
    included_atom_ids: tuple[str, ...] = ()
    included_source_keys: tuple[str, ...] = ()
    included_referent_ids: tuple[str, ...] = ()
    included_aggregate_ids: tuple[str, ...] = ()
    included_aggregate_metadata: tuple[Mapping[str, object], ...] = ()
    included_derivation_ids: tuple[str, ...] = ()
    included_derivation_metadata: tuple[Mapping[str, object], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        value = json.loads(self.text)
        if not isinstance(value, dict):
            raise SurfaceCompilationError("answer context is not a JSON object")
        return value


def compile_answer_context_packet(
    certificate: EvidenceCertificateV2,
    *,
    max_chars: int = 12_000,
) -> AnswerContextPacket:
    """Compile the only certificate projection allowed into the main prompt."""

    bound = int(max_chars)
    if bound <= 0:
        raise ValueError("max_chars must be positive")
    payload = _answer_context_payload(certificate)
    encoded = _encode(payload)
    if len(encoded) > bound:
        raise SurfaceCompilationError(
            "mandatory answer context exceeds max_chars; refusing truncation"
        )
    return AnswerContextPacket(
        text=encoded,
        required_fact_count=len(certificate.required_atoms),
        included_atom_ids=tuple(atom.atom_id for atom in _answer_atoms(certificate)),
        included_source_keys=_answer_source_keys(certificate),
        included_referent_ids=tuple(item.referent_id for item in _answer_referents(certificate)),
        included_aggregate_ids=tuple(item.aggregate_id for item in _answer_aggregates(certificate)),
        included_aggregate_metadata=_aggregate_metadata(certificate),
        included_derivation_ids=tuple(item.derivation_id for item in _answer_derivations(certificate)),
        included_derivation_metadata=_derivation_metadata(certificate),
    )


def _answer_source_keys(certificate: EvidenceCertificateV2) -> tuple[str, ...]:
    references = _answer_referents(certificate) if certificate.referents else certificate.subjects
    return tuple(dict.fromkeys(
        key for item in (*_answer_atoms(certificate), *references, *certificate.conflicts, *certificate.unresolved)
        for key in item.source_keys
    ))


def validate_answer_context_packet(
    packet: AnswerContextPacket,
    certificate: EvidenceCertificateV2,
) -> None:
    """Reject any answer brief that exposes or changes the public projection."""

    try:
        raw = json.loads(packet.text)
    except json.JSONDecodeError as exc:
        raise SurfaceCompilationError("answer context is invalid JSON") from exc
    if not isinstance(raw, dict):
        raise SurfaceCompilationError("answer context must be a JSON object")
    if _encode(raw) != packet.text:
        raise SurfaceCompilationError("answer context is not canonically encoded")
    if raw != _answer_context_payload(certificate):
        raise SurfaceCompilationError("answer context changed its public projection")
    if packet.required_fact_count != len(certificate.required_atoms):
        raise SurfaceCompilationError("answer context fact count mismatch")
    if packet.included_atom_ids != tuple(atom.atom_id for atom in _answer_atoms(certificate)):
        raise SurfaceCompilationError("answer context atom metadata mismatch")
    if packet.included_source_keys != _answer_source_keys(certificate):
        raise SurfaceCompilationError("answer context source metadata mismatch")
    if packet.included_referent_ids != tuple(item.referent_id for item in _answer_referents(certificate)):
        raise SurfaceCompilationError("answer context referent metadata mismatch")
    if packet.included_aggregate_ids != tuple(item.aggregate_id for item in _answer_aggregates(certificate)):
        raise SurfaceCompilationError("answer context aggregate metadata mismatch")
    if packet.included_aggregate_metadata != _aggregate_metadata(certificate):
        raise SurfaceCompilationError("answer context aggregate provenance mismatch")
    if packet.included_derivation_ids != tuple(item.derivation_id for item in _answer_derivations(certificate)):
        raise SurfaceCompilationError("answer context derivation metadata mismatch")
    if packet.included_derivation_metadata != _derivation_metadata(certificate):
        raise SurfaceCompilationError("answer context derivation provenance mismatch")


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
        normalized = _normalized(subject.reference)
        if normalized and normalized not in bucket:
            bucket.append(normalized)
    return {key: tuple(value) for key, value in references.items()}


@dataclass(frozen=True, slots=True)
class SurfaceVerification:
    passed: None
    lexical_checks_passed: bool
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
            "verifier_kind": "lexical_diagnostics",
            "semantic_status": "NOT_EVALUATED",
            "lexical_checks_passed": self.lexical_checks_passed,
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
    """Record lexical signals only; neither a pass nor a semantic verdict.

    Paraphrases, negation and local attribution cannot be evaluated reliably
    by substring matching. Absence or presence is a review hint, not evidence
    of answer correctness. Raw excerpts never count as retained facts.
    """

    normalized_answer = _normalized(answer)
    references = _references_by_participant(certificate)
    missing_required: list[str] = []
    attribution_violations: list[str] = []
    for atom in certificate.required_atoms:
        fragments = [atom.statement]
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
            aliases = references.get(participant_key, ())
            if aliases and not any(
                alias and alias in normalized_answer for alias in aliases
            ):
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
        passed=None,
        lexical_checks_passed=not any(
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
