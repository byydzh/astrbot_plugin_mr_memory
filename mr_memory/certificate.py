from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .activity_statistics import validate_activity_window_statistics
from .derivations import MAX_STORED_DERIVATIONS, StoredDerivationEvidence
from .snapshot import RequestSnapshot, stable_sha256


CERTIFICATE_SCHEMA_VERSION = "evidence-certificate.v2"
MAX_CERTIFICATE_SOURCE_KEYS = 64
MAX_CERTIFICATE_AGGREGATES = 8
# Contract-to-certificate conversion may combine independent bounded channels.
# Conflicts can contain the 16-item brief plus 16 contested interpretations;
# unresolved conditions can contain the 16-item brief, 16 interpretations,
# 16 uncertainty constraints, and 16 guarded claims.  These are structural
# capacities, not quality thresholds, and conversion must never silently drop
# an already host-validated condition.
MAX_CERTIFICATE_CONFLICTS = 32
MAX_CERTIFICATE_UNRESOLVED = 64
CERTIFICATE_STATUSES = {
    "CERTIFIED",
    "PARTIAL",
    "SEMANTIC_NONE",
    "SAFETY_ABSTAIN",
    "REQUEST_L3",
}
ATTRIBUTION_KINDS = {
    "DIRECT_SPEAKER_STATEMENT",
    "OTHER_SPEAKER_REPORT",
    "OBSERVER_SUMMARY",
    "HOST_IDENTITY",
    "DERIVED_INTERPRETATION",
    "BEHAVIORAL_FEEDBACK",
    "HOST_ACTIVITY_STATISTIC",
}
ATOM_STANCES = {"SUPPORTED", "REFUTED", "CONTESTED", "UNRESOLVED"}
ATOM_IMPORTANCE = {"REQUIRED", "OPTIONAL"}
EVIDENCE_ROLES = {"USER", "BOT", "SYSTEM", "UNKNOWN"}
SUBJECT_BINDING_MODES = {
    "HOST",
    "STRUCTURED_REF",
    "UNIQUE_ALIAS",
    "AMBIGUOUS",
    "UNBOUND",
}
REFERENT_TYPES = {"PARTICIPANT", "WORK", "ENTITY", "TOPIC"}
STOP_REASONS = {
    "CERTIFIED_CLOSE",
    "SEMANTIC_NONE",
    "SAFETY_ABSTAIN",
    "FRONTIER_EXHAUSTED",
    "SATURATED",
    "REQUEST_L3",
}

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _text(
    value: object,
    field: str,
    *,
    limit: int,
    required: bool = True,
) -> str:
    result = " ".join(str(value or "").strip().split())
    if required and not result:
        raise ValueError(f"{field} is required")
    if len(result) > limit:
        raise ValueError(f"{field} exceeds {limit} characters")
    return result


def _identifier(value: object, field: str) -> str:
    result = str(value or "").strip()
    if not _ID_RE.fullmatch(result):
        raise ValueError(f"{field} is not a bounded identifier")
    return result


def _sha256(value: object, field: str) -> str:
    result = str(value or "").strip().casefold()
    if not _SHA256_RE.fullmatch(result):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return result


def _exact_fields(value: Mapping[str, Any], allowed: set[str], field: str) -> None:
    keys = {str(key) for key in value}
    if keys == allowed:
        return
    missing = sorted(allowed - keys)
    unknown = sorted(keys - allowed)
    details = []
    if missing:
        details.append("missing " + ", ".join(missing))
    if unknown:
        details.append("unknown " + ", ".join(unknown))
    raise ValueError(f"{field} fields are invalid: " + "; ".join(details))


def _extract_object(value: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    text = str(value or "").strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("evidence certificate must be exactly one JSON object") from exc
    if not isinstance(parsed, dict):
        raise ValueError("evidence certificate must be exactly one JSON object")
    return parsed


def _string_tuple(
    value: object,
    field: str,
    *,
    limit: int,
    item_limit: int,
    required: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    if required and not value:
        raise ValueError(f"{field} must not be empty")
    if len(value) > limit:
        raise ValueError(f"{field} exceeds {limit} items")
    result = tuple(
        dict.fromkeys(
            _text(item, f"{field}[]", limit=item_limit) for item in value
        )
    )
    if required and not result:
        raise ValueError(f"{field} must not be empty")
    return result


def _source_keys(
    value: object,
    field: str,
    *,
    allowed: set[str],
    required: bool = True,
) -> tuple[str, ...]:
    result = _string_tuple(
        value,
        field,
        limit=MAX_CERTIFICATE_SOURCE_KEYS,
        item_limit=1000,
        required=required,
    )
    if not set(result).issubset(allowed):
        raise ValueError(f"{field} cites evidence outside the delivered allowlist")
    return result


def _participant(
    value: object,
    field: str,
    *,
    allowed: set[str],
    required: bool = False,
) -> str:
    result = _text(value, field, limit=256, required=required)
    if result and result not in allowed:
        raise ValueError(f"{field} is not host-authorized")
    return result


@dataclass(frozen=True, slots=True)
class CertificateSubject:
    reference: str
    participant_key: str
    reference_mode: str
    candidate_participant_keys: tuple[str, ...]
    source_keys: tuple[str, ...]
    valid_at: int | None

    @classmethod
    def from_value(
        cls,
        value: object,
        *,
        allowed_sources: set[str],
        allowed_participants: set[str],
        cutoff_at: int,
        field: str,
    ) -> CertificateSubject:
        if not isinstance(value, Mapping):
            raise ValueError(f"{field} must be an object")
        _exact_fields(
            value,
            {
                "reference",
                "participant_key",
                "reference_mode",
                "candidate_participant_keys",
                "source_keys",
                "valid_at",
            },
            field,
        )
        mode = str(value.get("reference_mode") or "").strip().upper()
        if mode not in SUBJECT_BINDING_MODES:
            raise ValueError(f"{field}.reference_mode is unsupported")
        participant_key = _participant(
            value.get("participant_key"),
            f"{field}.participant_key",
            allowed=allowed_participants,
        )
        candidates = _string_tuple(
            value.get("candidate_participant_keys"),
            f"{field}.candidate_participant_keys",
            limit=20,
            item_limit=256,
        )
        if not set(candidates).issubset(allowed_participants):
            raise ValueError(f"{field} contains a non-authorized identity candidate")
        if mode in {"HOST", "STRUCTURED_REF", "UNIQUE_ALIAS"}:
            if not participant_key or candidates:
                raise ValueError(f"{field} resolved mode requires one participant_key")
        elif mode == "AMBIGUOUS":
            if participant_key or len(candidates) < 2:
                raise ValueError(f"{field} ambiguous mode requires two candidates")
        elif participant_key or candidates:
            raise ValueError(f"{field} unbound mode cannot select an identity")
        sources = _source_keys(
            value.get("source_keys"),
            f"{field}.source_keys",
            allowed=allowed_sources,
            required=(mode == "UNIQUE_ALIAS"),
        )
        raw_valid_at = value.get("valid_at")
        valid_at = None if raw_valid_at in (None, "") else int(raw_valid_at)
        if valid_at is not None and (valid_at <= 0 or valid_at >= int(cutoff_at)):
            raise ValueError(f"{field}.valid_at must be strictly before the cutoff")
        return cls(
            reference=_text(value.get("reference"), f"{field}.reference", limit=240),
            participant_key=participant_key,
            reference_mode=mode,
            candidate_participant_keys=candidates,
            source_keys=sources,
            valid_at=valid_at,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "reference": self.reference,
            "participant_key": self.participant_key,
            "reference_mode": self.reference_mode,
            "candidate_participant_keys": list(self.candidate_participant_keys),
            "source_keys": list(self.source_keys),
            "valid_at": self.valid_at,
        }


@dataclass(frozen=True, slots=True)
class CertificateReferent:
    """A semantic subject; source authorship does not determine its type."""

    referent_id: str
    reference: str
    referent_type: str
    participant_key: str
    reference_mode: str
    candidate_participant_keys: tuple[str, ...]
    source_keys: tuple[str, ...]
    valid_at: int | None
    derivation_ids: tuple[str, ...] = ()

    @classmethod
    def from_value(
        cls,
        value: object,
        *,
        allowed_sources: set[str],
        allowed_participants: set[str],
        cutoff_at: int,
        field: str,
        allowed_derivation_ids: set[str] | None = None,
    ) -> CertificateReferent:
        if not isinstance(value, Mapping):
            raise ValueError(f"{field} must be an object")
        _exact_fields({key: item for key, item in value.items() if key != "derivation_ids"}, {
            "id", "reference", "referent_type", "participant_key",
            "reference_mode", "candidate_participant_keys", "source_keys", "valid_at",
        }, field)
        kind = str(value.get("referent_type") or "").strip().upper()
        if kind not in REFERENT_TYPES:
            raise ValueError(f"{field}.referent_type is unsupported")
        derivation_ids = _string_tuple(value.get("derivation_ids", []), f"{field}.derivation_ids",
                                      limit=MAX_STORED_DERIVATIONS, item_limit=80)
        if not set(derivation_ids).issubset(allowed_derivation_ids or set()):
            raise ValueError(f"{field}.derivation_ids are outside the host allowlist")
        if derivation_ids and kind == "PARTICIPANT":
            raise ValueError(f"{field} stored derivations cannot establish participant identity")
        subject_value = {
            key: item for key, item in value.items()
            if key not in {"id", "referent_type", "derivation_ids"}
        }
        if kind != "PARTICIPANT":
            if value.get("participant_key") != "" or value.get("candidate_participant_keys") != []:
                raise ValueError(f"{field} non-participant referent cannot carry participant keys")
            if value.get("reference_mode") != "EVIDENCE_REF":
                raise ValueError(f"{field} non-participant referent requires EVIDENCE_REF")
            _source_keys(value.get("source_keys"), f"{field}.source_keys", allowed=allowed_sources,
                         required=not derivation_ids)
            # Reuse the bounded text/time validation without claiming a person.
            subject_value["reference_mode"] = "UNBOUND"
        subject = CertificateSubject.from_value(
            subject_value, allowed_sources=allowed_sources,
            allowed_participants=allowed_participants, cutoff_at=cutoff_at, field=field,
        )
        return cls(
            referent_id=_identifier(value.get("id"), f"{field}.id"),
            reference=subject.reference, referent_type=kind,
            participant_key=subject.participant_key,
            reference_mode=subject.reference_mode if kind == "PARTICIPANT" else "EVIDENCE_REF",
            candidate_participant_keys=subject.candidate_participant_keys,
            source_keys=subject.source_keys, valid_at=subject.valid_at,
            derivation_ids=derivation_ids,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.referent_id, "reference": self.reference,
            "referent_type": self.referent_type, "participant_key": self.participant_key,
            "reference_mode": self.reference_mode,
            "candidate_participant_keys": list(self.candidate_participant_keys),
            "source_keys": list(self.source_keys), "valid_at": self.valid_at,
            **({"derivation_ids": list(self.derivation_ids)} if self.derivation_ids else {}),
        }

    def as_subject(self) -> CertificateSubject:
        if self.referent_type != "PARTICIPANT":
            raise ValueError("only participant referents have a legacy subject projection")
        return CertificateSubject(
            self.reference, self.participant_key, self.reference_mode,
            self.candidate_participant_keys, self.source_keys, self.valid_at,
        )


@dataclass(frozen=True, slots=True)
class ActivityEvidenceAggregate:
    """Immutable host statistics, separate from individually delivered messages."""

    aggregate_id: str
    descriptor_json: str

    @classmethod
    def from_value(cls, value: object, *, snapshot: RequestSnapshot,
                   allowed_participants: set[str]) -> ActivityEvidenceAggregate:
        descriptor = validate_activity_window_statistics(value)
        scope = descriptor["scope"]
        if scope["umo"] != snapshot.umo:
            raise ValueError("activity aggregate scope differs from the host snapshot")
        if scope["end_sent_at_exclusive"] > snapshot.cutoff_at:
            raise ValueError("activity aggregate exceeds the host cutoff")
        if scope["message_upper_bound"] != snapshot.message_upper_bound:
            raise ValueError("activity aggregate message bound differs from the host snapshot")
        if scope["participant_key"] not in allowed_participants:
            raise ValueError("activity aggregate participant is not host-authorized")
        return cls(descriptor["aggregate_id"], json.dumps(descriptor, ensure_ascii=False,
                                                        sort_keys=True, separators=(",", ":")))

    def as_dict(self) -> dict[str, Any]:
        return json.loads(self.descriptor_json)

    @property
    def participant_key(self) -> str:
        return self.as_dict()["scope"]["participant_key"]


@dataclass(frozen=True, slots=True)
class EvidenceAtom:
    atom_id: str
    statement: str
    speaker_participant_key: str
    subject_participant_key: str
    attribution: str
    stance: str
    source_keys: tuple[str, ...]
    source_spans: tuple[str, ...]
    importance: str
    confidence: float
    subject_referent_id: str = ""
    aggregate_ids: tuple[str, ...] = ()
    evidence_roles: tuple[str, ...] = ("UNKNOWN",)
    derivation_ids: tuple[str, ...] = ()

    @classmethod
    def from_value(
        cls,
        value: object,
        *,
        allowed_sources: set[str],
        allowed_participants: set[str],
        field: str,
        allowed_aggregate_ids: set[str] | None = None,
        source_roles: Mapping[str, str] | None = None,
        allowed_derivations: Mapping[str, StoredDerivationEvidence] | None = None,
    ) -> EvidenceAtom:
        if not isinstance(value, Mapping):
            raise ValueError(f"{field} must be an object")
        _exact_fields(
            {key: item for key, item in value.items()
             if key not in {"subject_referent_id", "aggregate_ids", "evidence_roles", "derivation_ids"}},
            {
                "id",
                "statement",
                "speaker_participant_key",
                "subject_participant_key",
                "attribution",
                "stance",
                "source_keys",
                "source_spans",
                "importance",
                "confidence",
            },
            field,
        )
        attribution = str(value.get("attribution") or "").strip().upper()
        stance = str(value.get("stance") or "").strip().upper()
        importance = str(value.get("importance") or "").strip().upper()
        if attribution not in ATTRIBUTION_KINDS:
            raise ValueError(f"{field}.attribution is unsupported")
        if stance not in ATOM_STANCES:
            raise ValueError(f"{field}.stance is unsupported")
        if importance not in ATOM_IMPORTANCE:
            raise ValueError(f"{field}.importance is unsupported")
        confidence = float(value.get("confidence"))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"{field}.confidence must be 0..1")
        aggregate_ids = _string_tuple(value.get("aggregate_ids", []), f"{field}.aggregate_ids",
                                      limit=MAX_CERTIFICATE_AGGREGATES, item_limit=80)
        if not set(aggregate_ids).issubset(allowed_aggregate_ids or set()):
            raise ValueError(f"{field}.aggregate_ids are outside the host allowlist")
        if aggregate_ids and attribution not in {"HOST_ACTIVITY_STATISTIC", "DERIVED_INTERPRETATION"}:
            raise ValueError(
                f"{field} aggregate evidence requires HOST_ACTIVITY_STATISTIC or DERIVED_INTERPRETATION"
            )
        if attribution == "HOST_ACTIVITY_STATISTIC" and not aggregate_ids:
            raise ValueError(f"{field} HOST_ACTIVITY_STATISTIC requires aggregate_ids")
        derivation_ids = _string_tuple(value.get("derivation_ids", []), f"{field}.derivation_ids",
                                      limit=MAX_STORED_DERIVATIONS, item_limit=80)
        if not set(derivation_ids).issubset(allowed_derivations or {}):
            raise ValueError(f"{field}.derivation_ids are outside the host allowlist")
        if derivation_ids and attribution not in {"OBSERVER_SUMMARY", "DERIVED_INTERPRETATION"}:
            raise ValueError(f"{field} stored derivations require summary or inference attribution")
        source_keys = _source_keys(
            value.get("source_keys"),
            f"{field}.source_keys",
            allowed=allowed_sources,
            required=not (aggregate_ids or derivation_ids),
        )
        source_spans = _string_tuple(
            value.get("source_spans"),
            f"{field}.source_spans",
            limit=MAX_CERTIFICATE_SOURCE_KEYS,
            item_limit=500,
        )
        if len(source_spans) > len(source_keys):
            raise ValueError(f"{field}.source_spans exceeds cited sources")
        if aggregate_ids and (source_spans or value.get("speaker_participant_key")):
            raise ValueError(f"{field} activity statistics cannot assert a speaker or quotation")
        if derivation_ids and (source_spans or value.get("speaker_participant_key")):
            raise ValueError(f"{field} stored derivations cannot assert a speaker or quotation")
        evidence_roles = _string_tuple(
            value.get("evidence_roles", ["UNKNOWN"]), f"{field}.evidence_roles",
            limit=len(EVIDENCE_ROLES), item_limit=7, required=True,
        )
        if (not set(evidence_roles).issubset(EVIDENCE_ROLES)
                or len(evidence_roles) != len(value.get("evidence_roles", ["UNKNOWN"]))):
            raise ValueError(f"{field}.evidence_roles must be unique host source roles")
        if source_roles is not None:
            expected_roles = {source_roles.get(key, "UNKNOWN") for key in source_keys}
            expected_roles.update(role for key in derivation_ids
                                  for role in allowed_derivations[key].source_roles)
            expected_roles = expected_roles or {"UNKNOWN"}
            if set(evidence_roles) != expected_roles:
                raise ValueError(f"{field}.evidence_roles differ from host source metadata")
        elif derivation_ids and not {role for key in derivation_ids
                                    for role in allowed_derivations[key].source_roles}.issubset(evidence_roles):
            raise ValueError(f"{field}.evidence_roles omit stored derivation source roles")
        return cls(
            atom_id=_identifier(value.get("id"), f"{field}.id"),
            statement=_text(value.get("statement"), f"{field}.statement", limit=2000),
            speaker_participant_key=_participant(
                value.get("speaker_participant_key"),
                f"{field}.speaker_participant_key",
                allowed=allowed_participants,
            ),
            subject_participant_key=_participant(
                value.get("subject_participant_key"),
                f"{field}.subject_participant_key",
                allowed=allowed_participants,
            ),
            attribution=attribution,
            stance=stance,
            source_keys=source_keys,
            source_spans=source_spans,
            importance=importance,
            confidence=confidence,
            subject_referent_id=(
                _identifier(value["subject_referent_id"], f"{field}.subject_referent_id")
                if value.get("subject_referent_id") else ""
            ),
            aggregate_ids=aggregate_ids,
            evidence_roles=tuple(sorted(evidence_roles)),
            derivation_ids=derivation_ids,
        )

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "id": self.atom_id,
            "statement": self.statement,
            "speaker_participant_key": self.speaker_participant_key,
            "subject_participant_key": self.subject_participant_key,
            "attribution": self.attribution,
            "stance": self.stance,
            "source_keys": list(self.source_keys),
            "source_spans": list(self.source_spans),
            "importance": self.importance,
            "confidence": self.confidence,
        }
        if self.subject_referent_id:
            result["subject_referent_id"] = self.subject_referent_id
        if self.aggregate_ids:
            result["aggregate_ids"] = list(self.aggregate_ids)
        if self.evidence_roles != ("UNKNOWN",):
            result["evidence_roles"] = list(self.evidence_roles)
        if self.derivation_ids:
            result["derivation_ids"] = list(self.derivation_ids)
        return result


@dataclass(frozen=True, slots=True)
class UpgradeGuard:
    observed: str
    forbidden: tuple[str, ...]
    atom_ids: tuple[str, ...]
    reason: str

    @classmethod
    def from_value(cls, value: object, *, field: str) -> UpgradeGuard:
        if not isinstance(value, Mapping):
            raise ValueError(f"{field} must be an object")
        _exact_fields(value, {"observed", "forbidden", "atom_ids", "reason"}, field)
        forbidden = _string_tuple(
            value.get("forbidden"),
            f"{field}.forbidden",
            limit=16,
            item_limit=240,
            required=True,
        )
        atom_ids = _string_tuple(
            value.get("atom_ids"),
            f"{field}.atom_ids",
            limit=16,
            item_limit=80,
            required=True,
        )
        if any(not _ID_RE.fullmatch(item) for item in atom_ids):
            raise ValueError(f"{field}.atom_ids contains an invalid identifier")
        return cls(
            observed=_text(value.get("observed"), f"{field}.observed", limit=240),
            forbidden=forbidden,
            atom_ids=atom_ids,
            reason=_text(value.get("reason"), f"{field}.reason", limit=800),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "observed": self.observed,
            "forbidden": list(self.forbidden),
            "atom_ids": list(self.atom_ids),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class CertificateQualification:
    statement: str
    source_keys: tuple[str, ...]
    atom_ids: tuple[str, ...]
    basis: str = "EVIDENCE"

    @classmethod
    def from_value(
        cls,
        value: object,
        *,
        allowed_sources: set[str],
        field: str,
        allow_packet_coverage_gap: bool = False,
    ) -> CertificateQualification:
        if not isinstance(value, Mapping):
            raise ValueError(f"{field} must be an object")
        _exact_fields(
            {key: item for key, item in value.items() if key != "basis"},
            {"statement", "source_keys", "atom_ids"}, field,
        )
        sources = _source_keys(
            value.get("source_keys"),
            f"{field}.source_keys",
            allowed=allowed_sources,
            required=False,
        )
        atom_ids = _string_tuple(
            value.get("atom_ids"),
            f"{field}.atom_ids",
            limit=32,
            item_limit=80,
        )
        if any(not _ID_RE.fullmatch(item) for item in atom_ids):
            raise ValueError(f"{field}.atom_ids contains an invalid identifier")
        # No citation can prove an absence in the retrieved packet. Preserve
        # that open question as a non-factual, packet-bound coverage gap. The
        # enclosing certificate binds it to the host snapshot and packet hash.
        basis = value.get("basis", "EVIDENCE" if sources or atom_ids else "PACKET_COVERAGE_GAP")
        if not isinstance(basis, str) or basis not in {"EVIDENCE", "PACKET_COVERAGE_GAP"}:
            raise ValueError(f"{field}.basis is unsupported")
        if basis == "PACKET_COVERAGE_GAP":
            if not allow_packet_coverage_gap:
                raise ValueError(f"{field} requires source_keys or atom_ids")
            if sources or atom_ids:
                raise ValueError(f"{field} packet coverage gap cannot carry evidence references")
        elif not sources and not atom_ids:
            raise ValueError(f"{field} requires source_keys or atom_ids")
        return cls(
            statement=_text(value.get("statement"), f"{field}.statement", limit=1200),
            source_keys=sources,
            atom_ids=atom_ids,
            basis=basis,
        )

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "statement": self.statement,
            "source_keys": list(self.source_keys),
            "atom_ids": list(self.atom_ids),
        }
        if self.basis == "PACKET_COVERAGE_GAP":
            result["basis"] = self.basis
        return result


@dataclass(frozen=True, slots=True)
class OpenObligation:
    obligation_id: str
    question: str
    critical: bool
    competing_interpretation_ids: tuple[str, ...]
    discriminator: str
    expected_information_gain: str

    @classmethod
    def from_value(cls, value: object, *, field: str) -> OpenObligation:
        if not isinstance(value, Mapping):
            raise ValueError(f"{field} must be an object")
        _exact_fields(
            value,
            {
                "id",
                "question",
                "critical",
                "competing_interpretation_ids",
                "discriminator",
                "expected_information_gain",
            },
            field,
        )
        interpretation_ids = _string_tuple(
            value.get("competing_interpretation_ids"),
            f"{field}.competing_interpretation_ids",
            limit=16,
            item_limit=80,
        )
        if any(not _ID_RE.fullmatch(item) for item in interpretation_ids):
            raise ValueError(
                f"{field}.competing_interpretation_ids contains an invalid identifier"
            )
        if not isinstance(value.get("critical"), bool):
            raise ValueError(f"{field}.critical must be boolean")
        return cls(
            obligation_id=_identifier(value.get("id"), f"{field}.id"),
            question=_text(value.get("question"), f"{field}.question", limit=1000),
            critical=bool(value.get("critical")),
            competing_interpretation_ids=interpretation_ids,
            discriminator=_text(
                value.get("discriminator"),
                f"{field}.discriminator",
                limit=600,
                required=False,
            ),
            expected_information_gain=_text(
                value.get("expected_information_gain"),
                f"{field}.expected_information_gain",
                limit=600,
                required=False,
            ),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.obligation_id,
            "question": self.question,
            "critical": self.critical,
            "competing_interpretation_ids": list(self.competing_interpretation_ids),
            "discriminator": self.discriminator,
            "expected_information_gain": self.expected_information_gain,
        }


@dataclass(frozen=True, slots=True)
class EvidenceCertificateV2:
    status: str
    scope_snapshot: RequestSnapshot
    packet_sha256: str
    subjects: tuple[CertificateSubject, ...]
    atoms: tuple[EvidenceAtom, ...]
    must_include: tuple[str, ...]
    must_not_upgrade: tuple[UpgradeGuard, ...]
    conflicts: tuple[CertificateQualification, ...]
    unresolved: tuple[CertificateQualification, ...]
    open_obligations: tuple[OpenObligation, ...]
    stop_reason: str
    pack_read_complete: bool
    host_validated: bool
    # Optional v2 extension. Legacy certificates retain their canonical digest.
    referents: tuple[CertificateReferent, ...] = ()
    aggregates: tuple[ActivityEvidenceAggregate, ...] = ()
    derivations: tuple[StoredDerivationEvidence, ...] = ()

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": CERTIFICATE_SCHEMA_VERSION,
            "status": self.status,
            "scope_snapshot": self.scope_snapshot.as_dict(),
            "data_revision": self.scope_snapshot.data_revision.as_dict(),
            "inference_revision": self.scope_snapshot.inference_revision.as_dict(),
            "packet_sha256": self.packet_sha256,
            "subjects": [item.as_dict() for item in self.subjects],
            "atoms": [item.as_dict() for item in self.atoms],
            "must_include": list(self.must_include),
            "must_not_upgrade": [item.as_dict() for item in self.must_not_upgrade],
            "conflicts": [item.as_dict() for item in self.conflicts],
            "unresolved": [item.as_dict() for item in self.unresolved],
            "open_obligations": [item.as_dict() for item in self.open_obligations],
            "stop_reason": self.stop_reason,
            "validation": {
                "pack_read_complete": self.pack_read_complete,
                "host_validated": self.host_validated,
            },
        }
        if self.referents:
            result["referents"] = [item.as_dict() for item in self.referents]
        if self.aggregates:
            result["aggregates"] = [item.as_dict() for item in self.aggregates]
        if self.derivations:
            result["derivations"] = [item.as_dict() for item in self.derivations]
        return result

    @property
    def digest(self) -> str:
        return stable_sha256(self.as_dict())

    @property
    def required_atoms(self) -> tuple[EvidenceAtom, ...]:
        by_id = {item.atom_id: item for item in self.atoms}
        return tuple(by_id[item] for item in self.must_include)


def _parse_items(
    value: object,
    field: str,
    *,
    limit: int,
    parser: Any,
) -> tuple[Any, ...]:
    if not isinstance(value, list) or len(value) > limit:
        raise ValueError(f"{field} must be an array with at most {limit} items")
    return tuple(parser(item, index) for index, item in enumerate(value))


def parse_evidence_certificate(
    value: str | Mapping[str, Any],
    *,
    expected_snapshot: RequestSnapshot,
    expected_packet_sha256: str,
    allowed_source_keys: Iterable[str],
    allowed_participant_keys: Iterable[str] = (),
    pack_read_complete: bool,
    host_validated: bool = True,
    allowed_aggregates: Mapping[str, Mapping[str, object]] | None = None,
    source_roles: Mapping[str, str] | None = None,
    allowed_derivations: Mapping[str, Mapping[str, object]] | None = None,
) -> EvidenceCertificateV2:
    """Parse one host-bound certificate; model-owned scope/revisions are rejected."""

    raw = _extract_object(value)
    _exact_fields(
        {key: item for key, item in raw.items() if key not in {"referents", "aggregates", "derivations"}},
        {
            "schema_version",
            "status",
            "scope_snapshot",
            "data_revision",
            "inference_revision",
            "packet_sha256",
            "subjects",
            "atoms",
            "must_include",
            "must_not_upgrade",
            "conflicts",
            "unresolved",
            "open_obligations",
            "stop_reason",
            "validation",
        },
        "certificate",
    )
    if str(raw.get("schema_version") or "") != CERTIFICATE_SCHEMA_VERSION:
        raise ValueError("unsupported evidence certificate schema_version")
    status = str(raw.get("status") or "").strip().upper()
    stop_reason = str(raw.get("stop_reason") or "").strip().upper()
    if status not in CERTIFICATE_STATUSES:
        raise ValueError("certificate.status is unsupported")
    if stop_reason not in STOP_REASONS:
        raise ValueError("certificate.stop_reason is unsupported")
    parsed_snapshot = RequestSnapshot.from_value(raw.get("scope_snapshot"))
    if parsed_snapshot != expected_snapshot:
        raise ValueError("certificate scope_snapshot differs from the host snapshot")
    if raw.get("data_revision") != expected_snapshot.data_revision.as_dict():
        raise ValueError("certificate data_revision differs from the host snapshot")
    if raw.get("inference_revision") != expected_snapshot.inference_revision.as_dict():
        raise ValueError("certificate inference_revision differs from the host snapshot")
    packet_sha256 = _sha256(raw.get("packet_sha256"), "certificate.packet_sha256")
    if packet_sha256 != _sha256(
        expected_packet_sha256, "expected_packet_sha256"
    ):
        raise ValueError("certificate packet_sha256 differs from the delivered packet")

    validation = raw.get("validation")
    if not isinstance(validation, Mapping):
        raise ValueError("certificate.validation must be an object")
    _exact_fields(validation, {"pack_read_complete", "host_validated"}, "validation")
    if not isinstance(validation.get("pack_read_complete"), bool) or not isinstance(
        validation.get("host_validated"), bool
    ):
        raise ValueError("certificate.validation values must be boolean")
    if bool(validation.get("pack_read_complete")) != bool(pack_read_complete):
        raise ValueError("certificate cannot choose pack_read_complete")
    if bool(validation.get("host_validated")) != bool(host_validated):
        raise ValueError("certificate cannot choose host_validated")

    allowed_sources = {str(item) for item in allowed_source_keys if str(item)}
    allowed_participants = {
        str(item) for item in allowed_participant_keys if str(item)
    }
    aggregates = _parse_items(
        raw.get("aggregates", []), "aggregates", limit=MAX_CERTIFICATE_AGGREGATES,
        parser=lambda item, index: ActivityEvidenceAggregate.from_value(
            item, snapshot=expected_snapshot, allowed_participants=allowed_participants),
    )
    aggregates_by_id = {item.aggregate_id: item for item in aggregates}
    if len(aggregates_by_id) != len(aggregates):
        raise ValueError("certificate aggregates contain duplicate IDs")
    for aggregate in aggregates:
        authorized = (allowed_aggregates or {}).get(aggregate.aggregate_id)
        if authorized is None or aggregate.as_dict() != authorized:
            raise ValueError("certificate aggregate differs from the host allowlist")
    derivations = _parse_items(raw.get("derivations", []), "derivations", limit=MAX_STORED_DERIVATIONS,
                              parser=lambda item, index: StoredDerivationEvidence.from_value(
                                  item, snapshot=expected_snapshot))
    derivations_by_id = {item.derivation_id: item for item in derivations}
    if len(derivations_by_id) != len(derivations):
        raise ValueError("certificate derivations contain duplicate IDs")
    for derivation in derivations:
        if derivation.as_dict() != (allowed_derivations or {}).get(derivation.derivation_id):
            raise ValueError("certificate derivation differs from the host allowlist")
    subjects = _parse_items(
        raw.get("subjects"),
        "subjects",
        limit=16,
        parser=lambda item, index: CertificateSubject.from_value(
            item,
            allowed_sources=allowed_sources,
            allowed_participants=allowed_participants,
            cutoff_at=expected_snapshot.cutoff_at,
            field=f"subjects[{index}]",
        ),
    )
    referents = _parse_items(
        raw.get("referents", []), "referents", limit=16,
        parser=lambda item, index: CertificateReferent.from_value(
            item, allowed_sources=allowed_sources,
            allowed_participants=allowed_participants,
            cutoff_at=expected_snapshot.cutoff_at, field=f"referents[{index}]",
            allowed_derivation_ids=set(derivations_by_id),
        ),
    )
    referents_by_id = {item.referent_id: item for item in referents}
    if len(referents_by_id) != len(referents):
        raise ValueError("certificate referents contain duplicate IDs")
    if referents and subjects != tuple(
        item.as_subject() for item in referents if item.referent_type == "PARTICIPANT"
    ):
        raise ValueError("certificate subjects differ from the typed participant projection")
    atoms = _parse_items(
        raw.get("atoms"),
        "atoms",
        limit=32,
        parser=lambda item, index: EvidenceAtom.from_value(
            item,
            allowed_sources=allowed_sources,
            allowed_participants=allowed_participants,
            field=f"atoms[{index}]",
            allowed_aggregate_ids=set(aggregates_by_id),
            source_roles=source_roles,
            allowed_derivations=derivations_by_id,
        ),
    )
    if {key for atom in atoms for key in atom.aggregate_ids} != set(aggregates_by_id):
        raise ValueError("certificate aggregates must match the referenced aggregate IDs")
    if {key for item in (*atoms, *referents) for key in item.derivation_ids} != set(derivations_by_id):
        raise ValueError("certificate derivations must match the referenced derivation IDs")
    for atom in atoms:
        if atom.aggregate_ids and any(
            aggregates_by_id[key].participant_key != atom.subject_participant_key
            for key in atom.aggregate_ids
        ):
            raise ValueError("activity statistic subject differs from its aggregate participant")
        if not atom.subject_referent_id:
            if referents and atom.subject_participant_key:
                raise ValueError("typed atom participant requires subject_referent_id")
            continue
        referent = referents_by_id.get(atom.subject_referent_id)
        if referent is None:
            raise ValueError("atom subject_referent_id references an unknown referent")
        if atom.subject_participant_key != referent.participant_key:
            raise ValueError("atom subject participant differs from its typed referent")
    atom_ids = [item.atom_id for item in atoms]
    if len(atom_ids) != len(set(atom_ids)):
        raise ValueError("certificate atoms contain duplicate IDs")
    atom_id_set = set(atom_ids)
    required_ids = {
        item.atom_id for item in atoms if item.importance == "REQUIRED"
    }
    must_include = _string_tuple(
        raw.get("must_include"),
        "must_include",
        limit=32,
        item_limit=80,
    )
    if any(not _ID_RE.fullmatch(item) for item in must_include):
        raise ValueError("must_include contains an invalid atom ID")
    if set(must_include) != required_ids:
        raise ValueError("must_include must name every and only REQUIRED atom")

    guards = _parse_items(
        raw.get("must_not_upgrade"),
        "must_not_upgrade",
        limit=16,
        parser=lambda item, index: UpgradeGuard.from_value(
            item, field=f"must_not_upgrade[{index}]"
        ),
    )
    conflicts = _parse_items(
        raw.get("conflicts"),
        "conflicts",
        limit=MAX_CERTIFICATE_CONFLICTS,
        parser=lambda item, index: CertificateQualification.from_value(
            item,
            allowed_sources=allowed_sources,
            field=f"conflicts[{index}]",
        ),
    )
    unresolved = _parse_items(
        raw.get("unresolved"),
        "unresolved",
        limit=MAX_CERTIFICATE_UNRESOLVED,
        parser=lambda item, index: CertificateQualification.from_value(
            item,
            allowed_sources=allowed_sources,
            field=f"unresolved[{index}]",
            allow_packet_coverage_gap=True,
        ),
    )
    obligations = _parse_items(
        raw.get("open_obligations"),
        "open_obligations",
        limit=24,
        parser=lambda item, index: OpenObligation.from_value(
            item, field=f"open_obligations[{index}]"
        ),
    )
    obligation_ids = [item.obligation_id for item in obligations]
    if len(obligation_ids) != len(set(obligation_ids)):
        raise ValueError("open_obligations contains duplicate IDs")
    referenced_atom_ids = {
        atom_id
        for guard in guards
        for atom_id in guard.atom_ids
    } | {
        atom_id
        for qualification in (*conflicts, *unresolved)
        for atom_id in qualification.atom_ids
    }
    if not referenced_atom_ids.issubset(atom_id_set):
        raise ValueError("certificate references an unknown atom ID")

    has_ambiguous_subject = any(
        item.reference_mode in {"AMBIGUOUS", "UNBOUND"} for item in subjects
    )
    has_critical_open = any(item.critical for item in obligations)
    if status == "CERTIFIED":
        if any(item.basis == "PACKET_COVERAGE_GAP" for item in unresolved):
            raise ValueError("CERTIFIED cannot retain packet coverage gaps")
        if not host_validated or not pack_read_complete:
            raise ValueError("CERTIFIED requires a complete host-validated packet")
        if not atoms:
            raise ValueError("CERTIFIED requires at least one evidence atom")
        if not must_include:
            raise ValueError("CERTIFIED requires at least one REQUIRED atom")
        if has_critical_open or has_ambiguous_subject:
            raise ValueError("CERTIFIED cannot retain critical or identity ambiguity")
        if stop_reason != "CERTIFIED_CLOSE":
            raise ValueError("CERTIFIED requires stop_reason=CERTIFIED_CLOSE")
    elif status == "SEMANTIC_NONE":
        if not host_validated or not pack_read_complete:
            raise ValueError("SEMANTIC_NONE requires a complete host-validated packet")
        if any((subjects, referents, aggregates, derivations, atoms, must_include, guards, conflicts, unresolved, obligations)):
            raise ValueError("SEMANTIC_NONE cannot carry evidence or open obligations")
        if stop_reason != "SEMANTIC_NONE":
            raise ValueError("SEMANTIC_NONE requires stop_reason=SEMANTIC_NONE")
    elif status == "REQUEST_L3":
        if stop_reason != "REQUEST_L3" or not obligations:
            raise ValueError("REQUEST_L3 requires open_obligations")
        if not any(
            item.discriminator and item.expected_information_gain
            for item in obligations
        ):
            raise ValueError(
                "REQUEST_L3 requires a discriminator and expected information gain"
            )
    elif status == "SAFETY_ABSTAIN":
        if stop_reason != "SAFETY_ABSTAIN":
            raise ValueError("SAFETY_ABSTAIN requires matching stop_reason")
        if not (unresolved or conflicts or obligations or has_ambiguous_subject):
            raise ValueError("SAFETY_ABSTAIN must expose why it abstained")
    elif status == "PARTIAL":
        if stop_reason not in {
            "FRONTIER_EXHAUSTED",
            "SATURATED",
        }:
            raise ValueError("PARTIAL requires an incomplete stop reason")
        if not (unresolved or conflicts or obligations):
            raise ValueError("PARTIAL must expose an unresolved condition")

    return EvidenceCertificateV2(
        status=status,
        scope_snapshot=parsed_snapshot,
        packet_sha256=packet_sha256,
        subjects=subjects,
        atoms=atoms,
        must_include=must_include,
        must_not_upgrade=guards,
        conflicts=conflicts,
        unresolved=unresolved,
        open_obligations=obligations,
        stop_reason=stop_reason,
        pack_read_complete=bool(pack_read_complete),
        host_validated=bool(host_validated),
        referents=referents,
        aggregates=aggregates,
        derivations=derivations,
    )
