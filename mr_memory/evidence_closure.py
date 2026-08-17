from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .brief import EvidenceBrief, parse_evidence_brief


OBLIGATION_STATUSES = {
    "OPEN",
    "SUPPORTED",
    "REFUTED",
    "CONTESTED",
    "AMBIGUOUS",
    "EXHAUSTED",
}
TERMINAL_OBLIGATION_STATUSES = OBLIGATION_STATUSES - {"OPEN"}
INTERPRETATION_STATUSES = {
    "CANDIDATE",
    "SUPPORTED",
    "REFUTED",
    "CONTESTED",
    "UNRESOLVED",
}
UNCERTAINTY_STATUSES = {"OPEN", "PRESERVED", "RESOLVED"}
BINDING_MODES = {
    "HOST",
    "STRUCTURED_REF",
    "UNIQUE_ALIAS",
    "AMBIGUOUS",
    "UNBOUND",
}
STOP_REASONS = {
    "CONTINUE",
    "CERTIFIED_CLOSE",
    "SAFETY_ABSTAIN",
    "FRONTIER_EXHAUSTED",
    "SATURATED",
    "BUDGET_EXHAUSTED",
}

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}$")
_TENANT_ARGUMENT_KEYS = {
    "umo",
    "scope",
    "scope_id",
    "group",
    "group_id",
    "platform_id",
    "cutoff_at",
    "before_sent_at",
}
_REQUIRED_REVISIONS = {
    "message",
    "graph",
    "identity",
    "relation",
    "feedback",
    "protocol",
}


def _text(
    value: object,
    field: str,
    *,
    limit: int,
    required: bool = False,
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
    if not _HASH_RE.fullmatch(result):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return result


def _string_tuple(
    value: object,
    field: str,
    *,
    limit: int,
    item_limit: int,
) -> tuple[str, ...]:
    if value is None:
        raw: list[object] = []
    elif isinstance(value, list):
        raw = value
    else:
        raise ValueError(f"{field} must be an array")
    if len(raw) > limit:
        raise ValueError(f"{field} exceeds {limit} items")
    return tuple(
        dict.fromkeys(
            _text(item, f"{field}[]", limit=item_limit, required=True)
            for item in raw
        )
    )


def _integer_tuple(
    value: object,
    field: str,
    *,
    limit: int,
) -> tuple[int, ...]:
    if value is None:
        raw: list[object] = []
    elif isinstance(value, list):
        raw = value
    else:
        raise ValueError(f"{field} must be an array")
    if len(raw) > limit:
        raise ValueError(f"{field} exceeds {limit} items")
    result: list[int] = []
    for index, item in enumerate(raw):
        try:
            identifier = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field}[{index}] must be an integer") from exc
        if identifier <= 0:
            raise ValueError(f"{field}[{index}] must be positive")
        if identifier not in result:
            result.append(identifier)
    return tuple(result)


def _source_tuple(
    value: object,
    field: str,
    *,
    allowed: set[str],
    limit: int = 64,
) -> tuple[str, ...]:
    result = _string_tuple(value, field, limit=limit, item_limit=256)
    if not set(result).issubset(allowed):
        raise ValueError(f"{field} cites evidence outside the delivered allowlist")
    return result


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
        raise ValueError("closure response must be one JSON object") from exc
    if not isinstance(parsed, dict):
        raise ValueError("closure response must be one JSON object")
    return parsed


def _canonical_json(value: object, *, field: str, max_chars: int) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be JSON serializable") from exc
    if len(encoded) > max_chars:
        raise ValueError(f"{field} exceeds {max_chars} serialized characters")
    return encoded


def _contains_tenant_argument(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).strip().casefold() in _TENANT_ARGUMENT_KEYS:
                return True
            if _contains_tenant_argument(child):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_tenant_argument(item) for item in value)
    return False


@dataclass(frozen=True, slots=True)
class RevisionVector:
    values: tuple[tuple[str, str], ...]

    @classmethod
    def from_value(cls, value: object) -> RevisionVector:
        if not isinstance(value, Mapping):
            raise ValueError("contract.revision_vector must be an object")
        normalized: dict[str, str] = {}
        for raw_key, raw_value in value.items():
            key = _identifier(raw_key, "contract.revision_vector key").casefold()
            normalized[key] = _text(
                raw_value,
                f"contract.revision_vector.{key}",
                limit=160,
                required=True,
            )
        missing = _REQUIRED_REVISIONS - set(normalized)
        if missing:
            raise ValueError(
                "contract.revision_vector is missing: " + ", ".join(sorted(missing))
            )
        return cls(tuple(sorted(normalized.items())))

    def as_dict(self) -> dict[str, str]:
        return dict(self.values)


@dataclass(frozen=True, slots=True)
class SubjectBinding:
    reference: str
    participant_key: str
    mode: str
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
        field: str,
    ) -> SubjectBinding:
        if not isinstance(value, Mapping):
            raise ValueError(f"{field} must be an object")
        mode = str(value.get("mode") or "").strip().upper()
        if mode not in BINDING_MODES:
            raise ValueError(f"{field}.mode is unsupported")
        participant_key = _text(
            value.get("participant_key"),
            f"{field}.participant_key",
            limit=256,
        )
        candidates = _string_tuple(
            value.get("candidate_participant_keys"),
            f"{field}.candidate_participant_keys",
            limit=20,
            item_limit=256,
        )
        if participant_key and participant_key not in allowed_participants:
            raise ValueError(f"{field}.participant_key is not host-authorized")
        if not set(candidates).issubset(allowed_participants):
            raise ValueError(f"{field} contains a non-authorized identity candidate")
        if mode in {"HOST", "STRUCTURED_REF", "UNIQUE_ALIAS"}:
            if not participant_key or candidates:
                raise ValueError(
                    f"{field} resolved modes require exactly one participant_key"
                )
        elif mode == "AMBIGUOUS":
            if participant_key or len(candidates) < 2:
                raise ValueError(
                    f"{field} ambiguous mode requires at least two candidates"
                )
        elif participant_key or candidates:
            raise ValueError(f"{field} unbound mode cannot select an identity")
        raw_valid_at = value.get("valid_at")
        valid_at = None if raw_valid_at in (None, "") else int(raw_valid_at)
        if valid_at is not None and valid_at <= 0:
            raise ValueError(f"{field}.valid_at must be positive")
        return cls(
            reference=_text(
                value.get("reference"),
                f"{field}.reference",
                limit=240,
                required=True,
            ),
            participant_key=participant_key,
            mode=mode,
            candidate_participant_keys=candidates,
            source_keys=_source_tuple(
                value.get("source_keys"),
                f"{field}.source_keys",
                allowed=allowed_sources,
                limit=16,
            ),
            valid_at=valid_at,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "reference": self.reference,
            "participant_key": self.participant_key,
            "mode": self.mode,
            "candidate_participant_keys": list(self.candidate_participant_keys),
            "source_keys": list(self.source_keys),
            "valid_at": self.valid_at,
        }


@dataclass(frozen=True, slots=True)
class EvidenceObligation:
    obligation_id: str
    kind: str
    question: str
    critical: bool
    status: str
    support_keys: tuple[str, ...]
    counter_keys: tuple[str, ...]
    last_changed_step: int

    @classmethod
    def from_value(
        cls,
        value: object,
        *,
        allowed_sources: set[str],
        field: str,
    ) -> EvidenceObligation:
        if not isinstance(value, Mapping):
            raise ValueError(f"{field} must be an object")
        status = str(value.get("status") or "").strip().upper()
        if status not in OBLIGATION_STATUSES:
            raise ValueError(f"{field}.status is unsupported")
        changed = int(value.get("last_changed_step") or 0)
        if changed < 0:
            raise ValueError(f"{field}.last_changed_step must be non-negative")
        support = _source_tuple(
            value.get("support_keys"),
            f"{field}.support_keys",
            allowed=allowed_sources,
        )
        counter = _source_tuple(
            value.get("counter_keys"),
            f"{field}.counter_keys",
            allowed=allowed_sources,
        )
        if status == "SUPPORTED" and not support:
            raise ValueError(f"{field} cannot be supported without evidence")
        if status == "REFUTED" and not counter:
            raise ValueError(f"{field} cannot be refuted without counterevidence")
        if status == "CONTESTED" and (not support or not counter):
            raise ValueError(f"{field} contested status needs both evidence sides")
        return cls(
            obligation_id=_identifier(
                value.get("id"),
                f"{field}.id",
            ),
            kind=_identifier(value.get("kind"), f"{field}.kind").casefold(),
            question=_text(
                value.get("question"),
                f"{field}.question",
                limit=800,
                required=True,
            ),
            critical=bool(value.get("critical", False)),
            status=status,
            support_keys=support,
            counter_keys=counter,
            last_changed_step=changed,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.obligation_id,
            "kind": self.kind,
            "question": self.question,
            "critical": self.critical,
            "status": self.status,
            "support_keys": list(self.support_keys),
            "counter_keys": list(self.counter_keys),
            "last_changed_step": self.last_changed_step,
        }


@dataclass(frozen=True, slots=True)
class CompetingInterpretation:
    interpretation_id: str
    statement: str
    status: str
    support_keys: tuple[str, ...]
    counter_keys: tuple[str, ...]
    uncertainty: str

    @classmethod
    def from_value(
        cls,
        value: object,
        *,
        allowed_sources: set[str],
        field: str,
    ) -> CompetingInterpretation:
        if not isinstance(value, Mapping):
            raise ValueError(f"{field} must be an object")
        status = str(value.get("status") or "").strip().upper()
        if status not in INTERPRETATION_STATUSES:
            raise ValueError(f"{field}.status is unsupported")
        support = _source_tuple(
            value.get("support_keys"),
            f"{field}.support_keys",
            allowed=allowed_sources,
        )
        counter = _source_tuple(
            value.get("counter_keys"),
            f"{field}.counter_keys",
            allowed=allowed_sources,
        )
        if status == "SUPPORTED" and not support:
            raise ValueError(f"{field} supported status needs evidence")
        if status == "REFUTED" and not counter:
            raise ValueError(f"{field} refuted status needs counterevidence")
        if status == "CONTESTED" and (not support or not counter):
            raise ValueError(f"{field} contested status needs both evidence sides")
        return cls(
            interpretation_id=_identifier(value.get("id"), f"{field}.id"),
            statement=_text(
                value.get("statement"),
                f"{field}.statement",
                limit=1200,
                required=True,
            ),
            status=status,
            support_keys=support,
            counter_keys=counter,
            uncertainty=_text(
                value.get("uncertainty"),
                f"{field}.uncertainty",
                limit=800,
            ),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.interpretation_id,
            "statement": self.statement,
            "status": self.status,
            "support_keys": list(self.support_keys),
            "counter_keys": list(self.counter_keys),
            "uncertainty": self.uncertainty,
        }


@dataclass(frozen=True, slots=True)
class UncertaintyConstraint:
    constraint_id: str
    statement: str
    status: str
    source_keys: tuple[str, ...]

    @classmethod
    def from_value(
        cls,
        value: object,
        *,
        allowed_sources: set[str],
        field: str,
    ) -> UncertaintyConstraint:
        if not isinstance(value, Mapping):
            raise ValueError(f"{field} must be an object")
        status = str(value.get("status") or "").strip().upper()
        if status not in UNCERTAINTY_STATUSES:
            raise ValueError(f"{field}.status is unsupported")
        sources = _source_tuple(
            value.get("source_keys"),
            f"{field}.source_keys",
            allowed=allowed_sources,
        )
        if status != "OPEN" and not sources:
            raise ValueError(f"{field} closed uncertainty needs source evidence")
        return cls(
            constraint_id=_identifier(value.get("id"), f"{field}.id"),
            statement=_text(
                value.get("statement"),
                f"{field}.statement",
                limit=800,
                required=True,
            ),
            status=status,
            source_keys=sources,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.constraint_id,
            "statement": self.statement,
            "status": self.status,
            "source_keys": list(self.source_keys),
        }


@dataclass(frozen=True, slots=True)
class RetrievalAction:
    obligation_id: str
    tool_name: str
    arguments: dict[str, object]
    discriminator: str
    expected_delta: str

    @classmethod
    def from_value(cls, value: object, *, field: str) -> RetrievalAction:
        if not isinstance(value, Mapping):
            raise ValueError(f"{field} must be an object")
        arguments = value.get("arguments")
        if not isinstance(arguments, Mapping):
            raise ValueError(f"{field}.arguments must be an object")
        copied_arguments = dict(arguments)
        _canonical_json(
            copied_arguments,
            field=f"{field}.arguments",
            max_chars=8000,
        )
        if _contains_tenant_argument(copied_arguments):
            raise ValueError(
                f"{field}.arguments cannot choose scope, tenant, or cutoff"
            )
        return cls(
            obligation_id=_identifier(
                value.get("obligation_id"),
                f"{field}.obligation_id",
            ),
            tool_name=_identifier(
                value.get("tool_name"),
                f"{field}.tool_name",
            ),
            arguments=copied_arguments,
            discriminator=_text(
                value.get("discriminator"),
                f"{field}.discriminator",
                limit=500,
                required=True,
            ),
            expected_delta=_text(
                value.get("expected_delta"),
                f"{field}.expected_delta",
                limit=500,
                required=True,
            ),
        )

    def signature(self) -> str:
        encoded = _canonical_json(
            {
                "obligation_id": self.obligation_id,
                "tool_name": self.tool_name,
                "arguments": self.arguments,
                "discriminator": self.discriminator,
            },
            field="action signature",
            max_chars=10000,
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, object]:
        return {
            "obligation_id": self.obligation_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "discriminator": self.discriminator,
            "expected_delta": self.expected_delta,
            "signature": self.signature(),
        }


@dataclass(frozen=True, slots=True)
class ReconstructionContract:
    contract_id: str
    scope_sha256: str
    query_sha256: str
    cutoff_at: int
    revision_vector: RevisionVector
    step_index: int
    subjects: tuple[SubjectBinding, ...]
    obligations: tuple[EvidenceObligation, ...]
    interpretations: tuple[CompetingInterpretation, ...]
    uncertainties: tuple[UncertaintyConstraint, ...]
    guarded_claims: tuple[str, ...]
    visited_source_keys: tuple[str, ...]
    selected_edge_ids: tuple[int, ...]
    selected_hypothesis_ids: tuple[int, ...]
    tried_action_signatures: tuple[str, ...]
    exhausted_discriminators: tuple[str, ...]
    frontier_discriminators: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_id": self.contract_id,
            "scope_sha256": self.scope_sha256,
            "query_sha256": self.query_sha256,
            "cutoff_at": self.cutoff_at,
            "revision_vector": self.revision_vector.as_dict(),
            "step_index": self.step_index,
            "subjects": [item.as_dict() for item in self.subjects],
            "obligations": [item.as_dict() for item in self.obligations],
            "interpretations": [item.as_dict() for item in self.interpretations],
            "uncertainties": [item.as_dict() for item in self.uncertainties],
            "guarded_claims": list(self.guarded_claims),
            "visited_source_keys": list(self.visited_source_keys),
            "selected_edge_ids": list(self.selected_edge_ids),
            "selected_hypothesis_ids": list(self.selected_hypothesis_ids),
            "tried_action_signatures": list(self.tried_action_signatures),
            "exhausted_discriminators": list(self.exhausted_discriminators),
            "frontier_discriminators": list(self.frontier_discriminators),
        }


@dataclass(frozen=True, slots=True)
class ContractTurn:
    contract: ReconstructionContract
    actions: tuple[RetrievalAction, ...]
    brief: EvidenceBrief | None
    terminal: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "contract": self.contract.as_dict(),
            "actions": [item.as_dict() for item in self.actions],
            "memory_brief": self.brief.as_dict() if self.brief is not None else None,
            "terminal": self.terminal,
        }


@dataclass(frozen=True, slots=True)
class GainVector:
    new_source_keys: tuple[str, ...] = ()
    new_graph_anchors: tuple[str, ...] = ()
    new_result_hashes: tuple[str, ...] = ()
    identity_changes: tuple[str, ...] = ()
    obligation_transitions: tuple[str, ...] = ()
    interpretation_transitions: tuple[str, ...] = ()
    uncertainty_transitions: tuple[str, ...] = ()
    frontier_expansions: tuple[str, ...] = ()

    @property
    def has_progress(self) -> bool:
        return any(
            (
                self.new_source_keys,
                self.new_graph_anchors,
                self.new_result_hashes,
                self.identity_changes,
                self.obligation_transitions,
                self.interpretation_transitions,
                self.uncertainty_transitions,
                self.frontier_expansions,
            )
        )

    @property
    def has_semantic_progress(self) -> bool:
        return any(
            (
                self.identity_changes,
                self.obligation_transitions,
                self.interpretation_transitions,
                self.uncertainty_transitions,
            )
        )

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "new_source_keys": list(self.new_source_keys),
            "new_graph_anchors": list(self.new_graph_anchors),
            "new_result_hashes": list(self.new_result_hashes),
            "identity_changes": list(self.identity_changes),
            "obligation_transitions": list(self.obligation_transitions),
            "interpretation_transitions": list(self.interpretation_transitions),
            "uncertainty_transitions": list(self.uncertainty_transitions),
            "frontier_expansions": list(self.frontier_expansions),
        }


@dataclass(frozen=True, slots=True)
class BudgetState:
    model_calls: int
    max_model_calls: int
    retrieval_rounds: int
    max_retrieval_rounds: int
    elapsed_ms: float = 0.0
    deadline_ms: float = 0.0
    measured_tokens: int = 0
    max_measured_tokens: int = 0

    def exhausted(self) -> bool:
        return any(
            (
                self.model_calls >= self.max_model_calls,
                self.retrieval_rounds >= self.max_retrieval_rounds,
                self.deadline_ms > 0 and self.elapsed_ms >= self.deadline_ms,
                self.max_measured_tokens > 0
                and self.measured_tokens >= self.max_measured_tokens,
            )
        )


@dataclass(frozen=True, slots=True)
class StopDecision:
    stop: bool
    reason: str
    force_unresolved: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "stop": self.stop,
            "reason": self.reason,
            "force_unresolved": self.force_unresolved,
        }


def _parse_contract(
    value: object,
    *,
    allowed_sources: set[str],
    allowed_participants: set[str],
    allowed_edge_ids: set[int],
    allowed_hypothesis_ids: set[int],
) -> ReconstructionContract:
    if not isinstance(value, Mapping):
        raise ValueError("contract must be an object")
    step_index = int(value.get("step_index") or 0)
    if step_index < 0:
        raise ValueError("contract.step_index must be non-negative")
    cutoff_at = int(value.get("cutoff_at") or 0)
    if cutoff_at <= 0:
        raise ValueError("contract.cutoff_at must be positive")

    raw_subjects = value.get("subjects") or []
    raw_obligations = value.get("obligations") or []
    raw_interpretations = value.get("interpretations") or []
    raw_uncertainties = value.get("uncertainties") or []
    for raw, field, limit in (
        (raw_subjects, "contract.subjects", 16),
        (raw_obligations, "contract.obligations", 24),
        (raw_interpretations, "contract.interpretations", 16),
        (raw_uncertainties, "contract.uncertainties", 16),
    ):
        if not isinstance(raw, list) or len(raw) > limit:
            raise ValueError(f"{field} must be an array with at most {limit} items")
    if not raw_obligations:
        raise ValueError("contract requires at least one evidence obligation")

    subjects = tuple(
        SubjectBinding.from_value(
            item,
            allowed_sources=allowed_sources,
            allowed_participants=allowed_participants,
            field=f"contract.subjects[{index}]",
        )
        for index, item in enumerate(raw_subjects)
    )
    obligations = tuple(
        EvidenceObligation.from_value(
            item,
            allowed_sources=allowed_sources,
            field=f"contract.obligations[{index}]",
        )
        for index, item in enumerate(raw_obligations)
    )
    if not any(item.critical for item in obligations):
        raise ValueError("contract requires at least one critical obligation")
    if not any(item.critical for item in obligations):
        raise ValueError("contract requires at least one critical evidence obligation")
    interpretations = tuple(
        CompetingInterpretation.from_value(
            item,
            allowed_sources=allowed_sources,
            field=f"contract.interpretations[{index}]",
        )
        for index, item in enumerate(raw_interpretations)
    )
    uncertainties = tuple(
        UncertaintyConstraint.from_value(
            item,
            allowed_sources=allowed_sources,
            field=f"contract.uncertainties[{index}]",
        )
        for index, item in enumerate(raw_uncertainties)
    )
    for field, items in (
        ("contract.subjects", [item.reference.casefold() for item in subjects]),
        (
            "contract.obligations",
            [item.obligation_id for item in obligations],
        ),
        (
            "contract.interpretations",
            [item.interpretation_id for item in interpretations],
        ),
        (
            "contract.uncertainties",
            [item.constraint_id for item in uncertainties],
        ),
    ):
        if len(items) != len(set(items)):
            raise ValueError(f"{field} contains duplicate identities")

    edge_ids = _integer_tuple(
        value.get("selected_edge_ids"),
        "contract.selected_edge_ids",
        limit=64,
    )
    hypothesis_ids = _integer_tuple(
        value.get("selected_hypothesis_ids"),
        "contract.selected_hypothesis_ids",
        limit=64,
    )
    if not set(edge_ids).issubset(allowed_edge_ids):
        raise ValueError("contract.selected_edge_ids contains an unavailable edge")
    if not set(hypothesis_ids).issubset(allowed_hypothesis_ids):
        raise ValueError(
            "contract.selected_hypothesis_ids contains an unavailable hypothesis"
        )

    signatures = _string_tuple(
        value.get("tried_action_signatures"),
        "contract.tried_action_signatures",
        limit=64,
        item_limit=64,
    )
    if any(not _HASH_RE.fullmatch(item) for item in signatures):
        raise ValueError("contract.tried_action_signatures must contain SHA-256 values")
    visited_sources = _source_tuple(
        value.get("visited_source_keys"),
        "contract.visited_source_keys",
        allowed=allowed_sources,
        limit=160,
    )
    referenced_sources: set[str] = set()
    for item in subjects:
        referenced_sources.update(item.source_keys)
    for item in obligations:
        referenced_sources.update(item.support_keys)
        referenced_sources.update(item.counter_keys)
        if item.last_changed_step > step_index:
            raise ValueError(
                f"obligation {item.obligation_id} changed after the current step"
            )
    for item in interpretations:
        referenced_sources.update(item.support_keys)
        referenced_sources.update(item.counter_keys)
    for item in uncertainties:
        referenced_sources.update(item.source_keys)
    if not referenced_sources.issubset(visited_sources):
        raise ValueError("contract state cites evidence not marked as visited")
    return ReconstructionContract(
        contract_id=_identifier(value.get("contract_id"), "contract.contract_id"),
        scope_sha256=_sha256(value.get("scope_sha256"), "contract.scope_sha256"),
        query_sha256=_sha256(value.get("query_sha256"), "contract.query_sha256"),
        cutoff_at=cutoff_at,
        revision_vector=RevisionVector.from_value(value.get("revision_vector")),
        step_index=step_index,
        subjects=subjects,
        obligations=obligations,
        interpretations=interpretations,
        uncertainties=uncertainties,
        guarded_claims=_string_tuple(
            value.get("guarded_claims"),
            "contract.guarded_claims",
            limit=16,
            item_limit=800,
        ),
        visited_source_keys=visited_sources,
        selected_edge_ids=edge_ids,
        selected_hypothesis_ids=hypothesis_ids,
        tried_action_signatures=signatures,
        exhausted_discriminators=_string_tuple(
            value.get("exhausted_discriminators"),
            "contract.exhausted_discriminators",
            limit=64,
            item_limit=500,
        ),
        frontier_discriminators=_string_tuple(
            value.get("frontier_discriminators"),
            "contract.frontier_discriminators",
            limit=32,
            item_limit=500,
        ),
    )


def _binding_signature(binding: SubjectBinding) -> tuple[object, ...]:
    return (
        binding.reference.casefold(),
        binding.mode,
        binding.participant_key,
        binding.candidate_participant_keys,
        binding.source_keys,
    )


def _validate_transition(
    previous: ReconstructionContract,
    current: ReconstructionContract,
) -> None:
    immutable = (
        ("contract_id", previous.contract_id, current.contract_id),
        ("scope_sha256", previous.scope_sha256, current.scope_sha256),
        ("query_sha256", previous.query_sha256, current.query_sha256),
        ("cutoff_at", previous.cutoff_at, current.cutoff_at),
        ("revision_vector", previous.revision_vector, current.revision_vector),
    )
    for field, old, new in immutable:
        if old != new:
            raise ValueError(f"contract transition changed immutable {field}")
    if current.step_index != previous.step_index + 1:
        raise ValueError("contract.step_index must advance by exactly one")
    if not set(previous.visited_source_keys).issubset(current.visited_source_keys):
        raise ValueError("contract transition removed visited evidence")
    if not set(previous.tried_action_signatures).issubset(
        current.tried_action_signatures
    ):
        raise ValueError("contract transition removed tried action signatures")

    old_obligations = {item.obligation_id: item for item in previous.obligations}
    new_obligations = {item.obligation_id: item for item in current.obligations}
    if old_obligations.keys() != new_obligations.keys():
        raise ValueError("contract transition changed the obligation set")
    binding_changed = {
        item.reference.casefold(): _binding_signature(item)
        for item in previous.subjects
    } != {
        item.reference.casefold(): _binding_signature(item) for item in current.subjects
    }
    frontier_progress = bool(
        set(current.exhausted_discriminators)
        - set(previous.exhausted_discriminators)
        or set(current.tried_action_signatures)
        - set(previous.tried_action_signatures)
    )
    old_subjects = {item.reference.casefold(): item for item in previous.subjects}
    new_subjects = {item.reference.casefold(): item for item in current.subjects}
    if old_subjects.keys() != new_subjects.keys():
        raise ValueError("contract transition changed the subject-reference set")
    for reference, old_subject in old_subjects.items():
        new_subject = new_subjects[reference]
        if not set(old_subject.source_keys).issubset(new_subject.source_keys):
            raise ValueError(f"subject binding {reference} removed evidence")
        if old_subject.mode in {"HOST", "STRUCTURED_REF"} and (
            new_subject.mode != old_subject.mode
            or new_subject.participant_key != old_subject.participant_key
        ):
            raise ValueError(
                f"subject binding {reference} attempted to rewrite host identity"
            )
        if _binding_signature(old_subject) != _binding_signature(new_subject):
            if not (
                set(new_subject.source_keys) - set(old_subject.source_keys)
                or new_subject.mode in {"HOST", "STRUCTURED_REF"}
                or (
                    new_subject.mode == "AMBIGUOUS"
                    and new_subject.candidate_participant_keys
                    != old_subject.candidate_participant_keys
                )
            ):
                raise ValueError(
                    f"subject binding {reference} changed without host evidence"
                )
    for obligation_id, old in old_obligations.items():
        new = new_obligations[obligation_id]
        if (old.kind, old.question, old.critical) != (
            new.kind,
            new.question,
            new.critical,
        ):
            raise ValueError(f"obligation {obligation_id} changed its definition")
        if not set(old.support_keys).issubset(new.support_keys) or not set(
            old.counter_keys
        ).issubset(new.counter_keys):
            raise ValueError(f"obligation {obligation_id} removed evidence")
        if old.status != new.status:
            has_new_evidence = bool(
                set(new.support_keys) - set(old.support_keys)
                or set(new.counter_keys) - set(old.counter_keys)
            )
            existing_evidence_supports_target = bool(
                (new.status == "SUPPORTED" and new.support_keys)
                or (new.status == "REFUTED" and new.counter_keys)
                or (
                    new.status == "CONTESTED"
                    and new.support_keys
                    and new.counter_keys
                )
                or (
                    new.status in {"OPEN", "AMBIGUOUS"}
                    and (new.support_keys or new.counter_keys)
                )
            )
            if new.status == "EXHAUSTED":
                transition_is_grounded = frontier_progress
            elif new.status == "AMBIGUOUS":
                transition_is_grounded = has_new_evidence or (
                    binding_changed and old.kind == "identity"
                ) or (frontier_progress and existing_evidence_supports_target)
            else:
                transition_is_grounded = has_new_evidence or (
                    binding_changed and old.kind == "identity"
                ) or (frontier_progress and existing_evidence_supports_target)
            if not transition_is_grounded:
                raise ValueError(
                    f"obligation {obligation_id} changed status without new evidence "
                    "or a host-observable frontier/binding transition"
                )
            if new.last_changed_step != current.step_index:
                raise ValueError(
                    f"obligation {obligation_id} did not record its transition step"
                )

    old_interpretations = {
        item.interpretation_id: item for item in previous.interpretations
    }
    new_interpretations = {
        item.interpretation_id: item for item in current.interpretations
    }
    if old_interpretations.keys() != new_interpretations.keys():
        raise ValueError("contract transition changed the interpretation set")
    for item_id, old_item in old_interpretations.items():
        new_item = new_interpretations[item_id]
        if old_item.statement != new_item.statement:
            raise ValueError(f"contract transition rewrote interpretation {item_id}")
        if not set(old_item.support_keys).issubset(new_item.support_keys) or not set(
            old_item.counter_keys
        ).issubset(new_item.counter_keys):
            raise ValueError(f"interpretation {item_id} removed evidence")
        if (
            old_item.status != new_item.status
            or old_item.uncertainty != new_item.uncertainty
        ) and not (
            set(new_item.support_keys) - set(old_item.support_keys)
            or set(new_item.counter_keys) - set(old_item.counter_keys)
            or (new_item.status == "UNRESOLVED" and frontier_progress)
            or (
                frontier_progress
                and (
                    (new_item.status == "SUPPORTED" and new_item.support_keys)
                    or (new_item.status == "REFUTED" and new_item.counter_keys)
                    or (
                        new_item.status == "CONTESTED"
                        and new_item.support_keys
                        and new_item.counter_keys
                    )
                )
            )
        ):
            raise ValueError(
                f"interpretation {item_id} changed without new evidence"
            )

    old_uncertainties = {
        item.constraint_id: item for item in previous.uncertainties
    }
    new_uncertainties = {
        item.constraint_id: item for item in current.uncertainties
    }
    if old_uncertainties.keys() != new_uncertainties.keys():
        raise ValueError("contract transition changed the uncertainty set")
    for item_id, old_item in old_uncertainties.items():
        new_item = new_uncertainties[item_id]
        if old_item.statement != new_item.statement:
            raise ValueError(f"contract transition rewrote uncertainty {item_id}")
        if not set(old_item.source_keys).issubset(new_item.source_keys):
            raise ValueError(f"uncertainty {item_id} removed evidence")
        if old_item.status != new_item.status and not (
            set(new_item.source_keys) - set(old_item.source_keys)
            or (new_item.status == "PRESERVED" and frontier_progress)
        ):
            raise ValueError(f"uncertainty {item_id} changed without new evidence")
    if previous.guarded_claims != current.guarded_claims:
        raise ValueError("contract transition rewrote guarded claims")


def validate_actions(
    turn: ContractTurn,
    *,
    allowed_tool_names: Iterable[str],
    tried_signatures: Iterable[str] = (),
    max_actions: int = 3,
) -> tuple[RetrievalAction, ...]:
    if len(turn.actions) > max(0, int(max_actions)):
        raise ValueError("closure turn requested too many retrieval actions")
    allowed_tools = {str(item) for item in allowed_tool_names if str(item)}
    obligations = {
        item.obligation_id: item.status for item in turn.contract.obligations
    }
    seen = {str(item) for item in tried_signatures if str(item)}
    seen.update(turn.contract.tried_action_signatures)
    accepted: list[RetrievalAction] = []
    for action in turn.actions:
        if action.tool_name not in allowed_tools:
            raise ValueError(f"retrieval tool is not allowed: {action.tool_name}")
        status = obligations.get(action.obligation_id)
        if status not in {"OPEN", "CONTESTED", "AMBIGUOUS"}:
            raise ValueError(
                "retrieval action must target an unresolved evidence obligation"
            )
        if action.discriminator in turn.contract.exhausted_discriminators:
            raise ValueError("retrieval action repeats an exhausted discriminator")
        signature = action.signature()
        if signature in seen:
            raise ValueError("duplicate or previously tried retrieval action")
        seen.add(signature)
        accepted.append(action)
    return tuple(accepted)


def parse_contract_turn(
    value: str | Mapping[str, Any],
    *,
    allowed_source_keys: Iterable[str],
    allowed_participant_keys: Iterable[str] = (),
    allowed_edge_ids: Iterable[int] = (),
    allowed_hypothesis_ids: Iterable[int] = (),
    allowed_tool_names: Iterable[str] = (),
    previous: ReconstructionContract | None = None,
    tried_action_signatures: Iterable[str] = (),
    max_actions: int = 3,
) -> ContractTurn:
    raw = _extract_object(value)
    allowed_sources = {str(item) for item in allowed_source_keys if str(item)}
    contract = _parse_contract(
        raw.get("contract"),
        allowed_sources=allowed_sources,
        allowed_participants={
            str(item) for item in allowed_participant_keys if str(item)
        },
        allowed_edge_ids={int(item) for item in allowed_edge_ids if int(item) > 0},
        allowed_hypothesis_ids={
            int(item) for item in allowed_hypothesis_ids if int(item) > 0
        },
    )
    if previous is not None:
        _validate_transition(previous, contract)
    required_tried = {str(item) for item in tried_action_signatures if str(item)}
    if not required_tried.issubset(contract.tried_action_signatures):
        raise ValueError("contract omitted a host-recorded tried action signature")
    raw_actions = raw.get("actions") or []
    if not isinstance(raw_actions, list):
        raise ValueError("actions must be an array")
    actions = tuple(
        RetrievalAction.from_value(item, field=f"actions[{index}]")
        for index, item in enumerate(raw_actions)
    )
    raw_brief = raw.get("memory_brief")
    brief = None
    if raw_brief is not None:
        if not isinstance(raw_brief, Mapping):
            raise ValueError("memory_brief must be an object or null")
        brief = parse_evidence_brief(
            json.dumps(raw_brief, ensure_ascii=False, separators=(",", ":")),
            allowed_source_keys=contract.visited_source_keys,
        )
    terminal = bool(raw.get("terminal", False))
    turn = ContractTurn(
        contract=contract,
        actions=actions,
        brief=brief,
        terminal=terminal,
    )
    validate_actions(
        turn,
        allowed_tool_names=allowed_tool_names,
        tried_signatures=(
            *(previous.tried_action_signatures if previous else ()),
            *required_tried,
        ),
        max_actions=max_actions,
    )
    if terminal and actions:
        raise ValueError("terminal closure turns cannot request retrieval actions")
    if terminal:
        if any(
            item.critical and item.status == "OPEN" for item in contract.obligations
        ):
            raise ValueError("terminal closure left a critical obligation open")
        if any(item.status == "OPEN" for item in contract.uncertainties):
            raise ValueError("terminal closure left an uncertainty constraint open")
        has_evidence = any(
            item.support_keys or item.counter_keys for item in contract.obligations
        )
        if has_evidence and brief is None:
            raise ValueError("evidence-bearing terminal closure requires a brief")
        has_qualification = any(
            item.critical
            and item.status in {"CONTESTED", "AMBIGUOUS", "EXHAUSTED"}
            for item in contract.obligations
        ) or any(
            item.status == "PRESERVED" for item in contract.uncertainties
        ) or any(
            item.mode in {"AMBIGUOUS", "UNBOUND"} for item in contract.subjects
        )
        if has_qualification and (
            brief is None or not (brief.conflicts or brief.unresolved)
        ):
            raise ValueError(
                "terminal closure requires an explicit unresolved brief for every "
                "critical conflict or uncertainty"
            )
    return turn


def compile_or_update_contract(
    value: str | Mapping[str, Any],
    **kwargs: object,
) -> ContractTurn:
    """Parse a first contract or one evidence-bound contract revision.

    The function is deliberately pure: callers supply every allowlist and retain
    ownership of provider calls, scoped retrieval, persistence, and timeouts.
    """

    return parse_contract_turn(value, **kwargs)  # type: ignore[arg-type]


def evidence_gain(
    previous: ReconstructionContract,
    current: ReconstructionContract,
    *,
    delivered_source_keys: Iterable[str] = (),
    delivered_graph_anchors: Iterable[str] = (),
    previous_graph_anchors: Iterable[str] = (),
    result_hashes: Iterable[str] = (),
    previous_result_hashes: Iterable[str] = (),
) -> GainVector:
    old_sources = set(previous.visited_source_keys)
    delivered_sources = {str(item) for item in delivered_source_keys if str(item)}
    current_sources = set(current.visited_source_keys)
    new_sources = (current_sources | delivered_sources) - old_sources

    old_subjects = {
        item.reference.casefold(): _binding_signature(item)
        for item in previous.subjects
    }
    current_subjects = {
        item.reference.casefold(): _binding_signature(item) for item in current.subjects
    }
    identity_changes = tuple(
        sorted(
            reference
            for reference in old_subjects.keys() | current_subjects.keys()
            if old_subjects.get(reference) != current_subjects.get(reference)
        )
    )

    old_obligations = {item.obligation_id: item for item in previous.obligations}
    obligation_transitions = tuple(
        sorted(
            f"{item.obligation_id}:{old_obligations[item.obligation_id].status}->{item.status}"
            for item in current.obligations
            if item.obligation_id in old_obligations
            and (
                old_obligations[item.obligation_id].status != item.status
                or old_obligations[item.obligation_id].support_keys
                != item.support_keys
                or old_obligations[item.obligation_id].counter_keys
                != item.counter_keys
            )
        )
    )
    old_interpretations = {
        item.interpretation_id: item for item in previous.interpretations
    }
    interpretation_transitions = tuple(
        sorted(
            f"{item.interpretation_id}:{old_interpretations[item.interpretation_id].status}->{item.status}"
            for item in current.interpretations
            if item.interpretation_id in old_interpretations
            and (
                old_interpretations[item.interpretation_id].status != item.status
                or old_interpretations[item.interpretation_id].support_keys
                != item.support_keys
                or old_interpretations[item.interpretation_id].counter_keys
                != item.counter_keys
                or old_interpretations[item.interpretation_id].uncertainty
                != item.uncertainty
            )
        )
    )
    old_uncertainties = {
        item.constraint_id: item for item in previous.uncertainties
    }
    uncertainty_transitions = tuple(
        sorted(
            f"{item.constraint_id}:{old_uncertainties[item.constraint_id].status}->{item.status}"
            for item in current.uncertainties
            if item.constraint_id in old_uncertainties
            and (
                old_uncertainties[item.constraint_id].status != item.status
                or old_uncertainties[item.constraint_id].source_keys
                != item.source_keys
            )
        )
    )
    return GainVector(
        new_source_keys=tuple(sorted(new_sources)),
        new_graph_anchors=tuple(
            sorted(
                {str(item) for item in delivered_graph_anchors if str(item)}
                - {str(item) for item in previous_graph_anchors if str(item)}
            )
        ),
        new_result_hashes=tuple(
            sorted(
                {str(item) for item in result_hashes if str(item)}
                - {str(item) for item in previous_result_hashes if str(item)}
            )
        ),
        identity_changes=identity_changes,
        obligation_transitions=obligation_transitions,
        interpretation_transitions=interpretation_transitions,
        uncertainty_transitions=uncertainty_transitions,
        frontier_expansions=tuple(
            sorted(
                set(current.frontier_discriminators)
                - set(previous.frontier_discriminators)
            )
        ),
    )


def should_stop(
    contract: ReconstructionContract,
    *,
    actions: Iterable[RetrievalAction] = (),
    budget: BudgetState,
    gain: GainVector | None = None,
    consecutive_no_progress_rounds: int = 0,
    saturation_rounds: int = 2,
) -> StopDecision:
    action_list = tuple(actions)
    critical = tuple(item for item in contract.obligations if item.critical)
    unresolved_identity = any(
        item.mode in {"AMBIGUOUS", "UNBOUND"} for item in contract.subjects
    ) or any(
        item.kind == "identity" and item.status in {"AMBIGUOUS", "EXHAUSTED"}
        for item in critical
    )
    all_critical_closed = all(
        item.status in TERMINAL_OBLIGATION_STATUSES for item in critical
    )
    all_uncertainty_closed = all(
        item.status != "OPEN" for item in contract.uncertainties
    )

    if all_critical_closed and all_uncertainty_closed and not action_list:
        if unresolved_identity:
            return StopDecision(True, "SAFETY_ABSTAIN", True)
        return StopDecision(True, "CERTIFIED_CLOSE", False)
    if budget.exhausted():
        return StopDecision(True, "BUDGET_EXHAUSTED", True)
    if (
        gain is not None
        and not gain.has_progress
        and consecutive_no_progress_rounds >= max(1, int(saturation_rounds))
    ):
        return StopDecision(True, "SATURATED", True)
    if not action_list:
        return StopDecision(True, "FRONTIER_EXHAUSTED", True)
    return StopDecision(False, "CONTINUE", False)
