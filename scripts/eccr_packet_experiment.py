from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mr_memory.backtest import canonical_json
from mr_memory.brief import EvidenceBrief, parse_evidence_brief
from mr_memory.evidence_closure import ContractTurn, ReconstructionContract
from mr_memory.runtime import (
    FAST_RECONSTRUCTION_SYSTEM_PROMPT,
    parse_reconstruction_plan,
    parse_structured_response,
    reconstruction_packet_allowlist,
)
from scripts.masked_ab_experiment import (
    ECCR_SYSTEM_PROMPT,
    PilotBudget,
    _assert_usage_resumable,
    _eccr_host_fields,
    _eccr_participant_allowlist,
    _file_sha256,
    _parse_eccr_turn,
    _pilot_completion,
    _pilot_run_usage,
    _provider_config,
    _provider_fingerprint,
    _score_pilot_gold,
    _stable_json_hash,
    _usage_ledger_audit,
)

SCHEMA_VERSION = "eccr.packet.experiment.v1"
CASE_SCHEMA_VERSION = "eccr.packet.case.v1"
GOLD_SCHEMA_VERSION = "eccr.packet.gold.v1"
DIAGNOSTIC_LAYER = "oracle_synthesis_diagnostic"
ARMS = {"one-pass", "eccr", "eccr-audit", "deterministic"}

_PACKET_AUDIT_DISCRIMINATOR = "fixed-packet-counterexample-coverage-audit"

ECCR_AUDIT_SYSTEM_PROMPT = (
    ECCR_SYSTEM_PROMPT
    + "\nThis is a fixed-packet, two-call oracle-synthesis audit. No retrieval "
    "tools exist. This two-call protocol is the explicit exception to the "
    "single-call close rule above: Call 1 only compiles open obligations, competing candidate "
    "interpretations, open uncertainty constraints, and immutable guarded "
    "claims; it must return terminal=false, memory_brief=null, and actions=[]. "
    "Call 2 receives that strict contract plus an audit-marked copy of the same "
    "packet. It must inspect counterexamples and coverage, preserve every "
    "obligation, interpretation, uncertainty, and guarded claim, then return "
    "terminal=true with actions=[]. Never claim this evaluates retrieval."
)

ONE_PASS_SYSTEM_PROMPT = (
    FAST_RECONSTRUCTION_SYSTEM_PROMPT
    + "\nThis is a packet-level synthesis diagnostic. The complete fixed evidence "
    "packet is already supplied and no retrieval is available. Do not claim that "
    "this run evaluated candidate generation or end-to-end retrieval."
)

_CASE_GOLD_KEYS = {
    "gold",
    "gold_answer",
    "evidence_groups",
    "required_semantics",
    "required_uncertainty",
    "forbidden_conclusions",
}


@dataclass(frozen=True, slots=True)
class CaseBundle:
    case_dir: Path
    case_path: Path
    packet_path: Path
    gold_path: Path
    case: dict[str, Any]
    packet: dict[str, Any]
    gold: dict[str, Any]
    hashes: dict[str, str]

    @property
    def case_id(self) -> str:
        return str(self.case["case_id"])


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return value


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _bounded_text(value: object, *, field: str, limit: int) -> str:
    text = str(value or "").strip()
    if not text or len(text) > limit:
        raise ValueError(f"{field} must contain 1..{limit} characters")
    return text


def _validate_packet_cutoff(value: Any, *, umo: str, cutoff_at: int) -> dict[str, Any]:
    timestamps: list[int] = []
    foreign_scopes: set[str] = set()
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            source_key = str(current.get("source_key") or "")
            if source_key and current.get("sent_at") not in (None, ""):
                sent_at = int(current["sent_at"])
                timestamps.append(sent_at)
                if sent_at >= cutoff_at:
                    raise ValueError(
                        f"packet source {source_key!r} is not strictly before cutoff"
                    )
            scoped_umo = str(current.get("umo") or "")
            if scoped_umo and scoped_umo != umo:
                foreign_scopes.add(scoped_umo)
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    if foreign_scopes:
        raise ValueError(f"packet crosses group scopes: {sorted(foreign_scopes)}")
    return {
        "timestamped_evidence_items": len(timestamps),
        "strictly_before_cutoff": True,
        "maximum_sent_at": max(timestamps, default=None),
    }


def _gold_source_keys(gold: dict[str, Any]) -> set[str]:
    groups = gold.get("evidence_groups") or {}
    if not isinstance(groups, dict):
        raise ValueError("gold.evidence_groups must be an object")
    return {
        str(item)
        for raw in groups.values()
        if isinstance(raw, dict)
        for field in ("required_any", "support")
        for item in raw.get(field, [])
        if str(item)
    }


def load_case_bundle(case_dir: str | Path) -> CaseBundle:
    root = Path(case_dir).resolve()
    case_path = root / "case.json"
    packet_path = root / "evidence_packet.json"
    gold_path = root / "gold.json"
    paths = (case_path, packet_path, gold_path)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing packet diagnostic inputs: " + ", ".join(missing)
        )
    if len({path.resolve() for path in paths}) != 3:
        raise ValueError(
            "case, evidence packet and gold must be physically separate files"
        )

    case = _load_object(case_path)
    packet = _load_object(packet_path)
    gold = _load_object(gold_path)
    if _CASE_GOLD_KEYS & {str(key).casefold() for key in case}:
        raise ValueError("case.json contains evaluation-only gold fields")
    if str(case.get("schema_version") or CASE_SCHEMA_VERSION) != CASE_SCHEMA_VERSION:
        raise ValueError("unsupported packet case schema_version")
    if str(gold.get("schema_version") or GOLD_SCHEMA_VERSION) != GOLD_SCHEMA_VERSION:
        raise ValueError("unsupported packet gold schema_version")
    case_id = _bounded_text(case.get("case_id"), field="case.case_id", limit=120)
    if str(gold.get("case_id") or case_id) != case_id:
        raise ValueError("case and gold case_id differ")
    _bounded_text(case.get("query"), field="case.query", limit=4000)
    umo = _bounded_text(case.get("umo"), field="case.umo", limit=500)
    cutoff_at = int(case.get("cutoff_at") or 0)
    if cutoff_at <= 0:
        raise ValueError("case.cutoff_at must be positive")
    layer = str(case.get("layer") or DIAGNOSTIC_LAYER)
    if layer != DIAGNOSTIC_LAYER:
        raise ValueError(f"case.layer must be {DIAGNOSTIC_LAYER!r}")
    packet_audit = _validate_packet_cutoff(packet, umo=umo, cutoff_at=cutoff_at)
    delivered, _, _ = reconstruction_packet_allowlist(packet)
    gold_sources = _gold_source_keys(gold)
    missing_gold = sorted(gold_sources - delivered)
    if missing_gold:
        raise ValueError(
            "oracle packet omits gold evidence source keys: " + ", ".join(missing_gold)
        )
    hashes = {
        "case_sha256": _file_sha256(case_path),
        "evidence_packet_sha256": _file_sha256(packet_path),
        "gold_sha256": _file_sha256(gold_path),
        "delivered_source_allowlist_sha256": _stable_json_hash(sorted(delivered)),
        "packet_cutoff_audit_sha256": _stable_json_hash(packet_audit),
    }
    return CaseBundle(
        root, case_path, packet_path, gold_path, case, packet, gold, hashes
    )


def _authorized_participants(case: dict[str, Any], packet: dict[str, Any]) -> set[str]:
    explicit = {
        str(item) for item in case.get("authorized_participant_keys", []) if str(item)
    }
    return explicit | _eccr_participant_allowlist(packet)


def _host_subject_bindings(case: dict[str, Any]) -> list[dict[str, Any]]:
    value = case.get("host_subject_bindings") or []
    if not isinstance(value, list):
        raise ValueError("case.host_subject_bindings must be an array")
    return [dict(item) for item in value if isinstance(item, dict)]


def _packet_source_participants(packet: dict[str, Any]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    stack: list[Any] = [packet]
    participant_fields = (
        "participant_key",
        "sender_participant_key",
        "subject_participant_key",
        "sender_id",
        "account_id",
    )
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            sources = {
                str(current.get("source_key") or ""),
                *(str(item) for item in current.get("source_keys", []) if str(item)),
            } - {""}
            participants = {
                str(current.get(field) or "")
                for field in participant_fields
                if str(current.get(field) or "")
            }
            for source in sources:
                result.setdefault(source, set()).update(participants)
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    return result


def _validate_subject_bindings(
    *,
    case: dict[str, Any],
    packet: dict[str, Any],
    subjects: list[dict[str, Any]],
) -> None:
    cutoff_at = int(case["cutoff_at"])
    host_bindings = {
        str(item.get("reference") or "").strip().casefold(): item
        for item in _host_subject_bindings(case)
        if str(item.get("reference") or "").strip()
    }
    source_participants = _packet_source_participants(packet)
    for subject in subjects:
        reference = str(subject.get("reference") or "").strip().casefold()
        participant = str(subject.get("participant_key") or "")
        mode = str(subject.get("mode") or "").strip().upper()
        candidates = {
            str(item)
            for item in subject.get("candidate_participant_keys", [])
            if str(item)
        }
        source_keys = {
            str(item) for item in subject.get("source_keys", []) if str(item)
        }
        valid_at = subject.get("valid_at")
        if valid_at not in (None, "") and int(valid_at) >= cutoff_at:
            raise ValueError("subject binding valid_at is not strictly before cutoff")

        host = host_bindings.get(reference)
        if host is not None:
            expected_participant = str(host.get("participant_key") or "")
            expected_candidates = {
                str(item)
                for item in host.get("candidate_participant_keys", [])
                if str(item)
            }
            expected_mode = str(host.get("mode") or "").strip().upper()
            if (
                participant != expected_participant
                or candidates != expected_candidates
                or mode != expected_mode
            ):
                raise ValueError("model subject binding contradicts host binding")
            continue

        if mode in {"HOST", "STRUCTURED_REF", "UNIQUE_ALIAS"}:
            attributable = [
                source_participants[source]
                for source in source_keys
                if source_participants.get(source)
            ]
            if not attributable or any(
                participant not in participants for participants in attributable
            ):
                raise ValueError(
                    "resolved subject binding is not attributable to its cited sources"
                )


def _normalize_eccr_visited_citations(value: str) -> str:
    raw = json.loads(value)
    if not isinstance(raw, dict) or not isinstance(raw.get("contract"), dict):
        return value
    contract = raw["contract"]
    cited = {
        str(item)
        for field in ("subjects", "obligations", "interpretations", "uncertainties")
        for row in contract.get(field, [])
        if isinstance(row, dict)
        for source_field in ("source_keys", "support_keys", "counter_keys")
        for item in row.get(source_field, [])
        if str(item)
    }
    brief = raw.get("memory_brief")
    if isinstance(brief, dict):
        cited.update(
            str(item)
            for field in ("claims", "conflicts", "unresolved")
            for row in brief.get(field, [])
            if isinstance(row, dict)
            for item in row.get("source_keys", [])
            if str(item)
        )
    visited = {
        str(item) for item in contract.get("visited_source_keys", []) if str(item)
    }
    contract["visited_source_keys"] = sorted(visited | cited)
    return canonical_json(raw)


def build_one_pass_messages(
    case: dict[str, Any], packet: dict[str, Any]
) -> list[dict[str, str]]:
    payload = {
        "diagnostic_layer": DIAGNOSTIC_LAYER,
        "end_to_end_retrieval": False,
        "task": "synthesize one grounded memory brief from the fixed evidence packet",
        "query": str(case["query"]),
        "scope": {"umo": str(case["umo"]), "cutoff_at": int(case["cutoff_at"])},
        "authorized_participant_keys": sorted(_authorized_participants(case, packet)),
        "host_subject_bindings": _host_subject_bindings(case),
        "evidence_packet": packet,
        "retrieval": {"available": False, "reason": "oracle packet diagnostic"},
    }
    prompt = canonical_json(payload)
    if len(prompt) > 100_000:
        raise ValueError("one-pass packet prompt exceeds 100000 characters")
    return [
        {"role": "system", "content": ONE_PASS_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def _eccr_runtime(
    case: dict[str, Any], packet: dict[str, Any]
) -> tuple[dict[str, Any], set[str], set[str], set[int], set[int]]:
    delivered, hypothesis_ids, edge_ids = reconstruction_packet_allowlist(packet)
    participants = _authorized_participants(case, packet)
    call = {
        "query": str(case["query"]),
        "umo": str(case["umo"]),
        "cutoff_at": int(case["cutoff_at"]),
    }
    host_fields = _eccr_host_fields(
        call=call,
        packet=packet,
        participants=participants,
        edge_ids=set(edge_ids),
        hypothesis_ids=set(hypothesis_ids),
    )
    return host_fields, set(delivered), participants, set(edge_ids), set(hypothesis_ids)


def build_eccr_messages(
    case: dict[str, Any], packet: dict[str, Any]
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    host_fields, _, participants, edge_ids, hypothesis_ids = _eccr_runtime(case, packet)
    payload = {
        "diagnostic_layer": DIAGNOSTIC_LAYER,
        "end_to_end_retrieval": False,
        "task": (
            "compile and close one evidence contract from the complete fixed packet; "
            "return terminal=true and actions=[] because retrieval is unavailable"
        ),
        "target_query": str(case["query"]),
        "host_contract_fields": host_fields,
        "authorized_participant_keys": sorted(participants),
        "host_subject_bindings": _host_subject_bindings(case),
        "authorized_edge_ids": sorted(edge_ids),
        "authorized_hypothesis_ids": sorted(hypothesis_ids),
        "allowed_retrieval_tools": [],
        "initial_memory_packet": packet,
        "retrieval_budget": {
            "max_actions": 0,
            "max_retrieval_rounds": 0,
            "max_total_model_calls": 1,
        },
        "required_memory_brief_schema": {
            "claims": [
                {
                    "statement": "grounded claim",
                    "source_keys": ["exact delivered source_key"],
                    "confidence": 0.0,
                }
            ],
            "conflicts": [
                {
                    "statement": "grounded conflict",
                    "source_keys": ["exact delivered source_key"],
                }
            ],
            "unresolved": [
                {
                    "statement": "uncertainty that must remain visible",
                    "source_keys": ["exact delivered source_key"],
                }
            ],
        },
    }
    prompt = canonical_json(payload)
    if len(prompt) > 100_000:
        raise ValueError("ECCR packet prompt exceeds 100000 characters")
    return (
        [
            {"role": "system", "content": ECCR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        host_fields,
    )


def _eccr_audit_runtime(
    case: dict[str, Any], packet: dict[str, Any]
) -> tuple[dict[str, Any], set[str], set[str], set[int], set[int]]:
    host_fields, delivered, participants, edge_ids, hypothesis_ids = _eccr_runtime(
        case, packet
    )
    host_fields = dict(host_fields)
    revision_vector = dict(host_fields["revision_vector"])
    revision_vector["protocol"] = hashlib.sha256(
        ECCR_AUDIT_SYSTEM_PROMPT.encode("utf-8")
    ).hexdigest()
    host_fields["revision_vector"] = revision_vector
    return host_fields, delivered, participants, edge_ids, hypothesis_ids


def _audit_protocol_state(value: str, *, phase: str) -> str:
    """Attach the host-observable two-phase audit frontier to one model turn."""

    raw = json.loads(value)
    if not isinstance(raw, dict) or not isinstance(raw.get("contract"), dict):
        return value
    contract = raw["contract"]
    frontier = [
        str(item)
        for item in contract.get("frontier_discriminators", [])
        if str(item) and str(item) != _PACKET_AUDIT_DISCRIMINATOR
    ]
    exhausted = [
        str(item)
        for item in contract.get("exhausted_discriminators", [])
        if str(item) and str(item) != _PACKET_AUDIT_DISCRIMINATOR
    ]
    if phase == "compile":
        frontier.append(_PACKET_AUDIT_DISCRIMINATOR)
    elif phase == "coverage-audit":
        exhausted.append(_PACKET_AUDIT_DISCRIMINATOR)
    else:
        raise ValueError(f"unsupported ECCR packet audit phase: {phase}")
    contract["frontier_discriminators"] = list(dict.fromkeys(frontier))
    contract["exhausted_discriminators"] = list(dict.fromkeys(exhausted))
    return canonical_json(raw)


def _restore_previous_contract_evidence(
    value: str,
    *,
    previous: ReconstructionContract,
) -> tuple[str, list[dict[str, Any]]]:
    """Restore only evidence keys that a later model turn accidentally omitted.

    The previous contract has already passed the host allowlists and schema
    validation.  This normalization is therefore deliberately narrower than a
    general repair pass: it does not create entities, remove current values, or
    alter definitions, identities, states, conclusions, or transition steps.
    The strict contract parser still sees (and rejects) every other mutation.
    """

    raw = json.loads(value)
    if not isinstance(raw, dict) or not isinstance(raw.get("contract"), dict):
        return value, []
    contract = raw["contract"]
    audit: list[dict[str, Any]] = []

    def restore_collection(
        *,
        collection: str,
        identity_field: str,
        previous_rows: list[dict[str, Any]],
        evidence_fields: tuple[str, ...],
        casefold_identity: bool = False,
    ) -> None:
        current_rows = contract.get(collection)
        if not isinstance(current_rows, list):
            return

        def identity(row: dict[str, Any]) -> str:
            key = str(row.get(identity_field) or "")
            return key.casefold() if casefold_identity else key

        previous_by_id = {
            identity(row): row
            for row in previous_rows
            if isinstance(row, dict) and identity(row)
        }
        current_ids = [
            identity(row)
            for row in current_rows
            if isinstance(row, dict) and identity(row)
        ]
        # Do not touch an ambiguous/duplicated entity set; the strict parser
        # must reject it without host assistance.
        if len(current_ids) != len(set(current_ids)):
            return

        for row in current_rows:
            if not isinstance(row, dict):
                continue
            entity_id = identity(row)
            old = previous_by_id.get(entity_id)
            if old is None:
                continue
            for field in evidence_fields:
                old_sources = old.get(field)
                current_sources = row.get(field)
                if not isinstance(old_sources, list):
                    continue
                if current_sources is None:
                    current_sources = []
                if not isinstance(current_sources, list):
                    # A malformed scalar is not an omission and must fail the
                    # normal schema validation.
                    continue
                present = {str(item) for item in current_sources if str(item)}
                restored = [
                    str(item)
                    for item in old_sources
                    if str(item) and str(item) not in present
                ]
                if not restored:
                    continue
                row[field] = [*current_sources, *restored]
                audit.append(
                    {
                        "normalization": "monotonic_previous_evidence_union",
                        "entity_type": {
                            "subjects": "subject",
                            "obligations": "obligation",
                            "interpretations": "interpretation",
                            "uncertainties": "uncertainty",
                        }[collection],
                        "entity_id": entity_id,
                        "field": field,
                        "restored_source_keys": restored,
                    }
                )

    previous_dict = previous.as_dict()
    restore_collection(
        collection="subjects",
        identity_field="reference",
        previous_rows=list(previous_dict["subjects"]),
        evidence_fields=("source_keys",),
        casefold_identity=True,
    )
    restore_collection(
        collection="obligations",
        identity_field="id",
        previous_rows=list(previous_dict["obligations"]),
        evidence_fields=("support_keys", "counter_keys"),
    )
    restore_collection(
        collection="interpretations",
        identity_field="id",
        previous_rows=list(previous_dict["interpretations"]),
        evidence_fields=("support_keys", "counter_keys"),
    )
    restore_collection(
        collection="uncertainties",
        identity_field="id",
        previous_rows=list(previous_dict["uncertainties"]),
        evidence_fields=("source_keys",),
    )
    return canonical_json(raw), audit


def _contract_selected_source_keys(contract: ReconstructionContract) -> set[str]:
    selected: set[str] = set()
    for subject in contract.subjects:
        selected.update(subject.source_keys)
    for obligation in contract.obligations:
        selected.update(obligation.support_keys)
        selected.update(obligation.counter_keys)
    for interpretation in contract.interpretations:
        selected.update(interpretation.support_keys)
        selected.update(interpretation.counter_keys)
    for uncertainty in contract.uncertainties:
        selected.update(uncertainty.source_keys)
    return selected


def _audit_annotated_packet(
    value: Any,
    *,
    selected_source_keys: set[str],
    path: str = "$",
) -> Any:
    """Preserve packet neighborhoods/snapshots and mark every direct citation."""

    if isinstance(value, dict):
        copied = {
            str(key): _audit_annotated_packet(
                item,
                selected_source_keys=selected_source_keys,
                path=f"{path}.{key}",
            )
            for key, item in value.items()
        }
        direct_sources: set[str] = set()
        for field in ("source_key", "request_source_key"):
            source = value.get(field)
            if isinstance(source, str) and source:
                direct_sources.add(source)
        for field in ("source_keys", "sample_source_keys"):
            sources = value.get(field)
            if isinstance(sources, list):
                direct_sources.update(str(item) for item in sources if str(item))
        if direct_sources:
            copied["_eccr_audit"] = {
                "packet_path": path,
                "source_selection": [
                    {
                        "source_key": source,
                        "first_round_selected": source in selected_source_keys,
                    }
                    for source in sorted(direct_sources)
                ],
            }
        return copied
    if isinstance(value, list):
        return [
            _audit_annotated_packet(
                item,
                selected_source_keys=selected_source_keys,
                path=f"{path}[{index}]",
            )
            for index, item in enumerate(value)
        ]
    return value


def build_eccr_audit_compile_messages(
    case: dict[str, Any], packet: dict[str, Any]
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    host_fields, _, participants, edge_ids, hypothesis_ids = _eccr_audit_runtime(
        case, packet
    )
    payload = {
        "diagnostic_layer": DIAGNOSTIC_LAYER,
        "end_to_end_retrieval": False,
        "audit_phase": "compile",
        "task": (
            "Call 1 of exactly 2. Compile the evidence questions before deciding. "
            "Return terminal=false, actions=[], and memory_brief=null. Every "
            "critical obligation must remain OPEN; interpretations must remain "
            "CANDIDATE; uncertainties must remain OPEN. Include at least two "
            "competing interpretations, one uncertainty constraint, and one "
            "guarded claim. Cite only the evidence selected for this first pass."
        ),
        "target_query": str(case["query"]),
        "host_contract_fields": host_fields,
        "authorized_participant_keys": sorted(participants),
        "host_subject_bindings": _host_subject_bindings(case),
        "authorized_edge_ids": sorted(edge_ids),
        "authorized_hypothesis_ids": sorted(hypothesis_ids),
        "allowed_retrieval_tools": [],
        "initial_memory_packet": packet,
        "audit_protocol": {
            "total_provider_calls": 2,
            "this_call": 1,
            "retrieval_available": False,
            "host_frontier_discriminator": _PACKET_AUDIT_DISCRIMINATOR,
        },
    }
    prompt = canonical_json(payload)
    if len(prompt) > 100_000:
        raise ValueError("ECCR packet audit compile prompt exceeds 100000 characters")
    return (
        [
            {"role": "system", "content": ECCR_AUDIT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        host_fields,
    )


def build_eccr_audit_review_messages(
    case: dict[str, Any],
    packet: dict[str, Any],
    *,
    previous_turn: ContractTurn,
) -> list[dict[str, str]]:
    delivered = set(reconstruction_packet_allowlist(packet)[0])
    selected = _contract_selected_source_keys(previous_turn.contract)
    if not selected.issubset(delivered):
        raise ValueError("first audit turn selected evidence outside the packet")
    unselected = delivered - selected
    payload = {
        "diagnostic_layer": DIAGNOSTIC_LAYER,
        "end_to_end_retrieval": False,
        "audit_phase": "counterexample_coverage_audit",
        "task": (
            "Call 2 of exactly 2. Audit counterexamples and coverage over every "
            "UNSELECTED source in the original packet structure, revise only "
            "evidence/status fields permitted by the strict transition, preserve "
            "all obligation, interpretation, uncertainty, and guarded-claim "
            "identities, then return terminal=true and actions=[]. The memory "
            "brief must expose every preserved uncertainty or contested claim."
        ),
        "target_query": str(case["query"]),
        "scope": {"umo": str(case["umo"]), "cutoff_at": int(case["cutoff_at"])},
        "previous_turn": previous_turn.as_dict(),
        "first_round_selected_evidence": {
            "source_keys": sorted(selected),
            "selection_rule": (
                "source cited by a subject, obligation, interpretation, or "
                "uncertainty in call 1"
            ),
        },
        "unselected_evidence": {
            "source_keys": sorted(unselected),
            "count": len(unselected),
        },
        "audit_marked_packet_preserving_original_neighborhoods_and_snapshots": (
            _audit_annotated_packet(packet, selected_source_keys=selected)
        ),
        "audit_protocol": {
            "total_provider_calls": 2,
            "this_call": 2,
            "retrieval_available": False,
            "host_exhausted_discriminator": _PACKET_AUDIT_DISCRIMINATOR,
        },
    }
    prompt = canonical_json(payload)
    if len(prompt) > 140_000:
        raise ValueError("ECCR packet audit review prompt exceeds 140000 characters")
    return [
        {"role": "system", "content": ECCR_AUDIT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def _response_audit(message: Any) -> dict[str, Any]:
    completion_text = str(getattr(message, "content", "") or "")
    reasoning = str(getattr(message, "reasoning_content", "") or "")
    return {
        "completion_text": completion_text,
        "completion_text_sha256": hashlib.sha256(
            completion_text.encode("utf-8")
        ).hexdigest(),
        "reasoning_content_present": bool(reasoning),
        "reasoning_content_chars": len(reasoning),
        "reasoning_content_sha256": (
            hashlib.sha256(reasoning.encode("utf-8")).hexdigest() if reasoning else None
        ),
    }


def _private_completion_record(
    message: Any,
    *,
    call_index: int,
    phase: str,
) -> dict[str, Any]:
    """Persist the provider-visible completion without hidden reasoning text."""

    content = str(getattr(message, "content", "") or "")
    return {
        "call_index": call_index,
        "phase": phase,
        "provider_visible_completion_content": content,
        "provider_visible_completion_sha256": hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest(),
    }


def _write_audit_checkpoint(
    path: Path | None,
    state: dict[str, Any],
    *,
    status: str,
    error: Exception | None = None,
) -> None:
    if path is None:
        return
    state["status"] = status
    state["updated_at"] = _utc_now()
    if error is None:
        state.pop("error", None)
    else:
        state["error"] = {
            "type": type(error).__name__,
            "detail": str(error)[:2000],
        }
    _atomic_write_json(path, state)


def _run_one_pass(
    *,
    case: dict[str, Any],
    packet: dict[str, Any],
    client: Any,
    provider_id: str,
    model: str,
    provider_extra_body: dict[str, Any],
    max_output_tokens: int,
    thinking_mode: str,
    deadline_seconds: float,
    ledger_path: Path,
    budget: PilotBudget,
    run_id: str,
    repetition: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    messages = build_one_pass_messages(case, packet)
    completion = _pilot_completion(
        client=client,
        model=model,
        provider_id=provider_id,
        messages=messages,
        provider_extra_body=provider_extra_body,
        tools=None,
        max_output_tokens=max_output_tokens,
        thinking_mode=thinking_mode,
        json_object=True,
        ledger_path=ledger_path,
        budget=budget,
        run_id=run_id,
        arm="one-pass",
        repetition=repetition,
        phase="oracle_packet_synthesis",
        call_index=0,
        request_timeout_seconds=deadline_seconds,
    )
    message = completion.choices[0].message
    delivered, hypothesis_ids, edge_ids = reconstruction_packet_allowlist(packet)
    plan, response_source = parse_structured_response(
        completion_text=getattr(message, "content", ""),
        reasoning_content=getattr(message, "reasoning_content", ""),
        parser=lambda value: parse_reconstruction_plan(
            value,
            allowed_source_keys=delivered,
            allowed_hypothesis_ids=hypothesis_ids,
            allowed_edge_ids=edge_ids,
        ),
    )
    return {
        "decision": plan.decision,
        "brief": plan.brief.as_dict() if plan.brief is not None else None,
        "visited_source_keys": sorted(delivered),
        "selected_edge_ids": [item[0] for item in plan.edge_activations],
        "selected_hypothesis_ids": [item[0] for item in plan.hypothesis_activations],
        "escalation_question": plan.escalation_question,
        "contract": None,
        "subjects": [],
        "model_calls": 1,
        "retrieval_rounds": 0,
        "retrieval_available": False,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "response_source": response_source,
        "model_input_sha256": _stable_json_hash(messages),
        "raw_response": _response_audit(message),
    }


def _run_eccr(
    *,
    case: dict[str, Any],
    packet: dict[str, Any],
    client: Any,
    provider_id: str,
    model: str,
    provider_extra_body: dict[str, Any],
    max_output_tokens: int,
    thinking_mode: str,
    deadline_seconds: float,
    ledger_path: Path,
    budget: PilotBudget,
    run_id: str,
    repetition: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    messages, host_fields = build_eccr_messages(case, packet)
    completion = _pilot_completion(
        client=client,
        model=model,
        provider_id=provider_id,
        messages=messages,
        provider_extra_body=provider_extra_body,
        tools=None,
        max_output_tokens=max_output_tokens,
        thinking_mode=thinking_mode,
        json_object=True,
        ledger_path=ledger_path,
        budget=budget,
        run_id=run_id,
        arm="eccr",
        repetition=repetition,
        phase="oracle_packet_contract",
        call_index=0,
        request_timeout_seconds=deadline_seconds,
    )
    message = completion.choices[0].message
    _, allowed_sources, participants, edge_ids, hypothesis_ids = _eccr_runtime(
        case, packet
    )
    turn, response_source = parse_structured_response(
        completion_text=getattr(message, "content", ""),
        reasoning_content=getattr(message, "reasoning_content", ""),
        parser=lambda value: _parse_eccr_turn(
            _normalize_eccr_visited_citations(value),
            host_fields=host_fields,
            allowed_source_keys=allowed_sources,
            allowed_participant_keys=participants,
            allowed_edge_ids=edge_ids,
            allowed_hypothesis_ids=hypothesis_ids,
            allowed_tool_names=set(),
            previous=None,
            tried_signatures=set(),
        ),
    )
    if not turn.terminal:
        raise ValueError("packet-level ECCR must close in its only model call")
    if turn.actions:
        raise ValueError("packet-level ECCR cannot request retrieval actions")
    qualified_critical = any(
        item.critical and item.status in {"AMBIGUOUS", "CONTESTED", "EXHAUSTED"}
        for item in turn.contract.obligations
    )
    if allowed_sources and qualified_critical and turn.brief is None:
        raise ValueError(
            "qualified terminal ECCR closure must emit an explicit unresolved brief"
        )
    subjects = [item.as_dict() for item in turn.contract.subjects]
    _validate_subject_bindings(case=case, packet=packet, subjects=subjects)
    return {
        "decision": "brief" if turn.brief is not None else "none",
        "brief": turn.brief.as_dict() if turn.brief is not None else None,
        "visited_source_keys": list(turn.contract.visited_source_keys),
        "selected_edge_ids": list(turn.contract.selected_edge_ids),
        "selected_hypothesis_ids": list(turn.contract.selected_hypothesis_ids),
        "contract": turn.contract.as_dict(),
        "subjects": subjects,
        "model_calls": 1,
        "retrieval_rounds": 0,
        "retrieval_available": False,
        "terminal": True,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "response_source": response_source,
        "model_input_sha256": _stable_json_hash(messages),
        "raw_response": _response_audit(message),
    }


def _run_eccr_audit(
    *,
    case: dict[str, Any],
    packet: dict[str, Any],
    client: Any,
    provider_id: str,
    model: str,
    provider_extra_body: dict[str, Any],
    max_output_tokens: int,
    thinking_mode: str,
    deadline_seconds: float,
    ledger_path: Path,
    budget: PilotBudget,
    run_id: str,
    repetition: int,
    private_checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    checkpoint_state: dict[str, Any] = {
        "schema_version": "eccr.packet.audit.checkpoint.private.v1",
        "private": True,
        "run_id": run_id,
        "created_at": _utc_now(),
        "rounds": [],
    }
    compile_messages, host_fields = build_eccr_audit_compile_messages(case, packet)
    _, allowed_sources, participants, edge_ids, hypothesis_ids = _eccr_audit_runtime(
        case, packet
    )
    compile_completion = _pilot_completion(
        client=client,
        model=model,
        provider_id=provider_id,
        messages=compile_messages,
        provider_extra_body=provider_extra_body,
        tools=None,
        max_output_tokens=max_output_tokens,
        thinking_mode=thinking_mode,
        json_object=True,
        ledger_path=ledger_path,
        budget=budget,
        run_id=run_id,
        arm="eccr-audit",
        repetition=repetition,
        phase="oracle_packet_audit_compile",
        call_index=0,
        request_timeout_seconds=deadline_seconds,
    )
    compile_message = compile_completion.choices[0].message
    compile_checkpoint = _private_completion_record(
        compile_message,
        call_index=0,
        phase="compile",
    )
    checkpoint_state["rounds"].append(compile_checkpoint)
    _write_audit_checkpoint(
        private_checkpoint_path,
        checkpoint_state,
        status="COMPILE_COMPLETION_RECEIVED",
    )
    try:
        compile_turn, compile_response_source = parse_structured_response(
            completion_text=getattr(compile_message, "content", ""),
            reasoning_content=getattr(compile_message, "reasoning_content", ""),
            parser=lambda value: _parse_eccr_turn(
                _normalize_eccr_visited_citations(
                    _audit_protocol_state(value, phase="compile")
                ),
                host_fields=host_fields,
                allowed_source_keys=allowed_sources,
                allowed_participant_keys=participants,
                allowed_edge_ids=edge_ids,
                allowed_hypothesis_ids=hypothesis_ids,
                allowed_tool_names=set(),
                previous=None,
                tried_signatures=set(),
            ),
        )
        if compile_turn.terminal:
            raise ValueError("ECCR audit call 1 must be nonterminal")
        if compile_turn.actions:
            raise ValueError("ECCR audit call 1 cannot request retrieval actions")
        if compile_turn.brief is not None:
            raise ValueError("ECCR audit call 1 cannot emit a memory brief")
        if any(item.status != "OPEN" for item in compile_turn.contract.obligations):
            raise ValueError("ECCR audit call 1 must leave every obligation OPEN")
        if len(compile_turn.contract.interpretations) < 2 or any(
            item.status != "CANDIDATE" for item in compile_turn.contract.interpretations
        ):
            raise ValueError(
                "ECCR audit call 1 requires at least two CANDIDATE interpretations"
            )
        if not compile_turn.contract.uncertainties or any(
            item.status != "OPEN" for item in compile_turn.contract.uncertainties
        ):
            raise ValueError("ECCR audit call 1 requires at least one OPEN uncertainty")
        if not compile_turn.contract.guarded_claims:
            raise ValueError("ECCR audit call 1 requires at least one guarded claim")
        compile_subjects = [item.as_dict() for item in compile_turn.contract.subjects]
        _validate_subject_bindings(
            case=case,
            packet=packet,
            subjects=compile_subjects,
        )
    except Exception as exc:
        _write_audit_checkpoint(
            private_checkpoint_path,
            checkpoint_state,
            status="COMPILE_VALIDATION_FAILED",
            error=exc,
        )
        raise

    compile_checkpoint["response_source"] = compile_response_source
    compile_checkpoint["parsed_turn"] = compile_turn.as_dict()
    _write_audit_checkpoint(
        private_checkpoint_path,
        checkpoint_state,
        status="COMPILE_PARSED",
    )

    selected = _contract_selected_source_keys(compile_turn.contract)
    review_messages = build_eccr_audit_review_messages(
        case,
        packet,
        previous_turn=compile_turn,
    )
    try:
        review_completion = _pilot_completion(
            client=client,
            model=model,
            provider_id=provider_id,
            messages=review_messages,
            provider_extra_body=provider_extra_body,
            tools=None,
            max_output_tokens=max_output_tokens,
            thinking_mode=thinking_mode,
            json_object=True,
            ledger_path=ledger_path,
            budget=budget,
            run_id=run_id,
            arm="eccr-audit",
            repetition=repetition,
            phase="oracle_packet_counterexample_coverage_audit",
            call_index=1,
            request_timeout_seconds=deadline_seconds,
        )
    except Exception as exc:
        _write_audit_checkpoint(
            private_checkpoint_path,
            checkpoint_state,
            status="REVIEW_CALL_FAILED",
            error=exc,
        )
        raise
    review_message = review_completion.choices[0].message
    review_checkpoint = _private_completion_record(
        review_message,
        call_index=1,
        phase="counterexample_coverage_audit",
    )
    checkpoint_state["rounds"].append(review_checkpoint)
    _write_audit_checkpoint(
        private_checkpoint_path,
        checkpoint_state,
        status="REVIEW_COMPLETION_RECEIVED",
    )
    normalization_audit: list[dict[str, Any]] = []

    def parse_review(value: str) -> ContractTurn:
        normalized, candidate_audit = _restore_previous_contract_evidence(
            _audit_protocol_state(value, phase="coverage-audit"),
            previous=compile_turn.contract,
        )
        turn = _parse_eccr_turn(
            _normalize_eccr_visited_citations(normalized),
            host_fields=host_fields,
            allowed_source_keys=allowed_sources,
            allowed_participant_keys=participants,
            allowed_edge_ids=edge_ids,
            allowed_hypothesis_ids=hypothesis_ids,
            allowed_tool_names=set(),
            previous=compile_turn.contract,
            tried_signatures=set(),
        )
        normalization_audit[:] = candidate_audit
        return turn

    try:
        review_turn, review_response_source = parse_structured_response(
            completion_text=getattr(review_message, "content", ""),
            reasoning_content=getattr(review_message, "reasoning_content", ""),
            parser=parse_review,
        )
        if not review_turn.terminal:
            raise ValueError("ECCR audit call 2 must terminate")
        if review_turn.actions:
            raise ValueError("ECCR audit call 2 cannot request retrieval actions")
        if _PACKET_AUDIT_DISCRIMINATOR not in set(
            review_turn.contract.exhausted_discriminators
        ):
            raise ValueError("ECCR audit call 2 did not complete the coverage frontier")
        qualified_critical = any(
            item.critical and item.status in {"AMBIGUOUS", "CONTESTED", "EXHAUSTED"}
            for item in review_turn.contract.obligations
        )
        if allowed_sources and qualified_critical and review_turn.brief is None:
            raise ValueError(
                "qualified terminal ECCR audit must emit an explicit unresolved brief"
            )
        subjects = [item.as_dict() for item in review_turn.contract.subjects]
        _validate_subject_bindings(case=case, packet=packet, subjects=subjects)
    except Exception as exc:
        _write_audit_checkpoint(
            private_checkpoint_path,
            checkpoint_state,
            status="REVIEW_VALIDATION_FAILED",
            error=exc,
        )
        raise

    review_checkpoint["response_source"] = review_response_source
    review_checkpoint["parsed_turn"] = review_turn.as_dict()
    review_checkpoint["normalization_audit"] = normalization_audit
    _write_audit_checkpoint(
        private_checkpoint_path,
        checkpoint_state,
        status="COMPLETED",
    )

    compile_hash = _stable_json_hash(compile_messages)
    review_hash = _stable_json_hash(review_messages)
    return {
        "decision": "brief" if review_turn.brief is not None else "none",
        "brief": (
            review_turn.brief.as_dict() if review_turn.brief is not None else None
        ),
        "visited_source_keys": list(review_turn.contract.visited_source_keys),
        "selected_edge_ids": list(review_turn.contract.selected_edge_ids),
        "selected_hypothesis_ids": list(review_turn.contract.selected_hypothesis_ids),
        "contract": review_turn.contract.as_dict(),
        "subjects": subjects,
        "model_calls": 2,
        "retrieval_rounds": 0,
        "retrieval_available": False,
        "terminal": True,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "response_source": review_response_source,
        "response_sources_by_round": [
            compile_response_source,
            review_response_source,
        ],
        "model_input_sha256": _stable_json_hash([compile_hash, review_hash]),
        "model_input_sha256_by_round": [compile_hash, review_hash],
        "first_round_selected_source_keys": sorted(selected),
        "first_round_unselected_source_keys": sorted(allowed_sources - selected),
        "normalization_audit": normalization_audit,
        "audit_protocol": {
            "provider_calls_required": 2,
            "provider_calls_completed": 2,
            "retrieval_available": False,
            "coverage_discriminator": _PACKET_AUDIT_DISCRIMINATOR,
        },
        "rounds": [
            {
                "call_index": 0,
                "phase": "compile",
                "terminal": False,
                "contract": compile_turn.contract.as_dict(),
                "brief": None,
                "model_input_sha256": compile_hash,
                "response_source": compile_response_source,
                "raw_response": _response_audit(compile_message),
            },
            {
                "call_index": 1,
                "phase": "counterexample_coverage_audit",
                "terminal": True,
                "contract": review_turn.contract.as_dict(),
                "brief": (
                    review_turn.brief.as_dict()
                    if review_turn.brief is not None
                    else None
                ),
                "model_input_sha256": review_hash,
                "response_source": review_response_source,
                "normalization_audit": normalization_audit,
                "raw_response": _response_audit(review_message),
            },
        ],
        "raw_response": _response_audit(review_message),
    }


def _message_scope(message: dict[str, Any]) -> str:
    return str(message.get("umo") or message.get("scope_token") or "")


def _message_participant(message: dict[str, Any]) -> str:
    return str(
        message.get("sender_participant_key")
        or message.get("participant_key")
        or message.get("sender_id")
        or message.get("account_id")
        or ""
    )


def _message_display_name(message: dict[str, Any]) -> str:
    return str(message.get("display_name") or message.get("sender_name") or "").strip()


def _deterministic_identity_snapshots(
    *,
    case: dict[str, Any],
    packet: dict[str, Any],
    bindings: list[dict[str, Any]],
    allowed_sources: set[str],
) -> list[dict[str, Any]]:
    raw_snapshots = packet.get("snapshots") or []
    if not raw_snapshots:
        return []
    if not isinstance(raw_snapshots, list):
        raise ValueError("packet.snapshots must be an array")

    results: list[dict[str, Any]] = []
    for index, raw_snapshot in enumerate(raw_snapshots):
        if not isinstance(raw_snapshot, dict):
            raise ValueError(f"packet.snapshots[{index}] must be an object")
        cutoff_at = int(raw_snapshot.get("cutoff_at") or 0)
        if cutoff_at <= 0:
            raise ValueError(f"packet.snapshots[{index}].cutoff_at must be positive")
        scope = _bounded_text(
            raw_snapshot.get("umo")
            or raw_snapshot.get("query_scope_token")
            or case.get("umo"),
            field=f"packet.snapshots[{index}].scope",
            limit=500,
        )
        raw_messages = raw_snapshot.get("messages") or []
        if not isinstance(raw_messages, list):
            raise ValueError(f"packet.snapshots[{index}].messages must be an array")

        in_scope: dict[str, dict[str, Any]] = {}
        cross_scope_participants: set[str] = set()
        for message_index, raw_message in enumerate(raw_messages):
            if not isinstance(raw_message, dict):
                raise ValueError(
                    f"packet.snapshots[{index}].messages[{message_index}] "
                    "must be an object"
                )
            sent_at = int(raw_message.get("sent_at") or 0)
            if sent_at <= 0 or sent_at >= cutoff_at:
                continue
            participant = _message_participant(raw_message)
            message_scope = _message_scope(raw_message)
            if message_scope != scope:
                if participant:
                    cross_scope_participants.add(participant)
                continue
            source_key = str(raw_message.get("source_key") or "")
            if not source_key or source_key not in allowed_sources:
                raise ValueError("snapshot message lacks a delivered source_key")
            existing = in_scope.get(source_key)
            if existing is not None and existing != raw_message:
                raise ValueError(
                    "snapshot reuses one source_key for different messages"
                )
            in_scope[source_key] = raw_message

        filtered_bindings: list[dict[str, Any]] = []
        selected_participants: set[str] = set()
        for binding in bindings:
            participant = str(binding.get("participant_key") or "")
            filtered_sources = sorted(
                source
                for source in binding.get("source_keys", [])
                if source in in_scope
                and (
                    not participant
                    or _message_participant(in_scope[source]) == participant
                )
            )
            if not filtered_sources:
                continue
            filtered = dict(binding)
            filtered["source_keys"] = filtered_sources
            filtered["valid_at"] = max(
                int(in_scope[source].get("sent_at") or 0) for source in filtered_sources
            )
            filtered_bindings.append(filtered)
            if participant:
                selected_participants.add(participant)

        selected_messages = [
            message
            for message in in_scope.values()
            if _message_participant(message) in selected_participants
        ]
        alias_participants: dict[str, set[str]] = {}
        alias_first_seen: dict[str, int] = {}
        for message in selected_messages:
            alias = _message_display_name(message)
            participant = _message_participant(message)
            if not alias or not participant:
                continue
            alias_participants.setdefault(alias, set()).add(participant)
            alias_first_seen[alias] = min(
                alias_first_seen.get(alias, int(message["sent_at"])),
                int(message["sent_at"]),
            )
        ordered_aliases = sorted(
            alias_participants,
            key=lambda alias: (alias_first_seen[alias], alias),
        )
        named_messages = [
            message for message in selected_messages if _message_display_name(message)
        ]
        latest = (
            max(
                named_messages,
                key=lambda message: (
                    int(message.get("sent_at") or 0),
                    str(message.get("source_key") or ""),
                ),
            )
            if named_messages
            else None
        )
        results.append(
            {
                "cutoff_id": _bounded_text(
                    raw_snapshot.get("cutoff_id") or f"snapshot-{index + 1}",
                    field=f"packet.snapshots[{index}].cutoff_id",
                    limit=120,
                ),
                "cutoff_at": cutoff_at,
                "scope_token": scope,
                "filtered_host_bindings": filtered_bindings,
                "selected_participant_keys": sorted(selected_participants),
                "selected_source_keys": sorted(
                    str(message["source_key"]) for message in selected_messages
                ),
                "visible_aliases": ordered_aliases,
                "visible_alias_bindings": [
                    {
                        "alias": alias,
                        "participant_keys": sorted(alias_participants[alias]),
                    }
                    for alias in ordered_aliases
                ],
                "latest_display_name": (
                    _message_display_name(latest) if latest is not None else None
                ),
                "latest_display_name_sent_at": (
                    int(latest["sent_at"]) if latest is not None else None
                ),
                "cross_scope_available_participant_keys": sorted(
                    cross_scope_participants
                ),
                # selected_messages is constructed only from the requested
                # scope.  Keep the source-level audit explicit instead of
                # assuming that an account appearing in two groups was used.
                "cross_scope_selected_source_keys": [],
            }
        )
    return results


def _run_deterministic(
    *, case: dict[str, Any], packet: dict[str, Any]
) -> dict[str, Any]:
    started = time.perf_counter()
    raw_bindings = case.get("host_subject_bindings")
    if not isinstance(raw_bindings, list) or not raw_bindings:
        raise ValueError("deterministic arm requires case.host_subject_bindings")
    allowed_sources = reconstruction_packet_allowlist(packet)[0]
    participants = _authorized_participants(case, packet)
    bindings: list[dict[str, Any]] = []
    visited: set[str] = set()
    for index, raw in enumerate(raw_bindings):
        if not isinstance(raw, dict):
            raise ValueError(f"host_subject_bindings[{index}] must be an object")
        mode = str(raw.get("mode") or "").strip().upper()
        participant_key = str(raw.get("participant_key") or "")
        candidates = [str(item) for item in raw.get("candidate_participant_keys", [])]
        sources = [str(item) for item in raw.get("source_keys", [])]
        valid_at = raw.get("valid_at")
        if valid_at not in (None, "") and int(valid_at) >= int(case["cutoff_at"]):
            raise ValueError(
                "deterministic binding valid_at is not strictly before cutoff"
            )
        if participant_key and participant_key not in participants:
            raise ValueError(
                "deterministic binding selects an unauthorized participant"
            )
        if not set(candidates).issubset(participants):
            raise ValueError("deterministic binding has unauthorized candidates")
        if not set(sources).issubset(allowed_sources):
            raise ValueError("deterministic binding cites evidence outside the packet")
        if mode in {"HOST", "STRUCTURED_REF", "UNIQUE_ALIAS"} and (
            not participant_key or candidates
        ):
            raise ValueError("resolved deterministic binding requires one participant")
        if mode == "AMBIGUOUS" and (participant_key or len(candidates) < 2):
            raise ValueError("ambiguous deterministic binding requires two candidates")
        if mode == "UNBOUND" and (participant_key or candidates):
            raise ValueError("unbound deterministic binding cannot select participants")
        if mode not in {
            "HOST",
            "STRUCTURED_REF",
            "UNIQUE_ALIAS",
            "AMBIGUOUS",
            "UNBOUND",
        }:
            raise ValueError("unsupported deterministic binding mode")
        visited.update(sources)
        bindings.append(
            {
                "reference": _bounded_text(
                    raw.get("reference"),
                    field=f"host_subject_bindings[{index}].reference",
                    limit=240,
                ),
                "participant_key": participant_key,
                "mode": mode,
                "candidate_participant_keys": candidates,
                "source_keys": sources,
                "valid_at": valid_at,
            }
        )
    _validate_subject_bindings(case=case, packet=packet, subjects=bindings)
    identity_snapshots = _deterministic_identity_snapshots(
        case=case,
        packet=packet,
        bindings=bindings,
        allowed_sources=allowed_sources,
    )
    if identity_snapshots:
        visited = {
            str(source)
            for snapshot in identity_snapshots
            for source in snapshot["selected_source_keys"]
        }
    return {
        "decision": "host_identity_binding",
        "brief": None,
        "visited_source_keys": sorted(visited),
        "selected_edge_ids": [],
        "selected_hypothesis_ids": [],
        "contract": None,
        "subjects": bindings,
        "identity_snapshots": identity_snapshots,
        "model_calls": 0,
        "retrieval_rounds": 0,
        "retrieval_available": False,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "response_source": "host_deterministic_binding",
        "model_input_sha256": None,
        "raw_response": None,
    }


def _result_brief(result: dict[str, Any]) -> EvidenceBrief | None:
    raw = result.get("brief")
    if raw is None:
        return None
    return parse_evidence_brief(
        json.dumps(raw, ensure_ascii=False, separators=(",", ":")),
        allowed_source_keys=result.get("visited_source_keys", []),
    )


def _identity_score(result: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    identity = gold.get("identity") or {}
    if not isinstance(identity, dict):
        raise ValueError("gold.identity must be an object")
    expected = {
        str(item)
        for field in ("expected_participant_keys", "expected_sender_ids")
        for item in identity.get(field, [])
        if str(item)
    }
    expected_modes = {
        str(item).upper()
        for item in identity.get("expected_binding_modes", [])
        if str(item)
    }
    subjects = result.get("subjects") or []
    selected = {
        str(item.get("participant_key") or "")
        for item in subjects
        if isinstance(item, dict) and str(item.get("participant_key") or "")
    }
    candidates = {
        str(candidate)
        for item in subjects
        if isinstance(item, dict)
        for candidate in item.get("candidate_participant_keys", [])
        if str(candidate)
    }
    modes = {
        str(item.get("mode") or "").upper()
        for item in subjects
        if isinstance(item, dict) and str(item.get("mode") or "")
    }
    return {
        "expected_participant_keys": sorted(expected),
        "selected_participant_keys": sorted(selected),
        "candidate_participant_keys": sorted(candidates),
        "binding_modes": sorted(modes),
        "expected_binding_modes": sorted(expected_modes),
        "participant_exact_match": (selected == expected if expected else None),
        "binding_mode_match": (
            bool(modes & expected_modes) if expected_modes else None
        ),
    }


def _identity_snapshot_score(
    result: dict[str, Any], gold: dict[str, Any]
) -> dict[str, Any]:
    raw_gold = gold.get("snapshots") or []
    if not raw_gold:
        return {
            "applicable": False,
            "snapshot_count": 0,
            "passed_snapshots": 0,
            "snapshot_pass_rate": None,
            "all_passed": None,
            "snapshots": [],
        }
    if not isinstance(raw_gold, list):
        raise ValueError("gold.snapshots must be an array")
    raw_result = result.get("identity_snapshots") or []
    if not isinstance(raw_result, list):
        raise ValueError("result.identity_snapshots must be an array")
    by_cutoff = {
        str(item.get("cutoff_id") or ""): item
        for item in raw_result
        if isinstance(item, dict) and str(item.get("cutoff_id") or "")
    }

    snapshots: list[dict[str, Any]] = []
    for index, expected in enumerate(raw_gold):
        if not isinstance(expected, dict):
            raise ValueError(f"gold.snapshots[{index}] must be an object")
        cutoff_id = _bounded_text(
            expected.get("cutoff_id") or f"snapshot-{index + 1}",
            field=f"gold.snapshots[{index}].cutoff_id",
            limit=120,
        )
        observed = by_cutoff.get(cutoff_id) or {}
        expected_participant = str(
            expected.get("target_participant_key")
            or expected.get("target_actor_token")
            or ""
        )
        expected_scope = str(
            expected.get("target_scope_token") or expected.get("umo") or ""
        )
        expected_aliases = {
            str(item)
            for item in expected.get("visible_aliases_same_actor", [])
            if str(item)
        }
        observed_aliases = {
            str(item) for item in observed.get("visible_aliases", []) if str(item)
        }
        alias_bindings = {
            str(item.get("alias") or ""): {
                str(participant)
                for participant in item.get("participant_keys", [])
                if str(participant)
            }
            for item in observed.get("visible_alias_bindings", [])
            if isinstance(item, dict) and str(item.get("alias") or "")
        }
        same_actor = bool(expected_aliases) and observed_aliases == expected_aliases
        if expected_participant:
            same_actor = same_actor and all(
                alias_bindings.get(alias) == {expected_participant}
                for alias in expected_aliases
            )
        latest_expected = str(expected.get("expected_latest_display_name") or "")
        latest_observed = str(observed.get("latest_display_name") or "")
        latest_name = bool(latest_expected) and latest_observed == latest_expected
        forbidden_aliases = {
            str(item)
            for item in expected.get("post_cutoff_aliases_forbidden", [])
            if str(item)
        }
        observed_binding_references = {
            str(item.get("reference") or "")
            for item in observed.get("filtered_host_bindings", [])
            if isinstance(item, dict) and str(item.get("reference") or "")
        }
        post_cutoff_alias = not bool(
            forbidden_aliases & (observed_aliases | observed_binding_references)
        )
        forbidden_cross_scope = {
            str(item)
            for item in expected.get("cross_scope_actor_tokens_forbidden", [])
            if str(item)
        }
        selected_participants = {
            str(item)
            for item in observed.get("selected_participant_keys", [])
            if str(item)
        }
        cross_scope_selected = {
            str(item)
            for item in observed.get("cross_scope_selected_source_keys", [])
            if str(item)
        }
        cross_scope = (
            not bool(forbidden_cross_scope & selected_participants)
            and not cross_scope_selected
            and (
                not expected_scope
                or str(observed.get("scope_token") or "") == expected_scope
            )
        )
        passed = same_actor and latest_name and post_cutoff_alias and cross_scope
        snapshots.append(
            {
                "cutoff_id": cutoff_id,
                "same_actor_exact_match": same_actor,
                "latest_name_exact_match": latest_name,
                "post_cutoff_aliases_excluded": post_cutoff_alias,
                "cross_scope_participants_excluded": cross_scope,
                "passed": passed,
                "expected_visible_aliases": sorted(expected_aliases),
                "observed_visible_aliases": sorted(observed_aliases),
                "expected_latest_display_name": latest_expected or None,
                "observed_latest_display_name": latest_observed or None,
            }
        )
    passed_snapshots = sum(bool(item["passed"]) for item in snapshots)
    return {
        "applicable": True,
        "snapshot_count": len(snapshots),
        "passed_snapshots": passed_snapshots,
        "snapshot_pass_rate": round(passed_snapshots / len(snapshots), 3),
        "all_passed": passed_snapshots == len(snapshots),
        "snapshots": snapshots,
    }


def score_result(result: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    score = _score_pilot_gold(
        brief=_result_brief(result),
        visited_source_keys={
            str(item) for item in result.get("visited_source_keys", []) if str(item)
        },
        gold=gold,
    )
    score["identity_contract"] = _identity_score(result, gold)
    score["identity_snapshots"] = _identity_snapshot_score(result, gold)
    deterministic_identity = (
        str(result.get("response_source") or "") == "host_deterministic_binding"
    )
    score["brief_evidence_recall_applicable"] = not deterministic_identity
    if deterministic_identity:
        # A zero-call identity resolver emits no natural-language brief, so
        # brief citation recall is not a meaningful success criterion.
        score["required_group_recall"] = None
    score["evaluation_layer"] = DIAGNOSTIC_LAYER
    score["end_to_end_retrieval_evaluated"] = False
    return score


def _run_arm(
    *,
    arm: str,
    case: dict[str, Any],
    packet: dict[str, Any],
    client: Any,
    provider_id: str,
    model: str,
    provider_extra_body: dict[str, Any],
    max_output_tokens: int,
    thinking_mode: str,
    deadline_seconds: float,
    ledger_path: Path,
    budget: PilotBudget,
    run_id: str,
    repetition: int,
    private_checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    if arm == "deterministic":
        return _run_deterministic(case=case, packet=packet)
    common = {
        "case": case,
        "packet": packet,
        "client": client,
        "provider_id": provider_id,
        "model": model,
        "provider_extra_body": provider_extra_body,
        "max_output_tokens": max_output_tokens,
        "thinking_mode": thinking_mode,
        "deadline_seconds": deadline_seconds,
        "ledger_path": ledger_path,
        "budget": budget,
        "run_id": run_id,
        "repetition": repetition,
    }
    if arm == "one-pass":
        return _run_one_pass(**common)
    if arm == "eccr":
        return _run_eccr(**common)
    if arm == "eccr-audit":
        return _run_eccr_audit(
            **common,
            private_checkpoint_path=private_checkpoint_path,
        )
    raise ValueError(f"unsupported arm: {arm}")


def run_case_arm(
    *,
    bundle: CaseBundle,
    arm: str,
    repetition: int,
    output_dir: Path,
    ledger_path: Path,
    budget: PilotBudget,
    client: Any,
    provider_id: str,
    model: str,
    provider_extra_body: dict[str, Any],
    provider_fingerprint: dict[str, Any],
    max_output_tokens: int,
    thinking_mode: str,
    deadline_seconds: float,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unsupported arm: {arm}")
    run_id = f"{bundle.case_id}:{arm}:rep-{repetition:03d}"
    result_path = (
        output_dir
        / "runs"
        / bundle.case_id
        / arm
        / f"rep-{repetition:03d}"
        / "result.private.json"
    )
    if result_path.exists():
        existing = _load_object(result_path)
        if str(existing.get("run_id") or "") != run_id:
            raise ValueError(f"result path contains a different run: {result_path}")
        return existing
    base = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "case_id": bundle.case_id,
        "arm": arm,
        "repetition": repetition,
        "diagnostic_layer": DIAGNOSTIC_LAYER,
        "end_to_end_retrieval_evaluated": False,
        "candidate_generation_evaluated": False,
        "retrieval_policy": "fixed oracle evidence packet; no retrieval tools",
        "input_hashes": bundle.hashes,
        "provider": provider_fingerprint,
        "provider_id": provider_id,
        "model": model,
        "started_at": _utc_now(),
    }
    try:
        arm_result = _run_arm(
            arm=arm,
            case=bundle.case,
            packet=bundle.packet,
            client=client,
            provider_id=provider_id,
            model=model,
            provider_extra_body=provider_extra_body,
            max_output_tokens=max_output_tokens,
            thinking_mode=thinking_mode,
            deadline_seconds=deadline_seconds,
            ledger_path=ledger_path,
            budget=budget,
            run_id=run_id,
            repetition=repetition,
            private_checkpoint_path=(
                result_path.with_name("audit-checkpoint.private.json")
                if arm == "eccr-audit"
                else None
            ),
        )
        # Gold crosses the boundary only here, after the provider result is fixed.
        evaluation = score_result(arm_result, bundle.gold)
        final = {
            **base,
            "status": "COMPLETED",
            "finished_at": _utc_now(),
            "result": arm_result,
            "usage": _pilot_run_usage(ledger_path, run_id),
            "evaluation": evaluation,
        }
    except Exception as exc:
        final = {
            **base,
            "status": "FAILED",
            "finished_at": _utc_now(),
            "error_type": type(exc).__name__,
            "error_detail": str(exc)[:2000],
            "usage": _pilot_run_usage(ledger_path, run_id),
            "evaluation": None,
        }
    _atomic_write_json(result_path, final)
    return final


def _mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return round(statistics.fmean(items), 3) if items else None


def build_summary(results: list[dict[str, Any]], ledger_path: Path) -> dict[str, Any]:
    by_arm: dict[str, dict[str, Any]] = {}
    for arm in sorted({str(item.get("arm") or "") for item in results}):
        rows = [item for item in results if str(item.get("arm") or "") == arm]
        completed = [item for item in rows if item.get("status") == "COMPLETED"]
        recalls = [
            float(value)
            for item in completed
            for value in [(item.get("evaluation") or {}).get("required_group_recall")]
            if value is not None
        ]
        identity_exact = [
            bool(value)
            for item in completed
            for value in [
                ((item.get("evaluation") or {}).get("identity_contract") or {}).get(
                    "participant_exact_match"
                )
            ]
            if value is not None
        ]
        identity_snapshot_scores = [
            (item.get("evaluation") or {}).get("identity_snapshots") or {}
            for item in completed
        ]
        identity_snapshot_scores = [
            item for item in identity_snapshot_scores if item.get("applicable") is True
        ]
        identity_snapshot_count = sum(
            int(item.get("snapshot_count") or 0) for item in identity_snapshot_scores
        )
        identity_snapshot_passed = sum(
            int(item.get("passed_snapshots") or 0) for item in identity_snapshot_scores
        )
        attempt_latencies = [
            float((item.get("usage") or {}).get("elapsed_ms") or 0)
            for item in rows
            if int((item.get("usage") or {}).get("calls") or 0) > 0
        ]
        completed_latencies = [
            float((item.get("usage") or {}).get("elapsed_ms") or 0)
            for item in completed
            if int((item.get("usage") or {}).get("calls") or 0) > 0
        ]
        by_arm[arm] = {
            "runs": len(rows),
            "completed": len(completed),
            "failed": len(rows) - len(completed),
            "provider_calls": sum(
                int((item.get("usage") or {}).get("calls") or 0) for item in rows
            ),
            "provider_tokens_measured_lower_bound": sum(
                int((item.get("usage") or {}).get("total_measured_lower_bound") or 0)
                for item in rows
            ),
            "mean_attempt_provider_latency_ms": _mean(attempt_latencies),
            "mean_completed_run_provider_latency_ms": _mean(completed_latencies),
            "mean_required_group_recall": _mean(recalls),
            "identity_exact_evaluated_runs": len(identity_exact),
            "identity_exact_passed_runs": sum(identity_exact),
            "identity_exact_pass_rate": _mean(
                1.0 if value else 0.0 for value in identity_exact
            ),
            "identity_snapshots_evaluated": identity_snapshot_count,
            "identity_snapshots_passed": identity_snapshot_passed,
            "identity_snapshot_pass_rate": (
                round(identity_snapshot_passed / identity_snapshot_count, 3)
                if identity_snapshot_count
                else None
            ),
            "identity_snapshot_all_passed_runs": sum(
                item.get("all_passed") is True for item in identity_snapshot_scores
            ),
            "semantic_judgment_status": "PENDING_BLIND_HUMAN_REVIEW",
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "diagnostic_layer": DIAGNOSTIC_LAYER,
        "end_to_end_retrieval_evaluated": False,
        "candidate_generation_evaluated": False,
        "interpretation": (
            "These results isolate synthesis/control over a fixed oracle evidence "
            "packet. They must not be reported as end-to-end retrieval quality."
        ),
        "runs": len(results),
        "completed": sum(item.get("status") == "COMPLETED" for item in results),
        "failed": sum(item.get("status") == "FAILED" for item in results),
        "arms": by_arm,
        "usage_ledger_audit": _usage_ledger_audit(ledger_path),
        "results": [
            {
                "run_id": item.get("run_id"),
                "case_id": item.get("case_id"),
                "arm": item.get("arm"),
                "status": item.get("status"),
                "decision": (item.get("result") or {}).get("decision"),
                "provider_calls": (item.get("usage") or {}).get("calls", 0),
                "tokens": (item.get("usage") or {}).get(
                    "total_measured_lower_bound", 0
                ),
                "latency_ms": (item.get("usage") or {}).get("elapsed_ms", 0),
                "required_group_recall": (item.get("evaluation") or {}).get(
                    "required_group_recall"
                ),
                "identity_exact_match": (
                    ((item.get("evaluation") or {}).get("identity_contract") or {}).get(
                        "participant_exact_match"
                    )
                ),
                "identity_snapshot_pass": (
                    (
                        (item.get("evaluation") or {}).get("identity_snapshots") or {}
                    ).get("all_passed")
                ),
                "identity_snapshots_passed": (
                    (
                        (item.get("evaluation") or {}).get("identity_snapshots") or {}
                    ).get("passed_snapshots")
                ),
                "identity_snapshots_evaluated": (
                    (
                        (item.get("evaluation") or {}).get("identity_snapshots") or {}
                    ).get("snapshot_count")
                ),
                "semantic_judgment_status": (
                    (item.get("evaluation") or {}).get("semantic_judgment_status")
                ),
                "error_type": item.get("error_type"),
            }
            for item in results
        ],
    }


def _parse_arms(value: str) -> list[str]:
    arms = list(
        dict.fromkeys(item.strip() for item in value.split(",") if item.strip())
    )
    unknown = sorted(set(arms) - ARMS)
    if unknown:
        raise ValueError(f"unknown packet experiment arms: {unknown}")
    if not arms:
        raise ValueError("at least one arm is required")
    return arms


def _resolve_case_dirs(args: argparse.Namespace) -> list[Path]:
    roots = [Path(item).resolve() for item in (args.case_dir or [])]
    if args.cases_root:
        cases_root = Path(args.cases_root).resolve()
        roots.extend(
            sorted(
                path
                for path in cases_root.iterdir()
                if path.is_dir()
                and (path / "case.json").is_file()
                and (path / "evidence_packet.json").is_file()
                and (path / "gold.json").is_file()
            )
        )
    unique = list(dict.fromkeys(roots))
    if not unique:
        raise ValueError("no packet diagnostic cases were selected")
    return unique


def validate_command(args: argparse.Namespace) -> dict[str, Any]:
    bundles = [load_case_bundle(path) for path in _resolve_case_dirs(args)]
    return {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_layer": DIAGNOSTIC_LAYER,
        "end_to_end_retrieval_evaluated": False,
        "cases": [
            {
                "case_id": bundle.case_id,
                "case_dir": str(bundle.case_dir),
                "hashes": bundle.hashes,
                "delivered_source_keys": len(
                    reconstruction_packet_allowlist(bundle.packet)[0]
                ),
            }
            for bundle in bundles
        ],
    }


def run_command(args: argparse.Namespace) -> dict[str, Any]:
    bundles = [load_case_bundle(path) for path in _resolve_case_dirs(args)]
    arms = _parse_arms(args.arms)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / "usage.jsonl"
    existing = _usage_ledger_audit(ledger_path)
    _assert_usage_resumable(existing)
    budget = PilotBudget(
        max_calls=int(args.max_provider_calls),
        soft_token_limit=int(args.soft_token_limit),
        calls=int(existing["attempted_calls"]),
        tokens=int(existing["provider_tokens_measured_lower_bound"]),
    )
    provider_fingerprint = _provider_fingerprint(args.config, args.provider_id)
    needs_provider = any(arm != "deterministic" for arm in arms)
    if needs_provider:
        client, model, provider_extra_body = _provider_config(
            args.config, args.provider_id
        )
    else:
        client, model, provider_extra_body = None, "deterministic-host", {}

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "diagnostic_layer": DIAGNOSTIC_LAYER,
        "end_to_end_retrieval_evaluated": False,
        "candidate_generation_evaluated": False,
        "arms": arms,
        "repetitions": int(args.repetitions),
        "provider": provider_fingerprint,
        "model": model,
        "case_hashes": {bundle.case_id: bundle.hashes for bundle in bundles},
        "experiment_parameters": {
            "max_provider_calls": int(args.max_provider_calls),
            "soft_token_limit": int(args.soft_token_limit),
            "max_output_tokens": int(args.max_output_tokens),
            "thinking_mode": str(args.thinking_mode),
            "deadline_seconds": float(args.deadline_seconds),
            "eccr_max_model_calls": 1,
            "eccr_audit_model_calls": 2,
            "retrieval_rounds": 0,
        },
        "gold_model_boundary": (
            "prompt builders and arm execution receive only case+packet; gold is "
            "passed only to post-response scoring"
        ),
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        old = _load_object(manifest_path)
        for field in (
            "schema_version",
            "arms",
            "repetitions",
            "case_hashes",
            "provider",
            "model",
            "experiment_parameters",
        ):
            if old.get(field) != manifest.get(field):
                raise ValueError(f"output resume manifest mismatch: {field}")
    else:
        _atomic_write_json(manifest_path, manifest)

    results: list[dict[str, Any]] = []
    for bundle in bundles:
        for arm in arms:
            if arm == "deterministic" and not bundle.case.get("host_subject_bindings"):
                continue
            for repetition in range(1, int(args.repetitions) + 1):
                results.append(
                    run_case_arm(
                        bundle=bundle,
                        arm=arm,
                        repetition=repetition,
                        output_dir=output_dir,
                        ledger_path=ledger_path,
                        budget=budget,
                        client=client,
                        provider_id=args.provider_id,
                        model=model,
                        provider_extra_body=provider_extra_body,
                        provider_fingerprint=provider_fingerprint,
                        max_output_tokens=int(args.max_output_tokens),
                        thinking_mode=args.thinking_mode,
                        deadline_seconds=float(args.deadline_seconds),
                    )
                )
    summary = build_summary(results, ledger_path)
    _atomic_write_json(output_dir / "summary.json", summary)
    return summary


def _add_cases(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--case-dir", action="append", default=[])
    parser.add_argument("--cases-root")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Auditable fixed-packet ECCR diagnostic. This does not evaluate "
            "end-to-end retrieval."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    _add_cases(validate_parser)
    validate_parser.set_defaults(handler=validate_command)

    run_parser = subparsers.add_parser("run")
    _add_cases(run_parser)
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--provider-id", required=True)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--arms", default="one-pass,eccr,deterministic")
    run_parser.add_argument("--repetitions", type=int, default=1)
    run_parser.add_argument("--max-provider-calls", type=int, default=4)
    run_parser.add_argument("--soft-token-limit", type=int, default=0)
    run_parser.add_argument("--max-output-tokens", type=int, default=384000)
    run_parser.add_argument(
        "--thinking-mode", choices=("enabled", "disabled"), default="enabled"
    )
    run_parser.add_argument("--deadline-seconds", type=float, default=180.0)
    run_parser.set_defaults(handler=run_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "repetitions", 1) <= 0:
        raise ValueError("repetitions must be positive")
    if getattr(args, "max_provider_calls", 0) < 0:
        raise ValueError("max-provider-calls must be non-negative")
    result = args.handler(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
