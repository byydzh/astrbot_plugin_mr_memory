from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal, Mapping


GraphMutationOperation = Literal[
    "upsert_edge",
    "revise_edge",
    "reinforce_edge",
    "inhibit_edge",
    "retire_edge",
    "revise_relation",
    "deprecate_relation",
    "merge_nodes",
]

PlasticEpistemicState = Literal[
    "HYPOTHESIS",
    "SUPPORTED",
    "CONTESTED",
    "CONFIRMED",
]

PlasticNodeKind = Literal[
    "concept",
    "behavior",
    "symbol",
    "topic",
    "preference",
    "procedure",
]


_RELATION_KEY_RE = re.compile(r"^[a-z][a-z0-9_:-]{0,79}$")
_NODE_KEY_RE = re.compile(r"^plastic:[a-z][a-z0-9_-]{0,31}:[0-9a-f]{20}$")
_ALLOWED_NODE_KINDS = {
    "concept",
    "behavior",
    "symbol",
    "topic",
    "preference",
    "procedure",
}
_ALLOWED_EPISTEMIC_STATES = {
    "HYPOTHESIS",
    "SUPPORTED",
    "CONTESTED",
    "CONFIRMED",
}


PLASTIC_GRAPH_MAINTENANCE_PROMPT = """You maintain a group-scoped plastic
associative graph. Raw messages, account identities, authorship, replies, recalls,
and evidence provenance are immutable host truth and must never be rewritten.

Use graph mutations only for learned associations such as local meanings, symbols,
behavioral expectations, preferences, procedures, and useful traversal paths. Search
the existing graph before inventing a relation type. Prefer reinforcing or revising
an existing relation over creating synonyms. A relation revision creates a new
version; old meanings remain auditable. Negative feedback should usually inhibit a
path before retiring it. Never use a model-created edge as evidence for itself.

Epistemic state is explicit and is never derived from an embedding threshold:
- HYPOTHESIS: one plausible reading with material doubt.
- SUPPORTED: contextual evidence supports it, but alternatives remain possible.
- CONTESTED: incompatible readings have live evidence; keep the alternatives.
- CONFIRMED: direct human clarification or unusually explicit evidence confirms it.
For slang, euphemism, irony, reclaimed insults, and jokes, preserve uncertainty and
competing edges instead of collapsing them into one sanitized summary. Use
revise_edge when later evidence changes the state, statement, or uncertainty note.

Every mutation must cite exact source keys from the current maintenance evidence.
Factual confidence and retrieval utility are different: usefulness feedback changes
utility, while contradictions must remain explicit evidence. Return bounded JSON
matching the host mutation schema; the host enforces group scope, evidence existence,
allowed node kinds, versioning, and reversible retirement.
"""


def _bounded_text(
    value: Any,
    name: str,
    *,
    limit: int,
    required: bool = False,
) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    if required and not text:
        raise ValueError(f"{name} is required")
    if len(text) > limit:
        raise ValueError(f"{name} exceeds {limit} characters")
    return text


def _bounded_float(value: Any, name: str, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number) or number < low or number > high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return number


def _string_list(
    value: Any,
    name: str,
    *,
    limit: int,
    item_limit: int,
    required: bool = False,
) -> tuple[str, ...]:
    if value is None:
        raw: list[Any] = []
    elif isinstance(value, list):
        raw = value
    else:
        raise ValueError(f"{name} must be a list")
    if len(raw) > limit:
        raise ValueError(f"{name} has more than {limit} items")
    result = tuple(
        dict.fromkeys(
            _bounded_text(item, f"{name}[]", limit=item_limit, required=True)
            for item in raw
        )
    )
    if required and not result:
        raise ValueError(f"{name} must not be empty")
    return result


def canonical_relation_key(value: Any, *, name: str = "") -> str:
    key = unicodedata.normalize("NFKC", str(value or "")).strip().casefold()
    key = key.replace(" ", "_").replace("-", "_")
    key = re.sub(r"[^a-z0-9_:-]+", "", key)
    if not key:
        label = unicodedata.normalize("NFKC", str(name or "")).strip().casefold()
        digest = hashlib.sha256(label.encode("utf-8")).hexdigest()[:16]
        key = f"learned:{digest}"
    if not _RELATION_KEY_RE.fullmatch(key):
        raise ValueError("relation.key must be a stable lowercase ASCII key")
    return key


def canonical_plastic_node_key(kind: str, label: str) -> str:
    normalized_kind = str(kind or "").strip().casefold()
    if normalized_kind not in _ALLOWED_NODE_KINDS:
        raise ValueError(f"unsupported plastic node kind: {normalized_kind}")
    normalized_label = " ".join(
        unicodedata.normalize("NFKC", str(label or "")).strip().casefold().split()
    )
    if not normalized_label:
        raise ValueError("plastic node label is required")
    digest = hashlib.sha256(normalized_label.encode("utf-8")).hexdigest()[:20]
    return f"plastic:{normalized_kind}:{digest}"


@dataclass(frozen=True, slots=True)
class PlasticNodeProposal:
    kind: PlasticNodeKind
    label: str
    description: str = ""
    node_key: str = ""

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        required: bool = True,
    ) -> PlasticNodeProposal | None:
        if value is None and not required:
            return None
        if not isinstance(value, Mapping):
            raise ValueError("node proposal must be an object")
        kind = _bounded_text(
            value.get("kind"), "node.kind", limit=32, required=True
        ).casefold()
        if kind not in _ALLOWED_NODE_KINDS:
            raise ValueError(f"unsupported plastic node kind: {kind}")
        label = _bounded_text(
            value.get("label"), "node.label", limit=160, required=True
        )
        supplied_key = _bounded_text(
            value.get("node_key"), "node.node_key", limit=80
        )
        expected_key = canonical_plastic_node_key(kind, label)
        if supplied_key and supplied_key != expected_key:
            raise ValueError("node_key does not match the host-derived node identity")
        return cls(
            kind=kind,  # type: ignore[arg-type]
            label=label,
            description=_bounded_text(
                value.get("description"), "node.description", limit=800
            ),
            node_key=expected_key,
        )


@dataclass(frozen=True, slots=True)
class RelationTypeProposal:
    key: str
    name: str
    description: str
    source_kinds: tuple[str, ...]
    target_kinds: tuple[str, ...]
    inverse_key: str = ""
    symmetric: bool = False
    risk_class: str = "normal"

    @classmethod
    def from_value(cls, value: Any) -> RelationTypeProposal:
        if not isinstance(value, Mapping):
            raise ValueError("relation proposal must be an object")
        name = _bounded_text(
            value.get("name"), "relation.name", limit=120, required=True
        )
        key = canonical_relation_key(value.get("key"), name=name)
        source_kinds = _string_list(
            value.get("source_kinds"),
            "relation.source_kinds",
            limit=12,
            item_limit=32,
            required=True,
        )
        target_kinds = _string_list(
            value.get("target_kinds"),
            "relation.target_kinds",
            limit=12,
            item_limit=32,
            required=True,
        )
        if not set(source_kinds).issubset(_ALLOWED_NODE_KINDS):
            raise ValueError("relation source kinds must be plastic node kinds")
        if not set(target_kinds).issubset(_ALLOWED_NODE_KINDS):
            raise ValueError("relation target kinds must be plastic node kinds")
        inverse_raw = _bounded_text(
            value.get("inverse_key"), "relation.inverse_key", limit=80
        )
        inverse_key = (
            canonical_relation_key(inverse_raw) if inverse_raw else ""
        )
        risk_class = _bounded_text(
            value.get("risk_class") or "normal",
            "relation.risk_class",
            limit=24,
            required=True,
        ).casefold()
        if risk_class not in {"normal", "sensitive"}:
            raise ValueError("plastic relations cannot use identity or privilege risk")
        return cls(
            key=key,
            name=name,
            description=_bounded_text(
                value.get("description"),
                "relation.description",
                limit=1000,
                required=True,
            ),
            source_kinds=source_kinds,
            target_kinds=target_kinds,
            inverse_key=inverse_key,
            symmetric=bool(value.get("symmetric", False)),
            risk_class=risk_class,
        )


@dataclass(frozen=True, slots=True)
class GraphMutation:
    operation: GraphMutationOperation
    evidence_source_keys: tuple[str, ...]
    confidence: float
    utility_delta: float
    statement: str = ""
    epistemic_state: PlasticEpistemicState | None = None
    uncertainty: str = ""
    source: PlasticNodeProposal | None = None
    target: PlasticNodeProposal | None = None
    relation: RelationTypeProposal | None = None
    edge_id: int | None = None
    source_node_key: str = ""
    target_node_key: str = ""

    def as_dict(self) -> dict[str, object]:
        def node_value(node: PlasticNodeProposal | None) -> object:
            if node is None:
                return None
            return {
                "kind": node.kind,
                "label": node.label,
                "description": node.description,
                "node_key": node.node_key,
            }

        relation_value: object = None
        if self.relation is not None:
            relation_value = {
                "key": self.relation.key,
                "name": self.relation.name,
                "description": self.relation.description,
                "source_kinds": list(self.relation.source_kinds),
                "target_kinds": list(self.relation.target_kinds),
                "inverse_key": self.relation.inverse_key,
                "symmetric": self.relation.symmetric,
                "risk_class": self.relation.risk_class,
            }
        return {
            "operation": self.operation,
            "evidence_source_keys": list(self.evidence_source_keys),
            "confidence": self.confidence,
            "utility_delta": self.utility_delta,
            "statement": self.statement,
            "epistemic_state": self.epistemic_state,
            "uncertainty": self.uncertainty,
            "source": node_value(self.source),
            "target": node_value(self.target),
            "relation": relation_value,
            "edge_id": self.edge_id,
            "source_node_key": self.source_node_key,
            "target_node_key": self.target_node_key,
        }


def parse_graph_mutation(value: str | Mapping[str, Any]) -> GraphMutation:
    if isinstance(value, str):
        text = value.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1)
        try:
            raw: Any = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("graph mutation must be one JSON object") from exc
    else:
        raw = dict(value)
    if not isinstance(raw, dict):
        raise ValueError("graph mutation must be one object")

    operation = _bounded_text(
        raw.get("operation"), "operation", limit=32, required=True
    ).casefold()
    supported = {
        "upsert_edge",
        "revise_edge",
        "reinforce_edge",
        "inhibit_edge",
        "retire_edge",
        "revise_relation",
        "deprecate_relation",
        "merge_nodes",
    }
    if operation not in supported:
        raise ValueError(f"unsupported graph mutation operation: {operation}")
    evidence = _string_list(
        raw.get("evidence_source_keys"),
        "evidence_source_keys",
        limit=32,
        item_limit=240,
        required=True,
    )
    confidence = _bounded_float(
        raw.get("confidence", 0.0), "confidence", 0.0, 1.0
    )
    utility_delta = _bounded_float(
        raw.get("utility_delta", 0.0), "utility_delta", -2.0, 2.0
    )
    edge_id_raw = raw.get("edge_id")
    edge_id = None if edge_id_raw in (None, "") else int(edge_id_raw)
    if edge_id is not None and edge_id <= 0:
        raise ValueError("edge_id must be positive")

    relation = None
    source = None
    target = None
    source_node_key = _bounded_text(
        raw.get("source_node_key"), "source_node_key", limit=80
    )
    target_node_key = _bounded_text(
        raw.get("target_node_key"), "target_node_key", limit=80
    )
    if source_node_key and not _NODE_KEY_RE.fullmatch(source_node_key):
        raise ValueError("source_node_key is not a host-derived plastic node key")
    if target_node_key and not _NODE_KEY_RE.fullmatch(target_node_key):
        raise ValueError("target_node_key is not a host-derived plastic node key")

    if operation == "upsert_edge":
        source = PlasticNodeProposal.from_value(raw.get("source"))
        target = PlasticNodeProposal.from_value(raw.get("target"))
        relation = RelationTypeProposal.from_value(raw.get("relation"))
        assert source is not None and target is not None
        if source.kind not in relation.source_kinds:
            raise ValueError("source node kind is outside the relation schema")
        if target.kind not in relation.target_kinds:
            raise ValueError("target node kind is outside the relation schema")
    elif operation == "revise_edge":
        if edge_id is None:
            raise ValueError("revise_edge requires edge_id")
    elif operation == "revise_relation":
        relation = RelationTypeProposal.from_value(raw.get("relation"))
    elif operation == "deprecate_relation":
        relation_raw = raw.get("relation")
        if not isinstance(relation_raw, Mapping):
            raise ValueError("deprecate_relation requires relation.key")
        key = canonical_relation_key(relation_raw.get("key"))
        relation = RelationTypeProposal(
            key=key,
            name=_bounded_text(
                relation_raw.get("name") or key,
                "relation.name",
                limit=120,
                required=True,
            ),
            description=_bounded_text(
                relation_raw.get("description") or "deprecated",
                "relation.description",
                limit=1000,
                required=True,
            ),
            source_kinds=("concept",),
            target_kinds=("concept",),
        )
    elif operation in {"reinforce_edge", "inhibit_edge", "retire_edge"}:
        if edge_id is None:
            raise ValueError(f"{operation} requires edge_id")
        if operation == "reinforce_edge" and utility_delta <= 0:
            raise ValueError("reinforce_edge requires a positive utility_delta")
        if operation in {"inhibit_edge", "retire_edge"} and utility_delta > 0:
            raise ValueError(f"{operation} cannot increase utility")
    elif operation == "merge_nodes":
        if not source_node_key or not target_node_key:
            raise ValueError("merge_nodes requires both node keys")
        if source_node_key == target_node_key:
            raise ValueError("cannot merge a node into itself")

    raw_epistemic_state = _bounded_text(
        raw.get("epistemic_state"), "epistemic_state", limit=24
    ).upper()
    epistemic_state: PlasticEpistemicState | None = None
    if raw_epistemic_state:
        if raw_epistemic_state not in _ALLOWED_EPISTEMIC_STATES:
            raise ValueError("epistemic_state is invalid")
        epistemic_state = raw_epistemic_state  # type: ignore[assignment]
    if operation == "upsert_edge" and epistemic_state is None:
        epistemic_state = "HYPOTHESIS"
    if operation == "revise_edge" and epistemic_state is None:
        raise ValueError("revise_edge requires epistemic_state")
    uncertainty = _bounded_text(
        raw.get("uncertainty"), "uncertainty", limit=1200
    )
    if epistemic_state in {"HYPOTHESIS", "CONTESTED"} and not uncertainty:
        if operation == "upsert_edge" and not raw_epistemic_state:
            uncertainty = "The evidence does not yet settle this association."
        else:
            raise ValueError(
                "uncertain or contested edges require an explicit uncertainty note"
            )

    return GraphMutation(
        operation=operation,  # type: ignore[arg-type]
        evidence_source_keys=evidence,
        confidence=confidence,
        utility_delta=utility_delta,
        statement=_bounded_text(raw.get("statement"), "statement", limit=1200),
        epistemic_state=epistemic_state,
        uncertainty=uncertainty,
        source=source,
        target=target,
        relation=relation,
        edge_id=edge_id,
        source_node_key=source_node_key,
        target_node_key=target_node_key,
    )
