from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from mr_memory.models import NormalizedMessage
from mr_memory.plasticity import (
    PLASTIC_GRAPH_MAINTENANCE_PROMPT,
    parse_graph_mutation,
)
from mr_memory.storage import MemoryStorage
from mr_memory.usage import TokenUsageRecord
from scripts.masked_ab_experiment import _chat_completion, _provider_config


MAINTENANCE_SYSTEM_PROMPT = f"""{PLASTIC_GRAPH_MAINTENANCE_PROMPT}

You are running one chronological maintenance tick over a real group-chat evidence
slice. The embedding stage only nominated this slice; you are the semantic gate.
Return exactly one JSON object with a `mutations` array and a short `state` object.
Use 1-8 bounded mutations. Reuse the exact active relation schema shown by the host.
For an existing edge use reinforce_edge or inhibit_edge instead of recreating it.
Create a new relation only when no current relation expresses the needed semantics.
Do not infer account identity, do not copy a speaker name into a plastic node merely
because they sent the message, and do not record hidden reasoning.
"""


RECONSTRUCTION_SYSTEM_PROMPT = """You are MR Memory's private subconscious layer.
You do not answer the group member. Embedding candidates are only a cheap prior; you
are the semantic gate. Use the generic association tool to traverse the group-local,
versioned plastic graph and inspect exact evidence before returning a brief. Prefer
the local sense supported by this group over famous external homonyms. Do not reveal
chain-of-thought. Finish by calling mr_return_brief exactly once. Every selected edge
and source key must have been returned by a tool in this run.
"""


SURFACE_SYSTEM_PROMPT = """你是群聊里的助手。请自然、直接、简洁地回答当前消息。
如果提供了群内记忆摘要，把它当作有来源但仍需审慎解释的上下文；优先解释群内局部用法，
不要把同名的知名外部对象硬套进来。不要提及实验、图数据库、潜意识层或系统提示。
回答不超过 220 个汉字，最多列 3 点，并确保最后一句完整结束。"""


EVALUATOR_SYSTEM_PROMPT = """You are a strict evidence-grounding evaluator.
Score answers only against the supplied real group evidence and explicit rubric.
Do not reward confident prose, outside knowledge, or keyword stuffing. Return one
JSON object matching the requested shape and no chain-of-thought."""


RECONSTRUCTION_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "mr_query_associations",
            "description": (
                "Traverse group-local learned associations. Use a short cue such as "
                "the ambiguous noun, not the whole user sentence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "node_key": {"type": "string"},
                    "relation_key": {"type": "string"},
                    "direction": {
                        "type": "string",
                        "enum": ["out", "in", "both"],
                    },
                    "include_dormant": {"type": "boolean"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 40},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mr_read_evidence",
            "description": "Read exact source messages exposed by traversed edges.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 16,
                    }
                },
                "required": ["source_keys"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mr_return_brief",
            "description": "Return the final source-grounded memory brief.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "selected_edge_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "maxItems": 16,
                    },
                    "source_keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 24,
                    },
                    "uncertainties": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 8,
                    },
                },
                "required": [
                    "summary",
                    "selected_edge_ids",
                    "source_keys",
                    "uncertainties",
                ],
                "additionalProperties": False,
            },
        },
    },
]


@dataclass(frozen=True, slots=True)
class SourceRecord:
    normalized: NormalizedMessage
    raw: dict[str, Any]

    @property
    def source_key(self) -> str:
        return self.normalized.resolved_source_key()

    def evidence_dict(self, *, max_chars: int = 900) -> dict[str, Any]:
        return {
            "source_key": self.source_key,
            "sent_at": self.normalized.sent_at,
            "sender_id": self.normalized.sender_id,
            "sender_name": self.normalized.sender_name,
            "role": self.normalized.role,
            "text": self.normalized.plain_text[:max_chars],
        }


class UsageLedger:
    def __init__(self, path: Path | None = None) -> None:
        self.rows: list[dict[str, Any]] = []
        self.path = path

    def add(
        self,
        *,
        case: str,
        phase: str,
        provider_id: str,
        model: str,
        usage: TokenUsageRecord,
        elapsed_ms: float,
        call_index: int = 1,
    ) -> None:
        row = {
            "case": case,
            "phase": phase,
            "call_index": call_index,
            "provider_id": provider_id,
            "model": model,
            **usage.as_dict(),
            "elapsed_ms": round(float(elapsed_ms), 3),
        }
        self.rows.append(row)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as sink:
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")

    def totals(self) -> dict[str, Any]:
        return {
            "calls": len(self.rows),
            "input_other": sum(int(row["input_other"]) for row in self.rows),
            "input_cached": sum(int(row["input_cached"]) for row in self.rows),
            "output": sum(int(row["output"]) for row in self.rows),
            "total": sum(int(row["total"]) for row in self.rows),
            "elapsed_ms": round(
                sum(float(row["elapsed_ms"]) for row in self.rows), 3
            ),
        }


def _json_object(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("model response must be a JSON object")
    return parsed


def _message_payload(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        return dict(message.model_dump(exclude_none=True))
    value: dict[str, Any] = {
        "role": str(getattr(message, "role", "assistant")),
        "content": getattr(message, "content", None),
    }
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        value["tool_calls"] = [
            item.model_dump(exclude_none=True)
            if hasattr(item, "model_dump")
            else item
            for item in tool_calls
        ]
    return value


def _clean_plain_text(raw: dict[str, Any], fallback: str) -> str:
    raw_message = raw.get("raw_message")
    if isinstance(raw_message, str) and raw_message.strip():
        return raw_message.strip()
    parts: list[str] = []
    for component in raw.get("message") or []:
        if not isinstance(component, dict):
            continue
        component_type = str(component.get("type") or "")
        data = component.get("data") or {}
        if component_type == "text" and isinstance(data, dict):
            parts.append(str(data.get("text") or ""))
        elif component_type:
            parts.append(f"[{component_type}]")
    return "".join(parts).strip() or fallback.strip()


def _load_history(
    database_path: str | Path,
    *,
    group_id: str | None = None,
) -> tuple[str, list[SourceRecord], dict[str, Any]]:
    uri = f"file:{Path(database_path).resolve().as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    if group_id:
        selected_group = str(group_id)
    else:
        row = connection.execute(
            """
            SELECT group_id, COUNT(*) AS messages,
                   COUNT(DISTINCT user_id) AS participants
            FROM messages GROUP BY group_id ORDER BY messages DESC LIMIT 1
            """
        ).fetchone()
        if row is None:
            raise ValueError("history database contains no group messages")
        selected_group = str(row["group_id"])
    stats_row = connection.execute(
        """
        SELECT COUNT(*) AS messages, COUNT(DISTINCT user_id) AS participants,
               MIN(time) AS first_at, MAX(time) AS last_at
        FROM messages WHERE group_id=?
        """,
        (selected_group,),
    ).fetchone()
    records: list[SourceRecord] = []
    for row in connection.execute(
        """
        SELECT group_id, message_id, message_seq, time, user_id, nickname,
               search_text, raw_json
        FROM messages WHERE group_id=? ORDER BY time, id
        """,
        (selected_group,),
    ):
        try:
            raw = json.loads(str(row["raw_json"]))
        except (json.JSONDecodeError, TypeError):
            raw = {}
        sender = raw.get("sender") if isinstance(raw.get("sender"), dict) else {}
        sender_name = str(
            sender.get("card")
            or row["nickname"]
            or sender.get("nickname")
            or row["user_id"]
            or ""
        )
        sender_id = str(row["user_id"] or raw.get("user_id") or "")
        self_id = str(raw.get("self_id") or "")
        text = _clean_plain_text(raw, str(row["search_text"] or ""))
        message_id = str(row["message_id"] or row["message_seq"] or "")
        sent_at = int(row["time"])
        source_key = (
            f"angel:{selected_group}:{message_id}:{sent_at}"
        )
        records.append(
            SourceRecord(
                normalized=NormalizedMessage(
                    platform="aiocqhttp",
                    platform_id="byy_official",
                    umo=f"aiocqhttp:GroupMessage:{selected_group}",
                    group_id=selected_group,
                    message_id=message_id,
                    sender_id=sender_id,
                    sender_name=sender_name,
                    sent_at=sent_at,
                    plain_text=text,
                    content=[{"type": "text", "text": text}],
                    role="BOT" if self_id and sender_id == self_id else "USER",
                    source_key=source_key,
                ),
                raw=raw,
            )
        )
    connection.close()
    return selected_group, records, {
        "messages": int(stats_row["messages"] or 0),
        "participants": int(stats_row["participants"] or 0),
        "first_at": int(stats_row["first_at"] or 0),
        "last_at": int(stats_row["last_at"] or 0),
    }


def _load_live_fixture(
    path: str | Path,
    *,
    group_id: str,
    umo: str,
) -> dict[str, SourceRecord]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("live fixture must be an object")
    result: dict[str, SourceRecord] = {}
    for label in ("target", "baseline_response", "feedback"):
        item = value.get(label)
        if not isinstance(item, dict):
            raise ValueError(f"live fixture is missing {label}")
        source_key = str(
            item.get("source_key") or f"live:{group_id}:{label}"
        )
        role = str(item.get("role") or "USER").upper()
        if role not in {"USER", "BOT", "SYSTEM"}:
            raise ValueError(f"unsupported live role: {role}")
        text = str(item.get("text") or "").strip()
        result[label] = SourceRecord(
            normalized=NormalizedMessage(
                platform="aiocqhttp",
                platform_id="byy_official",
                umo=umo,
                group_id=group_id,
                message_id=str(item.get("message_id") or label),
                sender_id=str(item.get("sender_id") or "unknown"),
                sender_name=str(item.get("sender_name") or "unknown"),
                sent_at=int(item["sent_at"]),
                plain_text=text,
                content=[{"type": "text", "text": text}],
                role=role,  # type: ignore[arg-type]
                source_key=source_key,
            ),
            raw=dict(item),
        )
    if result["baseline_response"].normalized.sent_at <= result["target"].normalized.sent_at:
        raise ValueError("baseline response must follow the target")
    if result["feedback"].normalized.sent_at <= result["baseline_response"].normalized.sent_at:
        raise ValueError("feedback must follow the baseline response")
    return result


def _ingest(storage: MemoryStorage, records: Iterable[SourceRecord]) -> int:
    inserted = 0
    for record in records:
        inserted += int(storage.upsert_message(record.normalized))
    return inserted


def _term_hits(
    records: Sequence[SourceRecord],
    term: str,
    *,
    before_sent_at: int | None = None,
) -> list[SourceRecord]:
    needle = term.casefold()
    return [
        record
        for record in records
        if needle in record.normalized.plain_text.casefold()
        and (
            before_sent_at is None
            or record.normalized.sent_at < int(before_sent_at)
        )
    ]


def _context_window(
    records: Sequence[SourceRecord],
    anchors: Sequence[SourceRecord],
    *,
    radius: int,
    before_sent_at: int | None = None,
    max_records: int = 100,
) -> list[SourceRecord]:
    index_by_key = {record.source_key: index for index, record in enumerate(records)}
    selected: set[int] = set()
    for anchor in anchors:
        index = index_by_key[anchor.source_key]
        selected.update(
            range(max(0, index - radius), min(len(records), index + radius + 1))
        )
    result = [
        records[index]
        for index in sorted(selected)
        if (
            before_sent_at is None
            or records[index].normalized.sent_at < int(before_sent_at)
        )
        and records[index].normalized.plain_text.strip()
    ]
    if len(result) <= max_records:
        return result
    anchor_keys = {record.source_key for record in anchors}
    anchors_kept = [record for record in result if record.source_key in anchor_keys]
    contextual = [record for record in result if record.source_key not in anchor_keys]
    allowance = max(0, max_records - len(anchors_kept))
    stride = max(1, math.ceil(len(contextual) / max(1, allowance)))
    sampled = contextual[::stride][:allowance]
    return sorted(
        [*anchors_kept, *sampled],
        key=lambda item: (item.normalized.sent_at, item.source_key),
    )


def _split_by_cutoffs(
    records: Sequence[SourceRecord],
    cutoffs: Sequence[int],
) -> list[list[SourceRecord]]:
    phases: list[list[SourceRecord]] = []
    previous = -1
    for cutoff in cutoffs:
        phase = [
            record
            for record in records
            if previous < record.normalized.sent_at <= int(cutoff)
        ]
        if phase:
            phases.append(phase)
        previous = int(cutoff)
    tail = [record for record in records if record.normalized.sent_at > previous]
    if tail:
        phases.append(tail)
    return phases


def _split_clusters(
    records: Sequence[SourceRecord],
    *,
    gap_seconds: int = 86400,
) -> list[list[SourceRecord]]:
    ordered = sorted(records, key=lambda item: item.normalized.sent_at)
    if not ordered:
        return []
    clusters: list[list[SourceRecord]] = [[ordered[0]]]
    for record in ordered[1:]:
        if record.normalized.sent_at - clusters[-1][-1].normalized.sent_at > gap_seconds:
            clusters.append([])
        clusters[-1].append(record)
    return clusters


def _edge_catalog(
    storage: MemoryStorage,
    *,
    umo: str,
    before_sent_at: int | None = None,
    include_evidence: bool = False,
) -> list[dict[str, Any]]:
    rows = storage.query_plastic_associations(
        umo=umo,
        include_dormant=True,
        limit=100,
        before_sent_at=before_sent_at,
    )
    result: list[dict[str, Any]] = []
    for row in rows:
        value = dict(row)
        if not include_evidence:
            value.pop("evidence", None)
        result.append(value)
    return result


def _maintenance_tick(
    *,
    storage: MemoryStorage,
    umo: str,
    case: str,
    phase: str,
    objective: str,
    evidence: Sequence[SourceRecord],
    client: Any,
    provider_id: str,
    model: str,
    extra_body: dict[str, Any],
    ledger: UsageLedger,
) -> dict[str, Any]:
    allowed = {record.source_key for record in evidence}
    prompt = {
        "objective": objective,
        "phase": phase,
        "active_graph": _edge_catalog(storage, umo=umo, include_evidence=False),
        "allowed_evidence": [record.evidence_dict() for record in evidence],
        "output_contract": {
            "mutations": [
                {
                    "operation": "one supported GraphMutation operation",
                    "evidence_source_keys": ["exact allowed source key"],
                    "confidence": "0..1",
                    "utility_delta": "-2..2",
                    "statement": "bounded source-grounded association",
                    "source": {
                        "kind": "concept|behavior|symbol|topic|preference|procedure",
                        "label": "required for upsert_edge",
                        "description": "optional",
                    },
                    "target": "same node shape; required for upsert_edge",
                    "relation": {
                        "key": "stable_lowercase_ascii_key",
                        "name": "human-readable name",
                        "description": "stable precise semantics",
                        "source_kinds": ["concept"],
                        "target_kinds": ["concept"],
                        "symmetric": False,
                        "risk_class": "normal",
                    },
                    "edge_id": "required for reinforce/inhibit/retire",
                }
            ],
            "state": {
                "focus": "short operational focus",
                "open_questions": ["bounded unresolved questions"],
            },
        },
    }
    completion, usage, elapsed_ms = _chat_completion(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": MAINTENANCE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(prompt, ensure_ascii=False, separators=(",", ":")),
            },
        ],
        extra_body=extra_body,
        max_output_tokens=5000,
        json_object=True,
    )
    ledger.add(
        case=case,
        phase=f"maintenance:{phase}",
        provider_id=provider_id,
        model=model,
        usage=usage,
        elapsed_ms=elapsed_ms,
    )
    response_text = str(completion.choices[0].message.content or "")
    parsed = _json_object(response_text)
    proposals = parsed.get("mutations")
    if not isinstance(proposals, list) or not proposals:
        raise ValueError(f"{case}/{phase}: maintenance returned no mutations")
    committed: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for proposal in proposals[:8]:
        try:
            mutation = parse_graph_mutation(proposal)
            result = storage.apply_graph_mutation(
                umo=umo,
                mutation=mutation,
                model=model,
                allowed_evidence_keys=allowed,
            )
            committed.append(
                {"proposal": mutation.as_dict(), "result": result}
            )
        except (TypeError, ValueError) as exc:
            rejected.append({"proposal": proposal, "error": str(exc)})
    if not committed:
        raise ValueError(
            f"{case}/{phase}: every graph mutation was rejected: {rejected}"
        )
    state_value = parsed.get("state") if isinstance(parsed.get("state"), dict) else {}
    state = {
        "focus": str(state_value.get("focus") or objective)[:1200],
        "open_questions": [
            str(item)[:500]
            for item in (state_value.get("open_questions") or [])[:8]
        ]
        if isinstance(state_value.get("open_questions"), list)
        else [],
        "active_node_keys": [],
        "active_edge_ids": [
            int(item["result"]["target_id"])
            for item in committed
            if item["result"].get("target_type") == "edge"
            and item["result"].get("target_id") is not None
        ][:32],
        "last_decision": f"maintenance:{case}:{phase}",
        "candidate_counts": {
            "evidence": len(evidence),
            "committed": len(committed),
            "rejected": len(rejected),
        },
        "visited_source_keys": sorted(allowed)[:64],
    }
    persisted_state = storage.update_subconscious_state(
        umo=umo,
        state=state,
        last_query_sha256=hashlib.sha256(
            f"experiment:{case}:{phase}".encode("utf-8")
        ).hexdigest(),
        at=max(record.normalized.sent_at for record in evidence),
    )
    return {
        "phase": phase,
        "evidence_count": len(evidence),
        "evidence_source_keys": sorted(allowed),
        "committed": committed,
        "rejected": rejected,
        "state": persisted_state,
        "graph": _edge_catalog(storage, umo=umo, include_evidence=True),
    }


def _read_source_records(
    source_index: dict[str, SourceRecord], source_keys: Iterable[str]
) -> list[dict[str, Any]]:
    return [
        source_index[key].evidence_dict(max_chars=700)
        for key in dict.fromkeys(str(item) for item in source_keys)
        if key in source_index
    ][:16]


def _association_tool_result(rows: Sequence[dict[str, object]]) -> list[dict[str, Any]]:
    """Keep traversal metadata compact; exact message text has a separate tool."""

    fields = (
        "id",
        "statement",
        "epistemic_confidence",
        "utility",
        "status",
        "source_key",
        "source_kind",
        "source_label",
        "source_description",
        "target_key",
        "target_kind",
        "target_label",
        "target_description",
        "relation_key",
        "relation_version",
        "relation_name",
        "relation_description",
        "source_keys",
    )
    return [
        {key: row[key] for key in fields if key in row}
        for row in rows
    ]


def _reconstruct(
    *,
    storage: MemoryStorage,
    source_index: dict[str, SourceRecord],
    umo: str,
    case: str,
    phase: str,
    query: str,
    cutoff_at: int,
    client: Any,
    provider_id: str,
    model: str,
    extra_body: dict[str, Any],
    ledger: UsageLedger,
    max_steps: int = 8,
) -> dict[str, Any]:
    catalog = _edge_catalog(
        storage,
        umo=umo,
        before_sent_at=cutoff_at,
        include_evidence=False,
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": RECONSTRUCTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "query": query,
                    "cutoff_at": cutoff_at,
                    "embedding_prior_candidates": catalog,
                    "instruction": (
                        "Resolve the group-local noun/phrase. Traverse with a short "
                        "cue, inspect sources, then return a compact brief."
                    ),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]
    exposed_edge_ids: set[int] = {int(row["id"]) for row in catalog}
    exposed_source_keys: set[str] = set()
    visited_source_keys: set[str] = set()
    tool_log: list[dict[str, Any]] = []
    for step in range(1, max_steps + 1):
        completion, usage, elapsed_ms = _chat_completion(
            client=client,
            model=model,
            messages=messages,
            extra_body=extra_body,
            tools=RECONSTRUCTION_TOOLS,
            max_output_tokens=3000,
        )
        ledger.add(
            case=case,
            phase=f"reconstruct:{phase}",
            provider_id=provider_id,
            model=model,
            usage=usage,
            elapsed_ms=elapsed_ms,
            call_index=step,
        )
        response = completion.choices[0].message
        tool_calls = list(getattr(response, "tool_calls", None) or [])
        if not tool_calls:
            messages.append(_message_payload(response))
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Host validation requires a tool call. Continue with the "
                        "needed traversal/evidence tools and finish only by calling "
                        "mr_return_brief."
                    ),
                }
            )
            tool_log.append(
                {
                    "step": step,
                    "tool": "host_validation",
                    "error": "assistant returned without a tool call",
                }
            )
            continue
        messages.append(_message_payload(response))
        for call in tool_calls:
            name = str(call.function.name)
            arguments = _json_object(str(call.function.arguments or "{}"))
            if name == "mr_query_associations":
                rows = storage.query_plastic_associations(
                    umo=umo,
                    query=str(arguments.get("query") or ""),
                    node_key=str(arguments.get("node_key") or ""),
                    relation_key=str(arguments.get("relation_key") or ""),
                    direction=str(arguments.get("direction") or "both"),
                    include_dormant=bool(arguments.get("include_dormant", False)),
                    limit=int(arguments.get("limit") or 20),
                    before_sent_at=cutoff_at,
                )
                exposed_edge_ids.update(int(row["id"]) for row in rows)
                for row in rows:
                    exposed_source_keys.update(str(key) for key in row["source_keys"])
                result: Any = _association_tool_result(rows)
            elif name == "mr_read_evidence":
                requested = [
                    str(item)
                    for item in (arguments.get("source_keys") or [])
                    if str(item) in exposed_source_keys
                ][:16]
                result = _read_source_records(source_index, requested)
                visited_source_keys.update(item["source_key"] for item in result)
            elif name == "mr_return_brief":
                edge_ids = tuple(
                    dict.fromkeys(int(item) for item in arguments["selected_edge_ids"])
                )
                source_keys = tuple(
                    dict.fromkeys(str(item) for item in arguments["source_keys"])
                )
                validation_error = ""
                if not set(edge_ids).issubset(exposed_edge_ids):
                    validation_error = "selected_edge_ids contains an unexposed edge"
                elif not set(source_keys).issubset(visited_source_keys):
                    validation_error = (
                        "source_keys contains evidence not read in this run; call "
                        "mr_read_evidence first or cite only visited keys"
                    )
                if validation_error:
                    result = {
                        "error": validation_error,
                        "visited_source_keys": sorted(visited_source_keys),
                        "exposed_edge_ids": sorted(exposed_edge_ids),
                    }
                    tool_log.append(
                        {
                            "step": step,
                            "tool": name,
                            "arguments": arguments,
                            "error": validation_error,
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": str(call.id),
                            "content": json.dumps(
                                result,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        }
                    )
                    continue
                storage.activate_plastic_edges(
                    umo=umo,
                    edge_ids=edge_ids,
                    at=cutoff_at,
                    relevance=1.0,
                )
                brief = {
                    "summary": str(arguments["summary"])[:5000],
                    "selected_edge_ids": list(edge_ids),
                    "source_keys": list(source_keys),
                    "uncertainties": [
                        str(item)[:500]
                        for item in arguments.get("uncertainties", [])[:8]
                    ],
                }
                tool_log.append(
                    {"step": step, "tool": name, "arguments": arguments}
                )
                return {
                    "phase": phase,
                    "brief": brief,
                    "tool_log": tool_log,
                    "candidate_edge_ids": sorted(exposed_edge_ids),
                    "visited_source_keys": sorted(visited_source_keys),
                }
            else:
                raise ValueError(f"unknown reconstruction tool: {name}")
            tool_log.append(
                {
                    "step": step,
                    "tool": name,
                    "arguments": arguments,
                    "result_count": len(result) if isinstance(result, list) else 1,
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(call.id),
                    "content": json.dumps(
                        result, ensure_ascii=False, separators=(",", ":"), default=str
                    ),
                }
            )
    raise ValueError(f"{case}/{phase}: reconstruction exceeded {max_steps} steps")


def _surface_answer(
    *,
    case: str,
    phase: str,
    query: str,
    brief: dict[str, Any] | None,
    client: Any,
    provider_id: str,
    model: str,
    extra_body: dict[str, Any],
    ledger: UsageLedger,
) -> str:
    payload = {"当前消息": query}
    if brief is not None:
        payload["群内记忆摘要"] = brief
    completion, usage, elapsed_ms = _chat_completion(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": SURFACE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            },
        ],
        extra_body=extra_body,
        max_output_tokens=1200,
    )
    ledger.add(
        case=case,
        phase=f"surface:{phase}",
        provider_id=provider_id,
        model=model,
        usage=usage,
        elapsed_ms=elapsed_ms,
    )
    return str(completion.choices[0].message.content or "").strip()


def _evaluate(
    *,
    case: str,
    rubric: list[str],
    evidence: Sequence[SourceRecord],
    arms: list[dict[str, Any]],
    client: Any,
    provider_id: str,
    model: str,
    extra_body: dict[str, Any],
    ledger: UsageLedger,
) -> dict[str, Any]:
    payload = {
        "case": case,
        "rubric": rubric,
        "evidence": [record.evidence_dict(max_chars=700) for record in evidence],
        "answers": [
            {"arm": arm["arm"], "answer": arm["answer"]} for arm in arms
        ],
        "required_output": {
            "arms": [
                {
                    "arm": "exact arm label",
                    "score": "integer 0..5",
                    "local_meaning_resolved": "boolean",
                    "unsupported_homonym": "boolean",
                    "grounded_points": ["short evidence-grounded points"],
                    "failures": ["short failures"],
                    "verdict": "one sentence",
                }
            ]
        },
    }
    completion, usage, elapsed_ms = _chat_completion(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            },
        ],
        extra_body=extra_body,
        max_output_tokens=3000,
        json_object=True,
    )
    ledger.add(
        case=case,
        phase="evaluation",
        provider_id=provider_id,
        model=model,
        usage=usage,
        elapsed_ms=elapsed_ms,
    )
    return _json_object(str(completion.choices[0].message.content or ""))


def _evaluation_index(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = value.get("arms")
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("arm")): row
        for row in rows
        if isinstance(row, dict) and row.get("arm")
    }


def _graph_snapshot(storage: MemoryStorage, *, umo: str) -> dict[str, Any]:
    associations = _edge_catalog(storage, umo=umo, include_evidence=True)
    nodes: dict[str, dict[str, Any]] = {}
    for edge in associations:
        nodes[str(edge["source_key"])] = {
            "key": str(edge["source_key"]),
            "kind": str(edge["source_kind"]),
            "label": str(edge["source_label"]),
        }
        nodes[str(edge["target_key"])] = {
            "key": str(edge["target_key"]),
            "kind": str(edge["target_kind"]),
            "label": str(edge["target_label"]),
        }
    return {"nodes": list(nodes.values()), "edges": associations}


def _matrix_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in ("good_girl", "arale"):
        evaluation = _evaluation_index(result[case]["evaluation"])
        for arm in result[case]["arms"]:
            score = evaluation.get(str(arm["arm"]), {})
            rows.append(
                {
                    "case": case,
                    "arm": arm["arm"],
                    "graph_edges": arm.get("graph_edges", 0),
                    "selected_edges": len(
                        (arm.get("reconstruction") or {})
                        .get("brief", {})
                        .get("selected_edge_ids", [])
                    ),
                    "visited_sources": len(
                        (arm.get("reconstruction") or {}).get(
                            "visited_source_keys", []
                        )
                    ),
                    "score": score.get("score"),
                    "resolved": score.get("local_meaning_resolved"),
                    "unsupported_homonym": score.get("unsupported_homonym"),
                    "verdict": score.get("verdict", ""),
                    "answer": arm["answer"],
                }
            )
    return rows


def _svg_graph(graph: dict[str, Any], *, title: str) -> str:
    nodes = list(graph.get("nodes") or [])
    edges = list(graph.get("edges") or [])
    width, height = 880, max(480, 100 + len(nodes) * 18)
    cx, cy = width / 2, height / 2
    radius = min(width, height) * 0.34
    positions: dict[str, tuple[float, float]] = {}
    for index, node in enumerate(nodes):
        angle = -math.pi / 2 + (2 * math.pi * index / max(1, len(nodes)))
        positions[str(node["key"])] = (
            cx + radius * math.cos(angle),
            cy + radius * math.sin(angle),
        )
    output = [
        f'<h3>{html.escape(title)}</h3>',
        f'<svg class="graph" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(title)}">',
        "<defs><marker id=\"arrow\" markerWidth=\"8\" markerHeight=\"8\" "
        "refX=\"7\" refY=\"3\" orient=\"auto\"><path d=\"M0,0 L0,6 L8,3 z\" "
        "fill=\"#74829b\"/></marker></defs>",
    ]
    for edge in edges:
        source = positions.get(str(edge["source_key"]))
        target = positions.get(str(edge["target_key"]))
        if source is None or target is None:
            continue
        x1, y1 = source
        x2, y2 = target
        output.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" '
            f'y2="{y2:.1f}" marker-end="url(#arrow)"/>'
        )
        output.append(
            f'<text class="edge-label" x="{(x1+x2)/2:.1f}" '
            f'y="{(y1+y2)/2:.1f}">{html.escape(str(edge["relation_name"]))}</text>'
        )
    colors = {
        "concept": "#5ba7ff",
        "behavior": "#f0a35a",
        "symbol": "#ad82f2",
        "topic": "#4cc7a5",
        "preference": "#ef718a",
        "procedure": "#d4b84f",
    }
    for node in nodes:
        x, y = positions[str(node["key"])]
        label = str(node["label"])
        color = colors.get(str(node["kind"]), "#8793a8")
        output.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="25" fill="{color}"/>'
        )
        output.append(
            f'<text class="node-label" x="{x:.1f}" y="{y + 40:.1f}">'
            f'{html.escape(label[:24])}</text>'
        )
    output.append("</svg>")
    return "".join(output)


def _write_reports(output_dir: Path, result: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    matrix = _matrix_rows(result)
    markdown = [
        "# 群聊局部语义与可塑图记忆 A/B",
        "",
        f"- 最大群：`{result['dataset']['group_id']}`",
        f"- 历史消息：{result['dataset']['messages']}",
        f"- 参与账户：{result['dataset']['participants']}",
        f"- 总调用 token：{result['usage']['totals']['total']}",
        "",
        "| 案例 | Arm | 图边 | 选中边 | 已读证据 | 评分 | 识别局部语义 | 外部同名幻觉 | 结论 |",
        "|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    if result.get("attempt_notes"):
        markdown[6:6] = [
            "- 运行备注：" + "；".join(str(item) for item in result["attempt_notes"]),
            "",
        ]
    for row in matrix:
        markdown.append(
            "| {case} | {arm} | {graph_edges} | {selected_edges} | "
            "{visited_sources} | {score} | {resolved} | {unsupported_homonym} | {verdict} |".format(
                **{key: str(value).replace("|", "\\|") for key, value in row.items()}
            )
        )
    policy = result.get("negative_credit_policy_replay")
    if isinstance(policy, dict):
        markdown.extend(
            [
                "",
                "## 反馈归因修复复验",
                "",
                f"- 原始反馈提案：{policy.get('proposed', 0)}",
                f"- 修复后提交：{policy.get('committed', 0)}",
                f"- 因未参与错误回答而拒绝：{policy.get('rejected', 0)}",
                f"- 正确主路径效用：{policy.get('utility_before_feedback')} → {policy.get('utility_after_feedback')}",
                f"- 宿主结论：{policy.get('verdict', '')}",
                "- 运行时写门：对应反馈 proposal 必须先达到 `COMMITTED`，随后同一维护轮才可修改可塑图。",
            ]
        )
    markdown.extend(["", "## 实际回答", ""])
    for row in matrix:
        markdown.extend(
            [
                f"### {row['case']} / {row['arm']}",
                "",
                str(row["answer"]),
                "",
            ]
        )
    (output_dir / "REPORT.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )

    rows_html = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(row[column]))}</td>"
            for column in (
                "case",
                "arm",
                "graph_edges",
                "selected_edges",
                "visited_sources",
                "score",
                "resolved",
                "unsupported_homonym",
                "verdict",
            )
        )
        + "</tr>"
        for row in matrix
    )
    answers_html = "".join(
        f"<article><h3>{html.escape(str(row['case']))} / "
        f"{html.escape(str(row['arm']))}</h3><pre>{html.escape(str(row['answer']))}</pre></article>"
        for row in matrix
    )
    usage_rows = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(row.get(column, '')))}</td>"
            for column in (
                "case",
                "phase",
                "call_index",
                "model",
                "input",
                "output",
                "total",
                "elapsed_ms",
            )
        )
        + "</tr>"
        for row in result["usage"]["calls"]
    )
    graph_html = _svg_graph(
        result["good_girl"]["final_graph"], title="好女孩：最终可塑子图"
    ) + _svg_graph(result["arale"]["final_graph"], title="阿拉蕾：反馈后可塑子图")
    note_html = "".join(
        f"<li>{html.escape(str(item))}</li>"
        for item in result.get("attempt_notes", [])
    )
    if note_html:
        note_html = f'<div class="panel"><strong>运行备注</strong><ul>{note_html}</ul></div>'
    policy_html = ""
    policy = result.get("negative_credit_policy_replay")
    if isinstance(policy, dict):
        policy_html = (
            '<div class="panel"><strong>反馈归因修复复验</strong><p>'
            f"原始提案 {html.escape(str(policy.get('proposed', 0)))} 条；"
            f"修复后提交 {html.escape(str(policy.get('committed', 0)))} 条，拒绝 "
            f"{html.escape(str(policy.get('rejected', 0)))} 条未参与错误回答的负向修改。"
            f"正确主路径效用 {html.escape(str(policy.get('utility_before_feedback')))} → "
            f"{html.escape(str(policy.get('utility_after_feedback')))}。"
            f"{html.escape(str(policy.get('verdict', '')))}"
            "运行时还要求对应反馈 proposal 先达到 COMMITTED，随后同一维护轮才可修改可塑图。"
            "</p></div>"
        )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MR Memory 局部语义 A/B</title><style>
:root{{--bg:#0d1117;--card:#151b24;--ink:#edf2f7;--muted:#9aa8ba;--line:#2b3545;--accent:#65b5ff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1240px;margin:auto;padding:32px}}h1,h2,h3{{line-height:1.2}}h1{{font-size:30px}}h2{{margin-top:38px;color:#b9dbff}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}}.card,article,.panel{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px}}
.metric{{font-size:28px;font-weight:750;color:var(--accent)}}.label{{color:var(--muted)}}.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:14px}}
table{{border-collapse:collapse;width:100%;background:var(--card)}}th,td{{padding:10px 12px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left}}th{{position:sticky;top:0;background:#202938}}
pre{{white-space:pre-wrap;word-break:break-word;color:#dce9f8;font:14px/1.6 system-ui,"Microsoft YaHei",sans-serif}}article{{margin:12px 0}}
.graph{{display:block;width:100%;max-height:720px;background:#101722;border:1px solid var(--line);border-radius:14px;margin:10px 0 28px}}.graph line{{stroke:#74829b;stroke-width:1.4;opacity:.75}}
.node-label{{fill:#e8f1ff;text-anchor:middle;font-size:13px;paint-order:stroke;stroke:#0d1117;stroke-width:4px}}.edge-label{{fill:#aab8ca;text-anchor:middle;font-size:11px;paint-order:stroke;stroke:#101722;stroke-width:4px}}
</style></head><body><main>
<h1>MR Memory 群聊局部语义 A/B</h1>
{note_html}
<div class="cards"><div class="card"><div class="metric">{result['dataset']['messages']}</div><div class="label">最大群真实消息</div></div>
<div class="card"><div class="metric">{result['dataset']['participants']}</div><div class="label">账户主体</div></div>
<div class="card"><div class="metric">{result['usage']['totals']['calls']}</div><div class="label">模型调用</div></div>
<div class="card"><div class="metric">{result['usage']['totals']['total']}</div><div class="label">总 token</div></div></div>
<h2>结果矩阵</h2><div class="table-wrap"><table><thead><tr><th>案例</th><th>Arm</th><th>图边</th><th>选中边</th><th>已读证据</th><th>评分</th><th>局部语义</th><th>同名幻觉</th><th>结论</th></tr></thead><tbody>{rows_html}</tbody></table></div>
{policy_html}
<h2>实际回答</h2>{answers_html}<h2>最终图</h2>{graph_html}
<h2>Token / 延迟账本</h2><div class="table-wrap"><table><thead><tr><th>案例</th><th>阶段</th><th>轮次</th><th>模型</th><th>输入</th><th>输出</th><th>合计</th><th>ms</th></tr></thead><tbody>{usage_rows}</tbody></table></div>
</main></body></html>"""
    (output_dir / "report.html").write_text(document, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_dir / "experiment.db"
    if database_path.exists() and not args.resume:
        database_path.unlink()
    group_id, history, dataset_stats = _load_history(
        args.history_db, group_id=args.group_id
    )
    umo = f"aiocqhttp:GroupMessage:{group_id}"
    live = _load_live_fixture(args.live_fixture, group_id=group_id, umo=umo)
    target_cutoff = live["target"].normalized.sent_at
    if any(
        record.normalized.group_id != group_id for record in live.values()
    ):
        raise ValueError("live fixture crosses the selected group")

    storage = MemoryStorage(database_path)
    storage.bind_scope(umo=umo, platform_id="byy_official", group_id=group_id)
    inserted = _ingest(storage, history)
    source_index = {record.source_key: record for record in history}

    subconscious_client, subconscious_model, subconscious_extra = _provider_config(
        args.config, args.subconscious_provider_id
    )
    subconscious_extra = {
        **subconscious_extra,
        "thinking": {"type": "disabled"},
    }
    surface_client, surface_model, surface_extra = _provider_config(
        args.config, args.surface_provider_id
    )
    live_usage_path = output_dir / "usage_live.jsonl"
    if live_usage_path.exists() and not args.resume:
        live_usage_path.unlink()
    ledger = UsageLedger(live_usage_path)

    good_hits = _term_hits(history, args.good_term)
    if not good_hits:
        raise ValueError(f"no history evidence for {args.good_term!r}")
    cutoffs = [
        int(item)
        for item in str(args.good_phase_cutoffs or "").split(",")
        if item.strip()
    ]
    if cutoffs:
        good_phases = _split_by_cutoffs(good_hits, cutoffs)
    else:
        clusters = _split_clusters(good_hits, gap_seconds=3 * 86400)
        good_phases = clusters[:3]
        if len(clusters) > 3:
            good_phases.append([item for group in clusters[3:] for item in group])
    good_query = f"在这个群里，大家说“{args.good_term}”到底是什么意思？"
    good_arms: list[dict[str, Any]] = [
        {
            "arm": "A0 无群记忆",
            "answer": _surface_answer(
                case="good_girl",
                phase="control",
                query=good_query,
                brief=None,
                client=surface_client,
                provider_id=args.surface_provider_id,
                model=surface_model,
                extra_body=surface_extra,
                ledger=ledger,
            ),
            "graph_edges": 0,
            "reconstruction": None,
        }
    ]
    good_ticks: list[dict[str, Any]] = []
    for index, phase_evidence in enumerate(good_phases, start=1):
        phase_name = f"P{index}"
        tick = _maintenance_tick(
            storage=storage,
            umo=umo,
            case="good_girl",
            phase=phase_name,
            objective=(
                f"Incrementally learn how the expression {args.good_term!r} is used "
                "in this group, including origin, pragmatic force, changing scope, "
                "contradictions, and joke/approval behavior."
            ),
            evidence=phase_evidence,
            client=subconscious_client,
            provider_id=args.subconscious_provider_id,
            model=subconscious_model,
            extra_body=subconscious_extra,
            ledger=ledger,
        )
        good_ticks.append(tick)
        reconstruction = _reconstruct(
            storage=storage,
            source_index=source_index,
            umo=umo,
            case="good_girl",
            phase=phase_name,
            query=good_query,
            cutoff_at=max(item.normalized.sent_at for item in phase_evidence) + 1,
            client=subconscious_client,
            provider_id=args.subconscious_provider_id,
            model=subconscious_model,
            extra_body=subconscious_extra,
            ledger=ledger,
        )
        answer = _surface_answer(
            case="good_girl",
            phase=phase_name,
            query=good_query,
            brief=reconstruction["brief"],
            client=surface_client,
            provider_id=args.surface_provider_id,
            model=surface_model,
            extra_body=surface_extra,
            ledger=ledger,
        )
        good_arms.append(
            {
                "arm": f"B{index} 图记忆 {phase_name}",
                "answer": answer,
                "graph_edges": len(tick["graph"]),
                "reconstruction": reconstruction,
            }
        )
    good_final_graph = _graph_snapshot(storage, umo=umo)

    arale_hits = _term_hits(
        history, args.arale_term, before_sent_at=target_cutoff
    )
    if not arale_hits:
        raise ValueError(f"no pre-query evidence for {args.arale_term!r}")
    arale_clusters = _split_clusters(arale_hits, gap_seconds=3 * 86400)
    arale_ticks: list[dict[str, Any]] = []
    for index, cluster in enumerate(arale_clusters, start=1):
        evidence = _context_window(
            history,
            cluster,
            radius=18,
            before_sent_at=target_cutoff,
            max_records=60,
        )
        tick = _maintenance_tick(
            storage=storage,
            umo=umo,
            case="arale",
            phase=f"H{index}",
            objective=(
                f"Resolve the group-local referent and useful associations of the "
                f"ambiguous noun {args.arale_term!r}; preserve evidence-based traits "
                "and links to neighboring group vocabulary."
            ),
            evidence=evidence,
            client=subconscious_client,
            provider_id=args.subconscious_provider_id,
            model=subconscious_model,
            extra_body=subconscious_extra,
            ledger=ledger,
        )
        arale_ticks.append(tick)
    arale_query = live["target"].normalized.plain_text
    pre_feedback_reconstruction = _reconstruct(
        storage=storage,
        source_index=source_index,
        umo=umo,
        case="arale",
        phase="pre_feedback",
        query=arale_query,
        cutoff_at=target_cutoff,
        client=subconscious_client,
        provider_id=args.subconscious_provider_id,
        model=subconscious_model,
        extra_body=subconscious_extra,
        ledger=ledger,
    )
    pre_feedback_answer = _surface_answer(
        case="arale",
        phase="pre_feedback",
        query=arale_query,
        brief=pre_feedback_reconstruction["brief"],
        client=surface_client,
        provider_id=args.surface_provider_id,
        model=surface_model,
        extra_body=surface_extra,
        ledger=ledger,
    )
    arale_arms: list[dict[str, Any]] = [
        {
            "arm": "A0 线上真实旧系统",
            "answer": live["baseline_response"].normalized.plain_text,
            "graph_edges": 0,
            "reconstruction": None,
        },
        {
            "arm": "B1 查询前图记忆",
            "answer": pre_feedback_answer,
            "graph_edges": len(_edge_catalog(storage, umo=umo)),
            "reconstruction": pre_feedback_reconstruction,
        },
    ]

    _ingest(storage, live.values())
    source_index.update({record.source_key: record for record in live.values()})
    feedback_tick = _maintenance_tick(
        storage=storage,
        umo=umo,
        case="arale",
        phase="F1_observed_negative_feedback",
        objective=(
            "Assign credit for the failed response: the member's immediate follow-up "
            "shows that the answer missed the group-local noun and its neighboring "
            "vocabulary. Reinforce grounded local disambiguation paths and inhibit "
            "unsupported famous-homonym behavior without rewriting raw evidence."
        ),
        evidence=[live["target"], live["baseline_response"], live["feedback"]],
        client=subconscious_client,
        provider_id=args.subconscious_provider_id,
        model=subconscious_model,
        extra_body=subconscious_extra,
        ledger=ledger,
    )
    arale_ticks.append(feedback_tick)
    after_feedback_reconstruction = _reconstruct(
        storage=storage,
        source_index=source_index,
        umo=umo,
        case="arale",
        phase="after_feedback",
        query=arale_query,
        cutoff_at=live["feedback"].normalized.sent_at + 1,
        client=subconscious_client,
        provider_id=args.subconscious_provider_id,
        model=subconscious_model,
        extra_body=subconscious_extra,
        ledger=ledger,
    )
    after_feedback_answer = _surface_answer(
        case="arale",
        phase="after_feedback",
        query=arale_query,
        brief=after_feedback_reconstruction["brief"],
        client=surface_client,
        provider_id=args.surface_provider_id,
        model=surface_model,
        extra_body=surface_extra,
        ledger=ledger,
    )
    arale_arms.append(
        {
            "arm": "B2 负反馈写图后",
            "answer": after_feedback_answer,
            "graph_edges": len(_edge_catalog(storage, umo=umo)),
            "reconstruction": after_feedback_reconstruction,
        }
    )

    good_evaluation = _evaluate(
        case="good_girl",
        rubric=[
            "Explain the group's local pragmatic meaning, not merely literal gender or morality.",
            "Recognize playful/ironic approval or certification usage.",
            "Ground the expression's 若叶睦 meme connection when supported.",
            "Recognize that later usage generalizes across people, bot, cat, food, and behavior.",
            "State uncertainty instead of inventing a rigid universal definition.",
        ],
        evidence=good_hits,
        arms=good_arms,
        client=subconscious_client,
        provider_id=args.subconscious_provider_id,
        model=subconscious_model,
        extra_body=subconscious_extra,
        ledger=ledger,
    )
    arale_evidence = _context_window(
        history,
        arale_hits,
        radius=18,
        before_sent_at=target_cutoff,
        max_records=60,
    )
    arale_evaluation = _evaluate(
        case="arale",
        rubric=[
            "Resolve 阿拉蕾 as the group-local 梦限大-related referent.",
            "Use grounded traits such as noisy/cute, very fast speech, and awkwardness where supported.",
            "Interpret 挺点 cautiously as likely asking for appealing/萌 points rather than inventing a new term.",
            "Do not substitute Digimon, Dr. Slump, or another famous homonym.",
            "Keep claims traceable to pre-query group evidence.",
        ],
        evidence=arale_evidence,
        arms=arale_arms,
        client=subconscious_client,
        provider_id=args.subconscious_provider_id,
        model=subconscious_model,
        extra_body=subconscious_extra,
        ledger=ledger,
    )

    result = {
        "run_id": f"plastic-semantics-{int(time.time())}",
        "generated_at": int(time.time()),
        "dataset": {
            "group_id": group_id,
            "umo": umo,
            **dataset_stats,
            "inserted_into_isolated_store": inserted,
            "good_term_hits": len(good_hits),
            "arale_term_hits_before_query": len(arale_hits),
            "target_cutoff": target_cutoff,
        },
        "providers": {
            "subconscious": {
                "provider_id": args.subconscious_provider_id,
                "model": subconscious_model,
            },
            "surface": {
                "provider_id": args.surface_provider_id,
                "model": surface_model,
            },
        },
        "attempt_notes": list(args.attempt_note or []),
        "good_girl": {
            "query": good_query,
            "maintenance_ticks": good_ticks,
            "arms": good_arms,
            "evaluation": good_evaluation,
            "final_graph": good_final_graph,
        },
        "arale": {
            "query": arale_query,
            "observed_feedback": live["feedback"].normalized.plain_text,
            "maintenance_ticks": arale_ticks,
            "arms": arale_arms,
            "evaluation": arale_evaluation,
            "final_graph": _graph_snapshot(storage, umo=umo),
        },
        "usage": {"calls": ledger.rows, "totals": ledger.totals()},
    }
    _write_reports(output_dir, result)
    storage.close()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run chronological plastic-graph semantic A/B experiments."
    )
    parser.add_argument("--history-db", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--live-fixture", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-id", default="")
    parser.add_argument("--good-term", default="好女孩")
    parser.add_argument("--arale-term", default="阿拉蕾")
    parser.add_argument("--good-phase-cutoffs", default="")
    parser.add_argument(
        "--subconscious-provider-id", default="deepseek/deepseek-v4-flash"
    )
    parser.add_argument(
        "--surface-provider-id", default="openai/gemini-3.5-flash"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--attempt-note", action="append", default=[])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "dataset": result["dataset"],
                "matrix": _matrix_rows(result),
                "usage": result["usage"]["totals"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
