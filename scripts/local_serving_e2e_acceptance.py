from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Mapping

from mr_memory.local_serving import compile_local_serving_envelope
from mr_memory.runtime import materialize_reconstruction_packet


REPORT_SCHEMA_VERSION = "mr-memory.local-serving-component-acceptance.v2"
EXECUTION_SCOPE = "OFFLINE_PIPELINE_COMPONENT_ACCEPTANCE"

def read_usage_ledger(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"usage ledger line {line_number} is not an object")
        rows.append(value)
    return rows


def provider_turn_report(
    rows: Iterable[Mapping[str, Any]],
    *,
    run_id: str,
    expect_zero_calls: bool = False,
    non_llm_run_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a closed, per-call report from the existing Provider ledger.

    The ledger remains the authority.  Missing terminal rows or duplicate events are
    errors, rather than being converted into zero usage or a successful call.
    """

    selected = [
        (line_number, dict(row))
        for line_number, row in enumerate(rows, start=1)
        if str(row.get("run_id") or "") == run_id
    ]
    if not selected:
        if not expect_zero_calls:
            raise ValueError(f"provider ledger has no rows for run_id: {run_id}")
        evidence = dict(non_llm_run_evidence or {})
        if not evidence or str(evidence.get("run_id") or "") != run_id:
            raise ValueError(
                "zero-call report requires matching non_llm_run_evidence"
            )
        if (
            evidence.get("operational_status") != "COMPLETED"
            or evidence.get("path") != "materialized_local"
            or evidence.get("memory_provider_calls") != 0
        ):
            raise ValueError(
                "zero-call evidence must be a completed materialized_local run "
                "with memory_provider_calls=0"
            )
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "execution_scope": EXECUTION_SCOPE,
            "run_id": run_id,
            "pairwise_closed": True,
            "expected_zero_calls": True,
            "non_llm_run_evidence": evidence,
            "attempted_calls": 0,
            "completed_calls": 0,
            "failed_calls": 0,
            "usage_complete": True,
            "measured_completed_input_other_lower_bound": 0,
            "measured_completed_input_cached_lower_bound": 0,
            "measured_completed_input_lower_bound": 0,
            "measured_completed_output_lower_bound": 0,
            "measured_completed_total_lower_bound": 0,
            "elapsed_ms_sum": 0.0,
            "currency_cost": "NO_PROVIDER_CALLS_CONFIRMED_BY_NON_LLM_RUN_EVIDENCE",
            "turns": [],
        }
    if expect_zero_calls:
        raise ValueError(f"expected zero calls but ledger contains rows: {run_id}")

    by_request: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for line_number, row in selected:
        request_id = str(row.get("request_id") or "")
        if not request_id:
            raise ValueError("provider ledger row is missing request_id")
        by_request.setdefault(request_id, []).append((line_number, row))

    turns: list[dict[str, Any]] = []
    for request_id, located_rows in by_request.items():
        request_rows = [row for _, row in located_rows]
        attempted = [
            (line, row)
            for line, row in located_rows
            if row.get("event") == "attempted"
        ]
        terminal = [
            (line, row)
            for line, row in located_rows
            if row.get("event") in {"completed", "failed"}
        ]
        if len(attempted) != 1 or len(terminal) != 1 or len(request_rows) != 2:
            raise ValueError(f"provider ledger request is not pairwise closed: {request_id}")
        attempted_line, start = attempted[0]
        terminal_line, end = terminal[0]
        if attempted_line >= terminal_line:
            raise ValueError(
                f"provider ledger terminal precedes attempted: {request_id}"
            )
        identity_fields = (
            "run_id",
            "arm",
            "repetition",
            "phase",
            "call_index",
            "provider_id",
            "model",
            "thinking",
            "max_tokens",
            "options_sha256",
            "payload_sha256",
        )
        for field in identity_fields:
            if start.get(field) != end.get(field):
                raise ValueError(
                    f"provider ledger identity changed within {request_id}: {field}"
                )
        expected_request_id = (
            f"{run_id}:{str(start.get('phase') or '')}:"
            f"{int(start.get('call_index') or 0)}"
        )
        if request_id != expected_request_id:
            raise ValueError(
                f"provider ledger request_id is noncanonical: {request_id}"
            )
        status = str(end["event"]).upper()
        turn = {
            "request_id": request_id,
            "attempted_line": attempted_line,
            "terminal_line": terminal_line,
            **{field: start.get(field) for field in identity_fields},
            "status": status,
            "elapsed_ms": float(end.get("elapsed_ms") or 0.0),
            "usage_present": end.get("usage_present") if status == "COMPLETED" else None,
            "input_other": int(end.get("input_other") or 0),
            "input_cached": int(end.get("input_cached") or 0),
            "input": int(end.get("input") or 0),
            "output": int(end.get("output") or 0),
            "total": int(end.get("total") or 0),
            "error_type": str(end.get("error_type") or ""),
            "error_detail": str(end.get("error_detail") or ""),
            "currency_cost": "UNKNOWN_PROVIDER_PRICING_NOT_IN_LEDGER",
        }
        if status == "COMPLETED":
            if turn["usage_present"] is not True:
                raise ValueError(f"completed provider turn has no usage: {request_id}")
            if turn["input"] != turn["input_other"] + turn["input_cached"]:
                raise ValueError(f"provider input arithmetic mismatch: {request_id}")
            if turn["total"] != turn["input"] + turn["output"]:
                raise ValueError(f"provider total arithmetic mismatch: {request_id}")
        turns.append(turn)

    turns.sort(key=lambda item: (int(item.get("call_index") or 0), item["request_id"]))
    call_indexes = [int(turn.get("call_index") or 0) for turn in turns]
    if len(call_indexes) != len(set(call_indexes)):
        raise ValueError("provider ledger call_index values are not unique")
    if call_indexes != list(range(len(call_indexes))):
        raise ValueError("provider ledger call_index values are not contiguous from zero")
    completed = [turn for turn in turns if turn["status"] == "COMPLETED"]
    failed_calls = len(turns) - len(completed)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "execution_scope": EXECUTION_SCOPE,
        "run_id": run_id,
        "pairwise_closed": True,
        "attempted_calls": len(turns),
        "completed_calls": len(completed),
        "failed_calls": failed_calls,
        "usage_complete": failed_calls == 0,
        "measured_completed_input_other_lower_bound": sum(
            turn["input_other"] for turn in completed
        ),
        "measured_completed_input_cached_lower_bound": sum(
            turn["input_cached"] for turn in completed
        ),
        "measured_completed_input_lower_bound": sum(
            turn["input"] for turn in completed
        ),
        "measured_completed_output_lower_bound": sum(
            turn["output"] for turn in completed
        ),
        "measured_completed_total_lower_bound": sum(
            turn["total"] for turn in completed
        ),
        "elapsed_ms_sum": round(sum(turn["elapsed_ms"] for turn in turns), 3),
        "currency_cost": "UNKNOWN_PROVIDER_PRICING_NOT_IN_LEDGER",
        "turns": turns,
    }


async def run_local_serving_case(
    *,
    query: str,
    normalized: Any,
    snapshot: Any,
    service: Any,
    classify_request: Callable[..., str],
    activity_analysis: Callable[[str], bool],
    identity_question: Callable[[str], bool],
    reference_question: Callable[[str], bool],
    build_identity_packet: Callable[..., Awaitable[tuple[dict[str, object], str, set[str], set[str], str]]],
    build_full_packet: Callable[..., Awaitable[tuple[dict[str, object], str, set[str], set[str], str]]],
    max_items: int = 12,
    chat_max_chars: int = 3000,
    memory_query_max_chars: int = 12000,
) -> dict[str, Any]:
    """Exercise real leaf functions from natural text to a component envelope.

    This helper deliberately remains a component runner: it mirrors routing so its
    output can be inspected without AstrBot imports, but it does not prove the
    production top-level orchestration.  The separate integration test executes
    ``_execute_local_memory_serving`` itself for that contract.
    """

    content = getattr(normalized, "content", None) or []
    has_structured_reference = any(
        isinstance(item, Mapping) and item.get("type") in {"reply", "mention"}
        for item in content
    )
    request_kind = classify_request(
        query,
        force=False,
        has_structured_reference=has_structured_reference,
    )
    include_activity = bool(activity_analysis(query))
    identity_match = bool(
        identity_question(
            query,
            has_structured_reference=has_structured_reference,
        )
    )
    reference_match = bool(reference_question(query))
    direct_candidate = include_activity or identity_match or (
        reference_match and has_structured_reference
    )
    if direct_candidate:
        direct = await build_identity_packet(
            service=service,
            snapshot=snapshot,
            normalized=normalized,
            query=query,
            include_participant_activity=include_activity,
        )
        resolution = direct[0].get("query_alias_resolution")
        resolved = resolution.get("participants") if isinstance(resolution, dict) else []
        ambiguous = bool(resolution.get("ambiguous")) if isinstance(resolution, dict) else False
        classified_mentions = (
            resolution.get("mentions")
            if isinstance(resolution, dict)
            and isinstance(resolution.get("mentions"), list)
            else []
        )
        use_direct = include_activity or bool(
            (identity_match or reference_match)
            and (
                resolved
                or ambiguous
                or classified_mentions
                or has_structured_reference
            )
        )
    else:
        direct = None
        use_direct = False
    packet_result = direct if use_direct else await build_full_packet(
        service=service,
        snapshot=snapshot,
        normalized=normalized,
        query=query,
    )
    packet = packet_result[0]
    materialized = materialize_reconstruction_packet(
        packet, query=query, max_items=max_items
    )
    envelope = compile_local_serving_envelope(
        packet,
        materialized,
        request_kind=request_kind,
        max_chars=(chat_max_chars if request_kind == "CHAT" else memory_query_max_chars),
    )
    envelope_value = json.loads(envelope.json_text)
    resolution = packet.get("query_alias_resolution")
    participants = resolution.get("participants") if isinstance(resolution, dict) else []
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "execution_scope": EXECUTION_SCOPE,
        "status": "COMPLETED",
        "query": query,
        "request_kind": request_kind,
        "route": "LOCAL_DIRECT" if use_direct else "FULL_RETRIEVAL",
        "resolved_participant_keys": [
            str(item.get("canonical_key") or "")
            for item in participants or []
            if isinstance(item, Mapping)
        ],
        "resolved_account_ids": [
            str(item.get("account_id") or "")
            for item in participants or []
            if isinstance(item, Mapping)
        ],
        "source_keys": list(envelope.source_keys),
        "semantic_status": envelope.semantic_status,
        "truncated": envelope.truncated,
        "envelope": envelope_value,
        "provider_turns": "EXTRACT_SEPARATELY_FROM_USAGE_LEDGER",
    }
