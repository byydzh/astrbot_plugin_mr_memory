from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mr_memory.backtest import canonical_json
from scripts.surface_brief_ab_experiment import (
    ARM_SCHEMA_VERSION,
    CASE_SCHEMA_VERSION,
    load_arms,
    load_case,
)


SUITE_SCHEMA_VERSION = "mr-memory.three-case.suite.v1"
REPORT_SCHEMA_VERSION = "mr-memory.three-case.report.v1"
_ARM_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _object(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be one JSON object")
    return value


def _extract_layer_body(wrapper: Mapping[str, Any]) -> dict[str, Any]:
    body = wrapper.get("result")
    if isinstance(body, dict):
        return dict(body)
    # Legacy masked-pilot outputs put generation data at the top level.  The
    # post-run gold score is not a provider output and is deliberately omitted.
    return {
        str(key): value
        for key, value in wrapper.items()
        if str(key) not in {"gold_score", "evaluation"}
    }


def _extract_brief(layer_body: Mapping[str, Any]) -> dict[str, Any]:
    brief = layer_body.get("brief") or layer_body.get("memory_brief")
    if not isinstance(brief, dict):
        raise ValueError("layer result has no structured memory brief")
    return dict(brief)


def prepare_surface_inputs(
    *,
    memory_case_path: Path,
    output_dir: Path,
    surface_case_template_path: Path | None = None,
    layer_result_path: Path | None = None,
    surface_case_id: str | None = None,
    arm_id: str = "layered-memory",
    arm_label: str = "Layered memory",
) -> dict[str, Any]:
    memory_case = _object(_load_json(memory_case_path), field="memory case")
    template: dict[str, Any] | None = None
    if surface_case_template_path is not None:
        template = load_case(surface_case_template_path)
    case_id = str(
        surface_case_id
        or (template or {}).get("case_id")
        or f"{memory_case.get('case_id')}-surface"
    ).strip()
    if not case_id:
        raise ValueError("surface case_id must not be empty")
    surface_case = {
        "schema_version": CASE_SCHEMA_VERSION,
        "case_id": case_id,
        "cutoff_at": int(
            (template or {}).get("cutoff_at")
            or memory_case.get("cutoff_at")
            or 0
        ),
        "query": str((template or {}).get("query") or memory_case.get("query") or ""),
        "recent_context": list((template or {}).get("recent_context") or []),
    }
    if surface_case["cutoff_at"] <= 0 or not surface_case["query"].strip():
        raise ValueError("memory case must provide a positive cutoff and non-empty query")
    output_dir.mkdir(parents=True, exist_ok=True)
    case_path = output_dir / "case.json"
    _atomic_write_json(case_path, surface_case)
    load_case(case_path)

    result: dict[str, Any] = {
        "schema_version": "mr-memory.surface-input.preparation.v1",
        "case_path": str(case_path),
        "case_sha256": _file_sha256(case_path),
        "arm_path": None,
        "layer_result_sha256": None,
        "actual_output_sha256": None,
    }
    if layer_result_path is None:
        return result
    if not _ARM_ID_RE.fullmatch(arm_id):
        raise ValueError(f"invalid arm_id: {arm_id!r}")
    wrapper = _object(_load_json(layer_result_path), field="layer result")
    if str(wrapper.get("status") or "").upper() != "COMPLETED":
        raise ValueError("layer result must have status=COMPLETED")
    layer_body = _extract_layer_body(wrapper)
    brief = _extract_brief(layer_body)
    arm = {
        "schema_version": ARM_SCHEMA_VERSION,
        "case_id": case_id,
        "arm_id": arm_id,
        "label": arm_label,
        "memory_brief": brief,
    }
    arm_path = output_dir / f"{arm_id}.json"
    _atomic_write_json(arm_path, arm)
    load_arms([arm_path], case_id=case_id)

    # This private file is intentionally unabridged.  It gives the exact
    # provider-stage object fed into the surface preparation audit, including
    # per-round raw responses when the runner recorded them.
    provenance = {
        "schema_version": "mr-memory.surface-input.provenance.v1",
        "created_at": _utc_now(),
        "source_path": str(layer_result_path.resolve()),
        "source_file_sha256": _file_sha256(layer_result_path),
        "actual_layer_output": layer_body,
        "actual_layer_output_sha256": _stable_sha256(layer_body),
        "memory_brief_sha256": _stable_sha256(brief),
        "excluded_non_generation_wrapper_fields": [
            field for field in ("gold_score", "evaluation") if field in wrapper
        ],
    }
    provenance_path = output_dir / "layer-output.private.json"
    _atomic_write_json(provenance_path, provenance)
    result.update(
        {
            "arm_path": str(arm_path),
            "arm_sha256": _file_sha256(arm_path),
            "layer_result_sha256": _file_sha256(layer_result_path),
            "actual_output_sha256": provenance["actual_layer_output_sha256"],
            "provenance_path": str(provenance_path),
        }
    )
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _ledger_calls(path: Path | None, run_id: str) -> dict[str, Any]:
    if path is None:
        return {
            "ledger_path": None,
            "ledger_sha256": None,
            "run_id": run_id,
            "attempted_calls": None,
            "terminal_calls": None,
            "usage_complete": None,
            "calls": [],
            "stage_costs": [],
        }
    rows = [row for row in _read_jsonl(path) if str(row.get("run_id") or "") == run_id]
    attempts = [row for row in rows if str(row.get("event") or "") == "attempted"]
    terminal = [
        row for row in rows if str(row.get("event") or "") in {"completed", "failed"}
    ]
    terminal_by_request = {str(row.get("request_id") or ""): row for row in terminal}
    attempt_ids = [str(row.get("request_id") or "") for row in attempts]
    usage_complete = bool(attempts) and all(
        request_id in terminal_by_request
        and terminal_by_request[request_id].get("event") == "completed"
        and bool(terminal_by_request[request_id].get("usage_present"))
        for request_id in attempt_ids
    )
    stage_costs = []
    for row in terminal:
        event = str(row.get("event") or "")
        usage_present = bool(row.get("usage_present"))
        total = row.get("total")
        if total is None and usage_present:
            components = (
                row.get("input_other"),
                row.get("input_cached"),
                row.get("output"),
            )
            if all(value is not None for value in components):
                total = sum(int(value) for value in components)
        stage_costs.append(
            {
                "request_id": row.get("request_id"),
                "phase": row.get("phase"),
                "call_index": row.get("call_index"),
                "event": event,
                "input_other": row.get("input_other") if usage_present else None,
                "input_cached": row.get("input_cached") if usage_present else None,
                "output": row.get("output") if usage_present else None,
                "total": total if usage_present else None,
                "elapsed_ms": row.get("elapsed_ms"),
                "usage_complete": event == "completed" and usage_present,
            }
        )
    return {
        "ledger_path": str(path.resolve()),
        "ledger_sha256": _file_sha256(path),
        "run_id": run_id,
        "attempted_calls": len(attempts),
        "terminal_calls": len(terminal),
        "usage_complete": usage_complete,
        "calls": terminal,
        "stage_costs": stage_costs,
    }


def _normalized_usage(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {
            "calls": None,
            "input_other": None,
            "input_cached": None,
            "output": None,
            "total": None,
            "elapsed_ms": None,
            "usage_complete": None,
        }
    total = value.get("total")
    if total is None:
        total = value.get("total_measured_lower_bound")
    return {
        "calls": value.get("calls"),
        "input_other": value.get("input_other"),
        "input_cached": value.get("input_cached"),
        "output": value.get("output"),
        "total": total,
        "elapsed_ms": value.get("elapsed_ms"),
        "usage_complete": value.get("usage_complete"),
        "unknown_usage_calls": value.get("unknown_usage_calls"),
    }


def _sum_usage(parts: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(parts)
    numeric = ("calls", "input_other", "input_cached", "output", "total", "elapsed_ms")
    result: dict[str, Any] = {}
    for field in numeric:
        values = [row.get(field) for row in rows]
        result[field] = (
            round(sum(float(value) for value in values), 3)
            if values and all(value is not None for value in values)
            else None
        )
        if field != "elapsed_ms" and result[field] is not None:
            result[field] = int(result[field])
    completeness = [row.get("usage_complete") for row in rows]
    result["usage_complete"] = (
        all(value is True for value in completeness)
        if completeness and all(value is not None for value in completeness)
        else None
    )
    result["elapsed_semantics"] = "sum_of_provider_call_elapsed_ms"
    return result


def _require_measured_usage(
    usage: Mapping[str, Any], ledger: Mapping[str, Any], *, field: str
) -> None:
    required = (
        "calls",
        "input_other",
        "input_cached",
        "output",
        "total",
        "elapsed_ms",
    )
    missing = [name for name in required if usage.get(name) is None]
    if missing:
        raise ValueError(f"{field} has incomplete measured usage: {missing}")
    if usage.get("usage_complete") is not True:
        raise ValueError(f"{field} usage_complete is not true")
    if ledger.get("usage_complete") is not True:
        raise ValueError(f"{field} ledger usage is incomplete")
    calls = int(usage["calls"])
    if calls <= 0:
        raise ValueError(f"{field} must contain at least one provider call")
    if int(ledger.get("attempted_calls") or -1) != calls:
        raise ValueError(f"{field} embedded and ledger call counts differ")
    if len(ledger.get("stage_costs", [])) != calls:
        raise ValueError(f"{field} ledger does not contain one cost row per call")


def _wall_elapsed_ms(*values: object) -> float | None:
    for value in values:
        if value is not None:
            return round(float(value), 3)
    return None


def _sum_wall_elapsed(parts: Iterable[object]) -> float | None:
    values = list(parts)
    if not values or any(value is None for value in values):
        return None
    return round(sum(float(value) for value in values), 3)


def _matching_stage_costs(
    ledger: Mapping[str, Any], *, phase: object, call_index: object
) -> list[dict[str, Any]]:
    costs = [
        dict(item)
        for item in ledger.get("stage_costs", [])
        if isinstance(item, dict)
    ]
    exact = [
        item
        for item in costs
        if item.get("phase") == phase and item.get("call_index") == call_index
    ]
    if exact:
        return exact
    by_index = [item for item in costs if item.get("call_index") == call_index]
    return by_index


def _surface_results(
    path: Path, *, selected_arm_ids: set[str] | None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    wrapper = _object(_load_json(path), field="surface private results")
    raw_results = wrapper.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("surface private results has no results array")
    selected = [
        dict(item)
        for item in raw_results
        if isinstance(item, dict)
        and (selected_arm_ids is None or str(item.get("arm_id") or "") in selected_arm_ids)
    ]
    if not selected:
        raise ValueError("surface result selection is empty")
    for item in selected:
        if str(item.get("status") or "").upper() != "COMPLETED":
            raise ValueError("every selected surface run must have status=COMPLETED")
        answer = str(item.get("answer") or "")
        if not answer:
            raise ValueError("every selected surface run must contain a visible answer")
        recorded_sha = str(item.get("answer_sha256") or "")
        actual_sha = hashlib.sha256(answer.encode("utf-8")).hexdigest()
        if recorded_sha and recorded_sha != actual_sha:
            raise ValueError("surface answer sha256 does not match its full text")
    return wrapper, selected


def _absolute(spec_dir: Path, value: object, *, required: bool = True) -> Path | None:
    text = str(value or "").strip()
    if not text:
        if required:
            raise ValueError("required path is empty")
        return None
    path = Path(text)
    return (spec_dir / path).resolve() if not path.is_absolute() else path.resolve()


def _build_case_report(spec_dir: Path, item: Mapping[str, Any]) -> dict[str, Any]:
    case_id = str(item.get("case_id") or "").strip()
    if not case_id:
        raise ValueError("suite case_id must not be empty")
    case_path = _absolute(spec_dir, item.get("case_path"))
    layer_path = _absolute(spec_dir, item.get("layer_result_path"))
    surface_path = _absolute(spec_dir, item.get("surface_results_path"))
    gold_path = _absolute(spec_dir, item.get("gold_path"))
    assert case_path and layer_path and surface_path and gold_path
    layer_ledger = _absolute(
        spec_dir, item.get("layer_ledger_path"), required=False
    )
    surface_ledger = _absolute(
        spec_dir, item.get("surface_ledger_path"), required=False
    )
    case_input = _object(_load_json(case_path), field=f"{case_id}.case")
    layer_wrapper = _object(_load_json(layer_path), field=f"{case_id}.layer")
    if str(layer_wrapper.get("status") or "").upper() != "COMPLETED":
        raise ValueError(f"{case_id}.layer must have status=COMPLETED")
    layer_body = _extract_layer_body(layer_wrapper)
    layer_run_id = str(layer_wrapper.get("run_id") or layer_body.get("run_id") or "")
    selected_ids = {
        str(value) for value in item.get("surface_arm_ids", []) if str(value)
    } or None
    surface_wrapper, surface_rows = _surface_results(
        surface_path, selected_arm_ids=selected_ids
    )
    gold = _object(_load_json(gold_path), field=f"{case_id}.post-run gold")

    layer_usage = _normalized_usage(layer_wrapper.get("usage") or layer_body.get("usage"))
    layer_wall_elapsed = _wall_elapsed_ms(
        layer_body.get("elapsed_ms"), layer_wrapper.get("elapsed_ms")
    )
    layer_ledger_report = _ledger_calls(layer_ledger, layer_run_id)
    _require_measured_usage(
        layer_usage, layer_ledger_report, field=f"{case_id}.layer"
    )
    surface_outputs: list[dict[str, Any]] = []
    surface_usage_parts: list[dict[str, Any]] = []
    surface_wall_parts: list[float | None] = []
    for row in surface_rows:
        usage = _normalized_usage(row.get("usage"))
        surface_usage_parts.append(usage)
        wall_elapsed = _wall_elapsed_ms(row.get("elapsed_ms"))
        surface_wall_parts.append(wall_elapsed)
        run_id = str(row.get("run_id") or "")
        answer = str(row.get("answer") or "")
        surface_ledger_report = _ledger_calls(surface_ledger, run_id)
        _require_measured_usage(
            usage, surface_ledger_report, field=f"{case_id}.surface[{run_id}]"
        )
        surface_outputs.append(
            {
                "run_id": run_id,
                "arm_id": row.get("arm_id"),
                "status": row.get("status"),
                "actual_answer": answer,
                "actual_answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
                "recorded_answer_sha256": row.get("answer_sha256"),
                "actual_result": row,
                "actual_result_sha256": _stable_sha256(row),
                "usage": usage,
                "wall_elapsed_ms": wall_elapsed,
                "wall_elapsed_semantics": "surface_generation_run_wall_elapsed_ms",
                "ledger": surface_ledger_report,
            }
        )
    surface_usage = _sum_usage(surface_usage_parts)
    surface_wall_elapsed = _sum_wall_elapsed(surface_wall_parts)
    combined_usage = _sum_usage([layer_usage, surface_usage])
    combined_usage["wall_elapsed_ms"] = _sum_wall_elapsed(
        [layer_wall_elapsed, surface_wall_elapsed]
    )
    combined_usage["wall_elapsed_semantics"] = (
        "sum_of_dependency_ordered_layer_and_surface_run_wall_elapsed_ms"
    )
    if layer_wall_elapsed is None or surface_wall_elapsed is None:
        raise ValueError(f"{case_id} is missing measured layer or surface wall elapsed_ms")

    stage_outputs: list[dict[str, Any]] = []
    rounds = layer_body.get("rounds")
    if isinstance(rounds, list) and rounds:
        for index, round_value in enumerate(rounds):
            if isinstance(round_value, dict):
                stage_outputs.append(
                    {
                        "stage_index": index,
                        "phase": round_value.get("phase"),
                        "call_index": round_value.get("call_index"),
                        "actual_output": round_value,
                        "actual_output_sha256": _stable_sha256(round_value),
                        "costs": _matching_stage_costs(
                            layer_ledger_report,
                            phase=round_value.get("phase"),
                            call_index=round_value.get("call_index"),
                        ),
                    }
                )
    else:
        stage_outputs.append(
            {
                "stage_index": 0,
                "phase": item.get("route") or layer_body.get("arm"),
                "call_index": None,
                "actual_output": layer_body,
                "actual_output_sha256": _stable_sha256(layer_body),
                "costs": list(layer_ledger_report.get("stage_costs", [])),
            }
        )
    return {
        "case_id": case_id,
        "title": str(item.get("title") or case_id),
        "request_provenance": str(item.get("request_provenance") or "unspecified"),
        "route": str(item.get("route") or "unspecified"),
        "input": {
            "query": case_input.get("query"),
            "recent_context": case_input.get("recent_context", []),
            "cutoff_at": case_input.get("cutoff_at"),
            "scope": case_input.get("umo") or case_input.get("scope_id"),
            "case_path": str(case_path),
            "case_sha256": _file_sha256(case_path),
            "actual_case": case_input,
            "actual_case_sha256": _stable_sha256(case_input),
        },
        "layer": {
            "status": layer_wrapper.get("status") or layer_body.get("status"),
            "run_id": layer_run_id,
            "result_path": str(layer_path),
            "result_file_sha256": _file_sha256(layer_path),
            "actual_output": layer_body,
            "actual_output_sha256": _stable_sha256(layer_body),
            "stages": stage_outputs,
            "usage": layer_usage,
            "wall_elapsed_ms": layer_wall_elapsed,
            "wall_elapsed_semantics": "layer_generation_run_wall_elapsed_ms",
            "ledger": layer_ledger_report,
        },
        "surface": {
            "execution_semantics": sorted(
                {
                    str(output["actual_result"].get("execution_semantics") or "")
                    for output in surface_outputs
                    if str(output["actual_result"].get("execution_semantics") or "")
                }
            ),
            "results_path": str(surface_path),
            "results_file_sha256": _file_sha256(surface_path),
            "manifest_sha256": surface_wrapper.get("manifest_sha256"),
            "outputs": surface_outputs,
            "usage": surface_usage,
            "wall_elapsed_ms": surface_wall_elapsed,
            "wall_elapsed_semantics": "sum_of_selected_surface_run_wall_elapsed_ms",
        },
        "combined_cost": combined_usage,
        "post_run_human_reference": {
            "loaded_during_generation": False,
            "path": str(gold_path),
            "sha256": _file_sha256(gold_path),
            "content": gold,
        },
        "human_review": {
            "core_facts_and_relationships": None,
            "subject_and_attribution": None,
            "unsupported_upgrades": None,
            "uncertainty_preserved": None,
            "notes": None,
        },
    }


def _build_prior_failure_report(
    spec_dir: Path, item: Mapping[str, Any]
) -> dict[str, Any]:
    failure_id = str(item.get("failure_id") or "").strip()
    if not failure_id:
        raise ValueError("prior failure_id must not be empty")
    result_path = _absolute(spec_dir, item.get("result_path"))
    ledger_path = _absolute(spec_dir, item.get("ledger_path"))
    assert result_path and ledger_path
    result = _object(_load_json(result_path), field=f"{failure_id}.failure result")
    if str(result.get("status") or "").upper() != "FAILED":
        raise ValueError(f"{failure_id}.result must have status=FAILED")
    run_id = str(result.get("run_id") or "").strip()
    if not run_id:
        raise ValueError(f"{failure_id}.result must contain run_id")
    error_type = str(result.get("error_type") or "").strip()
    if not error_type:
        raise ValueError(f"{failure_id}.result must contain error_type")
    usage = _normalized_usage(result.get("usage"))
    ledger = _ledger_calls(ledger_path, run_id)
    _require_measured_usage(usage, ledger, field=f"{failure_id}.failure")
    wall_elapsed = _wall_elapsed_ms(result.get("elapsed_ms"))
    if wall_elapsed is None:
        raise ValueError(f"{failure_id}.result is missing wall elapsed_ms")
    expected_total = item.get("expected_total")
    if expected_total is not None and int(expected_total) != int(usage["total"]):
        raise ValueError(
            f"{failure_id}.total differs from expected_total: "
            f"{usage['total']} != {expected_total}"
        )
    phases = sorted(
        {
            str(row.get("phase") or "unspecified")
            for row in ledger.get("stage_costs", [])
            if isinstance(row, dict)
        }
    )
    return {
        "failure_id": failure_id,
        "title": str(item.get("title") or failure_id),
        "status": "FAILED",
        "run_id": run_id,
        "phases": phases,
        "error_type": error_type,
        "error_detail": str(result.get("error_detail") or ""),
        "usage": usage,
        "wall_elapsed_ms": wall_elapsed,
        "wall_elapsed_semantics": "failed_runner_wall_elapsed_ms",
        "result_path": str(result_path),
        "result_file_sha256": _file_sha256(result_path),
        "ledger": ledger,
    }


def build_report(spec_path: Path, output_dir: Path) -> dict[str, Any]:
    suite = _object(_load_json(spec_path), field="suite")
    if str(suite.get("schema_version") or "") != SUITE_SCHEMA_VERSION:
        raise ValueError("unsupported three-case suite schema_version")
    raw_cases = suite.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != 3:
        raise ValueError("suite must contain exactly three cases")
    reports = [
        _build_case_report(spec_path.parent.resolve(), _object(item, field="suite case"))
        for item in raw_cases
    ]
    ids = [item["case_id"] for item in reports]
    if len(set(ids)) != 3:
        raise ValueError("suite case_ids must be unique")
    raw_prior_failures = suite.get("prior_failures", [])
    if not isinstance(raw_prior_failures, list):
        raise ValueError("suite prior_failures must be an array")
    prior_failures = [
        _build_prior_failure_report(
            spec_path.parent.resolve(), _object(item, field="suite prior failure")
        )
        for item in raw_prior_failures
    ]
    failure_ids = [item["failure_id"] for item in prior_failures]
    if len(set(failure_ids)) != len(failure_ids):
        raise ValueError("suite prior failure_ids must be unique")
    successful_cost = _sum_usage(item["combined_cost"] for item in reports)
    successful_cost["wall_elapsed_ms"] = _sum_wall_elapsed(
        item["combined_cost"].get("wall_elapsed_ms") for item in reports
    )
    prior_failure_cost = _sum_usage(item["usage"] for item in prior_failures)
    prior_failure_cost["wall_elapsed_ms"] = _sum_wall_elapsed(
        item.get("wall_elapsed_ms") for item in prior_failures
    )
    all_measured_cost = _sum_usage(
        [*(item["combined_cost"] for item in reports), *(item["usage"] for item in prior_failures)]
    )
    all_measured_cost["wall_elapsed_ms"] = _sum_wall_elapsed(
        [
            *(item["combined_cost"].get("wall_elapsed_ms") for item in reports),
            *(item.get("wall_elapsed_ms") for item in prior_failures),
        ]
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "suite_path": str(spec_path.resolve()),
        "suite_sha256": _file_sha256(spec_path),
        "semantic_evaluation": "NOT_PERFORMED_USER_REVIEW_REQUIRED",
        "automatic_quality_scores": None,
        "surface_execution_scope": (
            "Controlled calls to the configured main Provider with the production "
            "EvidenceCertificateV2 injection wording; not full AstrBot E2E persona, "
            "tools, plugin chain, or hidden context."
        ),
        "cost_definition": {
            "tokens": "provider-reported input_other + input_cached + output",
            "provider_latency": "sum of provider-call elapsed_ms from the usage ledger",
            "wall": (
                "runner-recorded layer/surface elapsed_ms; combined wall is the "
                "dependency-ordered per-case stage sum, not the whole parallel suite wall"
            ),
            "money": "not computed because no immutable price record is part of the fixture",
        },
        "cost_summary": {
            "completed_three_case_suite": successful_cost,
            "prior_failed_attempts": prior_failure_cost,
            "all_measured_attempts": all_measured_cost,
        },
        "prior_failures": prior_failures,
        "cases": reports,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    private_path = output_dir / "three-case-report.private.json"
    markdown_path = output_dir / "three-case-report.md"
    _atomic_write_json(private_path, report)
    _atomic_write_text(markdown_path, render_markdown(report, private_path))
    return {
        "report_path": str(private_path),
        "report_sha256": _file_sha256(private_path),
        "markdown_path": str(markdown_path),
        "markdown_sha256": _file_sha256(markdown_path),
        "cases": ids,
        "prior_failures": failure_ids,
        "semantic_evaluation": report["semantic_evaluation"],
    }


def _json_block(value: object) -> str:
    return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n```"


def _fmt(value: object) -> str:
    if value is None:
        return "未记录"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown(report: Mapping[str, Any], private_path: Path) -> str:
    completed_cost = report["cost_summary"]["completed_three_case_suite"]
    failed_cost = report["cost_summary"]["prior_failed_attempts"]
    all_cost = report["cost_summary"]["all_measured_attempts"]
    lines = [
        "# 三案例真实输出与成本记录",
        "",
        "## 技术摘要",
        "",
        "本报告不做自动语义评分，也不宣布哪个结果更好。下面直接保留输入、实际输出、人工证据与精确 Provider 账单，供人工检查。",
        "",
        "表层回答是使用线上同 Provider 与生产 EvidenceCertificateV2 注入文案的受控调用；它不复现完整 AstrBot persona、工具、插件链或其他隐藏上下文，不能标作完整 AstrBot E2E 回答。",
        "",
        "已完成三案例成本：{calls} 次 Provider 调用、{tokens} Token、Provider 调用累计 {elapsed} ms、依赖顺序案例链路累计 {wall} ms。先前失败尝试另计 {failed_calls} 次调用、{failed_tokens} Token；所有已测尝试合计 {all_tokens} Token。".format(
            calls=_fmt(completed_cost.get("calls")),
            tokens=_fmt(completed_cost.get("total")),
            elapsed=_fmt(completed_cost.get("elapsed_ms")),
            wall=_fmt(completed_cost.get("wall_elapsed_ms")),
            failed_calls=_fmt(failed_cost.get("calls")),
            failed_tokens=_fmt(failed_cost.get("total")),
            all_tokens=_fmt(all_cost.get("total")),
        ),
        "",
        f"完整未截断机器可读记录：`{private_path.resolve()}`",
        "",
        "## 三案例成本与时延",
        "",
        "三个案例的任务不同、路由不同，因此本表用于精确查账，不把 Token 或时延排序解释为语义质量排序。",
        "",
        "| 案例 | 路由 | 调用数 | Token | Provider 时延 (ms) | 案例链路 wall (ms) | 用量完整 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in report["cases"]:
        cost = case["combined_cost"]
        lines.append(
            "| {title} | {route} | {calls} | {tokens} | {latency} | {wall} | {complete} |".format(
                title=case["title"],
                route=case["route"],
                calls=_fmt(cost.get("calls")),
                tokens=_fmt(cost.get("total")),
                latency=_fmt(cost.get("elapsed_ms")),
                wall=_fmt(cost.get("wall_elapsed_ms")),
                complete=_fmt(cost.get("usage_complete")),
            )
        )
    if report.get("prior_failures"):
        lines.extend(
            [
                "",
                "## 先前协议失败也计入研发成本",
                "",
                "这些失败不是三案例成功结果。它们只用于记录为得到当前结果已经发生的 Provider 消耗；公开报告不展开失败调用的模型输出正文。",
                "",
                "| 尝试 | 阶段 | 错误类型 | calls | input_other | input_cached | output | total | provider elapsed_ms | wall elapsed_ms | 账本 SHA-256 |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for failure in report["prior_failures"]:
            usage = failure["usage"]
            lines.append(
                "| {title} | {phases} | {error_type} | {calls} | {io} | {cached} | {out} | {total} | {elapsed} | {wall} | `{ledger_sha}` |".format(
                    title=failure["title"],
                    phases=", ".join(failure["phases"]),
                    error_type=failure["error_type"],
                    calls=_fmt(usage.get("calls")),
                    io=_fmt(usage.get("input_other")),
                    cached=_fmt(usage.get("input_cached")),
                    out=_fmt(usage.get("output")),
                    total=_fmt(usage.get("total")),
                    elapsed=_fmt(usage.get("elapsed_ms")),
                    wall=_fmt(failure.get("wall_elapsed_ms")),
                    ledger_sha=failure["ledger"]["ledger_sha256"],
                )
            )
            lines.extend(
                [
                    "",
                    "逐调用成本（不含模型输出正文）：",
                    "",
                    _json_block(failure["ledger"]["stage_costs"]),
                ]
            )
    lines.extend(
        [
            "",
            "## 范围、数据和指标定义",
            "",
            "- `input_other`、`input_cached`、`output` 与 `total` 均直接来自 Provider usage；`total = input_other + input_cached + output`。",
            "- `provider elapsed_ms` 是 usage ledger 中 Provider 调用耗时之和；`wall elapsed_ms` 是 runner 对记忆层或表层执行单元测得的墙钟时间。",
            "- 每例合计 wall 是依赖顺序的记忆层与表层 wall 之和；不是三个案例并行执行时的套件总墙钟时间。",
            "- Gold 在生成结束后才加载，只作为给人的证据参考；不进入 Provider 输入。",
            "- 三例是定向案例研究，不构成随机样本、准确率或总体质量估计。",
            "",
            "## 三个案例的完整可审查记录",
        ]
    )
    for index, case in enumerate(report["cases"], start=1):
        lines.extend(
            [
                "",
                f"## 案例 {index}：{case['title']}",
                "",
                f"- 来源：{case['request_provenance']}",
                f"- 路由：{case['route']}",
                f"- 截止时间：{case['input']['cutoff_at']}",
                f"- 输入 SHA-256：`{case['input']['case_sha256']}`",
                "",
                "### 统一输入",
                "",
                _json_block(case["input"]["actual_case"]),
                "",
                "### L2/L3 实际记忆输出",
                "",
                f"完整输出 SHA-256：`{case['layer']['actual_output_sha256']}`",
                "",
                _json_block(case["layer"]["actual_output"]),
                "",
                "### 主 LLM 实际可见回答",
                "",
            ]
        )
        for output in case["surface"]["outputs"]:
            lines.extend(
                [
                    f"#### {output['arm_id']} / {output['run_id']}",
                    "",
                    f"回答 SHA-256：`{output['actual_answer_sha256']}`",
                    "",
                    output["actual_answer"],
                    "",
                ]
            )
        lines.extend(
            [
                "### 调用与成本",
                "",
                "| 阶段 | calls | input_other | input_cached | output | total | provider elapsed_ms | wall elapsed_ms | usage_complete |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for label, usage, wall_elapsed in (
            ("记忆层", case["layer"]["usage"], case["layer"]["wall_elapsed_ms"]),
            ("表层主 LLM", case["surface"]["usage"], case["surface"]["wall_elapsed_ms"]),
            ("合计", case["combined_cost"], case["combined_cost"]["wall_elapsed_ms"]),
        ):
            lines.append(
                "| {label} | {calls} | {io} | {cached} | {out} | {total} | {elapsed} | {wall} | {complete} |".format(
                    label=label,
                    calls=_fmt(usage.get("calls")),
                    io=_fmt(usage.get("input_other")),
                    cached=_fmt(usage.get("input_cached")),
                    out=_fmt(usage.get("output")),
                    total=_fmt(usage.get("total")),
                    elapsed=_fmt(usage.get("elapsed_ms")),
                    wall=_fmt(wall_elapsed),
                    complete=_fmt(usage.get("usage_complete")),
                )
            )
        lines.extend(
            [
                "",
                "逐调用明细（原始账单事件）：",
                "",
                _json_block(
                    {
                        "memory_layer": case["layer"]["ledger"]["calls"],
                        "memory_layer_stage_costs": case["layer"]["ledger"]["stage_costs"],
                        "surface": [
                            {
                                "run_id": item["run_id"],
                                "calls": item["ledger"]["calls"],
                                "stage_costs": item["ledger"]["stage_costs"],
                            }
                            for item in case["surface"]["outputs"]
                        ],
                    }
                ),
                "",
                "### 跑后才加载的人工证据",
                "",
                f"Gold SHA-256：`{case['post_run_human_reference']['sha256']}`",
                "",
                _json_block(case["post_run_human_reference"]["content"]),
                "",
                "### 人工检查（留空）",
                "",
                "- 核心事实/关系：",
                "- 主体与归因：",
                "- 是否有证据外升级：",
                "- 不确定性是否保留：",
                "- 备注：",
            ]
        )
    lines.extend(
        [
            "",
            "## 方法与完整性检查",
            "",
            "每例必须同时满足：记忆层 `COMPLETED`、surface `COMPLETED`、可见回答非空且 SHA-256 匹配、嵌入 usage 与独立 JSONL ledger 调用数一致、所有调用都有完整 usage。任一条件不满足，聚合器直接失败，不生成一份看似完整的报告。",
            "",
            "逐案例使用冻结输入和截止时间。生成阶段不接受 Gold 路径；报告聚合结束后才读取 Gold。完整输入、记忆层输出、surface 结果、账本和 Gold 都在私有机器可读报告中带文件或稳定对象哈希。",
            "",
            "## 限制与待人工判断",
            "",
            "本报告只证明这些具体调用实际返回了什么、消耗了多少，以及账本是否闭合；它不证明答案正确，也不把 surface verifier 通过当作语义质量。三例的人工检查栏保持空白，等待用户逐例判断核心事实、主体归因、证据外升级和不确定性保留。",
            "",
            "## 下一步",
            "",
            "1. 用户完成三例人工检查并记录具体错漏，不把单一总分替代逐项证据审查。",
            "2. 根据人工结果决定是否扩展为未污染 holdout；在此之前不从三例外推上线质量。",
        ]
    )
    return "\n".join(lines) + "\n"


def _prepare_surface_command(args: argparse.Namespace) -> dict[str, Any]:
    return prepare_surface_inputs(
        memory_case_path=Path(args.memory_case).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        surface_case_template_path=(
            Path(args.surface_case_template).resolve()
            if args.surface_case_template
            else None
        ),
        layer_result_path=(
            Path(args.layer_result).resolve() if args.layer_result else None
        ),
        surface_case_id=args.surface_case_id,
        arm_id=args.arm_id,
        arm_label=args.arm_label,
    )


def _aggregate_command(args: argparse.Namespace) -> dict[str, Any]:
    return build_report(Path(args.suite).resolve(), Path(args.output_dir).resolve())


def _module_command(module: str, *arguments: object) -> list[str]:
    return [sys.executable, "-m", module, *(str(item) for item in arguments)]


def _parse_memory_checkpoint_imports(values: Sequence[str] | None) -> dict[str, Path]:
    allowed = {"call-726", "good-girl", "q0030"}
    imports: dict[str, Path] = {}
    for raw in values or ():
        case_key, separator, raw_path = str(raw).partition("=")
        case_key = case_key.strip()
        if separator != "=" or case_key not in allowed or not raw_path.strip():
            raise ValueError(
                "--import-memory-checkpoint must be CASE=PATH where CASE is "
                "call-726, good-girl, or q0030"
            )
        if case_key in imports:
            raise ValueError(f"duplicate memory checkpoint import for {case_key}")
        path = Path(raw_path.strip()).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"memory checkpoint import missing: {path}")
        imports[case_key] = path
    return imports


def _parse_case_imports(
    values: Sequence[str] | None,
    *,
    option: str,
    allowed: set[str],
) -> dict[str, Path]:
    imports: dict[str, Path] = {}
    for raw in values or ():
        case_key, separator, raw_path = str(raw).partition("=")
        case_key = case_key.strip()
        if separator != "=" or case_key not in allowed or not raw_path.strip():
            raise ValueError(
                f"{option} must be CASE=PATH where CASE is "
                + ", ".join(sorted(allowed))
            )
        if case_key in imports:
            raise ValueError(f"duplicate {option} for {case_key}")
        path = Path(raw_path.strip()).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{option} missing: {path}")
        imports[case_key] = path
    return imports


def _completed_case_checkpoint_plan(path: Path, *, case_key: str) -> dict[str, Any]:
    from scripts.layered_case_generation import (
        _memory_ledger_rows,
        _usage_from_ledger_rows,
    )

    source_dir = path.parent
    manifest_path = source_dir / "manifest.json"
    memory_path = source_dir / "memory.private.json"
    ledger_path = source_dir / "usage.jsonl"
    surface_path = source_dir / "surface" / "result.private.json"
    surface_private_path = source_dir / "surface" / "private_results.json"
    artifacts = (path, manifest_path, memory_path, ledger_path, surface_path, surface_private_path)
    for artifact in artifacts:
        if not artifact.is_file():
            raise FileNotFoundError(f"completed-case import artifact missing: {artifact}")
    top = _object(_load_json(path), field="completed-case source result")
    manifest = _object(_load_json(manifest_path), field="completed-case source manifest")
    memory = _object(_load_json(memory_path), field="completed-case source memory")
    surface = _object(_load_json(surface_path), field="completed-case source surface")
    if any(
        str(value.get("status") or "").upper() != "COMPLETED"
        for value in (top, memory, surface)
    ):
        raise ValueError("completed-case import source is not fully COMPLETED")
    memory_rows = _memory_ledger_rows(
        ledger_path, run_id=str(memory.get("run_id") or "")
    )
    surface_rows = _memory_ledger_rows(
        ledger_path, run_id=str(surface.get("run_id") or "")
    )
    if len(memory_rows) != 6 or len(surface_rows) != 2:
        raise ValueError("completed-case import must contain 3 memory + 1 surface calls")
    rows = _read_jsonl(ledger_path)
    if rows != [*memory_rows, *surface_rows]:
        raise ValueError("completed-case import ledger has orphan/interleaved rows")
    memory_usage = _usage_from_ledger_rows(memory_rows)
    surface_usage = _usage_from_ledger_rows(surface_rows)
    if memory.get("usage") != memory_usage or surface.get("usage") != surface_usage:
        raise ValueError("completed-case import layer usage mismatch")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("completed-case import source manifest has no inputs")
    source_suite_root = path.parents[2]
    database_sha256 = inputs.get("database_sha256")
    bound_packet_path = None
    bound_database_path = None
    if database_sha256 is not None:
        packet_path = (
            source_suite_root / "prepared-input" / case_key / "evidence_packet.json"
        )
        database_path = source_suite_root / "prepared-input" / case_key / "scope.db"
        if (
            not packet_path.is_file()
            or _file_sha256(packet_path) != inputs.get("packet_file_sha256")
            or not database_path.is_file()
            or _file_sha256(database_path) != database_sha256
        ):
            raise ValueError(
                "completed-case import suite-bound frozen packet/database is unavailable"
            )
        bound_packet_path = str(packet_path.resolve())
        bound_database_path = str(database_path.resolve())
    attempt_manifest = source_suite_root / "suite.json"
    if not attempt_manifest.is_file():
        attempt_manifest = source_suite_root / "run-plan.json"
    if not attempt_manifest.is_file():
        raise FileNotFoundError("completed-case source has no suite/run-plan identity")
    attempt_object = _object(
        _load_json(attempt_manifest), field="completed-case source attempt manifest"
    )
    attempt_schema = str(attempt_object.get("schema_version") or "")
    if attempt_schema not in {
        "mr-memory.three-case.run-plan.v1",
        "mr-memory.three-case.run-plan.v2",
        "mr-memory.three-case.run-plan.v3",
        SUITE_SCHEMA_VERSION,
    }:
        raise ValueError("completed-case source attempt manifest schema is invalid")
    attempt_manifest_sha256 = _file_sha256(attempt_manifest)
    source_attempt_id = f"{source_suite_root.name}@{attempt_manifest_sha256[:16]}"
    return {
        "path": str(path),
        "sha256": _file_sha256(path),
        "artifact_sha256": {
            str(artifact.relative_to(source_dir)): _file_sha256(artifact)
            for artifact in artifacts
        },
        "selected_ledger_rows_sha256": _stable_sha256(rows),
        "request_ids": [
            str(row.get("request_id"))
            for row in rows
            if row.get("event") == "attempted"
        ],
        "logical_imported_calls": 4,
        "logical_imported_tokens": int(memory_usage["total"])
        + int(surface_usage["total"]),
        "source_attempt_id": source_attempt_id,
        "source_attempt_manifest_path": str(attempt_manifest.resolve()),
        "source_attempt_manifest_sha256": attempt_manifest_sha256,
        "source_suite_manifest_sha256": attempt_manifest_sha256,
        "source_attempt_manifest_schema": attempt_schema,
        "bound_packet_path": bound_packet_path,
        "bound_packet_sha256": inputs.get("packet_file_sha256"),
        "bound_database_path": bound_database_path,
        "bound_database_sha256": database_sha256,
        "new_provider_calls_upper_bound": 0,
    }


def _provider_stage_checkpoint_plan(path: Path) -> dict[str, Any]:
    from scripts.layered_case_generation import (
        _memory_ledger_rows,
        _usage_from_ledger_rows,
    )

    source_dir = path.parent
    manifest_path = source_dir / "manifest.json"
    stages_path = source_dir / "memory-stages.private.json"
    ledger_path = source_dir / "usage.jsonl"
    case_path = source_dir / "case.input.json"
    packet_path = source_dir / "evidence.input.json"
    artifacts = (path, manifest_path, stages_path, ledger_path, case_path, packet_path)
    for artifact in artifacts:
        if not artifact.is_file():
            raise FileNotFoundError(f"provider-stage import artifact missing: {artifact}")
    failed = _object(_load_json(path), field="provider-stage source memory")
    if (
        str(failed.get("status") or "").upper() != "FAILED"
        or failed.get("error_type") != "ValueError"
        or failed.get("error_detail")
        != "unresolved must be an array with at most 16 items"
    ):
        raise ValueError("provider-stage source is not the accepted assembly failure")
    rows = _memory_ledger_rows(
        ledger_path, run_id=str(failed.get("run_id") or "")
    )
    if _read_jsonl(ledger_path) != rows or len(rows) != 6:
        raise ValueError("provider-stage source must contain exactly three closed calls")
    usage = _usage_from_ledger_rows(rows)
    if failed.get("usage") != usage:
        raise ValueError("provider-stage source usage mismatch")
    stages = _object(_load_json(stages_path), field="provider-stage source stages")
    if not isinstance(stages.get("stages"), list) or len(stages["stages"]) != 3:
        raise ValueError("provider-stage source must contain exactly three stages")
    return {
        "path": str(path),
        "sha256": _file_sha256(path),
        "artifact_sha256": {
            str(artifact.relative_to(source_dir)): _file_sha256(artifact)
            for artifact in artifacts
        },
        "selected_ledger_rows_sha256": _stable_sha256(rows),
        "request_ids": [
            str(row.get("request_id"))
            for row in rows
            if row.get("event") == "attempted"
        ],
        "logical_imported_calls": 3,
        "logical_imported_tokens": int(usage["total"]),
        "new_provider_calls_upper_bound": 1,
    }


def _memory_checkpoint_plan(path: Path, *, case_key: str) -> dict[str, Any]:
    memory = _object(_load_json(path), field="memory checkpoint import")
    if str(memory.get("status") or "").upper() != "COMPLETED":
        raise ValueError("suite import requires a COMPLETED memory checkpoint")
    run_id = str(memory.get("run_id") or "")
    usage = memory.get("usage")
    if not run_id or not isinstance(usage, Mapping):
        raise ValueError("suite memory checkpoint import has no run_id/usage")
    if not bool(usage.get("usage_complete")) or int(usage.get("calls") or 0) <= 0:
        raise ValueError("suite memory checkpoint import has incomplete usage")
    manifest_path = path.parent / "manifest.json"
    ledger_path = path.parent / "usage.jsonl"
    if not manifest_path.is_file() or not ledger_path.is_file():
        raise FileNotFoundError(
            "suite import requires sibling manifest.json and usage.jsonl"
        )
    ledger = _ledger_calls(ledger_path, run_id)
    measured_total = sum(
        int(row.get("total") or 0)
        for row in ledger["calls"]
        if str(row.get("event") or "") == "completed"
    )
    if (
        not bool(ledger["usage_complete"])
        or int(ledger["attempted_calls"] or 0) != int(usage["calls"])
        or measured_total != int(usage.get("total") or 0)
    ):
        raise ValueError("suite memory checkpoint import ledger/usage mismatch")
    surface_path = path.parent / "surface" / "result.private.json"
    surface_status = None
    if surface_path.is_file():
        surface = _object(_load_json(surface_path), field="source surface checkpoint")
        surface_status = str(surface.get("status") or "").upper() or None
    source_manifest = _object(
        _load_json(manifest_path), field="memory checkpoint source manifest"
    )
    source_inputs = source_manifest.get("inputs")
    if not isinstance(source_inputs, Mapping):
        raise ValueError("memory checkpoint source manifest has no inputs object")
    database_sha256 = source_inputs.get("database_sha256")
    bound_database_path = None
    if database_sha256 is not None:
        source_suite_root = path.parent.parent.parent
        database_path = source_suite_root / "prepared-input" / case_key / "scope.db"
        if not database_path.is_file() or _file_sha256(database_path) != str(
            database_sha256
        ):
            raise ValueError(
                "memory checkpoint source database is missing or does not match its manifest"
            )
        bound_database_path = str(database_path)
    return {
        "path": str(path),
        "sha256": _file_sha256(path),
        "status": "COMPLETED",
        "run_id": run_id,
        "calls": int(usage["calls"]),
        "tokens": int(usage.get("total") or 0),
        "usage_complete": True,
        "manifest_sha256": _file_sha256(manifest_path),
        "ledger_sha256": _file_sha256(ledger_path),
        "source_surface_status": surface_status,
        "bound_database_path": bound_database_path,
        "bound_database_sha256": database_sha256,
        "new_provider_calls_upper_bound": 1,
    }


def _run_suite_command(args: argparse.Namespace) -> dict[str, Any]:
    """Run the frozen #726 / 好女孩 / q0030 generation pipeline.

    Without the acknowledgement flag this command is a pure dry run and only
    prints the exact argv vectors.  No shell is involved and gold paths do not
    appear in any provider-stage command.
    """

    dev_root = Path(args.dev_root).resolve()
    output_root = Path(args.output_root).resolve()
    config_path = Path(args.config).resolve()
    memory_imports = _parse_memory_checkpoint_imports(
        getattr(args, "import_memory_checkpoint", None)
    )
    memory_import_plans = {
        case_key: _memory_checkpoint_plan(path, case_key=case_key)
        for case_key, path in memory_imports.items()
    }
    completed_case_imports = _parse_case_imports(
        getattr(args, "import_completed_case", None),
        option="--import-completed-case",
        allowed={"call-726", "good-girl"},
    )
    provider_stage_imports = _parse_case_imports(
        getattr(args, "import_provider_stages_checkpoint", None),
        option="--import-provider-stages-checkpoint",
        allowed={"good-girl", "q0030"},
    )
    if set(memory_imports) & (
        set(completed_case_imports) | set(provider_stage_imports)
    ):
        raise ValueError("a case cannot use two checkpoint import modes")
    completed_case_import_plans = {
        case_key: _completed_case_checkpoint_plan(path, case_key=case_key)
        for case_key, path in completed_case_imports.items()
    }
    provider_stage_import_plans = {
        case_key: _provider_stage_checkpoint_plan(path)
        for case_key, path in provider_stage_imports.items()
    }
    v7_import_mode = (
        set(completed_case_imports) == {"call-726"}
        and set(provider_stage_imports) == {"good-girl"}
    )
    v8_import_mode = (
        set(completed_case_imports) == {"call-726", "good-girl"}
        and not provider_stage_imports
    )
    any_case_import_mode = bool(completed_case_imports or provider_stage_imports)
    if any_case_import_mode and not (v7_import_mode or v8_import_mode):
        raise ValueError(
            "case import mode must be v7 (call-726 full + good-girl stages) "
            "or v8 (call-726 + good-girl full; q0030 fresh)"
        )
    if v7_import_mode and (
        set(completed_case_imports) != {"call-726"}
        or set(provider_stage_imports) != {"good-girl"}
    ):
        raise ValueError(
            "v7 import mode requires full call-726 and provider-stage good-girl imports"
        )
    plugin_root = Path(__file__).resolve().parents[1]
    call_root = dev_root / "experiments" / "masked-call-726"
    good_root = dev_root / "eccr_cases" / "good_girl"
    q_root = dev_root / "experiments" / "layered-three-case" / "fixtures" / "q0030"
    required = [
        config_path,
        call_root / "call_r4.json",
        call_root / "messages.jsonl",
        call_root / "graph_r4.db",
        call_root / "candidates.json",
        call_root / "surface-ab-v1-input" / "case.json",
        good_root / "case.json",
        good_root / "evidence_packet.json",
        q_root / "case.json",
        q_root / "evidence_packet.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("three-case runner inputs missing: " + ", ".join(missing))

    prepared_726 = output_root / "prepared-input" / "call-726"
    case_726 = output_root / "cases" / "call-726"
    case_good = output_root / "cases" / "good-girl"
    case_q = output_root / "cases" / "q0030"
    result_726 = case_726 / "result.private.json"
    result_good = case_good / "result.private.json"
    result_q = case_q / "result.private.json"

    commands: list[dict[str, Any]] = [
        {
            "phase": "prepare_frozen_packet_call_726",
            "provider_calls_upper_bound": 0,
            "argv": _module_command(
                "scripts.layered_case_generation",
                "prepare-masked-packet",
                "--call",
                call_root / "call_r4.json",
                "--messages",
                call_root / "messages.jsonl",
                "--base-db",
                call_root / "graph_r4.db",
                "--candidates",
                call_root / "candidates.json",
                "--output-dir",
                prepared_726,
                "--resume",
            ),
        },
        {
            "phase": "production_l3_and_surface_call_726",
            "provider_calls_upper_bound": 4,
            "argv": _module_command(
                "scripts.layered_case_generation",
                "generate",
                "--case",
                call_root / "call_r4.json",
                "--evidence-packet",
                prepared_726 / "evidence_packet.json",
                "--surface-case-template",
                call_root / "surface-ab-v1-input" / "case.json",
                "--database",
                prepared_726 / "scope.db",
                "--config",
                config_path,
                "--subconscious-provider-id",
                args.subconscious_provider_id,
                "--main-provider-id",
                args.main_provider_id,
                "--route",
                "l3",
                "--output-dir",
                case_726,
                "--max-provider-calls",
                4,
                "--max-output-tokens",
                args.max_output_tokens,
                "--surface-max-output-tokens",
                args.surface_max_output_tokens,
                "--deadline-seconds",
                args.deadline_seconds,
                "--l3-max-model-calls",
                3,
                "--l3-max-retrieval-rounds",
                2,
                "--resume",
                "--authorize-provider-calls",
            ),
        },
        {
            "phase": "production_l3_and_surface_good_girl",
            "provider_calls_upper_bound": 4,
            "argv": _module_command(
                "scripts.layered_case_generation",
                "generate",
                "--case",
                good_root / "case.json",
                "--evidence-packet",
                good_root / "evidence_packet.json",
                "--config",
                config_path,
                "--subconscious-provider-id",
                args.subconscious_provider_id,
                "--main-provider-id",
                args.main_provider_id,
                "--route",
                "l3",
                "--output-dir",
                case_good,
                "--max-provider-calls",
                4,
                "--max-output-tokens",
                args.max_output_tokens,
                "--surface-max-output-tokens",
                args.surface_max_output_tokens,
                "--deadline-seconds",
                args.deadline_seconds,
                "--l3-max-model-calls",
                3,
                "--l3-max-retrieval-rounds",
                0,
                "--resume",
                "--authorize-provider-calls",
            ),
        },
        {
            "phase": "production_l2_and_surface_q0030",
            "provider_calls_upper_bound": 3,
            "argv": _module_command(
                "scripts.layered_case_generation",
                "generate",
                "--case",
                q_root / "case.json",
                "--evidence-packet",
                q_root / "evidence_packet.json",
                "--config",
                config_path,
                "--subconscious-provider-id",
                args.subconscious_provider_id,
                "--main-provider-id",
                args.main_provider_id,
                "--route",
                "l2",
                "--output-dir",
                case_q,
                "--max-provider-calls",
                3,
                "--max-output-tokens",
                args.max_output_tokens,
                "--surface-max-output-tokens",
                args.surface_max_output_tokens,
                "--deadline-seconds",
                args.deadline_seconds,
                "--resume",
                "--authorize-provider-calls",
            ),
        },
    ]
    if v8_import_mode:
        call_plan = completed_case_import_plans["call-726"]
        good_plan = completed_case_import_plans["good-girl"]
        call_packet = Path(str(call_plan["bound_packet_path"]))
        call_database = Path(str(call_plan["bound_database_path"]))
        commands = [
            {
                "phase": "validated_full_case_import_call_726",
                "provider_calls_upper_bound": 0,
                "new_provider_calls_upper_bound": 0,
                "limits": {
                    "subconscious_max_output_tokens": int(args.max_output_tokens),
                    "surface_max_output_tokens": int(args.surface_max_output_tokens),
                    "surface_max_chars": 12000,
                },
                "completed_case_import": call_plan,
                "argv": _module_command(
                    "scripts.layered_case_generation",
                    "import-completed-case",
                    "--source-result",
                    completed_case_imports["call-726"],
                    "--source-attempt-id",
                    call_plan["source_attempt_id"],
                    "--source-attempt-manifest",
                    call_plan["source_attempt_manifest_path"],
                    "--source-attempt-manifest-sha256",
                    call_plan["source_attempt_manifest_sha256"],
                    "--target-dir",
                    case_726,
                    "--case",
                    call_root / "call_r4.json",
                    "--evidence-packet",
                    call_packet,
                    "--surface-case-template",
                    call_root / "surface-ab-v1-input" / "case.json",
                    "--database",
                    call_database,
                    "--config",
                    config_path,
                    "--subconscious-provider-id",
                    args.subconscious_provider_id,
                    "--main-provider-id",
                    args.main_provider_id,
                    "--route",
                    "l3",
                    "--max-provider-calls",
                    4,
                    "--max-output-tokens",
                    args.max_output_tokens,
                    "--surface-max-output-tokens",
                    args.surface_max_output_tokens,
                    "--deadline-seconds",
                    args.deadline_seconds,
                    "--l3-max-model-calls",
                    3,
                    "--l3-max-retrieval-rounds",
                    2,
                    "--surface-max-chars",
                    12000,
                    "--commit",
                ),
            },
            {
                "phase": "validated_full_case_import_good_girl",
                "provider_calls_upper_bound": 0,
                "new_provider_calls_upper_bound": 0,
                "limits": {
                    "subconscious_max_output_tokens": int(args.max_output_tokens),
                    "surface_max_output_tokens": int(args.surface_max_output_tokens),
                    "surface_max_chars": 24000,
                },
                "completed_case_import": good_plan,
                "argv": _module_command(
                    "scripts.layered_case_generation",
                    "import-completed-case",
                    "--source-result",
                    completed_case_imports["good-girl"],
                    "--source-attempt-id",
                    good_plan["source_attempt_id"],
                    "--source-attempt-manifest",
                    good_plan["source_attempt_manifest_path"],
                    "--source-attempt-manifest-sha256",
                    good_plan["source_attempt_manifest_sha256"],
                    "--target-dir",
                    case_good,
                    "--case",
                    good_root / "case.json",
                    "--evidence-packet",
                    good_root / "evidence_packet.json",
                    "--config",
                    config_path,
                    "--subconscious-provider-id",
                    args.subconscious_provider_id,
                    "--main-provider-id",
                    args.main_provider_id,
                    "--route",
                    "l3",
                    "--max-provider-calls",
                    4,
                    "--max-output-tokens",
                    args.max_output_tokens,
                    "--surface-max-output-tokens",
                    args.surface_max_output_tokens,
                    "--deadline-seconds",
                    args.deadline_seconds,
                    "--l3-max-model-calls",
                    3,
                    "--l3-max-retrieval-rounds",
                    0,
                    "--surface-max-chars",
                    24000,
                    "--commit",
                ),
            },
            commands[3],
        ]
        q_command = commands[2]
        q_command["argv"].extend(["--surface-max-chars", "24000"])
        q_command["provider_calls_upper_bound"] = 3
        q_command["new_provider_calls_upper_bound"] = 3
        q_command["limits"] = {
            "subconscious_max_output_tokens": int(args.max_output_tokens),
            "surface_max_output_tokens": int(args.surface_max_output_tokens),
            "surface_max_chars": 24000,
        }
    elif v7_import_mode:
        source_result = completed_case_imports["call-726"]
        source_import_plan = completed_case_import_plans["call-726"]
        source_packet_path = Path(source_import_plan["bound_packet_path"])
        source_database_path = Path(source_import_plan["bound_database_path"])
        commands = [
            {
                "phase": "validated_full_case_import_call_726",
                "provider_calls_upper_bound": 0,
                "new_provider_calls_upper_bound": 0,
                "limits": {
                    "subconscious_max_output_tokens": int(args.max_output_tokens),
                    "surface_max_output_tokens": int(args.surface_max_output_tokens),
                    "surface_max_chars": 12000,
                },
                "completed_case_import": completed_case_import_plans["call-726"],
                "argv": _module_command(
                    "scripts.layered_case_generation",
                    "import-completed-case",
                    "--source-result",
                    source_result,
                    "--source-attempt-id",
                    source_import_plan["source_attempt_id"],
                    "--source-attempt-manifest",
                    source_import_plan["source_attempt_manifest_path"],
                    "--source-attempt-manifest-sha256",
                    source_import_plan["source_attempt_manifest_sha256"],
                    "--target-dir",
                    case_726,
                    "--case",
                    call_root / "call_r4.json",
                    "--evidence-packet",
                    source_packet_path,
                    "--surface-case-template",
                    call_root / "surface-ab-v1-input" / "case.json",
                    "--database",
                    source_database_path,
                    "--config",
                    config_path,
                    "--subconscious-provider-id",
                    args.subconscious_provider_id,
                    "--main-provider-id",
                    args.main_provider_id,
                    "--route",
                    "l3",
                    "--max-provider-calls",
                    4,
                    "--max-output-tokens",
                    args.max_output_tokens,
                    "--surface-max-output-tokens",
                    args.surface_max_output_tokens,
                    "--deadline-seconds",
                    args.deadline_seconds,
                    "--l3-max-model-calls",
                    3,
                    "--l3-max-retrieval-rounds",
                    2,
                    "--surface-max-chars",
                    12000,
                    "--commit",
                ),
            },
            commands[2],
            commands[3],
        ]
        good_command = commands[1]
        good_command["argv"].extend(
            [
                "--surface-max-chars",
                "24000",
                "--import-provider-stages-checkpoint",
                str(provider_stage_imports["good-girl"]),
            ]
        )
        good_command["provider_calls_upper_bound"] = 1
        good_command["new_provider_calls_upper_bound"] = 1
        good_command["provider_stage_import"] = provider_stage_import_plans[
            "good-girl"
        ]
        good_command["limits"] = {
            "subconscious_max_output_tokens": int(args.max_output_tokens),
            "surface_max_output_tokens": int(args.surface_max_output_tokens),
            "surface_max_chars": 24000,
        }
        q_command = commands[2]
        q_command["argv"].extend(["--surface-max-chars", "24000"])
        q_command["provider_calls_upper_bound"] = 3
        q_command["new_provider_calls_upper_bound"] = 3
        q_command["limits"] = {
            "subconscious_max_output_tokens": int(args.max_output_tokens),
            "surface_max_output_tokens": int(args.surface_max_output_tokens),
            "surface_max_chars": 24000,
        }
    else:
        generation_commands = commands[1:]
        generation_case_keys = ("call-726", "good-girl", "q0030")
        for item, case_key in zip(generation_commands, generation_case_keys):
            item["limits"] = {
                "subconscious_max_output_tokens": int(args.max_output_tokens),
                "surface_max_output_tokens": int(args.surface_max_output_tokens),
            }
            import_path = memory_imports.get(case_key)
            item["memory_checkpoint_import"] = (
                memory_import_plans[case_key]
                if import_path is not None
                else None
            )
            item["new_provider_calls_upper_bound"] = (
                1 if import_path is not None else item["provider_calls_upper_bound"]
            )
            if import_path is not None:
                bound_database_path = memory_import_plans[case_key].get(
                    "bound_database_path"
                )
                if bound_database_path is not None:
                    if "--database" not in item["argv"]:
                        raise ValueError(
                            f"{case_key} import is database-bound but command has no database"
                        )
                    database_index = item["argv"].index("--database") + 1
                    item["argv"][database_index] = str(bound_database_path)
                item["argv"].extend(
                    ["--import-memory-checkpoint", str(import_path)]
                )

    plan = {
        "schema_version": (
            "mr-memory.three-case.run-plan.v3"
            if v8_import_mode
            else (
                "mr-memory.three-case.run-plan.v2"
                if v7_import_mode
                else "mr-memory.three-case.run-plan.v1"
            )
        ),
        "created_at": _utc_now(),
        "provider_calls_upper_bound": sum(
            int(item["provider_calls_upper_bound"]) for item in commands
        ),
        "new_provider_calls_upper_bound": sum(
            int(
                item.get(
                    "new_provider_calls_upper_bound",
                    item["provider_calls_upper_bound"],
                )
            )
            for item in commands
        ),
        "limits": {
            "subconscious_max_output_tokens": int(args.max_output_tokens),
            "surface_max_output_tokens": int(args.surface_max_output_tokens),
        },
        "memory_checkpoint_imports": {
            case_key: memory_import_plans[case_key]
            for case_key in sorted(memory_import_plans)
        },
        "completed_case_imports": {
            case_key: completed_case_import_plans[case_key]
            for case_key in sorted(completed_case_import_plans)
        },
        "provider_stage_imports": {
            case_key: provider_stage_import_plans[case_key]
            for case_key in sorted(provider_stage_import_plans)
        },
        "gold_paths_present_in_provider_commands": any(
            str(gold.resolve()) in {str(argument) for argument in item["argv"]}
            for gold in (
                call_root / "gold_v2.json",
                good_root / "gold.json",
                q_root / "gold.json",
            )
            for item in commands
            if int(item["provider_calls_upper_bound"]) > 0
        ),
        "provider_config_validation": (
            "performed by each billable child before writing an attempted ledger event"
        ),
        "commands": commands,
    }
    if not bool(args.authorize_provider_calls):
        plan["status"] = "DRY_RUN_NOT_EXECUTED"
        return plan

    output_root.mkdir(parents=True, exist_ok=True)
    plan_path = output_root / "run-plan.json"
    if plan_path.exists():
        previous_plan = _object(_load_json(plan_path), field="existing run plan")
        for field in (
            "schema_version",
            "provider_calls_upper_bound",
            "new_provider_calls_upper_bound",
            "limits",
            "memory_checkpoint_imports",
            "completed_case_imports",
            "provider_stage_imports",
            "gold_paths_present_in_provider_commands",
            "provider_config_validation",
            "commands",
        ):
            if previous_plan.get(field) != plan.get(field):
                raise ValueError(f"three-case resume run-plan mismatch: {field}")
        plan = previous_plan
    else:
        _atomic_write_json(plan_path, plan)
    for item in commands:
        subprocess.run(item["argv"], cwd=plugin_root, check=True)

    suite = {
        "schema_version": SUITE_SCHEMA_VERSION,
        "cases": [
            {
                "case_id": "masked-call-726",
                "title": "#726 类魂/首发/口嫌体正直",
                "request_provenance": "observed online /chat call at the frozen cutoff",
                "route": "L3 bounded evidence-closure retrieval",
                "case_path": str(case_726 / "case.input.json"),
                "layer_result_path": str(result_726),
                "layer_ledger_path": str(case_726 / "usage.jsonl"),
                "surface_results_path": str(case_726 / "surface" / "private_results.json"),
                "surface_ledger_path": str(case_726 / "usage.jsonl"),
                "surface_arm_ids": ["production-l3"],
                "gold_path": str(call_root / "gold_v2.json"),
            },
            {
                "case_id": "good-girl-competing-meaning-v1",
                "title": "“好女孩”竞争释义与二次玩梗",
                "request_provenance": "research replay over real anonymized group-chat evidence",
                "route": "L3 fixed-packet counterexample audit",
                "case_path": str(case_good / "case.input.json"),
                "layer_result_path": str(result_good),
                "layer_ledger_path": str(case_good / "usage.jsonl"),
                "surface_results_path": str(case_good / "surface" / "private_results.json"),
                "surface_ledger_path": str(case_good / "usage.jsonl"),
                "surface_arm_ids": ["production-l3"],
                "gold_path": str(good_root / "gold.json"),
            },
            {
                "case_id": "q0030-mujica-yumemita",
                "title": "Mujica 与梦限大关系",
                "request_provenance": "human-approved research query over real anonymized group messages; not an observed /chat call",
                "route": "L2 blind BM25 neighborhood + fixed-packet reader",
                "case_path": str(case_q / "case.input.json"),
                "layer_result_path": str(result_q),
                "layer_ledger_path": str(case_q / "usage.jsonl"),
                "surface_results_path": str(case_q / "surface" / "private_results.json"),
                "surface_ledger_path": str(case_q / "usage.jsonl"),
                "surface_arm_ids": ["production-l2"],
                "gold_path": str(q_root / "gold.json"),
            },
        ],
    }
    suite_path = output_root / "suite.json"
    _atomic_write_json(suite_path, suite)
    aggregate = build_report(suite_path, output_root / "report")
    return {
        **plan,
        "status": "COMPLETED",
        "suite_path": str(suite_path),
        "aggregate": aggregate,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and aggregate the three-case layered-memory experiment."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-surface")
    prepare.add_argument("--memory-case", required=True)
    prepare.add_argument("--surface-case-template")
    prepare.add_argument("--layer-result")
    prepare.add_argument("--surface-case-id")
    prepare.add_argument("--arm-id", default="layered-memory")
    prepare.add_argument("--arm-label", default="Layered memory")
    prepare.add_argument("--output-dir", required=True)
    prepare.set_defaults(handler=_prepare_surface_command)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--suite", required=True)
    aggregate.add_argument("--output-dir", required=True)
    aggregate.set_defaults(handler=_aggregate_command)

    run = subparsers.add_parser("run")
    run.add_argument("--dev-root", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--output-root", required=True)
    run.add_argument(
        "--subconscious-provider-id", default="deepseek/deepseek-v4-flash"
    )
    run.add_argument("--main-provider-id", default="openai/gemini-3.5-flash")
    run.add_argument("--max-output-tokens", type=int, default=384000)
    run.add_argument("--surface-max-output-tokens", type=int, default=65536)
    run.add_argument("--deadline-seconds", type=float, default=600.0)
    run.add_argument(
        "--import-memory-checkpoint",
        action="append",
        default=[],
        metavar="CASE=PATH",
        help=(
            "Reuse one validated COMPLETED memory checkpoint in a new suite output. "
            "Repeat for call-726, good-girl, or q0030."
        ),
    )
    run.add_argument(
        "--import-completed-case",
        action="append",
        default=[],
        metavar="CASE=PATH",
        help=(
            "Import a fully validated completed case without any new Provider "
            "call. v8 accepts call-726 and good-girl result.private.json files."
        ),
    )
    run.add_argument(
        "--import-provider-stages-checkpoint",
        action="append",
        default=[],
        metavar="CASE=PATH",
        help=(
            "Replay already-paid, exactly reproducible hashed stages into a new "
            "case. Legacy v7 accepts good-girl; non-reconstructible q0030 stages "
            "are rejected rather than weakening provenance."
        ),
    )
    run.add_argument(
        "--authorize-provider-calls",
        action="store_true",
        help="Without this flag, only print the exact dry-run plan.",
    )
    run.set_defaults(handler=_run_suite_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = args.handler(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
