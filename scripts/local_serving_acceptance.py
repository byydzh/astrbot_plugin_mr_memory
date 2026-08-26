from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mr_memory.local_serving import (  # noqa: E402
    LOCAL_SERVING_SCHEMA_VERSION,
    LocalServingEnvelope,
    compile_local_serving_envelope,
)
from mr_memory.runtime import (  # noqa: E402
    MaterializedReconstruction,
    materialize_reconstruction_packet,
)


CASE_KEYS = ("call-726", "good-girl", "q0030")
EXPECTED_CASE_IDS = {
    "call-726": "masked-call-726-r4",
    "good-girl": "good-girl-competing-meaning-v1",
    "q0030": "q0030-mujica-yumemita",
}
REPORT_SCHEMA_VERSION = "mr-memory.local-serving-acceptance.v1"
CASE_REPORT_SCHEMA_VERSION = "mr-memory.local-serving-acceptance.case.v1"


@dataclass(frozen=True, slots=True)
class FrozenServingCase:
    case_key: str
    case_dir: Path
    case_input: dict[str, Any]
    original_packet: dict[str, Any]
    serving_packet: dict[str, Any]
    adapter_mode: str
    evidence_policy: dict[str, Any]
    fixture_source_keys: tuple[str, ...]
    episode_source_groups: tuple[tuple[str, ...], ...]
    load_ms: float
    adapt_ms: float

    @property
    def query(self) -> str:
        return str(self.case_input["query"])


@dataclass(frozen=True, slots=True)
class CaseAcceptanceResult:
    frozen: FrozenServingCase
    materialized: MaterializedReconstruction
    envelope: LocalServingEnvelope
    envelope_value: dict[str, Any]
    metrics: dict[str, Any]
    assertions: tuple[str, ...]
    warm_samples_ms: tuple[float, ...]


def _load_object(path: Path, *, field: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{field} must contain one JSON object: {path}")
    return value


def _nonempty_text(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _ordered_unique(values: Iterable[object]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _record_source_keys(value: object) -> tuple[str, ...]:
    found: list[str] = []

    def visit(item: object) -> None:
        if isinstance(item, Mapping):
            source_key = str(item.get("source_key") or "").strip()
            if source_key and source_key not in found:
                found.append(source_key)
            for nested in item.values():
                visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)

    visit(value)
    return tuple(found)


def _episode_groups_from_reconstruction_packet(
    packet: Mapping[str, object],
) -> tuple[tuple[str, ...], ...]:
    groups: list[tuple[str, ...]] = []
    raw_episodes = packet.get("expanded_episodes")
    for episode in raw_episodes if isinstance(raw_episodes, list) else []:
        if not isinstance(episode, Mapping):
            continue
        messages = episode.get("messages")
        sources = _ordered_unique(
            message.get("source_key")
            for message in (messages if isinstance(messages, list) else [])
            if isinstance(message, Mapping)
        )
        if sources:
            groups.append(sources)
    return tuple(groups)


def _fixture_role(value: object) -> str:
    role = str(value or "").strip().casefold()
    mapping = {
        "human": "USER",
        "anonymized_group_member": "USER",
        "assistant": "BOT",
    }
    if role not in mapping:
        raise ValueError(f"unsupported frozen fixture actor_role: {role!r}")
    return mapping[role]


def _verbatim_episode_transcript(messages: Sequence[Mapping[str, object]]) -> str:
    """Encode only fixture fields; no interpretation or semantic summary."""

    rows = [
        {
            "speaker_label": str(message.get("sender_name") or ""),
            "role": str(message.get("role") or ""),
            "sent_at": int(message.get("sent_at") or 0),
            "text": str(message.get("plain_text") or ""),
        }
        for message in messages
    ]
    return json.dumps(
        {"kind": "verbatim_transcript", "messages": rows},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _adapt_fixed_packet(
    packet: Mapping[str, object],
    *,
    case_key: str,
    cutoff_at: int,
    umo: str,
) -> tuple[dict[str, Any], tuple[str, ...], tuple[tuple[str, ...], ...]]:
    raw_messages = packet.get("messages")
    raw_episodes = packet.get("episodes")
    evidence_policy = packet.get("evidence_policy")
    if not isinstance(raw_messages, list) or not isinstance(raw_episodes, list):
        raise ValueError(f"{case_key} fixed packet lacks messages/episodes")
    if not isinstance(evidence_policy, Mapping):
        raise ValueError(f"{case_key} fixed packet lacks evidence_policy")

    message_index: dict[str, Mapping[str, object]] = {}
    message_order: list[str] = []
    for raw in raw_messages:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{case_key} message must be an object")
        source_key = _nonempty_text(
            raw.get("source_key"), field=f"{case_key}.message.source_key"
        )
        if source_key in message_index:
            raise ValueError(f"{case_key} duplicate source_key: {source_key}")
        sent_at = int(raw.get("sent_at") or 0)
        if sent_at <= 0 or sent_at >= cutoff_at:
            raise ValueError(
                f"{case_key} source {source_key!r} is not strictly before cutoff"
            )
        source_umo = str(raw.get("umo") or "")
        if source_umo and source_umo != umo:
            raise ValueError(f"{case_key} source {source_key!r} crosses scope")
        message_index[source_key] = raw
        message_order.append(source_key)

    stable_anonymous_accounts = bool(
        evidence_policy.get("anonymized_speakers_are_stable_accounts")
    )
    expanded_episodes: list[dict[str, Any]] = []
    groups: list[tuple[str, ...]] = []
    referenced: list[str] = []
    for index, raw_episode in enumerate(raw_episodes, start=1):
        if not isinstance(raw_episode, Mapping):
            raise ValueError(f"{case_key} episode must be an object")
        source_values = raw_episode.get("source_keys")
        if not isinstance(source_values, list):
            raise ValueError(f"{case_key} episode source_keys must be an array")
        source_keys = _ordered_unique(source_values)
        if not source_keys:
            raise ValueError(f"{case_key} episode cannot be empty")
        missing = [key for key in source_keys if key not in message_index]
        if missing:
            raise ValueError(
                f"{case_key} episode references missing sources: {', '.join(missing)}"
            )
        normalized_messages: list[dict[str, object]] = []
        for source_key in source_keys:
            raw = message_index[source_key]
            actor_role = str(raw.get("actor_role") or "").strip().casefold()
            participant_token = str(
                raw.get("sender_participant_key") or ""
            ).strip()
            speaker_label = str(raw.get("speaker_label") or "").strip()
            # q0030 explicitly says anonymized speaker labels are not stable
            # accounts. Preserve the visible label without manufacturing identity.
            sender_id = (
                participant_token
                if actor_role != "anonymized_group_member"
                or stable_anonymous_accounts
                else ""
            )
            normalized_messages.append(
                {
                    "source_key": source_key,
                    "sent_at": int(raw.get("sent_at") or 0),
                    "sender_id": sender_id,
                    "sender_name": speaker_label or participant_token,
                    "role": _fixture_role(actor_role),
                    "plain_text": str(raw.get("plain_text") or ""),
                }
            )
            if source_key not in referenced:
                referenced.append(source_key)
        groups.append(source_keys)
        expanded_episodes.append(
            {
                "id": index,
                "started_at": min(
                    int(message["sent_at"]) for message in normalized_messages
                ),
                "ended_at": max(
                    int(message["sent_at"]) for message in normalized_messages
                ),
                "title": f"frozen-evidence-group-{index:02d}",
                "summary": _verbatim_episode_transcript(normalized_messages),
                "keywords": [],
                "messages": normalized_messages,
            }
        )

    unreferenced = [key for key in message_order if key not in referenced]
    if unreferenced:
        raise ValueError(
            f"{case_key} packet contains messages outside every episode: "
            + ", ".join(unreferenced)
        )
    serving_packet = {
        "host_notice": (
            "Frozen private acceptance packet. Episode summaries are mechanical "
            "verbatim transcripts, not semantic conclusions."
        ),
        "evidence_policy": dict(evidence_policy),
        "fixture_provenance": dict(packet.get("retrieval") or {}),
        "candidates": {},
        "expanded_episodes": expanded_episodes,
        "semantic_evidence": [],
        "feedback_hypothesis_evidence": [],
        "source_count": len(message_order),
    }
    return serving_packet, tuple(message_order), tuple(groups)


def load_private_case(suite_root: str | Path, case_key: str) -> FrozenServingCase:
    if case_key not in CASE_KEYS:
        raise ValueError(f"unsupported private case: {case_key}")
    started = time.perf_counter_ns()
    case_dir = Path(suite_root).resolve() / "cases" / case_key
    paths = {
        "case": case_dir / "case.input.json",
        "packet": case_dir / "evidence.input.json",
        "manifest": case_dir / "manifest.json",
        "result": case_dir / "result.private.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("private acceptance inputs missing: " + ", ".join(missing))
    case_input = _load_object(paths["case"], field=f"{case_key} case input")
    packet = _load_object(paths["packet"], field=f"{case_key} evidence input")
    manifest = _load_object(paths["manifest"], field=f"{case_key} manifest")
    result = _load_object(paths["result"], field=f"{case_key} result")
    load_ms = (time.perf_counter_ns() - started) / 1_000_000

    expected_case_id = EXPECTED_CASE_IDS[case_key]
    if str(case_input.get("case_id") or "") != expected_case_id:
        raise ValueError(f"{case_key} case_id mismatch")
    if str(manifest.get("case_id") or "") != expected_case_id:
        raise ValueError(f"{case_key} manifest case_id mismatch")
    if str(result.get("case_id") or "") != expected_case_id:
        raise ValueError(f"{case_key} result case_id mismatch")
    if str(result.get("status") or "").upper() != "COMPLETED":
        raise ValueError(f"{case_key} private source run is not COMPLETED")
    query = _nonempty_text(case_input.get("query"), field=f"{case_key}.query")
    cutoff_at = int(case_input.get("cutoff_at") or 0)
    umo = _nonempty_text(case_input.get("umo"), field=f"{case_key}.umo")
    if cutoff_at <= 0:
        raise ValueError(f"{case_key}.cutoff_at must be positive")
    packet_query = str(packet.get("query") or "")
    if packet_query and packet_query != query:
        raise ValueError(f"{case_key} packet query differs from case query")

    adapt_started = time.perf_counter_ns()
    if case_key == "call-726":
        required = {"candidates", "expanded_episodes", "semantic_evidence"}
        if not required.issubset(packet):
            raise ValueError("call-726 is not a reconstruction evidence packet")
        serving_packet = packet
        fixture_sources = _record_source_keys(packet)
        groups = _episode_groups_from_reconstruction_packet(packet)
        adapter_mode = "native_reconstruction_packet"
        evidence_policy: dict[str, Any] = {}
        declared = int(packet.get("source_count") or 0)
        if declared and declared != len(fixture_sources):
            raise ValueError(
                f"call-726 source_count mismatch: {declared}!={len(fixture_sources)}"
            )
    else:
        serving_packet, fixture_sources, groups = _adapt_fixed_packet(
            packet,
            case_key=case_key,
            cutoff_at=cutoff_at,
            umo=umo,
        )
        adapter_mode = "fixed_packet_verbatim_adapter"
        raw_policy = packet.get("evidence_policy")
        assert isinstance(raw_policy, Mapping)
        evidence_policy = dict(raw_policy)
        if serving_packet.get("evidence_policy") != evidence_policy:
            raise AssertionError(f"{case_key} evidence_policy was not preserved")
    adapt_ms = (time.perf_counter_ns() - adapt_started) / 1_000_000
    if not fixture_sources or not groups:
        raise ValueError(f"{case_key} contains no source-backed episodes")

    return FrozenServingCase(
        case_key=case_key,
        case_dir=case_dir,
        case_input=case_input,
        original_packet=packet,
        serving_packet=serving_packet,
        adapter_mode=adapter_mode,
        evidence_policy=evidence_policy,
        fixture_source_keys=fixture_sources,
        episode_source_groups=groups,
        load_ms=load_ms,
        adapt_ms=adapt_ms,
    )


def _percentile_nearest_rank(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil(float(quantile) * len(ordered)))
    return ordered[min(len(ordered), rank) - 1]


def _source_alias_references(value: object) -> tuple[str, ...]:
    found: list[str] = []

    def visit(item: object) -> None:
        if isinstance(item, Mapping):
            for raw_key, nested in item.items():
                key = str(raw_key)
                if (key == "source_id" or key.endswith("_source_id")) and isinstance(
                    nested, str
                ):
                    if nested and nested not in found:
                        found.append(nested)
                elif (key == "source_ids" or key.endswith("_source_ids")) and isinstance(
                    nested, list
                ):
                    for raw in nested:
                        alias = str(raw or "")
                        if alias and alias not in found:
                            found.append(alias)
                else:
                    visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)

    visit(value)
    return tuple(found)


def _contains_true_truncation_marker(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).endswith("_truncated") and nested is True:
                return True
            if _contains_true_truncation_marker(nested):
                return True
    elif isinstance(value, list):
        return any(_contains_true_truncation_marker(item) for item in value)
    return False


def _coverage(selected: Iterable[str], eligible: Iterable[str]) -> float:
    selected_set = set(selected)
    eligible_set = set(eligible)
    if not eligible_set:
        return 1.0
    return len(selected_set & eligible_set) / len(eligible_set)


def _episode_coverage(
    selected: Iterable[str], groups: Sequence[Sequence[str]]
) -> tuple[int, int, float]:
    selected_set = set(selected)
    covered = sum(1 for group in groups if selected_set.intersection(group))
    total = len(groups)
    return covered, total, (covered / total if total else 1.0)


def _validate_envelope(
    frozen: FrozenServingCase,
    materialized: MaterializedReconstruction,
    envelope: LocalServingEnvelope,
    value: Mapping[str, object],
    *,
    max_chars: int,
) -> tuple[str, ...]:
    passed: list[str] = []

    def check(condition: bool, name: str, detail: str) -> None:
        if not condition:
            raise AssertionError(f"{frozen.case_key}: {name}: {detail}")
        passed.append(name)

    check(
        value.get("schema_version") == LOCAL_SERVING_SCHEMA_VERSION,
        "schema_version",
        "unexpected local serving schema",
    )
    check(
        value.get("operational_status") == "COMPLETED",
        "operational_status",
        "local serving was not completed",
    )
    check(envelope.usable, "semantic_status", "private case produced no usable evidence")
    check(
        len(envelope.json_text) <= max_chars,
        "character_budget",
        "envelope exceeds requested max_chars",
    )

    retrieval = value.get("retrieval")
    check(isinstance(retrieval, Mapping), "retrieval_object", "retrieval is missing")
    assert isinstance(retrieval, Mapping)
    check(
        retrieval.get("memory_provider_calls") == 0,
        "memory_provider_calls_zero",
        "local serving recorded a memory provider call",
    )
    check(
        retrieval.get("memory_provider_tokens") == 0,
        "memory_provider_tokens_zero",
        "local serving recorded provider tokens",
    )
    check(
        retrieval.get("memory_provider_external_api_cost") == 0,
        "memory_provider_cost_zero",
        "local serving recorded external API cost",
    )
    check(
        retrieval.get("main_model_incremental_cost") == "UNKNOWN_NOT_MEASURED",
        "main_model_cost_not_conflated",
        "the envelope runner must not claim main-model cost",
    )
    check(
        int(retrieval.get("source_count") or -1) == len(envelope.source_keys),
        "source_count_exact",
        "retrieval source_count differs from returned source_keys",
    )

    fixture_sources = set(frozen.fixture_source_keys)
    check(
        set(envelope.source_keys).issubset(fixture_sources),
        "source_subset",
        "envelope contains a source outside the frozen packet",
    )
    records = value.get("source_records")
    check(isinstance(records, list), "source_records_array", "source_records is missing")
    assert isinstance(records, list)
    record_ids = [
        str(item.get("id") or "")
        for item in records
        if isinstance(item, Mapping)
    ]
    check(
        len(record_ids) == len(envelope.source_keys) == len(set(record_ids)),
        "source_alias_bijection",
        "source aliases are not one-to-one with source_keys",
    )
    references = set(_source_alias_references(value))
    check(
        references.issubset(set(record_ids)),
        "source_alias_resolution",
        "an envelope source_id has no source_record",
    )
    leaked = [key for key in envelope.source_keys if key in envelope.json_text]
    check(not leaked, "raw_source_keys_hidden", "raw source keys leaked into JSON")

    expected_truncated = bool(
        len(envelope.source_keys) < len(set(materialized.source_keys))
        or _contains_true_truncation_marker(value)
    )
    check(
        bool(retrieval.get("truncated")) == envelope.truncated,
        "truncation_contract",
        "dataclass and JSON truncation flags differ",
    )
    check(
        not expected_truncated or envelope.truncated,
        "truncation_honest",
        "sources or fields were omitted without a truncation flag",
    )

    covered, total, _ = _episode_coverage(
        envelope.source_keys, frozen.episode_source_groups
    )
    check(
        covered == total,
        "episode_stratification",
        f"only {covered}/{total} frozen episodes retained at least one source",
    )
    constraints = " ".join(
        str(item) for item in value.get("constraints", [])
    ).casefold()
    check(
        "time adjacency is not reply" in constraints,
        "adjacency_policy",
        "time adjacency safeguard is absent",
    )
    check(
        "bot/assistant text is not independent truth" in constraints,
        "assistant_truth_policy",
        "assistant truth safeguard is absent",
    )
    check(
        "anonymous/name-matched speakers are not account identity" in constraints,
        "anonymous_identity_policy",
        "anonymous identity safeguard is absent",
    )
    if frozen.evidence_policy:
        check(
            frozen.serving_packet.get("evidence_policy") == frozen.evidence_policy,
            "evidence_policy_preserved",
            "fixed-packet evidence_policy changed during adaptation",
        )
    return tuple(passed)


def run_private_case(
    frozen: FrozenServingCase,
    *,
    max_chars: int = 12000,
    repeat: int = 100,
) -> CaseAcceptanceResult:
    if repeat <= 0:
        raise ValueError("repeat must be positive")
    materialize_started = time.perf_counter_ns()
    materialized = materialize_reconstruction_packet(
        frozen.serving_packet,
        query=frozen.query,
        max_items=12,
    )
    materialize_ms = (time.perf_counter_ns() - materialize_started) / 1_000_000
    compile_started = time.perf_counter_ns()
    envelope = compile_local_serving_envelope(
        frozen.serving_packet,
        materialized,
        request_kind="MEMORY_QUERY",
        max_chars=max_chars,
    )
    compile_ms = (time.perf_counter_ns() - compile_started) / 1_000_000
    value = json.loads(envelope.json_text)
    if not isinstance(value, dict):
        raise AssertionError(f"{frozen.case_key}: envelope must be a JSON object")
    assertions = _validate_envelope(
        frozen,
        materialized,
        envelope,
        value,
        max_chars=max_chars,
    )

    warm_samples: list[float] = []
    for _ in range(repeat):
        started = time.perf_counter_ns()
        warm_materialized = materialize_reconstruction_packet(
            frozen.serving_packet,
            query=frozen.query,
            max_items=12,
        )
        warm_envelope = compile_local_serving_envelope(
            frozen.serving_packet,
            warm_materialized,
            request_kind="MEMORY_QUERY",
            max_chars=max_chars,
        )
        warm_samples.append((time.perf_counter_ns() - started) / 1_000_000)
        if warm_envelope != envelope or warm_materialized != materialized:
            raise AssertionError(
                f"{frozen.case_key}: local serving output is not deterministic"
            )

    envelope_episode_count, episode_count, episode_coverage = _episode_coverage(
        envelope.source_keys, frozen.episode_source_groups
    )
    metrics = {
        "status": "COMPILER_ONLY_COMPLETED",
        "execution_scope": "FROZEN_PACKET_COMPILER_ONLY",
        "schema_version": REPORT_SCHEMA_VERSION,
        "case_key": frozen.case_key,
        "adapter_mode": frozen.adapter_mode,
        "semantic_status": envelope.semantic_status,
        "fixture_sources": len(frozen.fixture_source_keys),
        "materialized_sources": len(set(materialized.source_keys)),
        "envelope_sources": len(envelope.source_keys),
        "fixture_to_materialized_coverage": round(
            _coverage(materialized.source_keys, frozen.fixture_source_keys), 6
        ),
        "fixture_to_envelope_coverage": round(
            _coverage(envelope.source_keys, frozen.fixture_source_keys), 6
        ),
        "materialized_to_envelope_coverage": round(
            _coverage(envelope.source_keys, materialized.source_keys), 6
        ),
        "episodes_covered": envelope_episode_count,
        "episodes_total": episode_count,
        "episode_coverage": round(episode_coverage, 6),
        "envelope_chars": len(envelope.json_text),
        "truncated": envelope.truncated,
        "load_ms": round(frozen.load_ms, 3),
        "adapt_ms": round(frozen.adapt_ms, 3),
        "cold_materialize_ms": round(materialize_ms, 3),
        "cold_compile_ms": round(compile_ms, 3),
        "cold_serving_ms": round(materialize_ms + compile_ms, 3),
        "warm_repeats": repeat,
        "warm_p50_ms": round(statistics.median(warm_samples), 3),
        "warm_p95_ms": round(_percentile_nearest_rank(warm_samples, 0.95), 3),
        "memory_provider_calls": 0,
        "memory_provider_input_tokens": 0,
        "memory_provider_output_tokens": 0,
        "memory_provider_external_api_cost": 0,
        "main_model_calls": "NOT_RUN",
        "main_model_tokens": "UNKNOWN_NOT_RUN",
        "main_model_cost": "UNKNOWN_NOT_MEASURED",
        "total_provider_calls": "UNKNOWN_NOT_RUN",
        "total_tokens": "UNKNOWN_NOT_RUN",
        "total_external_api_cost": "UNKNOWN_NOT_MEASURED",
        "deterministic": True,
        "assertion_count": len(assertions),
    }
    return CaseAcceptanceResult(
        frozen=frozen,
        materialized=materialized,
        envelope=envelope,
        envelope_value=value,
        metrics=metrics,
        assertions=assertions,
        warm_samples_ms=tuple(warm_samples),
    )


def _case_private_payload(result: CaseAcceptanceResult) -> dict[str, Any]:
    source_records = result.envelope_value.get("source_records")
    source_records = source_records if isinstance(source_records, list) else []
    aliases = [
        str(item.get("id") or "")
        for item in source_records
        if isinstance(item, Mapping)
    ]
    return {
        "schema_version": CASE_REPORT_SCHEMA_VERSION,
        "case_key": result.frozen.case_key,
        "case_id": result.frozen.case_input["case_id"],
        "query": result.frozen.query,
        "cutoff_at": int(result.frozen.case_input["cutoff_at"]),
        "adapter_mode": result.frozen.adapter_mode,
        "adapter_semantics": (
            "native_reconstruction_packet_may_contain_precompiled_semantics"
            if result.frozen.adapter_mode == "native_reconstruction_packet"
            else "verbatim_transcript_only_no_semantic_summary"
        ),
        "evidence_policy": result.frozen.evidence_policy,
        "fixture_boundary": {
            "end_to_end_retrieval_claim": result.frozen.original_packet.get(
                "end_to_end_retrieval_claim"
            ),
            "diagnostic_type": result.frozen.original_packet.get("diagnostic_type"),
            "this_run_replayed_frozen_packet": True,
            "this_run_reran_candidate_retrieval": False,
        },
        "envelope": result.envelope_value,
        "private_provenance": {
            "fixture_source_keys": list(result.frozen.fixture_source_keys),
            "materialized_source_keys": list(result.materialized.source_keys),
            "envelope_source_keys": list(result.envelope.source_keys),
            "source_alias_map": dict(zip(aliases, result.envelope.source_keys)),
            "episode_source_groups": [
                list(group) for group in result.frozen.episode_source_groups
            ],
        },
        "metrics": result.metrics,
        "assertions_passed": list(result.assertions),
        "cost_boundary": {
            "execution_scope": "FROZEN_PACKET_COMPILER_ONLY",
            "memory_provider_calls": 0,
            "memory_provider_input_tokens": 0,
            "memory_provider_output_tokens": 0,
            "memory_provider_external_api_cost": 0,
            "main_model_not_run_by_this_runner": True,
            "total_cost_unknown_because_main_model_not_run": True,
            "historical_fixture_construction_cost_excluded": True,
        },
    }


def _atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_suite(
    suite_root: str | Path,
    *,
    output_dir: str | Path | None,
    max_chars: int = 12000,
    repeat: int = 100,
    case_keys: Sequence[str] = CASE_KEYS,
) -> tuple[CaseAcceptanceResult, ...]:
    if tuple(case_keys) != CASE_KEYS:
        raise ValueError("acceptance suite must run call-726, good-girl, q0030 in order")
    results = tuple(
        run_private_case(
            load_private_case(suite_root, case_key),
            max_chars=max_chars,
            repeat=repeat,
        )
        for case_key in case_keys
    )
    if output_dir is not None:
        target = Path(output_dir).resolve()
        if target.exists():
            raise FileExistsError(f"private output directory already exists: {target}")
        target.mkdir(parents=True)
        for result in results:
            _atomic_write_json(
                target / f"{result.frozen.case_key}.private.json",
                _case_private_payload(result),
            )
        aggregate_samples = [
            sample for result in results for sample in result.warm_samples_ms
        ]
        _atomic_write_json(
            target / "summary.private.json",
            {
                "schema_version": REPORT_SCHEMA_VERSION,
                "status": "COMPILER_ONLY_COMPLETED",
                "execution_scope": "FROZEN_PACKET_COMPILER_ONLY",
                "cases": [result.metrics for result in results],
                "aggregate": {
                    "case_count": len(results),
                    "warm_samples": len(aggregate_samples),
                    "warm_p50_ms": round(statistics.median(aggregate_samples), 3),
                    "warm_p95_ms": round(
                        _percentile_nearest_rank(aggregate_samples, 0.95), 3
                    ),
                    "memory_provider_calls": 0,
                    "memory_provider_input_tokens": 0,
                    "memory_provider_output_tokens": 0,
                    "memory_provider_external_api_cost": 0,
                    "main_model_calls": "NOT_RUN",
                    "main_model_tokens": "UNKNOWN_NOT_RUN",
                    "total_provider_calls": "UNKNOWN_NOT_RUN",
                    "total_tokens": "UNKNOWN_NOT_RUN",
                    "total_external_api_cost": "UNKNOWN_NOT_MEASURED",
                },
            },
        )
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the frozen #726 / good-girl / q0030 packets through the "
            "provider-free local serving compiler. Private text is written only "
            "to --output-dir; stdout contains metrics."
        )
    )
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-chars", type=int, default=12000)
    parser.add_argument("--repeat", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.repeat < 100:
        raise ValueError("CLI acceptance requires at least 100 warm repeats")
    results = run_suite(
        args.suite,
        output_dir=args.output_dir,
        max_chars=args.max_chars,
        repeat=args.repeat,
    )
    for result in results:
        print(json.dumps(result.metrics, ensure_ascii=False, separators=(",", ":")))
    aggregate_samples = [
        sample for result in results for sample in result.warm_samples_ms
    ]
    print(
        json.dumps(
            {
                "schema_version": REPORT_SCHEMA_VERSION,
                "status": "COMPILER_ONLY_COMPLETED",
                "execution_scope": "FROZEN_PACKET_COMPILER_ONLY",
                "case_count": len(results),
                "warm_samples": len(aggregate_samples),
                "warm_p50_ms": round(statistics.median(aggregate_samples), 3),
                "warm_p95_ms": round(
                    _percentile_nearest_rank(aggregate_samples, 0.95), 3
                ),
                "memory_provider_calls": 0,
                "memory_provider_input_tokens": 0,
                "memory_provider_output_tokens": 0,
                "memory_provider_external_api_cost": 0,
                "main_model_calls": "NOT_RUN",
                "main_model_tokens": "UNKNOWN_NOT_RUN",
                "total_provider_calls": "UNKNOWN_NOT_RUN",
                "total_tokens": "UNKNOWN_NOT_RUN",
                "total_external_api_cost": "UNKNOWN_NOT_MEASURED",
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
