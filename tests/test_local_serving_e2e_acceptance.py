from __future__ import annotations

import ast
import asyncio
import copy
import json
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from mr_memory.certificate import parse_evidence_certificate
from mr_memory.identity import build_request_identity_context
from mr_memory.models import NormalizedMessage
from mr_memory.reader import (
    L2_READER_PROTOCOL,
    build_l2_reader_prompt,
    parse_l2_reader_response,
)
from mr_memory.snapshot import (
    DataRevisionVector,
    InferenceRevisionVector,
    RequestSnapshot,
    stable_sha256,
)
from mr_memory.surface import (
    SURFACE_SCHEMA_VERSION,
    compile_surface_packet,
    validate_surface_packet,
)
from scripts.local_serving_e2e_acceptance import provider_turn_report


QUERY = "/chat 回忆参与者甲的作品态度变化"
SOURCE_KEYS = {"source-old", "source-new", "source-candidate-only"}
SURFACE_SOURCE_KEYS = {"source-old", "source-new"}
PARTICIPANT_KEYS = {"participant:a", "participant:asker"}
MAIN_SOURCE = (Path(__file__).resolve().parents[1] / "main.py").read_text(
    encoding="utf-8"
)
MAIN_TREE = ast.parse(MAIN_SOURCE)


def main_method(name: str, **namespace: object):
    original = next(
        node
        for node in ast.walk(MAIN_TREE)
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == name
    )
    node = copy.deepcopy(original)
    node.decorator_list = []
    node.returns = None
    for argument in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
        argument.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    values = dict(namespace)
    exec(compile(module, f"<main.py:{name}>", "exec"), values)
    return values[name]


@dataclass
class LocalOutcome:
    operational_status: str
    semantic_status: str = "UNKNOWN"
    envelope_text: str = ""
    run_id: str = ""
    detail: str = ""
    elapsed_ms: float = 0.0
    source_keys: tuple[str, ...] = ()
    selected_edge_ids: tuple[int, ...] = ()
    selected_hypothesis_ids: tuple[int, ...] = ()
    truncated: bool = False
    ledger_result: dict[str, object] | None = None

    @property
    def usable(self) -> bool:
        return bool(
            self.envelope_text
            and self.operational_status == "COMPLETED"
            and self.semantic_status in {"CERTIFIED", "PARTIAL", "SAFETY_ABSTAIN"}
        )


class FakeTextPart:
    def __init__(self, *, text: str):
        self.text = text

    def mark_as_temp(self):
        return self


def synthetic_snapshot() -> RequestSnapshot:
    return RequestSnapshot.create(
        snapshot_id="snapshot-online-e2e",
        umo="bot:GroupMessage:synthetic",
        cutoff_at=2_000,
        message_upper_bound=20,
        request_source_key="source-query",
        sender_participant_key="participant:asker",
        reply_source_key="",
        query=QUERY,
        context={"case": "online-resident-reader"},
        data_revision=DataRevisionVector.from_value(
            {
                "message": 20,
                "deletion": 0,
                "identity": 3,
                "graph": 5,
                "relation": 2,
                "feedback": 1,
            }
        ),
        inference_revision=InferenceRevisionVector.from_value(
            {
                "retriever": "synthetic-full-retrieval",
                "embedding_model": "disabled",
                "fusion_policy": "synthetic-fusion",
                "reader_model": "synthetic-resident-reader",
                "reader_protocol": L2_READER_PROTOCOL,
                "certificate_schema": "evidence-certificate.v2",
                "surface_compiler": SURFACE_SCHEMA_VERSION,
                "route_policy": "resident-one-pass",
            }
        ),
        captured_at=2_001,
    )


def synthetic_packet() -> dict[str, object]:
    return {
        "semantic_evidence": [],
        "expanded_episodes": [
            {
                "title": "合成作品态度记录",
                "summary": "参与者甲先表达保留，后来明确表示愿意尝试。",
                "messages": [
                    {
                        "source_key": "source-old",
                        "sent_at": 1_000,
                        "sender_id": "account-a",
                        "sender_name": "参与者甲",
                        "sender_participant_key": "participant:a",
                        "role": "USER",
                        "plain_text": "我暂时没有兴趣。",
                    },
                    {
                        "source_key": "source-new",
                        "sent_at": 1_500,
                        "sender_id": "account-a",
                        "sender_name": "参与者甲",
                        "sender_participant_key": "participant:a",
                        "role": "USER",
                        "plain_text": "现在愿意试试看。",
                    },
                ],
            }
        ],
        "request_identity_context": {
            "authority": "current_platform_event",
            "sender": {
                "participant_key": "participant:asker",
                "platform_id": "bot",
                "account_id": "asker",
                "display_name": "提问者",
                "binding_basis": "message_sender",
                "same_account_as_sender": True,
            },
            "mentions": [],
            "reply_target": None,
        },
        "query_alias_resolution": {
            "query": QUERY,
            "ambiguous": False,
            "participants": [],
            "ambiguous_aliases": [],
        },
        "participant_activity": [],
        "participant_history": [],
        "feedback_hypothesis_evidence": [
            {
                "hypothesis": {
                    "id": 702,
                    "activation_mode": "always",
                    "prospective_cue": "合成候选偏好",
                },
                "evidence": [{"source_key": "source-new"}],
            }
        ],
        "participant_source_keys": {
            "participant:a": ["source-old", "source-new"],
            "participant:asker": [],
        },
        "retrieval_coverage": {
            "semantic_none_allowed": True,
            "raw_history_complete": True,
            "snapshot_visible_message_total": 2,
            "packet_full_message_sources": 2,
        },
        "reply_context": None,
        # Raw retrieval candidates are visible to the Reader, but their database
        # identifiers are not evidence that the compiled surface presented them.
        "candidates": {
            "associations": [
                {
                    "id": 701,
                    "score": 0.91,
                    "statement": "合成候选关系",
                    "source_keys": ["source-candidate-only"],
                    "epistemic_state": "HYPOTHESIS",
                }
            ]
        },
    }


def synthetic_certificate_response(
    snapshot: RequestSnapshot,
    packet_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": "evidence-certificate.v2",
        "status": "CERTIFIED",
        "scope_snapshot": snapshot.as_dict(),
        "data_revision": snapshot.data_revision.as_dict(),
        "inference_revision": snapshot.inference_revision.as_dict(),
        "packet_sha256": packet_sha256,
        "subjects": [
            {
                "reference": "参与者甲",
                "participant_key": "participant:a",
                "reference_mode": "UNIQUE_ALIAS",
                "candidate_participant_keys": [],
                "source_keys": ["source-old", "source-new"],
                "valid_at": 1_500,
            }
        ],
        "atoms": [
            {
                "id": "attitude_change",
                "statement": "参与者甲的表达从保留变为愿意尝试。",
                "speaker_participant_key": "",
                "subject_participant_key": "participant:a",
                "attribution": "DERIVED_INTERPRETATION",
                "stance": "SUPPORTED",
                "source_keys": ["source-old", "source-new"],
                "source_spans": ["我暂时没有兴趣。", "现在愿意试试看。"],
                "importance": "REQUIRED",
                "confidence": 0.86,
            }
        ],
        "must_include": ["attitude_change"],
        "must_not_upgrade": [
            {
                "observed": "愿意尝试",
                "forbidden": ["已经购买", "已经完成体验"],
                "atom_ids": ["attitude_change"],
                "reason": "尝试意向不等于已经采取行动。",
            }
        ],
        "conflicts": [],
        "unresolved": [],
        "open_obligations": [],
        "stop_reason": "CERTIFIED_CLOSE",
        "validation": {"pack_read_complete": True, "host_validated": True},
    }


_READER_HOST_FIELDS = {
    "schema_version",
    "scope_snapshot",
    "data_revision",
    "inference_revision",
    "packet_sha256",
    "validation",
}


def synthetic_reader_delta(
    certificate: dict[str, object],
) -> dict[str, object]:
    """Encode a model-owned v3 response using the prompt's short ids."""

    source_aliases = {
        source_key: f"s{index}"
        for index, source_key in enumerate(sorted(SOURCE_KEYS), start=1)
    }
    participant_aliases = {
        participant_key: f"p{index}"
        for index, participant_key in enumerate(sorted(PARTICIPANT_KEYS), start=1)
    }

    def aliased(value: object, *, field: str = "") -> object:
        if isinstance(value, dict):
            return {
                str(key): aliased(nested, field=str(key))
                for key, nested in value.items()
                if str(key) != "speaker_participant_key"
            }
        if isinstance(value, list):
            if field == "source_keys" or field.endswith("_source_keys"):
                return [
                    source_aliases.get(item, item) if isinstance(item, str) else item
                    for item in value
                ]
            if field == "participant_keys" or field.endswith("_participant_keys"):
                return [
                    participant_aliases.get(item, item)
                    if isinstance(item, str)
                    else item
                    for item in value
                ]
            return [aliased(item) for item in value]
        if isinstance(value, str):
            if field == "source_key" or field.endswith("_source_key"):
                return source_aliases.get(value, value)
            if field == "participant_key" or field.endswith("_participant_key"):
                return participant_aliases.get(value, value)
        return value

    semantic = {
        key: copy.deepcopy(value)
        for key, value in certificate.items()
        if key not in _READER_HOST_FIELDS
    }
    result = aliased(semantic)
    assert isinstance(result, dict)
    return result


def production_execute_method():
    return main_method(
        "_execute_local_memory_serving",
        asyncio=asyncio,
        time=time,
        json=json,
        _runtime_run_id=lambda phase: f"{phase}-e2e-run",
        _stable_hash=lambda value: stable_sha256(value),
        _LocalMemoryOutcome=LocalOutcome,
        RoutePolicy=lambda **kwargs: SimpleNamespace(**kwargs),
        build_request_identity_context=build_request_identity_context,
        _collect_source_keys=main_method("_collect_source_keys"),
        compile_surface_packet=compile_surface_packet,
        validate_surface_packet=validate_surface_packet,
        L2_READER_PROTOCOL=L2_READER_PROTOCOL,
        DistillationSnapshotChanged=RuntimeError,
        logger=SimpleNamespace(
            error=lambda *args, **kwargs: None,
            exception=lambda *args, **kwargs: None,
        ),
    )


def production_reader_method():
    return main_method(
        "_read_l2_certificate",
        json=json,
        build_l2_reader_prompt=build_l2_reader_prompt,
        parse_l2_reader_response=parse_l2_reader_response,
        stable_sha256=stable_sha256,
    )


def production_inject_method():
    return main_method(
        "inject_subconscious_memory",
        json=json,
        SURFACE_SCHEMA_VERSION=SURFACE_SCHEMA_VERSION,
        TextPart=FakeTextPart,
        logger=SimpleNamespace(
            error=lambda *args, **kwargs: None,
            exception=lambda *args, **kwargs: None,
        ),
        GroupScopeError=ValueError,
    )


class OnlineResidentServingAcceptanceTests(unittest.IsolatedAsyncioTestCase):
    def production_host(
        self,
        *,
        normalized: NormalizedMessage,
        snapshot: RequestSnapshot,
        packet: dict[str, object],
        service: object,
    ) -> SimpleNamespace:
        packet_sha256 = stable_sha256(packet)
        response = SimpleNamespace(
            completion_text=json.dumps(
                synthetic_reader_delta(
                    synthetic_certificate_response(snapshot, packet_sha256)
                ),
                ensure_ascii=False,
            )
        )
        provider = SimpleNamespace(name="synthetic-reader")
        host = SimpleNamespace(
            _inflight_runtime_tasks=set(),
            feedback_learning_enabled=False,
            local_serving_enabled=True,
            # Production uses this only as an abnormal-hang guard.
            local_serving_timeout_seconds=180.0,
            subconscious_provider_id="resident-reader",
            context=SimpleNamespace(
                get_provider_by_id=lambda provider_id: (
                    provider if provider_id == "resident-reader" else None
                )
            ),
            _runtime_initialized=True,
            _runtime_bootstrap_task=None,
            _initialize_runtime=AsyncMock(),
            max_query_chars=4_000,
            local_serving_max_items=12,
            local_serving_max_chars=12_000,
            max_brief_chars=12_000,
            embedding_enabled=False,
            _embedding_preload_complete=True,
            _embedding_preload_error="",
            _active_interaction_traces={},
            _begin_interaction_trace=AsyncMock(),
            _local_serving_guard=lambda event: "",
            _group_scope=lambda event: SimpleNamespace(
                key=normalized.umo,
                storage_id="scope-online-e2e",
            ),
            _service_for_scope=lambda scope: service,
            _normalize_event=lambda event: normalized,
            _runtime_request_kind=lambda query, **kwargs: "MEMORY_QUERY",
            _runtime_activity_analysis=lambda query: False,
            _capture_layered_snapshot=AsyncMock(return_value=snapshot),
            _layered_evidence_packet=AsyncMock(
                return_value=(
                    packet,
                    packet_sha256,
                    set(SOURCE_KEYS),
                    set(PARTICIPANT_KEYS),
                    "LOCAL_FRESH",
                )
            ),
            _assert_snapshot_fresh=AsyncMock(),
            distillation_thinking_mode="enabled",
            local_serving_reader_thinking_mode="disabled",
            _run_fast_reconstruction_with_ledger=AsyncMock(
                return_value=(response, 8.5)
            ),
        )

        reader = production_reader_method()

        async def read_once(**kwargs):
            return await reader(host, **kwargs)

        host._read_l2_certificate = AsyncMock(side_effect=read_once)
        return host

    @staticmethod
    def normalized_message(snapshot: RequestSnapshot) -> NormalizedMessage:
        return NormalizedMessage(
            platform="aiocqhttp",
            platform_id="bot",
            umo=snapshot.umo,
            group_id="synthetic",
            message_id="query",
            sender_id="asker",
            sender_name="提问者",
            sent_at=snapshot.cutoff_at,
            plain_text=QUERY,
            content=[{"type": "plain", "text": QUERY}],
        )

    async def test_fresh_full_retrieval_one_reader_and_surface_injection(self) -> None:
        snapshot = synthetic_snapshot()
        packet = synthetic_packet()
        normalized = self.normalized_message(snapshot)
        service = SimpleNamespace(
            start_experiment=AsyncMock(),
            finish_experiment=AsyncMock(),
            audit_snapshot_sources=AsyncMock(),
            record_reconstruction_step=AsyncMock(),
        )
        host = self.production_host(
            normalized=normalized,
            snapshot=snapshot,
            packet=packet,
            service=service,
        )

        event = SimpleNamespace(message_obj=SimpleNamespace(message_str=QUERY))
        outcome = await production_execute_method()(host, event, QUERY)

        self.assertTrue(outcome.usable)
        self.assertEqual(outcome.operational_status, "COMPLETED")
        self.assertEqual(outcome.semantic_status, "CERTIFIED")
        host._layered_evidence_packet.assert_awaited_once()
        retrieval = host._layered_evidence_packet.await_args.kwargs
        self.assertFalse(retrieval["resolve_query_aliases"])
        self.assertFalse(retrieval["include_participant_activity"])
        self.assertFalse(retrieval["use_cache"])
        self.assertFalse(retrieval["finalize_packet"])

        host._read_l2_certificate.assert_awaited_once()
        reader_request = host._read_l2_certificate.await_args.kwargs
        self.assertFalse(reader_request["allow_l3"])
        self.assertEqual(reader_request["packet"], packet)
        host._run_fast_reconstruction_with_ledger.assert_awaited_once()
        provider_request = host._run_fast_reconstruction_with_ledger.await_args.kwargs
        self.assertEqual(provider_request["phase"], "resident_evidence_reader")
        self.assertEqual(provider_request["usage_source"], "resident_reader_one_pass")
        self.assertIn("resident one-pass", provider_request["system_prompt"])
        self.assertIn("不得返回 REQUEST_L3", provider_request["system_prompt"])

        ledger = outcome.ledger_result
        self.assertEqual(ledger["path"], "resident_reader_one_pass")
        self.assertEqual(ledger["retrieval_mode"], "FRESH")
        self.assertEqual(ledger["cache_layer"], "LOCAL_FRESH")
        self.assertEqual(ledger["memory_provider_calls"], 1)
        self.assertFalse(ledger["repair_attempted"])
        self.assertFalse(ledger["l3_attempted"])
        self.assertEqual(ledger["selected_edge_ids"], [])
        self.assertEqual(ledger["selected_hypothesis_ids"], [])
        self.assertEqual(
            set(ledger["visited_source_keys"]),
            SOURCE_KEYS,
        )
        self.assertEqual(
            set(ledger["presented_source_keys"]),
            SURFACE_SOURCE_KEYS,
        )
        self.assertEqual(set(outcome.source_keys), SURFACE_SOURCE_KEYS)
        self.assertEqual(outcome.selected_edge_ids, ())
        self.assertEqual(outcome.selected_hypothesis_ids, ())
        self.assertFalse(outcome.truncated)
        self.assertEqual(ledger["surface_omitted_optional"], 0)
        self.assertFalse(ledger["surface_truncated"])
        self.assertEqual(
            ledger["surface_injection_status"],
            "COMPILED_NOT_YET_INJECTED",
        )
        self.assertIn("FULL_RETRIEVAL", ledger["stage_elapsed_ms"])
        self.assertIn("RESIDENT_READER", ledger["stage_elapsed_ms"])
        self.assertEqual(service.audit_snapshot_sources.await_count, 2)
        self.assertEqual(
            service.audit_snapshot_sources.await_args.kwargs["source_keys"],
            SOURCE_KEYS,
        )
        self.assertEqual(host._assert_snapshot_fresh.await_count, 2)
        service.record_reconstruction_step.assert_awaited_once()
        reconstruction = service.record_reconstruction_step.await_args.kwargs
        self.assertEqual(reconstruction["tool_name"], "resident_reader_one_pass")
        self.assertEqual(reconstruction["arguments"]["reader_calls"], 1)

        self.assertEqual(service.finish_experiment.await_count, 0)
        request = SimpleNamespace(prompt="", extra_user_content_parts=[])
        host._session_allowed = lambda umo: True
        host._scope_event_carriers = {}
        host._feedback_candidate_ids = {}
        host._local_memory_for_request = AsyncMock(return_value=outcome)
        host._local_serving_injected = set()

        await production_inject_method()(host, event, request)

        self.assertEqual(len(request.extra_user_content_parts), 1)
        injected = request.extra_user_content_parts[0].text
        marker = "<mr_memory_surface>"
        self.assertIn(marker, injected)
        surface_text = injected.split(marker, 1)[1].split(
            "</mr_memory_surface>", 1
        )[0]
        surface = json.loads(surface_text)
        self.assertEqual(surface["schema_version"], SURFACE_SCHEMA_VERSION)
        self.assertEqual(surface["status"], "CERTIFIED")
        self.assertIn(id(event), host._local_serving_injected)
        service.finish_experiment.assert_awaited_once()
        terminal = service.finish_experiment.await_args.kwargs
        self.assertEqual(terminal["status"], "completed")
        self.assertEqual(
            terminal["result"]["surface_injection_status"],
            "INJECTED_IN_REQUEST_HOOK",
        )
        self.assertEqual(terminal["result"]["memory_provider_calls"], 1)

    async def test_surface_optional_omission_is_reported_as_truncation(self) -> None:
        snapshot = synthetic_snapshot()
        packet = synthetic_packet()
        packet_sha256 = stable_sha256(packet)
        raw_certificate = synthetic_certificate_response(snapshot, packet_sha256)
        raw_certificate["atoms"].append(
            {
                "id": "optional_context",
                "statement": "可选的合成背景。" * 80,
                "speaker_participant_key": "participant:a",
                "subject_participant_key": "participant:a",
                "attribution": "DIRECT_SPEAKER_STATEMENT",
                "stance": "SUPPORTED",
                "source_keys": ["source-old"],
                "source_spans": ["我暂时没有兴趣。"],
                "importance": "OPTIONAL",
                "confidence": 0.7,
            }
        )
        certificate = parse_evidence_certificate(
            raw_certificate,
            expected_snapshot=snapshot,
            expected_packet_sha256=packet_sha256,
            allowed_source_keys=SOURCE_KEYS,
            allowed_participant_keys=PARTICIPANT_KEYS,
            pack_read_complete=True,
        )
        full_surface = compile_surface_packet(certificate, max_chars=50_000)
        self.assertEqual(full_surface.omitted_optional, 0)

        normalized = self.normalized_message(snapshot)
        service = SimpleNamespace(
            start_experiment=AsyncMock(),
            finish_experiment=AsyncMock(),
            audit_snapshot_sources=AsyncMock(),
            record_reconstruction_step=AsyncMock(),
        )
        host = self.production_host(
            normalized=normalized,
            snapshot=snapshot,
            packet=packet,
            service=service,
        )
        host.local_serving_max_chars = len(full_surface.text) - 1
        host._run_fast_reconstruction_with_ledger = AsyncMock(
            return_value=(
                SimpleNamespace(
                    completion_text=json.dumps(
                        synthetic_reader_delta(raw_certificate),
                        ensure_ascii=False,
                    )
                ),
                8.5,
            )
        )

        event = SimpleNamespace(message_obj=SimpleNamespace(message_str=QUERY))
        outcome = await production_execute_method()(host, event, QUERY)

        self.assertTrue(outcome.usable)
        self.assertTrue(outcome.truncated)
        self.assertEqual(outcome.ledger_result["surface_omitted_optional"], 1)
        self.assertTrue(outcome.ledger_result["surface_truncated"])
        surface = json.loads(outcome.envelope_text)
        self.assertEqual(surface["omitted_optional"], 1)
        self.assertEqual(surface["evidence"]["optional"], [])

        request = SimpleNamespace(prompt="", extra_user_content_parts=[])
        host._session_allowed = lambda umo: True
        host._scope_event_carriers = {}
        host._feedback_candidate_ids = {}
        host._local_memory_for_request = AsyncMock(return_value=outcome)
        host._local_serving_injected = set()
        await production_inject_method()(host, event, request)
        terminal = service.finish_experiment.await_args.kwargs["result"]
        self.assertEqual(terminal["surface_omitted_optional"], 1)
        self.assertTrue(terminal["surface_truncated"])

    async def test_selected_source_edit_during_reader_fails_closed(self) -> None:
        snapshot = synthetic_snapshot()
        packet = synthetic_packet()
        normalized = self.normalized_message(snapshot)
        first = {
            "source_fingerprints": {
                "source-old": {
                    "message_id": 1,
                    "sent_at": 1_000,
                    "revision_no": 1,
                    "content_sha256": "a" * 64,
                }
            }
        }
        changed = copy.deepcopy(first)
        changed["source_fingerprints"]["source-old"]["revision_no"] = 2
        service = SimpleNamespace(
            start_experiment=AsyncMock(),
            finish_experiment=AsyncMock(),
            audit_snapshot_sources=AsyncMock(side_effect=[first, changed]),
            record_reconstruction_step=AsyncMock(),
        )
        host = self.production_host(
            normalized=normalized,
            snapshot=snapshot,
            packet=packet,
            service=service,
        )

        event = SimpleNamespace(message_obj=SimpleNamespace(message_str=QUERY))
        outcome = await production_execute_method()(host, event, QUERY)

        self.assertFalse(outcome.usable)
        self.assertEqual(outcome.operational_status, "FAILED")
        self.assertIn("selected evidence changed", outcome.detail)
        host._run_fast_reconstruction_with_ledger.assert_awaited_once()
        self.assertEqual(service.audit_snapshot_sources.await_count, 2)

    async def test_180_seconds_is_only_an_abnormal_hang_guard(self) -> None:
        snapshot = synthetic_snapshot()
        packet = synthetic_packet()
        normalized = self.normalized_message(snapshot)
        service = SimpleNamespace(
            start_experiment=AsyncMock(),
            finish_experiment=AsyncMock(),
            audit_snapshot_sources=AsyncMock(),
            record_reconstruction_step=AsyncMock(),
        )
        host = self.production_host(
            normalized=normalized,
            snapshot=snapshot,
            packet=packet,
            service=service,
        )
        self.assertEqual(host.local_serving_timeout_seconds, 180.0)

        # Compress only the wall-clock duration of the synthetic hang. The same
        # production timeout branch remains under test.
        host.local_serving_timeout_seconds = 0.01

        async def hung_full_retrieval(**kwargs):
            await asyncio.sleep(1)
            return (
                packet,
                stable_sha256(packet),
                set(SOURCE_KEYS),
                set(PARTICIPANT_KEYS),
                "NONE",
            )

        host._layered_evidence_packet = AsyncMock(side_effect=hung_full_retrieval)
        event = SimpleNamespace(message_obj=SimpleNamespace(message_str=QUERY))
        outcome = await production_execute_method()(host, event, QUERY)

        self.assertEqual(outcome.operational_status, "FAILED")
        self.assertEqual(outcome.semantic_status, "UNKNOWN")
        self.assertFalse(outcome.envelope_text)
        self.assertIn("abnormal-hang guard", outcome.detail)
        host._read_l2_certificate.assert_not_awaited()
        host._run_fast_reconstruction_with_ledger.assert_not_awaited()
        service.finish_experiment.assert_awaited_once()
        failure = service.finish_experiment.await_args.kwargs
        self.assertEqual(failure["status"], "failed")
        self.assertEqual(failure["result"]["last_stage"], "FULL_RETRIEVAL")
        self.assertIn(
            "FULL_RETRIEVAL_INCOMPLETE",
            failure["result"]["stage_elapsed_ms"],
        )

    def test_provider_ledger_reports_exactly_one_resident_reader_turn(self) -> None:
        common = {
            "request_id": "run-1:resident_evidence_reader:0",
            "run_id": "run-1",
            "arm": "memory",
            "repetition": 1,
            "phase": "resident_evidence_reader",
            "call_index": 0,
            "provider_id": "resident-reader",
            "model": "synthetic-model",
            "thinking": "enabled",
            "max_tokens": 8_192,
            "options_sha256": "options",
            "payload_sha256": "payload",
        }
        report = provider_turn_report(
            [
                {**common, "event": "attempted"},
                {
                    **common,
                    "event": "completed",
                    "usage_present": True,
                    "input_other": 1_000,
                    "input_cached": 200,
                    "input": 1_200,
                    "output": 180,
                    "total": 1_380,
                    "elapsed_ms": 320.5,
                },
            ],
            run_id="run-1",
        )
        self.assertTrue(report["pairwise_closed"])
        self.assertEqual(report["attempted_calls"], 1)
        self.assertEqual(report["completed_calls"], 1)
        self.assertEqual(report["failed_calls"], 0)
        self.assertEqual(report["turns"][0]["phase"], "resident_evidence_reader")
        self.assertEqual(report["measured_completed_total_lower_bound"], 1_380)


if __name__ == "__main__":
    unittest.main()
