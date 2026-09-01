from __future__ import annotations

import ast
import asyncio
import copy
import json
import re
import time
import unittest
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from mr_memory.identity import build_request_identity_context
from mr_memory.local_serving import (
    LOCAL_SERVING_SCHEMA_VERSION,
    compile_local_serving_envelope,
)
from mr_memory.models import NormalizedMessage
from mr_memory.runtime import materialize_reconstruction_packet
from mr_memory.service import MemoryService
from mr_memory.snapshot import stable_sha256
from mr_memory.storage import MemoryStorage
from scripts.local_serving_e2e_acceptance import (
    EXECUTION_SCOPE,
    provider_turn_report,
    release_gate_coverage,
    run_local_serving_case,
)
from tests import test_local_serving as local_serving_fixtures


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


def collect_source_keys(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            if key == "source_key" and isinstance(nested, str) and nested:
                found.add(nested)
            elif isinstance(nested, (dict, list, tuple)):
                found.update(collect_source_keys(nested))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            found.update(collect_source_keys(nested))
    return found


@dataclass
class LocalOutcome:
    operational_status: str
    semantic_status: str = "UNKNOWN"
    envelope_text: str = ""
    run_id: str = ""
    elapsed_ms: float = 0.0
    source_keys: tuple[str, ...] = ()
    selected_edge_ids: tuple[int, ...] = ()
    selected_hypothesis_ids: tuple[int, ...] = ()
    truncated: bool = False
    detail: str = ""
    ledger_result: dict[str, object] | None = None

    @property
    def usable(self) -> bool:
        return bool(
            self.envelope_text
            and self.operational_status == "COMPLETED"
            and self.semantic_status
            in {
                "EVIDENCE_AVAILABLE",
                "IDENTITY_AMBIGUOUS",
                "IDENTITY_UNRESOLVED",
            }
        )


class FakeTextPart:
    def __init__(self, *, text: str):
        self.text = text

    def mark_as_temp(self):
        return self


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
        materialize_reconstruction_packet=materialize_reconstruction_packet,
        compile_local_serving_envelope=compile_local_serving_envelope,
        LOCAL_SERVING_SCHEMA_VERSION=LOCAL_SERVING_SCHEMA_VERSION,
        logger=SimpleNamespace(
            error=lambda *args, **kwargs: None,
            exception=lambda *args, **kwargs: None,
        ),
    )


def production_classifier():
    activity_analysis = main_method("_runtime_activity_analysis")
    explicit_identity_intent = main_method("_explicit_identity_intent", re=re)
    owner = SimpleNamespace(
        _runtime_activity_analysis=activity_analysis,
        _explicit_identity_intent=explicit_identity_intent,
    )
    return (
        main_method("_runtime_request_kind", MrMemoryPlugin=owner),
        activity_analysis,
        main_method("_local_direct_identity_question", MrMemoryPlugin=owner),
    )


async def run_identity_case(
    *,
    storage: MemoryStorage,
    query: str,
    normalized: NormalizedMessage,
    snapshot: object,
) -> dict[str, object]:
    service = MemoryService(storage)
    service.audit_snapshot_sources = AsyncMock()
    host = SimpleNamespace(
        feedback_learning_enabled=False,
        _assert_snapshot_fresh=AsyncMock(),
    )
    identity_packet = main_method(
        "_local_identity_evidence_packet",
        build_request_identity_context=build_request_identity_context,
        _collect_source_keys=collect_source_keys,
        _collect_participant_keys=lambda value: set(),
        stable_sha256=stable_sha256,
    )

    async def build_identity_packet(**kwargs):
        return await identity_packet(host, **kwargs)

    async def forbidden_full_packet(**kwargs):
        raise AssertionError("identity query must not use full retrieval")

    classifier, activity_analysis, identity_question = production_classifier()
    return await run_local_serving_case(
        query=query,
        normalized=normalized,
        snapshot=snapshot,
        service=service,
        classify_request=classifier,
        activity_analysis=activity_analysis,
        identity_question=identity_question,
        reference_question=main_method(
            "_local_direct_reference_question", re=__import__("re")
        ),
        build_identity_packet=build_identity_packet,
        build_full_packet=forbidden_full_packet,
    )


class LocalServingEndToEndAcceptanceTests(unittest.IsolatedAsyncioTestCase):
    def production_host(
        self,
        *,
        normalized: NormalizedMessage,
        service: object,
        snapshot: object,
        packet: dict[str, object],
        timeout: float = 1.0,
    ) -> SimpleNamespace:
        classifier, activity_analysis, identity_question = production_classifier()
        return SimpleNamespace(
            _inflight_runtime_tasks=set(),
            feedback_learning_enabled=False,
            local_serving_enabled=True,
            local_serving_timeout_seconds=timeout,
            _runtime_initialized=True,
            _runtime_bootstrap_task=None,
            _initialize_runtime=AsyncMock(),
            max_query_chars=4000,
            local_serving_max_items=12,
            local_serving_max_chars=12000,
            embedding_enabled=False,
            _embedding_preload_complete=True,
            _embedding_preload_error="",
            _active_interaction_traces={},
            _begin_interaction_trace=AsyncMock(),
            _local_serving_guard=lambda event: "",
            _group_scope=lambda event: SimpleNamespace(
                key=normalized.umo, storage_id="scope-e2e"
            ),
            _service_for_scope=lambda scope: service,
            _normalize_event=lambda event: normalized,
            _runtime_request_kind=classifier,
            _runtime_activity_analysis=activity_analysis,
            _capture_layered_snapshot=AsyncMock(return_value=snapshot),
            _local_serving_inference_revision=lambda: {},
            _local_direct_identity_question=identity_question,
            _local_direct_reference_question=main_method(
                "_local_direct_reference_question", re=__import__("re")
            ),
            _local_identity_evidence_packet=AsyncMock(
                side_effect=AssertionError("full retrieval case must not route direct")
            ),
            _layered_evidence_packet=AsyncMock(
                return_value=(packet, stable_sha256(packet), set(), set(), "LOCAL_FRESH")
            ),
            _assert_snapshot_fresh=AsyncMock(),
        )



    async def test_production_execute_full_retrieval_audits_ledgers_and_injects(self) -> None:
        query = "/chat 解释作品态度前后变化"
        normalized = NormalizedMessage(
            platform="aiocqhttp",
            platform_id="bot",
            umo="bot:GroupMessage:full-e2e",
            group_id="full-e2e",
            message_id="query-full",
            sender_id="asker",
            sender_name="提问者",
            sent_at=400,
            plain_text=query,
            content=[{"type": "plain", "text": query}],
        )
        packet = local_serving_fixtures.LocalServingEnvelopeTests.packet()
        service = SimpleNamespace(
            start_experiment=AsyncMock(),
            finish_experiment=AsyncMock(),
            audit_snapshot_sources=AsyncMock(),
            record_reconstruction_step=AsyncMock(),
        )
        snapshot = SimpleNamespace(
            umo=normalized.umo,
            cutoff_at=400,
            message_upper_bound=10,
            snapshot_id="snapshot-full-e2e",
            digest="snapshot-digest",
            sender_participant_key='participant:["bot","asker"]',
            reply_source_key="",
        )
        host = self.production_host(
            normalized=normalized,
            service=service,
            snapshot=snapshot,
            packet=packet,
        )
        packet_sources = {"packet-source-selected", "packet-source-budget-omitted"}
        host._layered_evidence_packet = AsyncMock(
            return_value=(
                packet,
                stable_sha256(packet),
                packet_sources,
                set(),
                "LOCAL_FRESH",
            )
        )
        outcome = await production_execute_method()(host, object(), query)

        self.assertTrue(outcome.usable)
        self.assertEqual(outcome.operational_status, "COMPLETED")
        self.assertEqual(outcome.ledger_result["surface_injection_status"], "COMPILED_NOT_YET_INJECTED")
        self.assertIn("FULL_RETRIEVAL", outcome.ledger_result["stage_elapsed_ms"])
        self.assertIn("LEDGER_RECORD", outcome.ledger_result["stage_elapsed_ms"])
        self.assertIn("RUNTIME_READY", outcome.ledger_result["stage_elapsed_ms"])
        service.start_experiment.assert_awaited_once()
        service.audit_snapshot_sources.assert_awaited_once()
        self.assertEqual(
            service.audit_snapshot_sources.await_args.kwargs["source_keys"],
            packet_sources,
        )
        service.record_reconstruction_step.assert_awaited_once()
        host._assert_snapshot_fresh.assert_awaited_once()

        inject = main_method(
            "inject_subconscious_memory",
            asyncio=asyncio,
            json=json,
            LOCAL_SERVING_SCHEMA_VERSION=LOCAL_SERVING_SCHEMA_VERSION,
            TextPart=FakeTextPart,
            logger=SimpleNamespace(
                error=lambda *args, **kwargs: None,
                exception=lambda *args, **kwargs: None,
            ),
            GroupScopeError=ValueError,
        )
        event = SimpleNamespace(message_obj=SimpleNamespace(message_str=query))
        request = SimpleNamespace(prompt="", extra_user_content_parts=[])
        host.context = SimpleNamespace(
            get_provider_by_id=lambda value: (_ for _ in ()).throw(
                AssertionError("local injection must not look up a provider")
            )
        )
        host._group_scope = lambda value: SimpleNamespace(
            key=normalized.umo, storage_id="scope-e2e"
        )
        host._session_allowed = lambda value: True
        host._scope_event_carriers = {}
        host._services = {normalized.umo: service}
        host._service_scopes = {normalized.umo: host._group_scope(event)}
        host._test_service = service
        host.feedback_learning_enabled = False
        host._local_memory_for_request = AsyncMock(return_value=outcome)
        host._feedback_candidate_ids = {}
        host._local_serving_injected = set()

        await inject(host, event, request)
        self.assertEqual(len(request.extra_user_content_parts), 1)
        self.assertIn("<mr_memory_local_evidence>", request.extra_user_content_parts[0].text)
        self.assertIn(id(event), host._local_serving_injected)
        service.finish_experiment.assert_awaited_once()
        self.assertEqual(
            service.finish_experiment.await_args.kwargs["result"]["surface_injection_status"],
            "INJECTED_IN_REQUEST_HOOK",
        )

    async def test_normal_full_retrieval_over_old_two_second_gate_completes(self) -> None:
        query = "/chat 解释作品态度前后变化"
        normalized = NormalizedMessage(
            platform="aiocqhttp",
            platform_id="bot",
            umo="bot:GroupMessage:slow-e2e",
            group_id="slow-e2e",
            message_id="query-slow",
            sender_id="asker",
            sender_name="提问者",
            sent_at=450,
            plain_text=query,
            content=[{"type": "plain", "text": query}],
        )
        packet = local_serving_fixtures.LocalServingEnvelopeTests.packet()
        service = SimpleNamespace(
            start_experiment=AsyncMock(),
            finish_experiment=AsyncMock(),
            audit_snapshot_sources=AsyncMock(),
            record_reconstruction_step=AsyncMock(),
        )
        snapshot = SimpleNamespace(
            umo=normalized.umo,
            cutoff_at=450,
            message_upper_bound=10,
            snapshot_id="snapshot-slow-e2e",
            digest="snapshot-digest",
            sender_participant_key='participant:["bot","asker"]',
            reply_source_key="",
        )
        host = self.production_host(
            normalized=normalized,
            service=service,
            snapshot=snapshot,
            packet=packet,
            timeout=180.0,
        )

        async def slow_full_retrieval(**kwargs):
            await asyncio.sleep(2.05)
            return (packet, stable_sha256(packet), set(), set(), "LOCAL_FRESH")

        host._layered_evidence_packet = AsyncMock(side_effect=slow_full_retrieval)
        outcome = await production_execute_method()(host, object(), query)

        self.assertEqual(outcome.operational_status, "COMPLETED")
        self.assertTrue(outcome.usable)
        self.assertGreaterEqual(
            outcome.ledger_result["stage_elapsed_ms"]["FULL_RETRIEVAL"],
            2000.0,
        )
        self.assertEqual(outcome.ledger_result["hang_guard_seconds"], 180.0)

    async def test_production_execute_hang_guard_records_failed_retrieval_stage(self) -> None:
        query = "/chat 解释作品态度前后变化"
        normalized = NormalizedMessage(
            platform="aiocqhttp", platform_id="bot",
            umo="bot:GroupMessage:hang-e2e", group_id="hang-e2e",
            message_id="query-hang", sender_id="asker", sender_name="提问者",
            sent_at=500, plain_text=query,
            content=[{"type": "plain", "text": query}],
        )
        packet = local_serving_fixtures.LocalServingEnvelopeTests.packet()
        service = SimpleNamespace(
            start_experiment=AsyncMock(),
            finish_experiment=AsyncMock(),
            audit_snapshot_sources=AsyncMock(),
            record_reconstruction_step=AsyncMock(),
        )
        snapshot = SimpleNamespace(
            umo=normalized.umo, cutoff_at=500, message_upper_bound=10,
            snapshot_id="snapshot-hang", digest="snapshot-digest",
            sender_participant_key="", reply_source_key="",
        )
        host = self.production_host(
            normalized=normalized, service=service, snapshot=snapshot,
            packet=packet, timeout=0.01,
        )
        async def hung_retrieval(**kwargs):
            await asyncio.sleep(1)
            return (packet, stable_sha256(packet), set(), set(), "LOCAL_FRESH")

        host._layered_evidence_packet = AsyncMock(side_effect=hung_retrieval)
        outcome = await production_execute_method()(host, object(), query)

        self.assertEqual(outcome.operational_status, "FAILED")
        self.assertIn("abnormal-hang guard", outcome.detail)
        service.finish_experiment.assert_awaited_once()
        failure = service.finish_experiment.await_args.kwargs
        self.assertEqual(failure["status"], "failed")
        self.assertEqual(failure["result"]["last_stage"], "FULL_RETRIEVAL")
        self.assertIn(
            "FULL_RETRIEVAL_INCOMPLETE",
            failure["result"]["stage_elapsed_ms"],
        )

    async def test_full_retrieval_requests_are_not_globally_serialized(self) -> None:
        query = "/chat 解释作品态度前后变化"
        normalized = NormalizedMessage(
            platform="aiocqhttp", platform_id="bot",
            umo="bot:GroupMessage:concurrent-e2e", group_id="concurrent-e2e",
            message_id="query-concurrent", sender_id="asker",
            sender_name="提问者", sent_at=550, plain_text=query,
            content=[{"type": "plain", "text": query}],
        )
        packet = local_serving_fixtures.LocalServingEnvelopeTests.packet()
        service = SimpleNamespace(
            start_experiment=AsyncMock(), finish_experiment=AsyncMock(),
            audit_snapshot_sources=AsyncMock(),
            record_reconstruction_step=AsyncMock(),
        )
        snapshot = SimpleNamespace(
            umo=normalized.umo, cutoff_at=550, message_upper_bound=10,
            snapshot_id="snapshot-concurrent", digest="snapshot-digest",
            sender_participant_key="", reply_source_key="",
        )
        host = self.production_host(
            normalized=normalized, service=service, snapshot=snapshot,
            packet=packet, timeout=1.0,
        )
        both_entered = asyncio.Event()
        active = 0
        maximum_active = 0

        async def concurrent_retrieval(**kwargs):
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            if active == 2:
                both_entered.set()
            try:
                await both_entered.wait()
                return (packet, stable_sha256(packet), set(), set(), "LOCAL_FRESH")
            finally:
                active -= 1

        host._layered_evidence_packet = AsyncMock(side_effect=concurrent_retrieval)
        outcomes = await asyncio.wait_for(
            asyncio.gather(
                production_execute_method()(host, object(), query),
                production_execute_method()(host, object(), query),
            ),
            timeout=0.5,
        )
        self.assertEqual(maximum_active, 2)
        self.assertTrue(all(item.operational_status == "COMPLETED" for item in outcomes))


    async def test_provider_turn_report_is_pairwise_closed_and_explicit(self) -> None:
        common = {
            "request_id": "run-1:analysis:0", "run_id": "run-1", "arm": "memory",
            "repetition": 1, "phase": "analysis", "call_index": 0,
            "provider_id": "provider", "model": "model", "thinking": "enabled",
            "max_tokens": 1000, "options_sha256": "options", "payload_sha256": "payload",
        }
        report = provider_turn_report(
            [
                {**common, "event": "attempted"},
                {**common, "event": "completed", "usage_present": True,
                 "input_other": 100, "input_cached": 20, "input": 120,
                 "output": 30, "total": 150, "elapsed_ms": 250.5},
            ],
            run_id="run-1",
        )
        self.assertTrue(report["pairwise_closed"])
        self.assertEqual(report["attempted_calls"], 1)
        self.assertEqual(report["measured_completed_total_lower_bound"], 150)
        self.assertTrue(report["usage_complete"])
        self.assertEqual(report["turns"][0]["payload_sha256"], "payload")
        self.assertEqual(
            report["currency_cost"], "UNKNOWN_PROVIDER_PRICING_NOT_IN_LEDGER"
        )

        with self.assertRaisesRegex(ValueError, "pairwise closed"):
            provider_turn_report([{**common, "event": "attempted"}], run_id="run-1")

        with self.assertRaisesRegex(ValueError, "no rows"):
            provider_turn_report([], run_id="run-1")
        zero = provider_turn_report(
            [],
            run_id="local-run",
            expect_zero_calls=True,
            non_llm_run_evidence={
                "run_id": "local-run",
                "operational_status": "COMPLETED",
                "path": "materialized_local",
                "memory_provider_calls": 0,
            },
        )
        self.assertTrue(zero["expected_zero_calls"])
        self.assertEqual(zero["attempted_calls"], 0)

        with self.assertRaisesRegex(ValueError, "terminal precedes"):
            provider_turn_report(
                [
                    {**common, "event": "completed", "usage_present": True,
                     "input_other": 0, "input_cached": 0, "input": 0,
                     "output": 0, "total": 0},
                    {**common, "event": "attempted"},
                ],
                run_id="run-1",
            )

        failed = provider_turn_report(
            [
                {**common, "event": "attempted"},
                {**common, "event": "failed", "elapsed_ms": 10,
                 "error_type": "TimeoutError", "error_detail": "timeout"},
            ],
            run_id="run-1",
        )
        self.assertFalse(failed["usage_complete"])
        self.assertEqual(failed["failed_calls"], 1)
        self.assertEqual(failed["measured_completed_total_lower_bound"], 0)


if __name__ == "__main__":
    unittest.main()
