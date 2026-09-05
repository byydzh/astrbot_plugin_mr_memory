from __future__ import annotations

import ast
import asyncio
import copy
import json
import re
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from mr_memory.evidence_pack import (
    EVIDENCE_ATOM_PACK_FORMAT,
    compile_evidence_atom_pack,
    hydrate_evidence_atom_pack,
    participant_source_bindings as compile_participant_source_bindings,
)
from mr_memory.derivations import MAX_STORED_DERIVATIONS
from mr_memory.identity import build_request_identity_context
from mr_memory.reader import build_l2_reader_prompt
from mr_memory.snapshot import stable_sha256
from mr_memory.provider_compat import ProviderStreamError
from mr_memory.usage import TokenUsageRecord
from mr_memory.distillation import distillation_generation_options
from tests.test_certificate_v2 import _snapshot as l2_snapshot

MAIN_SOURCE = (
    Path(__file__).resolve().parents[1] / "main.py"
).read_text(encoding="utf-8")
MAIN_TREE = ast.parse(MAIN_SOURCE)


class _GroupScopeError(ValueError):
    pass


class _FakeTextPart:
    def __init__(self, *, text: str):
        self.text = text

    def mark_as_temp(self) -> "_FakeTextPart":
        return self


def _main_method(name: str, **namespace: object):
    """Execute one real main.py method without importing the AstrBot runtime.

    AstrBot is intentionally not a test dependency in this repository.  Compiling
    the selected method body lets these tests exercise its async control flow with
    provider/service spies instead of merely matching source strings.
    """

    original = next(
        node
        for node in ast.walk(MAIN_TREE)
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
        and node.name == name
    )
    node = copy.deepcopy(original)
    node.decorator_list = []
    node.returns = None
    arguments = [
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
    ]
    if node.args.vararg is not None:
        arguments.append(node.args.vararg)
    if node.args.kwarg is not None:
        arguments.append(node.args.kwarg)
    for argument in arguments:
        argument.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    values = dict(namespace)
    exec(compile(module, f"<main.py:{name}>", "exec"), values)
    return values[name]


class MainLocalBehaviorTests(unittest.IsolatedAsyncioTestCase):
    async def test_reader_provider_override_is_isolated_and_tracks_actual_model(self):
        schema = json.loads((Path(__file__).resolve().parents[1] / "_conf_schema.json").read_text(encoding="utf-8"))
        self.assertEqual(schema["local_serving_reader_provider_id"]["default"], "")
        self.assertEqual(schema["local_serving_reader_provider_id"]["_special"], "select_provider")
        select = _main_method("_local_reader_provider_id")
        host = SimpleNamespace(subconscious_provider_id="synthetic-maintenance",
                               local_serving_reader_provider_id="", distillation_max_output_tokens=1234,
                               embedding_enabled=False, _last_reader_model_revision="synthetic-maintenance|model=old")
        self.assertEqual(select(host), "synthetic-maintenance")
        host.local_serving_reader_provider_id = " synthetic-reader "
        selected = select(host)
        self.assertEqual(selected, "synthetic-reader")
        response = SimpleNamespace(usage={"input_other": 7, "output": 3})
        generate = AsyncMock(return_value=response)
        generate_method = _main_method(
            "_run_fast_reconstruction_with_ledger", time=time,
            FAST_RECONSTRUCTION_SYSTEM_PROMPT="synthetic system",
            distillation_generation_options=distillation_generation_options,
            _provider_model_name=lambda provider: "gemini-3.5-flash",
            _generate_with_failure_ledger=generate, TokenUsageRecord=TokenUsageRecord,
        )
        service = SimpleNamespace(record_llm_usage=AsyncMock())
        await generate_method(host, provider=object(), service=service, run_id="synthetic-reader-run",
                              prompt="synthetic evidence", provider_id=selected, thinking_mode="disabled",
                              phase="resident_evidence_reader", max_output_tokens=8192)
        self.assertEqual(generate.await_args.kwargs["provider_id"], selected)
        self.assertEqual(generate.await_args.kwargs["model"], "gemini-3.5-flash")
        self.assertEqual(generate.await_args.kwargs["options"]["max_tokens"], 8192)
        self.assertEqual(service.record_llm_usage.await_args.kwargs["provider_id"], selected)
        self.assertEqual(host.subconscious_provider_id, "synthetic-maintenance")
        revision = _main_method(
            "_runtime_inference_revision", _provider_model_name=lambda provider: "synthetic-reader-model",
            L2_READER_PROTOCOL="synthetic-protocol", CERTIFICATE_SCHEMA_VERSION="synthetic-certificate",
            ANSWER_CONTEXT_SCHEMA_VERSION="synthetic-answer-context",
        )
        actual_revision = revision(host, provider=object(), policy=SimpleNamespace(revision="same-policy"),
                                   provider_id=selected)
        self.assertEqual(actual_revision["reader_model"], selected + "|model=synthetic-reader-model")
        self.assertEqual(actual_revision["surface_compiler"], "synthetic-answer-context")

        # A missing explicit Reader must not cause another provider lookup.
        lookup = Mock(return_value=None)
        host.context = SimpleNamespace(get_provider_by_id=lookup)
        host._local_reader_provider_id = lambda: select(host)
        host._inflight_runtime_tasks = set()
        host.local_serving_enabled = True
        host._local_serving_guard = lambda event: ""
        host.max_query_chars = 100
        serving = _main_method(
            "_execute_local_memory_serving", asyncio=asyncio, time=time,
            _runtime_run_id=lambda kind: "synthetic-unavailable-reader", _LocalMemoryOutcome=SimpleNamespace,
            logger=Mock(),
        )
        outcome = await serving(host, object(), "synthetic query")
        self.assertEqual(outcome.operational_status, "PROVIDER_UNAVAILABLE")
        lookup.assert_called_once_with(selected)

    def test_feedback_compact_packet_preserves_host_relative_times(self):
        method = _main_method(
            "_compact_feedback_inspection",
            _feedback_text=_main_method("_feedback_text", re=re),
        )
        inspected = {
            "proposal_id": 1,
            "feedback": {"source_key": "synthetic:feedback", "sent_at": 100},
            "candidate_traces": [
                {"trace_id": "synthetic:old", "request_sent_at": 10,
                 "request_age_seconds": 90, "response_at": 20,
                 "response_age_seconds": 80},
                {"trace_id": "synthetic:pending", "request_sent_at": 90,
                 "request_age_seconds": 10, "response_at": None,
                 "response_age_seconds": None},
            ],
            "context": [
                {"source_key": "synthetic:near", "sent_at": 99, "age_seconds": 1},
                {"source_key": "synthetic:same-second", "sent_at": 100, "age_seconds": 0},
            ],
        }
        packet = method(inspected)
        self.assertEqual(
            [(row["request_age_seconds"], row["response_age_seconds"])
             for row in packet["candidate_traces"]],
            [(90, 80), (10, None)],
        )
        self.assertEqual([row["age_seconds"] for row in packet["context"]], [1, 0])
        self.assertEqual(packet["candidate_traces"][0]["response_at"], 20)
        self.assertEqual(packet["feedback"]["sent_at"], 100)

    async def test_failed_provider_call_records_partial_usage_and_stays_failed(self):
        error = ProviderStreamError(
            "incomplete stream",
            partial_usage={"input_other": 19, "input_cached": 4, "output": 7},
            chunk_count=2,
            failure_kind="incomplete_stream",
        )
        provider_call = AsyncMock(side_effect=error)
        method = _main_method(
            "_generate_with_failure_ledger",
            generate_with_enforced_options=provider_call,
            asyncio=asyncio,
            TokenUsageRecord=TokenUsageRecord,
            time=time,
        )
        service = SimpleNamespace(record_llm_usage=AsyncMock())
        with self.assertRaises(ProviderStreamError) as caught:
            await method(
                service=service, run_id="synthetic-run", phase="reader",
                provider_id="synthetic-provider", model="synthetic-model",
                started=time.perf_counter(), prompt="synthetic input",
            )
        self.assertIs(caught.exception, error)
        provider_call.assert_awaited_once()
        recorded = service.record_llm_usage.await_args.kwargs
        self.assertEqual(
            (recorded["input_other"], recorded["input_cached"], recorded["output"]),
            (19, 4, 7),
        )
        self.assertEqual(recorded["usage_source"], "provider_failure_partial_usage")

    async def test_default_distillation_schedule_uses_next_batch_identity(self):
        method = _main_method("_schedule_maintenance", time=time)
        service = SimpleNamespace(
            enqueue_pending_distillation_job=AsyncMock(return_value=17),
            enqueue_maintenance_job=AsyncMock(),
        )
        host = SimpleNamespace(
            _scope_event_carriers={},
            _service_for_scope=lambda scope: service,
            distillation_max_messages=40,
            _queue_existing_maintenance=AsyncMock(return_value=True),
            _schedule_maintenance_wakeup=Mock(),
        )
        scope = SimpleNamespace(key="synthetic-scope")
        self.assertTrue(await method(host, kind="distill", scope=scope))
        service.enqueue_pending_distillation_job.assert_awaited_once_with(
            umo="synthetic-scope", limit=40, available_at=None,
        )
        service.enqueue_maintenance_job.assert_not_awaited()
        service.enqueue_pending_distillation_job.return_value = None
        self.assertFalse(await method(host, kind="distill", scope=scope))
        self.assertEqual(host._queue_existing_maintenance.await_count, 1)

    async def test_budget_wait_defers_without_failing_or_calling_provider(self):
        method = _main_method(
            "_maintenance_worker", asyncio=asyncio, time=time, logger=Mock(),
            scoped_job_key=lambda **values: (values["umo"], values["job_id"]),
        )
        service = SimpleNamespace(
            claim_maintenance_job=AsyncMock(return_value={"payload": {}}),
            defer_maintenance_job=AsyncMock(return_value=True),
            fail_maintenance_job=AsyncMock(),
        )
        host = SimpleNamespace(
            _maintenance_enqueued=set(), _scope_event_carriers={},
            _service_for_scope=lambda scope: service,
            maintenance_llm_timeout_seconds=30,
            _private_budget_available=AsyncMock(return_value=False),
            _schedule_maintenance_wakeup=Mock(), _distill_scope=AsyncMock(),
        )
        queue = asyncio.Queue()
        queue.put_nowait((17, "distill", SimpleNamespace(key="synthetic-scope")))
        task = asyncio.create_task(method(host, queue=queue, worker_name="test"))
        try:
            await asyncio.wait_for(queue.join(), timeout=1)
            service.defer_maintenance_job.assert_awaited_once()
            service.fail_maintenance_job.assert_not_awaited()
            host._distill_scope.assert_not_awaited()
            host._schedule_maintenance_wakeup.assert_called_once()
            self.assertEqual(
                service.defer_maintenance_job.await_args.kwargs["reason"],
                "budget_exhausted:online",
            )
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

    async def test_feedback_schedule_freezes_next_pending_batch(self):
        method = _main_method("_schedule_maintenance", time=time)
        service = SimpleNamespace(
            enqueue_pending_feedback_job=AsyncMock(return_value=23),
            enqueue_maintenance_job=AsyncMock(),
        )
        host = SimpleNamespace(
            _scope_event_carriers={}, _service_for_scope=lambda scope: service,
            feedback_max_pending_per_wake=6,
            _queue_existing_maintenance=AsyncMock(return_value=True),
            _schedule_maintenance_wakeup=Mock(),
        )
        scope = SimpleNamespace(key="synthetic-scope")
        self.assertTrue(await method(host, kind="feedback", scope=scope))
        service.enqueue_pending_feedback_job.assert_awaited_once_with(
            umo=scope.key, limit=6, available_at=None,
        )
        service.enqueue_maintenance_job.assert_not_awaited()

    async def test_feedback_thinking_config_keeps_stream_and_normal_usage(self):
        schema = json.loads((Path(__file__).resolve().parents[1] / "_conf_schema.json").read_text(encoding="utf-8"))
        self.assertEqual(schema["feedback_thinking_mode"]["default"], "enabled")
        self.assertEqual(schema["feedback_thinking_mode"]["options"], ["enabled", "disabled"])
        response = SimpleNamespace(usage={"input_other": 13, "output": 5})
        generate = AsyncMock(return_value=response)
        method = _main_method(
            "_run_feedback_batch_with_ledger", time=time,
            distillation_generation_options=distillation_generation_options,
            _provider_model_name=lambda provider: "deepseek-v4-flash",
            _generate_with_failure_ledger=generate,
            FEEDBACK_BATCH_SYSTEM_PROMPT="synthetic feedback contract",
            PLASTIC_GRAPH_MAINTENANCE_PROMPT="synthetic graph contract",
            TokenUsageRecord=TokenUsageRecord,
        )
        host = SimpleNamespace(
            feedback_thinking_mode="disabled", distillation_thinking_mode="enabled",
            distillation_max_output_tokens=384000,
            subconscious_provider_id="synthetic-provider",
        )
        service = SimpleNamespace(record_llm_usage=AsyncMock())
        actual, _ = await method(
            host, provider=object(), service=service,
            run_id="synthetic-feedback", prompt="synthetic evidence", call_index=0,
        )
        self.assertIs(actual, response)
        generate.assert_awaited_once()
        sent = generate.await_args.kwargs
        self.assertEqual(sent["options"]["thinking"], {"type": "disabled"})
        self.assertEqual(sent["options"]["response_format"], {"type": "json_object"})
        self.assertEqual(sent["options"]["max_tokens"], 384000)
        self.assertTrue(sent["stream"])
        self.assertEqual(host.distillation_thinking_mode, "enabled")
        service.record_llm_usage.assert_awaited_once()
        self.assertEqual(
            service.record_llm_usage.await_args.kwargs["usage_source"],
            "astrbot_response_one_pass_batch",
        )
        batch = next(node for node in ast.walk(MAIN_TREE) if isinstance(node, ast.AsyncFunctionDef) and node.name == "_run_feedback_maintenance_batch")
        call = next(node for node in ast.walk(batch) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "_run_feedback_batch_with_ledger")
        self.assertEqual(
            next(ast.unparse(item.value) for item in call.keywords if item.arg == "thinking_mode"),
            "self.feedback_thinking_mode",
        )

    async def test_feedback_failure_preserves_error_and_only_fails_attempted_versions(self):
        method = _main_method("_run_feedback_maintenance", asyncio=asyncio, logger=Mock())
        scope = SimpleNamespace(key="synthetic-scope")
        first = {"id": 3, "feedback_revision_no": 1, "feedback_content_sha256": "a" * 64}
        second = {"id": 4, "feedback_revision_no": 1, "feedback_content_sha256": "b" * 64}
        service = SimpleNamespace(fail_feedback_proposals=AsyncMock())
        error = ValueError("invalid decision")
        async def fail_batch(**kwargs):
            self.assertEqual(kwargs["proposal_snapshots"], [first, second])
            kwargs["attempted"].append(first)
            raise error
        host = SimpleNamespace(_run_feedback_maintenance_batch=fail_batch)
        with self.assertRaises(ValueError) as caught:
            await method(host, scope=scope, service=service, proposal_snapshots=[first, second])
        self.assertIs(caught.exception, error)
        service.fail_feedback_proposals.assert_awaited_once_with(
            umo=scope.key, snapshots=[first], error="ValueError: invalid decision",
        )

    async def test_unchanged_capture_redelivery_has_no_maintenance_side_effects(self) -> None:
        logger = SimpleNamespace(info=Mock(), debug=Mock(), exception=Mock())
        method = _main_method(
            "capture_group_message",
            GroupScopeError=_GroupScopeError,
            logger=logger,
            time=time,
        )
        scope = SimpleNamespace(key="scope-1", platform_id="shadow")
        normalized = SimpleNamespace(
            message_id="platform-message-1",
            sender_id="synthetic-account",
            plain_text="synthetic text",
            content=[{"type": "plain", "text": "synthetic text"}],
            resolved_source_key=lambda: "shadow|scope-1|platform-message-1",
        )
        writes = [
            SimpleNamespace(status="INSERTED", needs_processing=True),
            SimpleNamespace(status="UNCHANGED", needs_processing=False),
        ]
        service = SimpleNamespace(
            is_account_forgotten=AsyncMock(return_value=False),
            ingest_with_outcome=AsyncMock(side_effect=writes),
            enqueue_feedback_candidate=AsyncMock(return_value=None),
        )
        host = SimpleNamespace(
            capture_enabled=True,
            feedback_learning_enabled=True,
            feedback_window_seconds=3600,
            auto_distillation_enabled=True,
            log_message_content=False,
            _scope_event_carriers={},
            _group_scope=lambda event: scope,
            _session_allowed=lambda umo: True,
            _service_for_scope=lambda value: service,
            _normalize_event=lambda event: normalized,
            _debounce_feedback=Mock(),
            _ensure_distillation_deadline=AsyncMock(return_value=True),
        )
        event = SimpleNamespace(
            message_obj=SimpleNamespace(raw_message={}),
            get_sender_id=lambda: normalized.sender_id,
        )

        await method(host, event)
        await method(host, event)

        self.assertEqual(service.ingest_with_outcome.await_count, 2)
        service.enqueue_feedback_candidate.assert_awaited_once()
        host._ensure_distillation_deadline.assert_awaited_once()
        logger.exception.assert_not_called()

    async def test_live_capture_rejects_missing_platform_message_id(self) -> None:
        logger = SimpleNamespace(info=Mock(), debug=Mock(), exception=Mock())
        method = _main_method(
            "capture_group_message",
            GroupScopeError=_GroupScopeError,
            logger=logger,
            time=time,
        )
        scope = SimpleNamespace(key="scope-1", platform_id="shadow")
        normalized = SimpleNamespace(
            message_id="",
            sender_id="synthetic-account",
            plain_text="synthetic text",
            content=[{"type": "plain", "text": "synthetic text"}],
        )
        service = SimpleNamespace(
            is_account_forgotten=AsyncMock(return_value=False),
            ingest_with_outcome=AsyncMock(),
            enqueue_feedback_candidate=AsyncMock(),
        )
        host = SimpleNamespace(
            capture_enabled=True,
            feedback_learning_enabled=True,
            feedback_window_seconds=3600,
            auto_distillation_enabled=True,
            log_message_content=False,
            _scope_event_carriers={},
            _group_scope=lambda event: scope,
            _session_allowed=lambda umo: True,
            _service_for_scope=lambda value: service,
            _normalize_event=lambda event: normalized,
            _debounce_feedback=Mock(),
            _ensure_distillation_deadline=AsyncMock(),
        )
        event = SimpleNamespace(
            message_obj=SimpleNamespace(raw_message={}),
            get_sender_id=lambda: normalized.sender_id,
        )

        await method(host, event)

        service.ingest_with_outcome.assert_not_called()
        service.enqueue_feedback_candidate.assert_not_called()
        host._ensure_distillation_deadline.assert_not_called()
        logger.exception.assert_called_once_with(
            "MR Memory failed to capture a group message."
        )

    async def test_opening_feedback_trace_does_not_activate_a_hypothesis(self) -> None:
        method = _main_method(
            "_begin_interaction_trace",
            asyncio=asyncio,
            time=time,
            _runtime_run_id=lambda phase: f"{phase}-run",
        )
        event = object()
        normalized = SimpleNamespace(
            sender_id="sender-1",
            sent_at=123,
            resolved_source_key=lambda: "source-request",
        )
        service = SimpleNamespace(
            start_interaction_trace=AsyncMock(),
            activate_feedback_hypotheses=AsyncMock(),
        )
        host = SimpleNamespace(
            _inflight_interaction_tasks=set(),
            _active_interaction_traces={},
            _trace_tool_counters={},
            _pending_main_tools={},
            _feedback_candidate_ids={},
            _normalize_event=lambda value: normalized,
            max_query_chars=4000,
            feedback_trace_ttl_seconds=86400,
        )

        result = await method(
            host,
            event=event,
            scope=SimpleNamespace(key="scope-1"),
            service=service,
            query="普通聊天",
        )

        self.assertEqual(result, [])
        service.start_interaction_trace.assert_awaited_once()
        service.activate_feedback_hypotheses.assert_not_called()

    async def test_local_guard_does_not_depend_on_background_model_toggle(self) -> None:
        method = _main_method(
            "_local_serving_guard",
            GroupScopeError=_GroupScopeError,
        )
        provider_spy = Mock(side_effect=AssertionError("provider lookup is forbidden"))
        host = SimpleNamespace(
            subconscious_enabled=False,
            context=SimpleNamespace(get_provider_by_id=provider_spy),
            _group_scope=lambda event: SimpleNamespace(key="scope-1"),
            _session_allowed=lambda umo: umo == "scope-1",
        )

        self.assertEqual(method(host, object()), "")
        provider_spy.assert_not_called()




    async def test_source_bound_person_candidate_is_expanded_for_reader_reasoning(
        self,
    ) -> None:
        collect_source_keys = _main_method("_collect_source_keys")
        collect_full_message_source_keys = _main_method(
            "_collect_full_message_source_keys"
        )
        collect_participant_keys = _main_method("_collect_participant_keys")
        collect_participant_keys_in_order = _main_method(
            "_collect_participant_keys_in_order"
        )
        participant_source_bindings = _main_method(
            "_participant_source_bindings",
            participant_source_bindings=compile_participant_source_bindings,
        )
        method = _main_method(
            "_layered_evidence_packet",
            asyncio=asyncio,
            re=re,
            time=time,
            build_request_identity_context=build_request_identity_context,
            _collect_source_keys=collect_source_keys,
            _collect_full_message_source_keys=collect_full_message_source_keys,
            _collect_participant_keys=collect_participant_keys,
            _collect_participant_keys_in_order=collect_participant_keys_in_order,
            _participant_source_bindings=participant_source_bindings,
            EVIDENCE_ATOM_PACK_FORMAT=EVIDENCE_ATOM_PACK_FORMAT,
            MAX_STORED_DERIVATIONS=MAX_STORED_DERIVATIONS,
            compile_evidence_atom_pack=compile_evidence_atom_pack,
            hydrate_evidence_atom_pack=hydrate_evidence_atom_pack,
            stable_sha256=stable_sha256,
        )
        recent_participant = 'participant:["synthetic","recent-account"]'
        incidental_episode_speaker = (
            'participant:["synthetic","incidental-episode-speaker"]'
        )

        class Service:
            def __init__(self) -> None:
                self.history_keys: list[str] = []
                self.activity_keys: list[str] = []
                self.search_calls: list[dict[str, object]] = []
                self.audit_candidate_memory_closures = AsyncMock(return_value={
                    "stored_derivations": [], "dependency_source_keys": [],
                    "rejected": [],
                })
                self.query_lexical_context = AsyncMock(return_value=[{
                    "anchor_source_key": "synthetic-recent-source",
                    "context_relation": "chronological_neighbor_not_reply",
                    "messages": [{"source_key": "synthetic-recent-source", "relative_seconds": 0}],
                }])
                self.resolve_query_participants = AsyncMock(
                    side_effect=AssertionError(
                        "text alias parser must not decide resident identity"
                    )
                )

            async def query_plastic_associations(self, **_kwargs: object) -> list:
                return []

            async def query_matching_cues(self, **_kwargs: object) -> list:
                return []

            async def search_messages(self, **kwargs: object) -> list:
                self.search_calls.append(dict(kwargs))
                return [SimpleNamespace(
                    source_key="synthetic-recent-source", sent_at=90,
                    sender_id="recent-account", sender_name="近期成员",
                    sender_participant_key=recent_participant, role="USER",
                    plain_text="一条合成近期消息", reply_to_source_key="",
                    mentions=(), revision_no=1,
                )]

            async def reconstruction_evidence_packet(
                self, **_kwargs: object
            ) -> dict[str, object]:
                return {
                    "candidates": {"participants": []},
                    "semantic_evidence": [],
                    "expanded_episodes": [
                        {
                            "id": 7,
                            "title": "合成事件",
                            "messages": [
                                {
                                    "source_key": "synthetic-episode-source",
                                    "sent_at": 40,
                                    "sender_participant_key": (
                                        incidental_episode_speaker
                                    ),
                                    "plain_text": "旁观成员在事件里说过一句话",
                                }
                            ],
                        }
                    ],
                }

            async def query_recent_context(
                self, **_kwargs: object
            ) -> list[dict[str, object]]:
                return [
                    {
                        "source_key": "synthetic-recent-source",
                        "sent_at": 90,
                        "sender_id": "recent-account",
                        "sender_name": "近期成员",
                        "sender_participant_key": recent_participant,
                        "plain_text": "一条合成近期消息",
                    }
                ]

            async def query_person_reference_candidates(
                self, **_kwargs: object
            ) -> dict[str, object]:
                return {
                    "host_decision": "NONE",
                    "references": [
                        {
                            "reference": "近期成员",
                            "candidate_participants": [
                                {
                                    "participant_key": recent_participant,
                                    "platform_id": "synthetic",
                                    "account_id": "recent-account",
                                    "alias_observations": [
                                        {
                                            "alias": "近期成员",
                                            "normalized_alias": "近期成员",
                                            "source_key": "synthetic-recent-source",
                                            "sent_at": 90,
                                            "relation": "SPEAKER",
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                    "coverage": {
                        "matched_reference_count_total": 1,
                        "matched_reference_count_returned": 1,
                        "candidate_count_total": 1,
                        "candidate_count_returned": 1,
                        "distinct_candidate_count_total": 1,
                        "distinct_candidate_count_returned": 1,
                        "truncated": False,
                    },
                }

            async def count_snapshot_messages(self, **_kwargs: object) -> int:
                return 2

            async def query_participant_history(
                self, **kwargs: object
            ) -> dict[str, object]:
                participant_key = str(kwargs["participant_key"])
                self.history_keys.append(participant_key)
                return {
                    "participant_key": participant_key,
                    "status": "SOURCE_BACKED",
                    "messages": [
                        {
                            "source_key": "synthetic-history-source",
                            "sent_at": 50,
                            "plain_text": "一条合成历史消息",
                        }
                    ],
                }

            async def query_participant_activity(
                self, **kwargs: object
            ) -> dict[str, object]:
                participant_key = str(kwargs["participant_key"])
                self.activity_keys.append(participant_key)
                return {
                    "participant_key": participant_key,
                    "found": True,
                    "source_keys": ["synthetic-history-source"],
                }

            async def message_for_source(self, **_kwargs: object) -> None:
                return None

            async def messages_for_sources(
                self, **kwargs: object
            ) -> list[dict[str, object]]:
                requested = set(kwargs["source_keys"])
                authoritative = {
                    "synthetic-episode-source": {
                        "source_key": "synthetic-episode-source",
                        "sent_at": 40,
                        "sender_id": "incidental-account",
                        "sender_name": "旁观成员",
                        "sender_participant_key": incidental_episode_speaker,
                        "role": "user",
                        "plain_text": "旁观成员在事件里说过一句话",
                        "components": [
                            {
                                "type": "plain",
                                "text": "旁观成员在事件里说过一句话",
                            }
                        ],
                    },
                    "synthetic-history-source": {
                        "source_key": "synthetic-history-source",
                        "sent_at": 50,
                        "sender_id": "recent-account",
                        "sender_name": "近期成员",
                        "sender_participant_key": recent_participant,
                        "role": "user",
                        "plain_text": "一条合成历史消息",
                        "components": [
                            {"type": "plain", "text": "一条合成历史消息"}
                        ],
                    },
                    "synthetic-recent-source": {
                        "source_key": "synthetic-recent-source",
                        "sent_at": 90,
                        "sender_id": "recent-account",
                        "sender_name": "近期成员",
                        "sender_participant_key": recent_participant,
                        "role": "user",
                        "plain_text": "一条合成近期消息",
                        "components": [
                            {"type": "plain", "text": "一条合成近期消息"}
                        ],
                    },
                }
                return [
                    authoritative[source_key]
                    for source_key in kwargs["source_keys"]
                    if source_key in requested and source_key in authoritative
                ]

        service = Service()
        host = SimpleNamespace(
            _layered_pack_key=lambda *_args, **_kwargs: "synthetic-pack",
            feedback_learning_enabled=False,
            embedding_top_k=8,
            local_serving_max_items=12,
            candidate_seed_floor=0.0,
            _embedding_backend=lambda: None,
        )
        snapshot = SimpleNamespace(
            umo="synthetic:GroupMessage:scope",
            cutoff_at=100,
            message_upper_bound=99,
            request_source_key="synthetic-current-source",
            sender_participant_key='participant:["synthetic","requester"]',
            reply_source_key="",
            snapshot_id="synthetic-snapshot",
        )
        normalized = SimpleNamespace(
            platform_id="synthetic",
            sender_id="requester",
            sender_name="提问者",
            content=[{"type": "plain", "text": "/chat 合成人物问题"}],
        )

        packet, _digest, source_keys, participant_keys, cache_layer = await method(
            host,
            service=service,
            snapshot=snapshot,
            normalized=normalized,
            query="/chat 合成人物问题",
            resolve_query_aliases=False,
            include_participant_activity=True,
            use_cache=False,
            finalize_packet=False,
        )

        service.resolve_query_participants.assert_not_awaited()
        service.audit_candidate_memory_closures.assert_awaited_once_with(
            umo=snapshot.umo,
            candidates={"semantic_memories": [], "episodes": [{"id": 7}],
                        "associations": []},
            before_sent_at=snapshot.cutoff_at,
            message_upper_bound=snapshot.message_upper_bound,
        )
        service.query_lexical_context.assert_awaited_once_with(
            umo=snapshot.umo, source_keys=["synthetic-recent-source"],
            before_sent_at=snapshot.cutoff_at, message_upper_bound=snapshot.message_upper_bound,
            exclude_source_key=snapshot.request_source_key,
        )
        self.assertEqual(len(packet["lexical_context"]), 1)
        self.assertEqual(packet["lexical_context"][0]["anchor_source_key"], "synthetic-recent-source")
        self.assertEqual(packet["lexical_context"][0]["context_relation"], "chronological_neighbor_not_reply")
        self.assertEqual(packet["lexical_context"][0]["messages"][0]["source_key"], "synthetic-recent-source")
        self.assertEqual(service.history_keys, [recent_participant])
        self.assertEqual(service.activity_keys, [recent_participant])
        self.assertNotIn(incidental_episode_speaker, service.history_keys)
        self.assertNotIn(incidental_episode_speaker, service.activity_keys)
        self.assertEqual(
            packet["person_reasoning_candidates"]["participant_keys"],
            [recent_participant],
        )
        self.assertEqual(packet["participant_history"][0]["status"], "SOURCE_BACKED")
        self.assertEqual(
            packet["recent_context"][0]["source_key"],
            "synthetic-recent-source",
        )
        self.assertEqual(
            source_keys,
            {
                "synthetic-episode-source",
                "synthetic-history-source",
                "synthetic-recent-source",
            },
        )
        self.assertEqual(
            packet["participant_source_keys"][recent_participant],
            ["synthetic-history-source", "synthetic-recent-source"],
        )
        self.assertEqual(
            packet["participant_speaker_source_keys"][recent_participant],
            ["synthetic-history-source", "synthetic-recent-source"],
        )
        self.assertEqual(
            packet["retrieval_coverage"]["source_hydration_returned"],
            3,
        )
        self.assertTrue(packet["retrieval_coverage"]["semantic_none_allowed"])
        self.assertIn(recent_participant, participant_keys)
        self.assertEqual(cache_layer, "LOCAL_FRESH")
        self.assertEqual(
            service.search_calls,
            [
                {
                    "umo": snapshot.umo,
                    "query": "合成人物问题",
                    "limit": host.local_serving_max_items,
                    "before_sent_at": snapshot.cutoff_at,
                    "message_upper_bound": snapshot.message_upper_bound,
                    "match_mode": "recall",
                    "exclude_source_key": snapshot.request_source_key,
                }
            ],
        )

    def test_identity_canonical_key_is_aliased_across_main_and_reader(self) -> None:
        collect_participant_keys = _main_method("_collect_participant_keys")
        participant_key = 'participant:["synthetic","account-a"]'
        graph_key = "graph-node:synthetic-topic"
        packet = {
            "identity_candidates": [
                {
                    "canonical_key": participant_key,
                    "account_id": "account-a",
                    "current_display_name": "合成成员甲",
                }
            ],
            "graph_metadata": {
                "canonical_key": graph_key,
                "kind": "topic",
            },
        }

        participant_keys = collect_participant_keys(packet)
        self.assertEqual(participant_keys, {participant_key})
        request = build_l2_reader_prompt(
            query="纸鹤计划进展如何",
            evidence_packet=packet,
            snapshot=l2_snapshot(),
            allowed_source_keys=(),
            allowed_participant_keys=participant_keys,
            pack_read_complete=True,
        )
        prompt_packet = json.loads(request.user_prompt)["evidence_packet"]

        self.assertEqual(
            prompt_packet["identity_candidates"][0]["canonical_key"],
            "p1",
        )
        self.assertEqual(prompt_packet["graph_metadata"]["canonical_key"], graph_key)
        self.assertNotIn(participant_key, request.user_prompt)

    def test_participant_source_bindings_do_not_cross_candidate_siblings(
        self,
    ) -> None:
        method = _main_method(
            "_participant_source_bindings",
            participant_source_bindings=compile_participant_source_bindings,
        )

        bindings = method(
            {
                "references": [
                    {
                        "candidate_participants": [
                            {
                                "participant_key": "synthetic-participant-a",
                                "alias_observations": [
                                    {"source_key": "synthetic-source-a"}
                                ],
                            },
                            {
                                "participant_key": "synthetic-participant-b",
                                "alias_observations": [
                                    {"source_key": "synthetic-source-b"}
                                ],
                            },
                        ]
                    }
                ]
            }
        )

        self.assertEqual(
            bindings,
            {
                "synthetic-participant-a": ["synthetic-source-a"],
                "synthetic-participant-b": ["synthetic-source-b"],
            },
        )

    def _hook_host(self, local_result: object, *, timeout: float = 0.2):
        provider_spy = Mock(side_effect=AssertionError("provider lookup is forbidden"))
        logger = Mock()
        scope = SimpleNamespace(key="scope-1")
        service = SimpleNamespace(finish_experiment=AsyncMock())
        host = SimpleNamespace(
            context=SimpleNamespace(get_provider_by_id=provider_spy),
            _group_scope=lambda event: scope,
            _session_allowed=lambda umo: True,
            _scope_event_carriers={},
            _services={scope.key: service},
            _service_scopes={scope.key: scope},
            _service_for_scope=Mock(return_value=service),
            _test_service=service,
            feedback_learning_enabled=False,
            local_serving_enabled=True,
            local_serving_timeout_seconds=timeout,
            local_serving_max_chars=12_000,
            _local_memory_for_request=local_result,
            _feedback_candidate_ids={},
            _local_serving_injected=set(),
            _active_surface_certificates={},
        )
        answer_context_value = {
            "schema_version": "memory-answer-context.v2",
            "memory_state": "supported",
            "referents": [],
            "facts": [{"statement": "synthetic-public-summary"}],
            "qualifications": {
                "do_not_upgrade": [],
                "conflicts": [],
                "unresolved": [],
            },
        }
        answer_context = SimpleNamespace(
            text=json.dumps(answer_context_value, ensure_ascii=False),
            required_fact_count=1,
            as_dict=Mock(return_value=answer_context_value),
        )
        method = _main_method(
            "inject_subconscious_memory",
            asyncio=asyncio,
            GroupScopeError=_GroupScopeError,
            logger=logger,
            json=json,
            ANSWER_CONTEXT_SCHEMA_VERSION="memory-answer-context.v2",
            SURFACE_SCHEMA_VERSION="memory-surface.v1",
            SurfaceCompilationError=ValueError,
            compile_answer_context_packet=Mock(return_value=answer_context),
            validate_answer_context_packet=Mock(),
            TextPart=_FakeTextPart,
        )
        return method, host, provider_spy, logger

    @staticmethod
    def _synthetic_surface(status: str) -> dict[str, object]:
        return {
            "schema_version": "memory-answer-context.v2",
            "memory_state": "supported",
            "referents": [],
            "facts": [{"statement": "synthetic-public-summary"}],
            "qualifications": {
                "do_not_upgrade": [],
                "conflicts": [],
                "unresolved": [],
            },
        }

    async def test_synthetic_memory_surface_is_injected_by_request_hook(self) -> None:
        for semantic_status in ("CERTIFIED", "PARTIAL", "SAFETY_ABSTAIN"):
            with self.subTest(semantic_status=semantic_status):
                surface = self._synthetic_surface(semantic_status)
                outcome = SimpleNamespace(
                    operational_status="COMPLETED",
                    semantic_status=semantic_status,
                    run_id=f"synthetic-{semantic_status.casefold()}",
                    detail="",
                    usable=True,
                    envelope_text=json.dumps(surface, ensure_ascii=False),
                    certificate=SimpleNamespace(
                        digest="synthetic-certificate-digest"
                    ),
                    ledger_result={
                        "surface_injection_status": "COMPILED_NOT_YET_INJECTED"
                    },
                )
                local_result = AsyncMock(return_value=outcome)
                method, host, provider_spy, _logger = self._hook_host(local_result)
                event = SimpleNamespace(
                    message_obj=SimpleNamespace(message_str="/chat 合成回忆请求")
                )
                request = SimpleNamespace(
                    prompt="合成宿主提示",
                    extra_user_content_parts=[],
                )

                returned = await method(host, event, request)

                self.assertIsNone(returned)
                self.assertEqual(request.prompt, "合成宿主提示")
                self.assertEqual(len(request.extra_user_content_parts), 1)
                injected = request.extra_user_content_parts[0].text
                opening = "<mr_memory_answer_context>"
                closing = "</mr_memory_answer_context>"
                self.assertEqual(injected.count(opening), 1)
                self.assertEqual(injected.count(closing), 1)
                encoded_context = injected.split(opening, 1)[1].split(closing, 1)[0]
                decoded_context = json.loads(encoded_context)
                self.assertEqual(
                    decoded_context["schema_version"],
                    "memory-answer-context.v2",
                )
                self.assertEqual(
                    decoded_context["facts"][0]["statement"],
                    "synthetic-public-summary",
                )
                self.assertNotIn("synthetic-certificate-digest", injected)
                self.assertIn(id(event), host._local_serving_injected)
                self.assertEqual(
                    host._active_surface_certificates[id(event)],
                    (
                        "scope-1",
                        f"synthetic-{semantic_status.casefold()}",
                        outcome.certificate,
                    ),
                )
                host._test_service.finish_experiment.assert_awaited_once()
                final = host._test_service.finish_experiment.await_args.kwargs
                self.assertEqual(final["status"], "completed")
                self.assertEqual(
                    final["result"]["surface_injection_status"],
                    "INJECTED_IN_REQUEST_HOOK",
                )
                self.assertFalse(final["result"]["raw_surface_injected"])
                self.assertEqual(final["result"]["answer_context_required_facts"], 1)
                provider_spy.assert_not_called()

    async def test_hook_rejects_wrong_protocol_and_unvalidated_envelope_content(self) -> None:
        wrong_protocol = self._synthetic_surface("CERTIFIED")
        wrong_protocol["schema_version"] = "unrelated-envelope"
        altered_content = self._synthetic_surface("CERTIFIED")
        altered_content["facts"] = [{"statement": "synthetic-unvalidated-content"}]
        for envelope, expected_status in (
            (wrong_protocol, "NOT_INJECTED_WRONG_PROTOCOL"),
            (altered_content, "NOT_INJECTED_INVALID_ANSWER_CONTEXT"),
        ):
            with self.subTest(expected_status=expected_status):
                outcome = SimpleNamespace(
                    operational_status="COMPLETED", semantic_status="CERTIFIED",
                    run_id="synthetic-envelope-boundary", detail="", usable=True,
                    envelope_text=json.dumps(envelope), certificate=object(),
                    ledger_result={},
                )
                method, host, provider_spy, logger = self._hook_host(AsyncMock(return_value=outcome))
                event = SimpleNamespace(message_obj=SimpleNamespace(message_str="/chat 合成回忆请求"))
                existing = _FakeTextPart(text="合成宿主既有上下文")
                request = SimpleNamespace(prompt="合成宿主提示", extra_user_content_parts=[existing])
                await method(host, event, request)
                self.assertEqual(request.prompt, "合成宿主提示")
                self.assertEqual(request.extra_user_content_parts, [existing])
                self.assertNotIn(id(event), host._local_serving_injected)
                self.assertNotIn(id(event), host._active_surface_certificates)
                host._test_service.finish_experiment.assert_awaited_once()
                final = host._test_service.finish_experiment.await_args.kwargs
                self.assertEqual(final["status"], "failed")
                self.assertEqual(final["result"]["surface_injection_status"], expected_status)
                logger.error.assert_called_once()
                provider_spy.assert_not_called()

    async def test_reader_provider_failure_does_not_inject_or_block_host(self) -> None:
        local_result = AsyncMock(
            side_effect=RuntimeError("synthetic resident reader provider failure")
        )
        method, host, provider_spy, logger = self._hook_host(local_result)
        event = SimpleNamespace(
            message_obj=SimpleNamespace(message_str="/chat 合成失败请求")
        )
        existing = _FakeTextPart(text="合成宿主既有上下文")
        request = SimpleNamespace(
            prompt="合成宿主提示",
            extra_user_content_parts=[existing],
        )

        returned = await method(host, event, request)

        self.assertIsNone(returned)
        self.assertEqual(request.prompt, "合成宿主提示")
        self.assertEqual(request.extra_user_content_parts, [existing])
        self.assertNotIn("<mr_memory_surface>", existing.text)
        self.assertNotIn(id(event), host._local_serving_injected)
        local_result.assert_awaited_once_with(event, "/chat 合成失败请求")
        host._test_service.finish_experiment.assert_not_awaited()
        logger.exception.assert_called_once()
        provider_spy.assert_not_called()

    async def test_sent_trace_uses_only_identifiers_from_the_final_outcome(
        self,
    ) -> None:
        method = _main_method(
            "trace_sent_artifacts",
            json=json,
            logger=Mock(),
        )
        event = SimpleNamespace(
            get_result=lambda: SimpleNamespace(chain=[]),
        )
        event_key = id(event)
        local_outcome = SimpleNamespace(
            operational_status="COMPLETED",
            semantic_status="CERTIFIED",
            run_id="local-run",
            usable=True,
            envelope_text=json.dumps(
                self._synthetic_surface("CERTIFIED"),
                ensure_ascii=False,
            ),
            source_keys=("source-1",),
            selected_edge_ids=(9,),
            selected_hypothesis_ids=(77,),
            ledger_result={"presented_aggregate_metadata": [{
                "aggregate_id": "activity-window:" + "a" * 64,
                "source_revision_sha256": "b" * 64,
                "source_count": 40,
                "scope": {"umo": "scope-1"},
            }]},
        )
        service = SimpleNamespace(
            experiment_report=AsyncMock(
                return_value={"run": {"status": "completed", "result": {}}}
            ),
            finish_experiment=AsyncMock(),
            record_memory_brief_trace=AsyncMock(),
            record_trace_node=AsyncMock(),
            record_trace_edge=AsyncMock(),
        )
        host = SimpleNamespace(
            _capture_visible_bot_output=AsyncMock(),
            _active_surface_certificates={event_key: object()},
            _local_serving_outcomes={
                event_key: (("scope-1", "source-request", "query-hash"), local_outcome)
            },
            _local_serving_injected={event_key},
            _local_serving_tasks={},
            _services={"scope-1": service},
            feedback_learning_enabled=True,
            _active_interaction_traces={
                event_key: ("scope-1", "trace-1", "source-request")
            },
            _trace_tool_counters={"trace-1": 0},
            _pending_main_tools={},
        )

        await method(host, event)

        service.record_memory_brief_trace.assert_awaited_once()
        self.assertNotIn(event_key, host._active_surface_certificates)
        trace_call = service.record_memory_brief_trace.await_args.kwargs
        self.assertEqual(trace_call["presented_edge_ids"], (9,))
        self.assertEqual(trace_call["presented_hypothesis_ids"], (77,))
        self.assertEqual(trace_call["source_keys"], ("source-1",))
        self.assertEqual(trace_call["presented_aggregate_metadata"],
                         local_outcome.ledger_result["presented_aggregate_metadata"])

    async def test_sent_trace_never_overwrites_a_failed_local_run(self) -> None:
        method = _main_method(
            "trace_sent_artifacts",
            json=json,
            logger=Mock(),
        )
        event = SimpleNamespace(get_result=lambda: SimpleNamespace(chain=[]))
        event_key = id(event)
        local_outcome = SimpleNamespace(
            operational_status="FAILED",
            run_id="failed-run",
        )
        service = SimpleNamespace(
            experiment_report=AsyncMock(),
            finish_experiment=AsyncMock(),
        )
        host = SimpleNamespace(
            _capture_visible_bot_output=AsyncMock(),
            _active_surface_certificates={event_key: object()},
            _local_serving_outcomes={
                event_key: (("scope-1", "source-request", "query-hash"), local_outcome)
            },
            _local_serving_injected=set(),
            _local_serving_tasks={},
            _services={"scope-1": service},
            feedback_learning_enabled=False,
        )

        await method(host, event)

        service.experiment_report.assert_not_awaited()
        self.assertNotIn(event_key, host._active_surface_certificates)
        service.finish_experiment.assert_not_awaited()

    async def test_sent_trace_never_reopens_failed_post_compile_run(self) -> None:
        method = _main_method(
            "trace_sent_artifacts",
            json=json,
            logger=Mock(),
        )
        event = SimpleNamespace(get_result=lambda: SimpleNamespace(chain=[]))
        event_key = id(event)
        local_outcome = SimpleNamespace(
            operational_status="COMPLETED",
            run_id="append-failed-run",
        )
        service = SimpleNamespace(
            experiment_report=AsyncMock(
                return_value={
                    "run": {
                        "status": "failed",
                        "result": {
                            "surface_injection_status": "NOT_INJECTED_APPEND_FAILED"
                        },
                    }
                }
            ),
            finish_experiment=AsyncMock(),
        )
        host = SimpleNamespace(
            _capture_visible_bot_output=AsyncMock(),
            _active_surface_certificates={event_key: object()},
            _local_serving_outcomes={
                event_key: (("scope-1", "source-request", "query-hash"), local_outcome)
            },
            _local_serving_injected=set(),
            _local_serving_tasks={},
            _services={"scope-1": service},
            feedback_learning_enabled=False,
        )

        await method(host, event)

        service.experiment_report.assert_awaited_once_with(run_id="append-failed-run")
        service.finish_experiment.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
