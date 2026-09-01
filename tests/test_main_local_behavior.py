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

from mr_memory.identity import build_request_identity_context
from mr_memory.snapshot import stable_sha256

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
            Any=object,
            _collect_source_keys=collect_source_keys,
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
            stable_sha256=stable_sha256,
        )
        recent_participant = 'participant:["synthetic","recent-account"]'

        class Service:
            def __init__(self) -> None:
                self.history_keys: list[str] = []
                self.activity_keys: list[str] = []
                self.resolve_query_participants = AsyncMock(
                    side_effect=AssertionError(
                        "text alias parser must not decide resident identity"
                    )
                )

            async def query_plastic_associations(self, **_kwargs: object) -> list:
                return []

            async def query_matching_cues(self, **_kwargs: object) -> list:
                return []

            async def reconstruction_evidence_packet(
                self, **_kwargs: object
            ) -> dict[str, object]:
                return {
                    "candidates": {"participants": []},
                    "semantic_evidence": [],
                    "expanded_episodes": [],
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

        service = Service()
        host = SimpleNamespace(
            _layered_pack_key=lambda *_args, **_kwargs: "synthetic-pack",
            feedback_learning_enabled=False,
            embedding_top_k=8,
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
        self.assertEqual(service.history_keys, [recent_participant])
        self.assertEqual(service.activity_keys, [recent_participant])
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
            {"synthetic-history-source", "synthetic-recent-source"},
        )
        self.assertEqual(
            packet["participant_source_keys"][recent_participant],
            ["synthetic-history-source", "synthetic-recent-source"],
        )
        self.assertTrue(packet["retrieval_coverage"]["semantic_none_allowed"])
        self.assertIn(recent_participant, participant_keys)
        self.assertEqual(cache_layer, "LOCAL_FRESH")

    def test_participant_source_bindings_do_not_cross_candidate_siblings(
        self,
    ) -> None:
        collect_source_keys = _main_method("_collect_source_keys")
        method = _main_method(
            "_participant_source_bindings",
            Any=object,
            _collect_source_keys=collect_source_keys,
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
            _local_memory_for_request=local_result,
            _feedback_candidate_ids={},
            _local_serving_injected=set(),
        )
        method = _main_method(
            "inject_subconscious_memory",
            asyncio=asyncio,
            GroupScopeError=_GroupScopeError,
            logger=logger,
            json=json,
            SURFACE_SCHEMA_VERSION="memory-surface.v1",
            TextPart=_FakeTextPart,
        )
        return method, host, provider_spy, logger

    @staticmethod
    def _synthetic_surface(status: str) -> dict[str, object]:
        return {
            "schema_version": "memory-surface.v1",
            "certificate_sha256": "synthetic-certificate-digest",
            "snapshot_sha256": "synthetic-snapshot-digest",
            "status": status,
            "scope": {"umo": "synthetic:GroupMessage:scope", "cutoff_at": 123},
            "subjects": [],
            "evidence": {"required": [], "optional": []},
            "contract": {
                "must_include": [],
                "must_not_upgrade": [],
                "conflicts": [],
                "unresolved": [],
                "open_obligations": [],
            },
            "stop_reason": "SYNTHETIC_TEST",
            "omitted_optional": 0,
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
                opening = "<mr_memory_surface>"
                closing = "</mr_memory_surface>"
                self.assertEqual(injected.count(opening), 1)
                self.assertEqual(injected.count(closing), 1)
                encoded_surface = injected.split(opening, 1)[1].split(closing, 1)[0]
                self.assertEqual(json.loads(encoded_surface), surface)
                self.assertIn(id(event), host._local_serving_injected)
                host._test_service.finish_experiment.assert_awaited_once()
                final = host._test_service.finish_experiment.await_args.kwargs
                self.assertEqual(final["status"], "completed")
                self.assertEqual(
                    final["result"]["surface_injection_status"],
                    "INJECTED_IN_REQUEST_HOOK",
                )
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
        trace_call = service.record_memory_brief_trace.await_args.kwargs
        self.assertEqual(trace_call["presented_edge_ids"], (9,))
        self.assertEqual(trace_call["presented_hypothesis_ids"], (77,))
        self.assertEqual(trace_call["source_keys"], ("source-1",))

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
