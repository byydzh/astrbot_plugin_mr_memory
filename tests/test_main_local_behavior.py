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


def _collect_source_keys(value: object) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            if key == "source_key" and isinstance(nested, str) and nested:
                result.add(nested)
            elif (
                (key == "source_keys" or key.endswith("_source_keys"))
                and isinstance(nested, list)
            ):
                result.update(str(item) for item in nested if str(item))
            else:
                result.update(_collect_source_keys(nested))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            result.update(_collect_source_keys(nested))
    return result


class MainLocalBehaviorTests(unittest.IsolatedAsyncioTestCase):
    def test_short_deictic_reference_requires_a_specific_identity_question(self) -> None:
        method = _main_method("_local_direct_reference_question", re=re)

        self.assertTrue(method("/chat 这人啥物种？"))
        self.assertTrue(method("/chat 这个人是什么品种"))
        self.assertTrue(method("/chat 在这个时间发这个表情包的人是什么猪"))
        self.assertFalse(method("/chat 这人喜欢什么"))
        self.assertFalse(method("/chat 啥物种？"))
        self.assertFalse(method("/chat 喜欢这个表情包的人是什么品种"))
        self.assertFalse(method("/chat 回忆这人以前说过什么"))

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
            _service_for_scope=lambda value: service,
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
            LOCAL_SERVING_SCHEMA_VERSION="mr-local-serving.v1",
            TextPart=_FakeTextPart,
        )
        return method, host, provider_spy, logger

    async def test_success_is_recorded_only_after_prompt_injection(self) -> None:
        outcome = SimpleNamespace(
            operational_status="COMPLETED",
            semantic_status="EVIDENCE_AVAILABLE",
            run_id="local-run",
            detail="",
            usable=True,
            envelope_text=json.dumps(
                {
                    "schema_version": "mr-local-serving.v1",
                    "source_records": [{"id": "s1", "text": "证据"}],
                },
                ensure_ascii=False,
            ),
            ledger_result={"surface_injection_status": "COMPILED_NOT_YET_INJECTED"},
        )
        method, host, provider_spy, _logger = self._hook_host(
            AsyncMock(return_value=outcome)
        )
        event = SimpleNamespace(message_obj=SimpleNamespace(message_str="/chat 回忆"))
        request = SimpleNamespace(prompt="", extra_user_content_parts=[])

        await method(host, event, request)

        self.assertEqual(len(request.extra_user_content_parts), 1)
        self.assertIn(id(event), host._local_serving_injected)
        host._test_service.finish_experiment.assert_awaited_once()
        final = host._test_service.finish_experiment.await_args.kwargs
        self.assertEqual(final["status"], "completed")
        self.assertEqual(
            final["result"]["surface_injection_status"],
            "INJECTED_IN_REQUEST_HOOK",
        )
        provider_spy.assert_not_called()

    async def test_ledger_failure_rolls_back_injected_evidence(self) -> None:
        outcome = SimpleNamespace(
            operational_status="COMPLETED",
            semantic_status="EVIDENCE_AVAILABLE",
            run_id="local-run",
            detail="",
            usable=True,
            envelope_text=json.dumps(
                {
                    "schema_version": "mr-local-serving.v1",
                    "source_records": [{"id": "s1", "text": "证据"}],
                },
                ensure_ascii=False,
            ),
            ledger_result={"surface_injection_status": "COMPILED_NOT_YET_INJECTED"},
        )
        method, host, provider_spy, logger = self._hook_host(
            AsyncMock(return_value=outcome)
        )
        host._test_service.finish_experiment.side_effect = RuntimeError(
            "ledger unavailable"
        )
        event = SimpleNamespace(message_obj=SimpleNamespace(message_str="/chat 回忆"))
        request = SimpleNamespace(prompt="", extra_user_content_parts=[])

        await method(host, event, request)

        self.assertEqual(request.extra_user_content_parts, [])
        self.assertNotIn(id(event), host._local_serving_injected)
        host._test_service.finish_experiment.assert_awaited_once()
        logger.exception.assert_called_once()
        logger.error.assert_called_once()
        provider_spy.assert_not_called()

    async def test_append_failure_is_terminal_failed_not_completed(self) -> None:
        class RejectingParts(list):
            def append(self, _value: object) -> None:
                raise RuntimeError("prompt is sealed")

        outcome = SimpleNamespace(
            operational_status="COMPLETED",
            semantic_status="EVIDENCE_AVAILABLE",
            run_id="local-run",
            detail="",
            usable=True,
            envelope_text=json.dumps(
                {"schema_version": "mr-local-serving.v1", "source_records": []}
            ),
            ledger_result={"surface_injection_status": "COMPILED_NOT_YET_INJECTED"},
        )
        method, host, provider_spy, logger = self._hook_host(
            AsyncMock(return_value=outcome)
        )
        event = SimpleNamespace(message_obj=SimpleNamespace(message_str="/chat 回忆"))
        request = SimpleNamespace(prompt="", extra_user_content_parts=RejectingParts())

        await method(host, event, request)

        self.assertNotIn(id(event), host._local_serving_injected)
        final = host._test_service.finish_experiment.await_args.kwargs
        self.assertEqual(final["status"], "failed")
        self.assertEqual(
            final["result"]["surface_injection_status"],
            "NOT_INJECTED_APPEND_FAILED",
        )
        self.assertEqual(final["result"]["operational_status"], "FAILED")
        logger.exception.assert_called_once()
        provider_spy.assert_not_called()

    async def test_hook_does_not_race_the_owned_retrieval_deadline(self) -> None:
        async def delayed_result(event: object, query: str) -> object:
            await asyncio.sleep(0.01)
            return SimpleNamespace(
                operational_status="FAILED",
                semantic_status="UNKNOWN",
                run_id="timed-out-local-run",
                detail="Local memory serving exceeded its owned hard deadline",
                usable=False,
                envelope_text="",
                ledger_result=None,
            )

        method, host, provider_spy, logger = self._hook_host(
            delayed_result,
            timeout=0.001,
        )
        event = SimpleNamespace(
            message_obj=SimpleNamespace(message_str="/chat 回忆一下")
        )
        request = SimpleNamespace(prompt="", extra_user_content_parts=[])

        await method(host, event, request)

        self.assertEqual(request.extra_user_content_parts, [])
        self.assertNotIn(id(event), host._local_serving_injected)
        provider_spy.assert_not_called()
        logger.error.assert_called_once()
        self.assertIn(
            "produced no injectable result",
            logger.error.call_args.args[0],
        )

    async def test_missing_preloaded_service_does_not_block_host(self) -> None:
        local_result = AsyncMock(
            side_effect=AssertionError("local retrieval must not run")
        )
        method, host, provider_spy, logger = self._hook_host(local_result)
        host._services = {}
        event = SimpleNamespace(
            message_obj=SimpleNamespace(message_str="/chat 回忆一下")
        )
        request = SimpleNamespace(prompt="", extra_user_content_parts=[])

        await method(host, event, request)

        self.assertEqual(request.extra_user_content_parts, [])
        self.assertNotIn(id(event), host._local_serving_injected)
        local_result.assert_not_awaited()
        provider_spy.assert_not_called()
        logger.error.assert_called_once()
        self.assertIn(
            "preloaded local service is unavailable",
            logger.error.call_args.args[0],
        )

    async def test_failed_local_result_does_not_inject_or_lookup_a_provider(self) -> None:
        failed = SimpleNamespace(
            operational_status="FAILED",
            semantic_status="UNKNOWN",
            run_id="failed-run",
            detail="local retrieval failed",
            usable=False,
            envelope_text="",
        )
        local_result = AsyncMock(return_value=failed)
        method, host, provider_spy, logger = self._hook_host(local_result)
        event = SimpleNamespace(
            message_obj=SimpleNamespace(message_str="/chat 回忆一下")
        )
        request = SimpleNamespace(prompt="", extra_user_content_parts=[])

        await method(host, event, request)

        self.assertEqual(request.extra_user_content_parts, [])
        self.assertNotIn(id(event), host._local_serving_injected)
        provider_spy.assert_not_called()
        logger.error.assert_called_once()

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
            run_id="local-run",
            usable=True,
            envelope_text=json.dumps(
                {
                    "memory_brief": {"claims": [], "conflicts": [], "unresolved": []},
                    "graph_connections": [{"edge_id": 9}],
                    "learned_patterns": [{"hypothesis_id": 77}],
                }
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
