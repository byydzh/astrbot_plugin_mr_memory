from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

from mr_memory.routing import RouteFeatures, RoutePolicy


class MainLayeredWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (Path.cwd() / "main.py").read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    @classmethod
    def _method(cls, name: str) -> ast.AsyncFunctionDef | ast.FunctionDef:
        for node in ast.walk(cls.tree):
            if (
                isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
                and node.name == name
            ):
                return node
        raise AssertionError(f"main.py does not define {name}")

    def _runtime_request_classifier(self):
        activity_method = self._method("_runtime_activity_analysis")
        isolated_activity = ast.FunctionDef(
            name="_runtime_activity_analysis",
            args=activity_method.args,
            body=activity_method.body,
            decorator_list=[],
            returns=activity_method.returns,
            type_comment=activity_method.type_comment,
        )
        identity_method = self._method("_explicit_identity_intent")
        isolated_identity = ast.FunctionDef(
            name="_explicit_identity_intent",
            args=identity_method.args,
            body=identity_method.body,
            decorator_list=[],
            returns=identity_method.returns,
            type_comment=identity_method.type_comment,
        )
        method = self._method("_runtime_request_kind")
        isolated_classifier = ast.FunctionDef(
            name="classify",
            args=method.args,
            body=method.body,
            decorator_list=[],
            returns=method.returns,
            type_comment=method.type_comment,
        )
        namespace: dict[str, object] = {"re": re}
        helpers = ast.fix_missing_locations(
            ast.Module(
                body=[isolated_activity, isolated_identity],
                type_ignores=[],
            )
        )
        exec(compile(helpers, "<runtime-request-helpers>", "exec"), namespace)
        namespace["MrMemoryPlugin"] = type(
            "MrMemoryPlugin",
            (),
            {
                "_runtime_activity_analysis": staticmethod(
                    namespace["_runtime_activity_analysis"]
                ),
                "_explicit_identity_intent": staticmethod(
                    namespace["_explicit_identity_intent"]
                ),
            },
        )
        classifier_module = ast.fix_missing_locations(
            ast.Module(body=[isolated_classifier], type_ignores=[])
        )
        exec(
            compile(classifier_module, "<runtime-request-kind>", "exec"),
            namespace,
        )
        return namespace["classify"]

    def test_production_local_outcome_keeps_unresolved_identity_injectable(self) -> None:
        outcome_class = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.ClassDef) and node.name == "_LocalMemoryOutcome"
        )
        usable = next(
            node
            for node in outcome_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "usable"
        )
        body = ast.unparse(usable)
        self.assertIn("EVIDENCE_AVAILABLE", body)
        self.assertIn("IDENTITY_AMBIGUOUS", body)
        self.assertIn("IDENTITY_UNRESOLVED", body)

    def test_host_return_decision_cannot_fall_through_to_provider(self) -> None:
        method = self._method("_run_layered_subconscious")
        body = ast.unparse(method)
        return_guard = body.index("decision.execution == 'RETURN'")
        provider_guard = body.index("if provider is None")
        start = body.index("self._runtime_singleflight.start")
        self.assertLess(return_guard, provider_guard)
        self.assertLess(return_guard, start)
        self.assertIn("semantic_status=decision.semantic_status", body)

    def test_missing_feedback_evidence_is_rejected_without_retry(self) -> None:
        method = self._method("_run_feedback_maintenance")
        body = ast.unparse(method)
        self.assertIn("except FeedbackEvidenceUnavailableError as exc", body)
        self.assertIn("await service.reject_feedback_proposal", body)
        self.assertIn("proposals = attributable_proposals", body)

    def test_singleflight_is_bound_to_snapshot_and_route(self) -> None:
        method = self._method("_run_layered_subconscious")
        body = ast.unparse(method)
        self.assertIn("'snapshot_sha256': snapshot.digest", body)
        self.assertIn("'route_level': route_level", body)
        self.assertIn("self._runtime_singleflight.start(flight_key, factory", body)
        self.assertNotIn(
            "self._runtime_singleflight.start(certificate_key, factory", body
        )

    def test_layered_run_uses_only_the_request_bound_interaction_trace(self) -> None:
        method = self._method("_run_layered_subconscious")
        body = ast.unparse(method)
        self.assertIn("request_source_key = normalized.resolved_source_key()", body)
        self.assertIn("self._active_interaction_traces.get(id(event))", body)
        self.assertIn("active_trace[0] == scope.key", body)
        self.assertIn("active_trace[2] == request_source_key", body)
        self.assertGreaterEqual(
            body.count("interaction_trace_id=interaction_trace_id"),
            2,
        )

        producer = self._method("_execute_layered_reconstruction")
        producer_body = ast.unparse(producer)
        self.assertIn(
            "interaction_trace_id",
            [arg.arg for arg in producer.args.kwonlyargs],
        )
        self.assertIn(
            "interaction_trace_id=interaction_trace_id",
            producer_body,
        )

    def test_layered_run_persists_trace_id_in_all_terminal_shapes(self) -> None:
        producer = self._method("_execute_layered_reconstruction_started")
        producer_body = ast.unparse(producer)
        self.assertIn(
            "interaction_trace_id",
            [arg.arg for arg in producer.args.kwonlyargs],
        )
        self.assertGreaterEqual(
            producer_body.count("'trace_id': interaction_trace_id"),
            3,
            "start metadata plus completed and failed results must remain linked",
        )

        budget = self._method("_record_layered_budget_block")
        budget_body = ast.unparse(budget)
        self.assertIn(
            "interaction_trace_id",
            [arg.arg for arg in budget.args.kwonlyargs],
        )
        self.assertEqual(
            budget_body.count("'trace_id': interaction_trace_id"),
            2,
            "budget-blocked metadata and result must use the same trace",
        )

    def test_only_singleflight_producer_runs_budget_preflight(self) -> None:
        method = self._method("_run_layered_subconscious")
        outer_budget_calls = 0
        factory_budget_calls = 0
        for node in ast.walk(method):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "_private_budget_available":
                continue
            enclosing_factory = any(
                isinstance(parent, (ast.AsyncFunctionDef, ast.FunctionDef))
                and parent.name == "factory"
                and node in tuple(ast.walk(parent))
                for parent in method.body
            )
            if enclosing_factory:
                factory_budget_calls += 1
            else:
                outer_budget_calls += 1
        self.assertEqual(factory_budget_calls, 1)
        self.assertEqual(outer_budget_calls, 0)

    def test_explicit_recall_uses_provider_deadline_not_chat_wait_budget(self) -> None:
        method = self._method("_run_layered_subconscious")
        body = ast.unparse(method)
        self.assertIn(
            "request_kind in {'MEMORY_QUERY', 'DEEP_RECALL'}",
            body,
        )
        self.assertIn("producer_deadline_seconds =", body)
        self.assertIn("self.subconscious_timeout_seconds", body)
        self.assertIn("self.runtime_l3_deadline_seconds", body)
        self.assertIn("timeout_seconds = producer_deadline_seconds + 5", body)
        self.assertIn("elif decision.execution == 'SYNC'", body)

        hard_sync_assignment = next(
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "hard_sync"
                for target in node.targets
            )
        )
        self.assertEqual(
            ast.unparse(hard_sync_assignment.value),
            "force or request_kind in {'MEMORY_QUERY', 'DEEP_RECALL'}",
        )
        self.assertNotIn("route_level", ast.unparse(hard_sync_assignment.value))

        timeout_try = next(
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Try)
            and "asyncio.timeout(timeout_seconds)"
            in "\n".join(ast.unparse(item) for item in node.body)
        )
        timeout_handler = next(
            handler
            for handler in timeout_try.handlers
            if handler.type is not None
            and "TimeoutError" in ast.unparse(handler.type)
        )
        cancel_guard = next(
            node
            for node in timeout_handler.body
            if isinstance(node, ast.If)
        )
        self.assertEqual(ast.unparse(cancel_guard.test), "hard_sync")


    def test_ordinary_chat_uses_full_local_retrieval_with_small_envelope(self) -> None:
        method = self._method("_execute_local_memory_serving")
        body = ast.unparse(method)
        self.assertIn("identity_question = self._local_direct_identity_question", body)
        self.assertIn("if include_participant_activity or identity_question", body)
        self.assertNotIn("request_kind == 'CHAT' and", body)
        self.assertIn("await self._layered_evidence_packet", body)
        self.assertIn(
            "min(self.local_serving_max_chars, 3000) if request_kind == 'CHAT'",
            body,
        )


    def test_existing_feedback_trace_reuses_local_request_outcome(
        self,
    ) -> None:
        """Repeated hooks keep feedback tracing but join one local serving result."""

        method = self._method("inject_subconscious_memory")
        hook_body = ast.unparse(method)
        self.assertIn("await self._local_memory_for_request(event, query)", hook_body)
        self.assertNotIn(
            "asyncio.timeout(self.local_serving_timeout_seconds)",
            hook_body,
        )
        executor_body = ast.unparse(self._method("_execute_local_memory_serving"))
        self.assertEqual(
            executor_body.count(
                "asyncio.timeout(self.local_serving_timeout_seconds)"
            ),
            1,
        )
        self.assertIn("await self._begin_interaction_trace", executor_body)
        self.assertNotIn("serving_deadline.reschedule", executor_body)
        self.assertIn("'last_stage': last_stage", executor_body)

    def test_full_retrieval_has_no_short_stage_deadline(self) -> None:
        body = ast.unparse(self._method("_execute_local_memory_serving"))
        full_branch = body[body.index("begin_stage('FULL_RETRIEVAL_PRECHECK')") :]

        self.assertNotIn("serving_deadline.reschedule", full_branch)
        self.assertNotIn("self.local_serving_timeout_seconds * 4.0", full_branch)
        self.assertIn("'stage_elapsed_ms': dict(stage_elapsed_ms)", body)
        for stage in (
            "SNAPSHOT_CAPTURE",
            "RUNTIME_READY",
            "SERVICE_READY",
            "INTERACTION_TRACE",
            "EXPERIMENT_START",
            "DIRECT_RETRIEVAL",
            "FULL_RETRIEVAL",
            "PACKET_MATERIALIZE",
            "ENVELOPE_COMPILE",
            "SOURCE_AUDIT",
            "LEDGER_RECORD",
        ):
            self.assertIn(f"begin_stage('{stage}')", body)
            self.assertIn(f"finish_stage('{stage}')", body)
        self.assertNotIn("_local_full_retrieval_lock", body)
        self.assertIn("finalize_packet=False", full_branch)
        self.assertIn("stage_elapsed_ms=stage_elapsed_ms", full_branch)
        local_method = ast.unparse(self._method("_local_memory_for_request"))
        self.assertIn("self._local_serving_outcomes.get(event_key)", local_method)
        self.assertIn("self._local_serving_tasks.get(event_key)", local_method)
        self.assertIn("await running[1]", local_method)

    def test_structured_short_reference_routes_before_full_retrieval(self) -> None:
        body = ast.unparse(self._method("_execute_local_memory_serving"))
        reference_index = body.index(
            "reference_question = self._local_direct_reference_question"
        )
        structured_index = body.index("has_structured_reference = bool")
        direct_index = body.index("await self._local_identity_evidence_packet")
        full_index = body.index("await self._layered_evidence_packet")

        self.assertLess(reference_index, direct_index)
        self.assertLess(structured_index, direct_index)
        self.assertLess(direct_index, full_index)
        self.assertIn(
            "reference_question and has_structured_reference",
            body,
        )

    def test_local_serving_failure_is_logged_but_never_injected(self) -> None:
        """Failure is ledger-only; it must not become substitute prompt content."""

        method = self._method("inject_subconscious_memory")
        method_body = ast.unparse(method)
        self.assertNotIn("MEMORY_LOOKUP_OPERATIONAL_FAILURE", method_body)
        self.assertNotIn("mr_memory_operational_failure", method_body)
        unusable_guard = None
        for node in ast.walk(method):
            if not isinstance(node, ast.If):
                continue
            if ast.unparse(node.test) == "not outcome.usable":
                unusable_guard = node
                break
        self.assertIsNotNone(unusable_guard)
        assert unusable_guard is not None
        guard_body = "\n".join(ast.unparse(node) for node in unusable_guard.body)
        self.assertNotIn("req.extra_user_content_parts.append", guard_body)
        self.assertIn("logger.error", guard_body)
        self.assertNotIn("_stop_failed_memory_query", guard_body)
        self.assertIn("outcome.operational_status", guard_body)
        self.assertIn("outcome.detail", guard_body)

        # Local orchestration exceptions are explicit and never become prompt text.
        run_try = next(
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Try)
            and "await self._local_memory_for_request(event, query)"
            in "\n".join(ast.unparse(item) for item in node.body)
        )
        self.assertEqual(len(run_try.handlers), 1)
        for handler in run_try.handlers:
            with self.subTest(handler=ast.unparse(handler.type)):
                handler_body = "\n".join(
                    ast.unparse(node) for node in handler.body
                )
                self.assertTrue(
                    "logger.error" in handler_body
                    or "logger.exception" in handler_body
                )
                self.assertIn("return", handler_body)
                self.assertNotIn("req.extra_user_content_parts.append", handler_body)

        self.assertNotIn("prospective_part", method_body)
        self.assertNotIn("_run_subconscious", method_body)
        self.assertNotIn("get_provider_by_id", method_body)

        feedback_try = run_try
        feedback_failures = "\n".join(
            ast.unparse(node)
            for handler in feedback_try.handlers
            for node in handler.body
        )
        self.assertIn("no memory injected", feedback_failures)
        self.assertIn("return", feedback_failures)
        self.assertNotIn("failed open", feedback_failures.casefold())

        parse_try = next(
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Try)
            and "envelope_value = json.loads(outcome.envelope_text)"
            in "\n".join(ast.unparse(item) for item in node.body)
        )
        parse_failure_body = "\n".join(
            ast.unparse(node) for node in parse_try.handlers[0].body
        )
        self.assertIn("logger.error", parse_failure_body)
        self.assertNotIn("_stop_failed_memory_query", parse_failure_body)
        self.assertNotIn(
            "req.extra_user_content_parts.append",
            parse_failure_body,
        )

    def test_completed_empty_local_result_is_not_rewritten_as_failure(self) -> None:
        method = self._method("inject_subconscious_memory")
        completed_guard = next(
            node
            for node in method.body
            if isinstance(node, ast.If)
            and "outcome.operational_status == 'COMPLETED'" in ast.unparse(node.test)
            and "not outcome.usable" in ast.unparse(node.test)
        )
        guard_body = "\n".join(ast.unparse(node) for node in completed_guard.body)
        self.assertIn("return", guard_body)
        self.assertNotIn("logger.error", guard_body)
        self.assertNotIn("req.extra_user_content_parts.append", guard_body)

        completed_index = method.body.index(completed_guard)
        failure_index = next(
            index
            for index, node in enumerate(method.body)
            if isinstance(node, ast.If) and ast.unparse(node.test) == "not outcome.usable"
        )
        self.assertLess(completed_index, failure_index)

    def test_subconscious_failure_does_not_hijack_the_host_event(self) -> None:
        body = ast.unparse(self._method("inject_subconscious_memory"))
        self.assertNotIn("event.set_result", body)
        self.assertNotIn("event.plain_result", body)
        self.assertNotIn("event.stop_event", body)
        self.assertNotIn("_stop_failed_memory_query", self.source)

    def test_sync_waiter_timeout_cancels_producer_and_is_terminal(self) -> None:
        method = self._method("_run_layered_subconscious")
        timeout_try = next(
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Try)
            and "asyncio.timeout(timeout_seconds)"
            in "\n".join(ast.unparse(item) for item in node.body)
        )
        timeout_handler = next(
            handler
            for handler in timeout_try.handlers
            if handler.type is not None
            and "TimeoutError" in ast.unparse(handler.type)
        )
        timeout_body = "\n".join(
            ast.unparse(node) for node in timeout_handler.body
        )
        hard_guard = next(
            node
            for node in timeout_handler.body
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "hard_sync"
        )
        hard_body = ast.unparse(hard_guard)
        self.assertIn("task.cancel()", hard_body)
        self.assertIn("await task", hard_body)
        self.assertIn("operational_status='TIMEOUT'", hard_body)
        self.assertIn("run_id=cancelled_run_id", hard_body)
        self.assertIn("failure_persisted=failure_persisted", hard_body)
        self.assertIn("mr_memory_run_id", hard_body)
        self.assertIn("mr_memory_failure_persisted", hard_body)
        self.assertNotIn("operational_status='RUNNING'", hard_body)
        self.assertNotIn("continues", timeout_body)

        cancellation_handler = next(
            handler
            for handler in timeout_try.handlers
            if handler.type is not None
            and ast.unparse(handler.type) == "asyncio.CancelledError"
        )
        cancellation_body = "\n".join(
            ast.unparse(node) for node in cancellation_handler.body
        )
        self.assertIn("if hard_sync and (not task.done())", cancellation_body)
        self.assertIn("task.cancel()", cancellation_body)
        self.assertIn("await task", cancellation_body)
        self.assertIn("producer cancelled", cancellation_body)
        self.assertTrue(
            isinstance(cancellation_handler.body[-1], ast.Raise)
            and cancellation_handler.body[-1].exc is None
        )

        producer = self._method("_execute_layered_reconstruction_started")
        producer_body = ast.unparse(producer)
        self.assertIn("setattr(exc, 'mr_memory_run_id', run_id)", producer_body)
        self.assertIn(
            "setattr(exc, 'mr_memory_failure_persisted', failure_persisted)",
            producer_body,
        )

    def test_subconscious_failure_ledger_is_terminal_and_privacy_safe(self) -> None:
        method = self._method("_record_subconscious_surface_failure")
        body = ast.unparse(method)
        self.assertIn("experiment_type='runtime_layered_reconstruction'", body)
        self.assertIn("status='failed'", body)
        self.assertIn("'surface_injection_status': 'FAILED'", body)
        self.assertIn("query_sha256=_stable_hash(query)", body)
        self.assertNotIn("'query': query", body)
        self.assertNotIn("outcome.failure_persisted and (not run_id)", body)
        self.assertIn("await service.experiment_report(run_id=run_id)", body)
        self.assertIn("existing_experiment = False", body)
        self.assertIn("if run_id and existing_experiment", body)

    def test_reader_revision_includes_actual_provider_model(self) -> None:
        method = self._method("_runtime_inference_revision")
        body = ast.unparse(method)
        self.assertIn("_provider_model_name(provider)", body)
        self.assertIn("self._last_reader_model_revision", body)
        self.assertIn("'reader_model': reader_model_revision", body)

    def test_certificate_store_rechecks_snapshot_after_source_audit(self) -> None:
        method = self._method("_store_layered_certificate")
        body = ast.unparse(method)
        self.assertNotIn("PROTOCOL_DEGRADED", body)
        self.assertGreaterEqual(body.count("self._assert_snapshot_fresh"), 2)
        audit = body.index("await service.audit_snapshot_sources")
        last_fresh = body.rindex("await self._assert_snapshot_fresh")
        put = body.index("await service.put_memory_certificate")
        self.assertLess(audit, last_fresh)
        self.assertLess(last_fresh, put)

    def test_l3_trace_serializes_repair_audit_without_degraded_success(self) -> None:
        method = self._method("_run_l3_certificate")
        body = ast.unparse(method)
        self.assertIn("'repair_attempted': result.repair_attempted", body)
        self.assertIn("item.as_dict() for item in result.protocol_failures", body)
        self.assertNotIn("result.degraded", body)
        self.assertNotIn("PROTOCOL_DEGRADED", body)

        producer = ast.unparse(self._method("_execute_layered_reconstruction_started"))
        self.assertIn("getattr(exc, 'protocol_failures', ())", producer)
        self.assertIn("'protocol_failures': protocol_failures", producer)
        self.assertIn("'repair_attempted': protocol_repair_attempted", producer)

    def test_reply_target_is_snapshot_bounded_packet_evidence(self) -> None:
        revision = ast.unparse(self._method("_runtime_inference_revision"))
        self.assertIn("host-prefetch.snapshot.v6", revision)
        self.assertIn("lexical-plus-embedding-plus-activity-plus-graph.v5", revision)

        method = self._method("_layered_evidence_packet")
        body = ast.unparse(method)
        reply_lookup = body.index("await service.message_for_source")
        packet_hash = body.index("packet_sha256 = stable_sha256(packet)")
        cache_write = body.index("await service.put_evidence_pack_cache")
        self.assertIn("packet['reply_context']", body)
        self.assertIn("source_key=snapshot.reply_source_key", body)
        self.assertIn("before_sent_at=snapshot.cutoff_at", body)
        self.assertIn("message_upper_bound=snapshot.message_upper_bound", body)
        self.assertLess(reply_lookup, packet_hash)
        self.assertLess(packet_hash, cache_write)

    def test_current_request_identity_is_snapshot_bound_and_prefetched(self) -> None:
        capture = ast.unparse(self._method("_capture_layered_snapshot"))
        self.assertIn("build_request_identity_context", capture)
        self.assertIn("'request_identity_context': request_identity_context", capture)
        self.assertIn("identity_snapshot=request_identity_context", capture)

        packet = ast.unparse(self._method("_layered_evidence_packet"))
        pack_key = ast.unparse(self._method("_layered_pack_key"))
        self.assertIn("'resolve_query_aliases': bool(resolve_query_aliases)", pack_key)
        self.assertIn(
            "'include_participant_activity': bool(include_participant_activity)",
            pack_key,
        )
        self.assertIn(
            "resolve_query_aliases=resolve_query_aliases",
            packet,
        )
        self.assertIn(
            "include_participant_activity=include_participant_activity",
            packet,
        )
        self.assertIn("request_identity_context['mentions']", packet)
        self.assertIn("reference=account_id", packet)
        self.assertIn("packet['request_identity_context']", packet)
        self.assertIn("await service.resolve_query_participants", packet)
        self.assertIn("await service.query_participant_activity", packet)
        self.assertIn("if include_participant_activity", packet)
        self.assertIn("limit=64", packet)
        self.assertIn("packet['query_alias_resolution']", packet)
        self.assertIn("packet['participant_activity']", packet)
        self.assertIn(
            "packet['source_count'] = len(_collect_source_keys(packet))",
            packet,
        )
        self.assertIn("before_sent_at=snapshot.cutoff_at", packet)
        self.assertIn("message_upper_bound=snapshot.message_upper_bound", packet)

    def test_lexical_and_embedding_candidates_are_combined_without_fallback(
        self,
    ) -> None:
        method = self._method("_layered_evidence_packet")
        body = ast.unparse(method)
        lexical_call = body.index("await service.query_matching_cues")
        embedding_call = body.index("await service.initialize_candidates")
        self.assertLess(lexical_call, embedding_call)
        self.assertIn("[*initial['cues'], *embedded_cues]", body)
        self.assertIn("cue_by_text", body)
        self.assertNotIn("if lexical_matches", body)
        self.assertIn("max_episodes=min(6, self.embedding_top_k)", body)
        self.assertIn("max_messages=48", body)
        self.assertFalse(
            any(
                isinstance(node, ast.Try)
                and "service.initialize_candidates" in ast.unparse(node)
                for node in ast.walk(method)
            ),
            "embedding errors must reach the existing ERROR/failed-ledger path",
        )

    def test_l2_reader_uses_disabled_thinking_and_independent_output_cap(
        self,
    ) -> None:
        method = self._method("_read_l2_certificate")
        calls = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_run_fast_reconstruction_with_ledger"
        ]
        self.assertEqual(len(calls), 2)
        for call in calls:
            keywords = {item.arg: item.value for item in call.keywords if item.arg}
            with self.subTest(call=ast.unparse(call)):
                self.assertEqual(ast.literal_eval(keywords["thinking_mode"]), "disabled")
                self.assertEqual(ast.literal_eval(keywords["max_output_tokens"]), 8192)

        runner = ast.unparse(self._method("_run_fast_reconstruction_with_ledger"))
        self.assertIn("if max_output_tokens is None", runner)
        self.assertIn("else max(1, int(max_output_tokens))", runner)
        self.assertNotIn(
            "max_output_tokens=8192",
            ast.unparse(self._method("_run_l3_certificate")),
        )

    def test_l2_reader_never_uses_hidden_reasoning_as_a_fallback(self) -> None:
        body = ast.unparse(self._method("_read_l2_certificate"))
        self.assertNotIn("parse_structured_response", body)
        self.assertNotIn("reasoning_content", body)
        self.assertIn("certificate = parse(", body)
        self.assertIn("certificate = parse_repair(", body)
        self.assertIn("response_source = 'completion'", body)

    def test_hot_reload_never_unboundedly_gathers_cancelled_tasks(self) -> None:
        method = self._method("terminate")
        body = ast.unparse(method)
        self.assertNotIn("await asyncio.gather", body)
        self.assertIn("await asyncio.wait({drain_task}, timeout=15)", body)
        self.assertIn("self._runtime_singleflight.drain(cancel=True)", body)
        self.assertIn("await asyncio.wait(pending_inflight, timeout=5)", body)
        self.assertIn("await asyncio.wait(set(tasks), timeout=10)", body)

    def test_invalid_runtime_backend_configuration_never_falls_back(self) -> None:
        plugin_class = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.ClassDef) and node.name == "MrMemoryPlugin"
        )
        plugin_init = next(
            node
            for node in plugin_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        body = ast.unparse(plugin_init)
        self.assertIn("Unsupported MR Memory embedding_backend", body)
        self.assertIn("Unsupported MR Memory distillation_thinking_mode", body)
        self.assertGreaterEqual(body.count("raise ValueError"), 2)
        self.assertNotIn("using fastembed", body)
        self.assertNotIn("using enabled", body)

    def test_tool_state_errors_and_mismatches_are_terminal(self) -> None:
        method = self._method("_apply_tool_state")
        body = ast.unparse(method)
        self.assertFalse(
            any(isinstance(node, ast.Try) for node in ast.walk(method)),
            "tool-state application must not swallow activation errors",
        )
        self.assertIn("missing tool", body)
        self.assertIn("tool state verification failed", body)
        self.assertNotIn("logger.warning", body)
        self.assertNotIn("self.expose_traversal_tools", body)
        self.assertIn("tool_name: False", body)
        self.assertIn("manager.get_func(tool_name)", body)
        self.assertGreaterEqual(body.count("raise RuntimeError"), 2)

    def test_distillation_validation_failure_never_repairs_or_sanitizes(self) -> None:
        method = self._method("_distill_scope")
        body = ast.unparse(method)
        provider_calls = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "generate_with_enforced_options"
        ]
        self.assertEqual(len(provider_calls), 1)
        self.assertIn("batch = parse_distillation_response", body)
        self.assertIn("no repair or fallback will run", body)
        self.assertNotIn("parse_distillation_response_resilient", body)
        self.assertNotIn("build_distillation_repair_prompt", body)
        self.assertNotIn("construction_repair", body)
        self.assertIn("if index_error", body)
        self.assertIn(
            "MR Memory graph committed but embedding refresh failed",
            body,
        )
        self.assertNotIn("logger.warning", body[body.index("if index_error") :])
        prompt_tries = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Try)
            and "build_distillation_prompt(" in ast.unparse(node)
        ]
        self.assertEqual(len(prompt_tries), 1)
        prompt_handler = "\n".join(
            ast.unparse(statement)
            for handler in prompt_tries[0].handlers
            for statement in handler.body
        )
        self.assertIn("stage': 'prompt_construction", prompt_handler)
        self.assertIn("status='failed'", prompt_handler)
        self.assertIn("raise", prompt_handler)

    def test_missing_feedback_provider_consumes_a_bounded_job_attempt(self) -> None:
        worker = ast.unparse(self._method("_maintenance_worker"))
        feedback = ast.unparse(self._method("_run_feedback_maintenance"))
        distill_schedule = ast.unparse(self._method("_ensure_distillation_deadline"))
        feedback_schedule = ast.unparse(self._method("_schedule_pending_feedback"))
        self.assertNotIn("kind == 'feedback' and self.context.get_provider_by_id", worker)
        self.assertIn("await service.claim_maintenance_job", worker)
        self.assertIn("await service.fail_maintenance_job", worker)
        self.assertNotIn("max_attempts", worker)
        self.assertNotIn("release_maintenance_job", worker)
        self.assertIn("cancelled maintenance did not enter terminal FAILED", worker)
        self.assertIn("budget_exhausted:", worker)
        self.assertNotIn("defer_maintenance_job_for_budget", worker)
        self.assertNotIn("resume_budget_wait", worker)
        failure_block = worker.split("except Exception as exc:", 1)[1]
        self.assertNotIn("_schedule_maintenance_wakeup", failure_block)
        self.assertNotIn("job state persisted", failure_block)
        self.assertIn("job_state=%s", failure_block)
        self.assertIn("'FAILED' if failure_persisted else 'UNKNOWN'", failure_block)
        self.assertNotIn("retry_failed=True", distill_schedule)
        self.assertNotIn("retry_failed=True", feedback_schedule)
        self.assertIn("feedback provider is unavailable", feedback)
        self.assertIn("raise RuntimeError", feedback)

    def test_feedback_failure_never_repairs_synthesizes_or_fails_open(self) -> None:
        method = self._method("_run_feedback_maintenance")
        body = ast.unparse(method)
        provider_calls = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_run_feedback_batch_with_ledger"
        ]
        self.assertEqual(len(provider_calls), 1)
        self.assertIn("plans = parse_gate", body)
        self.assertIn("gate_response_source = 'completion'", body)
        self.assertNotIn("parse_structured_response", body)
        self.assertNotIn("feedback_decision_graph_mutation", body)
        self.assertNotIn("fallback_mutation", body)
        self.assertNotIn("repairing once", body)
        mutation_calls = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "apply_graph_mutation"
        ]
        self.assertEqual(len(mutation_calls), 1)
        mutation_tries = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Try)
            and "apply_graph_mutation" in ast.unparse(node)
        ]
        self.assertEqual(
            len(mutation_tries),
            1,
            "graph mutation must not have an inner fail-open handler",
        )
        handler_body = "\n".join(
            ast.unparse(statement)
            for handler in mutation_tries[0].handlers
            for statement in handler.body
        )
        self.assertIn("status='failed'", handler_body)
        self.assertIn("raise", handler_body)
        self.assertIn("indexed_edge = await service.index_plastic_edge", body)
        self.assertIn("if not indexed_edge", body)
        self.assertIn("edge embedding document is unavailable", body)
        self.assertLess(
            body.index("await service.compact_feedback_memory"),
            body.index("status='completed'"),
        )

    def test_graph_tool_reports_missing_edge_embedding_as_an_error(self) -> None:
        body = ast.unparse(self._method("mr_graph_mutate"))
        self.assertIn("indexed_edge = await service.index_plastic_edge", body)
        self.assertIn("if not indexed_edge", body)
        self.assertIn("edge embedding document is unavailable", body)

    def test_after_send_observation_cannot_overwrite_failed_local_run(self) -> None:
        body = ast.unparse(self._method("trace_sent_artifacts"))
        status_guard = "local_outcome.operational_status == 'COMPLETED'"
        self.assertIn(status_guard, body)
        self.assertLess(body.index(status_guard), body.index("status='completed'"))
        self.assertIn("run_status == 'completed'", body)
        self.assertLess(
            body.index("run_status == 'completed'"),
            body.index("status='completed'"),
        )


if __name__ == "__main__":
    unittest.main()
