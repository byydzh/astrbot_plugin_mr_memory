from __future__ import annotations

import ast
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
        method = self._method("_runtime_request_kind")
        isolated_classifier = ast.FunctionDef(
            name="classify",
            args=method.args,
            body=method.body,
            decorator_list=[],
            returns=method.returns,
            type_comment=method.type_comment,
        )
        module = ast.fix_missing_locations(
            ast.Module(
                body=[isolated_activity, isolated_classifier],
                type_ignores=[],
            )
        )
        namespace: dict[str, object] = {}
        activity_only = ast.fix_missing_locations(
            ast.Module(body=[isolated_activity], type_ignores=[])
        )
        exec(compile(activity_only, "<runtime-activity-analysis>", "exec"), namespace)
        namespace["MrMemoryPlugin"] = type(
            "MrMemoryPlugin",
            (),
            {
                "_runtime_activity_analysis": staticmethod(
                    namespace["_runtime_activity_analysis"]
                )
            },
        )
        exec(compile(module, "<runtime-request-kind>", "exec"), namespace)
        return namespace["classify"]

    def test_host_return_decision_cannot_fall_through_to_provider(self) -> None:
        method = self._method("_run_layered_subconscious")
        body = ast.unparse(method)
        return_guard = body.index("decision.execution == 'RETURN'")
        provider_guard = body.index("if provider is None")
        start = body.index("self._runtime_singleflight.start")
        self.assertLess(return_guard, provider_guard)
        self.assertLess(return_guard, start)
        self.assertIn("semantic_status=decision.semantic_status", body)

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

    def test_explicit_person_lookup_is_a_synchronous_memory_query(self) -> None:
        """The exact user-facing lookup must not be routed as ordinary chat."""

        classify = self._runtime_request_classifier()
        self.assertTrue(callable(classify))
        self.assertEqual(classify("/chat 你好", force=False), "CHAT")
        request_kind = classify(
            "/chat 群里有d老师吗，请把他找出来",
            force=False,
        )
        self.assertEqual(request_kind, "MEMORY_QUERY")
        decision = RoutePolicy(l2_deadline_ms=1750).decide(
            RouteFeatures(request_kind=request_kind)
        )
        self.assertEqual(decision.execution, "SYNC")
        self.assertEqual(decision.deadline_ms, 1750)

    def test_recent_participant_activity_analysis_is_a_memory_query(self) -> None:
        classify = self._runtime_request_classifier()
        self.assertTrue(callable(classify))
        self.assertEqual(
            classify(
                "/chat 通过最近几天mllop的发言时间来预测它什么时候醒",
                force=False,
            ),
            "MEMORY_QUERY",
        )
        self.assertEqual(classify("/chat 最近天气什么时候好", force=False), "CHAT")
        self.assertEqual(classify("/chat 他发言好有趣", force=False), "CHAT")
        self.assertEqual(
            classify("/chat 最近原神新角色什么时候上线", force=False),
            "CHAT",
        )
        self.assertEqual(
            classify("/chat 最近这个消息什么时候发布", force=False),
            "CHAT",
        )

    def test_existing_feedback_trace_does_not_bypass_memory_reconstruction(
        self,
    ) -> None:
        """Repeated host hooks may reuse a trace, but still need memory evidence."""

        method = self._method("inject_subconscious_memory")
        trace_guard = next(
            node
            for node in method.body
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "self.feedback_learning_enabled"
            and "self._active_interaction_traces.get(id(event))" in ast.unparse(node)
        )
        self.assertFalse(
            any(isinstance(node, ast.Return) for node in ast.walk(trace_guard)),
            "reusing a feedback trace must not skip _run_subconscious",
        )
        self.assertIn(
            "await self._run_subconscious(event, query)",
            ast.unparse(method),
        )

    def test_subconscious_failure_is_logged_but_never_injected(self) -> None:
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
        self.assertIn("_record_subconscious_surface_failure", guard_body)
        self.assertNotIn("_stop_failed_memory_query", guard_body)
        self.assertIn("outcome.operational_status", guard_body)
        self.assertIn("outcome.detail", guard_body)

        # Provider exceptions happen before ``outcome`` exists.  They become a
        # ledger-only outcome and flow through the same no-injection guard.
        run_try = next(
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Try)
            and "await self._run_subconscious(event, query)"
            in "\n".join(ast.unparse(item) for item in node.body)
        )
        self.assertGreaterEqual(len(run_try.handlers), 2)
        for handler in run_try.handlers:
            with self.subTest(handler=ast.unparse(handler.type)):
                handler_body = "\n".join(
                    ast.unparse(node) for node in handler.body
                )
                assigns_outcome = any(
                    isinstance(node, (ast.Assign, ast.AnnAssign))
                    and any(
                        isinstance(target, ast.Name) and target.id == "outcome"
                        for target in (
                            node.targets
                            if isinstance(node, ast.Assign)
                            else [node.target]
                        )
                    )
                    for node in handler.body
                )
                self.assertTrue(
                    assigns_outcome,
                    "provider failures must become a ledger-only outcome",
                )
                self.assertNotIn("req.extra_user_content_parts.append", handler_body)

        # Prospective hypotheses are staged and appended only after a usable,
        # JSON-valid evidence packet, so a failed wake injects no memory at all.
        json_parse = method_body.index("evidence_value = json.loads")
        prospective_append = method_body.index(
            "req.extra_user_content_parts.append(prospective_part)",
            json_parse,
        )
        self.assertGreater(prospective_append, json_parse)

        feedback_try = next(
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Try)
            and "await self._begin_interaction_trace"
            in "\n".join(ast.unparse(item) for item in node.body)
        )
        feedback_failure = "\n".join(
            ast.unparse(node) for node in feedback_try.handlers[0].body
        )
        self.assertIn("logger.exception", feedback_failure)
        self.assertIn("return", feedback_failure)
        self.assertNotIn("prospective = []", feedback_failure)
        self.assertNotIn("failed open", feedback_failure.casefold())

        parse_try = next(
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Try)
            and "evidence_value = json.loads(outcome.surface_text)"
            in "\n".join(ast.unparse(item) for item in node.body)
        )
        parse_failure_body = "\n".join(
            ast.unparse(node) for node in parse_try.handlers[0].body
        )
        self.assertIn("logger.error", parse_failure_body)
        self.assertIn("_record_subconscious_surface_failure", parse_failure_body)
        self.assertNotIn("_stop_failed_memory_query", parse_failure_body)
        self.assertNotIn(
            "req.extra_user_content_parts.append",
            parse_failure_body,
        )

    def test_completed_empty_semantic_result_is_not_rewritten_as_failure(self) -> None:
        method = self._method("inject_subconscious_memory")
        completed_guard = next(
            node
            for node in method.body
            if isinstance(node, ast.If)
            and "outcome.operational_status == 'COMPLETED'" in ast.unparse(node.test)
            and "SEMANTIC_NONE" in ast.unparse(node.test)
            and "REQUEST_L3" in ast.unparse(node.test)
        )
        guard_body = "\n".join(ast.unparse(node) for node in completed_guard.body)
        self.assertIn("req.extra_user_content_parts.append(prospective_part)", guard_body)
        self.assertIn("return", guard_body)
        self.assertNotIn("_record_subconscious_surface_failure", guard_body)

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


if __name__ == "__main__":
    unittest.main()
