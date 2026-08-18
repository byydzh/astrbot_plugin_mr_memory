from __future__ import annotations

import ast
import unittest
from pathlib import Path


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

    def test_sync_l2_wait_includes_bounded_l3_escalation(self) -> None:
        method = self._method("_run_layered_subconscious")
        body = ast.unparse(method)
        self.assertIn("if policy.allow_l3", body)
        self.assertIn("timeout_seconds += self.runtime_l3_deadline_seconds", body)

        producer = self._method("_execute_layered_reconstruction_started")
        producer_body = ast.unparse(producer)
        self.assertIn(
            "self.subconscious_timeout_seconds + (self.runtime_l3_deadline_seconds "
            "if policy.allow_l3 else 0)",
            producer_body,
        )

    def test_reader_revision_includes_actual_provider_model(self) -> None:
        method = self._method("_runtime_inference_revision")
        body = ast.unparse(method)
        self.assertIn("_provider_model_name(provider)", body)
        self.assertIn("self._last_reader_model_revision", body)
        self.assertIn("'reader_model': reader_model_revision", body)

    def test_certificate_store_rechecks_snapshot_after_source_audit(self) -> None:
        method = self._method("_store_layered_certificate")
        body = ast.unparse(method)
        degraded_guard = body.index(
            "certificate.stop_reason == 'PROTOCOL_DEGRADED'"
        )
        self.assertGreaterEqual(body.count("self._assert_snapshot_fresh"), 2)
        audit = body.index("await service.audit_snapshot_sources")
        last_fresh = body.rindex("await self._assert_snapshot_fresh")
        put = body.index("await service.put_memory_certificate")
        self.assertLess(degraded_guard, audit)
        self.assertLess(audit, last_fresh)
        self.assertLess(last_fresh, put)

    def test_l3_trace_serializes_protocol_degradation_audit(self) -> None:
        method = self._method("_run_l3_certificate")
        body = ast.unparse(method)
        self.assertIn("'repair_attempted': result.repair_attempted", body)
        self.assertIn("'degraded': result.degraded", body)
        self.assertIn("item.as_dict() for item in result.protocol_failures", body)

    def test_reply_target_is_snapshot_bounded_packet_evidence(self) -> None:
        revision = ast.unparse(self._method("_runtime_inference_revision"))
        self.assertIn("host-prefetch.snapshot.v3", revision)

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
