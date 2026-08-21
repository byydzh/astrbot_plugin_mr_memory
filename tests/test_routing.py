from __future__ import annotations

import unittest

from mr_memory.routing import RouteFeatures, RoutePolicy


class RoutePolicyTests(unittest.TestCase):
    def test_deterministic_identity_path_never_calls_provider(self) -> None:
        decision = RoutePolicy().decide(
            RouteFeatures(identity_only=True, operational_status="FAILED")
        )
        self.assertEqual((decision.level, decision.action), ("L0", "RETURN_DETERMINISTIC"))
        self.assertFalse(decision.allow_provider_call)
        self.assertEqual(decision.semantic_status, "CERTIFIED")

    def test_valid_l1b_certificate_preserves_semantic_none(self) -> None:
        decision = RoutePolicy().decide(
            RouteFeatures(
                l1b_cache_state="HIT",
                l1b_semantic_status="SEMANTIC_NONE",
            )
        )
        self.assertEqual(decision.action, "RETURN_CERTIFICATE")
        self.assertEqual(decision.cache_layer, "L1B")
        self.assertEqual(decision.semantic_status, "SEMANTIC_NONE")
        self.assertEqual(decision.operational_status, "READY")

    def test_operational_failures_are_never_semantic_none(self) -> None:
        for status, action in (
            ("BUDGET_BLOCKED", "BLOCK_BUDGET"),
            ("PROVIDER_UNAVAILABLE", "BLOCK_PROVIDER"),
            ("FAILED", "REPORT_FAILURE"),
        ):
            with self.subTest(status=status):
                decision = RoutePolicy().decide(
                    RouteFeatures(operational_status=status)
                )
                self.assertEqual(decision.action, action)
                self.assertEqual(decision.execution, "BLOCKED")
                self.assertEqual(decision.semantic_status, "UNKNOWN")
                self.assertEqual(decision.operational_status, status)

    def test_balanced_chat_and_memory_query_are_bounded_sync(self) -> None:
        policy = RoutePolicy(mode="BALANCED", l2_deadline_ms=1750)
        chat = policy.decide(RouteFeatures(request_kind="CHAT"))
        query = policy.decide(RouteFeatures(request_kind="MEMORY_QUERY"))
        low_latency = RoutePolicy(mode="LOW_LATENCY", l2_deadline_ms=1750).decide(
            RouteFeatures(request_kind="CHAT")
        )
        self.assertEqual((chat.level, chat.execution, chat.deadline_ms), ("L2", "SYNC", 1750))
        self.assertEqual((query.level, query.execution, query.deadline_ms), ("L2", "SYNC", 1750))
        self.assertEqual(
            (low_latency.level, low_latency.execution, low_latency.deadline_ms),
            ("L2", "ASYNC", 0),
        )

    def test_singleflight_joins_instead_of_starting_duplicate_work(self) -> None:
        decision = RoutePolicy().decide(
            RouteFeatures(
                request_kind="MEMORY_QUERY",
                l1a_cache_state="HIT",
                singleflight_running=True,
                operational_status="RUNNING",
            )
        )
        self.assertEqual(decision.action, "JOIN_L2")
        self.assertTrue(decision.join_singleflight)
        self.assertFalse(decision.allow_provider_call)
        self.assertEqual(decision.cache_layer, "L1A")

    def test_reader_request_l3_is_advisory_to_host_policy(self) -> None:
        allowed = RoutePolicy(allow_l3=True).decide(
            RouteFeatures(
                request_kind="MEMORY_QUERY",
                live_semantic_status="REQUEST_L3",
            )
        )
        denied = RoutePolicy(allow_l3=False).decide(
            RouteFeatures(
                request_kind="MEMORY_QUERY",
                live_semantic_status="REQUEST_L3",
            )
        )
        self.assertEqual((allowed.level, allowed.action), ("L3", "START_L3"))
        self.assertIn("host policy approved", " ".join(allowed.reasons))
        self.assertEqual(denied.action, "RETURN_SAFETY_ABSTAIN")
        self.assertEqual(denied.semantic_status, "SAFETY_ABSTAIN")

    def test_identity_ambiguity_abstains_unless_deep_route_is_authorized(self) -> None:
        shallow = RoutePolicy().decide(RouteFeatures(identity_ambiguous=True))
        deep = RoutePolicy().decide(
            RouteFeatures(identity_ambiguous=True, explicit_deep=True)
        )
        self.assertEqual(shallow.action, "RETURN_SAFETY_ABSTAIN")
        self.assertEqual(deep.level, "L3")

    def test_complex_revision_case_selects_l3(self) -> None:
        decision = RoutePolicy(mode="RESEARCH").decide(
            RouteFeatures(
                request_kind="CHAT",
                multi_event=True,
                conflicting_evidence=True,
                revision_question=True,
            )
        )
        self.assertEqual((decision.level, decision.execution), ("L3", "SYNC"))
        self.assertIn("belief revision question", decision.reasons)


if __name__ == "__main__":
    unittest.main()
