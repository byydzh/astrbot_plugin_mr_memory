from __future__ import annotations

import unittest
import uuid
import time
from pathlib import Path

from mr_memory.feedback import FeedbackDecision
from mr_memory.models import NormalizedMessage
from mr_memory.plasticity import parse_graph_mutation
from mr_memory.storage import MemoryStorage


class PlasticGraphTests(unittest.TestCase):
    umo = "shadow:GroupMessage:group-a"

    def setUp(self) -> None:
        test_root = Path.cwd() / ".dev" / "test-tmp"
        test_root.mkdir(parents=True, exist_ok=True)
        self.database_path = test_root / f"{uuid.uuid4().hex}.db"
        self.storage = MemoryStorage(self.database_path)
        self.storage.bind_scope(
            umo=self.umo,
            platform_id="shadow",
            group_id="group-a",
        )

    def tearDown(self) -> None:
        self.storage.close()
        for suffix in ("", "-wal", "-shm"):
            Path(f"{self.database_path}{suffix}").unlink(missing_ok=True)

    def message(
        self,
        message_id: str,
        text: str,
        *,
        sent_at: int,
        sender_id: str = "user-a",
    ) -> NormalizedMessage:
        return NormalizedMessage(
            platform="aiocqhttp",
            platform_id="shadow",
            umo=self.umo,
            group_id="group-a",
            message_id=message_id,
            sender_id=sender_id,
            sender_name=sender_id,
            sent_at=sent_at,
            plain_text=text,
            content=[{"type": "plain", "text": text}],
        )

    @staticmethod
    def edge_payload(source_key: str) -> dict[str, object]:
        return {
            "operation": "upsert_edge",
            "evidence_source_keys": [source_key],
            "confidence": 0.87,
            "utility_delta": 0.6,
            "statement": "‘好女孩’在本群常用作戏谑性的认可标签。",
            "source": {
                "kind": "symbol",
                "label": "好女孩",
                "description": "群内反复出现的表达",
            },
            "relation": {
                "key": "group_usage",
                "name": "群内用法",
                "description": "一个表达在当前群聊中的语用含义",
                "source_kinds": ["symbol"],
                "target_kinds": ["concept"],
            },
            "target": {
                "kind": "concept",
                "label": "戏谑性的认可",
                "description": "不是性别或品德事实判断",
            },
        }

    def test_parser_forbids_identity_nodes(self) -> None:
        payload = self.edge_payload("source-1")
        payload["source"] = {"kind": "participant", "label": "某人"}
        with self.assertRaisesRegex(ValueError, "unsupported plastic node kind"):
            parse_graph_mutation(payload)

    def test_evidence_bound_edge_and_relation_revision(self) -> None:
        evidence = self.message("e-1", "鉴定为好女孩", sent_at=100)
        self.storage.upsert_message(evidence)
        mutation = parse_graph_mutation(
            self.edge_payload(evidence.resolved_source_key())
        )
        committed = self.storage.apply_graph_mutation(
            umo=self.umo,
            mutation=mutation,
            model="test-model",
            allowed_evidence_keys={evidence.resolved_source_key()},
        )
        self.assertEqual(committed["status"], "COMMITTED")
        self.assertEqual(committed["relation_version"], 1)
        rows = self.storage.query_plastic_associations(
            umo=self.umo,
            query="好女孩",
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["relation_key"], "group_usage")
        self.assertEqual(rows[0]["source_keys"], [evidence.resolved_source_key()])

        revision = parse_graph_mutation(
            {
                "operation": "revise_relation",
                "evidence_source_keys": [evidence.resolved_source_key()],
                "confidence": 0.9,
                "utility_delta": 0,
                "relation": {
                    "key": "group_usage",
                    "name": "本群语用",
                    "description": "表达在群聊语境中的动态、非字面含义",
                    "source_kinds": ["symbol"],
                    "target_kinds": ["concept"],
                },
            }
        )
        result = self.storage.apply_graph_mutation(
            umo=self.umo,
            mutation=revision,
            allowed_evidence_keys={evidence.resolved_source_key()},
        )
        self.assertEqual(result["version"], 2)
        self.assertEqual(
            self.storage.query_plastic_associations(umo=self.umo, query="好女孩")[0][
                "relation_version"
            ],
            2,
        )

    def test_upsert_reuses_active_relation_definition_until_explicit_revision(self) -> None:
        first = self.message("e-1", "鉴定为好女孩", sent_at=100)
        second = self.message("e-2", "这里也是群内反话", sent_at=110)
        self.storage.upsert_message(first)
        self.storage.upsert_message(second)
        self.storage.apply_graph_mutation(
            umo=self.umo,
            mutation=parse_graph_mutation(
                self.edge_payload(first.resolved_source_key())
            ),
        )

        repeated = self.edge_payload(second.resolved_source_key())
        repeated["relation"] = {
            "key": "group_usage",
            "name": "群聊特殊用法",
            "description": "模型对同一稳定关系键生成了不同措辞",
            "source_kinds": ["symbol"],
            "target_kinds": ["concept"],
        }
        repeated["target"] = {
            "kind": "concept",
            "label": "反话",
            "description": "局部语境中的非字面表达",
        }
        result = self.storage.apply_graph_mutation(
            umo=self.umo,
            mutation=parse_graph_mutation(repeated),
        )
        self.assertEqual(result["relation_version"], 1)
        self.assertTrue(result["relation_definition_reused"])
        rows = self.storage._connection.execute(
            """
            SELECT version, canonical_name FROM relation_types
            WHERE umo=? AND relation_key='group_usage'
            """,
            (self.umo,),
        ).fetchall()
        self.assertEqual(
            [(int(row["version"]), str(row["canonical_name"])) for row in rows],
            [(1, "群内用法")],
        )

    def test_competing_meanings_keep_doubt_until_evidence_revision(self) -> None:
        first = self.message("e-1", "鉴定为好女孩", sent_at=100)
        second = self.message("e-2", "这里的好女孩是不是反话？", sent_at=110)
        correction = self.message(
            "e-3", "对，好女孩在这里就是拿臭婊子开玩笑", sent_at=120
        )
        for message in (first, second, correction):
            self.storage.upsert_message(message)

        base = self.edge_payload(first.resolved_source_key())
        base.update(
            {
                "epistemic_state": "HYPOTHESIS",
                "uncertainty": "可能是反话，也可能只是戏谑性的夸奖。",
            }
        )
        praise = self.storage.apply_graph_mutation(
            umo=self.umo,
            mutation=parse_graph_mutation(base),
        )
        euphemism = dict(base)
        euphemism.update(
            {
                "evidence_source_keys": [second.resolved_source_key()],
                "statement": "‘好女孩’可能是对‘臭婊子’的群内委婉反称。",
                "relation": {
                    "key": "possible_euphemism_for",
                    "name": "可能委婉指代",
                    "description": "群聊表达可能委婉或反向指代另一表达",
                    "source_kinds": ["symbol"],
                    "target_kinds": ["concept"],
                },
                "target": {
                    "kind": "concept",
                    "label": "臭婊子",
                    "description": "可能被重新引义的冒犯性原词",
                },
            }
        )
        candidate = self.storage.apply_graph_mutation(
            umo=self.umo,
            mutation=parse_graph_mutation(euphemism),
        )
        rows = self.storage.query_plastic_associations(umo=self.umo, query="好女孩")
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["epistemic_state"] for row in rows}, {"HYPOTHESIS"})
        self.assertTrue(all(row["uncertainty"] for row in rows))

        revised = parse_graph_mutation(
            {
                "operation": "revise_edge",
                "edge_id": candidate["target_id"],
                "evidence_source_keys": [correction.resolved_source_key()],
                "confidence": 0.94,
                "utility_delta": 0.2,
                "statement": "‘好女孩’在该段群聊中是对‘臭婊子’的玩笑式反称。",
                "epistemic_state": "CONFIRMED",
                "uncertainty": "",
            }
        )
        result = self.storage.apply_graph_mutation(
            umo=self.umo,
            mutation=revised,
        )
        self.assertEqual(result["epistemic_state"], "CONFIRMED")
        refreshed = {
            row["id"]: row
            for row in self.storage.query_plastic_associations(
                umo=self.umo, query="好女孩"
            )
        }
        self.assertEqual(
            refreshed[int(candidate["target_id"])]["epistemic_state"],
            "CONFIRMED",
        )
        self.assertEqual(
            refreshed[int(praise["target_id"])]["epistemic_state"],
            "HYPOTHESIS",
        )
        repeated = dict(euphemism)
        repeated["evidence_source_keys"] = [correction.resolved_source_key()]
        repeated["confidence"] = 0.4
        repeated["uncertainty"] = "一次较弱的新推测不应覆盖显式修订。"
        self.storage.apply_graph_mutation(
            umo=self.umo,
            mutation=parse_graph_mutation(repeated),
        )
        still_confirmed = {
            row["id"]: row
            for row in self.storage.query_plastic_associations(
                umo=self.umo, query="好女孩"
            )
        }
        self.assertEqual(
            still_confirmed[int(candidate["target_id"])]["epistemic_state"],
            "CONFIRMED",
        )
        self.assertEqual(
            still_confirmed[int(candidate["target_id"])]["uncertainty"], ""
        )

    def test_cross_group_or_uninspected_evidence_is_rejected(self) -> None:
        evidence = self.message("e-1", "好女孩", sent_at=100)
        self.storage.upsert_message(evidence)
        mutation = parse_graph_mutation(
            self.edge_payload(evidence.resolved_source_key())
        )
        with self.assertRaisesRegex(ValueError, "outside the inspected set"):
            self.storage.apply_graph_mutation(
                umo=self.umo,
                mutation=mutation,
                allowed_evidence_keys={"not-inspected"},
            )

    def test_uncommitted_feedback_cannot_mutate_plastic_graph(self) -> None:
        evidence = self.message("e-1", "鉴定为好女孩", sent_at=90)
        request = self.message("q-1", "这是好女孩吗", sent_at=100)
        feedback = self.message("f-1", "不对，不是这个意思", sent_at=110)
        self.storage.upsert_message(evidence)
        self.storage.upsert_message(request)
        self.storage.start_interaction_trace(
            trace_id="trace-uncommitted",
            umo=self.umo,
            sender_id="user-a",
            request_source_key=request.resolved_source_key(),
            request_sent_at=100,
            query=request.plain_text,
        )
        self.storage.finish_interaction_trace(
            trace_id="trace-uncommitted",
            umo=self.umo,
            response_text="这是字面上的夸奖。",
            response_at=101,
        )
        self.storage.upsert_message(feedback)
        proposal_id = self.storage.enqueue_feedback_candidate(
            umo=self.umo,
            feedback_source_key=feedback.resolved_source_key(),
        )
        self.assertIsNotNone(proposal_id)
        with self.assertRaisesRegex(ValueError, "host-committed"):
            self.storage.apply_graph_mutation(
                umo=self.umo,
                mutation=parse_graph_mutation(
                    self.edge_payload(evidence.resolved_source_key())
                ),
                allowed_evidence_keys={evidence.resolved_source_key()},
                feedback_proposal_id=int(proposal_id),
            )
        self.assertEqual(
            self.storage.query_plastic_associations(
                umo=self.umo,
                query="好女孩",
            ),
            [],
        )

    def test_serialized_state_is_bounded_and_versioned(self) -> None:
        digest = "a" * 64
        first = self.storage.update_subconscious_state(
            umo=self.umo,
            state={"focus": ["好女孩"], "active_edge_ids": [1]},
            last_query_sha256=digest,
            at=100,
        )
        second = self.storage.update_subconscious_state(
            umo=self.umo,
            state={"last_decision": "NO_RELEVANT_MEMORY"},
            last_query_sha256=digest,
            at=200,
        )
        self.assertEqual(first["revision"], 1)
        self.assertEqual(second["revision"], 2)
        with self.assertRaisesRegex(ValueError, "unsupported fields"):
            self.storage.update_subconscious_state(
                umo=self.umo,
                state={"chain_of_thought": "do not persist this"},
                last_query_sha256=digest,
            )

    def test_running_maintenance_job_recovers_after_reopen(self) -> None:
        job_id = self.storage.enqueue_maintenance_job(
            umo=self.umo,
            job_type="plasticity",
            dedupe_key="feedback:e-1",
            payload={"source_key": "e-1"},
            available_at=100,
        )
        claimed = self.storage.claim_maintenance_job(
            umo=self.umo,
            job_id=job_id,
            now=100,
        )
        self.assertIsNotNone(claimed)
        self.storage.close()
        self.storage = MemoryStorage(self.database_path)
        pending = self.storage.pending_maintenance_jobs(
            umo=self.umo,
            now=101,
        )
        self.assertEqual([row["id"] for row in pending], [job_id])

    def test_unexpired_maintenance_lease_survives_an_observer_reopen(self) -> None:
        now = int(time.time())
        job_id = self.storage.enqueue_maintenance_job(
            umo=self.umo,
            job_type="feedback",
            dedupe_key="feedback:live-lease",
            available_at=now,
        )
        claimed = self.storage.claim_maintenance_job(
            umo=self.umo,
            job_id=job_id,
            now=now,
            lease_seconds=3600,
        )
        self.assertIsNotNone(claimed)
        self.storage.close()
        self.storage = MemoryStorage(self.database_path)
        self.assertEqual(
            self.storage.pending_maintenance_jobs(umo=self.umo, now=now + 1),
            [],
        )
        self.storage.finish_maintenance_job(umo=self.umo, job_id=job_id)

    def test_cancelled_worker_releases_maintenance_lease_without_spending_attempt(self) -> None:
        job_id = self.storage.enqueue_maintenance_job(
            umo=self.umo,
            job_type="feedback",
            dedupe_key="feedback:cancelled-worker",
            available_at=100,
        )
        claimed = self.storage.claim_maintenance_job(
            umo=self.umo,
            job_id=job_id,
            now=100,
            lease_seconds=3600,
        )
        self.assertEqual(claimed["attempts"], 1)

        self.assertTrue(
            self.storage.release_maintenance_job(
                umo=self.umo,
                job_id=job_id,
                now=101,
            )
        )
        pending = self.storage.pending_maintenance_jobs(umo=self.umo, now=101)
        self.assertEqual([row["id"] for row in pending], [job_id])
        self.assertEqual(pending[0]["attempts"], 0)

    def test_terminal_maintenance_job_only_retries_when_explicit(self) -> None:
        job_id = self.storage.enqueue_maintenance_job(
            umo=self.umo,
            job_type="distill",
            dedupe_key="distill:pending",
            available_at=100,
        )
        self.assertIsNotNone(
            self.storage.claim_maintenance_job(
                umo=self.umo,
                job_id=job_id,
                now=100,
            )
        )
        self.assertEqual(
            self.storage.fail_maintenance_job(
                umo=self.umo,
                job_id=job_id,
                error="old implementation failed",
                now=101,
                max_attempts=1,
            ),
            "FAILED",
        )
        self.storage.enqueue_maintenance_job(
            umo=self.umo,
            job_type="distill",
            dedupe_key="distill:pending",
            available_at=102,
        )
        self.assertEqual(
            self.storage.pending_maintenance_jobs(umo=self.umo, now=102),
            [],
        )
        retried_id = self.storage.enqueue_maintenance_job(
            umo=self.umo,
            job_type="distill",
            dedupe_key="distill:pending",
            available_at=103,
            retry_failed=True,
        )
        self.assertEqual(retried_id, job_id)
        pending = self.storage.pending_maintenance_jobs(umo=self.umo, now=103)
        self.assertEqual([row["id"] for row in pending], [job_id])
        self.assertEqual(pending[0]["attempts"], 0)

    def test_human_feedback_assigns_credit_to_activated_plastic_path(self) -> None:
        request = self.message("q-1", "这是好女孩吗", sent_at=100)
        evidence = self.message("e-1", "鉴定为好女孩", sent_at=90)
        self.storage.upsert_message(evidence)
        self.storage.upsert_message(request)
        edge = self.storage.apply_graph_mutation(
            umo=self.umo,
            mutation=parse_graph_mutation(
                self.edge_payload(evidence.resolved_source_key())
            ),
        )
        trace_id = "trace-plastic"
        self.storage.start_interaction_trace(
            trace_id=trace_id,
            umo=self.umo,
            sender_id="user-a",
            request_source_key=request.resolved_source_key(),
            request_sent_at=100,
            query=request.plain_text,
        )
        self.storage.activate_plastic_edges(
            umo=self.umo,
            edge_ids=[int(edge["target_id"])],
            at=100,
            trace_id=trace_id,
            relevance=0.9,
        )
        self.storage.finish_interaction_trace(
            trace_id=trace_id,
            umo=self.umo,
            response_text="这是群里的认可梗。",
            response_at=101,
        )
        feedback = self.message("f-1", "对，不错，就是这个意思", sent_at=110)
        self.storage.upsert_message(feedback)
        proposal_id = self.storage.enqueue_feedback_candidate(
            umo=self.umo,
            feedback_source_key=feedback.resolved_source_key(),
        )
        self.assertIsNotNone(proposal_id)
        result = self.storage.apply_feedback_decision(
            umo=self.umo,
            proposal_id=int(proposal_id),
            decision=FeedbackDecision(
                target_trace_id=trace_id,
                mutation="upsert",
                feedback_valence=1.0,
                confidence=0.9,
                scope_type="sender",
                scope_key="user-a",
                aspect="group_slang",
                statement="正确理解了好女孩的群内含义",
                prospective_cue="遇到好女孩时优先按群内认可梗理解",
                trigger_cues=("好女孩",),
                activation_mode="semantic",
            ),
        )
        self.assertEqual(result["plastic_edges_credited"], 1)
        self.assertGreater(result["plastic_backward_credit"], 0)
        refreshed = self.storage.query_plastic_associations(
            umo=self.umo, query="好女孩"
        )[0]
        self.assertGreater(float(refreshed["utility"]), 0.6)
        roles = {item["evidence_role"] for item in refreshed["evidence"]}
        self.assertIn("FEEDBACK_POSITIVE", roles)

    def test_negative_feedback_cannot_blame_an_unactivated_path(self) -> None:
        evidence = self.message("e-1", "鉴定为好女孩", sent_at=90)
        request = self.message("q-1", "这是好女孩吗", sent_at=100)
        self.storage.upsert_message(evidence)
        self.storage.upsert_message(request)
        first = self.storage.apply_graph_mutation(
            umo=self.umo,
            mutation=parse_graph_mutation(
                self.edge_payload(evidence.resolved_source_key())
            ),
        )
        other_payload = self.edge_payload(evidence.resolved_source_key())
        other_payload["source"] = {
            "kind": "symbol",
            "label": "好孩子",
            "description": "另一个未参与回答的群内表达",
        }
        second = self.storage.apply_graph_mutation(
            umo=self.umo,
            mutation=parse_graph_mutation(other_payload),
        )
        trace_id = "trace-negative-gate"
        self.storage.start_interaction_trace(
            trace_id=trace_id,
            umo=self.umo,
            sender_id="user-a",
            request_source_key=request.resolved_source_key(),
            request_sent_at=100,
            query=request.plain_text,
        )
        self.storage.activate_plastic_edges(
            umo=self.umo,
            edge_ids=[int(first["target_id"])],
            at=100,
            trace_id=trace_id,
            relevance=0.9,
        )
        self.storage.finish_interaction_trace(
            trace_id=trace_id,
            umo=self.umo,
            response_text="误解了这个词。",
            response_at=101,
        )
        feedback = self.message("f-1", "不对，不是这个意思", sent_at=110)
        self.storage.upsert_message(feedback)
        proposal_id = self.storage.enqueue_feedback_candidate(
            umo=self.umo,
            feedback_source_key=feedback.resolved_source_key(),
        )
        inspected = self.storage.inspect_feedback_proposal(
            umo=self.umo,
            proposal_id=int(proposal_id),
        )
        self.assertEqual(
            [row["edge_id"] for row in inspected["activated_plastic_edges"]],
            [first["target_id"]],
        )
        unactivated_inhibition = parse_graph_mutation(
            {
                "operation": "inhibit_edge",
                "edge_id": second["target_id"],
                "evidence_source_keys": [feedback.resolved_source_key()],
                "confidence": 0.9,
                "utility_delta": -0.5,
                "statement": "negative feedback",
            }
        )
        with self.assertRaisesRegex(ValueError, "actually influenced"):
            self.storage.apply_graph_mutation(
                umo=self.umo,
                mutation=unactivated_inhibition,
                allowed_evidence_keys={feedback.resolved_source_key()},
                allowed_negative_edge_ids={int(first["target_id"])},
            )
        activated_inhibition = parse_graph_mutation(
            {
                **unactivated_inhibition.as_dict(),
                "edge_id": first["target_id"],
            }
        )
        result = self.storage.apply_graph_mutation(
            umo=self.umo,
            mutation=activated_inhibition,
            allowed_evidence_keys={feedback.resolved_source_key()},
            allowed_negative_edge_ids={int(first["target_id"])},
        )
        self.assertEqual(result["status"], "COMMITTED")


if __name__ == "__main__":
    unittest.main()
