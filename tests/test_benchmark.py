from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.apply_human_review import apply_review
from scripts.apply_model_revisions import apply_revisions
from scripts.finalize_gold_benchmark import finalize_gold
from scripts.materialize_retrieval_benchmark import materialize


class RetrievalBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.corpus = [
            {
                "doc_id": "d1",
                "scope_id": "scope-a",
                "sent_at": 10,
                "speaker": "成员001",
                "text": "早期证据",
            },
            {
                "doc_id": "d2",
                "scope_id": "scope-a",
                "sent_at": 20,
                "speaker": "成员002",
                "text": "后续更新",
            },
            {
                "doc_id": "d3",
                "scope_id": "scope-b",
                "sent_at": 5,
                "speaker": "成员001",
                "text": "另一群消息",
            },
        ]
        self.candidates = [
            {
                "candidate_id": "c1",
                "scope_id": "scope-a",
                "query_doc_id": "d2",
                "query_time": 20,
                "hard_negatives": [{"doc_id": "d3"}],
            }
        ]

    def test_materializes_temporal_item_and_keeps_scope_boundary(self) -> None:
        annotations = [
            {
                "id": "q1",
                "candidate_id": "c1",
                "query": "事情后来如何变化？",
                "positive_doc_ids": ["d1", "d2"],
                "query_time_policy": "after_observed_query",
                "memory_type": "temporal_update",
                "epistemic": "corrected",
                "confidence": 1.0,
                "needs_review": False,
            }
        ]
        benchmark, manifest = materialize(self.corpus, self.candidates, annotations)
        self.assertEqual(benchmark[0]["query_time"], 21)
        self.assertEqual(benchmark[0]["lexical_decoy_doc_ids"], [])
        self.assertEqual(benchmark[0]["split"], "ai_high_confidence")
        self.assertFalse(benchmark[0]["provenance"]["human_reviewed"])
        self.assertTrue(manifest["evaluation_contract"]["scope_filter_required"])

    def test_rejects_positive_from_another_group(self) -> None:
        annotations = [
            {
                "id": "q1",
                "candidate_id": "c1",
                "query": "跨群证据不允许",
                "positive_doc_ids": ["d3"],
                "query_time_policy": "at_observed_query",
            }
        ]
        with self.assertRaisesRegex(ValueError, "crosses group scope"):
            materialize(self.corpus, self.candidates, annotations)

    def test_after_policy_requires_observed_message_as_evidence(self) -> None:
        annotations = [
            {
                "id": "q1",
                "candidate_id": "c1",
                "query": "更新后的状态是什么？",
                "positive_doc_ids": ["d1"],
                "query_time_policy": "after_observed_query",
            }
        ]
        with self.assertRaisesRegex(ValueError, "preventing query leakage"):
            materialize(self.corpus, self.candidates, annotations)

    def test_human_review_promotes_only_accepted_items_and_moves_cutoff(self) -> None:
        benchmark = [
            {
                "id": "q1",
                "scope_id": "scope-a",
                "query": "原问题是什么？",
                "query_time": 20,
                "positive_doc_ids": ["d1"],
                "lexical_decoy_doc_ids": [],
                "memory_type": "event",
                "epistemic": "asserted",
                "confidence": 0.9,
                "split": "ai_high_confidence",
                "provenance": {"candidate_id": "c1", "human_reviewed": False},
            }
        ]
        review = {
            "dataset_fingerprint": "test-fingerprint",
            "reviewer_type": "human",
            "exported_at": "2026-08-05T00:00:00Z",
            "reviews": [
                {
                    "id": "q1",
                    "decision": "accept",
                    "query": "补充证据后的问题是什么？",
                    "memory_type": "event",
                    "epistemic": "asserted",
                    "positive_doc_ids": ["d1", "d2"],
                    "notes": "补充后续证据",
                }
            ],
        }

        def fake_read(path):
            return {
                "benchmark.jsonl": benchmark,
                "corpus.jsonl": self.corpus,
                "candidates.jsonl": self.candidates,
            }[path.name]

        with patch(
            "scripts.apply_human_review.dataset_fingerprint",
            return_value="test-fingerprint",
        ), patch("scripts.apply_human_review.read_jsonl", side_effect=fake_read):
            reviewed, gold, summary = apply_review(directory=Path("."), review=review)
        self.assertEqual(len(gold), 1)
        self.assertEqual(reviewed[0]["split"], "gold")
        self.assertEqual(reviewed[0]["query_time"], 21)
        self.assertTrue(reviewed[0]["provenance"]["human_approved"])
        self.assertEqual(summary["query_time_adjusted_items"], ["q1"])

    def test_model_revision_stays_pending_until_second_human_review(self) -> None:
        reviewed = [
            {
                "id": "q1",
                "scope_id": "scope-a",
                "query": "旧问题",
                "query_time": 20,
                "positive_doc_ids": ["d1"],
                "memory_type": "event",
                "epistemic": "asserted",
                "split": "human_revision_required",
                "provenance": {"candidate_id": "c1", "human_decision": "edit"},
            }
        ]
        revisions = [
            {
                "id": "q1",
                "query": "按反馈修改后的问题",
                "positive_doc_ids": ["d1", "d2"],
                "memory_type": "temporal_update",
                "epistemic": "corrected",
                "rationale": "按人类意见修正范围",
            }
        ]
        output = apply_revisions(
            reviewed=reviewed,
            corpus=self.corpus,
            revisions=revisions,
        )
        self.assertEqual(output[0]["split"], "ai_revision_pending_human_review")
        self.assertFalse(output[0]["provenance"]["revision_human_reviewed"])
        self.assertEqual(output[0]["query_time"], 21)

    def test_explicit_round2_approval_promotes_revision_to_gold(self) -> None:
        base_gold = [
            {
                "id": "q1",
                "scope_id": "scope-a",
                "query": "已经由人类接受的问题",
                "query_time": 20,
                "positive_doc_ids": ["d1"],
                "memory_type": "event",
                "epistemic": "asserted",
                "split": "gold",
                "provenance": {"human_approved": True},
            }
        ]
        revisions = [
            {
                "id": "q2",
                "scope_id": "scope-a",
                "query": "按反馈修改并再次接受的问题",
                "query_time": 21,
                "positive_doc_ids": ["d2"],
                "memory_type": "temporal_update",
                "epistemic": "corrected",
                "split": "ai_revision_pending_human_review",
                "provenance": {
                    "human_approved": False,
                    "prior_human_decision": "edit",
                    "revision_human_reviewed": False,
                },
            }
        ]

        output, summary = finalize_gold(
            base_gold=base_gold,
            approved_revisions=revisions,
            corpus=self.corpus,
            approval_source="explicit_test_approval",
            approved_at="2026-08-05T00:00:00+00:00",
        )

        self.assertEqual([item["id"] for item in output], ["q1", "q2"])
        self.assertEqual(output[1]["split"], "gold")
        self.assertTrue(output[1]["provenance"]["human_approved"])
        self.assertTrue(output[1]["provenance"]["revision_human_reviewed"])
        self.assertEqual(summary["items"], 2)


if __name__ == "__main__":
    unittest.main()
