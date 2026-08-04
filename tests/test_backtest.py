from __future__ import annotations

import unittest
from dataclasses import replace

from mr_memory.backtest import build_reverse_replay_windows, direct_evidence_gate
from mr_memory.models import NormalizedMessage
from mr_memory.usage import TokenUsageRecord


class MaskedBacktestTests(unittest.TestCase):
    umo = "shadow:GroupMessage:group-a"

    def message(self, index: int, sent_at: int) -> NormalizedMessage:
        return NormalizedMessage(
            platform="aiocqhttp",
            platform_id="shadow",
            umo=self.umo,
            group_id="group-a",
            message_id=str(index),
            sender_id="user-a",
            sender_name="甲",
            sent_at=sent_at,
            plain_text=f"message-{index}",
        )

    def test_reverse_windows_exclude_cutoff_and_keep_batch_order(self) -> None:
        messages = [self.message(index, index * 10) for index in range(1, 11)]
        windows = build_reverse_replay_windows(
            reversed(messages),
            umo=self.umo,
            cutoff_at=91,
            batch_size=3,
        )
        self.assertEqual(
            [[item.sent_at for item in window.messages] for window in windows],
            [[70, 80, 90], [40, 50, 60], [10, 20, 30]],
        )
        self.assertTrue(
            all(item.sent_at < 91 for window in windows for item in window.messages)
        )

    def test_reverse_windows_reject_cross_group_records(self) -> None:
        foreign = self.message(1, 10)
        foreign = replace(foreign, umo="shadow:GroupMessage:group-b")
        with self.assertRaisesRegex(ValueError, "cross group scopes"):
            build_reverse_replay_windows(
                [self.message(2, 20), foreign],
                umo=self.umo,
                cutoff_at=30,
            )

    def test_usage_normalizes_provider_shapes(self) -> None:
        usage = TokenUsageRecord.from_value(
            {
                "prompt_tokens": 100,
                "cached_tokens": 40,
                "completion_tokens": 25,
            }
        )
        self.assertEqual(usage.input_other, 60)
        self.assertEqual(usage.input_cached, 40)
        self.assertEqual(usage.output, 25)
        self.assertEqual(usage.total, 125)

    def test_direct_evidence_gate_requires_candidate_and_source_overlap(self) -> None:
        decision = direct_evidence_gate(
            query="上次说讨厌太空采矿的成员是谁",
            tool_name="query_event_context",
            arguments={"event_id": 55},
            result=[
                {
                    "source_key": "source-1",
                    "sender_name": "测试用户",
                    "plain_text": "太空采矿玩吐了",
                }
            ],
            initial_candidates={"episodes": [{"id": 55, "score": 0.51}]},
        )
        self.assertTrue(decision.sufficient)
        self.assertIn("采矿", decision.matched_terms)
        unrelated = direct_evidence_gate(
            query="上次说讨厌太空采矿的成员是谁",
            tool_name="query_event_context",
            arguments={"event_id": 99},
            result=[{"source_key": "source-2", "plain_text": "今天天气不错"}],
            initial_candidates={"episodes": [{"id": 55, "score": 0.51}]},
        )
        self.assertFalse(unrelated.sufficient)


if __name__ == "__main__":
    unittest.main()
