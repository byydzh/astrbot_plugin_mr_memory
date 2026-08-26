from __future__ import annotations

import contextlib
import io
import json
import shutil
import unittest
import uuid
from pathlib import Path

from scripts.local_serving_acceptance import (
    CASE_KEYS,
    EXPECTED_CASE_IDS,
    load_private_case,
    main,
    run_suite,
)


class LocalServingAcceptanceTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary_root = Path.cwd() / ".dev" / "test-tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.root = temporary_root / f"local-serving-{uuid.uuid4().hex}"
        self.root.mkdir()
        self.suite = self.root / "suite"
        self._build_suite()

    def tearDown(self) -> None:
        shutil.rmtree(self.root)

    @staticmethod
    def _write(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _case_shell(
        self,
        case_key: str,
        *,
        query: str,
        cutoff_at: int = 1000,
    ) -> tuple[Path, dict[str, object]]:
        case_dir = self.suite / "cases" / case_key
        case_id = EXPECTED_CASE_IDS[case_key]
        case = {
            "schema_version": "fixture-test",
            "case_id": case_id,
            "umo": f"test:GroupMessage:{case_key}",
            "cutoff_at": cutoff_at,
            "query": query,
        }
        self._write(case_dir / "case.input.json", case)
        self._write(case_dir / "manifest.json", {"case_id": case_id})
        self._write(
            case_dir / "result.private.json",
            {"case_id": case_id, "status": "COMPLETED"},
        )
        return case_dir, case

    def _build_call_726(self) -> None:
        case_dir, _ = self._case_shell("call-726", query="回忆前后态度")
        messages = [
            {
                "source_key": "call-source-old",
                "sent_at": 100,
                "sender_id": "account-a",
                "sender_name": "甲",
                "role": "USER",
                "plain_text": "之前不感兴趣",
            },
            {
                "source_key": "call-source-new",
                "sent_at": 200,
                "sender_id": "account-a",
                "sender_name": "甲",
                "role": "USER",
                "plain_text": "后来决定购买",
            },
        ]
        packet = {
            "host_notice": "test",
            "candidates": {
                "episodes": [{"id": 1, "score": 1.0}],
                "associations": [],
            },
            "expanded_episodes": [
                {
                    "id": 1,
                    "title": "发言变化",
                    "summary": "甲前后有两次不同表述。",
                    "messages": messages,
                }
            ],
            "semantic_evidence": [
                {
                    "memory": {
                        "content": "甲后来决定购买。",
                        "confidence": 0.9,
                    },
                    "evidence": [messages[1]],
                }
            ],
            "feedback_hypothesis_evidence": [],
            "source_count": 2,
        }
        self._write(case_dir / "evidence.input.json", packet)

    def _fixed_packet(
        self,
        case_key: str,
        *,
        count: int,
        episode_count: int,
        actor_role: str,
        evidence_policy: dict[str, object],
    ) -> None:
        case_dir, case = self._case_shell(case_key, query=f"{case_key} 私有问题")
        messages = []
        for index in range(count):
            messages.append(
                {
                    "source_key": f"{case_key}-source-{index:03d}",
                    "umo": case["umo"],
                    "sent_at": 100 + index,
                    "sender_participant_key": f"actor-{index % 3}",
                    "speaker_label": f"成员{index % 3}",
                    "actor_role": actor_role,
                    "plain_text": f"private-{case_key}-message-{index:03d}",
                }
            )
        groups: list[list[str]] = [[] for _ in range(episode_count)]
        for index, message in enumerate(messages):
            groups[index % episode_count].append(str(message["source_key"]))
        packet = {
            "schema_version": 1,
            "case_id": EXPECTED_CASE_IDS[case_key],
            "query": case["query"],
            "query_scope_token": case["umo"],
            "cutoff_at": case["cutoff_at"],
            "diagnostic_type": "fixed-packet-test",
            "end_to_end_retrieval_claim": case_key == "q0030",
            "evidence_policy": evidence_policy,
            "episodes": [
                {
                    "episode_token": f"episode-{index}",
                    "source_keys": source_keys,
                }
                for index, source_keys in enumerate(groups)
            ],
            "messages": messages,
        }
        self._write(case_dir / "evidence.input.json", packet)

    def _build_suite(self) -> None:
        self._build_call_726()
        self._fixed_packet(
            "good-girl",
            count=52,
            episode_count=4,
            actor_role="human",
            evidence_policy={
                "assistant_messages_are_independent_ground_truth": False,
                "temporal_adjacency_is_explicit_reply_to": False,
            },
        )
        self._fixed_packet(
            "q0030",
            count=6,
            episode_count=2,
            actor_role="anonymized_group_member",
            evidence_policy={
                "anonymized_speakers_are_stable_accounts": False,
                "same_scope_only": True,
                "strictly_before_cutoff": True,
                "temporal_adjacency_is_explicit_reply_to": False,
            },
        )

    def test_fixed_packet_adapter_is_verbatim_and_preserves_policy(self) -> None:
        for case_key in ("good-girl", "q0030"):
            frozen = load_private_case(self.suite, case_key)
            self.assertEqual(
                frozen.serving_packet["evidence_policy"], frozen.evidence_policy
            )
            raw_texts = {
                str(item["plain_text"])
                for item in frozen.original_packet["messages"]
            }
            adapted_texts: set[str] = set()
            for episode in frozen.serving_packet["expanded_episodes"]:
                transcript = json.loads(str(episode["summary"]))
                self.assertEqual(transcript["kind"], "verbatim_transcript")
                adapted_texts.update(
                    str(item["text"]) for item in transcript["messages"]
                )
            self.assertEqual(adapted_texts, raw_texts)

        q0030 = load_private_case(self.suite, "q0030")
        for episode in q0030.serving_packet["expanded_episodes"]:
            for message in episode["messages"]:
                self.assertEqual(message["sender_id"], "")
                self.assertTrue(message["sender_name"])

    def test_three_case_suite_writes_private_envelopes_and_metrics(self) -> None:
        output = self.root / "private-output"
        results = run_suite(
            self.suite,
            output_dir=output,
            max_chars=12000,
            repeat=3,
        )

        self.assertEqual(tuple(item.frozen.case_key for item in results), CASE_KEYS)
        for result in results:
            self.assertTrue(result.metrics["deterministic"])
            self.assertEqual(
                result.metrics["memory_provider_calls"], 0
            )
            self.assertEqual(result.metrics["episode_coverage"], 1.0)
            self.assertEqual(result.metrics["warm_repeats"], 3)
            self.assertGreater(result.metrics["assertion_count"], 10)
            private_path = output / f"{result.frozen.case_key}.private.json"
            private = json.loads(private_path.read_text(encoding="utf-8"))
            self.assertEqual(private["query"], result.frozen.query)
            self.assertEqual(private["envelope"], result.envelope_value)
            self.assertEqual(
                private["cost_boundary"]["memory_provider_calls"],
                0,
            )
            expected_semantics = (
                "native_reconstruction_packet_may_contain_precompiled_semantics"
                if result.frozen.case_key == "call-726"
                else "verbatim_transcript_only_no_semantic_summary"
            )
            self.assertEqual(private["adapter_semantics"], expected_semantics)
        self.assertTrue(results[1].envelope.truncated)
        summary = json.loads(
            (output / "summary.private.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["status"], "COMPILER_ONLY_COMPLETED")
        self.assertEqual(summary["execution_scope"], "FROZEN_PACKET_COMPILER_ONLY")
        self.assertEqual(summary["aggregate"]["case_count"], 3)

    def test_cli_runs_100_warm_repeats_and_stdout_contains_no_private_text(self) -> None:
        output = self.root / "cli-private-output"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            status = main(
                [
                    "--suite",
                    str(self.suite),
                    "--output-dir",
                    str(output),
                    "--repeat",
                    "100",
                ]
            )
        self.assertEqual(status, 0)
        rows = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row.get("warm_repeats") == 100 for row in rows[:3]))
        self.assertEqual(rows[-1]["warm_samples"], 300)
        self.assertEqual(rows[-1]["memory_provider_calls"], 0)
        self.assertEqual(rows[-1]["total_tokens"], "UNKNOWN_NOT_RUN")
        self.assertNotIn("private-good-girl-message", stdout.getvalue())
        self.assertNotIn("private-q0030-message", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
