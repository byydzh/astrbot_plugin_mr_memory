from __future__ import annotations

import json
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

from scripts.build_three_case_fixtures import build_case_c_fixture
from scripts.eccr_packet_experiment import load_case_bundle


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


@contextmanager
def _workspace_tempdir():
    parent = Path.cwd() / ".test-artifacts"
    parent.mkdir(exist_ok=True)
    path = parent / uuid.uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class ThreeCaseFixtureBuilderTest(unittest.TestCase):
    def test_case_c_uses_blind_ranking_then_neighbor_expansion(self) -> None:
        with _workspace_tempdir() as root:
            benchmark_dir = root / "benchmark"
            output_dir = root / "fixture"
            benchmark_dir.mkdir()
            corpus = [
                {
                    "doc_id": "synthetic-doc-a",
                    "scope_id": "scope-a",
                    "sent_at": 100,
                    "speaker": "成员003",
                    "text": "合成纸鹤活动的签到台安排在东门。",
                },
                {
                    "doc_id": "synthetic-doc-b",
                    "scope_id": "scope-a",
                    "sent_at": 107,
                    "speaker": "成员015",
                    "text": "参加纸鹤活动前需要先去西门领材料吗？",
                },
                {
                    "doc_id": "synthetic-doc-c",
                    "scope_id": "scope-a",
                    "sent_at": 108,
                    "speaker": "成员003",
                    "text": "直接到东门签到即可。",
                },
                {
                    "doc_id": "synthetic-doc-d",
                    "scope_id": "scope-a",
                    "sent_at": 109,
                    "speaker": "成员003",
                    "text": "材料会在签到台发放，不用去西门。",
                },
                {
                    "doc_id": "future",
                    "scope_id": "scope-a",
                    "sent_at": 111,
                    "speaker": "成员003",
                    "text": "未来消息不得进入证据包",
                },
            ]
            benchmark = [
                {
                    "id": "case-c",
                    "scope_id": "scope-a",
                    "query": "参加合成纸鹤活动应去哪签到，材料怎么领？",
                    "query_time": 110,
                    "positive_doc_ids": [
                        "synthetic-doc-a",
                        "synthetic-doc-b",
                        "synthetic-doc-c",
                        "synthetic-doc-d",
                    ],
                    "provenance": {
                        "human_approved": True,
                        "positive_basis": "reviewed reply chain",
                        "human_review_dataset_fingerprint": "fixture-test",
                    },
                }
            ]
            _write_jsonl(benchmark_dir / "corpus.jsonl", corpus)
            _write_jsonl(benchmark_dir / "benchmark_gold_final.jsonl", benchmark)

            with self.assertRaisesRegex(ValueError, "private answer_rubric is required"):
                build_case_c_fixture(
                    benchmark_dir=benchmark_dir,
                    output_dir=output_dir,
                    top_k=1,
                    neighbor_radius=10,
                    secondary_neighbor_radius=0,
                )
            manifest = build_case_c_fixture(
                benchmark_dir=benchmark_dir,
                output_dir=output_dir,
                top_k=1,
                neighbor_radius=10,
                secondary_neighbor_radius=0,
                answer_rubric={
                    "required_semantics": ["合成活动在东门签到，材料在签到台发放。"],
                    "required_uncertainty": [],
                    "forbidden_conclusions": ["需要先去西门领取材料。"],
                },
            )

            self.assertFalse(manifest["retrieval_audit"]["selection_used_gold"])
            self.assertEqual(manifest["retrieval_audit"]["positive_sources_delivered"], 4)
            packet = json.loads((output_dir / "evidence_packet.json").read_text("utf-8"))
            source_keys = {item["source_key"] for item in packet["messages"]}
            self.assertNotIn("future", source_keys)
            self.assertTrue(
                {"synthetic-doc-a", "synthetic-doc-b", "synthetic-doc-c", "synthetic-doc-d"}.issubset(source_keys)
            )
            case = json.loads((output_dir / "case.json").read_text("utf-8"))
            self.assertNotIn("required_semantics", case)
            self.assertNotIn("positive_doc_ids", case)
            gold = json.loads((output_dir / "gold.json").read_text("utf-8"))
            self.assertEqual(gold["required_semantics"], ["合成活动在东门签到，材料在签到台发放。"])
            load_case_bundle(output_dir)

    def test_missing_blind_evidence_fails_instead_of_gold_injection(self) -> None:
        with _workspace_tempdir() as root:
            benchmark_dir = root / "benchmark"
            benchmark_dir.mkdir()
            _write_jsonl(
                benchmark_dir / "corpus.jsonl",
                [
                    {
                        "doc_id": "anchor",
                        "scope_id": "scope-a",
                        "sent_at": 1,
                        "speaker": "成员001",
                        "text": "纸鹤活动",
                    },
                    {
                        "doc_id": "gold-too-far",
                        "scope_id": "scope-a",
                        "sent_at": 50,
                        "speaker": "成员002",
                        "text": "完全无关词",
                    },
                ],
            )
            _write_jsonl(
                benchmark_dir / "benchmark_gold_final.jsonl",
                [
                    {
                        "id": "case-c",
                        "scope_id": "scope-a",
                        "query": "纸鹤活动",
                        "query_time": 100,
                        "positive_doc_ids": ["gold-too-far"],
                        "provenance": {"human_approved": True},
                    }
                ],
            )
            with self.assertRaisesRegex(ValueError, "did not deliver"):
                build_case_c_fixture(
                    benchmark_dir=benchmark_dir,
                    output_dir=root / "fixture",
                    top_k=1,
                    neighbor_radius=0,
                    secondary_neighbor_radius=0,
                )


if __name__ == "__main__":
    unittest.main()
