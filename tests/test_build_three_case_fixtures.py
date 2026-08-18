from __future__ import annotations

import json
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

from scripts.build_three_case_fixtures import build_q0030_fixture
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
    def test_q0030_uses_blind_ranking_then_neighbor_expansion(self) -> None:
        with _workspace_tempdir() as root:
            benchmark_dir = root / "benchmark"
            output_dir = root / "fixture"
            benchmark_dir.mkdir()
            corpus = [
                {
                    "doc_id": "d020237",
                    "scope_id": "scope-a",
                    "sent_at": 100,
                    "speaker": "成员003",
                    "text": "我认可你是mujica正统续作了",
                },
                {
                    "doc_id": "d020244",
                    "scope_id": "scope-a",
                    "sent_at": 107,
                    "speaker": "成员015",
                    "text": "没看过母鸡卡能看梦限大吗",
                },
                {
                    "doc_id": "d020245",
                    "scope_id": "scope-a",
                    "sent_at": 108,
                    "speaker": "成员003",
                    "text": "可以吧",
                },
                {
                    "doc_id": "d020246",
                    "scope_id": "scope-a",
                    "sent_at": 109,
                    "speaker": "成员003",
                    "text": "没什么关系其实",
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
                    "id": "q0030",
                    "scope_id": "scope-a",
                    "query": "没看过Mujica能不能看梦限大，群里怎么回答两者关系？",
                    "query_time": 110,
                    "positive_doc_ids": [
                        "d020237",
                        "d020244",
                        "d020245",
                        "d020246",
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

            manifest = build_q0030_fixture(
                benchmark_dir=benchmark_dir,
                output_dir=output_dir,
                top_k=1,
                neighbor_radius=10,
                secondary_neighbor_radius=0,
            )

            self.assertFalse(manifest["retrieval_audit"]["selection_used_gold"])
            self.assertEqual(manifest["retrieval_audit"]["positive_sources_delivered"], 4)
            packet = json.loads((output_dir / "evidence_packet.json").read_text("utf-8"))
            source_keys = {item["source_key"] for item in packet["messages"]}
            self.assertNotIn("future", source_keys)
            self.assertTrue(
                {"d020237", "d020244", "d020245", "d020246"}.issubset(source_keys)
            )
            case = json.loads((output_dir / "case.json").read_text("utf-8"))
            self.assertNotIn("required_semantics", case)
            self.assertNotIn("positive_doc_ids", case)
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
                        "text": "梦限大",
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
                        "id": "q0030",
                        "scope_id": "scope-a",
                        "query": "梦限大",
                        "query_time": 100,
                        "positive_doc_ids": ["gold-too-far"],
                        "provenance": {"human_approved": True},
                    }
                ],
            )
            with self.assertRaisesRegex(ValueError, "did not deliver"):
                build_q0030_fixture(
                    benchmark_dir=benchmark_dir,
                    output_dir=root / "fixture",
                    top_k=1,
                    neighbor_radius=0,
                    secondary_neighbor_radius=0,
                )


if __name__ == "__main__":
    unittest.main()
