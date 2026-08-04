from __future__ import annotations

import argparse
import collections
import json
import math
import re
from pathlib import Path
from typing import Any

try:
    from .materialize_retrieval_benchmark import read_jsonl
except ImportError:  # Direct script execution.
    from materialize_retrieval_benchmark import read_jsonl


WORD_RE = re.compile(r"[\u3400-\u9fff]|[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    units = WORD_RE.findall(text.casefold())
    chinese = "".join(unit for unit in units if len(unit) == 1 and "\u3400" <= unit <= "\u9fff")
    bigrams = [chinese[index : index + 2] for index in range(max(0, len(chinese) - 1))]
    latin = [unit for unit in units if len(unit) > 1]
    return units + bigrams + latin


class BM25Index:
    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self.documents = documents
        self.lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
        for index, document in enumerate(documents):
            counts = collections.Counter(
                tokenize(f"{document.get('speaker', '')} {document.get('text', '')}")
            )
            self.lengths.append(sum(counts.values()))
            for token, frequency in counts.items():
                self.postings[token].append((index, frequency))
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))

    def rank(
        self,
        query: str,
        *,
        scope_id: str,
        before: int,
        limit: int,
    ) -> list[str]:
        query_terms = set(tokenize(query))
        scores: collections.defaultdict[int, float] = collections.defaultdict(float)
        document_count = len(self.documents)
        k1 = 1.5
        b = 0.75
        for token in query_terms:
            postings = self.postings.get(token) or []
            if not postings:
                continue
            idf = math.log(1 + (document_count - len(postings) + 0.5) / (len(postings) + 0.5))
            for index, frequency in postings:
                document = self.documents[index]
                if str(document.get("scope_id")) != scope_id:
                    continue
                if int(document.get("sent_at", 0)) >= before:
                    continue
                length_norm = 1 - b + b * self.lengths[index] / max(self.average_length, 1.0)
                scores[index] += idf * (frequency * (k1 + 1)) / (frequency + k1 * length_norm)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
        return [str(self.documents[index]["doc_id"]) for index, _score in ranked]


def evaluate(
    corpus: list[dict[str, Any]],
    benchmark: list[dict[str, Any]],
    *,
    split: str,
    cutoffs: tuple[int, ...] = (1, 5, 10, 20),
) -> dict[str, Any]:
    selected = [item for item in benchmark if split == "all" or item.get("split") == split]
    if not selected:
        raise ValueError(f"benchmark split is empty: {split}")
    index = BM25Index(corpus)
    maximum = max(cutoffs)
    hit_totals = collections.Counter({cutoff: 0.0 for cutoff in cutoffs})
    recall_totals = collections.Counter({cutoff: 0.0 for cutoff in cutoffs})
    reciprocal_rank = 0.0
    all_evidence_at_20 = 0
    misses: list[str] = []
    for item in selected:
        ranking = index.rank(
            str(item["query"]),
            scope_id=str(item["scope_id"]),
            before=int(item["query_time"]),
            limit=maximum,
        )
        positives = set(str(value) for value in item["positive_doc_ids"])
        ranks = [index + 1 for index, doc_id in enumerate(ranking) if doc_id in positives]
        if ranks:
            reciprocal_rank += 1.0 / min(ranks)
        else:
            misses.append(str(item["id"]))
        for cutoff in cutoffs:
            found = positives & set(ranking[:cutoff])
            hit_totals[cutoff] += float(bool(found))
            recall_totals[cutoff] += len(found) / len(positives)
        if positives.issubset(set(ranking[:20])):
            all_evidence_at_20 += 1

    count = len(selected)
    return {
        "backend": "bm25-char-bigram",
        "split": split,
        "queries": count,
        "mrr_at_20": round(reciprocal_rank / count, 4),
        "hit_rate": {
            f"@{cutoff}": round(hit_totals[cutoff] / count, 4) for cutoff in cutoffs
        },
        "mean_evidence_recall": {
            f"@{cutoff}": round(recall_totals[cutoff] / count, 4) for cutoff in cutoffs
        },
        "all_evidence_at_20": round(all_evidence_at_20 / count, 4),
        "misses_at_20": misses,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local lexical sanity baseline.")
    parser.add_argument("benchmark_dir", type=Path)
    parser.add_argument(
        "--benchmark-file",
        type=Path,
        help="Benchmark JSONL to evaluate; defaults to benchmark.jsonl.",
    )
    parser.add_argument(
        "--split",
        choices=(
            "ai_high_confidence",
            "ai_low_confidence",
            "gold",
            "human_revision_required",
            "ai_revision_pending_human_review",
            "rejected",
            "all",
        ),
        default="ai_high_confidence",
    )
    args = parser.parse_args()
    benchmark_path = args.benchmark_file or args.benchmark_dir / "benchmark.jsonl"
    result = evaluate(
        read_jsonl(args.benchmark_dir / "corpus.jsonl"),
        read_jsonl(benchmark_path),
        split=args.split,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
