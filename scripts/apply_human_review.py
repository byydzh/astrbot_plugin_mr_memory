from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

try:
    from .materialize_retrieval_benchmark import read_jsonl, write_jsonl
except ImportError:  # Direct script execution.
    from materialize_retrieval_benchmark import read_jsonl, write_jsonl


VALID_DECISIONS = {"accept", "edit", "reject", ""}


def dataset_fingerprint(directory: Path, benchmark_path: Path | None = None) -> str:
    benchmark_path = benchmark_path or directory / "benchmark.jsonl"
    return hashlib.sha256(
        benchmark_path.read_bytes()
        + b"\0"
        + (directory / "corpus.jsonl").read_bytes()
    ).hexdigest()[:20]


def apply_review(
    *,
    directory: Path,
    review: dict[str, Any],
    benchmark_path: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    benchmark_path = benchmark_path or directory / "benchmark.jsonl"
    expected_fingerprint = dataset_fingerprint(directory, benchmark_path)
    actual_fingerprint = str(review.get("dataset_fingerprint") or "")
    if actual_fingerprint != expected_fingerprint:
        raise ValueError(
            "human review fingerprint mismatch: "
            f"expected {expected_fingerprint}, got {actual_fingerprint}"
        )
    if str(review.get("reviewer_type") or "") != "human":
        raise ValueError("review export does not identify a human reviewer")

    benchmark = read_jsonl(benchmark_path)
    corpus = {str(item["doc_id"]): item for item in read_jsonl(directory / "corpus.jsonl")}
    candidates = {
        str(item["candidate_id"]): item
        for item in read_jsonl(directory / "candidates.jsonl")
    }
    reviews = review.get("reviews") or []
    review_index: dict[str, dict[str, Any]] = {}
    for item in reviews:
        if not isinstance(item, dict):
            raise ValueError("human review contains a non-object entry")
        item_id = str(item.get("id") or "")
        if not item_id or item_id in review_index:
            raise ValueError(f"invalid or duplicate human review id: {item_id!r}")
        review_index[item_id] = item

    benchmark_ids = {str(item["id"]) for item in benchmark}
    if set(review_index) != benchmark_ids:
        missing = sorted(benchmark_ids - set(review_index))
        unknown = sorted(set(review_index) - benchmark_ids)
        raise ValueError(f"review coverage mismatch: missing={missing}, unknown={unknown}")

    reviewed: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    decisions: collections.Counter[str] = collections.Counter()
    changed_queries = 0
    changed_evidence = 0
    changed_types = 0
    changed_epistemic = 0
    adjusted_query_times: list[str] = []

    for source in benchmark:
        item_id = str(source["id"])
        human = review_index[item_id]
        decision = str(human.get("decision") or "")
        if decision not in VALID_DECISIONS:
            raise ValueError(f"{item_id}: unsupported human decision: {decision}")
        decisions[decision or "unreviewed"] += 1

        query = str(human.get("query") or "").strip()
        memory_type = str(human.get("memory_type") or "").strip()
        epistemic = str(human.get("epistemic") or "").strip()
        positives = [str(value) for value in human.get("positive_doc_ids") or []]
        if decision != "reject" and (len(query) < 4 or not memory_type or not epistemic):
            raise ValueError(f"{item_id}: accepted/revisable review has incomplete labels")
        if decision != "reject" and (not positives or len(positives) != len(set(positives))):
            raise ValueError(f"{item_id}: reviewed evidence must be non-empty and unique")

        scope_id = str(source["scope_id"])
        for doc_id in positives:
            document = corpus.get(doc_id)
            if document is None:
                raise ValueError(f"{item_id}: reviewed evidence does not exist: {doc_id}")
            if str(document.get("scope_id")) != scope_id:
                raise ValueError(f"{item_id}: reviewed evidence crosses group scope: {doc_id}")

        original_positives = [str(value) for value in source["positive_doc_ids"]]
        query_time = int(source["query_time"])
        query_time_reason = "preserved_model_cutoff"
        if positives:
            latest_evidence = max(int(corpus[doc_id]["sent_at"]) for doc_id in positives)
            candidate_id = str((source.get("provenance") or {}).get("candidate_id") or "")
            candidate_query_doc = str(candidates[candidate_id].get("query_doc_id") or "")
            removed_observed_query = (
                candidate_query_doc in original_positives
                and candidate_query_doc not in positives
            )
            if latest_evidence >= query_time:
                query_time = latest_evidence + 1
                query_time_reason = "moved_after_human_added_evidence"
            elif removed_observed_query:
                query_time = latest_evidence + 1
                query_time_reason = "moved_before_removed_observed_reply"
            if any(int(corpus[doc_id]["sent_at"]) >= query_time for doc_id in positives):
                raise ValueError(f"{item_id}: reviewed evidence violates temporal cutoff")
        if query_time_reason != "preserved_model_cutoff":
            adjusted_query_times.append(item_id)

        changed_queries += int(query != str(source["query"]))
        changed_evidence += int(positives != original_positives)
        changed_types += int(memory_type != str(source["memory_type"]))
        changed_epistemic += int(epistemic != str(source["epistemic"]))

        split = {
            "accept": "gold",
            "edit": "human_revision_required",
            "reject": "rejected",
            "": "unreviewed",
        }[decision]
        provenance = dict(source.get("provenance") or {})
        provenance.update(
            {
                "human_reviewed": bool(decision),
                "human_approved": decision == "accept",
                "human_decision": decision or "unreviewed",
                "human_review_dataset_fingerprint": actual_fingerprint,
                "human_review_exported_at": str(review.get("exported_at") or ""),
                "query_time_review_rule": query_time_reason,
            }
        )
        merged = {
            **source,
            "query": query,
            "query_time": query_time,
            "positive_doc_ids": positives,
            "memory_type": memory_type,
            "epistemic": epistemic,
            "split": split,
            "human_review_notes": str(human.get("notes") or "").strip(),
            "provenance": provenance,
        }
        reviewed.append(merged)
        if decision == "accept":
            gold.append(merged)

    summary = {
        "format_version": 1,
        "dataset_fingerprint": actual_fingerprint,
        "review_exported_at": str(review.get("exported_at") or ""),
        "items": len(reviewed),
        "decisions": dict(sorted(decisions.items())),
        "gold_items": len(gold),
        "changed_queries": changed_queries,
        "changed_evidence_sets": changed_evidence,
        "changed_memory_types": changed_types,
        "changed_epistemic_labels": changed_epistemic,
        "query_time_adjusted_items": adjusted_query_times,
    }
    return reviewed, gold, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a fingerprinted human review export.")
    parser.add_argument("benchmark_dir", type=Path)
    parser.add_argument("review_export", type=Path)
    parser.add_argument(
        "--benchmark-file",
        type=Path,
        help="Reviewed benchmark JSONL; defaults to benchmark.jsonl.",
    )
    parser.add_argument(
        "--output-prefix",
        help="Suffix for round-specific outputs, for example round2.",
    )
    args = parser.parse_args()

    review = json.loads(args.review_export.read_text(encoding="utf-8"))
    if not isinstance(review, dict):
        raise ValueError("human review export must be a JSON object")
    benchmark_path = args.benchmark_file or args.benchmark_dir / "benchmark.jsonl"
    reviewed, gold, summary = apply_review(
        directory=args.benchmark_dir,
        review=review,
        benchmark_path=benchmark_path,
    )
    prefix = str(args.output_prefix or "").strip()
    if not prefix and benchmark_path.name != "benchmark.jsonl":
        prefix = benchmark_path.stem.removeprefix("benchmark_")
    suffix = f"_{prefix}" if prefix else ""
    reviewed_path = args.benchmark_dir / f"benchmark{suffix}_human_reviewed.jsonl"
    gold_path = args.benchmark_dir / f"benchmark{suffix}_gold.jsonl"
    review_copy_path = args.benchmark_dir / f"human_review{suffix}.json"
    summary_path = args.benchmark_dir / f"human_review{suffix}_summary.json"
    write_jsonl(reviewed_path, reviewed)
    write_jsonl(gold_path, gold)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.copyfile(args.review_export, review_copy_path)

    manifest_path = args.benchmark_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[f"human_review{suffix}"] = summary
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
