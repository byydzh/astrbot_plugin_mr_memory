from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected one JSON object")
            records.append(value)
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _unique_index(records: list[dict[str, Any]], key: str, source: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for record in records:
        value = str(record.get(key) or "")
        if not value:
            raise ValueError(f"{source}: missing {key}")
        if value in output:
            raise ValueError(f"{source}: duplicate {key}: {value}")
        output[value] = record
    return output


def materialize(
    corpus: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    docs = _unique_index(corpus, "doc_id", "corpus")
    candidate_index = _unique_index(candidates, "candidate_id", "candidates")
    annotation_ids: set[str] = set()
    benchmark: list[dict[str, Any]] = []

    for annotation in annotations:
        item_id = str(annotation.get("id") or "")
        if not item_id or item_id in annotation_ids:
            raise ValueError(f"invalid or duplicate annotation id: {item_id!r}")
        annotation_ids.add(item_id)
        candidate_id = str(annotation.get("candidate_id") or "")
        if candidate_id not in candidate_index:
            raise ValueError(f"{item_id}: unknown candidate: {candidate_id}")
        candidate = candidate_index[candidate_id]
        scope_id = str(candidate["scope_id"])
        query = str(annotation.get("query") or "").strip()
        if len(query) < 4:
            raise ValueError(f"{item_id}: query is too short")

        positive_ids = [str(value) for value in annotation.get("positive_doc_ids") or []]
        if not positive_ids or len(positive_ids) != len(set(positive_ids)):
            raise ValueError(f"{item_id}: positive_doc_ids must be non-empty and unique")
        for doc_id in positive_ids:
            document = docs.get(doc_id)
            if document is None:
                raise ValueError(f"{item_id}: unknown positive document: {doc_id}")
            if str(document.get("scope_id")) != scope_id:
                raise ValueError(f"{item_id}: positive document crosses group scope: {doc_id}")

        policy = str(annotation.get("query_time_policy") or "")
        if policy == "at_observed_query":
            query_time = int(candidate["query_time"])
        elif policy == "after_observed_query":
            query_doc_id = str(candidate["query_doc_id"])
            if query_doc_id not in positive_ids:
                raise ValueError(
                    f"{item_id}: after_observed_query requires the observed query document "
                    "to be positive, preventing query leakage"
                )
            query_time = max(int(docs[doc_id]["sent_at"]) for doc_id in positive_ids) + 1
        else:
            raise ValueError(f"{item_id}: unknown query_time_policy: {policy}")

        future = [doc_id for doc_id in positive_ids if int(docs[doc_id]["sent_at"]) >= query_time]
        if future:
            raise ValueError(f"{item_id}: positive evidence is not before query_time: {future}")

        decoys: list[str] = []
        for record in candidate.get("hard_negatives") or []:
            doc_id = str(record.get("doc_id") or "")
            document = docs.get(doc_id)
            if (
                document is None
                or doc_id in positive_ids
                or str(document.get("scope_id")) != scope_id
                or int(document["sent_at"]) >= query_time
            ):
                continue
            if doc_id not in decoys:
                decoys.append(doc_id)

        needs_review = bool(annotation.get("needs_review", False))
        benchmark.append(
            {
                "id": item_id,
                "scope_id": scope_id,
                "query": query,
                "query_time": query_time,
                "positive_doc_ids": positive_ids,
                "lexical_decoy_doc_ids": decoys,
                "memory_type": str(annotation.get("memory_type") or ""),
                "epistemic": str(annotation.get("epistemic") or ""),
                "confidence": float(annotation.get("confidence", 0.0)),
                "split": "ai_low_confidence" if needs_review else "ai_high_confidence",
                "provenance": {
                    "candidate_id": candidate_id,
                    "annotation_method": "model_direct_evidence_first_pass",
                    "human_reviewed": False,
                    "human_review_priority": "high" if needs_review else "normal",
                    "positive_basis": "real delayed reply chain plus surrounding evidence",
                    "decoy_method": "automatic lexical selection; not a verified negative label",
                },
            }
        )

    split_counts = collections.Counter(item["split"] for item in benchmark)
    type_counts = collections.Counter(item["memory_type"] for item in benchmark)
    epistemic_counts = collections.Counter(item["epistemic"] for item in benchmark)
    manifest = {
        "format_version": 1,
        "items": len(benchmark),
        "splits": dict(sorted(split_counts.items())),
        "memory_types": dict(sorted(type_counts.items())),
        "epistemic_labels": dict(sorted(epistemic_counts.items())),
        "evaluation_contract": {
            "scope_filter_required": True,
            "candidate_time_rule": "document.sent_at < query.query_time",
            "default_split": "ai_high_confidence",
            "positive_doc_ids_are_model_annotated": True,
            "positive_doc_ids_are_human_verified": False,
            "lexical_decoys_are_verified_negatives": False,
        },
    }
    return benchmark, manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and materialize model-produced first-pass retrieval labels."
    )
    parser.add_argument("benchmark_dir", type=Path)
    args = parser.parse_args()

    directory = args.benchmark_dir
    corpus = read_jsonl(directory / "corpus.jsonl")
    candidates = read_jsonl(directory / "candidates.jsonl")
    annotations = read_jsonl(directory / "annotations.jsonl")
    benchmark, summary = materialize(corpus, candidates, annotations)
    write_jsonl(directory / "benchmark.jsonl", benchmark)

    source_manifest_path = directory / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_manifest.pop("manual_benchmark", None)
    source_manifest["model_labeled_benchmark"] = summary
    source_manifest_path.write_text(
        json.dumps(source_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
