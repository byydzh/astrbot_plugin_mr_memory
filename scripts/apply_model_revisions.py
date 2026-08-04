from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .materialize_retrieval_benchmark import read_jsonl, write_jsonl
except ImportError:  # Direct script execution.
    from materialize_retrieval_benchmark import read_jsonl, write_jsonl


def apply_revisions(
    *,
    reviewed: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    revisions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    documents = {str(item["doc_id"]): item for item in corpus}
    editable = {
        str(item["id"]): item
        for item in reviewed
        if item.get("split") == "human_revision_required"
    }
    revision_index = {str(item.get("id") or ""): item for item in revisions}
    if not revision_index or "" in revision_index or len(revision_index) != len(revisions):
        raise ValueError("revision ids must be non-empty and unique")
    if set(revision_index) != set(editable):
        raise ValueError(
            "revisions must cover exactly the human_revision_required items: "
            f"expected={sorted(editable)}, got={sorted(revision_index)}"
        )

    output: list[dict[str, Any]] = []
    for item_id, source in editable.items():
        revision = revision_index[item_id]
        query = str(revision.get("query") or "").strip()
        positives = [str(value) for value in revision.get("positive_doc_ids") or []]
        memory_type = str(revision.get("memory_type") or "").strip()
        epistemic = str(revision.get("epistemic") or "").strip()
        rationale = str(revision.get("rationale") or "").strip()
        if len(query) < 4 or not positives or len(positives) != len(set(positives)):
            raise ValueError(f"{item_id}: incomplete model revision")
        if not memory_type or not epistemic or not rationale:
            raise ValueError(f"{item_id}: revised labels and rationale are required")
        for doc_id in positives:
            document = documents.get(doc_id)
            if document is None:
                raise ValueError(f"{item_id}: revision references unknown evidence: {doc_id}")
            if str(document.get("scope_id")) != str(source["scope_id"]):
                raise ValueError(f"{item_id}: revision crosses group scope: {doc_id}")
        query_time = max(int(documents[doc_id]["sent_at"]) for doc_id in positives) + 1
        provenance = dict(source.get("provenance") or {})
        provenance.update(
            {
                "prior_human_decision": "edit",
                "model_revision_method": "direct_response_to_human_feedback",
                "revision_human_reviewed": False,
                "human_approved": False,
            }
        )
        output.append(
            {
                **source,
                "query": query,
                "query_time": query_time,
                "positive_doc_ids": positives,
                "memory_type": memory_type,
                "epistemic": epistemic,
                "split": "ai_revision_pending_human_review",
                "model_revision_rationale": rationale,
                "provenance": provenance,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Revise items returned by a human reviewer.")
    parser.add_argument("benchmark_dir", type=Path)
    parser.add_argument("revisions", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    revision_values = read_jsonl(args.revisions)
    output = apply_revisions(
        reviewed=read_jsonl(args.benchmark_dir / "benchmark_human_reviewed.jsonl"),
        corpus=read_jsonl(args.benchmark_dir / "corpus.jsonl"),
        revisions=revision_values,
    )
    output_path = args.output or args.benchmark_dir / "benchmark_round2.jsonl"
    write_jsonl(output_path, output)
    print(json.dumps({"output": str(output_path.resolve()), "items": len(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
