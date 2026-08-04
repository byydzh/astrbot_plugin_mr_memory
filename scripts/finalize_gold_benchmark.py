from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .materialize_retrieval_benchmark import read_jsonl, write_jsonl
except ImportError:  # Direct script execution.
    from materialize_retrieval_benchmark import read_jsonl, write_jsonl


ITEM_NUMBER_RE = re.compile(r"(\d+)$")


def _item_sort_key(item: dict[str, Any]) -> tuple[int, str]:
    item_id = str(item.get("id") or "")
    match = ITEM_NUMBER_RE.search(item_id)
    return (int(match.group(1)) if match else 2**31, item_id)


def _validate_evidence(
    item: dict[str, Any],
    *,
    documents: dict[str, dict[str, Any]],
) -> None:
    item_id = str(item.get("id") or "")
    scope_id = str(item.get("scope_id") or "")
    query_time = int(item.get("query_time", 0))
    positives = [str(value) for value in item.get("positive_doc_ids") or []]
    if not item_id or not scope_id or len(str(item.get("query") or "").strip()) < 4:
        raise ValueError(f"invalid benchmark item: {item_id!r}")
    if not positives or len(positives) != len(set(positives)):
        raise ValueError(f"{item_id}: positive evidence must be non-empty and unique")
    for doc_id in positives:
        document = documents.get(doc_id)
        if document is None:
            raise ValueError(f"{item_id}: unknown positive evidence: {doc_id}")
        if str(document.get("scope_id")) != scope_id:
            raise ValueError(f"{item_id}: positive evidence crosses group scope: {doc_id}")
        if int(document.get("sent_at", 0)) >= query_time:
            raise ValueError(f"{item_id}: positive evidence leaks past query_time: {doc_id}")


def finalize_gold(
    *,
    base_gold: list[dict[str, Any]],
    approved_revisions: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    approval_source: str,
    approved_at: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    documents = {str(item.get("doc_id") or ""): item for item in corpus}
    if "" in documents or len(documents) != len(corpus):
        raise ValueError("corpus doc_ids must be non-empty and unique")

    base_ids = [str(item.get("id") or "") for item in base_gold]
    revision_ids = [str(item.get("id") or "") for item in approved_revisions]
    if "" in base_ids or len(base_ids) != len(set(base_ids)):
        raise ValueError("base gold ids must be non-empty and unique")
    if "" in revision_ids or len(revision_ids) != len(set(revision_ids)):
        raise ValueError("revision ids must be non-empty and unique")
    overlap = set(base_ids) & set(revision_ids)
    if overlap:
        raise ValueError(f"base gold and revisions overlap: {sorted(overlap)}")
    if not approval_source.strip():
        raise ValueError("approval_source is required")

    finalized: list[dict[str, Any]] = []
    for source in base_gold:
        item = copy.deepcopy(source)
        if item.get("split") != "gold":
            raise ValueError(f"{item.get('id')}: base item is not gold")
        if not bool((item.get("provenance") or {}).get("human_approved")):
            raise ValueError(f"{item.get('id')}: base item lacks human approval")
        _validate_evidence(item, documents=documents)
        finalized.append(item)

    for source in approved_revisions:
        item = copy.deepcopy(source)
        if item.get("split") != "ai_revision_pending_human_review":
            raise ValueError(f"{item.get('id')}: revision is not pending human review")
        _validate_evidence(item, documents=documents)
        provenance = dict(item.get("provenance") or {})
        provenance.update(
            {
                "human_reviewed": True,
                "human_approved": True,
                "human_decision": "accept",
                "revision_human_reviewed": True,
                "revision_human_decision": "accept",
                "revision_human_approval_source": approval_source,
                "revision_human_approved_at": approved_at,
            }
        )
        item["split"] = "gold"
        item["provenance"] = provenance
        finalized.append(item)

    finalized.sort(key=_item_sort_key)
    summary = {
        "format_version": 1,
        "items": len(finalized),
        "base_human_approved_items": len(base_gold),
        "round2_human_approved_items": len(approved_revisions),
        "round2_item_ids": sorted(revision_ids),
        "approval_source": approval_source,
        "approved_at": approved_at,
        "splits": dict(sorted(Counter(item["split"] for item in finalized).items())),
        "memory_types": dict(
            sorted(Counter(str(item.get("memory_type")) for item in finalized).items())
        ),
        "epistemic_labels": dict(
            sorted(Counter(str(item.get("epistemic")) for item in finalized).items())
        ),
        "evaluation_contract": {
            "scope_filter_required": True,
            "candidate_time_rule": "document.sent_at < query.query_time",
            "positive_doc_ids_are_human_verified": True,
            "lexical_decoys_are_verified_negatives": False,
        },
    }
    return finalized, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote explicitly approved round-two revisions into final gold."
    )
    parser.add_argument("benchmark_dir", type=Path)
    parser.add_argument(
        "--approval-source",
        default="explicit_user_approval_in_codex_task",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    directory = args.benchmark_dir
    approved_at = datetime.now(timezone.utc).isoformat()
    finalized, summary = finalize_gold(
        base_gold=read_jsonl(directory / "benchmark_gold.jsonl"),
        approved_revisions=read_jsonl(directory / "benchmark_round2.jsonl"),
        corpus=read_jsonl(directory / "corpus.jsonl"),
        approval_source=args.approval_source,
        approved_at=approved_at,
    )
    output_path = args.output or directory / "benchmark_gold_final.jsonl"
    write_jsonl(output_path, finalized)
    (directory / "benchmark_gold_final_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["final_gold"] = summary
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output_path.resolve()), **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
