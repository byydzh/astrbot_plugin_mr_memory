from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

try:
    from .evaluate_retrieval_benchmark import BM25Index
    from .materialize_retrieval_benchmark import read_jsonl
except ImportError:  # Direct script execution.
    from evaluate_retrieval_benchmark import BM25Index
    from materialize_retrieval_benchmark import read_jsonl


CASE_SCHEMA_VERSION = "eccr.packet.case.v1"
GOLD_SCHEMA_VERSION = "eccr.packet.gold.v1"
PACKET_SCHEMA_VERSION = 1


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _one(items: Iterable[dict[str, Any]], *, field: str, value: str) -> dict[str, Any]:
    selected = [item for item in items if str(item.get(field) or "") == value]
    if len(selected) != 1:
        raise ValueError(f"expected exactly one row with {field}={value!r}, got {len(selected)}")
    return selected[0]


def _actor_token(scope_id: str, speaker: str) -> str:
    payload = f"{scope_id}\0{speaker}".encode("utf-8")
    return "actor_" + hashlib.sha256(payload).hexdigest()[:14]


def _expand_neighbors(
    corpus: list[dict[str, Any]],
    *,
    scope_id: str,
    cutoff_at: int,
    anchor_ids: list[str],
    primary_radius: int,
    secondary_radius: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible = [
        item
        for item in corpus
        if str(item.get("scope_id") or "") == scope_id
        and int(item.get("sent_at") or 0) < cutoff_at
    ]
    eligible.sort(key=lambda item: (int(item.get("sent_at") or 0), str(item.get("doc_id") or "")))
    positions = {str(item["doc_id"]): index for index, item in enumerate(eligible)}
    missing = [doc_id for doc_id in anchor_ids if doc_id not in positions]
    if missing:
        raise ValueError(f"retrieval anchors are absent before cutoff: {missing}")

    selected_positions: set[int] = set()
    episodes: list[dict[str, Any]] = []
    for ordinal, anchor_id in enumerate(anchor_ids, start=1):
        radius = primary_radius if ordinal == 1 else secondary_radius
        position = positions[anchor_id]
        start = max(0, position - radius)
        stop = min(len(eligible), position + radius + 1)
        window = eligible[start:stop]
        selected_positions.update(range(start, stop))
        episodes.append(
            {
                "episode_token": f"episode_{ordinal:02d}",
                "anchor_source_key": anchor_id,
                "neighborhood_type": "blind_bm25_chronological_neighbor_expansion",
                "source_keys": [str(item["doc_id"]) for item in window],
            }
        )
    messages = [eligible[index] for index in sorted(selected_positions)]
    return episodes, messages


def build_q0030_fixture(
    *,
    benchmark_dir: Path,
    output_dir: Path,
    top_k: int = 6,
    neighbor_radius: int = 10,
    secondary_neighbor_radius: int = 1,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if neighbor_radius < 0:
        raise ValueError("neighbor_radius must be non-negative")
    if secondary_neighbor_radius < 0:
        raise ValueError("secondary_neighbor_radius must be non-negative")

    corpus_path = benchmark_dir / "corpus.jsonl"
    benchmark_path = benchmark_dir / "benchmark_gold_final.jsonl"
    corpus = read_jsonl(corpus_path)
    benchmark = read_jsonl(benchmark_path)
    row = _one(benchmark, field="id", value="q0030")
    if not bool((row.get("provenance") or {}).get("human_approved")):
        raise ValueError("q0030 must remain human-approved before fixture construction")

    scope_id = str(row["scope_id"])
    cutoff_at = int(row["query_time"])
    query = str(row["query"])
    ranking = BM25Index(corpus).rank(
        query,
        scope_id=scope_id,
        before=cutoff_at,
        limit=top_k,
    )
    episodes, selected = _expand_neighbors(
        corpus,
        scope_id=scope_id,
        cutoff_at=cutoff_at,
        anchor_ids=ranking,
        primary_radius=neighbor_radius,
        secondary_radius=secondary_neighbor_radius,
    )
    delivered_ids = {str(item["doc_id"]) for item in selected}
    positive_ids = [str(item) for item in row.get("positive_doc_ids", [])]
    missing_positive_ids = sorted(set(positive_ids) - delivered_ids)
    if missing_positive_ids:
        raise ValueError(
            "blind BM25 plus neighbor expansion did not deliver the frozen human evidence: "
            + ", ".join(missing_positive_ids)
        )

    participants = sorted(
        {
            _actor_token(scope_id, str(item.get("speaker") or "unknown"))
            for item in selected
        }
    )
    case = {
        "schema_version": CASE_SCHEMA_VERSION,
        "case_id": "q0030-mujica-yumemita",
        "layer": "oracle_synthesis_diagnostic",
        "umo": scope_id,
        "cutoff_at": cutoff_at,
        "query": query,
        "authorized_participant_keys": participants,
        "fixture_provenance": {
            "messages_are_real_anonymized_group_chat": True,
            "query_is_human_reviewed_research_restating": True,
            "observed_online_chat_call": False,
            "retrieval_selection_used_gold": False,
        },
    }
    packet_messages = [
        {
            "source_key": str(item["doc_id"]),
            "umo": scope_id,
            "sent_at": int(item["sent_at"]),
            "sender_participant_key": _actor_token(
                scope_id, str(item.get("speaker") or "unknown")
            ),
            "speaker_label": str(item.get("speaker") or "unknown"),
            "actor_role": "anonymized_group_member",
            "plain_text": str(item.get("text") or ""),
        }
        for item in selected
    ]
    packet = {
        "schema_version": PACKET_SCHEMA_VERSION,
        "case_id": case["case_id"],
        "query": query,
        "query_scope_token": scope_id,
        "cutoff_at": cutoff_at,
        "diagnostic_type": "blind_bm25_neighbor_expansion_then_layered_reader",
        "end_to_end_retrieval_claim": True,
        "retrieval": {
            "backend": "bm25-char-bigram",
            "top_k": top_k,
            "primary_neighbor_radius": neighbor_radius,
            "secondary_neighbor_radius": secondary_neighbor_radius,
            "anchor_source_keys": ranking,
            "selection_input_sha256": _stable_hash(
                {"scope_id": scope_id, "cutoff_at": cutoff_at, "query": query}
            ),
            "gold_loaded_after_selection": True,
        },
        "evidence_policy": {
            "strictly_before_cutoff": True,
            "same_scope_only": True,
            "temporal_adjacency_is_explicit_reply_to": False,
            "anonymized_speakers_are_stable_accounts": False,
        },
        "episodes": episodes,
        "messages": packet_messages,
    }
    gold = {
        "schema_version": GOLD_SCHEMA_VERSION,
        "case_id": case["case_id"],
        "frozen_before_provider_run": True,
        "review_status": "human_approved_retrieval_evidence_pending_answer_review",
        "evidence_groups": {
            "mujica_yumemita_relation": {
                "required_any": positive_ids,
                "support": [],
            }
        },
        "required_semantics": [
            "群友回答没看过 Mujica 也可以看梦限大，并说两者其实没有什么关系。",
            "“Mujica 正统续作”是同一讨论中的群友评价，不应升级成作品官方关系。",
        ],
        "required_uncertainty": [
            "“正统续作”是群聊主观说法；“没什么关系”是对是否需要前作知识的直接回答。"
        ],
        "forbidden_conclusions": [
            "把群友的“正统续作”玩笑写成官方续作关系。",
            "声称没看过 Mujica 就不能看梦限大。",
        ],
        "source_annotation": {
            "benchmark_id": "q0030",
            "positive_basis": (row.get("provenance") or {}).get("positive_basis"),
            "human_review_dataset_fingerprint": (row.get("provenance") or {}).get(
                "human_review_dataset_fingerprint"
            ),
        },
    }

    _write_json(output_dir / "case.json", case)
    _write_json(output_dir / "evidence_packet.json", packet)
    _write_json(output_dir / "gold.json", gold)
    manifest = {
        "schema_version": "mr-memory.three-case-fixture.v1",
        "case_id": case["case_id"],
        "case_path": "case.json",
        "evidence_packet_path": "evidence_packet.json",
        "gold_path": "gold.json",
        "source_files": {
            "corpus": {
                "path": str(corpus_path.resolve()),
                "sha256": _file_hash(corpus_path),
            },
            "benchmark": {
                "path": str(benchmark_path.resolve()),
                "sha256": _file_hash(benchmark_path),
            },
        },
        "retrieval_audit": {
            "ranking": ranking,
            "delivered_source_count": len(packet_messages),
            "positive_source_count": len(positive_ids),
            "positive_sources_delivered": len(positive_ids) - len(missing_positive_ids),
            "selection_used_gold": False,
            "gold_opened_only_for_post_selection_coverage_audit": True,
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the q0030 real-group layered-memory fixture using blind BM25 "
            "selection followed by deterministic chronological neighbor expansion."
        )
    )
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--neighbor-radius", type=int, default=10)
    parser.add_argument("--secondary-neighbor-radius", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_q0030_fixture(
        benchmark_dir=args.benchmark_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        top_k=args.top_k,
        neighbor_radius=args.neighbor_radius,
        secondary_neighbor_radius=args.secondary_neighbor_radius,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
