from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mr_memory.plasticity import parse_graph_mutation
from mr_memory.storage import MemoryStorage
from scripts.plastic_semantics_ab_experiment import (
    _graph_snapshot,
    _load_history,
    _load_live_fixture,
    _write_reports,
)


def replay(args: argparse.Namespace) -> dict[str, Any]:
    result_path = Path(args.result).resolve()
    result = json.loads(result_path.read_text(encoding="utf-8-sig"))
    group_id = str(result["dataset"]["group_id"])
    umo = str(result["dataset"]["umo"])
    _, history, _ = _load_history(args.history_db, group_id=group_id)
    live = _load_live_fixture(args.live_fixture, group_id=group_id, umo=umo)
    source_index = {record.source_key: record for record in history}
    source_index.update({record.source_key: record for record in live.values()})

    database_path = result_path.parent / "negative-policy-replay.db"
    for suffix in ("", "-wal", "-shm"):
        Path(f"{database_path}{suffix}").unlink(missing_ok=True)
    storage = MemoryStorage(database_path)
    storage.bind_scope(umo=umo, platform_id="byy_official", group_id=group_id)

    tick_groups = [
        *result["good_girl"]["maintenance_ticks"],
        *result["arale"]["maintenance_ticks"],
    ]
    evidence_keys = {
        str(key)
        for tick in tick_groups
        for item in tick["committed"]
        for key in item["proposal"]["evidence_source_keys"]
    }
    missing = sorted(evidence_keys - set(source_index))
    if missing:
        raise ValueError(f"replay evidence is missing: {missing}")
    for key in sorted(
        evidence_keys,
        key=lambda item: (
            source_index[item].normalized.sent_at,
            item,
        ),
    ):
        storage.upsert_message(source_index[key].normalized)

    feedback_tick = result["arale"]["maintenance_ticks"][-1]
    earlier_ticks = tick_groups[:-1]
    for tick in earlier_ticks:
        allowed = set(str(key) for key in tick["evidence_source_keys"])
        for item in tick["committed"]:
            storage.apply_graph_mutation(
                umo=umo,
                mutation=parse_graph_mutation(item["proposal"]),
                model="policy-replay",
                allowed_evidence_keys=allowed,
            )
    before_rows = storage.query_plastic_associations(
        umo=umo,
        query=args.target_term,
        include_dormant=True,
        limit=100,
    )
    primary = next(
        row for row in before_rows if str(row["relation_key"]) == args.primary_relation
    )
    primary_edge_id = int(primary["id"])
    utility_before = float(primary["utility"])

    committed: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    allowed = set(str(key) for key in feedback_tick["evidence_source_keys"])
    for item in feedback_tick["committed"]:
        mutation = parse_graph_mutation(item["proposal"])
        try:
            value = storage.apply_graph_mutation(
                umo=umo,
                mutation=mutation,
                model="policy-replay",
                allowed_evidence_keys=allowed,
                allowed_negative_edge_ids=set(),
            )
            committed.append({"proposal": mutation.as_dict(), "result": value})
        except ValueError as exc:
            rejected.append(
                {"proposal": mutation.as_dict(), "error": str(exc)}
            )
    after_rows = storage.query_plastic_associations(
        umo=umo,
        query=args.target_term,
        include_dormant=True,
        limit=100,
    )
    primary_after = next(row for row in after_rows if int(row["id"]) == primary_edge_id)
    replay_result = {
        "proposed": len(feedback_tick["committed"]),
        "committed": len(committed),
        "rejected": len(rejected),
        "allowed_negative_edge_ids": [],
        "rejected_mutations": rejected,
        "utility_before_feedback": utility_before,
        "utility_after_feedback": float(primary_after["utility"]),
        "primary_edge_id": primary_edge_id,
        "verdict": (
            "旧系统没有激活 MR 路径，因此宿主拒绝负向归因；"
            "查询后发现的正确路径只能被强化，不能替旧回答背锅。"
        ),
    }
    result["negative_credit_policy_replay"] = replay_result
    result["arale"]["final_graph"] = _graph_snapshot(storage, umo=umo)
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_reports(result_path.parent, result)
    storage.close()
    for suffix in ("", "-wal", "-shm"):
        Path(f"{database_path}{suffix}").unlink(missing_ok=True)
    return replay_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay a feedback tick under activated-path negative credit policy."
    )
    parser.add_argument("--result", required=True)
    parser.add_argument("--history-db", required=True)
    parser.add_argument("--live-fixture", required=True)
    parser.add_argument("--target-term", default="阿拉蕾")
    parser.add_argument("--primary-relation", default="refers_to")
    return parser


def main() -> None:
    value = replay(build_parser().parse_args())
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
