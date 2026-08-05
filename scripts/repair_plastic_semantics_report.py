from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.masked_ab_experiment import _provider_config
from scripts.plastic_semantics_ab_experiment import (
    UsageLedger,
    _context_window,
    _evaluate,
    _load_history,
    _surface_answer,
    _term_hits,
    _write_reports,
)


def _totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "calls": len(rows),
        "input_other": sum(int(row["input_other"]) for row in rows),
        "input_cached": sum(int(row["input_cached"]) for row in rows),
        "output": sum(int(row["output"]) for row in rows),
        "total": sum(int(row["total"]) for row in rows),
        "elapsed_ms": round(sum(float(row["elapsed_ms"]) for row in rows), 3),
    }


def repair(args: argparse.Namespace) -> dict[str, Any]:
    result_path = Path(args.result).resolve()
    result = json.loads(result_path.read_text(encoding="utf-8-sig"))
    if not isinstance(result, dict):
        raise ValueError("result must be an object")
    group_id = str(result["dataset"]["group_id"])
    _, history, _ = _load_history(args.history_db, group_id=group_id)
    cutoff_at = int(result["dataset"]["target_cutoff"])
    arale_hits = _term_hits(history, args.arale_term, before_sent_at=cutoff_at)
    arale_evidence = _context_window(
        history,
        arale_hits,
        radius=18,
        before_sent_at=cutoff_at,
        max_records=60,
    )

    surface_client, surface_model, surface_extra = _provider_config(
        args.config, args.surface_provider_id
    )
    surface_extra = {**surface_extra, "thinking": {"type": "disabled"}}
    subconscious_client, subconscious_model, subconscious_extra = _provider_config(
        args.config, args.subconscious_provider_id
    )
    subconscious_extra = {
        **subconscious_extra,
        "thinking": {"type": "disabled"},
    }
    output_dir = result_path.parent
    ledger = UsageLedger(output_dir / "usage_live.jsonl")
    repaired_labels = set(args.arm)
    for arm in result["arale"]["arms"]:
        if str(arm["arm"]) not in repaired_labels:
            continue
        reconstruction = arm.get("reconstruction") or {}
        brief = reconstruction.get("brief")
        if not isinstance(brief, dict):
            raise ValueError(f"arm has no reconstruction brief: {arm['arm']}")
        arm["answer"] = _surface_answer(
            case="arale",
            phase=f"repair:{arm['arm']}",
            query=str(result["arale"]["query"]),
            brief=brief,
            client=surface_client,
            provider_id=args.surface_provider_id,
            model=surface_model,
            extra_body=surface_extra,
            ledger=ledger,
        )
    result["arale"]["evaluation"] = _evaluate(
        case="arale",
        rubric=[
            "Resolve 阿拉蕾 as the group-local 梦限大-related referent.",
            "Use grounded traits such as noisy/cute, very fast speech, and awkwardness where supported.",
            "Interpret 挺点 cautiously as likely asking for appealing/萌 points rather than inventing a new term.",
            "Do not substitute Digimon, Dr. Slump, or another famous homonym.",
            "Keep claims traceable to pre-query group evidence.",
        ],
        evidence=arale_evidence,
        arms=result["arale"]["arms"],
        client=subconscious_client,
        provider_id=args.subconscious_provider_id,
        model=subconscious_model,
        extra_body=subconscious_extra,
        ledger=ledger,
    )
    result.setdefault("repairs", []).append(
        {
            "kind": "surface_answer_completion",
            "arms": sorted(repaired_labels),
            "reason": "Original 600-token output cap truncated visible answers.",
            "new_calls": len(ledger.rows),
        }
    )
    combined_calls = [*result["usage"]["calls"], *ledger.rows]
    result["usage"] = {"calls": combined_calls, "totals": _totals(combined_calls)}
    _write_reports(output_dir, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repair truncated surface arms without rebuilding the graph."
    )
    parser.add_argument("--result", required=True)
    parser.add_argument("--history-db", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--arm",
        action="append",
        default=["B1 查询前图记忆", "B2 负反馈写图后"],
    )
    parser.add_argument("--arale-term", default="阿拉蕾")
    parser.add_argument(
        "--subconscious-provider-id", default="deepseek/deepseek-v4-flash"
    )
    parser.add_argument(
        "--surface-provider-id", default="openai/gemini-3.5-flash"
    )
    return parser


def main() -> None:
    result = repair(build_parser().parse_args())
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "arale_arms": [
                    {"arm": arm["arm"], "answer": arm["answer"]}
                    for arm in result["arale"]["arms"]
                ],
                "evaluation": result["arale"]["evaluation"],
                "usage": result["usage"]["totals"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
