from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sqlite3
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

DEFAULT_ONLINE_CUTOFF = "2026-08-06 19:21:33"
BOOTSTRAP_SEED = 20260817
BOOTSTRAP_SAMPLES = 20_000


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(item) for item in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "p95_linear": None,
            "max": None,
        }
    numeric = [float(item) for item in values]
    return {
        "n": len(numeric),
        "mean": statistics.fmean(numeric),
        "median": statistics.median(numeric),
        "p95_linear": _quantile(numeric, 0.95),
        "max": max(numeric),
    }


def _valid_json(raw: object) -> dict[str, Any]:
    try:
        value = json.loads(str(raw or "{}"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _online_profile(path: Path, cutoff_utc: str) -> dict[str, Any]:
    wal_path = Path(f"{path.resolve()}-wal")
    if wal_path.exists() and wal_path.stat().st_size > 0:
        raise ValueError(
            "online aggregate input has a non-empty WAL; freeze a logical SQLite "
            "snapshot before hashing or profiling it"
        )
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        foreign_key_rows = connection.execute("PRAGMA foreign_key_check").fetchall()
        rows = connection.execute(
            """
            SELECT run_id, status, metadata_json, result_json,
                   started_at, finished_at,
                   MAX(0.0, (julianday(finished_at) - julianday(started_at))
                       * 86400.0) AS wall_seconds
            FROM experiment_runs
            WHERE experiment_type = 'runtime_reconstruction'
              AND started_at >= ?
            ORDER BY started_at
            """,
            (cutoff_utc,),
        ).fetchall()
        run_ids = [str(row["run_id"]) for row in rows]
        usage_by_run: dict[str, dict[str, float]] = {
            run_id: {
                "calls": 0,
                "input_other": 0,
                "input_cached": 0,
                "output": 0,
                "total": 0,
                "elapsed_ms": 0.0,
            }
            for run_id in run_ids
        }
        if run_ids:
            placeholders = ",".join("?" for _ in run_ids)
            usage_rows = connection.execute(
                f"""
                SELECT run_id, COUNT(*) AS calls,
                       SUM(input_other) AS input_other,
                       SUM(input_cached) AS input_cached,
                       SUM(output) AS output,
                       SUM(input_other + input_cached + output) AS total,
                       SUM(elapsed_ms) AS elapsed_ms
                FROM llm_usage_events
                WHERE run_id IN ({placeholders})
                  AND phase = 'reconstruction'
                GROUP BY run_id
                """,
                run_ids,
            ).fetchall()
            for row in usage_rows:
                usage_by_run[str(row["run_id"])] = {
                    "calls": int(row["calls"] or 0),
                    "input_other": int(row["input_other"] or 0),
                    "input_cached": int(row["input_cached"] or 0),
                    "output": int(row["output"] or 0),
                    "total": int(row["total"] or 0),
                    "elapsed_ms": float(row["elapsed_ms"] or 0.0),
                }

        status_counts: Counter[str] = Counter()
        outcome_counts: Counter[str] = Counter()
        path_counts: Counter[str] = Counter()
        error_counts: Counter[str] = Counter()
        repair_runs = 0
        tool_steps: list[int] = []
        wall_seconds: list[float] = []
        tokens_measured_all_runs: list[float] = []
        tokens_per_run: list[float] = []
        outcome_values: dict[str, dict[str, list[float]]] = {}
        totals = Counter()
        for row in rows:
            status = str(row["status"] or "").upper()
            result = _valid_json(row["result_json"])
            metadata = _valid_json(row["metadata_json"])
            path_name = str(result.get("path") or metadata.get("path") or "unknown")
            if status == "COMPLETED":
                outcome = "none" if bool(result.get("no_relevant_memory")) else "brief"
            else:
                outcome = "failed"
            status_counts[status] += 1
            outcome_counts[outcome] += 1
            path_counts[path_name] += 1
            if result.get("error_type"):
                error_counts[str(result["error_type"])] += 1
            if bool(result.get("repair_attempted")):
                repair_runs += 1
            steps = int(result.get("tool_steps") or 0)
            tool_steps.append(steps)
            wall = float(row["wall_seconds"] or 0.0)
            wall_seconds.append(wall)
            usage = usage_by_run[str(row["run_id"])]
            token_total = float(usage["total"])
            tokens_measured_all_runs.append(token_total)
            if token_total > 0:
                tokens_per_run.append(token_total)
            for key in ("calls", "input_other", "input_cached", "output", "total"):
                totals[key] += int(usage[key])
            bucket = outcome_values.setdefault(
                outcome,
                {
                    "wall_seconds": [],
                    "tokens_measured_all_runs": [],
                    "tokens": [],
                },
            )
            bucket["wall_seconds"].append(wall)
            bucket["tokens_measured_all_runs"].append(token_total)
            if token_total > 0:
                bucket["tokens"].append(token_total)

        return {
            "source": {
                "artifact": path.name,
                "sha256": _file_sha256(path),
                "cutoff_utc": cutoff_utc,
                "database_scope_count": int(
                    connection.execute(
                        "SELECT COUNT(DISTINCT umo) FROM messages"
                    ).fetchone()[0]
                ),
            },
            "quality_checks": {
                "integrity_check": integrity,
                "foreign_key_violations": len(foreign_key_rows),
                "runs": len(rows),
            },
            "status_counts": dict(sorted(status_counts.items())),
            "outcome_counts": dict(sorted(outcome_counts.items())),
            "path_counts": dict(sorted(path_counts.items())),
            "error_counts": dict(sorted(error_counts.items())),
            "repair_runs": repair_runs,
            "tool_steps": _summary(tool_steps),
            "wall_seconds": _summary(wall_seconds),
            "tokens_measured_all_runs": _summary(tokens_measured_all_runs),
            "tokens_per_run_with_provider_usage": _summary(tokens_per_run),
            "usage_totals": dict(totals),
            "outcomes": {
                key: {
                    "wall_seconds": _summary(value["wall_seconds"]),
                    "tokens_measured_all_runs": _summary(
                        value["tokens_measured_all_runs"]
                    ),
                    "tokens_with_provider_usage": _summary(value["tokens"]),
                }
                for key, value in sorted(outcome_values.items())
            },
            "interpretation_boundary": (
                "These rows have no human semantic labels. brief/none/path describe "
                "system behavior, not correctness; timeout rows without provider "
                "usage make the ledger a measured total rather than a billing upper bound."
            ),
        }
    finally:
        connection.close()


def _model_key(row: dict[str, Any]) -> str:
    model = row.get("model")
    if isinstance(model, dict) and model.get("key"):
        return str(model["key"])
    return str(row.get("backend") or "unknown")


def _metric_by_query(row: dict[str, Any], metric: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for item in row.get("per_query", []):
        if not isinstance(item, dict) or not item.get("id"):
            continue
        if metric == "mrr_at_20":
            value = item.get("rr_at_20", 0.0)
        elif metric == "mean_evidence_recall_at_20":
            evidence_count = int(item.get("evidence_count") or 0)
            found = int(item.get("evidence_found_at_20") or 0)
            value = found / evidence_count if evidence_count else 0.0
        else:
            raise ValueError(f"unsupported paired metric: {metric}")
        values[str(item["id"])] = float(value)
    return values


def _paired_bootstrap(
    left: dict[str, float],
    right: dict[str, float],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    ids = sorted(set(left) & set(right))
    if not ids:
        raise ValueError("paired bootstrap has no shared query ids")
    differences = [left[item] - right[item] for item in ids]
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        estimates.append(
            statistics.fmean(differences[rng.randrange(len(differences))] for _ in ids)
        )
    return {
        "n_queries": len(ids),
        "samples": samples,
        "seed": seed,
        "mean_difference": statistics.fmean(differences),
        "ci95_low": float(_quantile(estimates, 0.025)),
        "ci95_high": float(_quantile(estimates, 0.975)),
    }


def _retrieval_profile(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    results = [item for item in value.get("results", []) if item.get("status") == "ok"]
    by_key = {_model_key(item): item for item in results}
    positive_annotation_counts = {
        sum(int(item.get("evidence_count") or 0) for item in row.get("per_query", []))
        for row in results
    }
    if len(positive_annotation_counts) != 1:
        raise ValueError(
            "retrieval arms disagree on the number of positive evidence annotations"
        )
    positive_evidence_annotations = next(iter(positive_annotation_counts), 0)
    models: list[dict[str, Any]] = []
    for key, row in by_key.items():
        metrics = row.get("metrics", {})
        models.append(
            {
                "key": key,
                "mrr_at_20": metrics.get("mrr_at_20"),
                "hit_at_20": (metrics.get("hit_rate") or {}).get("@20"),
                "mean_evidence_recall_at_20": (
                    metrics.get("mean_evidence_recall") or {}
                ).get("@20"),
                "all_evidence_at_20": metrics.get("all_evidence_at_20"),
            }
        )

    comparisons: list[dict[str, Any]] = []
    for left_key, right_key in (
        ("harrier-270m", "minilm-l12-v2"),
        ("harrier-270m", "bm25-char-bigram"),
        ("harrier-270m-group-memory", "harrier-270m"),
    ):
        left = by_key[left_key]
        right = by_key[right_key]
        for metric in ("mrr_at_20", "mean_evidence_recall_at_20"):
            comparisons.append(
                {
                    "left": left_key,
                    "right": right_key,
                    "metric": metric,
                    **_paired_bootstrap(
                        _metric_by_query(left, metric),
                        _metric_by_query(right, metric),
                    ),
                }
            )
    return {
        "source": {
            "artifact": path.name,
            "matrix_sha256": _file_sha256(path),
            "benchmark_sha256": value.get("benchmark", {}).get("sha256"),
            "corpus_sha256": value.get("benchmark", {}).get("corpus_sha256"),
            "queries": value.get("benchmark", {}).get("queries"),
            "corpus_documents": value.get("benchmark", {}).get("corpus_documents"),
            "positive_evidence_annotations": positive_evidence_annotations,
        },
        "models": models,
        "paired_bootstrap": comparisons,
        "interpretation_boundary": (
            "The gold set is one-group, reply-chain-derived and evidence-oriented. "
            "Lexical decoys are not verified negatives, so these metrics measure "
            "retrieval of annotated positives, not end-to-end semantic understanding "
            "or precision over all relevant documents."
        ),
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    generated = datetime.now(timezone.utc).isoformat()
    result = {
        "schema_version": 1,
        "generated_at": generated,
        "privacy": (
            "Aggregate-only artifact. It contains no group message text, account id, "
            "group id, source key, credential, or provider configuration."
        ),
        "online_runtime": _online_profile(Path(args.online_db), args.online_cutoff_utc),
        "candidate_retrieval": _retrieval_profile(Path(args.retrieval_matrix)),
    }
    _write_json(Path(args.output), result)
    return {
        "output": str(Path(args.output).resolve()),
        "online_runs": result["online_runtime"]["quality_checks"]["runs"],
        "retrieval_models": len(result["candidate_retrieval"]["models"]),
        "generated_at": generated,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a privacy-safe aggregate for the MR Memory runtime study."
    )
    parser.add_argument("--online-db", required=True)
    parser.add_argument("--retrieval-matrix", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--online-cutoff-utc", default=DEFAULT_ONLINE_CUTOFF)
    return parser


def main() -> None:
    print(json.dumps(analyze(_parser().parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
