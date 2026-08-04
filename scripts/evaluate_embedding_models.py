from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .evaluate_retrieval_benchmark import BM25Index
    from .materialize_retrieval_benchmark import read_jsonl
except ImportError:  # Direct script execution.
    from evaluate_retrieval_benchmark import BM25Index
    from materialize_retrieval_benchmark import read_jsonl


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    query_prefix: str = ""
    query_prompt_name: str = ""
    batch_size: int = 256


MODEL_SPECS = {
    spec.key: spec
    for spec in (
        ModelSpec(
            "minilm-l12-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ),
        ModelSpec(
            "bge-small-zh-v1.5",
            "BAAI/bge-small-zh-v1.5",
            query_prefix="为这个句子生成表示以用于检索相关文章：",
        ),
        ModelSpec(
            "bge-small-zh-v1.5-no-instruction",
            "BAAI/bge-small-zh-v1.5",
        ),
        ModelSpec(
            "granite-107m-r1",
            "ibm-granite/granite-embedding-107m-multilingual",
        ),
        ModelSpec(
            "granite-97m-r2",
            "ibm-granite/granite-embedding-97m-multilingual-r2",
        ),
        ModelSpec(
            "granite-311m-r2",
            "ibm-granite/granite-embedding-311m-multilingual-r2",
            batch_size=192,
        ),
        ModelSpec(
            "harrier-270m",
            "microsoft/harrier-oss-v1-270m",
            query_prompt_name="web_search_query",
            batch_size=192,
        ),
        ModelSpec(
            "harrier-270m-group-memory",
            "microsoft/harrier-oss-v1-270m",
            query_prefix=(
                "Instruct: Given a group-chat memory query, retrieve relevant prior "
                "messages that answer the query\nQuery: "
            ),
            batch_size=192,
        ),
    )
}

CUTOFFS = (1, 5, 10, 20)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _model_cache_size(cache_dir: Path, model_id: str) -> int:
    slug = "models--" + model_id.replace("/", "--")
    matches = [path for path in cache_dir.rglob(slug) if path.is_dir()]
    return sum(_directory_size(path) for path in matches)


def _document_text(document: dict[str, Any]) -> str:
    speaker = str(document.get("speaker") or "").strip()
    text = str(document.get("text") or "").strip()
    return f"{speaker}: {text}" if speaker else text


def _rank_dense(
    *,
    document_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    eligible_indices: np.ndarray,
    document_ids: np.ndarray,
    limit: int,
) -> tuple[list[str], list[float]]:
    if eligible_indices.size == 0:
        return [], []
    scores = document_embeddings[eligible_indices] @ query_embedding
    take = min(limit, scores.shape[0])
    if take == scores.shape[0]:
        candidates = np.arange(scores.shape[0])
    else:
        candidates = np.argpartition(scores, -take)[-take:]
    order = candidates[np.lexsort((eligible_indices[candidates], -scores[candidates]))]
    selected = eligible_indices[order]
    return document_ids[selected].tolist(), scores[order].astype(float).tolist()


def _score_rankings(
    benchmark: list[dict[str, Any]],
    rankings: dict[str, list[str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    hit_totals = collections.Counter({cutoff: 0.0 for cutoff in CUTOFFS})
    recall_totals = collections.Counter({cutoff: 0.0 for cutoff in CUTOFFS})
    reciprocal_rank = 0.0
    all_evidence_at_20 = 0
    misses: list[str] = []
    per_query: list[dict[str, Any]] = []

    for item in benchmark:
        item_id = str(item["id"])
        ranking = rankings[item_id]
        positives = {str(value) for value in item["positive_doc_ids"]}
        positive_ranks = [
            index + 1 for index, doc_id in enumerate(ranking) if doc_id in positives
        ]
        first_rank = min(positive_ranks) if positive_ranks else None
        rr_at_20 = 1.0 / first_rank if first_rank is not None and first_rank <= 20 else 0.0
        reciprocal_rank += rr_at_20
        if first_rank is None:
            misses.append(item_id)
        for cutoff in CUTOFFS:
            found = positives & set(ranking[:cutoff])
            hit_totals[cutoff] += float(bool(found))
            recall_totals[cutoff] += len(found) / len(positives)
        complete = positives.issubset(set(ranking[:20]))
        all_evidence_at_20 += int(complete)
        per_query.append(
            {
                "id": item_id,
                "first_positive_rank": first_rank,
                "rr_at_20": round(rr_at_20, 6),
                "evidence_count": len(positives),
                "evidence_found_at_20": len(positives & set(ranking[:20])),
                "all_evidence_at_20": complete,
                "memory_type": str(item.get("memory_type") or ""),
                "epistemic": str(item.get("epistemic") or ""),
            }
        )

    count = len(benchmark)
    metrics = {
        "queries": count,
        "mrr_at_20": round(reciprocal_rank / count, 6),
        "hit_rate": {
            f"@{cutoff}": round(hit_totals[cutoff] / count, 6) for cutoff in CUTOFFS
        },
        "mean_evidence_recall": {
            f"@{cutoff}": round(recall_totals[cutoff] / count, 6)
            for cutoff in CUTOFFS
        },
        "all_evidence_at_20": round(all_evidence_at_20 / count, 6),
        "misses_at_20": misses,
    }
    return metrics, per_query


def evaluate_bm25_detailed(
    corpus: list[dict[str, Any]], benchmark: list[dict[str, Any]]
) -> dict[str, Any]:
    started = time.perf_counter()
    index = BM25Index(corpus)
    rankings = {
        str(item["id"]): index.rank(
            str(item["query"]),
            scope_id=str(item["scope_id"]),
            before=int(item["query_time"]),
            limit=max(CUTOFFS),
        )
        for item in benchmark
    }
    metrics, per_query = _score_rankings(benchmark, rankings)
    return {
        "status": "ok",
        "backend": "bm25-char-bigram",
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "metrics": metrics,
        "per_query": per_query,
    }


def _load_sentence_transformer(spec: ModelSpec, *, device: str, cache_dir: Path) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        spec.model_id,
        device=device,
        cache_folder=str(cache_dir),
        trust_remote_code=False,
    )


def evaluate_dense_model(
    *,
    spec: ModelSpec,
    corpus: list[dict[str, Any]],
    benchmark: list[dict[str, Any]],
    cache_dir: Path,
    device: str,
    smoke: bool,
) -> dict[str, Any]:
    import torch

    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    load_started = time.perf_counter()
    model = _load_sentence_transformer(spec, device=device, cache_dir=cache_dir)
    load_seconds = time.perf_counter() - load_started
    parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    parameter_bytes = int(
        sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    )
    dimension_getter = getattr(model, "get_embedding_dimension", None)
    if dimension_getter is None:
        dimension_getter = model.get_sentence_embedding_dimension
    embedding_dimension = int(dimension_getter() or 0)
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})

    if smoke:
        sample_documents = [_document_text(item) for item in corpus[:128]]
        sample_queries = [str(item["query"]) for item in benchmark[:4]]
        if spec.query_prefix:
            sample_queries = [spec.query_prefix + query for query in sample_queries]
        query_kwargs: dict[str, Any] = {}
        if spec.query_prompt_name:
            query_kwargs["prompt_name"] = spec.query_prompt_name
        document_embeddings = model.encode(
            sample_documents,
            batch_size=min(spec.batch_size, 128),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        query_embeddings = model.encode(
            sample_queries,
            batch_size=4,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            **query_kwargs,
        )
        result = {
            "status": "smoke_ok",
            "model": asdict(spec),
            "load_seconds": round(load_seconds, 3),
            "parameters": parameters,
            "embedding_dimension": embedding_dimension,
            "document_shape": list(document_embeddings.shape),
            "query_shape": list(query_embeddings.shape),
        }
        del model, document_embeddings, query_embeddings
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        return result

    document_texts = [_document_text(item) for item in corpus]
    query_texts = [str(item["query"]) for item in benchmark]
    if spec.query_prefix:
        query_texts = [spec.query_prefix + query for query in query_texts]

    encode_started = time.perf_counter()
    document_embeddings = model.encode(
        document_texts,
        batch_size=spec.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32, copy=False)
    document_encode_seconds = time.perf_counter() - encode_started

    query_kwargs: dict[str, Any] = {}
    if spec.query_prompt_name:
        query_kwargs["prompt_name"] = spec.query_prompt_name
    query_started = time.perf_counter()
    query_embeddings = model.encode(
        query_texts,
        batch_size=min(spec.batch_size, max(1, len(query_texts))),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
        **query_kwargs,
    ).astype(np.float32, copy=False)
    query_encode_seconds = time.perf_counter() - query_started

    if document_embeddings.shape != (len(corpus), embedding_dimension):
        raise ValueError(f"{spec.key}: unexpected document embedding shape")
    if query_embeddings.shape != (len(benchmark), embedding_dimension):
        raise ValueError(f"{spec.key}: unexpected query embedding shape")

    document_ids = np.asarray([str(item["doc_id"]) for item in corpus], dtype=object)
    document_scopes = np.asarray([str(item["scope_id"]) for item in corpus], dtype=object)
    document_times = np.asarray([int(item["sent_at"]) for item in corpus], dtype=np.int64)
    rankings: dict[str, list[str]] = {}
    rank_started = time.perf_counter()
    for index, item in enumerate(benchmark):
        eligible = np.flatnonzero(
            (document_scopes == str(item["scope_id"]))
            & (document_times < int(item["query_time"]))
        )
        eligible_ids = set(document_ids[eligible].tolist())
        missing = set(str(value) for value in item["positive_doc_ids"]) - eligible_ids
        if missing:
            raise ValueError(f"{item['id']}: positive evidence is ineligible: {sorted(missing)}")
        ranking, _scores = _rank_dense(
            document_embeddings=document_embeddings,
            query_embedding=query_embeddings[index],
            eligible_indices=eligible,
            document_ids=document_ids,
            limit=max(CUTOFFS),
        )
        rankings[str(item["id"])] = ranking
    rank_seconds = time.perf_counter() - rank_started
    metrics, per_query = _score_rankings(benchmark, rankings)

    peak_gpu_bytes = 0
    if device.startswith("cuda"):
        peak_gpu_bytes = int(torch.cuda.max_memory_allocated())
    result = {
        "status": "ok",
        "backend": "sentence-transformers",
        "model": asdict(spec),
        "parameters": parameters,
        "parameter_bytes_in_memory": parameter_bytes,
        "parameter_dtypes": parameter_dtypes,
        "embedding_dimension": embedding_dimension,
        "max_sequence_length": int(model.max_seq_length),
        "cache_bytes": _model_cache_size(cache_dir, spec.model_id),
        "load_seconds": round(load_seconds, 6),
        "document_encode_seconds": round(document_encode_seconds, 6),
        "documents_per_second": round(len(corpus) / document_encode_seconds, 3),
        "query_encode_ms_per_query": round(
            query_encode_seconds * 1000 / len(benchmark), 3
        ),
        "ranking_ms_per_query": round(rank_seconds * 1000 / len(benchmark), 3),
        "peak_gpu_memory_bytes": peak_gpu_bytes,
        "metrics": metrics,
        "per_query": per_query,
    }

    del model, document_embeddings, query_embeddings
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def _runtime_metadata(device: str) -> dict[str, Any]:
    import sentence_transformers
    import torch
    import transformers

    gpu = None
    if device.startswith("cuda") and torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(torch.device(device))
        gpu = {
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "cuda_runtime": torch.version.cuda,
        }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "device": device,
        "gpu": gpu,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate local embedding models on the private group retrieval gold set."
    )
    parser.add_argument("benchmark_dir", type=Path)
    parser.add_argument(
        "--benchmark-file",
        type=Path,
        help="Defaults to benchmark_gold_final.jsonl.",
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=tuple(MODEL_SPECS),
        help="Repeat to select models; defaults to the full matrix.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--refresh-bm25", action="store_true")
    args = parser.parse_args()

    directory = args.benchmark_dir.resolve()
    benchmark_path = (args.benchmark_file or directory / "benchmark_gold_final.jsonl").resolve()
    cache_dir = (args.cache_dir or directory / "model_cache").resolve()
    output_path = (
        args.output
        or directory / ("embedding_smoke.json" if args.smoke else "embedding_matrix.json")
    ).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = directory / "corpus.jsonl"
    corpus = read_jsonl(corpus_path)
    benchmark = read_jsonl(benchmark_path)
    if not corpus or not benchmark:
        raise ValueError("corpus and benchmark must be non-empty")
    if any(item.get("split") != "gold" for item in benchmark):
        raise ValueError("embedding evaluation accepts only finalized gold items")

    selected_keys = args.model or list(MODEL_SPECS)
    payload: dict[str, Any] = {
        "format_version": 1,
        "status": "running",
        "private_local_evaluation": True,
        "benchmark": {
            "path": str(benchmark_path),
            "sha256": _sha256(benchmark_path),
            "queries": len(benchmark),
            "corpus_path": str(corpus_path),
            "corpus_sha256": _sha256(corpus_path),
            "corpus_documents": len(corpus),
            "document_representation": "speaker + ': ' + text",
            "scope_filter_required": True,
            "candidate_time_rule": "document.sent_at < query.query_time",
        },
        "runtime": _runtime_metadata(args.device),
        "cache_dir": str(cache_dir),
        "cache_bytes_before": _directory_size(cache_dir),
        "results": [],
    }
    if args.resume and output_path.exists():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        payload["results"] = list(previous.get("results") or [])
    completed = {
        str(item.get("model", {}).get("key"))
        for item in payload["results"]
        if item.get("status") in {"ok", "smoke_ok"}
    }

    has_bm25 = any(item.get("backend") == "bm25-char-bigram" for item in payload["results"])
    if not args.smoke and (args.refresh_bm25 or not has_bm25):
        payload["results"] = [
            item for item in payload["results"] if item.get("backend") != "bm25-char-bigram"
        ]
        bm25 = evaluate_bm25_detailed(corpus, benchmark)
        payload["results"].append(bm25)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    for key in selected_keys:
        if key in completed:
            continue
        spec = MODEL_SPECS[key]
        try:
            result = evaluate_dense_model(
                spec=spec,
                corpus=corpus,
                benchmark=benchmark,
                cache_dir=cache_dir,
                device=args.device,
                smoke=args.smoke,
            )
        except Exception as exc:
            result = {
                "status": "error",
                "model": asdict(spec),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        payload["results"] = [
            item
            for item in payload["results"]
            if str(item.get("model", {}).get("key")) != key
        ]
        payload["results"].append(result)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    for item in payload["results"]:
        model_id = str((item.get("model") or {}).get("model_id") or "")
        if model_id and item.get("status") in {"ok", "smoke_ok"}:
            item["cache_bytes"] = _model_cache_size(cache_dir, model_id)
    payload["status"] = "complete"
    payload["cache_bytes_after"] = _directory_size(cache_dir)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    concise = [
        {
            "backend": item.get("backend") or item.get("model", {}).get("key"),
            "status": item.get("status"),
            "mrr_at_20": (item.get("metrics") or item).get("mrr_at_20"),
            "hit_at_10": ((item.get("metrics") or item).get("hit_rate") or {}).get("@10"),
        }
        for item in payload["results"]
    ]
    print(json.dumps({"output": str(output_path), "results": concise}, ensure_ascii=False))


if __name__ == "__main__":
    main()
