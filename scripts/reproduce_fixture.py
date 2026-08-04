from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from mr_memory.distillation import parse_distillation_response
from mr_memory.embedding import HashEmbeddingBackend
from mr_memory.replay import iter_jsonl, replay_records
from mr_memory.service import MemoryService
from mr_memory.storage import MemoryStorage


async def reproduce(args: argparse.Namespace) -> dict[str, object]:
    storage = MemoryStorage(args.database)
    service = MemoryService(storage)
    try:
        inserted, updated = replay_records(iter_jsonl(args.messages), storage)
        messages = storage.search_messages(
            umo=args.umo,
            limit=args.message_limit,
        )
        response = Path(args.distillation).read_text(encoding="utf-8")
        batch = parse_distillation_response(response, messages)
        backend = HashEmbeddingBackend(dimensions=args.dimensions)
        persisted, indexed = await service.apply_distillation(
            batch,
            extractor_version="offline-reproduction-v1",
            embedding_backend=backend,
        )
        candidates = await service.initialize_candidates(
            umo=args.umo,
            query=args.query,
            embedding_backend=backend,
            limit=args.top_k,
        )
        return {
            "messages_inserted": inserted,
            "messages_updated": updated,
            "episodes": list(persisted.episode_ids),
            "semantic_memories": list(persisted.semantic_ids),
            "topics": list(persisted.topic_ids),
            "embedded_documents": indexed,
            "initial_active_set": candidates,
        }
    finally:
        await service.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the MRAgent construction and candidate-seeding path offline."
    )
    parser.add_argument("--messages", type=Path, required=True)
    parser.add_argument("--distillation", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--umo", required=True)
    parser.add_argument("--query", default="最后选择了哪个方案？")
    parser.add_argument("--message-limit", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--dimensions", type=int, default=256)
    args = parser.parse_args()
    result = asyncio.run(reproduce(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
