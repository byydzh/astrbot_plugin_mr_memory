from __future__ import annotations

import argparse
from pathlib import Path

from mr_memory.replay import iter_jsonl, replay_records
from mr_memory.storage import MemoryStorage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay normalized group messages into an isolated MR Memory database."
    )
    parser.add_argument("--input", required=True, type=Path, help="JSONL fixture")
    parser.add_argument("--database", required=True, type=Path, help="SQLite output")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    storage = MemoryStorage(args.database)
    try:
        inserted, updated = replay_records(iter_jsonl(args.input), storage)
        total = storage.count_messages()
    finally:
        storage.close()
    print(f"replay complete: inserted={inserted} updated={updated} total={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
