from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .models import NormalizedMessage
from .storage import MemoryStorage


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} must contain a JSON object")
            yield value


def replay_records(
    records: Iterable[dict[str, Any]],
    storage: MemoryStorage,
) -> tuple[int, int]:
    inserted = 0
    updated = 0
    for record in records:
        message = NormalizedMessage.from_mapping(record)
        if storage.upsert_message(message):
            inserted += 1
        else:
            updated += 1
    return inserted, updated
