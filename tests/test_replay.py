from __future__ import annotations

import unittest
import uuid
from pathlib import Path

from mr_memory.replay import iter_jsonl, replay_records
from mr_memory.storage import MemoryStorage


class ReplayTests(unittest.TestCase):
    def test_sample_fixture_replays_idempotently(self) -> None:
        fixture = (
            Path(__file__).parents[1]
            / "dev"
            / "fixtures"
            / "sample_messages.jsonl"
        )
        test_root = Path.cwd() / ".dev" / "test-tmp"
        test_root.mkdir(parents=True, exist_ok=True)
        database_path = test_root / f"{uuid.uuid4().hex}.db"
        storage = MemoryStorage(database_path)
        try:
            self.assertEqual(replay_records(iter_jsonl(fixture), storage), (5, 0))
            self.assertEqual(replay_records(iter_jsonl(fixture), storage), (0, 5))
            self.assertEqual(storage.count_messages(), 5)
            results = storage.search_messages(
                umo="shadow:GroupMessage:group-a",
                query="方案 B",
            )
            self.assertEqual(len(results), 2)
        finally:
            storage.close()
            for suffix in ("", "-wal", "-shm"):
                Path(f"{database_path}{suffix}").unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
