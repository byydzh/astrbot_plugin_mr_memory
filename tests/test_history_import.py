from __future__ import annotations

import json
import sqlite3
import unittest
import uuid
from pathlib import Path

from mr_memory.history_import import AngelEyeHistorySource, angel_eye_scope
from mr_memory.storage import MemoryStorage


class AngelEyeHistoryImportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path.cwd() / ".dev" / "test-tmp" / uuid.uuid4().hex
        self.root.mkdir(parents=True, exist_ok=True)
        self.source_path = self.root / "qq_history_cache.db"
        connection = sqlite3.connect(self.source_path)
        connection.executescript(
            """
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                message_seq INTEGER,
                time INTEGER NOT NULL,
                user_id TEXT,
                nickname TEXT,
                search_text TEXT,
                raw_json TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                UNIQUE(group_id, message_id)
            );
            CREATE TABLE sync_state (
                group_id TEXT PRIMARY KEY,
                oldest_seq INTEGER,
                covered_from INTEGER,
                covered_to INTEGER,
                history_exhausted INTEGER DEFAULT 0,
                last_sync_at INTEGER
            );
            """
        )
        first = {
            "post_type": "message",
            "message_type": "group",
            "message_id": 11,
            "time": 100,
            "group_id": 1000,
            "user_id": 42,
            "self_id": 999,
            "sender": {"user_id": 42, "nickname": "昵称", "card": "群名片"},
            "message": [
                {"type": "reply", "data": {"id": "10"}},
                {"type": "at", "data": {"qq": "7", "name": "对象"}},
                {"type": "text", "data": {"text": " 这张图 "}},
                {
                    "type": "image",
                    "data": {
                        "file": "meme.jpg",
                        "url": "https://private.example/image?token=secret",
                    },
                },
            ],
        }
        second = {
            "post_type": "message_sent",
            "message_type": "group",
            "message_id": 12,
            "time": 101,
            "group_id": 1000,
            "user_id": 999,
            "self_id": 999,
            "sender": {"user_id": 999, "nickname": "Bot", "card": ""},
            "message": [{"type": "text", "data": {"text": "收到"}}],
        }
        connection.executemany(
            """
            INSERT INTO messages(
                group_id, message_id, time, user_id, nickname,
                search_text, raw_json, created_at
            ) VALUES (?, ?, ?, ?, ?, '', ?, 200)
            """,
            [
                ("1000", "11", 100, "42", "昵称", json.dumps(first)),
                ("1000", "12", 101, "999", "Bot", json.dumps(second)),
                ("2000", "21", 102, "8", "坏消息", "not-json"),
            ],
        )
        connection.executemany(
            "INSERT INTO sync_state(group_id, history_exhausted) VALUES (?, ?)",
            [("1000", 1), ("2000", 0)],
        )
        connection.commit()
        connection.close()

    def tearDown(self) -> None:
        for name in (
            "qq_history_cache.db",
            "qq_history_cache.db-wal",
            "qq_history_cache.db-shm",
            "scope.db",
            "scope.db-wal",
            "scope.db-shm",
        ):
            (self.root / name).unlink(missing_ok=True)
        self.root.rmdir()


    def test_platform_instance_id_is_the_umo_prefix(self) -> None:
        scope = angel_eye_scope(platform_id="instance-a", group_id="123")
        self.assertEqual(scope.key, "instance-a:GroupMessage:123")
        with self.assertRaises(ValueError):
            angel_eye_scope(platform_id="bad:id", group_id="123")


if __name__ == "__main__":
    unittest.main()
