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

    def test_preview_normalization_and_idempotent_storage_import(self) -> None:
        source = AngelEyeHistorySource(self.source_path)
        preview = source.inspect()
        self.assertEqual([item.group_id for item in preview], ["1000", "2000"])
        self.assertEqual(preview[0].messages, 2)
        self.assertTrue(preview[0].history_exhausted)

        messages, last_row_id, skipped = source.load_batch(
            group_id="1000",
            platform_id="byy_official",
            after_row_id=0,
            through_row_id=preview[0].through_row_id,
        )
        self.assertEqual((len(messages), skipped), (2, 0))
        self.assertEqual(last_row_id, preview[0].through_row_id)
        self.assertEqual(messages[0].umo, "byy_official:GroupMessage:1000")
        self.assertEqual(messages[0].sender_name, "群名片")
        self.assertEqual(messages[0].role, "USER")
        self.assertEqual(messages[1].role, "BOT")
        self.assertIn("@对象", messages[0].plain_text)
        self.assertIn("[图片]", messages[0].plain_text)
        encoded_content = json.dumps(messages[0].content, ensure_ascii=False)
        self.assertNotIn("private.example", encoded_content)
        self.assertNotIn("token=secret", encoded_content)
        self.assertRegex(
            str(messages[0].content[-1]["reference_sha256"]),
            r"^[0-9a-f]{64}$",
        )

        scope = angel_eye_scope(platform_id="byy_official", group_id="1000")
        storage = MemoryStorage(self.root / "scope.db")
        storage.bind_scope(
            umo=scope.key,
            platform_id=scope.platform_id,
            group_id=scope.group_id,
        )
        first_result = storage.upsert_messages(messages, defer_media_index=True)
        second_result = storage.upsert_messages(messages, defer_media_index=True)
        storage.rebuild_media_fingerprints(umo=scope.key)
        self.assertEqual(first_result, {"processed": 2, "inserted": 2})
        self.assertEqual(second_result, {"processed": 2, "inserted": 0})
        self.assertEqual(storage.count_messages(umo=scope.key), 2)
        row = storage._connection.execute(
            "SELECT content_json FROM messages WHERE message_id='11'"
        ).fetchone()
        self.assertNotIn("private.example", str(row["content_json"]))
        storage.close()

        invalid, last_invalid, skipped_invalid = source.load_batch(
            group_id="2000",
            platform_id="byy_official",
            after_row_id=0,
            through_row_id=preview[1].through_row_id,
        )
        self.assertEqual(invalid, [])
        self.assertEqual(last_invalid, preview[1].through_row_id)
        self.assertEqual(skipped_invalid, 1)

    def test_platform_instance_id_is_the_umo_prefix(self) -> None:
        scope = angel_eye_scope(platform_id="instance-a", group_id="123")
        self.assertEqual(scope.key, "instance-a:GroupMessage:123")
        with self.assertRaises(ValueError):
            angel_eye_scope(platform_id="bad:id", group_id="123")


if __name__ == "__main__":
    unittest.main()
