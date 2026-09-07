import json
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from mr_memory.store import Store


class UsageTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / ".pytest_cache"
        scratch.mkdir(exist_ok=True)
        self.directory = scratch / ("usage-" + uuid.uuid4().hex)
        self.directory.mkdir()
        self.path = self.directory / "scope.db"
        self.umo = "demo:GroupMessage:42"
        self.now = 1_800_000_000
        self.store = Store(self.path, self.umo)

    def tearDown(self):
        self.store.close()
        for file in self.directory.iterdir():
            file.unlink()
        self.directory.rmdir()

    def test_rolling_budget_combines_legacy_resets_and_separate_new_usage(self):
        self.store.db.executescript("""
            CREATE TABLE experiment_runs(run_id TEXT PRIMARY KEY,umo TEXT);
            CREATE TABLE llm_usage_events(id INTEGER PRIMARY KEY,run_id TEXT,phase TEXT,
                input_other INTEGER,input_cached INTEGER,output INTEGER,created_at TEXT);
            CREATE TABLE token_budget_resets(id INTEGER PRIMARY KEY,umo TEXT,budget_class TEXT,
                reset_at INTEGER,usage_event_id INTEGER);
        """)
        self.store.tables.update({"experiment_runs", "llm_usage_events", "token_budget_resets"})
        with self.store.db:
            self.store.db.executemany("INSERT INTO experiment_runs VALUES(?,?)",
                                     [("ours", self.umo), ("other", "demo:GroupMessage:43")])
            events = [
                (1, "ours", "construction", 100, self.now - 100),  # excluded by reset id
                (2, "ours", "construction", 30, self.now - 100),
                (3, "ours", "eccr_reader", 40, self.now - 100),
                (4, "ours", "feedback_maintenance", 7, self.now - 100),
                (5, "ours", "history_construction", 900, self.now - 100),
                (6, "ours", "resident_evidence_reader", 900, self.now - 100),
                (7, "other", "construction", 900, self.now - 100),
                (8, "ours", "construction", 900, self.now - 90000),
            ]
            for id, run, phase, tokens, at in events:
                self.store.db.execute("INSERT INTO llm_usage_events VALUES(?,?,?, ?,0,0,datetime(?,'unixepoch'))",
                                      (id, run, phase, tokens, at))
            self.store.db.execute("INSERT INTO token_budget_resets VALUES(1,?,'online',?,1)",
                                  (self.umo, self.now - 200))
        background = self.store.reserve_usage("background", 50, at=self.now)
        self.store.reserve_usage("feedback", 13, at=self.now)
        self.assertEqual(self.store.usage_total("background", now=self.now), 120)
        self.assertEqual(self.store.usage_total("feedback", now=self.now), 20)
        self.store.settle_usage(background, 20)
        self.assertEqual(self.store.usage_total("background", now=self.now), 90)
        self.assertEqual(self.store.usage_total("background", now=self.now + 86401), 0)
        self.assertEqual(self.store.usage_total("feedback", now=self.now + 86401), 0)

    def test_carryover_is_one_labeled_aggregate_across_reloads(self):
        self.store.save_working_state({"background_tokens": 1234, "consolidated_at": self.now - 500,
                                      "background_day": (self.now + 28800) // 86400})
        self.assertEqual(self.store.usage_total("background", now=self.now), 1234)
        self.store.close()
        self.store = Store(self.path, self.umo)
        self.assertEqual(self.store.usage_total("background", now=self.now), 1234)
        rows = self.store.db.execute("SELECT * FROM mr_usage WHERE migration_key IS NOT NULL").fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "carryover_estimate")
        self.assertEqual(rows[0]["occurred_at"], self.now - 500)
        self.assertTrue(json.loads(rows[0]["detail_json"])["aggregate_only"])
        self.assertEqual(self.store.usage_total("background", now=self.now + 86401), 0)

    def message(self, number, text, role="USER", **extra):
        value = {"platform": "demo", "platform_id": "demo", "umo": self.umo,
                 "group_id": "42", "message_id": str(number), "sender_id": "bot" if role != "USER" else "person",
                 "sender_name": "Bot" if role != "USER" else "Person", "sent_at": self.now - 20 + number,
                 "plain_text": text, "content": [], "role": role, **extra}
        return self.store.append_message(value)

    def test_feedback_keeps_reply_context_and_does_not_mark_sources_distilled(self):
        question = self.message(1, "今天在哪里见面")
        first = self.message(2, "我记得是东门", "BOT", reply_to="1")
        second = self.message(3, "下午见", "BOT", reply_to="1")
        reaction = self.message(4, "改到西门了")
        tool = self.message(5, "", "SYSTEM", reply_to="1",
                            content=[{"type": "tool_result", "result": "活动通知"}])
        with patch("mr_memory.store.time.time", return_value=self.now):
            messages = self.store.feedback_messages(3600, after=second["id"])
            self.assertEqual([m["id"] for m in messages],
                             [question["id"], first["id"], second["id"], reaction["id"], tool["id"]])
            self.assertEqual(self.store.feedback_messages(3600, after=tool["id"]), [])
        self.store.save_memories([{"kind": "semantic", "content": "约定已改到西门"}],
                                 [reaction["source_key"]], mark_processed=False)
        self.assertEqual(len(self.store.pending_messages(1)), 1)
        self.assertEqual(self.store.pending_status(), {"count": 5, "oldest_at": question["sent_at"]})
        self.assertIn(reaction["id"], [m["id"] for m in self.store.pending_messages(20)])
        self.store.save_memories([], [reaction["source_key"]])
        self.assertEqual(self.store.pending_status()["count"], 4)
        self.assertNotIn(reaction["id"], [m["id"] for m in self.store.pending_messages(20)])
        self.store.save_working_state({"learned_feedback": [{"content": "地点以最新约定为准"}]})
        self.assertEqual(self.store.feedback()[0]["content"], "地点以最新约定为准")


if __name__ == "__main__":
    unittest.main()
