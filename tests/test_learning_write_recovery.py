import os
import copy
import json
import unittest
import uuid
from pathlib import Path

from mr_memory.learning_writes import LearningWriter
from mr_memory.store import Store


class LearningWriteRecoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.scratch = Path(os.environ['ASTRBOT_ROOT']) / ("write-recovery-" + uuid.uuid4().hex)
        self.scratch.mkdir(parents=True)
        self.store = Store(self.scratch / "group.db", "test:GroupMessage:42")
        self.rows = [self.store.append_message({"platform_id": "test", "umo": self.store.umo,
            "group_id": "42", "message_id": str(i), "sender_id": str(i), "sender_name": f"群友{i}",
            "sent_at": 1700000000 + i, "plain_text": text, "content": [], "role": "USER"})
            for i, text in enumerate(("周末改到西门见", "我带桌游来"), 1)]
        self.ids = [row["id"] for row in self.rows]
        self.sources = {row["id"]: row["source_key"] for row in self.rows}
        self.task = self.store.start_learning_task("background", self.ids)

    def tearDown(self):
        self.store.close()
        for path in self.scratch.iterdir():
            path.unlink()
        self.scratch.rmdir()

    def args(self):
        return {"items": [{"kind": "episode", "summary": "约在西门见", "source_ids": [999]},
                          {"kind": "semantic", "content": "群友2会带桌游", "source_ids": [self.ids[1]]}],
                "progress": {"completed_ids": self.ids, "checkpoint": "已读完，原文编号可继续查阅"}, "finish": True}

    def test_mutual_drafts_and_self_link_commit_once_with_receipts(self):
        writer = LearningWriter(self.store, self.sources, learning_kind="background")
        args = {"items": [
            {"kind": "episode", "handle": "trip", "content": "周末桌游", "source_ids": self.ids,
             "connections": [{"handle": "plan", "relation": "安排"}]},
            {"kind": "semantic", "handle": "plan", "content": "西门集合", "source_ids": self.ids,
             "connections": [{"handle": "trip", "relation": "出自", "purpose": "basis"},
                             {"handle": "plan", "relation": "自我说明"}]}]}
        first = writer.apply(args, "cycle")
        self.assertFalse(first.rejected)
        self.assertFalse(writer.pending_items)
        self.assertEqual(len(first.written), 2)
        refs = {row["handle"]: row for row in first.written}
        basis = self.store.memory("semantic", refs["plan"]["id"])["basis"]
        self.assertEqual(basis[0]["target"]["id"], refs["trip"]["id"])
        state = self.store.learning_task("background")["continuation"]["write_state"]
        self.assertEqual(set(state["receipts"]), {"cycle:0", "cycle:1"})
        again = LearningWriter(self.store, self.sources, learning_kind="background", **state)
        self.assertEqual(len(again.apply(args, "cycle").written), 2)
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM mr_memory_objects").fetchone()[0], 5)

    def test_cycle_failure_rolls_back_objects_edges_and_receipts_but_not_other_draft(self):
        writer = LearningWriter(self.store, self.sources, learning_kind="background")
        result = writer.apply({"items": [
            {"kind": "episode", "handle": "a", "content": "约定", "connections": [{"handle": "b", "relation": "相关"}]},
            {"kind": "semantic", "handle": "b", "content": "地点", "connections": [{"handle": "a"}]},
            {"kind": "episode", "content": "独立的另一项记忆"}]}, "broken-cycle")
        self.assertEqual(len(result.written), 1)
        self.assertEqual(len(writer.pending_items), 2)
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM mr_memory_objects").fetchone()[0], 1)
        self.assertEqual(set(self.store.learning_task("background")["continuation"]["write_state"]["receipts"]), {"broken-cycle:2"})
        repaired = writer.apply({"retry": [{"pending_id": "broken-cycle:0"},
            {"pending_id": "broken-cycle:1", "changes": {"connections": [{"handle": "a", "relation": "出自"}]}}]}, "retry")
        self.assertEqual(len(repaired.written), 2)
        self.assertFalse(writer.pending_items)

    async def test_partial_write_keeps_success_and_repairs_only_one_field_after_reopen(self):
        writer = LearningWriter(self.store, self.sources, learning_kind="background")
        outcome = writer.apply(self.args(), "save")
        self.assertEqual(len(outcome.written), 1)
        self.assertEqual(set(writer.pending_items), {"save:0"})
        self.assertEqual(writer.pending_items["save:0"]["item"]["source_ids"], [999])
        task = self.store.learning_task("background")
        self.assertEqual(task["completed_ids"], self.ids)
        self.assertEqual(task["checkpoint"], self.args()["progress"]["checkpoint"])
        self.assertEqual(task["memory_refs"], [{"kind": "semantic", "id": outcome.written[0]["id"]}])
        compact = json.loads(json.dumps(task["continuation"]["write_state"]))
        writer = LearningWriter(self.store, self.sources, learning_kind="background", **compact)
        writer.apply({"items": [], "progress": {"checkpoint": "仍需补来源"}}, "unrelated")
        self.assertEqual(set(writer.pending_items), {"save:0"})
        repaired = writer.apply({"retry": [{"pending_id": "save:0", "changes": {"source_ids": [self.ids[0]]}}]}, "repair")
        self.assertEqual(len(repaired.written), 1)
        self.assertEqual(repaired.written[0]["source_ids"], [self.ids[0]])
        self.assertEqual(repaired.written[0]["summary"], "约在西门见")
        self.assertFalse(writer.pending_items)
        self.assertEqual(self.store.learning_task("background")["completed_ids"], self.ids)
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM mr_memory_objects WHERE kind='semantic'").fetchone()[0], 1)

    def test_recovery_batch_keeps_dependencies_and_leaves_other_drafts_queued(self):
        writer = LearningWriter(self.store, self.sources, learning_kind="background", run_id=7)
        writer.apply(self.args(), "save")
        self.store.finish_learning_task("background")
        queued = self.store.load_working_state()["pending_learning_drafts"]["background"]
        queued["dependency"] = {**copy.deepcopy(queued["save:0"]), "origin_run_id": 8, "bindings": {}}
        queued["unrelated"] = {**copy.deepcopy(queued["save:0"]), "origin_run_id": 9, "bindings": {}}
        queued["save:0"]["bindings"] = {"next": {"receipt_key": "dependency"}}
        self.store.update_working_state({"pending_learning_drafts": {"background": queued}})
        selected = self.store.pending_learning_batch("background")
        self.assertEqual(set(selected), {"save:0", "dependency"})
        task = self.store.start_learning_task("background", [], pending_ids=list(selected))
        self.assertEqual(set(task["continuation"]["write_state"]["pending_items"]), set(selected))
        self.assertEqual(set(self.store.load_working_state()["pending_learning_drafts"]["background"]), {"unrelated"})
        self.store.finish_learning_task("background")
        self.assertEqual(set(self.store.load_working_state()["pending_learning_drafts"]["background"]), set(queued))

    async def test_duplicate_retry_does_not_create_two_associations(self):
        writer = LearningWriter(self.store, self.sources, learning_kind="background")
        writer.apply({"items": [{"kind": "association", "source": {"label": "小桥"},
            "target": {"label": "桌游"}, "relation": "携带", "statement": "小桥说会带桌游", "source_ids": [999]}]}, "draft")
        retry = {"pending_id": "draft:0", "changes": {"source_ids": self.ids}}
        outcome = writer.apply({"retry": [retry, retry]}, "repair")
        self.assertEqual(len(outcome.written), 1)
        self.assertEqual(len(outcome.rejected), 1)
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM mr_memory_objects WHERE kind='association'").fetchone()[0], 1)

if __name__ == "__main__":
    unittest.main()
