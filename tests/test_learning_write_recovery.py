import copy
import json
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mr_memory.agent import MemoryAgent, checkpoint_learning_task, estimate_learning_input
from mr_memory.learning_writes import LearningWriter
from mr_memory.store import Store


class LearningWriteRecoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.scratch = Path(__file__).resolve().parents[1] / ".pytest_cache" / ("write-recovery-" + uuid.uuid4().hex)
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
        return {"items": [{"kind": "episode", "summary": "约在西门见"},
                          {"kind": "semantic", "content": "群友2会带桌游", "source_ids": [self.ids[1]]}],
                "progress": {"completed_ids": self.ids, "checkpoint": "已读完，原文编号可继续查阅"}, "finish": True}

    async def test_partial_write_keeps_success_and_repairs_only_one_field_after_reopen(self):
        writer = LearningWriter(self.store, self.sources, learning_kind="background")
        outcome = writer.apply(self.args(), "save")
        self.assertEqual(len(outcome.written), 1)
        self.assertEqual(set(writer.pending_items), {"save:0"})
        self.assertNotIn("source_ids", writer.pending_items["save:0"]["item"])
        task = self.store.learning_task("background")
        self.assertEqual(task["completed_ids"], [])
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
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM semantic_memories").fetchone()[0], 1)

    async def test_association_receipt_and_pending_state_commit_with_the_write(self):
        writer = LearningWriter(self.store, self.sources, learning_kind="background")
        args = {"items": [{"kind": "association", "source": {"label": "小桥"},
            "target": {"label": "桌游"}, "relation": "携带", "statement": "小桥说会带桌游", "source_ids": self.ids}]}
        save = self.store.save_memories

        class Interrupted(BaseException):
            pass

        def stop_after_commit(*a, **kw):
            save(*a, **kw)
            raise Interrupted()

        with patch.object(self.store, "save_memories", side_effect=stop_after_commit), self.assertRaises(Interrupted):
            writer.apply(args, "one")
        state = self.store.learning_task("background")["continuation"]["write_state"]
        self.assertFalse(state["pending_items"])
        self.assertEqual(len(state["receipts"]["one:0"]), 1)
        resumed = LearningWriter(self.store, self.sources, learning_kind="background", **state)
        resumed.apply(args, "one")
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM plastic_edges").fetchone()[0], 1)

    async def test_duplicate_retry_does_not_create_two_associations(self):
        writer = LearningWriter(self.store, self.sources, learning_kind="background")
        writer.apply({"items": [{"kind": "association", "source": {"label": "小桥"},
            "target": {"label": "桌游"}, "relation": "携带", "statement": "小桥说会带桌游"}]}, "draft")
        retry = {"pending_id": "draft:0", "changes": {"source_ids": self.ids}}
        outcome = writer.apply({"retry": [retry, retry]}, "repair")
        self.assertEqual(len(outcome.written), 1)
        self.assertEqual(len(outcome.rejected), 1)
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM plastic_edges").fetchone()[0], 1)

    async def test_failed_detailed_receipt_does_not_repeat_a_committed_write(self):
        writer = LearningWriter(self.store, self.sources, learning_kind="background")
        save = self.store.save_memories

        def fail_receipt(*args, **kwargs):
            save(*args, **kwargs)
            raise RuntimeError("Detailed receipt read failed after commit")

        with patch.object(self.store, "save_memories", side_effect=fail_receipt):
            outcome = writer.apply({"items": [{"kind": "episode", "summary": "约在西门见",
                                                "source_ids": self.ids}]}, "save")
        self.assertEqual(len(outcome.written), 1)
        self.assertFalse(writer.pending_items)
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM episodes").fetchone()[0], 1)

    async def test_partial_checkpoint_repairs_one_draft_without_repeating_old_retrieval(self):
        args = self.args()
        writer = LearningWriter(self.store, self.sources, learning_kind="background")
        outcome = writer.apply(args, "save")
        old_body = "过去已读过的一大段无关检索原文" * 2000
        conversation = [{"role": "user", "content": old_body},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "save", "type": "function",
                "function": {"name": "remember", "arguments": json.dumps(args, ensure_ascii=False)}}]},
            {"role": "tool", "tool_call_id": "save", "content": json.dumps(outcome.receipt(writer.pending_items))}]
        task = self.store.learning_task("background")
        task["continuation"] = {"conversation": conversation, "sources": self.sources,
            "pending_write_errors": {"remember": "source_ids missing"}, "previous_input_tokens": 90000,
            "previous_input_bytes": 300000, "write_state": writer.state()}
        task = checkpoint_learning_task(task)
        self.store.update_learning_task("background", {"continuation": task["continuation"]})

        class RepairAgent(MemoryAgent):
            async def _model_turn(agent, messages, prompt, tools, turn, **kwargs):
                agent.requests.append(copy.deepcopy(messages))
                return SimpleNamespace(completion_text="", tools_call_name=["remember"],
                    tools_call_args=[{"retry": [{"pending_id": "save:0", "changes": {"source_ids": [self.ids[0]]}}],
                                      "finish": True}], tools_call_ids=["repair"], reasoning_content="",
                    usage={"input_other": 100, "output": 20}, raw_completion={"choices": [{"finish_reason": "tool_calls"}]})

        agent = RepairAgent(None, self.store, max_turns=1)
        agent.requests = []
        with patch("mr_memory.agent._tool_set", return_value=object()):
            result = await agent.consolidate(self.rows, {}, task=task, token_budget=30000)
        self.assertEqual(result.status, "completed", result.detail)
        self.assertEqual(len(result.written), 1)
        self.assertEqual(len(agent.requests), 1)
        self.assertNotIn(old_body, json.dumps(agent.requests, ensure_ascii=False))
        self.assertIn("save:0", json.dumps(agent.requests, ensure_ascii=False))
        self.assertEqual(self.store.learning_task("background")["completed_ids"], self.ids)

    async def test_checkpoint_keeps_density_for_estimate_and_admission_then_recalibrates(self):
        progress = {"completed_ids": self.ids, "checkpoint": "材料已理解，继续确认最后的进度"}
        self.store.save_learning_progress("background", progress)
        task = self.store.learning_task("background")
        task["continuation"] = {"previous_input_tokens": 90000, "previous_input_bytes": 900000,
            "conversation": [{"role": "user", "content": "已处理的长篇材料" * 10000},
                {"role": "assistant", "tool_calls": [{"id": "save", "type": "function", "function": {
                    "name": "remember", "arguments": json.dumps({"items": [], "progress": progress})}}]},
                {"role": "tool", "tool_call_id": "save", "content": "[]"}]}
        full_estimate = estimate_learning_input([], {}, task=task)
        self.assertGreaterEqual(full_estimate, 90000)
        compact = json.loads(json.dumps(checkpoint_learning_task(task)))
        self.assertEqual(compact["continuation"]["input_tokens_per_byte"], 0.1)
        compact_estimate = estimate_learning_input([], {}, task=compact)
        self.assertLess(compact_estimate, full_estimate)

        class FinishAgent(MemoryAgent):
            async def _model_turn(agent, messages, prompt, tools, turn, **kwargs):
                agent.calls += 1
                return SimpleNamespace(completion_text='{"items":[]}', tools_call_name=[], reasoning_content="",
                    usage={"input_other": 500, "output": 10}, raw_completion={"choices": [{"finish_reason": "stop"}]})

        agent = FinishAgent(None, self.store, max_turns=1)
        agent.calls = 0
        with patch("mr_memory.agent._tool_set", return_value=object()):
            result = await agent.consolidate([], {}, task=compact, token_budget=compact_estimate + 1000)
        self.assertEqual(agent.calls, 1)
        self.assertEqual(result.status, "completed", result.detail)
        updated = result.continuation
        self.assertEqual(updated["previous_input_tokens"], 500)
        self.assertEqual(updated["input_tokens_per_byte"], 500 / updated["previous_input_bytes"])
        self.assertEqual(self.store.learning_task("background")["continuation"]["input_tokens_per_byte"],
                         updated["input_tokens_per_byte"])


if __name__ == "__main__":
    unittest.main()
