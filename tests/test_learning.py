import copy
import json
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mr_memory.agent import MemoryAgent, checkpoint_learning_task, estimate_learning_input
from mr_memory.store import Store


class LearningTaskTests(unittest.TestCase):
    def setUp(self):
        self.scratch = Path(__file__).resolve().parents[1] / ".pytest_cache" / ("learning-" + uuid.uuid4().hex)
        self.scratch.mkdir(parents=True)
        self.path = self.scratch / "group.db"
        self.scope = "test:GroupMessage:42"
        self.store = Store(self.path, self.scope)

    def tearDown(self):
        self.store.close()
        for file in self.scratch.iterdir():
            file.unlink()
        self.scratch.rmdir()

    def message(self, number, text, *, role="USER", at=None, reply_to=None):
        return self.store.append_message(dict(platform="test", platform_id="test", umo=self.scope,
            group_id="42", message_id=str(number), sender_id="99" if role != "USER" else "10",
            sender_name="Test sender", sent_at=1_800_000_000 + number if at is None else at,
            plain_text=text, content=[{"type": "text", "text": text}], role=role, reply_to=reply_to))

    def test_partial_task_survives_reopen_and_keeps_exact_material(self):
        context = self.message(1, "之前的背景")
        first = self.message(2, "第一条待处理材料")
        second = self.message(3, "第二条待处理材料")
        task = self.store.start_learning_task("background", [first["id"], second["id"]], [context["id"]],
                                              {"plan": "理解这次对话"})
        self.store.update_learning_task("background", {"continuation": {"conversation": [{"role": "user", "content": "原始输入"}]}})
        self.store.save_learning_progress("background", {"completed_ids": [first["id"]], "checkpoint": "继续分析第二条"}, 8)
        self.store.save_working_state({"foreground_observation": "独立更新"})
        self.store.close()
        self.store = Store(self.path, self.scope)
        newer = self.message(4, "恢复前刚来的新材料")
        resumed = self.store.start_learning_task("background", [newer["id"]])
        self.assertEqual(resumed["material_ids"], task["material_ids"])
        self.assertEqual(resumed["completed_ids"], [first["id"]])
        self.assertEqual(resumed["checkpoint"], "继续分析第二条")
        self.assertEqual(resumed["run_id"], 8)
        self.assertEqual(resumed["continuation"]["conversation"], [{"role": "user", "content": "原始输入"}])
        rows = self.store.learning_messages(resumed)
        self.assertEqual([row["id"] for row in rows], [context["id"], first["id"], second["id"]])
        self.assertEqual([row["context_only"] for row in rows], [True, False, False])
        self.assertEqual({row["id"] for row in self.store.pending_messages(10)},
                         {context["id"], second["id"], newer["id"]})

    def test_memory_evidence_is_not_processing_progress_and_both_writes_are_atomic(self):
        evidence = self.message(1, "证据来自这一条")
        handled = self.message(2, "模型已处理但没有记忆价值的闲聊")
        task = self.store.start_learning_task("background", [evidence["id"], handled["id"]])
        item = {"kind": "semantic", "content": "模型的理解", "source_keys": [evidence["source_key"]]}
        with self.assertRaisesRegex(ValueError, "this task's material"):
            self.store.save_memories([item], [evidence["source_key"]], learning_kind="background",
                                      progress={"completed_ids": [999]}, run_id=3)
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM semantic_memories").fetchone()[0], 0)
        self.assertEqual(self.store.learning_task("background"), task)
        saved = self.store.save_memories([item], [evidence["source_key"]], learning_kind="background",
                                         progress={"completed_ids": [handled["id"]]}, run_id=4)
        task = self.store.learning_task("background")
        self.assertEqual(task["completed_ids"], [handled["id"]])
        self.assertEqual(task["memory_refs"], [{"kind": "semantic", "id": saved[0]["id"]}])
        self.assertEqual([row["id"] for row in self.store.pending_messages(10)], [evidence["id"]])
        self.message(2, "已处理之后收到的编辑，应重新等待理解")
        self.store.save_memories([item], [evidence["source_key"]], learning_kind="background")
        self.assertEqual(self.store.learning_task("background")["memory_refs"], task["memory_refs"])
        self.assertEqual([row["id"] for row in self.store.pending_messages(10)], [evidence["id"], handled["id"]])

    def test_feedback_completion_records_each_message_without_moving_legacy_floor(self):
        context = self.message(1, "机器人回答", role="BOT")
        first = self.message(2, "第一条反馈")
        second = self.message(3, "第二条反馈")
        self.store.start_learning_task("feedback", [first["id"], second["id"]], [context["id"]])
        self.store.save_working_state({"existing_setting": "保留", "feedback_after": context["id"]})
        self.store.save_learning_progress("feedback", {"completed_ids": [second["id"]]})
        self.assertEqual(self.store.load_working_state()["feedback_after"], context["id"])
        self.assertEqual([r[0] for r in self.store.db.execute("SELECT message_id FROM mr_feedback_processed")], [second["id"]])
        with self.assertRaisesRegex(ValueError, "not its context"):
            self.store.save_learning_progress("feedback", {"completed_ids": [context["id"]]})
        self.store.save_learning_progress("feedback", {"completed_ids": [first["id"]]})
        self.assertEqual(self.store.load_working_state(), {"existing_setting": "保留", "feedback_after": context["id"]})
        self.store.update_working_state({"foreground_observation": "新观察"})
        self.assertEqual(self.store.load_working_state()["feedback_after"], context["id"])
        self.assertEqual(self.store.pending_status()["count"], 3)
        self.store.start_learning_task("background", [context["id"]])
        self.store.finish_learning_task("feedback")
        self.assertIsNone(self.store.learning_task("feedback"))
        self.assertIsNotNone(self.store.learning_task("background"))
        self.store.forget("10")
        self.assertIsNone(self.store.learning_task("background"))

    def test_daytime_feedback_waits_without_expiring_and_keeps_old_original_question(self):
        question = self.message(1, "前一天提的问题", at=100)
        bot = self.message(2, "白天的回答", role="BOT", at=10_000, reply_to=question["source_key"])
        reaction = self.message(3, "一小时后的纠正", at=13_600)
        newer = self.message(4, "数据库 ID 更新但发送时间稍早的补充", at=13_599)
        with patch("mr_memory.store.time.time", return_value=100_000):
            rows = self.store.feedback_messages(21_600, 0)
        self.assertEqual([row["id"] for row in rows], [question["id"], bot["id"], newer["id"], reaction["id"]])
        self.assertEqual([row["context_only"] for row in rows], [True, True, False, False])
        late = self.store.feedback_messages(21_600, reaction["id"])
        self.assertEqual([row["id"] for row in late if not row["context_only"]], [newer["id"]])
        self.assertEqual(self.store.feedback_messages(21_600, newer["id"]), [])
        self.message(5, "距回答太远，尚无新机器人回答", at=50_000)
        self.assertEqual(self.store.feedback_messages(21_600, newer["id"]), [])

    def test_old_partial_directory_keeps_previous_writes_when_latest_run_wrote_nothing(self):
        self.assertIsNone(self.store.unfinished_learning("feedback"))
        self.store.record_run("feedback", 1, {"status": "completed", "written": [{"kind": "semantic", "id": 1}]})
        first = self.store.record_run("feedback", 2, {"status": "partial",
            "written": [{"kind": "semantic", "id": 7}, {"kind": "episode", "id": 3}]})
        self.store.record_run("background", 3, {"status": "partial", "written": [{"kind": "semantic", "id": 99}]})
        second = self.store.record_run("feedback", 4, {"status": "partial", "written": [], "detail": "额度已用完"})
        directory = self.store.unfinished_learning("feedback")
        self.assertEqual(directory["run_ids"], [first, second])
        self.assertEqual(directory["memory_refs"], [{"kind": "episode", "id": 3}, {"kind": "semantic", "id": 7}])
        self.assertEqual(directory["detail"], "额度已用完")
        self.assertNotIn("completed_ids", directory)
        self.store.record_run("feedback", 5, {"status": "completed", "written": []})
        self.assertIsNone(self.store.unfinished_learning("feedback"))

    def test_recent_feedback_uses_event_time_and_rebuilds_trimmed_interaction_prefix(self):
        old_bot = self.message(1, "旧回答", role="BOT", at=1000)
        self.message(2, "旧回答的反馈", at=1001)
        question = self.message(3, "新的问题", at=50000)
        bot = self.message(4, "新问题的回答", role="BOT", at=50001, reply_to=question["source_key"])
        first = self.message(5, "新回答有一处需要纠正", at=50002)
        second = self.message(6, "紧接着补充理由", at=50003)
        imported_bot = self.message(7, "后来导入的更早回答", role="BOT", at=100)
        imported = self.message(8, "后来导入的历史反馈", at=101)
        rows = self.store.feedback_messages(3600, 0, 5, newest=True)
        ids = [row["id"] for row in rows]
        self.assertLessEqual(len(rows), 5)
        self.assertTrue({question["id"], bot["id"], first["id"], second["id"]}.issubset(ids))
        self.assertNotIn(old_bot["id"], ids)
        self.assertNotIn(imported["id"], ids)
        self.assertNotIn(imported_bot["id"], ids)

    def test_recent_completion_preserves_old_queue_and_alternates_only_finished_batches(self):
        self.message(1, "旧回答", role="BOT", at=1000)
        old = [self.message(i, "旧反馈" + str(i), at=1000 + i) for i in range(2, 5)]
        self.message(5, "新回答", role="BOT", at=50000)
        recent = [self.message(i, "新反馈" + str(i), at=50000 + i) for i in range(6, 9)]
        rows = self.store.feedback_messages(3600, 0, 4, newest=True)
        ids = [row["id"] for row in rows if not row["context_only"]]
        self.assertEqual(ids, [row["id"] for row in recent])
        task = self.store.start_learning_task("feedback", ids, working={"feedback_order": "recent"})
        self.store.save_learning_progress("feedback", {"completed_ids": ids[:1]})
        self.assertEqual(self.store.load_working_state().get("feedback_next_order", "recent"), "recent")
        self.assertEqual(self.store.start_learning_task("feedback", [old[0]["id"]])["material_ids"], task["material_ids"])
        self.store.save_learning_progress("feedback", {"completed_ids": ids[1:]})
        self.store.finish_learning_task("feedback")
        self.assertEqual(self.store.load_working_state()["feedback_next_order"], "oldest")
        self.assertNotIn("feedback_after", self.store.load_working_state())
        older = self.store.feedback_messages(3600, 0, 4, newest=False)
        older_ids = [row["id"] for row in older if not row["context_only"]]
        self.assertEqual(older_ids, [row["id"] for row in old])
        self.store.start_learning_task("feedback", older_ids, working={"feedback_order": "oldest"})
        self.store.save_learning_progress("feedback", {"completed_ids": older_ids})
        self.store.finish_learning_task("feedback")
        self.assertEqual(self.store.load_working_state()["feedback_next_order"], "recent")
        self.store.start_learning_task("feedback", [], working={"feedback_order": "recent"})
        self.store.finish_learning_task("feedback")
        self.assertEqual(self.store.load_working_state()["feedback_next_order"], "recent")

    def test_legacy_cursor_and_sparse_unfinished_progress_survive_queue_migration(self):
        bot = self.message(1, "原回答", role="BOT", at=1000)
        first = self.message(2, "仍未处理", at=1001)
        second = self.message(3, "旧任务已单独处理", at=1002)
        self.store.save_working_state({"feedback_after": bot["id"], "unrelated": "保留"})
        self.store.start_learning_task("feedback", [first["id"], second["id"]])
        self.store.update_learning_task("feedback", {"completed_ids": [second["id"]],
            "checkpoint": "第二条已理解，第一条还需继续", "continuation": {"conversation": [{"role": "user", "content": "旧续接"}]}})
        with self.store.db:
            self.store.db.execute("DROP TABLE mr_feedback_processed")
        self.store.close()
        self.store = Store(self.path, self.scope)
        self.assertEqual(self.store.load_working_state()["feedback_after"], bot["id"])
        self.assertEqual(self.store.learning_task("feedback")["completed_ids"], [second["id"]])
        self.assertEqual(self.store.learning_task("feedback")["continuation"]["conversation"][0]["content"], "旧续接")
        rows = self.store.feedback_messages(3600, bot["id"], newest=False)
        self.assertEqual([row["id"] for row in rows if not row["context_only"]], [first["id"]])
        self.store.save_learning_progress("feedback", {"completed_ids": [first["id"]]})
        self.store.finish_learning_task("feedback")
        self.assertEqual(self.store.load_working_state()["feedback_next_order"], "recent")
        self.assertEqual(self.store.feedback_messages(3600, bot["id"]), [])


class LearningCompletionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.scratch = Path(__file__).resolve().parents[1] / ".pytest_cache" / ("completion-" + uuid.uuid4().hex)
        self.scratch.mkdir(parents=True)
        self.store = Store(self.scratch / "group.db", "test:GroupMessage:42")

    def tearDown(self):
        self.store.close()
        for file in self.scratch.iterdir():
            file.unlink()
        self.scratch.rmdir()

    def message(self, number, text):
        return self.store.append_message(dict(platform="test", platform_id="test", umo=self.store.umo,
            group_id="42", message_id=str(number), sender_id="10", sender_name="小桥",
            sent_at=1_700_000_000 + number, plain_text=text, content=[], role="USER"))

    async def run_agent(self, calls, messages, task, *, turns=1, working=None, budget=None):
        class RecordedAgent(MemoryAgent):
            async def _model_turn(agent, conversation, prompt, tools, turn, **kwargs):
                agent.requests.append(copy.deepcopy(conversation))
                batch = agent.calls.pop(0)
                return SimpleNamespace(completion_text="", tools_call_name=[row[0] for row in batch],
                    tools_call_args=[row[1] for row in batch], tools_call_ids=[row[2] for row in batch],
                    reasoning_content="", usage={"input_other": 100, "output": 10},
                    raw_completion={"choices": [{"finish_reason": "tool_calls"}]})
        agent = RecordedAgent(None, self.store, max_turns=turns)
        agent.requests, agent.calls = [], list(calls)
        with patch("mr_memory.agent._tool_set", return_value=object()):
            result = await agent.consolidate(messages, working or {}, task=task, token_budget=budget)
        return result, agent

    async def test_finish_saves_old_source_and_same_turn_reflection_without_ack_call(self):
        old = self.message(1, "星舟原来是同学制作的桌游")
        recent = self.message(2, "下次加上四人规则")
        task = self.store.start_learning_task("background", [recent["id"]])
        calls = [[("remember", {"items": [{"kind": "semantic", "content": "星舟是同学制作的桌游，有四人规则",
            "source_ids": [old["id"], recent["id"]]}], "progress": {"completed_ids": [recent["id"]],
            "checkpoint": "名称已弄清，等下次试玩反馈"}, "finish": True}, "save"),
            ("reflect", {"content": "等待试玩后的新反馈", "status": "waiting", "source_ids": [recent["id"]]}, "reflect")]]
        result, agent = await self.run_agent(calls, [recent], task)
        self.assertEqual(result.status, "completed", result.detail)
        self.assertEqual(len(agent.requests), 1)
        self.assertEqual(result.written[0]["source_ids"], [old["id"], recent["id"]])
        self.assertEqual(self.store.learning_task("background")["completed_ids"], [recent["id"]])
        self.assertEqual(self.store.reflections.get(1)["status"], "waiting")

    async def test_full_material_progress_does_not_end_active_reflection(self):
        row = self.message(1, "称呼已经理解，仍需想清与旧事的联系")
        task = self.store.start_learning_task("background", [row["id"]])
        result, _ = await self.run_agent([[("remember", {"progress": {
            "completed_ids": [row["id"]], "checkpoint": "材料已处理，下一步保留尚不明确的联系"}}, "save")]], [row], task)
        self.assertEqual(result.status, "partial")
        task = checkpoint_learning_task(self.store.learning_task("background"))
        result, agent = await self.run_agent([
            [("reflect", {"content": "称呼与旧事件的关系需等当事人补充", "status": "waiting", "source_ids": [row["id"]]}, "reflect")],
            [("remember", {"finish": True}, "finish")]], [], task, turns=2)
        self.assertEqual(result.status, "completed", result.detail)
        self.assertEqual(len(agent.requests), 2)
        self.assertEqual(self.store.reflections.get(1)["status"], "waiting")

    async def test_failed_same_turn_reflection_keeps_real_write_but_prevents_finish(self):
        row = self.message(1, "新认识")
        task = self.store.start_learning_task("background", [row["id"]])
        with patch.object(self.store.reflections, "save", side_effect=OSError("reflection write failed")):
            result, _ = await self.run_agent([[
                ("remember", {"items": [{"kind": "semantic", "content": "新的认识", "source_ids": [row["id"]]}],
                    "progress": {"completed_ids": [row["id"]]}, "finish": True}, "save"),
                ("reflect", {"content": "尚需下次验证", "status": "waiting"}, "reflect")]], [row], task)
        self.assertEqual(result.status, "partial")
        self.assertEqual(len(result.written), 1)
        self.assertIn("reflection write failed", result.continuation["pending_write_errors"]["reflect"])

    async def test_finish_cannot_skip_unprocessed_material(self):
        first, second = self.message(1, "已理解"), self.message(2, "还没处理")
        task = self.store.start_learning_task("background", [first["id"], second["id"]])
        result, _ = await self.run_agent([[("remember", {"items": [], "progress": {
            "completed_ids": [first["id"]]}, "finish": True}, "save")]], [first, second], task)
        self.assertEqual(result.status, "partial")
        self.assertEqual(self.store.learning_task("background")["completed_ids"], [first["id"]])

    async def test_checkpoint_resume_keeps_reads_after_save_without_replaying_old_input(self):
        old = self.message(1, "这段经历已经理解。" * 300)
        later = self.message(2, "刚读到的新线索，仍需想清楚")
        task = self.store.start_learning_task("background", [old["id"]])
        result, _ = await self.run_agent([
            [("remember", {"items": [], "progress": {"completed_ids": [old["id"]],
                "checkpoint": "已处理原材料，继续看新的线索"}}, "save")],
            [("context", {"message_id": later["id"], "before": 0, "after": 0}, "read")]], [old], task, turns=2)
        task = self.store.learning_task("background")
        compact = checkpoint_learning_task(task)
        self.assertNotIn("conversation", compact["continuation"])
        self.assertIn(later["plain_text"], json.dumps(compact["continuation"]["checkpoint_tail"], ensure_ascii=False))
        self.assertNotIn(old["plain_text"], json.dumps(compact["continuation"], ensure_ascii=False))
        self.assertEqual(compact["checkpoint"], task["checkpoint"])
        self.assertLess(estimate_learning_input([], {}, task=compact), estimate_learning_input([], {}, task={
            **task, "continuation": {**task["continuation"], "previous_input_tokens": 0, "previous_input_bytes": 0}}))
        result, agent = await self.run_agent([[("remember", {"items": [], "finish": True}, "finish")]], [], compact)
        self.assertEqual(result.status, "completed", result.detail)
        self.assertIn(later["plain_text"], json.dumps(agent.requests[0], ensure_ascii=False))

    async def test_old_encoded_continuation_is_cleaned_before_model_or_budget_estimate(self):
        row = self.message(1, "看图后的交流")
        task = self.store.start_learning_task("background", [row["id"]])
        encoded = "base64://" + "ABCD" * 15000
        task["continuation"] = {"conversation": [{"role": "user", "content": encoded}],
            "seen_messages": {str(row["id"]): {**row, "content": [{"type": "image", "file": encoded}]}},
            "previous_input_tokens": 80000, "previous_input_bytes": 85000}
        self.assertLess(estimate_learning_input([row], {}, task=task), 30000)
        result, agent = await self.run_agent([[("remember", {"items": [], "progress": {
            "completed_ids": [row["id"]]}, "finish": True}, "finish")]], [row], task, budget=30000)
        self.assertEqual(result.status, "completed", result.detail)
        self.assertNotIn(encoded, json.dumps(agent.requests))
        self.assertNotIn(encoded, json.dumps(result.continuation))
        self.assertEqual(task["continuation"]["conversation"][0]["content"], encoded)

    async def test_missing_source_is_reported_without_false_completion(self):
        row = self.message(1, "待理解")
        task = self.store.start_learning_task("background", [row["id"]])
        result, _ = await self.run_agent([[("remember", {"items": [{"kind": "semantic", "content": "新认识",
            "source_ids": [999]}], "progress": {"completed_ids": [row["id"]]}, "finish": True}, "save")]], [row], task)
        self.assertNotEqual(result.status, "completed")
        self.assertIn("unavailable in this group: [999]", result.tool_calls[0]["result"]["detail"])
        self.assertEqual(self.store.learning_task("background")["completed_ids"], [])


if __name__ == "__main__":
    unittest.main()
