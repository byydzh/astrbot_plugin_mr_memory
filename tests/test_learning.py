import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

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

    def test_feedback_completion_only_advances_over_contiguous_material(self):
        context = self.message(1, "机器人回答", role="BOT")
        first = self.message(2, "第一条反馈")
        second = self.message(3, "第二条反馈")
        self.store.start_learning_task("feedback", [first["id"], second["id"]], [context["id"]])
        self.store.save_working_state({"existing_setting": "保留"})
        self.store.save_learning_progress("feedback", {"completed_ids": [second["id"]]})
        self.assertNotIn("feedback_after", self.store.load_working_state())
        with self.assertRaisesRegex(ValueError, "not its context"):
            self.store.save_learning_progress("feedback", {"completed_ids": [context["id"]]})
        self.store.save_learning_progress("feedback", {"completed_ids": [first["id"]]})
        self.assertEqual(self.store.load_working_state(), {"existing_setting": "保留", "feedback_after": second["id"]})
        self.store.update_working_state({"foreground_observation": "新观察"})
        self.assertEqual(self.store.load_working_state()["feedback_after"], second["id"])
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
        self.assertEqual([row["id"] for row in rows], [question["id"], bot["id"], reaction["id"], newer["id"]])
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


if __name__ == "__main__":
    unittest.main()
