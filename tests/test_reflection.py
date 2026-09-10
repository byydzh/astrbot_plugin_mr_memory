import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from mr_memory.reflection import Reflection, SCHEMA
from mr_memory.store import Store


class ReflectionTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / ".pytest_cache"
        scratch.mkdir(exist_ok=True)
        self.scratch = scratch / ("reflection-" + uuid.uuid4().hex)
        self.scratch.mkdir()
        self.scope = "test:GroupMessage:42"
        self.store = Store(self.scratch / "group.db", self.scope)
        self.store.db.executescript(SCHEMA)
        self.reflections = Reflection(self.store)

    def tearDown(self):
        self.store.close()
        for file in self.scratch.iterdir():
            file.unlink()
        self.scratch.rmdir()

    def message(self, number, text, role="USER", request_id=None, event=None):
        content = [{"type": "text", "text": text}]
        if request_id:
            content.append({"type": "response_to", "message_id": str(request_id)})
        if event:
            content.append({"type": "bot_event", "event": event})
        return self.store.append_message(dict(platform="test", platform_id="test", umo=self.scope,
            group_id="42", message_id=str(number), sender_id="99" if role != "USER" else "10",
            sender_name="Test sender", sent_at=1_800_000_000 + number, plain_text=text,
            content=content, role=role, reply_to=str(request_id) if request_id else None))

    def test_concern_keeps_original_context_until_model_resolves_it(self):
        source = self.message(1, "被引用的是另一位成员的发言。")
        with patch("mr_memory.reflection.time.time", return_value=100):
            low = self.reflections.save({"content": "普通的待查问题", "priority": 1})
            concern = self.reflections.save({"content": "归属可能有误，需要回看原话及影响。", "priority": 9,
                "source_ids": [source["id"]], "memory_refs": [{"kind": "semantic", "id": 7}], "request_id": "1"})
        self.assertEqual([item["id"] for item in self.reflections.due(now=101)], [concern["id"], low["id"]])
        waiting = self.reflections.save({"id": concern["id"], "status": "waiting"})
        self.assertEqual(waiting["sources"][0]["plain_text"], source["plain_text"])
        self.assertEqual(self.reflections.associated("semantic", 7)[0]["id"], concern["id"])
        self.assertNotIn(concern["id"], [item["id"] for item in self.reflections.due(now=10**12)])
        self.reflections.save({"id": concern["id"], "next_review_at": 200})
        self.assertEqual(self.reflections.due(now=200)[0]["id"], concern["id"])
        self.reflections.save({"id": concern["id"], "status": "resolved", "content": "已结合引用澄清并修订。"})
        self.assertEqual(self.reflections.associated("semantic", 7), [])
        resolved = self.reflections.get(concern["id"])
        self.assertEqual(resolved["source_ids"], [source["id"]])
        self.assertEqual(resolved["request_id"], "1")
        self.assertEqual(self.reflections.waiting(), [])
        new = self.reflections.save({"id": 99, "content": "这是一条尚未分配地址的新关注"})
        self.assertNotEqual(new["id"], 99)
        updated = self.reflections.save({"id": new["id"], "content": "按返回地址继续更新"})
        self.assertEqual(updated["id"], new["id"])
        self.assertEqual(self.reflections.get(concern["id"])["status"], "resolved")

    def test_reviewed_does_not_overwrite_model_schedule_and_does_not_spin_waiting(self):
        with patch("mr_memory.reflection.time.time", return_value=100):
            untouched = self.reflections.save({"content": "还需继续核查"})
            scheduled = self.reflections.save({"content": "模型会安排下一次时间"})
            waiting = self.reflections.save({"content": "等待约定时间", "status": "waiting", "next_review_at": 110})
        with patch("mr_memory.reflection.time.time", return_value=120):
            self.reflections.save({"id": scheduled["id"], "next_review_at": 900})
            self.reflections.save({"id": untouched["id"], "content": "找到了部分线索，仍需继续核查"})
        with patch("mr_memory.reflection.time.time", return_value=130):
            self.reflections.reviewed([untouched["id"], scheduled["id"], waiting["id"]], 500, 115)
        self.assertEqual(self.reflections.get(untouched["id"])["next_review_at"], 500)
        self.assertEqual(self.reflections.get(scheduled["id"])["next_review_at"], 900)
        self.assertIsNone(self.reflections.get(waiting["id"])["next_review_at"])
        self.assertEqual(self.reflections.get(waiting["id"])["last_reviewed_at"], 130)
        self.assertEqual(self.reflections.due(now=131), [])

    def test_waiting_for_new_information_clears_an_old_pending_retry(self):
        task = self.reflections.save({"content": "核实转述对象", "next_review_at": 100})
        waiting = self.reflections.save({"id": task["id"], "status": "waiting"})
        self.assertIsNone(waiting["next_review_at"])
        self.assertEqual(self.reflections.due(now=200), [])
        self.reflections.save({"id": task["id"], "next_review_at": 300})
        self.assertEqual(self.reflections.due(now=300)[0]["id"], task["id"])
        cleared = self.reflections.save({"id": task["id"], "next_review_at": None})
        self.assertIsNone(cleared["next_review_at"])
        self.assertEqual(self.reflections.due(now=400), [])

    def test_only_model_appointments_bypass_preferred_window_and_automatic_retry_does_not(self):
        with patch("mr_memory.reflection.time.time", return_value=100):
            ordinary = self.reflections.save({"content": "下次整理时继续理解"})
            appointed = self.reflections.save({"content": "约定时间回来核实", "next_review_at": 200})
        self.assertEqual(self.reflections.due(now=199, scheduled_only=True), [])
        self.assertEqual([entry["id"] for entry in self.reflections.due(now=200, scheduled_only=True)], [appointed["id"]])
        with patch("mr_memory.reflection.time.time", return_value=210):
            self.reflections.reviewed([ordinary["id"], appointed["id"]], 300, 200)
        self.assertEqual(self.reflections.due(now=300, scheduled_only=True), [])
        self.assertEqual(len(self.reflections.due(now=300)), 2)
        with patch("mr_memory.reflection.time.time", return_value=310):
            # Choosing the same date as a system retry is still an explicit
            # model appointment and must survive this call's completion.
            self.reflections.save({"id": appointed["id"], "next_review_at": 300})
        with patch("mr_memory.reflection.time.time", return_value=320):
            self.reflections.reviewed([appointed["id"]], 500, 305)
        self.assertTrue(self.reflections.get(appointed["id"])["next_review_explicit"])
        self.assertEqual(self.reflections.get(appointed["id"])["next_review_at"], 300)

    def test_legacy_automatic_dates_are_not_inferred_to_be_model_appointments(self):
        self.store.db.execute("ALTER TABLE mr_reflections DROP COLUMN next_review_explicit")
        self.store.db.execute("""INSERT INTO mr_reflections(umo,content,status,next_review_at,created_at,updated_at,schedule_updated_at)
                              VALUES(?,?,'pending',100,50,50,50)""", (self.scope, "旧版本默认重排"))
        migrated = Reflection(self.store)
        self.assertEqual(len(migrated.due(now=200)), 1)
        self.assertEqual(migrated.due(now=200, scheduled_only=True), [])

    def test_feedback_opens_original_request_and_actual_stages_not_latest_question(self):
        question = self.message(1, "这句话是谁说的？")
        run = self.store.record_run("foreground", 10, {"request_id": "1", "question": question["plain_text"],
            "background": "候选记忆仍待核实", "tool_calls": [{"name": "memory"}]})
        self.store.append_run_step(run, 1, 11, "tool", "completed", "读取候选记忆",
                                   {"name": "memory", "result": {"kind": "semantic", "id": 7}})
        self.store.append_run_step(run, 2, 12, "inject", "completed", "交付背景", {"text": "保留说话人尚未确认的语境"})
        generated = self.message(2, "第一版生成内容", role="SYSTEM", request_id="1", event="generated")
        sent = self.message(3, "实际发送内容", role="BOT", request_id="1", event="sent")
        correction = self.message(4, "你把被引用的人和发送者弄反了。")
        latest = self.message(5, "明天几点集合？")
        self.store.record_run("foreground", 20, {"request_id": "5", "question": latest["plain_text"], "background": "新问题背景"})
        old_view = self.reflections.save({"content": "当时以为发言者和被引用者是同一人", "request_id": "1", "status": "resolved"})
        self.reflections.save({"content": "另一次互动", "request_id": "5"})
        context = self.reflections.feedback_context([generated, sent, correction, latest])
        self.assertEqual(len(context), 1)
        self.assertEqual(context[0]["request"]["id"], question["id"])
        self.assertEqual([item["id"] for item in context[0]["runs"]], [run])
        self.assertEqual([item["id"] for item in context[0]["past_reflections"]], [old_view["id"]])
        self.assertEqual(context[0]["past_reflections"][0]["status"], "resolved")
        self.assertEqual(context[0]["runs"][0]["injection_events"][0]["data"]["text"], "保留说话人尚未确认的语境")
        self.assertEqual([event["events"] for event in context[0]["response_events"]], [["generated"], ["sent"]])
        detailed = self.reflections.interaction(run_id=run, detailed=True)
        self.assertEqual(detailed["runs"][0]["steps"][0]["data"]["result"]["id"], 7)
        self.assertNotIn("tool_calls", detailed["runs"][0])
        self.assertEqual(detailed["response_events"][1]["plain_text"], "实际发送内容")
        self.assertEqual(detailed["subsequent_context"][0]["id"], correction["id"])
        self.assertEqual(self.reflections.feedback_context([latest]), [])
        self.assertEqual(self.reflections.interaction(request_id="missing")["status"], "not_found")
        legacy = self.store.record_run("foreground", 30, {"request_id": "legacy", "tool_calls": [{"name": "memory"}]})
        self.assertEqual(self.reflections.interaction(run_id=legacy, detailed=True)["runs"][0]["tool_calls"], [{"name": "memory"}])
        with self.assertRaisesRegex(ValueError, "different request"):
            self.reflections.interaction(request_id="5", run_id=run)

    def test_learning_run_can_be_reopened_without_a_foreground_request(self):
        source = self.message(1, "待分析的交流")
        run = self.store.record_run("background", 10, {"status": "partial", "detail": "等待额度恢复",
            "written": [{"kind": "semantic", "id": 7, "summary": "已经形成的理解"}],
            "response_text": "下一次继续检查后半段", "messages": [{"role": "user", "content": "模型输入"}]})
        self.store.append_run_step(run, 1, 11, "input", "completed", "本批材料", {"messages": [source]})
        self.store.append_run_step(run, 2, 12, "tool", "completed", "读取上下文", {"name": "context"})
        self.store.append_run_step(run, 3, 13, "write", "completed", "保存进度",
                                  {"checkpoint": "前半段已理解，继续核查后半段", "completed_ids": [source["id"]]})
        compact = self.reflections.interaction(run_id=run)
        self.assertEqual(compact["status"], "recorded")
        saved = compact["runs"][0]
        self.assertEqual(saved["material_ids"], [source["id"]])
        self.assertEqual(saved["memory_refs"], [{"kind": "semantic", "id": 7}])
        self.assertEqual(saved["checkpoint"], "前半段已理解，继续核查后半段")
        self.assertNotIn("steps", saved)
        detail = self.reflections.interaction(run_id=run, detailed=True)["runs"][0]
        self.assertEqual(detail["steps"][1]["data"]["name"], "context")
        self.assertNotIn("messages", detail)


if __name__ == "__main__":
    unittest.main()
