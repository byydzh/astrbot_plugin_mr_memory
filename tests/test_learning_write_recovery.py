import copy
import json
import unittest
import uuid
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mr_memory.agent import MemoryAgent, checkpoint_learning_task, estimate_learning_input, _memory_output
from mr_memory.learning_writes import LearningWriter
from mr_memory.store import Store
from mr_memory.trace import RunTrace


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
        return {"items": [{"kind": "episode", "summary": "约在西门见", "source_ids": [999]},
                          {"kind": "semantic", "content": "群友2会带桌游", "source_ids": [self.ids[1]]}],
                "progress": {"completed_ids": self.ids, "checkpoint": "已读完，原文编号可继续查阅"}, "finish": True}

    async def test_native_malformed_save_survives_sdk_empty_args_and_is_repaired(self):
        good = {"items": [{"kind": "episode", "summary": "周末在西门玩桌游", "source_ids": self.ids}],
                "progress": {"completed_ids": self.ids, "checkpoint": "见面地点与携带物已保存"}, "finish": True}
        wire = json.dumps(good, ensure_ascii=False) + ","
        owner = self

        class RepairAgent(MemoryAgent):
            async def _model_turn(agent, messages, *unused, **kwargs):
                agent.requests.append(copy.deepcopy(messages))
                if len(agent.requests) == 1:
                    return SimpleNamespace(completion_text="", tools_call_name=["remember"],
                        tools_call_args=[{}], tools_call_ids=["broken"], reasoning_content="保留原始思考",
                        usage={"input_other": 100, "output": 20}, raw_completion={"choices": [{
                            "finish_reason": "tool_calls", "message": {"tool_calls": [{"id": "broken",
                            "function": {"name": "remember", "arguments": wire}}]}}]})
                owner.assertEqual(owner.store.pending_status()["count"], 2)
                previous = next(row for row in messages if row.get("tool_calls"))
                owner.assertEqual(previous["tool_calls"][0]["function"]["arguments"], wire)
                owner.assertEqual(previous["reasoning_content"], "保留原始思考")
                failure = next(row for row in messages if row.get("tool_call_id") == "broken")
                owner.assertEqual(json.loads(failure["content"])["status"], "error")
                return SimpleNamespace(completion_text=json.dumps(good, ensure_ascii=False),
                    tools_call_name=[], reasoning_content="修正后保存", usage={"input_other": 120, "output": 30},
                    raw_completion={"choices": [{"finish_reason": "stop"}]})

        writer = LearningWriter(self.store, self.sources, learning_kind="background")
        with self.assertRaisesRegex(ValueError, "empty arguments"):
            writer.apply({}, "lost")
        agent = RepairAgent(None, self.store, max_turns=2)
        agent.requests = []
        with patch("mr_memory.agent._tool_set", return_value=object()):
            result = await agent.consolidate(self.rows, {}, task=self.task)
        self.assertEqual(result.status, "completed", result.detail)
        self.assertEqual(result.tool_calls[0]["raw_arguments"], wire)
        self.assertEqual(len(result.written), 1)
        self.assertEqual(self.store.pending_status()["count"], 0)

    async def test_last_affordable_call_saves_progress_through_native_write_tool(self):
        task = copy.deepcopy(self.task)
        task["continuation"] = {"input_tokens_per_byte": 0.3, "output_token_reserve": 6000}
        estimate = estimate_learning_input(self.rows, {}, task=task)
        owner = self
        schema = object()

        class ClosingAgent(MemoryAgent):
            async def _model_turn(agent, messages, prompt, tools, turn, **kwargs):
                owner.assertIs(tools, schema)
                owner.assertNotIn("tool_choice", kwargs)
                owner.assertTrue(json.loads(messages[-1]["content"])["resource_state"]["save_this_turn"])
                owner.assertEqual(turn, 1)
                return SimpleNamespace(completion_text=json.dumps({"items": [], "finish": True,
                    "progress": {"completed_ids": owner.ids, "checkpoint": "两条临时安排已理解，无长期信息"}}, ensure_ascii=False),
                    tools_call_name=[], reasoning_content="", usage={"input_other": estimate, "output": 80},
                    raw_completion={"choices": [{"finish_reason": "stop"}]})

        with patch("mr_memory.agent._tool_set", return_value=schema) as tool_set:
            result = await ClosingAgent(None, self.store, max_output_tokens=12000).consolidate(self.rows, {}, task=task,
                token_budget=estimate * 2 + 10000)
        self.assertEqual(result.status, "completed", result.detail)
        self.assertEqual(self.store.pending_status()["count"], 0)
        self.assertEqual(result.model_attempts, 1)
        self.assertEqual(result.continuation["output_token_reserve"], 6000)
        tool_set.assert_called_once_with(learning=True)

    async def test_final_native_dsml_preserves_complete_draft_and_progress(self):
        args = {"items": [{"kind": "episode", "summary": "小桥说 <北门> & 西门有别", "source_ids": self.ids}],
                "progress": {"completed_ids": self.ids, "checkpoint": "原文与作者已记录"}, "finish": True}
        parameters = "\n".join('<｜｜DSML｜｜ parameter name="'+key+'" string="false">'+
            json.dumps(value, ensure_ascii=False)+'</｜｜DSML｜｜ parameter>' for key, value in args.items())
        text = '<｜｜DSML｜｜ calls>\n<｜｜DSML｜｜ invoke name="remember">'+parameters+\
               '</｜｜DSML｜｜ invoke>\n</｜｜DSML｜｜ calls>'

        class NativeFinalAgent(MemoryAgent):
            async def _model_turn(agent, *unused, **kwargs):
                return SimpleNamespace(completion_text=text, tools_call_name=[], reasoning_content="思考保留",
                    usage={"input_other": 100, "output": 50}, raw_completion={"choices": [{"finish_reason": "stop"}]})

        with patch("mr_memory.agent._tool_set", return_value=object()):
            result = await NativeFinalAgent(None, self.store).consolidate(self.rows, {}, task=self.task)
        self.assertEqual(result.status, "completed", result.detail)
        self.assertEqual(result.written[0]["summary"], args["items"][0]["summary"])
        self.assertEqual(self.store.pending_status()["count"], 0)

    async def test_malformed_final_receives_failed_write_receipt_before_retry(self):
        owner = self

        class FinalRepairAgent(MemoryAgent):
            async def _model_turn(agent, messages, *unused, **kwargs):
                agent.calls += 1
                if agent.calls == 1:
                    text = '{"items":[],"progress":'
                else:
                    owner.assertEqual(owner.store.pending_status()["count"], 2)
                    owner.assertTrue(any("最后输出未保存" in (row.get("content") or "") for row in messages))
                    text = json.dumps({"items": [], "progress": {"completed_ids": owner.ids, "checkpoint": "已理解"}})
                return SimpleNamespace(completion_text=text, tools_call_name=[], reasoning_content="",
                    usage={"input_other": 100, "output": 10}, raw_completion={"choices": [{"finish_reason": "stop"}]})

        agent = FinalRepairAgent(None, self.store, max_turns=2); agent.calls = 0
        with patch("mr_memory.agent._tool_set", return_value=object()):
            result = await agent.consolidate(self.rows, {}, task=self.task)
        self.assertEqual(result.status, "completed", result.detail)
        self.assertEqual(self.store.pending_status()["count"], 0)

    def test_missing_outer_brace_keeps_values_but_incomplete_values_do_not_parse(self):
        complete = {"items": [{"kind": "episode", "summary": "保留原话}与括号", "source_ids": self.ids}],
                    "progress": {"completed_ids": self.ids, "checkpoint": "原文已理解"}}
        repaired = _memory_output(json.dumps(complete, ensure_ascii=False)[:-1])
        self.assertEqual(repaired["items"], complete["items"])
        self.assertEqual(repaired["progress"], complete["progress"])
        self.assertTrue(repaired["_format_repair"])
        for incomplete in ('{"items":[{"summary":"没说完', '{"items":[],"progress":',
                           '{"items":[],"progress":{"completed_ids":[1,'):
            with self.subTest(incomplete=incomplete), self.assertRaises(ValueError):
                _memory_output(incomplete)

    async def test_large_read_keeps_full_result_address_and_room_to_save(self):
        task = copy.deepcopy(self.task)
        task["continuation"] = {"input_tokens_per_byte": 0.3}
        trace = RunTrace(self.store, "background", time.time(), {})
        await trace.start()
        owner = self
        large = [{"kind": "semantic", "id": 17, "summary": "完整而尚未交给模型的旧记忆" * 12000}]

        class BudgetAgent(MemoryAgent):
            async def _model_turn(agent, messages, *unused, **kwargs):
                agent.calls += 1
                if agent.calls == 1:
                    return SimpleNamespace(completion_text="", tools_call_name=["search_memories"],
                        tools_call_args=[{"terms": ["西门"]}], tools_call_ids=["large"], reasoning_content="继续查以前的安排",
                        usage={"input_other": 4500, "output": 500}, raw_completion={"choices": [{"finish_reason": "tool_calls"}]})
                receipt = json.loads(next(row['content'] for row in messages if row.get('tool_call_id')=='large'))
                owner.assertEqual(receipt["status"], "deferred_for_budget")
                owner.assertNotIn(large[0]["summary"], json.dumps(messages, ensure_ascii=False))
                agent.ref = receipt["result_ref"]
                return SimpleNamespace(completion_text=json.dumps({"items": [], "progress": {
                    "completed_ids": [owner.ids[0]], "checkpoint": "地点原话已理解；待继续读保存的查询结果。"}},ensure_ascii=False),
                    tools_call_name=[], reasoning_content="", usage={"input_other": 4700, "output": 500},
                    raw_completion={"choices": [{"finish_reason": "stop"}]})

            async def _execute(agent, *unused):
                return large

        agent = BudgetAgent(None, self.store); agent.trace = trace; agent.calls = 0
        with patch("mr_memory.agent._tool_set", return_value=object()):
            result = await agent.consolidate(self.rows, {}, task=task, token_budget=16000)
        await trace.finish({"status": result.status})
        self.assertEqual(agent.calls, 2)
        self.assertEqual(self.store.learning_task("background")["completed_ids"], [self.ids[0]])
        read = self.store.reflections.interaction(**agent.ref)
        self.assertEqual(read["data"]["result"], large)
        self.assertEqual(self.store.pending_status()["count"], 1)

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

    async def test_completed_batch_transfers_pending_draft_to_next_batch_without_losing_it(self):
        writer = LearningWriter(self.store, self.sources, learning_kind="background", run_id=7)
        writer.apply(self.args(), "save")
        self.store.finish_learning_task("background")
        self.assertIsNone(self.store.learning_task("background"))
        self.store.close()
        self.store = Store(self.scratch / "group.db", "test:GroupMessage:42")
        task = self.store.start_learning_task("background", [])
        writes = task["continuation"]["write_state"]
        self.assertEqual(writes["pending_items"]["save:0"]["origin_run_id"], 7)
        self.assertEqual(writes["pending_items"]["save:0"]["saved_memory_refs"][0]["kind"], "semantic")
        self.assertFalse(self.store.load_working_state()["pending_learning_drafts"])
        resumed = LearningWriter(self.store, {}, learning_kind="background", **writes)
        repaired = resumed.apply({"retry": [{"pending_id": "save:0", "changes": {"source_ids": [self.ids[0]]}}]}, "repair")
        self.assertEqual(len(repaired.written), 1)
        self.assertEqual(self.store.learning_task("background")["material_ids"], [])
        self.assertFalse(resumed.pending_items)

    async def test_legacy_deferred_progress_resumes_without_requiring_a_model_call(self):
        self.store.update_learning_task("background", {"continuation": {"write_state": {
            "deferred_progress": self.args()["progress"],
            "pending_items": {"old:0": {"item": self.args()["items"][0], "item_index": 0, "detail": "missing source_ids"}}}}})
        task = self.store.resume_learning_task("background")
        self.assertEqual(task["completed_ids"], self.ids)
        self.assertFalse(task["continuation"]["write_state"]["deferred_progress"])
        self.assertIn("old:0", task["continuation"]["write_state"]["pending_items"])
        self.assertEqual(self.store.pending_status()["count"], 0)

    async def test_final_json_checkpoint_keeps_later_tools_and_can_end_with_pending_draft(self):
        args = self.args()
        writer = LearningWriter(self.store, self.sources, learning_kind="background")
        writer.apply(args, "save")
        task = self.store.learning_task("background")
        final = {"role": "assistant", "content": json.dumps(args, ensure_ascii=False)}
        later = {"role": "tool", "tool_call_id": "later", "content": "后来读到的原文"}
        task["continuation"].update({"conversation": [{"role": "user", "content": "旧材料" * 10000}, final,
            {"role": "assistant", "tool_calls": [{"id": "later", "function": {"name": "context", "arguments": "{}"}}]}, later]})
        compact = checkpoint_learning_task(task)
        self.assertEqual(compact["continuation"]["checkpoint_tail"][0], final)
        self.assertIn(later, compact["continuation"]["checkpoint_tail"])
        class FinishAgent(MemoryAgent):
            async def _model_turn(agent, *unused, **kwargs):
                return SimpleNamespace(completion_text='{"items":[]}', tools_call_name=[], reasoning_content="",
                    usage={"input_other": 100, "output": 10}, raw_completion={"choices": [{"finish_reason": "stop"}]})
        with patch("mr_memory.agent._tool_set", return_value=object()):
            result = await FinishAgent(None, self.store, max_turns=1).consolidate([], {}, task=compact)
        self.assertEqual(result.status, "completed", result.detail)
        self.assertIn("save:0", result.continuation["write_state"]["pending_items"])

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
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM mr_memory_objects WHERE kind='association'").fetchone()[0], 1)

    async def test_duplicate_retry_does_not_create_two_associations(self):
        writer = LearningWriter(self.store, self.sources, learning_kind="background")
        writer.apply({"items": [{"kind": "association", "source": {"label": "小桥"},
            "target": {"label": "桌游"}, "relation": "携带", "statement": "小桥说会带桌游", "source_ids": [999]}]}, "draft")
        retry = {"pending_id": "draft:0", "changes": {"source_ids": self.ids}}
        outcome = writer.apply({"retry": [retry, retry]}, "repair")
        self.assertEqual(len(outcome.written), 1)
        self.assertEqual(len(outcome.rejected), 1)
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM mr_memory_objects WHERE kind='association'").fetchone()[0], 1)

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
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM mr_memory_objects WHERE kind='episode'").fetchone()[0], 1)

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
