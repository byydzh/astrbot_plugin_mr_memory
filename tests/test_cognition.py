"""Continuity and graph-write contracts; these do not score model intelligence."""
import asyncio
import json
import unittest
from unittest.mock import patch

from mr_memory.agent import MemoryAgent, _json, estimate_learning_input
from mr_memory.learning_writes import LearningWriter
from mr_memory.store import Store
from tests.test_agent import Provider, response


class CognitionTests(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:", "test:GroupMessage:42")
        self.message = self.store.append_message({"platform": "test", "platform_id": "test",
            "umo": self.store.umo, "group_id": "42", "message_id": "request", "sender_id": "10",
            "sender_name": "甲", "sent_at": 1800000000, "plain_text": "继续讨论这个安排", "content": [], "role": "USER"})

    def tearDown(self):
        self.store.close()

    def run_agent(self, provider, *, task=None):
        async def run():
            with patch("mr_memory.agent._tool_set", return_value=object()):
                agent = MemoryAgent(provider, self.store)
                if task is not None:
                    return await agent.consolidate([self.message], {}, task=task)
                return await agent.reconstruct(self.message, [self.message], {})
        return asyncio.run(run())

    def test_shared_current_understanding_crosses_foreground_background_and_return(self):
        representation = {"roles": {"name": "任意结构", "id": 9, "participant_id": "语义内容，不是数据库字段"},
                          "possibilities": [{"created_at": 12, "content": "尚待展开"}]}
        first_provider = Provider([response(calls=[("complete", {"background": "当次语境", "items": [{
            "kind": "perspective", "title": "当前理解", "content": "初步理解", "representation": representation,
            "attention": {"reason": "下次继续想", "priority": "继续关注"}}], "recollection": "由模型自由组织的记录"}, "finish")])])
        first = self.run_agent(first_provider)
        self.assertEqual(first.status, "completed", first.detail)
        self.assertEqual(first.recollection, "由模型自由组织的记录")
        self.assertEqual(len(first_provider.requests), 1)
        self.assertNotIn("response_format", first_provider.requests[0][0])
        ref = first.written[0]
        task = self.store.start_learning_task("background", [self.message["id"]])
        learner = Provider([response(json.dumps({"items": [{"kind": ref["kind"], "id": ref["id"],
            "revision_no": 1, "content": "在后续经历中改写后的理解", "representation": {**representation, "new_relation": "可继续改变"}}],
            "progress": {"completed_ids": [self.message["id"]]}, "recollection": {"变化": "重新理解了原来的联系"}}))])
        learned = self.run_agent(learner, task=task)
        self.assertEqual(learned.status, "completed", learned.detail)
        learner_workspace = next(json.loads(m["content"])["workspace"] for m in learner.requests[0][0]["messages"]
            if m.get("role") == "user" and '"workspace":' in m.get("content", ""))
        self.assertEqual(learner_workspace["active"][0]["memory"]["representation"], representation)
        next_provider = Provider([response('{"background":"后续语境"}')])
        later = self.run_agent(next_provider)
        self.assertEqual(later.status, "completed", later.detail)
        sent = "\n".join(m.get("content", "") for m in next_provider.requests[0][0]["messages"])
        self.assertIn("在后续经历中改写后的理解", sent)
        self.assertNotIn('"summary":"初步理解"', sent)
        self.assertIn('"new_relation":"可继续改变"', sent)
        self.assertEqual(json.loads(_json({"representation": representation}))["representation"], representation)
        self.assertEqual(json.loads(_json({"attention": representation}))["attention"], representation)
        self.assertGreater(estimate_learning_input([], {}, workspace=self.store.workspace()),
                           estimate_learning_input([], {}))
        self.assertEqual(len(self.store.reconsider(kind="background")["items"]), 1)

    def test_batch_links_survive_draft_retry_and_attention_does_not_revise_meaning(self):
        writer = LearningWriter(self.store, {}, foreground=True)
        outcome = writer.apply({"items": [
            {"kind": "episode", "handle": "experience", "content": "一段共同经历"},
            {"kind": "perspective", "content": "从经历形成的认识", "attention": True,
             "connections": [{"handle": "experience", "relation": "来自", "purpose": "basis"},
                             {"kind": "message", "id": self.message["id"], "relation": "原话", "purpose": "basis"}], "source_ids": [999]}]}, "first")
        self.assertEqual(len(outcome.written), 1)
        resumed = LearningWriter(self.store, {}, foreground=True)
        outcome = resumed.apply({"retry": [{"pending_id": "first:1", "changes": {"source_ids": []}}]}, "second")
        self.assertFalse(outcome.rejected, outcome.rejected)
        selected = self.store.workspace()["active"][0]["memory"]
        self.assertEqual(selected["basis"][0]["target"]["kind"], "episode")
        self.assertEqual(self.store.memory("message", self.message["id"])["content"], self.message["plain_text"])
        resumed.apply({"items": [{"kind": "perspective", "id": selected["id"], "attention": None}]}, "release")
        self.assertEqual(self.store.workspace()["active"], [])
        self.assertEqual(self.store.memory("perspective", selected["id"])["revision_no"], selected["revision_no"])

    def test_reconsider_joins_later_actual_reply_without_claiming_visibility_is_use(self):
        self.store.save_memories([{"kind": "perspective", "content": "可供理解的认识", "attention": True}], mark_processed=False)
        self.run_agent(Provider([response('{"background":"当前背景","recollection":{"想法":"还想再联系另一次经历"}}')]))
        self.store.append_message({"platform": "test", "platform_id": "test", "umo": self.store.umo,
            "group_id": "42", "message_id": "reply", "sender_id": "bot", "sender_name": "机器人",
            "sent_at": 1800000001, "plain_text": "实际发出的回答", "role": "BOT", "reply_to": "request",
            "content": [{"type": "response_to", "message_id": "request"}]})
        experience = self.store.reconsider(ref={"kind": "perspective", "id": 1})["items"][0]
        self.assertEqual(experience["actual_response"][0]["plain_text"], "实际发出的回答")
        self.assertEqual(experience["available_memories"][0]["revision_no"], 1)
        self.assertEqual(experience["recollection"]["想法"], "还想再联系另一次经历")

    def test_complete_on_last_allowed_turn_is_a_completed_delivery(self):
        provider = Provider([response(calls=[("workspace", {}, "read")]),
                             response(calls=[("complete", {"background": "本次语境"}, "finish")])])
        async def run():
            with patch("mr_memory.agent._tool_set", return_value=object()):
                return await MemoryAgent(provider, self.store, max_turns=2).reconstruct(self.message, [self.message], {})
        result = asyncio.run(run())
        self.assertEqual(result.status, "completed", result.detail)
        self.assertEqual(result.background, "本次语境")
        self.assertIsNone(provider.requests[-1][0].get("tool_choice"))
        resources = [json.loads(row["content"])["resource_state"]
                     for row in provider.requests[-1][0]["messages"]
                     if row.get("role") == "user" and row.get("content", "").startswith('{"resource_state":')]
        self.assertEqual(resources[-1]["further_model_rounds"], 0)
        self.assertGreater(resources[-1]["remaining_seconds"], 0)

    def test_null_address_in_pending_draft_does_not_break_later_reconstruction(self):
        writer = LearningWriter(self.store, {}, foreground=True)
        failed = writer.apply({"items": [{"kind": "episode", "id": None, "revision_no": None,
            "action": None, "content": "尚未保存的经历", "source_ids": [999]}]}, "draft")
        self.assertTrue(failed.rejected)
        provider = Provider([
            response(calls=[("remember", {"retry": [{"pending_id": "draft:0", "changes": {"source_ids": []}}]}, "repair")]),
            response(calls=[("complete", {"background": "已继续理解"}, "finish")])])
        result = self.run_agent(provider)
        self.assertEqual(result.status, "completed", result.detail)
        self.assertEqual(len(result.written), 1)
        self.assertFalse(result.pending_items)
        self.assertEqual(self.store.memory("episode", result.written[0]["id"])["content"], "尚未保存的经历")


if __name__ == "__main__":
    unittest.main()
