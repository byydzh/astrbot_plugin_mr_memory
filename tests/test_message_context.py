"""Producer context survives storage, retrieval and selected-source handoff."""
import tempfile
import unittest
from pathlib import Path

from mr_memory.store import Store
from mr_memory.handoff import selected_context


class MessageContextTests(unittest.TestCase):
    def test_context_roundtrip_preserves_transport_identity_and_revises_in_place(self):
        scratch = Path(__file__).resolve().parents[1] / ".pytest_cache"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            store = Store(Path(directory) / "group.db", "test:GroupMessage:42")
            try:
                row = dict(platform_id="test", message_id="m1", sender_id="bot-account",
                           sender_name="CatGPT", sent_at=1000, role="BOT", plain_text="我在事务所练习。",
                           content=[{"type": "text", "text": "我在事务所练习。"}])
                original = store.append_message(row)
                context = {"type": "message_context", "version": 1, "producer": "example_rp",
                           "mode": "roleplay", "session_id": "call1", "turn_id": "turn1",
                           "speaker": {"id": "actor-a", "name": "角色甲"}, "utterance_id": "turn1"}
                annotated = store.annotate_message_context(original["source_key"], context)
                repeated = store.annotate_message_context(original["source_key"], context)
                self.assertEqual(annotated["id"], original["id"])
                self.assertEqual(repeated["revision_no"], annotated["revision_no"])
                self.assertEqual(annotated["sender_id"], "bot-account")
                self.assertEqual(len(store.recent()), 1)
                self.assertEqual(store.search_messages(["事务所"])[0]["content"][-1], context)
                self.assertEqual(store.pending_messages(5)[0]["content"][-1], context)
                self.assertIn("角色甲", store.pending_embeddings("synthetic-model")[0]["text"])
                raw = store.memory("message", original["id"])
                self.assertIn("角色甲", raw["title"])
                self.assertEqual(raw["source_contexts"][0]["speaker"], context["speaker"])
                selected = selected_context(store, {"kind": "message", "id": original["id"]})
                self.assertEqual(selected["content"][-1], context)
                store.save_memories([{"kind": "episode", "title": "电话对话", "content": "角色甲说在事务所练习。",
                                      "source_ids": [original["id"]]}], mark_processed=False)
                memory = store.memory("episode", 1, include_sources=False)
                self.assertEqual(memory["source_contexts"][0]["speaker"], context["speaker"])
                selected_memory = selected_context(store, {"kind": "episode", "id": 1})
                self.assertEqual(selected_memory["source_contexts"][0]["utterance_id"], "turn1")
                reply = store.append_message({**row, "message_id": "m2", "sender_id": "user", "role": "USER",
                                              "plain_text": "好的", "reply_to": "m1", "content": []})
                self.assertEqual(reply["reply_to_message"]["message_context"], [context])
                quoted = store.resolve_quotes([{"type": "reply", "message_id": "m1"}], "test")
                self.assertEqual(quoted[0]["message_context"], [context])
                self.assertIsNone(store.annotate_message_context("missing", context))
            finally:
                store.close()
