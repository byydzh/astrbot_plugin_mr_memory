import unittest
import uuid
from pathlib import Path

from mr_memory.agent import MemoryAgent
from mr_memory.store import Store


class SearchContextTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / ".pytest_cache"
        scratch.mkdir(exist_ok=True)
        self.scratch = scratch / ("search-context-" + uuid.uuid4().hex)
        self.scratch.mkdir()
        self.store = Store(self.scratch / "group.db", "test:GroupMessage:42")

    def tearDown(self):
        self.store.close()
        for file in self.scratch.iterdir():
            file.unlink()
        self.scratch.rmdir()

    def message(self, number, text, *, role="USER", sender="author", event=None):
        content = [{"type": "text", "text": text}]
        if event:
            content.extend([{"type": "response_to", "message_id": "1"},
                            {"type": "bot_event", "event": event}])
        return self.store.append_message(dict(
            platform="test", platform_id="test", umo=self.store.umo, group_id="42",
            message_id=str(number), sender_id=sender, sender_name=sender,
            sent_at=100 + number, plain_text=text, role=role, content=content))

    async def test_filtered_hits_keep_adjacent_correction_and_one_copy_of_each_stage(self):
        question = self.message(1, "画这张人物图")
        draft = self.message(2, "这是青叶的角色", sender="bot", role="SYSTEM", event="generated")
        sent = self.message(3, "这是青叶的角色", sender="bot", role="BOT", event="sent")
        correction = self.message(4, "不是，是我朋友的", sender="author")
        following = self.message(5, "我画的是朋友", sender="author")
        future = self.message(6, "后来的另一轮讨论")
        agent = MemoryAgent(None, self.store)
        result = await agent._execute("search_messages", {
            "terms": ["青叶"], "sender_id": "bot", "roles": ["BOT", "SYSTEM"],
            "end_at": 1000}, future["sent_at"])
        self.assertEqual(result["matches"], [sent["id"], draft["id"]])
        self.assertEqual([row["id"] for row in result["context"]],
                         [draft["id"], sent["id"], correction["id"]])
        self.assertEqual(result["context"][2]["sender_id"], "author")
        self.assertEqual(result["context"][2]["plain_text"], "不是，是我朋友的")
        self.assertEqual([part["event"] for row in result["context"] for part in row["content"]
                          if part["type"] == "bot_event"], ["generated", "sent"])
        self.assertEqual({part["message_id"] for row in result["context"] for part in row["content"]
                          if part["type"] == "response_to"}, {"1"})
        self.assertEqual(result["window"]["end_at"], future["sent_at"])
        expanded = await agent._execute("search_messages", {
            "terms": ["青叶"], "sender_id": "bot", "roles": ["BOT", "SYSTEM"],
            "before": 1, "after": 3, "end_at": 1000}, future["sent_at"])
        self.assertEqual([row["id"] for row in expanded["context"]],
                         [question["id"], draft["id"], sent["id"], correction["id"], following["id"]])

    def test_explicit_window_changes_context_without_changing_legacy_search_api(self):
        self.message(1, "之前的对话")
        hit = self.message(2, "纸鸢活动", sender="speaker")
        self.message(3, "下一句")
        query = {"terms": ["纸鸢"], "sender_id": "speaker"}
        self.assertEqual(self.store.search_messages(**query), [hit])
        result = self.store.search_message_context(before=0, after=0, **query)
        self.assertEqual(result["matches"], [hit["id"]])
        self.assertEqual(result["context"], [hit])
        self.assertEqual(self.store.search_message_context(terms=["不存在"])["context"], [])


if __name__ == "__main__":
    unittest.main()
