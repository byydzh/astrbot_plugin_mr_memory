"""Focused migration checks using synthetic groups; never send QQ messages."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mr_memory.identity import IdentityBindings, import_unlearned_messages, match_members
from mr_memory.store import Store


def person(uid, name):
    return {"account_id": uid, "name": name, "nickname": name, "card": name, "account_type": "USER"}


def message(umo, uid, name, mid, stamp=100):
    platform, _, group = umo.split(":")
    return dict(umo=umo, platform_id=platform, group_id=group, platform="qq_official" if platform == "new" else "aiocqhttp",
                sender_id=uid, sender_name=name, platform_nickname=name, group_card=name,
                message_id=mid, sent_at=stamp, plain_text="一起讨论天文", content=[], role="USER")


class MigrationTests(unittest.TestCase):
    def test_names_are_unique_and_later_collision_withdraws_only_automatic_link(self):
        old = [person("100", "小林"), person("200", "重名"), person("300", "重名")]
        now = [person("OPEN1", "小林"), person("OPEN2", "重名"), person("OPEN3", "新朋友")]
        matches = match_members(old, now)
        self.assertEqual(set(matches), {"OPEN1"})
        self.assertEqual(match_members(old, [person("OPEN1", "改名了")], matches), matches)
        now.append(person("OPEN4", "小林"))
        self.assertFalse(match_members(old, now, matches))
        matches["OPEN1"]["method"] = "manual"
        self.assertEqual(set(match_members(old, now, matches)), {"OPEN1"})

    def test_bound_identity_retrieves_both_histories_without_rewriting_authors_or_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = Store(root / "old.db", "old:GroupMessage:123")
            new = Store(root / "new.db", "new:GroupMessage:OPEN_GROUP")
            try:
                original = old.append_message(message(old.umo, "100", "小林", "old-message"))
                memory = old.save_memories([dict(kind="episode", content="旧群共同观星", source_ids=[original["id"]])], mark_processed=False)[0]
                old.edit_member("100", {"preferred_name": "林同学", "description": "喜欢天文"}, actor="admin")
                old.cache_roster([{"user_id": "100", "nickname": "小林", "card": "小林"}], 150)
                self.assertIsNotNone(old.roster())
                # Same nickname, distinct real author ID; migration preserves both.
                new.append_message(message(new.umo, "OPEN1", "小林", "new-message", 200))
                binding = IdentityBindings(root / "bindings.json")
                snapshot = old.profiles.members()
                self.assertEqual(import_unlearned_messages(new, old), 1)
                import_unlearned_messages(new, old)
                self.assertEqual(len(old.recent()), 2)
                binding.bind(new.umo, old.umo, snapshot)
                binding.refresh(new.umo, new.profiles.members())
                binding.configure(old)
                self.assertIsNone(old.roster())  # Old roster cannot hide newly observed OpenIDs.
                self.assertEqual(binding.canonical(new.umo), old.umo)
                self.assertEqual({r["sender_id"] for r in old.search_messages(sender_id="OPEN1")}, {"100", "OPEN1"})
                self.assertEqual({r["sender_id"] for r in old.search_messages(related_account_id="OPEN1")}, {"100", "OPEN1"})
                self.assertEqual(old.search_memories(related_account_id="OPEN1")[0]["id"], memory["id"])
                self.assertEqual(old.memory("episode", memory["id"])["source_ids"], [original["id"]])
                self.assertEqual(old.activity(account_id="OPEN1", start_at=0, end_at=300)["message_count"], 2)
                profiles = old.members(["OPEN1"])
                self.assertEqual(next(p for p in profiles if p["account_id"] == "OPEN1")["description"], "喜欢天文")
                # A nickname association does not grant authority to erase an old account.
                self.assertEqual(old.linked_accounts("OPEN1", confirmed=True), ["OPEN1"])
                restored = IdentityBindings(root / "bindings.json")
                restored.member(new.umo, "OPEN1", "")
                restored.configure(old)
                self.assertEqual([r["sender_id"] for r in old.search_messages(sender_id="OPEN1")], ["OPEN1"])
                restored.member(new.umo, "OPEN1", "100")
                restored.configure(old)
                self.assertEqual(set(old.linked_accounts("OPEN1", confirmed=True)), {"100", "OPEN1"})
                self.assertEqual(new.recent()[0]["sender_id"], "OPEN1")
            finally:
                old.close(); new.close()

    def test_existing_new_semantic_memory_is_not_overwritten(self):
        old, new = Store(":memory:", "old:GroupMessage:1"), Store(":memory:", "new:GroupMessage:G")
        try:
            row = new.append_message(message(new.umo, "OPEN", "同学", "1"))
            new.save_memories([dict(kind="episode", content="新群已经整理", source_ids=[row["id"]])], mark_processed=False)
            with self.assertRaisesRegex(ValueError, "已有独立整理"):
                import_unlearned_messages(new, old)
            self.assertEqual(old.recent(), [])
            self.assertEqual(len(new.search_memories()), 1)
        finally:
            old.close(); new.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
