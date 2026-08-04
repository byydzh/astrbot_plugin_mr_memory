from __future__ import annotations

import unittest

from mr_memory.scope import (
    GroupMemoryScope,
    GroupScopeError,
)


class GroupMemoryScopeTests(unittest.TestCase):
    def test_scope_is_derived_from_the_current_group_event(self) -> None:
        first = GroupMemoryScope.from_event_values(
            unified_msg_origin="adapter-a:GroupMessage:group-a",
            platform_id="adapter-a",
            group_id="group-a",
        )
        second = GroupMemoryScope.from_event_values(
            unified_msg_origin="adapter-a:GroupMessage:group-b",
            platform_id="adapter-a",
            group_id="group-b",
        )
        self.assertEqual(first.key, "adapter-a:GroupMessage:group-a")
        self.assertNotEqual(first.key, second.key)
        self.assertEqual(len(first.storage_id), 64)
        self.assertNotEqual(first.storage_id, second.storage_id)

    def test_non_group_or_incomplete_scope_is_rejected(self) -> None:
        cases = (
            {"unified_msg_origin": "", "platform_id": "p", "group_id": "g"},
            {"unified_msg_origin": "u", "platform_id": "", "group_id": "g"},
            {"unified_msg_origin": "u", "platform_id": "p", "group_id": ""},
        )
        for values in cases:
            with self.subTest(values=values), self.assertRaises(GroupScopeError):
                GroupMemoryScope.from_event_values(**values)


if __name__ == "__main__":
    unittest.main()
