"""Explicit member-profile command fields; no model calls."""
import unittest

from mr_memory.profiles import parse_profile_changes


class ProfileCommandTests(unittest.TestCase):
    def test_multiple_fields_and_clear(self):
        self.assertEqual(
            parse_profile_changes("称呼 小禾 别名 阿禾，禾同学 不用 旧外号 说明 平时 喜欢摄影"),
            {"preferred_name": "小禾", "aliases": ["阿禾", "禾同学"],
             "avoided_names": ["旧外号"], "description": "平时 喜欢摄影"},
        )
        self.assertEqual(parse_profile_changes("称呼\t-\n别名　-"), {"preferred_name": "", "aliases": []})
        self.assertEqual(parse_profile_changes(""), {})

    def test_quoted_field_names_are_values(self):
        self.assertEqual(parse_profile_changes('称呼 "别名" 说明 "别名 是说明中的词"'),
                         {"preferred_name": "别名", "description": "别名 是说明中的词"})
        self.assertEqual(parse_profile_changes("称呼 O'Neil"), {"preferred_name": "O'Neil"})

    def test_incomplete_edit_is_not_returned(self):
        for command in ["称呼 新称呼 别名", '称呼 新称呼 说明 "未闭合', "1002 称呼 另一人"]:
            with self.subTest(command=command), self.assertRaises(ValueError):
                parse_profile_changes(command)


if __name__ == '__main__':
    unittest.main()
