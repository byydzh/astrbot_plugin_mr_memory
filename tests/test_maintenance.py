from __future__ import annotations

import unittest

from mr_memory.maintenance import scoped_job_key


class MaintenanceQueueTests(unittest.TestCase):
    def test_same_local_job_id_in_two_group_databases_does_not_collide(self) -> None:
        first = scoped_job_key(
            umo="byy_official:GroupMessage:10001",
            job_id=1,
        )
        second = scoped_job_key(
            umo="byy_official:GroupMessage:10002",
            job_id=1,
        )

        self.assertNotEqual(first, second)
        self.assertEqual(len({first, second}), 2)

    def test_empty_scope_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            scoped_job_key(umo=" ", job_id=1)


if __name__ == "__main__":
    unittest.main()
