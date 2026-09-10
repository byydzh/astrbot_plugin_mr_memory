import time
import unittest
from unittest.mock import patch

from mr_memory.schedule import work_window


class ScheduleTests(unittest.TestCase):
    def at(self, hour, minute=0, **config):
        local = time.struct_time((2026, 9, 11, hour, minute, 0, 4, 254, -1))
        # The machine running tests may use a different timezone. Supply the
        # service's local clock so these expectations do not assume UTC+8.
        with patch("mr_memory.schedule.time.localtime", return_value=local), \
                patch("mr_memory.schedule.time.mktime", return_value=12345) as mktime:
            result = work_window(config, now=100)
        return result, mktime

    def test_default_window_follows_service_clock_and_ends_at_six(self):
        self.assertTrue(self.at(0)[0]["open"])
        self.assertTrue(self.at(5, 59)[0]["open"])
        closed, convert = self.at(6)
        self.assertFalse(closed["open"])
        self.assertEqual(closed["next_start"], 12345)
        self.assertTrue(closed["local_now"].startswith("2026-09-11 06:00:00"))
        self.assertEqual(convert.call_args.args[0][:6], (2026, 9, 12, 0, 0, 0))

    def test_custom_window_crosses_midnight_without_moving_daytime_to_tomorrow(self):
        config = {"learning_window_start": "22:30", "learning_window_end": "05:30"}
        self.assertTrue(self.at(22, 30, **config)[0]["open"])
        self.assertTrue(self.at(1, **config)[0]["open"])
        closed, convert = self.at(5, 30, **config)
        self.assertFalse(closed["open"])
        self.assertEqual(convert.call_args.args[0][:6], (2026, 9, 11, 22, 30, 0))

    def test_disabled_and_equal_bounds_allow_all_day(self):
        for config in ({"learning_window_enabled": False},
                       {"learning_window_start": "12:00", "learning_window_end": "12:00"}):
            result, convert = self.at(17, **config)
            self.assertTrue(result["open"])
            self.assertIsNone(result["next_start"])
            convert.assert_not_called()


if __name__ == "__main__":
    unittest.main()
