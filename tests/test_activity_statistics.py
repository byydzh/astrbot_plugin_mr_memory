from copy import deepcopy
from datetime import datetime
import unittest
from zoneinfo import ZoneInfo

from mr_memory.activity_statistics import activity_aggregate_id, validate_activity_window_statistics


class ActivityStatisticsTests(unittest.TestCase):
    def test_content_hash_and_exact_population_and_time_fields_are_checked(self) -> None:
        local = datetime(2025, 2, 3, 2, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
        epoch = int(local.timestamp())
        aggregate = {
            "schema_version": "mr-memory.activity-window.v1",
            "authority": "HOST_SQLITE_SNAPSHOT",
            "basis": "all_snapshot_visible_direct_speaker_messages",
            "scope": {"umo": "test:GroupMessage:synthetic", "participant_key": "synthetic-person",
                      "start_sent_at": epoch - 3600, "end_sent_at_exclusive": epoch + 3600,
                      "message_upper_bound": 10},
            "timezone": "Asia/Shanghai", "source_count": 1,
            "hour_histogram": {f"{hour:02d}": int(hour == 2) for hour in range(24)},
            "daily": [{"local_date": "2025-02-03", "source_count": 1,
                       "first_sent_at": epoch, "last_sent_at": epoch,
                       "first_local_datetime": local.isoformat(), "last_local_datetime": local.isoformat()}],
            "source_revision_sha256": "a" * 64,
        }
        aggregate["aggregate_id"] = activity_aggregate_id(aggregate)
        self.assertEqual(validate_activity_window_statistics(aggregate), aggregate)
        tampered = deepcopy(aggregate)
        tampered["source_count"] = 2
        with self.assertRaisesRegex(ValueError, "content hash"):
            validate_activity_window_statistics(tampered)
        tampered["aggregate_id"] = activity_aggregate_id(tampered)
        with self.assertRaisesRegex(ValueError, "histogram total"):
            validate_activity_window_statistics(tampered)
        wrong_time = deepcopy(aggregate)
        wrong_time["daily"][0]["first_local_datetime"] = "2025-02-03T03:30:00+08:00"
        wrong_time["aggregate_id"] = activity_aggregate_id(wrong_time)
        with self.assertRaisesRegex(ValueError, "local time differs"):
            validate_activity_window_statistics(wrong_time)
