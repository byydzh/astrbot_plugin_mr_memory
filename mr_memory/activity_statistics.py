"""Content-addressed host statistics, separate from sampled raw evidence.

The source revision digest is SHA-256 over canonical JSON arrays followed by LF:
``[source_key, revision_no, content_sha256, sent_at, message_row_id]`` in
``(sent_at, message_row_id)`` order.  It identifies the complete SQL-visible
source set without pretending that a digest is an original source key.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import re
from typing import Mapping
from zoneinfo import ZoneInfo

from .snapshot import stable_sha256

ACTIVITY_STATISTICS_SCHEMA = "mr-memory.activity-window.v1"
_FIELDS = frozenset({
    "schema_version", "aggregate_id", "authority", "basis", "scope", "timezone",
    "source_count", "hour_histogram", "daily", "source_revision_sha256",
})
_SCOPE_FIELDS = frozenset({
    "umo", "participant_key", "start_sent_at", "end_sent_at_exclusive",
    "message_upper_bound",
})
_DAY_FIELDS = frozenset({
    "local_date", "source_count", "first_sent_at", "last_sent_at",
    "first_local_datetime", "last_local_datetime",
})


def activity_aggregate_id(value: Mapping[str, object]) -> str:
    return "activity-window:" + stable_sha256({
        key: item for key, item in value.items() if key != "aggregate_id"
    })


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"activity aggregate {field} must be an integer >= {minimum}")
    return value


def validate_activity_window_statistics(value: object) -> dict[str, object]:
    """Validate host descriptor contents; callers bind it to their snapshot.

This checks an authenticated host's self-consistent descriptor, not the actual
database source set.  The host must construct it from its frozen SQL scope and
perform its existing snapshot freshness check before certificate publication.
"""
    if not isinstance(value, Mapping) or set(value) != _FIELDS:
        raise ValueError("activity aggregate fields are invalid")
    if value["schema_version"] != ACTIVITY_STATISTICS_SCHEMA:
        raise ValueError("activity aggregate schema is invalid")
    if value["authority"] != "HOST_SQLITE_SNAPSHOT":
        raise ValueError("activity aggregate authority is invalid")
    if value["basis"] != "all_snapshot_visible_direct_speaker_messages":
        raise ValueError("activity aggregate basis is invalid")
    if value["timezone"] != "Asia/Shanghai":
        raise ValueError("activity aggregate timezone is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", str(value["source_revision_sha256"])):
        raise ValueError("activity aggregate source revision digest is invalid")
    if value["aggregate_id"] != activity_aggregate_id(value):
        raise ValueError("activity aggregate content hash does not match")
    scope = value["scope"]
    if not isinstance(scope, Mapping) or set(scope) != _SCOPE_FIELDS:
        raise ValueError("activity aggregate scope fields are invalid")
    for key in ("umo", "participant_key"):
        if not isinstance(scope[key], str) or not scope[key].strip():
            raise ValueError(f"activity aggregate scope {key} is required")
    start = _integer(scope["start_sent_at"], "start_sent_at")
    end = _integer(scope["end_sent_at_exclusive"], "end_sent_at_exclusive", minimum=1)
    _integer(scope["message_upper_bound"], "message_upper_bound")
    if end <= start or end - start > 31 * 86400:
        raise ValueError("activity aggregate window is invalid")
    source_count = _integer(value["source_count"], "source_count")
    histogram = value["hour_histogram"]
    if not isinstance(histogram, Mapping) or set(histogram) != {f"{hour:02d}" for hour in range(24)}:
        raise ValueError("activity aggregate hour histogram fields are invalid")
    if sum(_integer(count, "hour count") for count in histogram.values()) != source_count:
        raise ValueError("activity aggregate histogram total differs from source count")
    daily = value["daily"]
    if not isinstance(daily, list) or len(daily) > 32:
        raise ValueError("activity aggregate daily statistics are invalid")
    zone = ZoneInfo("Asia/Shanghai")
    day_count = 0
    previous_date = ""
    for day in daily:
        if not isinstance(day, Mapping) or set(day) != _DAY_FIELDS:
            raise ValueError("activity aggregate daily fields are invalid")
        local_date = day["local_date"]
        if not isinstance(local_date, str) or local_date <= previous_date:
            raise ValueError("activity aggregate daily dates must be ordered and unique")
        first = _integer(day["first_sent_at"], "first_sent_at")
        last = _integer(day["last_sent_at"], "last_sent_at")
        if not start <= first <= last < end:
            raise ValueError("activity aggregate daily boundary is outside its window")
        for field, epoch in (("first_local_datetime", first), ("last_local_datetime", last)):
            local = datetime.fromtimestamp(epoch, tz=zone)
            if local.date().isoformat() != local_date or local.isoformat() != day[field]:
                raise ValueError("activity aggregate local time differs from its epoch")
        previous_date = local_date
        day_count += _integer(day["source_count"], "daily source count", minimum=1)
    if day_count != source_count:
        raise ValueError("activity aggregate daily total differs from source count")
    return deepcopy(dict(value))
