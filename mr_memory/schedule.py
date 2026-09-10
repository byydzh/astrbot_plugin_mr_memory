"""Preferred learning hours in the running service's local civil time."""
from __future__ import annotations

from datetime import date, timedelta
import time


def clock_minutes(value: str) -> int:
    """Read a configured HH:MM clock time, without supplying another timezone."""
    parts = str(value).strip().split(":")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError("后台工作时段请填写 HH:MM，例如 00:00 或 06:00")
    hour, minute = map(int, parts)
    if not 0 <= hour < 24 or not 0 <= minute < 60:
        raise ValueError("后台工作时段须在 00:00 至 23:59 之间")
    return hour * 60 + minute


def work_window(config, now: float | None = None) -> dict:
    now = time.time() if now is None else float(now)
    local = time.localtime(now)
    start = clock_minutes(config.get("learning_window_start", "00:00"))
    end = clock_minutes(config.get("learning_window_end", "06:00"))
    minute = local.tm_hour * 60 + local.tm_min
    enabled = bool(config.get("learning_window_enabled", True))
    opened = (not enabled or start == end or
              (start <= minute < end if start < end else minute >= start or minute < end))
    next_start = None
    if not opened:
        day = date(local.tm_year, local.tm_mon, local.tm_mday)
        if minute >= start:
            day += timedelta(days=1)
        # mktime uses the service process's timezone, including the rules on
        # this date; adding a fixed 24 hours would shift windows across DST.
        next_start = time.mktime((day.year, day.month, day.day, start // 60, start % 60, 0, 0, 0, -1))
    return {"open": opened, "next_start": next_start,
            "start": f"{start // 60:02d}:{start % 60:02d}", "end": f"{end // 60:02d}:{end % 60:02d}",
            "timezone": time.strftime("%Z", local),
            "local_now": time.strftime("%Y-%m-%d %H:%M:%S %z", local)}
