from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .identity import sanitize_components
from .models import NormalizedMessage
from .scope import GroupMemoryScope


ANGEL_EYE_REQUIRED_COLUMNS = {
    "id",
    "group_id",
    "message_id",
    "time",
    "user_id",
    "nickname",
    "raw_json",
}


@dataclass(frozen=True, slots=True)
class AngelEyeGroupSnapshot:
    group_id: str
    messages: int
    senders: int
    oldest_at: int
    newest_at: int
    through_row_id: int
    history_exhausted: bool


def angel_eye_scope(*, platform_id: str, group_id: str) -> GroupMemoryScope:
    platform = str(platform_id or "").strip()
    group = str(group_id or "").strip()
    if not platform or ":" in platform or "!" in platform:
        raise ValueError("OneBot 平台实例 ID 无效")
    return GroupMemoryScope.from_event_values(
        unified_msg_origin=f"{platform}:GroupMessage:{group}",
        platform_id=platform,
        group_id=group,
    )


class AngelEyeHistorySource:
    """Read-only adapter for AngelEye's QQ history cache."""

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)

    def _connect(self) -> sqlite3.Connection:
        if not self.database_path.is_file():
            raise FileNotFoundError("未找到 AngelEye 群聊历史缓存")
        connection = sqlite3.connect(
            f"file:{self.database_path.resolve()}?mode=ro",
            uri=True,
        )
        connection.row_factory = sqlite3.Row
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(messages)")
        }
        missing = ANGEL_EYE_REQUIRED_COLUMNS - columns
        if missing:
            connection.close()
            raise ValueError(
                "AngelEye 历史表缺少字段: " + ", ".join(sorted(missing))
            )
        return connection

    def inspect(self) -> list[AngelEyeGroupSnapshot]:
        connection = self._connect()
        try:
            try:
                sync_rows = connection.execute(
                    "SELECT group_id, history_exhausted FROM sync_state"
                ).fetchall()
            except sqlite3.OperationalError:
                sync_rows = []
            sync_state = {
                str(row["group_id"]): bool(row["history_exhausted"])
                for row in sync_rows
            }
            rows = connection.execute(
                """
                SELECT group_id, COUNT(*) AS messages,
                       COUNT(DISTINCT user_id) AS senders,
                       MIN(time) AS oldest_at, MAX(time) AS newest_at,
                       MAX(id) AS through_row_id
                FROM messages
                GROUP BY group_id
                ORDER BY messages DESC, group_id
                """
            ).fetchall()
            return [
                AngelEyeGroupSnapshot(
                    group_id=str(row["group_id"]),
                    messages=int(row["messages"] or 0),
                    senders=int(row["senders"] or 0),
                    oldest_at=int(row["oldest_at"] or 0),
                    newest_at=int(row["newest_at"] or 0),
                    through_row_id=int(row["through_row_id"] or 0),
                    history_exhausted=sync_state.get(
                        str(row["group_id"]), False
                    ),
                )
                for row in rows
            ]
        finally:
            connection.close()

    def load_batch(
        self,
        *,
        group_id: str,
        platform_id: str,
        after_row_id: int,
        through_row_id: int,
        limit: int = 250,
    ) -> tuple[list[NormalizedMessage], int, int]:
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT id, group_id, message_id, time, user_id, nickname,
                       raw_json
                FROM messages
                WHERE group_id=? AND id>? AND id<=?
                ORDER BY id
                LIMIT ?
                """,
                (
                    str(group_id),
                    int(after_row_id),
                    int(through_row_id),
                    max(1, min(2000, int(limit))),
                ),
            ).fetchall()
        finally:
            connection.close()
        messages: list[NormalizedMessage] = []
        skipped = 0
        last_row_id = int(after_row_id)
        for row in rows:
            last_row_id = int(row["id"])
            try:
                messages.append(
                    normalize_angel_eye_message(
                        row=dict(row),
                        platform_id=platform_id,
                    )
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                skipped += 1
        return messages, last_row_id, skipped


def _component_plain_text(content: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    placeholders = {
        "reply": "[引用消息]",
        "image": "[图片]",
        "file": "[文件]",
        "record": "[语音]",
        "audio": "[语音]",
        "video": "[视频]",
        "forward": "[转发消息]",
        "json": "[JSON消息]",
        "xml": "[XML消息]",
    }
    for component in content:
        kind = str(component.get("type") or "").casefold()
        if kind == "text":
            value = str(component.get("text") or "").strip()
            if value:
                parts.append(value)
        elif kind == "mention":
            label = str(
                component.get("display_name")
                or component.get("account_id")
                or ""
            ).strip()
            if label:
                parts.append(f"@{label}")
        elif kind in placeholders:
            parts.append(placeholders[kind])
    return " ".join(parts).strip()[:16000]


def normalize_angel_eye_message(
    *,
    row: dict[str, Any],
    platform_id: str,
) -> NormalizedMessage:
    raw = json.loads(str(row.get("raw_json") or "{}"))
    if not isinstance(raw, dict):
        raise ValueError("AngelEye raw_json 必须是对象")
    group_id = str(row.get("group_id") or raw.get("group_id") or "").strip()
    scope = angel_eye_scope(platform_id=platform_id, group_id=group_id)
    raw_content = raw.get("message") or []
    if not isinstance(raw_content, list):
        raw_content = []
    content = sanitize_components(
        item for item in raw_content if isinstance(item, dict)
    )
    sender = raw.get("sender") or {}
    if not isinstance(sender, dict):
        sender = {}
    sender_id = str(
        row.get("user_id") or sender.get("user_id") or raw.get("user_id") or ""
    ).strip()
    self_id = str(raw.get("self_id") or "").strip()
    role = (
        "BOT"
        if str(raw.get("post_type") or "").casefold() == "message_sent"
        or (sender_id and self_id and sender_id == self_id)
        else "USER"
    )
    sender_name = str(
        sender.get("card")
        or sender.get("nickname")
        or row.get("nickname")
        or ("AstrBot" if role == "BOT" else "")
    ).strip()
    message_id = str(row.get("message_id") or raw.get("message_id") or "").strip()
    sent_at = int(row.get("time") or raw.get("time") or 0)
    if not message_id or not sender_id or sent_at <= 0:
        raise ValueError("AngelEye 消息缺少稳定 ID、发送者或时间")
    return NormalizedMessage(
        platform="aiocqhttp",
        platform_id=scope.platform_id,
        umo=scope.key,
        group_id=scope.group_id,
        message_id=message_id,
        sender_id=sender_id,
        sender_name=sender_name,
        sent_at=sent_at,
        plain_text=_component_plain_text(content),
        content=content,
        role=role,  # type: ignore[arg-type]
    )
