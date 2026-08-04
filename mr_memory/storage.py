from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from .models import NormalizedMessage, StoredMessage


SCHEMA_VERSION = 2


class MemoryStorage:
    """SQLite truth store with an FTS5 index and reserved graph tables."""

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.database_path,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA synchronous = NORMAL")
        self._migrate()

    def _migrate(self) -> None:
        with self._lock, self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_key TEXT NOT NULL UNIQUE,
                    platform TEXT NOT NULL,
                    platform_id TEXT NOT NULL,
                    umo TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    sender_id TEXT NOT NULL,
                    sender_name TEXT NOT NULL,
                    sent_at INTEGER NOT NULL,
                    plain_text TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('USER', 'BOT', 'SYSTEM')),
                    is_deleted INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_messages_umo_time
                    ON messages (umo, sent_at, id);
                CREATE INDEX IF NOT EXISTS idx_messages_group_time
                    ON messages (platform_id, group_id, sent_at, id);
                CREATE INDEX IF NOT EXISTS idx_messages_sender_time
                    ON messages (umo, sender_id, sent_at, id);

                CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                    plain_text,
                    sender_name,
                    tokenize='trigram'
                );

                CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
                    INSERT INTO messages_fts(rowid, plain_text, sender_name)
                    VALUES (new.id, new.plain_text, new.sender_name);
                END;

                CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
                    DELETE FROM messages_fts WHERE rowid = old.id;
                END;

                CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
                    DELETE FROM messages_fts WHERE rowid = old.id;
                    INSERT INTO messages_fts(rowid, plain_text, sender_name)
                    VALUES (new.id, new.plain_text, new.sender_name);
                END;

                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    umo TEXT NOT NULL,
                    started_at INTEGER NOT NULL,
                    ended_at INTEGER NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    summary TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'OPEN',
                    extractor_version TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS episode_messages (
                    episode_id INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
                    message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                    position INTEGER NOT NULL,
                    PRIMARY KEY (episode_id, message_id)
                );

                CREATE TABLE IF NOT EXISTS episode_keywords (
                    episode_id INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
                    cue TEXT NOT NULL COLLATE NOCASE,
                    tag TEXT NOT NULL COLLATE NOCASE,
                    PRIMARY KEY (episode_id, cue, tag)
                );
                CREATE INDEX IF NOT EXISTS idx_episode_keywords_cue_tag
                    ON episode_keywords (cue, tag, episode_id);

                CREATE TABLE IF NOT EXISTS semantic_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    umo TEXT NOT NULL,
                    person_cue TEXT NOT NULL COLLATE NOCASE,
                    aspect_tag TEXT NOT NULL COLLATE NOCASE,
                    content TEXT NOT NULL,
                    source_message_id INTEGER REFERENCES messages(id),
                    confidence REAL NOT NULL DEFAULT 0,
                    extractor_version TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_semantic_person_aspect
                    ON semantic_memories (umo, person_cue, aspect_tag);

                CREATE TABLE IF NOT EXISTS topics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    umo TEXT NOT NULL,
                    name TEXT NOT NULL COLLATE NOCASE,
                    summary TEXT NOT NULL DEFAULT '',
                    extractor_version TEXT NOT NULL DEFAULT '',
                    UNIQUE (umo, name)
                );

                CREATE TABLE IF NOT EXISTS topic_episodes (
                    topic_id INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
                    episode_id INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
                    PRIMARY KEY (topic_id, episode_id)
                );

                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    umo TEXT NOT NULL,
                    canonical_name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    aliases_json TEXT NOT NULL DEFAULT '[]',
                    UNIQUE (umo, canonical_name, entity_type)
                );

                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    umo TEXT NOT NULL,
                    subject_entity_id INTEGER REFERENCES entities(id),
                    predicate TEXT NOT NULL,
                    object_text TEXT NOT NULL,
                    source_message_id INTEGER NOT NULL REFERENCES messages(id),
                    confidence REAL NOT NULL DEFAULT 0,
                    valid_from INTEGER,
                    valid_until INTEGER,
                    superseded_by INTEGER REFERENCES facts(id),
                    extractor_version TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    umo TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_id INTEGER NOT NULL,
                    relation TEXT NOT NULL,
                    target_type TEXT NOT NULL,
                    target_id INTEGER NOT NULL,
                    source_message_id INTEGER REFERENCES messages(id),
                    confidence REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS embeddings (
                    owner_type TEXT NOT NULL,
                    owner_id INTEGER NOT NULL,
                    model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (owner_type, owner_id, model)
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    available_at INTEGER NOT NULL,
                    last_error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            self._connection.execute(
                """
                INSERT INTO schema_meta(key, value) VALUES ('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (str(SCHEMA_VERSION),),
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def upsert_message(self, message: NormalizedMessage) -> bool:
        source_key = message.resolved_source_key()
        content_json = json.dumps(
            message.content,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._lock, self._connection:
            existed = self._connection.execute(
                "SELECT 1 FROM messages WHERE source_key = ?",
                (source_key,),
            ).fetchone()
            self._connection.execute(
                """
                INSERT INTO messages(
                    source_key, platform, platform_id, umo, group_id,
                    message_id, sender_id, sender_name, sent_at,
                    plain_text, content_json, role
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_key) DO UPDATE SET
                    sender_name=excluded.sender_name,
                    sent_at=excluded.sent_at,
                    plain_text=excluded.plain_text,
                    content_json=excluded.content_json,
                    role=excluded.role,
                    is_deleted=0,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    source_key,
                    message.platform,
                    message.platform_id,
                    message.umo,
                    message.group_id,
                    message.message_id,
                    message.sender_id,
                    message.sender_name,
                    message.sent_at,
                    message.plain_text,
                    content_json,
                    message.role,
                ),
            )
        return existed is None

    def count_messages(self, *, umo: str | None = None) -> int:
        with self._lock:
            if umo is None:
                row = self._connection.execute(
                    "SELECT COUNT(*) FROM messages WHERE is_deleted = 0"
                ).fetchone()
            else:
                row = self._connection.execute(
                    "SELECT COUNT(*) FROM messages WHERE is_deleted = 0 AND umo = ?",
                    (umo,),
                ).fetchone()
        return int(row[0])

    def count_graph_units(self, *, umo: str) -> int:
        """Count distilled units without counting raw source messages."""
        with self._lock:
            row = self._connection.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM episodes WHERE umo = ?) +
                    (SELECT COUNT(*) FROM semantic_memories WHERE umo = ?) +
                    (SELECT COUNT(*) FROM topics WHERE umo = ?)
                """,
                (umo, umo, umo),
            ).fetchone()
        return int(row[0])

    def store_episode(
        self,
        *,
        umo: str,
        started_at: int,
        ended_at: int,
        title: str,
        summary: str,
        source_keys: list[str],
        keywords: list[tuple[str, str]],
        extractor_version: str = "",
    ) -> int:
        """Persist one distilled Cue--Tag--Episode unit."""
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO episodes(
                    umo, started_at, ended_at, title, summary, status,
                    extractor_version
                ) VALUES (?, ?, ?, ?, ?, 'READY', ?)
                """,
                (
                    umo,
                    int(started_at),
                    int(ended_at),
                    title,
                    summary,
                    extractor_version,
                ),
            )
            episode_id = int(cursor.lastrowid)
            for position, source_key in enumerate(source_keys):
                row = self._connection.execute(
                    """
                    SELECT id FROM messages
                    WHERE umo = ? AND source_key = ? AND is_deleted = 0
                    """,
                    (umo, source_key),
                ).fetchone()
                if row:
                    self._connection.execute(
                        """
                        INSERT OR IGNORE INTO episode_messages(
                            episode_id, message_id, position
                        ) VALUES (?, ?, ?)
                        """,
                        (episode_id, int(row["id"]), position),
                    )
            self._connection.executemany(
                """
                INSERT OR IGNORE INTO episode_keywords(episode_id, cue, tag)
                VALUES (?, ?, ?)
                """,
                [
                    (episode_id, cue.strip(), tag.strip())
                    for cue, tag in keywords
                    if cue.strip() and tag.strip()
                ],
            )
        return episode_id

    def store_semantic_memory(
        self,
        *,
        umo: str,
        person: str,
        aspect: str,
        content: str,
        source_key: str = "",
        confidence: float = 0,
        extractor_version: str = "",
    ) -> int:
        """Persist one distilled Person--Aspect--Semantic unit."""
        with self._lock, self._connection:
            source_message_id = None
            if source_key:
                row = self._connection.execute(
                    """
                    SELECT id FROM messages
                    WHERE umo = ? AND source_key = ? AND is_deleted = 0
                    """,
                    (umo, source_key),
                ).fetchone()
                if row:
                    source_message_id = int(row["id"])
            cursor = self._connection.execute(
                """
                INSERT INTO semantic_memories(
                    umo, person_cue, aspect_tag, content, source_message_id,
                    confidence, extractor_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    umo,
                    person.strip(),
                    aspect.strip(),
                    content,
                    source_message_id,
                    max(0.0, min(1.0, float(confidence))),
                    extractor_version,
                ),
            )
        return int(cursor.lastrowid)

    def store_topic(
        self,
        *,
        umo: str,
        name: str,
        summary: str,
        event_ids: list[int],
        extractor_version: str = "",
    ) -> int:
        """Persist one Topic--Episode abstraction."""
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO topics(umo, name, summary, extractor_version)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(umo, name) DO UPDATE SET
                    summary=excluded.summary,
                    extractor_version=excluded.extractor_version
                """,
                (umo, name.strip(), summary, extractor_version),
            )
            row = self._connection.execute(
                "SELECT id FROM topics WHERE umo = ? AND name = ?",
                (umo, name.strip()),
            ).fetchone()
            topic_id = int(row["id"])
            self._connection.executemany(
                """
                INSERT OR IGNORE INTO topic_episodes(topic_id, episode_id)
                SELECT ?, id FROM episodes WHERE id = ? AND umo = ?
                """,
                [(topic_id, int(event_id), umo) for event_id in event_ids],
            )
        return topic_id

    def search_messages(
        self,
        *,
        umo: str,
        query: str = "",
        sender: str = "",
        limit: int = 20,
    ) -> list[StoredMessage]:
        safe_limit = max(1, min(50, int(limit)))
        sender_filter = sender.strip().casefold()
        query = query.strip()
        parameters: list[object] = [umo]
        sender_sql = ""
        if sender_filter:
            sender_sql = (
                " AND (instr(lower(m.sender_id), ?) > 0 "
                "OR instr(lower(m.sender_name), ?) > 0)"
            )
            parameters.extend((sender_filter, sender_filter))

        if query and len(query) >= 3:
            fts_query = self._make_fts_query(query)
            sql = f"""
                SELECT m.*
                FROM messages_fts
                JOIN messages AS m ON m.id = messages_fts.rowid
                WHERE messages_fts MATCH ?
                  AND m.umo = ?
                  AND m.is_deleted = 0
                  {sender_sql}
                ORDER BY bm25(messages_fts), m.sent_at DESC, m.id DESC
                LIMIT ?
            """
            parameters = [fts_query, *parameters, safe_limit]
        elif query:
            sql = f"""
                SELECT m.*
                FROM messages AS m
                WHERE m.umo = ?
                  AND m.is_deleted = 0
                  AND instr(lower(m.plain_text), lower(?)) > 0
                  {sender_sql}
                ORDER BY m.sent_at DESC, m.id DESC
                LIMIT ?
            """
            parameters = [umo, query]
            if sender_filter:
                parameters.extend((sender_filter, sender_filter))
            parameters.append(safe_limit)
        else:
            sql = f"""
                SELECT m.*
                FROM messages AS m
                WHERE m.umo = ?
                  AND m.is_deleted = 0
                  {sender_sql}
                ORDER BY m.sent_at DESC, m.id DESC
                LIMIT ?
            """
            parameters.append(safe_limit)

        with self._lock:
            rows = self._connection.execute(sql, parameters).fetchall()
        messages = [self._row_to_message(row) for row in rows]
        return sorted(messages, key=lambda item: (item.sent_at, item.id))

    def query_tag_events(
        self,
        *,
        umo: str,
        cue: str,
        tag: str,
        limit: int = 20,
    ) -> list[dict[str, object]]:
        """Paper mapping phi_(cue,tag)->event."""
        safe_limit = max(1, min(50, int(limit)))
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT DISTINCT e.id, e.started_at, e.ended_at, e.title, e.summary
                FROM episode_keywords AS k
                JOIN episodes AS e ON e.id = k.episode_id
                WHERE e.umo = ?
                  AND instr(lower(k.cue), lower(?)) > 0
                  AND instr(lower(k.tag), lower(?)) > 0
                ORDER BY e.ended_at DESC, e.id DESC
                LIMIT ?
                """,
                (umo, cue.strip(), tag.strip(), safe_limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def query_conversation_time(
        self, *, umo: str, event_id: int
    ) -> dict[str, object] | None:
        """Paper mapping phi_event->time."""
        with self._lock:
            row = self._connection.execute(
                """
                SELECT id, started_at, ended_at
                FROM episodes
                WHERE umo = ? AND id = ?
                """,
                (umo, int(event_id)),
            ).fetchone()
        return dict(row) if row else None

    def query_event_keywords(
        self, *, umo: str, event_id: int
    ) -> list[dict[str, object]]:
        """Paper reverse mapping phi_event->(cue,tag)."""
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT k.cue, k.tag
                FROM episode_keywords AS k
                JOIN episodes AS e ON e.id = k.episode_id
                WHERE e.umo = ? AND e.id = ?
                ORDER BY k.cue, k.tag
                """,
                (umo, int(event_id)),
            ).fetchall()
        return [dict(row) for row in rows]

    def query_event_context(
        self,
        *,
        umo: str,
        event_id: int,
        limit: int = 50,
    ) -> list[dict[str, object]]:
        """Paper mapping phi_event->context, grounded in raw messages."""
        safe_limit = max(1, min(100, int(limit)))
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT m.source_key, m.sent_at, m.sender_id, m.sender_name,
                       m.role, m.plain_text
                FROM episode_messages AS em
                JOIN episodes AS e ON e.id = em.episode_id
                JOIN messages AS m ON m.id = em.message_id
                WHERE e.umo = ?
                  AND e.id = ?
                  AND m.umo = e.umo
                  AND m.is_deleted = 0
                ORDER BY em.position, m.sent_at, m.id
                LIMIT ?
                """,
                (umo, int(event_id), safe_limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def query_personal_information(
        self, *, umo: str, person: str
    ) -> list[dict[str, object]]:
        """Paper mapping phi_person->semantic-aspects."""
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT aspect_tag, COUNT(*) AS evidence_count
                FROM semantic_memories
                WHERE umo = ? AND instr(lower(person_cue), lower(?)) > 0
                GROUP BY aspect_tag
                ORDER BY evidence_count DESC, aspect_tag
                """,
                (umo, person.strip()),
            ).fetchall()
        return [dict(row) for row in rows]

    def query_personal_aspect(
        self,
        *,
        umo: str,
        person: str,
        aspect: str,
        limit: int = 20,
    ) -> list[dict[str, object]]:
        """Paper mapping phi_(person,aspect)->semantic-content."""
        safe_limit = max(1, min(50, int(limit)))
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT id, person_cue, aspect_tag, content, source_message_id,
                       confidence
                FROM semantic_memories
                WHERE umo = ?
                  AND instr(lower(person_cue), lower(?)) > 0
                  AND instr(lower(aspect_tag), lower(?)) > 0
                ORDER BY confidence DESC, id DESC
                LIMIT ?
                """,
                (umo, person.strip(), aspect.strip(), safe_limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def query_topic_events(
        self,
        *,
        umo: str,
        topic: str,
        limit: int = 20,
    ) -> list[dict[str, object]]:
        """Paper mapping phi_topic->event."""
        safe_limit = max(1, min(50, int(limit)))
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT e.id, e.started_at, e.ended_at, e.title, e.summary,
                       t.name AS topic
                FROM topics AS t
                JOIN topic_episodes AS te ON te.topic_id = t.id
                JOIN episodes AS e ON e.id = te.episode_id
                WHERE t.umo = ?
                  AND e.umo = t.umo
                  AND instr(lower(t.name), lower(?)) > 0
                ORDER BY e.ended_at DESC, e.id DESC
                LIMIT ?
                """,
                (umo, topic.strip(), safe_limit),
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _make_fts_query(query: str) -> str:
        return f'"{query.replace(chr(34), chr(34) * 2)}"'

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> StoredMessage:
        return StoredMessage(
            id=int(row["id"]),
            source_key=str(row["source_key"]),
            platform=str(row["platform"]),
            platform_id=str(row["platform_id"]),
            umo=str(row["umo"]),
            group_id=str(row["group_id"]),
            message_id=str(row["message_id"]),
            sender_id=str(row["sender_id"]),
            sender_name=str(row["sender_name"]),
            sent_at=int(row["sent_at"]),
            plain_text=str(row["plain_text"]),
            content=json.loads(row["content_json"]),
            role=str(row["role"]),  # type: ignore[arg-type]
        )
