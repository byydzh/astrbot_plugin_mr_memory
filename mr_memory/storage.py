from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path

from .embedding import (
    cosine_similarity,
    decode_vector,
    encode_vector,
    normalize_vector,
)
from .models import NormalizedMessage, StoredMessage


SCHEMA_VERSION = 6


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

                CREATE TABLE IF NOT EXISTS scope_meta (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    umo TEXT NOT NULL UNIQUE,
                    platform_id TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
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

                CREATE TABLE IF NOT EXISTS memory_embeddings (
                    umo TEXT NOT NULL,
                    owner_type TEXT NOT NULL,
                    owner_key TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (umo, owner_type, owner_key, model)
                );
                CREATE INDEX IF NOT EXISTS idx_memory_embeddings_lookup
                    ON memory_embeddings (umo, model, owner_type);

                CREATE TABLE IF NOT EXISTS distilled_units (
                    umo TEXT NOT NULL,
                    unit_type TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    unit_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (umo, unit_type, fingerprint)
                );
                CREATE INDEX IF NOT EXISTS idx_distilled_units_owner
                    ON distilled_units (umo, unit_type, unit_id);

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

                CREATE TABLE IF NOT EXISTS experiment_runs (
                    run_id TEXT PRIMARY KEY,
                    umo TEXT NOT NULL,
                    experiment_type TEXT NOT NULL,
                    cutoff_at INTEGER,
                    query_sha256 TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'RUNNING',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    result_json TEXT NOT NULL DEFAULT '{}',
                    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    finished_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_experiment_runs_scope_time
                    ON experiment_runs (umo, started_at DESC);

                CREATE TABLE IF NOT EXISTS llm_usage_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES experiment_runs(run_id)
                        ON DELETE CASCADE,
                    phase TEXT NOT NULL,
                    arm TEXT NOT NULL DEFAULT '',
                    call_index INTEGER NOT NULL DEFAULT 0,
                    provider_id TEXT NOT NULL DEFAULT '',
                    model TEXT NOT NULL DEFAULT '',
                    input_other INTEGER NOT NULL DEFAULT 0,
                    input_cached INTEGER NOT NULL DEFAULT 0,
                    output INTEGER NOT NULL DEFAULT 0,
                    elapsed_ms REAL NOT NULL DEFAULT 0,
                    usage_source TEXT NOT NULL DEFAULT 'provider',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_llm_usage_run_phase
                    ON llm_usage_events (run_id, phase, arm, call_index);

                CREATE TABLE IF NOT EXISTS reconstruction_steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES experiment_runs(run_id)
                        ON DELETE CASCADE,
                    arm TEXT NOT NULL DEFAULT 'memory',
                    step_index INTEGER NOT NULL,
                    tool_name TEXT NOT NULL,
                    arguments_json TEXT NOT NULL DEFAULT '{}',
                    evidence_keys_json TEXT NOT NULL DEFAULT '[]',
                    result_sha256 TEXT NOT NULL DEFAULT '',
                    elapsed_ms REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_reconstruction_steps_run
                    ON reconstruction_steps (run_id, arm, step_index);
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

    def bind_scope(self, *, umo: str, platform_id: str, group_id: str) -> None:
        """Bind this physical database to exactly one server-derived group scope."""
        identity = (umo.strip(), platform_id.strip(), group_id.strip())
        if not all(identity):
            raise ValueError("scope identity fields cannot be empty")
        with self._lock, self._connection:
            existing = self._connection.execute(
                "SELECT umo, platform_id, group_id FROM scope_meta WHERE singleton = 1"
            ).fetchone()
            if existing:
                stored = (
                    str(existing["umo"]),
                    str(existing["platform_id"]),
                    str(existing["group_id"]),
                )
                if stored != identity:
                    raise ValueError("database is already bound to another group scope")
                return
            conflicting = self._connection.execute(
                "SELECT 1 FROM messages WHERE umo <> ? LIMIT 1",
                (identity[0],),
            ).fetchone()
            if conflicting:
                raise ValueError("database contains messages from another group scope")
            legacy_identities = self._connection.execute(
                """
                SELECT DISTINCT platform_id, group_id
                FROM messages
                WHERE umo = ?
                LIMIT 2
                """,
                (identity[0],),
            ).fetchall()
            if any(
                (str(row["platform_id"]), str(row["group_id"])) != identity[1:]
                for row in legacy_identities
            ):
                raise ValueError("database message metadata conflicts with group scope")
            self._connection.execute(
                """
                INSERT INTO scope_meta(singleton, umo, platform_id, group_id)
                VALUES (1, ?, ?, ?)
                """,
                identity,
            )

    def get_scope_identity(self) -> dict[str, str] | None:
        """Read the bound scope, falling back to legacy message metadata."""
        with self._lock:
            row = self._connection.execute(
                "SELECT umo, platform_id, group_id FROM scope_meta WHERE singleton = 1"
            ).fetchone()
            if row:
                return {
                    "umo": str(row["umo"]),
                    "platform_id": str(row["platform_id"]),
                    "group_id": str(row["group_id"]),
                }
            rows = self._connection.execute(
                """
                SELECT DISTINCT umo, platform_id, group_id
                FROM messages
                WHERE is_deleted = 0
                LIMIT 2
                """
            ).fetchall()
        if len(rows) != 1:
            return None
        return {
            "umo": str(rows[0]["umo"]),
            "platform_id": str(rows[0]["platform_id"]),
            "group_id": str(rows[0]["group_id"]),
        }

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

    def count_graph_units(
        self,
        *,
        umo: str,
        before_sent_at: int | None = None,
    ) -> int:
        """Count distilled units without counting raw source messages."""
        with self._lock:
            if before_sent_at is None:
                row = self._connection.execute(
                    """
                    SELECT
                        (SELECT COUNT(*) FROM episodes WHERE umo = ?) +
                        (SELECT COUNT(*) FROM semantic_memories WHERE umo = ?) +
                        (SELECT COUNT(*) FROM topics WHERE umo = ?)
                    """,
                    (umo, umo, umo),
                ).fetchone()
            else:
                cutoff = int(before_sent_at)
                row = self._connection.execute(
                    """
                    SELECT
                        (SELECT COUNT(*) FROM episodes
                         WHERE umo = ? AND ended_at < ?) +
                        (SELECT COUNT(*)
                         FROM semantic_memories AS s
                         JOIN messages AS m ON m.id = s.source_message_id
                         WHERE s.umo = ? AND m.umo = s.umo
                           AND m.sent_at < ? AND m.is_deleted = 0) +
                        (SELECT COUNT(*) FROM topics AS t
                         WHERE t.umo = ? AND EXISTS (
                             SELECT 1 FROM topic_episodes AS te
                             JOIN episodes AS e ON e.id = te.episode_id
                             WHERE te.topic_id = t.id AND e.umo = t.umo
                               AND e.ended_at < ?
                         ))
                    """,
                    (umo, cutoff, umo, cutoff, umo, cutoff),
                ).fetchone()
        return int(row[0])

    def start_experiment(
        self,
        *,
        run_id: str,
        umo: str,
        experiment_type: str,
        cutoff_at: int | None = None,
        query_sha256: str = "",
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Create a privacy-minimized developer run record."""

        if not run_id.strip():
            raise ValueError("run_id is required")
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO experiment_runs(
                    run_id, umo, experiment_type, cutoff_at, query_sha256,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id.strip(),
                    umo,
                    experiment_type.strip(),
                    int(cutoff_at) if cutoff_at is not None else None,
                    query_sha256.strip(),
                    json.dumps(
                        metadata or {},
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ),
            )

    def finish_experiment(
        self,
        *,
        run_id: str,
        status: str,
        result: dict[str, object] | None = None,
    ) -> None:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE experiment_runs
                SET status = ?, result_json = ?, finished_at = CURRENT_TIMESTAMP
                WHERE run_id = ?
                """,
                (
                    status.strip().upper(),
                    json.dumps(
                        result or {},
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    run_id.strip(),
                ),
            )
        if cursor.rowcount != 1:
            raise ValueError(f"unknown experiment run: {run_id}")

    def record_llm_usage(
        self,
        *,
        run_id: str,
        phase: str,
        arm: str = "",
        call_index: int = 0,
        provider_id: str = "",
        model: str = "",
        input_other: int = 0,
        input_cached: int = 0,
        output: int = 0,
        elapsed_ms: float = 0.0,
        usage_source: str = "provider",
    ) -> int:
        values = (int(input_other), int(input_cached), int(output))
        if any(value < 0 for value in values):
            raise ValueError("token counts cannot be negative")
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO llm_usage_events(
                    run_id, phase, arm, call_index, provider_id, model,
                    input_other, input_cached, output, elapsed_ms, usage_source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id.strip(),
                    phase.strip(),
                    arm.strip(),
                    int(call_index),
                    provider_id.strip(),
                    model.strip(),
                    *values,
                    max(0.0, float(elapsed_ms)),
                    usage_source.strip() or "provider",
                ),
            )
        return int(cursor.lastrowid)

    def record_reconstruction_step(
        self,
        *,
        run_id: str,
        step_index: int,
        tool_name: str,
        arguments: dict[str, object] | None = None,
        evidence_keys: list[str] | None = None,
        result_text: str = "",
        elapsed_ms: float = 0.0,
        arm: str = "memory",
    ) -> int:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO reconstruction_steps(
                    run_id, arm, step_index, tool_name, arguments_json,
                    evidence_keys_json, result_sha256, elapsed_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id.strip(),
                    arm.strip() or "memory",
                    int(step_index),
                    tool_name.strip(),
                    json.dumps(
                        arguments or {},
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    json.dumps(
                        sorted(set(evidence_keys or [])),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    hashlib.sha256(result_text.encode("utf-8")).hexdigest()
                    if result_text
                    else "",
                    max(0.0, float(elapsed_ms)),
                ),
            )
        return int(cursor.lastrowid)

    def experiment_report(self, *, run_id: str) -> dict[str, object] | None:
        with self._lock:
            run = self._connection.execute(
                "SELECT * FROM experiment_runs WHERE run_id = ?",
                (run_id.strip(),),
            ).fetchone()
            if run is None:
                return None
            usage = self._connection.execute(
                """
                SELECT phase, arm, COUNT(*) AS calls,
                       SUM(input_other) AS input_other,
                       SUM(input_cached) AS input_cached,
                       SUM(output) AS output,
                       SUM(input_other + input_cached + output) AS total,
                       SUM(elapsed_ms) AS elapsed_ms
                FROM llm_usage_events
                WHERE run_id = ?
                GROUP BY phase, arm
                ORDER BY MIN(id)
                """,
                (run_id.strip(),),
            ).fetchall()
            steps = self._connection.execute(
                """
                SELECT arm, step_index, tool_name, arguments_json,
                       evidence_keys_json, result_sha256, elapsed_ms
                FROM reconstruction_steps
                WHERE run_id = ?
                ORDER BY arm, step_index, id
                """,
                (run_id.strip(),),
            ).fetchall()
        run_value = dict(run)
        run_value["metadata"] = json.loads(str(run_value.pop("metadata_json")))
        run_value["result"] = json.loads(str(run_value.pop("result_json")))
        return {
            "run": run_value,
            "usage": [dict(row) for row in usage],
            "steps": [
                {
                    **dict(row),
                    "arguments": json.loads(str(row["arguments_json"])),
                    "evidence_keys": json.loads(str(row["evidence_keys_json"])),
                }
                for row in steps
            ],
        }

    def recent_experiments(
        self,
        *,
        umo: str,
        limit: int = 10,
    ) -> list[dict[str, object]]:
        """Return a bounded, privacy-safe token ledger for one group scope."""

        safe_limit = max(1, min(100, int(limit)))
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT r.run_id, r.experiment_type, r.status,
                       r.started_at, r.finished_at,
                       COUNT(u.id) AS usage_records,
                       COALESCE(SUM(u.input_other), 0) AS input_other,
                       COALESCE(SUM(u.input_cached), 0) AS input_cached,
                       COALESCE(SUM(u.output), 0) AS output,
                       COALESCE(SUM(
                           u.input_other + u.input_cached + u.output
                       ), 0) AS total,
                       COALESCE(SUM(u.elapsed_ms), 0) AS elapsed_ms
                FROM experiment_runs AS r
                LEFT JOIN llm_usage_events AS u ON u.run_id = r.run_id
                WHERE r.umo = ?
                GROUP BY r.run_id
                ORDER BY r.started_at DESC, r.run_id DESC
                LIMIT ?
                """,
                (umo, safe_limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def dashboard_summary(self, *, umo: str) -> dict[str, object]:
        """Return bounded operational metrics for the authenticated plugin page."""
        with self._lock:
            counts = self._connection.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM messages
                     WHERE umo = ? AND is_deleted = 0) AS messages,
                    (SELECT COUNT(*) FROM episodes WHERE umo = ?) AS episodes,
                    (SELECT COUNT(*) FROM semantic_memories WHERE umo = ?)
                        AS semantic_memories,
                    (SELECT COUNT(*) FROM topics WHERE umo = ?) AS topics,
                    (SELECT COUNT(*) FROM memory_embeddings WHERE umo = ?)
                        AS embeddings,
                    (SELECT COUNT(DISTINCT lower(k.cue))
                     FROM episode_keywords AS k
                     JOIN episodes AS e ON e.id = k.episode_id
                     WHERE e.umo = ?) AS cues,
                    (SELECT MAX(sent_at) FROM messages
                     WHERE umo = ? AND is_deleted = 0) AS last_message_at
                """,
                (umo, umo, umo, umo, umo, umo, umo),
            ).fetchone()
            last_update = self._connection.execute(
                """
                SELECT MAX(value) AS value FROM (
                    SELECT MAX(created_at) AS value FROM episodes WHERE umo = ?
                    UNION ALL
                    SELECT MAX(created_at) AS value
                    FROM semantic_memories WHERE umo = ?
                    UNION ALL
                    SELECT MAX(updated_at) AS value
                    FROM memory_embeddings WHERE umo = ?
                )
                """,
                (umo, umo, umo),
            ).fetchone()
            models = self._connection.execute(
                """
                SELECT model, dimensions, COUNT(*) AS document_count
                FROM memory_embeddings
                WHERE umo = ?
                GROUP BY model, dimensions
                ORDER BY model
                """,
                (umo,),
            ).fetchall()

        database_bytes = 0
        for suffix in ("", "-wal", "-shm"):
            path = Path(f"{self.database_path}{suffix}")
            try:
                database_bytes += path.stat().st_size
            except OSError:
                pass
        identity = self.get_scope_identity() or {
            "umo": umo,
            "platform_id": "",
            "group_id": "",
        }
        return {
            **identity,
            "storage_id": hashlib.sha256(umo.encode("utf-8")).hexdigest(),
            "messages": int(counts["messages"] or 0),
            "episodes": int(counts["episodes"] or 0),
            "semantic_memories": int(counts["semantic_memories"] or 0),
            "topics": int(counts["topics"] or 0),
            "embeddings": int(counts["embeddings"] or 0),
            "cues": int(counts["cues"] or 0),
            "last_message_at": (
                int(counts["last_message_at"])
                if counts["last_message_at"] is not None
                else None
            ),
            "last_graph_update": (
                str(last_update["value"]) if last_update["value"] else None
            ),
            "database_bytes": database_bytes,
            "embedding_models": [dict(row) for row in models],
        }

    def dashboard_graph(
        self,
        *,
        umo: str,
        limit: int = 200,
    ) -> dict[str, object]:
        """Build a compact Cue/Tag/Episode/Semantic/Topic graph for the UI."""
        safe_limit = max(1, min(500, int(limit)))
        with self._lock:
            episode_rows = self._connection.execute(
                """
                SELECT e.id, e.started_at, e.ended_at, e.title, e.summary,
                       COUNT(em.message_id) AS source_count
                FROM episodes AS e
                LEFT JOIN episode_messages AS em ON em.episode_id = e.id
                WHERE e.umo = ?
                GROUP BY e.id
                ORDER BY e.ended_at DESC, e.id DESC
                LIMIT ?
                """,
                (umo, safe_limit),
            ).fetchall()
            episode_ids = [int(row["id"]) for row in episode_rows]
            keyword_rows: list[sqlite3.Row] = []
            topic_rows: list[sqlite3.Row] = []
            if episode_ids:
                placeholders = ",".join("?" for _ in episode_ids)
                keyword_rows = self._connection.execute(
                    f"""
                    SELECT k.episode_id, k.cue, k.tag
                    FROM episode_keywords AS k
                    JOIN episodes AS e ON e.id = k.episode_id
                    WHERE e.umo = ? AND e.id IN ({placeholders})
                    ORDER BY k.episode_id, k.cue, k.tag
                    LIMIT ?
                    """,
                    (umo, *episode_ids, safe_limit * 4),
                ).fetchall()
                topic_rows = self._connection.execute(
                    f"""
                    SELECT t.id, t.name, t.summary, te.episode_id
                    FROM topics AS t
                    JOIN topic_episodes AS te ON te.topic_id = t.id
                    JOIN episodes AS e ON e.id = te.episode_id
                    WHERE t.umo = ? AND e.umo = t.umo
                      AND e.id IN ({placeholders})
                    ORDER BY t.name, te.episode_id
                    LIMIT ?
                    """,
                    (umo, *episode_ids, safe_limit * 2),
                ).fetchall()
            semantic_rows = self._connection.execute(
                """
                SELECT s.id, s.person_cue, s.aspect_tag, s.content,
                       s.confidence, m.source_key, m.plain_text
                FROM semantic_memories AS s
                LEFT JOIN messages AS m
                  ON m.id = s.source_message_id AND m.umo = s.umo
                WHERE s.umo = ?
                ORDER BY s.id DESC
                LIMIT ?
                """,
                (umo, safe_limit),
            ).fetchall()

        nodes: dict[str, dict[str, object]] = {}
        edges: list[dict[str, object]] = []

        def cue_node_id(cue: str) -> str:
            digest = hashlib.sha1(cue.casefold().encode("utf-8")).hexdigest()[:16]
            return f"cue:{digest}"

        def ensure_cue(cue: str) -> str:
            node_id = cue_node_id(cue)
            nodes.setdefault(
                node_id,
                {
                    "id": node_id,
                    "type": "cue",
                    "label": cue,
                    "detail": "关联线索",
                },
            )
            return node_id

        for row in reversed(episode_rows):
            node_id = f"episode:{int(row['id'])}"
            nodes[node_id] = {
                "id": node_id,
                "type": "episode",
                "entity_id": int(row["id"]),
                "label": str(row["title"] or f"Episode {row['id']}"),
                "detail": str(row["summary"] or ""),
                "started_at": int(row["started_at"]),
                "ended_at": int(row["ended_at"]),
                "source_count": int(row["source_count"] or 0),
            }
        for row in keyword_rows:
            cue_id = ensure_cue(str(row["cue"]))
            episode_id = f"episode:{int(row['episode_id'])}"
            edges.append(
                {
                    "source": cue_id,
                    "target": episode_id,
                    "relation": str(row["tag"]),
                    "type": "cue_episode",
                }
            )
        for row in topic_rows:
            topic_id = f"topic:{int(row['id'])}"
            nodes.setdefault(
                topic_id,
                {
                    "id": topic_id,
                    "type": "topic",
                    "entity_id": int(row["id"]),
                    "label": str(row["name"]),
                    "detail": str(row["summary"] or ""),
                },
            )
            edges.append(
                {
                    "source": topic_id,
                    "target": f"episode:{int(row['episode_id'])}",
                    "relation": "包含",
                    "type": "topic_episode",
                }
            )
        for row in reversed(semantic_rows):
            semantic_id = f"semantic:{int(row['id'])}"
            nodes[semantic_id] = {
                "id": semantic_id,
                "type": "semantic",
                "entity_id": int(row["id"]),
                "label": str(row["aspect_tag"]),
                "detail": str(row["content"]),
                "confidence": float(row["confidence"]),
                "source_key": str(row["source_key"] or ""),
                "source_text": str(row["plain_text"] or ""),
            }
            cue_id = ensure_cue(str(row["person_cue"]))
            edges.append(
                {
                    "source": cue_id,
                    "target": semantic_id,
                    "relation": str(row["aspect_tag"]),
                    "type": "cue_semantic",
                }
            )

        original_node_count = len(nodes)
        if original_node_count > safe_limit:
            degree: dict[str, int] = {node_id: 0 for node_id in nodes}
            for edge in edges:
                degree[str(edge["source"])] = degree.get(str(edge["source"]), 0) + 1
                degree[str(edge["target"])] = degree.get(str(edge["target"]), 0) + 1

            ratios = {
                "episode": 0.40,
                "cue": 0.30,
                "semantic": 0.15,
                "topic": 0.15,
            }

            def node_rank(item: dict[str, object]) -> tuple[object, ...]:
                node_type = str(item["type"])
                if node_type == "episode":
                    return (-int(item.get("ended_at") or 0), str(item["id"]))
                if node_type == "semantic":
                    return (-int(item.get("entity_id") or 0), str(item["id"]))
                return (-degree.get(str(item["id"]), 0), str(item["label"]).casefold())

            selected_ids: set[str] = set()
            for node_type in ("episode", "cue", "semantic", "topic"):
                quota = int(safe_limit * ratios[node_type])
                if quota <= 0 and not selected_ids:
                    quota = 1
                candidates = sorted(
                    (
                        item
                        for item in nodes.values()
                        if item["type"] == node_type
                    ),
                    key=node_rank,
                )
                for item in candidates[:quota]:
                    if len(selected_ids) >= safe_limit:
                        break
                    selected_ids.add(str(item["id"]))

            remaining = sorted(
                (
                    item
                    for item in nodes.values()
                    if str(item["id"]) not in selected_ids
                ),
                key=lambda item: (
                    -degree.get(str(item["id"]), 0),
                    node_rank(item),
                ),
            )
            for item in remaining:
                if len(selected_ids) >= safe_limit:
                    break
                selected_ids.add(str(item["id"]))
            nodes = {
                node_id: node
                for node_id, node in nodes.items()
                if node_id in selected_ids
            }
            edges = [
                edge
                for edge in edges
                if str(edge["source"]) in selected_ids
                and str(edge["target"]) in selected_ids
            ]

        type_order = {"cue": 0, "episode": 1, "semantic": 2, "topic": 3}
        ordered_nodes = sorted(
            nodes.values(),
            key=lambda item: (
                type_order.get(str(item["type"]), 99),
                str(item["label"]).casefold(),
            ),
        )
        summary = self.dashboard_summary(umo=umo)
        return {
            "scope": summary,
            "nodes": ordered_nodes,
            "edges": edges,
            "limit": safe_limit,
            "truncated": (
                int(summary["episodes"]) > len(episode_rows)
                or original_node_count > len(ordered_nodes)
            ),
        }

    def upsert_memory_embedding(
        self,
        *,
        umo: str,
        owner_type: str,
        owner_key: str,
        model: str,
        vector: list[float],
    ) -> None:
        normalized = normalize_vector(vector)
        if not umo.strip() or not owner_type.strip() or not owner_key.strip():
            raise ValueError("embedding scope, owner type and owner key are required")
        if not model.strip():
            raise ValueError("embedding model id is required")
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO memory_embeddings(
                    umo, owner_type, owner_key, model, dimensions, vector
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(umo, owner_type, owner_key, model) DO UPDATE SET
                    dimensions=excluded.dimensions,
                    vector=excluded.vector,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    umo,
                    owner_type.strip(),
                    owner_key.strip(),
                    model.strip(),
                    len(normalized),
                    encode_vector(normalized),
                ),
            )

    def find_distilled_unit(
        self,
        *,
        umo: str,
        unit_type: str,
        fingerprint: str,
    ) -> int | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT unit_id FROM distilled_units
                WHERE umo = ? AND unit_type = ? AND fingerprint = ?
                """,
                (umo, unit_type, fingerprint),
            ).fetchone()
        return int(row["unit_id"]) if row else None

    def record_distilled_unit(
        self,
        *,
        umo: str,
        unit_type: str,
        fingerprint: str,
        unit_id: int,
    ) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT OR IGNORE INTO distilled_units(
                    umo, unit_type, fingerprint, unit_id
                ) VALUES (?, ?, ?, ?)
                """,
                (umo, unit_type, fingerprint, int(unit_id)),
            )

    def search_memory_embeddings(
        self,
        *,
        umo: str,
        model: str,
        query_vector: list[float],
        owner_types: tuple[str, ...] = ("cue", "episode", "topic", "semantic"),
        limit: int = 12,
        min_score: float = -1.0,
        before_sent_at: int | None = None,
    ) -> list[dict[str, object]]:
        if not owner_types:
            return []
        normalized_query = normalize_vector(query_vector)
        placeholders = ",".join("?" for _ in owner_types)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT owner_type, owner_key, dimensions, vector
                FROM memory_embeddings
                WHERE umo = ? AND model = ?
                  AND owner_type IN ({placeholders})
                """,
                (umo, model, *owner_types),
            ).fetchall()
        scored: list[dict[str, object]] = []
        for row in rows:
            owner_type = str(row["owner_type"])
            owner_key = str(row["owner_key"])
            if not self._memory_owner_visible(
                umo=umo,
                owner_type=owner_type,
                owner_key=owner_key,
                before_sent_at=before_sent_at,
            ):
                continue
            dimensions = int(row["dimensions"])
            if dimensions != len(normalized_query):
                continue
            stored = decode_vector(bytes(row["vector"]), dimensions)
            score = cosine_similarity(normalized_query, stored)
            if score < float(min_score):
                continue
            scored.append(
                {
                    "owner_type": owner_type,
                    "owner_key": owner_key,
                    "score": round(score, 6),
                }
            )
        scored.sort(key=lambda item: float(item["score"]), reverse=True)
        return scored[: max(1, min(100, int(limit)))]

    def _memory_owner_visible(
        self,
        *,
        umo: str,
        owner_type: str,
        owner_key: str,
        before_sent_at: int | None,
    ) -> bool:
        if before_sent_at is None:
            return True
        cutoff = int(before_sent_at)
        with self._lock:
            if owner_type == "episode" and owner_key.isdigit():
                row = self._connection.execute(
                    """
                    SELECT 1 FROM episodes
                    WHERE umo = ? AND id = ? AND ended_at < ?
                    LIMIT 1
                    """,
                    (umo, int(owner_key), cutoff),
                ).fetchone()
            elif owner_type == "semantic" and owner_key.isdigit():
                row = self._connection.execute(
                    """
                    SELECT 1
                    FROM semantic_memories AS s
                    JOIN messages AS m ON m.id = s.source_message_id
                    WHERE s.umo = ? AND s.id = ? AND m.umo = s.umo
                      AND m.sent_at < ? AND m.is_deleted = 0
                    LIMIT 1
                    """,
                    (umo, int(owner_key), cutoff),
                ).fetchone()
            elif owner_type == "topic" and owner_key.isdigit():
                row = self._connection.execute(
                    """
                    SELECT 1
                    FROM topics AS t
                    JOIN topic_episodes AS te ON te.topic_id = t.id
                    JOIN episodes AS e ON e.id = te.episode_id
                    WHERE t.umo = ? AND t.id = ? AND e.umo = t.umo
                      AND e.ended_at < ?
                    LIMIT 1
                    """,
                    (umo, int(owner_key), cutoff),
                ).fetchone()
            elif owner_type == "cue":
                row = self._connection.execute(
                    """
                    SELECT 1
                    FROM episode_keywords AS k
                    JOIN episodes AS e ON e.id = k.episode_id
                    WHERE e.umo = ? AND lower(k.cue) = lower(?)
                      AND e.ended_at < ?
                    UNION ALL
                    SELECT 1
                    FROM semantic_memories AS s
                    JOIN messages AS m ON m.id = s.source_message_id
                    WHERE s.umo = ? AND lower(s.person_cue) = lower(?)
                      AND m.umo = s.umo AND m.sent_at < ?
                      AND m.is_deleted = 0
                    LIMIT 1
                    """,
                    (umo, owner_key, cutoff, umo, owner_key, cutoff),
                ).fetchone()
            else:
                return False
        return row is not None

    def query_cue_tags(
        self,
        *,
        umo: str,
        cue: str,
        limit: int = 20,
        before_sent_at: int | None = None,
    ) -> list[dict[str, object]]:
        safe_limit = max(1, min(50, int(limit)))
        cutoff_sql = " AND e.ended_at < ?" if before_sent_at is not None else ""
        parameters: list[object] = [umo, cue.strip()]
        if before_sent_at is not None:
            parameters.append(int(before_sent_at))
        parameters.append(safe_limit)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT k.cue, k.tag, COUNT(DISTINCT e.id) AS episode_count
                FROM episode_keywords AS k
                JOIN episodes AS e ON e.id = k.episode_id
                WHERE e.umo = ? AND lower(k.cue) = lower(?)
                {cutoff_sql}
                GROUP BY k.cue, k.tag
                ORDER BY episode_count DESC, k.tag
                LIMIT ?
                """,
                parameters,
            ).fetchall()
        return [dict(row) for row in rows]

    def expand_seed_candidates(
        self,
        *,
        umo: str,
        matches: list[dict[str, object]],
        before_sent_at: int | None = None,
    ) -> dict[str, list[dict[str, object]]]:
        result: dict[str, list[dict[str, object]]] = {
            "cues": [],
            "episodes": [],
            "topics": [],
            "semantic_memories": [],
        }
        with self._lock:
            for match in matches:
                owner_type = str(match["owner_type"])
                owner_key = str(match["owner_key"])
                score = float(match["score"])
                if owner_type == "cue":
                    tags = self.query_cue_tags(
                        umo=umo,
                        cue=owner_key,
                        before_sent_at=before_sent_at,
                    )
                    if not tags:
                        cutoff_sql = ""
                        parameters: list[object] = [umo, owner_key]
                        if before_sent_at is not None:
                            cutoff_sql = (
                                " AND EXISTS (SELECT 1 FROM messages AS m "
                                "WHERE m.id = semantic_memories.source_message_id "
                                "AND m.umo = semantic_memories.umo "
                                "AND m.sent_at < ? AND m.is_deleted = 0)"
                            )
                            parameters.append(int(before_sent_at))
                        semantic_rows = self._connection.execute(
                            f"""
                            SELECT DISTINCT aspect_tag AS tag,
                                   COUNT(*) AS episode_count
                            FROM semantic_memories
                            WHERE umo = ? AND lower(person_cue) = lower(?)
                            {cutoff_sql}
                            GROUP BY aspect_tag
                            ORDER BY episode_count DESC, aspect_tag
                            """,
                            parameters,
                        ).fetchall()
                        tags = [dict(row) for row in semantic_rows]
                    result["cues"].append(
                        {"cue": owner_key, "score": score, "tags": tags}
                    )
                elif owner_type == "episode" and owner_key.isdigit():
                    cutoff_sql = (
                        " AND ended_at < ?" if before_sent_at is not None else ""
                    )
                    parameters = [umo, int(owner_key)]
                    if before_sent_at is not None:
                        parameters.append(int(before_sent_at))
                    row = self._connection.execute(
                        f"""
                        SELECT id, started_at, ended_at, title, summary
                        FROM episodes WHERE umo = ? AND id = ?
                        {cutoff_sql}
                        """,
                        parameters,
                    ).fetchone()
                    if row:
                        result["episodes"].append({**dict(row), "score": score})
                elif owner_type == "topic" and owner_key.isdigit():
                    cutoff_sql = ""
                    parameters = [umo, int(owner_key)]
                    if before_sent_at is not None:
                        cutoff_sql = (
                            " AND EXISTS (SELECT 1 FROM topic_episodes AS te "
                            "JOIN episodes AS e ON e.id = te.episode_id "
                            "WHERE te.topic_id = topics.id AND e.umo = topics.umo "
                            "AND e.ended_at < ?)"
                        )
                        parameters.append(int(before_sent_at))
                    row = self._connection.execute(
                        f"""
                        SELECT id, name, summary
                        FROM topics WHERE umo = ? AND id = ?
                        {cutoff_sql}
                        """,
                        parameters,
                    ).fetchone()
                    if row:
                        result["topics"].append({**dict(row), "score": score})
                elif owner_type == "semantic" and owner_key.isdigit():
                    cutoff_sql = ""
                    parameters = [umo, int(owner_key)]
                    if before_sent_at is not None:
                        cutoff_sql = (
                            " AND EXISTS (SELECT 1 FROM messages AS m "
                            "WHERE m.id = semantic_memories.source_message_id "
                            "AND m.umo = semantic_memories.umo "
                            "AND m.sent_at < ? AND m.is_deleted = 0)"
                        )
                        parameters.append(int(before_sent_at))
                    row = self._connection.execute(
                        f"""
                        SELECT id, person_cue, aspect_tag, content, confidence
                        FROM semantic_memories WHERE umo = ? AND id = ?
                        {cutoff_sql}
                        """,
                        parameters,
                    ).fetchone()
                    if row:
                        result["semantic_memories"].append(
                            {**dict(row), "score": score}
                        )
        return result

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
        before_sent_at: int | None = None,
    ) -> list[StoredMessage]:
        safe_limit = max(1, min(500, int(limit)))
        sender_filter = sender.strip().casefold()
        query = query.strip()
        parameters: list[object] = [umo]
        cutoff_sql = ""
        if before_sent_at is not None:
            cutoff_sql = " AND m.sent_at < ?"
            parameters.append(int(before_sent_at))
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
                  {cutoff_sql}
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
                  {cutoff_sql}
                  {sender_sql}
                ORDER BY m.sent_at DESC, m.id DESC
                LIMIT ?
            """
            parameters = [umo, query]
            if before_sent_at is not None:
                parameters.append(int(before_sent_at))
            if sender_filter:
                parameters.extend((sender_filter, sender_filter))
            parameters.append(safe_limit)
        else:
            sql = f"""
                SELECT m.*
                FROM messages AS m
                WHERE m.umo = ?
                  AND m.is_deleted = 0
                  {cutoff_sql}
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
        before_sent_at: int | None = None,
    ) -> list[dict[str, object]]:
        """Paper mapping phi_(cue,tag)->event."""
        safe_limit = max(1, min(50, int(limit)))
        cutoff_sql = " AND e.ended_at < ?" if before_sent_at is not None else ""
        parameters: list[object] = [umo, cue.strip(), tag.strip()]
        if before_sent_at is not None:
            parameters.append(int(before_sent_at))
        parameters.append(safe_limit)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT DISTINCT e.id, e.started_at, e.ended_at, e.title, e.summary
                FROM episode_keywords AS k
                JOIN episodes AS e ON e.id = k.episode_id
                WHERE e.umo = ?
                  AND instr(lower(k.cue), lower(?)) > 0
                  AND instr(lower(k.tag), lower(?)) > 0
                  {cutoff_sql}
                ORDER BY e.ended_at DESC, e.id DESC
                LIMIT ?
                """,
                parameters,
            ).fetchall()
        return [dict(row) for row in rows]

    def query_conversation_time(
        self,
        *,
        umo: str,
        event_id: int,
        before_sent_at: int | None = None,
    ) -> dict[str, object] | None:
        """Paper mapping phi_event->time."""
        cutoff_sql = " AND ended_at < ?" if before_sent_at is not None else ""
        parameters: list[object] = [umo, int(event_id)]
        if before_sent_at is not None:
            parameters.append(int(before_sent_at))
        with self._lock:
            row = self._connection.execute(
                f"""
                SELECT id, started_at, ended_at
                FROM episodes
                WHERE umo = ? AND id = ?
                {cutoff_sql}
                """,
                parameters,
            ).fetchone()
        return dict(row) if row else None

    def query_event_keywords(
        self,
        *,
        umo: str,
        event_id: int,
        before_sent_at: int | None = None,
    ) -> list[dict[str, object]]:
        """Paper reverse mapping phi_event->(cue,tag)."""
        cutoff_sql = " AND e.ended_at < ?" if before_sent_at is not None else ""
        parameters: list[object] = [umo, int(event_id)]
        if before_sent_at is not None:
            parameters.append(int(before_sent_at))
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT k.cue, k.tag
                FROM episode_keywords AS k
                JOIN episodes AS e ON e.id = k.episode_id
                WHERE e.umo = ? AND e.id = ?
                {cutoff_sql}
                ORDER BY k.cue, k.tag
                """,
                parameters,
            ).fetchall()
        return [dict(row) for row in rows]

    def query_event_context(
        self,
        *,
        umo: str,
        event_id: int,
        limit: int = 50,
        before_sent_at: int | None = None,
    ) -> list[dict[str, object]]:
        """Paper mapping phi_event->context, grounded in raw messages."""
        safe_limit = max(1, min(100, int(limit)))
        cutoff_sql = ""
        parameters: list[object] = [umo, int(event_id)]
        if before_sent_at is not None:
            cutoff_sql = " AND e.ended_at < ? AND m.sent_at < ?"
            parameters.extend((int(before_sent_at), int(before_sent_at)))
        parameters.append(safe_limit)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT m.source_key, m.sent_at, m.sender_id, m.sender_name,
                       m.role, m.plain_text
                FROM episode_messages AS em
                JOIN episodes AS e ON e.id = em.episode_id
                JOIN messages AS m ON m.id = em.message_id
                WHERE e.umo = ?
                  AND e.id = ?
                  AND m.umo = e.umo
                  AND m.is_deleted = 0
                  {cutoff_sql}
                ORDER BY em.position, m.sent_at, m.id
                LIMIT ?
                """,
                parameters,
            ).fetchall()
        return [dict(row) for row in rows]

    def query_personal_information(
        self,
        *,
        umo: str,
        person: str,
        before_sent_at: int | None = None,
    ) -> list[dict[str, object]]:
        """Paper mapping phi_person->semantic-aspects."""
        cutoff_sql = ""
        parameters: list[object] = [umo, person.strip()]
        if before_sent_at is not None:
            cutoff_sql = (
                " AND EXISTS (SELECT 1 FROM messages AS m "
                "WHERE m.id = semantic_memories.source_message_id "
                "AND m.umo = semantic_memories.umo "
                "AND m.sent_at < ? AND m.is_deleted = 0)"
            )
            parameters.append(int(before_sent_at))
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT aspect_tag, COUNT(*) AS evidence_count
                FROM semantic_memories
                WHERE umo = ? AND instr(lower(person_cue), lower(?)) > 0
                {cutoff_sql}
                GROUP BY aspect_tag
                ORDER BY evidence_count DESC, aspect_tag
                """,
                parameters,
            ).fetchall()
        return [dict(row) for row in rows]

    def query_personal_aspect(
        self,
        *,
        umo: str,
        person: str,
        aspect: str,
        limit: int = 20,
        before_sent_at: int | None = None,
    ) -> list[dict[str, object]]:
        """Paper mapping phi_(person,aspect)->semantic-content."""
        safe_limit = max(1, min(50, int(limit)))
        cutoff_sql = ""
        parameters: list[object] = [umo, person.strip(), aspect.strip()]
        if before_sent_at is not None:
            cutoff_sql = (
                " AND EXISTS (SELECT 1 FROM messages AS m "
                "WHERE m.id = semantic_memories.source_message_id "
                "AND m.umo = semantic_memories.umo "
                "AND m.sent_at < ? AND m.is_deleted = 0)"
            )
            parameters.append(int(before_sent_at))
        parameters.append(safe_limit)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT id, person_cue, aspect_tag, content, source_message_id,
                       confidence
                FROM semantic_memories
                WHERE umo = ?
                  AND instr(lower(person_cue), lower(?)) > 0
                  AND instr(lower(aspect_tag), lower(?)) > 0
                  {cutoff_sql}
                ORDER BY confidence DESC, id DESC
                LIMIT ?
                """,
                parameters,
            ).fetchall()
        return [dict(row) for row in rows]

    def query_topic_events(
        self,
        *,
        umo: str,
        topic: str,
        limit: int = 20,
        before_sent_at: int | None = None,
    ) -> list[dict[str, object]]:
        """Paper mapping phi_topic->event."""
        safe_limit = max(1, min(50, int(limit)))
        cutoff_sql = " AND e.ended_at < ?" if before_sent_at is not None else ""
        parameters: list[object] = [umo, topic.strip()]
        if before_sent_at is not None:
            parameters.append(int(before_sent_at))
        parameters.append(safe_limit)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT e.id, e.started_at, e.ended_at, e.title, e.summary,
                       t.name AS topic
                FROM topics AS t
                JOIN topic_episodes AS te ON te.topic_id = t.id
                JOIN episodes AS e ON e.id = te.episode_id
                WHERE t.umo = ?
                  AND e.umo = t.umo
                  AND instr(lower(t.name), lower(?)) > 0
                  {cutoff_sql}
                ORDER BY e.ended_at DESC, e.id DESC
                LIMIT ?
                """,
                parameters,
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
