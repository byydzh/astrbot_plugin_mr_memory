from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import threading
import time
from pathlib import Path

from .embedding import (
    cosine_similarity,
    decode_vector,
    encode_vector,
    normalize_vector,
)
from .feedback import (
    FeedbackDecision,
    backward_credit_delta,
    feedback_surface_score,
    hypothesis_fingerprint,
    rank_hypotheses,
)
from .identity import (
    attachment_metadata,
    canonical_participant_key,
    content_fingerprint,
    extract_mentions,
    extract_reply,
    normalize_alias,
    sanitize_components,
)
from .models import DistillationWorkItem, NormalizedMessage, StoredMessage


SCHEMA_VERSION = 8


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
        self._restrict_file_permissions()

    def _restrict_file_permissions(self) -> None:
        """Keep plaintext group stores private to the AstrBot OS account."""

        if os.name == "nt":
            return
        try:
            os.chmod(self.database_path.parent, 0o700)
            for suffix in ("", "-wal", "-shm"):
                path = Path(f"{self.database_path}{suffix}")
                if path.exists():
                    os.chmod(path, 0o600)
        except OSError:
            # Deployment policy may own these modes; storage remains usable.
            pass

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

                CREATE TABLE IF NOT EXISTS participants (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    umo TEXT NOT NULL,
                    platform_id TEXT NOT NULL,
                    account_id TEXT NOT NULL,
                    canonical_key TEXT NOT NULL,
                    account_type TEXT NOT NULL DEFAULT 'USER',
                    current_display_name TEXT NOT NULL DEFAULT '',
                    first_seen_at INTEGER NOT NULL,
                    last_seen_at INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (umo, platform_id, account_id),
                    UNIQUE (umo, canonical_key)
                );
                CREATE INDEX IF NOT EXISTS idx_participants_scope_account
                    ON participants (umo, account_id);

                CREATE TABLE IF NOT EXISTS participant_aliases (
                    participant_id INTEGER NOT NULL
                        REFERENCES participants(id) ON DELETE CASCADE,
                    alias TEXT NOT NULL,
                    normalized_alias TEXT NOT NULL,
                    first_seen_at INTEGER NOT NULL,
                    last_seen_at INTEGER NOT NULL,
                    observation_count INTEGER NOT NULL DEFAULT 1,
                    source_kind TEXT NOT NULL DEFAULT 'observed',
                    confidence REAL NOT NULL DEFAULT 1,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (participant_id, normalized_alias)
                );
                CREATE INDEX IF NOT EXISTS idx_participant_alias_lookup
                    ON participant_aliases (normalized_alias, participant_id);

                CREATE TABLE IF NOT EXISTS forgotten_accounts (
                    umo TEXT NOT NULL,
                    platform_id TEXT NOT NULL,
                    account_hash TEXT NOT NULL,
                    requested_at INTEGER NOT NULL,
                    reason TEXT NOT NULL DEFAULT 'self_service',
                    PRIMARY KEY (umo, platform_id, account_hash)
                );

                CREATE TABLE IF NOT EXISTS message_revisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id INTEGER NOT NULL REFERENCES messages(id)
                        ON DELETE CASCADE,
                    revision_no INTEGER NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    plain_text TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    role TEXT NOT NULL,
                    revision_kind TEXT NOT NULL,
                    observed_at INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (message_id, revision_no)
                );

                CREATE TABLE IF NOT EXISTS message_processing (
                    message_id INTEGER PRIMARY KEY REFERENCES messages(id)
                        ON DELETE CASCADE,
                    content_sha256 TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    batch_key TEXT NOT NULL DEFAULT '',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT NOT NULL DEFAULT '',
                    distilled_at INTEGER,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_message_processing_pending
                    ON message_processing (status, message_id);

                CREATE TABLE IF NOT EXISTS message_participants (
                    message_id INTEGER NOT NULL REFERENCES messages(id)
                        ON DELETE CASCADE,
                    participant_id INTEGER NOT NULL REFERENCES participants(id),
                    relation TEXT NOT NULL,
                    position INTEGER NOT NULL DEFAULT 0,
                    evidence TEXT NOT NULL DEFAULT 'host',
                    PRIMARY KEY (message_id, participant_id, relation, position)
                );

                CREATE TABLE IF NOT EXISTS message_relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    umo TEXT NOT NULL,
                    source_message_id INTEGER NOT NULL REFERENCES messages(id)
                        ON DELETE CASCADE,
                    relation TEXT NOT NULL,
                    target_message_id INTEGER REFERENCES messages(id),
                    target_source_key TEXT NOT NULL DEFAULT '',
                    target_platform_message_id TEXT NOT NULL DEFAULT '',
                    target_participant_id INTEGER REFERENCES participants(id),
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (source_message_id, relation, target_source_key)
                );

                CREATE TABLE IF NOT EXISTS message_attachments (
                    message_id INTEGER NOT NULL REFERENCES messages(id)
                        ON DELETE CASCADE,
                    position INTEGER NOT NULL,
                    attachment_type TEXT NOT NULL,
                    name TEXT NOT NULL DEFAULT '',
                    reference_sha256 TEXT NOT NULL DEFAULT '',
                    extraction_status TEXT NOT NULL DEFAULT 'METADATA_ONLY',
                    descriptor_text TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (message_id, position)
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

                CREATE TABLE IF NOT EXISTS semantic_memory_sources (
                    semantic_memory_id INTEGER NOT NULL
                        REFERENCES semantic_memories(id) ON DELETE CASCADE,
                    message_id INTEGER NOT NULL REFERENCES messages(id),
                    evidence_role TEXT NOT NULL DEFAULT 'SUPPORT',
                    source_span TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (semantic_memory_id, message_id, evidence_role)
                );

                CREATE TABLE IF NOT EXISTS semantic_memory_revisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    semantic_memory_id INTEGER NOT NULL
                        REFERENCES semantic_memories(id) ON DELETE CASCADE,
                    previous_status TEXT NOT NULL,
                    new_status TEXT NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',
                    source_message_id INTEGER REFERENCES messages(id),
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

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

                CREATE TABLE IF NOT EXISTS topic_revisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic_id INTEGER NOT NULL REFERENCES topics(id)
                        ON DELETE CASCADE,
                    previous_summary TEXT NOT NULL,
                    proposed_summary TEXT NOT NULL,
                    extractor_version TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
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

                CREATE TABLE IF NOT EXISTS distillation_batches (
                    batch_key TEXT PRIMARY KEY,
                    umo TEXT NOT NULL,
                    target_source_keys_json TEXT NOT NULL,
                    target_hashes_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'RUNNING',
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    finished_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_distillation_batches_scope
                    ON distillation_batches (umo, created_at DESC);

                CREATE TABLE IF NOT EXISTS distillation_ignored_sources (
                    batch_key TEXT NOT NULL REFERENCES distillation_batches(batch_key)
                        ON DELETE CASCADE,
                    source_key TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (batch_key, source_key)
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

                CREATE TABLE IF NOT EXISTS interaction_traces (
                    trace_id TEXT PRIMARY KEY,
                    umo TEXT NOT NULL,
                    sender_id TEXT NOT NULL,
                    request_source_key TEXT NOT NULL DEFAULT '',
                    request_sent_at INTEGER NOT NULL,
                    request_sha256 TEXT NOT NULL,
                    request_excerpt TEXT NOT NULL DEFAULT '',
                    response_sha256 TEXT NOT NULL DEFAULT '',
                    response_excerpt TEXT NOT NULL DEFAULT '',
                    response_at INTEGER,
                    status TEXT NOT NULL DEFAULT 'OPEN',
                    expires_at INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_interaction_traces_scope_time
                    ON interaction_traces (umo, request_sent_at DESC, trace_id);
                CREATE INDEX IF NOT EXISTS idx_interaction_traces_feedback_window
                    ON interaction_traces (umo, status, response_at DESC);

                CREATE TABLE IF NOT EXISTS trace_nodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL REFERENCES interaction_traces(trace_id)
                        ON DELETE CASCADE,
                    umo TEXT NOT NULL,
                    node_key TEXT NOT NULL,
                    node_type TEXT NOT NULL,
                    content_json TEXT NOT NULL DEFAULT '{}',
                    activation REAL NOT NULL DEFAULT 0,
                    utility REAL NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'ACTIVE',
                    expires_at INTEGER,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (trace_id, node_key)
                );
                CREATE INDEX IF NOT EXISTS idx_trace_nodes_scope_type
                    ON trace_nodes (umo, node_type, id DESC);

                CREATE TABLE IF NOT EXISTS trace_edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL REFERENCES interaction_traces(trace_id)
                        ON DELETE CASCADE,
                    umo TEXT NOT NULL,
                    source_node_id INTEGER NOT NULL REFERENCES trace_nodes(id),
                    target_node_id INTEGER NOT NULL REFERENCES trace_nodes(id),
                    relation TEXT NOT NULL,
                    contribution REAL NOT NULL DEFAULT 0,
                    eligibility REAL NOT NULL DEFAULT 0,
                    credit REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (trace_id, source_node_id, target_node_id, relation)
                );

                CREATE TABLE IF NOT EXISTS feedback_hypotheses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    umo TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    scope_type TEXT NOT NULL,
                    scope_key TEXT NOT NULL,
                    aspect TEXT NOT NULL,
                    statement TEXT NOT NULL,
                    prospective_cue TEXT NOT NULL,
                    trigger_cues_json TEXT NOT NULL DEFAULT '[]',
                    activation_mode TEXT NOT NULL DEFAULT 'semantic',
                    evidence_confidence REAL NOT NULL DEFAULT 0,
                    utility REAL NOT NULL DEFAULT 0,
                    support_count INTEGER NOT NULL DEFAULT 0,
                    contradict_count INTEGER NOT NULL DEFAULT 0,
                    activation_count INTEGER NOT NULL DEFAULT 0,
                    learned_at INTEGER NOT NULL,
                    last_decay_at INTEGER NOT NULL,
                    last_activated_at INTEGER,
                    source_trace_id TEXT REFERENCES interaction_traces(trace_id),
                    status TEXT NOT NULL DEFAULT 'ACTIVE',
                    merged_into INTEGER REFERENCES feedback_hypotheses(id),
                    merge_previous_status TEXT NOT NULL DEFAULT '',
                    expires_at INTEGER,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (umo, fingerprint)
                );
                CREATE INDEX IF NOT EXISTS idx_feedback_hypotheses_activation
                    ON feedback_hypotheses (
                        umo, status, scope_type, scope_key, learned_at
                    );

                CREATE TABLE IF NOT EXISTS hypothesis_activations (
                    trace_id TEXT NOT NULL REFERENCES interaction_traces(trace_id)
                        ON DELETE CASCADE,
                    hypothesis_id INTEGER NOT NULL
                        REFERENCES feedback_hypotheses(id),
                    activation_score REAL NOT NULL,
                    contribution REAL NOT NULL DEFAULT 1,
                    credit REAL NOT NULL DEFAULT 0,
                    activation_method TEXT NOT NULL DEFAULT 'lexical',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (trace_id, hypothesis_id)
                );

                CREATE TABLE IF NOT EXISTS feedback_proposals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    umo TEXT NOT NULL,
                    feedback_source_key TEXT NOT NULL,
                    feedback_sent_at INTEGER NOT NULL,
                    candidate_trace_ids_json TEXT NOT NULL DEFAULT '[]',
                    surface_score REAL NOT NULL DEFAULT 0,
                    candidate_reason TEXT NOT NULL DEFAULT '',
                    decision_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    decided_at TEXT,
                    UNIQUE (umo, feedback_source_key)
                );
                CREATE INDEX IF NOT EXISTS idx_feedback_proposals_pending
                    ON feedback_proposals (umo, status, feedback_sent_at);

                CREATE TABLE IF NOT EXISTS feedback_links (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    umo TEXT NOT NULL,
                    trace_id TEXT NOT NULL REFERENCES interaction_traces(trace_id),
                    feedback_source_key TEXT NOT NULL,
                    feedback_sent_at INTEGER NOT NULL,
                    link_method TEXT NOT NULL,
                    link_confidence REAL NOT NULL,
                    feedback_valence REAL NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (umo, feedback_source_key, trace_id)
                );

                CREATE TABLE IF NOT EXISTS hypothesis_evidence (
                    hypothesis_id INTEGER NOT NULL
                        REFERENCES feedback_hypotheses(id),
                    feedback_source_key TEXT NOT NULL,
                    trace_id TEXT NOT NULL REFERENCES interaction_traces(trace_id),
                    relation TEXT NOT NULL,
                    valence REAL NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (hypothesis_id, feedback_source_key, relation)
                );
                """
            )

            def ensure_column(table: str, name: str, declaration: str) -> None:
                columns = {
                    str(row["name"])
                    for row in self._connection.execute(
                        f"PRAGMA table_info({table})"
                    ).fetchall()
                }
                if name not in columns:
                    self._connection.execute(
                        f"ALTER TABLE {table} ADD COLUMN {name} {declaration}"
                    )

            ensure_column(
                "messages",
                "sender_participant_id",
                "INTEGER REFERENCES participants(id)",
            )
            ensure_column("messages", "content_sha256", "TEXT NOT NULL DEFAULT ''")
            ensure_column("messages", "revision_no", "INTEGER NOT NULL DEFAULT 1")
            ensure_column("messages", "deleted_at", "INTEGER")
            ensure_column("episodes", "stable_key", "TEXT NOT NULL DEFAULT ''")
            ensure_column("episodes", "revision_no", "INTEGER NOT NULL DEFAULT 1")
            ensure_column(
                "episodes",
                "updated_at",
                "TEXT NOT NULL DEFAULT ''",
            )
            ensure_column(
                "semantic_memories",
                "stable_key",
                "TEXT NOT NULL DEFAULT ''",
            )
            ensure_column(
                "semantic_memories",
                "subject_participant_id",
                "INTEGER REFERENCES participants(id)",
            )
            ensure_column(
                "semantic_memories", "subject_text", "TEXT NOT NULL DEFAULT ''"
            )
            ensure_column(
                "semantic_memories", "claim_type", "TEXT NOT NULL DEFAULT 'FACT'"
            )
            ensure_column(
                "semantic_memories",
                "epistemic_status",
                "TEXT NOT NULL DEFAULT 'ASSERTED'",
            )
            ensure_column(
                "semantic_memories", "status", "TEXT NOT NULL DEFAULT 'ACTIVE'"
            )
            ensure_column(
                "semantic_memories",
                "superseded_by",
                "INTEGER REFERENCES semantic_memories(id)",
            )
            ensure_column(
                "semantic_memories",
                "updated_at",
                "TEXT NOT NULL DEFAULT ''",
            )
            self._connection.execute(
                "UPDATE episodes SET updated_at=CURRENT_TIMESTAMP WHERE updated_at=''"
            )
            self._connection.execute(
                """
                UPDATE semantic_memories SET updated_at=CURRENT_TIMESTAMP
                WHERE updated_at=''
                """
            )
            ensure_column(
                "feedback_proposals", "surface_score", "REAL NOT NULL DEFAULT 0"
            )
            ensure_column(
                "feedback_proposals",
                "candidate_reason",
                "TEXT NOT NULL DEFAULT ''",
            )
            self._connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_episodes_stable_key
                ON episodes (umo, stable_key) WHERE stable_key <> ''
                """
            )
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_semantic_stable_key
                ON semantic_memories (umo, stable_key, status)
                """
            )
            hypothesis_columns = {
                str(row["name"])
                for row in self._connection.execute(
                    "PRAGMA table_info(feedback_hypotheses)"
                ).fetchall()
            }
            if "activation_mode" not in hypothesis_columns:
                self._connection.execute(
                    """
                    ALTER TABLE feedback_hypotheses
                    ADD COLUMN activation_mode TEXT NOT NULL DEFAULT 'semantic'
                    """
                )
            if "merge_previous_status" not in hypothesis_columns:
                self._connection.execute(
                    """
                    ALTER TABLE feedback_hypotheses
                    ADD COLUMN merge_previous_status TEXT NOT NULL DEFAULT ''
                    """
                )
            self._connection.execute(
                """
                INSERT INTO schema_meta(key, value) VALUES ('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (str(SCHEMA_VERSION),),
            )
            backfill = self._connection.execute(
                "SELECT value FROM schema_meta WHERE key='truth_v2_backfill'"
            ).fetchone()
            if backfill is None or str(backfill["value"]) != str(SCHEMA_VERSION):
                self._backfill_truth_v2()
                self._connection.execute(
                    """
                    INSERT INTO schema_meta(key, value)
                    VALUES ('truth_v2_backfill', ?)
                    ON CONFLICT(key) DO UPDATE SET value=excluded.value
                    """,
                    (str(SCHEMA_VERSION),),
                )
            # A process kill or cancellation can occur after a batch claims rows but
            # before its normal completion handler runs. Recover those rows on every
            # open so the oldest-first checkpoint cannot become permanently stuck.
            self._connection.execute(
                """
                UPDATE distillation_batches
                SET status='FAILED', error='interrupted before completion',
                    finished_at=CURRENT_TIMESTAMP
                WHERE status='RUNNING'
                """
            )
            self._connection.execute(
                """
                UPDATE message_processing
                SET status='FAILED', batch_key='',
                    last_error='interrupted before completion',
                    updated_at=CURRENT_TIMESTAMP
                WHERE status='PROCESSING'
                """
            )

    def _upsert_alias_locked(
        self,
        *,
        participant_id: int,
        alias: str,
        seen_at: int,
        source_kind: str,
        confidence: float = 1.0,
    ) -> None:
        display = str(alias or "").strip()[:300]
        normalized = normalize_alias(display)
        if not normalized:
            return
        self._connection.execute(
            """
            INSERT INTO participant_aliases(
                participant_id, alias, normalized_alias, first_seen_at,
                last_seen_at, source_kind, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(participant_id, normalized_alias) DO UPDATE SET
                alias=CASE
                    WHEN excluded.last_seen_at >= participant_aliases.last_seen_at
                    THEN excluded.alias
                    ELSE participant_aliases.alias
                END,
                last_seen_at=MAX(participant_aliases.last_seen_at,
                                 excluded.last_seen_at),
                observation_count=participant_aliases.observation_count + 1,
                source_kind=CASE
                    WHEN participant_aliases.source_kind = 'administrator'
                    THEN participant_aliases.source_kind
                    ELSE excluded.source_kind
                END,
                confidence=MAX(participant_aliases.confidence,
                               excluded.confidence),
                is_active=1,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                int(participant_id),
                display,
                normalized,
                int(seen_at),
                int(seen_at),
                str(source_kind or "observed"),
                max(0.0, min(1.0, float(confidence))),
            ),
        )

    def _upsert_participant_locked(
        self,
        *,
        umo: str,
        platform_id: str,
        account_id: str,
        display_name: str,
        seen_at: int,
        account_type: str = "USER",
        alias_source: str = "observed",
    ) -> int | None:
        account = str(account_id or "").strip()
        platform = str(platform_id or "").strip()
        if not account or not platform or account.casefold() == "all":
            return None
        forgotten_hash = self._forgotten_account_hash(
            umo=umo,
            platform_id=platform,
            account_id=account,
        )
        if self._connection.execute(
            """
            SELECT 1 FROM forgotten_accounts
            WHERE umo = ? AND platform_id = ? AND account_hash = ?
            """,
            (umo, platform, forgotten_hash),
        ).fetchone():
            return None
        kind = str(account_type or "USER").strip().upper()
        if kind not in {"USER", "BOT", "UNKNOWN"}:
            kind = "UNKNOWN"
        key = canonical_participant_key(platform, account)
        display = str(display_name or "").strip()[:300]
        verified_display = display if alias_source == "observed" else ""
        self._connection.execute(
            """
            INSERT INTO participants(
                umo, platform_id, account_id, canonical_key, account_type,
                current_display_name, first_seen_at, last_seen_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(umo, platform_id, account_id) DO UPDATE SET
                account_type=CASE
                    WHEN participants.account_type = 'BOT' THEN 'BOT'
                    WHEN excluded.account_type = 'BOT' THEN 'BOT'
                    WHEN participants.account_type = 'UNKNOWN'
                    THEN excluded.account_type
                    ELSE participants.account_type
                END,
                current_display_name=CASE
                    WHEN excluded.current_display_name <> '' AND ? = 1
                         AND excluded.last_seen_at >= participants.last_seen_at
                    THEN excluded.current_display_name
                    WHEN participants.current_display_name = ''
                         AND excluded.current_display_name <> ''
                    THEN excluded.current_display_name
                    ELSE participants.current_display_name
                END,
                first_seen_at=MIN(participants.first_seen_at,
                                  excluded.first_seen_at),
                last_seen_at=MAX(participants.last_seen_at,
                                 excluded.last_seen_at),
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                umo,
                platform,
                account,
                key,
                kind,
                verified_display,
                int(seen_at),
                int(seen_at),
                1 if alias_source == "observed" else 0,
            ),
        )
        row = self._connection.execute(
            """
            SELECT id FROM participants
            WHERE umo = ? AND platform_id = ? AND account_id = ?
            """,
            (umo, platform, account),
        ).fetchone()
        if row is None:
            raise RuntimeError("participant upsert did not return an identity")
        participant_id = int(row["id"])
        self._upsert_alias_locked(
            participant_id=participant_id,
            alias=display,
            seen_at=seen_at,
            source_kind=alias_source,
        )
        return participant_id

    @staticmethod
    def _parse_content_json(value: object) -> list[dict[str, object]]:
        try:
            parsed = json.loads(str(value or "[]"))
        except (TypeError, json.JSONDecodeError):
            return []
        if not isinstance(parsed, list):
            return []
        return [item for item in parsed if isinstance(item, dict)]

    def _refresh_message_links_locked(
        self,
        *,
        message_id: int,
        umo: str,
        platform_id: str,
        source_key: str,
        platform_message_id: str,
        sender_participant_id: int | None,
        sent_at: int,
        content: list[dict[str, object]],
    ) -> None:
        self._connection.execute(
            "DELETE FROM message_participants WHERE message_id = ?",
            (int(message_id),),
        )
        self._connection.execute(
            "DELETE FROM message_relations WHERE source_message_id = ?",
            (int(message_id),),
        )
        self._connection.execute(
            "DELETE FROM message_attachments WHERE message_id = ?",
            (int(message_id),),
        )
        if sender_participant_id is not None:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO message_participants(
                    message_id, participant_id, relation, position, evidence
                ) VALUES (?, ?, 'SPEAKER', 0, 'platform_account')
                """,
                (int(message_id), int(sender_participant_id)),
            )

        for position, mention in enumerate(extract_mentions(content), start=1):
            participant_id = self._upsert_participant_locked(
                umo=umo,
                platform_id=platform_id,
                account_id=mention.account_id,
                display_name=mention.display_name,
                seen_at=sent_at,
                account_type="UNKNOWN",
                alias_source="mention",
            )
            if participant_id is None:
                continue
            self._connection.execute(
                """
                INSERT OR REPLACE INTO message_participants(
                    message_id, participant_id, relation, position, evidence
                ) VALUES (?, ?, 'MENTIONED', ?, 'platform_mention')
                """,
                (int(message_id), int(participant_id), int(position)),
            )

        reply = extract_reply(content)
        if reply is not None:
            target_participant_id = self._upsert_participant_locked(
                umo=umo,
                platform_id=platform_id,
                account_id=reply.sender_id,
                display_name=reply.sender_name,
                seen_at=reply.sent_at or sent_at,
                account_type="UNKNOWN",
                alias_source="reply",
            )
            target = self._connection.execute(
                """
                SELECT id, source_key FROM messages
                WHERE umo = ? AND platform_id = ? AND message_id = ?
                  AND is_deleted = 0
                ORDER BY id DESC LIMIT 1
                """,
                (umo, platform_id, reply.message_id),
            ).fetchone()
            if target is None and target_participant_id is not None:
                parameters: list[object] = [
                    umo,
                    int(target_participant_id),
                    int(reply.sent_at or sent_at),
                ]
                text_sql = ""
                if reply.plain_text:
                    text_sql = " AND plain_text = ?"
                    parameters.append(reply.plain_text)
                target = self._connection.execute(
                    f"""
                    SELECT id, source_key FROM messages
                    WHERE umo = ? AND sender_participant_id = ?
                      AND ABS(sent_at - ?) <= 180{text_sql}
                      AND is_deleted = 0
                    ORDER BY ABS(sent_at - ?), id DESC LIMIT 1
                    """,
                    (*parameters, int(reply.sent_at or sent_at)),
                ).fetchone()
            target_source_key = (
                str(target["source_key"])
                if target is not None
                else "|".join((platform_id, umo, reply.message_id))
            )
            metadata = {
                "reply_sender_id": reply.sender_id,
                "reply_sender_name": reply.sender_name,
                "reply_sent_at": reply.sent_at,
                "reply_text": reply.plain_text[:2000],
                "resolved": target is not None,
            }
            self._connection.execute(
                """
                INSERT OR REPLACE INTO message_relations(
                    umo, source_message_id, relation, target_message_id,
                    target_source_key, target_platform_message_id,
                    target_participant_id, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    umo,
                    int(message_id),
                    reply.relation,
                    int(target["id"]) if target is not None else None,
                    target_source_key,
                    reply.message_id,
                    target_participant_id,
                    json.dumps(
                        metadata,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ),
            )
            if target_participant_id is not None:
                self._connection.execute(
                    """
                    INSERT OR REPLACE INTO message_participants(
                        message_id, participant_id, relation, position, evidence
                    ) VALUES (?, ?, 'REPLY_TARGET', 0, 'platform_reply')
                    """,
                    (int(message_id), int(target_participant_id)),
                )

        for attachment in attachment_metadata(content):
            self._connection.execute(
                """
                INSERT INTO message_attachments(
                    message_id, position, attachment_type, name,
                    reference_sha256
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    int(message_id),
                    int(attachment["position"]),
                    str(attachment["kind"]),
                    str(attachment["name"]),
                    str(attachment["reference_sha256"]),
                ),
            )

    def _backfill_truth_v2(self) -> None:
        rows = self._connection.execute(
            """
            SELECT id, source_key, platform_id, umo, message_id, sender_id,
                   sender_name, sent_at, plain_text, content_json, role,
                   is_deleted, content_sha256, sender_participant_id
            FROM messages ORDER BY id
            """
        ).fetchall()
        for row in rows:
            content = self._scrub_forgotten_references_locked(
                umo=str(row["umo"]),
                platform_id=str(row["platform_id"]),
                content=sanitize_components(
                    self._parse_content_json(row["content_json"])
                ),
            )
            digest = content_fingerprint(
                sender_id=str(row["sender_id"]),
                role=str(row["role"]),
                plain_text=str(row["plain_text"]),
                content=content,
            )
            participant_id = row["sender_participant_id"]
            if participant_id is None:
                participant_id = self._upsert_participant_locked(
                    umo=str(row["umo"]),
                    platform_id=str(row["platform_id"]),
                    account_id=str(row["sender_id"]),
                    display_name=str(row["sender_name"]),
                    seen_at=int(row["sent_at"]),
                    account_type=("BOT" if str(row["role"]) == "BOT" else "USER"),
                )
            self._connection.execute(
                """
                UPDATE messages
                SET content_json = ?, content_sha256 = ?,
                    sender_participant_id = ?
                WHERE id = ?
                """,
                (
                    json.dumps(
                        content,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    digest,
                    participant_id,
                    int(row["id"]),
                ),
            )
            self._connection.execute(
                """
                INSERT INTO message_processing(
                    message_id, content_sha256, status
                ) VALUES (?, ?, ?)
                ON CONFLICT(message_id) DO UPDATE SET
                    content_sha256=excluded.content_sha256,
                    status=excluded.status, batch_key='', attempts=0,
                    last_error='', distilled_at=NULL,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    int(row["id"]),
                    digest,
                    "DELETED" if int(row["is_deleted"]) else "PENDING",
                ),
            )
            self._refresh_message_links_locked(
                message_id=int(row["id"]),
                umo=str(row["umo"]),
                platform_id=str(row["platform_id"]),
                source_key=str(row["source_key"]),
                platform_message_id=str(row["message_id"]),
                sender_participant_id=(
                    int(participant_id) if participant_id is not None else None
                ),
                sent_at=int(row["sent_at"]),
                content=content,
            )

        self._connection.execute(
            """
            INSERT OR IGNORE INTO semantic_memory_sources(
                semantic_memory_id, message_id, evidence_role, confidence
            )
            SELECT id, source_message_id, 'SUPPORT', confidence
            FROM semantic_memories WHERE source_message_id IS NOT NULL
            """
        )
        legacy_semantics = self._connection.execute(
            """
            SELECT id, umo, person_cue FROM semantic_memories
            WHERE subject_participant_id IS NULL
            """
        ).fetchall()
        for semantic in legacy_semantics:
            aliases = self._connection.execute(
                """
                SELECT DISTINCT p.id
                FROM participant_aliases AS a
                JOIN participants AS p ON p.id = a.participant_id
                WHERE p.umo = ? AND a.normalized_alias = ? AND a.is_active = 1
                """,
                (str(semantic["umo"]), normalize_alias(semantic["person_cue"])),
            ).fetchall()
            if len(aliases) == 1:
                self._connection.execute(
                    """
                    UPDATE semantic_memories
                    SET subject_participant_id = ?, subject_text = person_cue
                    WHERE id = ?
                    """,
                    (int(aliases[0]["id"]), int(semantic["id"])),
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

    def _invalidate_message_derivations_locked(
        self,
        *,
        message_id: int,
        reason: str,
    ) -> None:
        episode_rows = self._connection.execute(
            "SELECT episode_id FROM episode_messages WHERE message_id = ?",
            (int(message_id),),
        ).fetchall()
        episode_ids = [int(row["episode_id"]) for row in episode_rows]
        if episode_ids:
            placeholders = ",".join("?" for _ in episode_ids)
            self._connection.execute(
                f"""
                UPDATE episodes SET status = 'STALE', updated_at=CURRENT_TIMESTAMP
                WHERE id IN ({placeholders})
                """,
                episode_ids,
            )
            for episode_id in episode_ids:
                self._connection.execute(
                    """
                    DELETE FROM memory_embeddings
                    WHERE owner_type = 'episode' AND owner_key = ?
                    """,
                    (str(episode_id),),
                )

        semantic_rows = self._connection.execute(
            """
            SELECT DISTINCT semantic_memory_id
            FROM semantic_memory_sources WHERE message_id = ?
            """,
            (int(message_id),),
        ).fetchall()
        for row in semantic_rows:
            semantic_id = int(row["semantic_memory_id"])
            current = self._connection.execute(
                "SELECT status FROM semantic_memories WHERE id = ?",
                (semantic_id,),
            ).fetchone()
            if current is None or str(current["status"]) in {
                "RETRACTED",
                "SUPERSEDED",
            }:
                continue
            self._connection.execute(
                """
                UPDATE semantic_memories
                SET status = 'STALE', updated_at=CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (semantic_id,),
            )
            self._connection.execute(
                """
                INSERT INTO semantic_memory_revisions(
                    semantic_memory_id, previous_status, new_status, reason,
                    source_message_id
                ) VALUES (?, ?, 'STALE', ?, ?)
                """,
                (
                    semantic_id,
                    str(current["status"]),
                    str(reason)[:500],
                    int(message_id),
                ),
            )
            self._connection.execute(
                """
                DELETE FROM memory_embeddings
                WHERE owner_type = 'semantic' AND owner_key = ?
                """,
                (str(semantic_id),),
            )

    def upsert_message(self, message: NormalizedMessage) -> bool:
        source_key = message.resolved_source_key()
        with self._lock, self._connection:
            if self._is_account_forgotten_locked(
                umo=message.umo,
                platform_id=message.platform_id,
                account_id=message.sender_id,
            ):
                return False
            content = self._scrub_forgotten_references_locked(
                umo=message.umo,
                platform_id=message.platform_id,
                content=sanitize_components(message.content),
            )
            content_json = json.dumps(
                content,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            digest = content_fingerprint(
                sender_id=message.sender_id,
                role=message.role,
                plain_text=message.plain_text,
                content=content,
            )
            existing = self._connection.execute(
                """
                SELECT id, content_sha256, plain_text, content_json, role,
                       revision_no, is_deleted
                FROM messages WHERE source_key = ?
                """,
                (source_key,),
            ).fetchone()
            participant_id = self._upsert_participant_locked(
                umo=message.umo,
                platform_id=message.platform_id,
                account_id=message.sender_id,
                display_name=message.sender_name,
                seen_at=message.sent_at,
                account_type=("BOT" if message.role == "BOT" else "USER"),
            )
            changed = (
                existing is not None
                and (
                    str(existing["content_sha256"] or "") != digest
                    or int(existing["is_deleted"] or 0) != 0
                )
            )
            revision_no = 1
            if existing is not None:
                revision_no = int(existing["revision_no"] or 1)
                if changed:
                    self._connection.execute(
                        """
                        INSERT OR IGNORE INTO message_revisions(
                            message_id, revision_no, content_sha256, plain_text,
                            content_json, role, revision_kind, observed_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            int(existing["id"]),
                            revision_no,
                            str(existing["content_sha256"] or ""),
                            str(existing["plain_text"]),
                            str(existing["content_json"]),
                            str(existing["role"]),
                            (
                                "RESTORED"
                                if int(existing["is_deleted"] or 0)
                                else "EDITED"
                            ),
                            int(time.time()),
                        ),
                    )
                    revision_no += 1
                    self._invalidate_message_derivations_locked(
                        message_id=int(existing["id"]),
                        reason="source message edited or restored",
                    )
            self._connection.execute(
                """
                INSERT INTO messages(
                    source_key, platform, platform_id, umo, group_id,
                    message_id, sender_id, sender_name, sender_participant_id,
                    sent_at, plain_text, content_json, role, content_sha256,
                    revision_no
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_key) DO UPDATE SET
                    sender_name=excluded.sender_name,
                    sender_participant_id=excluded.sender_participant_id,
                    sent_at=excluded.sent_at,
                    plain_text=excluded.plain_text,
                    content_json=excluded.content_json,
                    role=excluded.role,
                    content_sha256=excluded.content_sha256,
                    revision_no=excluded.revision_no,
                    is_deleted=0,
                    deleted_at=NULL,
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
                    participant_id,
                    message.sent_at,
                    message.plain_text,
                    content_json,
                    message.role,
                    digest,
                    revision_no,
                ),
            )
            row = self._connection.execute(
                "SELECT id FROM messages WHERE source_key = ?",
                (source_key,),
            ).fetchone()
            if row is None:
                raise RuntimeError("message upsert did not return a row")
            stored_id = int(row["id"])
            if existing is None or changed:
                self._connection.execute(
                    """
                    INSERT INTO message_processing(
                        message_id, content_sha256, status, batch_key,
                        attempts, last_error, distilled_at
                    ) VALUES (?, ?, 'PENDING', '', 0, '', NULL)
                    ON CONFLICT(message_id) DO UPDATE SET
                        content_sha256=excluded.content_sha256,
                        status='PENDING', batch_key='', attempts=0,
                        last_error='', distilled_at=NULL,
                        updated_at=CURRENT_TIMESTAMP
                    """,
                    (stored_id, digest),
                )
            self._refresh_message_links_locked(
                message_id=stored_id,
                umo=message.umo,
                platform_id=message.platform_id,
                source_key=source_key,
                platform_message_id=message.message_id,
                sender_participant_id=participant_id,
                sent_at=message.sent_at,
                content=content,
            )
        return existing is None

    def mark_message_deleted(
        self,
        *,
        umo: str,
        platform_id: str,
        platform_message_id: str,
        deleted_at: int | None = None,
        reason: str = "platform_recall",
    ) -> bool:
        """Apply a platform recall without destroying the prior revision evidence."""

        when = int(deleted_at or time.time())
        with self._lock, self._connection:
            row = self._connection.execute(
                """
                SELECT id, content_sha256, plain_text, content_json, role,
                       revision_no, is_deleted
                FROM messages
                WHERE umo = ? AND platform_id = ? AND message_id = ?
                ORDER BY id DESC LIMIT 1
                """,
                (umo, platform_id, str(platform_message_id)),
            ).fetchone()
            if row is None or int(row["is_deleted"] or 0):
                return False
            message_id = int(row["id"])
            revision_no = int(row["revision_no"] or 1)
            self._connection.execute(
                """
                INSERT OR IGNORE INTO message_revisions(
                    message_id, revision_no, content_sha256, plain_text,
                    content_json, role, revision_kind, observed_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'DELETED', ?)
                """,
                (
                    message_id,
                    revision_no,
                    str(row["content_sha256"] or ""),
                    str(row["plain_text"]),
                    str(row["content_json"]),
                    str(row["role"]),
                    when,
                ),
            )
            self._connection.execute(
                """
                UPDATE messages
                SET is_deleted = 1, deleted_at = ?, revision_no = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (when, revision_no + 1, message_id),
            )
            self._connection.execute(
                """
                INSERT INTO message_processing(message_id, content_sha256, status)
                VALUES (?, ?, 'DELETED')
                ON CONFLICT(message_id) DO UPDATE SET
                    status='DELETED', batch_key='', last_error='',
                    updated_at=CURRENT_TIMESTAMP
                """,
                (message_id, str(row["content_sha256"] or "")),
            )
            self._invalidate_message_derivations_locked(
                message_id=message_id,
                reason=reason,
            )
        return True

    @staticmethod
    def _forgotten_account_hash(
        *,
        umo: str,
        platform_id: str,
        account_id: str,
    ) -> str:
        payload = "\x1f".join((umo, platform_id, account_id)).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def is_account_forgotten(
        self,
        *,
        umo: str,
        platform_id: str,
        account_id: str,
    ) -> bool:
        digest = self._forgotten_account_hash(
            umo=umo,
            platform_id=platform_id,
            account_id=str(account_id),
        )
        with self._lock:
            row = self._connection.execute(
                """
                SELECT 1 FROM forgotten_accounts
                WHERE umo = ? AND platform_id = ? AND account_hash = ?
                """,
                (umo, platform_id, digest),
            ).fetchone()
        return row is not None

    def _is_account_forgotten_locked(
        self,
        *,
        umo: str,
        platform_id: str,
        account_id: str,
    ) -> bool:
        account = str(account_id or "").strip()
        if not account:
            return False
        digest = self._forgotten_account_hash(
            umo=umo,
            platform_id=platform_id,
            account_id=account,
        )
        return self._connection.execute(
            """
            SELECT 1 FROM forgotten_accounts
            WHERE umo = ? AND platform_id = ? AND account_hash = ?
            """,
            (umo, platform_id, digest),
        ).fetchone() is not None

    def _scrub_forgotten_references_locked(
        self,
        *,
        umo: str,
        platform_id: str,
        content: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """Remove structured account bindings after a self-erasure request."""

        sanitized: list[dict[str, object]] = []
        for raw in content:
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            kind = str(item.get("type") or "").strip().casefold()
            if kind in {"mention", "at"}:
                account = str(
                    item.get("account_id")
                    or item.get("qq")
                    or item.get("user_id")
                    or ""
                ).strip()
                if self._is_account_forgotten_locked(
                    umo=umo,
                    platform_id=platform_id,
                    account_id=account,
                ):
                    for key in (
                        "account_id", "qq", "user_id", "display_name",
                        "name", "nickname",
                    ):
                        item.pop(key, None)
                    item["erased_participant"] = True
            elif kind in {"reply", "response_to", "quote"}:
                account = str(
                    item.get("sender_id")
                    or item.get("account_id")
                    or item.get("qq")
                    or ""
                ).strip()
                if self._is_account_forgotten_locked(
                    umo=umo,
                    platform_id=platform_id,
                    account_id=account,
                ):
                    for key in (
                        "sender_id", "account_id", "qq", "sender_name",
                        "sender_nickname", "nickname", "plain_text",
                        "message_str", "text",
                    ):
                        item.pop(key, None)
                    item["erased_participant"] = True
            sanitized.append(item)
        return sanitized

    def forget_account(
        self,
        *,
        umo: str,
        platform_id: str,
        account_id: str,
        requested_at: int | None = None,
    ) -> dict[str, int]:
        """Erase one account's derived and raw content, then suppress recapture."""

        account = str(account_id or "").strip()
        if not account:
            raise ValueError("account_id is required")
        when = int(requested_at or time.time())
        digest = self._forgotten_account_hash(
            umo=umo,
            platform_id=platform_id,
            account_id=account,
        )
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO forgotten_accounts(
                    umo, platform_id, account_hash, requested_at
                ) VALUES (?, ?, ?, ?)
                """,
                (umo, platform_id, digest, when),
            )
            participant = self._connection.execute(
                """
                SELECT id, canonical_key, current_display_name FROM participants
                WHERE umo = ? AND platform_id = ? AND account_id = ?
                """,
                (umo, platform_id, account),
            ).fetchone()
            if participant is None:
                return {"messages": 0, "episodes": 0, "claims": 0, "traces": 0}
            participant_id = int(participant["id"])
            alias_rows = self._connection.execute(
                """
                SELECT alias FROM participant_aliases
                WHERE participant_id = ?
                """,
                (participant_id,),
            ).fetchall()
            identity_cues = {
                account,
                str(participant["canonical_key"] or ""),
                str(participant["current_display_name"] or ""),
                *[str(row["alias"] or "") for row in alias_rows],
            }
            identity_cues.discard("")
            speaker_rows = self._connection.execute(
                """
                SELECT id, source_key, message_id FROM messages
                WHERE umo = ? AND sender_participant_id = ?
                """,
                (umo, participant_id),
            ).fetchall()
            message_ids = {int(row["id"]) for row in speaker_rows}
            erased_platform_message_ids = {
                str(row["message_id"]) for row in speaker_rows
            }
            if message_ids:
                placeholders = ",".join("?" for _ in message_ids)
                dependent = self._connection.execute(
                    f"""
                    SELECT source_message_id FROM message_relations
                    WHERE umo = ? AND target_message_id IN ({placeholders})
                      AND relation = 'RESPONDS_TO'
                    """,
                    (umo, *message_ids),
                ).fetchall()
                message_ids.update(int(row["source_message_id"]) for row in dependent)
                dependent_ids = [int(row["source_message_id"]) for row in dependent]
                if dependent_ids:
                    placeholders = ",".join("?" for _ in dependent_ids)
                    erased_platform_message_ids.update(
                        str(row["message_id"])
                        for row in self._connection.execute(
                            f"SELECT message_id FROM messages WHERE id IN ({placeholders})",
                            dependent_ids,
                        ).fetchall()
                    )

            reference_parameters: list[object] = [umo, participant_id]
            target_message_sql = ""
            if message_ids:
                target_message_sql = (
                    f" OR mr.target_message_id IN "
                    f"({','.join('?' for _ in message_ids)})"
                )
                reference_parameters.extend(message_ids)
            reference_rows = self._connection.execute(
                f"""
                SELECT DISTINCT m.id, m.source_key, m.platform_id, m.message_id,
                       m.sender_id, m.sender_participant_id, m.sent_at,
                       m.plain_text, m.content_json, m.role
                FROM messages AS m
                LEFT JOIN message_participants AS mp ON mp.message_id = m.id
                LEFT JOIN message_relations AS mr ON mr.source_message_id = m.id
                WHERE m.umo = ? AND m.is_deleted = 0
                  AND (mp.participant_id = ?
                       OR mr.target_participant_id = ?{target_message_sql})
                """,
                (
                    reference_parameters[0],
                    reference_parameters[1],
                    reference_parameters[1],
                    *reference_parameters[2:],
                ),
            ).fetchall()
            reference_rows = [
                row for row in reference_rows if int(row["id"]) not in message_ids
            ]

            episode_ids: set[int] = set()
            claim_ids: set[int] = set()
            source_keys: set[str] = set()
            if message_ids:
                placeholders = ",".join("?" for _ in message_ids)
                source_keys.update(
                    str(row["source_key"])
                    for row in self._connection.execute(
                        f"SELECT source_key FROM messages WHERE id IN ({placeholders})",
                        tuple(message_ids),
                    ).fetchall()
                )
                episode_ids.update(
                    int(row["episode_id"])
                    for row in self._connection.execute(
                        f"""
                        SELECT DISTINCT episode_id FROM episode_messages
                        WHERE message_id IN ({placeholders})
                        """,
                        tuple(message_ids),
                    ).fetchall()
                )
                claim_ids.update(
                    int(row["semantic_memory_id"])
                    for row in self._connection.execute(
                        f"""
                        SELECT DISTINCT semantic_memory_id
                        FROM semantic_memory_sources
                        WHERE message_id IN ({placeholders})
                        """,
                        tuple(message_ids),
                    ).fetchall()
                )
            claim_ids.update(
                int(row["id"])
                for row in self._connection.execute(
                    """
                    SELECT id FROM semantic_memories
                    WHERE umo = ? AND subject_participant_id = ?
                    """,
                    (umo, participant_id),
                ).fetchall()
            )

            if episode_ids:
                placeholders = ",".join("?" for _ in episode_ids)
                remaining = self._connection.execute(
                    f"""
                    SELECT DISTINCT em.message_id
                    FROM episode_messages AS em
                    JOIN messages AS m ON m.id = em.message_id
                    WHERE em.episode_id IN ({placeholders})
                      AND em.message_id NOT IN ({','.join('?' for _ in message_ids)})
                      AND m.is_deleted = 0
                    """,
                    (*episode_ids, *message_ids),
                ).fetchall() if message_ids else []
                self._connection.executemany(
                    """
                    UPDATE message_processing
                    SET status='PENDING', batch_key='', last_error='',
                        updated_at=CURRENT_TIMESTAMP
                    WHERE message_id = ?
                    """,
                    [(int(row["message_id"]),) for row in remaining],
                )
                self._connection.executemany(
                    """
                    DELETE FROM memory_embeddings
                    WHERE owner_type='episode' AND owner_key=?
                    """,
                    [(str(item),) for item in episode_ids],
                )
                self._connection.execute(
                    f"DELETE FROM distilled_units WHERE umo=? AND unit_type='episode' "
                    f"AND unit_id IN ({placeholders})",
                    (umo, *episode_ids),
                )
                self._connection.execute(
                    f"DELETE FROM episodes WHERE umo=? AND id IN ({placeholders})",
                    (umo, *episode_ids),
                )
            if claim_ids:
                placeholders = ",".join("?" for _ in claim_ids)
                dependent_claims = self._connection.execute(
                    f"""
                    SELECT id, status FROM semantic_memories
                    WHERE umo=? AND superseded_by IN ({placeholders})
                      AND id NOT IN ({placeholders})
                    """,
                    (umo, *claim_ids, *claim_ids),
                ).fetchall()
                if dependent_claims:
                    dependent_ids = [int(row["id"]) for row in dependent_claims]
                    dependent_placeholders = ",".join(
                        "?" for _ in dependent_ids
                    )
                    self._connection.execute(
                        f"""
                        UPDATE semantic_memories
                        SET status='STALE', superseded_by=NULL,
                            updated_at=CURRENT_TIMESTAMP
                        WHERE umo=? AND id IN ({dependent_placeholders})
                        """,
                        (umo, *dependent_ids),
                    )
                    self._connection.executemany(
                        """
                        INSERT INTO semantic_memory_revisions(
                            semantic_memory_id, previous_status, new_status,
                            reason
                        ) VALUES (?, ?, 'STALE',
                                  'superseding claim erased by participant')
                        """,
                        [
                            (int(row["id"]), str(row["status"]))
                            for row in dependent_claims
                        ],
                    )
                    self._connection.executemany(
                        """
                        DELETE FROM memory_embeddings
                        WHERE owner_type='semantic' AND owner_key=?
                        """,
                        [(str(item),) for item in dependent_ids],
                    )
                    source_rows = self._connection.execute(
                        f"""
                        SELECT DISTINCT message_id
                        FROM semantic_memory_sources
                        WHERE semantic_memory_id IN ({dependent_placeholders})
                        """,
                        dependent_ids,
                    ).fetchall()
                    self._connection.executemany(
                        """
                        UPDATE message_processing
                        SET status='PENDING', batch_key='', attempts=0,
                            last_error='', distilled_at=NULL,
                            updated_at=CURRENT_TIMESTAMP
                        WHERE message_id=?
                        """,
                        [(int(row["message_id"]),) for row in source_rows],
                    )
                self._connection.executemany(
                    """
                    DELETE FROM memory_embeddings
                    WHERE owner_type='semantic' AND owner_key=?
                    """,
                    [(str(item),) for item in claim_ids],
                )
                self._connection.execute(
                    f"DELETE FROM distilled_units WHERE umo=? AND unit_type='semantic' "
                    f"AND unit_id IN ({placeholders})",
                    (umo, *claim_ids),
                )
                self._connection.execute(
                    f"DELETE FROM semantic_memories WHERE umo=? AND id IN ({placeholders})",
                    (umo, *claim_ids),
                )

            self._connection.execute(
                """
                DELETE FROM memory_embeddings
                WHERE umo=? AND owner_type='participant' AND owner_key=?
                """,
                (umo, str(participant_id)),
            )
            if identity_cues:
                placeholders = ",".join("?" for _ in identity_cues)
                self._connection.execute(
                    f"""
                    DELETE FROM memory_embeddings
                    WHERE umo=? AND owner_type='cue'
                      AND owner_key IN ({placeholders})
                    """,
                    (umo, *sorted(identity_cues)),
                )

            trace_ids = [
                str(row["trace_id"])
                for row in self._connection.execute(
                    "SELECT trace_id FROM interaction_traces WHERE umo=? AND sender_id=?",
                    (umo, account),
                ).fetchall()
            ]
            if trace_ids:
                placeholders = ",".join("?" for _ in trace_ids)
                self._connection.execute(
                    f"""
                    UPDATE trace_nodes SET content_json='{{}}'
                    WHERE umo=? AND trace_id IN ({placeholders})
                    """,
                    (umo, *trace_ids),
                )
                self._connection.execute(
                    f"""
                    UPDATE interaction_traces
                    SET sender_id='', request_excerpt='', response_excerpt='',
                        updated_at=CURRENT_TIMESTAMP
                    WHERE umo=? AND trace_id IN ({placeholders})
                    """,
                    (umo, *trace_ids),
                )
            trace_count = len(trace_ids)
            if source_keys:
                placeholders = ",".join("?" for _ in source_keys)
                affected_hypotheses = [
                    int(row["hypothesis_id"])
                    for row in self._connection.execute(
                        f"""
                        SELECT DISTINCT hypothesis_id FROM hypothesis_evidence
                        WHERE feedback_source_key IN ({placeholders})
                        """,
                        tuple(source_keys),
                    ).fetchall()
                ]
                if affected_hypotheses:
                    hypothesis_placeholders = ",".join(
                        "?" for _ in affected_hypotheses
                    )
                    self._connection.execute(
                        f"""
                        UPDATE feedback_hypotheses
                        SET scope_key='', aspect='', statement='',
                            prospective_cue='', trigger_cues_json='[]',
                            evidence_confidence=0, utility=0,
                            status='DORMANT', updated_at=CURRENT_TIMESTAMP
                        WHERE umo=? AND id IN ({hypothesis_placeholders})
                        """,
                        (umo, *affected_hypotheses),
                    )
                self._connection.execute(
                    f"""
                    DELETE FROM hypothesis_evidence
                    WHERE feedback_source_key IN ({placeholders})
                    """,
                    tuple(source_keys),
                )
                self._connection.execute(
                    f"DELETE FROM feedback_proposals WHERE umo=? "
                    f"AND feedback_source_key IN ({placeholders})",
                    (umo, *source_keys),
                )
                self._connection.execute(
                    f"DELETE FROM feedback_links WHERE umo=? "
                    f"AND feedback_source_key IN ({placeholders})",
                    (umo, *source_keys),
                )
            self._connection.execute(
                """
                UPDATE feedback_hypotheses
                SET scope_key='', statement='', prospective_cue='',
                    trigger_cues_json='[]', status='DORMANT',
                    updated_at=CURRENT_TIMESTAMP
                WHERE umo=? AND scope_key=?
                """,
                (umo, account),
            )

            for row in reference_rows:
                message_id = int(row["id"])
                content = self._scrub_forgotten_references_locked(
                    umo=umo,
                    platform_id=str(row["platform_id"]),
                    content=self._parse_content_json(row["content_json"]),
                )
                for item in content:
                    kind = str(item.get("type") or "").casefold()
                    reply_message_id = str(
                        item.get("message_id") or item.get("id") or ""
                    )
                    if (
                        kind in {"reply", "response_to", "quote"}
                        and reply_message_id in erased_platform_message_ids
                    ):
                        for key in (
                            "sender_id", "account_id", "qq", "sender_name",
                            "sender_nickname", "nickname", "plain_text",
                            "message_str", "text",
                        ):
                            item.pop(key, None)
                        item["erased_participant"] = True
                content_json = json.dumps(
                    content,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                digest = content_fingerprint(
                    sender_id=str(row["sender_id"]),
                    role=str(row["role"]),
                    plain_text=str(row["plain_text"]),
                    content=content,
                )
                self._invalidate_message_derivations_locked(
                    message_id=message_id,
                    reason="structured participant reference erased",
                )
                self._connection.execute(
                    "DELETE FROM message_revisions WHERE message_id = ?",
                    (message_id,),
                )
                self._connection.execute(
                    """
                    UPDATE messages
                    SET content_json=?, content_sha256=?,
                        revision_no=revision_no+1,
                        updated_at=CURRENT_TIMESTAMP
                    WHERE id=?
                    """,
                    (content_json, digest, message_id),
                )
                self._connection.execute(
                    """
                    INSERT INTO message_processing(
                        message_id, content_sha256, status
                    ) VALUES (?, ?, 'PENDING')
                    ON CONFLICT(message_id) DO UPDATE SET
                        content_sha256=excluded.content_sha256,
                        status='PENDING', batch_key='', attempts=0,
                        last_error='', distilled_at=NULL,
                        updated_at=CURRENT_TIMESTAMP
                    """,
                    (message_id, digest),
                )
                self._refresh_message_links_locked(
                    message_id=message_id,
                    umo=umo,
                    platform_id=str(row["platform_id"]),
                    source_key=str(row["source_key"]),
                    platform_message_id=str(row["message_id"]),
                    sender_participant_id=(
                        int(row["sender_participant_id"])
                        if row["sender_participant_id"] is not None
                        else None
                    ),
                    sent_at=int(row["sent_at"]),
                    content=content,
                )

            for message_id in message_ids:
                self._connection.execute(
                    "DELETE FROM message_revisions WHERE message_id = ?",
                    (message_id,),
                )
                self._connection.execute(
                    """
                    UPDATE messages
                    SET sender_id='', sender_name='', plain_text='',
                        content_json='[]', content_sha256='', is_deleted=1,
                        deleted_at=?, updated_at=CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (when, message_id),
                )
                self._connection.execute(
                    """
                    UPDATE message_processing
                    SET status='DELETED', batch_key='', last_error='',
                        updated_at=CURRENT_TIMESTAMP
                    WHERE message_id = ?
                    """,
                    (message_id,),
                )
                self._connection.execute(
                    "DELETE FROM message_participants WHERE message_id = ?",
                    (message_id,),
                )
                self._connection.execute(
                    "DELETE FROM message_relations WHERE source_message_id = ?",
                    (message_id,),
                )
                self._connection.execute(
                    "DELETE FROM message_attachments WHERE message_id = ?",
                    (message_id,),
                )
            self._connection.execute(
                "DELETE FROM participant_aliases WHERE participant_id = ?",
                (participant_id,),
            )
            self._connection.execute(
                """
                UPDATE messages SET sender_participant_id=NULL
                WHERE sender_participant_id=?
                """,
                (participant_id,),
            )
            self._connection.execute(
                """
                UPDATE message_relations SET target_participant_id=NULL,
                    metadata_json='{}'
                WHERE target_participant_id=?
                """,
                (participant_id,),
            )
            self._connection.execute(
                "DELETE FROM message_participants WHERE participant_id = ?",
                (participant_id,),
            )
            self._connection.execute(
                "DELETE FROM participants WHERE id = ?",
                (participant_id,),
            )
            self._connection.execute(
                """
                DELETE FROM topics WHERE umo = ? AND NOT EXISTS (
                    SELECT 1 FROM topic_episodes AS te
                    WHERE te.topic_id = topics.id
                )
                """,
                (umo,),
            )
        return {
            "messages": len(message_ids),
            "episodes": len(episode_ids),
            "claims": len(claim_ids),
            "traces": int(trace_count),
        }

    def bind_participant_alias(
        self,
        *,
        umo: str,
        platform_id: str,
        account_id: str,
        alias: str,
        at: int | None = None,
    ) -> dict[str, object]:
        """Administrator-authoritative alias binding; account IDs never merge."""

        with self._lock, self._connection:
            participant_id = self._upsert_participant_locked(
                umo=umo,
                platform_id=platform_id,
                account_id=account_id,
                display_name="",
                seen_at=int(at or time.time()),
                account_type="USER",
                alias_source="administrator",
            )
            if participant_id is None:
                raise ValueError("account_id is required")
            self._upsert_alias_locked(
                participant_id=participant_id,
                alias=alias,
                seen_at=int(at or time.time()),
                source_kind="administrator",
                confidence=1.0,
            )
            row = self._connection.execute(
                "SELECT * FROM participants WHERE id = ?",
                (participant_id,),
            ).fetchone()
        return dict(row) if row is not None else {}

    def resolve_participants(
        self,
        *,
        umo: str,
        reference: str,
        limit: int = 20,
    ) -> dict[str, object]:
        """Resolve only exact IDs/keys/aliases and report alias ambiguity."""

        query = str(reference or "").strip()
        normalized = normalize_alias(query)
        safe_limit = max(1, min(100, int(limit)))
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT DISTINCT p.id, p.canonical_key, p.platform_id,
                       p.account_id, p.account_type, p.current_display_name,
                       p.first_seen_at, p.last_seen_at
                FROM participants AS p
                LEFT JOIN participant_aliases AS a
                  ON a.participant_id = p.id AND a.is_active = 1
                WHERE p.umo = ? AND (
                    p.account_id = ? OR p.canonical_key = ?
                    OR a.normalized_alias = ?
                )
                ORDER BY p.last_seen_at DESC, p.id
                LIMIT ?
                """,
                (umo, query, query, normalized, safe_limit),
            ).fetchall()
            result: list[dict[str, object]] = []
            for row in rows:
                aliases = self._connection.execute(
                    """
                    SELECT alias, first_seen_at, last_seen_at, source_kind,
                           confidence
                    FROM participant_aliases
                    WHERE participant_id = ? AND is_active = 1
                    ORDER BY last_seen_at DESC, alias
                    """,
                    (int(row["id"]),),
                ).fetchall()
                result.append(
                    {
                        **dict(row),
                        "aliases": [dict(alias) for alias in aliases],
                    }
                )
        alias_only = bool(query) and all(
            str(item["account_id"]) != query
            and str(item["canonical_key"]) != query
            for item in result
        )
        return {
            "reference": query,
            "ambiguous": alias_only and len(result) > 1,
            "participants": result,
        }

    def list_participants(
        self,
        *,
        umo: str,
        limit: int = 200,
    ) -> list[dict[str, object]]:
        safe_limit = max(1, min(2000, int(limit)))
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT id, canonical_key, platform_id, account_id, account_type,
                       current_display_name, first_seen_at, last_seen_at
                FROM participants WHERE umo = ?
                ORDER BY last_seen_at DESC, id LIMIT ?
                """,
                (umo, safe_limit),
            ).fetchall()
            result = []
            for row in rows:
                aliases = self._connection.execute(
                    """
                    SELECT alias, normalized_alias, first_seen_at, last_seen_at,
                           observation_count, source_kind, confidence
                    FROM participant_aliases
                    WHERE participant_id = ? AND is_active = 1
                    ORDER BY last_seen_at DESC
                    """,
                    (int(row["id"]),),
                ).fetchall()
                result.append({**dict(row), "aliases": [dict(a) for a in aliases]})
        return result

    def participant_embedding_documents(
        self,
        *,
        umo: str,
    ) -> list[dict[str, str]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT p.id, p.canonical_key, p.account_id,
                       p.current_display_name,
                       GROUP_CONCAT(a.alias, '\n') AS aliases
                FROM participants AS p
                LEFT JOIN participant_aliases AS a
                  ON a.participant_id = p.id AND a.is_active = 1
                WHERE p.umo = ?
                GROUP BY p.id ORDER BY p.id
                """,
                (umo,),
            ).fetchall()
        return [
            {
                "owner_key": str(row["id"]),
                "text": "\n".join(
                    item
                    for item in (
                        str(row["canonical_key"]),
                        str(row["account_id"]),
                        str(row["current_display_name"] or ""),
                        str(row["aliases"] or ""),
                    )
                    if item
                ),
            }
            for row in rows
        ]

    def distillation_identity_context(
        self,
        *,
        umo: str,
        source_keys: list[str],
        max_participants: int = 200,
        max_claims: int = 200,
    ) -> dict[str, object]:
        """Build the host-authoritative identity choices exposed to extraction.

        The model may select one of these keys, but it may never invent a new
        account binding. Text aliases that map to several accounts are explicitly
        marked ambiguous.
        """

        if not source_keys:
            return {"messages": {}, "participants": [], "ambiguous_aliases": {}}
        placeholders = ",".join("?" for _ in source_keys)
        safe_participants = max(1, min(1000, int(max_participants)))
        with self._lock:
            messages = self._connection.execute(
                f"""
                SELECT id, source_key, plain_text, sender_participant_id
                FROM messages
                WHERE umo = ? AND source_key IN ({placeholders})
                  AND is_deleted = 0
                """,
                (umo, *source_keys),
            ).fetchall()
            participant_ids: set[int] = {
                int(row["sender_participant_id"])
                for row in messages
                if row["sender_participant_id"] is not None
            }
            if messages:
                message_ids = [int(row["id"]) for row in messages]
                message_placeholders = ",".join("?" for _ in message_ids)
                linked = self._connection.execute(
                    f"""
                    SELECT DISTINCT participant_id FROM message_participants
                    WHERE message_id IN ({message_placeholders})
                    """,
                    message_ids,
                ).fetchall()
                participant_ids.update(int(row["participant_id"]) for row in linked)

            # Nicknames used as plain text become candidates, never automatic binds.
            texts = [str(row["plain_text"]).casefold() for row in messages]
            alias_rows = self._connection.execute(
                """
                SELECT a.normalized_alias, a.participant_id
                FROM participant_aliases AS a
                JOIN participants AS p ON p.id = a.participant_id
                WHERE p.umo = ? AND a.is_active = 1
                  AND length(a.normalized_alias) >= 2
                ORDER BY a.last_seen_at DESC LIMIT 5000
                """,
                (umo,),
            ).fetchall()
            alias_map: dict[str, set[int]] = {}
            for alias in alias_rows:
                normalized_alias = str(alias["normalized_alias"])
                if any(normalized_alias in text for text in texts):
                    participant_id = int(alias["participant_id"])
                    participant_ids.add(participant_id)
                    alias_map.setdefault(normalized_alias, set()).add(participant_id)

            selected_ids = sorted(participant_ids)[:safe_participants]
            participants: list[dict[str, object]] = []
            if selected_ids:
                selected_placeholders = ",".join("?" for _ in selected_ids)
                rows = self._connection.execute(
                    f"""
                    SELECT id, canonical_key, platform_id, account_id,
                           account_type, current_display_name,
                           first_seen_at, last_seen_at
                    FROM participants
                    WHERE umo = ? AND id IN ({selected_placeholders})
                    ORDER BY last_seen_at DESC, id
                    """,
                    (umo, *selected_ids),
                ).fetchall()
                for row in rows:
                    aliases = self._connection.execute(
                        """
                        SELECT alias FROM participant_aliases
                        WHERE participant_id = ? AND is_active = 1
                        ORDER BY last_seen_at DESC LIMIT 20
                        """,
                        (int(row["id"]),),
                    ).fetchall()
                    participants.append(
                        {
                            "participant_key": str(row["canonical_key"]),
                            "account_id": str(row["account_id"]),
                            "account_type": str(row["account_type"]),
                            "current_display_name": str(row["current_display_name"]),
                            "aliases": [str(alias["alias"]) for alias in aliases],
                        }
                    )

            message_context: dict[str, object] = {}
            for message in messages:
                speaker_key = ""
                if message["sender_participant_id"] is not None:
                    speaker = self._connection.execute(
                        "SELECT canonical_key FROM participants WHERE id = ?",
                        (int(message["sender_participant_id"]),),
                    ).fetchone()
                    if speaker is not None:
                        speaker_key = str(speaker["canonical_key"])
                links = self._connection.execute(
                    """
                    SELECT mp.relation, p.canonical_key
                    FROM message_participants AS mp
                    JOIN participants AS p ON p.id = mp.participant_id
                    WHERE mp.message_id = ? AND mp.relation <> 'SPEAKER'
                    ORDER BY mp.position, p.id
                    """,
                    (int(message["id"]),),
                ).fetchall()
                normalized_text = normalize_alias(message["plain_text"])
                unique_text_candidates: set[str] = set()
                ambiguous_text_aliases: dict[str, list[str]] = {}
                for alias, ids in alias_map.items():
                    if alias not in normalized_text:
                        continue
                    keys = [
                        self._participant_key_by_id(participant_id)
                        for participant_id in sorted(ids)
                    ]
                    keys = [key for key in keys if key]
                    if len(keys) == 1:
                        unique_text_candidates.add(keys[0])
                    elif keys:
                        ambiguous_text_aliases[alias] = keys
                message_context[str(message["source_key"])] = {
                    "speaker_participant_key": speaker_key,
                    "linked_participants": [
                        {
                            "relation": str(link["relation"]),
                            "participant_key": str(link["canonical_key"]),
                        }
                        for link in links
                    ],
                    "unique_text_candidate_participant_keys": sorted(
                        unique_text_candidates
                    ),
                    "ambiguous_text_aliases": ambiguous_text_aliases,
                }

            claims: list[dict[str, object]] = []
            if selected_ids:
                selected_placeholders = ",".join("?" for _ in selected_ids)
                claim_rows = self._connection.execute(
                    f"""
                    SELECT s.id, p.canonical_key AS subject_participant_key,
                           s.person_cue, s.aspect_tag, s.content, s.claim_type,
                           s.epistemic_status, s.status, s.confidence
                    FROM semantic_memories AS s
                    JOIN participants AS p ON p.id = s.subject_participant_id
                    WHERE s.umo = ? AND s.subject_participant_id
                          IN ({selected_placeholders})
                      AND s.status IN ('ACTIVE', 'CONFLICTED', 'QUARANTINED')
                    ORDER BY s.confidence DESC, s.id DESC LIMIT ?
                    """,
                    (umo, *selected_ids, max(1, min(1000, int(max_claims)))),
                ).fetchall()
                claims = [dict(row) for row in claim_rows]

        return {
            "messages": message_context,
            "participants": participants,
            "ambiguous_aliases": {
                alias: [
                    next(
                        (
                            str(item["participant_key"])
                            for item in participants
                            if str(item["participant_key"])
                            == self._participant_key_by_id(participant_id)
                        ),
                        "",
                    )
                    for participant_id in sorted(ids)
                ]
                for alias, ids in alias_map.items()
                if len(ids) > 1
            },
            "active_claims": claims,
        }

    def _participant_key_by_id(self, participant_id: int) -> str:
        with self._lock:
            row = self._connection.execute(
                "SELECT canonical_key FROM participants WHERE id = ?",
                (int(participant_id),),
            ).fetchone()
        return str(row["canonical_key"]) if row is not None else ""

    def pending_distillation_count(self, *, umo: str) -> int:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT COUNT(*)
                FROM message_processing AS p
                JOIN messages AS m ON m.id = p.message_id
                WHERE m.umo = ? AND m.is_deleted = 0
                  AND (p.status = 'PENDING'
                       OR (p.status = 'FAILED' AND p.attempts < 3))
                """,
                (umo,),
            ).fetchone()
        return int(row[0])

    def next_distillation_batch(
        self,
        *,
        umo: str,
        limit: int = 80,
        overlap: int = 12,
    ) -> DistillationWorkItem | None:
        safe_limit = max(1, min(500, int(limit)))
        safe_overlap = max(0, min(100, int(overlap)))
        with self._lock, self._connection:
            targets = self._connection.execute(
                """
                SELECT m.id, m.source_key, m.content_sha256, m.sent_at
                FROM message_processing AS p
                JOIN messages AS m ON m.id = p.message_id
                WHERE m.umo = ? AND m.is_deleted = 0
                  AND (p.status = 'PENDING'
                       OR (p.status = 'FAILED' AND p.attempts < 3))
                ORDER BY m.sent_at, m.id LIMIT ?
                """,
                (umo, safe_limit),
            ).fetchall()
            if not targets:
                return None
            first = targets[0]
            context = self._connection.execute(
                """
                SELECT m.*
                FROM messages AS m
                WHERE m.umo = ? AND m.is_deleted = 0
                  AND (m.sent_at < ? OR (m.sent_at = ? AND m.id < ?))
                ORDER BY m.sent_at DESC, m.id DESC LIMIT ?
                """,
                (
                    umo,
                    int(first["sent_at"]),
                    int(first["sent_at"]),
                    int(first["id"]),
                    safe_overlap,
                ),
            ).fetchall()
            target_ids = [int(row["id"]) for row in targets]
            placeholders = ",".join("?" for _ in target_ids)
            target_rows = self._connection.execute(
                f"SELECT * FROM messages WHERE id IN ({placeholders})",
                target_ids,
            ).fetchall()
            rows = sorted(
                [*context, *target_rows],
                key=lambda row: (int(row["sent_at"]), int(row["id"])),
            )
            target_hashes = tuple(
                (str(row["source_key"]), str(row["content_sha256"]))
                for row in targets
            )
            batch_payload = json.dumps(
                {"umo": umo, "targets": target_hashes},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            batch_key = hashlib.sha256(batch_payload.encode("utf-8")).hexdigest()
            self._connection.execute(
                """
                INSERT INTO distillation_batches(
                    batch_key, umo, target_source_keys_json,
                    target_hashes_json, status
                ) VALUES (?, ?, ?, ?, 'RUNNING')
                ON CONFLICT(batch_key) DO UPDATE SET
                    status='RUNNING', error='', finished_at=NULL
                """,
                (
                    batch_key,
                    umo,
                    json.dumps([item[0] for item in target_hashes]),
                    json.dumps(target_hashes),
                ),
            )
            self._connection.executemany(
                """
                UPDATE message_processing
                SET status='PROCESSING', batch_key=?, attempts=attempts+1,
                    last_error='', updated_at=CURRENT_TIMESTAMP
                WHERE message_id = ?
                """,
                [(batch_key, target_id) for target_id in target_ids],
            )
            stored = tuple(self._stored_message_from_row(row) for row in rows)
        return DistillationWorkItem(
            batch_key=batch_key,
            umo=umo,
            messages=stored,
            target_source_keys=tuple(item[0] for item in target_hashes),
            target_hashes=target_hashes,
        )

    def finish_distillation_batch(
        self,
        *,
        work_item: DistillationWorkItem,
        error: str = "",
    ) -> None:
        success = not str(error).strip()
        now = int(time.time())
        expected = dict(work_item.target_hashes)
        with self._lock, self._connection:
            for source_key, expected_hash in expected.items():
                row = self._connection.execute(
                    """
                    SELECT m.id, m.content_sha256, m.is_deleted
                    FROM messages AS m WHERE m.umo = ? AND m.source_key = ?
                    """,
                    (work_item.umo, source_key),
                ).fetchone()
                if row is None:
                    continue
                if int(row["is_deleted"]):
                    status = "DELETED"
                elif success and str(row["content_sha256"]) == expected_hash:
                    status = "DISTILLED"
                elif success:
                    status = "PENDING"
                else:
                    status = "FAILED"
                self._connection.execute(
                    """
                    UPDATE message_processing
                    SET status=?, batch_key='', last_error=?, distilled_at=?,
                        updated_at=CURRENT_TIMESTAMP
                    WHERE message_id = ?
                    """,
                    (
                        status,
                        str(error)[:1000],
                        now if status == "DISTILLED" else None,
                        int(row["id"]),
                    ),
                )
            self._connection.execute(
                """
                UPDATE distillation_batches
                SET status=?, error=?, finished_at=CURRENT_TIMESTAMP
                WHERE batch_key = ?
                """,
                (
                    "COMPLETED" if success else "FAILED",
                    str(error)[:1000],
                    work_item.batch_key,
                ),
            )

    def record_distillation_ignored_sources(
        self,
        *,
        umo: str,
        batch_key: str,
        items: list[dict[str, str]],
    ) -> None:
        """Persist the extractor's bounded coverage ledger for later audit."""

        with self._lock, self._connection:
            batch = self._connection.execute(
                """
                SELECT target_source_keys_json FROM distillation_batches
                WHERE batch_key=? AND umo=?
                """,
                (batch_key, umo),
            ).fetchone()
            if batch is None:
                raise ValueError("distillation batch is missing or outside scope")
            allowed = set(json.loads(str(batch["target_source_keys_json"])))
            self._connection.execute(
                "DELETE FROM distillation_ignored_sources WHERE batch_key=?",
                (batch_key,),
            )
            for item in items:
                source_key = str(item.get("source_key") or "")
                reason = str(item.get("reason") or "").strip()[:300]
                if source_key not in allowed or not reason:
                    raise ValueError("invalid ignored-source coverage record")
                self._connection.execute(
                    """
                    INSERT INTO distillation_ignored_sources(
                        batch_key, source_key, reason
                    ) VALUES (?, ?, ?)
                    ON CONFLICT(batch_key, source_key)
                    DO UPDATE SET reason=excluded.reason
                    """,
                    (batch_key, source_key, reason),
                )

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
                        (SELECT COUNT(*) FROM episodes
                         WHERE umo = ? AND status = 'READY') +
                        (SELECT COUNT(*) FROM semantic_memories
                         WHERE umo = ? AND status IN
                           ('ACTIVE', 'CONFLICTED')) +
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
                         WHERE umo = ? AND status = 'READY' AND ended_at < ?) +
                        (SELECT COUNT(*)
                         FROM semantic_memories AS s
                         WHERE s.umo = ?
                           AND s.status IN
                             ('ACTIVE', 'CONFLICTED')
                           AND EXISTS (
                             SELECT 1 FROM semantic_memory_sources AS ss
                             JOIN messages AS m ON m.id = ss.message_id
                             WHERE ss.semantic_memory_id = s.id
                               AND m.umo = s.umo AND m.sent_at < ?
                               AND m.is_deleted = 0
                           )) +
                        (SELECT COUNT(*) FROM topics AS t
                         WHERE t.umo = ? AND EXISTS (
                             SELECT 1 FROM topic_episodes AS te
                             JOIN episodes AS e ON e.id = te.episode_id
                             WHERE te.topic_id = t.id AND e.umo = t.umo
                               AND e.status = 'READY'
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

    def private_token_usage_since(
        self,
        *,
        umo: str,
        since: int,
    ) -> int:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT COALESCE(SUM(
                    u.input_other + u.input_cached + u.output
                ), 0)
                FROM llm_usage_events AS u
                JOIN experiment_runs AS r ON r.run_id = u.run_id
                WHERE r.umo = ? AND unixepoch(u.created_at) >= ?
                """,
                (umo, int(since)),
            ).fetchone()
        return int(row[0] or 0)

    def dashboard_summary(self, *, umo: str) -> dict[str, object]:
        """Return bounded operational metrics for the authenticated plugin page."""
        with self._lock:
            counts = self._connection.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM messages
                     WHERE umo = ? AND is_deleted = 0) AS messages,
                    (SELECT COUNT(*) FROM episodes
                     WHERE umo = ? AND status = 'READY') AS episodes,
                    (SELECT COUNT(*) FROM semantic_memories
                     WHERE umo = ? AND status IN
                       ('ACTIVE', 'CONFLICTED', 'QUARANTINED'))
                        AS semantic_memories,
                    (SELECT COUNT(*) FROM participants WHERE umo = ?)
                        AS participants,
                    (SELECT COUNT(*) FROM message_processing AS p
                     JOIN messages AS m ON m.id = p.message_id
                     WHERE m.umo = ? AND (
                       p.status = 'PENDING'
                       OR (p.status = 'FAILED' AND p.attempts < 3)
                     ))
                        AS pending_distillation,
                    (SELECT COUNT(*) FROM topics WHERE umo = ?) AS topics,
                    (SELECT COUNT(*) FROM memory_embeddings WHERE umo = ?)
                        AS embeddings,
                    (SELECT COUNT(*) FROM interaction_traces WHERE umo = ?)
                        AS interaction_traces,
                    (SELECT COUNT(*) FROM feedback_hypotheses
                     WHERE umo = ? AND status = 'ACTIVE')
                        AS active_hypotheses,
                    (SELECT COUNT(*) FROM feedback_links WHERE umo = ?)
                        AS feedback_links,
                    (SELECT COUNT(DISTINCT lower(k.cue))
                     FROM episode_keywords AS k
                     JOIN episodes AS e ON e.id = k.episode_id
                     WHERE e.umo = ? AND e.status = 'READY') AS cues,
                    (SELECT MAX(sent_at) FROM messages
                     WHERE umo = ? AND is_deleted = 0) AS last_message_at
                """,
                (
                    umo, umo, umo, umo, umo, umo,
                    umo, umo, umo, umo, umo, umo,
                ),
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
                    UNION ALL
                    SELECT MAX(updated_at) AS value
                    FROM feedback_hypotheses WHERE umo = ?
                )
                """,
                (umo, umo, umo, umo),
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
            "participants": int(counts["participants"] or 0),
            "pending_distillation": int(counts["pending_distillation"] or 0),
            "topics": int(counts["topics"] or 0),
            "embeddings": int(counts["embeddings"] or 0),
            "interaction_traces": int(counts["interaction_traces"] or 0),
            "active_hypotheses": int(counts["active_hypotheses"] or 0),
            "feedback_links": int(counts["feedback_links"] or 0),
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
        """Build a compact knowledge plus observable feedback-loop graph."""
        safe_limit = max(1, min(500, int(limit)))
        with self._lock:
            episode_rows = self._connection.execute(
                """
                SELECT e.id, e.started_at, e.ended_at, e.title, e.summary,
                       COUNT(em.message_id) AS source_count
                FROM episodes AS e
                LEFT JOIN episode_messages AS em ON em.episode_id = e.id
                WHERE e.umo = ? AND e.status = 'READY'
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
                    WHERE e.umo = ? AND e.status = 'READY'
                      AND e.id IN ({placeholders})
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
                      AND e.status = 'READY'
                      AND e.id IN ({placeholders})
                    ORDER BY t.name, te.episode_id
                    LIMIT ?
                    """,
                    (umo, *episode_ids, safe_limit * 2),
                ).fetchall()
            semantic_rows = self._connection.execute(
                """
                SELECT s.id, s.person_cue, s.subject_participant_id,
                       s.aspect_tag, s.content, s.claim_type,
                       s.epistemic_status, s.status, s.confidence,
                       m.source_key, m.plain_text
                FROM semantic_memories AS s
                LEFT JOIN messages AS m
                  ON m.id = s.source_message_id AND m.umo = s.umo
                WHERE s.umo = ? AND s.status IN
                  ('ACTIVE', 'CONFLICTED', 'QUARANTINED')
                ORDER BY s.id DESC
                LIMIT ?
                """,
                (umo, safe_limit),
            ).fetchall()
            participant_rows = self._connection.execute(
                """
                SELECT p.id, p.canonical_key, p.account_id, p.account_type,
                       p.current_display_name, p.first_seen_at, p.last_seen_at,
                       GROUP_CONCAT(a.alias, ' · ') AS aliases
                FROM participants AS p
                LEFT JOIN participant_aliases AS a
                  ON a.participant_id = p.id AND a.is_active = 1
                WHERE p.umo = ?
                GROUP BY p.id
                ORDER BY p.last_seen_at DESC, p.id LIMIT ?
                """,
                (umo, safe_limit),
            ).fetchall()
            hypothesis_rows = self._connection.execute(
                """
                SELECT id, scope_type, scope_key, aspect, statement,
                       prospective_cue, trigger_cues_json, activation_mode,
                       evidence_confidence, utility,
                       support_count, contradict_count, status, learned_at
                FROM feedback_hypotheses
                WHERE umo = ? AND status <> 'MERGED'
                ORDER BY learned_at DESC, id DESC
                LIMIT ?
                """,
                (umo, safe_limit),
            ).fetchall()
            feedback_rows = self._connection.execute(
                """
                SELECT l.id, l.trace_id, l.feedback_source_key,
                       l.feedback_sent_at, l.link_confidence,
                       l.feedback_valence, m.plain_text,
                       he.hypothesis_id, he.relation
                FROM feedback_links AS l
                LEFT JOIN messages AS m
                  ON m.umo = l.umo AND m.source_key = l.feedback_source_key
                LEFT JOIN hypothesis_evidence AS he
                  ON he.trace_id = l.trace_id
                 AND he.feedback_source_key = l.feedback_source_key
                WHERE l.umo = ?
                ORDER BY l.feedback_sent_at DESC, l.id DESC
                LIMIT ?
                """,
                (umo, safe_limit),
            ).fetchall()
            feedback_trace_ids = list(
                dict.fromkeys(str(row["trace_id"]) for row in feedback_rows)
            )
            trace_rows: list[sqlite3.Row] = []
            if feedback_trace_ids:
                placeholders = ",".join("?" for _ in feedback_trace_ids)
                trace_rows = self._connection.execute(
                    f"""
                    SELECT trace_id, sender_id, request_sent_at,
                           request_excerpt, response_excerpt, status
                    FROM interaction_traces
                    WHERE umo = ? AND trace_id IN ({placeholders})
                    ORDER BY request_sent_at DESC
                    """,
                    (umo, *feedback_trace_ids),
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

        for row in reversed(participant_rows):
            participant_node_id = f"participant:{int(row['id'])}"
            label = str(
                row["current_display_name"] or row["account_id"]
            )
            nodes[participant_node_id] = {
                "id": participant_node_id,
                "type": "participant",
                "entity_id": int(row["id"]),
                "label": label,
                "detail": str(row["aliases"] or label),
                "canonical_key": str(row["canonical_key"]),
                "account_id": str(row["account_id"]),
                "account_type": str(row["account_type"]),
                "first_seen_at": int(row["first_seen_at"]),
                "last_seen_at": int(row["last_seen_at"]),
            }

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
                "claim_type": str(row["claim_type"]),
                "epistemic_status": str(row["epistemic_status"]),
                "status": str(row["status"]),
                "confidence": float(row["confidence"]),
                "source_key": str(row["source_key"] or ""),
                "source_text": str(row["plain_text"] or ""),
            }
            if row["subject_participant_id"] is not None:
                subject_id = f"participant:{int(row['subject_participant_id'])}"
            else:
                subject_id = ensure_cue(str(row["person_cue"]))
            edges.append(
                {
                    "source": subject_id,
                    "target": semantic_id,
                    "relation": str(row["aspect_tag"]),
                    "type": "cue_semantic",
                }
            )
        for row in reversed(hypothesis_rows):
            hypothesis_id = f"hypothesis:{int(row['id'])}"
            nodes[hypothesis_id] = {
                "id": hypothesis_id,
                "type": "hypothesis",
                "entity_id": int(row["id"]),
                "label": str(row["aspect"]),
                "detail": str(row["prospective_cue"]),
                "statement": str(row["statement"]),
                "scope_type": str(row["scope_type"]),
                "scope_key": str(row["scope_key"]),
                "activation_mode": str(row["activation_mode"]),
                "trigger_cues": json.loads(str(row["trigger_cues_json"])),
                "confidence": float(row["evidence_confidence"]),
                "utility": float(row["utility"]),
                "status": str(row["status"]),
                "learned_at": int(row["learned_at"]),
                "support_count": int(row["support_count"]),
                "contradict_count": int(row["contradict_count"]),
            }
        for row in trace_rows:
            action_id = f"action:{str(row['trace_id'])}"
            nodes[action_id] = {
                "id": action_id,
                "type": "action",
                "label": str(row["request_excerpt"] or "主 Agent 调用"),
                "detail": str(row["response_excerpt"] or ""),
                "sender_id": str(row["sender_id"]),
                "started_at": int(row["request_sent_at"]),
                "status": str(row["status"]),
            }
        for row in reversed(feedback_rows):
            feedback_id = f"feedback:{int(row['id'])}"
            action_id = f"action:{str(row['trace_id'])}"
            nodes[feedback_id] = {
                "id": feedback_id,
                "type": "feedback",
                "entity_id": int(row["id"]),
                "label": str(row["plain_text"] or "后续反馈"),
                "detail": (
                    f"valence={float(row['feedback_valence']):.2f} · "
                    f"link={float(row['link_confidence']):.2f}"
                ),
                "source_key": str(row["feedback_source_key"]),
                "sent_at": int(row["feedback_sent_at"]),
            }
            if action_id in nodes:
                edges.append(
                    {
                        "source": action_id,
                        "target": feedback_id,
                        "relation": "后续反馈",
                        "type": "action_feedback",
                    }
                )
            if row["hypothesis_id"] is not None:
                hypothesis_id = f"hypothesis:{int(row['hypothesis_id'])}"
                if hypothesis_id in nodes:
                    edges.append(
                        {
                            "source": feedback_id,
                            "target": hypothesis_id,
                            "relation": str(row["relation"] or "更新"),
                            "type": "feedback_hypothesis",
                        }
                    )

        original_node_count = len(nodes)
        if original_node_count > safe_limit:
            degree: dict[str, int] = {node_id: 0 for node_id in nodes}
            for edge in edges:
                degree[str(edge["source"])] = degree.get(str(edge["source"]), 0) + 1
                degree[str(edge["target"])] = degree.get(str(edge["target"]), 0) + 1

            ratios = {
                "episode": 0.20,
                "cue": 0.15,
                "participant": 0.10,
                "semantic": 0.10,
                "topic": 0.10,
                "action": 0.10,
                "feedback": 0.10,
                "hypothesis": 0.15,
            }

            def node_rank(item: dict[str, object]) -> tuple[object, ...]:
                node_type = str(item["type"])
                if node_type == "episode":
                    return (-int(item.get("ended_at") or 0), str(item["id"]))
                if node_type == "semantic":
                    return (-int(item.get("entity_id") or 0), str(item["id"]))
                return (-degree.get(str(item["id"]), 0), str(item["label"]).casefold())

            selected_ids: set[str] = set()
            for node_type in (
                "episode",
                "cue",
                "participant",
                "semantic",
                "topic",
                "action",
                "feedback",
                "hypothesis",
            ):
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

        type_order = {
            "participant": 0,
            "cue": 1,
            "episode": 2,
            "semantic": 3,
            "topic": 4,
            "action": 5,
            "feedback": 6,
            "hypothesis": 7,
        }
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
        owner_types: tuple[str, ...] = (
            "participant",
            "cue",
            "episode",
            "topic",
            "semantic",
        ),
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
        safe_limit = max(1, min(100, int(limit)))
        scored.sort(key=lambda item: float(item["score"]), reverse=True)
        grouped: dict[str, list[dict[str, object]]] = {
            owner_type: [] for owner_type in owner_types
        }
        for item in scored:
            grouped[str(item["owner_type"])].append(item)
        nonempty = [owner_type for owner_type in owner_types if grouped[owner_type]]
        if not nonempty:
            return []
        quota = max(1, safe_limit // len(nonempty))
        selected: list[dict[str, object]] = []
        selected_keys: set[tuple[str, str]] = set()
        for owner_type in nonempty:
            for item in grouped[owner_type][:quota]:
                selected.append(item)
                selected_keys.add((str(item["owner_type"]), str(item["owner_key"])))
        if len(selected) < safe_limit:
            for item in scored:
                key = (str(item["owner_type"]), str(item["owner_key"]))
                if key in selected_keys:
                    continue
                selected.append(item)
                selected_keys.add(key)
                if len(selected) >= safe_limit:
                    break
        selected.sort(key=lambda item: float(item["score"]), reverse=True)
        return selected[:safe_limit]

    def _memory_owner_visible(
        self,
        *,
        umo: str,
        owner_type: str,
        owner_key: str,
        before_sent_at: int | None,
    ) -> bool:
        cutoff = int(before_sent_at) if before_sent_at is not None else None
        with self._lock:
            if owner_type == "episode" and owner_key.isdigit():
                sql = (
                    "SELECT 1 FROM episodes WHERE umo=? AND id=? "
                    "AND status='READY'"
                    + (" AND ended_at < ?" if cutoff is not None else "")
                    + " LIMIT 1"
                )
                parameters: tuple[object, ...] = (umo, int(owner_key))
                if cutoff is not None:
                    parameters = (*parameters, cutoff)
                row = self._connection.execute(sql, parameters).fetchone()
            elif owner_type == "semantic" and owner_key.isdigit():
                cutoff_sql = " AND m.sent_at < ?" if cutoff is not None else ""
                parameters = (umo, int(owner_key))
                if cutoff is not None:
                    parameters = (*parameters, cutoff)
                row = self._connection.execute(
                    f"""
                    SELECT 1
                    FROM semantic_memories AS s
                    JOIN semantic_memory_sources AS ss
                      ON ss.semantic_memory_id = s.id
                    JOIN messages AS m ON m.id = ss.message_id
                    WHERE s.umo = ? AND s.id = ? AND m.umo = s.umo
                      AND s.status IN ('ACTIVE', 'CONFLICTED')
                      AND m.is_deleted = 0{cutoff_sql}
                    LIMIT 1
                    """,
                    parameters,
                ).fetchone()
            elif owner_type == "topic" and owner_key.isdigit():
                cutoff_sql = " AND e.ended_at < ?" if cutoff is not None else ""
                parameters = (umo, int(owner_key))
                if cutoff is not None:
                    parameters = (*parameters, cutoff)
                row = self._connection.execute(
                    f"""
                    SELECT 1
                    FROM topics AS t
                    JOIN topic_episodes AS te ON te.topic_id = t.id
                    JOIN episodes AS e ON e.id = te.episode_id
                    WHERE t.umo = ? AND t.id = ? AND e.umo = t.umo
                      AND e.status = 'READY'{cutoff_sql}
                    LIMIT 1
                    """,
                    parameters,
                ).fetchone()
            elif owner_type == "cue":
                episode_cutoff = " AND e.ended_at < ?" if cutoff is not None else ""
                semantic_cutoff = " AND m.sent_at < ?" if cutoff is not None else ""
                parameters = (umo, owner_key)
                if cutoff is not None:
                    parameters = (*parameters, cutoff)
                parameters = (*parameters, umo, owner_key)
                if cutoff is not None:
                    parameters = (*parameters, cutoff)
                row = self._connection.execute(
                    f"""
                    SELECT 1
                    FROM episode_keywords AS k
                    JOIN episodes AS e ON e.id = k.episode_id
                    WHERE e.umo = ? AND lower(k.cue) = lower(?)
                      AND e.status = 'READY'{episode_cutoff}
                    UNION ALL
                    SELECT 1
                    FROM semantic_memories AS s
                    JOIN semantic_memory_sources AS ss
                      ON ss.semantic_memory_id = s.id
                    JOIN messages AS m ON m.id = ss.message_id
                    WHERE s.umo = ? AND lower(s.person_cue) = lower(?)
                      AND s.status IN ('ACTIVE', 'CONFLICTED')
                      AND m.umo = s.umo AND m.is_deleted = 0{semantic_cutoff}
                    LIMIT 1
                    """,
                    parameters,
                ).fetchone()
            elif owner_type == "participant" and owner_key.isdigit():
                cutoff_sql = " AND first_seen_at < ?" if cutoff is not None else ""
                parameters = (umo, int(owner_key))
                if cutoff is not None:
                    parameters = (*parameters, cutoff)
                row = self._connection.execute(
                    f"""
                    SELECT 1 FROM participants
                    WHERE umo = ? AND id = ?{cutoff_sql} LIMIT 1
                    """,
                    parameters,
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
                  AND e.status = 'READY'
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
            "participants": [],
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
                if owner_type == "participant" and owner_key.isdigit():
                    row = self._connection.execute(
                        """
                        SELECT id, canonical_key, account_id, account_type,
                               current_display_name
                        FROM participants WHERE umo = ? AND id = ?
                        """,
                        (umo, int(owner_key)),
                    ).fetchone()
                    if row:
                        aliases = self._connection.execute(
                            """
                            SELECT alias FROM participant_aliases
                            WHERE participant_id = ? AND is_active = 1
                            ORDER BY last_seen_at DESC LIMIT 20
                            """,
                            (int(owner_key),),
                        ).fetchall()
                        result["participants"].append(
                            {
                                **dict(row),
                                "aliases": [str(item["alias"]) for item in aliases],
                                "score": score,
                            }
                        )
                elif owner_type == "cue":
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
                              AND status IN ('ACTIVE', 'CONFLICTED')
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
                          AND status = 'READY'
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
                            "AND e.status = 'READY' "
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
                        SELECT id, person_cue, aspect_tag, content, claim_type,
                               epistemic_status, status, confidence
                        FROM semantic_memories WHERE umo = ? AND id = ?
                          AND status IN ('ACTIVE', 'CONFLICTED')
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
        stable_key: str = "",
    ) -> int:
        """Persist one distilled Cue--Tag--Episode unit."""
        with self._lock, self._connection:
            existing = None
            if stable_key:
                existing = self._connection.execute(
                    """
                    SELECT id FROM episodes
                    WHERE umo = ? AND stable_key = ? LIMIT 1
                    """,
                    (umo, stable_key),
                ).fetchone()
            if existing is None:
                cursor = self._connection.execute(
                    """
                    INSERT INTO episodes(
                        umo, started_at, ended_at, title, summary, status,
                        extractor_version, stable_key
                    ) VALUES (?, ?, ?, ?, ?, 'READY', ?, ?)
                    """,
                    (
                        umo,
                        int(started_at),
                        int(ended_at),
                        title,
                        summary,
                        extractor_version,
                        stable_key,
                    ),
                )
                episode_id = int(cursor.lastrowid)
            else:
                episode_id = int(existing["id"])
                self._connection.execute(
                    """
                    UPDATE episodes
                    SET started_at=?, ended_at=?, title=?, summary=?,
                        status='READY', extractor_version=?,
                        revision_no=revision_no+1, updated_at=CURRENT_TIMESTAMP
                    WHERE id = ? AND umo = ?
                    """,
                    (
                        int(started_at),
                        int(ended_at),
                        title,
                        summary,
                        extractor_version,
                        episode_id,
                        umo,
                    ),
                )
                self._connection.execute(
                    "DELETE FROM episode_messages WHERE episode_id = ?",
                    (episode_id,),
                )
                self._connection.execute(
                    "DELETE FROM episode_keywords WHERE episode_id = ?",
                    (episode_id,),
                )
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
        """Persist one distilled Participant--Aspect--Structured Claim unit."""
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

    def store_semantic_claim(
        self,
        *,
        umo: str,
        stable_key: str,
        subject_participant_key: str,
        subject_text: str,
        claim_type: str,
        aspect: str,
        content: str,
        epistemic_status: str,
        operation: str,
        target_claim_ids: list[int],
        evidence: list[dict[str, object]],
        confidence: float,
        extractor_version: str = "",
    ) -> int:
        """Persist a structured, multi-source claim and its revision transition."""

        if not evidence:
            raise ValueError("semantic claim needs source evidence")
        with self._lock, self._connection:
            participant_id = None
            person_cue = str(subject_text or "").strip()
            if subject_participant_key:
                participant = self._connection.execute(
                    """
                    SELECT id, current_display_name, account_id
                    FROM participants WHERE umo = ? AND canonical_key = ?
                    """,
                    (umo, subject_participant_key),
                ).fetchone()
                if participant is None:
                    raise ValueError("semantic claim subject is not a bound participant")
                participant_id = int(participant["id"])
                person_cue = str(
                    participant["current_display_name"]
                    or participant["account_id"]
                )
            if not person_cue:
                raise ValueError("semantic claim subject is empty")

            source_rows: list[tuple[int, str, str, float]] = []
            independent_user_speakers: set[int] = set()
            for item in evidence:
                source_key = str(item.get("source_key") or "")
                source = self._connection.execute(
                    """
                    SELECT id, plain_text, role, sender_participant_id
                    FROM messages
                    WHERE umo = ? AND source_key = ? AND is_deleted = 0
                    """,
                    (umo, source_key),
                ).fetchone()
                if source is None:
                    raise ValueError("semantic claim source is missing or deleted")
                span = str(item.get("span") or "")
                if span and span not in str(source["plain_text"]):
                    raise ValueError("semantic claim span is not in its source")
                source_rows.append(
                    (
                        int(source["id"]),
                        str(item.get("role") or "SUPPORT").upper(),
                        span,
                        max(0.0, min(1.0, float(item.get("confidence") or 0))),
                    )
                )
                if (
                    str(source["role"]) == "USER"
                    and source["sender_participant_id"] is not None
                ):
                    independent_user_speakers.add(
                        int(source["sender_participant_id"])
                    )

            targets: list[sqlite3.Row] = []
            if target_claim_ids:
                placeholders = ",".join("?" for _ in target_claim_ids)
                targets = self._connection.execute(
                    f"""
                    SELECT id, subject_participant_id, subject_text, status
                    FROM semantic_memories
                    WHERE umo = ? AND id IN ({placeholders})
                      AND status IN ('ACTIVE', 'CONFLICTED', 'QUARANTINED')
                    """,
                    (umo, *[int(item) for item in target_claim_ids]),
                ).fetchall()
                if len(targets) != len(set(target_claim_ids)):
                    raise ValueError("semantic revision target is not active in scope")
                for target in targets:
                    if participant_id is not None and target["subject_participant_id"] != participant_id:
                        raise ValueError("semantic revision crosses participant subjects")
                    if participant_id is None and normalize_alias(target["subject_text"]) != normalize_alias(person_cue):
                        raise ValueError("semantic revision crosses unresolved subjects")

            existing = self._connection.execute(
                """
                SELECT id, status, confidence FROM semantic_memories
                WHERE umo = ? AND stable_key = ?
                ORDER BY id DESC LIMIT 1
                """,
                (umo, stable_key),
            ).fetchone()
            if existing is not None and str(existing["status"]) in {
                "ACTIVE",
                "CONFLICTED",
                "QUARANTINED",
            }:
                prior_speakers = self._connection.execute(
                    """
                    SELECT DISTINCT m.sender_participant_id
                    FROM semantic_memory_sources AS ss
                    JOIN messages AS m ON m.id = ss.message_id
                    WHERE ss.semantic_memory_id = ? AND m.umo = ?
                      AND m.is_deleted = 0 AND m.role = 'USER'
                      AND m.sender_participant_id IS NOT NULL
                    """,
                    (int(existing["id"]), umo),
                ).fetchall()
                independent_user_speakers.update(
                    int(row["sender_participant_id"])
                    for row in prior_speakers
                )

            initial_status = "ACTIVE"
            if not subject_participant_key:
                initial_status = "UNRESOLVED"
            elif epistemic_status in {"UNCERTAIN", "HEARSAY", "JOKE"}:
                initial_status = "QUARANTINED"
            high_risk_text = " ".join(
                (str(claim_type), str(aspect), str(content))
            ).casefold()
            high_risk_markers = (
                "管理员", "群主", "权限", "真实身份", "admin", "owner",
            )
            if (
                subject_participant_key
                and (
                    str(claim_type).upper() == "IDENTITY"
                    or any(marker in high_risk_text for marker in high_risk_markers)
                )
                and len(independent_user_speakers) < 2
            ):
                initial_status = "QUARANTINED"

            if existing is None:
                cursor = self._connection.execute(
                    """
                    INSERT INTO semantic_memories(
                        umo, person_cue, aspect_tag, content,
                        source_message_id, confidence, extractor_version,
                        stable_key, subject_participant_id, subject_text,
                        claim_type, epistemic_status, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        umo,
                        person_cue,
                        aspect.strip(),
                        content,
                        source_rows[0][0],
                        max(0.0, min(1.0, float(confidence))),
                        extractor_version,
                        stable_key,
                        participant_id,
                        person_cue,
                        claim_type,
                        epistemic_status,
                        initial_status,
                    ),
                )
                semantic_id = int(cursor.lastrowid)
            else:
                semantic_id = int(existing["id"])
                previous_status = str(existing["status"])
                self._connection.execute(
                    """
                    UPDATE semantic_memories
                    SET person_cue=?, aspect_tag=?, content=?,
                        source_message_id=?, confidence=MAX(confidence, ?),
                        extractor_version=?, subject_participant_id=?,
                        subject_text=?, claim_type=?, epistemic_status=?,
                        status=?, updated_at=CURRENT_TIMESTAMP
                    WHERE id = ? AND umo = ?
                    """,
                    (
                        person_cue,
                        aspect.strip(),
                        content,
                        source_rows[0][0],
                        max(0.0, min(1.0, float(confidence))),
                        extractor_version,
                        participant_id,
                        person_cue,
                        claim_type,
                        epistemic_status,
                        initial_status,
                        semantic_id,
                        umo,
                    ),
                )
                if previous_status != initial_status:
                    self._connection.execute(
                        """
                        INSERT INTO semantic_memory_revisions(
                            semantic_memory_id, previous_status, new_status,
                            reason, source_message_id
                        ) VALUES (?, ?, ?, 'new supporting extraction', ?)
                        """,
                        (
                            semantic_id,
                            previous_status,
                            initial_status,
                            source_rows[0][0],
                        ),
                    )

            for source_id, role, span, evidence_confidence in source_rows:
                self._connection.execute(
                    """
                    INSERT INTO semantic_memory_sources(
                        semantic_memory_id, message_id, evidence_role,
                        source_span, confidence
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(semantic_memory_id, message_id, evidence_role)
                    DO UPDATE SET source_span=excluded.source_span,
                                  confidence=MAX(
                                      semantic_memory_sources.confidence,
                                      excluded.confidence
                                  )
                    """,
                    (semantic_id, source_id, role, span, evidence_confidence),
                )

            operation = str(operation or "ASSERT").upper()
            target_status = "SUPERSEDED" if operation == "SUPERSEDE" else "RETRACTED"
            if operation in {"SUPERSEDE", "RETRACT"}:
                for target in targets:
                    target_id = int(target["id"])
                    self._connection.execute(
                        """
                        UPDATE semantic_memories
                        SET status=?, superseded_by=?, updated_at=CURRENT_TIMESTAMP
                        WHERE id = ? AND umo = ?
                        """,
                        (
                            target_status,
                            semantic_id if operation == "SUPERSEDE" else None,
                            target_id,
                            umo,
                        ),
                    )
                    self._connection.execute(
                        """
                        INSERT INTO semantic_memory_revisions(
                            semantic_memory_id, previous_status, new_status,
                            reason, source_message_id
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            target_id,
                            str(target["status"]),
                            target_status,
                            f"{operation.lower()} by claim {semantic_id}",
                            source_rows[0][0],
                        ),
                    )
            elif initial_status == "ACTIVE":
                if participant_id is not None:
                    conflicting = self._connection.execute(
                        """
                        SELECT id, status FROM semantic_memories
                        WHERE umo = ? AND subject_participant_id = ?
                          AND lower(aspect_tag) = lower(?)
                          AND id <> ? AND stable_key <> ?
                          AND status = 'ACTIVE'
                        """,
                        (umo, participant_id, aspect, semantic_id, stable_key),
                    ).fetchall()
                else:
                    conflicting = []
                if any(role == "CONTRADICT" for _, role, _, _ in source_rows):
                    conflicting = [*conflicting]
                if conflicting:
                    conflict_ids = [semantic_id, *[int(row["id"]) for row in conflicting]]
                    placeholders = ",".join("?" for _ in conflict_ids)
                    self._connection.execute(
                        f"""
                        UPDATE semantic_memories SET status='CONFLICTED',
                            updated_at=CURRENT_TIMESTAMP
                        WHERE umo = ? AND id IN ({placeholders})
                        """,
                        (umo, *conflict_ids),
                    )
            return semantic_id

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
            row = self._connection.execute(
                "SELECT id FROM topics WHERE umo = ? AND name = ?",
                (umo, name.strip()),
            ).fetchone()
            if row is None:
                cursor = self._connection.execute(
                    """
                    INSERT INTO topics(umo, name, summary, extractor_version)
                    VALUES (?, ?, ?, ?)
                    """,
                    (umo, name.strip(), summary, extractor_version),
                )
                topic_id = int(cursor.lastrowid)
            else:
                topic_id = int(row["id"])
                previous = self._connection.execute(
                    "SELECT summary FROM topics WHERE id = ?",
                    (topic_id,),
                ).fetchone()
                if previous is not None and str(previous["summary"]) != summary:
                    self._connection.execute(
                        """
                        INSERT INTO topic_revisions(
                            topic_id, previous_summary, proposed_summary,
                            extractor_version
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (
                            topic_id,
                            str(previous["summary"]),
                            summary,
                            extractor_version,
                        ),
                    )
            self._connection.executemany(
                """
                INSERT OR IGNORE INTO topic_episodes(topic_id, episode_id)
                SELECT ?, id FROM episodes WHERE id = ? AND umo = ?
                """,
                [(topic_id, int(event_id), umo) for event_id in event_ids],
            )
            episode_rows = self._connection.execute(
                """
                SELECT e.summary
                FROM topic_episodes AS te
                JOIN episodes AS e ON e.id = te.episode_id
                WHERE te.topic_id = ? AND e.umo = ? AND e.status = 'READY'
                ORDER BY e.ended_at DESC, e.id DESC LIMIT 20
                """,
                (topic_id, umo),
            ).fetchall()
            summaries = list(
                dict.fromkeys(
                    str(item["summary"]).strip()
                    for item in reversed(episode_rows)
                    if str(item["summary"]).strip()
                )
            )
            aggregate = "；".join(summaries)
            if len(aggregate) > 4000:
                aggregate = aggregate[-4000:]
            self._connection.execute(
                """
                UPDATE topics SET summary = ?, extractor_version = ?
                WHERE id = ? AND umo = ?
                """,
                (aggregate or summary, extractor_version, topic_id, umo),
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
        messages = [self._stored_message_from_row(row) for row in rows]
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
                  AND e.status = 'READY'
                  AND lower(k.cue) = lower(?)
                  AND lower(k.tag) = lower(?)
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
                WHERE umo = ? AND id = ? AND status = 'READY'
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
                WHERE e.umo = ? AND e.id = ? AND e.status = 'READY'
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
                SELECT m.*
                FROM episode_messages AS em
                JOIN episodes AS e ON e.id = em.episode_id
                JOIN messages AS m ON m.id = em.message_id
                WHERE e.umo = ?
                  AND e.id = ?
                  AND e.status = 'READY'
                  AND m.umo = e.umo
                  AND m.is_deleted = 0
                  {cutoff_sql}
                ORDER BY em.position, m.sent_at, m.id
                LIMIT ?
                """,
                parameters,
            ).fetchall()
            messages = [self._stored_message_from_row(row) for row in rows]
        return [
            {
                "source_key": message.source_key,
                "sent_at": message.sent_at,
                "sender_id": message.sender_id,
                "sender_name": message.sender_name,
                "sender_participant_key": message.sender_participant_key,
                "role": message.role,
                "plain_text": message.plain_text,
                "reply_to_source_key": message.reply_to_source_key,
                "mentions": list(message.mentions),
                "components": message.content,
                "revision_no": message.revision_no,
            }
            for message in messages
        ]

    def query_personal_information(
        self,
        *,
        umo: str,
        person: str,
        before_sent_at: int | None = None,
    ) -> list[dict[str, object]]:
        """Paper mapping phi_person->semantic-aspects."""
        with self._lock:
            resolved = self.resolve_participants(umo=umo, reference=person)
            participants = resolved["participants"]
            assert isinstance(participants, list)
            if resolved["ambiguous"]:
                return [
                    {
                        "identity_ambiguous": True,
                        "reference": str(resolved["reference"]),
                        "candidate_participants": [
                            {
                                "canonical_key": item["canonical_key"],
                                "account_id": item["account_id"],
                                "current_display_name": item[
                                    "current_display_name"
                                ],
                                "aliases": item["aliases"],
                            }
                            for item in participants
                        ],
                        "notice": (
                            "昵称对应多个账户；必须改用 account_id 或 "
                            "canonical_key 后才能读取人物记忆。"
                        ),
                    }
                ]
            participant_ids = [int(item["id"]) for item in participants]
            parameters: list[object] = [umo]
            if participant_ids:
                placeholders = ",".join("?" for _ in participant_ids)
                identity_sql = f"s.subject_participant_id IN ({placeholders})"
                parameters.extend(participant_ids)
            else:
                identity_sql = (
                    "s.subject_participant_id IS NULL "
                    "AND lower(s.person_cue) = lower(?)"
                )
                parameters.append(person.strip())
            cutoff_sql = ""
            if before_sent_at is not None:
                cutoff_sql = (
                    " AND EXISTS (SELECT 1 "
                    "FROM semantic_memory_sources AS ss "
                    "JOIN messages AS m ON m.id = ss.message_id "
                    "WHERE ss.semantic_memory_id = s.id AND m.umo = s.umo "
                    "AND m.sent_at < ? AND m.is_deleted = 0)"
                )
                parameters.append(int(before_sent_at))
            rows = self._connection.execute(
                f"""
                SELECT s.subject_participant_id, p.canonical_key,
                       COALESCE(p.current_display_name, s.subject_text,
                                s.person_cue) AS subject_display_name,
                       s.aspect_tag,
                       COUNT(DISTINCT ss.message_id) AS evidence_count,
                       GROUP_CONCAT(DISTINCT s.status) AS statuses
                FROM semantic_memories AS s
                LEFT JOIN participants AS p ON p.id = s.subject_participant_id
                LEFT JOIN semantic_memory_sources AS ss
                  ON ss.semantic_memory_id = s.id
                WHERE s.umo = ? AND ({identity_sql})
                  AND s.status IN ('ACTIVE', 'CONFLICTED')
                {cutoff_sql}
                GROUP BY s.subject_participant_id, p.canonical_key,
                         subject_display_name, s.aspect_tag
                ORDER BY evidence_count DESC, s.aspect_tag
                """,
                parameters,
            ).fetchall()
        return [
            {
                **dict(row),
                "identity_ambiguous": bool(resolved["ambiguous"]),
            }
            for row in rows
        ]

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
        with self._lock:
            resolved = self.resolve_participants(umo=umo, reference=person)
            participants = resolved["participants"]
            assert isinstance(participants, list)
            if resolved["ambiguous"]:
                return [
                    {
                        "identity_ambiguous": True,
                        "reference": str(resolved["reference"]),
                        "candidate_participants": [
                            {
                                "canonical_key": item["canonical_key"],
                                "account_id": item["account_id"],
                                "current_display_name": item[
                                    "current_display_name"
                                ],
                                "aliases": item["aliases"],
                            }
                            for item in participants
                        ],
                        "notice": (
                            "昵称对应多个账户；必须改用 account_id 或 "
                            "canonical_key 后才能读取人物记忆。"
                        ),
                    }
                ]
            participant_ids = [int(item["id"]) for item in participants]
            parameters: list[object] = [umo]
            if participant_ids:
                placeholders = ",".join("?" for _ in participant_ids)
                identity_sql = f"s.subject_participant_id IN ({placeholders})"
                parameters.extend(participant_ids)
            else:
                identity_sql = (
                    "s.subject_participant_id IS NULL "
                    "AND lower(s.person_cue) = lower(?)"
                )
                parameters.append(person.strip())
            parameters.append(aspect.strip())
            cutoff_sql = ""
            if before_sent_at is not None:
                cutoff_sql = (
                    " AND EXISTS (SELECT 1 "
                    "FROM semantic_memory_sources AS sx "
                    "JOIN messages AS mx ON mx.id = sx.message_id "
                    "WHERE sx.semantic_memory_id = s.id AND mx.umo = s.umo "
                    "AND mx.sent_at < ? AND mx.is_deleted = 0)"
                )
                parameters.append(int(before_sent_at))
            parameters.append(safe_limit)
            rows = self._connection.execute(
                f"""
                SELECT s.id, s.person_cue, p.canonical_key,
                       s.aspect_tag, s.content, s.claim_type,
                       s.epistemic_status, s.status, s.confidence,
                       COUNT(DISTINCT ss.message_id) AS evidence_count,
                       GROUP_CONCAT(DISTINCT m.source_key) AS source_keys_csv
                FROM semantic_memories AS s
                LEFT JOIN participants AS p ON p.id = s.subject_participant_id
                LEFT JOIN semantic_memory_sources AS ss
                  ON ss.semantic_memory_id = s.id
                LEFT JOIN messages AS m
                  ON m.id = ss.message_id AND m.is_deleted = 0
                WHERE s.umo = ? AND ({identity_sql})
                  AND lower(s.aspect_tag) = lower(?)
                  AND s.status IN ('ACTIVE', 'CONFLICTED')
                  {cutoff_sql}
                GROUP BY s.id
                ORDER BY s.confidence DESC, s.id DESC
                LIMIT ?
                """,
                parameters,
            ).fetchall()
        return [
            {
                **{key: value for key, value in dict(row).items() if key != "source_keys_csv"},
                "source_keys": [
                    item
                    for item in str(row["source_keys_csv"] or "").split(",")
                    if item
                ],
                "identity_ambiguous": bool(resolved["ambiguous"]),
            }
            for row in rows
        ]

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
                  AND e.status = 'READY'
                  AND instr(lower(t.name), lower(?)) > 0
                  {cutoff_sql}
                ORDER BY e.ended_at DESC, e.id DESC
                LIMIT ?
                """,
                parameters,
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _bounded_json(value: object, *, max_chars: int = 12000) -> str:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        if len(encoded) > max_chars:
            raise ValueError(f"structured trace payload exceeds {max_chars} characters")
        return encoded

    def _assert_scope(self, umo: str) -> None:
        identity = self.get_scope_identity()
        if identity is not None and identity["umo"] != umo:
            raise ValueError("operation crosses the database group boundary")

    def start_interaction_trace(
        self,
        *,
        trace_id: str,
        umo: str,
        sender_id: str,
        request_source_key: str,
        request_sent_at: int,
        query: str,
        trace_ttl_seconds: int = 86400,
    ) -> str:
        """Open a bounded, externally inspectable working graph."""

        self._assert_scope(umo)
        normalized_id = trace_id.strip()
        if not normalized_id or len(normalized_id) > 160:
            raise ValueError("invalid trace_id")
        sent_at = int(request_sent_at)
        if sent_at <= 0:
            raise ValueError("request_sent_at must be positive")
        excerpt = str(query or "").strip()[:2000]
        digest = hashlib.sha256(str(query or "").encode("utf-8")).hexdigest()
        expires_at = sent_at + max(300, min(604800, int(trace_ttl_seconds)))
        with self._lock, self._connection:
            if request_source_key:
                source = self._connection.execute(
                    "SELECT umo FROM messages WHERE source_key = ?",
                    (request_source_key,),
                ).fetchone()
                if source is not None and str(source["umo"]) != umo:
                    raise ValueError("request source belongs to another group")
            self._connection.execute(
                """
                INSERT INTO interaction_traces(
                    trace_id, umo, sender_id, request_source_key,
                    request_sent_at, request_sha256, request_excerpt, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized_id,
                    umo,
                    sender_id.strip(),
                    request_source_key.strip(),
                    sent_at,
                    digest,
                    excerpt,
                    expires_at,
                ),
            )
            self._connection.execute(
                """
                INSERT INTO trace_nodes(
                    trace_id, umo, node_key, node_type, content_json,
                    activation, expires_at
                ) VALUES (?, ?, 'request', 'request', ?, 1, ?)
                """,
                (
                    normalized_id,
                    umo,
                    self._bounded_json(
                        {
                            "source_key": request_source_key.strip(),
                            "sender_id": sender_id.strip(),
                            "sent_at": sent_at,
                            "excerpt": excerpt,
                            "sha256": digest,
                        }
                    ),
                    expires_at,
                ),
            )
        return normalized_id

    def record_trace_node(
        self,
        *,
        trace_id: str,
        umo: str,
        node_key: str,
        node_type: str,
        content: dict[str, object] | None = None,
        activation: float = 0.0,
        utility: float = 0.0,
        expires_at: int | None = None,
    ) -> int:
        self._assert_scope(umo)
        key = node_key.strip()
        kind = node_type.strip().lower()
        if not key or len(key) > 200 or not kind or len(kind) > 80:
            raise ValueError("invalid trace node identity")
        with self._lock, self._connection:
            trace = self._connection.execute(
                "SELECT umo, expires_at FROM interaction_traces WHERE trace_id = ?",
                (trace_id,),
            ).fetchone()
            if trace is None or str(trace["umo"]) != umo:
                raise ValueError("trace does not belong to this group")
            self._connection.execute(
                """
                INSERT INTO trace_nodes(
                    trace_id, umo, node_key, node_type, content_json,
                    activation, utility, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trace_id, node_key) DO UPDATE SET
                    node_type=excluded.node_type,
                    content_json=excluded.content_json,
                    activation=excluded.activation,
                    utility=excluded.utility
                """,
                (
                    trace_id,
                    umo,
                    key,
                    kind,
                    self._bounded_json(content or {}, max_chars=8000),
                    max(0.0, min(1.0, float(activation))),
                    max(-4.0, min(4.0, float(utility))),
                    int(expires_at) if expires_at is not None else int(trace["expires_at"]),
                ),
            )
            row = self._connection.execute(
                "SELECT id FROM trace_nodes WHERE trace_id = ? AND node_key = ?",
                (trace_id, key),
            ).fetchone()
        return int(row["id"])

    def record_trace_edge(
        self,
        *,
        trace_id: str,
        umo: str,
        source_key: str,
        target_key: str,
        relation: str,
        contribution: float = 0.0,
        eligibility: float = 0.0,
    ) -> int:
        self._assert_scope(umo)
        normalized_relation = relation.strip().upper()
        if not normalized_relation or len(normalized_relation) > 80:
            raise ValueError("invalid trace relation")
        with self._lock, self._connection:
            rows = self._connection.execute(
                """
                SELECT id, node_key, umo FROM trace_nodes
                WHERE trace_id = ? AND node_key IN (?, ?)
                """,
                (trace_id, source_key, target_key),
            ).fetchall()
            node_ids = {
                str(row["node_key"]): int(row["id"])
                for row in rows
                if str(row["umo"]) == umo
            }
            if source_key not in node_ids or target_key not in node_ids:
                raise ValueError("trace edge references an unknown node")
            self._connection.execute(
                """
                INSERT INTO trace_edges(
                    trace_id, umo, source_node_id, target_node_id, relation,
                    contribution, eligibility
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trace_id, source_node_id, target_node_id, relation)
                DO UPDATE SET contribution=excluded.contribution,
                              eligibility=excluded.eligibility
                """,
                (
                    trace_id,
                    umo,
                    node_ids[source_key],
                    node_ids[target_key],
                    normalized_relation,
                    max(-1.0, min(1.0, float(contribution))),
                    max(0.0, min(1.0, float(eligibility))),
                ),
            )
            row = self._connection.execute(
                """
                SELECT id FROM trace_edges
                WHERE trace_id = ? AND source_node_id = ?
                  AND target_node_id = ? AND relation = ?
                """,
                (
                    trace_id,
                    node_ids[source_key],
                    node_ids[target_key],
                    normalized_relation,
                ),
            ).fetchone()
        return int(row["id"])

    def finish_interaction_trace(
        self,
        *,
        trace_id: str,
        umo: str,
        response_text: str,
        response_at: int | None = None,
    ) -> None:
        self._assert_scope(umo)
        raw_response = str(response_text or "")
        excerpt = raw_response.strip()[:3000]
        digest = hashlib.sha256(raw_response.encode("utf-8")).hexdigest()
        finished_at = int(response_at or time.time())
        with self._lock, self._connection:
            trace = self._connection.execute(
                "SELECT umo, request_sent_at, expires_at FROM interaction_traces WHERE trace_id = ?",
                (trace_id,),
            ).fetchone()
            if trace is None or str(trace["umo"]) != umo:
                raise ValueError("trace does not belong to this group")
            finished_at = max(finished_at, int(trace["request_sent_at"]))
            self._connection.execute(
                """
                UPDATE interaction_traces
                SET response_sha256 = ?, response_excerpt = ?, response_at = ?,
                    status = 'RESPONDED', updated_at = CURRENT_TIMESTAMP
                WHERE trace_id = ? AND umo = ?
                """,
                (digest, excerpt, finished_at, trace_id, umo),
            )
            self._connection.execute(
                """
                INSERT INTO trace_nodes(
                    trace_id, umo, node_key, node_type, content_json,
                    activation, expires_at
                ) VALUES (?, ?, 'response', 'response', ?, 1, ?)
                ON CONFLICT(trace_id, node_key) DO UPDATE SET
                    content_json=excluded.content_json, activation=1
                """,
                (
                    trace_id,
                    umo,
                    self._bounded_json(
                        {
                            "sent_at": finished_at,
                            "excerpt": excerpt,
                            "sha256": digest,
                        }
                    ),
                    int(trace["expires_at"]),
                ),
            )
            request_node = self._connection.execute(
                "SELECT id FROM trace_nodes WHERE trace_id = ? AND node_key = 'request'",
                (trace_id,),
            ).fetchone()
            response_node = self._connection.execute(
                "SELECT id FROM trace_nodes WHERE trace_id = ? AND node_key = 'response'",
                (trace_id,),
            ).fetchone()
            self._connection.execute(
                """
                INSERT OR IGNORE INTO trace_edges(
                    trace_id, umo, source_node_id, target_node_id, relation,
                    contribution, eligibility
                ) VALUES (?, ?, ?, ?, 'PRODUCES', 1, 1)
                """,
                (trace_id, umo, int(request_node["id"]), int(response_node["id"])),
            )
            activated = self._connection.execute(
                """
                SELECT a.hypothesis_id, a.activation_score, n.id AS node_id
                FROM hypothesis_activations AS a
                JOIN trace_nodes AS n
                  ON n.trace_id = a.trace_id
                 AND n.node_key = 'hypothesis:' || a.hypothesis_id
                WHERE a.trace_id = ? AND n.umo = ?
                """,
                (trace_id, umo),
            ).fetchall()
            for row in activated:
                score = max(0.0, min(1.0, float(row["activation_score"])))
                self._connection.execute(
                    """
                    INSERT OR IGNORE INTO trace_edges(
                        trace_id, umo, source_node_id, target_node_id, relation,
                        contribution, eligibility
                    ) VALUES (?, ?, ?, ?, 'INFLUENCES', ?, ?)
                    """,
                    (
                        trace_id,
                        umo,
                        int(row["node_id"]),
                        int(response_node["id"]),
                        score,
                        score,
                    ),
                )

    def enqueue_feedback_candidate(
        self,
        *,
        umo: str,
        feedback_source_key: str,
        feedback_window_seconds: int = 21600,
        candidate_limit: int = 3,
    ) -> int | None:
        """Queue later text against recent responses; the LLM decides semantics."""

        self._assert_scope(umo)
        with self._lock, self._connection:
            feedback = self._connection.execute(
                """
                SELECT id, source_key, sender_id, sent_at, plain_text
                FROM messages
                WHERE umo = ? AND source_key = ? AND is_deleted = 0
                """,
                (umo, feedback_source_key),
            ).fetchone()
            if feedback is None or not str(feedback["plain_text"]).strip():
                return None
            sent_at = int(feedback["sent_at"])
            lower = sent_at - max(60, min(604800, int(feedback_window_seconds)))
            traces = self._connection.execute(
                """
                SELECT trace_id, sender_id, response_at
                FROM interaction_traces
                WHERE umo = ? AND status IN ('RESPONDED', 'FEEDBACK')
                  AND response_at IS NOT NULL
                  AND response_at < ? AND response_at >= ?
                  AND request_source_key <> ?
                ORDER BY (sender_id = ?) DESC, response_at DESC
                LIMIT ?
                """,
                (
                    umo,
                    sent_at,
                    lower,
                    feedback_source_key,
                    str(feedback["sender_id"]),
                    max(1, min(8, int(candidate_limit))),
                ),
            ).fetchall()
            trace_ids = [str(row["trace_id"]) for row in traces]
            if not trace_ids:
                return None
            reply_to_bot = self._connection.execute(
                """
                SELECT 1
                FROM message_relations AS r
                LEFT JOIN messages AS target ON target.id = r.target_message_id
                LEFT JOIN participants AS p ON p.id = r.target_participant_id
                WHERE r.source_message_id = ?
                  AND r.relation = 'REPLY_TO'
                  AND (target.role = 'BOT' OR p.account_type = 'BOT')
                LIMIT 1
                """,
                (int(feedback["id"]),),
            ).fetchone() is not None
            closest = traces[0]
            score, reasons = feedback_surface_score(
                str(feedback["plain_text"]),
                reply_to_bot=reply_to_bot,
                seconds_after_response=(
                    sent_at - int(closest["response_at"] or sent_at)
                ),
                same_sender=(
                    str(feedback["sender_id"]) == str(closest["sender_id"])
                ),
            )
            if score < 0.25:
                return None
            self._connection.execute(
                """
                INSERT INTO feedback_proposals(
                    umo, feedback_source_key, feedback_sent_at,
                    candidate_trace_ids_json, surface_score, candidate_reason
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(umo, feedback_source_key) DO NOTHING
                """,
                (
                    umo,
                    feedback_source_key,
                    sent_at,
                    self._bounded_json(trace_ids, max_chars=2000),
                    score,
                    ",".join(reasons),
                ),
            )
            row = self._connection.execute(
                """
                SELECT id FROM feedback_proposals
                WHERE umo = ? AND feedback_source_key = ?
                """,
                (umo, feedback_source_key),
            ).fetchone()
        return int(row["id"])

    def pending_feedback_proposals(
        self, *, umo: str, limit: int = 3
    ) -> list[dict[str, object]]:
        self._assert_scope(umo)
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT id, feedback_source_key, feedback_sent_at,
                       candidate_trace_ids_json, surface_score,
                       candidate_reason, created_at
                FROM feedback_proposals
                WHERE umo = ? AND status = 'PENDING'
                ORDER BY feedback_sent_at, id
                LIMIT ?
                """,
                (umo, max(1, min(20, int(limit)))),
            ).fetchall()
        return [
            {
                **dict(row),
                "candidate_trace_ids": json.loads(
                    str(row["candidate_trace_ids_json"])
                ),
            }
            for row in rows
        ]

    @staticmethod
    def _component_evidence(content_json: str) -> dict[str, object]:
        try:
            content: Any = json.loads(content_json)
        except (TypeError, json.JSONDecodeError):
            content = []
        types: list[str] = []
        reply_ids: list[str] = []

        def walk(value: Any) -> None:
            if isinstance(value, dict):
                kind = value.get("type") or value.get("__class__")
                if kind:
                    types.append(str(kind)[:80])
                for key, item in value.items():
                    lowered = str(key).casefold()
                    if lowered in {"reply_id", "message_id", "id"} and (
                        "reply" in str(kind).casefold() or "reply" in lowered
                    ):
                        reply_ids.append(str(item)[:160])
                    elif isinstance(item, (dict, list)):
                        walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        walk(content)
        return {
            "component_types": list(dict.fromkeys(types))[:20],
            "reply_ids": list(dict.fromkeys(reply_ids))[:8],
        }

    def inspect_feedback_proposal(
        self, *, umo: str, proposal_id: int, context_limit: int = 16
    ) -> dict[str, object]:
        """Return bounded evidence for the private agent, with no media URLs/blobs."""

        self._assert_scope(umo)
        with self._lock:
            proposal = self._connection.execute(
                "SELECT * FROM feedback_proposals WHERE id = ? AND umo = ?",
                (int(proposal_id), umo),
            ).fetchone()
            if proposal is None:
                raise ValueError("unknown feedback proposal")
            feedback = self._connection.execute(
                """
                SELECT source_key, sender_id, sender_name, sent_at, plain_text,
                       content_json
                FROM messages
                WHERE umo = ? AND source_key = ? AND is_deleted = 0
                """,
                (umo, str(proposal["feedback_source_key"])),
            ).fetchone()
            if feedback is None:
                raise ValueError("feedback evidence no longer exists")
            trace_ids = json.loads(str(proposal["candidate_trace_ids_json"]))
            observable_nodes: list[sqlite3.Row] = []
            if not trace_ids:
                traces: list[sqlite3.Row] = []
            else:
                placeholders = ",".join("?" for _ in trace_ids)
                traces = self._connection.execute(
                    f"""
                    SELECT trace_id, sender_id, request_source_key,
                           request_sent_at, request_excerpt, response_at,
                           response_excerpt, status
                    FROM interaction_traces
                    WHERE umo = ? AND trace_id IN ({placeholders})
                    ORDER BY response_at DESC
                    """,
                    (umo, *trace_ids),
                ).fetchall()
                observable_nodes = self._connection.execute(
                    f"""
                    SELECT trace_id, node_key, node_type, content_json
                    FROM trace_nodes
                    WHERE umo = ? AND trace_id IN ({placeholders})
                      AND node_type IN ('tool_call', 'tool_result', 'artifact')
                    ORDER BY trace_id, id
                    LIMIT 64
                    """,
                    (umo, *trace_ids),
                ).fetchall()
            earliest = min(
                [int(row["request_sent_at"]) for row in traces]
                or [int(feedback["sent_at"]) - 300]
            )
            context = self._connection.execute(
                """
                SELECT source_key, sender_id, sender_name, sent_at, plain_text, role
                FROM messages
                WHERE umo = ? AND is_deleted = 0
                  AND sent_at >= ? AND sent_at <= ?
                ORDER BY sent_at, id
                LIMIT ?
                """,
                (
                    umo,
                    earliest,
                    int(feedback["sent_at"]),
                    max(2, min(40, int(context_limit))),
                ),
            ).fetchall()
            activation_rows = self._connection.execute(
                f"""
                SELECT a.trace_id, h.id AS hypothesis_id, h.aspect,
                       h.prospective_cue, a.activation_score, a.contribution
                FROM hypothesis_activations AS a
                JOIN feedback_hypotheses AS h ON h.id = a.hypothesis_id
                WHERE h.umo = ?
                  AND a.trace_id IN ({','.join('?' for _ in trace_ids) if trace_ids else "''"})
                ORDER BY a.trace_id, a.activation_score DESC
                """,
                (umo, *trace_ids),
            ).fetchall()
        feedback_value = {
            key: feedback[key]
            for key in ("source_key", "sender_id", "sender_name", "sent_at", "plain_text")
        }
        feedback_value.update(self._component_evidence(str(feedback["content_json"])))
        return {
            "proposal_id": int(proposal["id"]),
            "status": str(proposal["status"]),
            "umo": umo,
            "feedback": feedback_value,
            "candidate_traces": [dict(row) for row in traces],
            "observable_actions": [
                {
                    "trace_id": str(row["trace_id"]),
                    "node_key": str(row["node_key"]),
                    "node_type": str(row["node_type"]),
                    "content": json.loads(str(row["content_json"])),
                }
                for row in observable_nodes
            ],
            "activated_hypotheses": [dict(row) for row in activation_rows],
            "context": [dict(row) for row in context],
        }

    def search_feedback_hypotheses(
        self,
        *,
        umo: str,
        sender_id: str,
        query: str,
        at: int | None = None,
        limit: int = 10,
        include_inactive: bool = False,
    ) -> list[dict[str, object]]:
        self._assert_scope(umo)
        cutoff = int(at or time.time())
        status_sql = "" if include_inactive else " AND status = 'ACTIVE'"
        expiry_sql = (
            "" if include_inactive else " AND (expires_at IS NULL OR expires_at > ?)"
        )
        parameters: tuple[object, ...] = (
            (umo, cutoff, sender_id)
            if include_inactive
            else (umo, cutoff, cutoff, sender_id)
        )
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT id, umo, scope_type, scope_key, aspect, statement,
                       prospective_cue, trigger_cues_json, activation_mode,
                       evidence_confidence, utility, support_count,
                       contradict_count, activation_count, learned_at,
                       last_activated_at, status, merged_into
                FROM feedback_hypotheses
                WHERE umo = ? AND learned_at < ?
                  {expiry_sql}
                  AND (scope_type = 'group'
                       OR (scope_type = 'sender' AND scope_key = ?))
                  {status_sql}
                ORDER BY utility DESC, evidence_confidence DESC, id DESC
                LIMIT 200
                """,
                parameters,
            ).fetchall()
        if include_inactive:
            return [
                {
                    **dict(row),
                    "trigger_cues": json.loads(str(row["trigger_cues_json"])),
                }
                for row in rows[: max(1, min(50, int(limit)))]
            ]
        return rank_hypotheses(
            [dict(row) for row in rows],
            sender_id=sender_id,
            query=query,
            limit=limit,
        )

    def feedback_hypothesis_candidates(
        self,
        *,
        umo: str,
        sender_id: str,
        at: int,
        limit: int = 16,
    ) -> list[dict[str, object]]:
        """Bounded active view for semantic selection by the private agent."""

        self._assert_scope(umo)
        cutoff = int(at)
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT id, scope_type, scope_key, aspect, statement,
                       prospective_cue, trigger_cues_json, activation_mode,
                       evidence_confidence, utility, support_count,
                       contradict_count, learned_at
                FROM feedback_hypotheses
                WHERE umo = ? AND status = 'ACTIVE' AND learned_at < ?
                  AND (expires_at IS NULL OR expires_at > ?)
                  AND (scope_type = 'group'
                       OR (scope_type = 'sender' AND scope_key = ?))
                ORDER BY utility DESC, evidence_confidence DESC,
                         COALESCE(last_activated_at, learned_at) DESC, id DESC
                LIMIT ?
                """,
                (
                    umo,
                    cutoff,
                    cutoff,
                    sender_id,
                    max(1, min(50, int(limit))),
                ),
            ).fetchall()
        return [
            {
                **dict(row),
                "trigger_cues": json.loads(str(row["trigger_cues_json"])),
            }
            for row in rows
        ]

    def activate_feedback_hypotheses(
        self,
        *,
        umo: str,
        sender_id: str,
        query: str,
        at: int,
        trace_id: str | None = None,
        limit: int = 6,
        selected: list[dict[str, object]] | None = None,
        activation_method: str = "lexical",
    ) -> list[dict[str, object]]:
        if selected is None:
            ranked = self.search_feedback_hypotheses(
                umo=umo,
                sender_id=sender_id,
                query=query,
                at=at,
                limit=limit,
            )
        else:
            candidates = {
                int(row["id"]): row
                for row in self.feedback_hypothesis_candidates(
                    umo=umo,
                    sender_id=sender_id,
                    at=at,
                    limit=50,
                )
            }
            ranked = []
            for item in selected[: max(1, min(20, int(limit)))]:
                hypothesis_id = int(item.get("id") or 0)
                row = candidates.get(hypothesis_id)
                if row is None:
                    raise ValueError("selected hypothesis is outside the active scope")
                score = max(
                    0.0,
                    min(1.0, float(item.get("activation_score") or 0.0)),
                )
                if score <= 0:
                    continue
                ranked.append({**row, "activation_score": score})
        if not ranked:
            return []
        method = str(activation_method or "lexical").strip().lower()[:40]
        with self._lock, self._connection:
            trace = None
            if trace_id:
                trace = self._connection.execute(
                    "SELECT umo, expires_at FROM interaction_traces WHERE trace_id = ?",
                    (trace_id,),
                ).fetchone()
                if trace is None or str(trace["umo"]) != umo:
                    raise ValueError("activation trace belongs to another group")
            for row in ranked:
                hypothesis_id = int(row["id"])
                score = max(0.0, min(1.0, float(row["activation_score"])))
                existing_activation = None
                if trace_id:
                    existing_activation = self._connection.execute(
                        """
                        SELECT 1 FROM hypothesis_activations
                        WHERE trace_id = ? AND hypothesis_id = ?
                        """,
                        (trace_id, hypothesis_id),
                    ).fetchone()
                self._connection.execute(
                    """
                    UPDATE feedback_hypotheses
                    SET activation_count = activation_count + ?,
                        last_activated_at = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ? AND umo = ?
                    """,
                    (
                        0 if existing_activation is not None else 1,
                        int(at),
                        hypothesis_id,
                        umo,
                    ),
                )
                if not trace_id or trace is None:
                    continue
                self._connection.execute(
                    """
                    INSERT INTO hypothesis_activations(
                        trace_id, hypothesis_id, activation_score, contribution,
                        activation_method
                    ) VALUES (?, ?, ?, 1, ?)
                    ON CONFLICT(trace_id, hypothesis_id) DO UPDATE SET
                        activation_score=max(
                            hypothesis_activations.activation_score,
                            excluded.activation_score
                        ),
                        activation_method=excluded.activation_method
                    """,
                    (trace_id, hypothesis_id, score, method),
                )
                node_key = f"hypothesis:{hypothesis_id}"
                self._connection.execute(
                    """
                    INSERT INTO trace_nodes(
                        trace_id, umo, node_key, node_type, content_json,
                        activation, utility, expires_at
                    ) VALUES (?, ?, ?, 'hypothesis', ?, ?, ?, ?)
                    ON CONFLICT(trace_id, node_key) DO UPDATE SET
                        content_json=excluded.content_json,
                        activation=excluded.activation,
                        utility=excluded.utility
                    """,
                    (
                        trace_id,
                        umo,
                        node_key,
                        self._bounded_json(
                            {
                                "hypothesis_id": hypothesis_id,
                                "aspect": row["aspect"],
                                "prospective_cue": row["prospective_cue"],
                                "activation_mode": row["activation_mode"],
                                "activation_method": method,
                            }
                        ),
                        score,
                        max(-4.0, min(4.0, float(row["utility"]))),
                        int(trace["expires_at"]),
                    ),
                )
                request_node = self._connection.execute(
                    "SELECT id FROM trace_nodes WHERE trace_id = ? AND node_key = 'request'",
                    (trace_id,),
                ).fetchone()
                hypothesis_node = self._connection.execute(
                    "SELECT id FROM trace_nodes WHERE trace_id = ? AND node_key = ?",
                    (trace_id, node_key),
                ).fetchone()
                self._connection.execute(
                    """
                    INSERT OR IGNORE INTO trace_edges(
                        trace_id, umo, source_node_id, target_node_id, relation,
                        contribution, eligibility
                    ) VALUES (?, ?, ?, ?, 'ACTIVATES', ?, ?)
                    """,
                    (
                        trace_id,
                        umo,
                        int(request_node["id"]),
                        int(hypothesis_node["id"]),
                        score,
                        score,
                    ),
                )
        return ranked

    def apply_feedback_decision(
        self,
        *,
        umo: str,
        proposal_id: int,
        decision: FeedbackDecision,
        hypothesis_ttl_seconds: int = 15552000,
        min_commit_score: float = 0.65,
    ) -> dict[str, object]:
        """Atomically validate evidence, assign credit, and mutate active memory."""

        self._assert_scope(umo)
        with self._lock, self._connection:
            proposal = self._connection.execute(
                "SELECT * FROM feedback_proposals WHERE id = ? AND umo = ?",
                (int(proposal_id), umo),
            ).fetchone()
            if proposal is None or str(proposal["status"]) != "PENDING":
                raise ValueError("feedback proposal is not pending")
            feedback = self._connection.execute(
                """
                SELECT sender_id, sent_at, plain_text FROM messages
                WHERE umo = ? AND source_key = ? AND is_deleted = 0
                """,
                (umo, str(proposal["feedback_source_key"])),
            ).fetchone()
            if feedback is None:
                raise ValueError("feedback source evidence is missing")
            if decision.mutation == "ignore":
                self._connection.execute(
                    """
                    UPDATE feedback_proposals
                    SET status = 'IGNORED', decision_json = ?,
                        decided_at = CURRENT_TIMESTAMP
                    WHERE id = ? AND umo = ?
                    """,
                    (
                        self._bounded_json(decision.as_dict()),
                        int(proposal_id),
                        umo,
                    ),
                )
                return {"status": "IGNORED", "proposal_id": int(proposal_id)}

            commit_score = abs(decision.feedback_valence) * decision.confidence
            threshold = max(0.05, min(1.0, float(min_commit_score)))
            if decision.scope_type == "group":
                threshold = max(threshold, 0.8)
            if commit_score < threshold:
                raise ValueError(
                    f"feedback evidence score {commit_score:.3f} is below "
                    f"the commit threshold {threshold:.3f}"
                )
            if decision.activation_mode == "always" and decision.trigger_cues:
                raise ValueError("always hypotheses must not define trigger cues")
            if decision.activation_mode == "semantic" and not decision.trigger_cues:
                raise ValueError("semantic hypotheses require trigger cues")

            candidates = set(json.loads(str(proposal["candidate_trace_ids_json"])))
            if decision.target_trace_id not in candidates:
                raise ValueError("target trace was not an eligible candidate")
            trace = self._connection.execute(
                "SELECT * FROM interaction_traces WHERE trace_id = ? AND umo = ?",
                (decision.target_trace_id, umo),
            ).fetchone()
            if trace is None:
                raise ValueError("target trace belongs to another group")
            response_at = int(trace["response_at"] or trace["request_sent_at"])
            feedback_sent_at = int(feedback["sent_at"])
            if feedback_sent_at <= response_at:
                raise ValueError("feedback must occur after the target response")
            if decision.scope_type == "sender":
                allowed_scope_keys = {
                    str(feedback["sender_id"]),
                    str(trace["sender_id"]),
                }
                if decision.scope_key not in allowed_scope_keys:
                    raise ValueError("sender scope lacks source evidence")
            elif decision.scope_key != umo:
                raise ValueError("group-scoped hypothesis must use the current UMO")

            activation_rows = self._connection.execute(
                """
                SELECT a.hypothesis_id, a.activation_score, a.contribution
                FROM hypothesis_activations AS a
                JOIN feedback_hypotheses AS h ON h.id = a.hypothesis_id
                WHERE a.trace_id = ? AND h.umo = ?
                """,
                (decision.target_trace_id, umo),
            ).fetchall()
            total_credit = 0.0
            for activation in activation_rows:
                delta = backward_credit_delta(
                    feedback_valence=decision.feedback_valence,
                    feedback_confidence=decision.confidence,
                    eligibility=float(activation["activation_score"]),
                    contribution=float(activation["contribution"]),
                )
                total_credit += delta
                self._connection.execute(
                    """
                    UPDATE feedback_hypotheses
                    SET utility = min(4, max(-4, utility + ?)),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ? AND umo = ?
                    """,
                    (delta, int(activation["hypothesis_id"]), umo),
                )
                self._connection.execute(
                    """
                    UPDATE hypothesis_activations
                    SET credit = credit + ?
                    WHERE trace_id = ? AND hypothesis_id = ?
                    """,
                    (
                        delta,
                        decision.target_trace_id,
                        int(activation["hypothesis_id"]),
                    ),
                )

            hypothesis_id: int
            relation: str
            if decision.mutation == "upsert":
                fingerprint = hypothesis_fingerprint(
                    umo=umo,
                    scope_type=decision.scope_type,
                    scope_key=decision.scope_key,
                    aspect=decision.aspect,
                    prospective_cue=decision.prospective_cue,
                    trigger_cues=decision.trigger_cues,
                    activation_mode=decision.activation_mode,
                )
                increment = max(0.05, abs(decision.feedback_valence) * decision.confidence)
                expires_at = feedback_sent_at + max(
                    86400, min(63072000, int(hypothesis_ttl_seconds))
                )
                self._connection.execute(
                    """
                    INSERT INTO feedback_hypotheses(
                        umo, fingerprint, scope_type, scope_key, aspect,
                        statement, prospective_cue, trigger_cues_json,
                        activation_mode,
                        evidence_confidence, utility, support_count, learned_at,
                        last_decay_at, source_trace_id, expires_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
                    ON CONFLICT(umo, fingerprint) DO UPDATE SET
                        statement=excluded.statement,
                        prospective_cue=excluded.prospective_cue,
                        trigger_cues_json=excluded.trigger_cues_json,
                        activation_mode=excluded.activation_mode,
                        evidence_confidence=max(
                            feedback_hypotheses.evidence_confidence,
                            excluded.evidence_confidence
                        ),
                        utility=min(4, feedback_hypotheses.utility + excluded.utility),
                        support_count=feedback_hypotheses.support_count + 1,
                        status='ACTIVE', merged_into=NULL,
                        merge_previous_status='',
                        expires_at=max(feedback_hypotheses.expires_at, excluded.expires_at),
                        updated_at=CURRENT_TIMESTAMP
                    """,
                    (
                        umo,
                        fingerprint,
                        decision.scope_type,
                        decision.scope_key,
                        decision.aspect,
                        decision.statement,
                        decision.prospective_cue,
                        self._bounded_json(list(decision.trigger_cues), max_chars=1200),
                        decision.activation_mode,
                        decision.confidence,
                        increment,
                        feedback_sent_at,
                        feedback_sent_at,
                        decision.target_trace_id,
                        expires_at,
                    ),
                )
                hypothesis = self._connection.execute(
                    "SELECT id FROM feedback_hypotheses WHERE umo = ? AND fingerprint = ?",
                    (umo, fingerprint),
                ).fetchone()
                hypothesis_id = int(hypothesis["id"])
                relation = "SUPPORTS_CORRECTION"
            else:
                hypothesis_id = int(decision.target_hypothesis_id or 0)
                hypothesis = self._connection.execute(
                    """
                    SELECT id, scope_type, scope_key, learned_at, status
                    FROM feedback_hypotheses WHERE id = ? AND umo = ?
                    """,
                    (hypothesis_id, umo),
                ).fetchone()
                if hypothesis is None:
                    raise ValueError("target hypothesis belongs to another group")
                if (
                    str(hypothesis["scope_type"]) != decision.scope_type
                    or str(hypothesis["scope_key"]) != decision.scope_key
                ):
                    raise ValueError("target hypothesis is outside the decision scope")
                if int(hypothesis["learned_at"]) >= feedback_sent_at:
                    raise ValueError("target hypothesis was not available at feedback time")
                if str(hypothesis["status"]) == "MERGED":
                    raise ValueError("target hypothesis is a merged materialized view")
                amount = abs(decision.feedback_valence) * decision.confidence
                if decision.mutation == "reinforce":
                    self._connection.execute(
                        """
                        UPDATE feedback_hypotheses
                        SET utility=min(4, utility + ?),
                            evidence_confidence=max(evidence_confidence, ?),
                            support_count=support_count + 1,
                            status='ACTIVE', updated_at=CURRENT_TIMESTAMP
                        WHERE id = ? AND umo = ?
                        """,
                        (amount, decision.confidence, hypothesis_id, umo),
                    )
                    relation = "SUPPORTS"
                else:
                    self._connection.execute(
                        """
                        UPDATE feedback_hypotheses
                        SET utility=max(-4, utility - ?),
                            contradict_count=contradict_count + 1,
                            status=CASE WHEN utility - ? <= -1 THEN 'DORMANT'
                                        ELSE status END,
                            updated_at=CURRENT_TIMESTAMP
                        WHERE id = ? AND umo = ?
                        """,
                        (amount, amount, hypothesis_id, umo),
                    )
                    relation = "CONTRADICTS"

            source_key = str(proposal["feedback_source_key"])
            self._connection.execute(
                """
                INSERT OR IGNORE INTO hypothesis_evidence(
                    hypothesis_id, feedback_source_key, trace_id, relation,
                    valence, confidence
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    hypothesis_id,
                    source_key,
                    decision.target_trace_id,
                    relation,
                    decision.feedback_valence,
                    decision.confidence,
                ),
            )
            self._connection.execute(
                """
                INSERT OR IGNORE INTO feedback_links(
                    umo, trace_id, feedback_source_key, feedback_sent_at,
                    link_method, link_confidence, feedback_valence
                ) VALUES (?, ?, ?, ?, 'maintenance_agent', ?, ?)
                """,
                (
                    umo,
                    decision.target_trace_id,
                    source_key,
                    feedback_sent_at,
                    decision.confidence,
                    decision.feedback_valence,
                ),
            )
            expires_at = int(trace["expires_at"])
            feedback_node_key = f"feedback:{int(proposal_id)}"
            hypothesis_node_key = f"hypothesis:{hypothesis_id}"
            self._connection.execute(
                """
                INSERT INTO trace_nodes(
                    trace_id, umo, node_key, node_type, content_json,
                    activation, utility, expires_at
                ) VALUES (?, ?, ?, 'feedback', ?, ?, ?, ?)
                ON CONFLICT(trace_id, node_key) DO UPDATE SET
                    content_json=excluded.content_json,
                    activation=excluded.activation,
                    utility=excluded.utility
                """,
                (
                    decision.target_trace_id,
                    umo,
                    feedback_node_key,
                    self._bounded_json(
                        {
                            "source_key": source_key,
                            "sent_at": feedback_sent_at,
                            "excerpt": str(feedback["plain_text"])[:1200],
                            "valence": decision.feedback_valence,
                            "confidence": decision.confidence,
                        }
                    ),
                    decision.confidence,
                    decision.feedback_valence,
                    expires_at,
                ),
            )
            hypothesis_row = self._connection.execute(
                """
                SELECT aspect, prospective_cue, activation_mode, utility
                FROM feedback_hypotheses WHERE id = ?
                """,
                (hypothesis_id,),
            ).fetchone()
            self._connection.execute(
                """
                INSERT INTO trace_nodes(
                    trace_id, umo, node_key, node_type, content_json,
                    activation, utility, expires_at
                ) VALUES (?, ?, ?, 'hypothesis', ?, ?, ?, ?)
                ON CONFLICT(trace_id, node_key) DO UPDATE SET
                    content_json=excluded.content_json,
                    activation=max(trace_nodes.activation, excluded.activation),
                    utility=excluded.utility
                """,
                (
                    decision.target_trace_id,
                    umo,
                    hypothesis_node_key,
                    self._bounded_json(
                        {
                            "hypothesis_id": hypothesis_id,
                            "aspect": hypothesis_row["aspect"],
                            "prospective_cue": hypothesis_row["prospective_cue"],
                            "activation_mode": hypothesis_row["activation_mode"],
                        }
                    ),
                    decision.confidence,
                    float(hypothesis_row["utility"]),
                    expires_at,
                ),
            )
            nodes = self._connection.execute(
                """
                SELECT node_key, id FROM trace_nodes
                WHERE trace_id = ? AND node_key IN ('response', ?, ?)
                """,
                (decision.target_trace_id, feedback_node_key, hypothesis_node_key),
            ).fetchall()
            node_ids = {str(row["node_key"]): int(row["id"]) for row in nodes}
            if "response" in node_ids:
                self._connection.execute(
                    """
                    INSERT OR IGNORE INTO trace_edges(
                        trace_id, umo, source_node_id, target_node_id, relation,
                        contribution, eligibility
                    ) VALUES (?, ?, ?, ?, 'RECEIVES_FEEDBACK', ?, 1)
                    """,
                    (
                        decision.target_trace_id,
                        umo,
                        node_ids["response"],
                        node_ids[feedback_node_key],
                        decision.feedback_valence,
                    ),
                )
            self._connection.execute(
                """
                INSERT OR IGNORE INTO trace_edges(
                    trace_id, umo, source_node_id, target_node_id, relation,
                    contribution, eligibility
                ) VALUES (?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    decision.target_trace_id,
                    umo,
                    node_ids[feedback_node_key],
                    node_ids[hypothesis_node_key],
                    relation,
                    decision.confidence,
                ),
            )
            self._connection.execute(
                """
                UPDATE interaction_traces
                SET status='FEEDBACK', updated_at=CURRENT_TIMESTAMP
                WHERE trace_id = ? AND umo = ?
                """,
                (decision.target_trace_id, umo),
            )
            self._connection.execute(
                """
                UPDATE feedback_proposals
                SET status='COMMITTED', decision_json=?,
                    decided_at=CURRENT_TIMESTAMP
                WHERE id = ? AND umo = ?
                """,
                (
                    self._bounded_json(decision.as_dict()),
                    int(proposal_id),
                    umo,
                ),
            )
        return {
            "status": "COMMITTED",
            "proposal_id": int(proposal_id),
            "trace_id": decision.target_trace_id,
            "hypothesis_id": hypothesis_id,
            "backward_credit": round(total_credit, 6),
        }

    def reject_feedback_proposal(
        self, *, umo: str, proposal_id: int, error: str
    ) -> None:
        self._assert_scope(umo)
        with self._lock, self._connection:
            self._connection.execute(
                """
                UPDATE feedback_proposals
                SET status='REJECTED', error=?, decided_at=CURRENT_TIMESTAMP
                WHERE id = ? AND umo = ? AND status='PENDING'
                """,
                (str(error or "")[:500], int(proposal_id), umo),
            )

    def feedback_proposal_status(
        self, *, umo: str, proposal_id: int
    ) -> dict[str, object] | None:
        self._assert_scope(umo)
        with self._lock:
            row = self._connection.execute(
                """
                SELECT id, status, error, decision_json, decided_at
                FROM feedback_proposals WHERE id = ? AND umo = ?
                """,
                (int(proposal_id), umo),
            ).fetchone()
        if row is None:
            return None
        return {
            **dict(row),
            "decision": json.loads(str(row["decision_json"])),
        }

    def merge_feedback_hypotheses(
        self, *, umo: str, source_id: int, target_id: int
    ) -> None:
        """Reversible materialized-view merge; source evidence stays intact."""

        if int(source_id) == int(target_id):
            raise ValueError("cannot merge a hypothesis into itself")
        self._assert_scope(umo)
        with self._lock, self._connection:
            rows = self._connection.execute(
                """
                SELECT id, status, merged_into FROM feedback_hypotheses
                WHERE umo = ? AND id IN (?, ?)
                """,
                (umo, int(source_id), int(target_id)),
            ).fetchall()
            if {int(row["id"]) for row in rows} != {int(source_id), int(target_id)}:
                raise ValueError("merge crosses a group boundary or unknown hypothesis")
            by_id = {int(row["id"]): row for row in rows}
            source = by_id[int(source_id)]
            target = by_id[int(target_id)]
            if str(source["status"]) == "MERGED" or source["merged_into"] is not None:
                raise ValueError("source hypothesis is already merged")
            if str(target["status"]) != "ACTIVE" or target["merged_into"] is not None:
                raise ValueError("merge target must be an active root hypothesis")
            self._connection.execute(
                """
                UPDATE feedback_hypotheses
                SET merge_previous_status=status, status='MERGED', merged_into=?,
                    updated_at=CURRENT_TIMESTAMP
                WHERE id = ? AND umo = ?
                """,
                (int(target_id), int(source_id), umo),
            )

    def unmerge_feedback_hypothesis(self, *, umo: str, source_id: int) -> None:
        self._assert_scope(umo)
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE feedback_hypotheses
                SET status=CASE
                        WHEN merge_previous_status IN ('ACTIVE', 'DORMANT')
                        THEN merge_previous_status ELSE 'ACTIVE' END,
                    merged_into=NULL, merge_previous_status='',
                    updated_at=CURRENT_TIMESTAMP
                WHERE id = ? AND umo = ? AND status='MERGED'
                """,
                (int(source_id), umo),
            )
        if cursor.rowcount != 1:
            raise ValueError("hypothesis is not a merged member in this group")

    def compact_feedback_memory(
        self,
        *,
        umo: str,
        now: int | None = None,
        max_active_hypotheses: int = 200,
        utility_half_life_days: float = 90.0,
    ) -> dict[str, int]:
        """Decay the active view without deleting append-only evidence."""

        self._assert_scope(umo)
        current = int(now or time.time())
        half_life = max(1.0, min(3650.0, float(utility_half_life_days)))
        decayed = 0
        with self._lock, self._connection:
            rows = self._connection.execute(
                """
                SELECT id, utility, last_decay_at
                FROM feedback_hypotheses
                WHERE umo = ? AND status IN ('ACTIVE', 'DORMANT')
                  AND last_decay_at < ?
                """,
                (umo, current),
            ).fetchall()
            for row in rows:
                elapsed_days = max(0.0, (current - int(row["last_decay_at"])) / 86400)
                factor = math.pow(0.5, elapsed_days / half_life)
                self._connection.execute(
                    """
                    UPDATE feedback_hypotheses
                    SET utility=?, last_decay_at=?, updated_at=CURRENT_TIMESTAMP
                    WHERE id = ? AND umo = ?
                    """,
                    (float(row["utility"]) * factor, current, int(row["id"]), umo),
                )
                decayed += 1
            expired_traces = self._connection.execute(
                """
                UPDATE interaction_traces
                SET status='EXPIRED', updated_at=CURRENT_TIMESTAMP
                WHERE umo = ? AND expires_at <= ?
                  AND status NOT IN ('EXPIRED', 'COMPACTED')
                """,
                (umo, current),
            ).rowcount
            self._connection.execute(
                """
                UPDATE trace_nodes SET status='EXPIRED'
                WHERE umo = ? AND expires_at IS NOT NULL AND expires_at <= ?
                  AND status='ACTIVE'
                """,
                (umo, current),
            )
            expired_hypotheses = self._connection.execute(
                """
                UPDATE feedback_hypotheses
                SET status='DORMANT', updated_at=CURRENT_TIMESTAMP
                WHERE umo = ? AND status='ACTIVE'
                  AND expires_at IS NOT NULL AND expires_at <= ?
                """,
                (umo, current),
            ).rowcount
            keep = max(1, min(5000, int(max_active_hypotheses)))
            active = self._connection.execute(
                """
                SELECT id FROM feedback_hypotheses
                WHERE umo = ? AND status='ACTIVE'
                ORDER BY utility DESC, evidence_confidence DESC,
                         COALESCE(last_activated_at, learned_at) DESC, id DESC
                """,
                (umo,),
            ).fetchall()
            dormant_by_budget = 0
            if len(active) > keep:
                stale_ids = [int(row["id"]) for row in active[keep:]]
                placeholders = ",".join("?" for _ in stale_ids)
                dormant_by_budget = self._connection.execute(
                    f"""
                    UPDATE feedback_hypotheses
                    SET status='DORMANT', updated_at=CURRENT_TIMESTAMP
                    WHERE umo = ? AND id IN ({placeholders})
                    """,
                    (umo, *stale_ids),
                ).rowcount
            expired_proposals = self._connection.execute(
                """
                UPDATE feedback_proposals
                SET status='EXPIRED', error='maintenance deadline elapsed',
                    decided_at=CURRENT_TIMESTAMP
                WHERE umo = ? AND status='PENDING'
                  AND feedback_sent_at < ?
                """,
                (umo, current - 604800),
            ).rowcount
        return {
            "decayed_hypotheses": int(decayed),
            "expired_traces": int(expired_traces),
            "expired_hypotheses": int(expired_hypotheses),
            "dormant_by_budget": int(dormant_by_budget),
            "expired_proposals": int(expired_proposals),
        }

    def interaction_trace_graph(
        self, *, umo: str, trace_id: str
    ) -> dict[str, object] | None:
        self._assert_scope(umo)
        with self._lock:
            trace = self._connection.execute(
                "SELECT * FROM interaction_traces WHERE trace_id = ? AND umo = ?",
                (trace_id, umo),
            ).fetchone()
            if trace is None:
                return None
            nodes = self._connection.execute(
                """
                SELECT id, node_key, node_type, content_json, activation,
                       utility, status, expires_at, created_at
                FROM trace_nodes WHERE trace_id = ? AND umo = ? ORDER BY id
                """,
                (trace_id, umo),
            ).fetchall()
            edges = self._connection.execute(
                """
                SELECT e.id, s.node_key AS source, t.node_key AS target,
                       e.relation, e.contribution, e.eligibility, e.credit
                FROM trace_edges AS e
                JOIN trace_nodes AS s ON s.id = e.source_node_id
                JOIN trace_nodes AS t ON t.id = e.target_node_id
                WHERE e.trace_id = ? AND e.umo = ? ORDER BY e.id
                """,
                (trace_id, umo),
            ).fetchall()
        return {
            "trace": dict(trace),
            "nodes": [
                {**dict(row), "content": json.loads(str(row["content_json"]))}
                for row in nodes
            ],
            "edges": [dict(row) for row in edges],
        }

    @staticmethod
    def _make_fts_query(query: str) -> str:
        return f'"{query.replace(chr(34), chr(34) * 2)}"'

    def _stored_message_from_row(self, row: sqlite3.Row) -> StoredMessage:
        sender_participant_id = (
            int(row["sender_participant_id"])
            if "sender_participant_id" in row.keys()
            and row["sender_participant_id"] is not None
            else None
        )
        sender_key = ""
        if sender_participant_id is not None:
            participant = self._connection.execute(
                "SELECT canonical_key FROM participants WHERE id = ?",
                (sender_participant_id,),
            ).fetchone()
            if participant is not None:
                sender_key = str(participant["canonical_key"])
        reply = self._connection.execute(
            """
            SELECT target_source_key FROM message_relations
            WHERE source_message_id = ?
              AND relation IN ('REPLY_TO', 'RESPONDS_TO')
            ORDER BY id LIMIT 1
            """,
            (int(row["id"]),),
        ).fetchone()
        mentions = self._connection.execute(
            """
            SELECT p.canonical_key, p.account_id,
                   COALESCE(p.current_display_name, '') AS display_name
            FROM message_participants AS mp
            JOIN participants AS p ON p.id = mp.participant_id
            WHERE mp.message_id = ? AND mp.relation = 'MENTIONED'
            ORDER BY mp.position, p.id
            """,
            (int(row["id"]),),
        ).fetchall()
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
            content=self._parse_content_json(row["content_json"]),
            role=str(row["role"]),  # type: ignore[arg-type]
            sender_participant_id=sender_participant_id,
            sender_participant_key=sender_key,
            revision_no=(
                int(row["revision_no"] or 1)
                if "revision_no" in row.keys()
                else 1
            ),
            content_sha256=(
                str(row["content_sha256"] or "")
                if "content_sha256" in row.keys()
                else ""
            ),
            reply_to_source_key=(
                str(reply["target_source_key"]) if reply is not None else ""
            ),
            mentions=tuple(
                {
                    "participant_key": str(item["canonical_key"]),
                    "account_id": str(item["account_id"]),
                    "display_name": str(item["display_name"]),
                }
                for item in mentions
            ),
        )
