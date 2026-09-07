"""Small SQLite access layer for the existing, private, per-group MR database.

Stored prose is evidence for the model, never an identity decision made here.
Old tables remain in place. New databases get only the tables this module uses.
All methods are synchronous; one reentrant lock serializes this connection's
transactions and reads. No model or network calls happen while holding it.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
import time
from functools import wraps
from pathlib import Path
from threading import RLock
from typing import Any, Iterable
from uuid import uuid4


SCHEMA = """
CREATE TABLE IF NOT EXISTS scope_meta (singleton INTEGER PRIMARY KEY CHECK(singleton=1),umo TEXT NOT NULL UNIQUE,
 platform_id TEXT NOT NULL,group_id TEXT NOT NULL,created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS forgotten_accounts (umo TEXT NOT NULL,platform_id TEXT NOT NULL,account_hash TEXT NOT NULL,
 requested_at INTEGER NOT NULL,reason TEXT NOT NULL DEFAULT 'self_service',PRIMARY KEY(umo,platform_id,account_hash));
CREATE TABLE IF NOT EXISTS participants (
 id INTEGER PRIMARY KEY AUTOINCREMENT, umo TEXT NOT NULL, platform_id TEXT NOT NULL,
 account_id TEXT NOT NULL, canonical_key TEXT NOT NULL, account_type TEXT NOT NULL DEFAULT 'USER',
 current_display_name TEXT NOT NULL DEFAULT '', first_seen_at INTEGER NOT NULL,
 last_seen_at INTEGER NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 UNIQUE(umo,platform_id,account_id), UNIQUE(umo,canonical_key));
CREATE TABLE IF NOT EXISTS messages (
 id INTEGER PRIMARY KEY AUTOINCREMENT, source_key TEXT NOT NULL UNIQUE,
 platform TEXT NOT NULL, platform_id TEXT NOT NULL, umo TEXT NOT NULL, group_id TEXT NOT NULL,
 message_id TEXT NOT NULL, sender_id TEXT NOT NULL, sender_name TEXT NOT NULL,
 sent_at INTEGER NOT NULL, plain_text TEXT NOT NULL, content_json TEXT NOT NULL,
 role TEXT NOT NULL CHECK(role IN ('USER','BOT','SYSTEM')), is_deleted INTEGER NOT NULL DEFAULT 0,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 sender_participant_id INTEGER REFERENCES participants(id), content_sha256 TEXT NOT NULL DEFAULT '',
 revision_no INTEGER NOT NULL DEFAULT 1, deleted_at INTEGER);
CREATE INDEX IF NOT EXISTS idx_messages_umo_time ON messages(umo,sent_at,id);
CREATE INDEX IF NOT EXISTS idx_messages_sender_time ON messages(umo,sender_id,sent_at,id);
CREATE TABLE IF NOT EXISTS participant_aliases (
 participant_id INTEGER NOT NULL REFERENCES participants(id), alias TEXT NOT NULL,
 normalized_alias TEXT NOT NULL, first_seen_at INTEGER NOT NULL,last_seen_at INTEGER NOT NULL,
 observation_count INTEGER NOT NULL DEFAULT 1,source_kind TEXT NOT NULL DEFAULT 'observed',
 confidence REAL NOT NULL DEFAULT 1,is_active INTEGER NOT NULL DEFAULT 1,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 PRIMARY KEY(participant_id,normalized_alias));
CREATE TABLE IF NOT EXISTS participant_alias_observations (
 message_id INTEGER NOT NULL REFERENCES messages(id),participant_id INTEGER NOT NULL REFERENCES participants(id),
 umo TEXT NOT NULL,alias TEXT NOT NULL,normalized_alias TEXT NOT NULL,relation TEXT NOT NULL,
 position INTEGER NOT NULL DEFAULT 0,sent_at INTEGER NOT NULL,
 PRIMARY KEY(message_id,participant_id,normalized_alias,relation));
CREATE TABLE IF NOT EXISTS message_participants (
 message_id INTEGER NOT NULL REFERENCES messages(id),participant_id INTEGER NOT NULL REFERENCES participants(id),
 relation TEXT NOT NULL,position INTEGER NOT NULL DEFAULT 0,evidence TEXT NOT NULL DEFAULT 'host',
 PRIMARY KEY(message_id,participant_id,relation,position));
CREATE TABLE IF NOT EXISTS message_relations (
 id INTEGER PRIMARY KEY AUTOINCREMENT,umo TEXT NOT NULL,source_message_id INTEGER NOT NULL REFERENCES messages(id),
 relation TEXT NOT NULL,target_message_id INTEGER REFERENCES messages(id),target_source_key TEXT NOT NULL DEFAULT '',
 target_platform_message_id TEXT NOT NULL DEFAULT '',target_participant_id INTEGER REFERENCES participants(id),
 metadata_json TEXT NOT NULL DEFAULT '{}',created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 UNIQUE(source_message_id,relation,target_source_key));
CREATE TABLE IF NOT EXISTS message_revisions (
 id INTEGER PRIMARY KEY AUTOINCREMENT,message_id INTEGER NOT NULL REFERENCES messages(id),revision_no INTEGER NOT NULL,
 content_sha256 TEXT NOT NULL,plain_text TEXT NOT NULL,content_json TEXT NOT NULL,role TEXT NOT NULL,
 revision_kind TEXT NOT NULL,observed_at INTEGER NOT NULL,sent_at INTEGER NOT NULL DEFAULT 0,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,UNIQUE(message_id,revision_no));
CREATE TABLE IF NOT EXISTS message_processing (message_id INTEGER PRIMARY KEY REFERENCES messages(id),
 content_sha256 TEXT NOT NULL,status TEXT NOT NULL DEFAULT 'PENDING',batch_key TEXT NOT NULL DEFAULT '',
 attempts INTEGER NOT NULL DEFAULT 0,last_error TEXT NOT NULL DEFAULT '',distilled_at INTEGER,
 updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,processing_class TEXT NOT NULL DEFAULT 'LIVE',
 ingestion_source TEXT NOT NULL DEFAULT 'adapter_live');
CREATE TABLE IF NOT EXISTS episodes (
 id INTEGER PRIMARY KEY AUTOINCREMENT,umo TEXT NOT NULL,started_at INTEGER NOT NULL,ended_at INTEGER NOT NULL,
 title TEXT NOT NULL DEFAULT '',summary TEXT NOT NULL DEFAULT '',status TEXT NOT NULL DEFAULT 'OPEN',
 extractor_version TEXT NOT NULL DEFAULT '',created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 stable_key TEXT NOT NULL DEFAULT '',revision_no INTEGER NOT NULL DEFAULT 1,updated_at TEXT NOT NULL DEFAULT '');
CREATE TABLE IF NOT EXISTS episode_messages (
 episode_id INTEGER NOT NULL REFERENCES episodes(id),message_id INTEGER NOT NULL REFERENCES messages(id),
 position INTEGER NOT NULL,PRIMARY KEY(episode_id,message_id));
CREATE TABLE IF NOT EXISTS episode_keywords (
 episode_id INTEGER NOT NULL REFERENCES episodes(id),cue TEXT NOT NULL COLLATE NOCASE,
 tag TEXT NOT NULL COLLATE NOCASE,PRIMARY KEY(episode_id,cue,tag));
CREATE TABLE IF NOT EXISTS semantic_memories (
 id INTEGER PRIMARY KEY AUTOINCREMENT,umo TEXT NOT NULL,person_cue TEXT NOT NULL COLLATE NOCASE,
 aspect_tag TEXT NOT NULL COLLATE NOCASE,content TEXT NOT NULL,source_message_id INTEGER REFERENCES messages(id),
 confidence REAL NOT NULL DEFAULT 0,extractor_version TEXT NOT NULL DEFAULT '',
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,stable_key TEXT NOT NULL DEFAULT '',
 subject_participant_id INTEGER REFERENCES participants(id),subject_text TEXT NOT NULL DEFAULT '',
 claim_type TEXT NOT NULL DEFAULT 'FACT',epistemic_status TEXT NOT NULL DEFAULT 'ASSERTED',
 status TEXT NOT NULL DEFAULT 'ACTIVE',superseded_by INTEGER REFERENCES semantic_memories(id),
 updated_at TEXT NOT NULL DEFAULT '');
CREATE TABLE IF NOT EXISTS semantic_memory_sources (
 semantic_memory_id INTEGER NOT NULL REFERENCES semantic_memories(id),message_id INTEGER NOT NULL REFERENCES messages(id),
 evidence_role TEXT NOT NULL DEFAULT 'SUPPORT',source_span TEXT NOT NULL DEFAULT '',confidence REAL NOT NULL DEFAULT 0,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,PRIMARY KEY(semantic_memory_id,message_id,evidence_role));
CREATE TABLE IF NOT EXISTS topics (
 id INTEGER PRIMARY KEY AUTOINCREMENT,umo TEXT NOT NULL,name TEXT NOT NULL COLLATE NOCASE,
 summary TEXT NOT NULL DEFAULT '',extractor_version TEXT NOT NULL DEFAULT '',UNIQUE(umo,name));
CREATE TABLE IF NOT EXISTS topic_episodes (
 topic_id INTEGER NOT NULL REFERENCES topics(id),episode_id INTEGER NOT NULL REFERENCES episodes(id),
 PRIMARY KEY(topic_id,episode_id));
CREATE TABLE IF NOT EXISTS mr_topic_sources (
 umo TEXT NOT NULL,topic_id INTEGER NOT NULL REFERENCES topics(id),message_id INTEGER NOT NULL REFERENCES messages(id),
 PRIMARY KEY(umo,topic_id,message_id));
CREATE TABLE IF NOT EXISTS plastic_nodes (
 id INTEGER PRIMARY KEY AUTOINCREMENT,umo TEXT NOT NULL,node_key TEXT NOT NULL,node_kind TEXT NOT NULL,
 label TEXT NOT NULL,description TEXT NOT NULL DEFAULT '',epistemic_confidence REAL NOT NULL DEFAULT 0,
 utility REAL NOT NULL DEFAULT 0,activation_count INTEGER NOT NULL DEFAULT 0,last_activated_at INTEGER,
 status TEXT NOT NULL DEFAULT 'ACTIVE',merged_into INTEGER REFERENCES plastic_nodes(id),created_by TEXT NOT NULL DEFAULT '',
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,UNIQUE(umo,node_key));
CREATE TABLE IF NOT EXISTS mr_node_aliases (
 umo TEXT NOT NULL,node_id INTEGER NOT NULL REFERENCES plastic_nodes(id),alias TEXT NOT NULL,
 PRIMARY KEY(umo,node_id,alias));
CREATE TABLE IF NOT EXISTS relation_types (
 id INTEGER PRIMARY KEY AUTOINCREMENT,umo TEXT NOT NULL,relation_key TEXT NOT NULL,version INTEGER NOT NULL DEFAULT 1,
 canonical_name TEXT NOT NULL,description TEXT NOT NULL,source_kinds_json TEXT NOT NULL DEFAULT '[]',
 target_kinds_json TEXT NOT NULL DEFAULT '[]',inverse_key TEXT NOT NULL DEFAULT '',symmetric INTEGER NOT NULL DEFAULT 0,
 risk_class TEXT NOT NULL DEFAULT 'normal',status TEXT NOT NULL DEFAULT 'ACTIVE',
 predecessor_id INTEGER REFERENCES relation_types(id),created_by TEXT NOT NULL DEFAULT '',
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 UNIQUE(umo,relation_key,version));
CREATE TABLE IF NOT EXISTS plastic_edges (
 id INTEGER PRIMARY KEY AUTOINCREMENT,umo TEXT NOT NULL,stable_key TEXT NOT NULL,
 source_node_id INTEGER NOT NULL REFERENCES plastic_nodes(id),relation_type_id INTEGER NOT NULL REFERENCES relation_types(id),
 target_node_id INTEGER NOT NULL REFERENCES plastic_nodes(id),statement TEXT NOT NULL DEFAULT '',
 epistemic_confidence REAL NOT NULL DEFAULT 0,epistemic_state TEXT NOT NULL DEFAULT 'HYPOTHESIS',uncertainty TEXT NOT NULL DEFAULT '',
 utility REAL NOT NULL DEFAULT 0,activation_count INTEGER NOT NULL DEFAULT 0,support_count INTEGER NOT NULL DEFAULT 0,
 contradict_count INTEGER NOT NULL DEFAULT 0,last_activated_at INTEGER,status TEXT NOT NULL DEFAULT 'ACTIVE',
 superseded_by INTEGER REFERENCES plastic_edges(id),created_by TEXT NOT NULL DEFAULT '',
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 invalidation_reason TEXT NOT NULL DEFAULT '',invalidated_at TEXT,UNIQUE(umo,stable_key));
CREATE TABLE IF NOT EXISTS plastic_edge_evidence (
 edge_id INTEGER NOT NULL REFERENCES plastic_edges(id),message_id INTEGER NOT NULL REFERENCES messages(id),
 evidence_role TEXT NOT NULL,confidence REAL NOT NULL DEFAULT 0,created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 PRIMARY KEY(edge_id,message_id,evidence_role));
CREATE TABLE IF NOT EXISTS memory_embeddings (
 umo TEXT NOT NULL,owner_type TEXT NOT NULL,owner_key TEXT NOT NULL,model TEXT NOT NULL,dimensions INTEGER NOT NULL,
 vector BLOB NOT NULL,updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,PRIMARY KEY(umo,owner_type,owner_key,model));
CREATE TABLE IF NOT EXISTS mr_working_state (umo TEXT PRIMARY KEY,state_json TEXT NOT NULL,updated_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS mr_roster_cache (umo TEXT PRIMARY KEY,payload_json TEXT NOT NULL,fetched_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS mr_index_pending (umo TEXT NOT NULL,owner_type TEXT NOT NULL,owner_key TEXT NOT NULL,
 text TEXT NOT NULL,updated_at INTEGER NOT NULL,PRIMARY KEY(umo,owner_type,owner_key));
CREATE TABLE IF NOT EXISTS mr_runs (id INTEGER PRIMARY KEY AUTOINCREMENT,umo TEXT NOT NULL,
 kind TEXT NOT NULL,started_at REAL NOT NULL,payload_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS mr_run_steps (run_id INTEGER NOT NULL REFERENCES mr_runs(id),seq INTEGER NOT NULL,
 at REAL NOT NULL,phase TEXT NOT NULL,status TEXT NOT NULL,title TEXT NOT NULL,data_json TEXT NOT NULL,
 PRIMARY KEY(run_id,seq));
CREATE TABLE IF NOT EXISTS mr_usage (id INTEGER PRIMARY KEY AUTOINCREMENT,umo TEXT NOT NULL,
 kind TEXT NOT NULL CHECK(kind IN ('background','feedback')),tokens INTEGER NOT NULL,
 occurred_at INTEGER NOT NULL,status TEXT NOT NULL,detail_json TEXT NOT NULL DEFAULT '{}',
 migration_key TEXT,UNIQUE(umo,migration_key));
CREATE INDEX IF NOT EXISTS idx_mr_usage_scope_time ON mr_usage(umo,kind,occurred_at);
"""

# Callers scope the actual message row m to this group. These are transport
# relationships, not a claim that everything in the message is about its author.
RELATED_ACCOUNT_SQL = """(m.sender_id=? OR EXISTS(SELECT 1 FROM message_participants mp
    JOIN participants p ON p.id=mp.participant_id WHERE mp.message_id=m.id AND p.umo=m.umo AND p.account_id=?)
    OR EXISTS(SELECT 1 FROM message_relations r JOIN participants p ON p.id=r.target_participant_id
        WHERE r.source_message_id=m.id AND r.umo=m.umo AND p.account_id=?)
    OR EXISTS(SELECT 1 FROM json_each(m.content_json) c WHERE json_extract(c.value,'$.type')='reply'
        AND CAST(json_extract(c.value,'$.sender_id') AS TEXT)=?))"""


def _encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _terms(values: Iterable[str] | str) -> list[str]:
    return list(dict.fromkeys(str(v).strip() for v in ([values] if isinstance(values, str) else values) if str(v).strip()))


def _limit(value: int, maximum: int = 500) -> int:
    return max(1, min(maximum, int(value)))


def _serialized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return call


class Store:
    def __init__(self, path: str | Path, umo: str):
        self._lock = RLock()
        self.umo = str(umo).strip()
        if not self.umo:
            raise ValueError("Store requires an explicit group scope")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path), timeout=10, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        if self.db.execute("SELECT 1 FROM sqlite_master WHERE name='scope_meta'").fetchone():
            scope = self.db.execute("SELECT umo FROM scope_meta WHERE singleton=1").fetchone()
            if scope and scope[0] != self.umo:
                self.db.close()
                raise ValueError("Database belongs to another group")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(SCHEMA)
        parts = self.umo.split(":")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO scope_meta(singleton,umo,platform_id,group_id) VALUES(1,?,?,?)",
                            (self.umo, parts[0], parts[-1]))
        self.tables = {r[0] for r in self.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    @_serialized
    def close(self) -> None:
        self.db.close()

    def _rows(self, sql: str, params: Iterable[Any] = ()) -> list[dict]:
        return [dict(r) for r in self.db.execute(sql, tuple(params))]

    def _person(self, platform_id: str, account_id: str, name: str, at: int, role: str = "USER") -> int:
        key = "participant:" + _encode([platform_id, account_id])
        self.db.execute("""INSERT INTO participants(umo,platform_id,account_id,canonical_key,account_type,
            current_display_name,first_seen_at,last_seen_at) VALUES(?,?,?,?,?,?,?,?)
            ON CONFLICT(umo,platform_id,account_id) DO UPDATE SET
            account_type=CASE WHEN participants.account_type='BOT' OR excluded.account_type='BOT'
                THEN 'BOT' ELSE excluded.account_type END,
            current_display_name=CASE WHEN excluded.last_seen_at>=participants.last_seen_at
                AND excluded.current_display_name<>'' THEN excluded.current_display_name ELSE participants.current_display_name END,
            first_seen_at=min(participants.first_seen_at,excluded.first_seen_at),
            last_seen_at=max(participants.last_seen_at,excluded.last_seen_at),updated_at=CURRENT_TIMESTAMP""",
            (self.umo, platform_id, account_id, key, role, name, at, at))
        return int(self.db.execute("SELECT id FROM participants WHERE umo=? AND platform_id=? AND account_id=?",
                                   (self.umo, platform_id, account_id)).fetchone()[0])

    def _alias(self, participant_id: int, name: str, at: int, message_id: int, relation: str, position: int = 0) -> None:
        if not name:
            return
        normalized = name.casefold().strip()
        self.db.execute("""INSERT INTO participant_aliases(participant_id,alias,normalized_alias,first_seen_at,last_seen_at)
            VALUES(?,?,?,?,?) ON CONFLICT(participant_id,normalized_alias) DO UPDATE SET
            first_seen_at=min(participant_aliases.first_seen_at,excluded.first_seen_at),
            last_seen_at=max(participant_aliases.last_seen_at,excluded.last_seen_at),updated_at=CURRENT_TIMESTAMP""",
            (participant_id, name, normalized, at, at))
        self.db.execute("""INSERT OR IGNORE INTO participant_alias_observations
            (message_id,participant_id,umo,alias,normalized_alias,relation,position,sent_at) VALUES(?,?,?,?,?,?,?,?)""",
            (message_id, participant_id, self.umo, name, normalized, relation, position, at))

    @_serialized
    def append_message(self, message: dict) -> dict:
        if str(message.get("umo") or self.umo) != self.umo:
            raise ValueError("Message belongs to another group")
        platform_id = str(message.get("platform_id") or "")
        account = str(message.get("sender_id") or "")
        msg_id = str(message.get("message_id") or "")
        if not platform_id or not account or not msg_id:
            raise ValueError("Message requires platform_id, sender_id and message_id")
        if self._forgotten(platform_id, account):
            return {"ignored": "forgotten_account"}
        source = str(message.get("source_key") or f"{platform_id}|{self.umo}|{msg_id}")
        role = str(message.get("role") or "USER").upper()
        if role not in {"USER", "BOT", "SYSTEM"}:
            raise ValueError("Unsupported message role")
        at, name = int(message["sent_at"]), str(message.get("sender_name") or account)
        text, content = str(message.get("plain_text") or ""), message.get("content") or []
        encoded = _encode(content)
        digest = hashlib.sha256(_encode([text, content, role]).encode()).hexdigest()
        with self.db:
            old = self.db.execute("SELECT * FROM messages WHERE source_key=?", (source,)).fetchone()
            if old and old["umo"] != self.umo:
                raise ValueError("Source key belongs to another group")
            bot_event = role == "SYSTEM" and any(isinstance(c, dict) and c.get("type") == "bot_event" for c in content)
            participant = self._person(platform_id, account, name, at, "BOT" if role == "BOT" or bot_event else role)
            if old is None:
                cursor = self.db.execute("""INSERT INTO messages(source_key,platform,platform_id,umo,group_id,
                    message_id,sender_id,sender_name,sent_at,plain_text,content_json,role,sender_participant_id,content_sha256)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (source, str(message.get("platform") or "aiocqhttp"), platform_id, self.umo,
                     str(message.get("group_id") or ""), msg_id, account, name, at, text, encoded, role, participant, digest))
                row_id, revision = int(cursor.lastrowid), 1
            else:
                row_id, revision = int(old["id"]), int(old["revision_no"])
                # Preserve the old text before replacing a changed platform message.
                changed = (old["plain_text"], json.loads(old["content_json"]), old["role"]) != (text, content, role)
                if changed:
                    self._revision(dict(old), "PREVIOUS")
                    revision += 1
                    self.db.execute("""UPDATE messages SET plain_text=?,content_json=?,role=?,content_sha256=?,
                        revision_no=?,updated_at=CURRENT_TIMESTAMP WHERE id=?""", (text, encoded, role, digest, revision, row_id))
            self.db.execute("INSERT OR IGNORE INTO message_participants(message_id,participant_id,relation,evidence) VALUES(?,?,'SENDER','host')",
                            (row_id, participant))
            self._alias(participant, name, at, row_id, "SENDER")
            for position, component in enumerate(content):
                if not isinstance(component, dict) or str(component.get("type", "")).casefold() not in {"at", "mention"}:
                    continue
                target = str(component.get("qq") or component.get("account_id") or component.get("id") or "")
                if not target or target in {"all", "0"} or self._forgotten(platform_id, target):
                    continue
                target_name = str(component.get("display_name") or component.get("name") or component.get("display") or "")
                target_id = self._person(platform_id, target, target_name, at)
                self.db.execute("""INSERT OR IGNORE INTO message_participants
                    (message_id,participant_id,relation,position,evidence) VALUES(?,?,'MENTIONED',?,'platform_mention')""",
                    (row_id, target_id, position))
                self._alias(target_id, target_name, at, row_id, "MENTIONED", position)
            reply = message.get("reply_to")
            if reply:
                if isinstance(reply, dict):
                    reply = reply.get("source_key") or reply.get("message_id") or reply.get("id")
                reply = str(reply or "")
                target_key = reply if "|" in reply else f"{platform_id}|{self.umo}|{reply}"
                target_row = self.db.execute("SELECT id,sender_participant_id,message_id FROM messages WHERE umo=? AND source_key=?",
                                             (self.umo, target_key)).fetchone()
                relation = "RESPONDS_TO" if role == "BOT" else "REPLY_TO"
                self.db.execute("""INSERT INTO message_relations(umo,source_message_id,relation,target_message_id,
                    target_source_key,target_platform_message_id,target_participant_id) VALUES(?,?,?,?,?,?,?)
                    ON CONFLICT(source_message_id,relation,target_source_key) DO UPDATE SET
                    target_message_id=excluded.target_message_id,target_participant_id=excluded.target_participant_id""",
                    (self.umo, row_id, relation, target_row[0] if target_row else None, target_key,
                     target_row[2] if target_row else reply.rsplit("|", 1)[-1], target_row[1] if target_row else None))
            # A quoted message may arrive after its reply during history import.
            self.db.execute("""UPDATE message_relations SET target_message_id=?,target_participant_id=?
                WHERE umo=? AND target_source_key=? AND target_message_id IS NULL""", (row_id, participant, self.umo, source))
            current = dict(self.db.execute("SELECT * FROM messages WHERE id=?", (row_id,)).fetchone())
            self._revision(current, "CAPTURED")
            self.db.execute("""INSERT INTO message_processing(message_id,content_sha256) VALUES(?,?)
                ON CONFLICT(message_id) DO UPDATE SET status=CASE
                WHEN message_processing.content_sha256<>excluded.content_sha256 THEN 'PENDING'
                ELSE message_processing.status END,content_sha256=excluded.content_sha256""", (row_id, current["content_sha256"]))
        return self._message(current)

    def _forgotten(self, platform_id: str, account_id: str) -> bool:
        digest = hashlib.sha256("\x1f".join((self.umo, platform_id, str(account_id))).encode()).hexdigest()
        return self.db.execute("SELECT 1 FROM forgotten_accounts WHERE umo=? AND platform_id=? AND account_hash=?",
                               (self.umo, platform_id, digest)).fetchone() is not None

    def _invalidate_sources(self, ids: list[int]) -> None:
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        owners = {
            "episode": self._rows(f"SELECT DISTINCT episode_id id FROM episode_messages WHERE message_id IN({placeholders})", ids),
            "semantic": self._rows(f"SELECT DISTINCT semantic_memory_id id FROM semantic_memory_sources WHERE message_id IN({placeholders}) UNION SELECT id FROM semantic_memories WHERE source_message_id IN({placeholders})", [*ids, *ids]),
            "plastic_edge": self._rows(f"SELECT DISTINCT edge_id id FROM plastic_edge_evidence WHERE message_id IN({placeholders})", ids),
            "topic": self._rows(f"SELECT DISTINCT topic_id id FROM mr_topic_sources WHERE umo=? AND message_id IN({placeholders})", [self.umo, *ids]),
        }
        for kind, values in owners.items():
            for row in values:
                if kind != "topic":
                    table = {"episode": "episodes", "semantic": "semantic_memories", "plastic_edge": "plastic_edges"}[kind]
                    self.db.execute(f"UPDATE {table} SET status='INVALIDATED' WHERE umo=? AND id=?", (self.umo, row["id"]))
                self.db.execute("DELETE FROM memory_embeddings WHERE umo=? AND owner_type=? AND owner_key=?", (self.umo, kind, str(row["id"])))
                self.db.execute("DELETE FROM mr_index_pending WHERE umo=? AND owner_type=? AND owner_key=?", (self.umo, kind, str(row["id"])))
        if "hypothesis_evidence" in self.tables:
            self.db.execute(f"""UPDATE feedback_hypotheses SET invalidation_reason='source_removed',status='DORMANT'
                WHERE umo=? AND id IN(SELECT h.hypothesis_id FROM hypothesis_evidence h JOIN messages m ON m.source_key=h.feedback_source_key AND m.umo=? WHERE m.id IN({placeholders}))""", [self.umo, self.umo, *ids])

    @_serialized
    def delete_message(self, message_id: str) -> int:
        rows = self._rows("SELECT * FROM messages WHERE umo=? AND (message_id=? OR source_key=?) AND is_deleted=0", (self.umo, str(message_id), str(message_id)))
        with self.db:
            for row in rows:
                self.db.execute("UPDATE messages SET is_deleted=1,deleted_at=?,updated_at=CURRENT_TIMESTAMP WHERE id=?", (int(time.time()), row["id"]))
            self._invalidate_sources([r["id"] for r in rows])
        return len(rows)

    @_serialized
    def forget(self, account_id: str) -> dict:
        people = self._rows("SELECT * FROM participants WHERE umo=? AND account_id=?", (self.umo, str(account_id)))
        platform_ids = {p["platform_id"] for p in people} or {self.umo.split(":")[0]}
        rows = self._rows("SELECT id FROM messages WHERE umo=? AND sender_id=?", (self.umo, str(account_id)))
        ids = [r["id"] for r in rows]
        with self.db:
            for platform in platform_ids:
                digest = hashlib.sha256("\x1f".join((self.umo, platform, str(account_id))).encode()).hexdigest()
                self.db.execute("INSERT OR REPLACE INTO forgotten_accounts(umo,platform_id,account_hash,requested_at) VALUES(?,?,?,?)", (self.umo, platform, digest, int(time.time())))
            self._invalidate_sources(ids)
            for row_id in ids:
                self.db.execute("UPDATE messages SET is_deleted=1,plain_text='',content_json='[]',sender_name='',deleted_at=? WHERE id=?", (int(time.time()), row_id))
                self.db.execute("UPDATE message_revisions SET plain_text='',content_json='[]' WHERE message_id=?", (row_id,))
            for person in people:
                self.db.execute("DELETE FROM participant_aliases WHERE participant_id=?", (person["id"],))
                self.db.execute("DELETE FROM participant_alias_observations WHERE participant_id=?", (person["id"],))
                self.db.execute("UPDATE participants SET current_display_name='' WHERE id=?", (person["id"],))
                self.db.execute("DELETE FROM memory_embeddings WHERE umo=? AND owner_type='participant' AND owner_key=?", (self.umo, str(person["id"])))
            self.db.execute("DELETE FROM mr_working_state WHERE umo=?", (self.umo,))
            self.db.execute("DELETE FROM mr_roster_cache WHERE umo=?", (self.umo,))
        return {"forgotten": True, "messages_removed": len(ids)}

    def _revision(self, row: dict, kind: str) -> None:
        self.db.execute("""INSERT OR IGNORE INTO message_revisions
            (message_id,revision_no,content_sha256,plain_text,content_json,role,revision_kind,observed_at,sent_at)
            VALUES(?,?,?,?,?,?,?,?,?)""", (row["id"], row["revision_no"], row["content_sha256"], row["plain_text"],
            row["content_json"], row["role"], kind, int(time.time()), row["sent_at"]))

    @_serialized
    def resolve_quotes(self, content: Any, platform_id: str) -> Any:
        """Resolve quote authors and times without changing supplied content or storage."""
        def quote_content(value: Any) -> Any:
            if isinstance(value, list):
                return [quote_content(part) for part in value]
            if not isinstance(value, dict):
                return value
            part = {key: quote_content(item) for key, item in value.items()}
            if str(part.get("type", "")).casefold() != "reply":
                return part
            original = None
            fields = "id,source_key,sent_at,sender_id,sender_name,role"
            if part.get("source_key"):
                original = self.db.execute(f"SELECT {fields} FROM messages WHERE umo=? AND source_key=? AND is_deleted=0",
                                           (self.umo, str(part["source_key"]))).fetchone()
            if original is None and part.get("message_id"):
                original = self.db.execute(f"SELECT {fields} FROM messages WHERE umo=? AND platform_id=? AND message_id=? AND is_deleted=0",
                                           (self.umo, platform_id, str(part["message_id"]))).fetchone()
            # AstrBot's Reply.time can be the wrapper's fetch time. It does
            # not establish when the quoted message was originally sent.
            part.pop("time", None)
            part.pop("sent_at", None)
            part["time_status"] = "original_time_unknown"
            if original is not None:
                part.update(source_id=original["id"], source_key=original["source_key"],
                            sender_id=original["sender_id"], sender_name=original["sender_name"], role=original["role"])
                if original["sent_at"] > 0:
                    part.update(sent_at=original["sent_at"], time_status="original_message")
            return part
        return quote_content(content)

    def _message(self, row: dict) -> dict:
        relations = self._rows("SELECT relation,target_source_key,target_participant_id FROM message_relations WHERE umo=? AND source_message_id=?",
                               (self.umo, row["id"]))
        result = {key: row[key] for key in ("id", "source_key", "platform_id", "group_id", "sender_id", "sender_name", "sent_at", "plain_text", "role")}
        result.update(participant_id=row.get("sender_participant_id"), content=self.resolve_quotes(json.loads(row["content_json"]), row["platform_id"]),
                      reply_to=next((r["target_source_key"] for r in relations if r["relation"] in {"REPLY_TO", "RESPONDS_TO"}), None),
                      relations=relations, revision_no=row.get("revision_no", 1))
        if "message_attachments" in self.tables:
            result["attachments"] = self._rows("SELECT position,attachment_type,extraction_status,descriptor_text,reference_sha256 FROM message_attachments WHERE message_id=?", (row["id"],))
        return result

    @_serialized
    def recent(self, limit: int = 24, before: int | None = None) -> list[dict]:
        clauses, args = ["umo=?", "is_deleted=0"], [self.umo]
        if before is not None:
            clauses.append("sent_at<?")
            args.append(int(before))
        rows = self._rows(f"SELECT * FROM messages WHERE {' AND '.join(clauses)} ORDER BY sent_at DESC,id DESC LIMIT ?", [*args, _limit(limit)])
        return [self._message(r) for r in reversed(rows)]

    @_serialized
    def members(self, account_ids: Iterable[str] | None = None, name: str | None = None) -> list[dict]:
        clauses, args = ["p.umo=?"], [self.umo]
        if account_ids is not None:
            ids = _terms(account_ids)
            if not ids:
                return []
            clauses.append(f"p.account_id IN ({','.join('?' for _ in ids)})")
            args.extend(ids)
        if name:
            clauses.append("(instr(lower(p.current_display_name),lower(?))>0 OR EXISTS (SELECT 1 FROM participant_aliases a WHERE a.participant_id=p.id AND instr(lower(a.alias),lower(?))>0))")
            args.extend((str(name), str(name)))
        people = self._rows(f"SELECT p.* FROM participants p WHERE {' AND '.join(clauses)} ORDER BY p.last_seen_at DESC,p.id", args)
        people = [p for p in people if not self._forgotten(p["platform_id"], p["account_id"])]
        result = [dict(id=p["id"], account_id=p["account_id"], canonical_key=p["canonical_key"], name=p["current_display_name"],
                     aliases=[r["alias"] for r in self._rows("SELECT alias FROM participant_aliases WHERE participant_id=? ORDER BY last_seen_at DESC", (p["id"],))],
                     account_type=p["account_type"], last_seen_at=p["last_seen_at"], membership="observed_in_history") for p in people]
        cached = self.roster()
        by_account = {p["account_id"]: p for p in result}
        if cached:
            wanted = set(ids) if account_ids is not None else None
            for member in cached["members"]:
                if not isinstance(member, dict):
                    continue
                account = str(member.get("user_id") or member.get("account_id") or "")
                names = [str(member.get(k) or "") for k in ("card", "nickname", "name")]
                if not account or (wanted is not None and account not in wanted):
                    continue
                if self._forgotten(self.umo.split(":")[0], account):
                    continue
                person = by_account.get(account)
                if name and person is None and not any(str(name).casefold() in n.casefold() for n in names):
                    continue
                if person is None:
                    observed = self.db.execute("SELECT id,canonical_key FROM participants WHERE umo=? AND account_id=?", (self.umo, account)).fetchone()
                    person = dict(id=observed[0] if observed else None, account_id=account,
                                  canonical_key=observed[1] if observed else None, aliases=[], account_type="USER")
                    result.append(person)
                person.update(name=next((n for n in names if n), person.get("name", account)),
                              role=member.get("role"), membership="platform_roster",
                              roster_fetched_at=cached["fetched_at"])
        return result

    @_serialized
    def search_messages(self, terms: Iterable[str] = (), participant_id: int | None = None,
                        start_at: int | None = None, end_at: int | None = None, limit: int = 20,
                        *, sender_id: str | None = None, related_account_id: str | None = None) -> list[dict]:
        clauses, args = ["m.umo=?", "m.is_deleted=0"], [self.umo]
        words = _terms(terms)
        if words:
            clauses.append("(" + " OR ".join("(instr(lower(m.plain_text),lower(?))>0 OR instr(lower(m.content_json),lower(?))>0)" for _ in words) + ")")
            args.extend(word for word in words for _ in range(2))
        if participant_id is not None:
            clauses.append("(m.sender_participant_id=? OR EXISTS(SELECT 1 FROM message_participants mp WHERE mp.message_id=m.id AND mp.participant_id=?))")
            args.extend((int(participant_id), int(participant_id)))
        if sender_id is not None:
            clauses.append("m.sender_id=?")
            args.append(str(sender_id))
        if related_account_id is not None:
            # Involvement includes sending, mentions, and replies. It is a
            # separate query from the actual-author filter above.
            clauses.append(RELATED_ACCOUNT_SQL)
            args.extend([str(related_account_id)] * 4)
        for field, op, value in (("sent_at", ">=", start_at), ("sent_at", "<", end_at)):
            if value is not None:
                clauses.append(f"m.{field}{op}?")
                args.append(int(value))
        rows = self._rows(f"SELECT m.* FROM messages m WHERE {' AND '.join(clauses)} ORDER BY m.sent_at DESC,m.id DESC LIMIT ?", [*args, _limit(limit)])
        return [self._message(r) for r in rows]

    @_serialized
    def context(self, source_key: str | None = None, before: int = 8, after: int = 8,
                before_time: int | None = None, *, message_id: int | None = None) -> list[dict]:
        if (source_key is None) == (message_id is None):
            raise ValueError("Context requires exactly one source_key or internal message_id")
        column, value = ("source_key", source_key) if source_key is not None else ("id", int(message_id))
        anchor = self.db.execute(f"SELECT * FROM messages WHERE umo=? AND {column}=? AND is_deleted=0", (self.umo, value)).fetchone()
        if anchor is None or (before_time is not None and anchor["sent_at"] > int(before_time)):
            return []
        bound, params = (" AND sent_at<=?", [int(before_time)]) if before_time is not None else ("", [])
        left = self._rows(f"SELECT * FROM messages WHERE umo=? AND is_deleted=0 AND (sent_at,id)<(?,?){bound} ORDER BY sent_at DESC,id DESC LIMIT ?",
                         [self.umo, anchor["sent_at"], anchor["id"], *params, max(0, min(100, int(before)))])
        right = self._rows(f"SELECT * FROM messages WHERE umo=? AND is_deleted=0 AND (sent_at,id)>(?,?){bound} ORDER BY sent_at,id LIMIT ?",
                          [self.umo, anchor["sent_at"], anchor["id"], *params, max(0, min(100, int(after)))])
        return [self._message(r) for r in [*reversed(left), dict(anchor), *right]]

    @_serialized
    def activity(self, participant_id: int | None = None, start_at: int | None = None,
                 end_at: int | None = None, *, account_id: str | None = None) -> dict:
        if start_at is None or end_at is None or (participant_id is None and account_id is None):
            raise ValueError("Activity requires an account and time range")
        author = "sender_id" if account_id is not None else "sender_participant_id"
        args = (self.umo, str(account_id) if account_id is not None else int(participant_id), int(start_at), int(end_at))
        where = f"umo=? AND {author}=? AND sent_at>=? AND sent_at<? AND is_deleted=0"
        total = dict(self.db.execute(f"SELECT count(*) message_count,min(sent_at) first_at,max(sent_at) last_at FROM messages WHERE {where}", args).fetchone())
        days = self._rows(f"SELECT date(sent_at,'unixepoch','+8 hours') date,count(*) message_count,min(sent_at) first_at,max(sent_at) last_at FROM messages WHERE {where} GROUP BY date ORDER BY date", args)
        hours = {int(r["hour"]): r["message_count"] for r in self._rows(f"SELECT strftime('%H',sent_at,'unixepoch','+8 hours') hour,count(*) message_count FROM messages WHERE {where} GROUP BY hour", args)}
        return {**total, "days": days, "hour_counts": [{"hour": h, "message_count": hours.get(h, 0)} for h in range(24)], "timezone": "Asia/Shanghai", "start_at": int(start_at), "end_at": int(end_at)}

    def _sources(self, kind: str, row: dict) -> list[dict]:
        if kind == "episode":
            sql = "SELECT m.id,m.source_key,m.sender_participant_id,m.sender_id,m.sender_name,m.role FROM episode_messages x JOIN messages m ON m.id=x.message_id WHERE x.episode_id=? AND m.umo=? ORDER BY m.sent_at,m.id"
        elif kind == "semantic":
            sql = "SELECT m.id,m.source_key,m.sender_participant_id,m.sender_id,m.sender_name,m.role FROM messages m WHERE m.umo=? AND (m.id=? OR m.id IN(SELECT message_id FROM semantic_memory_sources WHERE semantic_memory_id=?)) ORDER BY m.sent_at,m.id"
            return self._rows(sql, (self.umo, row.get("source_message_id"), row["id"]))
        elif kind == "association":
            sql = "SELECT m.id,m.source_key,m.sender_participant_id,m.sender_id,m.sender_name,m.role FROM plastic_edge_evidence x JOIN messages m ON m.id=x.message_id WHERE x.edge_id=? AND m.umo=? ORDER BY m.sent_at,m.id"
        elif kind == "topic":
            sql = """SELECT m.id,m.source_key,m.sender_participant_id,m.sender_id,m.sender_name,m.role FROM messages m
                WHERE m.umo=? AND (m.id IN(SELECT x.message_id FROM topic_episodes t JOIN episode_messages x ON x.episode_id=t.episode_id WHERE t.topic_id=?)
                    OR m.id IN(SELECT message_id FROM mr_topic_sources WHERE umo=? AND topic_id=?)) ORDER BY m.sent_at,m.id"""
            return self._rows(sql, (self.umo, row["id"], self.umo, row["id"]))
        else:
            return []
        return self._rows(sql, (row["id"], self.umo))

    def _memory(self, kind: str, row: dict, include_sources: bool = True) -> dict:
        sources = self._sources(kind, row)
        result = {"kind": kind, "id": row["id"], "title": row.get("title") or row.get("name") or row.get("aspect_tag") or "",
                  "summary": row.get("summary") or row.get("content") or row.get("statement") or "",
                  "source_keys": [source["source_key"] for source in sources], "source_ids": [source["id"] for source in sources],
                  "participant_id": row.get("subject_participant_id"),
                  "status": row.get("status", "ACTIVE"), "narrative_bindings": None}
        if include_sources:
            result["sources"] = self._messages_for_sources(result["source_keys"])
        # Actual source authors are useful even when an old summary omitted its
        # local identity bindings. They do not imply a mapping to textual pN labels.
        speakers = dict.fromkeys((s["sender_participant_id"], s["sender_id"], s["sender_name"], s["role"]) for s in sources)
        result["source_speakers"] = [dict(participant_id=person, account_id=account, name_at_message=name, role=role)
                                     for person, account, name, role in speakers]
        subject = self.db.execute("SELECT account_id,current_display_name FROM participants WHERE umo=? AND id=?",
                                  (self.umo, row.get("subject_participant_id"))).fetchone()
        subject_name = row.get("subject_text") or row.get("person_cue") or (subject["current_display_name"] if subject else "")
        result["subject"] = {"name": subject_name, "account_id": subject["account_id"] if subject else None} if subject or subject_name else None
        for field in ("started_at", "ended_at", "created_at", "updated_at", "confidence", "person_cue", "subject_text", "epistemic_status", "uncertainty"):
            if field in row:
                result[field] = row[field]
        if "narrative_identity_bindings" in self.tables:
            binding = self.db.execute("SELECT metadata_json FROM narrative_identity_bindings WHERE umo=? AND owner_type=? AND owner_id=?",
                                      (self.umo, "plastic_edge" if kind == "association" else kind, row["id"])).fetchone()
            if binding:
                result["narrative_bindings"] = json.loads(binding[0])
        return result

    @_serialized
    def search_memories(self, kind: str = "all", terms: Iterable[str] = (), participant_id: int | None = None,
                        limit: int = 12, *, related_account_id: str | None = None, include_sources: bool = False) -> list[dict]:
        kinds = ("episode", "semantic", "topic") if kind == "all" else (kind,)
        outputs: list[dict] = []
        words = _terms(terms)
        prefix, prefix_args = "", []
        if related_account_id is not None:
            prefix = f"WITH related_messages AS MATERIALIZED (SELECT m.id FROM messages m WHERE m.umo=? AND m.is_deleted=0 AND {RELATED_ACCOUNT_SQL}) "
            prefix_args = [self.umo, *([str(related_account_id)] * 4)]
        for selected in kinds:
            if selected in {"association", "plastic_edge"}:
                outputs.extend(self.graph(terms=words, limit=limit, related_account_id=related_account_id, include_sources=include_sources))
                continue
            tables = {"episode": ("episodes", "title || ' ' || summary"), "semantic": ("semantic_memories", "person_cue || ' ' || aspect_tag || ' ' || content"), "topic": ("topics", "name || ' ' || summary")}
            if selected not in tables:
                raise ValueError("Unknown memory kind")
            table, text = tables[selected]
            clauses, args = ["umo=?"], [self.umo]
            if selected == "semantic":
                clauses.append("status='ACTIVE'")
            elif selected == "episode":
                clauses.append("status<>'INVALIDATED'")
            elif selected == "topic":
                clauses.append("NOT EXISTS(SELECT 1 FROM topic_episodes t JOIN episodes e ON e.id=t.episode_id WHERE t.topic_id=topics.id AND e.status='INVALIDATED')")
                clauses.append("NOT EXISTS(SELECT 1 FROM mr_topic_sources t JOIN messages m ON m.id=t.message_id WHERE t.umo=topics.umo AND t.topic_id=topics.id AND m.is_deleted=1)")
            if words:
                clauses.append("(" + " OR ".join(f"instr(lower({text}),lower(?))>0" for _ in words) + ")")
                args.extend(words)
            if participant_id is not None:
                if selected == "semantic":
                    clauses.append("subject_participant_id=?")
                elif selected == "episode":
                    clauses.append("id IN(SELECT em.episode_id FROM episode_messages em JOIN messages m ON m.id=em.message_id WHERE m.sender_participant_id=?)")
                else:
                    clauses.append("id IN(SELECT te.topic_id FROM topic_episodes te JOIN episode_messages em ON em.episode_id=te.episode_id JOIN messages m ON m.id=em.message_id WHERE m.sender_participant_id=?)")
                args.append(int(participant_id))
            if related_account_id is not None:
                if selected == "semantic":
                    clauses.append("""(subject_participant_id IN(SELECT id FROM participants WHERE umo=? AND account_id=?)
                        OR source_message_id IN(SELECT id FROM related_messages)
                        OR id IN(SELECT semantic_memory_id FROM semantic_memory_sources WHERE message_id IN(SELECT id FROM related_messages)))""")
                    args.extend((self.umo, str(related_account_id)))
                elif selected == "episode":
                    clauses.append("id IN(SELECT episode_id FROM episode_messages WHERE message_id IN(SELECT id FROM related_messages))")
                else:
                    clauses.append("""(id IN(SELECT t.topic_id FROM topic_episodes t JOIN episode_messages e ON e.episode_id=t.episode_id
                        WHERE e.message_id IN(SELECT id FROM related_messages))
                        OR id IN(SELECT topic_id FROM mr_topic_sources WHERE message_id IN(SELECT id FROM related_messages)))""")
            rows = self._rows(f"{prefix}SELECT * FROM {table} WHERE {' AND '.join(clauses)} ORDER BY id DESC LIMIT ?", [*prefix_args, *args, _limit(limit)])
            outputs.extend(self._memory(selected, r, include_sources=include_sources) for r in rows)
        # Round robin avoids filling the whole result with only episodes.
        if kind == "all":
            groups = [[r for r in outputs if r["kind"] == k] for k in kinds]
            outputs = [group[i] for i in range(max(map(len, groups), default=0)) for group in groups if i < len(group)]
        return outputs[:_limit(limit)]

    @_serialized
    def memory(self, kind: str, id: int | str, *, include_sources: bool = True) -> dict | None:
        if kind in {"association", "plastic_edge"}:
            rows = self._graph_rows(" AND e.id=?", [int(id)])
            return self._graph_record(rows[0], include_sources=include_sources) if rows else None
        if kind == "participant":
            rows = self._rows("SELECT * FROM participants WHERE umo=? AND id=?", (self.umo, int(id)))
            if not rows:
                return None
            people = self.members(account_ids=[rows[0]["account_id"]])
            if not people:
                return None
            person = people[0]
            return {"kind": kind, **person, "title": person["name"], "summary": " / ".join(person["aliases"]), "source_keys": [], "source_ids": []}
        if kind == "cue":
            rows = self.search_messages(terms=[str(id)], limit=12)
            return {"kind": "cue", "id": str(id), "title": str(id), "summary": str(id), "source_keys": [r["source_key"] for r in rows], "source_ids": [r["id"] for r in rows]}
        tables = {"episode": "episodes", "semantic": "semantic_memories", "topic": "topics"}
        if kind not in tables:
            raise ValueError("Unknown memory kind")
        rows = self._rows(f"SELECT * FROM {tables[kind]} WHERE umo=? AND id=?", (self.umo, int(id)))
        if not rows:
            return None
        if rows[0].get("status") in {"INVALIDATED", "SUPERSEDED", "RETRACTED"}:
            return None
        if kind == "topic" and self.db.execute("""SELECT 1 FROM topic_episodes t JOIN episodes e ON e.id=t.episode_id
            WHERE t.topic_id=? AND e.umo=? AND e.status='INVALIDATED' LIMIT 1""", (int(id), self.umo)).fetchone():
            return None
        if kind == "topic" and self.db.execute("""SELECT 1 FROM mr_topic_sources t JOIN messages m ON m.id=t.message_id
            WHERE t.umo=? AND t.topic_id=? AND m.is_deleted=1 LIMIT 1""", (self.umo, int(id))).fetchone():
            return None
        return self._memory(kind, rows[0], include_sources=include_sources)

    def _messages_for_sources(self, sources: Iterable[str]) -> list[dict]:
        keys = list(dict.fromkeys(sources))
        if not keys:
            return []
        rows = self._rows(f"SELECT * FROM messages WHERE umo=? AND source_key IN({','.join('?' for _ in keys)}) AND is_deleted=0 ORDER BY sent_at,id", [self.umo, *keys])
        return [self._message(r) for r in rows]

    @_serialized
    def memory_sources(self, kind: str, id: int | str) -> list[dict]:
        """Open a memory's retained original messages, including quote authors."""
        entry = self.memory(kind, id)
        return entry.get("sources", []) if entry else []

    def _graph_rows(self, extra: str = "", args: Iterable[Any] = (), limit: int = 500) -> list[dict]:
        return self._rows("""SELECT e.*,s.label AS source,t.label AS target,r.canonical_name AS relation
            FROM plastic_edges e JOIN plastic_nodes s ON s.id=e.source_node_id AND s.umo=e.umo
            JOIN plastic_nodes t ON t.id=e.target_node_id AND t.umo=e.umo
            JOIN relation_types r ON r.id=e.relation_type_id AND r.umo=e.umo
            WHERE e.umo=? AND e.status IN('ACTIVE','WEAKENED') AND e.invalidation_reason=''""" + extra + " ORDER BY e.id DESC LIMIT ?", [self.umo, *args, _limit(limit)])

    def _graph_record(self, row: dict, include_sources: bool = True) -> dict:
        result = self._memory("association", row, include_sources=include_sources)
        result.update({key: row[key] for key in ("statement", "source", "target", "relation", "source_node_id", "target_node_id", "epistemic_state", "uncertainty")})
        result["source_node"] = self._node(row["source_node_id"])
        result["target_node"] = self._node(row["target_node_id"])
        return result

    def _node(self, node_id: int) -> dict | None:
        row = self.db.execute("SELECT id,label,description FROM plastic_nodes WHERE umo=? AND id=?",
                              (self.umo, int(node_id))).fetchone()
        if row is None:
            return None
        return {"node_id": row["id"], "label": row["label"], "description": row["description"],
                "aliases": [r[0] for r in self.db.execute("SELECT alias FROM mr_node_aliases WHERE umo=? AND node_id=? ORDER BY alias",
                                                         (self.umo, row["id"]))]}

    def _save_node(self, endpoint: str | dict) -> int:
        # A label describes a node; only an explicit record address reuses it.
        value = {"label": endpoint} if isinstance(endpoint, str) else endpoint
        if not isinstance(value, dict):
            raise ValueError("An association endpoint requires node_id or a new label")
        node_id = value.get("node_id")
        if node_id is not None:
            current = self._node(int(node_id))
            if current is None:
                raise ValueError("Graph node does not exist in this group")
            label = str(value.get("label", current["label"])).strip()
            description = str(value.get("description", current["description"])).strip()
            if not label:
                raise ValueError("Graph node label must not be empty")
            if (label, description) != (current["label"], current["description"]):
                self.db.execute("UPDATE plastic_nodes SET label=?,description=?,updated_at=CURRENT_TIMESTAMP WHERE umo=? AND id=?",
                                (label, description, self.umo, int(node_id)))
        else:
            label = str(value.get("label") or "").strip()
            if not label:
                raise ValueError("A new graph node requires a label")
            node_id = int(self.db.execute("""INSERT INTO plastic_nodes(umo,node_key,node_kind,label,description,created_by)
                VALUES(?,'mr-node:'||lower(hex(randomblob(16))),'entity',?,?,'mr-simple')""",
                (self.umo, label, str(value.get("description") or "").strip())).lastrowid)
        aliases = value.get("aliases", [])
        if isinstance(aliases, str):
            aliases = [aliases]
        for alias in dict.fromkeys(str(a).strip() for a in aliases if str(a).strip()):
            self.db.execute("INSERT OR IGNORE INTO mr_node_aliases(umo,node_id,alias) VALUES(?,?,?)",
                            (self.umo, int(node_id), alias))
        if value.get("node_id") is not None and any(field in value for field in ("label", "description", "aliases")):
            for edge in self._rows("SELECT id FROM plastic_edges WHERE umo=? AND (source_node_id=? OR target_node_id=?)",
                                   (self.umo, int(node_id), int(node_id))):
                self._queue_embedding("association", edge["id"])
        return int(node_id)

    @_serialized
    def graph(self, node_id: int | None = None, terms: Iterable[str] = (), limit: int = 12,
              *, related_account_id: str | None = None, include_sources: bool = False) -> list[dict]:
        extra, args = "", []
        if node_id is not None:
            extra += " AND (e.source_node_id=? OR e.target_node_id=?)"
            args.extend((int(node_id), int(node_id)))
        if related_account_id is not None:
            extra += f" AND EXISTS(SELECT 1 FROM plastic_edge_evidence x JOIN messages m ON m.id=x.message_id WHERE x.edge_id=e.id AND m.umo=e.umo AND m.is_deleted=0 AND {RELATED_ACCOUNT_SQL})"
            args.extend([str(related_account_id)] * 4)
        words = _terms(terms)
        if words:
            extra += " AND (" + " OR ".join("""(instr(lower(s.label || ' ' || s.description || ' ' || t.label || ' ' || t.description || ' ' || e.statement || ' ' || r.canonical_name),lower(?))>0
                OR EXISTS(SELECT 1 FROM mr_node_aliases a WHERE a.umo=e.umo AND a.node_id IN(s.id,t.id) AND instr(lower(a.alias),lower(?))>0))""" for _ in words) + ")"
            args.extend(word for word in words for _ in range(2))
        return [self._graph_record(r, include_sources=include_sources) for r in self._graph_rows(extra, args, limit)]

    @_serialized
    def feedback_messages(self, window_seconds: int, after: int, limit: int = 500) -> list[dict]:
        """Return the next new reaction and its conversation, with an id watermark.

        The first unprocessed user message must follow a retained bot response.
        The model, not this query, determines whether it is meaningful feedback.
        """
        now = int(time.time())
        cutoff = now - int(window_seconds)
        first = self.db.execute("""SELECT u.* FROM messages u WHERE u.umo=? AND u.is_deleted=0
            AND u.role='USER' AND u.id>? AND u.sent_at BETWEEN ? AND ?
            AND EXISTS(SELECT 1 FROM messages b WHERE b.umo=u.umo AND b.is_deleted=0
                AND b.role='BOT' AND b.sent_at>=? AND (b.sent_at,b.id)<(u.sent_at,u.id))
            ORDER BY u.sent_at,u.id LIMIT 1""", (self.umo, int(after), cutoff, now, cutoff)).fetchone()
        if first is None:
            return []
        bot = self.db.execute("""SELECT * FROM messages WHERE umo=? AND is_deleted=0 AND role='BOT'
            AND sent_at>=? AND (sent_at,id)<(?,?) ORDER BY sent_at DESC,id DESC LIMIT 1""",
            (self.umo, cutoff, first["sent_at"], first["id"])).fetchone()
        bot_message = self._message(dict(bot))
        # Include the bot's original question when the send observation retained
        # a reply relation. This is structural association, not text matching.
        if bot_message["reply_to"]:
            rows = self._rows("""SELECT m.* FROM messages m WHERE m.umo=? AND m.is_deleted=0
                AND m.sent_at>=? AND (m.sent_at,m.id)<=(?,?) AND
                (m.source_key=? OR EXISTS(SELECT 1 FROM message_relations r WHERE r.source_message_id=m.id
                    AND r.umo=m.umo AND r.target_source_key=? AND r.relation IN('REPLY_TO','RESPONDS_TO')))
                ORDER BY m.sent_at,m.id""", (self.umo, cutoff, bot["sent_at"], bot["id"],
                                            bot_message["reply_to"], bot_message["reply_to"]))
            prefix = [self._message(row) for row in rows]
        else:
            prefix = [bot_message]
        prefix = prefix[-max(1, _limit(limit) - 1):]
        room = max(1, _limit(limit) - len(prefix))
        rows = self._rows("""SELECT * FROM messages WHERE umo=? AND is_deleted=0 AND sent_at<=?
            AND (sent_at,id)>=(?,?) ORDER BY sent_at,id LIMIT ?""",
            (self.umo, now, first["sent_at"], first["id"], room))
        return prefix + [self._message(row) for row in rows]

    @staticmethod
    def _usage_kind(kind: str) -> str:
        if kind not in {"background", "feedback"}:
            raise ValueError("Usage kind must be background or feedback")
        return kind

    def _carryover_usage(self) -> None:
        row = self.db.execute("SELECT state_json,updated_at FROM mr_working_state WHERE umo=?", (self.umo,)).fetchone()
        if row is None:
            return
        state = json.loads(row["state_json"])
        if "background_tokens" not in state:
            return
        # The removed implementation kept only a daily aggregate. Its event
        # times cannot be recovered; keep one explicitly estimated aggregate.
        at = int(state.get("consolidated_at") or row["updated_at"])
        detail = {"time_basis": "consolidated_at" if state.get("consolidated_at") else "working_state_updated_at",
                  "original_background_day": state.get("background_day"), "aggregate_only": True}
        with self.db:
            self.db.execute("""INSERT OR IGNORE INTO mr_usage
                (umo,kind,tokens,occurred_at,status,detail_json,migration_key)
                VALUES(?,'background',?,?,'carryover_estimate',?,'working_state_background_tokens')""",
                (self.umo, max(0, int(state["background_tokens"])), at, _encode(detail)))

    @_serialized
    def reserve_usage(self, kind: str, tokens: int, at: int | None = None) -> int:
        kind = self._usage_kind(kind)
        if int(tokens) < 0:
            raise ValueError("Usage tokens cannot be negative")
        self._carryover_usage()
        with self.db:
            return int(self.db.execute("""INSERT INTO mr_usage(umo,kind,tokens,occurred_at,status)
                VALUES(?,?,?,?,'reserved')""", (self.umo, kind, int(tokens), int(time.time()) if at is None else int(at))).lastrowid)

    @_serialized
    def settle_usage(self, id: int, tokens: int) -> None:
        if int(tokens) < 0:
            raise ValueError("Usage tokens cannot be negative")
        with self.db:
            changed = self.db.execute("""UPDATE mr_usage SET tokens=?,status='settled'
                WHERE umo=? AND id=? AND migration_key IS NULL""", (int(tokens), self.umo, int(id))).rowcount
            if not changed:
                raise ValueError("Usage reservation does not exist in this group")

    @_serialized
    def usage_total(self, kind: str, now: int | None = None) -> int:
        """Combine retained legacy usage and new reservations over rolling 24h."""
        kind = self._usage_kind(kind)
        self._carryover_usage()
        end = int(time.time()) if now is None else int(now)
        since, watermark = end - 86400, 0
        old_class = "online" if kind == "background" else "feedback"
        if "token_budget_resets" in self.tables:
            reset = self.db.execute("""SELECT reset_at,usage_event_id FROM token_budget_resets
                WHERE umo=? AND budget_class=? AND reset_at<=? ORDER BY reset_at DESC,id DESC LIMIT 1""",
                (self.umo, old_class, end)).fetchone()
            if reset:
                since, watermark = max(since, int(reset["reset_at"])), int(reset["usage_event_id"] or 0)
        total = int(self.db.execute("""SELECT COALESCE(SUM(tokens),0) FROM mr_usage
            WHERE umo=? AND kind=? AND occurred_at BETWEEN ? AND ?""", (self.umo, kind, since, end)).fetchone()[0])
        if {"llm_usage_events", "experiment_runs"}.issubset(self.tables):
            # Preserve the existing operator budget/reset contract. Historical
            # imports and resident_reader_one_pass do not belong to this class.
            phases = ("construction", "construction_repair", "reconstruction", "reconstruction_deep",
                      "certificate_reader") if kind == "background" else ("feedback_maintenance",)
            predicate = "u.phase IN (" + ",".join("?" for _ in phases) + ")"
            if kind == "background":
                predicate += " OR u.phase GLOB 'eccr_*'"
            legacy = self.db.execute(f"""SELECT COALESCE(SUM(u.input_other+u.input_cached+u.output),0)
                FROM llm_usage_events u JOIN experiment_runs r ON r.run_id=u.run_id
                WHERE r.umo=? AND unixepoch(u.created_at) BETWEEN ? AND ? AND u.id>? AND ({predicate})""",
                (self.umo, since, end, watermark, *phases)).fetchone()[0]
            total += int(legacy)
        return total

    @_serialized
    def load_working_state(self) -> Any:
        row = self.db.execute("SELECT state_json FROM mr_working_state WHERE umo=?", (self.umo,)).fetchone()
        return json.loads(row[0]) if row else {}

    @_serialized
    def save_working_state(self, state: Any) -> None:
        with self.db:
            self.db.execute("INSERT INTO mr_working_state VALUES(?,?,?) ON CONFLICT(umo) DO UPDATE SET state_json=excluded.state_json,updated_at=excluded.updated_at", (self.umo, _encode(state), int(time.time())))

    @_serialized
    def record_run(self, kind: str, started_at: float, payload: dict) -> int:
        with self.db:
            return int(self.db.execute("INSERT INTO mr_runs(umo,kind,started_at,payload_json) VALUES(?,?,?,?)",
                                      (self.umo, kind, started_at, _encode(payload))).lastrowid)

    @_serialized
    def append_run_step(self, run_id: int, seq: int, at: float, phase: str, status: str, title: str, data: dict) -> None:
        with self.db:
            changed = self.db.execute("""INSERT INTO mr_run_steps(run_id,seq,at,phase,status,title,data_json)
                SELECT id,?,?,?,?,?,? FROM mr_runs WHERE umo=? AND id=?""",
                (int(seq), float(at), phase, status, title, _encode(data), self.umo, int(run_id))).rowcount
            if not changed:
                raise ValueError("Run does not exist in this group")

    @_serialized
    def update_run(self, run_id: int, payload: dict) -> None:
        with self.db:
            row = self.db.execute("SELECT payload_json FROM mr_runs WHERE umo=? AND id=?", (self.umo, int(run_id))).fetchone()
            if row is None:
                raise ValueError("Run does not exist in this group")
            combined = {**json.loads(row["payload_json"]), **payload}
            self.db.execute("UPDATE mr_runs SET payload_json=? WHERE umo=? AND id=?", (_encode(combined), self.umo, int(run_id)))

    @_serialized
    def recent_runs(self, limit: int = 30) -> list[dict]:
        rows = self._rows("""SELECT r.*,s.at AS step_at,s.phase AS latest_phase,s.title AS latest_title,
            (SELECT count(*) FROM mr_run_steps n WHERE n.run_id=r.id) AS step_count FROM mr_runs r
            LEFT JOIN mr_run_steps s ON s.run_id=r.id AND s.seq=(SELECT max(n.seq) FROM mr_run_steps n WHERE n.run_id=r.id)
            WHERE r.umo=? ORDER BY r.id DESC LIMIT ?""", (self.umo, _limit(limit)))
        result = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            updated = payload.get("finished_at") or row["step_at"] or row["started_at"] if payload.get("trace_version") else None
            result.append({**{k: v for k, v in payload.items() if k not in ("messages", "tool_calls", "items", "written")},
                           "id": row["id"], "kind": row["kind"], "started_at": row["started_at"], "updated_at": updated,
                           "step_count": row["step_count"], "latest_phase": row["latest_phase"], "latest_title": row["latest_title"]})
        return result

    @_serialized
    def run_detail(self, run_id: int) -> dict | None:
        row = self.db.execute("SELECT * FROM mr_runs WHERE umo=? AND id=?", (self.umo, int(run_id))).fetchone()
        steps = self._rows("SELECT seq,at,phase,status,title,data_json FROM mr_run_steps WHERE run_id=? ORDER BY seq", (int(run_id),)) if row else []
        for step in steps:
            step["data"] = json.loads(step.pop("data_json"))
        return {**json.loads(row["payload_json"]), "id": row["id"], "kind": row["kind"],
                "started_at": row["started_at"], "steps": steps} if row else None

    @_serialized
    def response_events(self, request_id: str) -> list[dict]:
        """Read recorded main-agent activity linked to this one platform request."""
        rows = self._rows("""SELECT DISTINCT m.* FROM message_relations r JOIN messages m ON m.id=r.source_message_id
            WHERE r.umo=? AND m.umo=r.umo AND r.target_platform_message_id=? AND m.is_deleted=0
            ORDER BY m.sent_at,m.id""", (self.umo, str(request_id)))
        result = []
        for row in rows:
            content = json.loads(row["content_json"])
            if any(isinstance(part, dict) and part.get("type") == "bot_event"
                   and part.get("event") in {"generated", "tool_call", "tool_result", "sent"} for part in content):
                result.append(self._message(row))
        return result

    @_serialized
    def cache_roster(self, payload: Any, fetched_at: int) -> None:
        # Keep opt-outs when the platform refreshes its member list.
        payload = json.loads(_encode(payload))
        members = payload if isinstance(payload, list) else payload.get("members", payload.get("data", [])) if isinstance(payload, dict) else []
        if isinstance(members, list):
            members[:] = [m for m in members if not isinstance(m, dict) or not self._forgotten(
                self.umo.split(":")[0], str(m.get("user_id") or m.get("account_id") or ""))]
        with self.db:
            self.db.execute("INSERT INTO mr_roster_cache VALUES(?,?,?) ON CONFLICT(umo) DO UPDATE SET payload_json=excluded.payload_json,fetched_at=excluded.fetched_at", (self.umo, _encode(payload), int(fetched_at)))

    @_serialized
    def roster(self) -> dict | None:
        row = self.db.execute("SELECT payload_json,fetched_at FROM mr_roster_cache WHERE umo=?", (self.umo,)).fetchone()
        if not row:
            return None
        payload = json.loads(row[0])
        members = payload if isinstance(payload, list) else payload.get("members", payload.get("data", [])) if isinstance(payload, dict) else []
        return {"payload": payload, "members": members, "fetched_at": row[1]}

    @_serialized
    def pending_status(self) -> dict:
        row = self.db.execute("""SELECT count(*) AS count,min(m.sent_at) AS oldest_at
            FROM messages m LEFT JOIN message_processing p ON p.message_id=m.id
            WHERE m.umo=? AND m.is_deleted=0 AND COALESCE(p.status,'PENDING')<>'DISTILLED'""",
            (self.umo,)).fetchone()
        return dict(row)

    @_serialized
    def pending_messages(self, limit: int, newest: bool = True) -> list[dict]:
        # Live learning and historical recovery can alternate without dropping
        # old failed work. Both batches are read chronologically by the model.
        direction = "DESC" if newest else "ASC"
        rows = self._rows(f"""SELECT m.* FROM messages m LEFT JOIN message_processing p ON p.message_id=m.id
            WHERE m.umo=? AND m.is_deleted=0 AND COALESCE(p.status,'PENDING')<>'DISTILLED'
            ORDER BY m.sent_at {direction},m.id {direction} LIMIT ?""", (self.umo, _limit(limit)))
        return [self._message(r) for r in sorted(rows, key=lambda row: (row["sent_at"], row["id"]))]

    def _queue_embedding(self, kind: str, owner: int | str) -> None:
        entry = self.memory(kind, owner, include_sources=False)
        if not entry:
            return
        text = "\n".join(str(entry.get(k) or "") for k in ("title", "summary", "source", "relation", "target")).strip()
        if kind == "association":
            descriptions = [str(node.get("description") or "") + "\n" + " / ".join(node.get("aliases", []))
                            for node in (entry["source_node"], entry["target_node"])]
            text = "\n".join([text, *descriptions]).strip()
        owner_type = "plastic_edge" if kind == "association" else kind
        self.db.execute("""INSERT INTO mr_index_pending(umo,owner_type,owner_key,text,updated_at) VALUES(?,?,?,?,?)
            ON CONFLICT(umo,owner_type,owner_key) DO UPDATE SET text=excluded.text,updated_at=excluded.updated_at""", (self.umo, owner_type, str(owner), text, int(time.time())))

    @_serialized
    def pending_embeddings(self, model_id: str, limit: int = 16) -> list[dict]:
        queued = self._rows("SELECT owner_type,owner_key,text,updated_at FROM mr_index_pending WHERE umo=? ORDER BY updated_at DESC,owner_type,owner_key LIMIT ?", (self.umo, _limit(limit)))
        if queued:
            return queued
        with self.db:
            # Existing memories missing this model are also durable pending work.
            for kind, table in (("episode", "episodes"), ("semantic", "semantic_memories"), ("topic", "topics"), ("plastic_edge", "plastic_edges")):
                eligible = " AND x.status IN('ACTIVE','WEAKENED')" if kind == "plastic_edge" else " AND x.status='ACTIVE'" if kind == "semantic" else " AND x.status<>'INVALIDATED'" if kind == "episode" else ""
                if kind == "topic":
                    eligible += " AND NOT EXISTS(SELECT 1 FROM mr_topic_sources t JOIN messages m ON m.id=t.message_id WHERE t.umo=x.umo AND t.topic_id=x.id AND m.is_deleted=1)"
                rows = self._rows(f"""SELECT x.id FROM {table} x WHERE x.umo=? {eligible}
                    AND NOT EXISTS(SELECT 1 FROM memory_embeddings v WHERE v.umo=x.umo AND v.owner_type=? AND v.owner_key=CAST(x.id AS TEXT) AND v.model=?)
                    AND NOT EXISTS(SELECT 1 FROM mr_index_pending q WHERE q.umo=x.umo AND q.owner_type=? AND q.owner_key=CAST(x.id AS TEXT))
                    ORDER BY x.id LIMIT ?""", (self.umo, kind, model_id, kind, _limit(limit)))
                for row in rows:
                    self._queue_embedding(kind, row["id"])
        return self._rows("SELECT owner_type,owner_key,text,updated_at FROM mr_index_pending WHERE umo=? ORDER BY updated_at DESC,owner_type,owner_key LIMIT ?", (self.umo, _limit(limit)))

    @_serialized
    def save_embedding(self, doc: dict, model_id: str, vector: Iterable[float]) -> bool:
        current = self.db.execute("SELECT text FROM mr_index_pending WHERE umo=? AND owner_type=? AND owner_key=?", (self.umo, doc["owner_type"], str(doc["owner_key"]))).fetchone()
        if current is None or current[0] != doc["text"]:
            return False
        values = list(vector)
        blob = struct.pack(f"<{len(values)}f", *values)
        self.put_vector(doc["owner_type"], str(doc["owner_key"]), model_id, blob, len(values))
        with self.db:
            self.db.execute("DELETE FROM mr_index_pending WHERE umo=? AND owner_type=? AND owner_key=? AND text=?", (self.umo, doc["owner_type"], str(doc["owner_key"]), doc["text"]))
        return True

    @_serialized
    def vector_rows(self, model_id: str) -> list[dict]:
        rows = self._rows("SELECT model,owner_type,owner_key,dimensions,vector FROM memory_embeddings WHERE umo=? AND model=?", (self.umo, model_id))
        owners = {"episode": "episodes", "semantic": "semantic_memories", "topic": "topics", "plastic_edge": "plastic_edges", "participant": "participants"}
        allowed = {}
        for kind, table in owners.items():
            status = " AND status IN('ACTIVE','WEAKENED') AND invalidation_reason=''" if kind == "plastic_edge" else " AND status='ACTIVE'" if kind == "semantic" else " AND status<>'INVALIDATED'" if kind == "episode" else ""
            if kind == "topic":
                status += " AND NOT EXISTS(SELECT 1 FROM mr_topic_sources t JOIN messages m ON m.id=t.message_id WHERE t.umo=topics.umo AND t.topic_id=topics.id AND m.is_deleted=1)"
            allowed[kind] = {str(r[0]) for r in self.db.execute(f"SELECT id FROM {table} WHERE umo=?{status}", (self.umo,))}
        allowed["participant"] = {str(p["id"]) for p in self.members() if p["id"] is not None}
        rows = [r for r in rows if r["owner_type"] == "cue" or r["owner_key"] in allowed.get(r["owner_type"], set())]
        seen = {(r["owner_type"], r["owner_key"]) for r in rows}
        if "embeddings" in self.tables:
            for row in self._rows("SELECT model,owner_type,owner_id,dimensions,vector FROM embeddings WHERE model=?", (model_id,)):
                row["owner_key"] = str(row.pop("owner_id"))
                key = (row["owner_type"], row["owner_key"])
                if key not in seen and key[1] in allowed.get(key[0], set()):
                    rows.append(row)
                    seen.add(key)
        for row in rows:
            row["embedding_blob"] = row["vector"]
        return rows

    @_serialized
    def put_vector(self, owner_type: str, owner_key: str, model_id: str, vector: bytes, dimensions: int) -> None:
        if int(dimensions) <= 0 or len(vector) != int(dimensions) * 4:
            raise ValueError("Vector must be float32 bytes with matching dimensions")
        if self.memory(owner_type, owner_key) is None:
            raise ValueError("Vector owner is not in this group")
        with self.db:
            self.db.execute("""INSERT INTO memory_embeddings(umo,owner_type,owner_key,model,dimensions,vector)
                VALUES(?,?,?,?,?,?) ON CONFLICT(umo,owner_type,owner_key,model) DO UPDATE SET
                dimensions=excluded.dimensions,vector=excluded.vector,updated_at=CURRENT_TIMESTAMP""", (self.umo, owner_type, str(owner_key), model_id, int(dimensions), vector))

    @_serialized
    def save_memories(self, items: Iterable[dict], sources: Iterable[str], mark_processed: bool = True) -> list[dict]:
        keys = list(dict.fromkeys(str(s) for s in sources))
        source_rows = {r["source_key"]: r for r in self._messages_for_sources(keys)}
        if set(keys) != set(source_rows):
            raise ValueError("Memory sources must be retained messages in this group")
        outputs: list[tuple[str, int]] = []
        with self.db:
            for item in items:
                selected = list(dict.fromkeys(str(k) for k in item.get("source_keys", keys)))
                if not selected or not set(selected).issubset(source_rows):
                    raise ValueError("Each memory requires existing input source keys")
                evidence = [source_rows[k] for k in selected]
                kind = str(item.get("kind") or "semantic")
                kind = {"episodic": "episode", "plastic_edge": "association"}.get(kind, kind)
                table = {"episode": "episodes", "semantic": "semantic_memories", "association": "plastic_edges", "topic": "topics"}.get(kind)
                if table is None:
                    raise ValueError("Unsupported memory write kind")
                if kind == "topic" and item.get("id") is None:
                    raise ValueError("Topic revisions require an existing id; use episode for new experiences")
                stored = None
                if item.get("id") is not None:
                    stored = self.db.execute(f"SELECT * FROM {table} WHERE umo=? AND id=?", (self.umo, int(item["id"]))).fetchone()
                    if stored is None:
                        raise ValueError("Memory to update does not exist in this group")
                    stored = dict(stored)
                text_field = {"episode": "summary", "semantic": "content", "association": "statement", "topic": "summary"}[kind]
                summary = str(item.get("content") or item.get("summary") or item.get("statement") or (stored[text_field] if stored else "")).strip()
                if not summary:
                    raise ValueError("Memory content must not be empty")
                fingerprint = hashlib.sha256(_encode([kind, item, sorted(selected)]).encode()).hexdigest()
                stable = "mr-simple:" + fingerprint
                if kind == "episode":
                    if stored:
                        owner = int(stored["id"])
                        self.db.execute("""UPDATE episodes SET title=?,summary=?,started_at=?,ended_at=?,revision_no=revision_no+1,
                            updated_at=CURRENT_TIMESTAMP WHERE umo=? AND id=?""",
                            (str(item.get("title", stored["title"])), summary,
                             min(stored["started_at"], *(r["sent_at"] for r in evidence)),
                             max(stored["ended_at"], *(r["sent_at"] for r in evidence)), self.umo, owner))
                    else:
                        existing = self.db.execute("SELECT id FROM episodes WHERE umo=? AND stable_key=?", (self.umo, stable)).fetchone()
                        owner = int(existing[0]) if existing else int(self.db.execute("""INSERT INTO episodes
                            (umo,started_at,ended_at,title,summary,status,extractor_version,stable_key,updated_at)
                            VALUES(?,?,?,?,?,'CLOSED','mr-simple',?,CURRENT_TIMESTAMP)""", (self.umo, min(r["sent_at"] for r in evidence), max(r["sent_at"] for r in evidence), str(item.get("title") or ""), summary, stable)).lastrowid)
                    for position, message in enumerate(evidence):
                        self.db.execute("INSERT OR IGNORE INTO episode_messages VALUES(?,?,?)", (owner, message["id"], position))
                    cues = item.get("cues", item.get("keywords", []))
                    for cue in _terms(cues if isinstance(cues, (str, list, tuple)) else []):
                        self.db.execute("INSERT OR IGNORE INTO episode_keywords VALUES(?,?,?)", (owner, cue, "model"))
                elif kind == "semantic":
                    participant = item.get("participant_id", stored.get("subject_participant_id") if stored else None)
                    person = str(item.get("person", stored.get("person_cue", "") if stored else ""))
                    subject = item.get("subject", stored.get("subject_text", "") if stored else "")
                    if "person" in item and "subject" not in item:
                        subject = person
                    if "participant_id" not in item and ("person" in item or "subject" in item):
                        participant = None
                    if isinstance(subject, dict):
                        account = subject.get("account_id")
                        if account is not None:
                            author = self.db.execute("SELECT id FROM participants WHERE umo=? AND account_id=?", (self.umo, str(account))).fetchone()
                            if author is None:
                                raise ValueError("Memory subject account does not exist in this group")
                            participant = author[0]
                        subject = subject.get("name") or ""
                    if "subject" in item and "person" not in item:
                        person = str(subject)
                    if participant is not None and not self.db.execute("SELECT 1 FROM participants WHERE umo=? AND id=?", (self.umo, int(participant))).fetchone():
                        raise ValueError("Memory participant belongs to another group")
                    aspect = str(item.get("aspect", stored.get("aspect_tag", "") if stored else ""))
                    if stored:
                        owner = int(stored["id"])
                        self.db.execute("""UPDATE semantic_memories SET person_cue=?,aspect_tag=?,content=?,subject_participant_id=?,
                            subject_text=?,updated_at=CURRENT_TIMESTAMP WHERE umo=? AND id=?""",
                            (person, aspect, summary, participant, str(subject), self.umo, owner))
                    else:
                        existing = self.db.execute("SELECT id FROM semantic_memories WHERE umo=? AND stable_key=?", (self.umo, stable)).fetchone()
                        owner = int(existing[0]) if existing else int(self.db.execute("""INSERT INTO semantic_memories
                            (umo,person_cue,aspect_tag,content,source_message_id,extractor_version,stable_key,
                             subject_participant_id,subject_text,updated_at) VALUES(?,?,?,?,?,'mr-simple',?,?,?,CURRENT_TIMESTAMP)""",
                            (self.umo, person, aspect, summary, evidence[0]["id"], stable, participant, str(subject))).lastrowid)
                    for message in evidence:
                        self.db.execute("INSERT OR IGNORE INTO semantic_memory_sources(semantic_memory_id,message_id,evidence_role) VALUES(?,?,'SUPPORT')", (owner, message["id"]))
                elif kind == "topic":
                    owner = int(stored["id"])
                    name = str(item.get("name", item.get("title", stored["name"])))
                    self.db.execute("UPDATE topics SET name=?,summary=? WHERE umo=? AND id=?", (name, summary, self.umo, owner))
                    for message in evidence:
                        self.db.execute("INSERT OR IGNORE INTO mr_topic_sources(umo,topic_id,message_id) VALUES(?,?,?)",
                                        (self.umo, owner, message["id"]))
                elif kind == "association":
                    source = item.get("source", {"node_id": stored["source_node_id"]} if stored else None)
                    target = item.get("target", {"node_id": stored["target_node_id"]} if stored else None)
                    relation = item.get("relation")
                    if relation is None and stored:
                        relation = self.db.execute("SELECT canonical_name FROM relation_types WHERE umo=? AND id=?", (self.umo, stored["relation_type_id"])).fetchone()[0]
                    relation = str(relation or "").strip()
                    if not source or not target or not relation:
                        raise ValueError("Association needs source, target and relation")
                    node_ids = [self._save_node(endpoint) for endpoint in (source, target)]
                    relation_key = "mr-simple:" + hashlib.sha256(relation.encode()).hexdigest()
                    self.db.execute("INSERT OR IGNORE INTO relation_types(umo,relation_key,canonical_name,description,created_by) VALUES(?,?,?,?,'mr-simple')", (self.umo, relation_key, relation, relation))
                    relation_id = self.db.execute("SELECT id FROM relation_types WHERE umo=? AND relation_key=? ORDER BY version DESC LIMIT 1", (self.umo, relation_key)).fetchone()[0]
                    uncertainty = str(item.get("uncertainty", stored["uncertainty"] if stored else ""))
                    if stored:
                        owner = int(stored["id"])
                        self.db.execute("""UPDATE plastic_edges SET source_node_id=?,relation_type_id=?,target_node_id=?,statement=?,
                            uncertainty=?,updated_at=CURRENT_TIMESTAMP WHERE umo=? AND id=?""",
                            (node_ids[0], relation_id, node_ids[1], summary, uncertainty, self.umo, owner))
                    else:
                        owner = int(self.db.execute("""INSERT INTO plastic_edges(umo,stable_key,source_node_id,relation_type_id,target_node_id,
                            statement,uncertainty,created_by) VALUES(?,?,?,?,?,?,?,'mr-simple')""",
                            (self.umo, "mr-simple:" + uuid4().hex, node_ids[0], relation_id, node_ids[1], summary, uncertainty)).lastrowid)
                    for message in evidence:
                        self.db.execute("INSERT OR IGNORE INTO plastic_edge_evidence(edge_id,message_id,evidence_role) VALUES(?,?,'SUPPORT')", (owner, message["id"]))
                self._queue_embedding(kind, owner)
                outputs.append((kind, owner))
            for key in keys if mark_processed else ():
                message = source_rows[key]
                raw = self.db.execute("SELECT content_sha256 FROM messages WHERE id=?", (message["id"],)).fetchone()
                self.db.execute("""INSERT INTO message_processing(message_id,content_sha256,status,distilled_at)
                    VALUES(?,?,'DISTILLED',?) ON CONFLICT(message_id) DO UPDATE SET status='DISTILLED',
                    distilled_at=excluded.distilled_at,content_sha256=excluded.content_sha256,last_error=''""", (message["id"], raw[0], int(time.time())))
        return [self.memory(kind, owner) for kind, owner in outputs]
