"""Small SQLite access layer for the existing, private, per-group MR database.

Stored prose is evidence for the model, never an identity decision made here.
Legacy memory tables are imported once; live access uses MemoryGraph.
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

from .reflection import Reflection, SCHEMA as REFLECTION_SCHEMA
from .content import searchable_text, text_view
from .memory_graph import MemoryGraph, canonical


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
CREATE INDEX IF NOT EXISTS idx_messages_umo_id ON messages(umo,id);
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
CREATE TABLE IF NOT EXISTS memory_embeddings (
 umo TEXT NOT NULL,owner_type TEXT NOT NULL,owner_key TEXT NOT NULL,model TEXT NOT NULL,dimensions INTEGER NOT NULL,
 vector BLOB NOT NULL,updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,PRIMARY KEY(umo,owner_type,owner_key,model));
CREATE TABLE IF NOT EXISTS mr_working_state (umo TEXT PRIMARY KEY,state_json TEXT NOT NULL,updated_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS mr_memory_write_receipts (
 umo TEXT NOT NULL,receipt_key TEXT NOT NULL,refs_json TEXT NOT NULL,
 PRIMARY KEY(umo,receipt_key));
CREATE TABLE IF NOT EXISTS mr_learning_tasks (umo TEXT NOT NULL,kind TEXT NOT NULL
 CHECK(kind IN ('background','feedback')),task_json TEXT NOT NULL,updated_at INTEGER NOT NULL,
 PRIMARY KEY(umo,kind));
CREATE TABLE IF NOT EXISTS mr_feedback_processed (umo TEXT NOT NULL,message_id INTEGER NOT NULL REFERENCES messages(id),
 completed_at INTEGER NOT NULL,PRIMARY KEY(umo,message_id));
CREATE TABLE IF NOT EXISTS mr_roster_cache (umo TEXT PRIMARY KEY,payload_json TEXT NOT NULL,fetched_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS mr_index_pending (umo TEXT NOT NULL,owner_type TEXT NOT NULL,owner_key TEXT NOT NULL,
 text TEXT NOT NULL,updated_at INTEGER NOT NULL,PRIMARY KEY(umo,owner_type,owner_key));
CREATE TABLE IF NOT EXISTS mr_memory_revisions (id INTEGER PRIMARY KEY AUTOINCREMENT,umo TEXT NOT NULL,
 kind TEXT NOT NULL,owner_id INTEGER NOT NULL,operation TEXT NOT NULL,reason TEXT NOT NULL DEFAULT '',
 run_id INTEGER,snapshot_json TEXT NOT NULL,source_ids_json TEXT NOT NULL DEFAULT '[]',
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE INDEX IF NOT EXISTS idx_mr_memory_revisions_owner ON mr_memory_revisions(umo,kind,owner_id,id);
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
        self.db.create_function("mr_searchable_text", 1, searchable_text, deterministic=True)
        if self.db.execute("SELECT 1 FROM sqlite_master WHERE name='scope_meta'").fetchone():
            scope = self.db.execute("SELECT umo FROM scope_meta WHERE singleton=1").fetchone()
            if scope and scope[0] != self.umo:
                self.db.close()
                raise ValueError("Database belongs to another group")
        self.db.execute("PRAGMA foreign_keys=ON")
        feedback_queue_exists = self.db.execute("SELECT 1 FROM sqlite_master WHERE name='mr_feedback_processed'").fetchone()
        self.db.executescript(SCHEMA)
        self.db.executescript(REFLECTION_SCHEMA)
        parts = self.umo.split(":")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO scope_meta(singleton,umo,platform_id,group_id) VALUES(1,?,?,?)",
                            (self.umo, parts[0], parts[-1]))
        self.tables = {r[0] for r in self.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not feedback_queue_exists:
            # The old cursor remains a legacy floor. Preserve sparse progress
            # on the unfinished old task too, without advancing that floor.
            with self.db:
                self.db.execute("""INSERT OR IGNORE INTO mr_feedback_processed
                    SELECT t.umo,m.id,t.updated_at FROM mr_learning_tasks t,
                    json_each(t.task_json,'$.completed_ids') done JOIN messages m ON m.id=done.value AND m.umo=t.umo
                    WHERE t.umo=? AND t.kind='feedback' AND m.is_deleted=0""", (self.umo,))
        self.reflections = Reflection(self)
        self.memory_graph = MemoryGraph(self)
        from .cognition import Cognition
        self.cognition = Cognition(self)
        self.cognition.migrate_notes()

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
                    self.db.execute("DELETE FROM mr_feedback_processed WHERE umo=? AND message_id=?", (self.umo, row_id))
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

    def _invalidate_sources(self, ids):
        if not ids:
            return {}
        marks = ",".join("?" for _ in ids)
        affected = {(row["kind"], row["id"]) for row in self._rows(
            f"""SELECT kind,id FROM mr_memory_objects WHERE umo=? AND EXISTS(
                SELECT 1 FROM json_each(attributes_json,'$.source_ids') s WHERE s.value IN({marks}))""", [self.umo, *ids])}
        affected.update(("message", int(value)) for value in ids)
        edges = self._rows("SELECT id,attributes_json FROM mr_memory_objects WHERE umo=? AND kind='association'", (self.umo,))
        changed = True
        while changed:
            before = len(affected)
            for edge in edges:
                attrs = json.loads(edge["attributes_json"])
                source, target = attrs.get("source_ref", {}), attrs.get("target_ref", {})
                sr, tr = (source.get("kind"), source.get("id")), (target.get("kind"), target.get("id"))
                if sr in affected or tr in affected:
                    affected.add(("association", edge["id"]))
                if attrs.get("purpose") == "basis" and tr in affected:
                    affected.add(sr)
            changed = len(affected) != before
        owners = {}
        for kind, owner in affected:
            owners.setdefault(kind, []).append({"id": owner})
            self.db.execute("UPDATE mr_memory_objects SET status='INVALIDATED' WHERE umo=? AND kind=? AND id=?", (self.umo, kind, owner))
            self._clear_memory_derivatives(kind, owner)
        return owners

    def _forget_memory_context(self, messages, invalidated):
        if not messages:
            return
        ids = [message["id"] for message in messages]
        marks = ",".join("?" for _ in ids)
        refs = {(kind, row["id"]) for kind, rows in invalidated.items() for row in rows}
        history = self._rows(f"""SELECT kind,owner_id FROM mr_memory_revisions WHERE umo=? AND
            (EXISTS(SELECT 1 FROM json_each(source_ids_json) s WHERE s.value IN({marks})) OR
             EXISTS(SELECT 1 FROM json_each(json_extract(snapshot_json,'$.memory.source_ids')) s WHERE s.value IN({marks})))""",
            [self.umo, *ids, *ids])
        refs.update((row["kind"], row["owner_id"]) for row in history)
        self.db.executemany("DELETE FROM mr_memory_revisions WHERE umo=? AND kind=? AND owner_id=?",
                            [(self.umo, kind, owner) for kind, owner in refs])
        reflection_rows = self._rows(f"""SELECT id FROM mr_memory_objects WHERE umo=? AND kind='reflection' AND
            (EXISTS(SELECT 1 FROM json_each(attributes_json,'$.source_ids') s WHERE s.value IN({marks}))
             OR json_extract(attributes_json,'$.request_id') IN({marks}))""",
            [self.umo, *ids, *(message["message_id"] for message in messages)])
        for kind, owner in refs:
            reflection_rows.extend(self._rows("""SELECT id FROM mr_memory_objects WHERE umo=? AND kind='reflection'
                AND EXISTS(SELECT 1 FROM json_each(attributes_json,'$.memory_refs') ref
                WHERE json_extract(ref.value,'$.kind')=? AND CAST(json_extract(ref.value,'$.id') AS INTEGER)=?)""",
                (self.umo, kind, owner)))
        for row in reflection_rows:
            self.db.execute("UPDATE mr_memory_objects SET status='INVALIDATED' WHERE umo=? AND kind='reflection' AND id=?", (self.umo, row["id"]))
            self._clear_memory_derivatives("reflection", row["id"])
            self.db.execute("DELETE FROM mr_memory_revisions WHERE umo=? AND kind='reflection' AND owner_id=?", (self.umo, row["id"]))
        for kind, owner in refs:
            for table in ("mr_memory_recalls", "mr_memory_rehearsals"):
                if table in self.tables:
                    self.db.execute(f"DELETE FROM {table} WHERE umo=? AND kind=? AND owner_id=?", (self.umo, kind, owner))
            self.db.execute("DELETE FROM mr_working_set WHERE umo=? AND kind=? AND owner_id=?", (self.umo, kind, owner))
            self.db.execute("""DELETE FROM mr_cognition_turns WHERE umo=? AND EXISTS(
                SELECT 1 FROM json_each(payload_json,'$.available_memories') r
                WHERE json_extract(r.value,'$.kind')=? AND json_extract(r.value,'$.id')=?)""", (self.umo, kind, owner))
        self.db.execute(f"DELETE FROM mr_cognition_turns WHERE umo=? AND observation_id IN({marks})", [self.umo, *ids])

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
        rows = self._rows("SELECT id,message_id FROM messages WHERE umo=? AND sender_id=?", (self.umo, str(account_id)))
        ids = [r["id"] for r in rows]
        with self.db:
            for platform in platform_ids:
                digest = hashlib.sha256("\x1f".join((self.umo, platform, str(account_id))).encode()).hexdigest()
                self.db.execute("INSERT OR REPLACE INTO forgotten_accounts(umo,platform_id,account_hash,requested_at) VALUES(?,?,?,?)", (self.umo, platform, digest, int(time.time())))
            invalidated = self._invalidate_sources(ids)
            self._forget_memory_context(rows, invalidated)
            for row_id in ids:
                self.db.execute("UPDATE messages SET is_deleted=1,plain_text='',content_json='[]',sender_name='',deleted_at=? WHERE id=?", (int(time.time()), row_id))
                self.db.execute("UPDATE message_revisions SET plain_text='',content_json='[]' WHERE message_id=?", (row_id,))
            for person in people:
                self.db.execute("DELETE FROM participant_aliases WHERE participant_id=?", (person["id"],))
                self.db.execute("DELETE FROM participant_alias_observations WHERE participant_id=?", (person["id"],))
                self.db.execute("UPDATE participants SET current_display_name='' WHERE id=?", (person["id"],))
                self.db.execute("DELETE FROM memory_embeddings WHERE umo=? AND owner_type='participant' AND owner_key=?", (self.umo, str(person["id"])))
            self.db.execute("DELETE FROM mr_working_state WHERE umo=?", (self.umo,))
            self.db.execute("DELETE FROM mr_learning_tasks WHERE umo=?", (self.umo,))
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
        if result["reply_to"]:
            original = self.db.execute("""SELECT id,source_key,sender_id,sender_name,sent_at,plain_text,role
                FROM messages WHERE umo=? AND source_key=? AND is_deleted=0""", (self.umo, result["reply_to"])).fetchone()
            if original is not None:
                # A recent-message window may start at the tool result. Keep its
                # actual request and author even when that request is outside it.
                result["reply_to_message"] = dict(original)
        if "message_attachments" in self.tables:
            result["attachments"] = self._rows("SELECT position,attachment_type,extraction_status,descriptor_text,reference_sha256 FROM message_attachments WHERE message_id=?", (row["id"],))
        return text_view(result)

    @_serialized
    def messages(self, ids: Iterable[int]) -> list[dict]:
        """Open retained source addresses in this group, in the requested order."""
        ids = list(dict.fromkeys(int(value) for value in ids))
        if not ids:
            return []
        rows = self._rows(f"SELECT * FROM messages WHERE umo=? AND is_deleted=0 AND id IN ({','.join('?' for _ in ids)})",
                          (self.umo, *ids))
        by_id = {row["id"]: row for row in rows}
        return [self._message(by_id[value]) for value in ids if value in by_id]

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
                        *, sender_id: str | None = None, related_account_id: str | None = None,
                        roles: list[str] | None = None) -> list[dict]:
        clauses, args = ["m.umo=?", "m.is_deleted=0"], [self.umo]
        if roles is not None:
            if not roles:
                return []
            clauses.append(f"m.role IN ({','.join('?' for _ in roles)})")
            args.extend(roles)
        words = _terms(terms)
        if words:
            # Check raw text first; only candidate hits need media decoding removed.
            match = "((instr(lower(m.plain_text),lower(?))>0 AND instr(lower(mr_searchable_text(m.plain_text)),lower(?))>0) OR (instr(lower(m.content_json),lower(?))>0 AND instr(lower(mr_searchable_text(m.content_json)),lower(?))>0))"
            clauses.append("(" + " OR ".join(match for _ in words) + ")")
            args.extend(word for word in words for _ in range(4))
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
    def search_message_context(self, *, before: int = 0, after: int = 1, **query) -> dict:
        """Return literal hits with their adjacent exchange, without classifying it.

        Author/role filters select hits only. A reply or correction may be from
        somebody else and need not quote the hit. Overlapping windows share one
        chronological copy of each original, including generated/sent stages.
        """
        before, after = max(0, min(40, int(before))), max(0, min(40, int(after)))
        matches = self.search_messages(**query)
        end_at = query.get("end_at")
        before_time = int(end_at) - 1 if end_at is not None else None
        originals = {}
        for hit in matches:
            for row in self.context(message_id=hit["id"], before=before, after=after,
                                    before_time=before_time):
                originals[row["id"]] = row
        return {"matches": [row["id"] for row in matches],
                "context": sorted(originals.values(), key=lambda row: (row["sent_at"], row["id"])),
                "window": {"before": before, "after": after, "end_at": end_at}}

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

    @_serialized
    def search_memories(self, kind="all", terms=(), participant_id=None, limit=12, *,
                        related_account_id=None, include_sources=False):
        return self.memory_graph.search(kind, terms, participant_id, limit,
            related_account_id=related_account_id, include_sources=include_sources)

    @_serialized
    def memory(self, kind, id, *, include_sources=True, include_history=False):
        kind = canonical(kind)
        if kind == "participant":
            row = self.db.execute("SELECT account_id FROM participants WHERE umo=? AND id=?", (self.umo, int(id))).fetchone()
            people = self.members(account_ids=[row[0]]) if row else []
            return {"kind": kind, **people[0], "title": people[0]["name"],
                    "summary": " / ".join(people[0]["aliases"]), "source_keys": [], "source_ids": []} if people else None
        if kind == "cue":
            return {"kind": kind, "id": str(id), "title": str(id),
                    "navigation": self.memory_graph.navigate(cue=str(id)), "source_ids": [], "source_keys": []}
        return self.memory_graph.get(kind, id, include_sources=include_sources, include_history=include_history)

    def _clear_memory_derivatives(self, kind, owner):
        self.db.execute("DELETE FROM memory_embeddings WHERE umo=? AND owner_type=? AND owner_key=?", (self.umo, canonical(kind), str(owner)))
        self.db.execute("DELETE FROM mr_index_pending WHERE umo=? AND owner_type=? AND owner_key=?", (self.umo, canonical(kind), str(owner)))

    @_serialized
    def graph(self, node_id=None, terms=(), limit=12, *, ref=None, related_account_id=None, include_sources=False, offset=0):
        return self.memory_graph.graph(node_id, terms, limit, ref=ref, related_account_id=related_account_id,
                                       include_sources=include_sources, offset=offset)

    @_serialized
    def navigate(self, **query):
        return self.memory_graph.navigate(**query)

    @_serialized
    def reconsider(self, limit=8, offset=0, ref=None, after=0, kind=None):
        return self.cognition.reconsider(limit, offset, ref, after, kind)

    @_serialized
    def workspace(self):
        return self.cognition.workspace()

    @_serialized
    def record_cognition(self, **experience):
        with self.db:
            self.cognition.record(**experience)

    @_serialized
    def memory_changes(self, after=0, limit=20):
        return self.memory_graph.changes(after, limit)

    @_serialized
    def memory_directory(self, refs):
        return [self.memory_graph.brief(ref) for ref in refs]

    def _memory_history(self, kind: str, owner: int) -> list[dict]:
        rows = self._rows("SELECT id,operation,reason,run_id,snapshot_json,source_ids_json,created_at FROM mr_memory_revisions WHERE umo=? AND kind=? AND owner_id=? ORDER BY id",
                          (self.umo, kind, owner))
        for revision in rows:
            revision["snapshot"] = json.loads(revision.pop("snapshot_json"))
            revision["source_ids"] = json.loads(revision.pop("source_ids_json"))
        return rows

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

    @_serialized
    def feedback_messages(self, window_seconds: int, after: int, limit: int = 500, *, newest: bool = True) -> list[dict]:
        """Select a recent or historical conversation without skipping other work.

        `after` is the old FIFO cursor, retained only as a legacy floor. New
        progress is recorded per message, so completing a recent batch cannot
        erase the older queue. Relevance within this conversation is for the model.
        """
        pending = """m.umo=? AND m.is_deleted=0 AND m.id>?
            AND NOT EXISTS(SELECT 1 FROM mr_feedback_processed p WHERE p.umo=m.umo AND p.message_id=m.id)"""
        reaction = """m.role='USER' AND EXISTS(SELECT 1 FROM messages b WHERE b.umo=m.umo AND b.is_deleted=0
            AND b.role='BOT' AND b.sent_at>=m.sent_at-? AND (b.sent_at,b.id)<(m.sent_at,m.id))"""
        direction = "DESC" if newest else "ASC"
        first = self.db.execute(f"""SELECT m.* FROM messages m WHERE {pending} AND {reaction}
            ORDER BY m.sent_at {direction},m.id {direction} LIMIT 1""",
            (self.umo, int(after), int(window_seconds))).fetchone()
        if first is None:
            return []
        if newest:
            # Take a whole recent segment ending at the newest reaction, not
            # just that one line. A late historical import sorts by event time.
            candidates = self._rows(f"""SELECT m.*,({reaction}) AS is_reaction FROM messages m
                WHERE {pending} AND (m.sent_at,m.id)<=(?,?)
                ORDER BY m.sent_at DESC,m.id DESC LIMIT ?""",
                (int(window_seconds), self.umo, int(after), first["sent_at"], first["id"], _limit(limit)))
            candidates.reverse()
            start = next(index for index, row in enumerate(candidates) if row["is_reaction"])
            material_rows = candidates[start:]
        else:
            material_rows = self._rows(f"""SELECT m.*,({reaction}) AS is_reaction FROM messages m
                WHERE {pending} AND (m.sent_at,m.id)>=(?,?) ORDER BY m.sent_at,m.id LIMIT ?""",
                (int(window_seconds), self.umo, int(after), first["sent_at"], first["id"], _limit(limit)))
        while True:
            first = next(row for row in material_rows if row["is_reaction"])
            prefix = self._feedback_prefix(first, int(window_seconds))[-max(1, _limit(limit) - 1):]
            material_ids = {row["id"] for row in material_rows}
            prefix = [row for row in prefix if row["id"] not in material_ids]
            room = max(1, _limit(limit) - len(prefix))
            if len(material_rows) <= room:
                break
            material_rows = material_rows[-room:] if newest else material_rows[:room]
            # Trimming can move the first reaction into another interaction.
            # Rebuild its original bot/question prefix from that actual anchor.
        material = [self._message(row) for row in material_rows]
        # The window describes a reaction's relation to a bot response. Waiting
        # until tonight must not expire already observed daytime feedback.
        return [{**row, "context_only": True} for row in prefix] + [
            {**row, "context_only": False} for row in material]

    def _feedback_prefix(self, first: dict, window_seconds: int) -> list[dict]:
        bot = self.db.execute("""SELECT * FROM messages WHERE umo=? AND is_deleted=0 AND role='BOT'
            AND sent_at>=? AND (sent_at,id)<(?,?) ORDER BY sent_at DESC,id DESC LIMIT 1""",
            (self.umo, first["sent_at"] - window_seconds, first["sent_at"], first["id"])).fetchone()
        bot_message = self._message(dict(bot))
        if not bot_message["reply_to"]:
            return [bot_message]
        rows = self._rows("""SELECT m.* FROM messages m WHERE m.umo=? AND m.is_deleted=0
            AND (m.sent_at,m.id)<=(?,?) AND (m.source_key=? OR EXISTS(
                SELECT 1 FROM message_relations r WHERE r.source_message_id=m.id AND r.umo=m.umo
                AND r.target_source_key=? AND r.relation IN('REPLY_TO','RESPONDS_TO')))
            ORDER BY m.sent_at,m.id""", (self.umo, bot["sent_at"], bot["id"],
                                         bot_message["reply_to"], bot_message["reply_to"]))
        return [self._message(row) for row in rows]

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
    def update_working_state(self, patch: dict) -> dict:
        """Merge a foreground or worker update without overwriting another writer's fields."""
        state = self.load_working_state()
        state.update(patch)
        with self.db:
            self.db.execute("""INSERT INTO mr_working_state VALUES(?,?,?) ON CONFLICT(umo)
                DO UPDATE SET state_json=excluded.state_json,updated_at=excluded.updated_at""",
                (self.umo, _encode(state), int(time.time())))
        return state

    @_serialized
    def recall_drafts(self, pending=None, known=()) -> dict:
        """Merge this request's drafts without replacing concurrent requests."""
        state = self.load_working_state()
        drafts = state.get("pending_recall_writes", {})
        if pending is not None:
            for key in known:
                drafts.pop(key, None)
            drafts.update(pending)
        drafts = {key: value for key, value in drafts.items() if self.write_receipt(key) is None}
        if pending is not None or drafts != state.get("pending_recall_writes", {}):
            self.update_working_state({"pending_recall_writes": drafts})
        return drafts

    @_serialized
    def write_receipt(self, key):
        row = self.db.execute("SELECT refs_json FROM mr_memory_write_receipts WHERE umo=? AND receipt_key=?",
                              (self.umo, key)).fetchone()
        return json.loads(row[0]) if row else None

    @_serialized
    def learning_task(self, kind: str) -> dict | None:
        row = self.db.execute("SELECT task_json FROM mr_learning_tasks WHERE umo=? AND kind=?",
                              (self.umo, self._usage_kind(kind))).fetchone()
        return json.loads(row[0]) if row else None

    @_serialized
    def resume_learning_task(self, kind: str) -> dict | None:
        """Apply explicit progress retained by the old all-drafts barrier."""
        kind = self._usage_kind(kind)
        task = self.learning_task(kind)
        writes = (task or {}).get("continuation", {}).get("write_state", {})
        progress = writes.get("deferred_progress")
        if task is not None and progress:
            try:
                with self.db:
                    task = self._save_learning_progress(kind, progress, [], task.get("run_id"))
                    task["continuation"]["write_state"]["deferred_progress"] = {}
                    self._write_learning_task(kind, task)
            except (TypeError, ValueError) as exc:
                # An invalid saved proposal needs model correction, just like
                # a rejected live write. Do not crash before that model can run.
                task = self.learning_task(kind)
                errors = task["continuation"].setdefault("pending_write_errors", {})
                detail = f"Learning progress has not been saved: {type(exc).__name__}: {exc}"
                if errors.get("remember") != detail:
                    errors["remember"] = detail
                    with self.db:
                        self._write_learning_task(kind, task)
        return task

    @_serialized
    def unfinished_learning(self, kind: str) -> dict | None:
        """Locate real writes from older unfinished runs without inventing their progress."""
        kind = self._usage_kind(kind)
        latest = self.db.execute("""SELECT id,json_extract(payload_json,'$.status') AS status,
            substr(json_extract(payload_json,'$.detail'),1,512) AS detail FROM mr_runs
            WHERE umo=? AND kind=? ORDER BY id DESC LIMIT 1""", (self.umo, kind)).fetchone()
        if latest is None or latest["status"] == "completed":
            return None
        completed = self.db.execute("""SELECT COALESCE(max(id),0) FROM mr_runs WHERE umo=? AND kind=?
            AND json_extract(payload_json,'$.status')='completed'""", (self.umo, kind)).fetchone()[0]
        run_ids = [row[0] for row in self.db.execute("""SELECT id FROM mr_runs
            WHERE umo=? AND kind=? AND id>? ORDER BY id""", (self.umo, kind, completed))]
        refs = self._rows("""SELECT DISTINCT json_extract(w.value,'$.kind') AS kind,
            json_extract(w.value,'$.id') AS id FROM mr_runs r,json_each(r.payload_json,'$.written') w
            WHERE r.umo=? AND r.kind=? AND r.id>? AND json_type(w.value,'$.id')='integer'
            AND json_type(w.value,'$.kind')='text' ORDER BY kind,id""", (self.umo, kind, completed))
        return {"run_ids": run_ids, "memory_refs": refs, "last_status": latest["status"],
                "detail": latest["detail"] or "", "interpretation": "这些运行未完成，实际写入不等于整批材料已处理；按需回看运行和记忆继续。"}

    def _write_learning_task(self, kind: str, task: dict) -> None:
        task["updated_at"] = int(time.time())
        self.db.execute("""INSERT INTO mr_learning_tasks(umo,kind,task_json,updated_at) VALUES(?,?,?,?)
            ON CONFLICT(umo,kind) DO UPDATE SET task_json=excluded.task_json,updated_at=excluded.updated_at""",
            (self.umo, kind, _encode(task), task["updated_at"]))

    @_serialized
    def start_learning_task(self, kind: str, material_ids: list[int], context_ids: list[int] | None = None,
                            working: dict | None = None) -> dict:
        kind = self._usage_kind(kind)
        existing = self.learning_task(kind)
        if existing is not None:
            return existing
        material = list(dict.fromkeys(int(value) for value in material_ids))
        context = [value for value in dict.fromkeys(int(value) for value in (context_ids or []))
                   if value not in material]
        ids = context + material
        if ids:
            count = self.db.execute(f"""SELECT count(*) FROM messages WHERE umo=? AND is_deleted=0
                AND id IN ({','.join('?' for _ in ids)})""", (self.umo, *ids)).fetchone()[0]
            if count != len(ids):
                raise ValueError("Learning material must be retained messages in this group")
        task = {"kind": kind, "material_ids": material, "context_ids": context, "completed_ids": [],
                "checkpoint": "", "memory_refs": [], "working": working or {}, "created_at": int(time.time()),
                "updated_at": int(time.time()), "run_id": None, "continuation": {}}
        with self.db:
            state = self.load_working_state()
            drafts = state.get("pending_learning_drafts", {}).pop(kind, {})
            if drafts:
                task["continuation"]["write_state"] = {"pending_items": drafts}
                self.db.execute("""INSERT INTO mr_working_state VALUES(?,?,?) ON CONFLICT(umo)
                    DO UPDATE SET state_json=excluded.state_json,updated_at=excluded.updated_at""",
                    (self.umo, _encode(state), int(time.time())))
            self._write_learning_task(kind, task)
        return task

    @_serialized
    def update_learning_task(self, kind: str, patch: dict) -> dict:
        kind = self._usage_kind(kind)
        task = self.learning_task(kind)
        if task is None:
            raise ValueError("Learning task does not exist")
        task.update(patch)
        with self.db:
            self._write_learning_task(kind, task)
        return task

    @_serialized
    def finish_learning_task(self, kind: str) -> None:
        kind = self._usage_kind(kind)
        with self.db:
            task = self.learning_task(kind)
            state = self.load_working_state()
            drafts = (task or {}).get("continuation", {}).get("write_state", {}).get("pending_items", {})
            for row in drafts.values():
                row.setdefault("origin_run_id", task.get("run_id"))
                row.setdefault("saved_memory_refs", task.get("memory_refs", []))
            if drafts:
                state.setdefault("pending_learning_drafts", {})[kind] = drafts
            if (kind == "feedback" and task and task["material_ids"]
                    and set(task["material_ids"]) <= set(task["completed_ids"])):
                order = task.get("working", {}).get("feedback_order", "oldest")
                state["feedback_next_order"] = "oldest" if order == "recent" else "recent"
            self.db.execute("""INSERT INTO mr_working_state VALUES(?,?,?) ON CONFLICT(umo)
                DO UPDATE SET state_json=excluded.state_json,updated_at=excluded.updated_at""",
                (self.umo, _encode(state), int(time.time())))
            self.db.execute("DELETE FROM mr_learning_tasks WHERE umo=? AND kind=?",
                            (self.umo, self._usage_kind(kind)))

    @_serialized
    def learning_messages(self, task: dict) -> list[dict]:
        context = set(task.get("context_ids", []))
        ids = list(dict.fromkeys([*task.get("context_ids", []), *task.get("material_ids", [])]))
        if not ids:
            return []
        rows = self._rows(f"""SELECT * FROM messages WHERE umo=? AND is_deleted=0
            AND id IN ({','.join('?' for _ in ids)})""", (self.umo, *ids))
        by_id = {row["id"]: row for row in rows}
        return [{**self._message(by_id[value]), "context_only": value in context}
                for value in ids if value in by_id]

    def _save_learning_progress(self, kind: str, progress: dict | None, memory_refs: list[tuple[str, int]],
                                run_id: int | None) -> dict:
        """Apply the model's explicit progress inside the caller's write transaction."""
        task = self.learning_task(kind)
        if task is None:
            raise ValueError("Learning task does not exist")
        progress = progress or {}
        completed = set(int(value) for value in progress.get("completed_ids", []))
        material = task["material_ids"]
        if not completed.issubset(material):
            raise ValueError("Completed message ids must belong to this task's material, not its context or evidence")
        newly_completed = completed - set(task["completed_ids"])
        completed.update(task["completed_ids"])
        task["completed_ids"] = [value for value in material if value in completed]
        if "checkpoint" in progress:
            task["checkpoint"] = progress["checkpoint"]
        refs = {(ref["kind"], int(ref["id"])) for ref in task["memory_refs"]}
        for ref in memory_refs:
            if ref not in refs:
                task["memory_refs"].append({"kind": ref[0], "id": ref[1]})
                refs.add(ref)
        if run_id is not None:
            task["run_id"] = int(run_id)
        if kind == "background":
            for value in newly_completed:
                self.db.execute("""INSERT INTO message_processing(message_id,content_sha256,status,distilled_at)
                    SELECT id,content_sha256,'DISTILLED',? FROM messages WHERE umo=? AND id=? AND is_deleted=0
                    ON CONFLICT(message_id) DO UPDATE SET status='DISTILLED',distilled_at=excluded.distilled_at,
                    content_sha256=excluded.content_sha256,last_error=''""", (int(time.time()), self.umo, value))
        elif material:
            self.db.executemany("INSERT OR IGNORE INTO mr_feedback_processed VALUES(?,?,?)",
                [(self.umo, value, int(time.time())) for value in newly_completed])
        self._write_learning_task(kind, task)
        return task

    @_serialized
    def save_learning_progress(self, kind: str, progress: dict, run_id: int | None = None) -> dict:
        with self.db:
            return self._save_learning_progress(self._usage_kind(kind), progress, [], run_id)

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

    def _queue_embedding(self, kind, owner):
        kind = canonical(kind)
        entry = self.memory_graph.get(kind, owner, include_sources=False, attention=False)
        if not entry:
            return
        text = "\n".join(str(entry.get(k) or "") for k in ("title", "summary", "source", "relation", "target")).strip()
        text += "\n" + " / ".join(entry.get("aliases", []))
        if entry.get("representation"):
            text += "\n" + _encode(entry["representation"])
        text += "\n" + "\n".join(cue["cue"] + " / " + cue["aspect"] for cue in entry.get("cues", []))
        if kind == "association":
            text += "\n" + "\n".join(
                str(entry.get(side + "_node", {}).get("description") or "") + " / ".join(entry.get(side + "_node", {}).get("aliases", []))
                for side in ("source", "target"))
        if text.strip():
            self.db.execute("""INSERT INTO mr_index_pending(umo,owner_type,owner_key,text,updated_at) VALUES(?,?,?,?,?)
                ON CONFLICT(umo,owner_type,owner_key) DO UPDATE SET text=excluded.text,updated_at=excluded.updated_at""",
                (self.umo, kind, str(owner), text, int(time.time())))

    @_serialized
    def pending_embeddings(self, model_id, limit=16):
        with self.db:
            rows = self._rows("""SELECT m.kind,m.id FROM mr_memory_objects m WHERE m.umo=? AND m.status='ACTIVE'
                AND NOT EXISTS(SELECT 1 FROM memory_embeddings v WHERE v.umo=m.umo AND v.owner_type=m.kind AND v.owner_key=CAST(m.id AS TEXT) AND v.model=?)
                AND NOT EXISTS(SELECT 1 FROM mr_index_pending q WHERE q.umo=m.umo AND q.owner_type=m.kind AND q.owner_key=CAST(m.id AS TEXT))
                ORDER BY m.updated_at DESC,m.kind,m.id LIMIT ?""", (self.umo, model_id, _limit(limit)))
            for row in rows:
                self._queue_embedding(row["kind"], row["id"])
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
    def vector_rows(self, model_id):
        rows = self._rows("""SELECT v.model,v.owner_type,v.owner_key,v.dimensions,v.vector
            FROM memory_embeddings v LEFT JOIN mr_memory_objects m
            ON m.umo=v.umo AND m.kind=v.owner_type AND CAST(m.id AS TEXT)=v.owner_key
            WHERE v.umo=? AND v.model=? AND (m.status='ACTIVE' OR v.owner_type='cue' OR
                (v.owner_type='participant' AND EXISTS(SELECT 1 FROM participants p WHERE p.umo=v.umo
                    AND CAST(p.id AS TEXT)=v.owner_key AND p.current_display_name!='')))""", (self.umo, model_id))
        for row in rows:
            row["embedding_blob"] = row["vector"]
        return rows

    @_serialized
    def put_vector(self, owner_type: str, owner_key: str, model_id: str, vector: bytes, dimensions: int) -> None:
        if int(dimensions) <= 0 or len(vector) != int(dimensions) * 4:
            raise ValueError("Vector must be float32 bytes with matching dimensions")
        owner_type = canonical(owner_type)
        owner = (self.memory(owner_type, owner_key, include_sources=False) if owner_type in {"participant", "cue"}
                 else self.memory_graph.raw(owner_type, owner_key))
        if owner is None or owner.get("status", "ACTIVE") != "ACTIVE":
            raise ValueError("Vector owner is not in this group")
        with self.db:
            self.db.execute("""INSERT INTO memory_embeddings(umo,owner_type,owner_key,model,dimensions,vector)
                VALUES(?,?,?,?,?,?) ON CONFLICT(umo,owner_type,owner_key,model) DO UPDATE SET
                dimensions=excluded.dimensions,vector=excluded.vector,updated_at=CURRENT_TIMESTAMP""", (self.umo, owner_type, str(owner_key), model_id, int(dimensions), vector))

    @_serialized
    def save_memories(self, items, sources=(), mark_processed=True, *, run_id=None,
                      learning_kind=None, progress=None, learning_write=None, receipt_key=None):
        if receipt_key is not None:
            receipt = self.write_receipt(receipt_key)
            if receipt is not None:
                return receipt
        keys = list(dict.fromkeys(str(key) for key in sources))
        source_rows = {r["source_key"]: r for r in self._messages_for_sources(keys)}
        if set(keys) != set(source_rows):
            raise ValueError("Memory sources must be retained messages in this group")
        outputs = []
        with self.db:
            for item in items:
                value = dict(item)
                if "source_keys" in value:
                    selected = list(dict.fromkeys(value.pop("source_keys")))
                    if not set(selected) <= set(source_rows):
                        raise ValueError("A memory refers to an unavailable source key")
                    value["source_ids"] = [source_rows[key]["id"] for key in selected]
                saved = self.memory_graph.write(value, run_id=run_id)
                outputs.append((saved["kind"], saved["id"]))
            if receipt_key is not None:
                self.db.execute("INSERT INTO mr_memory_write_receipts VALUES(?,?,?)",
                    (self.umo, receipt_key, _encode([{"kind": kind, "id": owner} for kind, owner in outputs])))
            if learning_kind is not None:
                self._save_learning_progress(learning_kind, progress, outputs, run_id)
                if learning_write is not None:
                    task = self.learning_task(learning_kind)
                    state = dict(learning_write["state"])
                    receipts = dict(state.get("receipts", {}))
                    receipts[learning_write["pending_id"]] = [{"kind": kind, "id": owner} for kind, owner in outputs]
                    state["receipts"] = receipts
                    task["continuation"] = {**task.get("continuation", {}), "write_state": state}
                    self._write_learning_task(learning_kind, task)
            for key in keys if mark_processed and learning_kind is None else ():
                message = source_rows[key]
                raw = self.db.execute("SELECT content_sha256 FROM messages WHERE id=?", (message["id"],)).fetchone()
                self.db.execute("""INSERT INTO message_processing(message_id,content_sha256,status,distilled_at)
                    VALUES(?,?,'DISTILLED',?) ON CONFLICT(message_id) DO UPDATE SET status='DISTILLED',
                    distilled_at=excluded.distilled_at,content_sha256=excluded.content_sha256,last_error=''""", (message["id"], raw[0], int(time.time())))
        return [self.memory(kind, owner) or {"kind": kind, "id": owner,
                "status": self.memory_graph.raw(kind, owner)["status"]} for kind, owner in outputs]
