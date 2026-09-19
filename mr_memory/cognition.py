"""Model-selected working memories and the experience of using them.

Selection is an index into the revisable graph, never a second copy of its
meaning. The model supplies representations and decides what to keep in view.
"""
from __future__ import annotations

import json
import time

from .memory_graph import encode, reference


SCHEMA = """
CREATE TABLE IF NOT EXISTS mr_working_set (
 umo TEXT NOT NULL,kind TEXT NOT NULL,owner_id INTEGER NOT NULL,
 state_json TEXT NOT NULL,updated_at REAL NOT NULL,
 PRIMARY KEY(umo,kind,owner_id));
CREATE TABLE IF NOT EXISTS mr_cognition_turns (
 id INTEGER PRIMARY KEY AUTOINCREMENT,umo TEXT NOT NULL,run_key TEXT NOT NULL,
 run_id INTEGER,kind TEXT NOT NULL,at REAL NOT NULL,request_id TEXT,
 observation_id INTEGER,payload_json TEXT NOT NULL,UNIQUE(umo,run_key));
CREATE INDEX IF NOT EXISTS mr_cognition_recent ON mr_cognition_turns(umo,kind,id);
"""


def visible_memories(value):
    """Locate actual memory bodies in tool/input data, not guessed prose IDs."""
    found = {}

    def visit(part):
        if isinstance(part, list):
            for child in part:
                visit(child)
        elif isinstance(part, dict):
            if (type(part.get("revision_no")) is int and type(part.get("id")) is int
                    and isinstance(part.get("kind"), str)
                    and ("content" in part or "summary" in part)):
                ref = reference(part)
                found[(ref["kind"], ref["id"], part["revision_no"])] = {
                    **ref, "revision_no": part["revision_no"], "title": part.get("title", ""),
                    "belief": part.get("belief", {"stance": "unconfirmed"})}
            for key, child in part.items():
                if key in {"representation", "recollection", "attention", "belief", "pending_items", "pending_recall_writes", "rejected"}:
                    continue
                if isinstance(child, (list, dict)):
                    visit(child)

    visit(value)
    return list(found.values())


class Cognition:
    def __init__(self, store):
        self.store, self.db = store, store.db
        self.db.executescript(SCHEMA)

    def migrate_notes(self):
        """Move the old unconnected working prose into the shared graph once."""
        state = self.store.load_working_state()
        if "working_memory" not in state:
            return
        note = state.pop("working_memory")
        with self.db:
            if isinstance(note, str) and note.strip():
                self.store.memory_graph.write({"kind": "understanding", "title": "此前延续的理解",
                    "content": note, "attention": {"reason": "从旧工作笔记延续，结合当前经历继续理解"}})
            self.store.save_working_state(state)

    def select(self, ref, attention):
        ref = reference(ref)
        if attention is None or attention is False:
            self.db.execute("DELETE FROM mr_working_set WHERE umo=? AND kind=? AND owner_id=?",
                            (self.store.umo, ref["kind"], ref["id"]))
            return
        if attention is True:
            attention = {}
        self.db.execute("""INSERT INTO mr_working_set VALUES(?,?,?,?,?)
            ON CONFLICT(umo,kind,owner_id) DO UPDATE SET state_json=excluded.state_json,updated_at=excluded.updated_at""",
            (self.store.umo, ref["kind"], ref["id"], encode(attention), time.time()))

    def workspace(self):
        rows = self.store._rows("""SELECT * FROM mr_working_set
            WHERE umo=? ORDER BY updated_at,kind,owner_id""",
            (self.store.umo,))
        active = []
        for row in rows:
            memory = self.store.memory_graph.get(row["kind"], row["owner_id"], include_sources=False)
            if memory is None:
                continue
            memory["connections"] = self.store.memory_graph.connections({"kind": row["kind"], "id": row["owner_id"]})
            active.append({"memory": memory})
        catalog = self.store._rows("SELECT kind,count(*) count FROM mr_memory_objects WHERE umo=? AND status='ACTIVE' GROUP BY kind ORDER BY kind", (self.store.umo,))
        return {"active": active, "catalog": catalog,
            "meaning": "这些是你选择继续用于理解的认识，正文按图中当前版本读取。可用remember修改其内容、表示、联系及attention；attention=null只移出当前注意，记忆仍可搜索。"}

    def record(self, *, run_key, run_id, kind, current, payload):
        request_id = current.get("message_id")
        if not request_id and current.get("id"):
            original = self.db.execute("SELECT message_id FROM messages WHERE umo=? AND id=?", (self.store.umo, current["id"])).fetchone()
            request_id = original[0] if original else None
        self.db.execute("""INSERT INTO mr_cognition_turns(umo,run_key,run_id,kind,at,
            request_id,observation_id,payload_json) VALUES(?,?,?,?,?,?,?,?)
            ON CONFLICT(umo,run_key) DO UPDATE SET payload_json=excluded.payload_json""",
            (self.store.umo, run_key, run_id, kind, time.time(),
             str(request_id or "") or None, current.get("id"), encode(payload)))

    def reconsider(self, limit=8, offset=0, ref=None, after=0, kind=None):
        limit, offset = max(1, min(100, int(limit))), max(0, int(offset))
        clauses, args = ["umo=?", "id>?"], [self.store.umo, int(after)]
        if kind:
            clauses.append("kind=?")
            args.append(kind)
        if ref:
            ref = reference(ref)
            clauses.append("""EXISTS(SELECT 1 FROM json_each(payload_json,'$.available_memories') m
                WHERE json_extract(m.value,'$.kind')=? AND json_extract(m.value,'$.id')=?)""")
            args.extend([ref["kind"], ref["id"]])
        rows = self.store._rows("SELECT * FROM mr_cognition_turns WHERE " + " AND ".join(clauses)
                               + " ORDER BY id DESC LIMIT ? OFFSET ?", [*args, limit + 1, offset])
        experiences = []
        current_refs = {}
        for row in rows[:limit]:
            payload = json.loads(row["payload_json"])
            experience = {key: row[key] for key in ("id", "run_id", "kind", "at", "request_id", "observation_id")}
            experience.update(payload)
            if row["request_id"]:
                events, more = self.store.reflections._events(row["request_id"])
                experience["actual_response"] = [self.store.reflections._event_summary(event) for event in events]
                experience["more_response_events"] = more
                original = self.db.execute("SELECT id FROM messages WHERE umo=? AND message_id=? ORDER BY id LIMIT 1",
                                           (self.store.umo, row["request_id"])).fetchone()
                experience["subsequent_context"] = self.store.reflections.following_context(
                    events, {"id": original[0]} if original else None)
            for ref in payload.get("available_memories", []):
                current_refs[(ref["kind"], ref["id"])] = reference(ref)
            experiences.append(experience)
        cursor = self.db.execute("SELECT COALESCE(max(id),0) FROM mr_cognition_turns WHERE umo=? AND kind='foreground'",
                                 (self.store.umo,)).fetchone()[0]
        graph = self.store.memory_graph
        return {"items": experiences,
                "current_memories": [{**graph.brief(ref), "basis": graph.connections(ref, purpose="basis")}
                                     for ref in current_refs.values()],
                "cursor": cursor, "more": len(rows) > limit,
                "next_offset": offset + limit,
                "meaning": "available_memories保留当时可见的版本与把握，current_memories为现在的版本。recollection是模型自述，不能证明某记忆造成回答；actual_response是实际回复，subsequent_context是邻接交流，不自动算赞同或反对。理解其对象与含义后，可改变原认识、连接和把握；用interaction或context继续展开。"}
