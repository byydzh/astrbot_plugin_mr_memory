"""Visible memory references and compatibility with older working notes."""
from __future__ import annotations

import json

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
                    "content": note})
            self.store.save_working_state(state)

    def archived_workspace(self):
        rows = self.store._rows("""SELECT * FROM mr_working_set
            WHERE umo=? ORDER BY updated_at,kind,owner_id""",
            (self.store.umo,))
        active = []
        for row in rows:
            memory = self.store.memory_graph.get(row["kind"], row["owner_id"], include_sources=False)
            if memory is None:
                continue
            memory["connections"] = self.store.memory_graph.connections({"kind": row["kind"], "id": row["owner_id"]})
            active.append({"memory": memory, "characters": len(encode(memory))})
        catalog = self.store._rows("SELECT kind,count(*) count FROM mr_memory_objects WHERE umo=? AND status='ACTIVE' GROUP BY kind ORDER BY kind", (self.store.umo,))
        return {"active": active, "catalog": catalog,
            "characters": sum(row["characters"] for row in active),
            "meaning": "此前选中的记忆，按当前仍可读取的版本展示。"}

    def archived_experiences(self, ref, limit=5):
        rows = self.store._rows("""SELECT * FROM mr_cognition_turns WHERE umo=? AND EXISTS (
            SELECT 1 FROM json_each(payload_json,'$.available_memories') m
            WHERE json_extract(m.value,'$.kind')=? AND json_extract(m.value,'$.id')=?)
            ORDER BY id DESC LIMIT ?""", (self.store.umo, ref["kind"], ref["id"], limit + 1))
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
                "next_offset": limit,
                "meaning": "available_memories保留当时可见的版本与把握，current_memories为现在的版本。recollection是模型自述，不能证明某记忆造成回答；actual_response是实际回复，subsequent_context是邻接交流，不自动算赞同或反对。可用 interaction 或 context 继续回看原始材料。"}
