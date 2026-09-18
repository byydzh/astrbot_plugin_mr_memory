"""One-time import of legacy memory tables. Runtime readers never query them."""
from __future__ import annotations

import json
from datetime import datetime, timezone


def migrate(graph):
    db, umo = graph.db, graph.umo
    if db.execute("SELECT 1 FROM mr_graph_migrations WHERE umo=?", (umo,)).fetchone():
        return
    # The previous feedback handoff copied complete memories into working state.
    # Keep its useful recent focus as addresses, resolved against current content.
    working = db.execute("SELECT state_json FROM mr_working_state WHERE umo=?", (umo,)).fetchone()
    if working:
        state = json.loads(working[0])
        if isinstance(state.get("learned_feedback"), list):
            state["recent_learning_refs"] = [{"kind": item["kind"], "id": item["id"]}
                for item in state.pop("learned_feedback") if isinstance(item, dict) and "kind" in item and "id" in item]
            db.execute("UPDATE mr_working_state SET state_json=? WHERE umo=?", (json.dumps(state, ensure_ascii=False), umo))
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    def rows(table, condition="", args=()):
        if table not in tables:
            return []
        return [dict(row) for row in db.execute(f"SELECT * FROM {table} {condition}", args)]

    def sources(table, column, owner):
        return [row["message_id"] for row in rows(table, f"WHERE {column}=?", (owner,))]

    def insert(kind, row, title, content, attrs, cues=()):
        status = row.get("status", "ACTIVE")
        original_status = status
        if kind == "episode" and status not in {"INVALIDATED", "SUPERSEDED", "RETRACTED"}:
            status = "ACTIVE"
        status = "ACTIVE" if status in {"OPEN", "CLOSED", "ACTIVE", "WEAKENED"} else status
        if row.get("invalidation_reason"):
            status = "INVALIDATED"
        revision = 1 + db.execute("SELECT count(*) FROM mr_memory_revisions WHERE umo=? AND kind=? AND owner_id=?",
                                   (umo, kind, row["id"])).fetchone()[0]
        def timestamp(value):
            if isinstance(value, (int, float)):
                return value
            try:
                parsed = datetime.fromisoformat(str(value))
                return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).timestamp()
            except (TypeError, ValueError):
                return 0
        created = timestamp(row.get("created_at")) or timestamp(row.get("started_at"))
        updated = timestamp(row.get("updated_at")) or created
        # Preserve the original timestamps without pretending the import is a new memory event.
        attrs["legacy_created_at"] = row.get("created_at")
        attrs["legacy_updated_at"] = row.get("updated_at")
        attrs["legacy_status"] = original_status
        db.execute("INSERT INTO mr_memory_objects VALUES(?,?,?,?,?,?,?,?,?,?,0)",
                   (umo, kind, row["id"], title or "", content or "", json.dumps(attrs, ensure_ascii=False), revision, status, created, updated))
        for cue, aspect in cues:
            if cue:
                db.execute("INSERT OR IGNORE INTO mr_memory_cues VALUES(?,?,?,?,?)", (umo, kind, row["id"], str(cue), str(aspect or "")))

    for row in rows("episodes", "WHERE umo=?", (umo,)):
        attrs = {"source_ids": sources("episode_messages", "episode_id", row["id"]),
                 "started_at": row.get("started_at"), "ended_at": row.get("ended_at")}
        cues = [(r["cue"], r["tag"]) for r in rows("episode_keywords", "WHERE episode_id=?", (row["id"],))]
        insert("episode", row, row.get("title"), row.get("summary"), attrs, cues)
    for row in rows("semantic_memories", "WHERE umo=?", (umo,)):
        ids = sources("semantic_memory_sources", "semantic_memory_id", row["id"])
        if row.get("source_message_id") is not None:
            ids = list(dict.fromkeys([row["source_message_id"], *ids]))
        person = db.execute("SELECT account_id FROM participants WHERE umo=? AND id=?", (umo, row.get("subject_participant_id"))).fetchone()
        attrs = {"source_ids": ids, "participant_id": row.get("subject_participant_id"),
                 "person": row.get("person_cue", ""), "aspect": row.get("aspect_tag", ""),
                 "subject": {"name": row.get("subject_text") or row.get("person_cue", ""), "account_id": person[0] if person else None},
                 "epistemic_status": row.get("epistemic_status", "")}
        insert("semantic", row, row.get("aspect_tag"), row.get("content"), attrs,
               [(row.get("person_cue", ""), row.get("aspect_tag", ""))])
    for row in rows("topics", "WHERE umo=?", (umo,)):
        insert("topic", row, row.get("name"), row.get("summary"),
               {"source_ids": sources("mr_topic_sources", "topic_id", row["id"])}, [(row.get("name", ""), "主题")])
    for row in rows("plastic_nodes", "WHERE umo=?", (umo,)):
        aliases = [r["alias"] for r in rows("mr_node_aliases", "WHERE umo=? AND node_id=?", (umo, row["id"]))]
        insert("node", row, row.get("label"), row.get("description"), {"source_ids": [], "aliases": aliases},
               [(r, "") for r in [row.get("label", ""), *aliases]])
    for row in rows("plastic_edges", "WHERE umo=?", (umo,)):
        relation = next(iter(rows("relation_types", "WHERE umo=? AND id=?", (umo, row["relation_type_id"]))), {})
        attrs = {"source_ids": sources("plastic_edge_evidence", "edge_id", row["id"]),
                 "source_ref": {"kind": "node", "id": row["source_node_id"]},
                 "target_ref": {"kind": "node", "id": row["target_node_id"]},
                 "relation": relation.get("canonical_name", ""), "purpose": "association",
                 "uncertainty": row.get("uncertainty", ""), "epistemic_state": row.get("epistemic_state", "")}
        insert("association", row, attrs["relation"], row.get("statement"), attrs)
    # A theme's constituent episodes remain navigable, including after later edits.
    for row in rows("topic_episodes"):
        if not graph.raw("topic", row["topic_id"]) or not graph.raw("episode", row["episode_id"]):
            continue
        owner = db.execute("SELECT COALESCE(max(id),0)+1 FROM mr_memory_objects WHERE umo=? AND kind='association'", (umo,)).fetchone()[0]
        insert("association", {"id": owner}, "包含经历", "原主题关联的具体经历", {
            "source_ids": [], "source_ref": {"kind": "topic", "id": row["topic_id"]},
            "target_ref": {"kind": "episode", "id": row["episode_id"]}, "relation": "包含经历", "purpose": "basis",
            "target_revision": graph.raw("episode", row["episode_id"])["revision"]})
    # Existing embedding bytes still describe the same content. Only normalize addresses.
    db.execute("""INSERT OR IGNORE INTO memory_embeddings(umo,owner_type,owner_key,model,dimensions,vector,updated_at)
        SELECT umo,'association',owner_key,model,dimensions,vector,updated_at FROM memory_embeddings
        WHERE umo=? AND owner_type='plastic_edge'""", (umo,))
    db.execute("DELETE FROM memory_embeddings WHERE umo=? AND owner_type='plastic_edge'", (umo,))
    db.execute("""INSERT OR IGNORE INTO mr_index_pending SELECT umo,'association',owner_key,text,updated_at
        FROM mr_index_pending WHERE umo=? AND owner_type='plastic_edge'""", (umo,))
    db.execute("DELETE FROM mr_index_pending WHERE umo=? AND owner_type='plastic_edge'", (umo,))
    if "embeddings" in tables:
        for row in rows("embeddings"):
            kind = {"plastic_edge": "association", "episodic": "episode"}.get(row["owner_type"], row["owner_type"])
            if graph.raw(kind, row["owner_id"]):
                db.execute("INSERT OR IGNORE INTO memory_embeddings(umo,owner_type,owner_key,model,dimensions,vector) VALUES(?,?,?,?,?,?)",
                           (umo, kind, str(row["owner_id"]), row["model"], row["dimensions"], row["vector"]))
    migrate_reflections(graph)
    db.execute("INSERT INTO mr_graph_migrations VALUES(?,1)", (umo,))


def migrate_reflections(graph):
    """A continuing thought is a memory; scheduling is a view of its attributes."""
    db, umo = graph.db, graph.umo
    if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='mr_reflections'").fetchone():
        return
    for raw in db.execute("SELECT * FROM mr_reflections WHERE umo=?", (umo,)).fetchall():
        row = dict(raw)
        attrs = {key: row.get(key) for key in ("priority", "request_id", "next_review_at", "last_reviewed_at", "schedule_updated_at")}
        attrs.update(review_status=row["status"], next_review_explicit=row.get("next_review_explicit", 0),
                     source_ids=json.loads(row["source_ids_json"]), memory_refs=json.loads(row["memory_refs_json"]))
        db.execute("INSERT OR IGNORE INTO mr_memory_objects VALUES(?,?,?,?,?,?,1,'ACTIVE',?,?,0)",
            (umo, "reflection", row["id"], row["content"].splitlines()[0][:100], row["content"],
             json.dumps(attrs, ensure_ascii=False), row["created_at"], row["updated_at"]))
        graph.focus_connections(row["id"], attrs["memory_refs"])
