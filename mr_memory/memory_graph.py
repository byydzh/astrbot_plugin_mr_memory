"""One revisable memory graph, including the agent's experience of recalling it.

The graph records model-authored meaning. Access frequency schedules attention;
it never changes a claim's truth, merges people, or creates semantic edges.
All calls run under Store's connection lock and transaction boundary.
"""
from __future__ import annotations

import json
import time


SCHEMA = """
CREATE TABLE IF NOT EXISTS mr_memory_objects (
 umo TEXT NOT NULL,kind TEXT NOT NULL,id INTEGER NOT NULL,title TEXT NOT NULL,
 content TEXT NOT NULL,attributes_json TEXT NOT NULL DEFAULT '{}',
 revision INTEGER NOT NULL DEFAULT 1,status TEXT NOT NULL DEFAULT 'ACTIVE',
 created_at REAL NOT NULL,updated_at REAL NOT NULL,change_seq INTEGER NOT NULL DEFAULT 0,
 PRIMARY KEY(umo,kind,id));
CREATE INDEX IF NOT EXISTS mr_memory_source ON mr_memory_objects(umo,kind,
 json_extract(attributes_json,'$.source_ref.kind'),json_extract(attributes_json,'$.source_ref.id'));
CREATE INDEX IF NOT EXISTS mr_memory_target ON mr_memory_objects(umo,kind,
 json_extract(attributes_json,'$.target_ref.kind'),json_extract(attributes_json,'$.target_ref.id'));
CREATE TABLE IF NOT EXISTS mr_memory_cues (
 umo TEXT NOT NULL,kind TEXT NOT NULL,owner_id INTEGER NOT NULL,cue TEXT NOT NULL,
 aspect TEXT NOT NULL DEFAULT '',PRIMARY KEY(umo,kind,owner_id,cue,aspect));
CREATE INDEX IF NOT EXISTS mr_memory_cue_lookup ON mr_memory_cues(umo,cue,aspect);
CREATE TABLE IF NOT EXISTS mr_memory_changes (
 seq INTEGER PRIMARY KEY AUTOINCREMENT,umo TEXT NOT NULL,kind TEXT NOT NULL,
 owner_id INTEGER NOT NULL,revision INTEGER NOT NULL,at REAL NOT NULL,reason TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS mr_graph_migrations (umo TEXT PRIMARY KEY,version INTEGER NOT NULL);
"""


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def canonical(kind):
    return {"plastic_edge": "association", "episodic": "episode"}.get(kind, kind)


def reference(value):
    if not isinstance(value, dict):
        raise ValueError("A memory address needs kind and id")
    return {"kind": canonical(str(value["kind"])), "id": int(value["id"])}


class MemoryGraph:
    def __init__(self, store):
        self.store, self.db, self.umo = store, store.db, store.umo
        store.memory_graph = self
        self.db.executescript(SCHEMA)
        from .memory_migration import migrate
        with self.db:
            migrate(self)

    def raw(self, kind, owner):
        if kind == "message":
            row = self.db.execute("SELECT * FROM messages WHERE umo=? AND id=?", (self.umo, int(owner))).fetchone()
            if row is None:
                return None
            return {"umo": self.umo, "kind": "message", "id": row["id"],
                    "title": row["sender_name"] + "的原始发言", "content": row["plain_text"],
                    "attributes_json": encode({"source_ids": [row["id"]], "sender_id": row["sender_id"],
                                               "sender_name": row["sender_name"], "role": row["role"], "sent_at": row["sent_at"]}),
                    "revision": row["revision_no"], "status": "INVALIDATED" if row["is_deleted"] else "ACTIVE",
                    "created_at": row["created_at"], "updated_at": row["updated_at"]}
        row = self.db.execute("SELECT * FROM mr_memory_objects WHERE umo=? AND kind=? AND id=?",
                              (self.umo, canonical(kind), int(owner))).fetchone()
        return dict(row) if row else None

    def brief(self, ref):
        row = self.raw(ref["kind"], ref["id"])
        if not row:
            return {**ref, "status": "unavailable"}
        return {"kind": row["kind"], "id": row["id"], "title": row["title"],
                "revision_no": row["revision"], "status": row["status"]}

    def cues(self, kind, owner):
        return self.store._rows("SELECT cue,aspect FROM mr_memory_cues WHERE umo=? AND kind=? AND owner_id=? ORDER BY cue,aspect",
                                (self.umo, kind, owner))

    def source_rows(self, row, *, full=False):
        attrs = json.loads(row["attributes_json"])
        ids = attrs.get("source_ids", [])
        if full:
            return self.store.messages(ids)
        if not ids:
            return []
        return self.store._rows("SELECT id,source_key,sender_participant_id participant_id,sender_id,sender_name,role FROM messages WHERE umo=? AND is_deleted=0 AND id IN(" + ",".join("?" for _ in ids) + ")", [self.umo, *ids])

    def get(self, kind, owner, *, include_sources=True, include_history=False, attention=True):
        row = self.raw(kind, owner)
        if not row or row["status"] == "INVALIDATED" or (row["status"] != "ACTIVE" and not include_history):
            return None
        attrs = json.loads(row["attributes_json"])
        sources = self.source_rows(row, full=include_sources)
        result = {**attrs, "kind": row["kind"], "id": row["id"], "title": row["title"],
                  "summary": row["content"], "content": row["content"], "status": row["status"],
                  "revision_no": row["revision"], "created_at": row["created_at"], "updated_at": row["updated_at"],
                  "source_ids": [r["id"] for r in sources], "source_keys": [r["source_key"] for r in sources],
                  "cues": self.cues(row["kind"], row["id"])}
        speakers = {(r.get("participant_id"), r["sender_id"], r["sender_name"], r["role"]) for r in sources}
        result["source_speakers"] = [dict(participant_id=p, account_id=a, name_at_message=n, role=r)
                                     for p, a, n, r in sorted(speakers, key=lambda s: str(s[1:]))]
        if include_sources:
            result["sources"] = sources
        if row["kind"] == "node":
            result.update(node_id=row["id"], label=row["title"], description=row["content"], aliases=attrs.get("aliases", []))
        if row["kind"] == "association":
            result.update(statement=row["content"], relation=attrs.get("relation", ""))
            for side in ("source", "target"):
                ref = attrs.get(side + "_ref")
                if ref:
                    end = self.brief(ref)
                    end_raw = self.raw(ref["kind"], ref["id"])
                    end_attrs = json.loads(end_raw["attributes_json"]) if end_raw else {}
                    end.update(label=end.get("title", ""), description=end_raw["content"] if end_raw else "",
                               aliases=end_attrs.get("aliases", []))
                    if attention:
                        concerns = self.store.reflections.associated(ref["kind"], ref["id"], include_sources=include_sources)
                        if concerns:
                            end["reflections"] = concerns
                    result[side] = end["label"]
                    result[side + "_node"] = end
                    result[side + "_node_id"] = ref["id"]
        if attention:
            concerns = self.store.reflections.associated(row["kind"], row["id"], include_sources=include_sources)
            if concerns:
                result["reflections"] = concerns
            basis = self.connections({"kind": row["kind"], "id": row["id"]}, purpose="basis")
            if basis:
                result["basis"] = basis
            if "mr_memory_rehearsals" in self.store.tables:
                rehearsal = self.db.execute("SELECT through_seq,note,at FROM mr_memory_rehearsals WHERE umo=? AND kind=? AND owner_id=?",
                                            (self.umo, row["kind"], row["id"])).fetchone()
                if rehearsal:
                    result["last_reconsideration"] = dict(rehearsal)
        if include_history:
            result["history"] = self.store._memory_history(row["kind"], row["id"])
        if getattr(self.store, "cognition", None) is not None:
            selected = self.db.execute("SELECT state_json FROM mr_working_set WHERE umo=? AND kind=? AND owner_id=?",
                                       (self.umo, row["kind"], row["id"])).fetchone()
            result["attention"] = json.loads(selected[0]) if selected else None
        return result

    def connections(self, ref, *, purpose=None):
        rows = self.store._rows("""SELECT * FROM mr_memory_objects WHERE umo=? AND kind='association' AND status='ACTIVE'
            AND json_extract(attributes_json,'$.source_ref.kind')=?
            AND json_extract(attributes_json,'$.source_ref.id')=? ORDER BY id""", (self.umo, ref["kind"], ref["id"]))
        result = []
        for row in rows:
            attrs = json.loads(row["attributes_json"])
            if purpose is not None and attrs.get("purpose") != purpose:
                continue
            target = self.brief(attrs["target_ref"])
            result.append({"kind": "association", "id": row["id"], "relation": attrs.get("relation", ""),
                           "context": row["content"], "purpose": attrs.get("purpose", "association"),
                           "target": target, "based_on_revision": attrs.get("target_revision"),
                           "basis_changed": attrs.get("purpose") == "basis" and (
                               target.get("revision_no") != attrs.get("target_revision") or target.get("status") != "ACTIVE")})
        return result

    def search(self, kind="all", terms=(), participant_id=None, limit=12, *, related_account_id=None, include_sources=False):
        words = [terms] if isinstance(terms, str) else list(terms)
        prefix, prefix_args = "", []
        clauses, args = ["m.umo=?", "m.status='ACTIVE'"], [self.umo]
        if kind != "all":
            clauses.append("m.kind=?")
            args.append(canonical(kind))
        if words:
            clauses.append("(" + " OR ".join("(instr(lower(m.title||' '||m.content||' '||m.attributes_json),lower(?))>0 OR EXISTS(SELECT 1 FROM mr_memory_cues c WHERE c.umo=m.umo AND c.kind=m.kind AND c.owner_id=m.id AND instr(lower(c.cue||' '||c.aspect),lower(?))>0))" for _ in words) + ")")
            args.extend(word for word in words for _ in range(2))
        if participant_id is not None:
            clauses.append("(json_extract(m.attributes_json,'$.participant_id')=? OR EXISTS(SELECT 1 FROM json_each(m.attributes_json,'$.source_ids') s JOIN messages raw ON raw.id=s.value WHERE raw.umo=m.umo AND raw.sender_participant_id=?))")
            args.extend([int(participant_id)] * 2)
        if related_account_id is not None:
            from .store import RELATED_ACCOUNT_SQL
            # Follow explicit formation references as candidate provenance.
            # Being related through an experience does not assign its facts to a person.
            prefix = """WITH RECURSIVE related(kind,id) AS (
                SELECT o.kind,o.id FROM mr_memory_objects o WHERE o.umo=? AND o.status='ACTIVE' AND
                (CAST(json_extract(o.attributes_json,'$.subject.account_id') AS TEXT)=? OR EXISTS(
                    SELECT 1 FROM json_each(o.attributes_json,'$.source_ids') s JOIN messages m ON m.id=s.value
                    WHERE m.umo=o.umo AND m.is_deleted=0 AND """ + RELATED_ACCOUNT_SQL + """))
                UNION SELECT json_extract(e.attributes_json,'$.source_ref.kind'),json_extract(e.attributes_json,'$.source_ref.id')
                FROM related r JOIN mr_memory_objects e ON e.umo=? AND e.kind='association' AND e.status='ACTIVE'
                AND json_extract(e.attributes_json,'$.purpose')='basis'
                AND json_extract(e.attributes_json,'$.target_ref.kind')=r.kind
                AND json_extract(e.attributes_json,'$.target_ref.id')=r.id) """
            prefix_args = [self.umo, *([str(related_account_id)] * 5), self.umo]
            clauses.append("(m.kind,m.id) IN (SELECT kind,id FROM related)")
        rows = self.store._rows(prefix + "SELECT m.kind,m.id FROM mr_memory_objects m WHERE " + " AND ".join(clauses) + " ORDER BY m.updated_at DESC,m.kind,m.id DESC LIMIT ?", [*prefix_args, *args, max(1, min(500, int(limit)))])
        return [self.get(r["kind"], r["id"], include_sources=include_sources) for r in rows]

    def graph(self, node_id=None, terms=(), limit=12, *, ref=None, related_account_id=None, include_sources=False, offset=0):
        ref = reference(ref) if ref is not None else ({"kind": "node", "id": int(node_id)} if node_id is not None else None)
        clauses, args = ["umo=?", "kind='association'", "status='ACTIVE'"], [self.umo]
        if ref:
            clauses.append("((json_extract(attributes_json,'$.source_ref.kind')=? AND json_extract(attributes_json,'$.source_ref.id')=?) OR (json_extract(attributes_json,'$.target_ref.kind')=? AND json_extract(attributes_json,'$.target_ref.id')=?))")
            args.extend([ref["kind"], ref["id"]] * 2)
        rows = self.store._rows("SELECT id FROM mr_memory_objects WHERE " + " AND ".join(clauses) + " ORDER BY updated_at DESC,id DESC", args)
        words = [terms] if isinstance(terms, str) else list(terms)
        values = []
        for row in rows:
            value = self.get("association", row["id"], include_sources=include_sources)
            if words and not any(str(word).lower() in encode(value).lower() for word in words):
                continue
            if related_account_id is not None and not any(r["sender_id"] == str(related_account_id) for r in self.store.messages(value["source_ids"])):
                continue
            values.append(value)
            if len(values) >= max(0, int(offset)) + max(1, min(500, int(limit))):
                break
        return values[max(0, int(offset)):]

    def navigate(self, *, cue=None, aspect=None, ref=None, limit=12, offset=0):
        limit, offset = max(1, min(100, int(limit))), max(0, int(offset))
        if ref:
            ref = reference(ref)
            links = self.graph(ref=ref, limit=limit + 1, offset=offset)
            entries = [{key: value for key, value in row.items() if key in {
                "kind", "id", "source_ref", "target_ref", "relation", "summary", "purpose", "target_revision"}} for row in links]
            for entry in entries:
                entry["source_memory"] = self.brief(entry["source_ref"])
                entry["target_memory"] = self.brief(entry["target_ref"])
            return {"memory": self.brief(ref), "cues": self.cues(ref["kind"], ref["id"]),
                    "connections": entries[:limit], "more": len(entries) > limit, "next_offset": offset + limit}
        clauses, args = ["c.umo=?", "m.status='ACTIVE'"], [self.umo]
        if cue is not None:
            clauses.append("c.cue=? COLLATE NOCASE")
            args.append(str(cue))
        if aspect is not None:
            clauses.append("c.aspect=? COLLATE NOCASE")
            args.append(str(aspect))
        base = " FROM mr_memory_cues c JOIN mr_memory_objects m ON m.umo=c.umo AND m.kind=c.kind AND m.id=c.owner_id WHERE " + " AND ".join(clauses)
        if cue is not None and aspect is not None:
            rows = self.store._rows("SELECT m.kind,m.id,m.title,m.revision revision_no" + base + " ORDER BY m.updated_at DESC,m.id LIMIT ? OFFSET ?", [*args, limit + 1, offset])
        else:
            rows = self.store._rows("SELECT c.cue,c.aspect,count(*) memories" + base + " GROUP BY c.cue,c.aspect ORDER BY c.cue,c.aspect LIMIT ? OFFSET ?", [*args, limit + 1, offset])
        return {"items": rows[:limit], "more": len(rows) > limit, "next_offset": offset + limit}

    def _endpoint(self, value, sources, run_id):
        if isinstance(value, dict) and "kind" in value and "id" in value:
            ref = reference(value)
            if not self.raw(ref["kind"], ref["id"]):
                raise ValueError(f"Unknown memory endpoint {ref}")
            return ref
        value = {"label": value} if isinstance(value, str) else dict(value or {})
        if "node_id" in value:
            value["id"] = value.pop("node_id")
        value.update(kind="node", source_ids=sources)
        if "id" in value and set(value) <= {"id", "kind", "source_ids"}:
            ref = reference(value)
            if not self.raw(ref["kind"], ref["id"]):
                raise ValueError(f"Unknown memory endpoint {ref}")
            return ref
        saved = self.write(value, run_id=run_id)
        return reference(saved)

    def focus_connections(self, owner, refs):
        """Index a thought's explicit focus links without a second copy of prose."""
        current = self.connections({"kind": "reflection", "id": owner}, purpose="focus")
        wanted = {(canonical(ref["kind"]), int(ref["id"])) for ref in refs}
        retained = set()
        for edge in current:
            key = (edge["target"]["kind"], edge["target"]["id"])
            if key in wanted:
                retained.add(key)
            else:
                self.write({"kind": "association", "id": edge["id"], "action": "withdraw", "reason": "思路已改变关注对象"})
        for kind, target in wanted - retained:
            if self.raw(kind, target):
                self.write({"kind": "association", "source": {"kind": "reflection", "id": owner},
                    "target": {"kind": kind, "id": target}, "relation": "继续思考", "purpose": "focus",
                    "statement": "这项思路所关注的记忆"})

    def write(self, item, *, run_id=None):
        kind = canonical(str(item.get("kind") or ""))
        if not kind or kind in {"all", "cue", "participant"}:
            raise ValueError("Choose a stored memory kind (episode, semantic, topic, pattern, node, association, or another descriptive kind)")
        stored = self.raw(kind, item["id"]) if item.get("id") is not None else None
        if item.get("id") is not None and not stored:
            raise ValueError(f"No {kind} memory with id {item['id']}")
        if stored and "attention" in item and not (set(item) - {"kind", "id", "revision_no", "attention", "source_keys"}):
            self.store.cognition.select({"kind": kind, "id": stored["id"]}, item["attention"])
            return self.get(kind, stored["id"], include_sources=False)
        if kind == "message":
            raise ValueError("Original messages are observations maintained by capture; save interpretations as a memory connected to this message")
        if stored and item.get("revision_no") is not None and int(item["revision_no"]) != stored["revision"]:
            raise ValueError("Memory changed since it was read; open and merge its current version: " + encode(self.brief(reference(stored))))
        # Capture before an endpoint edit recursively changes an existing node.
        previous_memory = self.get(kind, stored["id"], include_sources=False, include_history=True, attention=False) if stored else None
        if previous_memory is not None:
            previous_memory.pop("history", None)
        attrs = json.loads(stored["attributes_json"]) if stored else {}
        old = dict(attrs)
        reserved = {"id", "kind", "title", "label", "name", "summary", "content", "statement", "description", "reason", "action", "revision_no", "cues", "connections", "source_keys", "source", "target", "attention"}
        attrs.update({key: value for key, value in item.items() if key not in reserved})
        if kind == "reflection":
            attrs.setdefault("review_status", "pending")
            attrs.setdefault("priority", 1)
            attrs.setdefault("memory_refs", [])
        sources = list(dict.fromkeys(int(v) for v in attrs.get("source_ids", [])))
        source_messages = self.store.messages(sources)
        if {r["id"] for r in source_messages} != set(sources):
            raise ValueError("A source message is unavailable in this group")
        attrs["source_ids"] = sources
        if kind == "episode" and source_messages and ("source_ids" in item or not stored):
            if "started_at" not in item:
                attrs["started_at"] = min(row["sent_at"] for row in source_messages)
            if "ended_at" not in item:
                attrs["ended_at"] = max(row["sent_at"] for row in source_messages)
        title = next((str(item[k]) for k in ("title", "label", "name") if k in item), stored["title"] if stored else str(item.get("aspect", item.get("person", ""))))
        content = next((str(item[k]) for k in ("content", "summary", "statement", "description") if k in item), stored["content"] if stored else "")
        action = item.get("action") if item.get("action") is not None else "revise"
        if action not in {"revise", "withdraw"}:
            raise ValueError("Use revise or withdraw")
        if action == "withdraw" and not stored:
            raise ValueError("Withdrawing needs an existing memory address")
        if not (content.strip() or (kind == "node" and title.strip())) and action != "withdraw":
            raise ValueError("A memory needs readable content")
        if kind == "association":
            for side in ("source", "target"):
                if side in item:
                    attrs[side + "_ref"] = self._endpoint(item[side], sources, run_id)
                if side + "_ref" not in attrs:
                    raise ValueError("A connection needs source and target memory addresses")
            if not attrs.get("relation"):
                raise ValueError("A connection needs a meaningful relation")
            if attrs.get("purpose") == "basis" and ("target" in item or "target_revision" not in attrs):
                attrs["target_revision"] = self.raw(**{"kind": attrs["target_ref"]["kind"], "owner": attrs["target_ref"]["id"]})["revision"]
        if isinstance(attrs.get("subject"), dict) and attrs["subject"].get("account_id") is not None:
            account = str(attrs["subject"]["account_id"])
            who = self.db.execute("SELECT id FROM participants WHERE umo=? AND account_id=?", (self.umo, account)).fetchone()
            attrs["subject"] = {**attrs["subject"], "account_id": account}
            attrs["participant_id"] = who[0] if who else None
        elif "subject" in item:
            attrs["participant_id"] = None
        elif "person" in item and "subject" not in item:
            attrs["subject"] = {"name": str(item["person"]), "account_id": None}
            attrs["participant_id"] = None
        # Exact repeated assertions with the same provenance are one write;
        # this does not infer identity or merge similarly named people/nodes.
        if not stored and kind not in {"node", "association"} and not item.get("connections") and not item.get("cues"):
            identical = self.db.execute("SELECT id FROM mr_memory_objects WHERE umo=? AND kind=? AND title=? AND content=? AND attributes_json=? AND status='ACTIVE'",
                                        (self.umo, kind, title, content, encode(attrs))).fetchone()
            if identical:
                if "attention" in item:
                    self.store.cognition.select({"kind": kind, "id": identical[0]}, item["attention"])
                return self.get(kind, identical[0], include_sources=False)
        owner = stored["id"] if stored else self.db.execute("SELECT COALESCE(max(id),0)+1 FROM mr_memory_objects WHERE umo=? AND kind=?", (self.umo, kind)).fetchone()[0]
        now, reason = time.time(), str(item.get("reason", ""))
        revision = stored["revision"] + 1 if stored else 1
        if stored:
            snapshot = {"record": stored, "memory": previous_memory}
            self.db.execute("INSERT INTO mr_memory_revisions(umo,kind,owner_id,operation,reason,run_id,snapshot_json,source_ids_json) VALUES(?,?,?,?,?,?,?,?)",
                            (self.umo, kind, owner, action, reason, run_id, encode(snapshot), encode(sources)))
        change = self.db.execute("INSERT INTO mr_memory_changes(umo,kind,owner_id,revision,at,reason) VALUES(?,?,?,?,?,?)",
                                 (self.umo, kind, owner, revision, now, reason)).lastrowid
        self.db.execute("""INSERT INTO mr_memory_objects VALUES(?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(umo,kind,id) DO UPDATE SET title=excluded.title,content=excluded.content,
            attributes_json=excluded.attributes_json,revision=excluded.revision,status=excluded.status,
            updated_at=excluded.updated_at,change_seq=excluded.change_seq""",
            (self.umo, kind, owner, title, content, encode(attrs), revision, "RETRACTED" if action == "withdraw" else "ACTIVE",
             stored["created_at"] if stored else now, now, change))
        if "cues" in item:
            self.db.execute("DELETE FROM mr_memory_cues WHERE umo=? AND kind=? AND owner_id=?", (self.umo, kind, owner))
            for cue in item["cues"]:
                cue = {"cue": cue, "aspect": item.get("aspect", "")} if isinstance(cue, str) else cue
                self.db.execute("INSERT OR IGNORE INTO mr_memory_cues VALUES(?,?,?,?,?)",
                                (self.umo, kind, owner, str(cue["cue"]), str(cue.get("aspect", ""))))
        self.store._clear_memory_derivatives(kind, owner)
        self.store._queue_embedding(kind, owner)
        if "attention" in item:
            self.store.cognition.select({"kind": kind, "id": owner}, item["attention"])
        if kind == "reflection" and "memory_refs" in item:
            self.focus_connections(owner, attrs["memory_refs"])
        # The model supplies connection meaning; the write merely connects addresses.
        for link in item.get("connections", []):
            target = reference(link)
            self.write({"kind": "association", "source": {"kind": kind, "id": owner}, "target": target,
                        "relation": link["relation"], "purpose": link.get("purpose", "association"),
                        "statement": link.get("context") or link["relation"], "source_ids": [],
                        **({"id": link["connection_id"]} if link.get("connection_id") is not None else {})}, run_id=run_id)
        # A relabelled endpoint changes the searchable description of its incident edges.
        if stored and (title != stored["title"] or content != stored["content"] or attrs.get("aliases") != old.get("aliases")):
            edges = self.store._rows("""SELECT id FROM mr_memory_objects WHERE umo=? AND kind='association'
                AND ((json_extract(attributes_json,'$.source_ref.kind')=? AND json_extract(attributes_json,'$.source_ref.id')=?)
                OR (json_extract(attributes_json,'$.target_ref.kind')=? AND json_extract(attributes_json,'$.target_ref.id')=?))""",
                (self.umo, kind, owner, kind, owner))
            for edge in edges:
                self.store._clear_memory_derivatives("association", edge["id"])
                self.store._queue_embedding("association", edge["id"])
        return self.get(kind, owner, include_sources=False, include_history=action == "withdraw")

    def changes(self, after=0, limit=20):
        rows = self.store._rows("SELECT kind,id,change_seq FROM mr_memory_objects WHERE umo=? AND change_seq>? ORDER BY change_seq LIMIT ?",
                                (self.umo, int(after), max(1, min(100, int(limit))) + 1))
        chosen = rows[:limit]
        return {"items": [{**self.brief(row), "change_seq": row["change_seq"]} for row in chosen],
                "cursor": chosen[-1]["change_seq"] if chosen else after, "more": len(rows) > limit}
