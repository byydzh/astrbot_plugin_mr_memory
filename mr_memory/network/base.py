"""Incremental writes and inspection against the migrated Plan D address space."""

import hashlib
import json
import sqlite3


def packed(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


class Network:
    def __init__(self, db):
        self.db = db
        db.row_factory = sqlite3.Row
        db.execute("pragma foreign_keys=on")
        exists = db.execute("select 1 from sqlite_master where type='table' and name='withdrawn'").fetchone()
        self.withdrawn = {r[0] for r in db.execute("select id from withdrawn")} if exists else set()

    def bindings(self, actor):
        return [
            r[0]
            for r in self.db.execute(
                """WITH RECURSIVE accounts(uid) AS (
            VALUES(?) UNION SELECT i.b FROM identity i JOIN accounts a ON i.a=a.uid
            UNION SELECT i.a FROM identity i JOIN accounts a ON i.b=a.uid) SELECT uid FROM accounts""",
                (actor,),
            )
        ]

    def openings(self, actors, per_actor=4, limit=16, *, recall_clock=False):
        order = "mr_recency(n.id,n.at)" if recall_clock else "n.at"
        pages = []
        matched = set()
        for actor in dict.fromkeys(actors):
            if actor in matched:
                continue
            group = self.bindings(actor)
            matched.update(group)
            items = {}
            for uid in group:
                rows = self.db.execute(
                    f"""select n.id,n.kind,n.body,n.at,n.revision from subject s
                    join node n on n.id=s.id where s.account=? and substr(s.id,1,2)='S|' and n.status='ACTIVE'
                    order by {order} desc,n.id limit ?""",
                    (uid, per_actor),
                )
                for r in rows:
                    items[r[0]] = {"id": r[0], "kind": r[1], "preview": r[2][:200], "revision": r[4]}
            if items:
                pages.append(list(items.values()))
        result = {}
        for i in range(max(map(len, pages), default=0)):
            for page in pages:
                if i < len(page):
                    result[page[i]["id"]] = page[i]
                    if len(result) >= limit:
                        return list(result.values())
        return list(result.values())

    def read(self, key):
        if key in self.withdrawn:
            return None
        r = self.db.execute("select * from node where id=?", (key,)).fetchone()
        if not r or r["status"] == "INVALIDATED":
            return None
        return dict(r) | {"subjects": json.loads(r["subjects"])}

    def walk(self, key, start=0, size=12):
        # Stored direction and availability travel with each edge, including source-only links.
        if not self.read(key):
            return []
        rows = self.db.execute(
            """select * from (
            select e.src,e.dst,e.rel,e.bin,e.active,e.direct,n.id,n.kind,n.body,n.revision,n.status
            from edge e join node n on n.id=e.dst where e.src=?
            union all
            select e.src,e.dst,e.rel,e.bin,e.active,e.direct,n.id,n.kind,n.body,n.revision,n.status
            from edge e join node n on n.id=e.src where e.dst=? and e.src<>e.dst)
            where status<>'INVALIDATED' order by active desc,direct desc,src,dst,rel limit ? offset ?""",
            (key, key, min(32, max(1, size)), max(0, start)),
        )
        return [
            {k: v for k, v in dict(r).items() if k != "body"} | {"preview": r["body"][:180]}
            for r in rows
            if r["id"] not in self.withdrawn
        ]

    def apply(self, batch, at, material, old, output):
        """Advance the input cursor and graph as one transaction. Replayed batches are idempotent."""
        digest = hashlib.sha256(packed(material).encode()).hexdigest()
        saved = self.db.execute("select receipt from batch where k=?", (batch,)).fetchone()
        if saved:
            receipt = json.loads(saved[0])
            if receipt["digest"] != digest:
                raise ValueError("Batch ID reused with different input")
            return receipt
        known = set(old)
        changes = output["changes"]
        edges = output["edges"]
        aliases = {}
        if not isinstance(changes, list) or not isinstance(edges, list):
            raise ValueError("Expected change and edge arrays")
        receipt = {
            "digest": digest,
            "input": len(material),
            "old_seen": len(known),
            "edits": [],
            "edges": [],
            "raw": [],
        }
        with self.db:
            previous = (
                self.db.execute(
                    """select id,at from node where region='E' and kind='message'
                and at<=? order by at desc limit 1""",
                    (material[0]["at"],),
                ).fetchone()
                if material
                else None
            )
            for raw in material:
                key = raw["id"]
                if self.read(key):
                    # Live capture publishes L0 before its scheduled consolidation.
                    # The queue transaction checks its revision and consumes it once.
                    existing = self.read(key)
                    if (existing["region"], existing["kind"], existing["at"], existing["subjects"]) != (
                        "E",
                        "message",
                        raw["at"],
                        [raw["uid"]],
                    ):
                        raise ValueError("Original address changed before consolidation: " + key)
                    known.add(key)
                    receipt["raw"].append(key)
                    previous = (key, raw["at"])
                    continue
                self.db.execute(
                    "insert into node values(?,?,?,?,?,?,?,1)",
                    (key, "E", "message", raw["text"], raw["at"], packed([raw["uid"]]), "ACTIVE"),
                )
                self.db.execute("insert into subject values(?,?)", (raw["uid"], key))
                if previous and 0 <= raw["at"] - previous[1] <= 600:
                    self._link(key, previous[0], "adjacent", at, True)
                if raw.get("reply") and self.read(raw["reply"]):
                    self._link(key, raw["reply"], "REPLY_TO", at, True)
                previous = (key, raw["at"])
                known.add(key)
                receipt["raw"].append(key)
            for i, op in enumerate(changes):
                addr = op["id"]
                if addr in aliases:
                    raise ValueError("Repeated local address")
                existing = self.read(addr)
                if existing:
                    if addr not in known or existing["revision"] != op.get("revision"):
                        raise ValueError("Read the current revision before editing " + addr)
                    key, region = addr, existing["region"]
                    if existing["kind"] in ("message", "encounter", "interaction"):
                        raise ValueError("Evidence is immutable")
                else:
                    if not addr.startswith("+") or op.get("region") not in ("M", "S", "K"):
                        raise ValueError("New addresses start with + and require region M, S or K")
                    region = op["region"]
                    key = f"{region}|delta:{batch}:{i+1}"
                basis = op.get("basis", [])
                if not basis or any(b not in known and b not in aliases for b in basis):
                    raise ValueError("A change must cite available evidence")
                if not op.get("text", "").strip() or not isinstance(op.get("subjects", []), list):
                    raise ValueError("Provide text and an array of actual subject IDs")
                subjects = list(dict.fromkeys(op.get("subjects", [])))
                if any(not isinstance(a, str) or not a.strip() for a in subjects):
                    raise ValueError("Invalid subject account")
                status = op.get("status", "ACTIVE")
                if status not in ("ACTIVE", "QUARANTINED", "RETRACTED", "CONFLICTED"):
                    raise ValueError("Invalid status")
                rev = existing["revision"] + 1 if existing else 1
                self.db.execute(
                    "insert into change values(?,?,?,?)",
                    (key, rev, at, packed(existing) if existing else None),
                )
                self.db.execute(
                    """insert into node values(?,?,?,?,?,?,?,?)
                    on conflict(id) do update set body=excluded.body, subjects=excluded.subjects,
                    status=excluded.status,revision=excluded.revision,at=excluded.at""",
                    (
                        key,
                        region,
                        existing["kind"] if existing else op["kind"],
                        op["text"],
                        at,
                        packed(subjects),
                        status,
                        rev,
                    ),
                )
                self.db.execute("delete from subject where id=?", (key,))
                self.db.executemany("insert into subject values(?,?)", [(u, key) for u in subjects])
                for b in basis:
                    self._link(key, aliases.get(b, b), "formed_from", at, True)
                aliases[addr] = key
                receipt["edits"].append({"id": key, "revision": rev, "status": status})
            for edge in edges:
                endpoints = [aliases.get(edge[side], edge[side]) for side in ("source", "target")]
                if any(k not in known and k not in aliases.values() for k in endpoints):
                    raise ValueError("Edge endpoint outside the visible neighborhood")
                if not isinstance(edge["relation"], str) or not edge["relation"].strip():
                    raise ValueError("Empty relation")
                self._link(*endpoints, edge["relation"], at, False, edge.get("effect") == "limits")
                receipt["edges"].append([*endpoints, edge["relation"]])
            self.db.execute("insert into batch values(?,?,?)", (batch, at, packed(receipt)))
        return receipt

    def _link(self, source, target, relation, at, direct, negative=False):
        if source == target:
            raise ValueError("Self link")
        slot = "negative" if negative else "inside" if source[0] == target[0] else "cross"
        self.db.execute(
            "insert or ignore into edge values(?,?,?,?,?,?,?,?,?)",
            (source, target, relation, slot, 1.0, 1, at, 1, int(direct)),
        )
