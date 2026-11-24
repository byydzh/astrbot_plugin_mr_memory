"""Persist semantic writes and derive the routing table in the same SQLite transaction."""

import json
from .base import Network, packed
from .projection import month_ranks, repair, full

L1 = {"episode", "event", "semantic"}
KINDS = {
    "M": L1 | {"bridge", "field"},
    "K": {"content", "scope", "veto"},
    "S": {
        "observation",
        "understanding",
        "situated_knowledge",
        "relationship",
        "atmosphere",
        "familiarity",
        "longitudinal_familiarity",
        "shared_social_state",
        "shared_social_state_verified",
    },
}


def prepare(db):
    db.execute("create index if not exists routing_far on edge(strength,at) where bin='far'")
    # Explicit witness edges remain openable. Physical M/L1 source edges retain Plan D's bounded cross slots.
    db.execute(
        """update edge set bin='source_only' where bin<>'source_only' and
      ((dst like 'E|message:%' and src not like 'E|%') or (src like 'E|message:%' and dst not like 'E|%'))
      and not (rel='source' and dst like 'E|message:%' and src like 'M|%'
               and exists(select 1 from node where id=edge.src and kind in ('episode','event','semantic')))"""
    )
    return full(db, write=True)[1]


class Graph(Network):
    def project_claims(self, rec, ops, lookup, at):
        pass

    def _link(self, a, b, rel, at, direct, negative=False):
        if a == b:
            raise ValueError("Self link")
        self.touched.update((a, b))
        raw = b.startswith("E|message:")
        kind = self.db.execute("select kind from node where id=?", (a,)).fetchone()[0]
        if rel == "formed_from":
            rel = ("source" if a[0] == "M" and kind in L1 else "witness") if raw else "grounded_in"
        slot = (
            "negative"
            if negative or rel in ("limits", "counterexample", "local_veto")
            else "cross" if a[0] != b[0] else "inside"
        )
        upper_raw = (raw and a[0] != "E") or (a.startswith("E|message:") and b[0] != "E")
        if upper_raw and not (rel == "source" and raw and a[0] == "M" and kind in L1):
            slot = "source_only"
        if rel == "bridge" and a[0] == b[0] == "M":
            slot = "far"
        weight, strength = (
            (0.6, 0)
            if rel == "adjacent"
            else (1.0, 3) if rel in ("source", "REPLY_TO", "same_l1") else (0.85, 1)
        )
        self.db.execute(
            """insert into edge values(?,?,?,?,?,?,?,0,?) on conflict(src,dst,rel) do update set
            strength=max(edge.strength,excluded.strength),at=max(edge.at,excluded.at),direct=max(edge.direct,excluded.direct)""",
            (a, b, rel, slot, weight, strength, at, int(direct and rel != "adjacent")),
        )

    def apply(self, batch, at, material, old, ops):
        existing = self.db.execute("select 1 from batch where k=?", (batch,)).fetchone()
        if existing:
            return super().apply(batch, at, material, old, ops)
        self.touched = set()
        far = month_ranks(self.db)
        with self.db:
            for op in ops["changes"]:
                prev = self.read(op["id"])
                region = prev["region"] if prev else op.get("region")
                kind = op.get("kind", prev["kind"] if prev else None)
                if kind not in KINDS.get(region, set()):
                    raise ValueError(f"Unknown or immutable {region}/{kind}")
                if prev and region == "M" and (kind in L1) != (prev["kind"] in L1):
                    raise ValueError("Keep the shared L1 address; create an abstraction supported by it")
            rec = super().apply(batch, at, material, old, ops)
            lookup = {op["id"]: e["id"] for op, e in zip(ops["changes"], rec["edits"], strict=True)}
            mirrors = []
            for op, edit in zip(ops["changes"], rec["edits"], strict=True):
                key = edit["id"]
                self.touched.add(key)
                if op.get("kind"):
                    self.db.execute("update node set kind=? where id=?", (op["kind"], key))
                r = self.read(key)
                if key[0] == "M" and r["kind"] in L1:
                    twin = "K" + key[1:]
                    old_twin = self.read(twin)
                    if old_twin:
                        self.db.execute(
                            "insert into change values(?,?,?,?)", (twin, r["revision"], at, packed(old_twin))
                        )
                    self.db.execute(
                        """insert or replace into node select ?, 'K',kind,body,at,subjects,status,revision from node where id=?""",
                        (twin, key),
                    )
                    self.db.execute("delete from subject where id=?", (twin,))
                    self.db.executemany("insert into subject values(?,?)", [(u, twin) for u in r["subjects"]])
                    self._link(twin, key, "same_l1", at, True)
                    mirrors.append(twin)
            for edge in ops["edges"]:
                k = tuple(lookup.get(edge[s], edge[s]) for s in ("source", "target")) + (edge["relation"],)
                evidence = [lookup.get(x, x) for x in edge.get("basis", [])]
                if not evidence:
                    evidence = list(
                        dict.fromkeys(
                            lookup.get(b, b)
                            for op in ops["changes"]
                            if lookup[op["id"]] in k[:2]
                            for b in op["basis"]
                        )
                    )
                allowed = set(old) | set(rec["raw"]) | set(lookup.values())
                if not evidence or not set(evidence) <= allowed:
                    raise ValueError("Relation needs available evidence")
                v = self.db.execute(
                    "select history from edge_basis where src=? and dst=? and rel=?", k
                ).fetchone()
                history = json.loads(v[0]) if v else []
                history.append({"batch": batch, "basis": evidence})
                self.db.execute("insert or replace into edge_basis values(?,?,?,?)", (*k, packed(history)))
                self.db.execute(
                    "update edge set strength=max(strength,?),direct=max(direct,?) where src=? and dst=? and rel=?",
                    (len(set(evidence)), int(any(i.startswith("E|message:") for i in evidence)), *k),
                )
            rec["mirrors"] = mirrors
            self.project_claims(rec, ops, lookup, at)
            rec["projection"] = repair(self.db, self.touched, far)
            self.db.execute("update batch set receipt=? where k=?", (packed(rec), batch))
        return rec
