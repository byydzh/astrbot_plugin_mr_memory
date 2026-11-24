"""Project model-confirmed anchors using the same physical views as the batch graph."""

from datetime import datetime, timezone, timedelta
import json
from .graph import Graph, L1


class AlignedGraph(Graph):
    def _link(self, a, b, rel, at, direct, negative=False):
        if rel == "formed_from" and direct and not a.startswith("E|"):
            left = self.read(a)
            right = self.read(b)
            if not left or not right:
                raise ValueError("Missing evidence endpoint")
            region = left["region"]
            same = region == right["region"]
            if left["kind"] not in L1:
                if right["region"] == "E":
                    rel = "backed_by" if region == "S" else "witness"
                elif region in ("M", "K") and same:
                    rel = "anchor" if right["kind"] in L1 else "inherits"
                    direct = False
                elif region == "S":
                    rel = "backed_by" if right["kind"] in L1 else "interprets"
                    direct = False
        super()._link(a, b, rel, at, direct, negative)
        if rel in ("inherits", "inherited_anchor"):
            self.db.execute("update edge set w=.6 where src=? and dst=? and rel=?", (a, b, rel))

    def project_claims(self, rec, ops, lookup, at):
        self.db.execute(
            "create table if not exists derived_shortcut(via text,src text,dst text,preexisting integer,primary key(via,src,dst)) without rowid"
        )
        for local in ops.get("program_local_projection", []):
            a = local["id"]
            selected = {k: (n, sim) for k, n, sim in local["selected"]}
            old = {v[0] for v in self.db.execute("select dst from edge where src=? and rel='local'", (a,))}
            for b in sorted(old - selected.keys()):
                self.db.execute("delete from edge where src=? and dst=? and rel='local'", (a, b))
                self.touched.update((a, b))
            for b, (shared, _) in sorted(selected.items()):
                self._link(a, b, "local", at, bool(shared))
                self.db.execute(
                    "update edge set strength=?,direct=? where src=? and dst=? and rel='local'",
                    (max(1, shared), int(bool(shared)), a, b),
                )
        projected = []
        for model in ops["changes"]:
            key = lookup[model["id"]]
            node = self.read(key)
            if node["region"] not in ("M", "K") or node["kind"] in L1:
                continue
            basis = [lookup.get(x, x) for x in model["basis"]]
            anchors = set()
            parents = []
            for x in basis:
                r = self.read(x)
                if r["kind"] in L1:
                    local = key[0] + x[1:]
                    if self.read(local):
                        anchors.add(local)
                        self._link(key, local, "anchor", at, False)
                elif r["region"] == node["region"]:
                    parents.append(x)
            inherited = {
                z[0]
                for p in parents
                for z in self.db.execute(
                    "select dst from edge where src=? and rel in ('anchor','inherited_anchor')", (p,)
                )
            }
            for x in sorted(inherited - anchors):
                self._link(key, x, "inherited_anchor", at, False)
            if node["region"] == "K":
                for x in sorted(anchors | inherited):
                    for (raw,) in self.db.execute(
                        "select dst from edge where src=? and rel='source' order by dst", ("M" + x[1:],)
                    ).fetchall():
                        self._link(key, raw, "witness", at, False)
            # Each active bridge revision certifies its cited remote shortcuts; other bridges can support the same pair.
            old = {
                (a, b): pre
                for a, b, pre in self.db.execute(
                    "select src,dst,preexisting from derived_shortcut where via=?", (key,)
                )
            }
            eligible = set()
            if node["kind"] == "bridge" and node["status"] == "ACTIVE":
                calendar = timezone(timedelta(hours=8))
                month = lambda t: datetime.fromtimestamp(t, calendar).strftime("%Y-%m")
                stamps = {x: self.read(x)["at"] for x in anchors}
                if "incoming_l1" in ops:
                    for x in anchors:
                        clock = self.db.execute("SELECT at FROM network_l1_clock WHERE id=?", (x,)).fetchone()
                        if clock:
                            stamps[x] = clock[0]
                    new = anchors.intersection(ops["incoming_l1"])
                    eligible = {
                        (a, b)
                        for a in new
                        for b in anchors
                        if 0 < stamps[b] < stamps[a] and month(stamps[b]) != month(stamps[a])
                    }
                else:
                    new = [x for x, t in stamps.items() if t == at]
                    remote = [x for x, t in stamps.items() if 0 < t < at and month(t) != month(at)]
                    eligible = {(a, b) for a in new for b in remote}
            for (a, b), prior in old.items():
                if (a, b) in eligible:
                    continue
                self.db.execute("delete from derived_shortcut where via=? and src=? and dst=?", (key, a, b))
                other = self.db.execute(
                    "select 1 from derived_shortcut where src=? and dst=? limit 1", (a, b)
                ).fetchone()
                explicit = self.db.execute(
                    "select 1 from edge_basis where src=? and dst=? and rel='bridge'", (a, b)
                ).fetchone()
                if not prior and not other and not explicit:
                    self.db.execute("delete from edge where src=? and dst=? and rel='bridge'", (a, b))
                    self.touched.update((a, b))
                projected.append([key, a, b, "release"])
            for a, b in sorted(eligible):
                if (a, b) not in old:
                    old_owner = self.db.execute(
                        "select preexisting from derived_shortcut where src=? and dst=? limit 1", (a, b)
                    ).fetchone()
                    prior = (
                        old_owner[0]
                        if old_owner
                        else int(
                            bool(
                                self.db.execute(
                                    "select 1 from edge where src=? and dst=? and rel='bridge'", (a, b)
                                ).fetchone()
                            )
                        )
                    )
                    self.db.execute("insert into derived_shortcut values(?,?,?,?)", (key, a, b, prior))
                self._link(a, b, "bridge", at, False)
                projected.append([key, a, b, "support"])
        rec["bridge_shortcuts"] = projected
