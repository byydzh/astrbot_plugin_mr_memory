"""Expose group-local graph addresses through the installed console's existing navigation."""

import base64
import json
from .base import packed
from .history_time import CLOCKS as _CLOCKS, DISPLAY_TIME as _DISPLAY_TIME, JOIN_CLOCK as _JOIN_CLOCK


def edge_id(a, b, r):
    return "graph~" + base64.urlsafe_b64encode(packed([a, b, r]).encode()).decode().rstrip("=")


def ref(db, k):
    row = db.execute("SELECT kind,body FROM node WHERE id=? AND status<>'INVALIDATED'", (k,)).fetchone()
    return {"id": k, "kind": row[0], "title": row[1][:80] or k} if row else None


def link(db, a, b, r, active, slot):
    left, right = ref(db, a), ref(db, b)
    if not left or not right:
        return None
    return dict(
        id=edge_id(a, b, r),
        source=left["title"],
        target=right["title"],
        source_key=a,
        target_key=b,
        source_ref=left,
        target_ref=right,
        relation=r,
        active=bool(active),
        slot=slot,
        kind="association",
    )


def region_filter(region):
    if region not in ("all", "M", "S", "K"):
        raise ValueError("region 须为 all、M、S 或 K")
    return region


def integer_param(name, value, minimum, maximum=None):
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} 须为整数") from None
    if value < minimum or (maximum is None and value > 2**63 - 1):
        raise ValueError(f"{name} 超出范围")
    return min(value, maximum) if maximum is not None else value


def _selector(kind, query, region, *, active):
    clauses = ["n.region<>'E'", "n.body IS NOT NULL", "n.status='ACTIVE'" if active else "n.status<>'INVALIDATED'"]
    args = {}
    if region_filter(region) != "all":
        clauses.append("n.region=:region")
        args["region"] = region
    if kind != "all":
        clauses.append("n.kind=:kind")
        args["kind"] = kind
    if query:
        clauses.append("instr(lower(n.body),lower(:query))>0")
        args["query"] = query
    return " AND ".join(clauses), args


_EDGE_COLUMNS = "e.src,e.dst,e.rel,e.active,e.bin"


def _backbone(db, where, args, limit):
    # Early graph snapshots only have node.at. Newer ones also retain the L1 witness clock.
    clocks, left_time, right_time = "", "a.at", "b.at"
    if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='network_l1_clock'").fetchone():
        clocks = "LEFT JOIN network_l1_clock ca ON ca.id=a.id LEFT JOIN network_l1_clock cb ON cb.id=b.id"
        left_time, right_time = "coalesce(ca.at,a.at)", "coalesce(cb.at,b.at)"
    bridges = db.execute(f"""
        WITH seed AS (SELECT n.id FROM node n WHERE {where})
        SELECT {_EDGE_COLUMNS} FROM edge e
        JOIN node a ON a.id=e.src JOIN node b ON b.id=e.dst
        {clocks}
        WHERE e.bin='far' AND a.region<>'E' AND b.region<>'E' AND a.status='ACTIVE' AND b.status='ACTIVE'
          AND (e.src IN seed OR e.dst IN seed)
          AND strftime('%Y-%m',{left_time},'unixepoch','+8 hours')
             <>strftime('%Y-%m',{right_time},'unixepoch','+8 hours')
        ORDER BY e.active DESC,e.strength DESC,e.at DESC,e.src,e.dst,e.rel LIMIT :lim
    """, {**args, "lim": limit}).fetchall()
    # Reserve representation for both old bridges and globally well-connected upper nodes.
    split = (limit + 1) // 2
    selected = {tuple(row[:3]): row for row in bridges[:split]}
    hubs = db.execute(f"""
        SELECT n.id,
          (SELECT count(*) FROM edge WHERE src=n.id AND active=1 AND dst NOT LIKE 'E|%') +
          (SELECT count(*) FROM edge WHERE dst=n.id AND active=1 AND src NOT LIKE 'E|%') AS degree
        FROM node n WHERE {where}
        ORDER BY degree DESC,n.id LIMIT :lim
    """, {**args, "lim": limit})
    # Take one incident edge per hub before giving a second seat to the same hub.
    neighbourhoods = [
        iter(db.execute(f"""
            SELECT {_EDGE_COLUMNS} FROM edge e WHERE e.active=1 AND
                ((e.src=? AND e.dst NOT LIKE 'E|%') OR (e.dst=? AND e.src NOT LIKE 'E|%'))
            ORDER BY e.strength DESC,e.at DESC,e.src,e.dst,e.rel LIMIT ?
        """, (hub, hub, limit))) for hub, degree in hubs if degree
    ]
    while neighbourhoods and len(selected) < limit:
        remaining = []
        for neighbours in neighbourhoods:
            for row in neighbours:
                key = tuple(row[:3])
                if key not in selected:
                    selected[key] = row
                    remaining.append(neighbours)
                    break
            if len(selected) == limit:
                break
        neighbourhoods = remaining
    for row in bridges[split:]:
        if len(selected) == limit:
            break
        selected[tuple(row[:3])] = row
    return selected.values()


def graph(store, query="", address=None, *, region="all", kind="all", view="recent", limit=80):
    db = store.db
    limit = integer_param("limit", limit, 1, 400)
    if view not in ("recent", "backbone"):
        raise ValueError("view 须为 recent 或 backbone")
    where, args = _selector(kind, query, region, active=True)
    if address:
        # On an ego query kind identifies the centre. Its neighbours can have other kinds.
        neighbour = "" if region == "all" else """AND EXISTS (
            SELECT 1 FROM node other WHERE other.id=CASE WHEN e.src=:id THEN e.dst ELSE e.src END
            AND other.region=:region)"""
        hits = db.execute(f"""SELECT {_EDGE_COLUMNS} FROM edge e WHERE (e.src=:id OR e.dst=:id) {neighbour}
            ORDER BY e.active DESC,e.direct DESC,e.at DESC LIMIT :lim""",
            {"id": address, "region": region, "lim": limit})
    elif view == "backbone":
        hits = _backbone(db, where, args, limit)
    else:
        hits = db.execute(f"""{_CLOCKS}, seed AS (SELECT n.id FROM node n {_JOIN_CLOCK} WHERE {where}
                ORDER BY {_DISPLAY_TIME} DESC LIMIT :seeds)
            SELECT {_EDGE_COLUMNS} FROM edge e WHERE e.src IN seed OR e.dst IN seed
            ORDER BY e.active DESC,e.at DESC LIMIT :lim""",
            {**args, "seeds": max(1, limit * 2 // 5), "lim": limit})
    edges = [v for row in hits if (v := link(db, *row))]
    nodes = {
        x["id"]: dict(key=x["id"], label=x["title"], **x)
        for e in edges
        for x in (e["source_ref"], e["target_ref"])
    }
    return {"nodes": list(nodes.values()), "edges": edges, "limit": limit, "network": "plan_d"}


def item(store, k, detail=True, *, clock_row=None):
    db = store.db
    if k.startswith("graph~"):
        try:
            a, b, r = json.loads(base64.urlsafe_b64decode(k[6:] + "==="))
        except (ValueError, TypeError):
            raise ValueError("图关系地址无效")
        row = db.execute(
            "SELECT src,dst,rel,active,bin FROM edge WHERE src=? AND dst=? AND rel=?", (a, b, r)
        ).fetchone()
        e = link(db, *row) if row else None
        return dict(e, summary=r, title=r, navigation={"connections": [e]}) if e else None
    row = clock_row if clock_row is not None else db.execute(f"""{_CLOCKS}
        SELECT n.*,c.id IS NOT NULL AS backfill,c.recorded_at,{_DISPLAY_TIME} AS display_at
        FROM node n {_JOIN_CLOCK} WHERE n.id=? AND n.status<>'INVALIDATED'""", (k,)).fetchone()
    if not row:
        return None
    raw = [
        int(r[0][10:])
        for r in db.execute(
            """SELECT dst FROM edge WHERE src=? AND dst LIKE 'E|message:%'
                 AND rel NOT IN ('adjacent','REPLY_TO') ORDER BY dst""",
            (k,),
        )
    ]
    if k.startswith("E|message:"):
        raw = [int(k[10:])]
    result = dict(
        kind=row["kind"],
        id=k,
        title=row["body"][:72] or k,
        summary=row["body"],
        updated_at=row["recorded_at"] if row["backfill"] else row["at"],
        status=row["status"],
        revision=row["revision"],
        source_ids=raw,
    )
    if row["backfill"] and row["recorded_at"] is None:
        result.update(time_origin="historical_replay", evidence_at=row["display_at"])
    if detail:
        result["navigation"] = {"connections": graph(store, address=k)["edges"]}
        result["history"] = [
            {"revision": v, "previous": json.loads(old) if old else None, "at": at,
             **({"time_origin": "historical_replay"} if unknown else {})}
            for v, old, at, unknown in db.execute(
                _CLOCKS + """SELECT h.revision,h.previous,CASE WHEN c.id IS NULL THEN h.at ELSE c.recorded_at END,
                    c.id IS NOT NULL AND c.recorded_at IS NULL FROM change h
                    LEFT JOIN clocks c ON c.id=h.id AND c.revision=h.revision AND c.at=h.at
                    WHERE h.id=? ORDER BY h.revision""", (k,)
            )
        ]
    return result


def item_count(store, kind, query, region):
    where, args = _selector(kind, query, region, active=False)
    return store.db.execute(f"SELECT count(*) FROM node n WHERE {where}", args).fetchone()[0]


def items(store, kind, query, *, region="all", offset=0, limit=60):
    where, args = _selector(kind, query, region, active=False)
    rows = store.db.execute(f"""{_CLOCKS}
        SELECT n.*,c.id IS NOT NULL AS backfill,c.recorded_at,{_DISPLAY_TIME} AS display_at
        FROM node n {_JOIN_CLOCK} WHERE {where} ORDER BY display_at DESC,n.id LIMIT :lim OFFSET :off""",
        {**args, "lim": integer_param("limit", limit, 1, 200), "off": integer_param("offset", offset, 0)})
    return [item(store, row["id"], False, clock_row=row) for row in rows]
