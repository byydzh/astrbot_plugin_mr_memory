"""One deterministic two-ended routing projection for imports and incremental transactions."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import heapq

DEAD = {"RETRACTED", "INVALIDATED", "SUPERSEDED"}


def bucket(node, slot):
    return "inside" if slot == "far" and node[0] != "M" else slot


def capacity(node, slot):
    return (4 if node[0] == "M" else 8) if slot == "inside" else 4


def month_ranks(db):
    groups = defaultdict(list)
    for a, b, ta, tb in db.execute("""select e.src,e.dst,a.at,b.at from edge e join node a on a.id=e.src
      join node b on b.id=e.dst where e.bin='far' order by e.strength desc,e.at desc,e.src,e.dst,e.rel"""):
        for u, t in ((a, tb), (b, ta)):
            groups[u, datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m")].append((a, b))
    ranks = {}
    for group in groups.values():
        for n, e in enumerate(group):
            ranks[e] = max(n, ranks.get(e, 0))
    return ranks


def rank(e, far):
    a, b, r, slot, d, s, t, on = e
    return far.get((a, b), 0), -d, -s, -t, a, b, r


COLUMNS = "src,dst,rel,bin,direct,strength,at,active"


def full(db, write=False):
    status = dict(db.execute("select id,status from node"))
    far = month_ranks(db)
    used = Counter()
    selected = set()
    rows = sorted(db.execute("select " + COLUMNS + " from edge"), key=lambda e: rank(e, far))
    changed = []
    for a, b, rel, slot, _, _, _, was in rows:
        x, y = (a, bucket(a, slot)), (b, bucket(b, slot))
        on = (
            slot != "source_only"
            and a != b
            and status[a] not in DEAD
            and status[b] not in DEAD
            and used[x] < capacity(*x)
            and used[y] < capacity(*y)
        )
        if on:
            selected.add((a, b, rel))
            used[x] += 1
            used[y] += 1
        if on != was:
            changed.append((int(on), a, b, rel))
    if write:
        db.executemany("update edge set active=? where src=? and dst=? and rel=?", changed)
    return selected, len(changed)


def repair(db, touched, old_far):
    """Only a lower-priority competitor at a shared slot can change after a decision flips."""
    far = month_ranks(db)
    for pair in old_far.keys() | far.keys():
        if old_far.get(pair, 0) != far.get(pair, 0):
            touched.update(pair)
    edges = {}
    active = {}
    heap = []
    queued = set()
    examined = 0
    flips = 0

    @lru_cache(None)
    def near(u):
        out = []
        for col in ("src", "dst"):
            for row in db.execute("select " + COLUMNS + " from edge where " + col + "=?", (u,)):
                k = tuple(row[:3])
                if k not in edges:
                    edges[k] = row
                    active[k] = bool(row[-1])
                out.append(k)
        return out

    @lru_cache(None)
    def alive(u):
        return db.execute("select status from node where id=?", (u,)).fetchone()[0] not in DEAD

    def queue(k):
        if k not in queued:
            queued.add(k)
            heapq.heappush(heap, (rank(edges[k], far), k))

    for u in touched:
        for k in near(u):
            queue(k)
    while heap:
        order, k = heapq.heappop(heap)
        queued.remove(k)
        examined += 1
        a, b, _ = k
        slot = edges[k][3]
        on = slot != "source_only" and a != b and alive(a) and alive(b)
        if on:
            for u in (a, b):
                lane = bucket(u, slot)
                taken = sum(
                    active[x] and bucket(u, edges[x][3]) == lane and rank(edges[x], far) < order
                    for x in near(u)
                )
                if taken >= capacity(u, lane):
                    on = False
                    break
        if active[k] == on:
            continue
        active[k] = on
        flips += 1
        db.execute("update edge set active=? where src=? and dst=? and rel=?", (int(on), *k))
        for u in (a, b):
            for x in near(u):
                if bucket(u, edges[x][3]) == bucket(u, slot) and rank(edges[x], far) > order:
                    queue(x)
    return {"examined": examined, "changed": flips}
