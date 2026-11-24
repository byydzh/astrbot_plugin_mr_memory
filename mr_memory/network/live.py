"""Translate the canonical scope's real messages into graph handles and back."""

from .base import packed
from .storage import opened
from .projection import repair, month_ranks


def token(db, account):
    if not account:
        return ""
    row = db.execute("SELECT token FROM network_account WHERE account=?", (account,)).fetchone()
    if row:
        return row[0]
    last = db.execute("SELECT max(CAST(substr(token,2) AS INTEGER)) FROM network_account").fetchone()[0] or 0
    handle = f"u{last+1:04d}"
    db.execute("INSERT INTO network_account VALUES(?,?)", (account, handle))
    return handle


def raw_view(store, row):
    db = store.db
    account = row["sender_id"]
    actor = token(db, account)
    people = [
        token(db, p[0])
        for p in db.execute(
            """SELECT p.account_id FROM message_participants m JOIN participants p
       ON p.id=m.participant_id WHERE m.message_id=? AND m.relation='MENTIONED' ORDER BY m.position""",
            (row["id"],),
        )
    ]
    reply = db.execute(
        """SELECT target_message_id FROM message_relations WHERE source_message_id=?
        AND relation IN ('REPLY_TO','RESPONDS_TO') AND target_message_id IS NOT NULL LIMIT 1""",
        (row["id"],),
    ).fetchone()
    return dict(
        id=f"E|message:{row['id']}",
        uid=actor,
        name=row["sender_name"],
        text=row["plain_text"],
        at=row["sent_at"],
        role=row["role"],
        mentions=people,
        reply=f"E|message:{reply[0]}" if reply else None,
    )


def sync_accounts(store):
    # The identity panel is authoritative even when a previous binding is removed.
    store.db.execute("DELETE FROM identity")
    for a, b in getattr(store, "identity_links", {}).items():
        store.db.execute(
            "INSERT OR IGNORE INTO identity VALUES(?,?)", (token(store.db, a), token(store.db, b))
        )


def captured(store, row):
    db = store.db
    val = raw_view(store, row)
    k, at = val["id"], val["at"]
    prior = db.execute("SELECT at,revision FROM node WHERE id=?", (k,)).fetchone()
    if prior and prior[1] == row["revision_no"]:
        return
    db.execute(
        """INSERT INTO node VALUES(?,'E','message',?,?,?,'ACTIVE',?)
       ON CONFLICT(id) DO UPDATE SET body=excluded.body,at=excluded.at,subjects=excluded.subjects,
       revision=excluded.revision""",
        (k, val["text"], at, packed([val["uid"]]), row["revision_no"]),
    )
    db.execute("INSERT OR IGNORE INTO subject VALUES(?,?)", (val["uid"], k))
    net = opened(db)
    net.touched = {k}
    prev = db.execute(
        """SELECT id,at FROM node WHERE region='E' AND kind='message' AND status='ACTIVE'
            AND at BETWEEN ? AND ? AND id<>? ORDER BY at DESC,rowid DESC LIMIT 1""",
        (at - 600, at, k),
    ).fetchone()
    if prev:
        net._link(k, prev[0], "adjacent", at, False)
    if val["reply"] and net.read(val["reply"]):
        net._link(k, val["reply"], "REPLY_TO", at, True)
    repair(db, net.touched, month_ranks(db))


def catch_up(store):
    db = store.db
    cursor = db.execute("SELECT cursor FROM network_ingest").fetchone()
    floor = (
        cursor[0]
        if cursor
        else db.execute("SELECT watermark FROM network_scope WHERE umo=?", (store.umo,)).fetchone()[0]
    )
    with db:
        for row in db.execute(
            "SELECT * FROM messages WHERE id>? AND is_deleted=0 ORDER BY id", (floor,)
        ).fetchall():
            captured(store, row)
        db.execute("INSERT OR REPLACE INTO network_ingest SELECT 1,coalesce(max(id),0) FROM messages")


def invalidate(store, mids, account=None):
    db = store.db
    db.execute("CREATE TEMP TABLE IF NOT EXISTS mr_invalid_graph(id TEXT PRIMARY KEY)")
    db.execute("DELETE FROM mr_invalid_graph")
    db.executemany(
        "INSERT OR IGNORE INTO mr_invalid_graph VALUES(?)", [(f"E|message:{mid}",) for mid in mids]
    )
    if account:
        db.execute(
            """INSERT OR IGNORE INTO mr_invalid_graph SELECT id FROM subject WHERE account=(
                   SELECT token FROM network_account WHERE account=?)""",
            (account,),
        )
    while db.execute(
        """INSERT OR IGNORE INTO mr_invalid_graph SELECT e.src FROM edge e
          JOIN mr_invalid_graph x ON e.dst=x.id WHERE e.rel IN
          ('source','same_l1','witness','backed_by','grounded_in','anchor','inherits','formed_from','recalls','interprets','inherited_anchor')"""
    ).rowcount:
        pass
    touched = {r[0] for r in db.execute("""SELECT src FROM edge WHERE dst IN mr_invalid_graph
              UNION SELECT dst FROM edge WHERE src IN mr_invalid_graph""")}
    db.execute("INSERT OR IGNORE INTO withdrawn SELECT id FROM mr_invalid_graph")
    db.execute("UPDATE node SET status='INVALIDATED',body='',subjects='[]' WHERE id IN mr_invalid_graph")
    for table in ("subject", "change", "network_vector", "network_l1_pending", "network_l1_clock"):
        db.execute(f"DELETE FROM {table} WHERE id IN mr_invalid_graph")
    if "imported_detail" in store.tables:
        db.execute("DELETE FROM imported_detail WHERE id IN mr_invalid_graph")
    db.execute("""DELETE FROM mr_network_run_changes WHERE EXISTS (SELECT 1 FROM json_tree(diff) j
                    JOIN mr_invalid_graph x ON x.id=j.atom)""")
    repair(db, touched, month_ranks(db))
