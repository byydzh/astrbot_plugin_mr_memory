"""MR Memory tests."""

import sqlite3
from contextlib import closing


def seed_legacy_scope(path, umo):
    """Write the persisted scope marker that existing pre-Plan-D groups have."""
    from mr_memory.store import SCHEMA
    with closing(sqlite3.connect(path)) as db, db:
        db.executescript(SCHEMA)
        platform, _, group = umo.split(":")
        db.execute("INSERT INTO scope_meta(singleton,umo,platform_id,group_id) VALUES(1,?,?,?)",
                   (umo, platform, group))
