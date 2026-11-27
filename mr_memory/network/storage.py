"""Keep graph edits, queue progress and receipts inside the group database."""

import itertools
import json
from types import SimpleNamespace

import numpy as np

from .typed import AlignedGraph

SCHEMA = """
CREATE TABLE IF NOT EXISTS node(id TEXT PRIMARY KEY,region TEXT,kind TEXT,body TEXT,at INTEGER,subjects TEXT,status TEXT,revision INTEGER NOT NULL DEFAULT 1);
CREATE TABLE IF NOT EXISTS edge(src TEXT,dst TEXT,rel TEXT,bin TEXT,w REAL,strength INTEGER,at INTEGER,active INTEGER DEFAULT 0,direct INTEGER DEFAULT 0,PRIMARY KEY(src,dst,rel)) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS incoming ON edge(dst,active);
CREATE INDEX IF NOT EXISTS forward ON edge(src,active);
CREATE INDEX IF NOT EXISTS temporal ON node(region,at);
CREATE TABLE IF NOT EXISTS edge_basis(src TEXT,dst TEXT,rel TEXT,history TEXT,PRIMARY KEY(src,dst,rel)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS subject(account TEXT NOT NULL,id TEXT NOT NULL REFERENCES node(id),PRIMARY KEY(account,id)) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS subject_node ON subject(id);
CREATE TABLE IF NOT EXISTS change(id TEXT NOT NULL,revision INTEGER NOT NULL,at INTEGER NOT NULL,previous TEXT,PRIMARY KEY(id,revision)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS batch(k TEXT PRIMARY KEY,at INTEGER NOT NULL,receipt TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS identity(a TEXT NOT NULL,b TEXT NOT NULL,PRIMARY KEY(a,b)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS withdrawn(id TEXT PRIMARY KEY) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS derived_shortcut(via TEXT,src TEXT,dst TEXT,preexisting INTEGER,PRIMARY KEY(via,src,dst)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS network_account(account TEXT PRIMARY KEY,token TEXT NOT NULL UNIQUE);
CREATE TABLE IF NOT EXISTS network_vector(id TEXT PRIMARY KEY,revision INTEGER NOT NULL,vector BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS network_l1_pending(id TEXT PRIMARY KEY,run_id INTEGER,day TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS network_work(singleton INTEGER PRIMARY KEY CHECK(singleton=1),horizon INTEGER NOT NULL,started INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS network_inputs(message_id INTEGER PRIMARY KEY,digest TEXT NOT NULL,run_id INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS network_consumed(message_id INTEGER PRIMARY KEY,digest TEXT NOT NULL,run_id INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS network_ingest(singleton INTEGER PRIMARY KEY CHECK(singleton=1),cursor INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS network_l1_clock(id TEXT PRIMARY KEY,at INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS network_clock_order ON network_l1_clock(at,id);
CREATE TABLE IF NOT EXISTS network_scope(umo TEXT PRIMARY KEY,model TEXT NOT NULL,watermark INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS routing_far ON edge(strength,at) WHERE bin='far';
"""


class AtomicConnection:
    """Nested graph operations must not commit their caller's SQLite transaction."""

    def __init__(self, db):
        self.raw = db
        self.levels = []
        self.names = itertools.count()

    def __getattr__(self, name):
        return getattr(self.raw, name)

    @property
    def row_factory(self):
        return self.raw.row_factory

    @row_factory.setter
    def row_factory(self, value):
        self.raw.row_factory = value

    def __enter__(self):
        name = f"mr_graph_{next(self.names)}"
        self.raw.execute("SAVEPOINT " + name)
        self.levels.append(name)
        return self

    def __exit__(self, typ, value, tb):
        name = self.levels.pop()
        if typ:
            self.raw.execute("ROLLBACK TO " + name)
        self.raw.execute("RELEASE " + name)


def opened(db):
    return AlignedGraph(AtomicConnection(db))


def upgrade_vector_space(db, umo):
    """Extend the three-column snapshot/import schema when opening a serving database."""
    def has_dimensions():
        return any(row[1] == "dimensions" for row in db.execute("PRAGMA table_info(network_scope)"))

    if has_dimensions():
        return
    with db:
        db.execute("BEGIN IMMEDIATE")
        if has_dimensions():
            return
        db.execute("ALTER TABLE network_scope ADD COLUMN dimensions INTEGER NOT NULL DEFAULT 0")
        row = db.execute("SELECT length(vector) FROM network_vector LIMIT 1").fetchone()
        if row:
            width, remainder = divmod(row[0], 4)
            if not width or remainder:
                raise ValueError("旧图中的向量不是有效的 float32 向量")
        else:
            raw = db.execute("""SELECT e.dimensions FROM memory_embeddings e JOIN network_scope s
                ON s.umo=e.umo AND s.model=e.model WHERE s.umo=? AND e.owner_type='message' LIMIT 1""",
                (umo,)).fetchone()
            width = raw[0] if raw else 0
        db.execute("UPDATE network_scope SET dimensions=? WHERE umo=?", (width, umo))


def bind_vector_space(db, umo, model, dimensions):
    """Pin an empty index on first use. Never relabel vectors from a previous space."""
    if not model or dimensions <= 0:
        raise ValueError("向量编码器须提供模型标识和有效维度")
    current = db.execute("SELECT model,dimensions FROM network_scope WHERE umo=?", (umo,)).fetchone()
    if current and tuple(current) == (model, dimensions):
        return
    updated = db.execute("""UPDATE network_scope SET model=?, dimensions=?
        WHERE umo=? AND model IN ('',?) AND dimensions IN (0,?)""", (model, dimensions, umo, model, dimensions))
    if not updated.rowcount:
        raise ValueError(f"向量空间不匹配：群库 {tuple(current) if current else None}，编码器 {(model, dimensions)}")


def upper_context(db, ops, ids, queries, raw):
    from .consolidation import context

    arrays = {region: ([], []) for region in "MSK"}
    for k, blob in db.execute(
        """SELECT v.id,v.vector FROM network_vector v JOIN node n
             ON v.id=n.id AND v.revision=n.revision WHERE n.region IN ('M','S','K')
             AND n.status='ACTIVE' AND NOT(n.region='K' AND n.kind IN ('episode','event','semantic')) ORDER BY n.id"""
    ):
        names, vec = arrays[k[0]]
        names.append(k)
        vec.append(np.frombuffer(blob, dtype="<f4"))
    state = SimpleNamespace(
        db=db,
        net=opened(db),
        idx=SimpleNamespace(arrays=arrays, overlay={}),
        query_vectors=np.array(queries, dtype=np.float32),
        rows=raw,
    )
    return context(
        state, [{"accepted_plan": {"changes": ops}, "receipt": {"edits": [{"id": key} for key in ids]}}]
    )


def upsert_vectors(db, keys, vectors):
    dimensions = db.execute("SELECT dimensions FROM network_scope").fetchone()[0]
    for k, v in zip(keys, vectors, strict=True):
        row = db.execute("SELECT revision FROM node WHERE id=?", (k,)).fetchone()
        arr = np.asarray(v, dtype="<f4")
        if not row or arr.shape != (dimensions,) or not np.isfinite(arr).all():
            raise ValueError(f"节点 {k} 不存在或向量无效（群库维度 {dimensions}，收到 {arr.shape}）")
        db.execute("INSERT OR REPLACE INTO network_vector VALUES(?,?,?)", (k, row[0], arr.tobytes()))


def parse_plan(text):
    s = text.strip()
    if s.startswith("```") and s.endswith("```"):
        s = s.split("\n", 1)[1][:-3].strip()
    result = json.loads(s)
    if not isinstance(result, dict):
        raise ValueError("Expected a graph operation object")
    return result
