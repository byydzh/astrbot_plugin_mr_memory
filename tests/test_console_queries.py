import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from astrbot_plugin_mr_memory.mr_memory import console
from astrbot_plugin_mr_memory.mr_memory.identity import IdentityBindings
from astrbot_plugin_mr_memory.mr_memory.network.storage import SCHEMA
from astrbot_plugin_mr_memory.mr_memory.store import Store
from tests.test_platform_migration import message
from tests import seed_legacy_scope


@pytest.fixture
def scope(tmp_path, request):
    if getattr(request, "param", None) == "legacy":
        seed_legacy_scope(tmp_path / "one.db", "test:GroupMessage:one")
    store = Store(tmp_path / "one.db", "test:GroupMessage:one")
    page = console.Console.__new__(console.Console)
    page.plugin = SimpleNamespace(scope_dir=tmp_path, identities=IdentityBindings(tmp_path / "bindings.json"))
    page.get_store = AsyncMock(return_value=store)
    try:
        yield store, page
    finally:
        store.close()


def call(page, handler, **params):
    with patch.object(console, "request", SimpleNamespace(query=params)):
        return asyncio.run(getattr(page, handler)("one"))


def add_node(db, key, kind, stamp, status="ACTIVE"):
    db.execute("INSERT INTO node(id,region,kind,body,at,status) VALUES(?,?,?,?,?,?)",
               (key, key.split("|")[0], kind, key, stamp, status))


def test_replay_clock_never_becomes_an_update_time(scope):
    from astrbot_plugin_mr_memory.mr_memory.network.console_view import item
    from astrbot_plugin_mr_memory.mr_memory.network import recall

    store, page = scope
    db = store.db
    db.executescript(SCHEMA)
    store.network_enabled = True
    with db:
        add_node(db, "M|old", "episode", 900)
        add_node(db, "K|old", "episode", 900)
        add_node(db, "S|live", "understanding", 200)
        add_node(db, "S|recorded", "understanding", 90)
        add_node(db, "E|message:1", "message", 100)
        db.execute("INSERT INTO edge(src,dst,rel) VALUES('M|old','E|message:1','source')")
        db.execute("INSERT INTO change(id,revision,at) VALUES('M|old',1,900)")
        for k, stamp, receipt in [
            ("history-1", 900, {"edits": [{"id": "M|old", "revision": 1}], "mirrors": ["K|old"]}),
            ("history-2", 90, {"edits": [{"id": "S|recorded", "revision": 1}], "recorded_at": 150})]:
            db.execute("INSERT INTO batch VALUES(?,?,?)", (k, stamp, json.dumps(receipt)))
        db.execute("UPDATE node SET subjects='[]'")
    rows = call(page, "memories", region="all")["items"]
    assert [r["id"] for r in rows] == ["S|live", "S|recorded", "M|old", "K|old"]
    assert rows[1]["updated_at"] == 150
    assert rows[2]["updated_at"] is None and rows[2]["evidence_at"] == 100
    assert rows[3]["updated_at"] is None and rows[3]["evidence_at"] is None
    assert item(store, "M|old")["history"][0]["at"] is None
    with patch.object(recall.time, "time", return_value=250):
        entrances = {n["id"]: n for n in recall.locate(store, "", [], {}, [])["旧图入口"]}
    assert entrances["M|old"]["at"] is None and entrances["M|old"]["evidence_at"] == 100
    assert entrances["S|recorded"]["at"] == 150 and entrances["S|live"]["at"] == 200
    opened = recall.lookup(store, {"read": ["K|old"]}, [], "")["K|old"]
    assert opened["at"] is None and opened["evidence_at"] is None
    with db:
        db.execute("UPDATE node SET revision=2,at=230 WHERE id='M|old'")
        db.execute("INSERT INTO change VALUES('M|old',2,230,'{}')")
    current = item(store, "M|old")
    assert current["updated_at"] == 230
    assert [r["at"] for r in current["history"]] == [None, 230]


def test_scope_activity_uses_visible_message_time(scope, tmp_path):
    store, page = scope
    Store(tmp_path / "two.db", "test:GroupMessage:two").close()
    for mid, stamp in [("late", 200), ("backfill", 100), ("removed", 300)]:
        row = store.append_message(message(store.umo, "10", "同学", mid, stamp))
    with store.db:
        store.db.execute("UPDATE messages SET is_deleted=1 WHERE id=?", (row["id"],))
    scopes, errors = page.discover()
    assert not errors
    assert {s["id"]: (s["last_message_at"], s["message_count"]) for s in scopes} == {
        "one": (200, 2), "two": (None, 0)}


def test_scope_activity_follows_receiving_endpoint_after_binding(scope, tmp_path):
    shared, page = scope
    endpoint = Store(tmp_path / "alias.db", "new:GroupMessage:open")
    try:
        page.plugin.identities.bind(endpoint.umo, shared.umo, [])
        historical = message(shared.umo, "a", "旧端", "old", 100)
        shared.append_message(historical)
        ingested = message(endpoint.umo, "b", "新端", "copied", 200)
        endpoint.append_message(ingested)
        shared.append_message(dict(ingested, umo=shared.umo))
        shared.append_message(dict(ingested, umo=shared.umo, message_id="live", sent_at=900000))
        shared.append_message(dict(historical, role="SYSTEM", message_id="manual", sent_at=910000))
        with patch.object(console.time, "time", return_value=920000):
            scopes, errors = page.discover()
        assert not errors
        assert {s["umo"]: (s["last_message_at"], s["message_count"], s["recent_message_count"])
                for s in scopes} == {shared.umo: (100, 1, 0), endpoint.umo: (900000, 2, 1)}
    finally:
        endpoint.close()


def test_memories_pages_include_retained_records_without_overlap(scope):
    store, page = scope
    store.db.executescript(SCHEMA)
    store.network_enabled = True
    with store.db:
        for i in range(204):
            add_node(store.db, f"M|episode:{i:03}", "episode", 200)
        add_node(store.db, "S|person", "understanding", 150)
        add_node(store.db, "K|rule", "veto", 120)
        add_node(store.db, "M|retired", "episode", 300, "INVALIDATED")
        add_node(store.db, "E|message:1", "message", 400)
        store.db.executemany("""INSERT INTO mr_memory_objects
            (umo,kind,id,title,content,created_at,updated_at) VALUES(?,'episode',?,'old','old',1,1)""",
            [(store.umo, i) for i in range(1, 64)])
    old = call(page, "memories")
    assert len(old["items"]) == 120 and old["has_more"]
    assert [r["id"] for r in old["items"][60:]] == list(range(63, 3, -1))
    first = call(page, "memories", region="all", limit="999")
    second = call(page, "memories", region="all", limit="200", offset="200")
    assert first["has_more"] and len(first["items"]) == 200 and not second["has_more"]
    assert [r["id"] for r in first["items"] + second["items"]] == [
        *(f"M|episode:{i:03}" for i in range(204)), "S|person", "K|rule", *range(63, 0, -1)]
    assert call(page, "memories", region="all", offset="269") == {"items": [], "has_more": False}
    assert call(page, "memories", region="S", kind="veto") == {"items": [], "has_more": False}
    assert [r["id"] for r in call(page, "memories", region="M", kind="episode",
                                   query="episode:203")["items"]] == ["M|episode:203"]
    with pytest.raises(ValueError, match="offset"):
        call(page, "memories", offset="-1")


def test_graph_backbone_samples_old_bridges_and_global_hubs(scope):
    store, page = scope
    store.db.executescript(SCHEMA)
    store.network_enabled = True
    jan, mar, dec = 1705276800, 1710460800, 1734220800
    with store.db:
        for key in ("M|a", "M|b", "M|c", "M|d"):
            add_node(store.db, key, "episode", mar)
        store.db.execute("INSERT INTO network_l1_clock VALUES('M|a',?)", (jan,))
        add_node(store.db, "S|hub", "understanding", mar)
        for i in range(5):
            add_node(store.db, f"K|{i}", "scope", mar)
        def edge(a, b, slot, stamp, strength=1):
            store.db.execute("""INSERT INTO edge(src,dst,rel,bin,at,strength,active)
                VALUES(?,?,'connection',?,?,?,1)""", (a, b, slot, stamp, strength))
        edge("M|a", "M|b", "far", mar)
        edge("M|c", "M|d", "far", mar, 10)  # Strong but within the same month.
        for i in range(5):
            edge("S|hub", f"K|{i}", "cross", mar)
        for i in range(45):
            add_node(store.db, f"M|new{i}", "episode", dec + i)
            if i:
                edge(f"M|new{i-1}", f"M|new{i}", "local", dec + i)
    default = call(page, "graph")
    assert default == call(page, "graph", view="recent", limit="80")
    assert all(e["source_key"].startswith("M|new") for e in default["edges"])
    sampled = call(page, "graph", view="backbone", limit="2")
    assert (sampled["edges"][0]["source_key"], sampled["edges"][0]["target_key"]) == ("M|a", "M|b")
    assert sampled["edges"][1]["source_key"] == "S|hub"
    scoped = call(page, "graph", region="S", kind="understanding", view="backbone", limit="800")
    assert scoped["limit"] == 400 and len(scoped["edges"]) == 5
    assert all(e["target_key"].startswith("K|") for e in scoped["edges"])
    assert call(page, "graph", region="S", kind="episode")["edges"] == []
    assert len(call(page, "graph", id="S|hub", kind="understanding")["edges"]) == 5
    assert not call(page, "graph", id="S|hub", kind="understanding", region="M")["edges"]
    with pytest.raises(ValueError, match="view"):
        call(page, "graph", view="bogus")

    with store.db:
        store.db.execute("DROP TABLE network_l1_clock")
        store.db.execute("UPDATE node SET at=? WHERE id='M|a'", (jan,))
    assert call(page, "graph", view="backbone", limit="2") == sampled


@pytest.mark.parametrize("scope", ["legacy"], indirect=True)
def test_legacy_graph_filters_before_sampling(scope):
    store, page = scope
    jan, jun = 1705276800, 1718409600
    objects = [("episode", 1, {}, jan), ("node", 2, {}, jun), ("semantic", 3, {}, jun)]
    for i, (kind, owner) in enumerate([("episode", 1), ("semantic", 3)], start=10):
        objects.append(("association", i, {"source_ref": {"kind": "node", "id": 2},
                                           "target_ref": {"kind": kind, "id": owner}}, jun))
    with store.db:
        store.db.executemany("""INSERT INTO mr_memory_objects
            (umo,kind,id,title,content,attributes_json,created_at,updated_at) VALUES(?,?,?,'title','text',?,?,?)""",
            [(store.umo, k, i, json.dumps(data), stamp, stamp) for k, i, data, stamp in objects])
    assert [e["id"] for e in call(page, "graph", view="recent", limit="1")["edges"]] == [11]
    assert [e["id"] for e in call(page, "graph", view="backbone", limit="1")["edges"]] == [10]
    assert [e["id"] for e in call(page, "graph", kind="episode", limit="1")["edges"]] == [10]
    assert [e["id"] for e in call(page, "graph", kind="semantic", view="backbone")["edges"]] == [11]
    assert call(page, "graph", region="M")["edges"] == []
