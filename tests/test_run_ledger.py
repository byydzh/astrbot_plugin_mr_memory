import asyncio
import json
from unittest.mock import AsyncMock

from mr_memory.store import Store
from mr_memory.run_ledger import STORAGE_KEY
from mr_memory.console import Console


def sample():
    body = {"title": "群里的桌游", "description": "北门见面后开始试玩。" * 150}
    text = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
    return body, {"status": "completed", "question": "在哪儿试玩", "request_id": "synthetic",
                  "tool_calls": [{"id": "call-1", "name": "memory", "result": body}],
                  "messages": [{"role": "tool", "tool_call_id": "call-1", "content": text},
                               {"role": "tool", "tool_call_id": "call-1", "content": text + "\n模型只看到这一版"}]}


def test_ledger_result_is_stored_once_and_both_views_round_trip(tmp_path):
    store = Store(tmp_path / "group.db", "test:GroupMessage:42")
    try:
        body, payload = sample()
        run = store.record_run("foreground", 1, {"trace_version": 1})
        store.append_run_step(run, 2, 2, "read", "completed", "读取 memory",
                              {"tool_call_id": "call-1", "name": "memory", "result": body})
        store.update_run(run, payload)
        store.update_run(run, {"detail": "最终回执"})
        raw = store.db.execute("SELECT payload_json FROM mr_runs WHERE id=?", (run,)).fetchone()[0]
        assert body["description"] not in raw
        assert json.loads(raw)[STORAGE_KEY]["results"] == [[0, 2]]
        assert STORAGE_KEY not in store.recent_runs()[0]
        store.db.execute("PRAGMA query_only=ON")
        detail = store.run_detail(run)
        assert detail["tool_calls"] == payload["tool_calls"]
        assert detail["messages"] == payload["messages"]
        assert detail["detail"] == "最终回执"
        assert store.reflections.interaction(run_id=run, step=2)["data"]["result"] == body
        view = store.reflections.interaction(run_id=run, detailed=True)["runs"][0]
        assert view["steps"][0]["data"]["result"] == body
    finally:
        store.close()


def test_missing_step_and_reused_call_id_never_substitute_another_result(tmp_path):
    store = Store(tmp_path / "group.db", "test:GroupMessage:42")
    try:
        body, payload = sample()
        run = store.record_run("background", 1, {})
        other = store.record_run("background", 2, {})
        store.append_run_step(other, 1, 3, "read", "completed", "读取 memory",
                              {"tool_call_id": "call-1", "name": "memory", "result": body})
        store.update_run(run, payload)
        assert STORAGE_KEY not in json.loads(store.db.execute(
            "SELECT payload_json FROM mr_runs WHERE id=?", (run,)).fetchone()[0])
        for seq, value in enumerate([{"title": "另一次"}, body], 1):
            store.append_run_step(run, seq, 3, "read", "completed", "读取 memory",
                                  {"tool_call_id": "call-1", "name": "memory", "result": value})
        store.update_run(run, payload)
        assert store.run_detail(run)["tool_calls"] == payload["tool_calls"]
        assert store.run_detail(run)["messages"] == payload["messages"]
    finally:
        store.close()


def test_old_inline_ledger_opens_read_only_in_console_and_interaction(tmp_path):
    store = Store(tmp_path / "old.db", "test:GroupMessage:42")
    try:
        body, payload = sample()
        run = store.record_run("foreground", 1, payload)
        store.append_run_step(run, 1, 2, "read", "completed", "读取 memory",
                              {"tool_call_id": "call-1", "name": "memory", "result": body})
        store.db.execute("PRAGMA query_only=ON")
        changes = store.db.total_changes
        console = object.__new__(Console)
        console.get_store = AsyncMock(return_value=store)
        details = asyncio.run(console.run("test", run))
        assert details["messages"] == payload["messages"]
        assert details["tool_calls"] == payload["tool_calls"]
        assert store.reflections.interaction(run_id=run, detailed=True)["runs"][0]["steps"] == details["steps"]
        assert store.db.total_changes == changes
    finally:
        store.close()



def test_archived_understanding_and_reflection_are_readable_without_experimental_writes(tmp_path):
    store = Store(tmp_path / "archive.db", "test:GroupMessage:42")
    try:
        # Columns and JSON encodings are those of the production-era tables.
        with store.db:
            for kind in ("understanding", "reflection"):
                store.db.execute("INSERT INTO mr_memory_objects VALUES(?,?,?,?,?,?,1,'ACTIVE',1,1,0)",
                                 (store.umo, kind, 9, "旧认识", "当时的解释", "{}"))
            store.db.execute("INSERT INTO mr_working_set VALUES(?,?,?,?,?)",
                             (store.umo, "understanding", 9, '{"reason":"旧关注"}', 1))
            store.db.execute("""INSERT INTO mr_cognition_turns
                (umo,run_key,kind,at,payload_json) VALUES(?,?,?,?,?)""",
                (store.umo, "old", "foreground", 1, '{"available_memories":[{"kind":"understanding","id":9}]}'))
        store.db.execute("PRAGMA query_only=ON")
        before = store.db.total_changes
        console = object.__new__(Console)
        console.get_store = AsyncMock(return_value=store)
        archived = asyncio.run(console.memory("test", "understanding", 9))
        assert archived["content"] == "当时的解释"
        assert archived["experiences"]["items"][0]["available_memories"][0]["id"] == 9
        assert store.cognition.archived_workspace()["active"][0]["memory"]["attention"]["reason"] == "旧关注"
        assert store.reflections.get(9)["content"] == "当时的解释"
        assert store.db.total_changes == before
    finally:
        store.close()
