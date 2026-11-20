import asyncio
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from astrbot.api.star import Context
from astrbot_plugin_mr_memory import main
from astrbot_plugin_mr_memory.mr_memory import console
from tests.test_main import event
from tests.test_platform_migration import message


def test_registered_console_routes_and_account_commands(tmp_path):
    asyncio.run(exercise_routes(tmp_path))


async def exercise_routes(root):
    host = Mock(spec=Context)
    host.registered_web_apis = []
    host.register_web_api.side_effect = lambda *route: host.registered_web_apis.append(route)
    host.get_platform_inst.return_value.meta.return_value.name = "aiocqhttp"
    manager = host.conversation_manager = Mock()
    manager.get_curr_conversation_id = AsyncMock(return_value="old")
    manager.get_conversation = AsyncMock(return_value=SimpleNamespace(
        title="history", history='[{"role":"user","content":"history"}]', platform_id="old", persona_id=None))
    manager.new_conversation = AsyncMock(return_value="fresh")
    with patch.object(main, "get_astrbot_data_path", return_value=str(root)):
        plugin = main.MrMemoryPlugin(host, {"embedding_enabled": False, "advanced_memory_enabled": False,
            "local_serving_enabled": True, "capture_enabled": True, "history_recovery_enabled": True,
            "subconscious_enabled": True, "auto_distillation_enabled": True, "learning_window_enabled": False})
    routes = {handler.args[0].__name__: (method[0], handler) for _, handler, method, _ in host.registered_web_apis}
    visited = set()

    async def call(name, verb="GET", body=None, **params):
        method, handler = routes[name]
        assert method == verb
        with patch.object(console, "request", SimpleNamespace(query={}, json=AsyncMock(return_value=body))):
            result = await handler(**params)
        assert result["status"] == "success", (name, result)
        visited.add(name)
        return result["data"]

    def seed(umo, uid):
        scope_id = hashlib.sha256(umo.encode()).hexdigest()
        store = main.Store(plugin.scope_dir / (scope_id + ".db"), umo)
        plugin.stores[umo] = store
        plugin.state_locks[umo] = asyncio.Lock()
        row = store.append_message(message(umo, uid, "同学", "first"))
        return {"scope_id": scope_id}, store, row

    try:
        old, store, row = seed("old:GroupMessage:123", "100")
        new, current, _ = seed("new:GroupMessage:open", "OPEN1")
        item = store.save_memories([dict(kind="episode", content="共同观星", source_ids=[row["id"]])])[0]
        rid = store.record_run("foreground", 1, {"status":"completed"})
        assert len((await call("overview"))["scopes"]) == 2
        assert (await call("scope_overview", **old))["counts"]["episodes"] == 1
        assert (await call("runs", **old))["runs"]
        assert (await call("run", run_id=rid, **old))["status"] == "completed"
        assert (await call("memories", **old))["items"]
        assert (await call("memory", kind="episode", item_id=item["id"], **old))["content"] == "共同观星"
        assert "edges" in await call("graph", **old)
        assert (await call("participants", **old))["participants"]
        await call("bind_alias", "POST", {"account_id":"100", "alias":"同窗"}, **old)
        await call("edit_profile", "POST", {"account_id":"100", "changes":{"description":"观星"}}, **old)
        profile = store.members(["100"])[0]
        assert "同窗" in profile["confirmed_aliases"] and profile["description"] == "观星"
        assert (await call("messages", **old))["messages"][0]["id"] == row["id"]
        assert (await call("message_context", message_id=row["id"], **old))["messages"]

        class Provider:
            provider_config = {}
            async def _prepare_chat_payload(self, **kwargs):
                return {"messages":kwargs["contexts"]}, None
            async def _query(self, *args, **kwargs):
                data = {"items":[{"kind":"episode","content":"再次观星","source_ids":[row["id"]]}],
                        "progress":{"completed_ids":[row["id"]], "checkpoint":"done"}}
                return SimpleNamespace(completion_text=json.dumps(data), tools_call_name=[], usage={"output":40})
        # Make the second message available to the actual manual consolidator.
        row = store.append_message(message(store.umo, "100", "同学", "second", 110))
        host.get_provider_by_id.return_value = Provider()
        assert (await call("distill", "POST", **old))["written_count"] == 1
        assert (await call("reset_conversation", "POST", **old))["preserved_messages"] == 1
        assert (await call("recover_history", "POST", **old))["queued"]
        assert store.umo in plugin.history.requested
        assert not (await call("migration", **new))["bound"]
        assert (await call("migration_bind", "POST", {"legacy_scope":old["scope_id"]}, **new))["imported_messages"] == 1
        await call("migration_member", "POST", {"account_id":"OPEN1","legacy_id":"100"}, **new)
        assert (await call("migration", **new))["matches"]["OPEN1"]["method"] == "manual"
        assert visited == routes.keys()

        current_event = event()
        current_event.message_str = "/uid 称呼 小林"
        replies = [r async for r in plugin.uid(current_event)]
        assert replies and "小林" in replies[0].chain[0].text
        owned = await plugin.store_for(current_event)
        assert owned.members(["1001"])[0]["preferred_name"] == "小林"
        replies = [r async for r in plugin.forget_me(current_event)]
        assert replies and "已删除" in replies[0].chain[0].text
        assert not owned.search_messages(sender_id="1001")
        assert store.search_messages(sender_id="100")
    finally:
        await plugin.terminate()
    assert not host.registered_web_apis
