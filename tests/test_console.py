"""One synthetic path through the restored console, including reload cleanup."""
import asyncio
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import AsyncMock, patch

from mr_memory.console import Console
from mr_memory.store import Store


@contextmanager
def scratch_directory():
    scratch = Path(__file__).resolve().parents[1] / ".pytest_cache" / ("console-" + uuid.uuid4().hex)
    scratch.mkdir(parents=True)
    try:
        yield scratch
    finally:
        for path in scratch.iterdir():
            path.unlink()
        scratch.rmdir()


class ConsoleTests(unittest.IsolatedAsyncioTestCase):
    async def test_read_manage_and_reload(self):
        with scratch_directory() as directory:
            path = Path(directory) / "synthetic.db"
            scope = "test:GroupMessage:42"
            store = Store(path, scope)
            for i in range(102):
                store.append_message(dict(platform="test", platform_id="test", umo=scope,
                    group_id="42", message_id=str(i), sender_id="100", sender_name="合成成员",
                    sent_at=1800000000, plain_text=f"合成消息 {i}", content=[], role="USER"))
            run_id = store.record_run("foreground", 1800000001, dict(status="completed", question="合成问题",
                background="合成背景", elapsed_ms=1400, usage={"input_tokens": 10}, tool_calls=[{"name": "members"}]))
            store.close()
            context = SimpleNamespace(registered_web_apis=[])
            def register(route, handler, methods, desc):
                for index, old in enumerate(context.registered_web_apis):
                    if old[0] == route and old[2] == methods:
                        context.registered_web_apis[index] = (route, handler, methods, desc)
                        return
                context.registered_web_apis.append((route, handler, methods, desc))
            context.register_web_api = register
            plugin = SimpleNamespace(context=context, scope_dir=Path(directory), config={"local_serving_enabled": False},
                stopping=False, stores={}, open_lock=asyncio.Lock(), state_locks={},
                embedder=SimpleNamespace(_model=object()),
                consolidate=AsyncMock(return_value={"status": "completed", "written_count": 1}))
            console = Console(plugin)
            with patch("mr_memory.console.request", SimpleNamespace(query={}, json=AsyncMock(return_value={"account_id": "100", "alias": "合成别名"}))):
                overview = await console.overview()
                self.assertFalse(overview["runtime"]["recall"])
                self.assertEqual(overview["scopes"][0]["id"], "synthetic")
                opened = await console.get_store("synthetic")
                self.assertIs(opened, plugin.stores[scope])
                first = await console.messages("synthetic")
                second = Console.read_messages(opened, "", first["next_before_id"])
                self.assertEqual(len(first["messages"]), 100)
                self.assertEqual(len(second["messages"]), 2)
                self.assertFalse(set(m["id"] for m in first["messages"]) & set(m["id"] for m in second["messages"]))
                await console.bind_alias("synthetic")
                self.assertIn("合成别名", (await console.participants("synthetic"))["participants"][0]["aliases"])
                self.assertEqual((await console.run("synthetic", str(run_id)))["background"], "合成背景")
                self.assertEqual((await console.scope_overview("synthetic"))["counts"]["pending"], 102)
                opened.start_learning_task("background", [1, 2])
                opened.save_learning_progress("background", {"completed_ids": [1], "checkpoint": "下一次继续理解第二条"}, run_id)
                opened.update_learning_task("background", {"conversation": [{"role": "user", "content": "完整模型输入"}]})
                plugin.learning_status = {scope: {"background": {"status": "deferred", "reason": "等待工作时段", "next_start": 1800000000}}}
                learning = await console.runs("synthetic")
                self.assertEqual(learning["learning"][0], {"kind": "background", "material_count": 2,
                    "completed_count": 1, "checkpoint": "下一次继续理解第二条", "memory_refs": [], "run_id": run_id})
                self.assertEqual(learning["learning_status"]["background"]["status"], "deferred")
                self.assertEqual(learning["learning_status"]["background"]["next_start_local"], console.local_date(1800000000))
                self.assertTrue(learning["learning_window"]["local_now"])
                await console.distill("synthetic")
                plugin.consolidate.assert_awaited_once_with(opened, force=True)
                with self.assertRaises(FileNotFoundError):
                    await console.get_store("../synthetic")
            successor = Console(plugin)
            self.assertEqual(len(context.registered_web_apis), 12)
            console.close()
            self.assertEqual(len(context.registered_web_apis), 12)
            successor.close()
            self.assertEqual(context.registered_web_apis, [])
            opened.close()


if __name__ == "__main__":
    unittest.main()
