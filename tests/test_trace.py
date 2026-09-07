import asyncio
import json
import time
import unittest
import uuid
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mr_memory.agent import MemoryAgent
from mr_memory.store import Store
from mr_memory.trace import RunTrace


def response(text="", calls=()):
    return SimpleNamespace(completion_text=text, tools_call_name=[call[0] for call in calls],
        tools_call_args=[call[1] for call in calls], tools_call_ids=[call[2] for call in calls],
        reasoning_content="secret-reasoning", usage={"input_other": 10, "output": 4})


class TraceTests(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_tool_events_are_incremental_paired_and_nonblocking_on_log_failure(self):
        scratch = Path(__file__).resolve().parents[1] / ".pytest_cache" / ("trace-" + uuid.uuid4().hex)
        scratch.mkdir(parents=True)
        store = Store(scratch / "group.db", "test:GroupMessage:trace")
        gates = {name: asyncio.Event() for name in ("slow", "fast")}
        entered = {name: asyncio.Event() for name in gates}
        fast_returned = asyncio.Event()

        class Agent(MemoryAgent):
            def __init__(self, replies):
                super().__init__(None, store)
                self.replies = replies

            async def _generate(self, *args, **kwargs):
                return self.replies.pop(0)

            async def _execute(self, name, arguments, cutoff):
                branch = arguments["branch"]
                entered[branch].set()
                await gates[branch].wait()
                if branch == "fast":
                    fast_returned.set()
                return {"branch": branch, "content": "真实读取结果"}

        task = None
        try:
            trace = RunTrace(store, "foreground", time.time(), {"question": "星舟是什么", "status": "cancelled"})
            run_id = await trace.start()
            self.assertEqual(store.recent_runs()[0]["status"], "running")
            agent = Agent([response(calls=[("search_messages", {"branch": "slow"}, "slow-call"),
                                          ("search_messages", {"branch": "fast"}, "fast-call")]),
                           response("星舟是共同设计的桌游。")])
            agent.trace = trace
            with patch("mr_memory.agent._tool_set", return_value=object()):
                task = asyncio.create_task(agent.reconstruct({"sent_at": 300}, [], {}))
                await asyncio.wait_for(asyncio.gather(*(event.wait() for event in entered.values())), 2)
                await trace._queue.join()
                visible = store.run_detail(run_id)
                starts = [step for step in visible["steps"] if step["phase"] == "read"]
                self.assertEqual([step["data"]["tool_call_id"] for step in starts], ["slow-call", "fast-call"])
                self.assertTrue(all(step["status"] == "running" and step["data"]["turn"] == 1 for step in starts))
                self.assertEqual(store.recent_runs()[0]["step_count"], len(visible["steps"]))
                self.assertEqual(visible["status"], "running")
                gates["fast"].set()
                await asyncio.wait_for(fast_returned.wait(), 2)
                await trace._queue.join()
                incremental = store.run_detail(run_id)
                finished = [step for step in incremental["steps"] if step["phase"] == "read" and step["status"] == "completed"]
                self.assertEqual([step["data"]["tool_call_id"] for step in finished], ["fast-call"])
                self.assertFalse(task.done())
                gates["slow"].set()
                result = await task
            await trace.finish({**asdict(result), "nested": {"reasoning_content": "secret-reasoning"}})
            detail = store.run_detail(run_id)
            steps = detail["steps"]
            self.assertEqual(detail["status"], "completed")
            self.assertEqual([step["seq"] for step in steps], list(range(1, len(steps) + 1)))
            self.assertEqual([step["data"]["tool_call_id"] for step in steps if step["phase"] == "read" and step["status"] == "completed"],
                             ["fast-call", "slow-call"])
            self.assertEqual(steps[-1]["phase"], "end")
            self.assertEqual(store.recent_runs()[0]["latest_phase"], "end")
            self.assertNotIn("reasoning_content", json.dumps(detail))
            self.assertNotIn("secret-reasoning", json.dumps(detail))

            broken = RunTrace(store, "foreground", time.time(), {})
            broken_id = await broken.start()
            agent = Agent([response("日志写入失败也照常产出背景。")])
            agent.trace = broken
            with patch.object(store, "append_run_step", side_effect=OSError("synthetic logging failure")), \
                    patch("mr_memory.agent._tool_set", return_value=object()), self.assertLogs("mr_memory.trace", level="WARNING"):
                unaffected = await agent.reconstruct({"sent_at": 300}, [], {})
                await broken.finish(asdict(unaffected))
            self.assertEqual(unaffected.status, "completed")
            self.assertIn("照常产出", unaffected.background)
            self.assertEqual(store.run_detail(broken_id)["status"], "completed")
            self.assertIn("synthetic logging failure", broken.error)
        finally:
            for gate in gates.values():
                gate.set()
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            store.close()
            for file in scratch.iterdir():
                file.unlink()
            scratch.rmdir()


if __name__ == "__main__":
    unittest.main()
