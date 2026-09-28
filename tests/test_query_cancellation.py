"""The timed-out SQLite worker must be gone, not merely its asyncio waiter."""
import asyncio
import threading
import time
import unittest
from unittest.mock import patch

from mr_memory.agent import MemoryAgent
from mr_memory.store import Store
from tests.test_agent import Provider, response


SLOW_SQL = """WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL
    SELECT x+1 FROM n WHERE x<100000000) SELECT sum(x) AS total FROM n"""


class QueryCancellationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = Store(":memory:", "test:GroupMessage:42")
        self.started = threading.Event()
        self.finished = threading.Event()

    def tearDown(self):
        self.store.db.interrupt()
        self.store.close()

    def slow_read(self, **kwargs):
        self.started.set()
        try:
            return self.store._rows(SLOW_SQL)
        finally:
            self.finished.set()

    async def wait_started(self):
        async with asyncio.timeout(1):
            while not self.started.is_set():
                await asyncio.sleep(0.001)

    async def test_agent_timeout_stops_sql_and_next_read_and_write_work(self):
        provider = Provider([response(calls=[("search_memories", {"related_account_id": "10"}, "s")])])
        with patch("mr_memory.agent._tool_set", return_value=object()), \
             patch.object(self.store, "search_memories", side_effect=self.slow_read):
            result = await asyncio.wait_for(MemoryAgent(provider, self.store, timeout_seconds=0.1).reconstruct(
                {"sent_at": 300}, [], {}), 2)
        self.assertTrue(self.started.is_set())
        self.assertTrue(self.finished.is_set(), "outer timeout left the SQL worker alive")
        self.assertEqual(result.status, "partial")
        self.assertIn("time budget", result.detail)
        await asyncio.wait_for(asyncio.to_thread(self.store.update_working_state, {"question": "next"}), 1)
        state = await asyncio.wait_for(self.store.read_async(self.store.load_working_state), 1)
        self.assertEqual(state["question"], "next")

    async def test_queued_read_can_cancel_without_poisoning_next_query(self):
        first = asyncio.create_task(self.store.read_async(self.slow_read))
        await self.wait_started()
        called = threading.Event()
        queued = asyncio.create_task(self.store.read_async(called.set))
        await asyncio.sleep(0)
        queued.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(queued, 1)
        self.assertFalse(called.is_set())
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(first, 1)
        self.assertTrue(self.finished.is_set())
        self.assertEqual(await self.store.read_async(self.store._rows, "SELECT 42 AS n"), [{"n": 42}])

    async def test_cancellation_while_waiting_for_connection_lock(self):
        held, release, called = threading.Event(), threading.Event(), threading.Event()

        def hold_lock():
            with self.store._lock:
                held.set()
                release.wait(2)

        owner = threading.Thread(target=hold_lock)
        owner.start()
        try:
            async with asyncio.timeout(1):
                while not held.is_set():
                    await asyncio.sleep(0.001)
            query = asyncio.create_task(self.store.read_async(called.set))
            await asyncio.sleep(0.03)
            started = time.monotonic()
            query.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(query, 1)
            self.assertLess(time.monotonic() - started, 0.5)
            self.assertFalse(called.is_set())
        finally:
            release.set()
            owner.join(1)


if __name__ == "__main__":
    unittest.main()
