from __future__ import annotations

import asyncio
import unittest

from mr_memory.singleflight import AsyncSingleFlight


class AsyncSingleFlightTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_key_runs_factory_once(self) -> None:
        flight: AsyncSingleFlight[int] = AsyncSingleFlight()
        calls = 0
        release = asyncio.Event()

        async def work() -> int:
            nonlocal calls
            calls += 1
            await release.wait()
            return 7

        first = asyncio.create_task(flight.run("same", work))
        second = asyncio.create_task(flight.run("same", work))
        await asyncio.sleep(0)
        release.set()
        values = await asyncio.gather(first, second)
        self.assertEqual([item[0] for item in values], [7, 7])
        self.assertEqual(sorted(item[1] for item in values), [False, True])
        self.assertEqual(calls, 1)

    async def test_waiter_timeout_does_not_cancel_shared_task(self) -> None:
        flight: AsyncSingleFlight[str] = AsyncSingleFlight()
        release = asyncio.Event()

        async def work() -> str:
            await release.wait()
            return "cached"

        task, created = await flight.start("key", work)
        self.assertTrue(created)
        with self.assertRaises(TimeoutError):
            await flight.run("key", work, timeout=0.01)
        self.assertFalse(task.cancelled())
        release.set()
        self.assertEqual(await task, "cached")

    async def test_different_keys_are_independent(self) -> None:
        flight: AsyncSingleFlight[str] = AsyncSingleFlight()

        async def work(value: str) -> str:
            await asyncio.sleep(0)
            return value

        values = await asyncio.gather(
            flight.run("a", lambda: work("a")),
            flight.run("b", lambda: work("b")),
        )
        self.assertEqual({item[0] for item in values}, {"a", "b"})


if __name__ == "__main__":
    unittest.main()
