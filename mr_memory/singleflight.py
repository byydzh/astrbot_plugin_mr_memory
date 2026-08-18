from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar


T = TypeVar("T")


class AsyncSingleFlight(Generic[T]):
    """Share one in-flight coroutine without coupling waiter cancellation.

    A caller deadline applies only to that caller.  The shared task continues and
    can populate a cache or serve a later request.  Completed tasks are removed
    only after their result has become observable to all current waiters.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._tasks: dict[str, asyncio.Task[T]] = {}

    async def start(
        self,
        key: str,
        factory: Callable[[], Awaitable[T]],
        *,
        task_name: str | None = None,
    ) -> tuple[asyncio.Task[T], bool]:
        normalized = str(key or "").strip()
        if not normalized:
            raise ValueError("singleflight key cannot be empty")
        async with self._lock:
            current = self._tasks.get(normalized)
            if current is not None and not current.done():
                return current, False
            task = asyncio.create_task(factory(), name=task_name)
            self._tasks[normalized] = task
            task.add_done_callback(
                lambda completed, active_key=normalized: asyncio.create_task(
                    self._discard(active_key, completed)
                )
            )
            return task, True

    async def run(
        self,
        key: str,
        factory: Callable[[], Awaitable[T]],
        *,
        timeout: float | None = None,
        task_name: str | None = None,
    ) -> tuple[T, bool]:
        task, created = await self.start(key, factory, task_name=task_name)
        protected = asyncio.shield(task)
        if timeout is None:
            return await protected, created
        try:
            async with asyncio.timeout(max(0.001, float(timeout))):
                return await protected, created
        except asyncio.TimeoutError as exc:
            raise TimeoutError("singleflight waiter timed out") from exc

    async def _discard(self, key: str, task: asyncio.Task[T]) -> None:
        async with self._lock:
            if self._tasks.get(key) is task:
                self._tasks.pop(key, None)

    async def active_keys(self) -> tuple[str, ...]:
        async with self._lock:
            return tuple(
                sorted(key for key, task in self._tasks.items() if not task.done())
            )

    async def drain(self, *, cancel: bool = False) -> None:
        async with self._lock:
            tasks = tuple(self._tasks.values())
        if cancel:
            for task in tasks:
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
