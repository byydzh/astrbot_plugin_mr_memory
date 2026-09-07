"""Incremental private run events; persistence never gates model or tool work."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any


def _visible(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _visible(item) for key, item in value.items() if str(key).lower() != "reasoning_content"}
    if isinstance(value, (list, tuple)):
        return [_visible(item) for item in value]
    return value


class RunTrace:
    def __init__(self, store, kind: str, started_at: float, initial: dict):
        self.store, self.kind, self.started_at, self.initial = store, kind, started_at, initial
        self.id: int | None = None
        self.error = ""
        self._started = False
        self._finished = False
        self._seq = 0
        self._queue = asyncio.Queue()
        self._writer = None
        self._finish_lock = asyncio.Lock()

    def _failed(self, exc: Exception) -> None:
        self.error = f"{type(exc).__name__}: {exc}"
        try:
            logging.getLogger(__name__).warning("MR run trace %s could not persist an event: %s", self.id, self.error)
        except Exception:
            pass

    async def start(self) -> int | None:
        if self._started:
            return self.id
        self._started = True
        try:
            payload = {**_visible(self.initial), "status": "running", "trace_version": 1}
            self.id = await asyncio.to_thread(self.store.record_run, self.kind, self.started_at, payload)
            self._writer = asyncio.create_task(self._write_steps())
        except Exception as exc:
            self._failed(exc)
        return self.id

    async def emit(self, phase: str, title: str, status: str = "completed", **data) -> None:
        if self.id is None or self._finished:
            return
        try:
            at = time.time()
            visible = _visible(data)
            self._seq += 1
            self._queue.put_nowait((self._seq, at, phase, status, title, visible))
        except Exception as exc:
            self._failed(exc)

    async def _write_steps(self) -> None:
        while True:
            step = await self._queue.get()
            try:
                if step is None:
                    return
                await asyncio.to_thread(self.store.append_run_step, self.id, *step)
            except Exception as exc:
                self._failed(exc)
            finally:
                self._queue.task_done()

    async def finish(self, payload: dict) -> None:
        async with self._finish_lock:
            if self._finished:
                return
            if self.id is None:
                self._finished = True
                return
            status = str(payload.get("status") or "completed")
            await self.emit("end", "本次运行结束", status=status,
                            **{key: payload[key] for key in ("detail", "elapsed_ms", "usage") if key in payload})
            self._finished = True
            self._queue.put_nowait(None)
            try:
                await self._writer
                final = {**_visible(payload), "status": status, "finished_at": time.time()}
                if self.error:
                    final["trace_error"] = self.error
                await asyncio.to_thread(self.store.update_run, self.id,
                                        final)
            except Exception as exc:
                self._failed(exc)
