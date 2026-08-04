from __future__ import annotations

import asyncio

from .models import NormalizedMessage, StoredMessage
from .storage import MemoryStorage


class MemoryService:
    """Async facade that keeps SQLite work outside AstrBot's event loop."""

    def __init__(self, storage: MemoryStorage):
        self.storage = storage

    async def ingest(self, message: NormalizedMessage) -> bool:
        return await asyncio.to_thread(self.storage.upsert_message, message)

    async def search(
        self,
        *,
        umo: str,
        query: str = "",
        sender: str = "",
        limit: int = 20,
    ) -> list[StoredMessage]:
        return await asyncio.to_thread(
            self.storage.search_messages,
            umo=umo,
            query=query,
            sender=sender,
            limit=limit,
        )

    async def count(self, *, umo: str | None = None) -> int:
        return await asyncio.to_thread(self.storage.count_messages, umo=umo)

    async def count_graph_units(self, *, umo: str) -> int:
        return await asyncio.to_thread(self.storage.count_graph_units, umo=umo)

    async def query_tag_events(
        self, *, umo: str, cue: str, tag: str, limit: int = 20
    ) -> list[dict[str, object]]:
        return await asyncio.to_thread(
            self.storage.query_tag_events,
            umo=umo,
            cue=cue,
            tag=tag,
            limit=limit,
        )

    async def query_conversation_time(
        self, *, umo: str, event_id: int
    ) -> dict[str, object] | None:
        return await asyncio.to_thread(
            self.storage.query_conversation_time,
            umo=umo,
            event_id=event_id,
        )

    async def query_event_keywords(
        self, *, umo: str, event_id: int
    ) -> list[dict[str, object]]:
        return await asyncio.to_thread(
            self.storage.query_event_keywords,
            umo=umo,
            event_id=event_id,
        )

    async def query_event_context(
        self, *, umo: str, event_id: int, limit: int = 50
    ) -> list[dict[str, object]]:
        return await asyncio.to_thread(
            self.storage.query_event_context,
            umo=umo,
            event_id=event_id,
            limit=limit,
        )

    async def query_personal_information(
        self, *, umo: str, person: str
    ) -> list[dict[str, object]]:
        return await asyncio.to_thread(
            self.storage.query_personal_information,
            umo=umo,
            person=person,
        )

    async def query_personal_aspect(
        self,
        *,
        umo: str,
        person: str,
        aspect: str,
        limit: int = 20,
    ) -> list[dict[str, object]]:
        return await asyncio.to_thread(
            self.storage.query_personal_aspect,
            umo=umo,
            person=person,
            aspect=aspect,
            limit=limit,
        )

    async def query_topic_events(
        self, *, umo: str, topic: str, limit: int = 20
    ) -> list[dict[str, object]]:
        return await asyncio.to_thread(
            self.storage.query_topic_events,
            umo=umo,
            topic=topic,
            limit=limit,
        )

    async def close(self) -> None:
        await asyncio.to_thread(self.storage.close)
