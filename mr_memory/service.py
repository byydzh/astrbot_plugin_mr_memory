from __future__ import annotations

import asyncio

from .distillation import (
    DistillationBatch,
    PersistedDistillation,
    index_distillation,
    persist_distillation,
)
from .embedding import EmbeddingBackend
from .feedback import FeedbackDecision
from .models import DistillationWorkItem, NormalizedMessage, StoredMessage
from .storage import MemoryStorage


class MemoryService:
    """Async facade that keeps SQLite work outside AstrBot's event loop."""

    def __init__(self, storage: MemoryStorage):
        self.storage = storage

    async def ingest(self, message: NormalizedMessage) -> bool:
        return await asyncio.to_thread(self.storage.upsert_message, message)

    async def mark_message_deleted(self, **kwargs: object) -> bool:
        return await asyncio.to_thread(self.storage.mark_message_deleted, **kwargs)

    async def is_account_forgotten(self, **kwargs: object) -> bool:
        return await asyncio.to_thread(self.storage.is_account_forgotten, **kwargs)

    async def forget_account(self, **kwargs: object) -> dict[str, int]:
        return await asyncio.to_thread(self.storage.forget_account, **kwargs)

    async def bind_participant_alias(self, **kwargs: object) -> dict[str, object]:
        return await asyncio.to_thread(self.storage.bind_participant_alias, **kwargs)

    async def resolve_participants(self, **kwargs: object) -> dict[str, object]:
        return await asyncio.to_thread(self.storage.resolve_participants, **kwargs)

    async def list_participants(self, **kwargs: object) -> list[dict[str, object]]:
        return await asyncio.to_thread(self.storage.list_participants, **kwargs)

    async def distillation_identity_context(
        self, **kwargs: object
    ) -> dict[str, object]:
        return await asyncio.to_thread(
            self.storage.distillation_identity_context, **kwargs
        )

    async def pending_distillation_count(self, *, umo: str) -> int:
        return await asyncio.to_thread(
            self.storage.pending_distillation_count, umo=umo
        )

    async def next_distillation_batch(
        self, **kwargs: object
    ) -> DistillationWorkItem | None:
        return await asyncio.to_thread(
            self.storage.next_distillation_batch, **kwargs
        )

    async def finish_distillation_batch(self, **kwargs: object) -> None:
        await asyncio.to_thread(self.storage.finish_distillation_batch, **kwargs)

    async def record_distillation_ignored_sources(
        self, **kwargs: object
    ) -> None:
        await asyncio.to_thread(
            self.storage.record_distillation_ignored_sources, **kwargs
        )

    async def search(
        self,
        *,
        umo: str,
        query: str = "",
        sender: str = "",
        limit: int = 20,
        before_sent_at: int | None = None,
    ) -> list[StoredMessage]:
        return await asyncio.to_thread(
            self.storage.search_messages,
            umo=umo,
            query=query,
            sender=sender,
            limit=limit,
            before_sent_at=before_sent_at,
        )

    async def count(self, *, umo: str | None = None) -> int:
        return await asyncio.to_thread(self.storage.count_messages, umo=umo)

    async def count_graph_units(
        self, *, umo: str, before_sent_at: int | None = None
    ) -> int:
        return await asyncio.to_thread(
            self.storage.count_graph_units,
            umo=umo,
            before_sent_at=before_sent_at,
        )

    async def dashboard_summary(self, *, umo: str) -> dict[str, object]:
        return await asyncio.to_thread(self.storage.dashboard_summary, umo=umo)

    async def dashboard_graph(
        self, *, umo: str, limit: int = 200
    ) -> dict[str, object]:
        return await asyncio.to_thread(
            self.storage.dashboard_graph,
            umo=umo,
            limit=limit,
        )

    async def start_experiment(self, **kwargs: object) -> None:
        await asyncio.to_thread(self.storage.start_experiment, **kwargs)

    async def finish_experiment(self, **kwargs: object) -> None:
        await asyncio.to_thread(self.storage.finish_experiment, **kwargs)

    async def record_llm_usage(self, **kwargs: object) -> int:
        return await asyncio.to_thread(self.storage.record_llm_usage, **kwargs)

    async def record_reconstruction_step(self, **kwargs: object) -> int:
        return await asyncio.to_thread(
            self.storage.record_reconstruction_step,
            **kwargs,
        )

    async def experiment_report(self, *, run_id: str) -> dict[str, object] | None:
        return await asyncio.to_thread(
            self.storage.experiment_report,
            run_id=run_id,
        )

    async def recent_experiments(
        self,
        *,
        umo: str,
        limit: int = 10,
    ) -> list[dict[str, object]]:
        return await asyncio.to_thread(
            self.storage.recent_experiments,
            umo=umo,
            limit=limit,
        )

    async def private_token_usage_since(self, *, umo: str, since: int) -> int:
        return await asyncio.to_thread(
            self.storage.private_token_usage_since,
            umo=umo,
            since=since,
        )

    async def apply_distillation(
        self,
        batch: DistillationBatch,
        *,
        extractor_version: str,
        embedding_backend: EmbeddingBackend | None = None,
    ) -> tuple[PersistedDistillation, int]:
        persisted = await asyncio.to_thread(
            persist_distillation,
            self.storage,
            batch,
            extractor_version=extractor_version,
        )
        indexed = 0
        if embedding_backend is not None:
            indexed = await index_distillation(
                self.storage,
                umo=batch.umo,
                persisted=persisted,
                backend=embedding_backend,
            )
        return persisted, indexed

    async def initialize_candidates(
        self,
        *,
        umo: str,
        query: str,
        embedding_backend: EmbeddingBackend,
        limit: int = 12,
        min_score: float = -1.0,
        before_sent_at: int | None = None,
    ) -> dict[str, list[dict[str, object]]]:
        query_vector = await embedding_backend.embed_query(query)
        matches = await asyncio.to_thread(
            self.storage.search_memory_embeddings,
            umo=umo,
            model=embedding_backend.model_id,
            query_vector=query_vector,
            limit=limit,
            min_score=min_score,
            before_sent_at=before_sent_at,
        )
        return await asyncio.to_thread(
            self.storage.expand_seed_candidates,
            umo=umo,
            matches=matches,
            before_sent_at=before_sent_at,
        )

    async def query_cue_tags(
        self,
        *,
        umo: str,
        cue: str,
        limit: int = 20,
        before_sent_at: int | None = None,
    ) -> list[dict[str, object]]:
        return await asyncio.to_thread(
            self.storage.query_cue_tags,
            umo=umo,
            cue=cue,
            limit=limit,
            before_sent_at=before_sent_at,
        )

    async def query_tag_events(
        self,
        *,
        umo: str,
        cue: str,
        tag: str,
        limit: int = 20,
        before_sent_at: int | None = None,
    ) -> list[dict[str, object]]:
        return await asyncio.to_thread(
            self.storage.query_tag_events,
            umo=umo,
            cue=cue,
            tag=tag,
            limit=limit,
            before_sent_at=before_sent_at,
        )

    async def query_conversation_time(
        self,
        *,
        umo: str,
        event_id: int,
        before_sent_at: int | None = None,
    ) -> dict[str, object] | None:
        return await asyncio.to_thread(
            self.storage.query_conversation_time,
            umo=umo,
            event_id=event_id,
            before_sent_at=before_sent_at,
        )

    async def query_event_keywords(
        self,
        *,
        umo: str,
        event_id: int,
        before_sent_at: int | None = None,
    ) -> list[dict[str, object]]:
        return await asyncio.to_thread(
            self.storage.query_event_keywords,
            umo=umo,
            event_id=event_id,
            before_sent_at=before_sent_at,
        )

    async def query_event_context(
        self,
        *,
        umo: str,
        event_id: int,
        limit: int = 50,
        before_sent_at: int | None = None,
    ) -> list[dict[str, object]]:
        return await asyncio.to_thread(
            self.storage.query_event_context,
            umo=umo,
            event_id=event_id,
            limit=limit,
            before_sent_at=before_sent_at,
        )

    async def query_personal_information(
        self,
        *,
        umo: str,
        person: str,
        before_sent_at: int | None = None,
    ) -> list[dict[str, object]]:
        return await asyncio.to_thread(
            self.storage.query_personal_information,
            umo=umo,
            person=person,
            before_sent_at=before_sent_at,
        )

    async def query_personal_aspect(
        self,
        *,
        umo: str,
        person: str,
        aspect: str,
        limit: int = 20,
        before_sent_at: int | None = None,
    ) -> list[dict[str, object]]:
        return await asyncio.to_thread(
            self.storage.query_personal_aspect,
            umo=umo,
            person=person,
            aspect=aspect,
            limit=limit,
            before_sent_at=before_sent_at,
        )

    async def query_topic_events(
        self,
        *,
        umo: str,
        topic: str,
        limit: int = 20,
        before_sent_at: int | None = None,
    ) -> list[dict[str, object]]:
        return await asyncio.to_thread(
            self.storage.query_topic_events,
            umo=umo,
            topic=topic,
            limit=limit,
            before_sent_at=before_sent_at,
        )

    async def start_interaction_trace(self, **kwargs: object) -> str:
        return await asyncio.to_thread(
            self.storage.start_interaction_trace, **kwargs
        )

    async def record_trace_node(self, **kwargs: object) -> int:
        return await asyncio.to_thread(self.storage.record_trace_node, **kwargs)

    async def record_trace_edge(self, **kwargs: object) -> int:
        return await asyncio.to_thread(self.storage.record_trace_edge, **kwargs)

    async def finish_interaction_trace(self, **kwargs: object) -> None:
        await asyncio.to_thread(self.storage.finish_interaction_trace, **kwargs)

    async def enqueue_feedback_candidate(self, **kwargs: object) -> int | None:
        return await asyncio.to_thread(
            self.storage.enqueue_feedback_candidate, **kwargs
        )

    async def pending_feedback_proposals(
        self, **kwargs: object
    ) -> list[dict[str, object]]:
        return await asyncio.to_thread(
            self.storage.pending_feedback_proposals, **kwargs
        )

    async def inspect_feedback_proposal(
        self, **kwargs: object
    ) -> dict[str, object]:
        return await asyncio.to_thread(
            self.storage.inspect_feedback_proposal, **kwargs
        )

    async def search_feedback_hypotheses(
        self, **kwargs: object
    ) -> list[dict[str, object]]:
        return await asyncio.to_thread(
            self.storage.search_feedback_hypotheses, **kwargs
        )

    async def feedback_hypothesis_candidates(
        self, **kwargs: object
    ) -> list[dict[str, object]]:
        return await asyncio.to_thread(
            self.storage.feedback_hypothesis_candidates, **kwargs
        )

    async def activate_feedback_hypotheses(
        self, **kwargs: object
    ) -> list[dict[str, object]]:
        return await asyncio.to_thread(
            self.storage.activate_feedback_hypotheses, **kwargs
        )

    async def apply_feedback_decision(
        self,
        *,
        decision: FeedbackDecision,
        **kwargs: object,
    ) -> dict[str, object]:
        return await asyncio.to_thread(
            self.storage.apply_feedback_decision,
            decision=decision,
            **kwargs,
        )

    async def reject_feedback_proposal(self, **kwargs: object) -> None:
        await asyncio.to_thread(self.storage.reject_feedback_proposal, **kwargs)

    async def feedback_proposal_status(
        self, **kwargs: object
    ) -> dict[str, object] | None:
        return await asyncio.to_thread(
            self.storage.feedback_proposal_status, **kwargs
        )

    async def compact_feedback_memory(
        self, **kwargs: object
    ) -> dict[str, int]:
        return await asyncio.to_thread(
            self.storage.compact_feedback_memory, **kwargs
        )

    async def interaction_trace_graph(
        self, **kwargs: object
    ) -> dict[str, object] | None:
        return await asyncio.to_thread(
            self.storage.interaction_trace_graph, **kwargs
        )

    async def close(self) -> None:
        await asyncio.to_thread(self.storage.close)
