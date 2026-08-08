from __future__ import annotations

import asyncio

from .distillation import (
    DistillationBatch,
    PersistedDistillation,
    commit_distillation_batch,
    index_distillation,
    persist_distillation,
)
from .embedding import EmbeddingBackend
from .feedback import FeedbackDecision
from .models import DistillationWorkItem, NormalizedMessage, StoredMessage
from .plasticity import GraphMutation
from .storage import MemoryStorage


class MemoryService:
    """Async facade that keeps SQLite work outside AstrBot's event loop."""

    def __init__(self, storage: MemoryStorage):
        self.storage = storage

    async def ingest(
        self,
        message: NormalizedMessage,
        *,
        processing_class: str = "LIVE",
        ingestion_source: str = "",
    ) -> bool:
        return await asyncio.to_thread(
            self.storage.upsert_message,
            message,
            processing_class=processing_class,
            ingestion_source=ingestion_source,
        )

    async def ingest_many(
        self,
        messages: list[NormalizedMessage],
        *,
        defer_media_index: bool = False,
        processing_class: str = "LIVE",
        ingestion_source: str = "",
    ) -> dict[str, int]:
        return await asyncio.to_thread(
            self.storage.upsert_messages,
            messages,
            defer_media_index=defer_media_index,
            processing_class=processing_class,
            ingestion_source=ingestion_source,
        )

    async def rebuild_media_fingerprints(self, *, umo: str) -> None:
        await asyncio.to_thread(
            self.storage.rebuild_media_fingerprints,
            umo=umo,
        )

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

    async def pending_distillation_count(
        self,
        *,
        umo: str,
        processing_class: str = "",
    ) -> int:
        return await asyncio.to_thread(
            self.storage.pending_distillation_count,
            umo=umo,
            processing_class=processing_class,
        )

    async def oldest_pending_distillation_at(
        self,
        *,
        umo: str,
        processing_class: str = "LIVE",
    ) -> int | None:
        return await asyncio.to_thread(
            self.storage.oldest_pending_distillation_at,
            umo=umo,
            processing_class=processing_class,
        )

    async def next_distillation_processing_class(
        self,
        *,
        umo: str,
    ) -> str | None:
        return await asyncio.to_thread(
            self.storage.next_distillation_processing_class,
            umo=umo,
        )

    async def retry_terminal_distillation_failures(
        self,
        *,
        umo: str,
        processing_class: str = "",
    ) -> int:
        return await asyncio.to_thread(
            self.storage.retry_terminal_distillation_failures,
            umo=umo,
            processing_class=processing_class,
        )

    async def next_distillation_batch(
        self, **kwargs: object
    ) -> DistillationWorkItem | None:
        return await asyncio.to_thread(self.storage.next_distillation_batch, **kwargs)

    async def finish_distillation_batch(self, **kwargs: object) -> None:
        await asyncio.to_thread(self.storage.finish_distillation_batch, **kwargs)

    async def record_distillation_ignored_sources(self, **kwargs: object) -> None:
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

    async def runtime_health_summary(self, **kwargs: object) -> dict[str, object]:
        return await asyncio.to_thread(
            self.storage.runtime_health_summary,
            **kwargs,
        )

    async def dashboard_graph(
        self,
        *,
        umo: str,
        limit: int = 200,
        query: str = "",
        focus_node_id: str = "",
        depth: int = 1,
        node_types: tuple[str, ...] = (),
        epistemic_states: tuple[str, ...] = (),
        relation: str = "",
        min_degree: int = 0,
        min_core: int = 0,
        structure_scope: str = "all",
        path_source: str = "",
        path_target: str = "",
    ) -> dict[str, object]:
        return await asyncio.to_thread(
            self.storage.dashboard_graph,
            umo=umo,
            limit=limit,
            query=query,
            focus_node_id=focus_node_id,
            depth=depth,
            node_types=node_types,
            epistemic_states=epistemic_states,
            relation=relation,
            min_degree=min_degree,
            min_core=min_core,
            structure_scope=structure_scope,
            path_source=path_source,
            path_target=path_target,
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

    async def private_token_usage_since(
        self,
        *,
        umo: str,
        since: int,
        budget_class: str = "all",
        apply_resets: bool = True,
    ) -> int:
        return await asyncio.to_thread(
            self.storage.private_token_usage_since,
            umo=umo,
            since=since,
            budget_class=budget_class,
            apply_resets=apply_resets,
        )

    async def reset_token_budget(self, **kwargs: object) -> dict[str, object]:
        return await asyncio.to_thread(self.storage.reset_token_budget, **kwargs)

    async def private_budget_retry_at(self, **kwargs: object) -> int:
        return await asyncio.to_thread(self.storage.private_budget_retry_at, **kwargs)

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

    async def commit_distillation_batch(
        self,
        batch: DistillationBatch,
        *,
        work_item: DistillationWorkItem,
        extractor_version: str,
        embedding_backend: EmbeddingBackend | None = None,
    ) -> tuple[PersistedDistillation, int, str]:
        persisted = await asyncio.to_thread(
            commit_distillation_batch,
            self.storage,
            batch,
            work_item=work_item,
            extractor_version=extractor_version,
        )
        indexed = 0
        index_error = ""
        if embedding_backend is not None:
            try:
                indexed = await index_distillation(
                    self.storage,
                    umo=batch.umo,
                    persisted=persisted,
                    backend=embedding_backend,
                )
            except Exception as exc:
                index_error = f"{type(exc).__name__}: {exc}"[:1000]
        return persisted, indexed, index_error

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

    async def reconstruction_evidence_packet(
        self, **kwargs: object
    ) -> dict[str, object]:
        return await asyncio.to_thread(
            self.storage.reconstruction_evidence_packet,
            **kwargs,
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

    async def apply_graph_mutation(
        self,
        *,
        mutation: GraphMutation,
        **kwargs: object,
    ) -> dict[str, object]:
        return await asyncio.to_thread(
            self.storage.apply_graph_mutation,
            mutation=mutation,
            **kwargs,
        )

    async def index_plastic_edge(
        self,
        *,
        umo: str,
        edge_id: int,
        embedding_backend: EmbeddingBackend,
    ) -> bool:
        document = await asyncio.to_thread(
            self.storage.plastic_edge_embedding_document,
            umo=umo,
            edge_id=int(edge_id),
        )
        if document is None:
            return False
        vectors = await embedding_backend.embed_texts([str(document["text"])])
        if len(vectors) != 1:
            raise ValueError("embedding backend returned the wrong vector count")
        await asyncio.to_thread(
            self.storage.upsert_memory_embedding,
            umo=umo,
            owner_type="plastic_edge",
            owner_key=str(document["owner_key"]),
            model=embedding_backend.model_id,
            vector=vectors[0],
        )
        return True

    async def query_plastic_associations(
        self, **kwargs: object
    ) -> list[dict[str, object]]:
        return await asyncio.to_thread(
            self.storage.query_plastic_associations, **kwargs
        )

    async def query_media_patterns(self, **kwargs: object) -> list[dict[str, object]]:
        return await asyncio.to_thread(self.storage.query_media_patterns, **kwargs)

    async def activate_plastic_edges(self, **kwargs: object) -> list[dict[str, object]]:
        return await asyncio.to_thread(self.storage.activate_plastic_edges, **kwargs)

    async def compact_plastic_graph(self, **kwargs: object) -> dict[str, int]:
        return await asyncio.to_thread(self.storage.compact_plastic_graph, **kwargs)

    async def subconscious_state(self, *, umo: str) -> dict[str, object]:
        return await asyncio.to_thread(self.storage.subconscious_state, umo=umo)

    async def update_subconscious_state(self, **kwargs: object) -> dict[str, object]:
        return await asyncio.to_thread(self.storage.update_subconscious_state, **kwargs)

    async def enqueue_maintenance_job(self, **kwargs: object) -> int:
        return await asyncio.to_thread(self.storage.enqueue_maintenance_job, **kwargs)

    async def pending_maintenance_jobs(
        self, **kwargs: object
    ) -> list[dict[str, object]]:
        return await asyncio.to_thread(self.storage.pending_maintenance_jobs, **kwargs)

    async def claim_maintenance_job(self, **kwargs: object) -> dict[str, object] | None:
        return await asyncio.to_thread(self.storage.claim_maintenance_job, **kwargs)

    async def maintenance_job_ready(self, **kwargs: object) -> bool:
        return await asyncio.to_thread(self.storage.maintenance_job_ready, **kwargs)

    async def defer_maintenance_job_for_budget(self, **kwargs: object) -> None:
        await asyncio.to_thread(self.storage.defer_maintenance_job_for_budget, **kwargs)

    async def resume_due_budget_jobs(self, **kwargs: object) -> int:
        return await asyncio.to_thread(self.storage.resume_due_budget_jobs, **kwargs)

    async def finish_maintenance_job(self, **kwargs: object) -> None:
        await asyncio.to_thread(self.storage.finish_maintenance_job, **kwargs)

    async def release_maintenance_job(self, **kwargs: object) -> bool:
        return await asyncio.to_thread(self.storage.release_maintenance_job, **kwargs)

    async def fail_maintenance_job(self, **kwargs: object) -> str:
        return await asyncio.to_thread(self.storage.fail_maintenance_job, **kwargs)

    async def start_interaction_trace(self, **kwargs: object) -> str:
        return await asyncio.to_thread(self.storage.start_interaction_trace, **kwargs)

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

    async def inspect_feedback_proposal(self, **kwargs: object) -> dict[str, object]:
        return await asyncio.to_thread(self.storage.inspect_feedback_proposal, **kwargs)

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
        return await asyncio.to_thread(self.storage.feedback_proposal_status, **kwargs)

    async def compact_feedback_memory(self, **kwargs: object) -> dict[str, int]:
        return await asyncio.to_thread(self.storage.compact_feedback_memory, **kwargs)

    async def interaction_trace_graph(
        self, **kwargs: object
    ) -> dict[str, object] | None:
        return await asyncio.to_thread(self.storage.interaction_trace_graph, **kwargs)

    async def close(self) -> None:
        await asyncio.to_thread(self.storage.close)
