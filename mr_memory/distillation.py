from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .embedding import EmbeddingBackend
from .models import StoredMessage
from .storage import MemoryStorage


DISTILLATION_SYSTEM_PROMPT = """You build an associative memory graph from group-chat evidence.
Return one JSON object and no prose.

For the supplied source messages:
1. Resolve pronouns and relative references using only the supplied context.
2. Segment the stream into coherent episodes. Do not omit relevant statements.
3. Give every episode one short associative tag and 2-30 explicit fine-grained cues.
4. Extract stable person facts separately; jokes, guesses and uncertain claims must not
   be promoted to stable facts. Confidence is 0.0-1.0.
5. Create topics only when they summarize one or more returned episodes.
6. Every source_key must be copied exactly from the input. Never invent an ID.

Schema:
{
  "episodes": [{
    "source_keys": ["exact input source_key"],
    "started_at": 0,
    "ended_at": 0,
    "title": "short title",
    "summary": "self-contained evidence summary",
    "tag": "short associative relation",
    "cues": ["entity", "action", "attribute"]
  }],
  "semantic_memories": [{
    "source_key": "exact input source_key",
    "person": "entity",
    "aspect": "aspect tag",
    "content": "stable fact",
    "confidence": 0.0
  }],
  "topics": [{
    "name": "topic name",
    "summary": "shared pattern",
    "episode_indices": [0]
  }]
}
"""


@dataclass(frozen=True, slots=True)
class EpisodeDraft:
    source_keys: tuple[str, ...]
    started_at: int
    ended_at: int
    title: str
    summary: str
    tag: str
    cues: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SemanticDraft:
    source_key: str
    person: str
    aspect: str
    content: str
    confidence: float


@dataclass(frozen=True, slots=True)
class TopicDraft:
    name: str
    summary: str
    episode_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DistillationBatch:
    umo: str
    episodes: tuple[EpisodeDraft, ...]
    semantic_memories: tuple[SemanticDraft, ...]
    topics: tuple[TopicDraft, ...]


@dataclass(frozen=True, slots=True)
class IndexDocument:
    owner_type: str
    owner_key: str
    text: str


@dataclass(frozen=True, slots=True)
class PersistedDistillation:
    episode_ids: tuple[int, ...]
    semantic_ids: tuple[int, ...]
    topic_ids: tuple[int, ...]
    index_documents: tuple[IndexDocument, ...]


def _fingerprint(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_distillation_prompt(messages: list[StoredMessage]) -> str:
    if not messages:
        raise ValueError("at least one source message is required")
    umos = {message.umo for message in messages}
    if len(umos) != 1:
        raise ValueError("a distillation batch cannot cross group scopes")
    payload = [
        {
            "source_key": message.source_key,
            "sent_at": message.sent_at,
            "sender_id": message.sender_id,
            "sender_name": message.sender_name,
            "text": message.plain_text,
        }
        for message in messages
    ]
    return json.dumps({"messages": payload}, ensure_ascii=False, separators=(",", ":"))


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = str(text).strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("distillation response does not contain a JSON object")
    try:
        value = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid distillation JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("distillation response must be a JSON object")
    return value


def _required_text(value: Any, field: str, *, max_chars: int = 4000) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be a non-empty string")
    if len(text) > max_chars:
        raise ValueError(f"{field} exceeds {max_chars} characters")
    return text


def _list(value: Any, field: str, *, max_items: int) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    if len(value) > max_items:
        raise ValueError(f"{field} exceeds {max_items} items")
    return value


def parse_distillation_response(
    text: str,
    messages: list[StoredMessage],
) -> DistillationBatch:
    if not messages:
        raise ValueError("cannot validate distillation without source messages")
    umos = {message.umo for message in messages}
    if len(umos) != 1:
        raise ValueError("source messages cross group scopes")
    umo = next(iter(umos))
    source_map = {message.source_key: message for message in messages}
    value = _extract_json_object(text)

    raw_episodes = _list(value.get("episodes"), "episodes", max_items=64)
    raw_semantics = _list(
        value.get("semantic_memories", []),
        "semantic_memories",
        max_items=128,
    )
    raw_topics = _list(value.get("topics", []), "topics", max_items=64)

    episodes: list[EpisodeDraft] = []
    for index, raw in enumerate(raw_episodes):
        if not isinstance(raw, dict):
            raise ValueError(f"episodes[{index}] must be an object")
        source_keys = tuple(
            dict.fromkeys(
                _required_text(item, f"episodes[{index}].source_keys[]", max_chars=512)
                for item in _list(
                    raw.get("source_keys"),
                    f"episodes[{index}].source_keys",
                    max_items=128,
                )
            )
        )
        if not source_keys:
            raise ValueError(f"episodes[{index}] has no source evidence")
        unknown = [key for key in source_keys if key not in source_map]
        if unknown:
            raise ValueError(f"episodes[{index}] invented source keys: {unknown}")
        source_times = [source_map[key].sent_at for key in source_keys]
        started_at = int(raw.get("started_at") or min(source_times))
        ended_at = int(raw.get("ended_at") or max(source_times))
        if started_at > ended_at:
            raise ValueError(f"episodes[{index}] starts after it ends")
        cues = tuple(
            dict.fromkeys(
                _required_text(item, f"episodes[{index}].cues[]", max_chars=200)
                for item in _list(
                    raw.get("cues"), f"episodes[{index}].cues", max_items=30
                )
            )
        )
        if not cues:
            raise ValueError(f"episodes[{index}] must contain at least one cue")
        episodes.append(
            EpisodeDraft(
                source_keys=source_keys,
                started_at=started_at,
                ended_at=ended_at,
                title=_required_text(raw.get("title"), f"episodes[{index}].title", max_chars=200),
                summary=_required_text(
                    raw.get("summary"), f"episodes[{index}].summary"
                ),
                tag=_required_text(raw.get("tag"), f"episodes[{index}].tag", max_chars=200),
                cues=cues,
            )
        )

    semantics: list[SemanticDraft] = []
    for index, raw in enumerate(raw_semantics):
        if not isinstance(raw, dict):
            raise ValueError(f"semantic_memories[{index}] must be an object")
        source_key = _required_text(
            raw.get("source_key"),
            f"semantic_memories[{index}].source_key",
            max_chars=512,
        )
        if source_key not in source_map:
            raise ValueError(
                f"semantic_memories[{index}] invented source key: {source_key}"
            )
        confidence = float(raw.get("confidence", 0.0))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"semantic_memories[{index}].confidence must be 0..1")
        semantics.append(
            SemanticDraft(
                source_key=source_key,
                person=_required_text(
                    raw.get("person"), f"semantic_memories[{index}].person", max_chars=200
                ),
                aspect=_required_text(
                    raw.get("aspect"), f"semantic_memories[{index}].aspect", max_chars=200
                ),
                content=_required_text(
                    raw.get("content"), f"semantic_memories[{index}].content"
                ),
                confidence=confidence,
            )
        )

    topics: list[TopicDraft] = []
    for index, raw in enumerate(raw_topics):
        if not isinstance(raw, dict):
            raise ValueError(f"topics[{index}] must be an object")
        raw_indices = _list(
            raw.get("episode_indices"),
            f"topics[{index}].episode_indices",
            max_items=64,
        )
        episode_indices = tuple(dict.fromkeys(int(item) for item in raw_indices))
        if not episode_indices:
            raise ValueError(f"topics[{index}] has no episodes")
        if any(item < 0 or item >= len(episodes) for item in episode_indices):
            raise ValueError(f"topics[{index}] references an unknown episode index")
        topics.append(
            TopicDraft(
                name=_required_text(raw.get("name"), f"topics[{index}].name", max_chars=200),
                summary=_required_text(raw.get("summary"), f"topics[{index}].summary"),
                episode_indices=episode_indices,
            )
        )

    return DistillationBatch(
        umo=umo,
        episodes=tuple(episodes),
        semantic_memories=tuple(semantics),
        topics=tuple(topics),
    )


def persist_distillation(
    storage: MemoryStorage,
    batch: DistillationBatch,
    *,
    extractor_version: str,
) -> PersistedDistillation:
    episode_ids: list[int] = []
    semantic_ids: list[int] = []
    topic_ids: list[int] = []
    documents: dict[tuple[str, str], IndexDocument] = {}

    for episode in batch.episodes:
        fingerprint = _fingerprint(
            {
                "source_keys": episode.source_keys,
                "started_at": episode.started_at,
                "ended_at": episode.ended_at,
                "title": episode.title,
                "summary": episode.summary,
                "tag": episode.tag,
                "cues": episode.cues,
            }
        )
        episode_id = storage.find_distilled_unit(
            umo=batch.umo,
            unit_type="episode",
            fingerprint=fingerprint,
        )
        if episode_id is None:
            episode_id = storage.store_episode(
                umo=batch.umo,
                started_at=episode.started_at,
                ended_at=episode.ended_at,
                title=episode.title,
                summary=episode.summary,
                source_keys=list(episode.source_keys),
                keywords=[(cue, episode.tag) for cue in episode.cues],
                extractor_version=extractor_version,
            )
            storage.record_distilled_unit(
                umo=batch.umo,
                unit_type="episode",
                fingerprint=fingerprint,
                unit_id=episode_id,
            )
        episode_ids.append(episode_id)
        documents[("episode", str(episode_id))] = IndexDocument(
            owner_type="episode",
            owner_key=str(episode_id),
            text=f"{episode.title}\n{episode.summary}",
        )
        for cue in episode.cues:
            documents[("cue", cue)] = IndexDocument(
                owner_type="cue", owner_key=cue, text=cue
            )

    for semantic in batch.semantic_memories:
        fingerprint = _fingerprint(
            {
                "source_key": semantic.source_key,
                "person": semantic.person,
                "aspect": semantic.aspect,
                "content": semantic.content,
                "confidence": semantic.confidence,
            }
        )
        semantic_id = storage.find_distilled_unit(
            umo=batch.umo,
            unit_type="semantic",
            fingerprint=fingerprint,
        )
        if semantic_id is None:
            semantic_id = storage.store_semantic_memory(
                umo=batch.umo,
                person=semantic.person,
                aspect=semantic.aspect,
                content=semantic.content,
                source_key=semantic.source_key,
                confidence=semantic.confidence,
                extractor_version=extractor_version,
            )
            storage.record_distilled_unit(
                umo=batch.umo,
                unit_type="semantic",
                fingerprint=fingerprint,
                unit_id=semantic_id,
            )
        semantic_ids.append(semantic_id)
        documents[("semantic", str(semantic_id))] = IndexDocument(
            owner_type="semantic",
            owner_key=str(semantic_id),
            text=f"{semantic.person}\n{semantic.aspect}\n{semantic.content}",
        )
        documents[("cue", semantic.person)] = IndexDocument(
            owner_type="cue", owner_key=semantic.person, text=semantic.person
        )

    for topic in batch.topics:
        event_ids = [episode_ids[index] for index in topic.episode_indices]
        topic_id = storage.store_topic(
            umo=batch.umo,
            name=topic.name,
            summary=topic.summary,
            event_ids=event_ids,
            extractor_version=extractor_version,
        )
        topic_ids.append(topic_id)
        documents[("topic", str(topic_id))] = IndexDocument(
            owner_type="topic",
            owner_key=str(topic_id),
            text=f"{topic.name}\n{topic.summary}",
        )

    return PersistedDistillation(
        episode_ids=tuple(episode_ids),
        semantic_ids=tuple(semantic_ids),
        topic_ids=tuple(topic_ids),
        index_documents=tuple(documents.values()),
    )


async def index_distillation(
    storage: MemoryStorage,
    *,
    umo: str,
    persisted: PersistedDistillation,
    backend: EmbeddingBackend,
) -> int:
    documents = list(persisted.index_documents)
    if not documents:
        return 0
    vectors = await backend.embed_texts([document.text for document in documents])
    if len(vectors) != len(documents):
        raise ValueError("embedding backend returned the wrong vector count")
    for document, vector in zip(documents, vectors, strict=True):
        storage.upsert_memory_embedding(
            umo=umo,
            owner_type=document.owner_type,
            owner_key=document.owner_key,
            model=backend.model_id,
            vector=vector,
        )
    return len(documents)
