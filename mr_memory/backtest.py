from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from .models import NormalizedMessage


@dataclass(frozen=True, slots=True)
class ReverseReplayWindow:
    """One leakage-safe construction batch, processed newest window first."""

    ordinal: int
    messages: tuple[NormalizedMessage, ...]

    @property
    def started_at(self) -> int:
        return self.messages[0].sent_at

    @property
    def ended_at(self) -> int:
        return self.messages[-1].sent_at

    @property
    def source_keys(self) -> tuple[str, ...]:
        return tuple(message.resolved_source_key() for message in self.messages)


@dataclass(frozen=True, slots=True)
class EvidenceGateDecision:
    sufficient: bool
    matched_terms: tuple[str, ...] = ()
    candidate_score: float = 0.0
    reason: str = ""


_GATE_STOPWORDS = {
    "上次",
    "之前",
    "群友",
    "游戏",
    "这个",
    "是不是",
    "今天",
    "记录",
    "聊天",
    "表示",
}


def _salient_terms(text: str) -> set[str]:
    terms = {
        item.casefold()
        for item in re.findall(r"[A-Za-z0-9_+-]{3,}", str(text))
    }
    for chunk in re.findall(r"[\u3400-\u9fff]{2,}", str(text)):
        terms.update(chunk[index : index + 2] for index in range(len(chunk) - 1))
    return {term for term in terms if term not in _GATE_STOPWORDS}


def direct_evidence_gate(
    *,
    query: str,
    tool_name: str,
    arguments: dict[str, Any],
    result: Any,
    initial_candidates: dict[str, Any],
    min_candidate_score: float = 0.48,
) -> EvidenceGateDecision:
    """Stop breadth expansion after a high-score episode is source-verified.

    This gate never declares the final answer true. It only decides that the
    controller has enough raw evidence to enter a final synthesis step.
    """

    if tool_name != "query_event_context" or not isinstance(result, list) or not result:
        return EvidenceGateDecision(False, reason="not grounded event context")
    event_id = arguments.get("event_id")
    if not isinstance(event_id, int):
        return EvidenceGateDecision(False, reason="missing event id")
    episodes = initial_candidates.get("episodes", [])
    candidate = next(
        (
            item
            for item in episodes
            if isinstance(item, dict) and int(item.get("id", -1)) == event_id
        ),
        None,
    )
    if candidate is None:
        return EvidenceGateDecision(False, reason="event was not an initial candidate")
    score = float(candidate.get("score", 0.0) or 0.0)
    if score < float(min_candidate_score):
        return EvidenceGateDecision(
            False,
            candidate_score=score,
            reason="candidate score below threshold",
        )
    query_terms = _salient_terms(query)
    evidence_text = "\n".join(
        str(item.get("plain_text") or "")
        for item in result
        if isinstance(item, dict)
    )
    matched = tuple(sorted(query_terms & _salient_terms(evidence_text)))
    if not matched:
        return EvidenceGateDecision(
            False,
            candidate_score=score,
            reason="no salient lexical overlap",
        )
    if not any(isinstance(item, dict) and item.get("source_key") for item in result):
        return EvidenceGateDecision(
            False,
            candidate_score=score,
            reason="event context lacks source keys",
        )
    return EvidenceGateDecision(
        True,
        matched_terms=matched,
        candidate_score=score,
        reason="high-score initial episode verified by raw source context",
    )


def build_reverse_replay_windows(
    records: Iterable[dict[str, Any] | NormalizedMessage],
    *,
    umo: str,
    cutoff_at: int,
    batch_size: int = 80,
    max_messages: int | None = None,
) -> tuple[ReverseReplayWindow, ...]:
    """Build historical graph batches without admitting the target call or future.

    Windows are returned from newest to oldest so an experiment can progressively
    backfill memory until useful evidence is found. Messages inside each window stay
    chronological because the distillation prompt depends on dialogue order.
    """

    if not umo.strip():
        raise ValueError("umo is required")
    if int(cutoff_at) <= 0:
        raise ValueError("cutoff_at must be positive")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if max_messages is not None and int(max_messages) <= 0:
        raise ValueError("max_messages must be positive when supplied")

    normalized = [
        item if isinstance(item, NormalizedMessage) else NormalizedMessage.from_mapping(item)
        for item in records
    ]
    foreign = {message.umo for message in normalized if message.umo != umo}
    if foreign:
        raise ValueError(f"records cross group scopes: {sorted(foreign)}")

    eligible = [message for message in normalized if message.sent_at < int(cutoff_at)]
    eligible.sort(
        key=lambda message: (
            message.sent_at,
            message.message_id,
            message.resolved_source_key(),
        )
    )
    if max_messages is not None:
        eligible = eligible[-int(max_messages) :]

    windows: list[ReverseReplayWindow] = []
    upper = len(eligible)
    ordinal = 0
    while upper > 0:
        lower = max(0, upper - int(batch_size))
        batch = tuple(eligible[lower:upper])
        if any(message.sent_at >= int(cutoff_at) for message in batch):
            raise AssertionError("future message entered a masked replay window")
        windows.append(ReverseReplayWindow(ordinal=ordinal, messages=batch))
        ordinal += 1
        upper = lower
    return tuple(windows)


def stable_text_hash(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def masked_call_manifest(
    *,
    run_id: str,
    umo: str,
    cutoff_at: int,
    query: str,
    observed_response: str,
    provider_stat_id: int | None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an auditable manifest without duplicating private text in indexes."""

    return {
        "schema_version": 1,
        "run_id": run_id,
        "umo": umo,
        "cutoff_at": int(cutoff_at),
        "query": query,
        "query_sha256": stable_text_hash(query),
        "observed_response": observed_response,
        "observed_response_sha256": stable_text_hash(observed_response),
        "provider_stat_id": provider_stat_id,
        "metadata": metadata or {},
    }


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
