"""Compile a retrieved evidence tree into a bounded, source-deduplicated pack.

Retrieval deliberately exposes several useful views of the same message (for
example an episode, a participant history and recent context).  Passing those
views through unchanged makes the reader pay for the same raw payload several
times and can accidentally make one message look like several independent
pieces of evidence.  This module keeps the views, but moves every raw message
payload into one source catalog and leaves source-bound references behind.

The compiler is deterministic and side-effect free.  It does not resolve
natural-language identity references: competing participant candidates and
their selected alias observations remain available to the semantic reader.
"""

from __future__ import annotations

import heapq
import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Mapping, TypeVar
from zoneinfo import ZoneInfo

from .activity_statistics import validate_activity_window_statistics
from .identity import canonical_participant_key
from .retrieval_terms import recall_coverage_terms


EVIDENCE_ATOM_PACK_FORMAT = "MR_MEMORY_EVIDENCE_ATOM_PACK_V1"
_SOURCE_TIMEZONE = ZoneInfo("Asia/Shanghai")
_DEFINITION_QUESTION_RE = re.compile(
    r"(?<!不)(?:是什么|是啥|是谁|指的是)|(?<![是不])什么意思"
)
_SOURCE_FIELD_CONTRACT = {
    "time": (
        "local_datetime is calculated by the host from this message's sent_at "
        "in Asia/Shanghai. Use it directly; do not recalculate epoch timestamps."
    ),
    "authorship": (
        "Top-level plain_text belongs to the top-level sender_participant_key. "
        "A reply component quotes another message: its sender_id and plain_text "
        "describe the quoted speaker and words, never the top-level author."
    ),
    "quoted_time": (
        "Reply-component timestamps are adapter-reported quote metadata, not "
        "independently verified original-message times. A quote is not a "
        "separate source unless its original message is in the source catalog."
    ),
    "activity": (
        "Message timestamps establish observed speaking activity, not waking "
        "or falling asleep. Distinguish a speaker's report, another person's "
        "question, and a prediction; never promote them to observed sleep events."
    ),
    "lexical_context": (
        "lexical_context contains bounded chronological neighbors of a literal "
        "query hit, not verified replies or evidence of an identity/relationship. "
        "Use each source's own author and time; context_truncated means some "
        "retrieved neighbors did not fit the shared source cap."
    ),
    "lexical_candidates": (
        "Structured-reference and definition-question facets preserve different "
        "literal-hit candidates for reading, not confirmed nickname bindings or "
        "answers. A reply relationship requires its original target source to "
        "remain in the catalog; chronological neighbors do not prove an answer."
    ),
    "derived_source_closure": (
        "source_closure counts only the evidence sources known in this input "
        "packet, not all historical evidence. SOURCE_CLOSURE_TRUNCATED means "
        "a stored derivation's text was withheld because some of its sources "
        "were omitted. Retrieve the missing evidence or retain explicit "
        "uncertainty; this is not an empty memory or proof of no event."
    ),
}

_DROP = object()
_T = TypeVar("_T")

# These fields belong to the raw message catalog.  Relation-specific fields
# such as ``confidence`` or ``evidence_role`` stay in the derived view that
# gives them meaning.
_SOURCE_CATALOG_FIELDS = (
    "sent_at",
    "local_datetime",
    "timezone",
    "sender_id",
    "sender_name",
    "sender_participant_key",
    "role",
    "plain_text",
    "reply_to_source_key",
    "mentions",
    "components",
    "component_types",
    "revision_no",
    "message_row_id",
    "source_message_id",
)

_RAW_REFERENCE_FIELDS = frozenset(_SOURCE_CATALOG_FIELDS)
_IMMUTABLE_CONFLICT_FIELDS = frozenset(
    {
        "sent_at",
        "sender_id",
        "sender_participant_key",
        "plain_text",
        "components",
        "revision_no",
    }
)

# The catalog cap is shared by several retrieval views.  A single flat sort
# lets a high-cardinality view (most notably alias observations or lexical
# hits) consume every slot before the reader sees any episodic or recent
# context.  These strata define one *minimum-coverage* objective per kind of
# evidence.  They do not increase the cap and they do not manufacture missing
# evidence: a stratum participates only when at least one collected source
# actually belongs to it.  Activity is a coverage objective only for an
# activity query; an unexpected activity view may still be retained later if
# spare capacity remains.
_COVERAGE_STRATA = (
    "reply",
    "person",
    "semantic",
    "activity",
    "lexical",
    "episode",
    "history",
    "recent",
    "resident",
)
_ORIGIN_STRATA = {
    "reply": "reply",
    "person_alias": "person",
    "person_alias_extra": "person",
    "semantic": "semantic",
    "feedback": "semantic",
    "activity": "activity",
    "lexical": "lexical",
    "episode": "episode",
    "history": "history",
    "recent": "recent",
    "resident": "resident",
}


@dataclass
class _SourceCandidate:
    source_key: str
    priority: int
    order: int
    origins: list[str] = field(default_factory=list)
    query_facets: list[str] = field(default_factory=list)
    continuity_facets: list[str] = field(default_factory=list)
    payload: dict[str, object] = field(default_factory=dict)
    conflicting_fields: set[str] = field(default_factory=set)


class _SourceCollector:
    def __init__(self) -> None:
        self._candidates: dict[str, _SourceCandidate] = {}
        self._next_order = 0

    @property
    def candidates(self) -> dict[str, _SourceCandidate]:
        return self._candidates

    def add(
        self,
        source_key: object,
        *,
        priority: int,
        origin: str,
        record: Mapping[str, object] | None = None,
        query_facets: Iterable[str] = (),
        continuity_facets: Iterable[str] = (),
    ) -> None:
        normalized = str(source_key or "").strip()
        if not normalized:
            return
        candidate = self._candidates.get(normalized)
        if candidate is None:
            candidate = _SourceCandidate(
                source_key=normalized,
                priority=int(priority),
                order=self._next_order,
            )
            self._next_order += 1
            self._candidates[normalized] = candidate
        elif priority < candidate.priority:
            candidate.priority = int(priority)
        if origin not in candidate.origins:
            candidate.origins.append(origin)
        for facet in query_facets:
            normalized_facet = str(facet or "").strip()
            if normalized_facet and normalized_facet not in candidate.query_facets:
                candidate.query_facets.append(normalized_facet)
        for facet in continuity_facets:
            normalized_facet = str(facet or "").strip()
            if (
                normalized_facet
                and normalized_facet not in candidate.continuity_facets
            ):
                candidate.continuity_facets.append(normalized_facet)
        if record is None:
            return
        for key in _SOURCE_CATALOG_FIELDS:
            if key not in record:
                continue
            value = record.get(key)
            if value is None or value == "":
                continue
            if key not in candidate.payload:
                candidate.payload[key] = deepcopy(value)
            elif candidate.payload[key] != value:
                # A frozen source should be immutable.  Keep the first value
                # (which came from the higher-priority view) and expose the
                # inconsistency as coverage metadata rather than duplicating
                # alternative message bodies.
                candidate.conflicting_fields.add(key)

    def walk(
        self,
        value: object,
        *,
        priority: int,
        origin: str,
        query_facets: Iterable[str] = (),
        continuity_facets: Iterable[str] = (),
    ) -> None:
        stable_query_facets = tuple(query_facets)
        stable_continuity_facets = tuple(continuity_facets)
        if isinstance(value, Mapping):
            direct_key = value.get("source_key")
            if direct_key:
                self.add(
                    direct_key,
                    priority=priority,
                    origin=origin,
                    record=value,
                    query_facets=stable_query_facets,
                    continuity_facets=stable_continuity_facets,
                )
            for key, item in value.items():
                if (
                    (key == "source_keys" or key.endswith("_source_keys"))
                    and isinstance(item, (list, tuple))
                ):
                    for source_key in item:
                        self.add(
                            source_key,
                            priority=priority,
                            origin=origin,
                            query_facets=stable_query_facets,
                            continuity_facets=stable_continuity_facets,
                        )
                self.walk(
                    item,
                    priority=priority,
                    origin=origin,
                    query_facets=stable_query_facets,
                    continuity_facets=stable_continuity_facets,
                )
        elif isinstance(value, (list, tuple)):
            for item in value:
                self.walk(
                    item,
                    priority=priority,
                    origin=origin,
                    query_facets=stable_query_facets,
                    continuity_facets=stable_continuity_facets,
                )


def _mapping_list(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _round_robin(groups: Iterable[list[_T]]) -> Iterable[_T]:
    pending = [group for group in groups if group]
    offset = 0
    while pending:
        next_pending: list[list[Mapping[str, object]]] = []
        for group in pending:
            if offset < len(group):
                yield group[offset]
            if offset + 1 < len(group):
                next_pending.append(group)
        pending = next_pending
        offset += 1


def _person_observation_groups(value: object) -> list[list[Mapping[str, object]]]:
    if not isinstance(value, Mapping):
        return []
    groups: list[list[Mapping[str, object]]] = []
    for reference in _mapping_list(value.get("references")):
        candidates = reference.get("candidate_participants")
        if not isinstance(candidates, (list, tuple)):
            candidates = reference.get("participants")
        for candidate in _mapping_list(candidates):
            observations = _mapping_list(candidate.get("alias_observations"))
            if observations:
                groups.append(observations)
    return groups


def _person_reference_observations(
    value: object,
) -> list[tuple[str, list[Mapping[str, object]], Mapping[str, object]]]:
    """Return one source-backed selection facet per textual reference.

    Facet identifiers are structural rather than copied from the query.  This
    keeps selection metadata safe to log while still preventing the first
    alias in an enumeration from consuming the complete source cap.
    """

    if not isinstance(value, Mapping):
        return []
    references: list[tuple[str, list[Mapping[str, object]], Mapping[str, object]]] = []
    for reference_index, reference in enumerate(_mapping_list(value.get("references"))):
        candidates = reference.get("candidate_participants")
        if not isinstance(candidates, (list, tuple)):
            candidates = reference.get("participants")
        candidate_groups = [
            observations
            for candidate in _mapping_list(candidates)
            if (observations := _mapping_list(candidate.get("alias_observations")))
        ]
        observations = list(_round_robin(candidate_groups))
        if observations:
            references.append(
                (
                    f"person_reference:{reference_index:03d}",
                    [dict(item) for item in observations if isinstance(item, Mapping)],
                    reference,
                )
            )
    return references


def _ordered_activity_messages(
    value: object,
) -> list[tuple[int, int, str, Mapping[str, object]]]:
    """Return source-bound activity messages with a validated time axis."""

    ordered: list[tuple[int, int, str, Mapping[str, object]]] = []
    for index, message in enumerate(_mapping_list(value)):
        sent_at = message.get("sent_at")
        if isinstance(sent_at, bool) or not isinstance(sent_at, int):
            raise ValueError("activity message sent_at must be an integer")
        source_key = str(message.get("source_key") or "").strip()
        if not source_key:
            continue
        ordered.append((sent_at, index, source_key, message))
    ordered.sort(key=lambda item: item[:3])
    return ordered


def _activity_boundary_source_keys(value: object) -> tuple[str, str] | None:
    """Return chronological activity boundaries after strict time validation."""

    ordered = _ordered_activity_messages(value)
    if not ordered:
        return None
    return ordered[0][2], ordered[-1][2]


def _positional_boundary_source_keys(value: object) -> tuple[str, str] | None:
    """Return the first and last source keys in the supplied list order."""

    source_keys = [
        source_key
        for message in _mapping_list(value)
        if (source_key := str(message.get("source_key") or "").strip())
    ]
    if not source_keys:
        return None
    return source_keys[0], source_keys[-1]


def _time_spread_messages(value: object) -> list[Mapping[str, object]]:
    """Return source messages in a deterministic time-spanning prefix order.

    Activity retrieval already returns a bounded chronological sample.  A
    second global source cap must not turn that sample back into an early-time
    prefix.  Admit the chronological endpoints first, then repeatedly admit
    the midpoint of the largest remaining rank interval.  Consequently every
    prefix of at least two records retains the complete observed time span,
    while longer prefixes progressively cover its interior.
    """

    ordered = _ordered_activity_messages(value)
    if len(ordered) <= 2:
        return [item[3] for item in ordered]

    spread = [ordered[0][3], ordered[-1][3]]
    intervals: list[tuple[int, int, int]] = [
        (-(len(ordered) - 1), 0, len(ordered) - 1)
    ]
    while intervals:
        _negative_span, left, right = heapq.heappop(intervals)
        if right - left <= 1:
            continue
        midpoint = (left + right) // 2
        spread.append(ordered[midpoint][3])
        if midpoint - left > 1:
            heapq.heappush(
                intervals,
                (-(midpoint - left), left, midpoint),
            )
        if right - midpoint > 1:
            heapq.heappush(
                intervals,
                (-(right - midpoint), midpoint, right),
            )
    return spread


def _evidence_groups(value: object) -> list[list[Mapping[str, object]]]:
    return [
        evidence
        for item in _mapping_list(value)
        if (evidence := _mapping_list(item.get("evidence")))
    ]


def _message_groups(value: object) -> list[list[Mapping[str, object]]]:
    return [
        messages
        for item in _mapping_list(value)
        if (messages := _mapping_list(item.get("messages")))
    ]


def _collect_ranked_sources(
    packet: Mapping[str, object],
    *,
    activity_mode: bool,
    person_primary_limit: int,
) -> _SourceCollector:
    collector = _SourceCollector()

    # Direct conversational anchors are always admitted first.
    collector.walk(packet.get("reply_context"), priority=0, origin="reply")

    # Give every competing person candidate one alias observation before any
    # candidate consumes a second slot.  This retains ambiguity for joint
    # semantic reasoning instead of turning catalog truncation into an identity
    # verdict.
    person_candidates = packet.get("person_reference_candidates")
    person_references = _person_reference_observations(person_candidates)
    # Interleave textual references before taking a second observation from
    # any one reference.  This is query-facet coverage, not an identity
    # decision: every retained record remains an undecided source-bound
    # candidate for the Reader.
    person_observations: list[tuple[Mapping[str, object], str]] = []
    offset = 0
    pending = list(person_references)
    while pending:
        next_pending: list[
            tuple[str, list[Mapping[str, object]], Mapping[str, object]]
        ] = []
        for facet, observations, reference in pending:
            if offset < len(observations):
                person_observations.append((observations[offset], facet))
            if offset + 1 < len(observations):
                next_pending.append((facet, observations, reference))
        pending = next_pending
        offset += 1
    for observation, facet in person_observations[
        : max(0, int(person_primary_limit))
    ]:
        collector.walk(
            observation,
            priority=10,
            origin="person_alias",
            query_facets=(facet,),
        )
    # Additional observations remain eligible, but do not starve the actual
    # semantic/lexical/history evidence when a common alias has many candidates.
    for facet, _observations, reference in person_references:
        collector.walk(
            reference,
            priority=65,
            origin="person_alias_extra",
            query_facets=(facet,),
        )

    semantic = _mapping_list(packet.get("semantic_evidence"))
    semantic_groups: list[list[tuple[Mapping[str, object], str]]] = []
    for semantic_index, item in enumerate(semantic):
        facet = f"semantic_entity:{semantic_index:03d}"
        evidence = _mapping_list(item.get("evidence"))
        if evidence:
            semantic_groups.append([(record, facet) for record in evidence])
    for record, facet in _round_robin(semantic_groups):
        collector.walk(
            record,
            priority=20,
            origin="semantic",
            query_facets=(facet,),
        )
    for semantic_index, item in enumerate(semantic):
        collector.walk(
            item,
            priority=20,
            origin="semantic",
            query_facets=(f"semantic_entity:{semantic_index:03d}",),
        )

    feedback = packet.get("feedback_hypothesis_evidence")
    for evidence in _round_robin(_evidence_groups(feedback)):
        collector.walk(evidence, priority=21, origin="feedback")
    collector.walk(feedback, priority=21, origin="feedback")

    activity_priority = 30 if activity_mode else 70
    activity = packet.get("participant_activity")
    if activity_mode:
        activity_items = _mapping_list(activity)
        activity_boundaries: list[tuple[str, str] | None] = [
            _activity_boundary_source_keys(item.get("messages"))
            for item in activity_items
        ]
        for activity_index, item in enumerate(activity_items):
            boundary = activity_boundaries[activity_index]
            for message in _time_spread_messages(item.get("messages")):
                source_key = str(message.get("source_key") or "").strip()
                facets: list[str] = []
                if boundary is not None and source_key == boundary[0]:
                    facets.append(f"activity:{activity_index:03d}:start")
                if boundary is not None and source_key == boundary[1]:
                    facets.append(f"activity:{activity_index:03d}:end")
                collector.walk(
                    message,
                    priority=activity_priority,
                    origin="activity",
                    query_facets=facets,
                )
        collector.walk(
            activity,
            priority=activity_priority,
            origin="activity",
        )

    # Raw FTS hits are an independent retrieval channel.  They should not be
    # rediscovered by the catch-all walk after lower-value context views have
    # already consumed the cap.
    collector.walk(
        packet.get("lexical_messages"),
        priority=35,
        origin="lexical",
    )
    for context_index, group in enumerate(_mapping_list(packet.get("lexical_context"))):
        anchor = str(group.get("anchor_source_key") or "").strip()
        messages = _mapping_list(group.get("messages"))
        anchor_position = next((index for index, message in enumerate(messages)
                                if str(message.get("source_key") or "").strip() == anchor), None)
        for message_index, message in enumerate(messages):
            source_key = str(message.get("source_key") or "").strip()
            role = "anchor" if source_key == anchor else f"neighbor-{message_index:02d}"
            if role != "anchor" and anchor_position is not None:
                direction = "following" if message_index > anchor_position else "preceding"
                role = f"{direction}-{message_index:02d}"
            collector.walk(
                message, priority=36, origin="lexical",
                continuity_facets=(f"lexical_context:{context_index:03d}:{role}",),
            )

    episodes = _mapping_list(packet.get("expanded_episodes"))
    episode_boundaries: list[tuple[str, str] | None] = [
        _positional_boundary_source_keys(item.get("messages")) for item in episodes
    ]
    for message in _round_robin(_message_groups(episodes)):
        source_key = str(message.get("source_key") or "").strip()
        continuity_facets: list[str] = []
        for episode_index, boundary in enumerate(episode_boundaries):
            if boundary is not None and source_key == boundary[0]:
                continuity_facets.append(
                    f"episode:{episode_index:03d}:anchor"
                )
            if boundary is not None and source_key == boundary[1]:
                continuity_facets.append(
                    f"episode:{episode_index:03d}:closure"
                )
        collector.walk(
            message,
            priority=40,
            origin="episode",
            continuity_facets=continuity_facets,
        )
    collector.walk(episodes, priority=40, origin="episode")

    history = packet.get("participant_history")
    for message in _round_robin(_message_groups(history)):
        collector.walk(message, priority=50, origin="history")
    collector.walk(history, priority=50, origin="history")

    collector.walk(packet.get("recent_context"), priority=60, origin="recent")

    resident = packet.get("resident_context")
    if isinstance(resident, Mapping):
        # One continuity stratum shares the existing global source cap. Prefer
        # anchors already recalled this turn so continuity does not spend a
        # second slot on the same source; at most two previous anchors compete.
        anchors = _mapping_list(resident.get("messages"))
        anchors = sorted(anchors, key=lambda item: (
            0 if str(item.get("source_key") or "") in collector.candidates else 1,
        ))[:2]
        collector.walk(anchors, priority=75, origin="resident")

    if not activity_mode:
        for message in _round_robin(_message_groups(activity)):
            collector.walk(message, priority=activity_priority, origin="activity")
        collector.walk(
            activity,
            priority=activity_priority,
            origin="activity",
        )

    # Pick up any source-bound graph/media/candidate records introduced by a
    # compatible future retriever.  They remain lower priority than the stable
    # online views above.
    collector.walk(packet.get("candidates"), priority=80, origin="candidate")
    ranked_top_levels = {
        "reply_context",
        "person_reference_candidates",
        "semantic_evidence",
        "feedback_hypothesis_evidence",
        "participant_activity",
        "lexical_messages",
        "lexical_context",
        "expanded_episodes",
        "participant_history",
        "recent_context",
        "resident_context",
        "stored_derivations",
        "candidates",
    }
    for key, value in packet.items():
        if key not in ranked_top_levels:
            collector.walk(value, priority=90, origin=f"other:{key}")
    return collector


def _candidate_coverage_strata(
    candidate: _SourceCandidate,
    *,
    activity_mode: bool,
) -> frozenset[str]:
    strata = {
        stratum
        for origin in candidate.origins
        if (stratum := _ORIGIN_STRATA.get(origin)) is not None
    }
    if not activity_mode:
        strata.discard("activity")
    return frozenset(strata)


def _select_stratified_sources(
    ranked: list[_SourceCandidate],
    *,
    max_sources: int,
    activity_mode: bool,
) -> list[_SourceCandidate]:
    """Select a strict-cap catalog with deterministic facet/layer coverage.

    A reply anchor and one source from every present retrieval stratum are
    admitted before explicit query facets can consume the remaining cap.
    Neighbors of selected lexical anchors follow query facets, before episode
    boundary continuity and stable-priority fill.
    Selection uses marginal coverage, so a deduplicated source shared by two
    still-uncovered objectives is preferred over a source that advances only
    one.

    When the cap is smaller than the number of disjoint present strata, the
    deterministic stratum order above breaks equal-gain ties; truncation is
    still exposed by the caller and therefore cannot be interpreted as
    semantic absence.
    """

    if not ranked or max_sources <= 0:
        return []

    strata_by_source = {
        candidate.source_key: _candidate_coverage_strata(
            candidate,
            activity_mode=activity_mode,
        )
        for candidate in ranked
    }
    query_facets_by_source = {
        candidate.source_key: frozenset(candidate.query_facets)
        for candidate in ranked
    }
    continuity_facets_by_source = {
        candidate.source_key: frozenset(candidate.continuity_facets)
        for candidate in ranked
    }
    present = {
        stratum
        for strata in strata_by_source.values()
        for stratum in strata
    }
    present_query_facets = {
        facet
        for facets in query_facets_by_source.values()
        for facet in facets
    }
    present_continuity_facets = {
        facet
        for facets in continuity_facets_by_source.values()
        for facet in facets
    }
    stratum_rank = {
        stratum: index for index, stratum in enumerate(_COVERAGE_STRATA)
    }
    selected_keys: set[str] = set()
    covered: set[str] = set()
    covered_query_facets: set[str] = set()
    covered_continuity_facets: set[str] = set()

    def stratum_selection_key(candidate: _SourceCandidate) -> tuple[object, ...]:
        newly_covered = strata_by_source[candidate.source_key].difference(covered)
        ordered_new = tuple(
            stratum_rank[stratum]
            for stratum in _COVERAGE_STRATA
            if stratum in newly_covered
        )
        return (
            -len(newly_covered),
            ordered_new,
            -len(query_facets_by_source[candidate.source_key].difference(covered_query_facets)),
            -len(continuity_facets_by_source[candidate.source_key]),
            candidate.priority,
            candidate.order,
            candidate.source_key,
        )

    def query_facet_selection_key(
        candidate: _SourceCandidate,
    ) -> tuple[object, ...]:
        new_query_facets = query_facets_by_source[candidate.source_key].difference(
            covered_query_facets
        )
        new_strata = strata_by_source[candidate.source_key].difference(covered)
        return (
            -len(new_query_facets),
            -len(new_strata),
            candidate.priority,
            candidate.order,
            candidate.source_key,
        )

    def continuity_selection_key(
        candidate: _SourceCandidate,
    ) -> tuple[object, ...]:
        new_facets = continuity_facets_by_source[candidate.source_key].difference(
            covered_continuity_facets
        )
        represented_groups = {
            facet.rpartition(":")[0] for facet in covered_continuity_facets
        }
        candidate_groups = {facet.rpartition(":")[0] for facet in new_facets}
        completes_selected_group = bool(
            represented_groups.intersection(candidate_groups)
        )
        return (
            -len(new_facets),
            0 if completes_selected_group else 1,
            candidate.priority,
            candidate.order,
            candidate.source_key,
        )

    def admit(candidate: _SourceCandidate) -> None:
        selected_keys.add(candidate.source_key)
        covered.update(strata_by_source[candidate.source_key])
        covered_query_facets.update(query_facets_by_source[candidate.source_key])
        covered_continuity_facets.update(
            continuity_facets_by_source[candidate.source_key]
        )

    # Quoted/replied-to content is the direct conversational anchor.  Preserve
    # it even if another source happens to cover more indirect strata.
    if "reply" in present:
        reply_candidates = [
            candidate
            for candidate in ranked
            if "reply" in strata_by_source[candidate.source_key]
        ]
        admit(min(reply_candidates, key=stratum_selection_key))

    while len(selected_keys) < max_sources and not present.issubset(covered):
        candidates = [
            candidate
            for candidate in ranked
            if candidate.source_key not in selected_keys
            and strata_by_source[candidate.source_key].difference(covered)
        ]
        if not candidates:
            break
        admit(min(candidates, key=stratum_selection_key))

    while (
        len(selected_keys) < max_sources
        and not present_query_facets.issubset(covered_query_facets)
    ):
        candidates = [
            candidate
            for candidate in ranked
            if candidate.source_key not in selected_keys
            and query_facets_by_source[candidate.source_key].difference(
                covered_query_facets
            )
        ]
        if not candidates:
            break
        admit(min(candidates, key=query_facet_selection_key))

    # All mandatory layers and query facets have already had their turn. Give
    # each selected human definition question at most one context slot before
    # ordinary neighbor expansion, preferring the next chronological message.
    # It is a reading opportunity, not a claim that the next message answers it.
    definition_groups = {
        facet.rpartition(":")[0]
        for candidate in ranked if candidate.source_key in selected_keys
        and any(value.startswith("lexical_definition:") for value in candidate.query_facets)
        for facet in candidate.continuity_facets
        if facet.startswith("lexical_context:") and facet.endswith(":anchor")
    }
    for group in sorted(definition_groups):
        following = [candidate for candidate in ranked if any(
            facet.startswith(group + ":following-") for facet in candidate.continuity_facets
        )]
        preceding = [candidate for candidate in ranked if any(
            facet.startswith(group + ":preceding-") for facet in candidate.continuity_facets
        )]
        preferred = following or preceding
        if not preferred or any(candidate.source_key in selected_keys for candidate in preferred):
            continue
        if len(selected_keys) >= max_sources:
            break
        admit(min(preferred, key=lambda candidate: (candidate.order, candidate.source_key)))

    # A lexical neighbor is valuable as context only after its literal anchor
    # was admitted. This never treats chronological adjacency as a reply edge.
    while len(selected_keys) < max_sources:
        anchor_groups = {
            facet.rpartition(":")[0] for facet in covered_continuity_facets
            if facet.startswith("lexical_context:") and facet.endswith(":anchor")
        }
        candidates = [
            candidate for candidate in ranked
            if candidate.source_key not in selected_keys
            and any(
                facet.rpartition(":")[0] in anchor_groups
                for facet in continuity_facets_by_source[candidate.source_key]
                .difference(covered_continuity_facets)
            )
        ]
        if not candidates:
            break
        admit(min(candidates, key=continuity_selection_key))

    while (
        len(selected_keys) < max_sources
        and not present_continuity_facets.issubset(covered_continuity_facets)
    ):
        candidates = [
            candidate
            for candidate in ranked
            if candidate.source_key not in selected_keys
            and any(
                facet.startswith("episode:")
                for facet in continuity_facets_by_source[candidate.source_key]
                .difference(covered_continuity_facets)
            )
        ]
        if not candidates:
            break
        admit(min(candidates, key=continuity_selection_key))

    if len(selected_keys) < max_sources:
        for candidate in ranked:
            if candidate.source_key in selected_keys:
                continue
            selected_keys.add(candidate.source_key)
            if len(selected_keys) >= max_sources:
                break

    # Catalog order stays compatible with the existing priority/order contract;
    # the stratified pass chooses the set, not a new presentation order.
    return [
        candidate
        for candidate in ranked
        if candidate.source_key in selected_keys
    ][:max_sources]


def _is_plural_source_reference(key: str) -> bool:
    return key == "source_keys" or key.endswith("_source_keys")


def _is_singular_source_reference(key: str) -> bool:
    return key != "source_key" and key.endswith("_source_key")


def _project_reference_tree(value: object, selected: set[str]) -> object:
    """Remove duplicate raw payloads and references to unselected sources."""

    if isinstance(value, Mapping):
        direct_source = str(value.get("source_key") or "").strip()
        if direct_source and direct_source not in selected:
            return _DROP

        result: dict[str, object] = {}
        had_plural_sources = False
        kept_plural_sources = False
        for key, item in value.items():
            if key == "source_key":
                if direct_source:
                    result[key] = direct_source
                continue
            if direct_source and key in _RAW_REFERENCE_FIELDS:
                # Centralized in ``sources`` below.
                continue
            if _is_plural_source_reference(key) and isinstance(item, (list, tuple)):
                had_plural_sources = had_plural_sources or bool(item)
                filtered: list[str] = []
                seen_sources: set[str] = set()
                for source in item:
                    normalized_source = str(source or "").strip()
                    if (
                        normalized_source in selected
                        and normalized_source not in seen_sources
                    ):
                        seen_sources.add(normalized_source)
                        filtered.append(normalized_source)
                if filtered:
                    kept_plural_sources = True
                    result[key] = filtered
                continue
            if _is_singular_source_reference(key):
                source_key = str(item or "").strip()
                if source_key in selected:
                    result[key] = source_key
                continue
            projected = _project_reference_tree(item, selected)
            if projected is not _DROP:
                result[key] = projected
        if had_plural_sources and not kept_plural_sources and not direct_source:
            return _DROP
        return result
    if isinstance(value, (list, tuple)):
        projected_items: list[object] = []
        for item in value:
            projected = _project_reference_tree(item, selected)
            if projected is not _DROP:
                projected_items.append(projected)
        return projected_items
    return deepcopy(value)


def _project_lexical_context(value: object, selected: set[str]) -> list[dict[str, object]]:
    """Keep bounded-neighbor coverage explicit without leaking dropped references."""
    result: list[dict[str, object]] = []
    for index, group in enumerate(_mapping_list(value)):
        available = {
            str(message.get("source_key") or "").strip()
            for message in _mapping_list(group.get("messages"))
            if str(message.get("source_key") or "").strip()
        }
        anchor = str(group.get("anchor_source_key") or "").strip()
        projected = _project_reference_tree(group, selected)
        group_copy = dict(projected) if isinstance(projected, Mapping) else {}
        retained = available.intersection(selected)
        group_copy.update({
            "context_group_id": f"lexical_context:{index:03d}",
            "context_relation": "chronological_neighbor_not_reply",
            "context_scope": "bounded_chronological_neighbors_only",
            "anchor_selected": bool(anchor and anchor in selected),
            "available_source_count": len(available),
            "selected_source_count": len(retained),
            "missing_source_count": len(available - retained),
            "context_truncated": bool(available - retained)
            or not bool(anchor and anchor in selected)
            or bool(group.get("messages_truncated")),
        })
        result.append(group_copy)
    return result


def _project_derived_source_closures(
    packet: Mapping[str, object], selected: set[str],
) -> tuple[dict[str, object], dict[str, object]]:
    """Restrict known derived units, never raw message records or host statistics."""
    prepared = deepcopy(dict(packet))
    units: list[tuple[tuple[str, str], dict[str, object], set[str]]] = []

    def add(kind: str, record: object, path: str, evidence: object = None) -> None:
        if not isinstance(record, dict) or record.get("source_key"):
            return
        identity = str(record.get("id") or record.get("stable_key") or path)
        units.append(((kind, identity), record,
                      _source_references(record) | _source_references(evidence)))

    buckets = (("semantic_memories", "semantic"), ("episodes", "episode"),
               ("topics", "topic"), ("associations", "association"),
               ("feedback_hypotheses", "feedback"))
    for container_name, container in (("root", prepared), ("candidates", prepared.get("candidates"))):
        if not isinstance(container, Mapping):
            continue
        for bucket, kind in buckets:
            for index, record in enumerate(_mapping_list(container.get(bucket))):
                add(kind, record, f"{container_name}.{bucket}.{index}")
    for index, record in enumerate(_mapping_list(prepared.get("expanded_episodes"))):
        add("episode", record, f"expanded_episodes.{index}", record.get("messages"))
    groups = (("semantic_evidence", "memory", "semantic"),
              ("feedback_hypothesis_evidence", "hypothesis", "feedback"))
    for bucket, field, kind in groups:
        for index, group in enumerate(_mapping_list(prepared.get(bucket))):
            add(kind, group.get(field), f"{bucket}.{index}.{field}", group.get("evidence"))

    sources_by_unit: dict[tuple[str, str], set[str]] = {}
    for key, _record, sources in units:
        sources_by_unit.setdefault(key, set()).update(sources)
    assertion_fields = {
        "title", "summary", "content", "statement", "uncertainty", "aspect_tag", "aspect",
        "person_cue", "subject_text", "name", "description", "canonical_name",
        "source_label", "source_description", "target_label", "target_description",
        "relation_name", "relation_description", "prospective_cue", "trigger_cues_json",
    }
    for key, record, _sources in units:
        available = sources_by_unit[key]
        retained = available.intersection(selected)
        missing = available - retained
        record["source_closure"] = {
            "basis": "input_packet_evidence_sources_only",
            "available_source_count": len(available), "selected_source_count": len(retained),
            "missing_source_count": len(missing), "truncated": bool(missing),
            "state": ("TRUNCATED_REQUIRES_SOURCE_RETRIEVAL" if missing else
                      "ALL_INPUT_SOURCES_RETAINED" if available else "INPUT_SOURCE_CLOSURE_UNKNOWN"),
        }
        if not missing:
            continue
        record["reader_evidence_basis"] = "STORED_DERIVATION_WITH_MISSING_SOURCES"
        withheld = list(record.get("reader_withheld_fields") or [])
        for field in sorted(assertion_fields):
            text = record.get(field)
            if not isinstance(text, str) or not text:
                continue
            record.pop(field)
            withheld.append({"field": field, "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                             "reason": "SOURCE_CLOSURE_TRUNCATED"})
            metadata = record.get("narrative_bindings")
            if isinstance(metadata, dict) and isinstance(metadata.get("fields"), dict):
                metadata["fields"].pop(field, None)
        record["reader_withheld_fields"] = withheld
    accepted_derivations = {
        ("association" if item.get("kind") == "plastic_edge" else str(item.get("kind")),
         str(item.get("owner_id"))): item
        for item in _mapping_list(prepared.get("stored_derivations"))
    }
    for key, record, _sources in units:
        descriptor = accepted_derivations.get(key)
        if descriptor is None:
            continue
        # Show a host-verified stored narrative once, in its independent
        # directory. Old raw-evidence views retain provenance/owner identity,
        # not another copy that could look like corroborating source words.
        for field in descriptor.get("text", {}):
            record.pop(field, None)
            metadata = record.get("narrative_bindings")
            if isinstance(metadata, dict) and isinstance(metadata.get("fields"), dict):
                metadata["fields"].pop(field, None)
        record["stored_derivation_id"] = descriptor["derivation_id"]
        record["reader_evidence_basis"] = "STORED_DERIVATION_DIRECTORY"
    for bucket, field, _kind in groups:
        for group in _mapping_list(prepared.get(bucket)):
            record = group.get(field)
            if isinstance(record, Mapping) and "source_closure" in record:
                group["source_closure"] = deepcopy(record["source_closure"])
    coverage = {
        "basis": "input_packet_evidence_sources_only", "units_total": len(sources_by_unit),
        "units_truncated": sum(bool(sources - selected) for sources in sources_by_unit.values()),
        "units_with_unknown_input_sources": sum(not sources for sources in sources_by_unit.values()),
    }
    return prepared, coverage


def _has_source_reference(value: object) -> bool:
    if isinstance(value, Mapping):
        if str(value.get("source_key") or "").strip():
            return True
        for key, item in value.items():
            if _is_plural_source_reference(key) and isinstance(item, (list, tuple)):
                if any(str(source or "").strip() for source in item):
                    return True
            if _has_source_reference(item):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_has_source_reference(item) for item in value)
    return False


def _source_references(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        direct = str(value.get("source_key") or "").strip()
        if direct:
            found.add(direct)
        for key, item in value.items():
            if _is_plural_source_reference(key) and isinstance(item, (list, tuple)):
                found.update(
                    str(source).strip()
                    for source in item
                    if str(source or "").strip()
                )
            else:
                found.update(_source_references(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.update(_source_references(item))
    return found


def _participant_relation_source_references(value: object) -> set[str]:
    """Collect raw-message references used to justify a participant relation.

    This collector intentionally mirrors the host's evidence-key namespace:
    the current request is not evidence, while singular and plural source-key
    fields inside an alias/history relation are admissible raw-message links.
    """

    found: set[str] = set()
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            if (
                key != "request_source_key"
                and (key == "source_key" or key.endswith("_source_key"))
                and isinstance(item, str)
            ):
                normalized = item.strip()
                if normalized:
                    found.add(normalized)
            elif _is_plural_source_reference(key) and isinstance(
                item, (list, tuple)
            ):
                found.update(
                    str(source).strip()
                    for source in item
                    if str(source or "").strip()
                )
            else:
                found.update(_participant_relation_source_references(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.update(_participant_relation_source_references(item))
    return found


def _raw_source_platform(source: Mapping[str, object]) -> str:
    key = str(source.get("sender_participant_key") or "")
    if not key.startswith("participant:"):
        return ""
    try:
        identity = json.loads(key.removeprefix("participant:"))
    except (TypeError, ValueError):
        return ""
    if (not isinstance(identity, list) or len(identity) != 2
            or not all(isinstance(part, str) and part for part in identity)
            or identity[1] != str(source.get("sender_id") or "")):
        return ""
    return identity[0] if canonical_participant_key(*identity) == key else ""


def _verified_structured_participants(
    source: Mapping[str, object], catalog: Mapping[str, Mapping[str, object]],
) -> set[str]:
    """Targets from raw top-level mentions or exact, verified catalog replies."""
    platform = _raw_source_platform(source)
    if not platform:
        return set()
    components = _mapping_list(source.get("components"))
    mention_accounts = {
        str(component.get("account_id") or "") for component in components
        if component.get("type") == "mention"
    }
    participants = set()
    for mention in _mapping_list(source.get("mentions")):
        account = str(mention.get("account_id") or "")
        participant = str(mention.get("participant_key") or "")
        if (account and account in mention_accounts
                and participant == canonical_participant_key(platform, account)):
            participants.add(participant)
    reply = catalog.get(str(source.get("reply_to_source_key") or ""))
    if reply is not None and _raw_source_platform(reply) == platform:
        participants.add(str(reply["sender_participant_key"]))
    return participants


def participant_source_bindings(value: object) -> dict[str, list[str]]:
    """Derive the Reader's participant identity/alias source allowlist.

    This is evidence binding, not identity resolution.  Only an explicit
    participant-scoped ``alias_observations``/``messages`` relation, raw author,
    canonical-account-verified mention, or catalog-resolved reply target may
    contribute. Generic ancestor sources and semantic subject bindings cannot.
    Participation in a message never proves an arbitrary nickname relation or
    authorship; direct-speaker evidence uses its separate allowlist below.
    """

    collected: dict[str, set[str]] = {}
    catalog_senders: dict[str, str] = {}
    catalog_records: dict[str, Mapping[str, object]] = {}
    if isinstance(value, Mapping):
        catalog = value.get("sources")
        if isinstance(catalog, (list, tuple)):
            for source in catalog:
                if not isinstance(source, Mapping):
                    continue
                source_key = str(source.get("source_key") or "").strip()
                sender_key = str(
                    source.get("sender_participant_key") or ""
                ).strip()
                if source_key and sender_key:
                    catalog_senders[source_key] = sender_key
                    catalog_records[source_key] = source

    def bind(participant: object, sources: Iterable[str]) -> None:
        participant_key = str(participant or "").strip()
        if not participant_key:
            return
        normalized = {
            str(source).strip()
            for source in sources
            if str(source or "").strip()
        }
        if normalized:
            collected.setdefault(participant_key, set()).update(normalized)

    def visit(node: object) -> None:
        if isinstance(node, Mapping):
            source_key = str(node.get("source_key") or "").strip()
            sender_key = str(node.get("sender_participant_key") or "").strip()
            if source_key and sender_key:
                bind(sender_key, (source_key,))
                for participant in _verified_structured_participants(node, catalog_records):
                    bind(participant, (source_key,))

            participant_key = str(node.get("participant_key") or "").strip()
            if participant_key:
                relation_sources: set[str] = set()
                for relation_key in ("alias_observations", "messages"):
                    relation_value = node.get(relation_key)
                    if isinstance(relation_value, (list, tuple)):
                        for relation_item in relation_value:
                            sources = _participant_relation_source_references(
                                relation_item
                            )
                            relation = (
                                str(relation_item.get("relation") or "").strip()
                                if isinstance(relation_item, Mapping)
                                else ""
                            )
                            requires_same_sender = relation_key == "messages" or (
                                relation_key == "alias_observations"
                                and relation == "SPEAKER"
                            )
                            if requires_same_sender:
                                mismatched = [
                                    source
                                    for source in sources
                                    if catalog_senders.get(source)
                                    and catalog_senders[source] != participant_key
                                ]
                                if mismatched:
                                    raise ValueError(
                                        "participant relation contradicts the "
                                        "authoritative message sender"
                                    )
                            relation_sources.update(sources)
                bind(participant_key, relation_sources)
            for item in node.values():
                visit(item)
        elif isinstance(node, (list, tuple)):
            for item in node:
                visit(item)

    visit(value)
    return {
        participant_key: sorted(source_keys)
        for participant_key, source_keys in sorted(collected.items())
        if source_keys
    }


def participant_speaker_source_bindings(value: object) -> dict[str, list[str]]:
    """Derive direct-speaker evidence only from same-record sender bindings.

    A source is admissible for a speaker exactly when the compact source record
    (or an equivalent uncompiled raw message record) carries both that
    ``source_key`` and ``sender_participant_key``.  Parent participant scopes,
    mentions, subject fields and semantic memories cannot create authorship.
    """

    collected: dict[str, set[str]] = {}

    def visit(node: object) -> None:
        if isinstance(node, Mapping):
            source_key = str(node.get("source_key") or "").strip()
            sender_key = str(node.get("sender_participant_key") or "").strip()
            if source_key and sender_key:
                collected.setdefault(sender_key, set()).add(source_key)
            for item in node.values():
                visit(item)
        elif isinstance(node, (list, tuple)):
            for item in node:
                visit(item)

    visit(value)
    return {
        participant_key: sorted(source_keys)
        for participant_key, source_keys in sorted(collected.items())
        if source_keys
    }


def _prune_source_bound_groups(pack: dict[str, object]) -> None:
    """Drop derived group records whose selected evidence became empty."""

    for key in (
        "semantic_evidence",
        "feedback_hypothesis_evidence",
    ):
        value = pack.get(key)
        if not isinstance(value, list):
            continue
        pack[key] = [
            item
            for item in value
            if isinstance(item, Mapping) and _has_source_reference(item.get("evidence"))
        ]

    for key in (
        "expanded_episodes",
        "participant_history",
    ):
        value = pack.get(key)
        if not isinstance(value, list):
            continue
        pack[key] = [
            item
            for item in value
            if isinstance(item, Mapping) and _has_source_reference(item.get("messages"))
        ]

    activity = pack.get("participant_activity")
    if isinstance(activity, list):
        projected_activity: list[dict[str, object]] = []
        for item in _mapping_list(activity):
            messages = [dict(message) for message in _mapping_list(item.get("messages"))]
            window_statistics = item.get("window_statistics")
            if window_statistics is not None:
                window_statistics = validate_activity_window_statistics(window_statistics)
                if window_statistics["scope"]["participant_key"] != item.get("participant_key"):
                    raise ValueError("activity aggregate participant differs from its activity view")
            if not messages and window_statistics is None:
                continue
            activity_copy = dict(item)
            if window_statistics is not None:
                # Independent host evidence survives raw sample projection,
                # including a zero-sample view. Never recompute this complete
                # population statistic from the retained source catalog.
                activity_copy["window_statistics"] = window_statistics
            original_returned = max(
                len(messages),
                int(item.get("message_count") or 0),
            )
            activity_copy["messages"] = messages
            activity_copy["sample_count"] = len(messages)
            if window_statistics is not None:
                # One authoritative distribution avoids mixing sample zeros
                # with nonzero hours in the complete window. Samples remain
                # available as original-message evidence, not a second total.
                activity_copy.pop("message_count", None)
                activity_copy.pop("hour_histogram", None)
                activity_copy["statistics_basis"] = "host_window_statistics"
            else:
                hour_histogram = {f"{hour:02d}": 0 for hour in range(24)}
                for message in messages:
                    try:
                        local_hour = int(message.get("local_hour"))
                    except (TypeError, ValueError):
                        continue
                    if 0 <= local_hour <= 23:
                        hour_histogram[f"{local_hour:02d}"] += 1
                activity_copy["message_count"] = len(messages)
                activity_copy["hour_histogram"] = hour_histogram
                activity_copy["statistics_basis"] = "atom_pack_selected_source_messages_only"
            activity_copy["sampling_method"] = (
                str(item.get("sampling_method") or "source_backed_samples")
                + "_then_atom_pack_source_cap"
            )
            activity_copy["messages_truncated"] = bool(
                item.get("messages_truncated")
            ) or original_returned > len(messages)
            projected_activity.append(activity_copy)
        pack["participant_activity"] = projected_activity

    person = pack.get("person_reference_candidates")
    if not isinstance(person, Mapping):
        return
    person_copy = dict(person)
    references: list[dict[str, object]] = []
    for reference in _mapping_list(person.get("references")):
        reference_copy = dict(reference)
        candidate_key = "candidate_participants"
        raw_candidates = reference.get(candidate_key)
        if not isinstance(raw_candidates, (list, tuple)):
            candidate_key = "participants"
            raw_candidates = reference.get(candidate_key)
        candidates: list[dict[str, object]] = []
        for candidate in _mapping_list(raw_candidates):
            observations = [
                dict(observation)
                for observation in _mapping_list(candidate.get("alias_observations"))
            ]
            if not observations:
                continue
            candidate_copy = dict(candidate)
            source_count_total = max(
                len(observations),
                int(candidate.get("source_count_total") or 0),
            )
            candidate_copy["alias_observations"] = observations
            candidate_copy["source_count_returned"] = len(observations)
            candidate_copy["observations_truncated"] = bool(
                candidate.get("observations_truncated")
            ) or source_count_total > len(observations)
            candidates.append(candidate_copy)
        if not candidates:
            continue
        reference_copy[candidate_key] = candidates
        candidate_count_total = max(
            len(candidates),
            int(reference.get("candidate_count_total") or 0),
        )
        reference_copy["candidate_count_returned"] = len(candidates)
        reference_copy["truncated"] = bool(
            reference.get("truncated")
        ) or candidate_count_total > len(candidates)
        references.append(reference_copy)
    person_copy["references"] = references
    coverage = person.get("coverage")
    if isinstance(coverage, Mapping):
        coverage_copy = dict(coverage)
        candidate_count_returned = sum(
            len(_mapping_list(reference.get("candidate_participants")))
            + len(_mapping_list(reference.get("participants")))
            for reference in references
        )
        distinct_returned = {
            str(candidate.get("participant_key") or "")
            for reference in references
            for key in ("candidate_participants", "participants")
            for candidate in _mapping_list(reference.get(key))
            if str(candidate.get("participant_key") or "")
        }
        coverage_copy["matched_reference_count_returned"] = len(references)
        coverage_copy["candidate_count_returned"] = candidate_count_returned
        coverage_copy["distinct_candidate_count_returned"] = len(
            distinct_returned
        )
        coverage_copy["truncated"] = bool(coverage.get("truncated")) or any(
            bool(reference.get("truncated")) for reference in references
        )
        person_copy["coverage"] = coverage_copy
    pack["person_reference_candidates"] = person_copy


def _catalog_components(
    value: object,
    *,
    has_plain_text: bool,
) -> object:
    """Avoid repeating the canonical plain text inside component payloads."""

    copied = deepcopy(value)
    if not isinstance(copied, list):
        return copied
    retained: list[object] = []
    for component in copied:
        if not isinstance(component, Mapping):
            retained.append(component)
            continue
        component_type = str(
            component.get("type") or component.get("component_type") or ""
        ).strip().casefold()
        if has_plain_text and component_type in {"plain", "plaintext", "text"}:
            continue
        if component_type in {"reply", "quote", "response_to"}:
            component = dict(component)
            component["content_scope"] = "QUOTED_MESSAGE"
            component["timestamp_basis"] = "ADAPTER_REPORTED_QUOTE_METADATA"
            _set_source_local_datetime(component)
        retained.append(component)
    return retained


def _set_source_local_datetime(source: dict[str, object]) -> None:
    """Calculate display time from authoritative epoch, without interpreting it."""
    source.pop("local_datetime", None)
    source.pop("timezone", None)
    sent_at = source.get("sent_at")
    if isinstance(sent_at, int) and not isinstance(sent_at, bool) and sent_at > 0:
        source["local_datetime"] = datetime.fromtimestamp(
            sent_at, tz=_SOURCE_TIMEZONE
        ).isoformat()
        source["timezone"] = _SOURCE_TIMEZONE.key


def compile_evidence_atom_pack(
    packet: Mapping[str, object],
    *,
    max_sources: int,
    activity_mode: bool = False,
    query: str = "",
) -> dict[str, object]:
    """Return a compact, bounded reader packet without mutating ``packet``.

    ``max_sources`` is a strict global cap.  It must be positive so a present
    reply source can always be retained.  If truncation occurs, the resulting
    coverage explicitly forbids interpreting the bounded pack as proof that no
    semantic memory exists outside it.
    """

    if not isinstance(packet, Mapping):
        raise TypeError("evidence packet must be a mapping")
    safe_max_sources = int(max_sources)
    if safe_max_sources <= 0:
        raise ValueError("max_sources must be positive")

    person_groups = _person_observation_groups(
        packet.get("person_reference_candidates")
    )
    person_primary_limit = max(2, min(6, safe_max_sources // 3))
    collector = _collect_ranked_sources(
        packet,
        activity_mode=bool(activity_mode),
        person_primary_limit=person_primary_limit,
    )
    # A graph can have no alias/entity node for a literal query term while its
    # raw-message hits are already in the candidate set.  Those hits must take
    # part in the same facet coverage objective before episode boundary fill;
    # otherwise an unrelated large episode can evict a whole query dimension.
    # These are literal units from the existing retriever, not identified
    # entities.  Only full source text can establish a match, never a derived
    # summary, participant name, or a guessed identity binding.
    lexical_terms = recall_coverage_terms(query)
    lexical_facets = {
        f"lexical_query:{index:03d}": term
        for index, term in enumerate(lexical_terms)
    }
    raw_catalog = {key: candidate.payload for key, candidate in collector.candidates.items()}
    structured_targets = {
        key: _verified_structured_participants(record, raw_catalog)
        for key, record in raw_catalog.items()
    }
    target_indices = {
        key: index for index, key in enumerate(sorted({
            target for targets in structured_targets.values() for target in targets
        }))
    }
    for candidate in collector.candidates.values():
        source_text = str(candidate.payload.get("plain_text") or "").casefold()
        for facet, term in lexical_facets.items():
            if term.casefold() in source_text:
                candidate.query_facets.append(facet)
                term_index = facet.rpartition(":")[2]
                for target in sorted(structured_targets[candidate.source_key]):
                    candidate.query_facets.append(
                        f"lexical_structured:{term_index}:{target_indices[target]:03d}"
                    )
                # This is a human wording feature, never confirmation that a
                # definition is true. Prior bot prose and negated fragments
                # must not consume the user's definition-question coverage.
                if (str(candidate.payload.get("role") or "").upper() == "USER"
                        and _DEFINITION_QUESTION_RE.search(source_text)):
                    candidate.query_facets.append(f"lexical_definition:{term_index}")
    ranked = sorted(
        collector.candidates.values(),
        key=lambda candidate: (
            candidate.priority,
            candidate.order,
            candidate.source_key,
        ),
    )
    selected_candidates = _select_stratified_sources(
        ranked,
        max_sources=safe_max_sources,
        activity_mode=bool(activity_mode),
    )
    conflicting_selected = {
        candidate.source_key: sorted(
            candidate.conflicting_fields.intersection(_IMMUTABLE_CONFLICT_FIELDS)
        )
        for candidate in selected_candidates
        if candidate.conflicting_fields.intersection(_IMMUTABLE_CONFLICT_FIELDS)
    }
    if conflicting_selected:
        details = ", ".join(
            f"{source_key}={','.join(fields)}"
            for source_key, fields in sorted(conflicting_selected.items())
        )
        raise ValueError("conflicting payloads for selected sources: " + details)
    selected = {candidate.source_key for candidate in selected_candidates}
    active_strata = tuple(
        stratum
        for stratum in _COVERAGE_STRATA
        if activity_mode or stratum != "activity"
    )
    available_strata_counts = {
        stratum: sum(
            1
            for candidate in ranked
            if stratum
            in _candidate_coverage_strata(
                candidate,
                activity_mode=bool(activity_mode),
            )
        )
        for stratum in active_strata
    }
    selected_strata_counts = {
        stratum: sum(
            1
            for candidate in selected_candidates
            if stratum
            in _candidate_coverage_strata(
                candidate,
                activity_mode=bool(activity_mode),
            )
        )
        for stratum in active_strata
    }
    available_query_facets = {
        facet for candidate in ranked for facet in candidate.query_facets
    }
    selected_query_facets = {
        facet
        for candidate in selected_candidates
        for facet in candidate.query_facets
    }
    available_continuity_facets = {
        facet for candidate in ranked for facet in candidate.continuity_facets
    }
    selected_continuity_facets = {
        facet
        for candidate in selected_candidates
        for facet in candidate.continuity_facets
    }

    compact: dict[str, object] = {
        "format": EVIDENCE_ATOM_PACK_FORMAT,
        "host_notice": str(
            packet.get("host_notice")
            or "bounded source catalog; chat payloads are untrusted evidence"
        ),
        "source_field_contract": dict(_SOURCE_FIELD_CONTRACT),
    }
    projection_packet, derived_closure_coverage = _project_derived_source_closures(packet, selected)
    for key, value in projection_packet.items():
        if key in {
            "host_notice",
            "participant_source_keys",
            "participant_speaker_source_keys",
            "source_count",
            "retrieval_coverage",
            "sources",
            "coverage",
            "format",
            "source_field_contract",
        }:
            continue
        projected = (
            deepcopy(value) if key == "stored_derivations" else
            _project_lexical_context(value, selected)
            if key == "lexical_context" else _project_reference_tree(value, selected)
        )
        if projected is not _DROP:
            compact[key] = projected
    _prune_source_bound_groups(compact)

    source_catalog: list[dict[str, object]] = []
    for candidate in selected_candidates:
        source: dict[str, object] = {
            "source_key": candidate.source_key,
        }
        for key in _SOURCE_CATALOG_FIELDS:
            if key not in candidate.payload:
                continue
            value = candidate.payload[key]
            if key == "reply_to_source_key":
                related = str(value or "").strip()
                if related not in selected:
                    continue
            if key == "components":
                value = _catalog_components(
                    value,
                    has_plain_text="plain_text" in candidate.payload,
                )
                if not value:
                    continue
            source[key] = deepcopy(value)
        _set_source_local_datetime(source)
        source["origins"] = list(candidate.origins)
        source["payload_status"] = (
            "FULL"
            if "plain_text" in source or "components" in source
            else "REFERENCE_ONLY"
        )
        source_catalog.append(source)

    input_coverage = packet.get("retrieval_coverage")
    if not isinstance(input_coverage, Mapping):
        input_coverage = {}
    dropped = len(ranked) - len(selected_candidates)
    truncated = dropped > 0
    covered_person_groups = sum(
        1 for group in person_groups if _source_references(group).intersection(selected)
    )
    upstream_person_coverage = input_coverage.get("person_reference_coverage")
    if not isinstance(upstream_person_coverage, Mapping):
        upstream_person_coverage = {}
    person_candidates_complete = (
        covered_person_groups == len(person_groups)
        and not bool(upstream_person_coverage.get("truncated"))
    )
    compact["sources"] = source_catalog
    compact["source_count"] = len(source_catalog)
    compact["retrieval_coverage"] = {
        "input_unique_sources": len(ranked),
        "output_unique_sources": len(source_catalog),
        "dropped": dropped,
        "truncated": truncated,
        "semantic_none_allowed": bool(
            input_coverage.get("semantic_none_allowed", False)
        )
        and not truncated,
        "truncation_meaning": (
            "UNKNOWN_BEYOND_SELECTED_SOURCES"
            if truncated
            else "NO_ADDITIONAL_CATALOG_TRUNCATION"
        ),
        "activity_mode": bool(activity_mode),
        "person_candidate_groups_total": len(person_groups),
        "person_candidate_groups_covered": covered_person_groups,
        "person_candidates_complete": person_candidates_complete,
        "strata_available_sources": available_strata_counts,
        "strata_selected_sources": selected_strata_counts,
        "query_facets_total": len(available_query_facets),
        "query_facets_covered": len(selected_query_facets),
        "query_facets_missing": sorted(
            available_query_facets.difference(selected_query_facets)
        ),
        "lexical_query_terms_total": len(lexical_facets),
        "lexical_query_terms_available": len(
            available_query_facets.intersection(lexical_facets)
        ),
        "lexical_query_terms_covered": len(
            selected_query_facets.intersection(lexical_facets)
        ),
        "lexical_query_terms_dropped": sorted(
            available_query_facets.intersection(lexical_facets).difference(
                selected_query_facets
            )
        ),
        "lexical_query_terms_unmatched": sorted(
            set(lexical_facets).difference(available_query_facets)
        ),
        "lexical_evidence_facets": {
            kind: {
                "total": sum(facet.startswith(prefix) for facet in available_query_facets),
                "covered": sum(facet.startswith(prefix) for facet in selected_query_facets),
                "missing": sorted(facet for facet in available_query_facets - selected_query_facets
                                  if facet.startswith(prefix)),
            }
            for kind, prefix in (("structured_candidates", "lexical_structured:"),
                                 ("definition_questions", "lexical_definition:"))
        },
        "continuity_facets_total": len(available_continuity_facets),
        "continuity_facets_covered": len(selected_continuity_facets),
        "continuity_facets_missing": sorted(
            available_continuity_facets.difference(selected_continuity_facets)
        ),
        "lexical_context_groups_total": len(_mapping_list(compact.get("lexical_context"))),
        "lexical_context_groups_fully_retained": sum(
            1 for group in _mapping_list(compact.get("lexical_context"))
            if not group.get("context_truncated")
        ),
        "lexical_context_groups_truncated": sum(
            1 for group in _mapping_list(compact.get("lexical_context"))
            if group.get("context_truncated")
        ),
        "source_field_conflicts": sum(
            len(candidate.conflicting_fields) for candidate in ranked
        ),
        "derived_source_closure": derived_closure_coverage,
        "upstream": deepcopy(dict(input_coverage)),
    }
    # A compact pack is a complete Reader input contract.  Keeping this map in
    # the compiler prevents alternate callers (tests, offline evaluation, and
    # future runtime paths) from accidentally omitting the attribution guard.
    compact["participant_source_keys"] = participant_source_bindings(compact)
    compact["participant_speaker_source_keys"] = (
        participant_speaker_source_bindings(compact)
    )
    return compact


def hydrate_evidence_atom_pack(
    pack: Mapping[str, object],
    messages: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    """Overlay authoritative snapshot messages onto a selected compact pack.

    Retrieval views may carry only a source reference or a semantic paraphrase.
    Once the global source cap has selected the bounded catalog, the host can
    hydrate those keys in one storage batch.  This function never adds sources;
    it replaces catalog metadata for already-selected keys and rebuilds both
    provenance maps from the resulting immutable message records.
    """

    if str(pack.get("format") or "") != EVIDENCE_ATOM_PACK_FORMAT:
        raise ValueError("source hydration requires an evidence atom pack")
    result = deepcopy(dict(pack))
    catalog = result.get("sources")
    if not isinstance(catalog, list):
        raise ValueError("evidence atom pack sources must be an array")
    selected_sources = {
        str(source.get("source_key") or "").strip(): source
        for source in catalog
        if isinstance(source, dict)
        and str(source.get("source_key") or "").strip()
    }
    hydrated: set[str] = set()
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError("source hydration records must be objects")
        source_key = str(message.get("source_key") or "").strip()
        if not source_key:
            raise ValueError("source hydration returned a blank source key")
        target = selected_sources.get(source_key)
        if target is None:
            raise ValueError("source hydration returned an unselected source")
        if source_key in hydrated:
            raise ValueError("source hydration returned a duplicate source")
        # The selected catalog may have been assembled from a semantic view or
        # a reference-only relation.  Remove every raw-message field first so
        # no stale component type or internal row id can survive beside the
        # authoritative snapshot payload.
        for key in _SOURCE_CATALOG_FIELDS:
            target.pop(key, None)
        for key in _SOURCE_CATALOG_FIELDS:
            if key not in message:
                continue
            value = deepcopy(message.get(key))
            if key == "reply_to_source_key":
                related = str(value or "").strip()
                if related and related not in selected_sources:
                    value = ""
            elif key == "components":
                value = _catalog_components(
                    value,
                    has_plain_text="plain_text" in message,
                )
            target[key] = value
        _set_source_local_datetime(target)
        target["payload_status"] = "FULL"
        target["snapshot_hydrated"] = True
        hydrated.add(source_key)

    missing_count = len(selected_sources) - len(hydrated)
    if missing_count:
        raise ValueError(
            "source hydration is incomplete: "
            f"requested={len(selected_sources)}, returned={len(hydrated)}"
        )

    coverage = result.get("retrieval_coverage")
    coverage_copy = dict(coverage) if isinstance(coverage, Mapping) else {}
    coverage_copy["source_hydration_requested"] = len(selected_sources)
    coverage_copy["source_hydration_returned"] = len(hydrated)
    coverage_copy["source_hydration_missing"] = 0
    result["retrieval_coverage"] = coverage_copy
    result["source_field_contract"] = dict(_SOURCE_FIELD_CONTRACT)
    result["participant_source_keys"] = participant_source_bindings(result)
    result["participant_speaker_source_keys"] = (
        participant_speaker_source_bindings(result)
    )
    return result


__all__ = [
    "EVIDENCE_ATOM_PACK_FORMAT",
    "compile_evidence_atom_pack",
    "hydrate_evidence_atom_pack",
    "participant_source_bindings",
    "participant_speaker_source_bindings",
]
