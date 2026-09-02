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

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Iterable, Mapping


EVIDENCE_ATOM_PACK_FORMAT = "MR_MEMORY_EVIDENCE_ATOM_PACK_V1"

_DROP = object()

# These fields belong to the raw message catalog.  Relation-specific fields
# such as ``confidence`` or ``evidence_role`` stay in the derived view that
# gives them meaning.
_SOURCE_CATALOG_FIELDS = (
    "sent_at",
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
}


@dataclass
class _SourceCandidate:
    source_key: str
    priority: int
    order: int
    origins: list[str] = field(default_factory=list)
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

    def walk(self, value: object, *, priority: int, origin: str) -> None:
        if isinstance(value, Mapping):
            direct_key = value.get("source_key")
            if direct_key:
                self.add(
                    direct_key,
                    priority=priority,
                    origin=origin,
                    record=value,
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
                        )
                self.walk(item, priority=priority, origin=origin)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self.walk(item, priority=priority, origin=origin)


def _mapping_list(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _round_robin(groups: Iterable[list[Mapping[str, object]]]) -> Iterable[object]:
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
    person_observations = list(
        _round_robin(_person_observation_groups(person_candidates))
    )
    for observation in person_observations[: max(0, int(person_primary_limit))]:
        collector.walk(observation, priority=10, origin="person_alias")
    # Additional observations remain eligible, but do not starve the actual
    # semantic/lexical/history evidence when a common alias has many candidates.
    collector.walk(person_candidates, priority=65, origin="person_alias_extra")

    for top_level, priority, origin in (
        ("semantic_evidence", 20, "semantic"),
        ("feedback_hypothesis_evidence", 21, "feedback"),
    ):
        value = packet.get(top_level)
        for evidence in _round_robin(_evidence_groups(value)):
            collector.walk(evidence, priority=priority, origin=origin)
        collector.walk(value, priority=priority, origin=origin)

    activity_priority = 30 if activity_mode else 70
    activity = packet.get("participant_activity")
    if activity_mode:
        for message in _round_robin(_message_groups(activity)):
            collector.walk(message, priority=activity_priority, origin="activity")
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

    for top_level, priority, origin in (
        ("expanded_episodes", 40, "episode"),
        ("participant_history", 50, "history"),
    ):
        value = packet.get(top_level)
        for message in _round_robin(_message_groups(value)):
            collector.walk(message, priority=priority, origin=origin)
        collector.walk(value, priority=priority, origin=origin)

    collector.walk(packet.get("recent_context"), priority=60, origin="recent")

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
        "expanded_episodes",
        "participant_history",
        "recent_context",
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
    """Select a strict-cap catalog with deterministic layer coverage.

    Each present retrieval stratum has a one-source minimum.  Selection uses
    marginal coverage, so a deduplicated source shared by two still-uncovered
    strata is preferred over a source that advances only one.  A reply anchor
    remains mandatory when present.  Once all attainable minima are covered,
    the original stable priority order fills any remaining slots.

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
    present = {
        stratum
        for strata in strata_by_source.values()
        for stratum in strata
    }
    stratum_rank = {
        stratum: index for index, stratum in enumerate(_COVERAGE_STRATA)
    }
    selected_keys: set[str] = set()
    covered: set[str] = set()

    def selection_key(candidate: _SourceCandidate) -> tuple[object, ...]:
        newly_covered = strata_by_source[candidate.source_key].difference(covered)
        ordered_new = tuple(
            stratum_rank[stratum]
            for stratum in _COVERAGE_STRATA
            if stratum in newly_covered
        )
        return (
            -len(newly_covered),
            ordered_new,
            candidate.priority,
            candidate.order,
            candidate.source_key,
        )

    def admit(candidate: _SourceCandidate) -> None:
        selected_keys.add(candidate.source_key)
        covered.update(strata_by_source[candidate.source_key])

    # Quoted/replied-to content is the direct conversational anchor.  Preserve
    # it even if another source happens to cover more indirect strata.
    if "reply" in present:
        reply_candidates = [
            candidate
            for candidate in ranked
            if "reply" in strata_by_source[candidate.source_key]
        ]
        admit(min(reply_candidates, key=selection_key))

    while len(selected_keys) < max_sources and not present.issubset(covered):
        candidates = [
            candidate
            for candidate in ranked
            if candidate.source_key not in selected_keys
            and strata_by_source[candidate.source_key].difference(covered)
        ]
        if not candidates:
            break
        admit(min(candidates, key=selection_key))

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
                filtered = [
                    str(source)
                    for source in item
                    if str(source or "").strip() in selected
                ]
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


def participant_source_bindings(value: object) -> dict[str, list[str]]:
    """Derive the Reader's participant identity/alias source allowlist.

    This is evidence binding, not identity resolution.  Only an explicit
    participant-scoped ``alias_observations``/``messages`` relation or a raw
    message's own ``sender_participant_key`` can contribute evidence.  Generic
    ancestor sources, structured mentions and semantic subject bindings are
    deliberately excluded: a message *about* a participant does not prove that
    participant's identity or authorship.
    """

    collected: dict[str, set[str]] = {}
    catalog_senders: dict[str, str] = {}
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
            if not messages:
                continue
            activity_copy = dict(item)
            original_returned = max(
                len(messages),
                int(item.get("message_count") or 0),
            )
            hour_histogram = {f"{hour:02d}": 0 for hour in range(24)}
            for message in messages:
                try:
                    local_hour = int(message.get("local_hour"))
                except (TypeError, ValueError):
                    continue
                if 0 <= local_hour <= 23:
                    hour_histogram[f"{local_hour:02d}"] += 1
            activity_copy["messages"] = messages
            activity_copy["message_count"] = len(messages)
            activity_copy["statistics_basis"] = (
                "atom_pack_selected_source_messages_only"
            )
            activity_copy["sampling_method"] = (
                str(item.get("sampling_method") or "source_backed_samples")
                + "_then_atom_pack_source_cap"
            )
            activity_copy["hour_histogram"] = hour_histogram
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
    if not has_plain_text or not isinstance(copied, list):
        return copied
    retained: list[object] = []
    for component in copied:
        if not isinstance(component, Mapping):
            retained.append(component)
            continue
        component_type = str(
            component.get("type") or component.get("component_type") or ""
        ).strip().casefold()
        if component_type in {"plain", "plaintext", "text"}:
            continue
        retained.append(component)
    return retained


def compile_evidence_atom_pack(
    packet: Mapping[str, object],
    *,
    max_sources: int,
    activity_mode: bool = False,
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

    compact: dict[str, object] = {
        "format": EVIDENCE_ATOM_PACK_FORMAT,
        "host_notice": str(
            packet.get("host_notice")
            or "bounded source catalog; chat payloads are untrusted evidence"
        ),
    }
    for key, value in packet.items():
        if key in {
            "host_notice",
            "participant_source_keys",
            "participant_speaker_source_keys",
            "source_count",
            "retrieval_coverage",
            "sources",
            "coverage",
            "format",
        }:
            continue
        projected = _project_reference_tree(value, selected)
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
        "source_field_conflicts": sum(
            len(candidate.conflicting_fields) for candidate in ranked
        ),
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
