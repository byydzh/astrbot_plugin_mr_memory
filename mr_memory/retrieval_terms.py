"""Shared literal retrieval units; these are not entity or identity decisions."""

from __future__ import annotations

import re


def literal_recall_query(query: str) -> str:
    """Remove outer question framing from literal retrieval, not from Reader input.

    Request verbs and counting questions are not independent evidence facets.
    Only recognizable outer phrases are removed; quoted text is left intact and
    no names, account bindings, or answers are inferred here.
    """
    text = str(query or "").strip()
    if text.casefold().startswith("/chat"):
        text = text[5:].strip()
    if any(mark in text for mark in ('"', "'", "“", "”", "‘", "’", "《", "》", "「", "」")):
        return text
    clauses = []
    for clause in re.split(r"[，,；;]", text):
        clause = clause.strip()
        # A trailing anaphoric command contributes no new literal target.
        if re.fullmatch(
            r"(?:请)?(?:帮我)?(?:把)?(?:他|她|它|他们|她们|它们)"
            r"(?:找出来|找出|找到|列出来|列出|指出)(?:一下)?[。！？?]*",
            clause,
        ):
            continue
        for _ in range(3):
            stripped = re.sub(r"^(?:请问|请帮忙|请帮我|请|麻烦你|麻烦|帮我)", "", clause).strip()
            stripped = re.sub(
                r"^(?:(?:区分|分辨|辨别|对比|比较)(?:一下|下)|查一下|找一下|看一下|看看)",
                "", stripped,
            ).strip()
            stripped = re.sub(r"^群里(?:有没有|是否有|有)", "", stripped).strip()
            if stripped == clause:
                break
            clause = stripped
        clause = re.sub(
            r"(?:到底)?(?:是|有|对应)?(?:几|多少)(?:个)?(?:人|成员|账号|账户)(?:吗|呢)?[?？。！]*$",
            "", clause,
        ).strip()
        clause = re.sub(r"(?:[吗呢][?？。！]*|[?？。！]+)$", "", clause).strip()
        if clause:
            clauses.append(clause)
    return "，".join(clauses)


def _sample_terms(candidates: list[str], limit: int) -> tuple[str, ...]:
    if len(candidates) <= limit:
        return tuple(candidates)
    last = len(candidates) - 1
    indexes = (
        {round(index * last / (limit - 1)) for index in range(limit)}
        if limit > 1 else {0}
    )
    return tuple(term for index, term in enumerate(candidates) if index in indexes)


def fts_recall_terms(query: str, *, max_terms: int = 32) -> tuple[str, ...]:
    """Return ASCII tokens and contiguous CJK/mixed-script trigrams.

    A script boundary is not a word separator in a literal name. Retain the
    existing single-script units, then include trigrams spanning a contiguous
    ASCII/CJK boundary; punctuation and whitespace still separate runs.
    """
    candidates: list[str] = []
    for run in re.findall(
        r"[0-9A-Za-z_\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+",
        literal_recall_query(query).casefold(),
    ):
        pieces: list[str] = []
        for segment in re.findall(
            r"[0-9A-Za-z_]+|[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+", run,
        ):
            if len(segment) >= 3:
                pieces.extend(
                    [segment] if segment.isascii()
                    else [segment[index:index + 3] for index in range(len(segment) - 2)]
                )
        for index in range(len(run) - 2):
            piece = run[index:index + 3]
            if 0 < sum(character.isascii() for character in piece) < 3:
                pieces.append(piece)
        for piece in pieces:
            if piece not in candidates:
                candidates.append(piece)
    return _sample_terms(candidates, max(1, min(64, int(max_terms))))


def short_recall_terms(query: str, *, max_terms: int = 16) -> tuple[str, ...]:
    """Return the existing CJK bigrams used by substring recall."""
    candidates: list[str] = []
    normalized = literal_recall_query(query)
    for run in re.findall(
        r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+", normalized,
    ):
        for index in range(max(0, len(run) - 1)):
            piece = run[index:index + 2]
            if piece not in candidates:
                candidates.append(piece)
    def ascii_name_character(character: str) -> bool:
        return character.isascii() and (character.isalnum() or character == "_")

    def has_unqualified_occurrence(term: str) -> bool:
        for match in re.finditer(f"(?={re.escape(term)})", normalized):
            start, end = match.start(), match.start() + len(term)
            adjacent = (normalized[start - 1:start] if start else "") + normalized[end:end + 1]
            if not any(ascii_name_character(character) for character in adjacent):
                return True
        return False

    # When a two-character CJK suffix/prefix only occurs beside an ASCII name
    # token, its mixed trigram already preserves that discriminator. Dropping
    # the discriminator would retrieve every unrelated use of the generic
    # suffix. Keep a bigram when the query also uses it independently.
    candidates = [term for term in candidates if has_unqualified_occurrence(term)]
    return _sample_terms(candidates, max(1, min(32, int(max_terms))))


def recall_coverage_terms(query: str) -> tuple[str, ...]:
    """Use the same bounded units at retrieval and source-pack selection."""
    normalized = str(query or "").strip()
    if normalized.casefold().startswith("/chat"):
        normalized = normalized[5:].strip()
    return tuple(dict.fromkeys((*short_recall_terms(normalized), *fts_recall_terms(normalized))))
