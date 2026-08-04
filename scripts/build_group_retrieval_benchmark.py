from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


URL_RE = re.compile(r"(?i)\b(?:https?|ftp)://\S+")
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
IP_RE = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")
LONG_NUMBER_RE = re.compile(r"(?<!\d)\d{7,}(?!\d)")
SECRET_RE = re.compile(
    r"(?i)\b(?:sk|api[_-]?key|token|bearer)[-_:= ]?[A-Za-z0-9._-]{12,}\b"
)
SPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[\u3400-\u9fff]|[A-Za-z0-9_]+")

MEDIA_LABELS = {
    "image": "[图片]",
    "record": "[语音]",
    "video": "[视频]",
    "file": "[文件]",
    "forward": "[转发消息]",
    "json": "[卡片]",
    "markdown": "[卡片]",
    "face": "[表情]",
}


@dataclass(slots=True)
class Message:
    ordinal: int
    source_id: str
    sent_at: int
    sender_id: str
    speaker: str
    text: str
    text_chars: int
    reply_source_id: str | None

    @property
    def doc_id(self) -> str:
        return f"d{self.ordinal:06d}"


def scope_hash(group_id: str) -> str:
    return hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:12]


def scrub_text(text: str) -> str:
    value = SECRET_RE.sub("[密钥]", text)
    value = URL_RE.sub("[链接]", value)
    value = EMAIL_RE.sub("[邮箱]", value)
    value = IP_RE.sub("[IP地址]", value)
    value = LONG_NUMBER_RE.sub("[数字标识]", value)
    return SPACE_RE.sub(" ", value).strip()


def _alias_for(sender_id: str, aliases: dict[str, str]) -> str:
    key = sender_id.strip() or "unknown"
    if key not in aliases:
        aliases[key] = f"成员{len(aliases) + 1:03d}"
    return aliases[key]


def parse_message(
    raw_json: str,
    *,
    aliases: dict[str, str],
) -> tuple[str, int, str | None]:
    payload = json.loads(raw_json)
    segments = payload.get("message") or []
    if not isinstance(segments, list):
        return "", 0, None

    text_parts: list[str] = []
    natural_text: list[str] = []
    reply_source_id: str | None = None
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        segment_type = str(segment.get("type") or "")
        data = segment.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        if segment_type == "text":
            fragment = scrub_text(str(data.get("text") or ""))
            if fragment:
                text_parts.append(fragment)
                natural_text.append(fragment)
        elif segment_type == "at":
            target = str(data.get("qq") or "")
            if target == "all":
                text_parts.append("@全体成员")
            elif target:
                text_parts.append(f"@{_alias_for(target, aliases)}")
        elif segment_type == "reply":
            target = data.get("id") or data.get("message_id") or data.get("seq")
            if target is not None:
                reply_source_id = str(target)
        elif segment_type in MEDIA_LABELS:
            text_parts.append(MEDIA_LABELS[segment_type])

    return " ".join(text_parts).strip(), len("".join(natural_text)), reply_source_id


def _tokenize(text: str) -> set[str]:
    units = WORD_RE.findall(text.lower())
    tokens = set(units)
    chinese = "".join(unit for unit in units if len(unit) == 1 and "\u3400" <= unit <= "\u9fff")
    tokens.update(chinese[index : index + 2] for index in range(max(0, len(chinese) - 1)))
    return {token for token in tokens if token.strip()}


def _informativeness(text: str) -> float:
    cleaned = re.sub(r"\[[^]]+]|@成员\d+|[\W_]+", "", text, flags=re.UNICODE)
    if not cleaned:
        return 0.0
    unique_ratio = len(set(cleaned)) / len(cleaned)
    generic = {"哈哈", "笑死", "草", "确实", "不是", "什么", "怎么", "好吧", "然后呢"}
    penalty = 0.15 if cleaned in generic else 0.0
    return min(len(cleaned), 40) / 40 + unique_ratio - penalty


def _context(
    messages: list[Message],
    index: int,
    *,
    radius: int = 2,
    max_gap_seconds: int = 600,
) -> list[dict[str, object]]:
    center = messages[index]
    output: list[dict[str, object]] = []
    for candidate in messages[max(0, index - radius) : index + radius + 1]:
        if abs(candidate.sent_at - center.sent_at) > max_gap_seconds:
            continue
        if not candidate.text:
            continue
        output.append(
            {
                "doc_id": candidate.doc_id,
                "sent_at": candidate.sent_at,
                "speaker": candidate.speaker,
                "text": candidate.text,
            }
        )
    return output


def _hard_negatives(
    messages: list[Message],
    *,
    query: Message,
    positive: Message,
    limit: int,
) -> list[dict[str, object]]:
    query_tokens = _tokenize(f"{query.text} {positive.text}")
    ranked: list[tuple[float, Message]] = []
    for candidate in messages:
        if candidate.doc_id in {query.doc_id, positive.doc_id} or candidate.text_chars < 4:
            continue
        if abs(candidate.sent_at - query.sent_at) <= 600:
            continue
        tokens = _tokenize(candidate.text)
        overlap = len(query_tokens & tokens)
        if overlap == 0:
            continue
        union = len(query_tokens | tokens) or 1
        same_speaker_bonus = 0.08 if candidate.sender_id == positive.sender_id else 0.0
        score = overlap / union + same_speaker_bonus
        ranked.append((score, candidate))
    ranked.sort(key=lambda item: (-item[0], item[1].sent_at, item[1].doc_id))
    return [
        {
            "doc_id": message.doc_id,
            "sent_at": message.sent_at,
            "speaker": message.speaker,
            "text": message.text,
        }
        for _, message in ranked[:limit]
    ]


def _select_group(connection: sqlite3.Connection, requested_hash: str | None) -> str:
    rows = connection.execute(
        "SELECT group_id, COUNT(*) AS c FROM messages GROUP BY group_id ORDER BY c DESC"
    ).fetchall()
    if not rows:
        raise RuntimeError("history database has no messages")
    if requested_hash is None:
        return str(rows[0][0])
    matches = [str(row[0]) for row in rows if scope_hash(str(row[0])) == requested_hash]
    if len(matches) != 1:
        raise ValueError(f"scope hash did not identify exactly one group: {requested_hash}")
    return matches[0]


def load_messages(
    connection: sqlite3.Connection,
    *,
    group_id: str,
) -> list[Message]:
    aliases: dict[str, str] = {}
    messages: list[Message] = []
    rows = connection.execute(
        """
        SELECT message_id, time, COALESCE(user_id, ''), raw_json
        FROM messages
        WHERE group_id = ?
        ORDER BY time, id
        """,
        (group_id,),
    )
    for ordinal, (message_id, sent_at, sender_id, raw_json) in enumerate(rows, start=1):
        text, text_chars, reply_source_id = parse_message(raw_json, aliases=aliases)
        messages.append(
            Message(
                ordinal=ordinal,
                source_id=str(message_id),
                sent_at=int(sent_at),
                sender_id=str(sender_id),
                speaker=_alias_for(str(sender_id), aliases),
                text=text,
                text_chars=text_chars,
                reply_source_id=reply_source_id,
            )
        )
    return messages


def write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a private, pseudonymized group-chat retrieval benchmark draft."
    )
    parser.add_argument("database", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scope-hash")
    parser.add_argument("--candidate-limit", type=int, default=120)
    parser.add_argument("--min-reply-gap", type=int, default=3600)
    parser.add_argument("--hard-negatives", type=int, default=8)
    args = parser.parse_args()

    uri = f"file:{args.database.as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        group_id = _select_group(connection, args.scope_hash)
        messages = load_messages(connection, group_id=group_id)
    finally:
        connection.close()

    source_to_index = {message.source_id: index for index, message in enumerate(messages)}
    candidates: list[tuple[float, int, int]] = []
    for query_index, query in enumerate(messages):
        if not query.reply_source_id or query.reply_source_id not in source_to_index:
            continue
        positive_index = source_to_index[query.reply_source_id]
        positive = messages[positive_index]
        gap = query.sent_at - positive.sent_at
        if gap < args.min_reply_gap:
            continue
        if query.text_chars < 5 or positive.text_chars < 8:
            continue
        if query.text.startswith(("/", "!", "。")):
            continue
        score = (
            math.log1p(gap / 3600)
            + _informativeness(query.text)
            + _informativeness(positive.text)
        )
        candidates.append((score, query_index, positive_index))

    candidates.sort(key=lambda item: (-item[0], messages[item[1]].sent_at))
    # Avoid letting one highly active speaker dominate the annotation queue.
    selected: list[tuple[float, int, int]] = []
    per_speaker: collections.Counter[str] = collections.Counter()
    speaker_cap = max(4, math.ceil(args.candidate_limit / 5))
    for item in candidates:
        query = messages[item[1]]
        if per_speaker[query.sender_id] >= speaker_cap:
            continue
        selected.append(item)
        per_speaker[query.sender_id] += 1
        if len(selected) >= args.candidate_limit:
            break

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    scope_id = f"scope-{scope_hash(group_id)}"
    corpus_records = (
        {
            "doc_id": message.doc_id,
            "scope_id": scope_id,
            "sent_at": message.sent_at,
            "speaker": message.speaker,
            "text": message.text,
        }
        for message in messages
        if message.text_chars > 0
    )
    corpus_count = write_jsonl(output / "corpus.jsonl", corpus_records)

    candidate_records: list[dict[str, object]] = []
    for ordinal, (score, query_index, positive_index) in enumerate(selected, start=1):
        query = messages[query_index]
        positive = messages[positive_index]
        candidate_records.append(
            {
                "candidate_id": f"c{ordinal:04d}",
                "scope_id": scope_id,
                "query_doc_id": query.doc_id,
                "query_time": query.sent_at,
                "observed_query": query.text,
                "positive_doc_id": positive.doc_id,
                "positive_text": positive.text,
                "positive_speaker": positive.speaker,
                "gap_seconds": query.sent_at - positive.sent_at,
                "selection_score": round(score, 4),
                "target_context": _context(messages, positive_index),
                "query_context": _context(messages, query_index),
                "hard_negatives": _hard_negatives(
                    messages,
                    query=query,
                    positive=positive,
                    limit=args.hard_negatives,
                ),
            }
        )
    candidate_count = write_jsonl(output / "candidates.jsonl", candidate_records)

    manifest = {
        "format_version": 1,
        "scope_id": scope_id,
        "source": "Angel Eye SQLite history cache (read-only export)",
        "privacy": {
            "group_id_exported": False,
            "sender_ids_exported": False,
            "nicknames_exported": False,
            "obvious_urls_emails_ips_long_numbers_redacted": True,
            "contains_private_message_text": True,
            "git_safe": False,
        },
        "messages_in_scope": len(messages),
        "corpus_documents": corpus_count,
        "reply_candidates_before_cap": len(candidates),
        "candidate_documents": candidate_count,
        "min_reply_gap_seconds": args.min_reply_gap,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
