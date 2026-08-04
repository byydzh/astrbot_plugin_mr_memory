from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import sqlite3
from pathlib import Path


def _scope_hash(group_id: str) -> str:
    return hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:12]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print privacy-safe aggregate statistics for an Angel Eye history DB."
    )
    parser.add_argument("database", type=Path)
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Also inspect message structure and reply-link coverage without printing text.",
    )
    args = parser.parse_args()

    uri = f"file:{args.database.as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        rows = connection.execute(
            """
            SELECT
                group_id,
                COUNT(*) AS message_count,
                MIN(time) AS first_time,
                MAX(time) AS last_time,
                COUNT(DISTINCT user_id) AS sender_count,
                ROUND(AVG(LENGTH(COALESCE(search_text, ''))), 1) AS mean_chars,
                SUM(CASE WHEN LENGTH(TRIM(COALESCE(search_text, ''))) >= 8
                         THEN 1 ELSE 0 END) AS usable_count
            FROM messages
            GROUP BY group_id
            ORDER BY message_count DESC
            """
        ).fetchall()
    finally:
        connection.close()

    print(f"groups\t{len(rows)}")
    print("scope_hash\tmessages\tfirst_time\tlast_time\tsenders\tmean_chars\tusable")
    for group_id, count, first, last, senders, mean_chars, usable in rows:
        first_iso = dt.datetime.fromtimestamp(first).isoformat() if first else ""
        last_iso = dt.datetime.fromtimestamp(last).isoformat() if last else ""
        print(
            f"{_scope_hash(str(group_id))}\t{count}\t{first_iso}\t{last_iso}"
            f"\t{senders}\t{mean_chars}\t{usable}"
        )

    if not args.deep:
        return

    connection = sqlite3.connect(uri, uri=True)
    try:
        records = connection.execute(
            "SELECT group_id, message_id, time, raw_json FROM messages ORDER BY group_id, time"
        )
        by_scope: dict[str, dict[str, object]] = {}
        for group_id, message_id, sent_at, raw_json in records:
            scope = by_scope.setdefault(
                str(group_id),
                {
                    "ids": {},
                    "text_lengths": [],
                    "segment_types": collections.Counter(),
                    "reply_links": [],
                    "parse_errors": 0,
                },
            )
            ids = scope["ids"]
            assert isinstance(ids, dict)
            ids[str(message_id)] = int(sent_at)
            try:
                payload = json.loads(raw_json)
            except (TypeError, json.JSONDecodeError):
                scope["parse_errors"] = int(scope["parse_errors"]) + 1
                continue
            segments = payload.get("message", [])
            if not isinstance(segments, list):
                continue
            text_parts: list[str] = []
            segment_types = scope["segment_types"]
            reply_links = scope["reply_links"]
            assert isinstance(segment_types, collections.Counter)
            assert isinstance(reply_links, list)
            for segment in segments:
                if not isinstance(segment, dict):
                    continue
                segment_type = str(segment.get("type") or "unknown")
                segment_types[segment_type] += 1
                data = segment.get("data") or {}
                if segment_type == "text" and isinstance(data, dict):
                    text_parts.append(str(data.get("text") or ""))
                elif segment_type == "reply" and isinstance(data, dict):
                    target = data.get("id") or data.get("message_id") or data.get("seq")
                    if target is not None:
                        reply_links.append((int(sent_at), str(target)))
            text_lengths = scope["text_lengths"]
            assert isinstance(text_lengths, list)
            text_lengths.append(len("".join(text_parts).strip()))

        print("\ndeep_structure")
        print(
            "scope_hash\tnonempty_text\tmedian_chars\tp90_chars\treply_messages"
            "\tmatched_replies\tmatch_gt_1h\tmatch_gt_1d\tsegment_types\tparse_errors"
        )
        for group_id, scope in sorted(
            by_scope.items(), key=lambda item: len(item[1]["text_lengths"]), reverse=True
        ):
            lengths = sorted(int(value) for value in scope["text_lengths"])
            ids = scope["ids"]
            replies = scope["reply_links"]
            assert isinstance(ids, dict)
            assert isinstance(replies, list)
            matched_delays = [
                reply_time - ids[target]
                for reply_time, target in replies
                if target in ids and reply_time >= ids[target]
            ]

            def percentile(fraction: float) -> int:
                if not lengths:
                    return 0
                return lengths[round((len(lengths) - 1) * fraction)]

            segment_types = scope["segment_types"]
            assert isinstance(segment_types, collections.Counter)
            type_summary = ",".join(
                f"{name}:{count}" for name, count in segment_types.most_common()
            )
            print(
                f"{_scope_hash(group_id)}\t{sum(value > 0 for value in lengths)}"
                f"\t{percentile(0.5)}\t{percentile(0.9)}\t{len(replies)}"
                f"\t{len(matched_delays)}\t{sum(value >= 3600 for value in matched_delays)}"
                f"\t{sum(value >= 86400 for value in matched_delays)}\t{type_summary}"
                f"\t{scope['parse_errors']}"
            )
    finally:
        connection.close()


if __name__ == "__main__":
    main()
