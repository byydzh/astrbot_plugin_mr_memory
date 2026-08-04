from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

from mr_memory.backtest import masked_call_manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_text(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    detail = payload.get("meta", {}).get("detail", {})
    news = detail.get("news", []) if isinstance(detail, dict) else []
    return [
        str(item.get("text") or "").strip()
        for item in news
        if isinstance(item, dict) and str(item.get("text") or "").strip()
    ]


def extract_visible_text(raw_json: str) -> tuple[str, list[str]]:
    """Extract text without retaining media URLs, cookies, or CQ payloads."""

    try:
        value = json.loads(raw_json)
    except (TypeError, json.JSONDecodeError):
        return "", []
    segments = value.get("message") or []
    texts: list[str] = []
    media_types: list[str] = []
    if not isinstance(segments, list):
        return "", []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        kind = str(segment.get("type") or "")
        data = segment.get("data") or {}
        if kind == "text" and isinstance(data, dict):
            text = str(data.get("text") or "").strip()
            if text:
                texts.append(text)
        elif kind == "json" and isinstance(data, dict):
            payload = data.get("data")
            if isinstance(payload, str):
                try:
                    payload = json.loads(html.unescape(payload))
                except json.JSONDecodeError:
                    payload = None
            texts.extend(_json_text(payload))
            media_types.append("forward")
        elif kind in {"image", "video", "record", "file"}:
            media_types.append(kind)
    return "\n".join(dict.fromkeys(texts)).strip(), list(dict.fromkeys(media_types))


def _clean_query(text: str) -> str:
    stripped = text.strip()
    if stripped.casefold().startswith("/chat"):
        stripped = stripped[5:].strip()
    return stripped


def export_call(args: argparse.Namespace) -> dict[str, Any]:
    astrbot_path = Path(args.astrbot_db).resolve()
    history_path = Path(args.history_db).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    astrbot = sqlite3.connect(astrbot_path)
    astrbot.row_factory = sqlite3.Row
    history = sqlite3.connect(history_path)
    history.row_factory = sqlite3.Row
    try:
        stat = astrbot.execute(
            "SELECT * FROM provider_stats WHERE id = ?", (int(args.call_id),)
        ).fetchone()
        if stat is None:
            raise ValueError(f"provider stat not found: {args.call_id}")
        umo = str(stat["umo"])
        parts = umo.split(":", 2)
        if len(parts) != 3 or parts[1] != "GroupMessage":
            raise ValueError(f"call is not a group message: {umo}")
        platform_id, _, group_id = parts
        started_at = float(stat["start_time"])

        prompt_candidates = history.execute(
            """
            SELECT * FROM messages
            WHERE group_id = ? AND time BETWEEN ? AND ?
              AND instr(search_text, ?) > 0
            ORDER BY abs(time - ?), message_seq, id
            """,
            (
                group_id,
                math.floor(started_at) - int(args.prompt_window_seconds),
                math.ceil(started_at) + 1,
                args.command_prefix,
                started_at,
            ),
        ).fetchall()
        if not prompt_candidates:
            raise ValueError("could not align provider stat to a /chat source message")
        prompt = prompt_candidates[0]
        prompt_text, prompt_media = extract_visible_text(str(prompt["raw_json"]))
        query = _clean_query(prompt_text)
        if not query:
            raise ValueError("aligned source message has no visible query text")
        cutoff_at = int(prompt["time"])

        response_rows = history.execute(
            """
            SELECT * FROM messages
            WHERE group_id = ? AND user_id = ?
              AND time >= ? AND time <= ?
            ORDER BY time, message_seq, id
            """,
            (
                group_id,
                str(args.bot_id),
                cutoff_at,
                math.ceil(float(stat["end_time"])) + int(args.response_grace_seconds),
            ),
        ).fetchall()
        response_parts: list[str] = []
        response_source_keys: list[str] = []
        for row in response_rows:
            text, _ = extract_visible_text(str(row["raw_json"]))
            if text:
                if text.startswith("AstrBot:"):
                    text = text[len("AstrBot:") :].strip()
                response_parts.append(text)
                response_source_keys.append(
                    f"angel:{group_id}:{str(row['message_id'])}"
                )
        observed_response = "\n".join(dict.fromkeys(response_parts)).strip()
        if not observed_response:
            raise ValueError("could not extract the observed bot response")

        rows = history.execute(
            """
            SELECT * FROM messages
            WHERE group_id = ? AND time < ?
            ORDER BY time DESC, message_seq DESC, id DESC
            LIMIT ?
            """,
            (group_id, cutoff_at, int(args.max_history_messages)),
        ).fetchall()
        rows = list(reversed(rows))
        normalized: list[dict[str, Any]] = []
        for row in rows:
            text, media_types = extract_visible_text(str(row["raw_json"]))
            if not text and not media_types:
                continue
            if not text:
                text = "[非文本消息:" + ",".join(media_types) + "]"
            role = "BOT" if str(row["user_id"]) == str(args.bot_id) else "USER"
            message_id = str(row["message_id"])
            normalized.append(
                {
                    "platform": "aiocqhttp",
                    "platform_id": platform_id,
                    "umo": umo,
                    "group_id": group_id,
                    "message_id": message_id,
                    "sender_id": str(row["user_id"] or ""),
                    "sender_name": str(row["nickname"] or row["user_id"] or ""),
                    "sent_at": int(row["time"]),
                    "plain_text": text,
                    "content": [{"type": "plain", "text": text}],
                    "role": role,
                    "source_key": f"angel:{group_id}:{message_id}",
                }
            )

        run_id = args.run_id or f"masked-call-{int(args.call_id)}"
        manifest = masked_call_manifest(
            run_id=run_id,
            umo=umo,
            cutoff_at=cutoff_at,
            query=query,
            observed_response=observed_response,
            provider_stat_id=int(stat["id"]),
            metadata={
                "group_id": group_id,
                "prompt_source_key": f"angel:{group_id}:{str(prompt['message_id'])}",
                "prompt_media_types": prompt_media,
                "response_source_keys": response_source_keys,
                "provider_id": str(stat["provider_id"]),
                "provider_model": str(stat["provider_model"] or ""),
                "observed_usage": {
                    "input_other": int(stat["token_input_other"]),
                    "input_cached": int(stat["token_input_cached"]),
                    "output": int(stat["token_output"]),
                },
                "provider_started_at": started_at,
                "provider_ended_at": float(stat["end_time"]),
                "exported_history_messages": len(normalized),
                "source_snapshots": {
                    "astrbot_db_sha256": _sha256(astrbot_path),
                    "history_db_sha256": _sha256(history_path),
                },
            },
        )
        (output_dir / "call.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with (output_dir / "messages.jsonl").open("w", encoding="utf-8") as sink:
            for record in normalized:
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {
            "run_id": run_id,
            "cutoff_at": cutoff_at,
            "messages": len(normalized),
            "query_sha256": manifest["query_sha256"],
            "observed_response_sha256": manifest["observed_response_sha256"],
            "output_dir": str(output_dir),
        }
    finally:
        history.close()
        astrbot.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export one real AstrBot call as a strict masked replay fixture."
    )
    parser.add_argument("--astrbot-db", required=True)
    parser.add_argument("--history-db", required=True)
    parser.add_argument("--call-id", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--bot-id", default="3997938224")
    parser.add_argument("--command-prefix", default="/chat")
    parser.add_argument("--max-history-messages", type=int, default=480)
    parser.add_argument("--prompt-window-seconds", type=int, default=3)
    parser.add_argument("--response-grace-seconds", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(export_call(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
