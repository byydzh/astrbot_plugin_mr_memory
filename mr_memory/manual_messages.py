"""Durable, group-local operator material and receipts for the console."""

import hashlib
import json
import re
import time

from .network.base import packed

SCHEMA = """
CREATE TABLE IF NOT EXISTS mr_manual_requests (
 request_id TEXT PRIMARY KEY, digest TEXT NOT NULL, trigger TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS mr_manual_items (
 message_id INTEGER PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
 request_id TEXT NOT NULL REFERENCES mr_manual_requests(request_id), run_id INTEGER);
CREATE INDEX IF NOT EXISTS mr_manual_request_items ON mr_manual_items(request_id,message_id);
CREATE TABLE IF NOT EXISTS mr_manual_run_history(message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,run_id INTEGER NOT NULL,PRIMARY KEY(message_id,run_id));
CREATE TABLE IF NOT EXISTS mr_network_run_changes (
 run_id INTEGER PRIMARY KEY REFERENCES mr_runs(id) ON DELETE CASCADE, diff TEXT NOT NULL);
"""


def ensure_schema(store):
    with store._lock:
        if "mr_manual_run_history" not in store.tables:
            store.db.executescript(SCHEMA)
            store.tables.update(
                ("mr_manual_requests", "mr_manual_items", "mr_manual_run_history", "mr_network_run_changes")
            )


def submit(store, body):
    if not isinstance(body, dict) or set(body) - {"request_id", "messages", "trigger"}:
        raise ValueError("请提供提交编号和待整理消息")
    key = body.get("request_id")
    if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", key):
        raise ValueError("提交编号无效")
    trigger = body.get("trigger", "next_window")
    rows = body.get("messages")
    if trigger not in ("next_window", "now") or not isinstance(rows, list) or not rows:
        raise ValueError("请选择整理方式并至少提供一条消息")
    normalized = []
    for row in rows:
        if not isinstance(row, dict) or set(row) - {"text", "mentioned_account_ids", "reply_to_message_id"}:
            raise ValueError("消息包含不能指定的字段")
        text = row.get("text")
        mentions = row.get("mentioned_account_ids", [])
        reply = row.get("reply_to_message_id")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("消息正文不能为空")
        if not isinstance(mentions, list) or not all(isinstance(u, str) and u for u in mentions):
            raise ValueError("涉及成员须使用本群账号")
        if reply is not None and (type(reply) is not int or reply < 1):
            raise ValueError("引用原话编号无效")
        normalized.append((text.strip(), list(dict.fromkeys(mentions)), reply))
    digest = hashlib.sha256(packed([trigger, normalized]).encode()).hexdigest()
    ensure_schema(store)
    db = store.db
    with store._lock, db:
        prior = db.execute("SELECT digest FROM mr_manual_requests WHERE request_id=?", (key,)).fetchone()
        if prior:
            if prior[0] != digest:
                raise ValueError("同一提交编号已用于不同的消息或触发方式")
            return key, False
        db.execute("INSERT INTO mr_manual_requests VALUES(?,?,?)", (key, digest, trigger))
        platform, group = db.execute(
            "SELECT platform_id,group_id FROM scope_meta WHERE umo=?", (store.umo,)
        ).fetchone()
        at = int(time.time())
        for pos, (text, mentions, reply) in enumerate(normalized):
            content = [{"type": "text", "text": text}]
            people = []
            for account in mentions:
                person = db.execute(
                    "SELECT id,current_display_name FROM participants WHERE umo=? AND account_id=?",
                    (store.umo, account),
                ).fetchone()
                if not person or store._forgotten(platform, account):
                    raise ValueError("涉及的账户不在本群可用成员中")
                content.append({"type": "mention", "account_id": account, "display_name": person[1]})
                people.append(person[0])
            target = None
            if reply is not None:
                target = db.execute(
                    "SELECT source_key,message_id,sender_participant_id FROM messages WHERE id=? AND umo=? AND is_deleted=0",
                    (reply, store.umo),
                ).fetchone()
                if not target:
                    raise ValueError("本群没有这条可引用原话")
            msg = f"manual:{key}:{pos}"
            checksum = hashlib.sha256(packed([text, content, "SYSTEM"]).encode()).hexdigest()
            mid = db.execute(
                """INSERT INTO messages(source_key,platform,platform_id,umo,group_id,message_id,
                    sender_id,sender_name,sent_at,plain_text,content_json,role,content_sha256)
                VALUES(?,'console',?,?,?,?, 'console:operator','管理员补充',?,?,?,'SYSTEM',?)""",
                (
                    f"console|{store.umo}|{msg}",
                    platform,
                    store.umo,
                    group,
                    msg,
                    at,
                    text,
                    packed(content),
                    checksum,
                ),
            ).lastrowid
            db.executemany(
                """INSERT INTO message_participants VALUES(?,?,'MENTIONED',?,'console_manual')""",
                [(mid, pid, i) for i, pid in enumerate(people)],
            )
            if target:
                db.execute(
                    """INSERT INTO message_relations(umo,source_message_id,relation,target_message_id,
                    target_source_key,target_platform_message_id,target_participant_id) VALUES(?,?,'REPLY_TO',?,?,?,?)""",
                    (store.umo, mid, reply, *target),
                )
            store._revision(
                dict(db.execute("SELECT * FROM messages WHERE id=?", (mid,)).fetchone()), "CAPTURED"
            )
            db.execute(
                "INSERT INTO message_processing(message_id,content_sha256,ingestion_source) VALUES(?,?,'console_manual')",
                (mid, checksum),
            )
            db.execute("INSERT INTO mr_manual_items(message_id,request_id) VALUES(?,?)", (mid, key))
            if store.network_enabled:
                from .network.live import captured

                captured(store, dict(db.execute("SELECT * FROM messages WHERE id=?", (mid,)).fetchone()))
    return key, True


def changes(store, run_id):
    with store._lock:
        if "mr_network_run_changes" not in store.tables:
            return {"changes": [], "edges": []}
        row = store.db.execute("SELECT diff FROM mr_network_run_changes WHERE run_id=?", (run_id,)).fetchone()
        return json.loads(row[0]) if row else {"changes": [], "edges": []}


def pending_force(store):
    with store._lock:
        if "mr_manual_requests" not in store.tables:
            return False
        return bool(
            store.db.execute("""SELECT 1 FROM mr_manual_requests r JOIN mr_manual_items i USING(request_id)
             JOIN message_processing p ON p.message_id=i.message_id WHERE r.trigger='now' AND p.status<>'DISTILLED'
             LIMIT 1""").fetchone()
        )


def receipt(store, key):
    with store._lock:
        if "mr_manual_requests" not in store.tables:
            raise ValueError("本群没有这张提交回执")
        req = store.db.execute("SELECT trigger FROM mr_manual_requests WHERE request_id=?", (key,)).fetchone()
        if not req:
            raise ValueError("本群没有这张提交回执")
        rows = store.db.execute(
            """SELECT i.message_id,p.status,i.run_id FROM mr_manual_items i
            JOIN message_processing p ON p.message_id=i.message_id WHERE request_id=? ORDER BY i.message_id""",
            (key,),
        )
        states = {
            "PENDING": "pending",
            "PROCESSING": "processing",
            "FAILED": "failed",
            "DISTILLED": "processed",
        }
        items = [
            {"message_id": mid, "state": states.get(s, "pending"), "run_id": rid} for mid, s, rid in rows
        ]
        kinds = {v["state"] for v in items}
        state = (
            "completed"
            if kinds == {"processed"}
            else (
                "failed"
                if kinds == {"failed"}
                else (
                    "running"
                    if "processing" in kinds
                    else "partial" if kinds & {"processed", "failed"} else "queued"
                )
            )
        )
        past = {
            r[0]
            for r in store.db.execute(
                """SELECT h.run_id FROM mr_manual_run_history h JOIN mr_manual_items i
                  ON i.message_id=h.message_id WHERE i.request_id=?""",
                (key,),
            )
        }
        runs = sorted(past | {i["run_id"] for i in items if i["run_id"] is not None})
        error = store.db.execute(
            """SELECT p.last_error FROM message_processing p JOIN mr_manual_items i USING(message_id)
               WHERE i.request_id=? AND p.status='FAILED' AND p.last_error<>'' ORDER BY i.message_id DESC LIMIT 1""",
            (key,),
        ).fetchone()
        return {
            "request_id": key,
            "trigger": req[0],
            "messages": items,
            "state": state,
            "reason": error[0] if error else "",
            "run_id": runs[-1] if runs else None,
            "runs": [{"run_id": rid, **changes(store, rid)} for rid in runs],
        }
