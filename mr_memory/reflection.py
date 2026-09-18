"""Persistent, model-directed reflection and the conversation it refers to.

This module records what the model wants to revisit. It neither diagnoses a
person nor decides that a retrieved memory caused a later answer.
"""
from __future__ import annotations

import json
import time
from functools import wraps


SCHEMA = """
CREATE VIEW IF NOT EXISTS mr_reflection_schedule AS
 SELECT id,umo,content,revision revision_no,created_at,updated_at,
 COALESCE(json_extract(attributes_json,'$.priority'),1) priority,
 COALESCE(json_extract(attributes_json,'$.review_status'),'pending') status,
 COALESCE(json_extract(attributes_json,'$.source_ids'),'[]') source_ids_json,
 COALESCE(json_extract(attributes_json,'$.memory_refs'),'[]') memory_refs_json,
 json_extract(attributes_json,'$.request_id') request_id,
 json_extract(attributes_json,'$.next_review_at') next_review_at,
 json_extract(attributes_json,'$.last_reviewed_at') last_reviewed_at,
 COALESCE(json_extract(attributes_json,'$.schedule_updated_at'),updated_at) schedule_updated_at,
 COALESCE(json_extract(attributes_json,'$.next_review_explicit'),0) next_review_explicit
 FROM mr_memory_objects WHERE kind='reflection' AND status='ACTIVE';
CREATE INDEX IF NOT EXISTS idx_mr_runs_reflection_request
 ON mr_runs(umo,CAST(json_extract(payload_json,'$.request_id') AS TEXT),id);
CREATE INDEX IF NOT EXISTS idx_messages_reflection_request ON messages(umo,message_id);
CREATE INDEX IF NOT EXISTS idx_relations_reflection_request
 ON message_relations(umo,target_platform_message_id,source_message_id);
CREATE INDEX IF NOT EXISTS idx_relations_reflection_source
 ON message_relations(umo,target_source_key,source_message_id);
"""


def _locked(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self.store._lock:
            return method(self, *args, **kwargs)
    return call


def _limit(value):
    return max(1, min(100, int(value)))


class Reflection:
    def __init__(self, store):
        self.store = store

    def _record(self, row, *, include_sources: bool = True):
        item = dict(row)
        item["kind"] = "reflection"
        item.pop("umo", None)
        item.pop("schedule_updated_at", None)
        item["next_review_explicit"] = bool(item["next_review_explicit"])
        item["source_ids"] = json.loads(item.pop("source_ids_json"))
        item["memory_refs"] = json.loads(item.pop("memory_refs_json"))
        ids = item["source_ids"]
        columns = "*" if include_sources else "id,source_key"
        rows = self.store._rows(
            f"SELECT {columns} FROM messages WHERE umo=? AND is_deleted=0 AND id IN ({','.join('?' for _ in ids)}) ORDER BY sent_at,id",
            [self.store.umo, *ids]) if ids else []
        item["source_keys"] = [row["source_key"] for row in rows]
        if include_sources:
            item["sources"] = [self.store._message(row) for row in rows]
        found = {row["id"] for row in rows}
        item["missing_source_ids"] = [id for id in ids if id not in found]
        return item

    @_locked
    def get(self, id: int, *, include_sources: bool = True) -> dict | None:
        row = self.store.db.execute("SELECT * FROM mr_reflection_schedule WHERE umo=? AND id=?",
                                    (self.store.umo, int(id))).fetchone()
        return self._record(row, include_sources=include_sources) if row else None

    @_locked
    def save(self, item: dict) -> dict:
        """Create a concern or update its supplied fields; omitted fields survive.

        Explicit source_ids and memory_refs replace their previous lists. The
        model can therefore correct a mistaken link as well as retain evidence.
        """
        now = time.time()
        id = item.get("id")
        old = self.get(int(id)) if id is not None else None
        # A complete new concern can be saved even when the caller has guessed
        # a not-yet-allocated address. Always return the actual allocated id.
        value = {"priority": 1, "status": "pending", "source_ids": [], "memory_refs": [],
                 "request_id": None, "next_review_at": None, **(old or {}), **item}
        if old and old["status"] != "waiting" and value["status"] == "waiting" and "next_review_at" not in item:
            # A previous pending retry is not an appointment to revisit a task
            # that the model has now left waiting for new information.
            value["next_review_at"] = None
        content = str(value.get("content") or "").strip()
        if not content:
            raise ValueError("Reflection content is required")
        if value["status"] not in {"pending", "waiting", "resolved"}:
            raise ValueError("Reflection status must be pending, waiting or resolved")
        explicit = (bool(old and old["next_review_explicit"]) if "next_review_at" not in item
                    else value["next_review_at"] is not None)
        if value["next_review_at"] is None or value["status"] == "resolved":
            explicit = False
        sources = list(dict.fromkeys(int(source) for source in value["source_ids"]))
        refs = []
        for ref in value["memory_refs"]:
            normalized = {"kind": str(ref["kind"]), "id": int(ref["id"])}
            if normalized not in refs:
                refs.append(normalized)
        # A model call may finish after /mrforget. Its old input must not write
        # the explicitly forgotten experience back through this newer endpoint.
        anchors = self.store._rows(
            f"SELECT DISTINCT platform_id,sender_id FROM messages WHERE umo=? AND id IN({','.join('?' for _ in sources)})",
            [self.store.umo, *sources]) if sources else []
        if value["request_id"] is not None:
            anchors.extend(self.store._rows("SELECT DISTINCT platform_id,sender_id FROM messages WHERE umo=? AND message_id=?",
                                            (self.store.umo, str(value["request_id"]))))
        if any(self.store._forgotten(row["platform_id"], row["sender_id"]) for row in anchors):
            raise ValueError("Reflection refers to an explicitly forgotten source or request")
        with self.store.db:
            schedule_changed = (not old or value["status"] != old["status"] or "next_review_at" in item
                                or explicit != old["next_review_explicit"])
            saved = self.store.memory_graph.write({"kind": "reflection", **({"id": int(id)} if old else {}),
                "content": content, "title": str(item.get("title") or content.splitlines()[0][:100]),
                "priority": float(value["priority"]), "review_status": value["status"],
                "source_ids": sources, "memory_refs": refs,
                "request_id": str(value["request_id"]) if value["request_id"] is not None else None,
                "next_review_at": float(value["next_review_at"]) if value["next_review_at"] is not None else None,
                "next_review_explicit": int(explicit),
                **({"schedule_updated_at": now} if schedule_changed else {})})
            id = saved["id"]
        return self.get(id)

    @_locked
    def due(self, limit: int = 8, now: float | None = None, *, scheduled_only: bool = False,
            include_sources: bool = True) -> list[dict]:
        now = time.time() if now is None else float(now)
        rows = self.store._rows("""SELECT * FROM mr_reflection_schedule WHERE umo=? AND
            ((status='pending' AND (next_review_at IS NULL OR next_review_at<=?)) OR
             (status='waiting' AND next_review_at IS NOT NULL AND next_review_at<=?))
            AND (?=0 OR (next_review_explicit=1 AND next_review_at IS NOT NULL))
            ORDER BY priority DESC,COALESCE(last_reviewed_at,created_at),id LIMIT ?""",
            (self.store.umo, now, now, int(scheduled_only), _limit(limit)))
        return [self._record(row, include_sources=include_sources) for row in rows]

    @_locked
    def waiting(self, limit: int = 8, *, include_sources: bool = True) -> list[dict]:
        rows = self.store._rows("""SELECT * FROM mr_reflection_schedule WHERE umo=? AND status='waiting'
            ORDER BY priority DESC,COALESCE(last_reviewed_at,created_at),id LIMIT ?""",
            (self.store.umo, _limit(limit)))
        return [self._record(row, include_sources=include_sources) for row in rows]

    @_locked
    def associated(self, kind: str, id: int, limit: int = 8, *, include_sources: bool = True) -> list[dict]:
        rows = self.store._rows("""SELECT r.* FROM mr_reflection_schedule r WHERE umo=? AND status!='resolved'
            AND EXISTS(SELECT 1 FROM json_each(r.memory_refs_json) ref
                WHERE json_extract(ref.value,'$.kind')=? AND CAST(json_extract(ref.value,'$.id') AS INTEGER)=?)
            ORDER BY priority DESC,COALESCE(last_reviewed_at,created_at),id LIMIT ?""",
            (self.store.umo, str(kind), int(id), _limit(limit)))
        return [self._record(row, include_sources=include_sources) for row in rows]

    @_locked
    def reviewed(self, ids, default_next_at: float, call_started_at: float) -> None:
        """Record the visit, preserving any schedule/status the model just chose."""
        now = time.time()
        with self.store.db:
            for id in dict.fromkeys(int(id) for id in ids):
                row = self.store.db.execute("SELECT * FROM mr_reflection_schedule WHERE umo=? AND id=?",
                                            (self.store.umo, id)).fetchone()
                if row is None:
                    continue
                next_at, explicit = row["next_review_at"], row["next_review_explicit"]
                if row["schedule_updated_at"] <= call_started_at:
                    if row["status"] == "pending":
                        next_at, explicit = float(default_next_at), 0
                    elif row["status"] == "waiting" and next_at is not None and next_at <= now:
                        next_at, explicit = None, 0
                # Scheduling changes are bookkeeping, not a new semantic revision.
                self.store.db.execute("""UPDATE mr_memory_objects SET attributes_json=json_set(attributes_json,
                    '$.last_reviewed_at',?,'$.next_review_at',?,'$.next_review_explicit',?)
                    WHERE umo=? AND kind='reflection' AND id=?""", (now, next_at, explicit, self.store.umo, id))

    def _events(self, request_id: str) -> tuple[list[dict], bool]:
        rows = self.store._rows("""SELECT DISTINCT m.* FROM message_relations r
            JOIN messages m ON m.id=r.source_message_id AND m.umo=r.umo
            WHERE r.umo=? AND r.target_platform_message_id=? AND m.is_deleted=0
              AND (m.role='BOT' OR EXISTS(SELECT 1 FROM json_each(m.content_json) c
                  WHERE json_extract(c.value,'$.type')='bot_event'))
            ORDER BY m.sent_at,m.id LIMIT 101""", (self.store.umo, request_id))
        return [self.store._message(row) for row in rows[:100]], len(rows) > 100

    @staticmethod
    def _event_summary(message):
        content = message.get("content", [])
        kinds = [part.get("event") for part in content if isinstance(part, dict) and part.get("type") == "bot_event"]
        result = {key: message[key] for key in ("id", "source_key", "sent_at", "role", "sender_id", "sender_name")}
        result["events"] = kinds
        if message["role"] == "BOT" or "generated" in kinds:
            result["plain_text"] = message["plain_text"]
        result["content_types"] = [part.get("type") for part in content if isinstance(part, dict)]
        result["tool_names"] = [part["name"] for part in content if isinstance(part, dict) and part.get("name")]
        return result

    @_locked
    def interaction(self, request_id: str | None = None, run_id: int | None = None,
                    detailed: bool = False, step: int | None = None) -> dict:
        """Read a specific request's recorded stages, without inferring causality.

        Detailed mode opens at most four linked MR runs and 100 response events.
        Later adjacent messages are context, not automatically user feedback.
        """
        if step is not None:
            if run_id is None or request_id is not None:
                raise ValueError("A recorded step is addressed by run_id and step")
            row = self.store.db.execute("""SELECT s.* FROM mr_run_steps s JOIN mr_runs r ON r.id=s.run_id
                WHERE r.umo=? AND r.id=? AND s.seq=?""", (self.store.umo, int(run_id), int(step))).fetchone()
            if row is None:
                return {"status": "step_not_found", "run_id": run_id, "step": step}
            value = dict(row)
            value["data"] = json.loads(value.pop("data_json"))
            return value
        selected = self.store.run_detail(int(run_id)) if run_id is not None else None
        if run_id is not None and selected is None:
            return {"status": "run_not_found", "run_id": run_id}
        linked = selected.get("request_id") if selected else None
        if request_id is not None and linked is not None and str(request_id) != str(linked):
            raise ValueError("Run belongs to a different request")
        request_id = str(request_id if request_id is not None else linked or "")
        if not request_id:
            if selected and selected["kind"] in {"background", "feedback"}:
                return self._learning_interaction(selected, detailed)
            return {"status": "request_link_missing", "run_id": run_id}
        request_row = self.store.db.execute("""SELECT * FROM messages WHERE umo=? AND message_id=?
            ORDER BY id DESC LIMIT 1""", (self.store.umo, request_id)).fetchone()
        if request_row and self.store._forgotten(request_row["platform_id"], request_row["sender_id"]):
            return {"status": "unavailable", "request_id": request_id, "detail": "原请求已按发言者要求遗忘"}
        rows = self.store._rows("""SELECT id FROM mr_runs WHERE umo=?
            AND CAST(json_extract(payload_json,'$.request_id') AS TEXT)=? ORDER BY id DESC LIMIT 5""",
            (self.store.umo, request_id))
        run_ids = [row["id"] for row in rows[:4]]
        if selected and selected["id"] not in run_ids:
            run_ids = [selected["id"], *run_ids[:3]]
        runs = [self.store.run_detail(id) for id in run_ids]
        request = self.store._message(dict(request_row)) if request_row and not request_row["is_deleted"] else None
        events, events_more = self._events(request_id)
        reflection_rows = self.store._rows("""SELECT id,status,content,priority,created_at,updated_at,
            source_ids_json,memory_refs_json FROM mr_reflection_schedule WHERE umo=? AND request_id=?
            ORDER BY updated_at DESC,id DESC LIMIT 13""", (self.store.umo, request_id))
        previous_understandings = []
        for row in reflection_rows[:12]:
            entry = dict(row)
            entry["source_ids"] = json.loads(entry.pop("source_ids_json"))
            entry["memory_refs"] = json.loads(entry.pop("memory_refs_json"))
            previous_understandings.append(entry)
        result = {"status": "recorded" if request or runs or events else "not_found", "request_id": request_id,
                  "request": request, "runs": [], "response_events": [],
                  "past_reflections": previous_understandings,
                  "truncated": {"runs": len(rows) > 4, "response_events": events_more,
                                "past_reflections": len(reflection_rows) > 12},
                  "interpretation": "记录呈现本轮实际输入、读取、注入、输出与后续交流；读取过某条记忆并不证明它造成了回答。"}
        result["reflection_context"] = "past_reflections是当时形成的理解；resolved表示当时认为已解决，后来仍可重新理解和修订。用reflect(id)展开来源。"
        for run in runs:
            delivered = [step for step in run.get("steps", []) if step["phase"] == "inject"]
            if detailed:
                view = dict(run)
                if run.get("steps"):
                    # Steps retain inputs, calls, results, generated output and
                    # actual injection. Avoid replaying the same trajectory as
                    # both serialized messages and a second tool-call list.
                    view.pop("messages", None)
                    view.pop("tool_calls", None)
            else:
                view = {key: run[key] for key in ("id", "kind", "started_at", "status", "question", "elapsed_ms") if key in run}
                view["tool_names"] = [call.get("name") for call in run.get("tool_calls", []) if isinstance(call, dict)]
            # A generated background is distinct from one actually handed to AstrBot.
            view["injection_events"] = delivered
            view["background"] = run.get("background")
            result["runs"].append(view)
        result["response_events"] = events if detailed else [self._event_summary(event) for event in events]
        anchors = [event for event in events if event["role"] == "BOT"]
        anchor = anchors[-1] if anchors else request
        if anchor:
            following = self.store.context(message_id=anchor["id"], before=0, after=12)
            result["subsequent_context"] = [message for message in following if message["id"] != anchor["id"]]
        else:
            result["subsequent_context"] = []
        return result

    @staticmethod
    def _learning_interaction(run: dict, detailed: bool) -> dict:
        """Open a previous learning session only when the model asks for its run id."""
        steps = run.get("steps", [])
        inputs = next((step["data"] for step in steps if step["phase"] == "input"
                       and isinstance(step.get("data", {}).get("messages"), list)), {})
        material = inputs.get("messages", run.get("messages", []))
        view = {key: run[key] for key in ("id", "kind", "started_at", "status", "detail", "elapsed_ms", "response_text")
                if key in run}
        view["material_ids"] = run.get("material_ids", [row["id"] for row in material
            if isinstance(row, dict) and "id" in row and not row.get("context_only")])
        view["context_ids"] = run.get("context_ids", [row["id"] for row in material
            if isinstance(row, dict) and "id" in row and row.get("context_only")])
        view["memory_refs"] = run.get("memory_refs", [{"kind": row["kind"], "id": row["id"]}
            for row in run.get("written", []) if isinstance(row, dict) and "id" in row and "kind" in row])
        progress = next((step["data"] for step in reversed(steps) if "checkpoint" in step.get("data", {})), {})
        view["checkpoint"] = run.get("checkpoint", run.get("progress", {}).get("checkpoint", progress.get("checkpoint", "")))
        view["completed_ids"] = run.get("completed_ids", progress.get("completed_ids", []))
        if detailed:
            if steps:
                # The retained steps already include material, calls and results.
                view["steps"] = steps
            else:
                view.update({key: run[key] for key in ("messages", "tool_calls") if key in run})
        return {"status": "recorded", "run_id": run["id"], "runs": [view],
                "interpretation": "这是之前一次学习的实际材料、写入和续接线索；按需展开详细记录，不需要从头重做。"}

    @_locked
    def feedback_context(self, messages: list[dict], limit: int = 8) -> list[dict]:
        """Link only request IDs retained on bot activity in this feedback batch."""
        requests = []
        for message in messages:
            content = message.get("content", [])
            bot_event = any(isinstance(part, dict) and part.get("type") == "bot_event" for part in content)
            if message.get("role") != "BOT" and not bot_event:
                continue
            linked = [str(part["message_id"]) for part in content if isinstance(part, dict)
                      and part.get("type") == "response_to" and part.get("message_id") is not None]
            if not linked and message.get("reply_to"):
                row = self.store.db.execute("SELECT message_id FROM messages WHERE umo=? AND source_key=?",
                                            (self.store.umo, message["reply_to"])).fetchone()
                if row:
                    linked = [str(row[0])]
            for request in linked:
                if request not in requests:
                    requests.append(request)
        return [self.interaction(request_id=request) for request in requests[-_limit(limit):]]
