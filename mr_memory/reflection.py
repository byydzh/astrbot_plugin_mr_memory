"""Read recorded interactions and existing reflections, including their original sources."""
from __future__ import annotations

import json
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
    def associated(self, kind: str, id: int, limit: int = 8, *, include_sources: bool = True) -> list[dict]:
        rows = self.store._rows("""SELECT r.* FROM mr_reflection_schedule r WHERE umo=? AND status!='resolved'
            AND EXISTS(SELECT 1 FROM json_each(r.memory_refs_json) ref
                WHERE json_extract(ref.value,'$.kind')=? AND CAST(json_extract(ref.value,'$.id') AS INTEGER)=?)
            ORDER BY priority DESC,COALESCE(last_reviewed_at,created_at),id LIMIT ?""",
            (self.store.umo, str(kind), int(id), _limit(limit)))
        return [self._record(row, include_sources=include_sources) for row in rows]

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

    def following_context(self, events, request):
        """Nearby interaction is material for interpretation, not a feedback label."""
        anchors = [event for event in events if event["role"] == "BOT"]
        anchor = anchors[-1] if anchors else request
        if not anchor:
            return []
        return [message for message in self.store.context(message_id=anchor["id"], before=0, after=12)
                if message["id"] != anchor["id"]]

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
        result["reflection_context"] = "past_reflections是当时形成的理解；resolved表示当时认为已解决，后来仍可重新理解和修订。用memory(kind='reflection', id=...)展开来源。"
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
        result["subsequent_context"] = self.following_context(events, request)
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
