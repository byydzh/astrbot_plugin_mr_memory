"""Apply independent model drafts and retain the exact drafts needing repair."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field


RETRY_SCHEMA = {"type": "array", "items": {"type": "object", "properties": {
    "pending_id": {"type": "string"},
    "changes": {"type": "object", "description": "只填写要修正的记忆字段，如source_ids；其他草稿内容保留。"},
    "discard_reason": {"type": "string", "description": "重新考虑后决定不保存这条草稿时填写原因。"}},
    "required": ["pending_id"], "additionalProperties": False}}


@dataclass
class WriteOutcome:
    written: list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    discarded: list = field(default_factory=list)
    progress_applied: dict = field(default_factory=dict)
    progress_error: str = ""

    def receipt(self, pending: dict) -> list | dict:
        if not pending and not self.rejected and not self.progress_error and not self.discarded:
            return self.written
        return {"status": "partial" if pending or self.rejected or self.progress_error else "completed",
                "written": self.written,
                "rejected": [{key: value for key, value in row.items() if key != "item"} for row in self.rejected],
                "pending_items": [{"pending_id": key, **value} for key, value in pending.items()],
                "discarded": self.discarded, "progress_applied": self.progress_applied,
                "detail": self.progress_error or "; ".join(row["detail"] for row in self.rejected),
                "next": "材料进度与草稿分别保存；待修草稿不阻止本批结束，会保留到后续学习。已保存项不用重发；retry使用pending_id补changes，或用discard_reason说明不再保存的原因。需要原文时仍可按消息编号读取。"}


class LearningWriter:
    """A draft owns its failure state; an unrelated success cannot clear it."""

    def __init__(self, store, sources: dict[int, str], *, run_id=None, learning_kind=None,
                 pending_items: dict | None = None, deferred_progress: dict | None = None,
                 receipts: dict | None = None):
        self.store, self.sources = store, sources
        self.run_id, self.learning_kind = run_id, learning_kind
        self.pending_items = copy.deepcopy(pending_items or {})
        self.deferred_progress = copy.deepcopy(deferred_progress or {})
        self.receipts = copy.deepcopy(receipts or {})

    def state(self) -> dict:
        return copy.deepcopy({"pending_items": self.pending_items,
                              "deferred_progress": self.deferred_progress, "receipts": self.receipts})

    def persist(self):
        if self.learning_kind is not None:
            task = self.store.learning_task(self.learning_kind)
            self.store.update_learning_task(self.learning_kind, {
                "continuation": {**task.get("continuation", {}), "write_state": self.state()}})

    def clean_item(self, item) -> dict:
        if not isinstance(item, dict):
            raise ValueError("Memory item must be an object")
        if "source_ids" not in item:
            raise ValueError("source_ids is missing; choose the actual evidence message IDs for this item")
        if not isinstance(item["source_ids"], list):
            raise ValueError("source_ids must be a list of evidence message IDs")
        requested = list(dict.fromkeys(int(key) for key in item["source_ids"]))
        missing = [key for key in requested if key not in self.sources]
        if missing:
            for row in self.store.messages(missing):
                self.sources[int(row["id"])] = str(row["source_key"])
            unavailable = [key for key in missing if key not in self.sources]
            if unavailable:
                raise ValueError(f"Source messages are unavailable in this group: {unavailable}")
        row = dict(item)
        row["source_keys"] = [self.sources[key] for key in requested]
        row.pop("source_ids")
        return row

    def apply(self, args: dict, call_id: str) -> WriteOutcome:
        if not isinstance(args, dict) or set(args) - {"items", "progress", "finish", "retry"}:
            raise ValueError("remember accepts items, progress, finish and retry")
        if not args:
            raise ValueError("remember received empty arguments; no memory or progress was saved. Supply items, progress, retry or finish.")
        items, retries, progress = args.get("items", []), args.get("retry", []), args.get("progress", {})
        if not isinstance(items, list) or not isinstance(retries, list):
            raise ValueError("remember.items and remember.retry must be lists")
        if not isinstance(progress, dict):
            raise ValueError("Learning progress must be an object")
        if "finish" in args and type(args["finish"]) is not bool:
            raise ValueError("remember.finish must be a boolean")
        if progress:
            completed = list(dict.fromkeys([*self.deferred_progress.get("completed_ids", []),
                                           *progress.get("completed_ids", [])]))
            self.deferred_progress.update(copy.deepcopy(progress))
            if "completed_ids" in progress:
                self.deferred_progress["completed_ids"] = completed
        outcome = WriteOutcome()
        work = []
        for index, item in enumerate(items):
            # Older prompts may resend an unchanged failed draft in full.
            pending_id = next((key for key, value in self.pending_items.items() if value["item"] == item),
                              f"{call_id}:{index}")
            if pending_id in self.receipts:
                outcome.written.extend(self.receipts[pending_id])
                continue
            work.append((pending_id, index, copy.deepcopy(item)))
        for index, retry in enumerate(retries):
            try:
                if not isinstance(retry, dict) or set(retry) - {"pending_id", "changes", "discard_reason"}:
                    raise ValueError("retry entries accept pending_id, changes and discard_reason")
                key = str(retry.get("pending_id", ""))
                if key not in self.pending_items:
                    raise ValueError(f"No pending draft {key!r}; use an ID from pending_items")
                if any(scheduled[0] == key for scheduled in work):
                    raise ValueError(f"Pending draft {key!r} is already included in this call")
                if retry.get("discard_reason"):
                    if retry.get("changes"):
                        raise ValueError("Choose changes or discard_reason for one pending draft")
                    self.pending_items.pop(key)
                    outcome.discarded.append({"pending_id": key, "reason": str(retry["discard_reason"])})
                    continue
                changes = retry.get("changes", {})
                if not isinstance(changes, dict):
                    raise ValueError("retry.changes must be an object of memory fields")
                original = self.pending_items[key]
                item = {**(original["item"] if isinstance(original["item"], dict) else {}), **changes}
                work.append((key, original["item_index"], item))
            except (TypeError, ValueError) as exc:
                outcome.rejected.append({"retry_index": index, "detail": str(exc)})
        options = {"mark_processed": False, "run_id": self.run_id}
        if self.learning_kind is not None:
            options["learning_kind"] = self.learning_kind
        for key, index, item in work:
            self.pending_items[key] = {**self.pending_items.get(key, {}), "item_index": index, "item": copy.deepcopy(item),
                                       "origin_run_id": self.pending_items.get(key, {}).get("origin_run_id", self.run_id),
                                       "detail": "This draft has not been saved yet"}
        self.persist()
        for key, index, item in work:
            if key in self.receipts:
                outcome.written.extend(self.receipts[key])
                self.pending_items.pop(key, None)
                continue
            try:
                cleaned = self.clean_item(item)
                write_options = dict(options)
                next_state = self.state()
                next_state["pending_items"].pop(key, None)
                if self.learning_kind is not None:
                    write_options["learning_write"] = {"pending_id": key, "state": next_state}
                # Only this item's evidence needs loading for its transaction.
                saved = self.store.save_memories([cleaned], cleaned["source_keys"], **write_options)
                outcome.written.extend(saved)
                self.pending_items.pop(key, None)
                self.receipts[key] = [{"kind": row["kind"], "id": row["id"]} for row in saved]
            except Exception as exc:
                if self.learning_kind is not None:
                    # The row may have committed before constructing its detailed
                    # read receipt failed. Its transactional address is decisive.
                    durable = self.store.learning_task(self.learning_kind).get("continuation", {}).get("write_state", {})
                    saved = durable.get("receipts", {}).get(key)
                    if saved is not None:
                        self.receipts[key] = saved
                        self.pending_items.pop(key, None)
                        outcome.written.extend(saved)
                        continue
                failure = {**self.pending_items[key], "item_index": index, "item": copy.deepcopy(item),
                           "detail": f"{type(exc).__name__}: {exc}"}
                self.pending_items[key] = failure
                outcome.rejected.append({"pending_id": key, **failure})
                self.persist()
        # Understanding a material and successfully persisting every proposed
        # memory are independent. Keep failed drafts, not the whole batch, pending.
        applicable = dict(self.deferred_progress)
        if applicable and self.learning_kind is not None:
            try:
                self.store.save_learning_progress(self.learning_kind, applicable, self.run_id)
                outcome.progress_applied = applicable
                self.deferred_progress = {}
            except Exception as exc:
                outcome.progress_error = f"Learning progress has not been saved: {type(exc).__name__}: {exc}"
        self.persist()
        return outcome
