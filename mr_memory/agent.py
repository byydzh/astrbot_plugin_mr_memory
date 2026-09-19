"""Private DeepSeek memory work, using the configured AstrBot provider."""
from __future__ import annotations

import asyncio
import copy
import itertools
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .content import text_view
from .learning_writes import LearningWriter
from .cognition import visible_memories
from .handoff import format_background as _background_text
from .memory_protocol import (RECONSTRUCTION_PROMPT, CONSOLIDATION_TASK, REFLECTION_TASK,
    CONSOLIDATION_PROMPT, TOOL_SCHEMAS, _tool_set, tool_definitions)
from .tool_calls import response_calls, memory_text_arguments



@dataclass
class ReconstructionResult:
    background: str = ""
    recollection: Any = field(default_factory=dict)
    status: str = "error"
    elapsed_ms: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list, repr=False)
    detail: str = ""
    written: list[dict[str, Any]] = field(default_factory=list)
    pending_items: dict = field(default_factory=dict)


@dataclass
class ConsolidationResult:
    items: list[dict[str, Any]] = field(default_factory=list)
    written: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list, repr=False)
    status: str = "retry"
    elapsed_ms: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)
    detail: str = ""
    response_text: str = ""
    progress: dict[str, Any] = field(default_factory=dict)
    continuation: dict[str, Any] = field(default_factory=dict, repr=False)
    model_attempts: int = 0
    unknown_usage_calls: int = 0
    recollection: Any = field(default_factory=dict)


def _model_view(value: Any) -> Any:
    """Remove duplicate transport fields while keeping message content and references."""
    if isinstance(value, list):
        return [_model_view(item) for item in value]
    if not isinstance(value, dict):
        return text_view(value)
    result = {key: item if key in {"representation", "recollection", "attention", "belief"} else _model_view(item) for key, item in value.items()}
    if result.get("context_only") is False:
        result.pop("context_only")
    # Revision snapshots retain the readable memory plus SQL-only information.
    # Drop only fields already present with the same meaning and value.
    if isinstance(result.get("record"), dict) and isinstance(result.get("memory"), dict):
        memory = result["memory"]
        aliases = {"content": "summary", "statement": "summary", "name": "title", "aspect_tag": "title"}
        result["record"] = {key: item for key, item in result["record"].items()
                            if not ((key in memory and memory[key] == item)
                                    or (aliases.get(key) in memory and memory[aliases[key]] == item))}
        if not result["record"]:
            result.pop("record")
    for key in ("participant_id", "sender_participant_id", "subject_participant_id",
                "target_participant_id", "canonical_key", "narrative_bindings"):
        result.pop(key, None)
    # A platform account is useful authorship data; its private SQL row number is not.
    if "account_id" in result and "name" in result and result.get("kind") in (None, "participant"):
        result.pop("id", None)
    for key in ("sent_at", "first_at", "last_at", "start_at", "end_at", "started_at", "ended_at", "request_at", "now_unix",
                "created_at", "updated_at", "last_reviewed_at", "next_review_at"):
        if type(value.get(key)) in (int, float) and value[key] > 0:
            local = datetime.fromtimestamp(value[key], timezone.utc).astimezone()
            result[key + "_local"] = local.isoformat() + " 星期" + "一二三四五六日"[local.weekday()]
    if "source_ids" in result:
        result.pop("source_keys", None)
    if "kind" in result and "summary" in result:
        for key in ("content", "statement", "description"):
            if result.get(key) == result["summary"]:
                result.pop(key, None)
    # The working view includes every connection; its basis subset need not
    # repeat the same edges. Standalone memory reads still carry their basis.
    if result.get("basis") and isinstance(result.get("connections"), list):
        if all(edge in result["connections"] for edge in result["basis"]):
            result.pop("basis")
    if "plain_text" in result and isinstance(result.get("id"), int):
        result.pop("source_key", None)
    body = result.get("plain_text")
    if isinstance(body, str) and body and isinstance(result.get("content"), list):
        # Keep nontext components (mentions, quotes, pictures, tool output) intact.
        result["content"] = [part for part in result["content"]
                             if not (isinstance(part, dict) and str(part.get("type", "")).lower() in {"text", "plain"}
                                     and isinstance(part.get("text"), str) and part["text"] in body)]
        if not result["content"]:
            result.pop("content")
    return result


def _json(value: Any, *, seen_messages: dict | None = None) -> str:
    seen = {} if seen_messages is None else seen_messages

    def once(part):
        if isinstance(part, list):
            return [once(item) for item in part]
        if not isinstance(part, dict):
            return part
        if type(part.get("id")) is int and "plain_text" in part:
            previous = seen.get(part["id"], {})
            if ("revision_no" in part and "revision_no" in previous
                    and part["revision_no"] != previous["revision_no"]):
                previous = {}
            # Quotes carry fewer fields than full messages. Retain the fields
            # already delivered instead of replacing that knowledge with a quote.
            seen[part["id"]] = {**previous, **part}
            repeated = [key for key in ("plain_text", "content", "attachments", "reply_to_message")
                        if key in part and key in previous and part[key] == previous[key]]
            if repeated:
                part = {key: item for key, item in part.items() if key not in repeated}
                part["source_ref"] = {"message_id": part["id"], "fields": repeated}
        if (isinstance(part.get("kind"), str) and type(part.get("id")) is int
                and (type(part.get("revision_no")) is int or "summary" in part)):
            # Reflections and connection views have content/context rather than
            # summary. They are the same addressed object, not new evidence.
            revision = part.get("revision_no")
            key = f"memory:{part['kind']}:{part['id']}:{revision}"
            previous = seen.get(key, {})
            seen[key] = {**previous, **part}
            repeated = [name for name in ("summary", "content", "statement", "description", "context",
                        "source_speakers", "subject", "representation", "belief", "reflections", "sources",
                        "basis", "connections", "cues", "attention")
                        if name in part and name in previous and part[name] == previous[name]]
            if repeated:
                part = {name: item for name, item in part.items() if name not in repeated}
                part["memory_ref"] = {"kind": part["kind"], "id": part["id"], "fields": repeated}
                if revision is not None:
                    part["memory_ref"]["revision_no"] = revision
        return {key: item if key in {"representation", "recollection", "attention", "belief"} else once(item) for key, item in part.items()}

    # Overlapping memories often cite the same dialogue. Keep its complete text
    # once in the conversation; references retain every memory's source links.
    return json.dumps(once(_model_view(value)), ensure_ascii=False, separators=(",", ":"))


def _restore_seen(values):
    return {int(key) if str(key).isdigit() else key: value for key, value in values.items()}


def _message_tables(value):
    """Share repeated field names/values; preserve every per-message value."""
    if isinstance(value, dict):
        return {key: item if key in {"representation", "recollection", "attention", "belief"} else _message_tables(item) for key, item in value.items()}
    if not isinstance(value, list):
        return value
    rows = [_message_tables(item) for item in value]
    if len(rows) < 2 or not all(isinstance(row, dict) and
            {"id", "sender_id", "sender_name", "role", "sent_at"} <= row.keys() for row in rows):
        return rows
    columns = [key for key in rows[0] if all(key in row for row in rows)]
    defaults = {key: rows[0][key] for key in columns if all(row[key] == rows[0][key] for row in rows)}
    columns = [key for key in columns if key not in defaults]
    extra = [dict((key, item) for key, item in row.items() if key not in columns and key not in defaults)
             for row in rows]
    has_extra = any(extra)
    return {"message_defaults": defaults, "message_columns": columns + (["fields"] if has_extra else []),
            "message_rows": [[row[key] for key in columns] + ([fields] if has_extra else [])
                             for row, fields in zip(rows, extra)]}


def _expand_message_tables(value):
    if isinstance(value, list):
        return [_expand_message_tables(item) for item in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {"message_defaults", "message_columns", "message_rows"}:
        rows = []
        for cells in value["message_rows"]:
            row = dict(zip(value["message_columns"], cells, strict=True))
            fields = row.pop("fields", {})
            rows.append(_expand_message_tables({**value["message_defaults"], **row, **fields}))
        return rows
    return {key: _expand_message_tables(item) for key, item in value.items()}


def _learning_json(value, *, seen_messages=None):
    visible = json.loads(_json(value, seen_messages=seen_messages))
    return json.dumps(_message_tables(visible), ensure_ascii=False, separators=(",", ":"))


def _add_usage(target: dict[str, int], response: Any) -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    for key in ("input_other", "input_cached", "output"):
        value = usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            target[key] = target.get(key, 0) + value


def _finish_reason(response: Any) -> str:
    raw = getattr(response, "raw_completion", None)
    choices = raw.get("choices", []) if isinstance(raw, dict) else getattr(raw, "choices", [])
    if not choices:
        return ""
    choice = choices[0]
    return str(choice.get("finish_reason", "") if isinstance(choice, dict) else getattr(choice, "finish_reason", ""))


def _memory_output(text: str) -> dict:
    """Read a complete final JSON envelope, allowing a preceding brief summary."""
    text = text.strip()
    native = memory_text_arguments(text)
    if native is not None:
        if not isinstance(native.get("items", []), list):
            raise ValueError("Memory output must contain an items list")
        return {"items": native.get("items", []), "progress": native.get("progress", {}),
                "retry": native.get("retry", []), "recollection": native.get("recollection", {}),
                **({"finish": native["finish"]} if "finish" in native else {})}
    if text.startswith("```json"):
        text = text[7:].lstrip()
    elif text.startswith("```"):
        text = text[3:].lstrip()
    starts = [0, *(match.end() for match in re.finditer(r"(?m)^[ \t]*(?=[{\[])", text))]
    for start in dict.fromkeys(starts):
        repair = ""
        try:
            payload, end = json.JSONDecoder().raw_decode(text, start)
        except json.JSONDecodeError as exc:
            # A finished response can omit only its outer closing brace. Accept
            # that one syntactic repair if the entire rest parses unchanged;
            # do not fill strings, arrays, fields or values. Length-truncated
            # responses are handled before this parser is called.
            if start or exc.pos != len(text) or not text.startswith("{") or text[-1:] not in {"}", "]"}:
                continue
            try:
                payload = json.loads(text + "}")
            except json.JSONDecodeError:
                continue
            end = len(text)
            repair = "补齐缺失的最外层闭合括号；原字段、值和原始输出保留。"
        tail = text[end:].strip()
        if tail.startswith("```"):
            tail = tail[3:].strip()
        if tail and not re.fullmatch(r"(?:\s*</[｜|]+DSML[｜|]+(?:parameter|invoke|tool_calls|function_calls)>\s*)+", tail):
            continue
        items = payload.get("items") if isinstance(payload, dict) else payload
        if isinstance(items, list):
            return {"items": items, "progress": payload.get("progress", {}) if isinstance(payload, dict) else {},
                    "retry": payload.get("retry", []) if isinstance(payload, dict) else [],
                    "recollection": payload.get("recollection", {}) if isinstance(payload, dict) else {},
                    **({"_format_repair": repair} if repair else {})}
    raise ValueError("Memory output must end with a complete JSON items list")


def _reconstruction_output(text: str, store=None, before=None) -> tuple[str, Any, str, dict]:
    """Read the delivered background, memory changes and experience of this turn."""
    value = text.strip()
    if value.startswith("```json"):
        value = value[7:].strip()
        if value.endswith("```"):
            value = value[:-3].strip()
    if not value.startswith("{"):
        return text, {}, "本次仅返回文本背景", {}
    try:
        result = json.loads(value, strict=False)
    except ValueError:
        return "", {}, "本次结构化输出未完整返回", {}
    writes = {key: result[key] for key in ("items", "retry", "recollection") if key in result}
    try:
        background = _background_text(result.get("background"), store, before)
    except ValueError as exc:
        return "", {}, str(exc), writes
    recollection = result.get("recollection", {})
    return background.strip(), recollection, "", writes


def _memory_items(text: str) -> list:
    return _memory_output(text)["items"]


def _learning_material(messages: list, working: dict, task: dict | None) -> dict:
    value = {"messages": messages, "working": working}
    if task is not None:
        value["learning_task"] = {key: task[key] for key in (
            "material_ids", "offered_ids", "context_ids", "completed_ids", "checkpoint", "memory_refs", "run_id") if key in task}
        errors = task.get("continuation", {}).get("pending_write_errors", {})
        if errors:
            value["unfinished_writes"] = errors
        writes = task.get("continuation", {}).get("write_state", {})
        if writes:
            value["pending_items"] = [{"pending_id": key, **row} for key, row in writes.get("pending_items", {}).items()]
            value["saved_write_receipts"] = writes.get("receipts", {})
            value["deferred_progress"] = writes.get("deferred_progress", {})
    return value


def _learning_continuation(task: dict | None) -> dict:
    original = (task or {}).get("continuation") or {}
    previous = text_view(original)
    if previous.get("conversation") != original.get("conversation"):
        # Old tasks may contain archived media bytes. Once those bytes are no
        # longer model input, the old observed token/byte ratio is inapplicable.
        previous["previous_input_tokens"] = 0
        previous["previous_input_bytes"] = 0
        previous["input_tokens_per_byte"] = 0.0
    return previous


def compact_learning_task(task: dict) -> dict:
    """Re-encode duplicate fields without discarding any model thought or result."""
    previous = _learning_continuation(task)
    seen = {}
    conversation = copy.deepcopy(previous.get("conversation", []))
    for message in conversation:
        if message.get("role") not in {"user", "tool"} or not isinstance(message.get("content"), str):
            continue
        try:
            value = json.loads(message["content"])
        except ValueError:
            continue
        message["content"] = _learning_json(_expand_message_tables(value), seen_messages=seen)
    if conversation == previous.get("conversation"):
        return task
    density = previous.get("input_tokens_per_byte", 0.0)
    if previous.get("previous_input_tokens") and previous.get("previous_input_bytes"):
        density = previous["previous_input_tokens"] / previous["previous_input_bytes"]
    return {**task, "continuation": {**previous, "conversation": conversation,
        "seen_messages": seen, "previous_input_tokens": 0, "previous_input_bytes": 0,
        "input_tokens_per_byte": density}}


def checkpoint_learning_task(task: dict) -> dict:
    """Resume from a saved model checkpoint without discarding work after it."""
    if not task.get("checkpoint"):
        return task
    previous = _learning_continuation(task)
    conversation = previous.get("conversation", [])
    boundary = None
    for index, message in enumerate(conversation):
        if message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls", [])
        if not calls:
            try:
                final = _memory_output(str(message.get("content") or ""))
            except (TypeError, ValueError):
                pass
            else:
                progress = final.get("progress")
                if isinstance(progress, dict) and progress.get("checkpoint") == task["checkpoint"]:
                    # Final JSON uses the same writer as remember. The matching
                    # persisted checkpoint is its write receipt.
                    boundary = index
        for call in calls:
            if call.get("function", {}).get("name") != "remember":
                continue
            try:
                args = json.loads(call["function"]["arguments"])
            except (KeyError, TypeError, ValueError):
                continue
            if args.get("progress", {}).get("checkpoint") != task["checkpoint"]:
                continue
            receipt = next((row for row in conversation[index + 1:]
                            if row.get("tool_call_id") == call["id"]), None)
            try:
                saved = json.loads(receipt["content"]) if receipt else None
            except (TypeError, ValueError):
                continue
            partial_checkpoint = (isinstance(saved, dict) and
                                  saved.get("progress_applied", {}).get("checkpoint") == task["checkpoint"])
            if not isinstance(saved, list) and not partial_checkpoint:
                continue
            # Keep the checkpoint-producing turn and every later read/write:
            # the checkpoint may precede a tool result that still needs thought.
            boundary = index
    if boundary is None:
        return task
    return {**task, "continuation": {
        "sources": previous.get("sources", {}),
        "pending_write_errors": previous.get("pending_write_errors", {}),
        "write_state": previous.get("write_state", {}),
        "output_token_reserve": previous.get("output_token_reserve", 0),
        "checkpoint_tail": conversation[boundary:],
        "input_tokens_per_byte": (previous["previous_input_tokens"] / previous["previous_input_bytes"]
                                  if previous.get("previous_input_tokens") and previous.get("previous_input_bytes")
                                  else previous.get("input_tokens_per_byte", 0.0)),
    }}


def _checkpoint_conversation(messages: list, working: dict, task: dict | None) -> list:
    conversation = [{"role": "user", "content": _learning_json(_learning_material(messages, working, task))}]
    tail = _learning_continuation(task).get("checkpoint_tail", [])
    if tail:
        conversation.extend(copy.deepcopy(tail))
        conversation.append({"role": "user", "content": "以上工具记录已实际执行；从当前checkpoint、实际写入回执与尚未完成的关注继续。材料已处理不等于反思必须立即结束，也不需要重做已经保存的部分。此处从checkpoint续接，部分source_ref引用的正文位于已省去的旧输入；地址仍有效，需要核对时用context(message_id)重新打开原文。"})
    return conversation


def _learning_resume_update(messages: list, working: dict, task: dict, previous: dict) -> dict:
    update = {"resume": "接着上次未完成的工作继续。"}
    seen = previous.get("seen_messages", {})
    fresh = [row for row in messages if _model_view(row) != seen.get(row.get("id"), seen.get(str(row.get("id"))))]
    if fresh:
        update["messages"] = fresh
    changes = {key: value for key, value in working.items() if previous.get("working", {}).get(key) != value}
    if changes:
        update["working_updates"] = changes
    task_state = _learning_material([], {}, task).get("learning_task", {})
    if previous.get("task_state") != task_state:
        update["learning_task"] = task_state
    return update


def _learning_bytes(conversation: list, feedback: bool) -> int:
    prompt = (REFLECTION_TASK if feedback else CONSOLIDATION_TASK) + CONSOLIDATION_PROMPT
    request = {"messages": [{"role": "system", "content": prompt}, *conversation],
               "tools": tool_definitions(learning=True)}
    return len(json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) + 1024


def _estimate_input_tokens(input_bytes: int, previous_tokens: int, previous_bytes: int,
                           tokens_per_byte: float) -> int:
    if previous_tokens and previous_bytes:
        return max(previous_tokens, round(input_bytes * previous_tokens / previous_bytes))
    # A checkpoint has a new prefix: retain observed density, not the old size.
    return round(input_bytes * tokens_per_byte) if tokens_per_byte else input_bytes


def estimate_learning_input(messages: list, working: dict, *, feedback: bool = False,
                            task: dict | None = None, input_tokens_per_byte: float = 0.0,
                            workspace: dict | None = None) -> int:
    """Budget the next input, preserving the observed cost of a resumable prefix."""
    previous = _learning_continuation(task)
    conversation = list(previous.get("conversation", []))
    if conversation:
        seen = _restore_seen(previous.get("seen_messages", {}))
        update = _learning_resume_update(messages, working, task, previous)
        conversation.append({"role": "user", "content": _learning_json(update, seen_messages=seen)})
    else:
        conversation = _checkpoint_conversation(messages, working, task)
        seen = {}
    if workspace is not None and workspace != previous.get("workspace"):
        conversation.append({"role": "user", "content": _json({"workspace": workspace}, seen_messages=seen)})
    # The resource report and possible save hint are appended immediately before
    # a call. Include their small transport overhead in scheduler admission.
    input_bytes = _learning_bytes(conversation, feedback) + 2048
    tokens, size = previous.get("previous_input_tokens", 0), previous.get("previous_input_bytes", 0)
    return _estimate_input_tokens(input_bytes, tokens, size, previous.get("input_tokens_per_byte", 0.0) or input_tokens_per_byte)


class MemoryAgent:
    def __init__(self, provider, store, embedder=None, timeout_seconds=20,
                 max_turns=4, max_output_tokens=1200, thinking_mode="disabled"):
        self.provider, self.store, self.embedder = provider, store, embedder
        self.timeout_seconds = max(0.01, float(timeout_seconds))
        self.max_turns = None if max_turns is None else max(1, int(max_turns))
        self.max_output_tokens = max(64, int(max_output_tokens))
        self.thinking_mode = thinking_mode
        self.trace = None
        self.run_key = uuid4().hex
        self.available_memories = {}
        self.last_workspace = None

    async def _workspace_update(self, messages, seen_messages):
        workspace = await asyncio.to_thread(self.store.workspace)
        if workspace != self.last_workspace:
            messages.append({"role": "user", "content": _json({"workspace": workspace}, seen_messages=seen_messages)})
            self.last_workspace = workspace
            await self._emit("input", "延续中的理解已更新", workspace=workspace)

    async def _record_experience(self, kind, current, result, writer):
        result.recollection = copy.deepcopy(writer.recollection)
        payload = {"status": result.status, "recollection": result.recollection,
                   "request": {key: current[key] for key in ("question", "plain_text", "sender_id", "sender_name", "sent_at") if key in current},
                   "available_memories": list(self.available_memories.values()),
                   "written": [{key: row[key] for key in ("kind", "id")} for row in result.written],
                   "background": getattr(result, "background", ""), "usage": result.usage,
                   "elapsed_ms": result.elapsed_ms}
        try:
            await asyncio.to_thread(self.store.record_cognition, run_key=self.run_key,
                                    run_id=getattr(self.trace, "id", None), kind=kind, current=current, payload=payload)
        except Exception as exc:
            result.detail = "; ".join(filter(None, [result.detail, f"Understanding experience was not recorded: {exc}"]))
            await self._emit("write", "理解过程记录未保存", "error", detail=str(exc))

    async def _emit(self, phase, title, status="completed", **data):
        if self.trace is not None:
            try:
                return await self.trace.emit(phase, title, status=status, **data)
            except Exception:
                pass

    async def _model_turn(self, messages, prompt, tools, turn, *, max_output_tokens=None, tool_choice=None,
                          response_format=None):
        # Count bodies actually sent to the model. A deferred tool result or a
        # model's own proposed memory is not evidence that it read that memory.
        for message in messages:
            if message.get("role") not in {"user", "tool"}:
                continue
            try:
                value = json.loads(message.get("content") or "")
            except (ValueError, TypeError):
                continue
            for ref in visible_memories(value):
                self.available_memories[(ref["kind"], ref["id"], ref["revision_no"])] = ref
        began = time.monotonic()
        await self._emit("model", f"第 {turn} 轮模型调用", "running", turn=turn)
        try:
            response = await self._generate(messages, prompt, tools, max_output_tokens=max_output_tokens,
                                            tool_choice=tool_choice, response_format=response_format)
        except asyncio.CancelledError:
            await self._emit("model", f"第 {turn} 轮模型调用已取消", "cancelled", turn=turn,
                             elapsed_ms=(time.monotonic() - began) * 1000)
            raise
        except Exception as exc:
            await self._emit("model", f"第 {turn} 轮模型调用失败", "error", turn=turn,
                             elapsed_ms=(time.monotonic() - began) * 1000, detail=f"{type(exc).__name__}: {exc}")
            raise
        usage = {}
        _add_usage(usage, response)
        await self._emit("model", f"第 {turn} 轮模型已返回", turn=turn,
                         elapsed_ms=(time.monotonic() - began) * 1000,
                         completion_text=str(getattr(response, "completion_text", "") or ""), usage=usage,
                         finish_reason=_finish_reason(response),
                         tool_names=list(getattr(response, "tools_call_name", None) or []))
        return response

    async def _generate(self, messages, system_prompt, tools=None, *, max_output_tokens=None, tool_choice=None,
                        response_format=None):
        # AstrBot's public text_chat drops generation kwargs on this provider.
        # Keep its client/parser while placing native options in the prepared payload.
        # Keep the configured client/model; isolate per-call thinking options from
        # other users of the same AstrBot provider instance.
        provider = copy.copy(self.provider)
        config = getattr(provider, "provider_config", {})
        provider.provider_config = {**config, "custom_extra_body": {
            **config.get("custom_extra_body", {}), "thinking": {"type": self.thinking_mode}}}
        payload, _ = await provider._prepare_chat_payload(
            prompt=None, contexts=messages, system_prompt=system_prompt)
        payload.update(thinking={"type": self.thinking_mode},
                       max_tokens=self.max_output_tokens if max_output_tokens is None else max_output_tokens)
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if response_format is not None:
            payload["response_format"] = response_format
        return await provider._query(payload, tools, request_max_retries=1)

    @staticmethod
    def _arguments(name, arguments):
        if name not in TOOL_SCHEMAS or not isinstance(arguments, dict):
            raise ValueError("Unknown tool or non-object arguments")
        schema = TOOL_SCHEMAS[name]["parameters"]

        def argument_error(problem):
            accepted = json.dumps(schema["properties"], ensure_ascii=False, separators=(",", ":"))
            return ValueError(f"Invalid arguments for {name}: {problem}. "
                              f"Accepted parameters (names and types): {accepted}. "
                              + TOOL_SCHEMAS[name]["description"])

        unknown = sorted(set(arguments) - set(schema["properties"]))
        missing = sorted(set(schema["required"]) - set(arguments))
        if unknown or missing:
            raise argument_error(f"unknown_fields={unknown}; missing_fields={missing}")
        for key, value in arguments.items():
            definition = schema["properties"][key]
            kinds = definition["type"] if isinstance(definition["type"], list) else [definition["type"]]
            valid = (("integer" in kinds and type(value) is int) or
                     ("string" in kinds and isinstance(value, str)) or
                     ("boolean" in kinds and type(value) is bool) or
                     ("object" in kinds and isinstance(value, dict)) or
                     ("null" in kinds and value is None) or
                     ("array" in kinds and isinstance(value, list)))
            if not valid or ("enum" in definition and value not in definition["enum"]):
                raise argument_error(f"invalid_field={key}; received_type={type(value).__name__}")
        result = dict(arguments)
        for key in ("sender_id", "related_account_id"):
            if key in result:
                result[key] = str(result[key])
        if "limit" in result:
            result["limit"] = max(1, min(80, result["limit"]))
        for key in ("before", "after") if name in {"context", "search_messages"} else ():
            if key in result:
                result[key] = max(0, min(40, result[key]))
        return result

    async def _execute(self, name, arguments, cutoff):
        args = self._arguments(name, arguments)
        if name == "main_context":
            context = getattr(self, "request_context", None)
            if context is None:
                return {"status": "unavailable", "detail": "This recorded run has no main-model request snapshot; its capabilities cannot be inferred from that absence."}
            return context.read(**args)
        if name == "interaction":
            return await asyncio.to_thread(self.store.reflections.interaction, **args)
        if name == "reflect":
            if set(args) == {"id"}:
                return await asyncio.to_thread(self.store.reflections.get, args["id"])
            return await asyncio.to_thread(self.store.reflections.save, args)
        if name == "semantic_search":
            if self.embedder is None:
                return {"status": "unavailable", "detail": "Semantic index is not configured; literal search remains available."}
            hits = await self.embedder.search(self.store, args["query"], limit=args.get("limit", 8))
            rows = await asyncio.gather(*(asyncio.to_thread(self.store.memory, hit["owner_type"], hit["owner_key"],
                                                          include_sources=False) for hit in hits))
            return [{"score": hit["score"], "memory": {key: value for key, value in row.items() if key != "sources"}}
                    for hit, row in zip(hits, rows) if row is not None]
        if name in {"search_messages", "activity"}:
            args["end_at"] = min(args.get("end_at", cutoff), cutoff)
            if args.get("start_at", 0) > args["end_at"]:
                raise ValueError("start_at is later than the allowed end_at")
        if name == "search_messages":
            return await asyncio.to_thread(self.store.search_message_context, **args)
        if name == "context":
            args["before_time"] = cutoff
        method = "members" if name == "member" else name
        value = await asyncio.to_thread(getattr(self.store, method), **args)
        if name == "context" and not value:
            raise ValueError("Context anchor not found in this group at or before the request time. "
                             "message_id is the MR internal message id (source_ids), not a platform message number. "
                             "For a platform message, use its existing complete source_key; do not construct one.")
        if name in {"search_memories", "graph"}:
            return [{key: part for key, part in row.items() if key != "sources"} for row in value]
        return value

    async def reconstruct(self, current: dict, recent: list, working: dict):
        started = time.monotonic()
        result = ReconstructionResult()
        cutoff = int(current.get("sent_at") or time.time())
        messages = []
        seen_messages = {}
        tools = _tool_set()
        sources = {row["id"]: row["source_key"] for row in [*recent, current] if "id" in row and "source_key" in row}
        writer = LearningWriter(self.store, sources, run_id=getattr(self.trace, "id", None),
                                foreground=True)
        working = {key: value for key, value in working.items() if key != "pending_recall_writes"}
        working["continuity"] = await asyncio.to_thread(self.store.continuity, cutoff)
        async def persist_changes(args, key):
            operation = asyncio.create_task(asyncio.to_thread(writer.apply, args, self.run_key + ":" + key))
            try:
                outcome = await asyncio.shield(operation)
            except asyncio.CancelledError:
                await operation
                raise
            result.written.extend(outcome.written)
            return outcome
        cancelled = False
        completed_response = False
        try:
            async with asyncio.timeout(self.timeout_seconds):
                messages.append({"role": "user", "content": _json({"current": current, "recent": recent,
                                "working": working,
                                "now_unix": cutoff, "service_local_time": datetime.fromtimestamp(cutoff, timezone.utc).astimezone().isoformat()}, seen_messages=seen_messages)})
                last_round_seconds = 0.0
                for turn in range(self.max_turns):
                    await self._workspace_update(messages, seen_messages)
                    round_started = time.monotonic()
                    remaining = self.timeout_seconds - (round_started - started)
                    messages.append({"role": "user", "content": _json({"resource_state": {
                        "remaining_seconds": round(max(0, remaining), 1),
                        "further_model_rounds": self.max_turns - turn - 1,
                        "previous_round_seconds": round(last_round_seconds, 1),
                        "usage_so_far": result.usage,
                        "meaning": "当前交流正在等待这份背景。结合继续思考可能增加的理解与代价，自行决定是否继续；资源是上限，不需要用完。仍值得探索的思路可以留待以后。"}})})
                    forced_finish = turn == self.max_turns - 1 or (turn > 0 and remaining <= 6 + last_round_seconds)
                    if forced_finish:
                        messages.append({"role": "user", "content":
                            "这是本次可用的最后一轮，交付已经形成的理解，不再发起工具调用。"
                            "直接返回 JSON 对象，字段与 complete 一致：background，以及可选的 items、retry、recollection。"
                            "背景仍保留当前把握；未解决的思路可以留在 recollection 或记忆里。"})
                    response = await self._model_turn(
                        messages, RECONSTRUCTION_PROMPT, tools, turn + 1,
                        tool_choice="none" if forced_finish else None,
                        response_format={"type": "json_object"} if forced_finish else None)
                    _add_usage(result.usage, response)
                    native_calls = response_calls(response)
                    names = [call.name for call in native_calls]
                    text = str(getattr(response, "completion_text", "") or "").strip()
                    if names and text:
                        # Tool-planning prose is part of this agent's conversation,
                        # not a completed background for the main model.
                        background, recollection, detail, _ = await asyncio.to_thread(_reconstruction_output, text, self.store, cutoff)
                        if not detail:
                            result.background = background
                        result.status = "partial"
                    if not names:
                        if not text:
                            result.detail = "Provider returned neither text nor tool calls"
                            break
                        messages.append({"role": "assistant", "content": text})
                        background, recollection, detail, final_writes = await asyncio.to_thread(_reconstruction_output, text, self.store, cutoff)
                        result.background = background
                        result.status = "partial" if _finish_reason(response) == "length" else "completed"
                        if not background and detail:
                            result.status = "partial"
                            result.detail = detail
                        if result.status == "partial":
                            result.detail = result.detail or "Model turn or output budget reached; background may be incomplete"
                        if final_writes:
                            await self._emit("write", "保存最终输出中的记忆变化", "running", operation_id="final_memories", items=final_writes)
                            try:
                                outcome = await persist_changes(final_writes, "final")
                                receipt = outcome.receipt(writer.pending_items)
                                if outcome.rejected:
                                    result.status, result.detail = "partial", "Background produced; some final memory changes remain unsaved"
                                await self._emit("write", "最终记忆变化保存结果", "partial" if outcome.rejected else "completed",
                                                 operation_id="final_memories", result=receipt)
                            except Exception as exc:
                                result.status, result.detail = "partial", f"Background produced; final memory write failed: {exc}"
                                await self._emit("write", "最终记忆变化未保存", "error", operation_id="final_memories", detail=str(exc))
                        break
                    tool_messages = [call.message() for call in native_calls]
                    assistant = {"role": "assistant", "content": text or None, "tool_calls": tool_messages,
                                 "reasoning_content": getattr(response, "reasoning_content", None) or ""}
                    messages.append(assistant)

                    async def execute_one(call):
                        nonlocal completed_response
                        name, args, call_id = call.name, call.arguments, call.id
                        began = time.monotonic()
                        record = {"name": name, "arguments": args, "id": call_id, "status": "running"}
                        if call.error:
                            record["raw_arguments"] = call.raw_arguments
                        result.tool_calls.append(record)
                        value = None
                        phase = "write" if name in {"remember", "complete"} or (name == "reflect" and set(args or {}) != {"id"}) else "read"
                        await self._emit(phase, f"{'关注' if phase == 'write' else '读取'} {name}", "running", turn=turn + 1,
                                         tool_call_id=call_id, name=name, arguments=args)
                        try:
                            if call.error:
                                raise ValueError(call.error)
                            if name == "complete":
                                result.background = await asyncio.to_thread(_background_text, args.get("background"), self.store, cutoff)
                                changes = {key: value for key, value in args.items() if key != "background"}
                                outcome = await persist_changes(changes, call_id) if changes else None
                                value = outcome.receipt(writer.pending_items) if outcome else {"status": "completed"}
                                result.status = "partial" if outcome and outcome.rejected else "completed"
                                if outcome and outcome.rejected:
                                    result.detail = "Background produced; some memory changes remain unsaved"
                                completed_response = True
                            elif name == "remember":
                                outcome = await persist_changes(args, call_id)
                                value = outcome.receipt(writer.pending_items)
                            else:
                                value = await self._execute(name, args, cutoff)
                            record["status"] = value.get("status", "completed") if isinstance(value, dict) else "completed"
                        except asyncio.CancelledError:
                            record["status"] = "cancelled"
                            raise
                        except Exception as exc:
                            value = {"status": "error", "detail": f"{type(exc).__name__}: {exc}"}
                            record["status"] = "error"
                        finally:
                            record["elapsed_ms"] = (time.monotonic() - began) * 1000
                            await self._emit(phase, f"{'关注' if phase == 'write' else '读取'} {name}", record["status"], turn=turn + 1,
                                             tool_call_id=call_id, name=name, elapsed_ms=record["elapsed_ms"], result=value)
                        record["result"] = value
                        return call_id, value

                    # Reads can overlap; a model-directed reflection may be read by a later call.
                    if "reflect" in names or "remember" in names or "complete" in names:
                        outputs = [await execute_one(call) for call in native_calls]
                    else:
                        outputs = await asyncio.gather(*(execute_one(call) for call in native_calls))
                    # Serialize in conversation order, after parallel reads finish:
                    # a reference must never precede the text it refers to.
                    messages.extend({"role": "tool", "tool_call_id": call_id,
                                     "content": _json(value, seen_messages=seen_messages)}
                                    for call_id, value in outputs)
                    if completed_response:
                        break
                    last_round_seconds = time.monotonic() - round_started
                else:
                    result.status, result.detail = "partial", "Model turn limit reached before a final background"
        except asyncio.CancelledError:
            cancelled = True
            raise
        except TimeoutError:
            result.status, result.detail = "partial", "Memory reconstruction time budget exhausted"
        except Exception as exc:
            result.status, result.detail = "error", f"{type(exc).__name__}: {exc}"
        finally:
            result.elapsed_ms = (time.monotonic() - started) * 1000
            result.pending_items = writer.pending_items
            await self._record_experience("foreground", current, result, writer)
            # Debug evidence is private; hidden model reasoning is not persisted.
            result.messages = [{key: value for key, value in message.items() if key != "reasoning_content"} for message in messages]
            await self._emit("output", "本次记忆背景产出", "cancelled" if cancelled else result.status,
                             background=result.background, recollection=result.recollection,
                             detail=result.detail, usage=result.usage, elapsed_ms=result.elapsed_ms)
        return result

    async def consolidate(self, messages: list, working: dict, *, feedback: bool = False,
                          token_budget: int | None = None, task: dict | None = None):
        started = time.monotonic()
        result = ConsolidationResult()
        previous = _learning_continuation(task)
        self.last_workspace = previous.get("workspace")
        conversation = copy.deepcopy(previous.get("conversation", []))
        seen_messages = _restore_seen(previous.get("seen_messages", {}))
        sources: dict[int, str] = {int(key): value for key, value in previous.get("sources", {}).items()}
        pending_write_errors = dict(previous.get("pending_write_errors", {}))
        previous_input_tokens = int(previous.get("previous_input_tokens", 0))
        previous_input_bytes = int(previous.get("previous_input_bytes", 0))
        input_tokens_per_byte = float(previous.get("input_tokens_per_byte", 0.0))
        usage_this_turn = dict(previous.get("previous_call_usage", {}))
        output_token_reserve = max(int(previous.get("output_token_reserve", 0)), usage_this_turn.get("output", 0))
        previous_call_seconds = float(previous.get("previous_call_seconds", 0))
        learning_kind = "feedback" if feedback else "background"
        writer = LearningWriter(self.store, sources, run_id=getattr(self.trace, "id", None),
                                learning_kind=learning_kind if task is not None else None,
                                recover_recall=not feedback,
                                **previous.get("write_state", {}))
        working = {key: value for key, value in working.items() if key != "pending_recall_writes"}
        cancelled = False

        def read_sources(value):
            if isinstance(value, list):
                for part in value:
                    read_sources(part)
            elif isinstance(value, dict):
                source_id = value.get("id", value.get("source_id"))
                if type(source_id) is int and value.get("source_key") and "plain_text" in value:
                    sources[source_id] = str(value["source_key"])
                for key, part in value.items():
                    if key in {"representation", "recollection", "attention", "belief"}:
                        continue
                    if isinstance(part, (list, dict)):
                        read_sources(part)

        async def clean_items(items):
            if not isinstance(items, list):
                raise ValueError("Memory output must contain an items list")
            return [await asyncio.to_thread(writer.clean_item, item) for item in items]

        async def apply_write(args, call_id):
            # Complete the bounded DB operation before unload persists the final
            # continuation; cancelling to_thread alone leaves its writer running.
            operation = asyncio.create_task(asyncio.to_thread(writer.apply, args, call_id))
            try:
                outcome = await asyncio.shield(operation)
            except asyncio.CancelledError:
                await operation
                raise
            latest = {(row["kind"], row["id"]): row for row in result.written + outcome.written}
            result.written = list(latest.values())
            if outcome.progress_error:
                pending_write_errors["remember"] = outcome.progress_error
            else:
                pending_write_errors.pop("remember", None)
            return outcome.receipt(writer.pending_items)

        try:
            read_sources(messages)
            read_sources(working)
            if not messages and not working and not conversation and not (task or {}).get("checkpoint") and not writer.pending_items:
                raise ValueError("Consolidation requires an experience or a reflection to revisit")
            tools = _tool_set(learning=True)
            cutoff = int(time.time())
            async with asyncio.timeout(self.timeout_seconds):
                purpose = REFLECTION_TASK if feedback else CONSOLIDATION_TASK
                material = _learning_material(messages, working, task)
                if writer.pending_items:
                    material["pending_items"] = [{"pending_id": key, **row} for key, row in writer.pending_items.items()]
                if not conversation:
                    conversation = _checkpoint_conversation(messages, working, task)
                    conversation[0]["content"] = _learning_json(material, seen_messages=seen_messages)
                else:
                    # Previous messages remain byte-for-byte intact for prefix reuse.
                    update = _learning_resume_update(messages, working, task, previous)
                    if writer.pending_items:
                        update["pending_items"] = [{"pending_id": key, **row} for key, row in writer.pending_items.items()]
                    conversation.append({"role": "user", "content": _learning_json(update, seen_messages=seen_messages)})
                saving = False
                for turn in itertools.count():
                    if self.max_turns is not None and turn >= self.max_turns:
                        result.status, result.detail = "partial", "Learning turn limit reached before the model finished"
                        break
                    if token_budget is not None and sum(result.usage.values()) >= token_budget:
                        result.status, result.detail = "partial", "Rolling token budget reached during learning"
                        break
                    await self._workspace_update(conversation, seen_messages)
                    remaining = max(0, token_budget - sum(result.usage.values())) if token_budget is not None else None
                    seconds_left = self.timeout_seconds - (time.monotonic() - started)
                    # Saving may need remember -> returned IDs -> reflect ->
                    # completion. Reserve several calls using observed usage,
                    # including room for the longer context after tool results.
                    begin_saving = not saving and (
                        (turn > 0 and self.max_turns is not None and turn >= self.max_turns - 3)
                        or (usage_this_turn and remaining is not None and remaining <= 4 * sum(usage_this_turn.values()))
                        or (previous_call_seconds > 0 and seconds_left <= 4 * previous_call_seconds))
                    saving = saving or begin_saving
                    resource_state = {
                        "now_unix": int(time.time()), "service_local_time": datetime.now().astimezone().isoformat(),
                        "remaining_total_tokens": remaining,
                        "previous_call_total_tokens": sum(usage_this_turn.values()),
                        "remaining_seconds": round(max(0, seconds_left), 1),
                        "estimated_input_tokens": 0, "remaining_after_current_input_tokens": None,
                        "meaning": "总余额是调用前数值，本次输入（含缓存）也要从中扣除。remaining_after_current_input_tokens才是扣除本次输入估算后可用于本次输出和后续调用的余额；下一次调用还要再次支付整个上下文。若不够再携带一次上下文，可在本次最后JSON的items和progress中直接保存认识与进度，无需再等待remember回执。未完思路写checkpoint，独立待查关注可用reflect。"}
                    resource_message = {"role": "user", "content": _json({"resource_state": resource_state})}
                    conversation.append(resource_message)
                    if begin_saving:
                        conversation.append({"role": "user", "content": "本次剩余资源适合收拢已经形成的理解。用remember保存修正和progress：已处理材料、目前理解、有关记忆与原文地址、还差什么及下一步线索。独立关注可用reflect更新，已解决的设为resolved。所有工具仍可用，由你判断完成保存所需的读取和写入，然后结束本轮，下次从进度继续。"})
                    input_bytes = _learning_bytes(conversation, feedback)
                    estimated_input = _estimate_input_tokens(input_bytes, previous_input_tokens, previous_input_bytes,
                                                             input_tokens_per_byte)
                    if remaining is not None and remaining <= estimated_input:
                        compact = compact_learning_task({"continuation": {"conversation": conversation,
                            "input_tokens_per_byte": input_tokens_per_byte}})["continuation"]
                        if compact["conversation"] != conversation:
                            conversation = compact["conversation"]
                            seen_messages = _restore_seen(compact["seen_messages"])
                            previous_input_tokens = previous_input_bytes = 0
                            input_bytes = _learning_bytes(conversation, feedback)
                            estimated_input = _estimate_input_tokens(input_bytes, 0, 0, input_tokens_per_byte)
                            # The old object is no longer part of the compacted transcript.
                            resource_message = next(row for row in reversed(conversation)
                                if row.get("role") == "user" and row.get("content", "").startswith('{"resource_state":'))
                    output_room = min(self.max_output_tokens, max(2048, output_token_reserve))
                    next_input = estimated_input + max(0, estimated_input - previous_input_tokens) if previous_input_tokens else estimated_input
                    save_this_turn = remaining is not None and remaining < estimated_input + next_input + 2 * output_room
                    resource_state["save_this_turn"] = save_this_turn
                    if save_this_turn:
                        resource_state["action"] = "剩余额度可能只够本次调用，请收拢已形成的理解并保存实际进度。工具保持可用；若继续读取，下一次携带上下文可能需要等额度释放。未完线索写入checkpoint，已经完整处理的材料可完成，无须穷尽所有关联。"
                    resource_state["estimated_input_tokens"] = estimated_input
                    resource_state["remaining_after_current_input_tokens"] = max(0, remaining - estimated_input) if remaining is not None else None
                    resource_message["content"] = _json({"resource_state": resource_state})
                    input_bytes = _learning_bytes(conversation, feedback)
                    estimated_input = _estimate_input_tokens(input_bytes, previous_input_tokens, previous_input_bytes,
                                                             input_tokens_per_byte)
                    if remaining is not None and remaining <= estimated_input:
                        result.status, result.detail = "partial", "Remaining rolling budget cannot cover the next learning input"
                        break
                    output_limit = (min(self.max_output_tokens, remaining - estimated_input)
                                    if remaining is not None else self.max_output_tokens)
                    call_started = time.monotonic()
                    result.model_attempts += 1
                    try:
                        response = await self._model_turn(conversation, purpose + CONSOLIDATION_PROMPT,
                                                         tools,
                                                         turn + 1, max_output_tokens=output_limit)
                    except (Exception, asyncio.CancelledError):
                        result.unknown_usage_calls += 1
                        raise
                    previous_call_seconds = time.monotonic() - call_started
                    usage_this_turn = {}
                    _add_usage(usage_this_turn, response)
                    _add_usage(result.usage, response)
                    # Search requests can be short; they must not overwrite the
                    # space already needed by a full thought-and-save response.
                    output_token_reserve = max(output_token_reserve, usage_this_turn.get("output", 0))
                    if "output" not in usage_this_turn or not ({"input_other", "input_cached"} & usage_this_turn.keys()):
                        result.unknown_usage_calls += 1
                    observed_input = usage_this_turn.get("input_other", 0) + usage_this_turn.get("input_cached", 0)
                    if observed_input:
                        previous_input_tokens, previous_input_bytes = observed_input, input_bytes
                        input_tokens_per_byte = observed_input / input_bytes
                    text = str(getattr(response, "completion_text", "") or "")
                    result.response_text = text
                    native_calls = response_calls(response)
                    if _finish_reason(response) == "length":
                        interrupted = {"role": "assistant", "content": text,
                                       "reasoning_content": getattr(response, "reasoning_content", None) or ""}
                        if native_calls:
                            interrupted["tool_calls"] = [call.message() for call in native_calls]
                        conversation.append(interrupted)
                        for call in native_calls:
                            conversation.append({"role": "tool", "tool_call_id": call.id, "content": _json({
                                "status": "error", "detail": "输出达到上限，本条调用未执行。原参数保留在上一条调用，请修正或拆小后重新保存。"})})
                            if call.name in {"remember", "reflect"}:
                                pending_write_errors[call.name] = "输出被截断，本条写入未执行；原始参数已保留。"
                        if not native_calls:
                            pending_write_errors["remember"] = "最后输出被截断，尚未保存；请根据上一条保留的原始输出补全或拆分保存。"
                        result.status, result.detail = "partial", "Consolidation output reached its token limit"
                        break
                    if not native_calls:
                        conversation.append({"role": "assistant", "content": text,
                                             "reasoning_content": getattr(response, "reasoning_content", None) or ""})
                        try:
                            envelope = _memory_output(text)
                        except ValueError as exc:
                            pending_write_errors["remember"] = f"最后输出未保存：{exc}"
                            conversation.append({"role": "user", "content": _json({"write_result": {
                                "status": "error", "detail": pending_write_errors["remember"],
                                "action": "原始输出已完整保留在上一条消息；修正参数格式后重新保存，无需重复已做的检索。"}})})
                            await self._emit("write", "最后输出未保存，等待模型修正", "error",
                                             target="long_term_memory", detail=str(exc))
                            continue
                        if repair := envelope.pop("_format_repair", ""):
                            await self._emit("write", "已读取模型的完整保存内容", target="long_term_memory",
                                             format_repair=repair)
                        if task is not None:
                            receipt = await apply_write(envelope, f"final:{getattr(self.trace, 'id', 'local')}:{turn}")
                            if pending_write_errors:
                                conversation.append({"role": "user", "content": _learning_json({"write_result": receipt,
                                    "unfinished_writes": pending_write_errors}, seen_messages=seen_messages)})
                                result.status, result.detail = "partial", "; ".join(pending_write_errors.values())
                                continue
                            result.status = "completed"
                            break
                        result.items = await clean_items(envelope["items"])
                        writer.recollection = envelope.get("recollection", {})
                        if not isinstance(envelope["progress"], dict):
                            raise ValueError("Learning progress must be an object")
                        result.progress = envelope["progress"]
                        if pending_write_errors and ("reflect" in pending_write_errors or not result.items):
                            result.status, result.detail = "partial", "; ".join(pending_write_errors.values())
                        else:
                            result.status = "completed"
                        break
                    conversation.append({"role": "assistant", "content": text or None, "tool_calls": [call.message() for call in native_calls],
                        "reasoning_content": getattr(response, "reasoning_content", None) or ""})
                    # Learning may read its own writes; execute in the model's order.
                    finish_requested = False
                    for call in native_calls:
                        name, args, call_id = call.name, call.arguments, call.id
                        began = time.monotonic()
                        record = {"name": name, "arguments": args, "id": call_id, "status": "running"}
                        if call.error:
                            record["raw_arguments"] = call.raw_arguments
                        result.tool_calls.append(record)
                        value = None
                        phase = "write" if name == "remember" or (name == "reflect" and set(args or {}) != {"id"}) else "read"
                        extra = {"target": "long_term_memory" if name == "remember" else "reflection"} if phase == "write" else {}
                        await self._emit(phase, f"{'保存' if phase == 'write' else '读取'} {name}", "running",
                                         turn=turn + 1, tool_call_id=call_id, name=name, arguments=args, **extra)
                        try:
                            if call.error:
                                raise ValueError(call.error)
                            if name == "remember":
                                write_id = f"{getattr(self.trace, 'id', 'local')}:{len(conversation)}:{call_id}"
                                value = await apply_write(args, write_id)
                                if "finish" in args:
                                    finish_requested = args["finish"]
                            else:
                                value = await self._execute(name, args, cutoff)
                                if name == "reflect" and set(args) != {"id"}:
                                    pending_write_errors.pop("reflect", None)
                            read_sources(value)
                            record["status"] = "partial" if name == "remember" and (
                                "remember" in pending_write_errors or isinstance(value, dict)
                                and value.get("status") == "partial") else "completed"
                        except asyncio.CancelledError:
                            record["status"] = "cancelled"
                            raise
                        except Exception as exc:
                            value = {"status": "error", "detail": f"{type(exc).__name__}: {exc}"}
                            record["status"] = "error"
                            if name == "remember" or (name == "reflect" and set(args or {}) != {"id"}):
                                pending_write_errors[name] = f"{name} write has not completed: " + value["detail"]
                        finally:
                            record["elapsed_ms"] = (time.monotonic() - began) * 1000
                            result_step = await self._emit(phase, f"{'保存' if phase == 'write' else '读取'} {name}", record["status"],
                                             turn=turn + 1, tool_call_id=call_id, name=name,
                                             elapsed_ms=record["elapsed_ms"], result=value, **extra)
                        record["result"] = value
                        # Keep the ordered timeline; bodies already read in this
                        # conversation are references, with authors still visible.
                        # A save acknowledges normalized content and real addresses;
                        # its top-level sources were already read to perform the save.
                        receipt = ([{key: part for key, part in row.items() if key != "sources"} for row in value]
                                   if name == "remember" and isinstance(value, list) else value)
                        candidate_seen = copy.deepcopy(seen_messages)
                        tool_message = {"role": "tool", "tool_call_id": call_id,
                            "content": _learning_json(receipt, seen_messages=candidate_seen)}
                        unread_later = False
                        if phase == "read" and record["status"] == "completed" and type(result_step) is int and remaining is not None:
                            next_bytes = _learning_bytes([*conversation, tool_message], feedback) + 2048
                            next_input = _estimate_input_tokens(next_bytes, previous_input_tokens, previous_input_bytes,
                                                                input_tokens_per_byte)
                            left = token_budget - sum(result.usage.values())
                            save_room = min(self.max_output_tokens, max(2048, output_token_reserve))
                            if next_input + save_room > left and await self.trace.persisted(result_step):
                                unread_later = True
                                record["deferred_for_model"] = True
                                tool_message["content"] = _json({"status": "deferred_for_budget",
                                    "result_ref": {"run_id": self.trace.id, "step": result_step},
                                    "detail": "查询结果已完整存档，但本轮剩余额度不足以携带全文并保存。你尚未读到这份结果；之后可用interaction(run_id=run_id,step=step)打开全文。先保存当前已理解的材料、未完成线索及此结果地址。"})
                        if not unread_later:
                            seen_messages = candidate_seen
                        conversation.append(tool_message)
                    if finish_requested:
                        progress = await asyncio.to_thread(self.store.learning_task, learning_kind) if task is not None else None
                        unfinished = set(progress["material_ids"]) - set(progress["completed_ids"]) if progress else set()
                        if not unfinished and not pending_write_errors:
                            result.status = "completed"
                            break
                        reason = (f"本批还有未处理材料 {sorted(unfinished)}。" if unfinished else "")
                        reason += "; ".join(pending_write_errors.values())
                        conversation.append({"role": "user", "content": "finish尚未完成，已有写入和进度已保留：" + reason})
        except asyncio.CancelledError:
            cancelled = True
            raise
        except Exception as exc:
            if result.written:
                result.status = "partial"
            result.detail = f"{type(exc).__name__}: {exc}"
        finally:
            result.elapsed_ms = (time.monotonic() - started) * 1000
            if result.status == "completed" and writer.pending_items:
                result.detail = f"本批材料已完成；{len(writer.pending_items)}条记忆草稿仍待修复，已独立保留"
            pending_calls = []
            for message in conversation:
                if message.get("role") == "assistant":
                    pending_calls = [call["id"] for call in message.get("tool_calls", [])]
                elif message.get("role") == "tool" and message.get("tool_call_id") in pending_calls:
                    pending_calls.remove(message["tool_call_id"])
            for call_id in pending_calls:
                conversation.append({"role": "tool", "tool_call_id": call_id, "content": _json({
                    "status": "interrupted", "detail": "调用中断，结果尚未收到；恢复后结合已保存的进度与实际记忆检查再继续。"})})
            input_bytes = _learning_bytes(conversation, feedback)
            result.continuation = {
                "conversation": conversation, "sources": sources, "seen_messages": seen_messages,
                "workspace": self.last_workspace,
                "write_state": writer.state(),
                "pending_write_errors": pending_write_errors, "previous_input_tokens": previous_input_tokens,
                "previous_input_bytes": previous_input_bytes, "previous_call_usage": usage_this_turn,
                "input_tokens_per_byte": input_tokens_per_byte,
                "output_token_reserve": output_token_reserve,
                "previous_call_seconds": previous_call_seconds, "working": working,
                "task_state": _learning_material([], {}, task).get("learning_task", {}),
                "estimated_next_input_tokens": _estimate_input_tokens(input_bytes, previous_input_tokens,
                                                                       previous_input_bytes, input_tokens_per_byte)}
            if task is not None:
                await asyncio.to_thread(self.store.update_learning_task, learning_kind, {
                    "continuation": result.continuation, "run_id": getattr(self.trace, "id", None),
                    "updated_at": time.time()})
            result.messages = [{key: value for key, value in message.items() if key != "reasoning_content"}
                               for message in conversation]
            await self._record_experience(learning_kind, {}, result, writer)
            await self._emit("output", "本次学习产出", "cancelled" if cancelled else result.status,
                             pending_item_count=len(result.items), written_count=len(result.written), detail=result.detail,
                             usage=result.usage, elapsed_ms=result.elapsed_ms)
        return result
