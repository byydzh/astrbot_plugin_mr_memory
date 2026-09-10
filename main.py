from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import time
import weakref
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star
from astrbot.core.agent.message import TextPart
from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

from .mr_memory.agent import MemoryAgent, estimate_learning_input
from .mr_memory.embedding import Embedder
from .mr_memory.store import Store
from .mr_memory.settings import normalize_settings
from .mr_memory.schedule import work_window
from .mr_memory.console import register_console
from .mr_memory.trace import RunTrace


def json_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "json") and hasattr(value, "dict"):
        return json.loads(value.json())
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def components(chain: list[Any]) -> list[dict]:
    """Keep the platform's message structure alongside its readable text."""
    result = []
    for item in chain:
        kind = item.__class__.__name__.lower()
        if kind in {"plain", "plaintext"}:
            result.append({"type": "text", "text": item.text})
        elif kind == "at":
            result.append({"type": "mention", "account_id": str(item.qq),
                           "display_name": str(getattr(item, "name", "") or "")})
        elif kind == "reply":
            result.append({"type": "reply", "message_id": str(item.id),
                           "sender_id": str(getattr(item, "sender_id", "") or ""),
                           "sender_name": str(getattr(item, "sender_nickname", "") or ""),
                           "sent_at": getattr(item, "time", 0) or 0,
                           "plain_text": str(getattr(item, "message_str", "") or ""),
                           "content": components(list(getattr(item, "chain", []) or []))})
        else:
            value = json_value(item)
            if not isinstance(value, dict):
                value = {"description": value}
            result.append({**value, "type": kind})
    return result


def readable(chain: list[dict]) -> str:
    return " ".join(str(item.get("text") or item.get("plain_text") or
                        item.get("description") or f"[{item['type']}]") for item in chain)


class OneBotSendObserver:
    """Observe one event's sends without modifying its shared CQHttp client."""

    def __init__(self, bot, record, original_content):
        self.bot, self.record = bot, record
        self.forward_content = [part for part in original_content if part["type"] in {"node", "nodes"}]
        self.media = {}
        for part in original_content:
            self.media.setdefault(part["type"], []).append(part)

    def __getattr__(self, name):
        return getattr(self.bot, name)

    def sent_content(self, payload):
        result = []
        for part in payload:
            kind, data = part["type"], part.get("data", {})
            if kind == "text":
                result.append({"type": "text", "text": data.get("text", "")})
            elif kind == "at":
                result.append({"type": "mention", "account_id": str(data.get("qq", ""))})
            elif originals := self.media.get(kind):
                # Keep original media references, not the transport's base64 copy.
                result.append(originals.pop(0))
            else:
                result.append({**data, "type": kind})
        return result

    async def send_group_msg(self, **params):
        receipt = await self.bot.send_group_msg(**params)
        try:
            await self.record(receipt, self.sent_content(params["message"]))
        except Exception:
            logger.exception("MR: could not observe OneBot send receipt")
        return receipt

    async def call_action(self, action, **params):
        receipt = await self.bot.call_action(action, **params)
        if action == "send_group_forward_msg":
            try:
                content = [self.forward_content.pop(0)] if self.forward_content else [{"type": "nodes", "content": params.get("messages", [])}]
                await self.record(receipt, content)
            except Exception:
                logger.exception("MR: could not observe OneBot send receipt")
        return receipt


class MrMemoryPlugin(Star):
    """Connect group experience and one active memory agent to AstrBot."""

    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config = normalize_settings(config)
        self.data_dir = Path(get_astrbot_data_path()) / "plugin_data" / "astrbot_plugin_mr_memory"
        self.scope_dir = self.data_dir / "scopes"
        self.scope_dir.mkdir(parents=True, exist_ok=True)
        self.stores: dict[str, Store] = {}
        self.active_scopes: set[str] = set()
        self.open_lock = asyncio.Lock()
        self.state_locks: dict[str, asyncio.Lock] = {}
        self.learning_locks: dict[str, asyncio.Lock] = {}
        self.tasks: set[asyncio.Task] = set()
        self.requests: set[asyncio.Task] = set()
        self.observed_events = weakref.WeakSet()
        self.last_result: dict[str, dict] = {}
        self.learning_status: dict[str, dict] = {}
        self.stopping = False
        self.embedder = Embedder(
            str(config.get("embedding_model_name", "microsoft/harrier-oss-v1-270m")),
            self.data_dir / "models" / "sentence_transformers",
            threads=int(config.get("embedding_cpu_threads", 1)),
            batch_size=int(config.get("embedding_batch_size", 1)),
            query_prompt=str(config.get("embedding_query_prompt_name", "web_search_query")),
            max_seq_length=int(config.get("embedding_max_seq_length", 512)),
        )
        self.console = register_console(self)

    def spawn(self, coroutine) -> asyncio.Task:
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    async def initialize(self) -> None:
        if self.config["embedding_enabled"]:
            await self.embedder.warmup()
        # Deferred messages and unfinished work must wake even if a group stays
        # quiet after reload. Opening a scope does not start a model call.
        def reflection_scopes():
            scopes = []
            allowed = self.config["allowed_umos"]
            for path in self.scope_dir.glob("*.db"):
                try:
                    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
                        row = db.execute("SELECT umo FROM scope_meta WHERE singleton=1").fetchone()
                        if not row or (allowed and row[0] not in allowed):
                            continue
                        scopes.append((path, row[0]))
                except sqlite3.Error:
                    logger.exception("MR: could not reopen reflection scope %s", path.name)
            return scopes

        for path, umo in await asyncio.to_thread(reflection_scopes):
            async with self.open_lock:
                if umo not in self.stores:
                    self.stores[umo] = await asyncio.to_thread(Store, path, umo)
                    self.state_locks[umo] = asyncio.Lock()
                self.active_scopes.add(umo)
        self.spawn(self.maintain())
        logger.info("MR: active memory reconstruction ready")

    def allowed(self, event: AstrMessageEvent) -> bool:
        scopes = self.config.get("allowed_umos", [])
        return bool(event.get_group_id()) and (not scopes or event.unified_msg_origin in scopes)

    async def store_for(self, event: AstrMessageEvent) -> Store:
        umo = event.unified_msg_origin
        self.active_scopes.add(umo)
        if umo not in self.stores:
            async with self.open_lock:
                if umo not in self.stores:
                    # Keep the existing on-disk group filename; this is not a semantic check.
                    path = self.scope_dir / (hashlib.sha256(umo.encode()).hexdigest() + ".db")
                    self.stores[umo] = await asyncio.to_thread(Store, path, umo)
                    self.state_locks[umo] = asyncio.Lock()
        return self.stores[umo]

    def agent(self, store: Store, *, background: bool = False, feedback: bool = False) -> MemoryAgent:
        provider_id = str(self.config["subconscious_provider_id"])
        if not background:
            provider_id = self.config["local_serving_reader_provider_id"] or provider_id
        provider = self.context.get_provider_by_id(provider_id)
        if provider is None:
            raise RuntimeError(f"MR provider is unavailable: {provider_id}")
        return MemoryAgent(provider, store, self.embedder if self.config["embedding_enabled"] else None,
                           timeout_seconds=float(self.config["maintenance_llm_timeout_seconds"] if background
                                                 else self.config["local_serving_timeout_seconds"]),
                           max_turns=None if background else int(self.config["max_loop_steps"]),
                           max_output_tokens=int(self.config["distillation_max_output_tokens"] if background
                                                 else self.config["memory_max_tokens"]),
                           thinking_mode=self.config["feedback_thinking_mode"] if feedback else
                           self.config["distillation_thinking_mode"] if background else
                           self.config["local_serving_reader_thinking_mode"])

    @staticmethod
    def message(event: AstrMessageEvent) -> dict:
        obj = event.message_obj
        content = components(list(obj.message or []))
        raw = getattr(obj, "raw_message", None)
        stamp = raw.get("time") if isinstance(raw, dict) else None
        stamp = stamp or obj.timestamp or time.time()
        if hasattr(stamp, "timestamp"):
            stamp = stamp.timestamp()
        return {"platform": event.get_platform_name(), "platform_id": event.get_platform_id(),
                "umo": event.unified_msg_origin, "group_id": str(event.get_group_id()),
                "message_id": str(obj.message_id), "sender_id": str(obj.sender.user_id),
                "sender_name": str(obj.sender.nickname or ""), "sent_at": int(stamp),
                "plain_text": str(obj.message_str or ""), "content": content, "role": "USER",
                "reply_to": next((x["message_id"] for x in content if x["type"] == "reply"), "")}

    async def update_state(self, store: Store, changes: dict) -> None:
        async with self.state_locks[store.umo]:
            await asyncio.to_thread(store.update_working_state, changes)

    def observe_sends(self, event: AstrMessageEvent) -> None:
        """Record successful sends on this event, including streamed segments."""
        if getattr(event, "_mr_memory_send_observed", None) is self:
            return
        original_send = getattr(event, "_mr_memory_original_send", event.send)
        observes_receipts = isinstance(event, AiocqhttpMessageEvent)

        if observes_receipts:
            original_transport = getattr(event, "_mr_memory_original_send_message", event.send_message)

            async def record_receipt(receipt, chain):
                try:
                    message_id = receipt.get("message_id") if isinstance(receipt, dict) else None
                    await self.record_bot_event(event, "sent", readable(chain), chain, message_id=message_id)
                except Exception:
                    logger.exception("MR: could not record sent response")

            async def send_message_and_record(bot, message_chain, *args, **kwargs):
                try:
                    original_content = components(list(message_chain.chain))
                except Exception:
                    logger.exception("MR: could not read original media references")
                    original_content = []
                observer = OneBotSendObserver(bot, record_receipt, original_content)
                return await original_transport(observer, message_chain, *args, **kwargs)

            event.send_message = send_message_and_record
            event._mr_memory_original_send_message = original_transport
            event._mr_memory_send_message_wrapper = send_message_and_record

        async def send_and_record(message, *args, **kwargs):
            receipt = await original_send(message, *args, **kwargs)
            try:
                if not self.stopping and not observes_receipts:
                    chain = components(list(getattr(message, "chain", []) or []))
                    await self.record_bot_event(event, "sent", readable(chain), chain)
            except Exception:
                logger.exception("MR: could not record sent response")
            return receipt

        event.send = send_and_record
        event._mr_memory_send_observed = self
        event._mr_memory_original_send = original_send
        event._mr_memory_send_wrapper = send_and_record
        self.observed_events.add(event)

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=1000)
    async def capture_group_message(self, event: AstrMessageEvent) -> None:
        if self.stopping or not self.allowed(event) or not self.config["capture_enabled"]:
            return
        self.observe_sends(event)
        try:
            store = await self.store_for(event)
            raw = getattr(event.message_obj, "raw_message", {})
            if isinstance(raw, dict) and raw.get("notice_type") == "group_recall":
                await asyncio.to_thread(store.delete_message, str(raw["message_id"]))
            else:
                await asyncio.to_thread(store.append_message, self.message(event))
        except Exception:
            logger.exception("MR: could not record group message")

    async def refresh_roster(self, store: Store, event: AstrMessageEvent) -> None:
        bot = getattr(event, "bot", None)
        if bot is None:
            return
        cached = await asyncio.to_thread(store.roster)
        if cached and time.time() - cached.get("fetched_at", 0) < 900:
            return
        try:
            routing = {}
            if self_id := getattr(event.message_obj, "self_id", None):
                routing["self_id"] = self_id
            async with asyncio.timeout(3):
                members = await bot.get_group_member_list(group_id=int(event.get_group_id()), **routing)
            await asyncio.to_thread(store.cache_roster, members, int(time.time()))
        except Exception as exc:
            logger.warning("MR: current group roster unavailable: %s", type(exc).__name__)

    @filter.on_llm_request()
    async def inject_subconscious_memory(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        if self.stopping:
            return
        task = asyncio.create_task(self.reconstruct_request(event, req))
        self.requests.add(task)
        try:
            await task
        finally:
            self.requests.discard(task)

    async def reconstruct_request(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        if not self.allowed(event):
            return
        if self.config["capture_enabled"]:
            self.observe_sends(event)
        if not self.config["local_serving_enabled"]:
            return
        started = time.perf_counter()
        started_at = time.time()
        store = await self.store_for(event)
        current = self.message(event)
        current["question"] = req.prompt or current["plain_text"]
        payload = {"status": "cancelled", "question": current["question"], "request_id": current["message_id"]}
        trace = RunTrace(store, "foreground", started_at, payload)
        await trace.start()
        prepared = False
        try:
            await trace.emit("input", "准备当前问题与群聊上下文", status="running", operation_id="prepare")
            recorded = await asyncio.to_thread(store.append_message, current) if self.config["capture_enabled"] else {}
            if "id" in recorded:
                current.update(id=recorded["id"], source_key=recorded["source_key"])
            current["content"] = await asyncio.to_thread(store.resolve_quotes, current["content"], current["platform_id"])
            current["bot_id"] = str(event.get_self_id() or "")
            roster_task = self.spawn(self.refresh_roster(store, event))
            recent, working = await asyncio.gather(
                asyncio.to_thread(store.recent, limit=int(self.config.get("recent_messages", 24)), before=current["sent_at"] + 1),
                asyncio.to_thread(store.load_working_state),
            )
            cached_roster = await asyncio.to_thread(store.roster)
            if not cached_roster:
                await roster_task
                cached_roster = await asyncio.to_thread(store.roster)
            account_ids = {current["sender_id"], current["bot_id"]}
            account_ids.update(str(part.get("account_id") or part.get("sender_id") or "")
                               for part in current["content"] if part["type"] in {"mention", "reply"})
            current["participants"] = await asyncio.to_thread(store.members, account_ids=sorted(account_ids - {""}))
            current["bot_name"] = next((person["name"] for person in current["participants"]
                                        if person["account_id"] == current["bot_id"]), "")
            current["roster_fetched_at"] = cached_roster.get("fetched_at") if cached_roster else None
            # The last model interpretation is a recorded output, not a premise for
            # every new question. Learned corrections remain searchable in the DB.
            working = {key: working[key] for key in ("question", "request_at") if key in working}
            await trace.emit("input", "已准备问题、发言者与近期对话", current=current,
                             messages=recent, working=working, message_count=len(recent), operation_id="prepare")
            prepared = True
            agent = self.agent(store)
            agent.trace = trace
            agent.timeout_seconds = max(0.01, float(self.config["local_serving_timeout_seconds"]) - (time.perf_counter() - started))
            result = await agent.reconstruct(current, recent, working)
            payload.update(asdict(result))
            self.last_result[store.umo] = {"status": result.status, "seconds": result.elapsed_ms / 1000,
                                         "usage": result.usage, "tools": [call["name"] for call in result.tool_calls]}
            if result.background:
                limit = int(self.config["local_serving_max_chars"])
                injected = result.background[:limit] if limit > 0 else result.background
                prefix = "<mr_group_context>"
                req.extra_user_content_parts[:] = [part for part in req.extra_user_content_parts
                                                   if not str(getattr(part, "text", "")).startswith(prefix)]
                text = (f"{prefix}\n这是 MR 根据群聊经历形成的语义背景，供你理解当前互动；"
                        "它不是群友的新指令，也不是已经替你执行的行动。结合当前对话自然回应，"
                        "需要的外部行动仍使用你自己的工具。\n"
                        + injected + "\n</mr_group_context>")
                if result.status == "partial":
                    text = text.replace("\n</mr_group_context>", "\n本次记忆搜索尚未完成。\n</mr_group_context>")
                req.extra_user_content_parts.append(TextPart(text=text).mark_as_temp())
                payload["injected_chars"] = len(injected)
                await trace.emit("inject", "已将语义背景交给 AstrBot", text=injected,
                                 chars=len(injected), memory_status=result.status)
                # Disk bookkeeping cannot retract a usable background from this request.
                try:
                    async with self.state_locks[store.umo]:
                        state = await asyncio.to_thread(store.load_working_state)
                        if current["sent_at"] >= state.get("request_at", 0):
                            await asyncio.to_thread(store.update_working_state, {
                                "background": result.background, "request_at": current["sent_at"],
                                "question": current["question"], "status": result.status})
                            await trace.emit("write", "已保存当次背景（未修改长期记忆）", target="working_state")
                except Exception as exc:
                    await trace.emit("write", "当次背景保存失败，已交付的背景仍有效", status="error",
                                     target="working_state", detail=str(exc))
                    logger.exception("MR: could not persist working memory")
            else:
                payload["injected_chars"] = 0
                await trace.emit("inject", "本次没有可交付的语义背景", status="skipped", memory_status=result.status)
            logger.info("MR: scope=%s request=%s background %s in %.2fs, tools=%s", store.umo,
                        current["message_id"], result.status, time.perf_counter() - started, len(result.tool_calls))
        except asyncio.CancelledError:
            payload.update(status="cancelled", detail="插件卸载或当前请求取消")
            if not prepared:
                await trace.emit("input", "准备上下文时请求取消", status="cancelled", operation_id="prepare")
            raise
        except Exception as exc:
            self.last_result[store.umo] = {"status": "error", "error": type(exc).__name__}
            payload.update(status="error", detail=str(exc))
            if not prepared:
                await trace.emit("input", "准备上下文失败", status="error", detail=str(exc), operation_id="prepare")
            logger.exception("MR: subconscious reconstruction failed")
        finally:
            payload["elapsed_ms"] = (time.perf_counter() - started) * 1000
            await trace.finish(payload)

    async def record_bot_event(self, event: AstrMessageEvent, kind: str, text: str, content: list[dict], *, message_id=None) -> None:
        if self.stopping or not self.allowed(event) or not self.config["capture_enabled"] or not (text or content):
            return
        store = await self.store_for(event)
        row = self.message(event)
        request_id = row["message_id"]
        bot_id = str(event.get_self_id() or "")
        bot = await asyncio.to_thread(store.members, account_ids=[bot_id])
        row.update(message_id=str(message_id) if message_id is not None else f"astrbot:{request_id}:{kind}:{time.time_ns()}",
                   sender_id=bot_id, sender_name=bot[0]["name"] if bot else "",
                   sent_at=int(time.time()), plain_text=text, role="BOT" if kind == "sent" else "SYSTEM",
                   content=[{"type": "response_to", "message_id": request_id},
                            {"type": "bot_event", "event": kind}, *content], reply_to=request_id)
        await asyncio.to_thread(store.append_message, row)

    @filter.on_llm_response()
    async def capture_response(self, event: AstrMessageEvent, response: Any) -> None:
        try:
            await self.record_bot_event(event, "generated", str(getattr(response, "completion_text", "") or ""), [])
        except Exception:
            logger.exception("MR: could not record generated response")

    @filter.on_using_llm_tool()
    async def capture_tool_call(self, event: AstrMessageEvent, tool: Any, tool_args: dict | None) -> None:
        try:
            await self.record_bot_event(event, "tool_call", "", [{"type": "tool_call",
                "name": str(getattr(tool, "name", "")), "arguments": tool_args or {}}])
        except Exception:
            logger.exception("MR: could not record tool call")

    @filter.on_llm_tool_respond()
    async def capture_tool_result(self, event: AstrMessageEvent, tool: Any,
                                  tool_args: dict | None, tool_result: Any) -> None:
        try:
            value = json_value(tool_result)
            await self.record_bot_event(event, "tool_result", "", [{"type": "tool_result",
                "name": str(getattr(tool, "name", "")), "arguments": tool_args or {}, "result": value}])
        except Exception:
            logger.exception("MR: could not record tool result")

    async def consolidate(self, store: Store, *, force: bool = False) -> dict:
        # Manual console requests participate in plugin unload just like the
        # background loop, so their database is not closed during a model call.
        return await self.spawn(self.learn(store, force=force))

    async def learn(self, store: Store, *, force: bool = False, feedback: bool = False) -> dict:
        kind = "feedback" if feedback else "background"
        if self.stopping or not self.config["subconscious_enabled"]:
            return {"status": "disabled", "reason": "后台潜意识已关闭"}
        allowed = self.config["allowed_umos"]
        if allowed and store.umo not in allowed:
            return {"status": "disabled", "reason": "此群未启用"}
        lock = self.learning_locks.setdefault(store.umo, asyncio.Lock())
        if lock.locked():
            return {"status": "busy", "reason": "此群正在整理记忆"}
        async with lock:
            def report(status, reason, **detail):
                value = {"status": status, "reason": reason, **detail}
                self.learning_status.setdefault(store.umo, {})[kind] = value
                return value

            window = work_window(self.config)
            scheduled = []
            if not force and not window["open"]:
                if feedback:
                    scheduled = await asyncio.to_thread(store.reflections.due, scheduled_only=True)
                if not scheduled:
                    return report("deferred", "等待服务当地工作时段", **window)
            state = await asyncio.to_thread(store.load_working_state)
            used = await asyncio.to_thread(store.usage_total, kind)
            budget = int(self.config["feedback_daily_token_budget" if feedback else "private_daily_token_budget"])
            if budget > 0 and used >= budget:
                return report("budget_exhausted", "滚动24小时额度已用完", used_tokens=used)
            remaining = budget - used if budget > 0 else None
            maximum = int(self.config["distillation_max_messages"])
            task = await asyncio.to_thread(store.learning_task, kind)
            resuming = task is not None
            reflection_tasks = []
            if task:
                messages = await asyncio.to_thread(store.learning_messages, task)
                done = set(task["completed_ids"])
                messages = [row for row in messages if row.get("context_only") or row["id"] not in done]
                working = dict(task["working"])
                # A model appointment can bring an unfinished feedback task
                # forward. It is not an invitation to process fresh backlog.
                if scheduled:
                    working["scheduled_reflections"] = scheduled
                    working["focus"] = "工作时段外先处理这些模型预约到期的关注；其余材料可继续留待工作时段。"
                refs = working.get("reflections", [])
                refreshed = [await asyncio.to_thread(store.reflections.get, row["id"]) for row in refs]
                reflection_tasks = [row for row in refreshed if row and row["status"] != "resolved"]
                working["reflections"] = reflection_tasks
                # Keep the native prefix while it fits. If it cannot continue
                # within available resources, a model-written checkpoint and
                # real saved memories let it resume without the whole trace.
                if (task.get("checkpoint") and task.get("continuation") and remaining is not None
                        and estimate_learning_input(messages, working, feedback=feedback, task=task) * 2 > remaining):
                    task = {**task, "continuation": {}}
                if not task.get("continuation"):
                    memories = [await asyncio.to_thread(store.memory, ref["kind"], ref["id"], include_sources=False)
                                for ref in task["memory_refs"]]
                    working["saved_memories"] = [row for row in memories if row]
            elif feedback:
                reflection_tasks = scheduled or await asyncio.to_thread(store.reflections.due)
                messages = await asyncio.to_thread(store.feedback_messages,
                    int(self.config["feedback_window_seconds"]), int(state.get("feedback_after", 0)), maximum) if not scheduled else []
                if messages and time.time() - messages[-1]["sent_at"] < self.config["feedback_debounce_seconds"]:
                    if not reflection_tasks:
                        return report("waiting", "等待当前互动结束")
                    messages = []
            else:
                pending = await asyncio.to_thread(store.pending_status)
                if not force and pending["count"] < self.config["auto_distillation_min_pending"]:
                    if pending["oldest_at"] is None or time.time() - pending["oldest_at"] < self.config["maintenance_interval_seconds"]:
                        return report("waiting", "等待积累消息或到达最长等待时间")
                # Live group experience gets the next batch; idle capacity catches up older history.
                messages = await asyncio.to_thread(store.pending_messages, maximum, newest=True)
            if not task:
                if not messages and not reflection_tasks:
                    return report("idle", "没有待处理互动")
                if feedback:
                    waiting = await asyncio.to_thread(store.reflections.waiting) if messages else []
                    reflection_tasks = list({item["id"]: item for item in reflection_tasks + waiting}.values())
                    working = {"interactions": await asyncio.to_thread(store.reflections.feedback_context, messages),
                               "reflections": reflection_tasks}
                else:
                    reflection_tasks = await asyncio.to_thread(store.reflections.waiting)
                    working = {"reflections": reflection_tasks} if reflection_tasks else {}
                previous_learning = await asyncio.to_thread(store.unfinished_learning, kind)
                if previous_learning:
                    working["previous_learning"] = previous_learning

            # The configured message count is a maximum. Fit whole messages to
            # the available input/continuation budget; omitted messages remain
            # queued and can also be opened by the model's context tool.
            estimate = estimate_learning_input(messages, working, feedback=feedback, task=task)
            while remaining is not None and estimate * 2 > remaining and not (task or {}).get("continuation"):
                material = [row for row in messages if not row.get("context_only")]
                if len(material) <= 1:
                    break
                context = [row for row in messages if row.get("context_only")]
                count = max(1, len(material) // 2)
                messages = context + (material[:count] if feedback or task else material[-count:])
                if feedback:
                    working["interactions"] = await asyncio.to_thread(store.reflections.feedback_context, messages)
                estimate = estimate_learning_input(messages, working, feedback=feedback, task=task)
            required = estimate + 1 if (task or {}).get("continuation") else estimate * 2
            if remaining is not None and required > remaining:
                return report("budget_wait", "剩余额度不足以读取并保存当前材料，等待滚动额度释放",
                              used_tokens=used, required_tokens=required)

            if task is None:
                task = await asyncio.to_thread(store.start_learning_task, kind,
                    [row["id"] for row in messages if not row.get("context_only")],
                    [row["id"] for row in messages if row.get("context_only")], working)
            task = {**task, "offered_ids": [row["id"] for row in messages if not row.get("context_only")]}
            agent = self.agent(store, background=True, feedback=feedback)
            reserved = remaining if remaining is not None else estimate + int(self.config["distillation_max_output_tokens"])
            started_at = time.time()
            usage_id = await asyncio.to_thread(store.reserve_usage, kind, reserved, started_at)
            payload = {"status": "interrupted", "messages": messages, "working": working,
                       "usage": {}, "usage_estimated": True, "reserved_tokens": reserved,
                       "material_ids": task["material_ids"], "context_ids": task["context_ids"],
                       "previous_run_id": task.get("run_id")}
            trace = RunTrace(store, kind, started_at, {"message_count": len(messages), "status": "running"})
            await trace.start()
            try:
                report("running", "接着上次已保存的进度继续" if resuming else "正在学习本批交流")
                await trace.emit("input", "继续上次未完成的学习" if resuming else "已选取原回答与后续交流" if feedback else "已选取待整理群聊",
                                 messages=messages, message_count=len(messages), working=working,
                                 budget_kind=kind, rolling_budget=budget, used_tokens=used,
                                 previous_run_id=task.get("run_id"), completed_ids=task["completed_ids"],
                                 checkpoint=task["checkpoint"], memory_refs=task["memory_refs"])
                agent.trace = trace
                result = await agent.consolidate(
                    messages, working, feedback=feedback, token_budget=remaining, task=task)
                payload.update({key: value for key, value in asdict(result).items() if key != "continuation"},
                               usage_estimated=bool(result.unknown_usage_calls) or not bool(result.usage))
                if result.usage and not result.unknown_usage_calls:
                    await asyncio.to_thread(store.settle_usage, usage_id, sum(result.usage.values()))
                elif result.model_attempts == 0:
                    await asyncio.to_thread(store.settle_usage, usage_id, 0)
                    payload["usage_estimated"] = False
                source_keys = [m["source_key"] for m in messages]
                learned_sources = list(dict.fromkeys(source_keys +
                    [key for item in result.items for key in item.get("source_keys", [])]))
                if result.items:
                    await trace.emit("write", "保存模型结束时形成的长期记忆", status="running",
                                     target="long_term_memory", items=result.items, operation_id="final_memories")
                try:
                    saved = await asyncio.to_thread(store.save_memories, result.items, learned_sources, mark_processed=False,
                                                    run_id=trace.id, learning_kind=kind, progress=result.progress)
                except BaseException as exc:
                    if result.items:
                        await trace.emit("write", "末次长期记忆保存未完成", target="long_term_memory", operation_id="final_memories",
                                         status="cancelled" if isinstance(exc, asyncio.CancelledError) else "error", detail=str(exc))
                    raise
                if result.items:
                    await trace.emit("write", "已保存末次长期记忆", target="long_term_memory", result=saved,
                                     written_count=len(saved), operation_id="final_memories")
                written = result.written + saved
                written = list({(row["kind"], row["id"]): row for row in written}.values())
                payload["written"] = written
                progress = await asyncio.to_thread(store.learning_task, kind)
                payload.update({key: progress[key] for key in ("completed_ids", "checkpoint", "memory_refs")})
                unfinished = set(progress["material_ids"]) - set(progress["completed_ids"])
                completed = result.status == "completed" and not unfinished
                await trace.emit("write", "已保存本批学习进度", target="processing_progress",
                                 completed_ids=progress["completed_ids"], remaining_message_count=len(unfinished),
                                 checkpoint=progress["checkpoint"], memory_refs=progress["memory_refs"])
                if feedback:
                    # LLM interpretations remain available to the next foreground turn,
                    # including corrections, while original dialogue remains in the store.
                    learned = [{key: value for key, value in row.items() if key in {
                        "kind", "id", "title", "summary", "content", "subject", "aspect",
                        "source", "target", "relation", "statement", "source_ids"}} for row in written]
                    if learned:
                        await self.update_state(store, {"feedback_at": int(time.time()), "learned_feedback": learned})
                elif completed:
                    await self.update_state(store, {"consolidated_at": int(time.time())})
                if completed and (reflection_tasks or scheduled):
                    await asyncio.to_thread(store.reflections.reviewed, [item["id"] for item in reflection_tasks + scheduled],
                        time.time() + float(self.config["maintenance_interval_seconds"]), started_at)
                elif result.status == "completed" and scheduled:
                    await asyncio.to_thread(store.reflections.reviewed, [item["id"] for item in scheduled],
                        time.time() + float(self.config["maintenance_interval_seconds"]), started_at)
                if completed:
                    await asyncio.to_thread(store.finish_learning_task, kind)
                else:
                    await asyncio.to_thread(store.update_learning_task, kind,
                        {"continuation": result.continuation, "run_id": trace.id})
                    payload["status"] = "partial" if result.status == "completed" else result.status
                    payload["detail"] = result.detail or "本批尚有材料未完成，已保存进度等待续接"
                payload["written_count"] = len(written)
                await self.index_pending(store, trace=trace)
                return report("completed" if completed else payload["status"], "本批已完成" if completed else payload["detail"],
                              written_count=len(written), message_count=len(messages), remaining_message_count=len(unfinished))
            except asyncio.CancelledError:
                payload.update(status="cancelled", detail="Plugin unloaded during this call; token reservation retained")
                raise
            except Exception as exc:
                payload.update(status="error", detail=str(exc))
                raise
            finally:
                payload["elapsed_ms"] = (time.time() - started_at) * 1000
                await trace.finish(payload)

    async def index_pending(self, store: Store, *, trace=None) -> None:
        if not self.config["embedding_enabled"]:
            return
        docs = await asyncio.to_thread(store.pending_embeddings, self.embedder.model_id, 16)
        if not docs:
            return
        if trace:
            await trace.emit("index", "更新本地语义索引", status="running", count=len(docs), model=self.embedder.model_id,
                             operation_id="index")
        try:
            vectors = await self.embedder.texts([d["text"] for d in docs])
            for doc, vector in zip(docs, vectors, strict=True):
                await asyncio.to_thread(store.save_embedding, doc, self.embedder.model_id, vector)
        except BaseException as exc:
            if trace:
                await trace.emit("index", "索引更新未完成", status="cancelled" if isinstance(exc, asyncio.CancelledError) else "error",
                                 detail=str(exc), count=len(docs), operation_id="index")
            raise
        if trace:
            await trace.emit("index", "已更新本地语义索引", count=len(docs),
                             memories=[{"kind": d["owner_type"], "id": d["owner_key"]} for d in docs], operation_id="index")

    async def maintain(self) -> None:
        while not self.stopping:
            await asyncio.sleep(float(self.config.get("background_interval_seconds", 60)))
            for store in list(self.stores.values()):
                if self.stopping:
                    return
                if store.umo not in self.active_scopes:
                    continue
                allowed = self.config["allowed_umos"]
                if allowed and store.umo not in allowed:
                    continue
                try:
                    await self.index_pending(store)
                    if self.config["feedback_learning_enabled"]:
                        await self.learn(store, feedback=True)
                    if self.config["auto_distillation_enabled"]:
                        await self.consolidate(store)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("MR: background update incomplete; remaining work stays pending")

    @filter.command("mrmem")
    async def memory_status(self, event: AstrMessageEvent):
        if not self.allowed(event):
            return
        store = await self.store_for(event)
        last = dict(self.last_result.get(store.umo, {"status": "尚无本次加载后的调用"}))
        for kind, key in (("background", "private_daily_token_budget"), ("feedback", "feedback_daily_token_budget")):
            last[kind + "_tokens_rolling24h"] = await asyncio.to_thread(store.usage_total, kind)
            last[kind + "_budget"] = self.config[key]
        yield event.plain_result("MR 潜意识：" + json.dumps(last, ensure_ascii=False, default=str))

    @filter.command("mrforget")
    async def forget_me(self, event: AstrMessageEvent):
        if not self.allowed(event):
            return
        store = await self.store_for(event)
        await asyncio.to_thread(store.forget, str(event.message_obj.sender.user_id))
        yield event.plain_result("已删除并停止记录你在本群的个人记忆。")

    async def terminate(self) -> None:
        self.stopping = True
        self.console.close()
        for event in self.observed_events:
            if getattr(event, "_mr_memory_send_observed", None) is self:
                if event.send is event._mr_memory_send_wrapper:
                    event.send = event._mr_memory_original_send
                if hasattr(event, "_mr_memory_send_message_wrapper"):
                    if event.send_message is event._mr_memory_send_message_wrapper:
                        event.send_message = event._mr_memory_original_send_message
                    del event._mr_memory_original_send_message
                    del event._mr_memory_send_message_wrapper
                for name in ("_mr_memory_send_observed", "_mr_memory_original_send", "_mr_memory_send_wrapper"):
                    delattr(event, name)
        self.observed_events.clear()
        for task in list(self.tasks):
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        # Hot reload drains bounded foreground searches without cancelling the
        # AstrBot response task or closing its database underneath a tool call.
        if self.requests:
            await asyncio.gather(*self.requests, return_exceptions=True)
        await self.embedder.close()
        for store in self.stores.values():
            await asyncio.to_thread(store.close)
        self.stores.clear()
