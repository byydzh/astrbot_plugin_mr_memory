"""AstrBot's authenticated Plugin Page for inspecting and managing MR memory."""
from __future__ import annotations

import asyncio
import sqlite3
import time
from contextlib import closing
from functools import partial

from astrbot.api import logger
from astrbot.api.web import request

from .store import Store
from .settings import DEFAULTS


class Console:
    def __init__(self, plugin):
        self.plugin = plugin
        self.handlers = []
        routes = (
            ("overview", self.overview, "GET"),
            ("scopes/<scope_id>/overview", self.scope_overview, "GET"),
            ("scopes/<scope_id>/runs", self.runs, "GET"),
            ("scopes/<scope_id>/runs/<run_id>", self.run, "GET"),
            ("scopes/<scope_id>/memories", self.memories, "GET"),
            ("scopes/<scope_id>/memory/<kind>/<item_id>", self.memory, "GET"),
            ("scopes/<scope_id>/graph", self.graph, "GET"),
            ("scopes/<scope_id>/participants", self.participants, "GET"),
            ("scopes/<scope_id>/participants/bind_alias", self.bind_alias, "POST"),
            ("scopes/<scope_id>/messages", self.messages, "GET"),
            ("scopes/<scope_id>/context/<message_id>", self.message_context, "GET"),
            ("scopes/<scope_id>/distill", self.distill, "POST"),
        )
        for path, handler, method in routes:
            wrapped = partial(self.dispatch, handler)
            self.handlers.append(wrapped)
            plugin.context.register_web_api(
                f"/astrbot_plugin_mr_memory/{path}", wrapped, [method], "MR 群聊记忆控制台")

    def close(self):
        # Remove only this instance's closures. A newly loaded instance may
        # already have replaced these routes during a plugin reload.
        apis = self.plugin.context.registered_web_apis
        apis[:] = [api for api in apis if not any(api[1] is fn for fn in self.handlers)]
        self.handlers.clear()

    async def dispatch(self, handler, **kwargs):
        try:
            if self.plugin.stopping:
                raise RuntimeError("MR 正在热重载，请稍后刷新")
            return {"status": "success", "data": await handler(**kwargs)}
        except (ValueError, FileNotFoundError) as exc:
            return {"status": "error", "message": str(exc), "data": {}}
        except Exception as exc:
            logger.exception("MR: console request failed")
            return {"status": "error", "message": f"读取或操作未完成：{exc}", "data": {}}

    def discover(self):
        scopes, errors = [], []
        for path in sorted(self.plugin.scope_dir.glob("*.db")):
            try:
                with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
                    row = db.execute("SELECT umo,platform_id,group_id FROM scope_meta WHERE singleton=1").fetchone()
                if row:
                    scopes.append(dict(id=path.stem, umo=row[0], platform_id=row[1], group_id=row[2]))
            except sqlite3.Error as exc:
                errors.append(f"{path.name}：{exc}")
        return scopes, errors

    async def get_store(self, scope_id):
        scopes, _ = await asyncio.to_thread(self.discover)
        scope = next((s for s in scopes if s["id"] == scope_id), None)
        if scope is None:
            raise FileNotFoundError("找不到这个群的记忆库，请刷新群列表")
        umo = scope["umo"]
        async with self.plugin.open_lock:
            if umo not in self.plugin.stores:
                self.plugin.stores[umo] = await asyncio.to_thread(
                    Store, self.plugin.scope_dir / f"{scope_id}.db", umo)
                self.plugin.state_locks[umo] = asyncio.Lock()
        return self.plugin.stores[umo]

    async def overview(self):
        scopes, errors = await asyncio.to_thread(self.discover)
        config = {**DEFAULTS, **self.plugin.config}
        allowed = config["allowed_umos"]
        for scope in scopes:
            scope["enabled"] = not allowed or scope["umo"] in allowed
        return {"scopes": scopes, "errors": errors, "runtime": {
            "provider": config["local_serving_reader_provider_id"] or config["subconscious_provider_id"],
            "background_provider": config["subconscious_provider_id"],
            "capture": config["capture_enabled"],
            "recall": config["local_serving_enabled"],
            "learning": config["subconscious_enabled"] and config["auto_distillation_enabled"],
            "feedback": config["subconscious_enabled"] and config["feedback_learning_enabled"],
            "embedding_enabled": config["embedding_enabled"],
            "embedding_loaded": getattr(self.plugin.embedder, "_model", None) is not None,
            "memory_timeout_seconds": config["local_serving_timeout_seconds"],
            "background_rolling24h_budget": config["private_daily_token_budget"],
            "feedback_rolling24h_budget": config["feedback_daily_token_budget"],
        }}

    @staticmethod
    def inventory(store):
        with store._lock:
            counts = {}
            for key, table, clause in (
                ("messages", "messages", " AND is_deleted=0"),
                ("episodes", "episodes", " AND status<>'INVALIDATED'"),
                ("semantics", "semantic_memories", " AND status='ACTIVE'"),
                ("associations", "plastic_edges", " AND status IN('ACTIVE','WEAKENED') AND invalidation_reason=''"),
                ("participants", "participants", ""),
                ("embeddings", "memory_embeddings", ""),
            ):
                counts[key] = store.db.execute(f"SELECT count(*) FROM {table} WHERE umo=?{clause}", (store.umo,)).fetchone()[0]
            counts["pending"] = store.db.execute("""SELECT count(*) FROM messages m
                LEFT JOIN message_processing p ON p.message_id=m.id
                WHERE m.umo=? AND m.is_deleted=0 AND (p.status IS NULL OR p.status<>'DISTILLED')""", (store.umo,)).fetchone()[0]
            counts["last_message_at"] = store.db.execute("SELECT max(sent_at) FROM messages WHERE umo=? AND is_deleted=0", (store.umo,)).fetchone()[0]
            return counts

    async def scope_overview(self, scope_id):
        store = await self.get_store(scope_id)
        counts = await asyncio.to_thread(self.inventory, store)
        state = await asyncio.to_thread(store.load_working_state)
        return {"counts": counts, "state": state,
                "background_tokens_rolling24h": await asyncio.to_thread(store.usage_total, "background"),
                "feedback_tokens_rolling24h": await asyncio.to_thread(store.usage_total, "feedback")}

    async def runs(self, scope_id):
        store = await self.get_store(scope_id)
        return {"runs": await asyncio.to_thread(store.recent_runs, limit=30)}

    async def run(self, scope_id, run_id):
        store = await self.get_store(scope_id)
        result = await asyncio.to_thread(store.run_detail, run_id)
        if result is None:
            raise FileNotFoundError("这次调用没有保存详情")
        if result.get("request_id"):
            result["response_events"] = await asyncio.to_thread(store.response_events, result["request_id"])
        return result

    async def memories(self, scope_id):
        store = await self.get_store(scope_id)
        terms = [request.query.get("query")] if request.query.get("query") else []
        kind = request.query.get("kind", "all")
        return {"items": await asyncio.to_thread(store.search_memories, kind=kind, terms=terms, limit=60)}

    async def memory(self, scope_id, kind, item_id):
        store = await self.get_store(scope_id)
        item = await asyncio.to_thread(store.memory, kind, item_id)
        if item is None:
            raise FileNotFoundError("这条记忆不存在或已失效")
        return item

    async def graph(self, scope_id):
        store = await self.get_store(scope_id)
        node = request.query.get("node_id")
        terms = [request.query.get("query")] if request.query.get("query") else []
        edges = await asyncio.to_thread(store.graph, node_id=int(node) if node else None, terms=terms, limit=80)
        nodes = {}
        for edge in edges:
            for side in ("source", "target"):
                node_id = edge[f"{side}_node_id"]
                nodes[node_id] = {"id": node_id, "label": edge[side]}
        return {"nodes": list(nodes.values()), "edges": edges, "limit": 80}

    async def participants(self, scope_id):
        store = await self.get_store(scope_id)
        query = request.query.get("query")
        people = await asyncio.to_thread(store.members, name=query or None)
        if query and str(query).isdigit():
            by_id = await asyncio.to_thread(store.members, account_ids=[query])
            existing = {p["account_id"] for p in people}
            people.extend(p for p in by_id if p["account_id"] not in existing)
        return {"participants": people}

    @staticmethod
    def save_alias(store, account, alias):
        if not account or not alias:
            raise ValueError("请填写账户 ID 和别名")
        with store._lock, store.db:
            person = store.db.execute("SELECT id FROM participants WHERE umo=? AND account_id=?", (store.umo, account)).fetchone()
            if person is None:
                raise ValueError("该账户还没有历史消息记录，请先从下方列表选择已记录的账户")
            now = int(time.time())
            store.db.execute("""INSERT INTO participant_aliases
                (participant_id,alias,normalized_alias,first_seen_at,last_seen_at,source_kind)
                VALUES(?,?,?,?,?,'admin_confirmed') ON CONFLICT(participant_id,normalized_alias)
                DO UPDATE SET alias=excluded.alias,is_active=1,source_kind='admin_confirmed',updated_at=CURRENT_TIMESTAMP""",
                (person[0], alias, alias.casefold().strip(), now, now))
        return {"message": "别名已保存到这个账户，后续成员检索可以读到"}

    async def bind_alias(self, scope_id):
        store = await self.get_store(scope_id)
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("请求内容应为账户与别名")
        return await asyncio.to_thread(self.save_alias, store, str(body.get("account_id", "")).strip(), str(body.get("alias", "")).strip())

    @staticmethod
    def read_messages(store, query, before_id):
        with store._lock:
            clauses, args = ["umo=?", "is_deleted=0"], [store.umo]
            if query:
                clauses.append("(instr(lower(plain_text),lower(?))>0 OR instr(lower(content_json),lower(?))>0)")
                args.extend([query, query])
            if before_id:
                anchor = store.db.execute("SELECT sent_at,id FROM messages WHERE umo=? AND id=?", (store.umo, int(before_id))).fetchone()
                if anchor is None:
                    raise ValueError("找不到翻页起点")
                clauses.append("(sent_at,id)<(?,?)")
                args.extend(anchor)
            rows = store._rows(f"SELECT * FROM messages WHERE {' AND '.join(clauses)} ORDER BY sent_at DESC,id DESC LIMIT 101", args)
            return {"messages": [store._message(row) for row in rows[:100]],
                    "next_before_id": rows[99]["id"] if len(rows) > 100 else None}

    async def messages(self, scope_id):
        store = await self.get_store(scope_id)
        return await asyncio.to_thread(self.read_messages, store, request.query.get("query", ""), request.query.get("before_id"))

    async def message_context(self, scope_id, message_id):
        store = await self.get_store(scope_id)
        return {"messages": await asyncio.to_thread(store.context, message_id=int(message_id), before=8, after=8)}

    async def distill(self, scope_id):
        store = await self.get_store(scope_id)
        return await self.plugin.consolidate(store, force=True)


def register_console(plugin):
    return Console(plugin)
