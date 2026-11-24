"""AstrBot's authenticated Plugin Page for inspecting and managing MR memory."""
from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import time
from contextlib import closing
from functools import partial

from astrbot.api import logger
from astrbot.api.web import request

from .store import Store
from . import manual_messages
from .settings import DEFAULTS
from .schedule import work_window
from .conversation import ConversationRepair
from .identity import match_members, match_bots
from .network import console_view


class Console:
    def __init__(self, plugin):
        self.plugin = plugin
        self.handlers = []
        self.conversation_repair = ConversationRepair(plugin.context)
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
            ("scopes/<scope_id>/participants/profile", self.edit_profile, "POST"),
            ("scopes/<scope_id>/messages", self.messages, "GET"),
            ("scopes/<scope_id>/manual-messages", self.submit_manual_messages, "POST"),
            ("scopes/<scope_id>/manual-messages/<request_id>", self.manual_receipt, "GET"),
            ("scopes/<scope_id>/context/<message_id>", self.message_context, "GET"),
            ("scopes/<scope_id>/distill", self.distill, "POST"),
            ("scopes/<scope_id>/conversation/reset", self.reset_conversation, "POST"),
            ("scopes/<scope_id>/history/recover", self.recover_history, "POST"),
            ("scopes/<scope_id>/migration", self.migration, "GET"),
            ("scopes/<scope_id>/migration/bind", self.migration_bind, "POST"),
            ("scopes/<scope_id>/migration/member", self.migration_member, "POST"),
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

    def discover(self, *, with_activity=True):
        scopes, errors = [], []
        activity = {}
        week_ago = int(time.time()) - 7 * 86400
        for path in sorted(self.plugin.scope_dir.glob("*.db")):
            try:
                with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
                    row = db.execute("SELECT umo,platform_id,group_id FROM scope_meta WHERE singleton=1").fetchone()
                    if row:
                        scope = dict(id=path.stem, umo=row[0], platform_id=row[1], group_id=row[2])
                        scopes.append(scope)
                        if with_activity and self.plugin.identities.canonical(row[0]) == row[0]:
                            # After a binding the shared store retains the original receiving platform and group.
                            activity[row[0]] = {
                                (platform, group): (last, total, recent)
                                for platform, group, last, total, recent in db.execute(
                                    """SELECT platform_id,group_id,max(sent_at),count(*),sum(sent_at>=?)
                                       FROM messages WHERE umo=? AND is_deleted=0 AND role IN ('USER','BOT')
                                       GROUP BY platform_id,group_id""", (week_ago, row[0]))
                            }
            except sqlite3.Error as exc:
                errors.append(f"{path.name}：{exc}")
        if with_activity:
            for scope in scopes:
                origins = activity.get(self.plugin.identities.canonical(scope["umo"]))
                counts = (None, None, None) if origins is None else origins.get(
                    (scope["platform_id"], scope["group_id"]), (None, 0, 0))
                scope["last_message_at"], scope["message_count"], scope["recent_message_count"] = counts
        return scopes, errors

    async def get_store(self, scope_id, *, raw=False):
        scopes, _ = await asyncio.to_thread(self.discover, with_activity=False)
        scope = next((s for s in scopes if s["id"] == scope_id), None)
        if scope is None:
            raise FileNotFoundError("找不到这个群的记忆库，请刷新群列表")
        umo = scope["umo"] if raw else self.plugin.identities.canonical(scope["umo"])
        async with self.plugin.open_lock:
            if umo not in self.plugin.stores:
                self.plugin.stores[umo] = await asyncio.to_thread(
                    Store, self.plugin.scope_dir / (hashlib.sha256(umo.encode()).hexdigest() + ".db"), umo)
                self.plugin.identities.configure(self.plugin.stores[umo])
                self.plugin.state_locks[umo] = asyncio.Lock()
        return self.plugin.stores[umo]

    async def overview(self):
        scopes, errors = await asyncio.to_thread(self.discover)
        config = {**DEFAULTS, **self.plugin.config}
        allowed = {self.plugin.identities.canonical(umo) for umo in config["allowed_umos"]}
        for scope in scopes:
            scope["enabled"] = not allowed or scope["umo"] in allowed or self.plugin.identities.canonical(scope["umo"]) in allowed
        return {"scopes": scopes, "errors": errors, "runtime": {
            "provider": config["local_serving_reader_provider_id"] or config["subconscious_provider_id"],
            "background_provider": config["subconscious_provider_id"],
            "capture": config["capture_enabled"],
            "history_recovery": config["capture_enabled"] and config["history_recovery_enabled"],
            "recall": config["local_serving_enabled"],
            "advanced": False,
            "member_prefix_capacity": config["member_prefix_capacity"],
            "learning": config["subconscious_enabled"] and config["auto_distillation_enabled"],
            "background_enabled": config["subconscious_enabled"],
            "feedback": False,
            "embedding_enabled": config["embedding_enabled"],
            "embedding_loaded": (self.plugin.embedder.loaded if hasattr(self.plugin.embedder, "loaded")
                                 else getattr(self.plugin.embedder, "_model", None) is not None),
            "memory_timeout_seconds": config["local_serving_timeout_seconds"],
            "background_rolling24h_budget": config["private_daily_token_budget"],
            "feedback_rolling24h_budget": config["feedback_daily_token_budget"],
            "learning_window": self.learning_window(),
        }}

    def learning_window(self):
        config = {**DEFAULTS, **self.plugin.config}
        window = work_window(config)
        window["enabled"] = config["learning_window_enabled"]
        window["next_start_local"] = self.local_date(window["next_start"])
        return window

    @staticmethod
    def local_date(value):
        return time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(float(value))) if value is not None else None

    @staticmethod
    def learning_tasks(store):
        # The persisted conversation can be large. Select only the small
        # reader-facing state rather than loading or sending that transcript.
        with store._lock:
            rows = store._rows("""SELECT kind,json_array_length(task_json,'$.material_ids') material_count,
                json_array_length(task_json,'$.completed_ids') completed_count,
                json_extract(task_json,'$.checkpoint') checkpoint,
                json_extract(task_json,'$.memory_refs') memory_refs,
                json_extract(task_json,'$.run_id') run_id,
                (SELECT count(*) FROM json_each(task_json,'$.continuation.write_state.pending_items')) pending_draft_count
                FROM mr_learning_tasks WHERE umo=? ORDER BY kind""", (store.umo,))
        for row in rows:
            row["memory_refs"] = json.loads(row["memory_refs"] or "[]")
        return rows

    async def learning_overview(self, store):
        statuses = {}
        for kind, value in getattr(self.plugin, "learning_status", {}).get(store.umo, {}).items():
            statuses[kind] = {key: value[key] for key in ("status", "reason", "next_start", "local_now", "timezone",
                             "remaining_message_count", "used_tokens") if key in value}
            statuses[kind]["next_start_local"] = self.local_date(value.get("next_start"))
        working = await asyncio.to_thread(store.load_working_state)
        return {"learning": await asyncio.to_thread(self.learning_tasks, store),
                "pending_learning_drafts": {kind: len(rows) for kind, rows in working.get("pending_learning_drafts", {}).items()},
                "learning_status": statuses, "learning_window": self.learning_window()}

    @staticmethod
    def inventory(store):
        with store._lock:
            counts = {}
            for key, table, clause in (
                ("messages", "messages", " AND is_deleted=0"),
                ("episodes", "mr_memory_objects", " AND kind='episode' AND status='ACTIVE'"),
                ("semantics", "mr_memory_objects", " AND kind='semantic' AND status='ACTIVE'"),
                ("associations", "mr_memory_objects", " AND kind='association' AND status='ACTIVE'"),
                ("patterns", "mr_memory_objects", " AND kind NOT IN('episode','semantic','association','node') AND status='ACTIVE'"),
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
        if store.network_enabled:
            def network_counts():
                db=store.db
                kinds=dict(db.execute("SELECT kind,count(*) FROM node WHERE region='M' AND status='ACTIVE' GROUP BY kind"))
                semantic_edges=db.execute("""SELECT count(*) FROM edge WHERE active=1 AND
                   rel NOT IN ('adjacent','source','REPLY_TO','same_l1','witness','voice','received','reply_to','reply_back')
                   AND src NOT LIKE 'E|message:%' AND dst NOT LIKE 'E|message:%'""").fetchone()[0]
                return {"engine":"plan_d","regions":dict(db.execute("SELECT region,count(*) FROM node GROUP BY region")),
                        "edges":store.db.execute("SELECT count(*) FROM edge").fetchone()[0],
                        "episodes":kinds.get("episode",0)+kinds.get("event",0),"semantics":kinds.get("semantic",0),
                        "semantic_edges":semantic_edges,
                        "seed_watermark":store.db.execute("SELECT watermark FROM network_scope WHERE umo=?",(store.umo,)).fetchone()[0]}
            counts["network"] = await store.read_async(network_counts)
            counts["legacy_counts"]={key:counts[key] for key in ("episodes","semantics","associations")}
            counts.update(episodes=counts["network"]["episodes"],semantics=counts["network"]["semantics"],
                          associations=counts["network"]["semantic_edges"])
        state = await asyncio.to_thread(store.load_working_state)
        history = await asyncio.to_thread(store.history_state)
        adapter = self.plugin.context.get_platform_inst(store.umo.split(":", 1)[0])
        if getattr(store, "platform_migrated", False) or (adapter and adapter.meta().name in {"qq_official", "qq_official_webhook"}):
            history = {**history, "status": "unsupported", "last_error": "官方 QQ 没有历史补拉接口；在线普通消息持续记录，旧库已接续的历史仍可检索。"}
        return {"counts": counts, "state": state, "history": history,
                "workspace": await store.read_async(store.cognition.archived_workspace), **await self.learning_overview(store),
                "background_tokens_rolling24h": await asyncio.to_thread(store.usage_total, "background"),
                "feedback_tokens_rolling24h": await asyncio.to_thread(store.usage_total, "feedback")}

    async def runs(self, scope_id):
        store = await self.get_store(scope_id)
        return {"runs": await asyncio.to_thread(store.recent_runs, limit=30), "workspace": await store.read_async(store.cognition.archived_workspace),
                **await self.learning_overview(store)}

    async def recover_history(self, scope_id):
        store = await self.get_store(scope_id)
        hours = float(self.plugin.config["history_recovery_initial_hours"])
        if hours <= 0:
            raise ValueError("补查历史时长必须大于零")
        if not self.plugin.history.request(store, since=time.time() - hours * 3600):
            raise ValueError("请启用消息记录和历史回补，并使用已配置的 OneBot 群会话")
        return {"queued": True, "message": f"已安排补查近 {hours:g} 小时历史；结果会显示在历史回补进度中"}

    async def run(self, scope_id, run_id):
        store = await self.get_store(scope_id)
        result = await asyncio.to_thread(store.run_detail, run_id)
        if result is None:
            raise FileNotFoundError("这次调用没有保存详情")
        if result.get("request_id"):
            result["response_events"] = await asyncio.to_thread(store.response_events, result["request_id"])
        for key,value in (await asyncio.to_thread(manual_messages.changes, store, run_id)).items():
            if value or key not in result:result[key]=value
        return result

    async def memories(self, scope_id):
        store = await self.get_store(scope_id)
        args = request.query
        query = args.get("query", "")
        kind = args.get("kind", "all")
        region = console_view.region_filter(args.get("region", "all"))
        offset = console_view.integer_param("offset", args.get("offset", 0), 0)
        limit = console_view.integer_param("limit", args.get("limit", 60), 1, 200)
        legacy_layout = not any(k in args for k in ("region", "offset", "limit"))

        def page():
            total = console_view.item_count(store, kind, query, region) if store.network_enabled else 0
            rows = (console_view.items(store, kind, query, region=region, offset=offset, limit=limit)
                    if offset < total else [])
            more = offset + len(rows) < total
            if region == "all" and (legacy_layout or not more):
                # Old clients received two lists of up to 60. V3 pages share a single limit
                # across the same ordered catalog: first graph addresses, then retained records.
                room = limit if legacy_layout else limit - len(rows)
                old = store.search_memories(kind=kind, terms=[query] if query else [],
                                            limit=room + 1, offset=max(0, offset - total))
                rows.extend(old[:room])
                more = more or len(old) > room
            return {"items": rows, "has_more": more}

        return await store.read_async(page)

    async def memory(self, scope_id, kind, item_id):
        store = await self.get_store(scope_id)
        if store.network_enabled and ("|" in item_id or item_id.startswith("graph~")):
            from .network.console_view import item
            view = await store.read_async(item,store,item_id)
            if view is None:raise FileNotFoundError("这条记忆不存在或已失效")
            return view
        item = await asyncio.to_thread(store.memory, kind, item_id, include_history=True)
        if item is None:
            raise FileNotFoundError("这条记忆不存在或已失效")
        item["navigation"] = await asyncio.to_thread(store.navigate, ref={"kind": kind, "id": int(item_id)})
        item["experiences"] = await store.read_async(store.cognition.archived_experiences, ref={"kind": kind, "id": int(item_id)}, limit=5)
        return item

    async def graph(self, scope_id):
        store = await self.get_store(scope_id)
        node = request.query.get("node_id")
        terms = [request.query.get("query")] if request.query.get("query") else []
        addr = request.query.get("id", "")
        limit = console_view.integer_param("limit", request.query.get("limit", 80), 1, 400)
        region = console_view.region_filter(request.query.get("region", "all"))
        kind = request.query.get("kind", "all")
        view = request.query.get("view", "recent")
        if view not in ("recent", "backbone"):
            raise ValueError("view 须为 recent 或 backbone")
        if store.network_enabled and not node and (not addr or "|" in addr):
            return await store.read_async(console_view.graph, store, terms[0] if terms else "", addr or None,
                                          region=region, kind=kind, view=view, limit=limit)
        if region != "all":
            return {"nodes": [], "edges": [], "limit": limit}
        ref = {"kind": kind, "id": int(addr)} if kind != "all" and addr else None
        edges = await store.read_async(store.graph, node_id=int(node) if node else None, ref=ref, terms=terms,
                                       limit=limit, endpoint_kind=kind if not ref else "all", view=view)
        nodes = {}
        for edge in edges:
            for side in ("source", "target"):
                address = edge[f"{side}_ref"]
                key = f"{address['kind']}:{address['id']}"
                nodes[key] = {"key": key, **address, "label": edge[side]}
                edge[f"{side}_key"] = key
        return {"nodes": list(nodes.values()), "edges": edges, "limit": limit}

    async def participants(self, scope_id):
        store = await self.get_store(scope_id)
        query = request.query.get("query")
        people = await asyncio.to_thread(store.members, name=query or None)
        if query and str(query).isdigit():
            by_id = await asyncio.to_thread(store.members, account_ids=[query])
            existing = {p["account_id"] for p in people}
            people.extend(p for p in by_id if p["account_id"] not in existing)
        pool = await asyncio.to_thread(store.member_pool, self.plugin.config["member_prefix_capacity"])
        return {"participants": people, "pool": pool}

    async def edit_profile(self, scope_id):
        store = await self.get_store(scope_id)
        body = await request.json()
        if not isinstance(body, dict) or not isinstance(body.get("changes"), dict):
            raise ValueError("请提供账户 UID 和资料字段")
        person = await asyncio.to_thread(store.edit_member, str(body.get("account_id", "")), body["changes"], actor="admin")
        return {"participant": person, "message": "已保存该群员资料，后续检索与前缀立即使用新资料"}

    @staticmethod
    def save_alias(store, account, alias):
        if not account or not alias:
            raise ValueError("请填写账户 ID 和别名")
        with store._lock:
            people = store.members([account])
            if not people:
                raise ValueError("找不到该群员")
            store.edit_member(account, {"aliases": [*people[0]["confirmed_aliases"], alias]}, actor="admin")
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

    async def submit_manual_messages(self, scope_id):
        store = await self.get_store(scope_id)
        allowed = {self.plugin.identities.canonical(umo) for umo in self.plugin.config["allowed_umos"]}
        if allowed and store.umo not in allowed:
            raise ValueError("此群未启用记忆")
        key, created = await asyncio.to_thread(manual_messages.submit, store, await request.json())
        result = await self.manual_receipt(scope_id, key)
        if created and result["trigger"] == "now" and self.plugin.config["subconscious_enabled"]:
            self.plugin.spawn(self.wake_manual(store, key))
        return result

    async def wake_manual(self, store, key):
        while not self.plugin.stopping:
            lock = self.plugin.learning_locks.setdefault(store.umo, asyncio.Lock())
            async with lock:
                pass
            previous = await asyncio.to_thread(manual_messages.receipt, store, key)
            if previous["state"] == "completed":
                return
            result = await self.plugin.learn(store, force=True)
            if result.get("status") == "busy":
                continue
            current = await asyncio.to_thread(manual_messages.receipt, store, key)
            if current["state"] == "completed" or (current["messages"] == previous["messages"] and
                not (store.network_enabled and result.get("status") == "completed")):
                return

    async def manual_receipt(self, scope_id, request_id):
        store = await self.get_store(scope_id)
        result = await asyncio.to_thread(manual_messages.receipt, store, request_id)
        status = self.plugin.learning_status.get(store.umo, {}).get("background", {}).get("status")
        lock = self.plugin.learning_locks.get(store.umo)
        if result["state"] in ("queued", "partial"):
            if not self.plugin.config["subconscious_enabled"] or (result["trigger"] == "next_window" and not self.plugin.config["auto_distillation_enabled"]):
                result["reason"] = "disabled"
            elif status in ("budget_exhausted", "budget_wait"):
                result["reason"] = "budget_exhausted"
            elif lock and lock.locked():
                result["reason"] = "busy"
            elif result["trigger"] == "next_window" and not work_window(self.plugin.config)["open"]:
                result["reason"] = "waiting_window"
        result.setdefault("reason", "")
        return result

    async def reset_conversation(self, scope_id):
        store = await self.get_store(scope_id)
        aliases = self.plugin.identities.aliases(store.umo)
        return await self.conversation_repair.reset_main_conversation(aliases[0] if aliases else store.umo)

    async def migration(self, scope_id):
        current = await self.get_store(scope_id, raw=True)
        scopes, _ = await asyncio.to_thread(self.discover, with_activity=False)
        bindings = self.plugin.identities
        binding = bindings.groups.get(current.umo)
        old_scope = next((s for s in scopes if s["umo"] == binding["legacy_umo"]), None) if binding else None
        selected = request.query.get("legacy_scope") or (old_scope or {}).get("id")
        old = await self.get_store(selected) if selected else None
        view = old if binding else current
        def people():
            with view._lock:
                platform = current.umo.split(":", 1)[0]
                ids = [r[0] for r in view.db.execute("SELECT account_id FROM participants WHERE platform_id=?", (platform,))]
                return view.profiles.members(ids)
        all_observed = await asyncio.to_thread(people)
        if binding:
            await self.plugin.refresh_identities(current.umo, view)
        observed = [p for p in all_observed if p["account_type"] == "USER"]
        bots = [p for p in all_observed if p["account_type"] == "BOT"]
        old_people = binding["old_people"] if binding else (await asyncio.to_thread(old.profiles.members) if old else [])
        matches = binding["members"] if binding else match_members(old_people, observed)
        with current._lock:
            official = bool(current.db.execute("SELECT 1 FROM messages WHERE platform IN ('qq_official','qq_official_webhook') LIMIT 1").fetchone())
        return {"official": official, "current_umo": current.umo, "bound": bool(binding),
                "legacy_scope": (old_scope or {}).get("id"),
                "choices": [s for s in scopes if s["group_id"].isdigit() and s["umo"] != current.umo],
                "old_people": old_people, "observed_people": observed, "matches": matches,
                "observed_bots": bots, "bot_matches": binding.get("bots", {}) if binding else match_bots(old_people, bots),
                "old_counts": await asyncio.to_thread(self.inventory, old) if old else None,
                "receiving_messages": len(observed) > 0,
                "message": "已接续旧库，后续成员发言会继续静默对照昵称。" if binding else "选择对应旧群，预览名称对照后接续。"}

    async def migration_bind(self, scope_id):
        data = await request.json()
        current = await self.get_store(scope_id, raw=True)
        legacy = await self.get_store(str(data.get("legacy_scope") or ""))
        with current._lock:
            official = current.db.execute("SELECT 1 FROM messages WHERE platform IN ('qq_official','qq_official_webhook') LIMIT 1").fetchone()
        if not official:
            raise ValueError("请在群选择器中选择新官方机器人已开始记录的群")
        result = await self.plugin.bind_scope(current, legacy)
        return {**result, "legacy_scope": str(data["legacy_scope"])}

    async def migration_member(self, scope_id):
        data = await request.json()
        current = await self.get_store(scope_id, raw=True)
        bindings = self.plugin.identities
        binding = bindings.groups.get(current.umo)
        if not binding:
            raise ValueError("请先绑定群，再处理个别成员")
        canonical = await self.get_store(hashlib.sha256(binding["legacy_umo"].encode()).hexdigest())
        account, legacy = str(data.get("account_id") or ""), str(data.get("legacy_id") or "")
        async with self.plugin.identity_lock:
            with canonical._lock:
                if not canonical.db.execute("SELECT 1 FROM participants WHERE platform_id=? AND account_id=? AND account_type=?",
                        (current.umo.split(":", 1)[0], account, "USER")).fetchone():
                    raise ValueError("该类型的账号尚未在当前群被观察到")
            bindings.member(current.umo, account, legacy)
            bindings.configure(canonical)
        return {"message": "成员关联已由管理员更新。"}


def register_console(plugin):
    return Console(plugin)
