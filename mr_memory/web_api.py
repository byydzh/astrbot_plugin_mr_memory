from __future__ import annotations

import time

from astrbot.api import logger
from quart import jsonify, request

PLUGIN_NAME = "astrbot_plugin_mr_memory"
WEB_LOG_PREFIX = "[MR Memory][WebUI]"


class WebConsoleMixin:
    """AstrBot Plugin Page API surface for the memory console."""

    def _register_memory_web_apis(self) -> None:
        self._register_memory_web_api(
            "overview",
            self._api_memory_overview,
            ["GET"],
            "MR Memory 运行概览与群范围列表",
        )
        self._register_memory_web_api(
            "scopes/<scope_id>/graph",
            self._api_memory_graph,
            ["GET"],
            "读取指定群范围的记忆图",
        )
        self._register_memory_web_api(
            "scopes/<scope_id>/participants",
            self._api_memory_participants,
            ["GET"],
            "读取指定群范围的账户主体与别名历史",
        )
        self._register_memory_web_api(
            "scopes/<scope_id>/participants/bind_alias",
            self._api_memory_bind_alias,
            ["POST"],
            "绑定管理员确认的账户别名",
        )
        self._register_memory_web_api(
            "scopes/<scope_id>/messages",
            self._api_memory_messages,
            ["GET"],
            "检索指定群范围的原始消息",
        )
        self._register_memory_web_api(
            "scopes/<scope_id>/episodes/<event_id>",
            self._api_memory_episode,
            ["GET"],
            "读取 episode 的关键词与原始证据",
        )
        self._register_memory_web_api(
            "scopes/<scope_id>/distill",
            self._api_memory_distill,
            ["POST"],
            "手动整理指定群范围的最近消息",
        )
        self._register_memory_web_api(
            "scopes/<scope_id>/budget/reset",
            self._api_memory_budget_reset,
            ["POST"],
            "重置指定群范围的在线回忆或反馈额度",
        )

    def _register_memory_web_api(self, route, handler, methods, desc) -> None:
        route_path = f"/{PLUGIN_NAME}/{route.strip('/')}"

        async def logged_handler(*args, **kwargs):
            started_at = time.monotonic()
            try:
                result = await handler(*args, **kwargs)
                elapsed_ms = int((time.monotonic() - started_at) * 1000)
                logger.info(
                    "%s %s %s completed in %d ms",
                    WEB_LOG_PREFIX,
                    request.method,
                    route_path,
                    elapsed_ms,
                )
                return result
            except FileNotFoundError as exc:
                return self._memory_web_error(str(exc), 404)
            except ValueError as exc:
                return self._memory_web_error(str(exc), 400)
            except TimeoutError:
                return self._memory_web_error("记忆整理超时", 504)
            except Exception:
                logger.exception(
                    "%s %s %s failed",
                    WEB_LOG_PREFIX,
                    request.method,
                    route_path,
                )
                return self._memory_web_error("MR Memory 控制台请求失败", 500)

        logged_handler.__name__ = f"mr_memory_web_{handler.__name__}"
        self.context.register_web_api(route_path, logged_handler, methods, desc)

    @staticmethod
    def _memory_web_success(data):
        return jsonify({"status": "success", "data": data})

    @staticmethod
    def _memory_web_error(message: str, status_code: int):
        return (
            jsonify({"status": "error", "message": message, "data": {}}),
            status_code,
        )

    @staticmethod
    def _bounded_int(value, *, default: int, minimum: int, maximum: int) -> int:
        if value in (None, ""):
            return default
        return max(minimum, min(maximum, int(value)))

    async def _api_memory_overview(self):
        return self._memory_web_success(await self._web_memory_overview())

    async def _api_memory_graph(self, scope_id: str):
        limit = self._bounded_int(
            request.args.get("limit"),
            default=200,
            minimum=1,
            maximum=500,
        )
        node_types = tuple(
            item.strip()
            for item in str(request.args.get("types") or "").split(",")
            if item.strip()
        )
        epistemic_states = tuple(
            item.strip()
            for item in str(request.args.get("epistemic") or "").split(",")
            if item.strip()
        )
        data = await self._web_memory_graph(
            scope_id=scope_id,
            limit=limit,
            query=str(request.args.get("query") or "")[:300],
            focus_node_id=str(request.args.get("focus") or "")[:300],
            depth=self._bounded_int(
                request.args.get("depth"), default=1, minimum=1, maximum=3
            ),
            node_types=node_types,
            epistemic_states=epistemic_states,
            relation=str(request.args.get("relation") or "")[:160],
            min_degree=self._bounded_int(
                request.args.get("min_degree"), default=0, minimum=0, maximum=1000
            ),
            min_core=self._bounded_int(
                request.args.get("min_core"), default=0, minimum=0, maximum=100
            ),
            structure_scope=str(request.args.get("structure") or "all"),
            path_source=str(request.args.get("path_source") or "")[:300],
            path_target=str(request.args.get("path_target") or "")[:300],
        )
        return self._memory_web_success(data)

    async def _api_memory_participants(self, scope_id: str):
        limit = self._bounded_int(
            request.args.get("limit"),
            default=200,
            minimum=1,
            maximum=2000,
        )
        data = await self._web_memory_participants(
            scope_id=scope_id,
            reference=str(request.args.get("reference") or ""),
            limit=limit,
        )
        return self._memory_web_success(data)

    async def _api_memory_bind_alias(self, scope_id: str):
        payload = await request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            raise ValueError("请求正文必须是 JSON 对象")
        data = await self._web_memory_bind_alias(
            scope_id=scope_id,
            account_id=str(payload.get("account_id") or ""),
            alias=str(payload.get("alias") or ""),
        )
        return self._memory_web_success(data)

    async def _api_memory_messages(self, scope_id: str):
        limit = self._bounded_int(
            request.args.get("limit"),
            default=60,
            minimum=1,
            maximum=200,
        )
        data = await self._web_memory_messages(
            scope_id=scope_id,
            query=str(request.args.get("query") or ""),
            sender=str(request.args.get("sender") or ""),
            limit=limit,
        )
        return self._memory_web_success(data)

    async def _api_memory_episode(self, scope_id: str, event_id: str):
        data = await self._web_memory_episode(
            scope_id=scope_id,
            event_id=self._bounded_int(
                event_id,
                default=1,
                minimum=1,
                maximum=2**63 - 1,
            ),
        )
        return self._memory_web_success(data)

    async def _api_memory_distill(self, scope_id: str):
        payload = await request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            raise ValueError("请求正文必须是 JSON 对象")
        limit = self._bounded_int(
            payload.get("limit"),
            default=0,
            minimum=0,
            maximum=500,
        )
        processing_class = str(payload.get("processing_class") or "").strip().upper()
        if processing_class not in {"", "LIVE", "BACKFILL"}:
            raise ValueError("processing_class 必须是 LIVE 或 BACKFILL")
        data = await self._web_memory_distill(
            scope_id=scope_id,
            limit=limit,
            processing_class=processing_class,
        )
        return self._memory_web_success(data)

    async def _api_memory_budget_reset(self, scope_id: str):
        payload = await request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            raise ValueError("请求正文必须是 JSON 对象")
        data = await self._web_memory_budget_reset(
            scope_id=scope_id,
            budget_class=str(payload.get("budget_class") or "online"),
        )
        return self._memory_web_success(data)
