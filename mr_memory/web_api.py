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
        data = await self._web_memory_graph(scope_id=scope_id, limit=limit)
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
        data = await self._web_memory_distill(scope_id=scope_id, limit=limit)
        return self._memory_web_success(data)
