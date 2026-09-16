"""Manual recovery of AstrBot's main conversation without erasing its history."""
from __future__ import annotations

import asyncio
from datetime import datetime
import json


class ConversationRepair:
    def __init__(self, context):
        self.context = context
        self._locks: dict[str, asyncio.Lock] = {}

    async def reset_main_conversation(self, umo: str) -> dict:
        """Select a fresh native conversation; leave the old conversation intact.

        AstrBot saves a running request to its captured conversation ID. Switching
        the selected conversation therefore also keeps that request's eventual
        history out of the fresh conversation. No provider or MR data is touched.
        """
        async with self._locks.setdefault(umo, asyncio.Lock()):
            manager = self.context.conversation_manager
            old_id = await manager.get_curr_conversation_id(umo)
            old = await manager.get_conversation(umo, old_id) if old_id else None
            result = {
                "old_conversation_id": old_id,
                "new_conversation_id": None,
                "preserved_messages": 0,
                "history_title": old.title if old else None,
            }
            if old is None:
                return {**result, "status": "no_conversation",
                        "message": "当前没有主会话，无需清理；未作更改。"}

            history = json.loads(old.history or "[]")
            if not history:
                return {**result, "status": "already_empty",
                        "message": "当前主会话已经是空的，无需再次清理；未作更改。"}

            new_id = await manager.new_conversation(
                umo,
                platform_id=old.platform_id,
                content=[],
                title=f"手动清理后 · {datetime.now().astimezone():%Y-%m-%d %H:%M:%S}",
                persona_id=old.persona_id,
            )
            return {
                **result,
                "status": "cleared",
                "new_conversation_id": new_id,
                "preserved_messages": len(history),
                "message": (f"已保留原主会话的 {len(history)} 条记录，并切换到空的新会话，"
                            "用于后续请求。MR 群聊记忆与模型设置未修改；"
                            "已经开始的回复仍可能继续完成。"),
            }
