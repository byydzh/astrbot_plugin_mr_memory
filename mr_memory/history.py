"""Archive missed OneBot group messages without replaying AstrBot events."""
from __future__ import annotations

import asyncio
import logging
import time


logger = logging.getLogger(__name__)
CONNECT_EVENT = "meta_event.lifecycle.connect"


def history_message(raw: dict, umo: str, self_id: str) -> dict:
    platform_id, _, group_id = umo.split(":", 2)
    if str(raw.get("group_id", group_id)) != group_id:
        raise ValueError("History response belongs to another group")
    sender = raw.get("sender") or {}
    account = str(sender.get("user_id") or raw.get("user_id") or "")
    message_id = str(raw.get("message_id") or "")
    if not account or not message_id or not raw.get("time"):
        raise ValueError("History message lacks its author, ID or timestamp")
    parts = raw.get("message")
    if isinstance(parts, str):
        from aiocqhttp import Message
        parts = list(Message(parts))
    if not isinstance(parts, list):
        raise ValueError("History message has no readable message components")
    content, plain = [], []
    for part in parts:
        kind, data = part["type"], part.get("data", {})
        if kind == "text":
            content.append({"type": "text", "text": data.get("text", "")})
            plain.append(str(data.get("text", "")))
        elif kind == "at":
            content.append({"type": "mention", "account_id": str(data.get("qq", "")),
                            "display_name": str(data.get("name") or "")})
            plain.append("@" + str(data.get("name") or data.get("qq", "")))
        elif kind == "reply":
            content.append({"type": "reply", "message_id": str(data.get("id", ""))})
            plain.append("[引用消息]")
        else:
            # Retain media/forward references without downloading or captioning.
            content.append({**data, "type": kind})
            plain.append(f"[{kind}]")
    return {"umo": umo, "platform": "aiocqhttp", "platform_id": platform_id,
            "group_id": group_id, "message_id": message_id, "sender_id": account,
            "sender_name": str(sender.get("card") or sender.get("nickname") or account),
            "platform_nickname": sender.get("nickname"), "group_card": sender.get("card"),
            "sent_at": int(raw["time"]), "plain_text": " ".join(plain), "content": content,
            "role": "BOT" if account == str(self_id) else "USER",
            "reply_to": next((part["message_id"] for part in content if part["type"] == "reply"), "")}


class HistoryRecovery:
    def __init__(self, plugin):
        self.plugin = plugin
        self.clients = {}
        self.requested = {}
        self.rewind = set()
        self.wake = asyncio.Event()
        self.closed = False

    def enabled(self, umo):
        config = self.plugin.config
        return (config["capture_enabled"] and config.get("history_recovery_enabled", True)
                and (not config["allowed_umos"] or umo in config["allowed_umos"]))

    def attach(self, platform_id):
        if platform_id in self.clients or self.closed:
            return
        adapter = self.plugin.context.get_platform_inst(platform_id)
        if adapter is None or adapter.meta().name != "aiocqhttp":
            return
        bot = adapter.get_client()

        async def connected(event):
            for store in list(self.plugin.stores.values()):
                if store.umo.split(":", 1)[0] == platform_id:
                    self.rewind.add(store.umo)
                    self.request(store)

        bot.subscribe(CONNECT_EVENT, connected)
        self.clients[platform_id] = (bot, connected)

    def request(self, store, since=None):
        if not self.enabled(store.umo) or self.closed:
            return False
        platform_id = store.umo.split(":", 1)[0]
        self.attach(platform_id)
        if platform_id not in self.clients:
            return False
        prior = self.requested.get(store.umo)
        if prior is not None:
            since = min(prior, since) if since is not None else prior
        self.requested[store.umo] = since
        self.wake.set()
        return True

    def start(self):
        for store in list(self.plugin.stores.values()):
            self.rewind.add(store.umo)
            self.request(store)
        self.plugin.spawn(self.maintain())

    def close(self):
        self.closed = True
        for bot, handler in self.clients.values():
            bot.unsubscribe(CONNECT_EVENT, handler)
        self.clients.clear()
        self.wake.set()

    async def maintain(self):
        while not self.closed:
            try:
                await asyncio.wait_for(self.wake.wait(), timeout=300)
            except TimeoutError:
                # A separate durable cursor also covers process downtime and
                # event losses without a reconnect notification.
                for store in list(self.plugin.stores.values()):
                    self.request(store)
            self.wake.clear()
            pending, self.requested = self.requested, {}
            for umo, since in pending.items():
                if self.closed:
                    return
                store = self.plugin.stores.get(umo)
                client = self.clients.get(umo.split(":", 1)[0])
                if store is not None and client and self.enabled(umo):
                    try:
                        rewind = umo in self.rewind
                        self.rewind.discard(umo)
                        await self.recover(store, client[0], since=since, rewind=rewind)
                    except Exception:
                        logger.exception("MR history recovery %s could not read or save progress", umo)

    async def recover(self, store, bot, *, since=None, rewind=False):
        state = await asyncio.to_thread(store.history_state)
        pending = state.get("pending")
        watermark = max(state.get("watermark", 0), (state.get("last_result") or {}).get("since", 0))
        floor = float(since) if since is not None else float(watermark or
            (time.time() - float(self.plugin.config.get("history_recovery_initial_hours", 24)) * 3600)) - 1
        if not pending or floor < pending["since"]:
            pending = {"since": floor, "cursor": "0", "head_at": 0, "pages": 0,
                       "inserted": 0, "existing": 0, "ignored": 0}
        elif rewind:
            # NapCat maps short message IDs to native IDs in process memory.
            # Rebuild pagination after reconnect, retaining the unfilled range.
            pending["cursor"] = "0"
        state.update(status="running", pending=pending, error="", updated_at=time.time())

        async def save():
            state["updated_at"] = time.time()
            await asyncio.to_thread(store.save_history_state, state)

        await save()
        try:
            async with asyncio.timeout(30):
                login = await bot.get_login_info()
            self_id = str(login["user_id"])
            group_id = store.umo.split(":", 2)[2]
            while not self.closed:
                async with asyncio.timeout(30):
                    result = await bot.get_group_msg_history(group_id=int(group_id), self_id=int(self_id),
                        message_seq=pending["cursor"], count=100, reverseOrder=True)
                raw = result.get("messages") if isinstance(result, dict) else None
                if not isinstance(raw, list):
                    raise ValueError("NapCat history response has no messages list")
                if not raw:
                    raise ValueError("NapCat did not return history; coverage is unverified")
                rows = sorted((history_message(item, store.umo, self_id) for item in raw), key=lambda row: row["sent_at"])
                pending["head_at"] = max(pending["head_at"], max(row["sent_at"] for row in rows))
                counts = await asyncio.to_thread(store.append_history, [row for row in rows if row["sent_at"] >= pending["since"]])
                for key in counts:
                    pending[key] += counts[key]
                state["total_inserted"] = state.get("total_inserted", 0) + counts["inserted"]
                pending["pages"] += 1
                oldest = rows[0]
                covered = oldest["sent_at"] < pending["since"]
                stalled = oldest["message_id"] == pending["cursor"]
                if covered or stalled:
                    state.update(status="completed" if covered else "partial",
                        watermark=max(watermark, pending["head_at"], pending["since"]),
                        last_result=dict(pending), pending=None, checked_at=time.time())
                    if stalled and not covered:
                        state["unavailable"] = {"since": pending["since"], "until": oldest["sent_at"],
                            "detail": "平台不再返回更早记录，这段范围无法确认补齐"}
                    elif state.get("unavailable") and pending["since"] <= state["unavailable"]["since"]:
                        state.pop("unavailable")
                    await save()
                    if pending["inserted"] or state["status"] != "completed":
                        logger.info("MR history recovery %s: %s, added=%s pages=%s", store.umo,
                                    state["status"], pending["inserted"], pending["pages"])
                    return state
                pending["cursor"] = oldest["message_id"]
                await save()
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            state["status"] = "cancelled"
            await save()
            raise
        except Exception as exc:
            state.update(status="error", error=f"{type(exc).__name__}: {exc}")
            await save()
            logger.warning("MR history recovery %s incomplete: %s", store.umo, state["error"])
        return state
