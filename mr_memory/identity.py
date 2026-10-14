"""Local, reversible platform bindings. Raw authors and old memory IDs stay intact."""
from __future__ import annotations

import json
import time
from pathlib import Path


def names(person):
    return {str(person.get(key) or "").strip() for key in
            ("card", "nickname", "name") if str(person.get(key) or "").strip()}


def match_bots(old, current):
    """BOT marks our own recorded replies, not other bots mentioned in the group."""
    old_ids = {str(p["account_id"]) for p in old if p.get("account_type") == "BOT"}
    if len(old_ids) != 1:
        return {}
    legacy = next(iter(old_ids))
    return {str(p["account_id"]): {"legacy_id": legacy, "method": "bot_self", "names": []}
            for p in current if p.get("account_type") == "BOT"}


def match_members(old, current, confirmed=None):
    """Only unique unchanged platform names qualify; collisions remain unresolved."""
    confirmed = confirmed or {}
    old = [p for p in old if p.get("account_type", "USER") == "USER"]
    current = [p for p in current if p.get("account_type", "USER") == "USER"]
    old_names, new_names = {}, {}
    for members, index in ((old, old_names), (current, new_names)):
        for person in members:
            for name in names(person):
                index.setdefault(name, set()).add(str(person["account_id"]))
    result = {}
    for person in current:
        account = str(person["account_id"])
        if confirmed.get(account, {}).get("method") in {"manual", "ignored", "self_reported", "self_ignored"}:
            result[account] = confirmed[account]
            continue
        if confirmed.get(account, {}).get("method") == "unchanged_name":
            if not any(len(new_names.get(name, ())) > 1 for name in confirmed[account].get("names", [])):
                result[account] = confirmed[account]
            continue
        matches = {uid for name in names(person) for uid in old_names.get(name, ())}
        unambiguous = {name for name in names(person)
                       if len(old_names.get(name, ())) == 1 and len(new_names.get(name, ())) == 1}
        if len(matches) == 1 and unambiguous:
            result[account] = {"legacy_id": matches.pop(), "method": "unchanged_name",
                               "names": sorted(unambiguous)}
    # Two new accounts must never inherit the same old identity automatically.
    owners = {}
    for account, value in result.items():
        if value["legacy_id"]:
            owners.setdefault(value["legacy_id"], []).append(account)
    for accounts in owners.values():
        if len(accounts) > 1:
            for account in accounts:
                if result[account]["method"] not in {"manual", "self_reported"}:
                    result.pop(account)
    return result


class IdentityBindings:
    def __init__(self, path: Path):
        self.path = path
        self.groups = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def canonical(self, umo):
        return self.groups.get(umo, {}).get("legacy_umo", umo)

    def aliases(self, canonical):
        return [umo for umo, value in self.groups.items() if value["legacy_umo"] == canonical]

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pending = self.path.with_suffix(".tmp")
        pending.write_text(json.dumps(self.groups, ensure_ascii=False, indent=2), encoding="utf-8")
        pending.replace(self.path)

    def bind(self, current, legacy, old_people):
        if current == legacy or current in self.groups or self.aliases(legacy):
            raise ValueError("该群已经接续；请先查看现有绑定，不能覆盖正在使用的群关系")
        self.groups[current] = {"legacy_umo": legacy, "bound_at": int(time.time()),
                                "old_people": old_people, "members": {}}
        self.save()

    def refresh(self, current, people):
        binding = self.groups.get(current)
        if not binding:
            return False
        members = match_members(binding["old_people"], people, binding["members"])
        bots = {**binding.get("bots", {}), **match_bots(binding["old_people"], people)}
        if members == binding["members"] and bots == binding.get("bots", {}):
            return False
        binding["members"] = members
        binding["bots"] = bots
        self.save()
        return True

    def member(self, current, account, legacy, *, self_reported=False):
        binding = self.groups.get(current)
        if not binding:
            raise ValueError("请先接续这个群的旧记忆库")
        existing = binding["members"].get(account, {})
        if self_reported and existing.get("method") in {"manual", "ignored"}:
            raise ValueError("你的关联已由管理员设置；如需更改，请联系管理员在迁移后台修改。")
        if not legacy:
            binding["members"][account] = {"legacy_id": "", "method": "self_ignored" if self_reported else "ignored", "names": []}
            self.save()
            return
        old_ids = {str(p["account_id"]) for p in binding["old_people"] if p.get("account_type", "USER") == "USER"}
        if legacy not in old_ids:
            raise ValueError("旧账号不属于所选群的历史成员")
        if any(k != account and v["legacy_id"] == legacy and v["method"] in ({"manual", "self_reported"} if self_reported else {"manual"})
               for k, v in binding["members"].items()):
            raise ValueError("该 QQ 号已被其他成员绑定，请联系管理员核对。")
        binding["members"][account] = {"legacy_id": legacy, "method": "self_reported" if self_reported else "manual", "names": []}
        for key, value in list(binding["members"].items()):
            if key != account and value["legacy_id"] == legacy:
                if not self_reported and value["method"] == "self_reported":
                    binding["members"][key] = {"legacy_id": "", "method": "ignored", "names": []}
                else:
                    del binding["members"][key]
        self.save()

    def configure(self, store):
        aliases = self.aliases(store.umo)
        store.identity_links = {account: value["legacy_id"] for alias in aliases
                                for account, value in self.groups[alias]["members"].items() if value["legacy_id"]}
        store.identity_confirmed = {account: value["legacy_id"] for alias in aliases
                                    for account, value in self.groups[alias]["members"].items()
                                    if value["legacy_id"] and value["method"] == "manual"}
        for alias in aliases:
            store.identity_links.update({account: value["legacy_id"]
                                         for account, value in self.groups[alias].get("bots", {}).items() if value["legacy_id"]})
        store.platform_migrated = bool(aliases)


def import_unlearned_messages(source, target):
    """Bring pre-binding live observations along; source remains an accessible archive."""
    if source is None:
        return 0
    with source._lock:
        if source.db.execute("SELECT count(*) FROM mr_memory_objects").fetchone()[0]:
            raise ValueError("新群已有独立整理的记忆，不能覆盖。两库已保留，需合并这些记忆后再接续")
        if source.db.execute("SELECT count(*) FROM mr_learning_tasks").fetchone()[0]:
            raise ValueError("新群有未完成整理任务，请先完成或处理该任务再接续")
        rows = source._rows("SELECT * FROM messages WHERE is_deleted=0 ORDER BY sent_at,id")
        profiles = source.profiles.members()
    for row in rows:
        row.update(umo=target.umo, content=json.loads(row.pop("content_json")))
        # Retain the original source key, platform IDs, author and timestamp.
        reply = next((p.get("message_id") for p in row["content"] if p.get("type") in {"reply", "response_to"}), "")
        row["reply_to"] = f"{row['platform_id']}|{source.umo}|{reply}" if reply else ""
        target.append_message(row)
    for person in profiles:
        if person.get("edited_at"):
            target.edit_member(person["account_id"], {
                "preferred_name": person["preferred_name"], "description": person["description"],
                "aliases": person["confirmed_aliases"], "avoided_names": person["avoided_names"]},
                actor=person["edited_by"])
    return len(rows)
