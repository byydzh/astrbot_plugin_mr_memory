"""Platform identities, editable member information and a stable daily prefix.

UIDs come from the adapter. A nickname never merges two accounts. The prefix
is a view of this directory, not its storage limit. Call under Store's lock.
"""
from __future__ import annotations

import json
import shlex
import time


SCHEMA = """
CREATE TABLE IF NOT EXISTS mr_member_profiles (
 umo TEXT NOT NULL,account_id TEXT NOT NULL,nickname TEXT NOT NULL DEFAULT '',
 card TEXT NOT NULL DEFAULT '',platform_updated_at INTEGER NOT NULL DEFAULT 0,
 preferred_name TEXT NOT NULL DEFAULT '',aliases_json TEXT NOT NULL DEFAULT '[]',
 avoided_names_json TEXT NOT NULL DEFAULT '[]',description TEXT NOT NULL DEFAULT '',
 edited_by TEXT NOT NULL DEFAULT '',edited_at INTEGER NOT NULL DEFAULT 0,
 PRIMARY KEY(umo,account_id));
CREATE TABLE IF NOT EXISTS mr_member_pool (
 umo TEXT PRIMARY KEY,day TEXT NOT NULL,capacity INTEGER NOT NULL,accounts_json TEXT NOT NULL,
 members_json TEXT NOT NULL DEFAULT '[]');
CREATE TABLE IF NOT EXISTS mr_member_names (
 umo TEXT NOT NULL,account_id TEXT NOT NULL,name TEXT NOT NULL,source TEXT NOT NULL,
 PRIMARY KEY(umo,account_id,name,source));
"""

PREFIX_GUIDE = """本群成员资料（按真实 UID 对应）：群名片、平台昵称和历史显示名来自平台；本人/管理员填写的称呼和说明标明来源。UID 标识发言账号，不表示这句话谈的经历都属于发言者。不同 UID 即使同名也分别列出，同 UID 的换名仍是同一账号。preferred_name 是明确选择的称呼；avoided_names 是明确不希望被叫的称呼，历史原话里的旧称呼不覆盖它。未指定称呼时，可用当前群名片、昵称或省略称呼，不把自行截取、拼接的缩名当作已确认昵称。description 是资料填写者的自述/说明，不是对其所有说法的事实认证。此名单只是活跃成员子集；缺席不表示不是群员，完整资料可按 UID/称呼查询。"""


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def parse_profile_changes(arguments: str) -> dict:
    """Read explicit field/value pairs, keeping quoted field names as text."""
    fields = {"称呼": "preferred_name", "别名": "aliases", "不用": "avoided_names", "说明": "description"}
    lexer = shlex.shlex(arguments, posix=False)
    lexer.whitespace_split = True
    lexer.whitespace += "\u3000"
    lexer.commenters = ""
    changes, values = {}, []
    field = None

    def finish_field():
        value = " ".join(values).strip()
        if not value:
            raise ValueError(f"{field}后缺少内容；如需清空请填 -。本次资料未修改。")
        value = "" if value == "-" else value
        changes[fields[field]] = ([part.strip() for part in value.replace("，", ",").split(",") if part.strip()]
                                  if field in {"别名", "不用"} else value)

    try:
        tokens = list(lexer)
    except ValueError as exc:
        raise ValueError("引号未闭合，请用成对的英文引号包住内容。本次资料未修改。") from exc
    for token in tokens:
        if token in fields:
            if field is not None:
                finish_field()
            field, values = token, []
        else:
            if field is None:
                raise ValueError("用法：/mr uid；修改本人：/mr uid 称呼 小林 别名 林同学；还可用 不用 / 说明，填 - 清空。不能指定他人的 UID。")
            # posix=False retains quotes, so a value such as "别名" is not a field.
            values.append(token[1:-1] if len(token) >= 2 and token[0] in "\"'" and token[-1] == token[0] else token)
    if field is not None:
        finish_field()
    return changes


class MemberProfiles:
    def __init__(self, store):
        self.store, self.db, self.umo = store, store.db, store.umo
        self.db.executescript(SCHEMA)
        if "members_json" not in {r[1] for r in self.db.execute("PRAGMA table_info(mr_member_pool)")}:
            self.db.execute("ALTER TABLE mr_member_pool ADD COLUMN members_json TEXT NOT NULL DEFAULT '[]'")
            self.db.commit()
        cached = store.roster()
        if cached:
            self.observe_roster(cached["members"], cached["fetched_at"])

    def observe(self, account, *, nickname=None, card=None, at=None):
        account = str(account)
        if not account or self.store._forgotten(self.umo.split(":")[0], account):
            return
        now = int(at or time.time())
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO mr_member_profiles(umo,account_id) VALUES(?,?)", (self.umo, account))
            fields, values = [], []
            for key, value in (("nickname", nickname), ("card", card)):
                if value is not None:
                    fields.append(key + "=?")
                    values.append(str(value))
                    if str(value).strip():
                        self.db.execute("INSERT OR IGNORE INTO mr_member_names VALUES(?,?,?,?)", (self.umo, account, str(value), key))
            if fields:
                self.db.execute(f"UPDATE mr_member_profiles SET {','.join(fields)},platform_updated_at=? "
                                "WHERE umo=? AND account_id=? AND platform_updated_at<=?",
                                (*values, now, self.umo, account, now))

    def observe_roster(self, members, at):
        for member in members:
            if isinstance(member, dict):
                self.observe(member.get("user_id") or member.get("account_id") or "",
                             nickname=member.get("nickname"), card=member.get("card"), at=at)

    def members(self, account_ids=None, name=None):
        wanted = None if account_ids is None else {str(uid) for uid in account_ids}
        people = {row["account_id"]: dict(row) for row in self.db.execute(
            "SELECT id,account_id,canonical_key,current_display_name name,account_type,last_seen_at "
            "FROM participants WHERE umo=?", (self.umo,))}
        profiles = {row["account_id"]: dict(row) for row in self.db.execute(
            "SELECT * FROM mr_member_profiles WHERE umo=?", (self.umo,))}
        roster = self.store.roster()
        live = {str(m.get("user_id") or m.get("account_id") or ""): m for m in (roster or {}).get("members", []) if isinstance(m, dict)}
        result = []
        aliases = {}
        names = {}
        for row in self.db.execute("SELECT account_id,name FROM mr_member_names WHERE umo=? ORDER BY name COLLATE BINARY", (self.umo,)):
            names.setdefault(row["account_id"], set()).add(row["name"])
        for row in self.db.execute("""SELECT p.account_id,a.alias,a.source_kind FROM participant_aliases a
            JOIN participants p ON p.id=a.participant_id WHERE p.umo=? AND a.is_active=1
            ORDER BY a.alias COLLATE BINARY""", (self.umo,)):
            aliases.setdefault(row["account_id"], []).append(dict(row))
        for account in sorted(people.keys() | profiles.keys() | live.keys()):
            if (wanted is not None and account not in wanted) or self.store._forgotten(self.umo.split(":")[0], account):
                continue
            person = people.get(account, {"id": None, "account_id": account, "account_type": "USER", "last_seen_at": 0, "name": ""})
            profile = profiles.get(account, {})
            observed = sorted(names.get(account, set()) | {a["alias"] for a in aliases.get(account, []) if a["source_kind"] == "observed"})
            confirmed = json.loads(profile.get("aliases_json", "[]"))
            # Preserve explicit admin aliases created before the profile editor.
            if not profile.get("edited_at"):
                confirmed += [a["alias"] for a in aliases.get(account, []) if a["source_kind"] == "admin_confirmed"]
            person.update({key: profile.get(key, "") for key in ("nickname", "card", "preferred_name", "description", "edited_by")})
            person.update(edited_at=profile.get("edited_at", 0), confirmed_aliases=sorted(set(confirmed)),
                          avoided_names=json.loads(profile.get("avoided_names_json", "[]")), observed_names=observed,
                          aliases=sorted(set(observed + confirmed)),
                          membership="platform_roster" if account in live else "observed_in_history",
                          roster_fetched_at=(roster or {}).get("fetched_at"), role=live.get(account, {}).get("role"))
            person["name"] = person["preferred_name"] or person["card"] or person["nickname"] or person["name"] or account
            if name and str(name).casefold() not in "\n".join([account, person["name"], person["nickname"], person["card"], *person["aliases"]]).casefold():
                continue
            result.append(person)
        return result

    def edit(self, account, changes, *, actor):
        account = str(account).strip()
        if not self.members([account]):
            raise ValueError("找不到该群员，请先刷新群成员资料或让该群员发一条消息")
        allowed = {"preferred_name", "aliases", "avoided_names", "description"}
        if not isinstance(changes, dict) or set(changes) - allowed:
            raise ValueError("可修改称呼、别名、不使用的称呼和本人说明；UID 与平台资料由平台提供")
        values, fields = [], []
        for key, value in changes.items():
            if key in {"aliases", "avoided_names"}:
                if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
                    raise ValueError("别名和不使用的称呼应为文字列表")
                value = encoded(sorted({x.strip() for x in value if x.strip()}))
                key += "_json"
            elif not isinstance(value, str):
                raise ValueError("称呼和说明应为文字")
            else:
                value = value.strip()
            fields.append(key + "=?")
            values.append(value)
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO mr_member_profiles(umo,account_id) VALUES(?,?)", (self.umo, account))
            if fields:
                self.db.execute(f"UPDATE mr_member_profiles SET {','.join(fields)},edited_by=?,edited_at=? WHERE umo=? AND account_id=?",
                                (*values, actor, int(time.time()), self.umo, account))
        member = self.members([account])[0]
        # Explicit corrections take effect now, without reranking the daily pool.
        pool = self.db.execute("SELECT members_json FROM mr_member_pool WHERE umo=?", (self.umo,)).fetchone()
        if pool and fields:
            rows = json.loads(pool[0])
            rows = [self.prefix_row(member) if row["uid"] == account else row for row in rows]
            with self.db:
                self.db.execute("UPDATE mr_member_pool SET members_json=? WHERE umo=?", (encoded(rows), self.umo))
        return member

    @staticmethod
    def prefix_row(person):
        row = {"uid": person["account_id"]}
        for source, target in (("card", "card"), ("nickname", "nickname"), ("observed_names", "historical_display_names"),
                               ("preferred_name", "preferred_name"), ("confirmed_aliases", "aliases"),
                               ("avoided_names", "avoided_names"), ("description", "description"), ("edited_by", "information_source")):
            if person[source]:
                row[target] = person[source]
        return row

    def pool(self, capacity=50, *, now=None):
        now = int(now or time.time())
        capacity = max(0, int(capacity))
        day = time.strftime("%Y-%m-%d", time.localtime(now))
        cached = self.db.execute("SELECT * FROM mr_member_pool WHERE umo=?", (self.umo,)).fetchone()
        if not cached or cached["day"] != day or cached["capacity"] != capacity:
            # Count actual group-member speech, not mentions or bot/tool traffic.
            ranked = self.db.execute("""SELECT sender_id,count(*) n,max(sent_at) last FROM messages
                WHERE umo=? AND role='USER' AND is_deleted=0 AND sent_at>=? AND sent_at<=?
                GROUP BY sender_id ORDER BY n DESC,last DESC,sender_id""", (self.umo, now - 7 * 86400, now))
            has_roster = bool(self.store.roster())
            available = {p["account_id"] for p in self.members() if p["account_type"] != "BOT"
                         and (not has_roster or p["membership"] == "platform_roster")}
            accounts = [row["sender_id"] for row in ranked if row["sender_id"] in available][:capacity]
            # Stable UID order prevents activity rank changes from rearranging the prefix.
            accounts.sort()
            rows = [self.prefix_row(person) for person in self.members(accounts)]
            with self.db:
                self.db.execute("INSERT INTO mr_member_pool VALUES(?,?,?,?,?) ON CONFLICT(umo) DO UPDATE SET "
                    "day=excluded.day,capacity=excluded.capacity,accounts_json=excluded.accounts_json,members_json=excluded.members_json",
                    (self.umo, day, capacity, encoded(accounts), encoded(rows)))
        else:
            accounts = json.loads(cached["accounts_json"])
            rows = json.loads(cached["members_json"])
            if accounts and not rows:  # Upgrade the earlier pool schema once.
                rows = [self.prefix_row(person) for person in self.members(accounts)]
                with self.db:
                    self.db.execute("UPDATE mr_member_pool SET members_json=? WHERE umo=?", (encoded(rows), self.umo))
        prefix = PREFIX_GUIDE + "\n" + encoded(rows) if rows else ""
        return {"day": day, "capacity": capacity, "account_ids": [r["uid"] for r in rows], "members": rows,
                "prefix": prefix, "characters": len(prefix),
                "notice": "仅限制固定前缀人数，完整资料持续保存。每日按近七天发言量选取、UID排序；当天名单与资料保持稳定，本人/管理员修改立即生效。缓存命中仍计费并占用模型上下文，未命中按普通输入计费。"}
