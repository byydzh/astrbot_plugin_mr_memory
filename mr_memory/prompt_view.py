"""Small model-facing views; the full profiles and source records stay in storage."""
from __future__ import annotations

import json

from .content import text_view


MEMBER_GUIDE = (
    "群员身份：uid及ids是已关联的同一账号，不按同名合并。name是显示名，names是其他记录名，"
    "call是指定称呼，avoid是不使用的称呼，aliases是确认别名，note/by是资料及填写者。"
    "称呼以call为先；旧聊天不覆盖纠正。账号身份不代表话中所有经历都属于本人。"
    "这里只列活跃成员；本轮members补充或更新此表。admin仅指AstrBot权限。"
)


def compact_members(people, links=None):
    links = links or {}
    groups = {}
    for person in people:
        account = str(person.get('uid') or person.get('account_id') or '')
        if account:
            groups.setdefault(links.get(account, account), []).append(person)
    result = []
    for uid, rows in sorted(groups.items()):
        primary = max(rows, key=lambda p: (p.get('edited_at') or 0,
                       str(p.get('uid') or p.get('account_id')) == uid))
        row = {'uid': uid}
        ids = {str(p.get('uid') or p.get('account_id')) for p in rows}
        ids.update(key for key, value in links.items() if value == uid)
        ids.update(str(v) for p in rows for v in p.get('linked_account_ids', []))
        ids.discard(uid)
        if ids:
            row['ids'] = sorted(ids)
        name = primary.get('card') or primary.get('nickname') or primary.get('name')
        if name and name != uid:
            row['name'] = name
        for source, target in (('preferred_name', 'call'), ('avoided_names', 'avoid'),
                               ('description', 'note'), ('information_source', 'by'), ('edited_by', 'by')):
            if primary.get(source):
                row[target] = primary[source]
        aliases = sorted(set(primary.get('confirmed_aliases', primary.get('aliases', []))))
        if aliases:
            row['aliases'] = aliases
        names = set()
        for person in rows:
            names.update(person.get('historical_display_names', person.get('observed_names', [])))
            names.update(person.get(k) for k in ('name', 'card', 'nickname') if person.get(k))
        names.difference_update([uid, row.get('name'), row.get('call'), *aliases])
        if names:
            row['names'] = sorted(names)
        if any(p.get('account_type') == 'BOT' for p in rows):
            row['bot'] = True
        if primary.get('role'):
            row['group_role'] = primary['role']
        result.append(row)
    return result


def directory_prefix(people, links=None):
    rows = compact_members(people, links)
    return MEMBER_GUIDE + '\n' + json.dumps(rows, ensure_ascii=False, separators=(',', ':')) if rows else ''


def member_updates(people, pool, links=None):
    known = {p['uid']: p for p in compact_members(pool, links)}
    result = []
    for person in compact_members(people, links):
        previous = known.get(person['uid'])
        if previous is None:
            result.append(person)
            continue
        changed = {key: person.get(key) for key in person.keys() | previous.keys()
                   if key != 'uid' and person.get(key) != previous.get(key)}
        if changed:
            result.append({'uid': person['uid'], **dict(sorted(changed.items()))})
    return result


def request_view(current, pool, links=None):
    runtime = current.get('astrbot_runtime') or {}
    view = {'speaker': current.get('sender_id'), 'name': current.get('sender_name'),
            'bot': current.get('bot_id'), 'admin': runtime.get('is_admin', False)}
    clock = current.get('clock') or {}
    now = (clock.get('now') or {}).get('local', '')
    sent = (clock.get('message') or {}).get('local', '')
    # Keep the service-local offset, without a second UTC spelling or microseconds.
    from datetime import datetime
    def seconds(value):
        return datetime.fromisoformat(value).isoformat(timespec='seconds') if value else ''
    if now:
        view['time'] = seconds(now)
    if sent and seconds(sent) != view.get('time'):
        view['sent'] = seconds(sent)
    updates = member_updates(current.get('participants', []), pool, links)
    if updates:
        view['members'] = updates
    return {key: value for key, value in view.items() if value is not None and value != ''}


def prepend_directory(prompt, prefix):
    start, end = '<mr_member_directory>', '</mr_member_directory>'
    if start in prompt and end in prompt:
        before, _, tail = prompt.partition(start)
        _, _, after = tail.partition(end)
        prompt = (before.rstrip() + after).strip()
    return (start + '\n' + prefix + '\n' + end + '\n\n' + prompt) if prefix else prompt


def learning_search_view(value, *, include_sources=True):
    """Keep candidate meaning and attribution without expanding adjacent memories."""
    if isinstance(value, list):
        return [learning_search_view(item, include_sources=include_sources) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: item if key in {'representation', 'belief', 'attention', 'recollection'}
              else learning_search_view(item, include_sources=include_sources) for key, item in value.items()}
    if 'kind' not in result or 'id' not in result:
        return result
    if not include_sources:
        result.pop('sources', None)
    for key in ('created_at', 'legacy_created_at', 'legacy_updated_at', 'legacy_status'):
        result.pop(key, None)
    for side in ('source_node', 'target_node'):
        endpoint = result.get(side)
        if isinstance(endpoint, dict):
            result[side] = {key: endpoint[key] for key in (
                'kind', 'id', 'title', 'revision_no', 'status', 'belief', 'change_reason') if key in endpoint}
    if result.get('kind') == 'reflection':
        for key in ('last_reviewed_at', 'next_review_at', 'next_review_explicit', 'request_id'):
            result.pop(key, None)
    return {key: item for key, item in result.items() if item is not None and item != [] and item != ''}


def memory_match_view(value):
    """Discovery cards. The memory tool opens the full record and its original evidence."""
    if isinstance(value, list):
        return [memory_match_view(row) for row in value]
    if not isinstance(value, dict):
        return value
    if isinstance(value.get('memory'), dict) and 'score' in value:
        return {**value, 'memory': memory_match_view(value['memory'])}
    if value.get('kind') not in {'event', 'semantic', 'association'} or type(value.get('id')) is not int:
        return learning_search_view(value)
    fields = ('kind', 'id', 'revision_no', 'title', 'summary', 'status', 'belief', 'subject',
              'source_ids', 'source_contexts', 'started_at', 'ended_at', 'relation', 'purpose',
              'source_ref', 'source', 'target_ref', 'target', 'change_reason')
    card = {key: value[key] for key in fields if value.get(key) not in (None, '', [])}
    if 'summary' not in card:
        for key in ('statement', 'content', 'context', 'description'):
            if value.get(key):
                card[key] = value[key]
                break
    return card


def learning_material_view(messages):
    """Offer long external tool bodies by address alongside their call and excerpt."""
    result = []
    for message in messages:
        row = dict(message)
        content = []
        for part in message.get('content', []):
            if (not isinstance(part, dict) or part.get('type') != 'tool_result'
                    or 'result' not in part or type(message.get('id')) is not int):
                content.append(part)
                continue
            body = text_view(part['result'])
            rendered = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False, separators=(',', ':'))
            if len(rendered) <= 2048:
                content.append(part)
                continue
            content.append({**{key: value for key, value in part.items() if key != 'result'},
                            'result_preview': rendered[:800],
                            'result_ref': {'message_id': message['id'], 'characters': len(rendered)}})
        if 'content' in message:
            row['content'] = content
        result.append(row)
    return result


def learning_address_view(value):
    """Use resolved local message addresses instead of repeated transport IDs."""
    if isinstance(value, list):
        return [learning_address_view(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: item if key in {'representation', 'belief', 'attention', 'recollection'}
              else learning_address_view(item) for key, item in value.items()}
    if result.get('type') in {'image', 'file', 'record', 'audio', 'video'}:
        addresses = set()
        for key in ('url', 'file', 'path'):
            address = result.get(key)
            if isinstance(address, str) and address:
                if address in addresses:
                    result.pop(key)
                else:
                    addresses.add(address)
    if result.get('type') == 'reply' and type(result.get('source_id')) is int:
        result.pop('message_id', None)
        result.pop('source_key', None)
    original = value.get('reply_to_message')
    if not isinstance(original, dict) or type(original.get('id')) is not int:
        return result
    source_key = original.get('source_key')
    if not source_key or value.get('reply_to') != source_key:
        return result
    identifier = original['id']
    result['reply_to'] = {'message_id': identifier}
    for relation in result.get('relations', []):
        if relation.get('target_source_key') == source_key:
            relation.pop('target_source_key')
            relation['target_message_id'] = identifier
    for part in result.get('content', []):
        if (isinstance(part, dict) and part.get('type') == 'response_to'
                and isinstance(part.get('message_id'), str)
                and source_key.endswith('|' + part['message_id'])):
            part.pop('message_id')
            part['source_id'] = identifier
    return result
