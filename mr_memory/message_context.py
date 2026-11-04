"""Optional producer-supplied context; transport identities remain authoritative."""
from copy import deepcopy


def from_content(content):
    return [deepcopy(item) for item in content
            if isinstance(item, dict) and item.get("type") == "message_context"]


def from_event(event, direction):
    value = getattr(event, "_mr_" + direction + "_context", None)
    return [{**deepcopy(value), "type": "message_context"}] if isinstance(value, dict) else []


def replace_context(content, context):
    return [item for item in content if not (
        isinstance(item, dict) and item.get("type") == "message_context"
        and item.get("producer") == context.get("producer"))] + [deepcopy(context)]
