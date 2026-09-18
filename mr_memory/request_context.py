"""Read-only access to the main model's current input, without executing its tools."""
from __future__ import annotations

import copy

from .content import text_view


def _plain(value):
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump(exclude_none=True))
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


class RequestContext:
    """A snapshot of public ProviderRequest fields at the MR hook boundary.

    The directory is cheap. The memory model can open full context or tool
    declarations when their content changes what background would be useful.
    """
    def __init__(self, request):
        tools = getattr(getattr(request, "func_tool", None), "tools", []) or []
        additional = [part for part in (getattr(request, "extra_user_content_parts", []) or [])
                      if not str(getattr(part, "text", "")).startswith(("<mr_group_context>", "<mr_request_context>"))]
        self.sections = copy.deepcopy(_plain({
            "request": [getattr(request, "prompt", "") or ""],
            "conversation": getattr(request, "contexts", []) or [],
            "instructions": [getattr(request, "system_prompt", "") or ""],
            "additional": additional,
            "tools": [{"name": tool.name, "description": tool.description, "parameters": tool.parameters}
                      for tool in tools],
            "media": [{"type": kind, "url": url} for kind, attr in (("image", "image_urls"), ("audio", "audio_urls"))
                      for url in (getattr(request, attr, []) or [])],
        }))

    def directory(self):
        return {"sections": {name: len(items) for name, items in self.sections.items()},
                "tool_names": [tool["name"] for tool in self.sections["tools"]],
                "meaning": "这是本次主模型请求已有的输入和工具目录，不是MR能执行的工具。用main_context按需打开。媒体列表表示实际输入，MR的文字视图未识别图片或音频；不能因此推断主模型也看不到、不能检索或不能行动。"}

    def read(self, section, offset=0, limit=8):
        if section not in self.sections:
            raise ValueError("Unknown section; choose " + ", ".join(self.sections))
        offset, limit = max(0, int(offset)), max(1, min(100, int(limit)))
        rows = self.sections[section]
        return {"section": section, "offset": offset, "total": len(rows),
                "items": text_view(rows[offset:offset + limit]), "more": offset + limit < len(rows),
                "next_offset": offset + limit}
