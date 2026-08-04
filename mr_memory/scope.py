from __future__ import annotations

import hashlib
from dataclasses import dataclass


class GroupScopeError(ValueError):
    """Raised when an event cannot be bound to one isolated group scope."""


@dataclass(frozen=True, slots=True)
class GroupMemoryScope:
    """Server-derived tenant boundary for every memory read and write."""

    key: str
    platform_id: str
    group_id: str

    @property
    def storage_id(self) -> str:
        """Opaque stable filename component; raw group identifiers stay in SQLite."""
        return hashlib.sha256(self.key.encode("utf-8")).hexdigest()

    @classmethod
    def from_event_values(
        cls,
        *,
        unified_msg_origin: str,
        platform_id: str,
        group_id: str,
    ) -> "GroupMemoryScope":
        normalized_umo = unified_msg_origin.strip()
        normalized_platform = platform_id.strip()
        normalized_group = group_id.strip()
        if not normalized_group:
            raise GroupScopeError("group_id is required")
        if not normalized_platform:
            raise GroupScopeError("platform_id is required")
        if not normalized_umo:
            raise GroupScopeError("unified_msg_origin is required")
        return cls(
            key=normalized_umo,
            platform_id=normalized_platform,
            group_id=normalized_group,
        )
