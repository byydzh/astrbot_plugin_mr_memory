from __future__ import annotations


def scoped_job_key(*, umo: str, job_id: int) -> tuple[str, int]:
    """Return a process-wide key for an ID allocated inside one scope DB."""

    scope_key = str(umo).strip()
    if not scope_key:
        raise ValueError("maintenance job scope cannot be empty")
    return scope_key, int(job_id)
