"""Separate a historical migration's logical clock from witnessed and recorded time."""

# Archived backfills used the following night's replay clock. Identify the exact
# revision from its receipt; an address can later receive a real nightly update.
CLOCKS = """WITH history_batches AS MATERIALIZED (
    SELECT k,at,receipt,json_extract(receipt,'$.recorded_at') AS recorded_at
    FROM batch WHERE k GLOB 'history-[0-9]*'
), edits AS MATERIALIZED (
    SELECT b.k,b.at,b.recorded_at,json_extract(j.value,'$.id') AS id,
           json_extract(j.value,'$.revision') AS revision
    FROM history_batches b,json_each(b.receipt,'$.edits') j
), clocks AS MATERIALIZED (
    SELECT id,revision,at,recorded_at FROM edits
    UNION ALL
    SELECT m.value,e.revision,e.at,e.recorded_at FROM history_batches b,json_each(b.receipt,'$.mirrors') m
    JOIN edits e ON e.k=b.k AND e.id='M'||substr(m.value,2)
) """
JOIN_CLOCK = "LEFT JOIN clocks c ON c.id=n.id AND c.revision=n.revision AND c.at=n.at"
# Start with this node's handful of links, even with the time index on E.
EVIDENCE_TIME = """(SELECT max(w.at) FROM edge e CROSS JOIN node w ON w.id=e.dst
    WHERE e.src=n.id AND w.region='E' AND w.kind='message' AND w.status<>'INVALIDATED'
    AND e.rel NOT IN ('adjacent','REPLY_TO'))"""
DISPLAY_TIME = f"CASE WHEN c.id IS NULL THEN n.at ELSE coalesce(c.recorded_at,{EVIDENCE_TIME}) END"


def recall_times(db):
    """Use this read transaction's effective clocks for ranking and model-visible dates."""
    rows = db.execute(f"""{CLOCKS} SELECT n.id,c.recorded_at,{EVIDENCE_TIME}
        FROM clocks c JOIN node n ON n.id=c.id AND n.revision=c.revision AND n.at=c.at""")
    times = {key: (written, evidence) for key, written, evidence in rows}

    def recency(key, stamp):
        if key not in times:
            return stamp
        written, evidence = times[key]
        return written if written is not None else evidence

    db.create_function("mr_recency", 2, recency, deterministic=True)
    return times
