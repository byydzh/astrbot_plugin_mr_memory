"""Scheduled, checkpointed L0 extraction followed by semantic integration of the same cohort."""

import asyncio
import hashlib
import json
import time
import numpy as np

from ..agent import _add_usage
from ..trace import RunTrace
from ..schedule import work_window
from .storage import opened, parse_plan, upper_context, upsert_vectors
from .base import packed
from .consolidation import EXTRACT, SYSTEM, extract_ops, raw_packet, semantic_ops
from .live import raw_view
from .recall import snapshot


class StaleMaterial(RuntimeError):
    """The next consolidation must reopen the changed evidence."""


def transaction(store, fn, *args):
    with store._lock:
        net = opened(store.db)
        with net.db:
            return fn(net, store, *args)


def plan(net, store, force, config):
    db = net.db
    if not force and not work_window(config)["open"]:
        return {"status": "deferred", "reason": "waiting_window"}
    work = db.execute("SELECT horizon,started FROM network_work").fetchone()
    if not work:
        count, oldest, horizon = db.execute(
            """SELECT count(*),min(m.sent_at),max(m.id) FROM messages m JOIN message_processing p
            ON m.id=p.message_id WHERE m.is_deleted=0 AND p.status<>'DISTILLED'"""
        ).fetchone()
        if not count:
            return {"status": "idle", "reason": "没有待整理消息"}
        if (
            not force
            and count < config["auto_distillation_min_pending"]
            and time.time() - oldest < config["maintenance_interval_seconds"]
        ):
            return {"status": "waiting", "reason": "等待积累材料"}
        started = int(time.time())
        db.execute("INSERT INTO network_work VALUES(1,?,?)", (horizon, started))
    else:
        horizon, started = work
    rows = db.execute(
        """SELECT m.* FROM messages m JOIN message_processing p ON m.id=p.message_id
       LEFT JOIN network_inputs i ON i.message_id=m.id AND i.digest=m.content_sha256 WHERE m.id<=? AND m.is_deleted=0
       AND i.message_id IS NULL AND p.status<>'DISTILLED' ORDER BY m.sent_at,m.id LIMIT ?""",
        (horizon, config["distillation_max_messages"]),
    ).fetchall()
    if rows:
        ids = {row["id"]: row["content_sha256"] for row in rows}
        key = "live-" + hashlib.sha256(packed(list(ids.items())).encode()).hexdigest()[:20]
        return dict(
            phase="extract", key=key, rows=[raw_view(store, r) for r in rows], digests=ids, at=started
        )
    ids = [r[0] for r in db.execute("SELECT id FROM network_l1_pending ORDER BY rowid")]
    raw = {}
    ops = []
    for k in ids:
        z = net.read(k)
        if not z or z["status"] != "ACTIVE":
            continue
        basis = [
            r[0] for r in db.execute("SELECT dst FROM edge WHERE src=? AND rel='source' ORDER BY dst", (k,))
        ]
        for addr in basis:
            if addr in raw:
                continue
            row = db.execute(
                "SELECT * FROM messages WHERE id=? AND is_deleted=0", (int(addr[10:]),)
            ).fetchone()
            if row:
                raw[addr] = raw_view(store, row)
        ops.append(
            dict(
                id=k,
                kind=z["kind"],
                subjects=z["subjects"],
                text=z["body"],
                basis=[a for a in basis if a in raw],
            )
        )
    return dict(phase="integration", key=f"integrate-{horizon}-{started}", at=started, rows=raw, ops=ops)


def local_projection(db, key, ops, raw, vectors):
    rows = {r["id"]: r["at"] for r in raw}
    own = {}
    for i, (v, op) in enumerate(zip(vectors, ops, strict=True), 1):
        addr = f"M|delta:{key}:{i}"
        refs = set(op["basis"])
        stamp = max(rows[r] for r in refs)
        own[addr] = (refs, np.asarray(v, dtype="<f4"))
        db.execute("INSERT OR REPLACE INTO network_l1_clock VALUES(?,?)", (addr, stamp))
    affected = set(own)
    for addr in own:
        at = db.execute("SELECT at FROM network_l1_clock WHERE id=?", (addr,)).fetchone()[0]
        affected.update(
            r[0]
            for r in db.execute(
                """SELECT id FROM network_l1_clock WHERE (at,id)>(?,?)
            AND at<=? ORDER BY at,id LIMIT 64""",
                (at, addr, at + 86400),
            )
        )

    def evidence(k):
        if k not in own:
            refs = {r[0] for r in db.execute("SELECT dst FROM edge WHERE src=? AND rel='source'", (k,))}
            hit = db.execute("SELECT vector FROM network_vector WHERE id=?", (k,)).fetchone()
            own[k] = (refs, np.frombuffer(hit[0], dtype="<f4") if hit else None)
        return own[k]

    result = []
    for addr in sorted(affected):
        at = db.execute("SELECT at FROM network_l1_clock WHERE id=?", (addr,)).fetchone()[0]
        previous = db.execute(
            """SELECT id FROM network_l1_clock WHERE (at,id)<(?,?) AND at>=?
                                ORDER BY at DESC,id DESC LIMIT 64""",
            (at, addr, at - 86400),
        ).fetchall()
        refs, v = evidence(addr)
        candidates = []
        for r in reversed(previous):
            old, vec = evidence(r[0])
            sim = float(v @ vec) if v is not None and vec is not None else -1
            candidates.append([r[0], len(old & refs), sim])
        top = sorted(candidates, key=lambda x: (x[1], x[2]), reverse=True)[:2]
        result.append({"id": addr, "selected": [r for r in top if r[1] or r[2] >= 0.65]})
    return result


def capture_edges(db):
    db.execute("CREATE TEMP TABLE IF NOT EXISTS mr_changed_edges(src,dst,rel,old,new)")
    db.execute("DELETE FROM mr_changed_edges")
    # Only this group connection sees TEMP triggers, and only during its graph write.
    for name, action, fields in (
        ("added", "INSERT", "new.src,new.dst,new.rel,NULL,new.active"),
        ("removed", "DELETE", "old.src,old.dst,old.rel,old.active,NULL"),
        ("updated", "UPDATE OF active", "new.src,new.dst,new.rel,old.active,new.active"),
    ):
        db.execute(f"""CREATE TEMP TRIGGER mr_net_{name} AFTER {action} ON main.edge
              BEGIN INSERT INTO mr_changed_edges VALUES({fields}); END""")


def write(net, store, p, ops, visible, vectors, run):
    db = net.db
    phase = p["phase"]
    if phase == "extract":
        for mid, digest in p["digests"].items():
            r = db.execute("SELECT content_sha256,is_deleted FROM messages WHERE id=?", (mid,)).fetchone()
            if not r or tuple(r) != (digest, 0):
                raise StaleMaterial("材料在本次调用中发生变更，须重读")
        ops["program_local_projection"] = local_projection(db, p["key"], ops["changes"], p["rows"], vectors)
    else:
        ops["incoming_l1"] = [op["id"] for op in p["ops"]]
        if db.execute("""SELECT 1 FROM network_inputs i JOIN messages m ON i.message_id=m.id
             WHERE m.is_deleted=0 AND m.content_sha256<>i.digest LIMIT 1""").fetchone():
            raise StaleMaterial("已提炼的原话出现新版本，下一批重读")
        for k in visible:
            if not net.read(k):
                raise StaleMaterial("本轮依据已撤回，须重读")
    for op in ops["changes"]:
        for uid in op.get("subjects", []):
            if not db.execute("SELECT 1 FROM network_account WHERE token=?", (uid,)).fetchone():
                raise ValueError("主体不在本群账户表：" + str(uid)[:40])
    capture_edges(db)
    prior = [net.read(op["id"]) for op in ops["changes"]]
    rec = net.apply(p["key"], p["at"], p["rows"] if phase == "extract" else [], visible, ops)
    edits = [e["id"] for e in rec["edits"]]
    aliases = {op["id"]: k for op, k in zip(ops["changes"], edits)}
    upsert_vectors(db, edits, vectors)
    by_address = dict(zip(edits, vectors, strict=True))
    twins = rec.get("mirrors", [])
    upsert_vectors(db, twins, [by_address["M" + k[1:]] for k in twins])
    changes = [
        {
            "address": k,
            "before": before["body"] if before else None,
            "after": net.read(k)["body"],
            "before_node": before,
            "after_node": net.read(k),
            "basis_addresses": [aliases.get(b, b) for b in op["basis"]],
        }
        for k, before, op in zip(edits, prior, ops["changes"], strict=True)
    ]
    for k in rec.get("mirrors", []):
        old = db.execute(
            "SELECT previous FROM change WHERE id=? AND revision=?", (k, net.read(k)["revision"])
        ).fetchone()
        prior_twin = json.loads(old[0]) if old and old[0] else None
        changes.append(
            {
                "address": k,
                "before": prior_twin["body"] if prior_twin else None,
                "after": net.read(k)["body"],
                "before_node": prior_twin,
                "after_node": net.read(k),
                "basis_addresses": ["M" + k[1:]],
            }
        )
    edges = {}
    for a, b, r, old, new in db.execute("SELECT * FROM mr_changed_edges ORDER BY rowid"):
        init = edges.get((a, b, r), (old, None))[0]
        edges[a, b, r] = (init, new)
    diff = {
        "changes": changes,
        "edges": [
            {
                "source": a,
                "target": b,
                "relation": r,
                "action": (
                    "added"
                    if old is None
                    else "removed" if new is None else "activated" if new else "deactivated"
                ),
                "active_before": old,
                "active_after": new,
            }
            for (a, b, r), (old, new) in edges.items()
            if old != new
        ],
    }
    for name in ("added", "removed", "updated"):
        db.execute("DROP TRIGGER temp.mr_net_" + name)
    db.execute("INSERT INTO mr_network_run_changes VALUES(?,?)", (run, packed(diff)))
    if phase == "extract":
        db.executemany(
            "INSERT OR REPLACE INTO network_inputs VALUES(?,?,?)",
            [(mid, d, run) for mid, d in p["digests"].items()],
        )
        db.executemany(
            "UPDATE mr_manual_items SET run_id=? WHERE message_id=?", [(run, mid) for mid in p["digests"]]
        )
        db.executemany(
            "UPDATE message_processing SET status='PROCESSING',last_error='' WHERE message_id=?",
            [(mid,) for mid in p["digests"]],
        )
        db.executemany(
            "INSERT OR REPLACE INTO network_l1_pending VALUES(?,?,?)", [(k, run, p["key"]) for k in edits]
        )
    else:
        db.execute(
            """INSERT OR REPLACE INTO network_consumed SELECT i.message_id,i.digest,? FROM network_inputs i
               JOIN message_processing p ON p.message_id=i.message_id AND p.content_sha256=i.digest""",
            (run,),
        )
        db.execute("""UPDATE message_processing SET status='DISTILLED',distilled_at=unixepoch(),last_error=''
            WHERE EXISTS(SELECT 1 FROM network_inputs i WHERE i.message_id=message_processing.message_id
                    AND i.digest=message_processing.content_sha256)""")
        db.execute(
            """INSERT OR IGNORE INTO mr_manual_run_history SELECT i.message_id,i.run_id FROM mr_manual_items i
                 JOIN network_inputs n ON i.message_id=n.message_id WHERE i.run_id IS NOT NULL"""
        )
        db.execute(
            "UPDATE mr_manual_items SET run_id=? WHERE message_id IN (SELECT message_id FROM network_inputs)",
            (run,),
        )
        for t in ("network_inputs", "network_l1_pending", "network_work"):
            db.execute("DELETE FROM " + t)
    return len(edits)


def attempt(net, store, p, run, state, reason=""):
    db = net.db
    mids = (
        list(p["digests"])
        if p["phase"] == "extract"
        else [r[0] for r in db.execute("SELECT message_id FROM network_inputs")]
    )
    for mid in mids:
        db.execute(
            "UPDATE message_processing SET status=?,last_error=? WHERE message_id=? AND status<>'DISTILLED'",
            (state, reason, mid),
        )
        db.execute(
            """INSERT OR IGNORE INTO mr_manual_run_history SELECT message_id,run_id FROM mr_manual_items
                    WHERE message_id=? AND run_id IS NOT NULL""",
            (mid,),
        )
        db.execute("UPDATE mr_manual_items SET run_id=? WHERE message_id=?", (run, mid))


async def consolidate(plugin, store, force):
    p = await asyncio.to_thread(transaction, store, plan, force, plugin.config)
    if "phase" not in p:
        return p
    agent = plugin.agent(store, background=True)
    emb = plugin.embedder
    if emb is None or not plugin.config["embedding_enabled"]:
        raise RuntimeError("四区整理需要启用当前群的向量编码器")
    phase = p["phase"]
    visible = []
    await asyncio.to_thread(store.bind_network_space, emb.model_id, emb.dimensions)
    if phase == "extract":
        inp = raw_packet(p["rows"])
        prompt = EXTRACT
    elif p["ops"]:
        keys = [x["id"] for x in p["ops"]]
        queries = [await emb.query(x["text"]) for x in p["ops"]]
        inp, ids, _ = await snapshot(
            store, lambda source: upper_context(source.db, p["ops"], keys, queries, p["rows"])
        )
        visible = list(ids.values())
        prompt = SYSTEM
    else:
        inp = None
    trace = RunTrace(store, "background", time.time(), {"phase": phase, "network": "plan_d"})
    await trace.start()
    if trace.id is None:
        raise RuntimeError("未能建立整理收据")
    agent.trace = trace
    result = {"phase": phase, "network": "plan_d", "run_id": trace.id, "status": "error"}
    started = False
    try:
        conv = [{"role": "user", "content": packed(inp)}]
        limit = min(
            agent.max_output_tokens,
            (
                max(5500, min(16000, len(p["rows"]) * 44))
                if phase == "extract"
                else max(16000, len(p["ops"]) * 650)
            ),
        )
        await trace.emit(
            "input",
            "打开本群新网整理材料",
            phase_name=phase,
            source_ids=list(p.get("digests", {})),
            opened=visible,
        )
        total = {}
        for i in range(4):
            ops = {"changes": [], "edges": []}
            if inp is not None:
                used = await asyncio.to_thread(store.usage_total, "background")
                budget = plugin.config["private_daily_token_budget"]
                reserve = limit + len(packed(conv).encode()) + len(prompt.encode()) + 1200
                if budget > 0 and used + reserve > budget:
                    result.update(status="budget_exhausted", reason="budget_exhausted")
                    if started:
                        await asyncio.to_thread(
                            transaction, store, attempt, p, trace.id, "PENDING", "budget_exhausted"
                        )
                    return result
                if not started:
                    await asyncio.to_thread(transaction, store, attempt, p, trace.id, "PROCESSING")
                    started = True
                claim = await asyncio.to_thread(store.reserve_usage, "background", reserve)
                async with asyncio.timeout(agent.timeout_seconds):
                    answer = await agent._model_turn(
                        conv,
                        prompt,
                        None,
                        i + 1,
                        max_output_tokens=limit,
                        response_format={"type": "json_object"},
                    )
                use = {}
                _add_usage(use, answer)
                if sum(use.values()):
                    await asyncio.to_thread(store.settle_usage, claim, sum(use.values()))
                for k, v in use.items():
                    total[k] = total.get(k, 0) + v
            try:
                if inp is not None:
                    obj = parse_plan(answer.completion_text or "")
                    ops = extract_ops(obj, p["rows"]) if phase == "extract" else semantic_ops(obj, ids)
                texts = [op["text"] for op in ops["changes"]]
                values = await emb.texts(texts) if texts else []
                count = await asyncio.to_thread(transaction, store, write, p, ops, visible, values, trace.id)
                result.update(status="completed", written_count=count, usage=total)
                return result
            except (KeyError, TypeError, ValueError) as exc:
                if i == 3 or inp is None:
                    raise
                conv.extend(
                    [
                        {"role": "assistant", "content": answer.completion_text},
                        {
                            "role": "user",
                            "content": "本次未提交，请修正并重新给出完整操作。" + str(exc)[:250],
                        },
                    ]
                )
    except BaseException as exc:
        result.update(reason=type(exc).__name__)
        if started:
            state = "PENDING" if isinstance(exc, asyncio.CancelledError) else "FAILED"
            await asyncio.to_thread(transaction, store, attempt, p, trace.id, state, result["reason"])
        raise
    finally:
        await trace.finish(result)
