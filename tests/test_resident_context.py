from __future__ import annotations

import asyncio
import contextlib
import json
import unittest
from types import SimpleNamespace

from mr_memory.evidence_pack import compile_evidence_atom_pack
from mr_memory.models import NormalizedMessage
from mr_memory.service import MemoryService
from mr_memory.storage import MemoryStorage
from mr_memory.storage import DistillationSnapshotChanged
from tests.test_main_local_behavior import _main_method


class ResidentContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_frozen_summary_survives_unrelated_background_write_only(self):
        guard = _main_method("_assert_snapshot_fresh",
                             DistillationSnapshotChanged=DistillationSnapshotChanged)
        recheck = _main_method("_assert_resident_dependencies_fresh",
                               DistillationSnapshotChanged=DistillationSnapshotChanged)
        host = SimpleNamespace()
        host._assert_snapshot_fresh = lambda **kw: guard(host, **kw)
        scope = "synthetic:GroupMessage:dependency-check"
        with contextlib.closing(MemoryStorage(":memory:")) as storage:
            def add(key, at):
                message = NormalizedMessage(
                    platform="synthetic", platform_id="synthetic", umo=scope,
                    group_id="dependency-check", message_id=key, sender_id="speaker",
                    sender_name="Synthetic", sent_at=at, plain_text=key,
                    content=[{"type": "plain", "text": key}],
                )
                storage.upsert_message(message)
                return message.resolved_source_key()
            source = add("prior", 50)
            selected = storage.store_episode(
                umo=scope, started_at=50, ended_at=50, title="Stored topic",
                summary="Prior summary", source_keys=[source], keywords=[],
            )
            add("request", 100)
            service = MemoryService(storage)
            snapshot = SimpleNamespace(
                umo=scope, cutoff_at=100, message_upper_bound=2,
                data_revision=SimpleNamespace(**{
                    field: storage.revision_vector(umo=scope)["data"].get(field, 0)
                    for field in ("message", "identity", "deletion", "graph", "relation", "feedback")
                }),
            )
            catalog = storage.audit_candidate_memory_closures(
                umo=scope, candidates={"episodes": [selected]},
                before_sent_at=100, message_upper_bound=2,
            )["stored_derivations"]
            self.assertEqual(len(catalog), 1)
            future = add("later", 110)
            storage.store_episode(
                umo=scope, started_at=110, ended_at=110, title="Unrelated later topic",
                summary="Only later sources", source_keys=[future], keywords=[],
            )
            with self.assertRaises(DistillationSnapshotChanged):
                await guard(host, service=service, snapshot=snapshot)
            await recheck(host, service=service, snapshot=snapshot, stored_derivations=catalog)
            with storage._connection:
                storage._connection.execute("UPDATE episodes SET summary=? WHERE id=?",
                                            ("Changed selected summary", selected))
            with self.assertRaisesRegex(DistillationSnapshotChanged, "selected stored memory"):
                await recheck(host, service=service, snapshot=snapshot, stored_derivations=catalog)

    async def test_next_turn_reopens_sources_with_current_snapshot_bounds(self):
        remember = _main_method("_remember_resident_context", asyncio=asyncio)
        reopen = _main_method(
            "_resident_context_sources",
            _collect_source_keys=_main_method("_collect_source_keys"),
        )
        host = SimpleNamespace(_wake_execution_locks={}, local_serving_max_items=12)
        scope = "synthetic:GroupMessage:continuity"
        with contextlib.closing(MemoryStorage(":memory:")) as storage:
            try:
                service = MemoryService(storage)

                def insert(key: str, text: str, at: int, group: str = scope):
                    storage.upsert_message(NormalizedMessage(
                        platform="synthetic", platform_id="synthetic", umo=group,
                        group_id=group.rsplit(":", 1)[-1], message_id=key,
                        sender_id="synthetic-member", sender_name="合成成员",
                        sent_at=at, plain_text=text, source_key=key,
                        content=[{"type": "plain", "text": text}],
                    ))

                insert("synthetic:remembered", "合成议题的原始消息", 50)
                first = SimpleNamespace(
                    umo=scope, cutoff_at=100, query_sha256="a" * 64,
                )
                certificate = SimpleNamespace(
                    status="CERTIFIED",
                    referents=(SimpleNamespace(reference="合成议题"),),
                    atoms=(SimpleNamespace(statement="THIS_OLD_ANSWER_MUST_NOT_BE_REUSED"),),
                )
                self.assertTrue(await remember(
                    host, service=service, snapshot=first, certificate=certificate,
                    source_keys={"synthetic:remembered"},
                ))
                saved = await service.subconscious_state(umo=scope)
                self.assertEqual(saved["last_tick_at"], 100)
                self.assertNotIn("THIS_OLD_ANSWER", json.dumps(saved))

                # Re-read authoritative source content after a correction. Old
                # model prose cannot survive through these operational pointers.
                insert("synthetic:remembered", "合成议题经过修订的原始消息", 50)
                insert("synthetic:at-cutoff", "不属于本快照的消息", 150)
                insert("synthetic:other-scope", "其他群消息", 60,
                       "synthetic:GroupMessage:other")
                # The fourth inserted row is late despite its old sent_at.
                insert("synthetic:late-row", "快照行上界之外", 70)
                await service.update_subconscious_state(
                    umo=scope, state={
                        **saved["state"],
                        "visited_source_keys": [
                            "synthetic:remembered", "synthetic:at-cutoff",
                            "synthetic:other-scope", "synthetic:late-row",
                        ],
                    }, last_query_sha256="a" * 64, at=100,
                )
                second = SimpleNamespace(
                    umo=scope, cutoff_at=150, message_upper_bound=3,
                    request_source_key="synthetic:request-2",
                )
                sources = await reopen(host, service=service, snapshot=second)
                self.assertEqual([row["source_key"] for row in sources],
                                 ["synthetic:remembered"])
                self.assertEqual(sources[0]["plain_text"], "合成议题经过修订的原始消息")
                packet = compile_evidence_atom_pack(
                    {"resident_context": {"messages": sources}}, max_sources=12,
                )
                self.assertEqual([row["source_key"] for row in packet["sources"]],
                                 ["synthetic:remembered"])
                self.assertNotIn("THIS_OLD_ANSWER", json.dumps(packet))
                competing = [
                    {"source_key":f"synthetic:current-{index}", "plain_text":"当前检索候选"}
                    for index in range(20)
                ]
                pressure_packet = compile_evidence_atom_pack({
                    "lexical_messages":competing,
                    "resident_context":{"messages":sources},
                }, max_sources=12)
                selected = [row["source_key"] for row in pressure_packet["sources"]]
                self.assertEqual(len(selected),12)
                self.assertIn("synthetic:remembered",selected)
                shared_packet = compile_evidence_atom_pack({
                    "lexical_messages":[*competing,*sources],
                    "resident_context":{"messages":sources},
                }, max_sources=12)
                self.assertEqual(sum(row["source_key"]=="synthetic:remembered"
                                     for row in shared_packet["sources"]),1)

                # An older concurrent request cannot overwrite a newer focus;
                # a historical snapshot cannot consume future operational state.
                older = SimpleNamespace(umo=scope, cutoff_at=90, query_sha256="b" * 64)
                self.assertFalse(await remember(
                    host, service=service, snapshot=older, certificate=certificate,
                    source_keys=set(),
                ))
                self.assertEqual(await reopen(
                    host, service=service, snapshot=older,
                ), [])
            finally:
                storage.close()
