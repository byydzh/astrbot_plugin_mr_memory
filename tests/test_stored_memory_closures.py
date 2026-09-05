from __future__ import annotations

import asyncio
import unittest
import uuid
from pathlib import Path

from mr_memory.models import NormalizedMessage
from mr_memory.plasticity import parse_graph_mutation
from mr_memory.service import MemoryService
from mr_memory.storage import MemoryStorage


class StoredMemoryClosureTests(unittest.TestCase):
    UMO = "synthetic:GroupMessage:source-closure"

    def setUp(self) -> None:
        folder = Path.cwd() / ".dev" / "test-tmp"
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / f"{uuid.uuid4().hex}.db"
        self.storage = MemoryStorage(self.path)
        self.messages = []
        for index in range(13):
            text = f"Synthetic observation {index}."
            message = NormalizedMessage(
                platform="synthetic", platform_id="synthetic", umo=self.UMO,
                group_id="source-closure", message_id=str(index),
                sender_id="synthetic-speaker", sender_name="Synthetic speaker",
                sent_at=100 + index, plain_text=text,
                content=[{"type": "plain", "text": text}],
                role="BOT" if index == 0 else "USER",
            )
            self.storage.upsert_message(message)
            self.messages.append(message)
        keys = [message.resolved_source_key() for message in self.messages]
        self.episode = self.storage.store_episode(
            umo=self.UMO, started_at=100, ended_at=112, title="Synthetic episode",
            summary="A stored summary backed by thirteen observations.",
            source_keys=keys, keywords=[],
        )
        self.semantic = self.storage.store_semantic_memory(
            umo=self.UMO, person="Synthetic speaker", aspect="synthetic aspect",
            content="Synthetic multi-source claim.", source_key=keys[0],
        )
        with self.storage._connection:
            self.storage._connection.execute(
                "INSERT INTO semantic_memory_sources(semantic_memory_id,message_id,evidence_role) "
                "SELECT ?,id,CASE WHEN id=(SELECT MAX(id) FROM messages) "
                "THEN 'CONTRADICT' ELSE 'SUPPORT' END FROM messages WHERE umo=?",
                (self.semantic, self.UMO),
            )
        edge = self.storage.apply_graph_mutation(
            umo=self.UMO,
            mutation=parse_graph_mutation({
                "operation": "upsert_edge", "evidence_source_keys": keys,
                "confidence": 0.7, "utility_delta": 0.5,
                "statement": "Synthetic source relates to synthetic target.",
                "source": {"kind": "concept", "label": "Synthetic source"},
                "target": {"kind": "concept", "label": "Synthetic target"},
                "relation": {
                    "key": "synthetic_relation", "name": "Synthetic relation",
                    "description": "Fixture relation", "source_kinds": ["concept"],
                    "target_kinds": ["concept"],
                },
            }),
        )
        self.candidates = {
            "episodes": [{"id": self.episode}],
            "semantic_memories": [{"id": self.semantic}],
            "associations": [{"id": edge["target_id"]}],
        }

    def tearDown(self) -> None:
        self.storage.close()
        for suffix in ("", "-wal", "-shm"):
            Path(f"{self.path}{suffix}").unlink(missing_ok=True)

    def audit(self, **kwargs: object) -> dict[str, object]:
        kwargs.setdefault("message_upper_bound", 13)
        return asyncio.run(MemoryService(self.storage).audit_candidate_memory_closures(
            umo=self.UMO, candidates=self.candidates, before_sent_at=200, **kwargs,
        ))

    def test_full_closure_exceeds_raw_presentation_budget(self) -> None:
        result = self.audit(message_upper_bound=13)
        self.assertEqual(result["rejected"], [])
        self.assertEqual(len(result["dependency_source_keys"]), 13)
        self.assertEqual(len(result["stored_derivations"]), 3)
        for descriptor in result["stored_derivations"]:
            self.assertEqual(len(descriptor["dependency_keys"]), 13)
            self.assertEqual(descriptor["source_roles"], ["BOT", "USER"])
            self.assertNotIn("source_keys", descriptor)
            self.assertTrue(descriptor["text"])
            self.assertIn("head_created_at", descriptor)
        semantic = next(item for item in result["stored_derivations"] if item["kind"] == "semantic")
        last_key = self.messages[-1].resolved_source_key()
        self.assertEqual(semantic["source_fingerprints"][last_key]["evidence_roles"], ["CONTRADICT"])

    def test_deleted_or_future_source_rejects_entire_closure(self) -> None:
        for field, value, reason in (
            ("is_deleted", 1, "SOURCE_DELETED"),
            ("sent_at", 200, "AT_OR_AFTER_CUTOFF"),
        ):
            with self.subTest(reason=reason):
                with self.storage._connection:
                    self.storage._connection.execute(
                        f"UPDATE messages SET {field}=? WHERE id=13", (value,),
                    )
                result = self.audit()
                self.assertEqual(result["stored_derivations"], [])
                self.assertEqual(result["dependency_source_keys"], [])
                self.assertEqual(len(result["rejected"]), 3)
                for rejected in result["rejected"]:
                    self.assertEqual(rejected["dependency_count"], 13)
                    self.assertIn(reason, {row["reason"] for row in rejected["violations"]})
                with self.storage._connection:
                    self.storage._connection.execute(
                        "UPDATE messages SET is_deleted=0,sent_at=112 WHERE id=13",
                    )

    def test_missing_relation_target_and_upper_bound_are_not_filtered_out(self) -> None:
        result = self.audit(message_upper_bound=12)
        self.assertEqual(result["stored_derivations"], [])
        self.assertTrue(all(
            "AFTER_MESSAGE_UPPER_BOUND" in {row["reason"] for row in item["violations"]}
            for item in result["rejected"]
        ))
        # Model a dangling legacy relation: the audit must see its missing join.
        self.storage._connection.execute("PRAGMA foreign_keys=OFF")
        with self.storage._connection:
            self.storage._connection.execute("DELETE FROM messages WHERE id=13")
        self.storage._connection.execute("PRAGMA foreign_keys=ON")
        result = self.audit()
        self.assertEqual(result["stored_derivations"], [])
        self.assertEqual(len(result["rejected"]), 3)
        for rejected in result["rejected"]:
            self.assertEqual(rejected["dependency_count"], 13)
            self.assertIn("SOURCE_NOT_FOUND", {row["reason"] for row in rejected["violations"]})


if __name__ == "__main__":
    unittest.main()
