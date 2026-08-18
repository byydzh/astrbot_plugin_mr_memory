from __future__ import annotations

import unittest

from mr_memory.snapshot import (
    DataRevisionVector,
    InferenceRevisionVector,
    RequestSnapshot,
    semantic_certificate_lookup_key,
)


def _data_revision() -> DataRevisionVector:
    return DataRevisionVector.from_value(
        {
            "message": 12,
            "deletion": 2,
            "identity": "id-4",
            "graph": 9,
            "relation": 3,
            "feedback": 7,
        }
    )


def _inference_revision() -> InferenceRevisionVector:
    return InferenceRevisionVector.from_value(
        {
            "retriever": "hybrid-v2",
            "embedding_model": "harrier-270m",
            "fusion_policy": "rrf-v1",
            "reader_model": "deepseek-v4-flash",
            "reader_protocol": "reader-v2",
            "certificate_schema": "evidence-certificate.v2",
            "surface_compiler": "memory-surface.v1",
            "route_policy": "host-route-policy.v1",
        }
    )


def _snapshot() -> RequestSnapshot:
    return RequestSnapshot.create(
        snapshot_id="request-726",
        umo="aiocqhttp:GroupMessage:123456",
        cutoff_at=1_800_000_000,
        message_upper_bound=726,
        request_source_key="msg:727",
        sender_participant_key="qq:10001",
        reply_source_key="msg:700",
        query="阐述阿拉蕾的挺点",
        context={"reply": "msg:700", "history": ["a", "b"]},
        data_revision=_data_revision(),
        inference_revision=_inference_revision(),
        captured_at=1_800_000_001,
    )


class RequestSnapshotTests(unittest.TestCase):
    def test_round_trip_and_digest_are_stable(self) -> None:
        snapshot = _snapshot()
        rebuilt = RequestSnapshot.from_value(snapshot.as_dict())
        self.assertEqual(rebuilt, snapshot)
        self.assertEqual(rebuilt.digest, snapshot.digest)
        self.assertEqual(len(snapshot.digest), 64)

    def test_create_uses_reader_query_canonicalization(self) -> None:
        first = _snapshot()
        second = RequestSnapshot.create(
            snapshot_id="request-query-casefold",
            umo=first.umo,
            cutoff_at=first.cutoff_at,
            message_upper_bound=first.message_upper_bound,
            request_source_key=first.request_source_key,
            sender_participant_key=first.sender_participant_key,
            reply_source_key=first.reply_source_key,
            query="  阐述阿拉蕾的挺点  ",
            context={"reply": "msg:700", "history": ["a", "b"]},
            data_revision=first.data_revision,
            inference_revision=first.inference_revision,
            captured_at=first.captured_at,
        )
        third = RequestSnapshot.create(
            snapshot_id="request-query-latin-case",
            umo=first.umo,
            cutoff_at=first.cutoff_at,
            message_upper_bound=first.message_upper_bound,
            request_source_key=first.request_source_key,
            sender_participant_key=first.sender_participant_key,
            reply_source_key=first.reply_source_key,
            query="Mujica   MEMORY",
            context={},
            data_revision=first.data_revision,
            inference_revision=first.inference_revision,
            captured_at=first.captured_at,
        )
        fourth = RequestSnapshot.create(
            snapshot_id="request-query-latin-lower",
            umo=first.umo,
            cutoff_at=first.cutoff_at,
            message_upper_bound=first.message_upper_bound,
            request_source_key=first.request_source_key,
            sender_participant_key=first.sender_participant_key,
            reply_source_key=first.reply_source_key,
            query="mujica memory",
            context={},
            data_revision=first.data_revision,
            inference_revision=first.inference_revision,
            captured_at=first.captured_at,
        )
        self.assertEqual(first.query_sha256, second.query_sha256)
        self.assertEqual(third.query_sha256, fourth.query_sha256)

    def test_cutoff_and_transaction_bound_are_both_enforced(self) -> None:
        snapshot = _snapshot()
        self.assertTrue(
            snapshot.allows_evidence(
                umo=snapshot.umo,
                sent_at=snapshot.cutoff_at - 1,
                message_row_id=726,
                source_key="msg:726",
            )
        )
        self.assertFalse(
            snapshot.allows_evidence(
                umo=snapshot.umo,
                sent_at=snapshot.cutoff_at,
                message_row_id=726,
                source_key="msg:726",
            )
        )
        self.assertFalse(
            snapshot.allows_evidence(
                umo=snapshot.umo,
                sent_at=snapshot.cutoff_at - 1,
                message_row_id=727,
                source_key="msg:726",
            )
        )
        self.assertFalse(
            snapshot.allows_evidence(
                umo=snapshot.umo,
                sent_at=snapshot.cutoff_at - 1,
                message_row_id=726,
                source_key="msg:727",
            )
        )
        self.assertFalse(
            snapshot.allows_evidence(
                umo="aiocqhttp:GroupMessage:other",
                sent_at=snapshot.cutoff_at - 1,
                message_row_id=726,
            )
        )

    def test_revision_vectors_reject_missing_and_unknown_fields(self) -> None:
        raw = _data_revision().as_dict()
        raw["future"] = "not-allowed"
        with self.assertRaisesRegex(ValueError, "unknown future"):
            DataRevisionVector.from_value(raw)
        raw = _inference_revision().as_dict()
        raw.pop("reader_protocol")
        with self.assertRaisesRegex(ValueError, "missing reader_protocol"):
            InferenceRevisionVector.from_value(raw)

    def test_snapshot_rejects_scope_or_hash_tampering(self) -> None:
        raw = _snapshot().as_dict()
        raw["umo"] = "aiocqhttp:GroupMessage:another"
        with self.assertRaisesRegex(ValueError, "scope_sha256"):
            RequestSnapshot.from_value(raw)
        raw = _snapshot().as_dict()
        raw["cutoff_mode"] = "latest"
        with self.assertRaisesRegex(ValueError, "unknown cutoff_mode"):
            RequestSnapshot.from_value(raw)

    def test_semantic_cache_key_survives_request_local_snapshot_changes(self) -> None:
        first = _snapshot()
        raw = first.as_dict()
        raw.update(
            {
                "snapshot_id": "request-727",
                "cutoff_at": first.cutoff_at + 60,
                "message_upper_bound": first.message_upper_bound + 1,
                "request_source_key": "msg:728",
                "captured_at": first.captured_at + 60,
                "data_revision": {
                    **first.data_revision.as_dict(),
                    "message": "13",
                },
            }
        )
        second = RequestSnapshot.from_value(raw)
        packet_hash = "a" * 64
        self.assertNotEqual(first.digest, second.digest)
        self.assertEqual(
            semantic_certificate_lookup_key(first, packet_sha256=packet_hash),
            semantic_certificate_lookup_key(second, packet_sha256=packet_hash),
        )
        self.assertNotEqual(
            semantic_certificate_lookup_key(first, packet_sha256=packet_hash),
            semantic_certificate_lookup_key(first, packet_sha256="b" * 64),
        )


if __name__ == "__main__":
    unittest.main()
