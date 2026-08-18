from __future__ import annotations

import copy
import json
import unittest

from mr_memory.certificate import (
    MAX_CERTIFICATE_CONFLICTS,
    MAX_CERTIFICATE_SOURCE_KEYS,
    MAX_CERTIFICATE_UNRESOLVED,
    parse_evidence_certificate,
)
from mr_memory.snapshot import (
    DataRevisionVector,
    InferenceRevisionVector,
    RequestSnapshot,
)


def _snapshot() -> RequestSnapshot:
    return RequestSnapshot.create(
        snapshot_id="snap-good-girl",
        umo="aiocqhttp:GroupMessage:42",
        cutoff_at=2_000,
        message_upper_bound=99,
        request_source_key="msg:100",
        sender_participant_key="p1",
        reply_source_key="msg:90",
        query="好女孩是什么意思",
        context={"reply": "msg:90"},
        data_revision=DataRevisionVector.from_value(
            {
                "message": 99,
                "deletion": 1,
                "identity": 5,
                "graph": 8,
                "relation": 3,
                "feedback": 2,
            }
        ),
        inference_revision=InferenceRevisionVector.from_value(
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
        ),
        captured_at=2_001,
    )


def _raw_certificate() -> dict[str, object]:
    snapshot = _snapshot()
    return {
        "schema_version": "evidence-certificate.v2",
        "status": "CERTIFIED",
        "scope_snapshot": snapshot.as_dict(),
        "data_revision": snapshot.data_revision.as_dict(),
        "inference_revision": snapshot.inference_revision.as_dict(),
        "packet_sha256": "a" * 64,
        "subjects": [
            {
                "reference": "byy",
                "participant_key": "p1",
                "reference_mode": "HOST",
                "candidate_participant_keys": [],
                "source_keys": [],
                "valid_at": None,
            }
        ],
        "atoms": [
            {
                "id": "a1",
                "statement": "byy早期明确说自己已经玩腻类魂。",
                "speaker_participant_key": "p1",
                "subject_participant_key": "p1",
                "attribution": "DIRECT_SPEAKER_STATEMENT",
                "stance": "SUPPORTED",
                "source_keys": ["s1"],
                "source_spans": ["类魂玩吐了"],
                "importance": "REQUIRED",
                "confidence": 0.94,
            },
            {
                "id": "a2",
                "statement": "后来又表达过购买意向。",
                "speaker_participant_key": "p1",
                "subject_participant_key": "p1",
                "attribution": "DIRECT_SPEAKER_STATEMENT",
                "stance": "SUPPORTED",
                "source_keys": ["s2"],
                "source_spans": ["我可能会买"],
                "importance": "OPTIONAL",
                "confidence": 0.76,
            },
        ],
        "must_include": ["a1"],
        "must_not_upgrade": [
            {
                "observed": "表达购买意向",
                "forbidden": ["已经付款", "已经预购", "一定会首发购买"],
                "atom_ids": ["a1"],
                "reason": "意向不等于已经完成购买。",
            }
        ],
        "conflicts": [],
        "unresolved": [],
        "open_obligations": [],
        "stop_reason": "CERTIFIED_CLOSE",
        "validation": {
            "pack_read_complete": True,
            "host_validated": True,
        },
    }


def _parse(raw: object):
    return parse_evidence_certificate(
        raw,
        expected_snapshot=_snapshot(),
        expected_packet_sha256="a" * 64,
        allowed_source_keys={"s1", "s2"},
        allowed_participant_keys={"p1", "p2", "p3"},
        pack_read_complete=True,
        host_validated=True,
    )


class EvidenceCertificateV2Tests(unittest.TestCase):
    def test_valid_certificate_round_trips_with_required_anchor(self) -> None:
        certificate = _parse(_raw_certificate())
        self.assertEqual(certificate.status, "CERTIFIED")
        self.assertEqual(
            [item.atom_id for item in certificate.required_atoms],
            ["a1"],
        )
        self.assertEqual(_parse(certificate.as_dict()), certificate)
        self.assertEqual(len(certificate.digest), 64)

    def test_parser_is_strict_about_json_envelope_and_fields(self) -> None:
        raw = json.dumps(_raw_certificate(), ensure_ascii=False)
        with self.assertRaisesRegex(ValueError, "exactly one JSON object"):
            _parse("analysis follows\n" + raw)
        with self.assertRaisesRegex(ValueError, "unknown surprise"):
            changed = _raw_certificate()
            changed["surprise"] = True
            _parse(changed)
        self.assertEqual(_parse(f"```json\n{raw}\n```").status, "CERTIFIED")

    def test_model_cannot_change_scope_revision_or_source_allowlist(self) -> None:
        changed = _raw_certificate()
        changed["scope_snapshot"]["cutoff_at"] = 2_001
        with self.assertRaisesRegex(ValueError, "host snapshot"):
            _parse(changed)
        changed = _raw_certificate()
        changed["data_revision"]["graph"] = "later"
        with self.assertRaisesRegex(ValueError, "data_revision"):
            _parse(changed)
        changed = _raw_certificate()
        changed["atoms"][0]["source_keys"] = ["not-delivered"]
        with self.assertRaisesRegex(ValueError, "allowlist"):
            _parse(changed)

    def test_source_lists_preserve_up_to_contract_limit_without_truncation(self) -> None:
        source_keys = [
            f"source-{index}" for index in range(MAX_CERTIFICATE_SOURCE_KEYS)
        ]
        source_spans = [
            f"span-{index}" for index in range(MAX_CERTIFICATE_SOURCE_KEYS)
        ]
        raw = _raw_certificate()
        raw["atoms"][0]["source_keys"] = source_keys
        raw["atoms"][0]["source_spans"] = source_spans
        raw["unresolved"] = [
            {
                "statement": "所有来源都属于同一项尚未解决的证据链。",
                "source_keys": source_keys,
                "atom_ids": [],
            }
        ]
        certificate = parse_evidence_certificate(
            raw,
            expected_snapshot=_snapshot(),
            expected_packet_sha256="a" * 64,
            allowed_source_keys={"s1", "s2", *source_keys},
            allowed_participant_keys={"p1", "p2", "p3"},
            pack_read_complete=True,
            host_validated=True,
        )
        self.assertEqual(certificate.atoms[0].source_keys, tuple(source_keys))
        self.assertEqual(certificate.atoms[0].source_spans, tuple(source_spans))
        self.assertEqual(certificate.unresolved[0].source_keys, tuple(source_keys))

        raw["unresolved"][0]["source_keys"] = [*source_keys, "source-over-limit"]
        with self.assertRaisesRegex(ValueError, "exceeds 64 items"):
            parse_evidence_certificate(
                raw,
                expected_snapshot=_snapshot(),
                expected_packet_sha256="a" * 64,
                allowed_source_keys={"s1", "s2", *source_keys, "source-over-limit"},
                allowed_participant_keys={"p1", "p2", "p3"},
                pack_read_complete=True,
                host_validated=True,
            )

    def test_conflict_and_unresolved_caps_enforce_exact_boundaries(self) -> None:
        self.assertEqual(MAX_CERTIFICATE_CONFLICTS, 32)
        self.assertEqual(MAX_CERTIFICATE_UNRESOLVED, 64)

        def qualification(kind: str, index: int) -> dict[str, object]:
            return {
                "statement": f"{kind}-{index}",
                "source_keys": ["s1"],
                "atom_ids": [],
            }

        raw = _raw_certificate()
        raw["conflicts"] = [
            qualification("conflict", index)
            for index in range(MAX_CERTIFICATE_CONFLICTS)
        ]
        raw["unresolved"] = [
            qualification("unresolved", index)
            for index in range(MAX_CERTIFICATE_UNRESOLVED)
        ]
        certificate = _parse(raw)
        self.assertEqual(len(certificate.conflicts), 32)
        self.assertEqual(len(certificate.unresolved), 64)

        too_many_conflicts = copy.deepcopy(raw)
        too_many_conflicts["conflicts"].append(
            qualification("conflict", MAX_CERTIFICATE_CONFLICTS)
        )
        with self.assertRaisesRegex(ValueError, "conflicts.*at most 32 items"):
            _parse(too_many_conflicts)

        too_many_unresolved = copy.deepcopy(raw)
        too_many_unresolved["unresolved"].append(
            qualification("unresolved", MAX_CERTIFICATE_UNRESOLVED)
        )
        with self.assertRaisesRegex(ValueError, "unresolved.*at most 64 items"):
            _parse(too_many_unresolved)

    def test_required_atoms_and_upgrade_guards_must_reference_known_atoms(self) -> None:
        changed = _raw_certificate()
        changed["must_include"] = []
        with self.assertRaisesRegex(ValueError, "every and only REQUIRED"):
            _parse(changed)
        changed = _raw_certificate()
        changed["must_not_upgrade"][0]["atom_ids"] = ["not-an-atom"]
        with self.assertRaisesRegex(ValueError, "unknown atom"):
            _parse(changed)

    def test_certified_rejects_ambiguous_identity(self) -> None:
        changed = _raw_certificate()
        changed["subjects"][0] = {
            "reference": "新昵称",
            "participant_key": "",
            "reference_mode": "AMBIGUOUS",
            "candidate_participant_keys": ["p2", "p3"],
            "source_keys": ["s1"],
            "valid_at": 1_999,
        }
        with self.assertRaisesRegex(ValueError, "identity ambiguity"):
            _parse(changed)

    def test_semantic_none_requires_complete_host_validation(self) -> None:
        raw = _raw_certificate()
        raw.update(
            {
                "status": "SEMANTIC_NONE",
                "subjects": [],
                "atoms": [],
                "must_include": [],
                "must_not_upgrade": [],
                "stop_reason": "SEMANTIC_NONE",
            }
        )
        self.assertEqual(_parse(raw).status, "SEMANTIC_NONE")
        raw = copy.deepcopy(raw)
        raw["validation"]["pack_read_complete"] = False
        with self.assertRaisesRegex(ValueError, "cannot choose"):
            _parse(raw)

    def test_request_l3_requires_actionable_discriminator(self) -> None:
        raw = _raw_certificate()
        raw.update(
            {
                "status": "REQUEST_L3",
                "stop_reason": "REQUEST_L3",
                "open_obligations": [
                    {
                        "id": "o1",
                        "question": "这个称呼是否在同一段对话中发生反讽反转？",
                        "critical": True,
                        "competing_interpretation_ids": ["literal", "ironic"],
                        "discriminator": "查找被称呼者随后的否认或负反馈",
                        "expected_information_gain": "区分字面夸奖和群内反讽",
                    }
                ],
            }
        )
        self.assertEqual(_parse(raw).status, "REQUEST_L3")
        raw["open_obligations"][0]["discriminator"] = ""
        with self.assertRaisesRegex(ValueError, "discriminator"):
            _parse(raw)


if __name__ == "__main__":
    unittest.main()
