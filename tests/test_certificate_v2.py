from __future__ import annotations

import copy
import json
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from mr_memory.activity_statistics import activity_aggregate_id

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
        snapshot_id="snap-synthetic-plan",
        umo="aiocqhttp:GroupMessage:42",
        cutoff_at=2_000,
        message_upper_bound=99,
        request_source_key="msg:100",
        sender_participant_key="p1",
        reply_source_key="msg:90",
        query="纸鹤计划进展如何",
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
                "reference": "合成人物甲",
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
                "statement": "合成人物甲曾表示自己已经玩腻拼图。",
                "speaker_participant_key": "p1",
                "subject_participant_key": "p1",
                "attribution": "DIRECT_SPEAKER_STATEMENT",
                "stance": "SUPPORTED",
                "source_keys": ["s1"],
                "source_spans": ["拼图玩腻了"],
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


def _parse(raw: object, *, allowed_aggregates=None, source_roles=None):
    return parse_evidence_certificate(
        raw,
        expected_snapshot=_snapshot(),
        expected_packet_sha256="a" * 64,
        allowed_source_keys={"s1", "s2"},
        allowed_participant_keys={"p1", "p2", "p3"},
        pack_read_complete=True,
        host_validated=True,
        allowed_aggregates=allowed_aggregates,
        source_roles=source_roles,
    )


def _raw_typed_certificate():
    raw = _raw_certificate()
    raw["subjects"] = [{
        "reference": "合成记录员", "participant_key": "p1", "reference_mode": "HOST",
        "candidate_participant_keys": [], "source_keys": ["s1"], "valid_at": 1800,
    }]
    raw["referents"] = [
        {"id": "referent-person", "referent_type": "PARTICIPANT", **raw["subjects"][0]},
        *[{
            "id": f"referent-{kind.casefold()}", "reference": "灯塔",
            "referent_type": kind, "participant_key": "", "reference_mode": "EVIDENCE_REF",
            "candidate_participant_keys": [], "source_keys": ["s2"], "valid_at": 1850,
        } for kind in ("WORK", "ENTITY", "TOPIC")],
    ]
    raw["atoms"] = [{
        "id": f"typed-{index}", "statement": f"合成事实 {index} 的对象有独立指称。",
        "speaker_participant_key": "p1", "subject_participant_key": item["participant_key"],
        "subject_referent_id": item["id"], "attribution": "DIRECT_SPEAKER_STATEMENT",
        "stance": "SUPPORTED", "source_keys": ["s1"], "source_spans": ["仅供内部审计的合成原文"],
        "importance": "REQUIRED", "confidence": 0.8,
    } for index, item in enumerate(raw["referents"])]
    raw["must_include"] = [item["id"] for item in raw["atoms"]]
    raw["must_not_upgrade"] = []
    return raw


def _raw_packet_gap_certificate():
    raw = _raw_certificate()
    raw.update(status="PARTIAL", stop_reason="FRONTIER_EXHAUSTED", must_not_upgrade=[])
    raw["atoms"] = [raw["atoms"][0]]
    raw["atoms"][0].update(statement="合成人物甲表示纸鹤活动的纸张已经到齐。", source_spans=["纸张已经到齐"])
    raw["unresolved"] = [{
        "statement": "本次证据包不足以判断纸鹤活动的场地是否已经确认。",
        "source_keys": [], "atom_ids": [],
    }]
    raw["open_obligations"] = [{
        "id": "venue-confirmation", "question": "纸鹤活动的场地是否已经确认？", "critical": True,
        "competing_interpretation_ids": [], "discriminator": "", "expected_information_gain": "",
    }]
    return raw


def _activity_descriptor():
    zone = ZoneInfo("Asia/Shanghai")
    first, last = 1000, 1900
    descriptor = {
        "schema_version": "mr-memory.activity-window.v1", "authority": "HOST_SQLITE_SNAPSHOT",
        "basis": "all_snapshot_visible_direct_speaker_messages", "timezone": "Asia/Shanghai",
        "scope": {"umo": _snapshot().umo, "participant_key": "p1", "start_sent_at": 0,
                  "end_sent_at_exclusive": _snapshot().cutoff_at, "message_upper_bound": _snapshot().message_upper_bound},
        "source_count": 3, "hour_histogram": {f"{hour:02}": 3 if hour == 8 else 0 for hour in range(24)},
        "daily": [{"local_date": "1970-01-01", "source_count": 3, "first_sent_at": first, "last_sent_at": last,
                   "first_local_datetime": datetime.fromtimestamp(first, zone).isoformat(),
                   "last_local_datetime": datetime.fromtimestamp(last, zone).isoformat()}],
        "source_revision_sha256": "c" * 64,
    }
    descriptor["aggregate_id"] = activity_aggregate_id(descriptor)
    return descriptor


def _raw_activity_certificate():
    raw = _raw_certificate()
    descriptor = _activity_descriptor()
    raw["aggregates"] = [descriptor]
    raw["atoms"] = [raw["atoms"][0]]
    raw["atoms"][0].update(
        statement="该观察窗口内共有三条本人发言。", attribution="HOST_ACTIVITY_STATISTIC",
        speaker_participant_key="", source_keys=[], source_spans=[], aggregate_ids=[descriptor["aggregate_id"]],
    )
    raw["must_not_upgrade"] = []
    return raw, {descriptor["aggregate_id"]: descriptor}


class EvidenceCertificateV2Tests(unittest.TestCase):
    def test_source_roles_round_trip_and_missing_legacy_roles_remain_unknown(self) -> None:
        raw = _raw_certificate()
        self.assertEqual(_parse(raw).atoms[0].evidence_roles, ("UNKNOWN",))
        raw["atoms"][0]["evidence_roles"] = ["BOT"]
        raw["atoms"][1]["evidence_roles"] = ["USER"]
        roles = {"s1": "BOT", "s2": "USER"}
        certificate = _parse(raw, source_roles=roles)
        self.assertEqual(_parse(certificate.as_dict()), certificate)
        raw["atoms"][0]["evidence_roles"] = ["USER"]
        with self.assertRaisesRegex(ValueError, "differ from host source metadata"):
            _parse(raw, source_roles=roles)

    def test_host_activity_aggregate_is_snapshot_bound_without_fabricated_raw_sources(self) -> None:
        raw, allowlist = _raw_activity_certificate()
        certificate = _parse(raw, allowed_aggregates=allowlist)
        self.assertEqual(certificate.atoms[0].source_keys, ())
        self.assertEqual(certificate.atoms[0].speaker_participant_key, "")
        self.assertEqual(certificate.aggregates[0].as_dict(), raw["aggregates"][0])
        self.assertEqual(_parse(certificate.as_dict(), allowed_aggregates=allowlist), certificate)
        self.assertEqual(_parse(_raw_certificate()).as_dict(), _raw_certificate())
        with self.assertRaisesRegex(ValueError, "host allowlist"):
            _parse(raw)

    def test_activity_aggregate_rejects_tampering_cross_scope_subject_and_direct_speech(self) -> None:
        raw, allowlist = _raw_activity_certificate()
        raw["aggregates"][0]["source_count"] = 4
        with self.assertRaisesRegex(ValueError, "content hash"):
            _parse(raw, allowed_aggregates=allowlist)
        raw, allowlist = _raw_activity_certificate()
        raw["atoms"][0]["subject_participant_key"] = "p2"
        with self.assertRaisesRegex(ValueError, "subject differs"):
            _parse(raw, allowed_aggregates=allowlist)
        raw, allowlist = _raw_activity_certificate()
        raw["atoms"][0]["attribution"] = "DIRECT_SPEAKER_STATEMENT"
        with self.assertRaisesRegex(ValueError, "requires HOST_ACTIVITY_STATISTIC"):
            _parse(raw, allowed_aggregates=allowlist)
        raw, allowlist = _raw_activity_certificate()
        descriptor = raw["aggregates"][0]
        descriptor["scope"]["umo"] = "synthetic-other-scope"
        descriptor["aggregate_id"] = activity_aggregate_id(descriptor)
        with self.assertRaisesRegex(ValueError, "scope differs"):
            _parse(raw, allowed_aggregates=allowlist)

    def test_aggregate_backed_derivation_retains_host_evidence_and_derived_attribution(self) -> None:
        raw, allowlist = _raw_activity_certificate()
        raw["atoms"][0].update(attribution="DERIVED_INTERPRETATION",
            statement="观察窗口内的发言集中在早间；其线下活动时间仍不确定。")
        certificate = _parse(raw, allowed_aggregates=allowlist)
        self.assertEqual(certificate.atoms[0].attribution, "DERIVED_INTERPRETATION")
        self.assertEqual(certificate.aggregates[0].as_dict(), raw["aggregates"][0])
        self.assertEqual(_parse(certificate.as_dict(), allowed_aggregates=allowlist), certificate)
        for field, value, error in (
            ("aggregate_ids", ["activity-window:" + "e" * 64], "host allowlist"),
            ("speaker_participant_key", "p1", "cannot assert a speaker"),
            ("subject_participant_key", "p2", "subject differs"),
        ):
            changed = copy.deepcopy(raw)
            changed["atoms"][0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, error):
                _parse(changed, allowed_aggregates=allowlist)

    def test_uncited_unresolved_round_trips_as_a_packet_bound_coverage_gap(self) -> None:
        raw = _raw_packet_gap_certificate()
        certificate = _parse(raw)
        gap = certificate.unresolved[0]
        self.assertEqual(gap.basis, "PACKET_COVERAGE_GAP")
        self.assertEqual((gap.source_keys, gap.atom_ids), ((), ()))
        self.assertEqual(gap.statement, raw["unresolved"][0]["statement"])
        self.assertEqual(_parse(certificate.as_dict()), certificate)
        self.assertEqual(certificate.scope_snapshot, _snapshot())
        self.assertEqual(certificate.packet_sha256, "a" * 64)
        tampered = certificate.as_dict()
        tampered["packet_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "packet_sha256"):
            _parse(tampered)

    def test_coverage_gaps_cannot_be_certified_conflicts_or_cited_facts(self) -> None:
        raw = _raw_packet_gap_certificate()
        raw.update(status="CERTIFIED", stop_reason="CERTIFIED_CLOSE")
        with self.assertRaisesRegex(ValueError, "CERTIFIED cannot retain packet coverage gaps"):
            _parse(raw)
        raw = _raw_packet_gap_certificate()
        raw["conflicts"] = copy.deepcopy(raw["unresolved"])
        with self.assertRaisesRegex(ValueError, "conflicts.*requires source_keys or atom_ids"):
            _parse(raw)
        raw = _raw_packet_gap_certificate()
        raw["unresolved"][0].update(basis="PACKET_COVERAGE_GAP", source_keys=["s1"])
        with self.assertRaisesRegex(ValueError, "gap cannot carry evidence references"):
            _parse(raw)
        raw["unresolved"][0].update(basis="EVIDENCE", source_keys=[])
        with self.assertRaisesRegex(ValueError, "requires source_keys or atom_ids"):
            _parse(raw)

    def test_certified_cannot_claim_success_with_only_optional_evidence(self) -> None:
        raw = _raw_certificate()
        for atom in raw["atoms"]:
            atom["importance"] = "OPTIONAL"
        raw["must_include"] = []
        with self.assertRaisesRegex(ValueError, "at least one REQUIRED atom"):
            _parse(raw)

    def test_typed_referents_round_trip_without_rebinding_same_named_subjects(self) -> None:
        raw = _raw_typed_certificate()
        certificate = _parse(raw)
        self.assertEqual(_parse(certificate.as_dict()), certificate)
        self.assertEqual([item.referent_type for item in certificate.referents],
                         ["PARTICIPANT", "WORK", "ENTITY", "TOPIC"])
        self.assertEqual([atom.subject_referent_id for atom in certificate.atoms],
                         [item["id"] for item in raw["referents"]])
        self.assertEqual(certificate.referents[1].valid_at, 1850)
        # The extension does not change old persisted certificate digests.
        legacy = _raw_certificate()
        self.assertEqual(_parse(legacy).as_dict(), legacy)

    def test_typed_subject_links_reject_orphans_cross_binding_and_person_coercion(self) -> None:
        for mutate, error in (
            (lambda raw: raw["atoms"][1].update(subject_referent_id="missing"), "unknown referent"),
            (lambda raw: raw["atoms"][1].update(subject_participant_key="p1"), "differs from its typed"),
            (lambda raw: raw["referents"][1].update(participant_key="p1"), "non-participant"),
            (lambda raw: raw["subjects"][0].update(reference="changed label"), "typed participant projection"),
        ):
            raw = _raw_typed_certificate()
            mutate(raw)
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                _parse(raw)

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
