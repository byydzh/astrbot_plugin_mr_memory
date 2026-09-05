from __future__ import annotations

import copy
import json
import unittest
from dataclasses import replace

from mr_memory.certificate import parse_evidence_certificate
from mr_memory.surface import (
    ANSWER_CONTEXT_SCHEMA_VERSION,
    SurfaceCompilationError,
    compile_answer_context_packet,
    compile_surface_packet,
    validate_answer_context_packet,
    validate_surface_packet,
    verify_surface_answer,
)
from tests.test_certificate_v2 import _parse, _raw_certificate, _raw_packet_gap_certificate, _raw_typed_certificate, _raw_activity_certificate, _snapshot


def _partial_certificate():
    raw = _raw_certificate()
    raw.update(
        {
            "status": "PARTIAL",
            "stop_reason": "FRONTIER_EXHAUSTED",
            "unresolved": [
                {
                    "statement": "尚不能确定购买意向后来是否兑现。",
                    "source_keys": ["s2"],
                    "atom_ids": ["a2"],
                }
            ],
        }
    )
    return parse_evidence_certificate(
        raw,
        expected_snapshot=_snapshot(),
        expected_packet_sha256="a" * 64,
        allowed_source_keys={"s1", "s2"},
        allowed_participant_keys={"p1"},
        pack_read_complete=True,
        host_validated=True,
    )


class SurfaceCompilerTests(unittest.TestCase):
    def test_inference_retains_its_meaning_and_robot_evidence_role(self) -> None:
        raw, allowlist = _raw_activity_certificate()
        raw["atoms"][0]["attribution"] = "DERIVED_INTERPRETATION"
        raw["atoms"][0]["statement"] = "该时段有活动记录，但仅凭这些记录无法确定是否刚刚醒来。"
        certificate = _parse(raw, allowed_aggregates=allowlist)
        fact = compile_answer_context_packet(certificate).as_dict()["facts"][0]
        self.assertEqual(fact["statement"], raw["atoms"][0]["statement"])
        self.assertEqual(fact["attribution"], "inference")
        self.assertTrue(fact["statistics"])

        certificate = _parse(_raw_certificate())
        certificate = replace(certificate, atoms=tuple(
            replace(atom, evidence_roles=("BOT",)) for atom in certificate.atoms
        ))
        brief = compile_answer_context_packet(certificate)
        self.assertTrue(all(fact["evidence_roles"] == ["BOT"] for fact in brief.as_dict()["facts"]))

    def test_activity_brief_projects_complete_host_statistics_and_separate_provenance(self) -> None:
        raw, allowlist = _raw_activity_certificate()
        certificate = _parse(raw, allowed_aggregates=allowlist)
        surface = compile_surface_packet(certificate)
        validate_surface_packet(surface, certificate)
        brief = compile_answer_context_packet(certificate)
        validate_answer_context_packet(brief, certificate)
        payload = brief.as_dict()
        statistics = payload["activity_statistics"][0]
        descriptor = raw["aggregates"][0]
        self.assertEqual(statistics["source_count"], 3)
        self.assertEqual(statistics["daily"], descriptor["daily"])
        self.assertEqual(statistics["hour_histogram"], descriptor["hour_histogram"])
        self.assertIn("不证明醒来、入睡或发言原文", statistics["interpretation_limit"])
        self.assertEqual(payload["facts"][0]["statistics"], [statistics["id"]])
        self.assertEqual(payload["facts"][0]["speaker"], "")
        self.assertEqual(brief.included_source_keys, ())
        self.assertEqual(brief.included_aggregate_ids, (descriptor["aggregate_id"],))
        self.assertEqual(brief.included_aggregate_metadata[0]["source_revision_sha256"], descriptor["source_revision_sha256"])
        self.assertNotIn(descriptor["aggregate_id"], brief.text)
        self.assertNotIn(descriptor["source_revision_sha256"], brief.text)

    def test_brief_preserves_packet_gap_as_uncertainty_without_fact_or_source(self) -> None:
        certificate = _parse(_raw_packet_gap_certificate())
        surface = compile_surface_packet(certificate)
        validate_surface_packet(surface, certificate)
        self.assertEqual(surface.as_dict()["contract"]["unresolved"][0]["basis"], "PACKET_COVERAGE_GAP")
        brief = compile_answer_context_packet(certificate)
        validate_answer_context_packet(brief, certificate)
        payload = brief.as_dict()
        gap = payload["qualifications"]["unresolved"][0]
        self.assertEqual(payload["memory_state"], "qualified")
        self.assertEqual(gap, {
            "statement": certificate.unresolved[0].statement,
            "kind": "coverage_gap", "scope": "current_evidence_packet",
            "not_evidence_of_absence": True,
        })
        self.assertEqual(len(payload["facts"]), 1)
        self.assertEqual(brief.included_source_keys, ("s1",))
        self.assertEqual(brief.included_atom_ids, ("a1",))
        self.assertEqual(payload["qualifications"]["open_questions"], [{
            "question": certificate.open_obligations[0].question, "critical": True,
        }])
        self.assertNotIn(certificate.packet_sha256, brief.text)

    def test_typed_brief_preserves_distinct_subjects_time_authorship_and_metadata(self) -> None:
        raw = _raw_typed_certificate()
        # A display name and a work title may be identical without sharing identity.
        raw["subjects"][0]["reference"] = "灯塔"
        raw["referents"][0]["reference"] = "灯塔"
        # A third-party source author is not a semantic subject.
        raw["atoms"][1]["speaker_participant_key"] = "p2"
        raw["atoms"][1]["attribution"] = "OTHER_SPEAKER_REPORT"
        certificate = _parse(raw)
        internal = compile_surface_packet(certificate)
        validate_surface_packet(internal, certificate)
        self.assertEqual(len(internal.as_dict()["referents"]), 4)
        packet = compile_answer_context_packet(certificate)
        validate_answer_context_packet(packet, certificate)
        payload = packet.as_dict()
        by_entity = {item["entity"]: item for item in payload["referents"]}
        self.assertEqual(
            [by_entity[item["subject"]]["type"] for item in payload["facts"]],
            ["participant", "work", "entity", "topic"],
        )
        self.assertEqual(len({item["subject"] for item in payload["facts"]}), 4)
        self.assertEqual(by_entity[payload["facts"][1]["speaker"]]["resolution"], "unlabeled")
        self.assertEqual(payload["facts"][1]["attribution"], "reported_statement")
        self.assertTrue(by_entity[payload["facts"][1]["subject"]]["valid_at"].endswith("Z"))
        self.assertEqual(packet.included_atom_ids, tuple(raw["must_include"]))
        self.assertEqual(set(packet.included_source_keys), {"s1", "s2"})
        self.assertEqual(len(packet.included_referent_ids), 4)
        for private_value in ("referent-work", "仅供内部审计的合成原文", "source_keys", "confidence"):
            self.assertNotIn(private_value, packet.text)

    def test_brief_includes_optional_fact_needed_to_understand_qualification(self) -> None:
        raw = _raw_certificate()
        raw["unresolved"] = [{
            "statement": "候选日期仍有两个，不能当成已确认安排。",
            "source_keys": ["s2"], "atom_ids": ["a2"],
        }]
        raw["open_obligations"] = [{
            "id": "calendar-check", "question": "需要确认哪个候选日期？", "critical": False,
            "competing_interpretation_ids": [], "discriminator": "", "expected_information_gain": "",
        }]
        packet = compile_answer_context_packet(_parse(raw))
        payload = packet.as_dict()
        self.assertEqual(packet.included_atom_ids, ("a1", "a2"))
        self.assertEqual(payload["qualifications"]["unresolved"][0]["facts"], ["fact_2"])
        self.assertEqual(payload["qualifications"]["open_questions"][0]["question"], "需要确认哪个候选日期？")

    def test_answer_context_exposes_summary_not_internal_evidence(self) -> None:
        raw = copy.deepcopy(_raw_certificate())
        private_participant = "participant-private-sentinel"
        private_source = "source-private-sentinel"
        private_atom_id = "atom-private-sentinel"
        raw["subjects"] = [
            {
                "reference": "public-reference-sentinel",
                "participant_key": private_participant,
                "reference_mode": "UNIQUE_ALIAS",
                "candidate_participant_keys": [],
                "source_keys": [private_source],
                "valid_at": None,
            }
        ]
        raw["atoms"] = [
            {
                "id": private_atom_id,
                "statement": "public-fact-sentinel",
                "speaker_participant_key": private_participant,
                "subject_participant_key": private_participant,
                "attribution": "DIRECT_SPEAKER_STATEMENT",
                "stance": "SUPPORTED",
                "source_keys": [private_source],
                "source_spans": ["private-dialogue-sentinel"],
                "importance": "REQUIRED",
                "confidence": 0.91,
            },
            {
                "id": "optional-private-atom",
                "statement": "optional-history-sentinel",
                "speaker_participant_key": private_participant,
                "subject_participant_key": private_participant,
                "attribution": "DIRECT_SPEAKER_STATEMENT",
                "stance": "SUPPORTED",
                "source_keys": [private_source],
                "source_spans": ["optional-private-dialogue"],
                "importance": "OPTIONAL",
                "confidence": 0.73,
            },
        ]
        raw["must_include"] = [private_atom_id]
        raw["must_not_upgrade"] = []
        raw["unresolved"] = []
        certificate = parse_evidence_certificate(
            raw,
            expected_snapshot=_snapshot(),
            expected_packet_sha256="a" * 64,
            allowed_source_keys={private_source},
            allowed_participant_keys={private_participant},
            pack_read_complete=True,
            host_validated=True,
        )

        packet = compile_answer_context_packet(certificate)
        validate_answer_context_packet(packet, certificate)
        payload = packet.as_dict()

        self.assertEqual(
            payload["schema_version"],
            ANSWER_CONTEXT_SCHEMA_VERSION,
        )
        self.assertEqual(
            payload["referents"][0]["mention"],
            "public-reference-sentinel",
        )
        self.assertEqual(
            payload["facts"][0]["statement"],
            "public-fact-sentinel",
        )
        for private_value in (
            private_participant,
            private_source,
            private_atom_id,
            "private-dialogue-sentinel",
            "optional-history-sentinel",
            certificate.digest,
            certificate.scope_snapshot.umo,
        ):
            self.assertNotIn(private_value, packet.text)

    def test_compiler_keeps_contract_and_validates_round_trip(self) -> None:
        certificate = _partial_certificate()
        packet = compile_surface_packet(certificate, max_chars=100_000)
        validate_surface_packet(packet, certificate)
        payload = packet.as_dict()
        self.assertEqual(
            [item["id"] for item in payload["evidence"]["required"]],
            ["a1"],
        )
        self.assertEqual(payload["contract"]["must_include"], ["a1"])
        self.assertEqual(len(payload["contract"]["must_not_upgrade"]), 1)
        self.assertEqual(len(payload["contract"]["unresolved"]), 1)
        self.assertEqual(packet.omitted_optional, 0)

    def test_only_optional_atoms_are_removed_to_fit(self) -> None:
        certificate = _partial_certificate()
        full = compile_surface_packet(certificate, max_chars=100_000)
        bounded = compile_surface_packet(
            certificate,
            max_chars=len(full.text) - 1,
        )
        validate_surface_packet(bounded, certificate)
        payload = bounded.as_dict()
        self.assertEqual(payload["evidence"]["optional"], [])
        self.assertEqual(packet_ids(payload["evidence"]["required"]), ["a1"])
        self.assertEqual(bounded.omitted_optional, 1)

    def test_oversized_optional_does_not_block_later_small_atom(self) -> None:
        raw = _raw_certificate()
        required = copy.deepcopy(raw["atoms"][0])
        required.update(
            {
                "id": "required_core",
                "statement": "合成核心事实必须保留。",
                "source_spans": ["核心事实"],
            }
        )
        oversized = copy.deepcopy(raw["atoms"][1])
        oversized.update(
            {
                "id": "optional_oversized",
                "statement": "大" * 1_500,
                "source_spans": ["大型可选证据"],
            }
        )
        small = copy.deepcopy(raw["atoms"][1])
        small.update(
            {
                "id": "optional_small",
                "statement": "后续短小的合成可选事实。",
                "source_spans": ["短小可选证据"],
            }
        )
        raw["subjects"][0]["reference"] = "参与者甲"
        raw["atoms"] = [required, oversized, small]
        raw["must_include"] = ["required_core"]
        raw["must_not_upgrade"] = [
            {
                "observed": "愿意尝试",
                "forbidden": ["已经完成"],
                "atom_ids": ["required_core"],
                "reason": "合成观察不等于已经完成。",
            }
        ]

        def parse_variant(atoms: list[dict[str, object]]):
            variant = copy.deepcopy(raw)
            variant["atoms"] = atoms
            return parse_evidence_certificate(
                variant,
                expected_snapshot=_snapshot(),
                expected_packet_sha256="a" * 64,
                allowed_source_keys={"s1", "s2"},
                allowed_participant_keys={"p1"},
                pack_read_complete=True,
                host_validated=True,
            )

        certificate = parse_variant([required, oversized, small])
        small_only = parse_variant([required, small])
        oversized_only = parse_variant([required, oversized])
        bound = len(compile_surface_packet(small_only, max_chars=100_000).text)
        self.assertGreater(
            len(compile_surface_packet(oversized_only, max_chars=100_000).text),
            bound,
        )

        packet = compile_surface_packet(certificate, max_chars=bound)

        validate_surface_packet(packet, certificate)
        self.assertLessEqual(len(packet.text), bound)
        self.assertEqual(packet.included_optional_atom_ids, ("optional_small",))
        self.assertEqual(packet.omitted_optional, 1)
        self.assertGreater(
            len(compile_surface_packet(certificate, max_chars=100_000).text),
            bound,
        )

    def test_compiler_fails_closed_if_mandatory_contract_does_not_fit(self) -> None:
        with self.assertRaisesRegex(SurfaceCompilationError, "refusing truncation"):
            compile_surface_packet(_partial_certificate(), max_chars=80)

    def test_validator_rejects_tampered_required_statement(self) -> None:
        certificate = _partial_certificate()
        packet = compile_surface_packet(certificate, max_chars=100_000)
        raw = packet.as_dict()
        raw["evidence"]["required"][0]["statement"] = "已经确定买了。"
        tampered = replace(
            packet,
            text=json.dumps(raw, ensure_ascii=False, separators=(",", ":")),
        )
        with self.assertRaisesRegex(SurfaceCompilationError, "required atom"):
            validate_surface_packet(tampered, certificate)

    def test_lexical_match_does_not_claim_semantic_acceptance(self) -> None:
        certificate = _partial_certificate()
        answer = (
            "合成人物甲曾表示自己已经玩腻拼图。"
            "尚不能确定购买意向后来是否兑现。"
        )
        result = verify_surface_answer(answer, certificate)
        self.assertIsNone(result.passed)
        self.assertTrue(result.lexical_checks_passed, result.as_dict())
        self.assertEqual(result.as_dict()["semantic_status"], "NOT_EVALUATED")
        self.assertEqual(result.required_matched, 1)
        self.assertEqual(result.unresolved_retained, 1)

    def test_shadow_verifier_catches_upgrade_uncertainty_and_attribution_loss(self) -> None:
        certificate = _partial_certificate()
        upgraded = verify_surface_answer(
            "合成人物甲说拼图玩腻了，但后来已经付款。",
            certificate,
        )
        self.assertIsNone(upgraded.passed)
        self.assertFalse(upgraded.lexical_checks_passed)
        self.assertEqual(upgraded.forbidden_upgrades, ("已经付款",))
        self.assertEqual(len(upgraded.missing_unresolved), 1)

        unattributed = verify_surface_answer(
            "拼图玩腻了。尚不能确定购买意向后来是否兑现。",
            certificate,
        )
        self.assertIsNone(unattributed.passed)
        # Copying a raw excerpt is not evidence that its certified meaning survived.
        self.assertEqual(unattributed.required_matched, 0)
        self.assertFalse(unattributed.lexical_checks_passed)


def packet_ids(values: list[dict[str, object]]) -> list[object]:
    return [item.get("id") for item in values]


if __name__ == "__main__":
    unittest.main()
