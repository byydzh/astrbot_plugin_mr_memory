from __future__ import annotations

import json
import unittest
from dataclasses import replace

from mr_memory.certificate import parse_evidence_certificate
from mr_memory.surface import (
    SurfaceCompilationError,
    compile_surface_packet,
    validate_surface_packet,
    verify_surface_answer,
)
from tests.test_certificate_v2 import _raw_certificate, _snapshot


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

    def test_shadow_verifier_accepts_attributed_qualified_answer(self) -> None:
        certificate = _partial_certificate()
        answer = (
            "byy早期明确说自己已经玩腻类魂。"
            "尚不能确定购买意向后来是否兑现。"
        )
        result = verify_surface_answer(answer, certificate)
        self.assertTrue(result.passed, result.as_dict())
        self.assertEqual(result.required_matched, 1)
        self.assertEqual(result.unresolved_retained, 1)

    def test_shadow_verifier_catches_upgrade_uncertainty_and_attribution_loss(self) -> None:
        certificate = _partial_certificate()
        upgraded = verify_surface_answer(
            "byy说类魂玩吐了，但后来已经付款。",
            certificate,
        )
        self.assertFalse(upgraded.passed)
        self.assertEqual(upgraded.forbidden_upgrades, ("已经付款",))
        self.assertEqual(len(upgraded.missing_unresolved), 1)

        unattributed = verify_surface_answer(
            "类魂玩吐了。尚不能确定购买意向后来是否兑现。",
            certificate,
        )
        self.assertFalse(unattributed.passed)
        self.assertEqual(unattributed.attribution_violations, ("a1:p1",))


def packet_ids(values: list[dict[str, object]]) -> list[object]:
    return [item.get("id") for item in values]


if __name__ == "__main__":
    unittest.main()
