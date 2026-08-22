from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import unittest
from dataclasses import replace

from mr_memory.evidence_closure import parse_contract_turn
from mr_memory.certificate import (
    MAX_CERTIFICATE_CONFLICTS,
    MAX_CERTIFICATE_SOURCE_KEYS,
    MAX_CERTIFICATE_UNRESOLVED,
)
from mr_memory.reader import (
    build_l2_reader_prompt,
    build_single_repair_prompt,
    certificate_from_contract_turn,
    evidence_certificate_v2_schema,
    normalize_l2_reader_response,
    parse_l2_reader_response,
)
from mr_memory.snapshot import (
    DataRevisionVector,
    InferenceRevisionVector,
    RequestSnapshot,
    stable_sha256,
)
from mr_memory.surface import compile_surface_packet, validate_surface_packet
from tests.test_certificate_v2 import _raw_certificate, _snapshot as l2_snapshot
from tests import test_evidence_closure as closure_fixtures


def _eccr_snapshot() -> RequestSnapshot:
    return RequestSnapshot.create(
        snapshot_id="eccr-case-a",
        umo="scope-a",
        cutoff_at=123456,
        message_upper_bound=20,
        request_source_key="source-current",
        sender_participant_key="participant:a",
        reply_source_key="",
        query="query-a",
        context={"case": "eccr"},
        data_revision=DataRevisionVector.from_value(
            {
                "message": "m1",
                "deletion": "d1",
                "identity": "i1",
                "graph": "g1",
                "relation": "r1",
                "feedback": "f1",
            }
        ),
        inference_revision=InferenceRevisionVector.from_value(
            {
                "retriever": "hybrid-v2",
                "embedding_model": "harrier-270m",
                "fusion_policy": "rrf-v1",
                "reader_model": "deepseek-v4-flash",
                "reader_protocol": "eccr-v1",
                "certificate_schema": "evidence-certificate.v2",
                "surface_compiler": "memory-surface.v1",
                "route_policy": "host-route-policy.v1",
            }
        ),
        captured_at=123457,
    )


def _terminal_turn(*, subject: dict[str, object] | None = None):
    contract = closure_fixtures.EvidenceClosureTests.contract(
        obligation_status="SUPPORTED",
        support_keys=["source-1"],
        visited=["source-1"],
        subject=subject,
        with_uncertainty=False,
    )
    return parse_contract_turn(
        closure_fixtures.EvidenceClosureTests.turn(
            contract,
            brief={
                "claims": [
                    {
                        "statement": "证据支持带保留地概括为前后反差。",
                        "source_keys": ["source-1"],
                        "confidence": 0.72,
                    }
                ],
                "conflicts": [],
                "unresolved": [
                    {
                        "statement": "不能据此断言逐字动机或明确预购。",
                        "source_keys": ["source-1"],
                    }
                ],
            },
            terminal=True,
        ),
        allowed_source_keys={"source-1"},
        allowed_participant_keys={"participant:a", "participant:b"},
        allowed_tool_names=closure_fixtures.EvidenceClosureTests.allowed_tools,
    )


class L2ReaderPromptTests(unittest.TestCase):
    def test_allowlist_order_is_stable_across_hash_seeds_and_input_order(self) -> None:
        code = (
            "import json; from mr_memory.reader import _allowlist; "
            "v={'source-z','source-a','source-m'}; "
            "print(json.dumps(_allowlist(v,'sources',limit=8,item_limit=40)))"
        )
        outputs = []
        for seed in ("1", "7", "8675309"):
            environment = {**os.environ, "PYTHONHASHSEED": seed}
            outputs.append(
                subprocess.check_output(
                    [sys.executable, "-c", code],
                    cwd=os.getcwd(),
                    env=environment,
                    text=True,
                ).strip()
            )
        self.assertEqual(len(set(outputs)), 1)
        self.assertEqual(
            json.loads(outputs[0]), ["source-a", "source-m", "source-z"]
        )

    def test_prompt_contains_full_host_bound_schema_and_packet(self) -> None:
        packet = {
            "messages": [
                {"source_key": "s1", "text": "类魂玩吐了"},
                {"source_key": "s2", "text": "我可能会买"},
            ]
        }
        request = build_l2_reader_prompt(
            query="好女孩是什么意思",
            evidence_packet=packet,
            snapshot=l2_snapshot(),
            allowed_source_keys={"s1", "s2"},
            allowed_participant_keys={"p1"},
            pack_read_complete=True,
        )
        self.assertEqual(request.packet_sha256, stable_sha256(packet))
        self.assertIn("host-bound JSON Schema", request.system_prompt)
        self.assertIn("must_not_upgrade", request.system_prompt)
        payload = json.loads(request.user_prompt)
        self.assertEqual(payload["scope_snapshot"], l2_snapshot().as_dict())
        self.assertEqual(payload["evidence_packet"], packet)
        self.assertEqual(request.messages()[0]["role"], "system")

        schema = evidence_certificate_v2_schema(
            snapshot=l2_snapshot(),
            packet_sha256=request.packet_sha256,
            allowed_source_keys=request.allowed_source_keys,
            allowed_participant_keys=request.allowed_participant_keys,
            pack_read_complete=True,
        )
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            schema["properties"]["scope_snapshot"]["const"],
            l2_snapshot().as_dict(),
        )
        self.assertEqual(
            schema["properties"]["packet_sha256"]["const"],
            request.packet_sha256,
        )
        for host_failure in ("PROTOCOL_DEGRADED", "BUDGET_EXHAUSTED"):
            self.assertNotIn(
                host_failure,
                schema["properties"]["stop_reason"]["enum"],
            )
        self.assertEqual(
            schema["properties"]["subjects"]["items"]["properties"]
            ["source_keys"]["maxItems"],
            MAX_CERTIFICATE_SOURCE_KEYS,
        )
        self.assertEqual(
            schema["properties"]["atoms"]["items"]["properties"]
            ["source_keys"]["maxItems"],
            MAX_CERTIFICATE_SOURCE_KEYS,
        )
        self.assertEqual(
            schema["properties"]["atoms"]["items"]["properties"]
            ["source_spans"]["maxItems"],
            MAX_CERTIFICATE_SOURCE_KEYS,
        )
        self.assertEqual(
            schema["properties"]["unresolved"]["items"]["properties"]
            ["source_keys"]["maxItems"],
            MAX_CERTIFICATE_SOURCE_KEYS,
        )
        self.assertEqual(
            schema["properties"]["conflicts"]["maxItems"],
            MAX_CERTIFICATE_CONFLICTS,
        )
        self.assertEqual(
            schema["properties"]["unresolved"]["maxItems"],
            MAX_CERTIFICATE_UNRESOLVED,
        )

    def test_prompt_rejects_query_or_packet_hash_tampering(self) -> None:
        with self.assertRaisesRegex(ValueError, "query differs"):
            build_l2_reader_prompt(
                query="另一个问题",
                evidence_packet={},
                snapshot=l2_snapshot(),
                allowed_source_keys=(),
                pack_read_complete=True,
            )
        with self.assertRaisesRegex(ValueError, "does not match"):
            build_l2_reader_prompt(
                query="好女孩是什么意思",
                evidence_packet={},
                packet_sha256="f" * 64,
                snapshot=l2_snapshot(),
                allowed_source_keys=(),
                pack_read_complete=True,
            )

    def test_response_parser_uses_request_allowlists_and_host_bindings(self) -> None:
        packet = {"sources": ["s1", "s2"]}
        request = build_l2_reader_prompt(
            query="好女孩是什么意思",
            evidence_packet=packet,
            snapshot=l2_snapshot(),
            allowed_source_keys={"s1", "s2"},
            allowed_participant_keys={"p1"},
            pack_read_complete=True,
        )
        raw = _raw_certificate()
        raw["packet_sha256"] = request.packet_sha256
        certificate = parse_l2_reader_response(raw, request)
        self.assertEqual(certificate.packet_sha256, request.packet_sha256)
        provider_degraded = copy.deepcopy(raw)
        provider_degraded["status"] = "PARTIAL"
        provider_degraded["stop_reason"] = "PROTOCOL_DEGRADED"
        with self.assertRaisesRegex(ValueError, "stop_reason is unsupported"):
            parse_l2_reader_response(provider_degraded, request)
        raw["atoms"][0]["source_keys"] = ["outside"]
        with self.assertRaisesRegex(ValueError, "allowlist"):
            parse_l2_reader_response(raw, request)

    def test_l2_host_normalizes_only_exact_singleton_and_safe_identity_downgrade(
        self,
    ) -> None:
        packet = {"sources": ["s1", "s2"]}
        request = build_l2_reader_prompt(
            query="好女孩是什么意思",
            evidence_packet=packet,
            snapshot=l2_snapshot(),
            allowed_source_keys={"s1", "s2"},
            allowed_participant_keys={"p1", "p2"},
            pack_read_complete=True,
        )
        raw = _raw_certificate()
        raw["packet_sha256"] = request.packet_sha256
        raw["subjects"][0]["candidate_participant_keys"] = ["p1"]
        raw["subjects"].extend(
            [
                {
                    "reference": "第二个确定称呼",
                    "participant_key": "p2",
                    "reference_mode": "UNIQUE_ALIAS",
                    "candidate_participant_keys": ["p2"],
                    "source_keys": ["s2"],
                    "valid_at": None,
                },
                {
                    "reference": "未绑定称呼一",
                    "participant_key": "",
                    "reference_mode": "UNBOUND",
                    "candidate_participant_keys": [],
                    "source_keys": ["s2"],
                    "valid_at": None,
                },
                {
                    "reference": "未绑定称呼二",
                    "participant_key": "",
                    "reference_mode": "UNBOUND",
                    "candidate_participant_keys": [],
                    "source_keys": ["s2"],
                    "valid_at": None,
                },
            ]
        )
        frozen_raw = copy.deepcopy(raw)
        audit: list[dict[str, object]] = []
        certificate = parse_l2_reader_response(
            raw, request, normalization_audit=audit
        )
        self.assertEqual(certificate.subjects[0].candidate_participant_keys, ())
        self.assertEqual(certificate.subjects[1].candidate_participant_keys, ())
        self.assertEqual(certificate.subjects[2].reference_mode, "UNBOUND")
        self.assertEqual(certificate.subjects[3].reference_mode, "UNBOUND")
        self.assertEqual(certificate.status, "SAFETY_ABSTAIN")
        self.assertEqual(certificate.stop_reason, "SAFETY_ABSTAIN")
        self.assertEqual(raw, frozen_raw)
        self.assertEqual(
            [item["action"] for item in audit],
            [
                "canonicalize_redundant_singleton",
                "canonicalize_redundant_singleton",
                "downgrade_identity_ambiguity",
            ],
        )
        normalized, repeated_audit = normalize_l2_reader_response(raw)
        self.assertEqual(repeated_audit, audit)
        self.assertEqual(normalized["status"], "SAFETY_ABSTAIN")
        self.assertEqual(normalized["subjects"][0]["candidate_participant_keys"], [])
        self.assertEqual(normalized["subjects"][1]["candidate_participant_keys"], [])

        mismatched = copy.deepcopy(raw)
        mismatched["subjects"][0]["candidate_participant_keys"] = ["p2"]
        with self.assertRaisesRegex(ValueError, "resolved mode requires"):
            parse_l2_reader_response(mismatched, request)

        multiple = copy.deepcopy(raw)
        multiple["subjects"][0]["candidate_participant_keys"] = ["p1", "p2"]
        with self.assertRaisesRegex(ValueError, "resolved mode requires"):
            parse_l2_reader_response(multiple, request)

    def test_l2_host_rebuilds_must_include_from_required_atoms_in_atom_order(
        self,
    ) -> None:
        request = build_l2_reader_prompt(
            query="好女孩是什么意思",
            evidence_packet={"sources": ["s1", "s2"]},
            snapshot=l2_snapshot(),
            allowed_source_keys={"s1", "s2"},
            allowed_participant_keys={"p1"},
            pack_read_complete=True,
        )
        raw = _raw_certificate()
        raw["packet_sha256"] = request.packet_sha256
        raw["atoms"][1]["importance"] = "REQUIRED"
        raw["must_include"] = ["a2", "a1"]
        frozen_atoms = copy.deepcopy(raw["atoms"])
        audit: list[dict[str, object]] = []

        certificate = parse_l2_reader_response(
            raw,
            request,
            normalization_audit=audit,
        )

        self.assertEqual(certificate.must_include, ("a1", "a2"))
        self.assertEqual(raw["atoms"], frozen_atoms)
        self.assertEqual(
            audit,
            [
                {
                    "action": "canonicalize_must_include_from_atoms",
                    "classification": "semantic_preserving_canonicalization",
                    "changed_paths": ["must_include"],
                    "required_atom_ids": ["a1", "a2"],
                }
            ],
        )

        invalid_atom = copy.deepcopy(raw)
        invalid_atom["atoms"][0]["id"] = "invalid atom id"
        with self.assertRaisesRegex(ValueError, "bounded identifier"):
            parse_l2_reader_response(invalid_atom, request)

    def test_repair_prompt_is_available_exactly_once(self) -> None:
        request = build_l2_reader_prompt(
            query="好女孩是什么意思",
            evidence_packet={},
            snapshot=l2_snapshot(),
            allowed_source_keys=(),
            pack_read_complete=True,
        )
        repaired = build_single_repair_prompt(
            request,
            invalid_response="not-json",
            validation_error=ValueError("certificate is invalid"),
        )
        self.assertEqual(repaired.repair_attempt, 1)
        payload = json.loads(repaired.user_prompt)
        self.assertEqual(payload["invalid_response"], "not-json")
        self.assertEqual(payload["original_request"], json.loads(request.user_prompt))
        self.assertIn("source_spans", payload["instruction"])
        self.assertIn("不得超过 source_keys", payload["instruction"])
        self.assertIn("source_spans", request.system_prompt)
        self.assertIn("participant_activity", request.system_prompt)
        self.assertIn("全部 source_key", request.system_prompt)
        self.assertIn("participant_activity", payload["instruction"])
        with self.assertRaisesRegex(ValueError, "already been used"):
            build_single_repair_prompt(
                repaired,
                invalid_response="still-invalid",
                validation_error="bad",
            )


class EccrCertificateAdapterTests(unittest.TestCase):
    def test_terminal_brief_maps_to_host_bound_derived_certificate(self) -> None:
        certificate = certificate_from_contract_turn(
            _terminal_turn(),
            snapshot=_eccr_snapshot(),
            packet_sha256="b" * 64,
            allowed_source_keys={"source-1"},
            allowed_participant_keys={"participant:a", "participant:b"},
            stop_reason="CERTIFIED_CLOSE",
            pack_read_complete=True,
        )
        self.assertEqual(certificate.status, "CERTIFIED")
        self.assertEqual(certificate.atoms[0].attribution, "DERIVED_INTERPRETATION")
        self.assertEqual(certificate.atoms[0].speaker_participant_key, "")
        self.assertEqual(
            certificate.atoms[0].subject_participant_key,
            "participant:a",
        )
        self.assertIn(
            "不能声称有逐字预购或明确动机。",
            [item.statement for item in certificate.unresolved],
        )
        self.assertEqual(certificate.scope_snapshot, _eccr_snapshot())

    def test_ambiguous_contract_maps_to_safety_abstain(self) -> None:
        turn = _terminal_turn(
            subject={
                "reference": "新昵称",
                "participant_key": "",
                "mode": "AMBIGUOUS",
                "candidate_participant_keys": ["participant:a", "participant:b"],
                "source_keys": ["source-1"],
                "valid_at": 123400,
            }
        )
        certificate = certificate_from_contract_turn(
            turn,
            snapshot=_eccr_snapshot(),
            packet_sha256="b" * 64,
            allowed_source_keys={"source-1"},
            allowed_participant_keys={"participant:a", "participant:b"},
            stop_reason="CERTIFIED_CLOSE",
            pack_read_complete=True,
        )
        self.assertEqual(certificate.status, "SAFETY_ABSTAIN")
        self.assertEqual(certificate.subjects[0].reference_mode, "AMBIGUOUS")

    def test_adapter_rejects_revision_and_allowlist_drift(self) -> None:
        snapshot_raw = _eccr_snapshot().as_dict()
        snapshot_raw["data_revision"]["graph"] = "new-graph"
        with self.assertRaisesRegex(ValueError, "revision mismatch: graph"):
            certificate_from_contract_turn(
                _terminal_turn(),
                snapshot=RequestSnapshot.from_value(snapshot_raw),
                packet_sha256="b" * 64,
                allowed_source_keys={"source-1"},
                allowed_participant_keys={"participant:a"},
                stop_reason="CERTIFIED_CLOSE",
                pack_read_complete=True,
            )
        with self.assertRaisesRegex(ValueError, "outside host allowlist"):
            certificate_from_contract_turn(
                _terminal_turn(),
                snapshot=_eccr_snapshot(),
                packet_sha256="b" * 64,
                allowed_source_keys={"different-source"},
                allowed_participant_keys={"participant:a"},
                stop_reason="CERTIFIED_CLOSE",
                pack_read_complete=True,
            )

    def test_adapter_rejects_nonterminal_certified_close(self) -> None:
        turn = _terminal_turn()
        turn = replace(turn, terminal=False)
        with self.assertRaisesRegex(ValueError, "CERTIFIED_CLOSE requires"):
            certificate_from_contract_turn(
                turn,
                snapshot=_eccr_snapshot(),
                packet_sha256="b" * 64,
                allowed_source_keys={"source-1"},
                allowed_participant_keys={"participant:a"},
                stop_reason="CERTIFIED_CLOSE",
                pack_read_complete=True,
            )

    def test_nonterminal_bounded_stops_map_to_partial_certificates(self) -> None:
        turn = replace(_terminal_turn(), terminal=False)
        for stop_reason in (
            "FRONTIER_EXHAUSTED",
            "SATURATED",
        ):
            with self.subTest(stop_reason=stop_reason):
                certificate = certificate_from_contract_turn(
                    turn,
                    snapshot=_eccr_snapshot(),
                    packet_sha256="b" * 64,
                    allowed_source_keys={"source-1"},
                    allowed_participant_keys={"participant:a"},
                    stop_reason=stop_reason,
                    pack_read_complete=True,
                )
                self.assertEqual(certificate.status, "PARTIAL")
                self.assertEqual(certificate.stop_reason, stop_reason)
                self.assertTrue(
                    certificate.unresolved or certificate.open_obligations
                )
        with self.assertRaisesRegex(ValueError, "cannot produce a certificate"):
            certificate_from_contract_turn(
                turn,
                snapshot=_eccr_snapshot(),
                packet_sha256="b" * 64,
                allowed_source_keys={"source-1"},
                allowed_participant_keys={"participant:a"},
                stop_reason="BUDGET_EXHAUSTED",
                pack_read_complete=True,
            )

    def test_protocol_degradation_cannot_produce_a_certificate(self) -> None:
        turn = _terminal_turn()
        with self.assertRaisesRegex(ValueError, "cannot produce a certificate"):
            certificate_from_contract_turn(
                turn,
                snapshot=_eccr_snapshot(),
                packet_sha256="b" * 64,
                allowed_source_keys={"source-1"},
                allowed_participant_keys={"participant:a"},
                stop_reason="PROTOCOL_DEGRADED",
                pack_read_complete=True,
            )


if __name__ == "__main__":
    unittest.main()
