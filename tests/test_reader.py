from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import unittest
from dataclasses import replace

from mr_memory.evidence_closure import parse_contract_turn
from mr_memory.evidence_pack import compile_evidence_atom_pack
from mr_memory.certificate import (
    MAX_CERTIFICATE_CONFLICTS,
    MAX_CERTIFICATE_SOURCE_KEYS,
    MAX_CERTIFICATE_UNRESOLVED,
)
from mr_memory.reader import (
    _alias_evidence_packet,
    build_l2_reader_prompt,
    certificate_from_contract_turn,
    evidence_certificate_v2_schema,
    parse_l2_reader_response,
    validate_certificate_source_bindings,
)
from mr_memory.snapshot import (
    DataRevisionVector,
    InferenceRevisionVector,
    RequestSnapshot,
    canonical_json,
    stable_sha256,
)
from mr_memory.surface import compile_surface_packet, validate_surface_packet
from tests.test_certificate_v2 import _raw_certificate, _snapshot as l2_snapshot
from tests import test_evidence_closure as closure_fixtures


_HOST_CERTIFICATE_FIELDS = {
    "schema_version",
    "scope_snapshot",
    "data_revision",
    "inference_revision",
    "packet_sha256",
    "validation",
}


def _semantic_delta(
    raw: dict[str, object],
    request: object,
) -> dict[str, object]:
    semantic = {
        key: copy.deepcopy(value)
        for key, value in raw.items()
        if key not in _HOST_CERTIFICATE_FIELDS
    }
    for atom in semantic.get("atoms", []):
        if isinstance(atom, dict):
            atom.pop("speaker_participant_key", None)
    aliased = _alias_evidence_packet(
        semantic,
        source_aliases=request.source_key_to_alias,
        participant_aliases=request.participant_key_to_alias,
    )
    if not isinstance(aliased, dict):  # pragma: no cover - helper guard
        raise AssertionError("semantic response must be an object")
    return aliased


def _large_prompt_char_counts() -> tuple[int, int]:
    """Reconstruct the v2 prompt baseline and compare the compact v3 prompt."""

    source_keys = [
        f"source-key-{index:04d}-with-a-realistically-long-host-namespace"
        for index in range(512)
    ]
    participant_keys = [
        f"participant-key-{index:04d}-with-a-long-canonical-namespace"
        for index in range(128)
    ]
    packet = {
        "messages": [
            {
                "source_key": source_key,
                "sender_participant_key": participant_keys[index % 128],
                "text": f"synthetic evidence {index}",
            }
            for index, source_key in enumerate(source_keys)
        ]
    }
    participant_sources = {
        participant: source_keys[index * 4 : (index + 1) * 4]
        for index, participant in enumerate(participant_keys)
    }
    request = build_l2_reader_prompt(
        query="好女孩是什么意思",
        evidence_packet=packet,
        snapshot=l2_snapshot(),
        allowed_source_keys=source_keys,
        allowed_participant_keys=participant_keys,
        participant_source_keys=participant_sources,
        pack_read_complete=True,
    )
    schema = evidence_certificate_v2_schema(
        snapshot=l2_snapshot(),
        packet_sha256=request.packet_sha256,
        allowed_source_keys=source_keys,
        allowed_participant_keys=participant_keys,
        pack_read_complete=True,
    )
    legacy_system_prefix = request.system_prompt.split(
        "\nCompact semantic contract:\n",
        1,
    )[0].replace("compact semantic contract", "host-bound JSON Schema")
    legacy_system = (
        legacy_system_prefix
        + "\nJSON Schema:\n"
        + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    )
    legacy_payload = {
        "protocol": "evidence-reader.v2",
        "query": "好女孩是什么意思",
        "scope_snapshot": l2_snapshot().as_dict(),
        "packet_sha256": request.packet_sha256,
        "allowed_source_keys": source_keys,
        "allowed_participant_keys": participant_keys,
        "participant_source_keys": participant_sources,
        "pack_read_complete": True,
        "semantic_none_allowed": True,
        "evidence_packet": packet,
    }
    old_chars = len(legacy_system) + len(canonical_json(legacy_payload))
    new_chars = len(request.system_prompt) + len(request.user_prompt)
    return old_chars, new_chars


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

    def test_prompt_uses_compact_alias_protocol_and_keeps_host_fields_local(
        self,
    ) -> None:
        packet = {
            "messages": [
                {
                    "source_key": "source-one-canonical",
                    "sender_participant_key": "participant-one-canonical",
                    "text": "类魂玩吐了",
                },
                {
                    "source_key": "source-two-canonical",
                    "sender_participant_key": "participant-one-canonical",
                    "text": "我可能会买",
                },
            ]
        }
        request = build_l2_reader_prompt(
            query="好女孩是什么意思",
            evidence_packet=packet,
            snapshot=l2_snapshot(),
            allowed_source_keys={
                "source-one-canonical",
                "source-two-canonical",
            },
            allowed_participant_keys={"participant-one-canonical"},
            participant_source_keys={
                "participant-one-canonical": {
                    "source-two-canonical",
                    "source-one-canonical",
                }
            },
            pack_read_complete=True,
        )
        self.assertEqual(request.packet_sha256, stable_sha256(packet))
        self.assertIn("compact semantic contract", request.system_prompt)
        self.assertNotIn("\nJSON Schema:\n", request.system_prompt)
        self.assertIn("must_not_upgrade", request.system_prompt)
        payload = json.loads(request.user_prompt)
        for host_field in (
            "scope_snapshot",
            "packet_sha256",
            "allowed_source_keys",
            "allowed_participant_keys",
        ):
            self.assertNotIn(host_field, payload)
        self.assertEqual(
            payload["evidence_packet"],
            {
                "messages": [
                    {
                        "source_key": "s1",
                        "sender_participant_key": "p1",
                        "text": "类魂玩吐了",
                    },
                    {
                        "source_key": "s2",
                        "sender_participant_key": "p1",
                        "text": "我可能会买",
                    },
                ]
            },
        )
        self.assertEqual(payload["participant_source_keys"], {"p1": ["s1", "s2"]})
        self.assertEqual(
            dict(request.source_alias_to_key),
            {
                "s1": "source-one-canonical",
                "s2": "source-two-canonical",
            },
        )
        self.assertEqual(
            dict(request.participant_alias_to_key),
            {"p1": "participant-one-canonical"},
        )
        self.assertTrue(payload["semantic_none_allowed"])
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
        certified_atoms = schema["allOf"][0]["then"]["properties"]["atoms"]
        self.assertEqual(
            certified_atoms["items"]["properties"]["source_keys"]["minItems"],
            1,
        )

    def test_compact_semantic_response_restores_aliases_and_injects_host_fields(
        self,
    ) -> None:
        source_one = "source-one-with-a-long-canonical-host-key"
        source_two = "source-two-with-a-long-canonical-host-key"
        participant = "participant-with-a-long-canonical-host-key"
        packet = {
            "messages": [
                {
                    "source_key": source_one,
                    "sender_participant_key": participant,
                    "text": "synthetic one",
                },
                {
                    "source_key": source_two,
                    "sender_participant_key": participant,
                    "text": "synthetic two",
                },
            ]
        }
        request = build_l2_reader_prompt(
            query="好女孩是什么意思",
            evidence_packet=packet,
            snapshot=l2_snapshot(),
            allowed_source_keys={source_one, source_two},
            allowed_participant_keys={participant},
            participant_source_keys={participant: {source_one, source_two}},
            pack_read_complete=True,
        )
        source_alias = request.source_key_to_alias[source_one]
        participant_alias = request.participant_key_to_alias[participant]
        response = {
            "status": "CERTIFIED",
            "subjects": [
                {
                    "reference": "synthetic subject",
                    "participant_key": participant_alias,
                    "reference_mode": "UNIQUE_ALIAS",
                    "candidate_participant_keys": [],
                    "source_keys": [source_alias],
                    "valid_at": None,
                }
            ],
            "atoms": [
                {
                    "id": "synthetic-atom-1",
                    "statement": "A synthetic supported statement.",
                    "subject_participant_key": participant_alias,
                    "attribution": "DIRECT_SPEAKER_STATEMENT",
                    "stance": "SUPPORTED",
                    "source_keys": [source_alias],
                    "source_spans": ["synthetic one"],
                    "importance": "REQUIRED",
                    "confidence": 0.9,
                }
            ],
            "must_include": ["synthetic-atom-1"],
            "must_not_upgrade": [],
            "conflicts": [],
            "unresolved": [],
            "open_obligations": [],
            "stop_reason": "CERTIFIED_CLOSE",
        }

        certificate = parse_l2_reader_response(response, request)

        self.assertEqual(certificate.scope_snapshot, l2_snapshot())
        self.assertEqual(certificate.packet_sha256, stable_sha256(packet))
        self.assertEqual(certificate.subjects[0].participant_key, participant)
        self.assertEqual(certificate.subjects[0].source_keys, (source_one,))
        self.assertEqual(
            certificate.atoms[0].speaker_participant_key,
            participant,
        )
        self.assertEqual(certificate.atoms[0].source_keys, (source_one,))

        supplied_speaker = copy.deepcopy(response)
        supplied_speaker["atoms"][0]["speaker_participant_key"] = participant_alias
        with self.assertRaisesRegex(ValueError, "speaker_participant_key is host-owned"):
            parse_l2_reader_response(supplied_speaker, request)

        supplied_host_fields = copy.deepcopy(response)
        supplied_host_fields["scope_snapshot"] = {"tampered": True}
        supplied_host_fields["packet_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "host-owned fields"):
            parse_l2_reader_response(supplied_host_fields, request)

    def test_compact_prompt_is_smaller_for_the_same_large_allowlist(self) -> None:
        old_chars, new_chars = _large_prompt_char_counts()
        self.assertLess(new_chars, old_chars)
        self.assertLess(new_chars / old_chars, 0.35)

    def test_resident_one_pass_schema_excludes_l3_and_bounds_uncertainty(
        self,
    ) -> None:
        packet = {"sources": ["s1", "s2"]}
        request = build_l2_reader_prompt(
            query="好女孩是什么意思",
            evidence_packet=packet,
            snapshot=l2_snapshot(),
            allowed_source_keys={"s1", "s2"},
            allowed_participant_keys={"p1"},
            participant_source_keys={"p1": {"s1", "s2"}},
            pack_read_complete=True,
            allow_l3=False,
        )
        schema = evidence_certificate_v2_schema(
            snapshot=l2_snapshot(),
            packet_sha256=request.packet_sha256,
            allowed_source_keys=request.allowed_source_keys,
            allowed_participant_keys=request.allowed_participant_keys,
            pack_read_complete=True,
            allow_l3=False,
        )
        self.assertFalse(request.allow_l3)
        self.assertNotIn(
            "REQUEST_L3",
            schema["properties"]["status"]["enum"],
        )
        self.assertNotIn(
            "REQUEST_L3",
            schema["properties"]["stop_reason"]["enum"],
        )
        self.assertNotIn(
            '"const": "REQUEST_L3"',
            json.dumps(schema["allOf"], sort_keys=True),
        )
        for directive in (
            "resident one-pass",
            "不得返回 REQUEST_L3",
            "PARTIAL",
            "SAFETY_ABSTAIN",
            "unresolved",
            "不得请求工具、修复或任何升级路径",
        ):
            self.assertIn(directive, request.system_prompt)

        offline_schema = evidence_certificate_v2_schema(
            snapshot=l2_snapshot(),
            packet_sha256=request.packet_sha256,
            allowed_source_keys=request.allowed_source_keys,
            allowed_participant_keys=request.allowed_participant_keys,
            pack_read_complete=True,
        )
        self.assertIn(
            "REQUEST_L3",
            offline_schema["properties"]["status"]["enum"],
        )
        self.assertIn(
            "REQUEST_L3",
            offline_schema["properties"]["stop_reason"]["enum"],
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

    def test_prompt_rejects_packet_identifiers_outside_host_allowlists(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the source allowlist"):
            build_l2_reader_prompt(
                query="好女孩是什么意思",
                evidence_packet={
                    "messages": [{"source_key": "source-outside"}]
                },
                snapshot=l2_snapshot(),
                allowed_source_keys={"source-allowed"},
                pack_read_complete=True,
            )

    def test_only_identity_shaped_canonical_keys_are_participant_aliased(self) -> None:
        participant_key = 'participant:["synthetic","account-a"]'
        packet = {
            "identity_candidates": [
                {
                    "canonical_key": participant_key,
                    "account_id": "account-a",
                    "current_display_name": "测试成员甲",
                }
            ],
            "graph_metadata": {
                "canonical_key": "graph-node:synthetic-topic",
                "kind": "topic",
            },
        }
        aliased = _alias_evidence_packet(
            packet,
            source_aliases={},
            participant_aliases={participant_key: "p1"},
        )
        self.assertEqual(
            aliased["identity_candidates"][0]["canonical_key"],
            "p1",
        )
        self.assertEqual(
            aliased["graph_metadata"]["canonical_key"],
            "graph-node:synthetic-topic",
        )

    def test_incomplete_person_coverage_forbids_unique_alias(self) -> None:
        packet = compile_evidence_atom_pack(
            {
                "person_reference_candidates": {
                    "references": [
                        {
                            "reference": "合成称呼",
                            "candidate_participants": [
                                {
                                    "participant_key": "p1",
                                    "alias_observations": [
                                        {
                                            "source_key": "s1",
                                            "sent_at": 1,
                                            "sender_id": "account-a",
                                            "plain_text": "第一条合成观察",
                                            "alias": "合成称呼",
                                            "relation": "SPEAKER",
                                        }
                                    ],
                                },
                                {
                                    "participant_key": "p2",
                                    "alias_observations": [
                                        {
                                            "source_key": "s2",
                                            "sent_at": 2,
                                            "sender_id": "account-b",
                                            "plain_text": "第二条合成观察",
                                            "alias": "合成称呼",
                                            "relation": "SPEAKER",
                                        }
                                    ],
                                },
                            ],
                        }
                    ]
                },
                "retrieval_coverage": {
                    "person_reference_coverage": {"truncated": True}
                },
            },
            max_sources=2,
        )
        coverage = packet["retrieval_coverage"]
        self.assertEqual(coverage["person_candidate_groups_total"], 2)
        self.assertEqual(coverage["person_candidate_groups_covered"], 2)
        self.assertFalse(coverage["truncated"])
        self.assertFalse(coverage["person_candidates_complete"])

        request = build_l2_reader_prompt(
            query="好女孩是什么意思",
            evidence_packet=packet,
            snapshot=l2_snapshot(),
            allowed_source_keys={"s1", "s2"},
            allowed_participant_keys={"p1", "p2"},
            participant_source_keys={"p1": {"s1"}, "p2": {"s2"}},
            person_candidates_complete=bool(
                coverage["person_candidates_complete"]
            ),
            pack_read_complete=True,
        )
        raw = _raw_certificate()
        raw["packet_sha256"] = request.packet_sha256
        raw["subjects"][0].update(
            {"reference_mode": "UNIQUE_ALIAS", "source_keys": ["s1"]}
        )
        with self.assertRaisesRegex(ValueError, "UNIQUE_ALIAS is forbidden"):
            parse_l2_reader_response(_semantic_delta(raw, request), request)
        self.assertIn("禁止输出 UNIQUE_ALIAS", request.system_prompt)

    def test_reader_rejects_markdown_fenced_json(self) -> None:
        request = build_l2_reader_prompt(
            query="好女孩是什么意思",
            evidence_packet={"messages": [{"source_key": "s1"}]},
            snapshot=l2_snapshot(),
            allowed_source_keys={"s1", "s2"},
            allowed_participant_keys={"p1"},
            participant_source_keys={"p1": {"s1", "s2"}},
            pack_read_complete=True,
        )
        raw = _raw_certificate()
        raw["packet_sha256"] = request.packet_sha256
        fenced = "```json\n" + json.dumps(_semantic_delta(raw, request)) + "\n```"
        with self.assertRaisesRegex(ValueError, "exactly one JSON object"):
            parse_l2_reader_response(fenced, request)
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
        packet = {
            "sources": [
                {"source_key": "s1", "sender_participant_key": "p1"},
                {"source_key": "s2", "sender_participant_key": "p1"},
            ]
        }
        request = build_l2_reader_prompt(
            query="好女孩是什么意思",
            evidence_packet=packet,
            snapshot=l2_snapshot(),
            allowed_source_keys={"s1", "s2"},
            allowed_participant_keys={"p1"},
            participant_source_keys={"p1": {"s1", "s2"}},
            pack_read_complete=True,
        )
        raw = _raw_certificate()
        raw["packet_sha256"] = request.packet_sha256
        certificate = parse_l2_reader_response(_semantic_delta(raw, request), request)
        self.assertEqual(certificate.packet_sha256, request.packet_sha256)
        provider_degraded = copy.deepcopy(raw)
        provider_degraded["status"] = "PARTIAL"
        provider_degraded["stop_reason"] = "PROTOCOL_DEGRADED"
        with self.assertRaisesRegex(ValueError, "stop_reason is unsupported"):
            parse_l2_reader_response(
                _semantic_delta(provider_degraded, request),
                request,
            )
        invalid_delta = _semantic_delta(raw, request)
        invalid_delta["atoms"][0]["source_keys"] = ["s999"]
        with self.assertRaisesRegex(ValueError, "outside the source alias map"):
            parse_l2_reader_response(invalid_delta, request)

    def test_resident_one_pass_rejects_l3_response_and_repair(self) -> None:
        packet = {
            "sources": [
                {"source_key": "s1", "sender_participant_key": "p1"},
                {"source_key": "s2", "sender_participant_key": "p1"},
            ]
        }
        offline_request = build_l2_reader_prompt(
            query="好女孩是什么意思",
            evidence_packet=packet,
            snapshot=l2_snapshot(),
            allowed_source_keys={"s1", "s2"},
            allowed_participant_keys={"p1"},
            participant_source_keys={"p1": {"s1", "s2"}},
            pack_read_complete=True,
        )
        raw = _raw_certificate()
        raw["packet_sha256"] = offline_request.packet_sha256
        raw.update(
            {
                "status": "REQUEST_L3",
                "stop_reason": "REQUEST_L3",
                "open_obligations": [
                    {
                        "id": "o1",
                        "question": "两个相邻称呼是否指向同一参与者？",
                        "critical": True,
                        "competing_interpretation_ids": ["same", "different"],
                        "discriminator": "检查另一份边界明确的合成证据",
                        "expected_information_gain": "区分两个合成候选解释",
                    }
                ],
            }
        )
        self.assertEqual(
            parse_l2_reader_response(
                _semantic_delta(raw, offline_request),
                offline_request,
            ).status,
            "REQUEST_L3",
        )

        resident_request = build_l2_reader_prompt(
            query="好女孩是什么意思",
            evidence_packet=packet,
            snapshot=l2_snapshot(),
            allowed_source_keys={"s1", "s2"},
            allowed_participant_keys={"p1"},
            participant_source_keys={"p1": {"s1", "s2"}},
            pack_read_complete=True,
            allow_l3=False,
        )
        with self.assertRaisesRegex(ValueError, "cannot request L3"):
            parse_l2_reader_response(
                _semantic_delta(raw, resident_request),
                resident_request,
            )

    def test_l2_reader_fails_closed_on_redundant_or_ambiguous_identity(
        self,
    ) -> None:
        packet = {
            "sources": [
                {"source_key": "s1", "sender_participant_key": "p1"},
                {"source_key": "s2", "sender_participant_key": "p1"},
            ]
        }
        request = build_l2_reader_prompt(
            query="好女孩是什么意思",
            evidence_packet=packet,
            snapshot=l2_snapshot(),
            allowed_source_keys={"s1", "s2"},
            allowed_participant_keys={"p1", "p2"},
            participant_source_keys={
                "p1": {"s1", "s2"},
                "p2": {"s2"},
            },
            pack_read_complete=True,
        )
        raw = _raw_certificate()
        raw["packet_sha256"] = request.packet_sha256
        raw["subjects"][0]["candidate_participant_keys"] = ["p1"]
        with self.assertRaisesRegex(ValueError, "resolved mode requires"):
            parse_l2_reader_response(_semantic_delta(raw, request), request)

        ambiguous = copy.deepcopy(raw)
        ambiguous["subjects"][0].update(
            {
                "participant_key": "",
                "reference_mode": "AMBIGUOUS",
                "candidate_participant_keys": ["p1", "p2"],
                "source_keys": ["s1", "s2"],
            }
        )
        with self.assertRaisesRegex(ValueError, "identity ambiguity"):
            parse_l2_reader_response(_semantic_delta(ambiguous, request), request)

    def test_l2_reader_preserves_valid_must_include_and_rejects_missing_ids(
        self,
    ) -> None:
        request = build_l2_reader_prompt(
            query="好女孩是什么意思",
            evidence_packet={
                "sources": [
                    {"source_key": "s1", "sender_participant_key": "p1"},
                    {"source_key": "s2", "sender_participant_key": "p1"},
                ]
            },
            snapshot=l2_snapshot(),
            allowed_source_keys={"s1", "s2"},
            allowed_participant_keys={"p1"},
            participant_source_keys={"p1": {"s1", "s2"}},
            pack_read_complete=True,
        )
        raw = _raw_certificate()
        raw["packet_sha256"] = request.packet_sha256
        raw["atoms"][1]["importance"] = "REQUIRED"
        raw["must_include"] = ["a2", "a1"]
        certificate = parse_l2_reader_response(
            _semantic_delta(raw, request),
            request,
        )
        self.assertEqual(certificate.must_include, ("a2", "a1"))

        missing = copy.deepcopy(raw)
        missing["must_include"] = ["a1"]
        with self.assertRaisesRegex(ValueError, "every and only REQUIRED"):
            parse_l2_reader_response(_semantic_delta(missing, request), request)

        invalid_atom = copy.deepcopy(raw)
        invalid_atom["atoms"][0]["id"] = "invalid atom id"
        with self.assertRaisesRegex(ValueError, "bounded identifier"):
            parse_l2_reader_response(_semantic_delta(invalid_atom, request), request)

    def test_certified_atoms_require_a_source_during_parsing(self) -> None:
        request = build_l2_reader_prompt(
            query="好女孩是什么意思",
            evidence_packet={"messages": [{"source_key": "source-alpha"}]},
            snapshot=l2_snapshot(),
            allowed_source_keys={"source-alpha", "s1", "s2"},
            allowed_participant_keys={"p1"},
            participant_source_keys={"p1": {"source-alpha", "s1", "s2"}},
            pack_read_complete=True,
        )
        raw = _raw_certificate()
        raw["packet_sha256"] = request.packet_sha256
        raw["atoms"][0]["source_keys"] = []
        raw["atoms"][0]["source_spans"] = []
        with self.assertRaisesRegex(
            ValueError,
            r"atoms\[0\] DIRECT_SPEAKER_STATEMENT requires one host-bound speaker",
        ):
            parse_l2_reader_response(_semantic_delta(raw, request), request)

    def test_semantic_none_policy_is_bound_into_schema_prompt_and_parser(
        self,
    ) -> None:
        packet = {"messages": []}
        allowed_request = build_l2_reader_prompt(
            query="好女孩是什么意思",
            evidence_packet=packet,
            snapshot=l2_snapshot(),
            allowed_source_keys=(),
            pack_read_complete=True,
        )
        forbidden_request = build_l2_reader_prompt(
            query="好女孩是什么意思",
            evidence_packet=packet,
            snapshot=l2_snapshot(),
            allowed_source_keys=(),
            pack_read_complete=True,
            semantic_none_allowed=False,
        )
        schema = evidence_certificate_v2_schema(
            snapshot=l2_snapshot(),
            packet_sha256=forbidden_request.packet_sha256,
            allowed_source_keys=(),
            pack_read_complete=True,
            semantic_none_allowed=False,
        )
        self.assertNotIn(
            "SEMANTIC_NONE",
            schema["properties"]["status"]["enum"],
        )
        self.assertNotIn(
            "SEMANTIC_NONE",
            schema["properties"]["stop_reason"]["enum"],
        )
        self.assertNotIn(
            '"const": "SEMANTIC_NONE"',
            json.dumps(schema["allOf"], sort_keys=True),
        )
        payload = json.loads(forbidden_request.user_prompt)
        self.assertFalse(payload["semantic_none_allowed"])
        self.assertIn("检索覆盖不足", forbidden_request.system_prompt)
        self.assertIn("禁止声称历史不存在", forbidden_request.system_prompt)

        raw = _raw_certificate()
        raw.update(
            {
                "packet_sha256": allowed_request.packet_sha256,
                "status": "SEMANTIC_NONE",
                "subjects": [],
                "atoms": [],
                "must_include": [],
                "must_not_upgrade": [],
                "conflicts": [],
                "unresolved": [],
                "open_obligations": [],
                "stop_reason": "SEMANTIC_NONE",
            }
        )
        self.assertEqual(
            parse_l2_reader_response(
                _semantic_delta(raw, allowed_request),
                allowed_request,
            ).status,
            "SEMANTIC_NONE",
        )
        with self.assertRaisesRegex(ValueError, "retrieval coverage is insufficient"):
            parse_l2_reader_response(
                _semantic_delta(raw, forbidden_request),
                forbidden_request,
            )

    def test_participant_sources_bind_aliases_and_speakers_not_atom_subjects(
        self,
    ) -> None:
        packet = {
            "messages": [
                {
                    "source_key": "source-alpha",
                    "sender_participant_key": "participant-alpha",
                    "text": "synthetic alpha",
                },
                {
                    "source_key": "source-beta",
                    "sender_participant_key": "participant-beta",
                    "text": "synthetic beta",
                },
            ]
        }
        request = build_l2_reader_prompt(
            query="好女孩是什么意思",
            evidence_packet=packet,
            snapshot=l2_snapshot(),
            allowed_source_keys={"source-beta", "source-alpha"},
            allowed_participant_keys={"participant-beta", "participant-alpha"},
            participant_source_keys={
                "participant-alpha": ["source-alpha", "source-alpha"],
                "participant-beta": ["source-beta"],
            },
            pack_read_complete=True,
        )
        self.assertEqual(
            dict(request.participant_source_keys),
            {
                "participant-alpha": ("source-alpha",),
                "participant-beta": ("source-beta",),
            },
        )
        self.assertEqual(
            json.loads(request.user_prompt)["participant_source_keys"],
            {
                "p1": ["s1"],
                "p2": ["s2"],
            },
        )
        self.assertEqual(
            json.loads(request.user_prompt)["participant_speaker_source_keys"],
            {
                "p1": ["s1"],
                "p2": ["s2"],
            },
        )

        raw = _raw_certificate()
        raw.update(
            {
                "packet_sha256": request.packet_sha256,
                "subjects": [
                    {
                        "reference": "synthetic-alias-alpha",
                        "participant_key": "participant-alpha",
                        "reference_mode": "UNIQUE_ALIAS",
                        "candidate_participant_keys": [],
                        "source_keys": ["source-alpha"],
                        "valid_at": None,
                    }
                ],
                "atoms": [
                    {
                        "id": "synthetic-fact-1",
                        "statement": "A synthetic statement about alpha.",
                        "speaker_participant_key": "participant-alpha",
                        "subject_participant_key": "participant-alpha",
                        "attribution": "DIRECT_SPEAKER_STATEMENT",
                        "stance": "SUPPORTED",
                        "source_keys": ["source-alpha"],
                        "source_spans": ["synthetic alpha"],
                        "importance": "REQUIRED",
                        "confidence": 0.9,
                    }
                ],
                "must_include": ["synthetic-fact-1"],
                "must_not_upgrade": [],
                "conflicts": [],
                "unresolved": [],
                "open_obligations": [],
            }
        )
        self.assertEqual(
            parse_l2_reader_response(_semantic_delta(raw, request), request).status,
            "CERTIFIED",
        )

        cross_alias = copy.deepcopy(raw)
        cross_alias["subjects"][0]["source_keys"] = ["source-beta"]
        with self.assertRaisesRegex(ValueError, r"subjects\[0\].*not admissible"):
            parse_l2_reader_response(_semantic_delta(cross_alias, request), request)

        third_party_about_subject = copy.deepcopy(raw)
        third_party_about_subject["subjects"] = [
            {
                "reference": "synthetic-alias-beta",
                "participant_key": "participant-beta",
                "reference_mode": "UNIQUE_ALIAS",
                "candidate_participant_keys": [],
                "source_keys": ["source-beta"],
                "valid_at": None,
            }
        ]
        third_party_about_subject["atoms"][0][
            "subject_participant_key"
        ] = "participant-beta"
        self.assertEqual(
            parse_l2_reader_response(
                _semantic_delta(third_party_about_subject, request),
                request,
            ).status,
            "CERTIFIED",
        )

        source_owned_by_beta = copy.deepcopy(raw)
        source_owned_by_beta["atoms"][0]["source_keys"] = ["source-beta"]
        rebound = parse_l2_reader_response(
            _semantic_delta(source_owned_by_beta, request),
            request,
        )
        self.assertEqual(
            rebound.atoms[0].speaker_participant_key,
            "participant-beta",
        )

        model_owned_speaker = _semantic_delta(raw, request)
        model_owned_speaker["atoms"][0]["speaker_participant_key"] = "p2"
        with self.assertRaisesRegex(ValueError, "speaker_participant_key is host-owned"):
            parse_l2_reader_response(model_owned_speaker, request)

        unresolved_subject = copy.deepcopy(raw)
        unresolved_subject["atoms"][0][
            "subject_participant_key"
        ] = "participant-beta"
        with self.assertRaisesRegex(ValueError, r"subject participant is not resolved"):
            parse_l2_reader_response(
                _semantic_delta(unresolved_subject, request),
                request,
            )

        unbound_request = build_l2_reader_prompt(
            query="好女孩是什么意思",
            evidence_packet=packet,
            snapshot=l2_snapshot(),
            allowed_source_keys={"source-beta", "source-alpha"},
            allowed_participant_keys={"participant-beta", "participant-alpha"},
            pack_read_complete=True,
        )
        with self.assertRaisesRegex(ValueError, r"subjects\[0\].*not admissible"):
            parse_l2_reader_response(
                _semantic_delta(raw, unbound_request),
                unbound_request,
            )

        with self.assertRaisesRegex(ValueError, "participant outside the allowlist"):
            build_l2_reader_prompt(
                query="好女孩是什么意思",
                evidence_packet=packet,
                snapshot=l2_snapshot(),
                allowed_source_keys={"source-alpha"},
                allowed_participant_keys={"participant-alpha"},
                participant_source_keys={"participant-gamma": ["source-alpha"]},
                pack_read_complete=True,
            )
        with self.assertRaisesRegex(ValueError, "source outside the allowlist"):
            build_l2_reader_prompt(
                query="好女孩是什么意思",
                evidence_packet=packet,
                snapshot=l2_snapshot(),
                allowed_source_keys={"source-alpha"},
                allowed_participant_keys={"participant-alpha"},
                participant_source_keys={"participant-alpha": ["source-beta"]},
                pack_read_complete=True,
            )

    def test_parsed_certificate_source_bindings_are_reusable_for_cache_checks(
        self,
    ) -> None:
        request = build_l2_reader_prompt(
            query="好女孩是什么意思",
            evidence_packet={
                "messages": [
                    {
                        "source_key": "source-alpha",
                        "sender_participant_key": "participant-alpha",
                    }
                ]
            },
            snapshot=l2_snapshot(),
            allowed_source_keys={"source-alpha", "source-beta"},
            allowed_participant_keys={"participant-alpha"},
            participant_source_keys={
                "participant-alpha": {"source-alpha", "source-beta"},
            },
            pack_read_complete=True,
        )
        raw = _raw_certificate()
        raw.update(
            {
                "packet_sha256": request.packet_sha256,
                "subjects": [
                    {
                        "reference": "current structured participant",
                        "participant_key": "participant-alpha",
                        "reference_mode": "HOST",
                        "candidate_participant_keys": [],
                        "source_keys": [],
                        "valid_at": None,
                    }
                ],
                "atoms": [
                    {
                        "id": "synthetic-fact-1",
                        "statement": "A synthetic participant-bound statement.",
                        "speaker_participant_key": "participant-alpha",
                        "subject_participant_key": "participant-alpha",
                        "attribution": "DIRECT_SPEAKER_STATEMENT",
                        "stance": "SUPPORTED",
                        "source_keys": ["source-alpha"],
                        "source_spans": ["synthetic evidence"],
                        "importance": "REQUIRED",
                        "confidence": 0.9,
                    }
                ],
                "must_include": ["synthetic-fact-1"],
                "must_not_upgrade": [],
                "conflicts": [],
                "unresolved": [],
                "open_obligations": [],
            }
        )
        certificate = parse_l2_reader_response(_semantic_delta(raw, request), request)
        inadmissible = {"participant-alpha": {"source-beta"}}

        speaker_only = replace(
            certificate,
            atoms=(
                replace(
                    certificate.atoms[0],
                    subject_participant_key="",
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, r"speaker participant"):
            validate_certificate_source_bindings(
                speaker_only,
                participant_source_keys=request.participant_source_keys,
                participant_speaker_source_keys=inadmissible,
            )

        subject_only = replace(
            certificate,
            atoms=(
                replace(
                    certificate.atoms[0],
                    speaker_participant_key="",
                ),
            ),
        )
        self.assertIs(
            validate_certificate_source_bindings(
                subject_only,
                participant_source_keys=inadmissible,
            ),
            subject_only,
        )

        orphan_subject = replace(subject_only, subjects=())
        with self.assertRaisesRegex(ValueError, r"subject participant is not resolved"):
            validate_certificate_source_bindings(
                orphan_subject,
                participant_source_keys=inadmissible,
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

    def test_adapter_allows_third_party_evidence_about_resolved_subject(self) -> None:
        certificate = certificate_from_contract_turn(
            _terminal_turn(),
            snapshot=_eccr_snapshot(),
            packet_sha256="b" * 64,
            allowed_source_keys={"source-1", "source-2"},
            allowed_participant_keys={"participant:a", "participant:b"},
            participant_source_keys={
                "participant:a": {"source-1"},
                "participant:b": {"source-2"},
            },
            stop_reason="CERTIFIED_CLOSE",
            pack_read_complete=True,
        )
        self.assertEqual(certificate.status, "CERTIFIED")

        third_party = certificate_from_contract_turn(
            _terminal_turn(),
            snapshot=_eccr_snapshot(),
            packet_sha256="b" * 64,
            allowed_source_keys={"source-1", "source-2"},
            allowed_participant_keys={"participant:a", "participant:b"},
            participant_source_keys={
                "participant:a": {"source-2"},
                "participant:b": {"source-1"},
            },
            stop_reason="CERTIFIED_CLOSE",
            pack_read_complete=True,
        )
        self.assertEqual(third_party.status, "CERTIFIED")

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
