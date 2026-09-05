from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import unittest
from dataclasses import replace

from mr_memory.evidence_closure import parse_contract_turn
from mr_memory.evidence_pack import compile_evidence_atom_pack
from mr_memory.identity import build_request_identity_context
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
from mr_memory.surface import compile_answer_context_packet, compile_surface_packet, validate_surface_packet
from tests.test_certificate_v2 import _raw_certificate, _raw_packet_gap_certificate, _raw_activity_certificate, _snapshot as l2_snapshot
from tests import test_evidence_closure as closure_fixtures

_HOST_CERTIFICATE_FIELDS = {
    "schema_version",
    "scope_snapshot",
    "data_revision",
    "inference_revision",
    "packet_sha256",
    "validation",
    "aggregates",
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


def _typed_resident_delta(
    raw: dict[str, object],
    request: object,
) -> dict[str, object]:
    """Translate a certificate fixture into the resident semantic contract."""

    semantic = _semantic_delta(raw, request)
    subjects = semantic.pop("subjects", [])
    referents: list[dict[str, object]] = []
    participant_referents: dict[str, str] = {}
    for index, subject in enumerate(subjects, start=1):
        if not isinstance(subject, dict):
            continue
        referent_id = f"referent-{index}"
        participant_key = str(subject.get("participant_key") or "")
        if participant_key:
            participant_referents[participant_key] = referent_id
        referents.append(
            {
                "id": referent_id,
                "reference": subject.get("reference"),
                "referent_type": "PARTICIPANT",
                "participant_key": participant_key,
                "reference_mode": subject.get("reference_mode"),
                "candidate_participant_keys": subject.get("candidate_participant_keys"),
                "source_keys": subject.get("source_keys"),
                "valid_at": subject.get("valid_at"),
            }
        )
    semantic["referents"] = referents
    reasoning_by_attribution = {
        "DIRECT_SPEAKER_STATEMENT": "EVIDENCE_STATEMENT",
        "OTHER_SPEAKER_REPORT": "EVIDENCE_STATEMENT",
        "OBSERVER_SUMMARY": "EVIDENCE_SUMMARY",
        "DERIVED_INTERPRETATION": "DERIVED_INFERENCE",
        "HOST_IDENTITY": "HOST_IDENTITY",
        "BEHAVIORAL_FEEDBACK": "BEHAVIORAL_FEEDBACK",
        "HOST_ACTIVITY_STATISTIC": "HOST_ACTIVITY_STATISTIC",
    }
    for atom in semantic.get("atoms", []):
        if not isinstance(atom, dict):
            continue
        participant_key = str(atom.pop("subject_participant_key", "") or "")
        attribution = str(atom.pop("attribution", "") or "").strip().upper()
        atom["reasoning_kind"] = reasoning_by_attribution[attribution]
        atom["subject_referent_id"] = participant_referents.get(
            participant_key,
            "",
        )
    return semantic


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
        query="纸鹤计划进展如何",
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
        "query": "纸鹤计划进展如何",
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
    def test_reply_without_author_keeps_quote_without_identity_authority(self) -> None:
        context = build_request_identity_context(
            platform_id="synthetic", sender_id="synthetic-speaker", sender_name="Synthetic speaker",
            content=[{"type": "reply", "message_id": "synthetic-quoted-message",
                      "plain_text": "Synthetic quoted observation with no supplied author."}],
        )
        packet = {
            "request_identity_context": context,
            "reply_context": {"message_id": "synthetic-quoted-message",
                              "plain_text": "Synthetic quoted observation with no supplied author."},
        }

        def prepare(value):
            return build_l2_reader_prompt(
                query="纸鹤计划进展如何", evidence_packet=value, snapshot=l2_snapshot(),
                allowed_source_keys=(), allowed_participant_keys=tuple(
                    [context["sender"]["participant_key"]]
                ), participant_source_keys={}, pack_read_complete=True, allow_l3=False,
            )

        request = prepare(packet)
        payload = json.loads(request.user_prompt)
        self.assertEqual(payload["structured_ref_participant_keys"]["reply_target"], [])
        self.assertIn("Synthetic quoted observation with no supplied author.", request.user_prompt)
        self.assertNotIn("participant_key", context["reply_target"])
        self.assertIsNone(context["reply_target"]["same_account_as_sender"])
        for changes in (
            {"participant_key": "participant-outside-the-host-allowlist"},
            {"participant_key": 0},
            {"account_id": "nonempty-author-without-key"},
            {"binding_basis": "model_guessed_reply"},
        ):
            with self.subTest(changes=changes):
                altered = copy.deepcopy(packet)
                altered["request_identity_context"]["reply_target"].update(changes)
                with self.assertRaises(ValueError):
                    prepare(altered)

    def test_source_roles_are_host_bound_and_keep_bot_evidence_distinct(self) -> None:
        raw = _raw_certificate()
        packet = {"sources": [
            {"source_key": "s1", "sender_participant_key": "p1", "role": "BOT", "plain_text": "合成机器人摘要"},
            {"source_key": "s2", "sender_participant_key": "p1", "role": "USER", "plain_text": "合成用户回应"},
        ]}
        request = build_l2_reader_prompt(
            query="纸鹤计划进展如何", evidence_packet=packet, snapshot=l2_snapshot(),
            allowed_source_keys={"s1", "s2"}, allowed_participant_keys={"p1"},
            participant_source_keys={"p1": {"s1", "s2"}}, pack_read_complete=True, allow_l3=False,
        )
        response = _typed_resident_delta(raw, request)
        certificate = parse_l2_reader_response(response, request)
        self.assertEqual(certificate.atoms[0].evidence_roles, ("BOT",))
        self.assertEqual(certificate.atoms[1].evidence_roles, ("USER",))
        self.assertEqual(certificate.as_dict()["atoms"][0]["evidence_roles"], ["BOT"])
        mixed = copy.deepcopy(response)
        mixed["atoms"][0].update(reasoning_kind="EVIDENCE_SUMMARY", source_keys=["s1", "s2"])
        self.assertEqual(parse_l2_reader_response(mixed, request).atoms[0].evidence_roles, ("BOT", "USER"))
        forged = copy.deepcopy(response)
        forged["atoms"][0]["evidence_roles"] = ["USER"]
        with self.assertRaisesRegex(ValueError, "evidence_roles"):
            parse_l2_reader_response(forged, request)

    def test_reader_prose_displays_only_current_verified_aliases(self) -> None:
        first = 'participant:["synthetic","account-spruce"]'
        second = 'participant:["synthetic","account-pebble"]'
        query = "合成材料及字面p2的说明"
        quote = "原文含foo_p1和p2；xp3及p3suffix只是标记。"
        snapshot = replace(l2_snapshot(), query_sha256=hashlib.sha256(query.encode("utf-8")).hexdigest())
        packet = {"sources": [{
            "source_key": "synthetic-display-source", "sender_participant_key": first,
            "sender_id": "account-spruce", "sender_name": "潮杉", "sent_at": 100,
            "plain_text": quote,
            "components": [{"type": "mention", "account_id": "account-pebble"}],
            "mentions": [{"participant_key": second, "account_id": "account-pebble", "display_name": "砾川"}],
        }]}

        def request_for(evidence):
            return build_l2_reader_prompt(
                query=query, evidence_packet=evidence, snapshot=snapshot,
                allowed_source_keys={"synthetic-display-source"},
                allowed_participant_keys={first, second},
                participant_source_keys={first: {"synthetic-display-source"}, second: {"synthetic-display-source"}},
                pack_read_complete=True, allow_l3=False,
            )

        request = request_for(packet)
        first_alias, second_alias = request.participant_key_to_alias[first], request.participant_key_to_alias[second]
        self.assertEqual({first_alias, second_alias}, {"p3", "p4"})
        self.assertEqual(dict(request.participant_alias_display_names), {first_alias: "潮杉", second_alias: "砾川"})
        self.assertTrue(json.loads(request.user_prompt)["protocol"].endswith(".v4"))
        response = {
            "status": "PARTIAL",
            "referents": [{"id": "material-author", "reference": f"潮杉（{first_alias}）",
                "referent_type": "PARTICIPANT", "participant_key": first_alias,
                "reference_mode": "RESOLVED", "candidate_participant_keys": [],
                "source_keys": ["s1"], "valid_at": None}],
            "atoms": [{"id": "material-update", "statement":
                f"潮杉（{first_alias}）向{second_alias}介绍材料；{first_alias}保留原标签foo_p1和p2，xp3及p3suffix不变；别名（{second_alias}）尚未确认。",
                "subject_referent_id": "material-author", "reasoning_kind": "EVIDENCE_STATEMENT",
                "stance": "SUPPORTED", "source_keys": ["s1"], "source_spans": [quote],
                "importance": "REQUIRED", "confidence": 0.9}],
            "must_not_upgrade": [{"observed": f"{first_alias}介绍材料", "forbidden": [f"{second_alias}已确认"],
                "atom_ids": ["material-update"], "reason": f"{second_alias}尚未答复"}],
            "conflicts": [], "unresolved": [{"statement": f"{second_alias}是否收到材料未定",
                "source_keys": ["s1"], "atom_ids": ["material-update"]}],
            "open_obligations": [{"id": "confirmation", "question": f"{second_alias}是否确认？", "critical": True,
                "competing_interpretation_ids": [], "discriminator": f"查找{second_alias}的答复",
                "expected_information_gain": f"确认{second_alias}是否收到"}],
            "stop_reason": "FRONTIER_EXHAUSTED",
        }
        before = copy.deepcopy(response)
        certificate = parse_l2_reader_response(response, request)
        self.assertEqual(response, before)
        self.assertEqual(certificate.referents[0].reference, "潮杉")
        self.assertEqual(certificate.referents[0].participant_key, first)
        self.assertEqual(certificate.atoms[0].speaker_participant_key, first)
        self.assertEqual(certificate.atoms[0].source_keys, ("synthetic-display-source",))
        self.assertEqual(certificate.atoms[0].source_spans, (quote,))
        self.assertEqual(certificate.atoms[0].statement,
            "潮杉向砾川介绍材料；潮杉保留原标签foo_p1和p2，xp3及p3suffix不变；别名（砾川）尚未确认。")
        self.assertEqual(certificate.must_not_upgrade[0].forbidden, ("砾川已确认",))
        self.assertEqual(certificate.unresolved[0].statement, "砾川是否收到材料未定")
        self.assertEqual(certificate.open_obligations[0].discriminator, "查找砾川的答复")
        self.assertEqual(certificate.open_obligations[0].expected_information_gain, "确认砾川是否收到")
        self.assertEqual(json.loads(request.user_prompt)["evidence_packet"]["sources"][0]["plain_text"], quote)
        brief = compile_answer_context_packet(certificate).text
        self.assertNotRegex(brief, r"(?<![A-Za-z0-9])p[34](?![A-Za-z0-9])")
        self.assertIn("foo_p1", brief)
        self.assertIn("p2", brief)

        missing = copy.deepcopy(packet)
        missing["sources"][0]["sender_name"] = ""
        anonymous_response = copy.deepcopy(response)
        anonymous_response["referents"][0]["reference"] = first_alias
        anonymous_response["atoms"][0]["statement"] = f"{first_alias}向{second_alias}介绍材料。"
        unknown = parse_l2_reader_response(anonymous_response, request_for(missing))
        self.assertEqual(unknown.referents[0].reference, "匿名成员2（显示名未知）")
        self.assertEqual(unknown.referents[0].participant_key, first)
        self.assertEqual(unknown.atoms[0].statement, "匿名成员2（显示名未知）向砾川介绍材料。")
        self.assertEqual(unknown.atoms[0].source_spans, (quote,))
        duplicate = copy.deepcopy(packet)
        duplicate["sources"][0]["mentions"][0]["display_name"] = "潮杉"
        distinguished = parse_l2_reader_response(anonymous_response, request_for(duplicate))
        self.assertEqual(distinguished.referents[0].reference, "潮杉（匿名成员2）")
        self.assertEqual(distinguished.referents[0].participant_key, first)
        self.assertEqual(distinguished.atoms[0].statement, "潮杉（匿名成员2）向潮杉（匿名成员1）介绍材料。")
        # Display projection never changes claims that contain no current alias.
        plain_response = copy.deepcopy(response)
        for records in (plain_response["referents"], plain_response["atoms"], plain_response["must_not_upgrade"],
                        plain_response["unresolved"], plain_response["open_obligations"]):
            for record in records:
                for key in ("reference", "statement", "observed", "reason", "question", "discriminator", "expected_information_gain"):
                    if key in record:
                        record[key] = "材料情况仍待核对。"
                if "forbidden" in record:
                    record["forbidden"] = ["材料已确认"]
        self.assertEqual(parse_l2_reader_response(plain_response, request_for(missing)).status, "PARTIAL")
        self.assertEqual(parse_l2_reader_response(plain_response, request_for(duplicate)).status, "PARTIAL")

    def test_resident_prompt_field_names_and_types_match_strict_parser(self) -> None:
        query = "合成档案何时更新"
        snapshot = replace(l2_snapshot(), query_sha256=hashlib.sha256(query.encode("utf-8")).hexdigest())
        request = build_l2_reader_prompt(
            query=query, evidence_packet={"sources": [
                {"source_key": "synthetic-source", "plain_text": "合成档案尚待核对。"},
            ]}, snapshot=snapshot, allowed_source_keys={"synthetic-source"},
            allowed_participant_keys=set(), pack_read_complete=True, allow_l3=False,
            semantic_none_allowed=False,
        )
        response = {
            "status": "PARTIAL",
            "referents": [{"id": "archive", "reference": "合成档案", "referent_type": "WORK",
                           "participant_key": "", "reference_mode": "EVIDENCE_REF",
                           "candidate_participant_keys": [], "source_keys": ["s1"], "valid_at": None}],
            "atoms": [], "must_not_upgrade": [], "conflicts": [],
            "unresolved": [{"statement": "当前证据不足以确定更新时间。", "source_keys": [], "atom_ids": []}],
            "open_obligations": [{"id": "update-date", "question": "更新时间仍待核对。", "critical": True,
                                  "competing_interpretation_ids": [], "discriminator": "",
                                  "expected_information_gain": "确认更新时间。"}],
            "stop_reason": "FRONTIER_EXHAUSTED",
        }
        contract = request.system_prompt.split("Compact semantic contract:\n", 1)[1]
        declared = re.search(r"referents<=16，每项字段恰为 ([^；]+)；", contract)
        self.assertIsNotNone(declared)
        self.assertEqual(set(declared.group(1).split(",")), set(response["referents"][0]))
        self.assertNotRegex(contract, r"\bcandidates\b")
        self.assertIn("candidate_participant_keys=[]", contract)
        self.assertIn("discriminator和expected_information_gain必须为字符串", contract)
        self.assertIn("可选字段aggregate_ids必须为字符串数组", contract)
        self.assertIn("aggregate_ids非空时reasoning_kind只能为HOST_ACTIVITY_STATISTIC或DERIVED_INFERENCE", contract)
        self.assertIn("source_keys和atom_ids必须为字符串数组", contract)
        self.assertIn("不得输出 must_include 或 source_spans", contract)
        certificate = parse_l2_reader_response(response, request)
        self.assertEqual(certificate.status, "PARTIAL")
        self.assertEqual(certificate.unresolved[0].basis, "PACKET_COVERAGE_GAP")
        self.assertEqual(certificate.open_obligations[0].expected_information_gain, "确认更新时间。")
        wrong = copy.deepcopy(response)
        wrong["referents"][0]["candidates"] = wrong["referents"][0].pop("candidate_participant_keys")
        with self.assertRaisesRegex(ValueError, "candidate_participant_keys.*candidates"):
            parse_l2_reader_response(wrong, request)

    def test_resident_activity_statistics_use_a_host_allowlist_separate_from_sample_sources(self) -> None:
        raw, allowlist = _raw_activity_certificate()
        descriptor = raw["aggregates"][0]
        packet = {
            "messages": [{"source_key": "s1", "sender_participant_key": "p1", "plain_text": "合成观察样本"}],
            "participant_activity": [{"participant_key": "p1", "message_count": 1,
                "messages": [{"source_key": "s1"}], "window_statistics": descriptor}],
        }
        request = build_l2_reader_prompt(
            query="纸鹤计划进展如何", evidence_packet=packet, snapshot=l2_snapshot(),
            allowed_source_keys={"s1"}, allowed_participant_keys={"p1"},
            participant_source_keys={"p1": {"s1"}}, pack_read_complete=True, allow_l3=False,
        )
        self.assertEqual(dict(request.allowed_aggregates), allowlist)
        schema = evidence_certificate_v2_schema(
            snapshot=l2_snapshot(), packet_sha256=request.packet_sha256,
            allowed_source_keys={"s1"}, allowed_participant_keys={"p1"},
            pack_read_complete=True, allowed_aggregates=allowlist,
        )
        self.assertEqual(schema["properties"]["aggregates"]["items"]["enum"], [descriptor])
        statistic_branch = schema["allOf"][0]["then"]["properties"]["atoms"]["items"]["anyOf"][1]
        self.assertEqual(statistic_branch["properties"]["attribution"]["enum"],
                         ["HOST_ACTIVITY_STATISTIC", "DERIVED_INTERPRETATION"])
        response = _typed_resident_delta(raw, request)
        certificate = parse_l2_reader_response(response, request)
        self.assertEqual(certificate.atoms[0].source_keys, ())
        self.assertEqual(certificate.atoms[0].aggregate_ids, (descriptor["aggregate_id"],))
        self.assertEqual(certificate.aggregates[0].as_dict()["source_count"], 3)
        derived_response = copy.deepcopy(response)
        derived_response["atoms"][0].update(reasoning_kind="DERIVED_INFERENCE",
            statement="完整观察窗口的发言集中在早间，但这不足以确定参与者的线下作息。")
        derived = parse_l2_reader_response(derived_response, request)
        self.assertEqual(derived.atoms[0].attribution, "DERIVED_INTERPRETATION")
        self.assertEqual(derived.atoms[0].source_keys, ())
        self.assertEqual(derived.atoms[0].speaker_participant_key, "")
        self.assertEqual(derived.aggregates, certificate.aggregates)
        changed = copy.deepcopy(response)
        changed["atoms"][0]["aggregate_ids"] = ["activity-window:" + "e" * 64]
        with self.assertRaisesRegex(ValueError, "outside the host allowlist"):
            parse_l2_reader_response(changed, request)
        changed = copy.deepcopy(response)
        changed["referents"][0]["reference_mode"] = "UNIQUE_ALIAS"
        with self.assertRaisesRegex(ValueError, "source_keys is required"):
            parse_l2_reader_response(changed, request)
        changed = copy.deepcopy(response)
        changed["aggregates"] = [descriptor]
        with self.assertRaisesRegex(ValueError, "host-owned fields"):
            parse_l2_reader_response(changed, request)

    def test_reader_delivery_preserves_raw_alias_evidence_and_marks_stored_inference(self) -> None:
        participant = "synthetic-reader-account"
        packet = {
            "sources": [{"source_key": "synthetic-source", "sender_participant_key": participant,
                         "plain_text": "蓝杉，材料到了。", "sender_name": "记录员"}],
            "person_reference_candidates": {"references": [{"reference": "记录员", "candidate_participants": [{
                "participant_key": participant, "alias_observations": [{"source_key": "synthetic-source",
                    "alias": "记录员", "relation": "SPEAKER"}],
            }]}]},
            "semantic_evidence": [{"memory": {"content": "p7曾整理材料", "source_keys": ["synthetic-source"]}}],
            "expanded_episodes": [{"summary": "p7整理了材料", "messages": [{"source_key": "synthetic-source"}]}],
        }
        before = copy.deepcopy(packet)
        request = build_l2_reader_prompt(
            query="纸鹤计划进展如何", evidence_packet=packet, snapshot=l2_snapshot(),
            allowed_source_keys={"synthetic-source"}, allowed_participant_keys={participant},
            participant_source_keys={participant: {"synthetic-source"}}, pack_read_complete=True, allow_l3=False,
        )
        delivered = json.loads(request.user_prompt)["evidence_packet"]
        aliased = _alias_evidence_packet(before, source_aliases=request.source_key_to_alias,
                                         participant_aliases=request.participant_key_to_alias)
        self.assertEqual(packet, before)
        self.assertEqual(delivered["sources"], aliased["sources"])
        self.assertEqual(delivered["person_reference_candidates"], aliased["person_reference_candidates"])
        self.assertNotIn("content", delivered["semantic_evidence"][0]["memory"])
        self.assertEqual(delivered["semantic_evidence"][0]["memory"]["reader_withheld_fields"][0]["field"], "content")
        for item in (delivered["semantic_evidence"][0]["memory"], delivered["expanded_episodes"][0]):
            self.assertEqual(item["reader_evidence_basis"], "STORED_DERIVATION")
            self.assertEqual(item["reader_text_identity_scope"], "HISTORICAL_UNMAPPED")
        self.assertIn("唯一可用的受限快照子集", request.system_prompt)
        self.assertNotIn("唯一且完整", request.system_prompt)
        self.assertIn("不能证明任意reference属于该账号", request.system_prompt)

    def test_participant_aliases_avoid_original_labels_without_rewriting_evidence(self) -> None:
        first, second = "synthetic-account-alpha", "synthetic-account-beta"
        historical_text = "p1提到材料，\np2记录了数量；xp3和p3suffix只是标记。"
        packet = {
            "sources": [
                {"source_key": "synthetic-source-alpha", "sender_participant_key": first,
                 "plain_text": "材料已分类。"},
                {"source_key": "synthetic-source-beta", "sender_participant_key": second,
                 "plain_text": "封套已备妥。"},
            ],
            "semantic_evidence": [{"memory": {"content": historical_text}}],
        }
        before = copy.deepcopy(packet)
        arguments = {
            "query": "纸鹤计划进展如何",
            "evidence_packet": packet,
            "snapshot": replace(l2_snapshot(), sender_participant_key=first),
            "allowed_source_keys": {"synthetic-source-alpha", "synthetic-source-beta"},
            "allowed_participant_keys": {first, second},
            "participant_source_keys": {
                first: {"synthetic-source-alpha"}, second: {"synthetic-source-beta"},
            },
            "pack_read_complete": True,
            "allow_l3": False,
        }
        request = build_l2_reader_prompt(**arguments)
        self.assertEqual(dict(request.participant_key_to_alias), {first: "p3", second: "p4"})
        self.assertEqual(dict(request.participant_alias_to_key), {"p3": first, "p4": second})
        delivered = json.loads(request.user_prompt)["evidence_packet"]
        self.assertEqual(packet, before)
        self.assertNotIn("content", delivered["semantic_evidence"][0]["memory"])
        self.assertEqual(delivered["semantic_evidence"][0]["memory"]["reader_withheld_fields"][0]["text_sha256"],
                         hashlib.sha256(historical_text.encode("utf-8")).hexdigest())
        self.assertEqual(delivered["sources"][0]["plain_text"], "材料已分类。")
        response = {
            "status": "CERTIFIED",
            "referents": [{"id": "synthetic-reference", "reference": "材料整理者",
                "referent_type": "PARTICIPANT", "participant_key": "p3",
                "reference_mode": "RESOLVED", "candidate_participant_keys": [],
                "source_keys": ["s1"], "valid_at": None}],
            "atoms": [{"id": "synthetic-atom", "statement": "材料整理者说材料已分类。",
                "subject_referent_id": "synthetic-reference", "reasoning_kind": "EVIDENCE_STATEMENT",
                "stance": "SUPPORTED", "source_keys": ["s1"], "source_spans": ["材料已分类。"],
                "importance": "REQUIRED", "confidence": 0.9}],
            "must_include": ["synthetic-atom"], "must_not_upgrade": [], "conflicts": [],
            "unresolved": [], "open_obligations": [], "stop_reason": "CERTIFIED_CLOSE",
        }
        certificate = parse_l2_reader_response(response, request)
        self.assertEqual(certificate.subjects[0].participant_key, first)
        response["referents"][0]["participant_key"] = "p1"
        with self.assertRaisesRegex(ValueError, "outside the participant alias map"):
            parse_l2_reader_response(response, request)

        query = "p3所指材料的情况"
        arguments.update(query=query, snapshot=replace(
            arguments["snapshot"], query_sha256=hashlib.sha256(query.encode("utf-8")).hexdigest(),
        ))
        query_request = build_l2_reader_prompt(**arguments)
        self.assertEqual(dict(query_request.participant_key_to_alias), {first: "p4", second: "p5"})
        self.assertEqual(packet, before)

    def test_resident_preserves_uncited_packet_gap_without_fabricating_evidence(self) -> None:
        packet = {"messages": [{
            "source_key": "s1", "sender_participant_key": "p1", "text": "纸张已经到齐",
        }]}
        request = build_l2_reader_prompt(
            query="纸鹤计划进展如何", evidence_packet=packet, snapshot=l2_snapshot(),
            allowed_source_keys={"s1"}, allowed_participant_keys={"p1"},
            participant_source_keys={"p1": {"s1"}},
            pack_read_complete=True, allow_l3=False,
        )
        raw = _raw_packet_gap_certificate()
        response = _typed_resident_delta(raw, request)
        certificate = parse_l2_reader_response(response, request)
        self.assertEqual(certificate.status, "PARTIAL")
        self.assertEqual(certificate.unresolved[0].basis, "PACKET_COVERAGE_GAP")
        self.assertEqual(certificate.unresolved[0].source_keys, ())
        self.assertEqual(certificate.unresolved[0].atom_ids, ())
        self.assertEqual(certificate.open_obligations[0].question, raw["open_obligations"][0]["question"])
        self.assertEqual(certificate.packet_sha256, request.packet_sha256)
        self.assertIn("不能据此断言全库或历史中不存在", request.system_prompt)
        response.update(status="CERTIFIED", stop_reason="CERTIFIED_CLOSE")
        with self.assertRaisesRegex(ValueError, "CERTIFIED cannot retain packet coverage gaps"):
            parse_l2_reader_response(response, request)

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
        self.assertEqual(json.loads(outputs[0]), ["source-a", "source-m", "source-z"])

    def test_prompt_uses_compact_alias_protocol_and_keeps_host_fields_local(
        self,
    ) -> None:
        packet = {
            "messages": [
                {
                    "source_key": "source-one-canonical",
                    "sender_participant_key": "participant-one-canonical",
                    "text": "拼图玩腻了",
                },
                {
                    "source_key": "source-two-canonical",
                    "sender_participant_key": "participant-one-canonical",
                    "text": "我可能会买",
                },
            ]
        }
        request = build_l2_reader_prompt(
            query="纸鹤计划进展如何",
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
                        "text": "拼图玩腻了",
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
            schema["properties"]["subjects"]["items"]["properties"]["source_keys"][
                "maxItems"
            ],
            MAX_CERTIFICATE_SOURCE_KEYS,
        )
        self.assertEqual(
            schema["properties"]["atoms"]["items"]["properties"]["source_keys"][
                "maxItems"
            ],
            MAX_CERTIFICATE_SOURCE_KEYS,
        )
        self.assertEqual(
            schema["properties"]["atoms"]["items"]["properties"]["source_spans"][
                "maxItems"
            ],
            MAX_CERTIFICATE_SOURCE_KEYS,
        )
        self.assertEqual(
            schema["properties"]["unresolved"]["items"]["properties"]["source_keys"][
                "maxItems"
            ],
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
            query="纸鹤计划进展如何",
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
        with self.assertRaisesRegex(
            ValueError, "speaker_participant_key is host-owned"
        ):
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
            query="纸鹤计划进展如何",
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

    def test_resident_contract_maps_only_verified_participant_referents(
        self,
    ) -> None:
        packet = {
            "messages": [
                {
                    "source_key": "source-alpha",
                    "sender_participant_key": "participant-alpha",
                    "text": "synthetic participant evidence",
                },
                {
                    "source_key": "source-beta",
                    "sender_participant_key": "participant-beta",
                    "text": "synthetic third-party identity evidence",
                }
            ]
        }
        request = build_l2_reader_prompt(
            query="纸鹤计划进展如何",
            evidence_packet=packet,
            snapshot=l2_snapshot(),
            allowed_source_keys={"source-alpha", "source-beta"},
            allowed_participant_keys={"participant-alpha", "participant-beta"},
            participant_source_keys={
                "participant-alpha": {"source-alpha"},
                "participant-beta": {"source-beta"},
            },
            pack_read_complete=True,
            allow_l3=False,
        )
        raw = _raw_certificate()
        raw.update(
            {
                "packet_sha256": request.packet_sha256,
                "subjects": [
                    {
                        "reference": "synthetic member",
                        "participant_key": "participant-alpha",
                        "reference_mode": "UNIQUE_ALIAS",
                        "candidate_participant_keys": [],
                        "source_keys": ["source-alpha"],
                        "valid_at": None,
                    }
                ],
                "atoms": [
                    {
                        "id": "participant-fact",
                        "statement": "A synthetic participant-bound statement.",
                        "speaker_participant_key": "participant-alpha",
                        "subject_participant_key": "participant-alpha",
                        "attribution": "DIRECT_SPEAKER_STATEMENT",
                        "stance": "SUPPORTED",
                        "source_keys": ["source-alpha"],
                        "source_spans": ["synthetic participant evidence"],
                        "importance": "REQUIRED",
                        "confidence": 0.9,
                    }
                ],
                "must_include": ["participant-fact"],
                "must_not_upgrade": [],
                "conflicts": [],
                "unresolved": [],
                "open_obligations": [],
            }
        )

        certificate = parse_l2_reader_response(
            _typed_resident_delta(raw, request),
            request,
        )

        self.assertEqual(certificate.subjects[0].participant_key, "participant-alpha")
        self.assertEqual(
            certificate.atoms[0].subject_participant_key,
            "participant-alpha",
        )
        self.assertEqual(
            certificate.atoms[0].speaker_participant_key,
            "participant-alpha",
        )
        self.assertEqual(
            certificate.atoms[0].attribution,
            "DIRECT_SPEAKER_STATEMENT",
        )

        third_party = _typed_resident_delta(raw, request)
        third_party["referents"].append(
            {
                "id": "referent-third-party",
                "reference": "synthetic third party",
                "referent_type": "PARTICIPANT",
                "participant_key": request.participant_key_to_alias[
                    "participant-beta"
                ],
                "reference_mode": "UNIQUE_ALIAS",
                "candidate_participant_keys": [],
                "source_keys": [request.source_key_to_alias["source-beta"]],
                "valid_at": None,
            }
        )
        third_party["atoms"][0]["subject_referent_id"] = "referent-third-party"
        reported = parse_l2_reader_response(third_party, request)
        self.assertEqual(reported.atoms[0].attribution, "OTHER_SPEAKER_REPORT")
        self.assertEqual(
            reported.atoms[0].speaker_participant_key,
            "participant-alpha",
        )
        for directive in (
            "referents",
            "referent_type",
            "subject_referent_id",
            "reasoning_kind",
            "不得输出 attribution",
            "PARTICIPANT",
            "WORK",
            "ENTITY",
            "TOPIC",
        ):
            self.assertIn(directive, request.system_prompt)

    def test_resident_host_and_structured_referents_use_host_owned_bindings(
        self,
    ) -> None:
        packet = {
            "request_identity_context": {
                "authority": "current_platform_event",
                "sender": {
                    "participant_key": "p1",
                    "binding_basis": "message_sender",
                },
                "mentions": [
                    {
                        "participant_key": "p2",
                        "binding_basis": "structured_mention",
                    }
                ],
                "reply_target": {
                    "participant_key": "p3",
                    "binding_basis": "structured_reply",
                },
            },
            "messages": [
                {
                    "source_key": "s1",
                    "sender_participant_key": "p1",
                    "text": "synthetic first source",
                },
                {
                    "source_key": "s2",
                    "sender_participant_key": "p1",
                    "text": "synthetic second source",
                },
            ],
        }
        request = build_l2_reader_prompt(
            query="纸鹤计划进展如何",
            evidence_packet=packet,
            snapshot=l2_snapshot(),
            allowed_source_keys={"s1", "s2"},
            allowed_participant_keys={"p1", "p2", "p3", "p4"},
            participant_source_keys={
                "p1": {"s1", "s2"},
                "p2": {"s1"},
                "p3": {"s2"},
                "p4": {"s1", "s2"},
            },
            pack_read_complete=True,
            allow_l3=False,
        )
        payload = json.loads(request.user_prompt)
        self.assertEqual(
            payload["current_sender_participant_key"], request.participant_key_to_alias["p1"],
        )
        self.assertEqual(
            payload["structured_ref_participant_keys"],
            {"mentions": [request.participant_key_to_alias["p2"]],
             "reply_target": [request.participant_key_to_alias["p3"]]},
        )

        def resident_delta(mode: str, participant_key: str) -> dict[str, object]:
            raw = _raw_certificate()
            raw["packet_sha256"] = request.packet_sha256
            raw["subjects"][0].update(
                {
                    "reference": "synthetic current-event reference",
                    "participant_key": participant_key,
                    "reference_mode": mode,
                    "candidate_participant_keys": [],
                    "source_keys": [],
                }
            )
            for atom in raw["atoms"]:
                atom["subject_participant_key"] = participant_key
            return _typed_resident_delta(raw, request)

        host_bound = parse_l2_reader_response(resident_delta("HOST", "p1"), request)
        self.assertEqual(host_bound.subjects[0].participant_key, "p1")
        for participant_key in ("p2", "p3"):
            structured = parse_l2_reader_response(
                resident_delta("STRUCTURED_REF", participant_key),
                request,
            )
            self.assertEqual(
                structured.subjects[0].participant_key,
                participant_key,
            )

        with self.assertRaisesRegex(ValueError, "not admissible for HOST"):
            parse_l2_reader_response(resident_delta("HOST", "p2"), request)
        with self.assertRaisesRegex(ValueError, "not admissible for STRUCTURED_REF"):
            parse_l2_reader_response(
                resident_delta("STRUCTURED_REF", "p4"),
                request,
            )
        for participant_key, expected_mode in (
            ("p1", "HOST"), ("p2", "STRUCTURED_REF"), ("p4", "UNIQUE_ALIAS"),
        ):
            slim = resident_delta("RESOLVED", participant_key)
            slim.pop("must_include")
            for atom in slim["atoms"]:
                atom.pop("source_spans")
            if participant_key == "p4":
                slim["referents"][0]["source_keys"] = ["s1"]
            with self.subTest(participant=participant_key):
                resolved = parse_l2_reader_response(slim, request)
                self.assertEqual(resolved.subjects[0].reference_mode, expected_mode)
                self.assertEqual(resolved.subjects[0].participant_key, participant_key)
                self.assertEqual(resolved.must_include, ("a1",))
                self.assertEqual(resolved.atoms[0].source_spans, ())
                self.assertEqual(resolved.referents[0].participant_key, participant_key)
        with self.assertRaisesRegex(ValueError, "source_keys is required"):
            parse_l2_reader_response(resident_delta("RESOLVED", "p4"), request)
        with self.assertRaisesRegex(ValueError, "coverage is incomplete"):
            parse_l2_reader_response(slim, replace(request, person_candidates_complete=False))
        contradictory = copy.deepcopy(slim)
        contradictory["must_include"] = []
        with self.assertRaisesRegex(ValueError, "every and only REQUIRED"):
            parse_l2_reader_response(contradictory, request)
        optional_only = copy.deepcopy(slim)
        for atom in optional_only["atoms"]:
            atom["importance"] = "OPTIONAL"
        with self.assertRaisesRegex(ValueError, "at least one REQUIRED atom"):
            parse_l2_reader_response(optional_only, request)
        self.assertIn("全局 participant allowlist", request.system_prompt)

    def test_resident_nonparticipant_referents_never_become_participants(
        self,
    ) -> None:
        packet = {
            "messages": [
                {
                    "source_key": "source-alpha",
                    "sender_participant_key": "participant-alpha",
                    "text": "synthetic evidence about several non-person referents",
                }
            ]
        }
        request = build_l2_reader_prompt(
            query="纸鹤计划进展如何",
            evidence_packet=packet,
            snapshot=l2_snapshot(),
            allowed_source_keys={"source-alpha"},
            allowed_participant_keys={"participant-alpha"},
            participant_source_keys={
                "participant-alpha": {"source-alpha"},
            },
            pack_read_complete=True,
            allow_l3=False,
        )
        response = {
            "status": "CERTIFIED",
            "referents": [
                {
                    "id": f"referent-{kind.casefold()}",
                    "reference": f"synthetic {kind.casefold()}",
                    "referent_type": kind,
                    "participant_key": "",
                    "reference_mode": "EVIDENCE_REF",
                    "candidate_participant_keys": [],
                    "source_keys": ["s1"],
                    "valid_at": None,
                }
                for kind in ("WORK", "ENTITY", "TOPIC")
            ],
            "atoms": [
                {
                    "id": f"fact-{kind.casefold()}",
                    "statement": f"A synthetic statement about a {kind.casefold()}.",
                    "subject_referent_id": f"referent-{kind.casefold()}",
                    "reasoning_kind": "EVIDENCE_STATEMENT",
                    "stance": "SUPPORTED",
                    "source_keys": ["s1"],
                    "source_spans": ["synthetic evidence"],
                    "importance": "REQUIRED" if kind == "WORK" else "OPTIONAL",
                    "confidence": 0.8,
                }
                for kind in ("WORK", "ENTITY", "TOPIC")
            ],
            "must_include": ["fact-work"],
            "must_not_upgrade": [],
            "conflicts": [],
            "unresolved": [],
            "open_obligations": [],
            "stop_reason": "CERTIFIED_CLOSE",
        }

        certificate = parse_l2_reader_response(response, request)

        self.assertEqual(certificate.subjects, ())
        self.assertEqual([item.referent_type for item in certificate.referents], ["WORK", "ENTITY", "TOPIC"])
        self.assertEqual(certificate.atoms[0].subject_referent_id, "referent-work")
        self.assertTrue(
            all(not atom.subject_participant_key for atom in certificate.atoms)
        )
        self.assertTrue(
            all(
                atom.speaker_participant_key == "participant-alpha"
                for atom in certificate.atoms
            )
        )
        surface_packet = compile_surface_packet(certificate)
        validate_surface_packet(surface_packet, certificate)
        self.assertIn(
            "A synthetic statement about a work.",
            surface_packet.text,
        )
        brief = compile_answer_context_packet(certificate).as_dict()
        subjects = {item["entity"]: item for item in brief["referents"]}
        self.assertEqual(subjects[brief["facts"][0]["subject"]]["type"], "work")

        no_explicit_subject = copy.deepcopy(response)
        no_explicit_subject["referents"] = []
        no_explicit_subject["atoms"] = [copy.deepcopy(response["atoms"][0])]
        no_explicit_subject["atoms"][0]["subject_referent_id"] = ""
        no_explicit_subject["must_include"] = ["fact-work"]
        unbound = parse_l2_reader_response(no_explicit_subject, request)
        self.assertEqual(unbound.atoms[0].speaker_participant_key, "participant-alpha")
        # Host-owned authorship is not a default semantic subject.
        self.assertEqual(unbound.atoms[0].subject_participant_key, "")

    def test_resident_multi_speaker_summary_has_no_speaker(self) -> None:
        packet = {
            "messages": [
                {
                    "source_key": "source-alpha",
                    "sender_participant_key": "participant-alpha",
                },
                {
                    "source_key": "source-beta",
                    "sender_participant_key": "participant-beta",
                },
            ]
        }
        request = build_l2_reader_prompt(
            query="纸鹤计划进展如何",
            evidence_packet=packet,
            snapshot=l2_snapshot(),
            allowed_source_keys={"source-alpha", "source-beta"},
            allowed_participant_keys={"participant-alpha", "participant-beta"},
            participant_source_keys={
                "participant-alpha": {"source-alpha"},
                "participant-beta": {"source-beta"},
            },
            pack_read_complete=True,
            allow_l3=False,
        )
        response = {
            "status": "CERTIFIED",
            "referents": [],
            "atoms": [
                {
                    "id": "exchange-summary",
                    "statement": "Two synthetic speakers participated.",
                    "subject_referent_id": "",
                    "reasoning_kind": "EVIDENCE_SUMMARY",
                    "stance": "SUPPORTED",
                    "source_keys": ["s1", "s2"],
                    "source_spans": [],
                    "importance": "REQUIRED",
                    "confidence": 0.8,
                }
            ],
            "must_include": ["exchange-summary"],
            "must_not_upgrade": [],
            "conflicts": [],
            "unresolved": [],
            "open_obligations": [],
            "stop_reason": "CERTIFIED_CLOSE",
        }

        certificate = parse_l2_reader_response(response, request)

        self.assertEqual(certificate.atoms[0].attribution, "OBSERVER_SUMMARY")
        self.assertEqual(certificate.atoms[0].speaker_participant_key, "")

        model_owned_attribution = copy.deepcopy(response)
        model_owned_attribution["atoms"][0]["attribution"] = "OBSERVER_SUMMARY"
        with self.assertRaisesRegex(ValueError, "attribution"):
            parse_l2_reader_response(model_owned_attribution, request)

        model_owned_speaker = copy.deepcopy(response)
        model_owned_speaker["atoms"][0]["speaker_participant_key"] = "p1"
        with self.assertRaisesRegex(ValueError, "speaker_participant_key"):
            parse_l2_reader_response(model_owned_speaker, request)

    def test_resident_activity_prediction_is_derived_not_direct(self) -> None:
        packet = {
            "messages": [
                {
                    "source_key": "source-alpha",
                    "sender_participant_key": "participant-alpha",
                },
                {
                    "source_key": "source-beta",
                    "sender_participant_key": "participant-alpha",
                },
            ]
        }
        request = build_l2_reader_prompt(
            query="纸鹤计划进展如何",
            evidence_packet=packet,
            snapshot=l2_snapshot(),
            allowed_source_keys={"source-alpha", "source-beta"},
            allowed_participant_keys={"participant-alpha"},
            participant_source_keys={
                "participant-alpha": {"source-alpha", "source-beta"},
            },
            pack_read_complete=True,
            allow_l3=False,
        )
        response = {
            "status": "CERTIFIED",
            "referents": [],
            "atoms": [
                {
                    "id": "activity-inference",
                    "statement": "A future activity window is inferred from prior timing.",
                    "subject_referent_id": "",
                    "reasoning_kind": "DERIVED_INFERENCE",
                    "stance": "SUPPORTED",
                    "source_keys": ["s1", "s2"],
                    "source_spans": [],
                    "importance": "REQUIRED",
                    "confidence": 0.7,
                }
            ],
            "must_include": ["activity-inference"],
            "must_not_upgrade": [],
            "conflicts": [],
            "unresolved": [],
            "open_obligations": [],
            "stop_reason": "CERTIFIED_CLOSE",
        }

        certificate = parse_l2_reader_response(response, request)

        self.assertEqual(certificate.atoms[0].attribution, "DERIVED_INTERPRETATION")
        self.assertEqual(certificate.atoms[0].speaker_participant_key, "")

    def test_resident_rejects_participant_fields_outside_participant_referents(
        self,
    ) -> None:
        packet = {
            "messages": [
                {
                    "source_key": "source-alpha",
                    "sender_participant_key": "participant-alpha",
                }
            ]
        }
        request = build_l2_reader_prompt(
            query="纸鹤计划进展如何",
            evidence_packet=packet,
            snapshot=l2_snapshot(),
            allowed_source_keys={"source-alpha"},
            allowed_participant_keys={"participant-alpha"},
            participant_source_keys={
                "participant-alpha": {"source-alpha"},
            },
            pack_read_complete=True,
            allow_l3=False,
        )
        response = {
            "status": "CERTIFIED",
            "referents": [
                {
                    "id": "referent-work",
                    "reference": "synthetic work",
                    "referent_type": "WORK",
                    "participant_key": "p1",
                    "reference_mode": "EVIDENCE_REF",
                    "candidate_participant_keys": [],
                    "source_keys": ["s1"],
                    "valid_at": None,
                }
            ],
            "atoms": [
                {
                    "id": "fact-work",
                    "statement": "A synthetic work statement.",
                    "subject_referent_id": "referent-work",
                    "reasoning_kind": "EVIDENCE_STATEMENT",
                    "stance": "SUPPORTED",
                    "source_keys": ["s1"],
                    "source_spans": ["synthetic evidence"],
                    "importance": "REQUIRED",
                    "confidence": 0.8,
                }
            ],
            "must_include": ["fact-work"],
            "must_not_upgrade": [],
            "conflicts": [],
            "unresolved": [],
            "open_obligations": [],
            "stop_reason": "CERTIFIED_CLOSE",
        }
        with self.assertRaisesRegex(ValueError, "non-participant referent"):
            parse_l2_reader_response(response, request)

        model_owned_subject = copy.deepcopy(response)
        model_owned_subject["referents"][0]["participant_key"] = ""
        model_owned_subject["atoms"][0]["subject_participant_key"] = "p1"
        with self.assertRaisesRegex(ValueError, "subject_participant_key"):
            parse_l2_reader_response(model_owned_subject, request)

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
                query="纸鹤计划进展如何",
                evidence_packet={"messages": [{"source_key": "source-outside"}]},
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
            query="纸鹤计划进展如何",
            evidence_packet=packet,
            snapshot=l2_snapshot(),
            allowed_source_keys={"s1", "s2"},
            allowed_participant_keys={"p1", "p2"},
            participant_source_keys={"p1": {"s1"}, "p2": {"s2"}},
            person_candidates_complete=bool(coverage["person_candidates_complete"]),
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
            query="纸鹤计划进展如何",
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
                query="纸鹤计划进展如何",
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
            query="纸鹤计划进展如何",
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
            query="纸鹤计划进展如何",
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
            query="纸鹤计划进展如何",
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
            query="纸鹤计划进展如何",
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
            query="纸鹤计划进展如何",
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
            query="纸鹤计划进展如何",
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
            query="纸鹤计划进展如何",
            evidence_packet=packet,
            snapshot=l2_snapshot(),
            allowed_source_keys=(),
            pack_read_complete=True,
        )
        forbidden_request = build_l2_reader_prompt(
            query="纸鹤计划进展如何",
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
            query="纸鹤计划进展如何",
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
        with self.assertRaisesRegex(
            ValueError, "speaker_participant_key is host-owned"
        ):
            parse_l2_reader_response(model_owned_speaker, request)

        unresolved_subject = copy.deepcopy(raw)
        unresolved_subject["atoms"][0]["subject_participant_key"] = "participant-beta"
        with self.assertRaisesRegex(ValueError, r"subject participant is not resolved"):
            parse_l2_reader_response(
                _semantic_delta(unresolved_subject, request),
                request,
            )

        unbound_request = build_l2_reader_prompt(
            query="纸鹤计划进展如何",
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
                query="纸鹤计划进展如何",
                evidence_packet=packet,
                snapshot=l2_snapshot(),
                allowed_source_keys={"source-alpha"},
                allowed_participant_keys={"participant-alpha"},
                participant_source_keys={"participant-gamma": ["source-alpha"]},
                pack_read_complete=True,
            )
        with self.assertRaisesRegex(ValueError, "source outside the allowlist"):
            build_l2_reader_prompt(
                query="纸鹤计划进展如何",
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
            query="纸鹤计划进展如何",
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
                self.assertTrue(certificate.unresolved or certificate.open_obligations)
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
