from __future__ import annotations

import copy
import json
import unittest

from mr_memory.certificate import parse_evidence_certificate
from mr_memory.derivations import build_stored_derivation, validate_stored_derivation
from mr_memory.reader import build_l2_reader_prompt, parse_l2_reader_response
from mr_memory.snapshot import stable_sha256
from mr_memory.surface import compile_answer_context_packet, validate_answer_context_packet
from tests.test_certificate_v2 import _snapshot


def _summary_descriptor():
    return build_stored_derivation(
        kind="episode", owner_id="synthetic-episode",
        text={"title": "合成纸鹤活动", "summary": "讨论决定先整理材料，场地仍待确认。"},
        scope={"umo": _snapshot().umo, "before_sent_at": _snapshot().cutoff_at,
               "message_upper_bound": _snapshot().message_upper_bound},
        source_fingerprints={f"private-source-{index}": {
            "message_id": index, "sent_at": 1000 + index, "revision_no": 1,
            "content_sha256": stable_sha256(f"synthetic source {index}"),
            "umo": _snapshot().umo, "is_deleted": False, "role": "BOT" if index == 13 else "USER",
            "evidence_roles": ["SUPPORT"],
        } for index in range(1, 14)},
    )


class StoredDerivationTests(unittest.TestCase):
    def test_stored_summary_survives_zero_raw_budget_through_reader_certificate_and_answer(self):
        descriptor = _summary_descriptor()
        key = descriptor["derivation_id"]
        request = build_l2_reader_prompt(
            query="纸鹤计划进展如何", evidence_packet={"sources": [], "stored_derivations": [descriptor]},
            snapshot=_snapshot(), allowed_source_keys=(), allowed_participant_keys=(),
            pack_read_complete=True, allow_l3=False,
        )
        delivered = json.loads(request.user_prompt)["evidence_packet"]["stored_derivations"][0]
        self.assertEqual(delivered["text"], descriptor["text"])
        self.assertEqual(delivered["source_count"], 13)
        self.assertNotIn("private-source-", request.user_prompt)
        self.assertNotIn("source_fingerprints", request.user_prompt)
        self.assertNotIn(descriptor["content_revision"], request.user_prompt)
        response = {
            "status": "PARTIAL", "referents": [{
                "id": "activity", "reference": "纸鹤活动", "referent_type": "TOPIC",
                "participant_key": "", "candidate_participant_keys": [], "reference_mode": "EVIDENCE_REF",
                "source_keys": [], "valid_at": None, "derivation_ids": [key],
            }],
            "atoms": [{"id": "preparation", "statement": "先整理材料；场地尚待确认。",
                       "subject_referent_id": "activity", "reasoning_kind": "EVIDENCE_SUMMARY",
                       "stance": "SUPPORTED", "source_keys": [], "derivation_ids": [key],
                       "importance": "REQUIRED", "confidence": 0.8}],
            "must_not_upgrade": [], "conflicts": [],
            "unresolved": [{"statement": "场地尚未确认。", "source_keys": [], "atom_ids": ["preparation"]}],
            "open_obligations": [], "stop_reason": "FRONTIER_EXHAUSTED",
        }
        certificate = parse_l2_reader_response(response, request)
        self.assertEqual(certificate.atoms[0].attribution, "OBSERVER_SUMMARY")
        self.assertEqual(certificate.atoms[0].evidence_roles, ("BOT", "USER"))
        self.assertEqual(certificate.atoms[0].speaker_participant_key, "")
        self.assertEqual(certificate.atoms[0].source_keys, ())
        self.assertEqual(len(certificate.derivations[0].dependency_keys), 13)
        cached = parse_evidence_certificate(certificate.as_dict(), expected_snapshot=_snapshot(),
            expected_packet_sha256=request.packet_sha256, allowed_source_keys=(),
            allowed_participant_keys=(), allowed_derivations=request.allowed_derivations, pack_read_complete=True)
        self.assertEqual(cached, certificate)
        brief = compile_answer_context_packet(cached)
        validate_answer_context_packet(brief, cached)
        fact = brief.as_dict()["facts"][0]
        self.assertEqual(fact["basis"], "stored_derivation")
        self.assertEqual(fact["evidence_roles"], ["BOT", "USER"])
        self.assertEqual(brief.included_derivation_ids, (key,))
        self.assertNotIn(key, brief.text)
        self.assertNotIn("private-source-", brief.text)
        self.assertNotIn("source_fingerprints", brief.text)
        inference = copy.deepcopy(response)
        inference["atoms"][0]["reasoning_kind"] = "DERIVED_INFERENCE"
        self.assertEqual(parse_l2_reader_response(inference, request).atoms[0].attribution, "DERIVED_INTERPRETATION")
        for changes, error in (
            ({"reasoning_kind": "EVIDENCE_STATEMENT"}, "host-bound speaker"),
            ({"derivation_ids": ["stored:" + "f" * 64]}, "host allowlist"),
        ):
            changed = copy.deepcopy(response)
            changed["atoms"][0].update(changes)
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, error):
                parse_l2_reader_response(changed, request)
        person = copy.deepcopy(response)
        person["referents"][0].update(referent_type="PARTICIPANT", reference_mode="UNBOUND")
        with self.assertRaisesRegex(ValueError, "cannot establish participant identity"):
            parse_l2_reader_response(person, request)

    def test_descriptor_cannot_turn_deleted_or_future_sources_into_valid_summary(self):
        for field, value in (("is_deleted", True), ("sent_at", 2000), ("message_id", 100)):
            descriptor = _summary_descriptor()
            descriptor["source_fingerprints"]["private-source-1"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_stored_derivation(descriptor, snapshot=_snapshot())
