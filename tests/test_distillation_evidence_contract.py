from __future__ import annotations

import json
import unittest
from dataclasses import replace

from mr_memory.distillation import (
    build_distillation_prompt,
    build_distillation_prompt_aliases,
    parse_distillation_response,
)
from mr_memory.models import StoredMessage


class DistillationEvidenceContractTests(unittest.TestCase):
    @staticmethod
    def message(index: int = 1) -> StoredMessage:
        return StoredMessage(
            id=index,
            source_key=f"synthetic-source-{index}",
            platform="aiocqhttp",
            platform_id="synthetic",
            umo="synthetic:GroupMessage:evidence-contract",
            group_id="evidence-contract",
            message_id=f"synthetic-message-{index}",
            sender_id=f"synthetic-account-{index}",
            sender_name=f"SyntheticParticipant{index}",
            sent_at=100 + index,
            plain_text="  SyntheticTopic may describe a shared preference.\nThis remains uncertain.  ",
            content=[],
            role="USER",
            sender_participant_key=f"synthetic-participant-{index}",
        )

    @staticmethod
    def response(source_key: str, subject: dict[str, str]) -> dict[str, object]:
        return {
            "episodes": [],
            "claims": [{
                "subject": subject,
                "claim_type": "PREFERENCE",
                "predicate": "synthetic_preference",
                "object": "A shared preference is possible.",
                "epistemic_status": "UNCERTAIN",
                "operation": "ASSERT",
                "confidence": 0.6,
                "evidence": [{"source_key": source_key, "role": "SUPPORT"}],
            }],
            "topics": [],
            "associations": [],
            "ignored_source_keys": [],
        }

    def test_unresolved_subject_remains_unbound_with_exact_whole_message_evidence(self) -> None:
        message = self.message()
        response = self.response(
            message.source_key,
            {"participant_key": "", "unresolved_text": "SyntheticTopic"},
        )
        batch = parse_distillation_response(json.dumps(response), [message])
        claim = batch.semantic_memories[0]
        self.assertEqual(claim.subject_participant_key, "")
        self.assertEqual(claim.subject_text, "SyntheticTopic")
        self.assertEqual(claim.epistemic_status, "UNCERTAIN")
        self.assertEqual(claim.evidence[0].span, message.plain_text)

        response["claims"][0]["evidence"][0]["span"] = "invented supporting quote"
        with self.assertRaisesRegex(ValueError, "not an exact source substring"):
            parse_distillation_response(json.dumps(response), [message])

    def test_subject_binding_uses_authoritative_message_relations_without_duplicate_context(self) -> None:
        original = self.message()
        replied_to = self.message(2)
        cases = (
            (original, original.sender_participant_key, [original]),
            (
                replace(original, mentions=({"participant_key": "synthetic-mentioned-account"},)),
                "synthetic-mentioned-account",
                [],
            ),
            (
                replace(original, reply_to_source_key=replied_to.source_key),
                replied_to.sender_participant_key,
                [replied_to],
            ),
        )
        for message, subject_key, context_messages in cases:
            with self.subTest(subject=subject_key):
                messages = [message, *(item for item in context_messages if item.id != message.id)]
                response = self.response(message.source_key, {"participant_key": subject_key})
                batch = parse_distillation_response(
                    json.dumps(response), messages, target_source_keys=[message.source_key]
                )
                self.assertEqual(batch.semantic_memories[0].subject_participant_key, subject_key)

    def test_known_but_unanchored_subject_is_still_rejected(self) -> None:
        message = self.message()
        unrelated = self.message(2)
        response = self.response(
            message.source_key, {"participant_key": unrelated.sender_participant_key}
        )
        with self.assertRaisesRegex(ValueError, "lacks deterministic"):
            parse_distillation_response(
                json.dumps(response),
                [message, unrelated],
                target_source_keys=[message.source_key],
            )

    def test_mention_alias_round_trip_preserves_host_binding_and_full_source(self) -> None:
        message = replace(
            self.message(), mentions=({"participant_key": "synthetic-mentioned-account"},)
        )
        aliases = build_distillation_prompt_aliases([message])
        prompt = build_distillation_prompt([message], aliases=aliases)
        self.assertNotIn("synthetic-mentioned-account", prompt)
        response = self.response(
            aliases.source_to_alias[message.source_key],
            {"participant_key": aliases.participant_to_alias["synthetic-mentioned-account"]},
        )
        batch = parse_distillation_response(json.dumps(response), [message], aliases=aliases)
        self.assertEqual(
            batch.semantic_memories[0].subject_participant_key, "synthetic-mentioned-account"
        )
        self.assertEqual(batch.semantic_memories[0].evidence[0].span, message.plain_text)


if __name__ == "__main__":
    unittest.main()
