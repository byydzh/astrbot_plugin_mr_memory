from __future__ import annotations

import json
import sqlite3
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from mr_memory.brief import EvidenceBrief, EvidenceClaim, EvidenceQualification
from mr_memory.models import NormalizedMessage
from mr_memory.storage import MemoryStorage, SCHEMA_VERSION
from mr_memory.usage import TokenUsageRecord
from scripts.masked_ab_experiment import (
    PilotBudget,
    PilotBudgetExceeded,
    _assert_no_nonempty_wal,
    _assert_usage_resumable,
    _collect_terminal_pilot_results,
    _evidence_keys,
    _execute_pilot_tool,
    _file_sha256,
    _pilot_completion,
    _prepare_pilot_base,
    _provider_fingerprint,
    _readonly_sqlite_backup,
    _run_pilot_b16,
    _run_pilot_full_mr,
    _score_pilot_gold,
    _usage_ledger_audit,
    _usage_totals,
    _validate_base_provenance,
    _validate_gold_sources,
    _validate_resume_migrated_database,
)


def _completion(*, content: str = "", tool_calls: list[object] | None = None):
    message = SimpleNamespace(
        content=content,
        reasoning_content="",
        tool_calls=tool_calls or [],
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class MaskedPilotTests(unittest.TestCase):
    umo = "shadow:GroupMessage:group-a"

    def test_source_is_backed_up_read_only_and_only_clone_is_migrated(self) -> None:
        test_root = Path.cwd() / ".dev" / "test-tmp"
        test_root.mkdir(parents=True, exist_ok=True)
        prefix = uuid.uuid4().hex
        paths = [
            test_root / f"{prefix}-{name}.db"
            for name in ("source", "base-v15", "arm-a", "arm-b")
        ]
        source, migrated, arm_a_path, arm_b_path = paths
        try:
            storage = MemoryStorage(source)
            storage.bind_scope(umo=self.umo, platform_id="shadow", group_id="group-a")
            storage.upsert_message(
                NormalizedMessage(
                    platform="aiocqhttp",
                    platform_id="shadow",
                    umo=self.umo,
                    group_id="group-a",
                    message_id="1",
                    sender_id="user-a",
                    sender_name="甲",
                    sent_at=100,
                    plain_text="只应存在于遮罩快照",
                )
            )
            storage.close()
            source_hash = _file_sha256(source)
            audit = _prepare_pilot_base(
                source,
                migrated,
                umo=self.umo,
                cutoff_at=200,
            )
            self.assertEqual(_file_sha256(source), source_hash)
            self.assertEqual(audit["source_sha256"], source_hash)
            self.assertEqual(audit["cutoff_audit"]["messages"], 1)
            connection = sqlite3.connect(migrated)
            try:
                version = connection.execute(
                    "SELECT value FROM schema_meta WHERE key='schema_version'"
                ).fetchone()[0]
            finally:
                connection.close()
            self.assertEqual(int(version), SCHEMA_VERSION)

            arm_a = _readonly_sqlite_backup(migrated, arm_a_path)
            arm_b = _readonly_sqlite_backup(migrated, arm_b_path)
            connection = sqlite3.connect(arm_a)
            try:
                connection.execute("CREATE TABLE pilot_marker(value TEXT)")
                connection.commit()
            finally:
                connection.close()
            connection = sqlite3.connect(arm_b)
            try:
                marker = connection.execute(
                    "SELECT name FROM sqlite_master WHERE name='pilot_marker'"
                ).fetchone()
            finally:
                connection.close()
            self.assertIsNone(marker)
        finally:
            for path in paths:
                for suffix in ("", "-wal", "-shm"):
                    Path(f"{path}{suffix}").unlink(missing_ok=True)

    def test_read_only_tool_forces_cutoff(self) -> None:
        storage = Mock()
        storage.query_event_context.return_value = [
            {"source_key": "source-1", "sent_at": 100, "plain_text": "证据"}
        ]
        result = _execute_pilot_tool(
            storage,
            umo=self.umo,
            cutoff_at=200,
            name="mr_query_event_context",
            arguments={"event_id": 7, "limit": 30},
        )
        self.assertEqual(result[0]["source_key"], "source-1")
        storage.query_event_context.assert_called_once_with(
            umo=self.umo,
            event_id=7,
            limit=30,
            before_sent_at=200,
        )

    def test_b16_is_exactly_one_provider_call_and_does_not_deepen(self) -> None:
        packet = {
            "candidates": {
                "feedback_hypotheses": [],
                "associations": [],
            },
            "expanded_episodes": [{"source_key": "source-1"}],
        }
        plan = {
            "decision": "escalate",
            "memory_brief": {"claims": [], "conflicts": [], "unresolved": []},
            "activate_hypotheses": [],
            "activate_edges": [],
            "escalation_question": "需要跨事件验证同一账号。",
        }
        with patch(
            "scripts.masked_ab_experiment._pilot_completion",
            return_value=_completion(content=json.dumps(plan, ensure_ascii=False)),
        ) as mocked:
            result = _run_pilot_b16(
                call={"query": "测试", "umo": self.umo, "cutoff_at": 200},
                packet=packet,
                client=object(),
                provider_id="deepseek/test",
                model="deepseek-v4-flash",
                provider_extra_body={},
                max_output_tokens=384000,
                thinking_mode="enabled",
                deadline_seconds=90,
                ledger_path=Path("unused.jsonl"),
                budget=PilotBudget(max_calls=30, soft_token_limit=600000),
                run_id="pilot-b16",
                repetition=1,
            )
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(result["model_calls"], 1)
        self.assertEqual(result["decision"], "escalate")
        self.assertEqual(result["brief"], None)

    def test_failed_provider_attempt_consumes_hard_call_budget(self) -> None:
        ledger = (
            Path.cwd() / ".dev" / "test-tmp" / f"pilot-usage-{uuid.uuid4().hex}.jsonl"
        )
        budget = PilotBudget(max_calls=1, soft_token_limit=0)
        try:
            with patch(
                "scripts.masked_ab_experiment._chat_completion",
                side_effect=TimeoutError("provider timed out"),
            ):
                with self.assertRaises(TimeoutError):
                    _pilot_completion(
                        client=object(),
                        model="deepseek-v4-flash",
                        provider_id="deepseek/test",
                        messages=[{"role": "user", "content": "test"}],
                        provider_extra_body={},
                        tools=None,
                        max_output_tokens=384000,
                        thinking_mode="enabled",
                        json_object=True,
                        ledger_path=ledger,
                        budget=budget,
                        run_id="pilot-failed",
                        arm="b16",
                        repetition=1,
                        phase="one-pass",
                        call_index=0,
                    )
            self.assertEqual(_usage_totals(ledger), (1, 0))
            self.assertEqual(budget.calls, 1)
            with self.assertRaisesRegex(PilotBudgetExceeded, "hard limit"):
                budget.before_call()
            events = [
                json.loads(line)
                for line in ledger.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [item["event"] for item in events], ["attempted", "failed"]
            )
            self.assertEqual(events[0]["options_sha256"], events[1]["options_sha256"])
            self.assertEqual(events[0]["payload_sha256"], events[1]["payload_sha256"])
        finally:
            ledger.unlink(missing_ok=True)

    def test_full_mr_uses_tool_then_returns_only_visited_sources(self) -> None:
        storage = Mock()
        storage.query_event_context.return_value = [
            {
                "source_key": "source-1",
                "sent_at": 100,
                "sender_name": "甲",
                "plain_text": "类魂玩吐了",
            }
        ]
        tool_call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(
                name="mr_query_event_context",
                arguments=json.dumps({"event_id": 7, "limit": 20}),
            ),
        )
        final = {
            "claims": [
                {
                    "statement": "甲说类魂玩吐了。",
                    "source_keys": ["source-1"],
                    "confidence": 0.9,
                }
            ],
            "conflicts": [],
            "unresolved": [],
        }
        with patch(
            "scripts.masked_ab_experiment._pilot_completion",
            side_effect=[
                _completion(tool_calls=[tool_call]),
                _completion(content=json.dumps(final, ensure_ascii=False)),
            ],
        ) as mocked:
            result = _run_pilot_full_mr(
                storage=storage,
                call={"query": "谁说类魂玩吐了", "umo": self.umo, "cutoff_at": 200},
                candidates={"episodes": [{"id": 7}]},
                client=object(),
                provider_id="deepseek/test",
                model="deepseek-v4-flash",
                provider_extra_body={},
                max_output_tokens=384000,
                thinking_mode="enabled",
                max_steps=8,
                deadline_seconds=90,
                ledger_path=Path("unused.jsonl"),
                budget=PilotBudget(max_calls=30, soft_token_limit=600000),
                run_id="pilot-full",
                repetition=1,
            )
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(result["model_calls"], 2)
        self.assertEqual(result["visited_source_keys"], ["source-1"])
        self.assertEqual(result["brief"]["claims"][0]["source_keys"], ["source-1"])
        storage.query_event_context.assert_called_once_with(
            umo=self.umo,
            event_id=7,
            limit=20,
            before_sent_at=200,
        )

    def test_gold_scoring_distinguishes_required_groups_and_calibration_terms(
        self,
    ) -> None:
        brief = EvidenceBrief(
            claims=(EvidenceClaim("守夜草说类魂玩吐了。", ("prior",), 0.9),),
            conflicts=(),
            unresolved=(
                EvidenceQualification(
                    "是否属于抢首发仍需保留措辞不确定性。", ("launch",)
                ),
            ),
        )
        score = _score_pilot_gold(
            brief=brief,
            visited_source_keys={"prior", "buy", "launch"},
            gold={
                "identity": {"expected_names": ["守夜草"]},
                "evidence_groups": {
                    "prior_dislike": {
                        "required_any": ["prior"],
                        "support": [],
                    },
                    "cross_episode": {
                        "required_any": ["buy"],
                        "support": ["launch"],
                    },
                },
                "forbidden_terms": ["雾影猎人"],
                "required_semantics": ["同一账号跨情节发生语义反转"],
                "required_uncertainty": ["抢首发不是逐字证据"],
                "forbidden_conclusions": ["完全没有证据"],
            },
        )
        self.assertEqual(score["required_group_recall"], 0.5)
        self.assertTrue(score["identity_text_match"])
        self.assertEqual(score["forbidden_term_hits"], [])
        self.assertEqual(
            score["evidence_groups"]["cross_episode"]["visited_required_source_keys"],
            ["buy"],
        )
        self.assertEqual(
            score["semantic_judgment_status"],
            "PENDING_BLIND_HUMAN_REVIEW",
        )
        self.assertIsNone(score["semantic_score"])
        self.assertEqual(
            score["semantic_rubric"]["required_uncertainty"],
            ["抢首发不是逐字证据"],
        )

    def test_provenance_binds_all_frozen_inputs_and_construction(self) -> None:
        provenance = {
            "sha256": "a" * 64,
            "messages_sha256": "b" * 64,
            "candidates_sha256": "c" * 64,
            "researcher_attested_pre_cutoff_derived_state": True,
            "message_count": 2,
            "maximum_message_sent_at": 100,
            "construction_protocol": "exact masked replay before the target cutoff",
        }
        value = _validate_base_provenance(
            gold={"base_db_provenance": provenance},
            source_sha256="a" * 64,
            messages_sha256="b" * 64,
            candidates_sha256="c" * 64,
            cutoff_audit={"messages": 2, "maximum_sent_at": 100},
        )
        self.assertEqual(value["messages_sha256"], "b" * 64)
        self.assertEqual(value["candidates_sha256"], "c" * 64)
        self.assertIn("masked replay", value["construction_protocol"])

        mutations = {
            "sha256": "d" * 64,
            "messages_sha256": "d" * 64,
            "candidates_sha256": "d" * 64,
            "researcher_attested_pre_cutoff_derived_state": False,
            "message_count": 3,
            "maximum_message_sent_at": 101,
            "construction_protocol": "",
        }
        for field, replacement in mutations.items():
            with self.subTest(field=field):
                changed = dict(provenance)
                changed[field] = replacement
                with self.assertRaises(ValueError):
                    _validate_base_provenance(
                        gold={"base_db_provenance": changed},
                        source_sha256="a" * 64,
                        messages_sha256="b" * 64,
                        candidates_sha256="c" * 64,
                        cutoff_audit={"messages": 2, "maximum_sent_at": 100},
                    )

    def test_resume_rejects_changed_migrated_database(self) -> None:
        previous = {"base": {"migrated_sha256": "a" * 64}}
        _validate_resume_migrated_database(previous, previous)
        with self.assertRaisesRegex(ValueError, "migrated"):
            _validate_resume_migrated_database(
                previous,
                {"base": {"migrated_sha256": "b" * 64}},
            )

    def test_nonempty_wal_is_not_accepted_as_a_frozen_hash(self) -> None:
        root = Path.cwd() / ".dev" / "test-tmp"
        root.mkdir(parents=True, exist_ok=True)
        database = root / f"wal-{uuid.uuid4().hex}.db"
        wal = Path(f"{database}-wal")
        try:
            database.write_bytes(b"database")
            wal.write_bytes(b"pending")
            with self.assertRaisesRegex(ValueError, "non-empty WAL"):
                _assert_no_nonempty_wal(database)
        finally:
            wal.unlink(missing_ok=True)
            database.unlink(missing_ok=True)

    def test_plural_source_keys_and_strict_media_signature(self) -> None:
        value = {
            "source_key": "a",
            "source_keys": ["b", ""],
            "sample_source_keys": ["c", "b"],
            "nested": [{"support_source_keys": ["d"]}],
        }
        self.assertEqual(_evidence_keys(value), ["a", "b", "c", "d"])

        class StrictMediaStorage:
            def __init__(self) -> None:
                self.arguments = None

            def query_media_patterns(
                self,
                *,
                umo,
                fingerprints=(),
                media_type="image",
                min_observations=2,
                limit=12,
            ):
                self.arguments = (
                    umo,
                    fingerprints,
                    media_type,
                    min_observations,
                    limit,
                )
                return [{"sample_source_keys": ["m1", "m2"]}]

        storage = StrictMediaStorage()
        result = _execute_pilot_tool(
            storage,
            umo=self.umo,
            cutoff_at=200,
            name="mr_query_media_patterns",
            arguments={"limit": 4},
        )
        self.assertEqual(_evidence_keys(result), ["m1", "m2"])
        self.assertEqual(
            storage.arguments,
            (self.umo, (), "image", 2, 4),
        )

    def test_usage_audit_blocks_unknown_provider_billing(self) -> None:
        ledger = Path.cwd() / ".dev" / "test-tmp" / f"audit-{uuid.uuid4().hex}.jsonl"
        ledger.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {"request_id": "r1", "event": "attempted"},
            {"request_id": "r1", "event": "completed", "total": 17},
            {"request_id": "r2", "event": "attempted"},
            {"request_id": "r2", "event": "failed"},
            {"request_id": "r3", "event": "attempted"},
        ]
        try:
            ledger.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            audit = _usage_ledger_audit(ledger)
            self.assertEqual(audit["attempted_calls"], 3)
            self.assertEqual(audit["completed_calls"], 1)
            self.assertEqual(audit["failed_calls"], 1)
            self.assertEqual(audit["unknown_usage_calls"], 2)
            self.assertEqual(audit["unknown_request_ids"], ["r2", "r3"])
            self.assertEqual(audit["provider_tokens_measured_lower_bound"], 17)
            self.assertFalse(audit["usage_complete"])
            with self.assertRaisesRegex(RuntimeError, "unknown provider billing"):
                _assert_usage_resumable(audit)
        finally:
            ledger.unlink(missing_ok=True)

    def test_completed_response_without_usage_remains_unknown(self) -> None:
        ledger = (
            Path.cwd() / ".dev" / "test-tmp" / f"missing-usage-{uuid.uuid4().hex}.jsonl"
        )
        ledger.parent.mkdir(parents=True, exist_ok=True)
        completion = _completion(content="{}")
        completion.usage = None
        try:
            with patch(
                "scripts.masked_ab_experiment._chat_completion",
                return_value=(completion, TokenUsageRecord(), 1.0),
            ):
                with self.assertRaisesRegex(RuntimeError, "omitted usage"):
                    _pilot_completion(
                        client=object(),
                        model="deepseek-v4-flash",
                        provider_id="deepseek/test",
                        messages=[{"role": "user", "content": "test"}],
                        provider_extra_body={},
                        tools=None,
                        max_output_tokens=384000,
                        thinking_mode="enabled",
                        json_object=True,
                        ledger_path=ledger,
                        budget=PilotBudget(max_calls=1, soft_token_limit=0),
                        run_id="pilot-missing-usage",
                        arm="b16",
                        repetition=1,
                        phase="one-pass",
                        call_index=0,
                    )
            audit = _usage_ledger_audit(ledger)
            self.assertEqual(audit["completed_calls"], 1)
            self.assertEqual(audit["unknown_usage_calls"], 1)
            self.assertEqual(audit["provider_tokens_measured_lower_bound"], 0)
            self.assertFalse(audit["usage_complete"])
        finally:
            ledger.unlink(missing_ok=True)

    def test_resume_summary_collects_results_from_all_arms(self) -> None:
        root = Path.cwd() / ".dev" / "test-tmp" / f"summary-{uuid.uuid4().hex}"
        cache = root / "runs" / "cache" / "rep-01" / "result.json"
        b16 = root / "runs" / "b16" / "rep-01" / "result.json"
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            b16.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(
                json.dumps(
                    {
                        "run_id": "pilot-cache-r01",
                        "arm": "cache",
                        "repetition": 1,
                        "status": "COMPLETED",
                    }
                ),
                encoding="utf-8",
            )
            b16.write_text(
                json.dumps(
                    {
                        "run_id": "pilot-b16-r01",
                        "arm": "b16",
                        "repetition": 1,
                        "status": "FAILED",
                    }
                ),
                encoding="utf-8",
            )
            values = _collect_terminal_pilot_results(root)
            self.assertEqual({item["arm"] for item in values}, {"cache", "b16"})
            self.assertEqual(len(values), 2)
        finally:
            for path in (cache, b16):
                path.unlink(missing_ok=True)

    def test_provider_fingerprint_never_hashes_auth_header_value(self) -> None:
        root = Path.cwd() / ".dev" / "test-tmp"
        root.mkdir(parents=True, exist_ok=True)
        first = root / f"provider-a-{uuid.uuid4().hex}.json"
        second = root / f"provider-b-{uuid.uuid4().hex}.json"

        def config(token: str) -> dict:
            return {
                "provider": [
                    {
                        "id": "deepseek/test",
                        "provider_source_id": "source-test",
                        "model": "model-test",
                    }
                ],
                "provider_sources": [
                    {
                        "id": "source-test",
                        "api_base": "https://example.invalid/v1",
                        "timeout": 120,
                        "custom_headers": {
                            "Authorization": token,
                            "X-Protocol": "stable",
                        },
                    }
                ],
            }

        try:
            first.write_text(json.dumps(config("Bearer secret-a")), encoding="utf-8")
            second.write_text(json.dumps(config("Bearer secret-b")), encoding="utf-8")
            self.assertEqual(
                _provider_fingerprint(first, "deepseek/test"),
                _provider_fingerprint(second, "deepseek/test"),
            )
        finally:
            first.unlink(missing_ok=True)
            second.unlink(missing_ok=True)

    def test_gold_required_evidence_is_bound_to_expected_sender(self) -> None:
        root = Path.cwd() / ".dev" / "test-tmp"
        root.mkdir(parents=True, exist_ok=True)
        database = root / f"gold-{uuid.uuid4().hex}.db"
        connection = sqlite3.connect(database)
        try:
            connection.execute(
                "CREATE TABLE messages(umo TEXT, source_key TEXT, sent_at INTEGER, "
                "sender_id TEXT)"
            )
            connection.executemany(
                "INSERT INTO messages VALUES(?,?,?,?)",
                [
                    (self.umo, "required", 100, "actor"),
                    (self.umo, "support", 110, "observer"),
                ],
            )
            connection.commit()
        finally:
            connection.close()
        gold = {
            "identity": {"expected_sender_ids": ["actor"]},
            "evidence_groups": {
                "cross_episode": {
                    "required_any": ["required"],
                    "support": ["support"],
                }
            },
        }
        try:
            audit = _validate_gold_sources(
                database,
                umo=self.umo,
                cutoff_at=200,
                gold=gold,
            )
            self.assertTrue(audit["required_sender_identity_match"])
            changed = json.loads(json.dumps(gold))
            changed["identity"]["expected_sender_ids"] = ["other"]
            with self.assertRaisesRegex(ValueError, "target account"):
                _validate_gold_sources(
                    database,
                    umo=self.umo,
                    cutoff_at=200,
                    gold=changed,
                )
        finally:
            database.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
