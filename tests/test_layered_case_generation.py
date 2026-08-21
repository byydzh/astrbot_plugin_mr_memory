from __future__ import annotations

import copy
import hashlib
import json
import shutil
import unittest
import uuid
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mr_memory.snapshot import RequestSnapshot
from mr_memory.snapshot import stable_sha256
from mr_memory.surface import compile_surface_packet, validate_surface_packet
from mr_memory.usage import TokenUsageRecord
from mr_memory.orchestrator import ECCR_TOOL_ACTION_CATALOG, EccrProtocolError
from scripts.layered_case_generation import (
    _READ_TOOLS,
    _completed_case_import,
    _execute_pilot_tool,
    _l3_certificate_packet_sha256,
    _participant_keys,
    _prepare_provider_stage_import,
    _provider_request_hashes,
    _run_l2,
    _run_l3,
    _sanitize_provider_input,
    _snapshot,
    _source_keys,
    _surface_messages,
    build_parser,
    generate,
)
from tests.test_certificate_v2 import _raw_certificate
from tests import test_orchestrator


@contextmanager
def _workspace_tempdir():
    parent = Path.cwd() / ".test-artifacts"
    parent.mkdir(exist_ok=True)
    path = parent / uuid.uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _completion(value: object) -> SimpleNamespace:
    content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, reasoning_content="")
            )
        ]
    )


def _case() -> dict[str, object]:
    return {
        "case_id": "layered-runner-test",
        "umo": "scope:test",
        "cutoff_at": 2_000,
        "query": "好女孩是什么意思",
        "authorized_participant_keys": ["p1", "p2"],
    }


def _packet() -> dict[str, object]:
    return {
        "diagnostic_type": "test-fixed-packet",
        "messages": [
            {
                "source_key": "s1",
                "participant_key": "p1",
                "sent_at": 1_900,
                "plain_text": "第一条证据",
            },
            {
                "source_key": "s2",
                "participant_key": "p1",
                "sent_at": 1_950,
                "plain_text": "第二条反例",
            },
        ],
    }


def _certificate_value(snapshot, packet) -> dict[str, object]:
    raw = _raw_certificate()
    raw["scope_snapshot"] = snapshot.as_dict()
    raw["data_revision"] = snapshot.data_revision.as_dict()
    raw["inference_revision"] = snapshot.inference_revision.as_dict()
    raw["packet_sha256"] = stable_sha256(packet)
    raw["atoms"] = [raw["atoms"][0]]
    raw["must_include"] = ["a1"]
    raw["must_not_upgrade"][0]["atom_ids"] = ["a1"]
    return raw


class LayeredGenerationBoundaryTests(unittest.TestCase):
    def test_real_v7_completed_case_with_legacy_stop_reason_is_rejected(
        self,
    ) -> None:
        source_suite = (
            Path.cwd()
            / ".dev"
            / "provider_results"
            / "three-case-layered-v7-partial"
        )
        source_result = source_suite / "cases" / "call-726" / "result.private.json"
        dev_root = Path.cwd().parent / ".dev"
        if not source_result.is_file() or not dev_root.is_dir():
            self.skipTest("private v7 completed-case fixture is not present")
        manifest = json.loads(
            (source_result.parent / "manifest.json").read_text("utf-8")
        )
        providers = manifest["provider"]
        database_hash = manifest["inputs"]["database_sha256"]
        source_database = source_suite / "prepared-input" / "call-726" / "scope.db"
        self.assertEqual(
            hashlib.sha256(source_database.read_bytes()).hexdigest(), database_hash
        )
        source_attempt_sha256 = hashlib.sha256(
            (source_suite / "run-plan.json").read_bytes()
        ).hexdigest()
        args = Namespace(
            source_result=str(source_result),
            source_attempt_id=f"{source_suite.name}@{source_attempt_sha256[:16]}",
            source_attempt_manifest=str(source_suite / "run-plan.json"),
            source_attempt_manifest_sha256=source_attempt_sha256,
            target_dir=str(source_suite.parent / "must-not-write"),
            case=str(dev_root / "experiments" / "masked-call-726" / "call_r4.json"),
            evidence_packet=str(
                source_suite / "prepared-input" / "call-726" / "evidence_packet.json"
            ),
            surface_case_template=str(
                dev_root
                / "experiments"
                / "masked-call-726"
                / "surface-ab-v1-input"
                / "case.json"
            ),
            database=str(source_database),
            config=str(source_suite / "run-plan.json"),
            subconscious_provider_id="deepseek/deepseek-v4-flash",
            main_provider_id="openai/gemini-3.5-flash",
            route="l3",
            max_provider_calls=4,
            max_output_tokens=384_000,
            surface_max_output_tokens=65_536,
            deadline_seconds=600.0,
            l3_max_model_calls=3,
            l3_max_retrieval_rounds=2,
            surface_max_chars=12_000,
            commit=False,
        )

        def provider_config(_path, provider_id):
            binding = providers[
                "memory" if provider_id.startswith("deepseek") else "surface"
            ]
            return object(), binding["model"], {}

        def provider_fingerprint(_path, provider_id):
            binding = providers[
                "memory" if provider_id.startswith("deepseek") else "surface"
            ]
            return {
                key: value
                for key, value in binding.items()
                if key not in {"provider_id", "model"}
            }

        with (
            patch(
                "scripts.layered_case_generation._provider_config",
                side_effect=provider_config,
            ),
            patch(
                "scripts.layered_case_generation._provider_fingerprint",
                side_effect=provider_fingerprint,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "stop_reason is unsupported"):
                _completed_case_import(args)

    def test_generation_requires_explicit_authorization_before_file_access(self) -> None:
        args = Namespace(authorize_provider_calls=False)
        with self.assertRaisesRegex(PermissionError, "authorize-provider-calls"):
            generate(args)

    def test_billable_parser_has_no_gold_input_channel(self) -> None:
        parser = build_parser()
        generation = next(
            action.choices["generate"]
            for action in parser._actions
            if getattr(action, "choices", None) and "generate" in action.choices
        )
        destinations = {action.dest for action in generation._actions}
        self.assertFalse(any("gold" in destination.casefold() for destination in destinations))
        self.assertIn("validate_import_only", destinations)

    def test_l3_tool_names_match_production_and_pilot_adapter_strips_prefix(self) -> None:
        self.assertEqual(_READ_TOOLS, set(ECCR_TOOL_ACTION_CATALOG))

        class Storage:
            def query_event_context(self, **kwargs):
                return kwargs

        result = _execute_pilot_tool(
            Storage(),
            umo="scope:test",
            cutoff_at=2_000,
            name="mr_query_event_context",
            arguments={"event_id": 7, "limit": 12},
        )
        self.assertEqual(result["umo"], "scope:test")
        self.assertEqual(result["event_id"], 7)
        self.assertEqual(result["limit"], 12)
        self.assertEqual(result["before_sent_at"], 2_000)

    def test_post_selection_audit_flags_are_not_visible_to_provider(self) -> None:
        sanitized, removed = _sanitize_provider_input(
            {
                "fixture_provenance": {"retrieval_selection_used_gold": False},
                "retrieval": {"gold_loaded_after_selection": True, "backend": "bm25"},
            }
        )
        self.assertEqual(
            sanitized,
            {
                "fixture_provenance": {},
                "retrieval": {"backend": "bm25"},
            },
        )
        self.assertEqual(len(removed), 2)
        with self.assertRaisesRegex(ValueError, "forbidden post-run field"):
            _sanitize_provider_input({"gold_answer": "must never reach generation"})

    def test_end_to_end_mocked_provider_chain_is_resume_safe(self) -> None:
        with _workspace_tempdir() as root:
            case_path = root / "case.json"
            packet_path = root / "packet.json"
            config_path = root / "private-config.json"
            output_dir = root / "output"
            _write_json(case_path, _case())
            _write_json(packet_path, _packet())
            _write_json(config_path, {})
            provider_calls: list[str] = []
            provider_limits: dict[str, int] = {}

            def fake_pilot_completion(**kwargs):
                budget = kwargs["budget"]
                budget.reserve_call()
                usage = TokenUsageRecord(input_other=10, input_cached=1, output=4)
                budget.observe(usage)
                phase = str(kwargs["phase"])
                provider_calls.append(phase)
                provider_limits[phase] = int(kwargs["max_output_tokens"])
                if bool(kwargs["json_object"]):
                    payload = json.loads(kwargs["messages"][1]["content"])
                    snapshot = RequestSnapshot.from_value(payload["scope_snapshot"])
                    raw = _certificate_value(snapshot, payload["evidence_packet"])
                    completion = _completion(raw)
                else:
                    completion = _completion("这是主模型基于证据约束生成的完整回答。")
                request_id = (
                    f"{kwargs['run_id']}:{phase}:{int(kwargs['call_index'])}"
                )
                common = {
                    "request_id": request_id,
                    "run_id": kwargs["run_id"],
                    "arm": kwargs["arm"],
                    "phase": phase,
                    "call_index": int(kwargs["call_index"]),
                    "provider_id": kwargs["provider_id"],
                    "model": kwargs["model"],
                    "options_sha256": _provider_request_hashes(
                        model=kwargs["model"],
                        messages=kwargs["messages"],
                        provider_extra_body=kwargs["provider_extra_body"],
                        max_output_tokens=kwargs["max_output_tokens"],
                        json_object=kwargs["json_object"],
                    )[0],
                    "payload_sha256": _provider_request_hashes(
                        model=kwargs["model"],
                        messages=kwargs["messages"],
                        provider_extra_body=kwargs["provider_extra_body"],
                        max_output_tokens=kwargs["max_output_tokens"],
                        json_object=kwargs["json_object"],
                    )[1],
                }
                ledger_path = Path(kwargs["ledger_path"])
                ledger_path.parent.mkdir(parents=True, exist_ok=True)
                with ledger_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({**common, "event": "attempted"}) + "\n")
                    handle.write(
                        json.dumps(
                            {
                                **common,
                                "event": "completed",
                                "usage_present": True,
                                **usage.as_dict(),
                                "elapsed_ms": 12.5,
                            }
                        )
                        + "\n"
                    )
                return completion

            args = Namespace(
                authorize_provider_calls=True,
                case=str(case_path),
                evidence_packet=str(packet_path),
                surface_case_template=None,
                database=None,
                config=str(config_path),
                subconscious_provider_id="deepseek/deepseek-v4-flash",
                main_provider_id="openai/gemini-3.5-flash",
                route="l2",
                output_dir=str(output_dir),
                max_provider_calls=3,
                max_output_tokens=384_000,
                surface_max_output_tokens=65_536,
                deadline_seconds=30.0,
                l3_max_model_calls=3,
                l3_max_retrieval_rounds=0,
                surface_max_chars=12_000,
                import_memory_checkpoint=None,
                resume=True,
            )
            with (
                patch(
                    "scripts.layered_case_generation._provider_config",
                    side_effect=lambda _path, provider_id: (
                        object(),
                        provider_id.rsplit("/", 1)[-1],
                        {},
                    ),
                ),
                patch(
                    "scripts.layered_case_generation._provider_fingerprint",
                    return_value={"transport_sha256": "f" * 64},
                ),
                patch(
                    "scripts.layered_case_generation._pilot_completion",
                    side_effect=fake_pilot_completion,
                ),
            ):
                first = generate(args)
                second = generate(args)
            self.assertEqual(first["provider_calls"], 2)
            self.assertEqual(second["provider_calls"], 2)
            self.assertEqual(provider_calls, ["reader_initial", "surface_answer"])
            self.assertEqual(
                provider_limits,
                {"reader_initial": 384_000, "surface_answer": 65_536},
            )
            final = json.loads((output_dir / "result.private.json").read_text("utf-8"))
            self.assertEqual(final["status"], "COMPLETED")
            self.assertEqual(
                final["result"]["certificate"]["schema_version"],
                "evidence-certificate.v2",
            )
            surface = json.loads(
                (output_dir / "surface" / "private_results.json").read_text("utf-8")
            )
            self.assertIn("完整回答", surface["results"][0]["answer"])
            self.assertTrue(final["cost"]["ledger_audit"]["usage_complete"])
            self.assertEqual(
                final["limits"],
                {
                    "subconscious_max_output_tokens": 384_000,
                    "surface_max_output_tokens": 65_536,
                },
            )
            manifest = json.loads((output_dir / "manifest.json").read_text("utf-8"))
            prompt_audit = manifest["l2_initial_prompt_audit"]
            self.assertEqual(
                prompt_audit["ordered_source_keys"],
                sorted(prompt_audit["ordered_source_keys"]),
            )
            self.assertEqual(
                prompt_audit["ordered_participant_keys"],
                sorted(prompt_audit["ordered_participant_keys"]),
            )
            self.assertEqual(len(prompt_audit["payload_sha256"]), 64)
            stages = json.loads(
                (output_dir / "memory-stages.private.json").read_text("utf-8")
            )["stages"]
            self.assertEqual(stages[0]["prompt_audit"], prompt_audit)
            self.assertEqual(manifest["limits"], {
                "max_provider_calls": 3,
                "provider_calls_upper_bound": 3,
                "subconscious_max_output_tokens": 384_000,
                "surface_max_output_tokens": 65_536,
                "deadline_seconds": 30.0,
                "l3_max_model_calls": 3,
                "l3_max_retrieval_rounds": 0,
                "surface_max_chars": 12_000,
            })
            changed = copy.copy(args)
            changed.surface_max_output_tokens = 32_768
            with (
                patch(
                    "scripts.layered_case_generation._provider_config",
                    side_effect=lambda _path, provider_id: (
                        object(), provider_id.rsplit("/", 1)[-1], {},
                    ),
                ),
                patch(
                    "scripts.layered_case_generation._provider_fingerprint",
                    return_value={"transport_sha256": "f" * 64},
                ),
            ):
                with self.assertRaisesRegex(ValueError, "resume manifest mismatch: limits"):
                    generate(changed)

    def test_import_completed_memory_excludes_failed_surface_and_does_not_repeat_reader(
        self,
    ) -> None:
        with _workspace_tempdir() as root:
            case_path = root / "case.json"
            packet_path = root / "packet.json"
            config_path = root / "private-config.json"
            source_dir = root / "v5" / "case"
            target_dir = root / "v6" / "case"
            _write_json(case_path, _case())
            _write_json(packet_path, _packet())
            _write_json(config_path, {})
            phases: list[tuple[str, str]] = []

            def fake_pilot_completion(**kwargs):
                budget = kwargs["budget"]
                budget.reserve_call()
                phase = str(kwargs["phase"])
                ledger_path = Path(kwargs["ledger_path"])
                location = "source" if source_dir in ledger_path.parents else "target"
                phases.append((location, phase))
                request_id = f"{kwargs['run_id']}:{phase}:{int(kwargs['call_index'])}"
                common = {
                    "request_id": request_id,
                    "run_id": kwargs["run_id"],
                    "arm": kwargs["arm"],
                    "phase": phase,
                    "call_index": int(kwargs["call_index"]),
                    "provider_id": kwargs["provider_id"],
                    "model": kwargs["model"],
                    "options_sha256": _provider_request_hashes(
                        model=kwargs["model"],
                        messages=kwargs["messages"],
                        provider_extra_body=kwargs["provider_extra_body"],
                        max_output_tokens=kwargs["max_output_tokens"],
                        json_object=kwargs["json_object"],
                    )[0],
                    "payload_sha256": _provider_request_hashes(
                        model=kwargs["model"],
                        messages=kwargs["messages"],
                        provider_extra_body=kwargs["provider_extra_body"],
                        max_output_tokens=kwargs["max_output_tokens"],
                        json_object=kwargs["json_object"],
                    )[1],
                }
                ledger_path.parent.mkdir(parents=True, exist_ok=True)
                with ledger_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({**common, "event": "attempted"}) + "\n")
                    if location == "source" and phase == "surface_answer":
                        handle.write(
                            json.dumps({**common, "event": "failed", "elapsed_ms": 1.0})
                            + "\n"
                        )
                        raise RuntimeError("simulated surface HTTP 400")
                    usage = TokenUsageRecord(input_other=10, input_cached=1, output=4)
                    budget.observe(usage)
                    handle.write(
                        json.dumps(
                            {
                                **common,
                                "event": "completed",
                                "usage_present": True,
                                **usage.as_dict(),
                                "elapsed_ms": 12.5,
                            }
                        )
                        + "\n"
                    )
                if bool(kwargs["json_object"]):
                    payload = json.loads(kwargs["messages"][1]["content"])
                    snapshot = RequestSnapshot.from_value(payload["scope_snapshot"])
                    return _completion(
                        _certificate_value(snapshot, payload["evidence_packet"])
                    )
                return _completion("导入记忆后仅重新生成表层回答。")

            common_args = dict(
                authorize_provider_calls=True,
                case=str(case_path),
                evidence_packet=str(packet_path),
                surface_case_template=None,
                database=None,
                config=str(config_path),
                subconscious_provider_id="deepseek/deepseek-v4-flash",
                main_provider_id="openai/gemini-3.5-flash",
                route="l2",
                max_provider_calls=3,
                max_output_tokens=384_000,
                deadline_seconds=30.0,
                l3_max_model_calls=3,
                l3_max_retrieval_rounds=0,
                surface_max_chars=12_000,
                resume=True,
            )
            source_args = Namespace(
                **common_args,
                output_dir=str(source_dir),
                surface_max_output_tokens=384_000,
                import_memory_checkpoint=None,
            )
            target_args = Namespace(
                **common_args,
                output_dir=str(target_dir),
                surface_max_output_tokens=65_536,
                import_memory_checkpoint=str(source_dir / "memory.private.json"),
            )
            patches = (
                patch(
                    "scripts.layered_case_generation._provider_config",
                    side_effect=lambda _path, provider_id: (
                        object(), provider_id.rsplit("/", 1)[-1], {},
                    ),
                ),
                patch(
                    "scripts.layered_case_generation._provider_fingerprint",
                    return_value={"transport_sha256": "f" * 64},
                ),
                patch(
                    "scripts.layered_case_generation._pilot_completion",
                    side_effect=fake_pilot_completion,
                ),
            )
            with patches[0], patches[1], patches[2]:
                with self.assertRaisesRegex(RuntimeError, "simulated surface HTTP 400"):
                    generate(source_args)
                # v5 used one shared field. The import path treats it only as
                # the memory-side cap and never inherits it for the new surface.
                source_manifest_path = source_dir / "manifest.json"
                source_manifest = json.loads(source_manifest_path.read_text("utf-8"))
                source_manifest["limits"].pop("subconscious_max_output_tokens")
                source_manifest["limits"].pop("surface_max_output_tokens")
                source_manifest["limits"]["max_output_tokens"] = 384_000
                _write_json(source_manifest_path, source_manifest)

                preflight = copy.copy(target_args)
                preflight.output_dir = str(root / "v6-preflight-must-not-exist")
                preflight.validate_import_only = True
                preflight.authorize_provider_calls = False
                preflight_result = generate(preflight)
                self.assertEqual(preflight_result["status"], "IMPORT_PREFLIGHT_OK")
                self.assertEqual(preflight_result["provider_calls"], 0)
                self.assertFalse(Path(preflight.output_dir).exists())

                wrong_database = root / "different-scope.db"
                wrong_database.write_bytes(b"not-the-source-database")
                database_mismatch = copy.copy(target_args)
                database_mismatch.output_dir = str(root / "v6-db-mismatch" / "case")
                database_mismatch.database = str(wrong_database)
                with self.assertRaisesRegex(
                    ValueError, "input mismatch: database_sha256"
                ):
                    generate(database_mismatch)

                tampered_certificate_dir = root / "v5-tampered-certificate" / "case"
                shutil.copytree(source_dir, tampered_certificate_dir)
                tampered_certificate_path = (
                    tampered_certificate_dir / "memory.private.json"
                )
                tampered_certificate = json.loads(
                    tampered_certificate_path.read_text("utf-8")
                )
                tampered_certificate["certificate_sha256"] = "0" * 64
                _write_json(tampered_certificate_path, tampered_certificate)
                certificate_mismatch = copy.copy(target_args)
                certificate_mismatch.output_dir = str(
                    root / "v6-certificate-mismatch" / "case"
                )
                certificate_mismatch.import_memory_checkpoint = str(
                    tampered_certificate_path
                )
                with self.assertRaisesRegex(
                    RuntimeError, "certificate_sha256 mismatch"
                ):
                    generate(certificate_mismatch)

                tampered_packet_dir = root / "v5-tampered-packet" / "case"
                shutil.copytree(source_dir, tampered_packet_dir)
                tampered_packet_path = tampered_packet_dir / "memory.private.json"
                tampered_packet = json.loads(tampered_packet_path.read_text("utf-8"))
                tampered_packet["surface_packet_text"] += "\nTAMPERED"
                tampered_packet["surface_packet_sha256"] = hashlib.sha256(
                    tampered_packet["surface_packet_text"].encode("utf-8")
                ).hexdigest()
                _write_json(tampered_packet_path, tampered_packet)
                packet_mismatch = copy.copy(target_args)
                packet_mismatch.output_dir = str(root / "v6-packet-mismatch" / "case")
                packet_mismatch.import_memory_checkpoint = str(tampered_packet_path)
                with self.assertRaisesRegex(
                    RuntimeError, "surface packet is not canonical"
                ):
                    generate(packet_mismatch)

                tampered_allowlist_dir = root / "v5-tampered-allowlist" / "case"
                shutil.copytree(source_dir, tampered_allowlist_dir)
                tampered_allowlist_path = tampered_allowlist_dir / "memory.private.json"
                tampered_allowlist = json.loads(
                    tampered_allowlist_path.read_text("utf-8")
                )
                tampered_allowlist["allowed_source_keys"].append("forged-source")
                _write_json(tampered_allowlist_path, tampered_allowlist)
                allowlist_mismatch = copy.copy(target_args)
                allowlist_mismatch.output_dir = str(
                    root / "v6-allowlist-mismatch" / "case"
                )
                allowlist_mismatch.import_memory_checkpoint = str(
                    tampered_allowlist_path
                )
                with self.assertRaisesRegex(
                    RuntimeError, "allowed_source_keys mismatch"
                ):
                    generate(allowlist_mismatch)

                tampered_ledger_dir = root / "v5-tampered-ledger" / "case"
                shutil.copytree(source_dir, tampered_ledger_dir)
                tampered_ledger_path = tampered_ledger_dir / "usage.jsonl"
                tampered_rows = [
                    json.loads(line)
                    for line in tampered_ledger_path.read_text("utf-8").splitlines()
                    if line.strip()
                ]
                for row in tampered_rows:
                    if (
                        row.get("event") == "completed"
                        and row.get("phase") != "surface_answer"
                    ):
                        row["total"] += 1
                        break
                tampered_ledger_path.write_text(
                    "".join(json.dumps(row) + "\n" for row in tampered_rows),
                    encoding="utf-8",
                )
                ledger_mismatch = copy.copy(target_args)
                ledger_mismatch.output_dir = str(root / "v6-ledger-mismatch" / "case")
                ledger_mismatch.import_memory_checkpoint = str(
                    tampered_ledger_dir / "memory.private.json"
                )
                with self.assertRaisesRegex(
                    ValueError, "usage arithmetic mismatch"
                ):
                    generate(ledger_mismatch)

                result = generate(target_args)

            self.assertEqual(
                phases,
                [
                    ("source", "reader_initial"),
                    ("source", "surface_answer"),
                    ("target", "surface_answer"),
                ],
            )
            self.assertEqual(result["status"], "COMPLETED")
            self.assertEqual(result["provider_calls_imported"], 1)
            self.assertEqual(result["provider_calls_new"], 1)
            self.assertIsNotNone(result["memory_checkpoint_import"])
            provenance = result["memory_checkpoint_import"]
            self.assertEqual(provenance["excluded_non_memory_ledger_row_count"], 2)
            target_rows = [
                json.loads(line)
                for line in (target_dir / "usage.jsonl").read_text("utf-8").splitlines()
                if line.strip()
            ]
            self.assertFalse(
                any(
                    row.get("event") == "failed"
                    or (
                        row.get("phase") == "surface_answer"
                        and row.get("provider_id") == "openai/gemini-3.5-flash"
                        and row.get("event") == "failed"
                    )
                    for row in target_rows
                )
            )
            self.assertEqual(len(target_rows), 4)


class ProductionLayerChainTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_v7_q_stages_fail_closed_when_prompt_order_was_not_frozen(
        self,
    ) -> None:
        source = (
            Path.cwd()
            / ".dev"
            / "provider_results"
            / "three-case-layered-v7-partial"
            / "cases"
            / "q0030"
        )
        if not (source / "memory.private.json").is_file():
            self.skipTest("private v7 q failure fixture is not present")
        manifest = json.loads((source / "manifest.json").read_text("utf-8"))
        effective_case = json.loads((source / "case.input.json").read_text("utf-8"))
        case = dict(effective_case)
        case.pop("recent_context", None)
        packet = json.loads((source / "evidence.input.json").read_text("utf-8"))
        with self.assertRaisesRegex(ValueError, "prompt/options/payload hash mismatch"):
            await _prepare_provider_stage_import(
                source / "memory.private.json",
                target_manifest=manifest,
                case=case,
                packet=packet,
                snapshot=RequestSnapshot.from_value(manifest["snapshot"]),
                source_keys=_source_keys(packet),
                participant_keys=_participant_keys(case, packet),
                memory_provider_extra={},
            )

    async def test_real_v6_provider_stages_with_invalid_audit_are_rejected(
        self,
    ) -> None:
        source = (
            Path.cwd()
            / ".dev"
            / "provider_results"
            / "three-case-layered-v6-failed"
            / "cases"
            / "good-girl"
        )
        if not (source / "memory.private.json").is_file():
            self.skipTest("private v6 replay fixture is not present")
        manifest = json.loads((source / "manifest.json").read_text("utf-8"))
        target_manifest = copy.deepcopy(manifest)
        target_manifest["limits"]["surface_max_chars"] = 24_000
        effective_case = json.loads((source / "case.input.json").read_text("utf-8"))
        case = dict(effective_case)
        case.pop("recent_context", None)
        packet = json.loads((source / "evidence.input.json").read_text("utf-8"))
        with self.assertRaisesRegex(
            EccrProtocolError,
            "must cite newly visited evidence",
        ):
            await _prepare_provider_stage_import(
                source / "memory.private.json",
                target_manifest=target_manifest,
                case=case,
                packet=packet,
                snapshot=RequestSnapshot.from_value(manifest["snapshot"]),
                source_keys=_source_keys(packet),
                participant_keys=_participant_keys(case, packet),
                memory_provider_extra={},
            )

    async def test_l2_repair_yields_certificate_v2_and_compiled_surface_packet(self) -> None:
        case = _case()
        packet = _packet()
        recent_context = [{"role": "user", "content": "上文"}]
        snapshot = _snapshot(
            case=case,
            packet=packet,
            provider_id="deepseek/deepseek-v4-flash",
            route="l2",
            recent_context=recent_context,
        )
        calls: list[tuple[int, str]] = []

        async def complete(_system: str, _prompt: str, index: int, phase: str):
            calls.append((index, phase))
            if index == 0:
                return _completion("not-json")
            return _completion(_certificate_value(snapshot, packet))

        certificate, stages, detail = await _run_l2(
            case=case,
            packet=packet,
            snapshot=snapshot,
            source_keys={"s1", "s2"},
            participant_keys={"p1", "p2"},
            complete=complete,
        )
        self.assertEqual(calls, [(0, "reader_initial"), (1, "reader_repair")])
        self.assertEqual(detail["route"], "L2")
        self.assertTrue(detail["repair_attempted"])
        self.assertEqual(len(stages), 2)
        packet_for_surface = compile_surface_packet(certificate)
        validate_surface_packet(packet_for_surface, certificate)
        messages = _surface_messages(
            case=case,
            recent_context=recent_context,
            surface_packet_text=packet_for_surface.text,
        )
        payload_text, injected_part = messages[1]["content"].split("\n", 1)
        surface_payload = json.loads(payload_text)
        self.assertEqual(
            surface_payload["execution_semantics"],
            "CONTROLLED_SAME_PROVIDER_SURFACE_CALL_NOT_FULL_ASTRBOT_E2E_PERSONA",
        )
        encoded = injected_part.split("<mr_memory_evidence>", 1)[1].split(
            "</mr_memory_evidence>", 1
        )[0]
        injected = json.loads(encoded)["evidence_certificate"]
        self.assertEqual(injected["certificate_sha256"], certificate.digest)
        self.assertEqual(surface_payload["current_message"], case["query"])

    async def test_l2_normalization_is_audited_and_reasoning_never_controls_flow(
        self,
    ) -> None:
        case = _case()
        packet = _packet()
        snapshot = _snapshot(
            case=case,
            packet=packet,
            provider_id="deepseek/deepseek-v4-flash",
            route="l2",
            recent_context=[],
        )
        raw = _certificate_value(snapshot, packet)
        raw["subjects"][0]["candidate_participant_keys"] = ["p1"]
        raw["subjects"].append(
            {
                "reference": "未绑定称呼",
                "participant_key": "",
                "reference_mode": "UNBOUND",
                "candidate_participant_keys": [],
                "source_keys": ["s2"],
                "valid_at": None,
            }
        )

        async def normalized_complete(_system: str, _prompt: str, index: int, _phase: str):
            self.assertEqual(index, 0)
            return _completion(raw)

        certificate, stages, detail = await _run_l2(
            case=case,
            packet=packet,
            snapshot=snapshot,
            source_keys={"s1", "s2"},
            participant_keys={"p1", "p2"},
            complete=normalized_complete,
        )
        self.assertEqual(certificate.status, "SAFETY_ABSTAIN")
        self.assertEqual(
            [item["action"] for item in detail["normalization_audit"]],
            [
                "canonicalize_redundant_singleton",
                "downgrade_identity_ambiguity",
            ],
        )
        self.assertEqual(
            stages[0]["normalized_certificate_sha256"], certificate.digest
        )

        repair_prompts: list[str] = []

        async def reasoning_complete(_system: str, prompt: str, index: int, _phase: str):
            repair_prompts.append(prompt)
            if index == 0:
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content="not-json",
                                reasoning_content=json.dumps(raw, ensure_ascii=False),
                            )
                        )
                    ]
                )
            return _completion(raw)

        _certificate, _stages, repaired_detail = await _run_l2(
            case=case,
            packet=packet,
            snapshot=snapshot,
            source_keys={"s1", "s2"},
            participant_keys={"p1", "p2"},
            complete=reasoning_complete,
        )
        self.assertTrue(repaired_detail["repair_attempted"])
        self.assertEqual(json.loads(repair_prompts[1])["invalid_response"], "not-json")

    async def test_l3_uses_bounded_orchestrator_then_certificate_v2(self) -> None:
        case = _case()
        packet = _packet()
        snapshot = _snapshot(
            case=case,
            packet=packet,
            provider_id="deepseek/deepseek-v4-flash",
            route="l3",
            recent_context=[],
        )
        fixture = test_orchestrator.EccrOrchestratorTests()
        fixture.setUp()
        fixture.host = {
            "scope_sha256": snapshot.scope_sha256,
            "query_sha256": snapshot.query_sha256,
            "cutoff_at": snapshot.cutoff_at,
            "revision_vector": {
                "message": snapshot.data_revision.message,
                "graph": snapshot.data_revision.graph,
                "identity": snapshot.data_revision.identity,
                "relation": snapshot.data_revision.relation,
                "feedback": snapshot.data_revision.feedback,
                "protocol": snapshot.inference_revision.reader_protocol,
            },
        }
        first = fixture.contract(0)
        second = copy.deepcopy(first)
        second["step_index"] = 1
        third = copy.deepcopy(second)
        third["step_index"] = 2
        third["visited_source_keys"] = ["s1", "s2"]
        third["obligations"][0].update(
            {
                "status": "CONTESTED",
                "support_keys": ["s1"],
                "counter_keys": ["s2"],
                "last_changed_step": 2,
            }
        )
        third["interpretations"].append(
            {
                "id": "i2",
                "statement": "反例解释",
                "status": "CONTESTED",
                "support_keys": ["s2"],
                "counter_keys": ["s1"],
                "uncertainty": "仅候选",
                "origin": "AUDIT_DISCOVERY",
                "discriminates_interpretation_ids": ["i1"],
            }
        )
        third["uncertainties"][0].update(
            {"status": "PRESERVED", "source_keys": ["s1", "s2"]}
        )
        responses = [
            {"contract": first, "actions": [], "memory_brief": None, "terminal": False},
            {"contract": second, "actions": [], "memory_brief": None, "terminal": False},
            {
                "contract": third,
                "actions": [],
                "memory_brief": {
                    "claims": [],
                    "conflicts": [
                        {"statement": "两种解释竞争", "source_keys": ["s1", "s2"]}
                    ],
                    "unresolved": [
                        {"statement": "仍不可升级", "source_keys": ["s1", "s2"]}
                    ],
                },
                "terminal": True,
            },
        ]
        calls: list[tuple[int, str]] = []

        async def complete(_system: str, _prompt: str, index: int, phase: str):
            calls.append((index, phase))
            return _completion(responses[index])

        certificate, stages, detail = await _run_l3(
            case=case,
            packet=packet,
            snapshot=snapshot,
            source_keys={"s1", "s2"},
            participant_keys={"p1", "p2"},
            complete=complete,
            storage=None,
            max_model_calls=3,
            max_retrieval_rounds=0,
            deadline_seconds=30.0,
        )
        self.assertEqual(len(calls), 3)
        self.assertEqual(detail["route"], "L3")
        self.assertEqual(detail["model_calls"], 3)
        self.assertFalse(detail["repair_attempted"])
        self.assertEqual(detail["protocol_failures"], [])
        self.assertEqual(len(stages), 3)
        self.assertIn(certificate.status, {"CERTIFIED", "PARTIAL", "SAFETY_ABSTAIN"})
        expected_packet_sha256 = _l3_certificate_packet_sha256(
            initial_packet=packet,
            retrieval_results=detail["retrieval_results"],
            final_contract=detail["trace"][-1]["contract"],
        )
        self.assertEqual(certificate.packet_sha256, expected_packet_sha256)
        self.assertNotEqual(certificate.packet_sha256, stable_sha256(packet))
        surface_packet = compile_surface_packet(certificate)
        validate_surface_packet(surface_packet, certificate)


if __name__ == "__main__":
    unittest.main()
