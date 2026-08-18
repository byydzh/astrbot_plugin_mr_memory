from __future__ import annotations

import json
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

from scripts.eccr_packet_experiment import (
    load_case_bundle,
    load_generation_bundle,
)
from scripts.masked_ab_experiment import _load_pilot_gold
from scripts.three_case_experiment import (
    REPORT_SCHEMA_VERSION,
    SUITE_SCHEMA_VERSION,
    _file_sha256,
    _run_suite_command,
    _surface_results,
    build_report,
    prepare_surface_inputs,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


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


def _brief(source: str = "src-1") -> dict[str, object]:
    return {
        "claims": [
            {"statement": "观察到一条事实", "source_keys": [source], "confidence": 0.8}
        ],
        "conflicts": [],
        "unresolved": [
            {"statement": "仍需保留疑虑", "source_keys": [source]}
        ],
    }


class GenerationOnlyBoundaryTest(unittest.TestCase):
    def test_generation_loader_does_not_require_or_parse_gold(self) -> None:
        with _workspace_tempdir() as root:
            _write_json(
                root / "case.json",
                {
                    "schema_version": "eccr.packet.case.v1",
                    "case_id": "case-a",
                    "query": "问题",
                    "umo": "scope-a",
                    "cutoff_at": 10,
                    "layer": "oracle_synthesis_diagnostic",
                },
            )
            _write_json(
                root / "evidence_packet.json",
                {
                    "messages": [
                        {
                            "source_key": "src-1",
                            "umo": "scope-a",
                            "sent_at": 9,
                            "text": "证据",
                        }
                    ]
                },
            )
            # Invalid JSON proves the generation loader never opens this file.
            (root / "gold.json").write_text("not-json", encoding="utf-8")
            bundle = load_generation_bundle(root)
            self.assertEqual(bundle.case_id, "case-a")
            self.assertNotIn("gold_sha256", bundle.hashes)
            with self.assertRaises(json.JSONDecodeError):
                load_case_bundle(root)

    def test_masked_l3_generation_does_not_parse_gold(self) -> None:
        with _workspace_tempdir() as root:
            invalid_gold = root / "gold.json"
            invalid_gold.write_text("not-json", encoding="utf-8")
            self.assertIsNone(
                _load_pilot_gold(invalid_gold, generation_only=True)
            )
            with self.assertRaises(json.JSONDecodeError):
                _load_pilot_gold(invalid_gold, generation_only=False)


class SurfacePreparationTest(unittest.TestCase):
    def test_all_three_real_result_shapes_reach_surface_without_truncation(self) -> None:
        with _workspace_tempdir() as root:
            case_path = root / "memory-case.json"
            _write_json(
                case_path,
                {
                    "case_id": "case-a",
                    "query": "这里是什么意思？",
                    "cutoff_at": 100,
                },
            )
            shapes = {
                "call-726-top-level": {
                    "status": "COMPLETED",
                    "brief": _brief(),
                    "raw_response": {"completion": "#726 完整原始输出"},
                    "gold_score": {"legacy": True},
                },
                "good-girl-multi-round": {
                    "status": "COMPLETED",
                    "result": {
                        "brief": _brief(),
                        "rounds": [
                            {
                                "phase": "audit_compile",
                                "call_index": 0,
                                "raw_response": {"completion": "好女孩第一轮完整输出"},
                            },
                            {
                                "phase": "audit_review",
                                "call_index": 1,
                                "raw_response": {"completion": "好女孩第二轮完整输出"},
                                "brief": _brief(),
                            },
                        ],
                    },
                    "evaluation": {"legacy": True},
                },
                "q0030-one-pass": {
                    "status": "COMPLETED",
                    "result": {
                        "brief": _brief(),
                        "raw_response": {"completion": "q0030 完整原始输出"},
                    },
                },
            }
            for index, (shape_name, wrapper) in enumerate(shapes.items()):
                with self.subTest(shape=shape_name):
                    result_path = root / f"layer-result-{index}.json"
                    output_dir = root / f"surface-{index}"
                    _write_json(result_path, wrapper)
                    prepared = prepare_surface_inputs(
                        memory_case_path=case_path,
                        output_dir=output_dir,
                        layer_result_path=result_path,
                        arm_id=f"layer-{index}",
                    )
                    provenance = json.loads(
                        (output_dir / "layer-output.private.json").read_text("utf-8")
                    )
                    expected_body = (
                        wrapper["result"]
                        if isinstance(wrapper.get("result"), dict)
                        else {
                            key: value
                            for key, value in wrapper.items()
                            if key not in {"gold_score", "evaluation"}
                        }
                    )
                    self.assertEqual(provenance["actual_layer_output"], expected_body)
                    self.assertEqual(
                        prepared["actual_output_sha256"],
                        provenance["actual_layer_output_sha256"],
                    )
                    arm = json.loads(Path(prepared["arm_path"]).read_text("utf-8"))
                    self.assertEqual(arm["memory_brief"], _brief())


class ReportAggregationTest(unittest.TestCase):
    def test_surface_report_rejects_failed_empty_or_hash_mismatched_answers(self) -> None:
        with _workspace_tempdir() as root:
            variants = [
                {"status": "FAILED", "run_id": "r1", "arm_id": "layered"},
                {"status": "COMPLETED", "run_id": "r1", "arm_id": "layered", "answer": ""},
                {
                    "status": "COMPLETED",
                    "run_id": "r1",
                    "arm_id": "layered",
                    "answer": "完整回答",
                    "answer_sha256": "0" * 64,
                },
            ]
            for index, row in enumerate(variants):
                with self.subTest(index=index):
                    path = root / f"surface-{index}.json"
                    _write_json(path, {"results": [row]})
                    with self.assertRaises(ValueError):
                        _surface_results(path, selected_arm_ids={"layered"})

    def test_report_keeps_actual_answers_and_leaves_human_review_blank(self) -> None:
        with _workspace_tempdir() as root:
            cases = []
            for index in range(3):
                case_dir = root / f"case-{index}"
                case_path = case_dir / "case.json"
                layer_path = case_dir / "layer.json"
                surface_path = case_dir / "surface.json"
                gold_path = case_dir / "gold.json"
                layer_ledger = case_dir / "layer-usage.jsonl"
                surface_ledger = case_dir / "surface-usage.jsonl"
                _write_json(
                    case_path,
                    {
                        "schema_version": "surface.brief.case.v1",
                        "case_id": f"surface-case-{index}",
                        "query": f"问题{index}",
                        "recent_context": [{"sender": "成员", "text": f"上下文{index}"}],
                        "cutoff_at": 100 + index,
                    },
                )
                _write_json(
                    layer_path,
                    {
                        "status": "COMPLETED",
                        "run_id": f"layer-{index}",
                        "result": {
                            "brief": _brief(),
                            "raw_response": "原始记忆输出",
                            "elapsed_ms": 22 + index,
                        },
                        "usage": {
                            "calls": 1,
                            "input_other": 10,
                            "input_cached": 2,
                            "output": 3,
                            "total": 15,
                            "elapsed_ms": 20,
                            "usage_complete": True,
                        },
                    },
                )
                surface_row = {
                    "status": "COMPLETED",
                    "run_id": f"surface-{index}",
                    "arm_id": "layered",
                    "answer": f"实际回答{index}",
                    "elapsed_ms": 8 + index,
                    "usage": {
                        "calls": 1,
                        "input_other": 4,
                        "input_cached": 0,
                        "output": 2,
                        "total": 6,
                        "elapsed_ms": 7,
                        "usage_complete": True,
                    },
                }
                _write_json(surface_path, {"results": [surface_row]})
                _write_json(gold_path, {"human_evidence": [f"证据{index}"]})
                layer_ledger.write_text(
                    json.dumps(
                        {"event": "attempted", "request_id": f"lr-{index}", "run_id": f"layer-{index}"}
                    )
                    + "\n"
                    + json.dumps(
                        {
                            "event": "completed",
                            "request_id": f"lr-{index}",
                            "run_id": f"layer-{index}",
                            "usage_present": True,
                            "phase": "memory_reader",
                            "call_index": 0,
                            "input_other": 10,
                            "input_cached": 2,
                            "output": 3,
                            "total": 15,
                            "elapsed_ms": 20,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                surface_ledger.write_text(
                    json.dumps(
                        {"event": "attempted", "request_id": f"sr-{index}", "run_id": f"surface-{index}"}
                    )
                    + "\n"
                    + json.dumps(
                        {
                            "event": "completed",
                            "request_id": f"sr-{index}",
                            "run_id": f"surface-{index}",
                            "usage_present": True,
                            "phase": "surface_answer",
                            "call_index": 0,
                            "input_other": 4,
                            "input_cached": 0,
                            "output": 2,
                            "total": 6,
                            "elapsed_ms": 7,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                cases.append(
                    {
                        "case_id": f"case-{index}",
                        "title": f"案例{index}",
                        "route": "L3" if index < 2 else "L2",
                        "case_path": str(case_path),
                        "layer_result_path": str(layer_path),
                        "layer_ledger_path": str(layer_ledger),
                        "surface_results_path": str(surface_path),
                        "surface_ledger_path": str(surface_ledger),
                        "surface_arm_ids": ["layered"],
                        "gold_path": str(gold_path),
                    }
                )
            suite_path = root / "suite.json"
            failure_result = root / "prior-failure" / "memory.private.json"
            failure_ledger = root / "prior-failure" / "usage.jsonl"
            _write_json(
                failure_result,
                {
                    "run_id": "failed-v1",
                    "status": "FAILED",
                    "error_type": "ValueError",
                    "error_detail": "strict contract rejected the provider response",
                    "elapsed_ms": 192028.783,
                    "usage": {
                        "calls": 2,
                        "input_other": 19661,
                        "input_cached": 18048,
                        "output": 24364,
                        "total": 62073,
                        "elapsed_ms": 191970.187,
                        "usage_complete": True,
                    },
                },
            )
            failure_rows = []
            for call_index, values in enumerate(
                ((18170, 0, 13585, 31755, 109059.077), (1491, 18048, 10779, 30318, 82911.11))
            ):
                request_id = f"failed-v1:eccr_compile:{call_index}"
                failure_rows.extend(
                    [
                        {
                            "event": "attempted",
                            "request_id": request_id,
                            "run_id": "failed-v1",
                            "phase": "eccr_compile",
                            "call_index": call_index,
                        },
                        {
                            "event": "completed",
                            "request_id": request_id,
                            "run_id": "failed-v1",
                            "phase": "eccr_compile",
                            "call_index": call_index,
                            "usage_present": True,
                            "input_other": values[0],
                            "input_cached": values[1],
                            "output": values[2],
                            "total": values[3],
                            "elapsed_ms": values[4],
                        },
                    ]
                )
            failure_ledger.parent.mkdir(parents=True, exist_ok=True)
            failure_ledger.write_text(
                "".join(json.dumps(row) + "\n" for row in failure_rows),
                encoding="utf-8",
            )
            _write_json(
                suite_path,
                {
                    "schema_version": SUITE_SCHEMA_VERSION,
                    "cases": cases,
                    "prior_failures": [
                        {
                            "failure_id": "v1-subject-contract",
                            "title": "v1 主体绑定协议失败",
                            "result_path": str(failure_result),
                            "ledger_path": str(failure_ledger),
                            "expected_total": 62073,
                        }
                    ],
                },
            )
            result = build_report(suite_path, root / "report")
            report = json.loads(Path(result["report_path"]).read_text("utf-8"))
            self.assertEqual(report["schema_version"], REPORT_SCHEMA_VERSION)
            self.assertEqual(report["automatic_quality_scores"], None)
            self.assertEqual(
                [
                    item["surface"]["outputs"][0]["actual_answer"]
                    for item in report["cases"]
                ],
                ["实际回答0", "实际回答1", "实际回答2"],
            )
            self.assertEqual(report["cases"][0]["combined_cost"]["total"], 21)
            self.assertEqual(report["cases"][0]["combined_cost"]["wall_elapsed_ms"], 30)
            self.assertEqual(report["prior_failures"][0]["usage"]["total"], 62073)
            self.assertEqual(
                report["prior_failures"][0]["phases"], ["eccr_compile"]
            )
            self.assertEqual(
                report["cost_summary"]["all_measured_attempts"]["total"],
                62073 + 3 * 21,
            )
            self.assertEqual(
                report["cases"][0]["input"]["actual_case"]["recent_context"][0]["text"],
                "上下文0",
            )
            self.assertEqual(
                report["cases"][0]["layer"]["ledger"]["stage_costs"][0],
                {
                    "request_id": f"lr-0",
                    "phase": "memory_reader",
                    "call_index": 0,
                    "event": "completed",
                    "input_other": 10,
                    "input_cached": 2,
                    "output": 3,
                    "total": 15,
                    "elapsed_ms": 20,
                    "usage_complete": True,
                },
            )
            self.assertTrue(
                all(value is None for value in report["cases"][0]["human_review"].values())
            )
            markdown = Path(result["markdown_path"]).read_text("utf-8")
            for index in range(3):
                self.assertIn(f"实际回答{index}", markdown)
                self.assertIn(f"上下文{index}", markdown)
            self.assertIn("62,073", f"{report['prior_failures'][0]['usage']['total']:,}")
            self.assertIn("62073", markdown)
            self.assertNotIn("strict contract rejected", markdown)
            failure_ledger.write_text(
                "".join(json.dumps(row) + "\n" for row in failure_rows[:-1]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "ledger usage is incomplete"):
                build_report(suite_path, root / "incomplete-failure-report")


class SuiteRunnerDryRunTest(unittest.TestCase):
    def test_real_v8_plan_imports_two_completed_cases_and_fresh_q_cap_three(
        self,
    ) -> None:
        plugin_root = Path.cwd()
        source = (
            plugin_root
            / ".dev"
            / "provider_results"
            / "three-case-layered-v7-partial"
        )
        dev_root = plugin_root.parent / ".dev"
        if not (source / "cases" / "good-girl" / "result.private.json").is_file():
            self.skipTest("private v7 completed cases are not present")
        args = type(
            "Args",
            (),
            {
                "dev_root": str(dev_root),
                "output_root": str(
                    plugin_root
                    / ".dev"
                    / "provider_results"
                    / "three-case-layered-v8-test-must-not-exist"
                ),
                "config": str(source / "run-plan.json"),
                "subconscious_provider_id": "deepseek/deepseek-v4-flash",
                "main_provider_id": "openai/gemini-3.5-flash",
                "deadline_seconds": 600.0,
                "max_output_tokens": 384000,
                "surface_max_output_tokens": 65536,
                "import_memory_checkpoint": [],
                "import_completed_case": [
                    "call-726="
                    + str(source / "cases" / "call-726" / "result.private.json"),
                    "good-girl="
                    + str(source / "cases" / "good-girl" / "result.private.json"),
                ],
                "import_provider_stages_checkpoint": [],
                "authorize_provider_calls": False,
            },
        )()
        plan = _run_suite_command(args)
        self.assertEqual(plan["schema_version"], "mr-memory.three-case.run-plan.v3")
        self.assertEqual(plan["provider_calls_upper_bound"], 3)
        self.assertEqual(plan["new_provider_calls_upper_bound"], 3)
        self.assertEqual(
            [item["provider_calls_upper_bound"] for item in plan["commands"]],
            [0, 0, 3],
        )
        self.assertEqual(
            [item["new_provider_calls_upper_bound"] for item in plan["commands"]],
            [0, 0, 3],
        )
        self.assertTrue(
            all(
                "--source-attempt-id" in item["argv"]
                for item in plan["commands"][:2]
            )
        )
        self.assertNotIn(
            "--import-provider-stages-checkpoint", plan["commands"][2]["argv"]
        )
        self.assertIn("--surface-max-chars", plan["commands"][2]["argv"])

    def test_real_v7_import_plan_has_four_new_call_hard_cap(self) -> None:
        plugin_root = Path.cwd()
        stage_source = (
            plugin_root
            / ".dev"
            / "provider_results"
            / "three-case-layered-v6-failed"
        )
        completed_source = (
            plugin_root
            / ".dev"
            / "provider_results"
            / "three-case-layered-v7-partial"
        )
        dev_root = plugin_root.parent / ".dev"
        if not (
            stage_source / "cases" / "good-girl" / "memory.private.json"
        ).is_file() or not (
            completed_source / "cases" / "call-726" / "result.private.json"
        ).is_file():
            self.skipTest("private v6 import fixtures are not present")
        args = type(
            "Args",
            (),
            {
                "dev_root": str(dev_root),
                "output_root": str(
                    plugin_root
                    / ".dev"
                    / "provider_results"
                    / "three-case-layered-v7-test-must-not-exist"
                ),
                "config": str(stage_source / "run-plan.json"),
                "subconscious_provider_id": "deepseek/deepseek-v4-flash",
                "main_provider_id": "openai/gemini-3.5-flash",
                "deadline_seconds": 600.0,
                "max_output_tokens": 384000,
                "surface_max_output_tokens": 65536,
                "import_memory_checkpoint": [],
                "import_completed_case": [
                    "call-726="
                    + str(
                        completed_source
                        / "cases"
                        / "call-726"
                        / "result.private.json"
                    )
                ],
                "import_provider_stages_checkpoint": [
                    "good-girl="
                    + str(
                        stage_source
                        / "cases"
                        / "good-girl"
                        / "memory.private.json"
                    )
                ],
                "authorize_provider_calls": False,
            },
        )()
        plan = _run_suite_command(args)
        self.assertEqual(plan["schema_version"], "mr-memory.three-case.run-plan.v2")
        self.assertEqual(plan["provider_calls_upper_bound"], 4)
        self.assertEqual(plan["new_provider_calls_upper_bound"], 4)
        self.assertEqual(
            [item["provider_calls_upper_bound"] for item in plan["commands"]],
            [0, 1, 3],
        )
        self.assertIn("import-completed-case", plan["commands"][0]["argv"])
        self.assertNotIn("--authorize-provider-calls", plan["commands"][0]["argv"])
        self.assertIn(
            "--import-provider-stages-checkpoint", plan["commands"][1]["argv"]
        )
        self.assertEqual(
            plan["commands"][1]["limits"]["surface_max_chars"], 24000
        )
        self.assertEqual(
            plan["commands"][2]["limits"]["surface_max_chars"], 24000
        )

    def test_dry_run_lists_production_chain_and_no_gold_provider_input(self) -> None:
        with _workspace_tempdir() as root:
            dev = root / ".dev"
            call = dev / "experiments" / "masked-call-726"
            good = dev / "eccr_cases" / "good_girl"
            q = dev / "experiments" / "layered-three-case" / "fixtures" / "q0030"
            imported_memory = root / "v5-call-726-memory.private.json"
            required = [
                root / "config.json",
                imported_memory,
                call / "call_r4.json",
                call / "messages.jsonl",
                call / "graph_r4.db",
                call / "candidates.json",
                call / "surface-ab-v1-input" / "case.json",
                good / "case.json",
                good / "evidence_packet.json",
                q / "case.json",
                q / "evidence_packet.json",
            ]
            for path in required:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
            imported_run_id = "layered-import-memory"
            _write_json(
                imported_memory,
                {
                    "status": "COMPLETED",
                    "run_id": imported_run_id,
                    "usage": {
                        "calls": 1,
                        "total": 15,
                        "usage_complete": True,
                    },
                },
            )
            _write_json(
                imported_memory.parent / "manifest.json",
                {"frozen": True, "inputs": {"database_sha256": None}},
            )
            (imported_memory.parent / "usage.jsonl").write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {
                            "event": "attempted",
                            "request_id": "memory:0",
                            "run_id": imported_run_id,
                        },
                        {
                            "event": "completed",
                            "request_id": "memory:0",
                            "run_id": imported_run_id,
                            "usage_present": True,
                            "input_other": 10,
                            "input_cached": 1,
                            "output": 4,
                            "total": 15,
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            args = type(
                "Args",
                (),
                {
                    "dev_root": str(dev),
                    "output_root": str(root / "output"),
                    "config": str(root / "config.json"),
                    "subconscious_provider_id": "deepseek/deepseek-v4-flash",
                    "main_provider_id": "openai/gemini-3.5-flash",
                    "deadline_seconds": 600.0,
                    "max_output_tokens": 384000,
                    "surface_max_output_tokens": 65536,
                    "import_memory_checkpoint": [
                        f"call-726={imported_memory}"
                    ],
                    "authorize_provider_calls": False,
                },
            )()
            plan = _run_suite_command(args)
            self.assertEqual(plan["status"], "DRY_RUN_NOT_EXECUTED")
            self.assertEqual(plan["provider_calls_upper_bound"], 11)
            self.assertEqual(plan["new_provider_calls_upper_bound"], 8)
            self.assertEqual(
                plan["limits"],
                {
                    "subconscious_max_output_tokens": 384000,
                    "surface_max_output_tokens": 65536,
                },
            )
            self.assertFalse(plan["gold_paths_present_in_provider_commands"])
            provider_argv = [
                item["argv"]
                for item in plan["commands"]
                if item["provider_calls_upper_bound"]
            ]
            self.assertEqual(len(provider_argv), 3)
            self.assertIn("--import-memory-checkpoint", provider_argv[0])
            self.assertNotIn("--import-memory-checkpoint", provider_argv[1])
            self.assertNotIn("--import-memory-checkpoint", provider_argv[2])
            self.assertEqual(
                plan["memory_checkpoint_imports"]["call-726"]["sha256"],
                _file_sha256(imported_memory),
            )
            self.assertEqual(
                plan["memory_checkpoint_imports"]["call-726"]["calls"], 1
            )
            self.assertTrue(
                all(
                    "scripts.layered_case_generation" in argv
                    and "generate" in argv
                    for argv in provider_argv
                )
            )
            forbidden_legacy_modules = {
                "scripts.masked_ab_experiment",
                "scripts.eccr_packet_experiment",
                "scripts.surface_brief_ab_experiment",
            }
            self.assertTrue(
                all(
                    forbidden_legacy_modules.isdisjoint(set(argv))
                    for argv in provider_argv
                )
            )
            self.assertTrue(
                all("gold" not in " ".join(argv).casefold() for argv in provider_argv)
            )
            self.assertTrue(
                all(
                    argv[argv.index("--max-output-tokens") + 1] == "384000"
                    and argv[argv.index("--surface-max-output-tokens") + 1]
                    == "65536"
                    for argv in provider_argv
                )
            )
            self.assertEqual(
                [item["provider_calls_upper_bound"] for item in plan["commands"]],
                [0, 4, 4, 3],
            )
            self.assertIn("--database", provider_argv[0])
            database_index = provider_argv[0].index("--database") + 1
            self.assertTrue(
                provider_argv[0][database_index].endswith(
                    str(Path("prepared-input") / "call-726" / "scope.db")
                )
            )


if __name__ == "__main__":
    unittest.main()
