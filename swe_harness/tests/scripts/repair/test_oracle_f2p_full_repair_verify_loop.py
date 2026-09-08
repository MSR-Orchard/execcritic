import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[3] / "scripts" / "repair" / "oracle_f2p_full_repair_verify_loop.py"
SPEC = importlib.util.spec_from_file_location("oracle_f2p_full_repair_verify_loop_test_module", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

SWEPRO_VERIFY_SCRIPT = (
    Path(__file__).parents[3]
    / "src"
    / "minisweagent"
    / "run"
    / "benchmarks"
    / "swebench_pro_verify_azure_modal.py"
)
SWEPRO_SPEC = importlib.util.spec_from_file_location("swebench_pro_verify_test_module", SWEPRO_VERIFY_SCRIPT)
SWEPRO_VERIFY = importlib.util.module_from_spec(SWEPRO_SPEC)
SWEPRO_SPEC.loader.exec_module(SWEPRO_VERIFY)


@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [([], True), (["--no-full-eval-script-gate"], False)],
)
def test_cli_full_eval_script_gate_default_and_opt_out(monkeypatch, extra_args, expected):
    original_parse_args = MODULE.argparse.ArgumentParser.parse_args

    class ParsedArguments(Exception):
        pass

    def parse_and_stop(parser):
        args = original_parse_args(parser)
        assert args.full_eval_script_gate is expected
        assert args.force_submit_on_turn_limit is True
        raise ParsedArguments

    monkeypatch.setattr(MODULE.argparse.ArgumentParser, "parse_args", parse_and_stop)
    monkeypatch.setattr(
        MODULE.sys,
        "argv",
        [
            "oracle_f2p_full_repair_verify_loop.py",
            "--input",
            "input.jsonl",
            "--output",
            "output",
            *extra_args,
        ],
    )

    with pytest.raises(ParsedArguments):
        MODULE.main()


def test_cli_generated_test_gate_isolation_defaults_on(monkeypatch):
    original_parse_args = MODULE.argparse.ArgumentParser.parse_args

    class ParsedArguments(Exception):
        pass

    def parse_and_stop(parser):
        args = original_parse_args(parser)
        assert args.isolated_generated_test_gate is True
        raise ParsedArguments

    monkeypatch.setattr(MODULE.argparse.ArgumentParser, "parse_args", parse_and_stop)
    monkeypatch.setattr(
        MODULE.sys,
        "argv",
        [
            "oracle_f2p_full_repair_verify_loop.py",
            "--input",
            "input.jsonl",
            "--output",
            "output",
        ],
    )

    with pytest.raises(ParsedArguments):
        MODULE.main()


@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [([], True), (["--no-force-submit-on-turn-limit"], False)],
)
def test_cli_force_submit_on_turn_limit_default_and_opt_out(monkeypatch, extra_args, expected):
    original_parse_args = MODULE.argparse.ArgumentParser.parse_args

    class ParsedArguments(Exception):
        pass

    def parse_and_stop(parser):
        args = original_parse_args(parser)
        assert args.force_submit_on_turn_limit is expected
        raise ParsedArguments

    monkeypatch.setattr(MODULE.argparse.ArgumentParser, "parse_args", parse_and_stop)
    monkeypatch.setattr(
        MODULE.sys,
        "argv",
        [
            "oracle_f2p_full_repair_verify_loop.py",
            "--input",
            "input.jsonl",
            "--output",
            "output",
            *extra_args,
        ],
    )

    with pytest.raises(ParsedArguments):
        MODULE.main()


def test_cli_sampling_and_workspace_replay_options(monkeypatch):
    original_parse_args = MODULE.argparse.ArgumentParser.parse_args

    class ParsedArguments(Exception):
        pass

    def parse_and_stop(parser):
        args = original_parse_args(parser)
        assert args.temperature == 0.95
        assert args.top_p == 0.95
        assert args.restore_mode == "replay"
        assert args.seed_trajectory is True
        assert args.max_gate_checks == 10
        assert args.continue_on_duplicate_patch is True
        assert args.reasoning_effort == "medium"
        raise ParsedArguments

    monkeypatch.setattr(MODULE.argparse.ArgumentParser, "parse_args", parse_and_stop)
    monkeypatch.setattr(
        MODULE.sys,
        "argv",
        [
            "oracle_f2p_full_repair_verify_loop.py",
            "--input",
            "input.jsonl",
            "--output",
            "output",
            "--temperature",
            "0.95",
            "--top-p",
            "0.95",
            "--restore-mode",
            "replay",
            "--seed-trajectory",
            "--max-gate-checks",
            "10",
            "--continue-on-duplicate-patch",
            "--reasoning-effort",
            "medium",
        ],
    )

    with pytest.raises(ParsedArguments):
        MODULE.main()


def test_normalize_patch_for_apply_restores_required_final_newline():
    patch = "diff --git a/pkg.py b/pkg.py\n@@ -1 +1 @@\n-old\n+new"

    normalized = MODULE.normalize_patch_for_apply(patch)

    assert normalized == patch + "\n"
    assert MODULE.normalize_patch_for_apply(normalized) == normalized
    assert MODULE.normalize_patch_for_apply("\n") == ""


def test_extract_patch_prefers_native_submission():
    patch = "diff --git a/pkg.py b/pkg.py\n@@ -1 +1 @@\n-old\n+new"

    extracted, info = MODULE.extract_patch({"info": {"submission": patch}, "messages": []})

    assert extracted == patch + "\n"
    assert info["source"] == "info.submission"
    assert info["valid"] is True


def test_source_patch_override_supports_empty_unknown_matched_round0(tmp_path):
    input_path = tmp_path / "input.jsonl"
    manifest_path = tmp_path / "manifest.jsonl"
    input_path.write_text(__import__("json").dumps({
        "metadata": {"instance_id": "org__repo-1", "verify_status": "unknown"},
        "source_patch_override": "",
        "messages": [{"role": "tool", "content": "diff --git a/stale.py b/stale.py"}],
    }) + "\n")

    summary = MODULE.build_source_manifest(
        input_path,
        manifest_path,
        {"org__repo-1": {"instance_id": "org__repo-1"}},
        rollouts_per_source=1,
        allow_empty_source_patch=True,
    )
    source_record = next(MODULE.source_records(manifest_path))
    row, patch = MODULE.load_source_row(input_path, source_record)

    assert summary["eligible_source_rows"] == 1
    assert source_record["patch_chars"] == 0
    assert source_record["patch_info"]["source"] == "source_patch_override"
    assert patch == ""
    assert row["messages"]


def test_build_source_manifest_reports_zero_eligible_rows(tmp_path):
    input_path = tmp_path / "input.jsonl"
    manifest_path = tmp_path / "manifest.jsonl"
    input_path.write_text(__import__("json").dumps({
        "metadata": {"instance_id": "org__repo-1", "verify_status": "resolved"},
        "source_patch_override": "",
    }) + "\n")

    summary = MODULE.build_source_manifest(
        input_path,
        manifest_path,
        {"org__repo-1": {"instance_id": "org__repo-1"}},
        rollouts_per_source=1,
        allow_empty_source_patch=True,
    )

    assert summary["eligible_source_rows"] == 0
    assert summary["total_attempt_units"] == 0


def test_build_dataset_manifest_selects_one_task_per_instance(tmp_path):
    rows = [{"instance_id": "org__repo-1"}, {"instance_id": "org__repo-2"}]
    manifest = tmp_path / "manifest.jsonl"

    summary = MODULE.build_dataset_manifest(rows, manifest, rollouts_per_source=1)
    records = [__import__("json").loads(line) for line in manifest.read_text().splitlines()]

    assert summary["eligible_source_rows"] == 2
    assert summary["selection_mode"] == "dataset_instances"
    assert [record["instance_id"] for record in records] == ["org__repo-1", "org__repo-2"]
    assert all(record["patch_sha256"] == "" for record in records)


def test_swebench_pro_instance_normalization_accepts_python_literal_lists():
    instance = SWEPRO_VERIFY.prepare_instance(
        {
            "instance_id": "instance_org__repo-deadbeef",
            "dockerhub_tag": "org.repo-org__repo-deadbeef",
            "problem_statement": "Fix the broken behavior.",
            "requirements": "- Preserve compatibility.",
            "interface": "Type: Function\nName: fix_behavior",
            "fail_to_pass": "['TestOne', 'TestTwo']",
            "pass_to_pass": '["TestStable"]',
            "selected_test_files_to_run": "['TestOne']",
        }
    )

    assert instance["FAIL_TO_PASS"] == ["TestOne", "TestTwo"]
    assert instance["PASS_TO_PASS"] == ["TestStable"]
    assert instance["image_name"] == "mirror.gcr.io/jefzda/sweap-images:org.repo-org__repo-deadbeef"
    assert instance["_swebench_pro"] is True
    assert instance["problem_statement"] == (
        "Fix the broken behavior.\n\n"
        "Requirements:\n- Preserve compatibility.\n\n"
        "New interfaces introduced:\nType: Function\nName: fix_behavior"
    )


def test_swebench_pro_problem_statement_normalization_is_idempotent():
    original = {
        "instance_id": "instance_org__repo-deadbeef",
        "dockerhub_tag": "org.repo-org__repo-deadbeef",
        "problem_statement": "Fix the broken behavior.",
        "requirements": "- Preserve compatibility.",
        "interface": "Type: Function\nName: fix_behavior",
        "fail_to_pass": '[]',
        "pass_to_pass": '[]',
        "selected_test_files_to_run": '[]',
    }

    prepared = SWEPRO_VERIFY.prepare_instance(original)
    prepared_again = SWEPRO_VERIFY.prepare_instance(prepared)

    assert prepared_again["problem_statement"] == prepared["problem_statement"]
    assert prepared_again["problem_statement"].count("Requirements:") == 1
    assert prepared_again["problem_statement"].count("New interfaces introduced:") == 1


def test_swebench_pro_focused_targets_and_full_grade_are_separate():
    instance = {
        "FAIL_TO_PASS": ["TestOne", "TestTwo"],
        "PASS_TO_PASS": ["TestStable"],
        "selected_test_files_to_run": ["all_tests.py"],
        "_swebench_pro_selected_nodes": ["TestTwo"],
    }

    assert SWEPRO_VERIFY._selected_targets(instance) == ["TestTwo"]
    assert SWEPRO_VERIFY._grade_results(
        instance,
        {"TestOne": "PASSED", "TestTwo": "PASSED", "TestStable": "PASSED"},
    )["resolved"] is True
    assert SWEPRO_VERIFY._grade_results(
        instance,
        {"TestOne": "PASSED", "TestTwo": "PASSED", "TestStable": "FAILED"},
    )["resolved"] is False


def test_swebench_pro_sandbox_boot_retries_transient_failures(monkeypatch):
    calls = []

    def factory():
        calls.append(len(calls) + 1)
        if len(calls) < 3:
            raise RuntimeError("transient create failure")
        return "sandbox"

    monkeypatch.setenv("SWEPRO_BOOT_RETRIES", "3")
    monkeypatch.setattr(SWEPRO_VERIFY.time, "sleep", lambda _delay: None)

    assert SWEPRO_VERIFY.create_sandbox_with_retry(factory, "instance-1") == "sandbox"
    assert calls == [1, 2, 3]


def test_swebench_pro_harness_root_validation(monkeypatch, tmp_path):
    checkout = tmp_path / "SWE-bench_Pro-os"
    (checkout / "run_scripts").mkdir(parents=True)
    monkeypatch.setenv("SWEPRO_HARNESS_ROOT", str(checkout))

    assert SWEPRO_VERIFY.validate_harness_root() == checkout


def test_swebench_pro_binary_patch_sections_are_ignored():
    text_diff = """diff --git a/pkg.py b/pkg.py
--- a/pkg.py
+++ b/pkg.py
@@ -1 +1 @@
-old
+new
"""
    binary_diff = """diff --git a/logo.png b/logo.png
GIT binary patch
literal 3
abc
"""

    assert SWEPRO_VERIFY.strip_binary_hunks(text_diff + binary_diff) == text_diff
