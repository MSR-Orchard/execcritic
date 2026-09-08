import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[3] / "scripts" / "repair" / "build_testpatch_testmap.py"
SPEC = importlib.util.spec_from_file_location("build_testpatch_testmap", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(instance_id="org__repo-1", run_key=None, **overrides):
    row = {
        "instance_id": instance_id,
        "run_key": run_key or instance_id,
        "submitted": True,
        "exit_status": "Submitted",
        "test_patch": "diff --git a/tests/test_x.py b/tests/test_x.py\n",
        "test_command": "python -m pytest -q tests/test_x.py::test_regression",
        "trajectory_path": f"trajectories/{instance_id}.traj.json",
        "base_clean_fail": True,
        "gold_pass": True,
        "base_fail_gold_pass": True,
    }
    row.update(overrides)
    return row


def test_build_map_uses_canonical_schema_without_gold_labels():
    test_map, summary = MODULE.build_test_map([_row()])

    assert test_map == {
        "org__repo-1": [
            {
                "test_patch": "diff --git a/tests/test_x.py b/tests/test_x.py\n",
                "test_command": "python -m pytest -q tests/test_x.py::test_regression",
                "source_trajectory": "trajectories/org__repo-1.traj.json",
                "source_exit_status": "Submitted",
            }
        ]
    }
    assert summary["test_map_ids"] == 1
    assert summary["gold_pass_audit_only"] == 1
    assert summary["gold_labels_used_for_selection"] is False
    assert "gold_pass" not in test_map["org__repo-1"][0]


def test_repeat_rows_are_deduplicated_then_capped():
    first = _row(run_key="org__repo-1#rep0")
    duplicate = _row(run_key="org__repo-1#rep1")
    second = _row(
        run_key="org__repo-1#rep2",
        test_patch="diff --git a/tests/test_y.py b/tests/test_y.py\n",
        test_command="python -m pytest -q tests/test_y.py::test_other",
    )

    test_map, summary = MODULE.build_test_map(
        [first, duplicate, second],
        max_tests_per_instance=1,
    )

    assert len(test_map["org__repo-1"]) == 1
    assert summary["duplicate_tests_dropped"] == 1
    assert summary["tests_truncated_by_cap"] == 1


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"submitted": False}, "not_submitted"),
        ({"test_patch": ""}, "empty_patch"),
        ({"test_command": ""}, "empty_command"),
    ],
)
def test_invalid_or_unsubmitted_rows_are_counted(overrides, reason):
    test_map, summary = MODULE.build_test_map([_row(**overrides)])

    assert test_map == {}
    assert summary["skipped"] == {reason: 1}


def test_rejects_duplicate_run_keys():
    with pytest.raises(ValueError, match="duplicate run_key"):
        MODULE.build_test_map([_row(), _row()])


def test_cli_writes_map_and_default_summary(tmp_path):
    results = tmp_path / "results.jsonl"
    results.write_text(json.dumps(_row()) + "\n", encoding="utf-8")
    output = tmp_path / "behavior_contract.testmap.json"

    subprocess.run(
        [sys.executable, str(SCRIPT), "--results", str(results), "--out", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(output.read_text())["org__repo-1"][0]["test_command"].endswith(
        "::test_regression"
    )
    assert output.with_suffix(".summary.json").is_file()
