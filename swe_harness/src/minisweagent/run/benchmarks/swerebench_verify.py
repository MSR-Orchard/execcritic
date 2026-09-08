#!/usr/bin/env python3

"""Verify patches for SWE-rebench instances.

Follows the evaluation approach from SWE-rebench-V2/scripts/eval.py:
- Uses `docker run --rm` with patches mounted as a volume
- Applies code patch + test_patch via `git apply`
- Runs test_cmd and parses output using the log_parser from install_config
- Checks FAIL_TO_PASS and PASS_TO_PASS against parsed test status map
"""

import json
import re
import subprocess
import tempfile
from pathlib import Path

import typer
from rich.console import Console

from minisweagent.utils.log import logger

console = Console(highlight=False)
app = typer.Typer(rich_markup_mode="rich", add_completion=False)

DATASET_MAPPING = {
    "rebench": "nebius/SWE-rebench",
    "rebench_v2": "nebius/SWE-rebench-V2",
}

# Timing patterns stripped from test names to avoid spurious mismatches
# (copied from SWE-rebench-V2/scripts/eval.py)
_TIMING_NORMALIZE_RES = [
    re.compile(r"\s*\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]\s*$", re.IGNORECASE),
    re.compile(r"\s+in\s+\d+(?:\.\d+)?\s+(?:msec|sec)\b", re.IGNORECASE),
    re.compile(r"\s*\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\)\s*$", re.IGNORECASE),
]


def _normalize_test_name(name: str) -> str:
    for pattern in _TIMING_NORMALIZE_RES:
        name = pattern.sub("", name)
    return name.strip()


def _run_in_docker(
    image: str,
    workdir: str,
    patch_dir: Path,
    test_cmds: list[str],
    *,
    timeout: int = 600,
) -> tuple[int, str]:
    """Run patches + tests in a container, uploading patches via docker cp."""
    import uuid

    container_name = f"swerebench-eval-{uuid.uuid4().hex[:8]}"
    # Start container
    subprocess.run(
        ["docker", "run", "-d", "--name", container_name, "--network", "host", "-w", workdir, image, "sleep", "2h"],
        check=True, capture_output=True, text=True, timeout=120,
    )
    try:
        # Upload patches
        subprocess.run(["docker", "exec", container_name, "mkdir", "-p", "/patches"], check=True, capture_output=True)
        subprocess.run(["docker", "cp", str(patch_dir / "patch.diff"), f"{container_name}:/patches/patch.diff"], check=True, capture_output=True, timeout=30)
        subprocess.run(["docker", "cp", str(patch_dir / "test_patch.diff"), f"{container_name}:/patches/test_patch.diff"], check=True, capture_output=True, timeout=30)

        # Run test script
        cmd_lines = [
            "set -e",
            "git reset --hard HEAD",
            "git apply -v --3way --recount --ignore-space-change --whitespace=nowarn /patches/patch.diff",
            "git apply -v --3way --recount --ignore-space-change --whitespace=nowarn /patches/test_patch.diff",
        ]
        cmd_lines.extend(test_cmds)
        script = "\n".join(cmd_lines)

        result = subprocess.run(
            ["docker", "exec", container_name, "/bin/bash", "-c", script],
            check=False, capture_output=True, text=True, timeout=timeout, errors="replace",
        )
        return result.returncode, (result.stdout or "") + (result.stderr or "")
    finally:
        subprocess.Popen(f"(timeout 60 docker stop {container_name} || docker rm -f {container_name}) >/dev/null 2>&1 &", shell=True)


def _parse_test_output(log: str, parser_name: str, image: str) -> dict[str, str]:
    """Use the log parser from the container's swebench_matterhorn to parse test output.

    We execute the parser inside the container to avoid needing to replicate
    all ~60 log parsers locally.
    """
    parse_script = f"""
import sys, json
sys.path.insert(0, '/swebench_matterhorn')
sys.path.insert(0, '/swebench_matterhorn/lib')
from agent.swe_log_parsers import NAME_TO_PARSER
parser = NAME_TO_PARSER["{parser_name}"]
log = sys.stdin.read()
result = parser(log)
print(json.dumps(result))
"""
    docker_cmd = [
        "docker", "run", "--rm", "-i",
        image,
        "python3", "-c", parse_script,
    ]
    result = subprocess.run(docker_cmd, input=log, capture_output=True, text=True, timeout=60, errors="replace")
    if result.returncode != 0:
        logger.warning(f"Log parser failed (rc={result.returncode}): {result.stderr[:200]}")
        return {}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        logger.warning(f"Log parser returned invalid JSON: {result.stdout[:200]}")
        return {}


def evaluate_instance(instance: dict, patch: str, *, logs_dir: Path | None = None) -> dict:
    """Evaluate a single instance following the SWE-rebench-V2 eval.py approach."""
    instance_id = instance["instance_id"]
    install_config = instance.get("install_config", {})
    image = instance["image_name"]
    repo = instance.get("repo", "")

    test_cmds = install_config.get("test_cmd", [])
    if isinstance(test_cmds, str):
        test_cmds = [test_cmds]

    parser_name = install_config.get("log_parser", "")
    test_patch = instance.get("test_patch", "")
    fail_to_pass = [_normalize_test_name(n) for n in instance.get("FAIL_TO_PASS", [])]
    pass_to_pass = [_normalize_test_name(n) for n in instance.get("PASS_TO_PASS", [])]

    workdir = f"/{repo.split('/')[1]}" if "/" in repo else "/"

    with tempfile.TemporaryDirectory(prefix="swerebench_eval_") as tmp:
        patch_dir = Path(tmp)
        (patch_dir / "patch.diff").write_text(patch or "")
        (patch_dir / "test_patch.diff").write_text(test_patch or "")

        logger.info(f"Running tests for {instance_id} in {image}")
        exit_code, output = _run_in_docker(image, workdir, patch_dir, test_cmds)

    if logs_dir:
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / f"{instance_id}_log.txt").write_text(output)

    # Parse test output using the log parser
    parsed = {}
    if parser_name:
        parsed = _parse_test_output(output, parser_name, image)
        parsed = {_normalize_test_name(k): v for k, v in parsed.items()}

    passed_actual = {k for k, v in parsed.items() if v == "PASSED"}

    f2p_passed = sorted(passed_actual & set(fail_to_pass))
    f2p_failed = sorted(set(fail_to_pass) - passed_actual)
    p2p_passed = sorted(passed_actual & set(pass_to_pass))
    p2p_failed = sorted(set(pass_to_pass) - passed_actual)

    resolved = len(fail_to_pass) > 0 and not f2p_failed and not p2p_failed

    return {
        "instance_id": instance_id,
        "resolved": resolved,
        "exit_code": exit_code,
        "fail_to_pass_passed": f2p_passed,
        "fail_to_pass_failed": f2p_failed,
        "pass_to_pass_passed": len(p2p_passed),
        "pass_to_pass_failed": p2p_failed,
        "parsed_tests_count": len(parsed),
        "error": "",
    }


# fmt: off
@app.command()
def main(
    preds_path: Path = typer.Argument(..., help="Path to preds.json from a swerebench rollout"),
    subset: str = typer.Option("rebench_v2", "--subset", help="SWE-rebench subset or dataset path"),
    split: str = typer.Option("train", "--split", help="Dataset split"),
    output: Path = typer.Option("verify_results.json", "-o", "--output", help="Output results file"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5')"),
    use_gold_patch: bool = typer.Option(False, "--use-gold-patch", help="Use golden patch from dataset instead of model patch"),
    logs_dir: Path = typer.Option("logs", "--logs-dir", help="Directory to save test logs"),
) -> None:
    # fmt: on
    """Verify patches from a swerebench rollout by running tests in Docker containers."""
    from datasets import load_dataset

    preds = json.loads(preds_path.read_text())
    console.print(f"Loaded {len(preds)} predictions from {preds_path}")

    dataset_path = DATASET_MAPPING.get(subset, subset)
    console.print(f"Loading dataset {dataset_path}, split {split}...")
    instances = {inst["instance_id"]: inst for inst in load_dataset(dataset_path, split=split)}

    instance_ids = list(preds.keys())
    if filter_spec:
        instance_ids = [iid for iid in instance_ids if re.match(filter_spec, iid)]
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instance_ids = instance_ids[slice(*values)]

    console.print(f"Verifying {len(instance_ids)} instances (gold_patch={use_gold_patch})...")

    results = {}
    resolved_count = 0
    for i, instance_id in enumerate(instance_ids):
        if instance_id not in instances:
            console.print(f"[yellow]Skipping {instance_id}: not found in dataset[/yellow]")
            continue

        if use_gold_patch:
            patch = instances[instance_id].get("patch", "")
        else:
            patch = preds[instance_id].get("model_patch", "")
        console.print(f"[{i+1}/{len(instance_ids)}] Verifying {instance_id}...")

        try:
            result = evaluate_instance(instances[instance_id], patch, logs_dir=logs_dir)
        except Exception as e:
            result = {"instance_id": instance_id, "resolved": False, "error": str(e),
                      "fail_to_pass_passed": [], "fail_to_pass_failed": [], "pass_to_pass_passed": 0, "pass_to_pass_failed": []}
            logger.error(f"Error verifying {instance_id}: {e}", exc_info=True)

        results[instance_id] = result

        status = "[green]RESOLVED[/green]" if result["resolved"] else "[red]FAILED[/red]"
        f2p_total = len(result.get("fail_to_pass_passed", [])) + len(result.get("fail_to_pass_failed", []))
        p2p_total = result.get("pass_to_pass_passed", 0) + len(result.get("pass_to_pass_failed", []))
        detail = f"f2p={len(result.get('fail_to_pass_passed', []))}/{f2p_total} p2p={result.get('pass_to_pass_passed', 0)}/{p2p_total}"
        if result.get("error"):
            detail = result["error"][:80]
        console.print(f"  {status} ({detail})")

        if result["resolved"]:
            resolved_count += 1

    output.write_text(json.dumps(results, indent=2))
    console.print(f"\nResults saved to {output}")
    console.print(f"Resolved: {resolved_count}/{len(results)} ({resolved_count/max(len(results),1)*100:.1f}%)")


if __name__ == "__main__":
    app()
