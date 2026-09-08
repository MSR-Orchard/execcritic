#!/usr/bin/env python3

"""Verify patches for SWE-bench instances using Azure Modal sandbox.

Reads patches from trajectory files in an output directory, then evaluates
each patch by running tests inside an Azure Modal sandbox.

Self-contained: no dependency on the ``swebench`` package.  Log parsers and
eval-script generation logic are inlined.
"""

import base64
import concurrent.futures
import json
import os
import re
import shlex
import threading
import time
import traceback
from pathlib import Path

import typer
from rich.console import Console
from rich.live import Live

from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.utils.log import add_file_handler, logger
from minisweagent.utils.serialize import UNSET, recursive_merge

console = Console(highlight=False)
app = typer.Typer(rich_markup_mode="rich", add_completion=False)
_OUTPUT_FILE_LOCK = threading.Lock()
_RESULTS_CACHE: dict | None = None
_RESULTS_DIRTY = False

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
    "smith": "SWE-bench/SWE-smith",
    "_test": "klieret/swe-bench-dummy-test-dataset",
    "pro": "ScaleAI/SWE-bench_Pro",
}

DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "swebench_azure_modal.yaml"

# From swebench.harness.constants
START_TEST_OUTPUT = ">>>>> Start Test Output"
END_TEST_OUTPUT = ">>>>> End Test Output"
APPLY_PATCH_FAIL = ">>>>> Patch Apply Failed"
RESET_FAILED = ">>>>> Reset Failed"
TESTS_ERROR = ">>>>> Tests Errored"
TESTS_TIMEOUT = ">>>>> Tests Timed Out"
DOCKER_WORKDIR = "/testbed"
FAIL_ONLY_REPOS = {"chartjs/Chart.js", "processing/p5.js", "markedjs/marked"}


# ---------------------------------------------------------------------------
# Inlined log parsers (from swebench.harness.log_parsers, Python repos only)
# ---------------------------------------------------------------------------

def _parse_log_pytest(log: str) -> dict[str, str]:
    sm = {}
    for line in log.split("\n"):
        if any(line.startswith(s) for s in ["PASSED", "FAILED", "ERROR", "SKIPPED", "XFAIL"]):
            if line.startswith("FAILED"):
                line = line.replace(" - ", " ")
            parts = line.split()
            if len(parts) >= 2:
                sm[parts[1]] = parts[0]
    return sm


def _parse_log_pytest_options(log: str) -> dict[str, str]:
    opt_re = re.compile(r"(.*?)\[(.*)\]")
    sm = {}
    for line in log.split("\n"):
        if any(line.startswith(s) for s in ["PASSED", "FAILED", "ERROR", "SKIPPED", "XFAIL"]):
            if line.startswith("FAILED"):
                line = line.replace(" - ", " ")
            parts = line.split()
            if len(parts) >= 2:
                m = opt_re.search(parts[1])
                if m:
                    main, opt = m.groups()
                    if opt.startswith("/") and not opt.startswith("//") and "*" not in opt:
                        opt = "/" + opt.split("/")[-1]
                    name = f"{main}[{opt}]"
                else:
                    name = parts[1]
                sm[name] = parts[0]
    return sm


def _parse_log_pytest_v2(log: str) -> dict[str, str]:
    sm = {}
    escapes = "".join(chr(c) for c in range(1, 32))
    for line in log.split("\n"):
        line = re.sub(r"\[(\d+)m", "", line).translate(str.maketrans("", "", escapes))
        if any(line.startswith(s) for s in ["PASSED", "FAILED", "ERROR", "SKIPPED", "XFAIL"]):
            if line.startswith("FAILED"):
                line = line.replace(" - ", " ")
            parts = line.split()
            if len(parts) >= 2:
                sm[parts[1]] = parts[0]
        elif any(line.endswith(s) for s in ["PASSED", "FAILED", "ERROR", "SKIPPED", "XFAIL"]):
            parts = line.split()
            if len(parts) >= 2:
                sm[parts[0]] = parts[1]
    return sm


def _parse_log_django(log: str) -> dict[str, str]:
    sm = {}
    prev_test = None
    for line in log.split("\n"):
        line = line.strip()
        if "--version is equivalent to version" in line:
            sm["--version is equivalent to version"] = "PASSED"
        if " ... " in line:
            prev_test = line.split(" ... ")[0]
        for suffix in (" ... ok", " ... OK", " ...  OK"):
            if line.endswith(suffix):
                sm[line.rsplit(suffix, 1)[0]] = "PASSED"
                break
        if " ... skipped" in line:
            sm[line.split(" ... skipped")[0]] = "SKIPPED"
        if line.endswith(" ... FAIL"):
            sm[line.split(" ... FAIL")[0]] = "FAILED"
        if line.startswith("FAIL:"):
            sm[line.split()[1].strip()] = "FAILED"
        if line.endswith(" ... ERROR"):
            sm[line.split(" ... ERROR")[0]] = "ERROR"
        if line.startswith("ERROR:"):
            sm[line.split()[1].strip()] = "ERROR"
        if line.lstrip().startswith("ok") and prev_test is not None:
            sm[prev_test] = "PASSED"
    return sm


def _parse_log_seaborn(log: str) -> dict[str, str]:
    sm = {}
    for line in log.split("\n"):
        if line.startswith("FAILED"):
            parts = line.split()
            if len(parts) >= 2:
                sm[parts[1]] = "FAILED"
        elif " PASSED " in line:
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "PASSED":
                sm[parts[0]] = "PASSED"
        elif line.startswith("PASSED"):
            parts = line.split()
            if len(parts) >= 2:
                sm[parts[1]] = "PASSED"
    return sm


def _parse_log_sympy(log: str) -> dict[str, str]:
    sm = {}
    for match in re.findall(r"(_*) (.*)\.py:(.*) (_*)", log):
        sm[f"{match[1]}.py:{match[2]}"] = "FAILED"
    for line in log.split("\n"):
        line = line.strip()
        if line.startswith("test_"):
            if line.endswith(" E"):
                sm[line.split()[0]] = "ERROR"
            elif line.endswith(" F"):
                sm[line.split()[0]] = "FAILED"
            elif line.endswith(" ok"):
                sm[line.split()[0]] = "PASSED"
    return sm


def _parse_log_matplotlib(log: str) -> dict[str, str]:
    log = log.replace("MouseButton.LEFT", "1").replace("MouseButton.RIGHT", "3")
    return _parse_log_pytest(log)


# Repo -> log parser mapping (all SWE-bench Python repos)
_REPO_TO_PARSER: dict[str, object] = {
    "astropy/astropy": _parse_log_pytest_v2,
    "django/django": _parse_log_django,
    "marshmallow-code/marshmallow": _parse_log_pytest,
    "matplotlib/matplotlib": _parse_log_matplotlib,
    "mwaskom/seaborn": _parse_log_seaborn,
    "pallets/flask": _parse_log_pytest,
    "psf/requests": _parse_log_pytest_options,
    "pvlib/pvlib-python": _parse_log_pytest,
    "pydata/xarray": _parse_log_pytest,
    "pydicom/pydicom": _parse_log_pytest_options,
    "pylint-dev/astroid": _parse_log_pytest,
    "pylint-dev/pylint": _parse_log_pytest_options,
    "pytest-dev/pytest": _parse_log_pytest,
    "pyvista/pyvista": _parse_log_pytest,
    "scikit-learn/scikit-learn": _parse_log_pytest_v2,
    "sqlfluff/sqlfluff": _parse_log_pytest,
    "sphinx-doc/sphinx": _parse_log_pytest_v2,
    "sympy/sympy": _parse_log_sympy,
}

# ---------------------------------------------------------------------------
# Eval-script building helpers (self-contained, no swebench dependency)
# ---------------------------------------------------------------------------

NON_TEST_EXTS = [
    ".json", ".png", "csv", ".txt", ".md", ".jpg", ".jpeg",
    ".pkl", ".yml", ".yaml", ".toml",
]

_DEFAULT_PYTEST_CMD = (
    "PYTEST_NO_HEADER=''; "
    "python -m pytest --help 2>/dev/null | grep -q -- '--no-header' && PYTEST_NO_HEADER=--no-header; "
    "python -m pytest $PYTEST_NO_HEADER -rA --tb=short -p no:cacheprovider"
)

# Repos with a non-default test command
_REPO_TEST_CMD: dict[str, str] = {
    "django/django": "./tests/runtests.py --verbosity 2",
    "sympy/sympy": "bin/test -C --verbose",
}

def _get_modified_files(patch: str) -> list[str]:
    """Files modified (not newly created) — used to reset test files."""
    return [line[6:] for line in patch.split("\n") if line.startswith("--- a/")]


def _get_test_directives(instance: dict) -> list[str]:
    """Extract test file paths from test_patch (mirrors swebench.harness.test_spec)."""
    repo = instance.get("repo", "")
    test_patch = instance.get("test_patch", "")

    if repo == "swe-bench/humaneval":
        return ["test.py"]

    directives = re.findall(r"diff --git a/.* b/(.*)", test_patch)
    directives = [d for d in directives if not any(d.endswith(ext) for ext in NON_TEST_EXTS)]

    if repo == "django/django":
        transformed = []
        for d in directives:
            d = d[:-len(".py")] if d.endswith(".py") else d
            d = d[len("tests/"):] if d.startswith("tests/") else d
            d = d.replace("/", ".")
            transformed.append(d)
        directives = transformed

    return directives


def _selected_test_directives(instance: dict) -> list[str]:
    """Return exact in-loop selectors when the dataset node format supports them."""
    nodes = [str(node) for node in instance.get("_swebench_selected_nodes", []) if str(node).strip()]
    if not nodes:
        return []

    repo = instance.get("repo", "")
    if repo == "django/django":
        labels = []
        for node in nodes:
            match = re.fullmatch(r"([^ ]+) \(([^)]+)\)", node)
            if not match:
                return []
            method, test_case = match.groups()
            labels.append(f"{test_case}.{method}")
        return labels
    if repo == "sympy/sympy":
        # SymPy F2P labels are usually bare function names. Its runner needs the
        # file directive from test_patch, so keep the canonical file-level run.
        return []
    if all("::" in node for node in nodes):
        return nodes
    return []


def _build_test_command(instance: dict) -> str:
    repo = instance["repo"]
    test_cmd = _REPO_TEST_CMD.get(repo, _DEFAULT_PYTEST_CMD)
    directives = _selected_test_directives(instance) or _get_test_directives(instance)
    return " ".join([test_cmd, *(shlex.quote(directive) for directive in directives)])


def _build_file_fallback_command(instance: dict) -> str:
    test_cmd = _REPO_TEST_CMD.get(instance["repo"], _DEFAULT_PYTEST_CMD)
    return " ".join([
        test_cmd,
        *(shlex.quote(directive) for directive in _get_test_directives(instance)),
    ])


def _test_runner(instance: dict) -> str:
    repo = instance.get("repo", "")
    if repo == "django/django":
        return "django runtests.py"
    if repo == "sympy/sympy":
        return "sympy bin/test"
    return "pytest"


# ---------------------------------------------------------------------------
# File upload utilities (mirrors swerebench_verify_azure_modal.py)
# ---------------------------------------------------------------------------

_B64WRITE_CHUNK_SIZE = 48000


def _b64write_cmd(path: str, content: str) -> str:
    b64 = base64.b64encode(content.encode()).decode()
    return f"python3 -c \"import base64; open({path!r},'wb').write(base64.b64decode({b64!r}))\""


def _upload_file(env, path: str, content: str, *, timeout: int = 30) -> None:
    raw = content.encode()
    if len(raw) <= _B64WRITE_CHUNK_SIZE:
        env.execute({"command": _b64write_cmd(path, content)}, timeout=timeout)
    else:
        env.execute({"command": f"rm -f {path}"}, timeout=10)
        for offset in range(0, len(raw), _B64WRITE_CHUNK_SIZE):
            chunk = raw[offset:offset + _B64WRITE_CHUNK_SIZE]
            b64 = base64.b64encode(chunk).decode()
            cmd = f"python3 -c \"import base64; open({path!r},'ab').write(base64.b64decode({b64!r}))\""
            env.execute({"command": cmd}, timeout=timeout)


# ---------------------------------------------------------------------------
# Docker image + eval script
# ---------------------------------------------------------------------------

def _get_swebench_docker_image(instance: dict) -> str:
    image_name = instance.get("image_name", None) or instance.get("docker_image", None)
    if image_name is None:
        iid = instance["instance_id"]
        id_docker_compatible = iid.replace("__", "_1776_")
        image_name = f"swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    if image_name.startswith("docker.io/"):
        image_name = image_name.replace("docker.io/", "mirror.gcr.io/")
    elif not image_name.startswith("mirror.gcr.io/"):
        image_name = "mirror.gcr.io/" + image_name
    return image_name


def _build_eval_script(instance: dict) -> str:
    """Build eval script for a SWE-bench instance (no swebench dependency).

    The test_patch is uploaded separately as /tmp/test_patch.diff.
    """
    base_commit = instance.get("_persistent_baseline_commit") or instance["base_commit"]
    test_patch = instance["test_patch"]
    test_command = _build_test_command(instance)
    file_fallback_command = _build_file_fallback_command(instance)
    has_focused_selector = bool(_selected_test_directives(instance))

    test_files = _get_modified_files(test_patch)
    reset_cmd = f"git checkout {base_commit} {' '.join(test_files)}" if test_files else 'echo "No test files to reset"'

    lines = [
        "#!/bin/bash",
        "set -o pipefail",
        "source /opt/miniconda3/bin/activate",
        "conda activate testbed",
        f"cd {DOCKER_WORKDIR}",
        f"git config --global --add safe.directory {DOCKER_WORKDIR}",
        f"cd {DOCKER_WORKDIR}",
        "git status",
        "git show",
        f"git -c core.fileMode=false diff {base_commit}",
        "source /opt/miniconda3/bin/activate",
        "conda activate testbed",
        reset_cmd,
        "git apply -v /tmp/test_patch.diff",
        f"echo '{START_TEST_OUTPUT}'",
    ]
    if has_focused_selector and file_fallback_command != test_command:
        lines.extend([
            f"({test_command}) > /tmp/swebench_focused_test.out 2>&1",
            "test_rc=$?",
            (
                "if [ \"$test_rc\" -eq 4 ] && "
                "grep -Eqi 'not found:|no tests ran|collected 0 items|file or directory not found' "
                "/tmp/swebench_focused_test.out; then"
            ),
            "  echo 'FOCUSED_SELECTOR_FALLBACK_TO_FILE'",
            f"  {file_fallback_command}",
            "  test_rc=$?",
            "else",
            "  cat /tmp/swebench_focused_test.out",
            "fi",
        ])
    else:
        lines.extend([test_command, "test_rc=$?"])
    lines.extend([
        f"echo '{END_TEST_OUTPUT}'",
        reset_cmd,
        "reset_rc=$?",
        f"if [ \"$reset_rc\" -ne 0 ]; then echo '{RESET_FAILED}'; exit \"$reset_rc\"; fi",
        "exit \"$test_rc\"",
    ])
    return "\n".join(lines) + "\n"


def _parse_test_output(log: str, instance: dict) -> dict[str, str]:
    """Parse test output using inlined log parsers."""
    repo = instance["repo"]
    parser = _REPO_TO_PARSER.get(repo)
    if parser is None:
        logger.warning(f"No log parser for repo {repo!r}, falling back to pytest_v2")
        parser = _parse_log_pytest_v2

    if START_TEST_OUTPUT in log and END_TEST_OUTPUT in log:
        test_content = log.split(START_TEST_OUTPUT)[1].split(END_TEST_OUTPUT)[0]
    else:
        test_content = log

    status_map = parser(test_content)
    if not status_map:
        status_map = parser(log)
    return status_map


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _from_json_or_obj(val) -> list[str]:
    if val is None:
        return []
    if isinstance(val, str):
        return json.loads(val)
    return list(val)


def create_environment(instance: dict, env_config: dict):
    """Create, but do not evaluate or clean up, one official verifier sandbox."""
    from minisweagent.environments.extra.azure_modal import AzureModalEnvironment

    image = _get_swebench_docker_image(instance)
    cfg = {k: v for k, v in env_config.items() if k not in ("environment_class",)}
    cfg.update(image=image, cwd=DOCKER_WORKDIR)
    return AzureModalEnvironment(**cfg)


def grade_eval_output(instance: dict, output: str, *, exit_code: int, patch_applied: bool) -> dict:
    """Grade official SWE-bench output independently of sandbox ownership."""
    instance_id = instance["instance_id"]
    repo = instance.get("repo", "")

    fail_to_pass = _from_json_or_obj(instance.get("FAIL_TO_PASS"))
    pass_to_pass = _from_json_or_obj(instance.get("PASS_TO_PASS"))
    fail_only = repo in FAIL_ONLY_REPOS

    # Check for bad-code signals (mirrors get_logs_eval in grading.py)
    bad_codes = [c for c in [APPLY_PATCH_FAIL, RESET_FAILED, TESTS_ERROR, TESTS_TIMEOUT] if c in output]
    if bad_codes or (START_TEST_OUTPUT not in output and END_TEST_OUTPUT not in output):
        if bad_codes:
            logger.warning(f"{instance_id}: bad signals in output: {bad_codes}")
        else:
            logger.warning(f"{instance_id}: START/END test markers not found in output")
        parsed: dict[str, str] = {}
    else:
        parsed = _parse_test_output(output, instance)

    passed = {k for k, v in parsed.items() if v in ("PASSED", "XFAIL")}

    def _check(case: str) -> bool:
        if fail_only:
            return case not in parsed or parsed.get(case) != "FAILED"
        return case in parsed and parsed[case] in ("PASSED", "XFAIL")

    f2p_passed = sorted(t for t in fail_to_pass if _check(t))
    f2p_failed = sorted(t for t in fail_to_pass if not _check(t))
    p2p_passed = sorted(passed & set(pass_to_pass))
    p2p_failed = sorted(set(pass_to_pass) - passed)

    if fail_only:
        resolved = len(fail_to_pass) > 0 and not f2p_failed
    else:
        resolved = len(fail_to_pass) > 0 and not f2p_failed and not p2p_failed

    return {
        "instance_id": instance_id,
        "resolved": resolved,
        "exit_code": exit_code,
        "patch_applied": patch_applied,
        "fail_to_pass_passed": f2p_passed,
        "fail_to_pass_failed": f2p_failed,
        "pass_to_pass_passed": len(p2p_passed),
        "pass_to_pass_failed": p2p_failed,
        "parsed_tests_count": len(parsed),
        "error": "",
    }


def evaluate_instance_in_environment(
    instance: dict,
    patch: str,
    env,
    env_config: dict,
    *,
    logs_dir: Path | None = None,
    sample_idx: int = 0,
) -> dict:
    """Evaluate in a caller-owned sandbox, resetting it to baseline first."""
    instance_id = instance["instance_id"]
    base_commit = instance.get("_persistent_baseline_commit")
    if base_commit:
        reset = env.execute(
            {
                "command": (
                    f"cd {DOCKER_WORKDIR} && git reset --hard {shlex.quote(str(base_commit))} "
                    "&& git clean -fdq"
                )
            },
            timeout=120,
        )
        if reset["returncode"] != 0:
            return {
                "instance_id": instance_id,
                "resolved": False,
                "exit_code": reset["returncode"],
                "patch_applied": False,
                "fail_to_pass_passed": [],
                "fail_to_pass_failed": _from_json_or_obj(instance.get("FAIL_TO_PASS")),
                "pass_to_pass_passed": 0,
                "pass_to_pass_failed": _from_json_or_obj(instance.get("PASS_TO_PASS")),
                "parsed_tests_count": 0,
                "error": f"baseline reset failed: {str(reset.get('output') or '')[-1000:]}",
            }

    test_command = _build_test_command(instance)
    file_fallback_command = _build_file_fallback_command(instance)
    eval_script = _build_eval_script(instance)
    test_timeout = env_config.get("sandbox_timeout", 600)
    _upload_file(env, "/tmp/patch.diff", patch or "", timeout=30)
    _upload_file(env, "/tmp/test_patch.diff", instance.get("test_patch", ""), timeout=60)
    _upload_file(env, "/tmp/eval.sh", eval_script, timeout=60)

    patch_applied = not bool((patch or "").strip())
    if not patch_applied:
        for git_apply_cmd in [
            "git apply --verbose /tmp/patch.diff",
            "git apply --verbose --reject /tmp/patch.diff",
            "patch --batch --fuzz=5 -p1 -i /tmp/patch.diff",
        ]:
            applied = env.execute(
                {"command": f"cd {DOCKER_WORKDIR} && {git_apply_cmd}"}, timeout=60
            )
            if applied["returncode"] == 0:
                patch_applied = True
                break
    if not patch_applied:
        logger.warning("%s: model patch failed to apply", instance_id)

    result = env.execute({"command": "bash /tmp/eval.sh"}, timeout=test_timeout)
    output = result["output"]
    if logs_dir:
        logs_dir.mkdir(parents=True, exist_ok=True)
        suffix = f".sample_{sample_idx}" if sample_idx > 0 else ""
        (logs_dir / f"{instance_id}{suffix}_log.txt").write_text(output)
    grade = grade_eval_output(
        instance,
        output,
        exit_code=result["returncode"],
        patch_applied=patch_applied,
    )
    return {
        **grade,
        "test_command": (
            f"{test_command} [fallback_on_selector_miss: {file_fallback_command}]"
            if _selected_test_directives(instance) and file_fallback_command != test_command
            else test_command
        ),
        "test_files": _get_modified_files(instance.get("test_patch", "")),
        "test_runner": _test_runner(instance),
        "focused_nodes": list(instance.get("_swebench_selected_nodes", [])),
    }


def evaluate_instance_azure_modal(
    instance: dict, patch: str, env_config: dict, *, logs_dir: Path | None = None, sample_idx: int = 0,
) -> dict:
    """Evaluate a single SWE-bench instance inside a fresh Azure sandbox."""
    env = create_environment(instance, env_config)
    try:
        return evaluate_instance_in_environment(
            instance,
            patch,
            env,
            env_config,
            logs_dir=logs_dir,
            sample_idx=sample_idx,
        )
    finally:
        env.cleanup()


# ---------------------------------------------------------------------------
# Trajectory scanning and I/O helpers
# ---------------------------------------------------------------------------

def load_patch_from_traj(traj_path: Path) -> str:
    data = json.loads(traj_path.read_text())
    patch = data.get("info", {}).get("submission", "")
    if patch:
        return patch
    # OpenHands format
    return data.get("metadata", {}).get("model_patch", "") or ""


def _parse_traj_filename(name: str, suffix: str) -> tuple[str, int]:
    name = name.removesuffix(suffix)
    if ".sample_" in name:
        instance_id, _, idx = name.rpartition(".sample_")
        return instance_id, int(idx)
    return name, 0


def scan_trajectories(output_dir: Path, traj_format: str = "auto") -> dict[tuple[str, int], Path]:
    """Return {(instance_id, sample_idx): traj_path} for all trajectory files."""
    trajs: dict[tuple[str, int], Path] = {}
    if traj_format in ("auto", "native"):
        for traj_file in sorted(output_dir.rglob("*.traj.json")):
            instance_id, sample_idx = _parse_traj_filename(traj_file.stem, ".traj")
            trajs[(instance_id, sample_idx)] = traj_file
    if traj_format in ("auto", "openhands"):
        for traj_file in sorted(output_dir.rglob("*.jsonl")):
            try:
                data = json.loads(traj_file.read_text())
                instance_id = data["instance_id"]
                sample_idx = int(data.get("sample_idx", 0))
            except (json.JSONDecodeError, KeyError, ValueError):
                logger.warning(f"Skipping unrecognised .jsonl file: {traj_file}")
                continue
            trajs[(instance_id, sample_idx)] = traj_file
    return trajs


def get_existing_verified_samples(results_path: Path) -> set[tuple[str, int]]:
    if not results_path.exists():
        return set()
    data = json.loads(results_path.read_text())
    existing = set()
    for instance_id, entry in data.items():
        for s in entry.get("samples", []):
            existing.add((instance_id, s["sample_idx"]))
        if "samples" not in entry:
            existing.add((instance_id, 0))
    return existing


def update_results_file(output_path: Path, instance_id: str, sample_idx: int, result: dict):
    global _RESULTS_CACHE, _RESULTS_DIRTY
    with _OUTPUT_FILE_LOCK:
        if _RESULTS_CACHE is None:
            _RESULTS_CACHE = json.loads(output_path.read_text()) if output_path.exists() and output_path.stat().st_size > 0 else {}
        entry = _RESULTS_CACHE.get(instance_id, {"instance_id": instance_id, "samples": []})
        samples = [s for s in entry.get("samples", []) if s.get("sample_idx") != sample_idx]
        result["sample_idx"] = sample_idx
        samples.append(result)
        samples.sort(key=lambda s: s["sample_idx"])
        entry["samples"] = samples
        entry["resolved"] = any(s.get("resolved") for s in samples)
        _RESULTS_CACHE[instance_id] = entry
        _RESULTS_DIRTY = True


def flush_results_file(output_path: Path):
    global _RESULTS_DIRTY
    with _OUTPUT_FILE_LOCK:
        if _RESULTS_CACHE is not None and _RESULTS_DIRTY:
            output_path.write_text(json.dumps(_RESULTS_CACHE, indent=2))
            _RESULTS_DIRTY = False


# ---------------------------------------------------------------------------
# Instance processing
# ---------------------------------------------------------------------------

def process_instance(
    instance: dict,
    traj_path: Path | None,
    results_path: Path,
    env_config: dict,
    logs_dir: Path,
    progress_manager: RunBatchProgressManager,
    sample_idx: int = 0,
    golden_patch: str | None = None,
    evaluator=None,
) -> None:
    instance_id = instance["instance_id"]
    task_label = f"{instance_id}#{sample_idx}" if sample_idx > 0 else instance_id
    progress_manager.on_instance_start(task_label)
    progress_manager.update_instance_status(task_label, "Starting sandbox")
    result: dict = {}
    exit_status = "error"
    try:
        patch = golden_patch if golden_patch is not None else load_patch_from_traj(traj_path)
        progress_manager.update_instance_status(task_label, "Running tests")
        evaluate = evaluator or evaluate_instance_azure_modal
        result = evaluate(instance, patch, env_config, logs_dir=logs_dir, sample_idx=sample_idx)
        exit_status = "resolved" if result["resolved"] else "failed"
    except Exception as e:
        logger.error(f"Error verifying {task_label}: {e}", exc_info=True)
        result = {
            "instance_id": instance_id,
            "resolved": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "fail_to_pass_passed": [],
            "fail_to_pass_failed": [],
            "pass_to_pass_passed": 0,
            "pass_to_pass_failed": [],
        }
    progress_manager.on_instance_end(task_label, exit_status)
    update_results_file(results_path, instance_id, sample_idx, result)


def _print_verification_report(results: dict, results_path: Path) -> None:
    total_instances = len(results)
    resolved_instances = sum(1 for r in results.values() if r.get("resolved"))
    total_samples = sum(len(r.get("samples", [])) for r in results.values())
    resolved_samples = sum(
        sum(1 for s in r.get("samples", []) if s.get("resolved"))
        for r in results.values()
    )

    console.print(f"\n[bold]{'=' * 80}[/bold]")
    console.print(f"[bold cyan]VERIFICATION REPORT[/bold cyan]".center(80))
    console.print(f"[bold]{'=' * 80}[/bold]")
    console.print(f"\nResults file: [bold green]{results_path}[/bold green]")
    console.print(f"\n[bold]Pass@k (instance-level):[/bold]")
    console.print(f"  Resolved: {resolved_instances}/{total_instances} ({resolved_instances / max(total_instances, 1) * 100:.1f}%)")
    console.print(f"\n[bold]Per-sample:[/bold]")
    console.print(f"  Resolved: {resolved_samples}/{total_samples} ({resolved_samples / max(total_samples, 1) * 100:.1f}%)")

    resolved_ids = [iid for iid, r in results.items() if r.get("resolved")]
    failed_ids = [iid for iid, r in results.items() if not r.get("resolved")]

    if resolved_ids:
        console.print(f"\n[bold green]Resolved instances ({len(resolved_ids)}):[/bold green]")
        for iid in sorted(resolved_ids)[:20]:
            console.print(f"  ✓ {iid}")
        if len(resolved_ids) > 20:
            console.print(f"  ... and {len(resolved_ids) - 20} more")

    if failed_ids:
        console.print(f"\n[bold red]Failed instances ({len(failed_ids)}):[/bold red]")
        for iid in sorted(failed_ids)[:20]:
            console.print(f"  ✗ {iid}")
        if len(failed_ids) > 20:
            console.print(f"  ... and {len(failed_ids) - 20} more")

    console.print(f"\n[bold]{'=' * 80}[/bold]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# fmt: off
@app.command()
def main(
    subset: str = typer.Option("lite", "--subset", help="SWE-bench subset (including 'pro') or dataset path", rich_help_panel="Data selection"),
    split: str = typer.Option("dev", "--split", help="Dataset split", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5')", rich_help_panel="Data selection"),
    output: str = typer.Option(..., "-o", "--output", help="Directory containing trajectory files to verify", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of parallel workers", rich_help_panel="Basic"),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config", help="Config files/specs for azure_modal environment", rich_help_panel="Basic"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Re-verify instances already in results file", rich_help_panel="Data selection"),
    logs_dir: str = typer.Option("", "--logs-dir", help="Directory to save test logs (default: <output>/verify_logs)", rich_help_panel="Advanced"),
    verify_output: str = typer.Option("", "--verify-output", help="Output results file (default: <output>/verify_results_azure_modal.json)", rich_help_panel="Basic"),
    golden: bool = typer.Option(False, "--golden", help="Sanity check: evaluate golden patches from the dataset instead of model patches", rich_help_panel="Advanced"),
    report_only: bool = typer.Option(False, "--report-only", help="Only print report from existing results file without running verification", rich_help_panel="Advanced"),
    traj_format: str = typer.Option("auto", "--traj-format", help="Trajectory format: 'native' (*.traj.json), 'openhands' (*.jsonl), or 'auto' (both)", rich_help_panel="Data selection"),
    swebench_pro_harness_root: str = typer.Option("", "--swebench-pro-harness-root", help="SWE-bench_Pro-os checkout containing run_scripts (Pro only)", rich_help_panel="Data selection"),
) -> None:
    # fmt: on
    """Verify patches from trajectory files using Azure Modal sandboxes."""
    output_path = Path(output)
    dataset_path = DATASET_MAPPING.get(subset, subset)
    is_swebench_pro = dataset_path == DATASET_MAPPING["pro"]
    if golden:
        result_name = "verify_results_swebench_pro_golden.json" if is_swebench_pro else "verify_results_golden.json"
        logs_name = "verify_logs_swebench_pro_golden" if is_swebench_pro else "verify_logs_golden"
    else:
        result_name = (
            "verify_results_swebench_pro_azure_modal.json"
            if is_swebench_pro
            else "verify_results_azure_modal.json"
        )
        logs_name = "verify_logs_swebench_pro" if is_swebench_pro else "verify_logs"
    default_results = output_path / result_name
    default_logs = output_path / logs_name
    results_path = Path(verify_output) if verify_output else default_results
    logs_path = Path(logs_dir) if logs_dir else default_logs

    # Report-only mode
    if report_only:
        if not results_path.exists():
            console.print(f"[red]Results file not found: {results_path}[/red]")
            raise typer.Exit(1)
        results = json.loads(results_path.read_text())
        if filter_spec or slice_spec:
            ids = sorted(results.keys())
            if filter_spec:
                ids = [iid for iid in ids if re.match(filter_spec, iid)]
            if slice_spec:
                values = [int(x) if x else None for x in slice_spec.split(":")]
                ids = ids[slice(*values)]
            results = {iid: results[iid] for iid in ids}
        _print_verification_report(results, results_path)
        return

    add_file_handler(output_path / "minisweagent_verify.log")

    from datasets import load_dataset

    logger.info(f"Loading dataset {dataset_path}, split {split}...")
    raw_instances = list(load_dataset(dataset_path, split=split))
    instance_evaluator = evaluate_instance_azure_modal
    if is_swebench_pro:
        from minisweagent.run.benchmarks import swebench_pro_verify_azure_modal

        if swebench_pro_harness_root:
            os.environ["SWEPRO_HARNESS_ROOT"] = str(Path(swebench_pro_harness_root).expanduser())
        try:
            harness_root = swebench_pro_verify_azure_modal.validate_harness_root()
        except FileNotFoundError as exc:
            raise typer.BadParameter(str(exc), param_hint="--swebench-pro-harness-root") from exc
        logger.info(f"Using SWE-bench Pro harness checkout {harness_root}")
        raw_instances = [swebench_pro_verify_azure_modal.prepare_instance(inst) for inst in raw_instances]
        instance_evaluator = swebench_pro_verify_azure_modal.evaluate_instance_azure_modal
    instances = {inst["instance_id"]: inst for inst in raw_instances}

    if golden:
        instance_ids = sorted(instances.keys())
    else:
        logger.info(f"Scanning trajectories in {output_path}")
        traj_map = scan_trajectories(output_path, traj_format=traj_format)
        if not traj_map:
            console.print(f"[yellow]No trajectory files found in {output_path}[/yellow]")
            raise typer.Exit(1)
        logger.info(f"Found {len(traj_map)} trajectory files")

        traj_instance_ids = sorted({iid for iid, _ in traj_map})
        instance_ids = [iid for iid in traj_instance_ids if iid in instances]
        missing = [iid for iid in traj_instance_ids if iid not in instances]
        if missing:
            logger.warning(f"{len(missing)} trajectory instance_ids not found in dataset: {missing[:5]}...")

    if filter_spec:
        instance_ids = [iid for iid in instance_ids if re.match(filter_spec, iid)]
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instance_ids = instance_ids[slice(*values)]

    if golden:
        work_items_golden = [(iid, 0) for iid in instance_ids]
        if not redo_existing:
            existing = get_existing_verified_samples(results_path)
            before = len(work_items_golden)
            work_items_golden = [(iid, sidx) for iid, sidx in work_items_golden if (iid, sidx) not in existing]
            if before > len(work_items_golden):
                logger.info(f"Skipping {before - len(work_items_golden)} already-verified instances")
        logger.info(f"[golden] Verifying {len(work_items_golden)} instances with golden patches...")
    else:
        instance_id_set = set(instance_ids)
        work_items = [(iid, sidx, traj_map[(iid, sidx)]) for iid, sidx in sorted(traj_map) if iid in instance_id_set]
        if not redo_existing:
            existing = get_existing_verified_samples(results_path)
            before = len(work_items)
            work_items = [(iid, sidx, tp) for iid, sidx, tp in work_items if (iid, sidx) not in existing]
            if before > len(work_items):
                logger.info(f"Skipping {before - len(work_items)} already-verified samples")
        logger.info(f"Verifying {len(work_items)} samples across {len({iid for iid, _, _ in work_items})} instances...")

    configs = [get_config_from_spec(spec) for spec in config_spec]
    config = recursive_merge(*configs)
    env_config = config.get("environment", {})

    n_jobs = len(work_items_golden) if golden else len(work_items)
    progress_manager = RunBatchProgressManager(
        n_jobs, output_path / f"verify_statuses_{time.time()}.yaml",
    )

    if workers <= 1:
        live_ctx = Live(progress_manager.render_group, refresh_per_second=4)
    else:
        progress_manager._main_progress_bar.live.redirect_stdout = False
        progress_manager._main_progress_bar.live.redirect_stderr = False
        live_ctx = progress_manager._main_progress_bar

    with live_ctx:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        futures = {}
        if golden:
            for iid, s_idx in work_items_golden:
                future = executor.submit(
                    process_instance,
                    instances[iid], None, results_path, env_config, logs_path, progress_manager, s_idx,
                    golden_patch=instances[iid].get("patch", ""),
                    evaluator=instance_evaluator,
                )
                futures[future] = iid
        else:
            for iid, s_idx, traj_path in work_items:
                task_label = f"{iid}#{s_idx}" if s_idx > 0 else iid
                future = executor.submit(
                    process_instance,
                    instances[iid], traj_path, results_path, env_config, logs_path, progress_manager, s_idx,
                    evaluator=instance_evaluator,
                )
                futures[future] = task_label
        last_flush = time.time()
        try:
            done: set = set()
            while len(done) < len(futures):
                try:
                    for future in concurrent.futures.as_completed(futures, timeout=2.0):
                        done.add(future)
                        try:
                            future.result()
                        except concurrent.futures.CancelledError:
                            pass
                        except Exception as e:
                            task_label = futures[future]
                            logger.error(f"Uncaught error for {task_label}: {e}", exc_info=True)
                            progress_manager.on_uncaught_exception(task_label, e)
                except concurrent.futures.TimeoutError:
                    pass
                if time.time() - last_flush > 30:
                    flush_results_file(results_path)
                    last_flush = time.time()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Cancelling pending tasks...")
            console.print("\n[yellow]Interrupted! Waiting for in-flight tasks to save results (up to 30s)...[/yellow]")
            executor.shutdown(wait=False, cancel_futures=True)
            in_flight = [f for f in futures if not f.done()]
            if in_flight:
                concurrent.futures.wait(in_flight, timeout=30)
            flush_results_file(results_path)
            console.print(f"[green]Partial results saved to {results_path}[/green]")
            console.print("[green]Re-run the same command to resume from where you left off.[/green]")
            raise SystemExit(1)
        finally:
            executor.shutdown(wait=False)

    flush_results_file(results_path)
    if _RESULTS_CACHE:
        _print_verification_report(_RESULTS_CACHE, results_path)


if __name__ == "__main__":
    app()
