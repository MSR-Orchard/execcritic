"""Submission gate for generated-test tasks.

This agent is intentionally narrow: it lets a model explore normally, but it
only accepts submission after the run has produced and exercised one fixed
generated test file. The gate is hard control flow, not prompt advice.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import re
import shlex
import time
from pathlib import PurePosixPath

from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.exceptions import FormatError, LimitsExceeded, Submitted


# Unrecoverable sandbox-connection signatures. When the remote sandbox agent is
# gone (pod killed / network dropped), every subsequent command comes back as an
# observation with returncode -1 and one of these phrases. Left unhandled, the
# agent loops into per-command timeouts until step_limit (~45 min of dead wall
# clock). We raise on the first occurrence so the rollout's retry loop can rebuild
# the env, and — if that also fails — mark the sample ABORTED with a zeroed loss
# mask instead of training on a garbage trajectory.
SANDBOX_CONNECTION_FAILURE = re.compile(
    r"Cannot connect to host|Connect call failed|Agent connection error|"
    r"Connection refused|Server disconnected|Connection reset by peer",
    re.IGNORECASE,
)


def _sandbox_connection_failure(output: dict) -> str:
    """Return the offending text if the output signals a dead sandbox, else ''."""
    if (output.get("returncode") if output else 0) == 0:
        return ""
    for field in ("output", "exception_info"):
        text = str((output or {}).get(field) or "")
        if SANDBOX_CONNECTION_FAILURE.search(text):
            return text
    extra = (output or {}).get("extra") or {}
    text = str(extra.get("exception") or "")
    if SANDBOX_CONNECTION_FAILURE.search(text):
        return text
    return ""


PASS = re.compile(r"\b(\d+) passed\b")
FAIL = re.compile(r"\b(\d+) failed\b")
ERRS = re.compile(r"\b(\d+) errors?\b")
UNITTEST_RAN = re.compile(r"\bRan (\d+) tests?\b")
UNITTEST_FAILED = re.compile(r"\bFAILED \(([^)]*)\)")
UNITTEST_OK = re.compile(r"^OK\b", re.M)
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
BLOCK = re.compile(
    r"No module named|ModuleNotFoundError|ImportError|cannot import name|collected 0 items|"
    r"no tests ran|Ran 0 tests?|\b(?:SyntaxError|IndentationError|TabError)\b|"
    r"INTERNALERROR|^ERROR:\s*(?:file or directory not found|not found):|"
    r"fixture .* not found",
    re.I | re.M,
)
MODULE_NOT_FOUND = re.compile(
    r"(?:ModuleNotFoundError:\s*)?No module named ['\"]([^'\"]+)['\"]|\bModuleNotFoundError\b",
    re.I,
)
IMPORT_NAME_MISSING = re.compile(r"cannot import name ['\"]?([^'\"\n]+?)['\"]?(?:\s+from|\s*$)", re.I)
GENERIC_IMPORT_ERROR = re.compile(r"^(?:E\s+)?ImportError:", re.I | re.M)
BAD_RELATIVE_IMPORT = re.compile(
    r"attempted relative import (?:beyond top-level package|with no known parent package)",
    re.I,
)
PYTHON_SYNTAX_ERROR = re.compile(r"\b(?:SyntaxError|IndentationError|TabError)\b", re.I)
FIXTURE_NOT_FOUND = re.compile(r"fixture ['\"]?([^'\"\s]+)['\"]? not found", re.I)
RUNNER_LABEL_ERROR = re.compile(
    r"collected 0 items|no tests ran|Ran 0 tests?|^ERROR:\s*(?:file or directory not found|not found):",
    re.I | re.M,
)
COLLECTION_ERROR = re.compile(r"ERROR collecting|ImportError while importing test module", re.I)
INTERNAL_TEST_RUNNER_ERROR = re.compile(r"INTERNALERROR", re.I)
RUNTIME_NAME_ERROR = re.compile(r"\b(?:NameError|UnboundLocalError):", re.I)
DATABASE_ACCESS_NOT_ALLOWED = re.compile(
    r"Database queries to ['\"][^'\"]+['\"] are not allowed",
    re.I,
)
TEST_RUNNER_USAGE_ERROR = re.compile(r"(?:^|\n)ERROR:\s+usage:|ExitCode\.USAGE_ERROR|\bUsageError:", re.I)
PERMISSION_SETUP_FAILURE = re.compile(
    r"(?:AssertionError:\s*)?403\s*!=\s*200|Response code was 403",
    re.I,
)
OUTPUT_LOG_PATH = re.compile(r"<GENTEST_OUTPUT_LOG>(.*?)</GENTEST_OUTPUT_LOG>", re.S)
ORACLE_CONTRACT_FIELDS = (
    "entrypoint",
    "public_integration_path",
    "trigger_input",
    "expected_output",
    "evidence",
    "non_assertions",
)
ORACLE_MAX_FIELD_CHARS = 2000
GROUNDING_PLAN_FIELDS = (
    "issue_summary",
    "public_entrypoint",
    "public_integration_path",
    "trigger_input",
    "expected_output",
    "evidence",
    "candidate_patch_hypothesis",
    "candidate_blind_spots",
    "discriminating_trigger_inputs",
    "why_this_test_should_distinguish_patch",
    "test_strategy",
    "exploration_plan",
    "non_assertions",
)
PATCH_RISK_PLAN_FIELDS = (
    "candidate_patch_hypothesis",
    "candidate_blind_spots",
    "discriminating_trigger_inputs",
    "why_this_test_should_distinguish_patch",
)
GROUNDING_PLAN_MAX_FIELD_CHARS = 4000
IMPLEMENTATION_ORACLE = re.compile(
    r"\b(source(?:_code| code| text)?|implementation|patch|diff|line numbers?|private method|internal)\b",
    re.I,
)
BROAD_EXPECTED_OUTPUT = re.compile(r"^\s*(works?|correct|passes?|no error|does not error|doesn't error)\.?\s*$", re.I)
BROAD_PUBLIC_INTEGRATION_PATH = re.compile(
    r"^\s*(public api|public entrypoint|public integration path|documented behavior|function|method|command|api)\.?\s*$",
    re.I,
)
PRIVATE_OR_HELPER_ENTRYPOINT = re.compile(
    r"\b(private|internal|implementation|helper|source(?:_code| code| text)?|patch|diff)\b|"
    r"(?:^|[\s./:])_[A-Za-z]\w*",
    re.I,
)
PUBLIC_INTEGRATION_BLOCKING_FLAGS = {
    "missing_public_integration_path",
    "broad_public_integration_path",
    "private_or_helper_entrypoint",
}
EXECUTION_CONTRACT_MAX_FIELD_CHARS = 4000
CONTROL_CHARACTER = re.compile(r"[\x00-\x1f\x7f]")


def validate_execution_contract(
    test_file_path,
    test_command,
    command_evidence,
) -> tuple[dict, str]:
    """Validate a model-inferred, focused execution contract without repository-specific rules."""
    path = str(test_file_path or "").strip()
    command = str(test_command or "").strip()
    evidence = re.sub(r"\s+", " ", str(command_evidence or "")).strip()
    if not path:
        return {}, "test_file_path is required."
    if len(path) > EXECUTION_CONTRACT_MAX_FIELD_CHARS or CONTROL_CHARACTER.search(path):
        return {}, "test_file_path contains control characters or is too long."
    if path.startswith(("/", "~")) or "\\" in path:
        return {}, "test_file_path must be a POSIX path relative to /testbed."
    parts = PurePosixPath(path).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        return {}, "test_file_path must not contain empty, current-directory, or parent-directory segments."
    normalized_path = str(PurePosixPath(*parts))
    name = PurePosixPath(normalized_path).name
    lower_name = name.lower()
    if not lower_name.endswith(".py") or not (
        lower_name.startswith("test_") or lower_name.endswith("_test.py")
    ):
        return {}, "test_file_path must use a Python test filename such as test_*.py or *_test.py."

    if not command:
        return {}, "test_command is required."
    command = re.sub(r"^cd\s+['\"]?/testbed['\"]?\s*(?:&&|;)\s*", "", command).strip()
    if not command:
        return {}, "test_command must contain a test runner after an optional 'cd /testbed'."
    if len(command) > EXECUTION_CONTRACT_MAX_FIELD_CHARS or CONTROL_CHARACTER.search(command):
        return {}, "test_command must be one line without control characters and within the size limit."
    if "`" in command or "$(" in command:
        return {}, "test_command must not contain shell command substitution."
    try:
        tokens = shlex.split(command)
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>")
        lexer.whitespace_split = True
        shell_tokens = list(lexer)
    except ValueError as exc:
        return {}, f"test_command is not valid shell syntax: {exc}."
    if not tokens:
        return {}, "test_command is required."
    if any(token and all(char in ";&|<>" for char in token) for token in shell_tokens):
        return {}, "test_command must be one focused invocation without shell chaining, pipes, or redirection."

    stem = name[:-3]
    module = normalized_path[:-3].replace("/", ".")
    module_without_tests = module[len("tests.") :] if module.startswith("tests.") else module
    target_references = {
        normalized_path.lower(),
        f"/testbed/{normalized_path}".lower(),
        name.lower(),
        stem.lower(),
        module.lower(),
        module_without_tests.lower(),
    }
    command_lower = command.lower()
    if not any(reference and reference in command_lower for reference in target_references):
        return {}, "test_command must explicitly target the selected generated test path, module, or test stem."

    discover_pattern_is_focused = "discover" in tokens and any(
        token in {name, stem, f"{stem}.py"} for token in tokens
    )
    broad_targets = {".", "./", "test", "test/", "tests", "tests/"}
    for token in tokens:
        raw_target = token.split("=", 1)[-1] if "=" in token else token
        raw_target = raw_target.split("::", 1)[0]
        if raw_target in broad_targets and not discover_pattern_is_focused:
            return {}, "test_command must run only the generated test, not a test directory or full suite."
        if not raw_target.lower().endswith(".py"):
            continue
        candidate_name = PurePosixPath(raw_target).name.lower()
        if candidate_name in {"python.py", "runtests.py", "setup.py"}:
            continue
        candidate = raw_target
        if candidate.startswith("/testbed/"):
            candidate = candidate[len("/testbed/") :]
        candidate = candidate.lstrip("./")
        if candidate not in {normalized_path, name}:
            return {}, f"test_command targets another Python file ({raw_target}); run only {normalized_path}."

    if not evidence:
        return {}, "command_evidence is required and must cite repository evidence for the path and command."
    if len(evidence) > EXECUTION_CONTRACT_MAX_FIELD_CHARS:
        return {}, "command_evidence is too long."
    return {
        "test_file_path": normalized_path,
        "test_command": command,
        "command_evidence": evidence,
    }, ""

INFRA_FAILURE_CATEGORIES = {
    "bad_relative_import",
    "collection_error",
    "command_exception",
    "empty_test_output",
    "fixture_not_found",
    "generic_import_error",
    "database_access_not_allowed",
    "permission_setup_failure",
    "import_name_missing",
    "internal_test_runner_error",
    "mixed_failure_with_errors",
    "module_not_found",
    "test_runner_usage_error",
    "test_runtime_errors",
    "runner_label_error",
    "runtime_name_error",
    "syntax_error_in_generated_test",
    "unknown_self_test_failure",
}
BLOCKING_FAILURE_CATEGORIES = {
    "bad_relative_import",
    "collection_error",
    "command_exception",
    "database_access_not_allowed",
    "permission_setup_failure",
    "fixture_not_found",
    "generic_import_error",
    "import_name_missing",
    "internal_test_runner_error",
    "module_not_found",
    "test_runner_usage_error",
    "runner_label_error",
    "runtime_name_error",
    "syntax_error_in_generated_test",
}
FAILURE_REASON_BY_CATEGORY = {
    "assertion_failure_clean": "",
    "bad_relative_import": "generated test uses a relative import that cannot resolve from the runner location",
    "base_passed": "generated test passed on the buggy base checkout",
    "collection_error": "test collection failed before reaching the target assertion",
    "command_exception": "generated-test command raised an execution exception",
    "database_access_not_allowed": "generated test setup used database access from a context where queries are disabled",
    "permission_setup_failure": "request setup returned a permission response before reaching the intended behavior",
    "empty_test_output": "test runner output did not contain a usable pass/fail/error summary",
    "fixture_not_found": "generated test depends on a fixture unavailable to this runner",
    "generic_import_error": "generated test has an import-time failure in this repository version",
    "import_name_missing": "generated test imports a symbol absent in this repository version",
    "internal_test_runner_error": "test runner hit an internal error before a useful assertion result",
    "mixed_failure_with_errors": "test produced assertion failures and errors; errors must be removed",
    "module_not_found": "generated test imports a module absent in this repository version",
    "not_clean_failure": "generated test did not produce exactly one clean assertion-failure style result",
    "test_runner_usage_error": "the selected test command or its options are invalid before reaching the target behavior",
    "test_runtime_errors": "generated test errored before or during assertion execution",
    "runner_label_error": "generated test was not collected by the selected repo test runner",
    "runtime_name_error": "generated test has a NameError or UnboundLocalError before reaching the target behavior",
    "syntax_error_in_generated_test": "generated test has invalid Python syntax or indentation",
    "unknown_self_test_failure": "generated-test result could not be classified",
}
class GeneratedTestSubmitAgentConfig(AgentConfig):
    test_file: str = "test_model_gen.py"
    """Initial generated test path for legacy fixed-path mode."""
    require_execution_contract: bool = False
    """Require write_generated_test to choose a safe path, focused command, and repository evidence."""
    require_self_test: bool = True
    """Require a test runner command mentioning test_file before submission."""
    require_self_test_clean_fail: bool = True
    """Require the latest self-test run for test_file to be a clean failure."""
    require_only_test_file_changes: bool = True
    """Reject submission if git shows edits outside test_file."""
    require_structured_test_write: bool = True
    """Require the current test hash to have been written through structured_test_tool before submission."""
    require_oracle_contract: bool = True
    """Require a non-empty oracle contract for the current structured test before submission."""
    require_public_integration_path: bool = False
    """Require the oracle to identify a concrete public integration path before submission."""
    require_single_test_case: bool = True
    """Require the generated test file to stay within the configured test-case limit."""
    max_generated_test_cases: int = 1
    """Maximum generated test_* functions or methods allowed before submission."""
    auto_run_self_test: bool = True
    """Run the configured repo-specific test command after test_file changes."""
    auto_submit_on_clean_self_test: bool = True
    """Submit immediately after the latest self-test already satisfies the generated-test gate."""
    require_explicit_submit: bool = False
    """Require the model to submit the latest self-tested candidate through submit_test_tool."""
    submit_test_tool: str = "submit_generated_test"
    """Structured terminal tool used when require_explicit_submit is enabled."""
    require_grounding_plan: bool = False
    """Require an initial structured grounding_plan call before any bash or generated-test write."""
    require_grounding_plan_first: bool = True
    """If true, the first executed model action must be a single grounding_plan call."""
    require_patch_risk_plan: bool = False
    """Require patch-risk fields in the grounding plan for patch-conditioned test generation."""
    min_inspection_commands_after_grounding_plan: int = 0
    """Require at least this many non-test bash commands after the grounding plan before submission."""
    grounding_plan_tool: str = "grounding_plan"
    """Tool action name for the initial generated-test grounding plan."""
    structured_test_tool: str = "write_generated_test"
    """Tool action name for harness-owned test file writes."""
    max_structured_test_writes: int = 5
    """Maximum successful structured generated-test writes; negative disables the cap."""
    force_submit_on_max_structured_test_writes: bool = False
    """Submit the current generated test after the final allowed structured write."""
    block_model_self_test_commands: bool = True
    """Skip model-issued generated-test runner commands; the harness owns self-test execution."""
    self_test_command: str = ""
    """Repo-specific command used only by legacy fixed-command mode."""
    self_test_timeout: int = 0
    """Optional timeout for the harness-run self-test; 0 uses the environment default."""
    candidate_patch_text: str = ""
    """Optional candidate patch text used for patch-gentest candidate replay."""
    candidate_replay_command: str = ""
    """Repo-specific command the harness runs after applying candidate_patch_text."""
    candidate_replay_timeout: int = 0
    """Optional timeout for candidate replay; 0 uses the environment default."""
    candidate_replay_feedback_on_pass: bool = False
    """If true, block auto-submit once when the generated test passes on the candidate patch."""
    initial_test_hash: str = ""
    """Fingerprint of the pre-created starter file; this version is not self-tested."""
    max_self_test_feedback_chars: int = 6000
    """Maximum raw test output tail included in structured feedback to the model."""
    allowed_generated_paths: list[str] = []
    """Additional generated-test mirror paths allowed after running the test command."""
    max_rejections: int = 3
    """Number of blocked submissions to report before optional bypass."""
    allow_after_max_rejections: bool = False
    """If true, allow submission after max_rejections; default keeps the gate strict."""
    submit_file_contents: bool = True
    """Submit the actual file contents instead of the model's echo text."""
    max_submission_chars: int = 200_000
    """Upper bound for submitted file contents."""
    max_consecutive_format_errors: int = 3
    """Stop after this many consecutive malformed model responses; negative disables the cap."""
    stop_on_no_tool_call_format_error: bool = False
    """Stop immediately when the model response contains no tool call."""


def classify_generated_test_output(command: str, output: dict) -> dict:
    raw = (output.get("output") or "") + ("\n" + output.get("exception_info") if output.get("exception_info") else "")
    clean = ANSI.sub("", raw)
    passed_match, failed_match, errors_match = PASS.search(clean), FAIL.search(clean), ERRS.search(clean)
    passed = int(passed_match.group(1)) if passed_match else 0
    failed = int(failed_match.group(1)) if failed_match else 0
    errors = int(errors_match.group(1)) if errors_match else 0
    verdict = "empty"

    ran = UNITTEST_RAN.search(clean)
    unittest_failed = UNITTEST_FAILED.search(clean)
    if ran and unittest_failed:
        summary = unittest_failed.group(1)
        failures = re.search(r"failures=(\d+)", summary)
        err_match = re.search(r"errors=(\d+)", summary)
        failed = int(failures.group(1)) if failures else 0
        errors = int(err_match.group(1)) if err_match else 0
        if failed == 0 and errors == 0:
            failed = 1
        verdict = "fail"
    elif ran and UNITTEST_OK.search(clean):
        passed = int(ran.group(1))
        verdict = "pass" if passed > 0 else "empty"
    elif failed > 0 or errors > 0:
        verdict = "fail"
    elif passed > 0:
        verdict = "pass"
    elif BLOCK.search(clean):
        verdict = "block"

    failure_category = _failure_category(clean, output, verdict, passed, failed, errors)
    entities = _failure_entities(clean)
    if failure_category in BLOCKING_FAILURE_CATEGORIES:
        verdict = "block"
    elif failure_category == "empty_test_output":
        verdict = "empty"
    elif failure_category == "base_passed":
        verdict = "pass"
    elif failure_category in {
        "assertion_failure_clean",
        "mixed_failure_with_errors",
        "not_clean_failure",
        "test_runtime_errors",
    }:
        verdict = "fail"

    clean_fail = failure_category == "assertion_failure_clean"
    infra_reason = failure_category if failure_category in INFRA_FAILURE_CATEGORIES else ""
    failure_reason = FAILURE_REASON_BY_CATEGORY.get(
        failure_category,
        FAILURE_REASON_BY_CATEGORY["unknown_self_test_failure"],
    )
    return {
        "command": command,
        "returncode": output.get("returncode"),
        "exception_info": output.get("exception_info"),
        "verdict": verdict,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "clean_fail": clean_fail,
        "infra_failure": bool(infra_reason),
        "infra_reason": infra_reason,
        "failure_category": failure_category,
        "failure_reason": failure_reason,
        "output_log_path": _first_group(OUTPUT_LOG_PATH, clean),
        **entities,
        "tail": clean[-1200:],
    }


def _failure_category(clean: str, output: dict, verdict: str, passed: int, failed: int, errors: int) -> str:
    if output.get("exception_info"):
        return "command_exception"
    if BAD_RELATIVE_IMPORT.search(clean):
        return "bad_relative_import"
    if MODULE_NOT_FOUND.search(clean):
        return "module_not_found"
    if IMPORT_NAME_MISSING.search(clean):
        return "import_name_missing"
    if GENERIC_IMPORT_ERROR.search(clean):
        return "generic_import_error"
    if PYTHON_SYNTAX_ERROR.search(clean):
        return "syntax_error_in_generated_test"
    if FIXTURE_NOT_FOUND.search(clean):
        return "fixture_not_found"
    if INTERNAL_TEST_RUNNER_ERROR.search(clean):
        return "internal_test_runner_error"
    if TEST_RUNNER_USAGE_ERROR.search(clean):
        return "test_runner_usage_error"
    if RUNTIME_NAME_ERROR.search(clean):
        return "runtime_name_error"
    if DATABASE_ACCESS_NOT_ALLOWED.search(clean):
        return "database_access_not_allowed"
    if PERMISSION_SETUP_FAILURE.search(clean):
        return "permission_setup_failure"
    if RUNNER_LABEL_ERROR.search(clean):
        return "runner_label_error"
    if COLLECTION_ERROR.search(clean):
        return "collection_error"
    if failed > 0 and errors > 0:
        return "mixed_failure_with_errors"
    if errors > 0:
        return "test_runtime_errors"
    if verdict == "pass" or (passed > 0 and failed == 0 and errors == 0):
        return "base_passed"
    if verdict == "empty":
        return "empty_test_output"
    if verdict == "fail" and failed > 0:
        return "assertion_failure_clean"
    if verdict == "fail":
        return "not_clean_failure"
    return "unknown_self_test_failure"


def _failure_entities(clean: str) -> dict:
    return {
        "missing_module": _first_group(MODULE_NOT_FOUND, clean),
        "missing_symbol": _first_group(IMPORT_NAME_MISSING, clean),
        "missing_fixture": _first_group(FIXTURE_NOT_FOUND, clean),
    }


def _first_group(pattern: re.Pattern, text: str) -> str:
    match = pattern.search(text or "")
    if not match:
        return ""
    for value in match.groups():
        if value:
            return value.strip().strip("'\"")
    return ""


def evaluate_oracle_contract(oracle: dict | None, test_code: str = "") -> dict:
    contract = _normalize_oracle_contract(oracle)
    flags: set[str] = set()
    if not contract["entrypoint"]:
        flags.add("missing_entrypoint")
    if not contract["public_integration_path"]:
        flags.add("missing_public_integration_path")
    if not contract["trigger_input"]:
        flags.add("missing_trigger_input")
    if not contract["expected_output"]:
        flags.add("missing_expected_output")
    if not contract["evidence"]:
        flags.add("missing_evidence")
    if IMPLEMENTATION_ORACLE.search(" ".join([contract["expected_output"], contract["evidence"]])):
        flags.add("implementation_oracle")
    if contract["expected_output"] and BROAD_EXPECTED_OUTPUT.match(contract["expected_output"]):
        flags.add("broad_oracle")
    if contract["public_integration_path"] and BROAD_PUBLIC_INTEGRATION_PATH.match(
        contract["public_integration_path"]
    ):
        flags.add("broad_public_integration_path")
    if PRIVATE_OR_HELPER_ENTRYPOINT.search(
        " ".join([contract["entrypoint"], contract["public_integration_path"]])
    ):
        flags.add("private_or_helper_entrypoint")
    if _has_over_pinned_assert_literal(test_code):
        flags.add("over_pinned_literal")
    return {
        "contract": contract,
        "quality_flags": sorted(flags),
        "grounding_tags": _oracle_grounding_tags(contract["evidence"]),
        "has_contract": any(contract.values()),
    }


def _normalize_oracle_contract(oracle: dict | None) -> dict:
    if not isinstance(oracle, dict):
        oracle = {}
    return {field: _compact_oracle_value(oracle.get(field)) for field in ORACLE_CONTRACT_FIELDS}


def _compact_oracle_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    elif isinstance(value, (list, tuple, set)):
        text = "; ".join(str(item) for item in value)
    else:
        text = str(value)
    return re.sub(r"\s+", " ", text).strip()[:ORACLE_MAX_FIELD_CHARS]


def _oracle_grounding_tags(evidence: str) -> list[str]:
    text = (evidence or "").lower()
    tags = []
    if any(token in text for token in ("issue", "bug report", "problem statement", "reported", "explicit")):
        tags.append("issue_grounded")
    if any(token in text for token in ("doc", "documentation", "documented")):
        tags.append("docs_grounded")
    if "existing" in text and "test" in text:
        tags.append("existing_tests_grounded")
    if any(token in text for token in ("public_invariant", "invariant", "contract", "semantics", "public api")):
        tags.append("invariant_grounded")
    if text and not tags:
        tags.append("stated_but_unverified")
    return tags


def _has_over_pinned_assert_literal(test_code: str) -> bool:
    try:
        tree = ast.parse(test_code or "")
    except SyntaxError:
        return False
    for expr in _assertion_expressions(tree):
        for node in ast.walk(expr):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if len(node.value) >= 160 or "\n" in node.value:
                    return True
    return False


def _assertion_expressions(tree: ast.AST) -> list[ast.AST]:
    expressions: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            expressions.append(node.test)
        elif isinstance(node, ast.Call) and _ast_call_name(node.func).split(".")[-1].startswith("assert"):
            expressions.extend(node.args)
    return expressions


def _ast_call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _ast_call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


class GeneratedTestSubmitAgent(DefaultAgent):
    def __init__(self, model, env, *, self_test_runner=None, **kwargs):
        self._self_test_runner = self_test_runner
        super().__init__(model, env, config_class=GeneratedTestSubmitAgentConfig, **kwargs)
        self._self_test_attempts: list[dict] = []
        self._rejections = 0
        self._last_gate: dict = {}
        self._test_file = self._normalize_relpath(self.config.test_file)
        self._test_command = self.config.self_test_command.strip()
        self._command_evidence = ""
        self._execution_contracts: dict[str, dict] = {}
        self._last_auto_execution_hash = ""
        self._last_oracle_quality: dict = {}
        self._oracle_by_hash: dict[str, dict] = {}
        self._grounding_plan: dict = {}
        self._grounding_plan_violations = 0
        self._inspection_commands_after_grounding_plan = 0
        self._candidate_replay_attempts: list[dict] = []
        self._last_candidate_replay_hash = ""
        self._candidate_pass_feedback_hashes: set[str] = set()
        self._structured_test_writes = 0
        self._explicit_submit_attempts: list[dict] = []
        self._format_errors = 0
        self._consecutive_format_errors = 0

    def step(self) -> dict:
        try:
            result = super().step()
            self._consecutive_format_errors = 0
            return result
        except FormatError as exc:
            self._format_errors += 1
            self._consecutive_format_errors += 1
            no_tool_call = any(
                "No tool calls found in the response" in str(message.get("content") or "")
                for message in exc.messages
            )
            if self.config.stop_on_no_tool_call_format_error and no_tool_call:
                raise LimitsExceeded(
                    *exc.messages,
                    self.model.format_message(
                        role="exit",
                        content="NoToolCallFormatError",
                        extra={
                            "exit_status": "NoToolCallFormatError",
                            "submission": "",
                            "format_errors": self._format_errors,
                            "consecutive_format_errors": self._consecutive_format_errors,
                        },
                    ),
                ) from exc
            limit = self.config.max_consecutive_format_errors
            if limit >= 0 and self._consecutive_format_errors >= limit:
                raise LimitsExceeded(
                    *exc.messages,
                    self.model.format_message(
                        role="exit",
                        content="FormatErrorLimitExceeded",
                        extra={
                            "exit_status": "FormatErrorLimitExceeded",
                            "submission": "",
                            "format_errors": self._format_errors,
                            "consecutive_format_errors": self._consecutive_format_errors,
                        },
                    ),
                ) from exc
            raise
        except Submitted:
            self._consecutive_format_errors = 0
            if self.config.require_explicit_submit:
                self._rejections += 1
                self.add_messages(
                    self.model.format_message(
                        role="user",
                        content=(
                            "SUBMISSION BLOCKED: this generated-test protocol requires the structured "
                            f"{self.config.submit_test_tool} tool. Review the latest harness-run self-test, then "
                            f"call {self.config.submit_test_tool} with its execution_hash."
                        ),
                    )
                )
                return {}
            ok, reason, gate = self._evaluate_gate()
            self._last_gate = gate
            may_bypass = (
                self.config.allow_after_max_rejections
                and self.config.max_rejections >= 0
                and self._rejections >= self.config.max_rejections
            )
            if ok or may_bypass:
                raise Submitted(
                    self._submission_message(
                        gate,
                        submitted_by="model_submit" if ok else "model_submit_after_max_rejections",
                    )
                )
            self._rejections += 1
            protocol = [
                (
                    f"Call {self.config.structured_test_tool} with the full Python test_code for "
                    f"/testbed/{self._test_file}."
                ),
                "Do not edit source files or unrelated tests.",
                "Do not run the generated test command or submit manually.",
                "After the generated test execution contract changes, the harness runs its exact focused command.",
                "If the structured self-test feedback is not a clean issue failure, revise the test file.",
            ]
            protocol_text = "\n".join(f"{idx}. {step}" for idx, step in enumerate(protocol, 1))
            self.add_messages(
                self.model.format_message(
                    role="user",
                    content=(
                        "SUBMISSION BLOCKED by the generated-test gate "
                        f"({self._rejections}/{self.config.max_rejections}).\n\n"
                        f"{reason}\n\n"
                        f"Required protocol:\n{protocol_text}"
                    ),
                )
            )
            return {}

    def execute_actions(self, message: dict) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        outputs = []
        explicit_submission = None
        n_self_tests_before = len(self._self_test_attempts)
        if self._must_block_until_grounded(actions):
            outputs = [self._grounding_plan_required_output(action) for action in actions]
            return self.add_messages(
                *self.model.format_observation_messages(
                    message, self._normalise_outputs(outputs), self.get_template_vars()
                )
            )
        explicit_submit_actions = [action for action in actions if self._is_explicit_submit_action(action)]
        if explicit_submit_actions and len(actions) != 1:
            self._rejections += 1
            self._explicit_submit_attempts.append(
                {
                    "requested_execution_hash": str(
                        explicit_submit_actions[0].get("execution_hash") or ""
                    ).strip(),
                    "current_execution_hash": "",
                    "accepted": False,
                    "reason": f"{self.config.submit_test_tool} must be the only tool call in the response.",
                    "protocol_rejected": True,
                }
            )
            outputs = [
                {
                    "output": (
                        f"GENTEST_PROTOCOL_BLOCKED: {self.config.submit_test_tool} must be the only tool call "
                        "in the response because submission is terminal."
                    ),
                    "returncode": 2,
                    "exception_info": None,
                    "extra": {"gentest_explicit_submit_rejected": True},
                }
                for _ in actions
            ]
            return self.add_messages(
                *self.model.format_observation_messages(
                    message, self._normalise_outputs(outputs), self.get_template_vars()
                )
            )
        for action in actions:
            try:
                if self._is_grounding_plan_action(action):
                    output = self._record_grounding_plan_action(action)
                elif self._is_explicit_submit_action(action):
                    output, explicit_submission = self._explicit_submit_action(action)
                elif self._is_structured_test_write(action):
                    output = self._write_structured_test_action(action)
                elif self.config.block_model_self_test_commands and self._is_test_command(action.get("command", "")):
                    output = self._blocked_model_self_test_output()
                else:
                    output = self.env.execute(action)
                    if failure := _sandbox_connection_failure(output):
                        # Dead sandbox: stop looping into timeouts. Propagate so the
                        # rollout retry loop can rebuild the env or abort the sample.
                        raise ConnectionError(f"Sandbox connection failed: {failure[:200]}")
            except Submitted:
                outputs.append(
                    {
                        "output": "Submission marker observed; generated-test gate is checking the submission.",
                        "returncode": 0,
                        "exception_info": None,
                        "extra": {},
                    }
                )
                self.add_messages(
                    *self.model.format_observation_messages(
                        message, self._normalise_outputs(outputs), self.get_template_vars()
                    )
                )
                raise
            outputs.append(output)
            if not (output.get("extra") or {}).get("gentest_blocked_model_self_test"):
                if self._is_inspection_command(action):
                    self._inspection_commands_after_grounding_plan += 1
                self._record_self_test_attempt(action, output)
        messages = self.add_messages(
            *self.model.format_observation_messages(message, self._normalise_outputs(outputs), self.get_template_vars())
        )
        if explicit_submission:
            self.add_messages(explicit_submission)
            return messages
        auto_attempt = self._run_harness_self_test_if_needed()
        auto_submit_message = None
        wrote_structured_test = any((output.get("extra") or {}).get("gentest_structured_write") for output in outputs)
        if len(self._self_test_attempts) > n_self_tests_before or wrote_structured_test:
            auto_submit_message = self._auto_submit_message_if_ready()
        if auto_submit_message:
            self.add_messages(auto_submit_message)
        elif self._should_force_submit_after_structured_write(wrote_structured_test):
            self.add_messages(self._forced_write_limit_submission_message())
        elif auto_attempt:
            self.add_messages(self._self_test_feedback_message(auto_attempt))
        return messages

    def _must_block_until_grounded(self, actions: list[dict]) -> bool:
        if not self.config.require_grounding_plan or self._grounding_plan:
            return False
        if len(actions) == 1 and self._is_grounding_plan_action(actions[0]):
            return False
        if self.config.require_grounding_plan_first:
            return True
        return any(
            self._is_structured_test_write(action) or self._is_explicit_submit_action(action)
            for action in actions
        )

    def _grounding_plan_required_output(self, action: dict) -> dict:
        self._grounding_plan_violations += 1
        attempted = action.get("tool") or ("bash" if action.get("command") else "unknown")
        required_fields = (
            "issue_summary, public_entrypoint, public_integration_path, trigger_input, "
            "expected_output, evidence, test_strategy, exploration_plan, and optional non_assertions"
        )
        return {
            "output": (
                "GENTEST_PROTOCOL_BLOCKED: the first generated-test phase must be a single "
                f"{self.config.grounding_plan_tool} tool call before bash inspection or "
                f"{self.config.structured_test_tool}.\n\n"
                f"Blocked attempted action: {attempted}.\n\n"
                f"Call {self.config.grounding_plan_tool} with {required_fields}. "
                "Use best current hypotheses from the issue; write unknowns explicitly and say how inspection "
                "will resolve them."
            ),
            "returncode": 2,
            "exception_info": None,
            "extra": {
                "gentest_grounding_required": True,
                "blocked_before_grounding_plan": attempted,
            },
        }

    def _record_grounding_plan_action(self, action: dict) -> dict:
        plan = self._normalize_grounding_plan(action)
        self._grounding_plan = plan
        required = tuple(field for field in GROUNDING_PLAN_FIELDS if field != "non_assertions")
        missing = [field for field in required if not plan.get(field)]
        if not self.config.require_patch_risk_plan:
            missing = [field for field in missing if field not in PATCH_RISK_PLAN_FIELDS]
        status = "recorded" if not missing else "recorded_with_missing_fields"
        missing_text = ", ".join(missing) or "none"
        return {
            "output": (
                f"Recorded initial {self.config.grounding_plan_tool}.\n"
                f"status={status}; missing_required_fields={missing_text}.\n"
                "Now inspect the repository to validate imports, fixtures, runner conventions, and the x->y "
                "mapping before writing the generated test."
            ),
            "returncode": 0,
            "exception_info": None,
            "extra": {
                "gentest_grounding_plan": True,
                "grounding_plan_status": status,
                "grounding_plan": self._compact_grounding_plan(plan),
            },
        }

    @staticmethod
    def _normalise_outputs(outputs: list[dict]) -> list[dict]:
        for output in outputs:
            if not isinstance(output, dict):
                continue
            output.setdefault("exception_info", None)
            output.setdefault("output", "")
            output.setdefault("returncode", None)
            output.setdefault("extra", {})
        return outputs

    @property
    def gate_info(self) -> dict:
        return {
            "require_execution_contract": bool(self.config.require_execution_contract),
            "test_file": self._test_file,
            "test_command": self._test_command,
            "command_evidence": self._command_evidence,
            "self_test_commands": [attempt.get("command", "") for attempt in self._self_test_attempts],
            "self_test_attempts": [self._compact_attempt(attempt) for attempt in self._self_test_attempts[-20:]],
            "structured_test_writes": self._structured_test_writes,
            "max_structured_test_writes": int(self.config.max_structured_test_writes),
            "require_explicit_submit": bool(self.config.require_explicit_submit),
            "submit_test_tool": self.config.submit_test_tool,
            "explicit_submit_attempts": [dict(attempt) for attempt in self._explicit_submit_attempts[-10:]],
            "last_oracle_quality": self._compact_oracle_quality(self._last_oracle_quality),
            "grounding_plan": self._compact_grounding_plan(self._grounding_plan),
            "grounding_plan_required": bool(self.config.require_grounding_plan),
            "grounding_plan_recorded": bool(self._grounding_plan),
            "grounding_plan_violations": self._grounding_plan_violations,
            "inspection_commands_after_grounding_plan": self._inspection_commands_after_grounding_plan,
            "min_inspection_commands_after_grounding_plan": int(
                self.config.min_inspection_commands_after_grounding_plan
            ),
            "candidate_replay_enabled": self._candidate_replay_enabled(),
            "candidate_replay_attempts": [
                self._compact_candidate_replay(attempt) for attempt in self._candidate_replay_attempts[-10:]
            ],
            "latest_candidate_replay": self._compact_candidate_replay(self._latest_candidate_replay_attempt()),
            "rejections": self._rejections,
            "last_gate": dict(self._last_gate),
        }

    def serialize(self, *extra_dicts) -> dict:
        return super().serialize({"info": {"gentest_gate": self.gate_info}}, *extra_dicts)

    def _evaluate_gate(self) -> tuple[bool, str, dict]:
        test_code = self._read_test_file()
        changed = self._changed_paths()
        allowed = {self._test_file}
        allowed.update(self._normalize_relpath(path) for path in self.config.allowed_generated_paths)
        outside_changes = sorted(path for path in changed if path not in allowed and not self._ignored_change(path))
        latest_self_test = self._latest_self_test_attempt()
        current_hash = self._fingerprint(test_code)
        current_execution_hash = self._execution_fingerprint(test_code, self._test_file, self._test_command)
        oracle_quality = self._oracle_quality_for_hash(current_hash, test_code)
        structured_test_write_recorded = current_hash in self._oracle_by_hash and (
            not self.config.require_execution_contract or current_execution_hash in self._execution_contracts
        )
        test_shape = self._generated_test_shape(test_code)
        gate = {
            "require_execution_contract": bool(self.config.require_execution_contract),
            "test_file": self._test_file,
            "test_command": self._test_command,
            "command_evidence": self._command_evidence,
            "test_hash": current_hash,
            "execution_hash": current_execution_hash,
            "has_test_file": bool(test_code.strip()),
            "test_code": test_code[: self.config.max_submission_chars],
            "oracle": dict(oracle_quality.get("contract") or {}),
            "oracle_quality": self._compact_oracle_quality(oracle_quality),
            "structured_test_write_recorded": structured_test_write_recorded,
            "structured_test_writes": self._structured_test_writes,
            "max_structured_test_writes": int(self.config.max_structured_test_writes),
            "max_generated_test_cases": self._max_generated_test_cases(),
            "test_shape": test_shape,
            "grounding_plan": self._compact_grounding_plan(self._grounding_plan),
            "grounding_plan_recorded": bool(self._grounding_plan),
            "inspection_commands_after_grounding_plan": self._inspection_commands_after_grounding_plan,
            "min_inspection_commands_after_grounding_plan": int(
                self.config.min_inspection_commands_after_grounding_plan
            ),
            "candidate_replay_enabled": self._candidate_replay_enabled(),
            "latest_candidate_replay": self._compact_candidate_replay(
                self._latest_candidate_replay_attempt(current_hash)
            ),
            "candidate_replay_attempts": [
                self._compact_candidate_replay(attempt) for attempt in self._candidate_replay_attempts[-10:]
            ],
            "self_tested": bool(self._self_test_attempts),
            "self_test_clean_fail": bool(latest_self_test and latest_self_test.get("clean_fail")),
            "latest_self_test": self._compact_attempt(latest_self_test),
            "self_test_commands": [attempt.get("command", "") for attempt in self._self_test_attempts[-10:]],
            "changed_paths": sorted(changed),
            "outside_changes": outside_changes,
            "rejections": self._rejections,
        }

        if self.config.require_execution_contract:
            _, contract_error = validate_execution_contract(
                self._test_file,
                self._test_command,
                self._command_evidence,
            )
            if contract_error:
                return False, f"Invalid generated-test execution contract: {contract_error}", gate
        if not test_code.strip():
            return False, f"/testbed/{self._test_file} is missing or empty.", gate
        if len(test_code) > self.config.max_submission_chars:
            return False, "The generated-test artifact is too large to submit safely.", gate
        if self.config.require_structured_test_write and not structured_test_write_recorded:
            return (
                False,
                (
                    f"The current /testbed/{self._test_file} execution contract was not written through "
                    f"{self.config.structured_test_tool}. Call {self.config.structured_test_tool} with the "
                    "full test_code and oracle contract "
                    "instead of editing the generated test through bash."
                ),
                gate,
            )
        if self.config.require_oracle_contract and not oracle_quality.get("has_contract"):
            return (
                False,
                (
                    "No oracle contract was recorded for the current generated test. Call "
                    f"{self.config.structured_test_tool} with the generated-test artifact and oracle fields "
                    "entrypoint, public_integration_path, trigger_input, expected_output, evidence, "
                    "and non_assertions."
                ),
                gate,
            )
        if self.config.require_public_integration_path:
            blocking_flags = sorted(
                set(oracle_quality.get("quality_flags") or []) & PUBLIC_INTEGRATION_BLOCKING_FLAGS
            )
            if blocking_flags:
                return (
                    False,
                    (
                        "Oracle contract must identify a concrete public integration path and avoid private "
                        f"or helper entrypoints (quality_flags={', '.join(blocking_flags)}). Call "
                        f"{self.config.structured_test_tool} with the generated-test artifact and oracle fields "
                        "entrypoint, public_integration_path, trigger_input, expected_output, evidence, "
                        "and non_assertions. The public_integration_path should describe the repo's user-facing "
                        "API, CLI, framework runner, documented object, or adjacent-test workflow that reaches "
                        "the issue behavior."
                    ),
                    gate,
                )
        if self.config.require_grounding_plan and not self._grounding_plan:
            return False, f"No initial {self.config.grounding_plan_tool} was recorded.", gate
        if self.config.require_patch_risk_plan:
            missing_patch_fields = [field for field in PATCH_RISK_PLAN_FIELDS if not self._grounding_plan.get(field)]
            if missing_patch_fields:
                return (
                    False,
                    (
                        "Grounding plan must include candidate patch risk analysis fields: "
                        f"{', '.join(missing_patch_fields)}. Call {self.config.grounding_plan_tool} again with "
                        "candidate_patch_hypothesis, candidate_blind_spots, discriminating_trigger_inputs, "
                        "and why_this_test_should_distinguish_patch."
                    ),
                    gate,
                )
        min_inspections = max(0, int(self.config.min_inspection_commands_after_grounding_plan))
        if self._inspection_commands_after_grounding_plan < min_inspections:
            return (
                False,
                (
                    "Inspect repository source/tests before writing the final generated test. "
                    f"Observed {self._inspection_commands_after_grounding_plan} inspection command(s) after "
                    f"grounding_plan; require at least {min_inspections}."
                ),
                gate,
            )
        if self.config.require_self_test and not self._self_test_attempts:
            return False, f"No test runner result for {self._test_file} was observed.", gate
        if self.config.require_self_test_clean_fail:
            if not latest_self_test:
                return False, f"No self-test result for {self._test_file} was observed.", gate
            if latest_self_test.get("execution_hash") != current_execution_hash:
                return False, f"The path, command, or contents changed after the latest self-test; rerun it.", gate
            if not latest_self_test.get("clean_fail"):
                detail = latest_self_test.get("verdict") or "unknown"
                category = latest_self_test.get("failure_category") or "unknown"
                reason = latest_self_test.get("infra_reason") or latest_self_test.get("failure_reason") or ""
                return (
                    False,
                    f"Latest self-test was not a clean failure "
                    f"(verdict={detail}; category={category}{'; ' + reason if reason else ''}).",
                    gate,
                )
        if self.config.require_single_test_case and test_shape.get("parse_ok"):
            max_generated_test_cases = self._max_generated_test_cases()
            test_count = int(test_shape.get("test_count") or 0)
            if not (1 <= test_count <= max_generated_test_cases):
                if max_generated_test_cases == 1:
                    count_requirement = "exactly one generated test case"
                    fix_instruction = (
                        "Keep one issue-backed test_* function or one test_* method and remove unrelated controls, "
                        "variants, and extra tests."
                    )
                else:
                    count_requirement = f"between 1 and {max_generated_test_cases} generated test cases"
                    fix_instruction = (
                        f"Keep at most {max_generated_test_cases} issue-backed test_* functions or test_* methods "
                        "and remove unrelated controls, variants, and extra tests."
                    )
                return (
                    False,
                    (
                        f"/testbed/{self._test_file} must contain {count_requirement}, "
                        f"but AST found {test_count}. {fix_instruction}"
                    ),
                    gate,
                )
            if int(test_shape.get("parametrize_count") or 0) > 0:
                return (
                    False,
                    (
                        f"/testbed/{self._test_file} must stay within the generated test case limit, "
                    "but parametrize expands one function into multiple cases. Replace it with one "
                        "concrete trigger input per generated test case."
                    ),
                    gate,
                )
        if self.config.require_only_test_file_changes and outside_changes:
            return False, f"Unexpected modified files: {', '.join(outside_changes[:20])}", gate
        return True, "ok", gate

    def _auto_submit_message_if_ready(self) -> dict | None:
        if self.config.require_explicit_submit or not self.config.auto_submit_on_clean_self_test:
            return None
        latest_self_test = self._latest_self_test_attempt()
        if not latest_self_test.get("clean_fail"):
            return None
        current_hash = latest_self_test.get("test_hash") or self._fingerprint(self._read_test_file())
        candidate_block_reason = self._candidate_replay_block_reason(current_hash)
        if candidate_block_reason:
            latest_self_test["auto_submit_blocked_reason"] = candidate_block_reason
            return None

        ok, reason, gate = self._evaluate_gate()
        self._last_gate = gate
        if not ok:
            self._last_gate["auto_submit_blocked_reason"] = reason
            latest_self_test["auto_submit_blocked_reason"] = reason
            return None
        return self._submission_message(gate, submitted_by="rule_based_clean_self_test")

    def _should_force_submit_after_structured_write(self, wrote_structured_test: bool) -> bool:
        max_writes = int(self.config.max_structured_test_writes)
        return (
            wrote_structured_test
            and not self.config.require_explicit_submit
            and self.config.force_submit_on_max_structured_test_writes
            and max_writes >= 0
            and self._structured_test_writes >= max_writes
        )

    def _forced_write_limit_submission_message(self) -> dict:
        ok, reason, gate = self._evaluate_gate()
        gate["forced_submission"] = True
        gate["forced_submission_reason"] = "max_structured_test_writes"
        gate["forced_submission_gate_passed"] = ok
        if not ok:
            gate["forced_submission_gate_reason"] = reason
        return self._submission_message(gate, submitted_by="max_structured_test_writes")

    def _run_harness_self_test_if_needed(self) -> dict | None:
        command = self._test_command_for_execution()
        if not self.config.auto_run_self_test or not command:
            return None

        test_code = self._read_test_file()
        if not test_code.strip():
            return None
        current_hash = self._fingerprint(test_code)
        execution_hash = self._execution_fingerprint(test_code, self._test_file, self._test_command)
        if (
            not self.config.require_execution_contract
            and current_hash == self.config.initial_test_hash
        ):
            return None
        latest_self_test = self._latest_self_test_attempt()
        if latest_self_test.get("execution_hash") == execution_hash:
            return None
        if self._last_auto_execution_hash == execution_hash:
            return None

        self_test_started = time.perf_counter()
        try:
            if self._self_test_runner is not None:
                attempt = dict(self._self_test_runner(test_code, self._test_file, self._test_command))
            else:
                if self.config.self_test_timeout > 0:
                    output = self.env.execute({"command": command}, timeout=self.config.self_test_timeout)
                else:
                    output = self.env.execute({"command": command})
                attempt = classify_generated_test_output(command, output)
        except Exception as exc:  # noqa: BLE001
            output = {
                "output": "",
                "returncode": None,
                "exception_info": f"{type(exc).__name__}: {str(exc)[:500]}",
                "extra": {},
            }
            attempt = classify_generated_test_output(command, output)
        attempt.setdefault("command", command)
        attempt["duration_sec"] = time.perf_counter() - self_test_started
        attempt["test_hash"] = current_hash
        attempt["execution_hash"] = execution_hash
        attempt["test_file_path"] = self._test_file
        attempt["test_command"] = self._test_command
        attempt["command_evidence"] = self._command_evidence
        attempt["source"] = "harness_auto"
        attempt["oracle_quality"] = self._oracle_quality_for_hash(current_hash, test_code)
        self._self_test_attempts.append(attempt)
        self._last_auto_execution_hash = execution_hash
        if attempt.get("clean_fail"):
            candidate_attempt = self._run_candidate_replay_if_needed(current_hash, test_code)
            if candidate_attempt:
                attempt["candidate_replay"] = self._compact_candidate_replay(candidate_attempt)
        return attempt

    def _self_test_feedback_message(self, attempt: dict) -> dict:
        tail = attempt.get("tail") or ""
        max_tail = max(0, int(self.config.max_self_test_feedback_chars))
        if max_tail and len(tail) > max_tail:
            tail = tail[-max_tail:]
        if attempt.get("auto_submit_blocked_reason"):
            status = "auto_submit_blocked"
        elif attempt.get("clean_fail"):
            status = "clean_failure_observed"
        else:
            status = "self_test_failed"
        gate_blocked_line = (
            f"  gate_blocked_reason: {attempt.get('auto_submit_blocked_reason')}\n"
            if attempt.get("auto_submit_blocked_reason")
            else ""
        )
        oracle_lines = self._oracle_feedback_lines(attempt.get("oracle_quality") or {})
        candidate_lines = self._candidate_replay_feedback_lines(attempt.get("test_hash", ""))
        content = (
            "generated_test_gate:\n"
            f"  status: {status}\n"
            f"  source: {attempt.get('source') or 'unknown'}\n"
            f"  verdict: {attempt.get('verdict') or 'unknown'}\n"
            f"  passed: {int(attempt.get('passed') or 0)}\n"
            f"  failed: {int(attempt.get('failed') or 0)}\n"
            f"  errors: {int(attempt.get('errors') or 0)}\n"
            f"  clean_fail: {bool(attempt.get('clean_fail'))}\n"
            f"{gate_blocked_line}"
            f"  command: {attempt.get('command') or ''}\n\n"
            f"  execution_hash: {attempt.get('execution_hash') or ''}\n\n"
            f"{candidate_lines}"
            f"{oracle_lines}"
            "repair_instructions:\n"
            "  The base self-test output below is from the buggy checkout after your generated test was written. "
            "Use it to diagnose why the current test is not acceptable.\n"
            "  If you need more evidence, continue exploring the repository with bash: inspect adjacent tests, "
            "docs, fixtures, imports, public APIs, and runner conventions until the x -> y behavior is grounded.\n"
            f"  When ready, call {self.config.structured_test_tool} with the full updated test_code and selected "
            "test_file_path, "
            "exact focused test_command, command_evidence, and oracle contract.\n"
            f"  Your next response must contain a tool call: bash for inspection or "
            f"{self.config.structured_test_tool} for the rewrite. Do not answer in prose only.\n"
            "  Do not run the generated test command yourself; the harness will run it automatically after each "
            "write.\n"
            + (
                f"  A clean failure is necessary but not sufficient: decide whether it fails for the issue-backed "
                f"behavior rather than setup, runner, or incidental reasons. If this exact candidate is ready, "
                f"call {self.config.submit_test_tool} with the execution_hash above as the only tool call. "
                f"Otherwise inspect more evidence or rewrite the candidate.\n\n"
                if self.config.require_explicit_submit
                else "\n"
            )
            + "base_self_test_output:\n"
            "<test_output_tail>\n"
            f"{tail}\n"
            "</test_output_tail>"
        )
        return self.model.format_message(role="user", content=content)

    def _submission_message(self, gate: dict, *, submitted_by: str) -> dict:
        gate = dict(gate)
        gate["submitted_by"] = submitted_by
        self._last_gate = dict(gate)
        submission = gate.get("test_code", "")
        if not self.config.submit_file_contents:
            submission = ""
        return self.model.format_message(
            role="exit",
            content=submission,
            extra={
                "exit_status": "Submitted",
                "submission": submission,
                "gentest_gate": gate,
            },
        )

    def _is_grounding_plan_action(self, action: dict) -> bool:
        return action.get("tool") == self.config.grounding_plan_tool

    def _is_explicit_submit_action(self, action: dict) -> bool:
        return action.get("tool") == self.config.submit_test_tool

    def _explicit_submit_action(self, action: dict) -> tuple[dict, dict | None]:
        requested_execution_hash = str(action.get("execution_hash") or "").strip()
        ok, reason, gate = self._evaluate_gate()
        current_execution_hash = str(gate.get("execution_hash") or "")
        if not self.config.require_explicit_submit:
            reason = "Explicit generated-test submission is not enabled for this protocol."
            ok = False
        elif not requested_execution_hash:
            reason = (
                f"{self.config.submit_test_tool} requires the execution_hash from the latest self-test feedback."
            )
            ok = False
        elif requested_execution_hash != current_execution_hash:
            reason = (
                "The requested execution_hash is stale or does not identify the current test contents, path, "
                f"and command (requested={requested_execution_hash}; current={current_execution_hash})."
            )
            ok = False

        attempt = {
            "requested_execution_hash": requested_execution_hash,
            "current_execution_hash": current_execution_hash,
            "accepted": bool(ok),
            "reason": "ok" if ok else reason,
        }
        self._explicit_submit_attempts.append(attempt)
        self._last_gate = gate
        if not ok:
            self._rejections += 1
            return (
                {
                    "output": (
                        f"GENTEST_SUBMISSION_BLOCKED ({self._rejections}/{self.config.max_rejections}): {reason}\n"
                        "Keep working from the latest candidate and self-test feedback, then explicitly submit "
                        "the exact ready candidate."
                    ),
                    "returncode": 2,
                    "exception_info": None,
                    "extra": {
                        "gentest_explicit_submit_rejected": True,
                        **attempt,
                    },
                },
                None,
            )

        gate["model_submit_requested_execution_hash"] = requested_execution_hash
        return (
            {
                "output": "Submission accepted; the latest self-tested generated-test candidate is final.",
                "returncode": 0,
                "exception_info": None,
                "extra": {
                    "gentest_explicit_submit_accepted": True,
                    **attempt,
                },
            },
            self._submission_message(gate, submitted_by="model_submit_tool"),
        )

    def _is_structured_test_write(self, action: dict) -> bool:
        return action.get("tool") == self.config.structured_test_tool or (
            "test_code" in action and "command" not in action
        )

    def _max_generated_test_cases(self) -> int:
        return max(1, int(self.config.max_generated_test_cases or 1))

    @staticmethod
    def _generated_test_shape(test_code: str) -> dict:
        try:
            tree = ast.parse(test_code or "")
        except SyntaxError as exc:
            return {
                "parse_ok": False,
                "parse_error": f"{exc.__class__.__name__}: {exc.msg}",
                "test_count": 0,
                "test_names": [],
                "parametrize_count": 0,
                "parametrize_names": [],
            }

        test_names: list[str] = []
        parametrize_names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                parent_name = GeneratedTestSubmitAgent._parent_class_name(tree, node)
                qualified_name = f"{parent_name}.{node.name}" if parent_name else node.name
                test_names.append(qualified_name)
                if any(GeneratedTestSubmitAgent._decorator_is_parametrize(decorator) for decorator in node.decorator_list):
                    parametrize_names.append(qualified_name)
        return {
            "parse_ok": True,
            "test_count": len(test_names),
            "test_names": test_names[:20],
            "all_test_names": test_names,
            "parametrize_count": len(parametrize_names),
            "parametrize_names": parametrize_names[:20],
            "all_parametrize_names": parametrize_names,
        }

    @staticmethod
    def _parent_class_name(tree: ast.AST, target: ast.AST) -> str:
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if any(child is target for child in node.body):
                return node.name
        return ""

    @staticmethod
    def _decorator_is_parametrize(decorator: ast.AST) -> bool:
        call = decorator.func if isinstance(decorator, ast.Call) else decorator
        return _ast_call_name(call).endswith("parametrize")

    @staticmethod
    def _normalize_grounding_plan(action: dict) -> dict:
        return {
            field: GeneratedTestSubmitAgent._compact_grounding_value(action.get(field))
            for field in GROUNDING_PLAN_FIELDS
        }

    def _write_structured_test_action(self, action: dict) -> dict:
        max_writes = int(self.config.max_structured_test_writes)
        if max_writes >= 0 and self._structured_test_writes >= max_writes:
            return self._structured_test_write_limit_output(max_writes)

        test_code = action.get("test_code")
        if not isinstance(test_code, str):
            return {
                "output": "",
                "returncode": 2,
                "exception_info": "write_generated_test action requires a string test_code field.",
                "extra": {"gentest_structured_write": True},
            }
        if len(test_code) > self.config.max_submission_chars:
            return {
                "output": "",
                "returncode": 2,
                "exception_info": (
                    f"write_generated_test action is too large: {len(test_code)} chars "
                    f"> {self.config.max_submission_chars}."
                ),
                "extra": {"gentest_structured_write": True},
            }

        execution_contract = {
            "test_file_path": self._test_file,
            "test_command": self._test_command,
            "command_evidence": self._command_evidence,
        }
        if self.config.require_execution_contract:
            execution_contract, contract_error = validate_execution_contract(
                action.get("test_file_path"),
                action.get("test_command"),
                action.get("command_evidence"),
            )
            if contract_error:
                return {
                    "output": "",
                    "returncode": 2,
                    "exception_info": f"Invalid generated-test execution contract: {contract_error}",
                    "extra": {"gentest_structured_write": True, "gentest_execution_contract_rejected": True},
                }

        cleanup = self._discard_non_generated_changes()
        if (cleanup.get("returncode") or 0) != 0 or cleanup.get("exception_info"):
            self._normalise_outputs([cleanup])
            cleanup.setdefault("extra", {})["gentest_structured_write"] = True
            return cleanup

        if self.config.require_execution_contract:
            selected_path = execution_contract["test_file_path"]
            path_probe = self.env.execute(
                {"command": f"cd /testbed && test ! -e {shlex.quote(selected_path)}"},
                timeout=30,
            )
            if failure := _sandbox_connection_failure(path_probe):
                raise ConnectionError(f"Sandbox connection failed: {failure[:200]}")
            if (path_probe.get("returncode") or 0) != 0 or path_probe.get("exception_info"):
                return {
                    "output": path_probe.get("output") or "",
                    "returncode": 2,
                    "exception_info": (
                        f"test_file_path already exists in the buggy repository: {selected_path}. "
                        "Choose a new adjacent test_*.py or *_test.py file instead of overwriting an existing test."
                    ),
                    "extra": {"gentest_structured_write": True, "gentest_execution_contract_rejected": True},
                }
            self._test_file = execution_contract["test_file_path"]
            self._test_command = execution_contract["test_command"]
            self._command_evidence = execution_contract["command_evidence"]
        out = self._write_test_file(test_code)
        if (out.get("returncode") or 0) != 0 or out.get("exception_info"):
            self._normalise_outputs([out])
            out.setdefault("extra", {})["gentest_structured_write"] = True
            return out
        oracle_quality = evaluate_oracle_contract(action.get("oracle"), test_code)
        self._last_oracle_quality = oracle_quality
        test_hash = self._fingerprint(test_code)
        execution_hash = self._execution_fingerprint(
            test_code,
            self._test_file,
            self._test_command,
        )
        self._oracle_by_hash[test_hash] = oracle_quality
        self._execution_contracts[execution_hash] = {
            "test_file_path": self._test_file,
            "test_command": self._test_command,
            "command_evidence": self._command_evidence,
        }
        flags = ", ".join(oracle_quality.get("quality_flags") or []) or "none"
        tags = ", ".join(oracle_quality.get("grounding_tags") or []) or "none"
        self._structured_test_writes += 1
        return {
            "output": (
                f"Wrote {len(test_code)} chars to /testbed/{self._test_file}.\n"
                f"Execution command: {self._test_command_for_execution()}\n"
                f"Recorded oracle quality_flags={flags}; grounding_tags={tags}.\n"
                "The harness will run the generated-test gate automatically."
            ),
            "returncode": 0,
            "exception_info": None,
            "extra": {
                "gentest_structured_write": True,
                "written_test_file": self._test_file,
                "test_command": self._test_command,
                "command_evidence": self._command_evidence,
                "execution_hash": execution_hash,
                "oracle_quality": self._compact_oracle_quality(oracle_quality),
            },
        }

    def _structured_test_write_limit_output(self, max_writes: int) -> dict:
        return {
            "output": (
                f"Skipped {self.config.structured_test_tool}: reached max_structured_test_writes="
                f"{max_writes}. The generated test will not be rewritten again, and the harness will not "
                "run another automatic self-test for this action."
            ),
            "returncode": 2,
            "exception_info": (
                f"max_structured_test_writes exceeded: {self._structured_test_writes}/{max_writes} "
                f"successful {self.config.structured_test_tool} calls already used."
            ),
            "extra": {"gentest_structured_write_blocked": True},
        }

    def _blocked_model_self_test_output(self) -> dict:
        return {
            "output": (
                "Skipped model-issued generated-test command. The harness owns self-test execution and "
                "will run the configured command automatically after the generated-test artifact changes."
            ),
            "returncode": 0,
            "exception_info": None,
            "extra": {"gentest_blocked_model_self_test": True},
        }

    def _write_test_file(self, text: str) -> dict:
        relpath = self._test_file
        qpath = shlex.quote(relpath)
        payload = base64.b64encode(text.encode("utf-8", "replace")).decode("ascii")
        parent = str(PurePosixPath(relpath).parent)
        commands = []
        if parent and parent != ".":
            commands.append(f"mkdir -p {shlex.quote(parent)}")
        commands.append(f"printf %s '{payload}' | base64 -d > {qpath}")
        return self.env.execute({"command": f"cd /testbed && {' && '.join(commands)}"}, timeout=30)

    def _discard_non_generated_changes(self) -> dict:
        return self.env.execute(
            {"command": "cd /testbed && git checkout -- . && git clean -fdq"},
            timeout=120,
        )

    def _read_file(self, path: str) -> str:
        qpath = shlex.quote(path)
        out = self.env.execute(
            {"command": f"cd /testbed && base64 -w0 {qpath} 2>/dev/null"},
            timeout=30,
        )
        blob = (out.get("output") or "").strip()
        if (out.get("returncode") or 0) != 0 or not blob:
            return ""
        try:
            return base64.b64decode(blob).decode("utf-8", "replace")
        except Exception:
            return ""

    def _read_test_file(self) -> str:
        return self._read_file(self._test_file)

    def _changed_paths(self) -> set[str]:
        out = self.env.execute(
            {
                "command": (
                    "cd /testbed && "
                    "{ git diff --name-only -- .; git ls-files --others --exclude-standard; } "
                    "2>/dev/null | sort -u"
                )
            },
            timeout=30,
        )
        if (out.get("returncode") or 0) != 0:
            return set()
        return {
            self._normalize_relpath(line)
            for line in (out.get("output") or "").splitlines()
            if line.strip()
        }

    def _candidate_replay_enabled(self) -> bool:
        return bool(self.config.candidate_patch_text.strip() and self.config.candidate_replay_command.strip())

    def _run_candidate_replay_if_needed(self, test_hash: str, test_code: str) -> dict | None:
        if not self._candidate_replay_enabled() or not test_hash:
            return None
        latest = self._latest_candidate_replay_attempt(test_hash)
        if latest:
            return latest
        if self._last_candidate_replay_hash == test_hash:
            return None

        attempt: dict = {
            "test_hash": test_hash,
            "source": "candidate_replay",
            "command": self.config.candidate_replay_command.strip(),
        }
        try:
            try:
                apply_output = self._apply_candidate_patch()
            except Exception as exc:  # noqa: BLE001
                apply_output = {
                    "output": "",
                    "returncode": None,
                    "exception_info": f"{type(exc).__name__}: {str(exc)[:500]}",
                    "extra": {},
                }
            attempt["apply_returncode"] = apply_output.get("returncode")
            attempt["apply_exception_info"] = apply_output.get("exception_info")
            if (apply_output.get("returncode") or 0) != 0 or apply_output.get("exception_info"):
                attempt.update(
                    {
                        "status": "candidate_apply_failure",
                        "candidate_pass": False,
                        "candidate_clean_fail": False,
                        "tail": ((apply_output.get("output") or "") + (apply_output.get("exception_info") or ""))[
                            -1200:
                        ],
                    }
                )
                return attempt

            command = self.config.candidate_replay_command.strip()
            try:
                timeout = self.config.candidate_replay_timeout or self.config.self_test_timeout
                if timeout > 0:
                    output = self.env.execute({"command": command}, timeout=timeout)
                else:
                    output = self.env.execute({"command": command})
            except Exception as exc:  # noqa: BLE001
                output = {
                    "output": "",
                    "returncode": None,
                    "exception_info": f"{type(exc).__name__}: {str(exc)[:500]}",
                    "extra": {},
                }
            classified = classify_generated_test_output(command, output)
            classified["test_hash"] = test_hash
            classified["source"] = "candidate_replay"
            classified["candidate_pass"] = classified.get("verdict") == "pass"
            classified["candidate_clean_fail"] = bool(classified.get("clean_fail"))
            if classified["candidate_pass"]:
                classified["status"] = "candidate_pass"
            elif classified.get("infra_failure"):
                classified["status"] = "candidate_infra_failure"
            elif classified["candidate_clean_fail"]:
                classified["status"] = "candidate_clean_fail"
            else:
                classified["status"] = "candidate_not_pass"
            attempt.update(classified)
            return attempt
        finally:
            try:
                restore_error = self._restore_test_after_candidate_replay(test_code)
            except Exception as exc:  # noqa: BLE001
                restore_error = f"{type(exc).__name__}: {str(exc)[:500]}"
            if restore_error:
                attempt["restore_error"] = restore_error
            self._candidate_replay_attempts.append(attempt)
            self._last_candidate_replay_hash = test_hash

    def _apply_candidate_patch(self) -> dict:
        payload = base64.b64encode(self.config.candidate_patch_text.encode("utf-8", "replace")).decode("ascii")
        command = (
            "cd /testbed && "
            f"printf %s '{payload}' | base64 -d > /tmp/gentest_candidate.diff && "
            "("
            "git apply --whitespace=nowarn /tmp/gentest_candidate.diff || "
            "git apply --3way --whitespace=nowarn /tmp/gentest_candidate.diff || "
            "patch --batch --fuzz=5 -p1 -i /tmp/gentest_candidate.diff"
            ")"
        )
        return self.env.execute({"command": command}, timeout=120)

    def _restore_test_after_candidate_replay(self, test_code: str) -> str:
        cleanup = self._discard_non_generated_changes()
        cleanup_text = (cleanup.get("output") or "") + (cleanup.get("exception_info") or "")
        if (cleanup.get("returncode") or 0) != 0 or cleanup.get("exception_info"):
            return cleanup_text[-1200:] or "candidate replay cleanup failed"
        out = self._write_test_file(test_code)
        write_text = (out.get("output") or "") + (out.get("exception_info") or "")
        if (out.get("returncode") or 0) != 0 or out.get("exception_info"):
            return write_text[-1200:] or "candidate replay test restore failed"
        return ""

    def _candidate_replay_block_reason(self, test_hash: str) -> str:
        if not self.config.candidate_replay_feedback_on_pass:
            return ""
        attempt = self._latest_candidate_replay_attempt(test_hash)
        if not attempt or not attempt.get("candidate_pass"):
            return ""
        if test_hash in self._candidate_pass_feedback_hashes:
            return ""
        self._candidate_pass_feedback_hashes.add(test_hash)
        return (
            "Candidate replay passed: the generated test cleanly fails on the buggy base, but it also passes "
            "after applying the candidate patch."
        )

    def _latest_candidate_replay_attempt(self, test_hash: str = "") -> dict:
        for attempt in reversed(self._candidate_replay_attempts):
            if not test_hash or attempt.get("test_hash") == test_hash:
                return attempt
        return {}

    @staticmethod
    def _compact_candidate_replay(attempt: dict | None) -> dict:
        if not attempt:
            return {}
        keys = (
            "status",
            "source",
            "command",
            "test_hash",
            "candidate_pass",
            "candidate_clean_fail",
            "verdict",
            "passed",
            "failed",
            "errors",
            "failure_category",
            "infra_reason",
            "returncode",
            "apply_returncode",
            "restore_error",
            "tail",
        )
        out = {key: attempt.get(key) for key in keys if key in attempt}
        if "tail" in out:
            out["tail"] = (out.get("tail") or "")[-1200:]
        if "restore_error" in out:
            out["restore_error"] = (out.get("restore_error") or "")[-1200:]
        return out

    def _record_self_test_attempt(self, action: dict, output: dict) -> None:
        command = action.get("command") if isinstance(action, dict) else ""
        if not command or not self._is_test_command(command):
            return
        attempt = classify_generated_test_output(command, output)
        test_code = self._read_test_file()
        attempt["test_hash"] = self._fingerprint(test_code)
        attempt["execution_hash"] = self._execution_fingerprint(test_code, self._test_file, self._test_command)
        attempt["test_file_path"] = self._test_file
        attempt["test_command"] = command
        attempt["command_evidence"] = self._command_evidence
        attempt["source"] = "model_command"
        attempt["oracle_quality"] = self._oracle_quality_for_hash(attempt["test_hash"], test_code)
        self._self_test_attempts.append(attempt)

    def _latest_self_test_attempt(self) -> dict:
        return self._self_test_attempts[-1] if self._self_test_attempts else {}

    def _is_inspection_command(self, action: dict) -> bool:
        command = action.get("command") if isinstance(action, dict) else ""
        return bool(self._grounding_plan and command and not self._is_test_command(command))

    def _is_test_command(self, command: str) -> bool:
        text = re.sub(r"\s+", " ", (command or "").lower())
        configured_command = re.sub(r"\s+", " ", self._test_command_for_execution().lower())
        if configured_command and text == configured_command:
            return True
        test_name = PurePosixPath(self._test_file).name.lower()
        test_stem = test_name[:-3] if test_name.endswith(".py") else test_name
        mentions_test = test_name in text or test_stem in text
        mentions_runner = (
            "pytest" in text
            or "runtests.py" in text
            or "unittest" in text
            or "tox" in text
            or re.search(r"(^|[;&|()\s])bin/test(\s|$)", text) is not None
        )
        return bool(mentions_test and mentions_runner)

    def _test_command_for_execution(self) -> str:
        command = self._test_command.strip()
        if not command:
            return ""
        if self.config.require_execution_contract:
            return f"cd /testbed && {command}"
        return command

    @staticmethod
    def _fingerprint(text: str) -> str:
        return hashlib.sha256((text or "").encode("utf-8", "replace")).hexdigest()

    @staticmethod
    def _execution_fingerprint(test_code: str, test_file_path: str, test_command: str) -> str:
        payload = "\0".join((test_code or "", test_file_path or "", test_command or ""))
        return hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()

    @staticmethod
    def _compact_attempt(attempt: dict | None) -> dict:
        if not attempt:
            return {}
        out = dict(attempt)
        if "tail" in out:
            out["tail"] = (out.get("tail") or "")[-1200:]
        if "oracle_quality" in out:
            out["oracle_quality"] = GeneratedTestSubmitAgent._compact_oracle_quality(out.get("oracle_quality"))
        return out

    @staticmethod
    def _compact_grounding_plan(plan: dict | None) -> dict:
        if not isinstance(plan, dict):
            return {}
        return {
            field: GeneratedTestSubmitAgent._compact_grounding_value(plan.get(field))
            for field in GROUNDING_PLAN_FIELDS
        }

    @staticmethod
    def _compact_grounding_value(value) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return re.sub(r"\s+", " ", value).strip()[:GROUNDING_PLAN_MAX_FIELD_CHARS]

    @staticmethod
    def _compact_oracle_quality(oracle_quality: dict | None) -> dict:
        if not isinstance(oracle_quality, dict):
            return {}
        contract = oracle_quality.get("contract") if isinstance(oracle_quality.get("contract"), dict) else {}
        return {
            "contract": {field: _compact_oracle_value(contract.get(field)) for field in ORACLE_CONTRACT_FIELDS},
            "quality_flags": list(oracle_quality.get("quality_flags") or []),
            "grounding_tags": list(oracle_quality.get("grounding_tags") or []),
            "has_contract": bool(oracle_quality.get("has_contract")),
        }

    def _oracle_quality_for_hash(self, test_hash: str, test_code: str) -> dict:
        oracle_quality = self._oracle_by_hash.get(test_hash)
        if oracle_quality:
            self._last_oracle_quality = oracle_quality
            return oracle_quality
        oracle_quality = evaluate_oracle_contract({}, test_code)
        self._last_oracle_quality = oracle_quality
        return oracle_quality

    def _candidate_replay_feedback_lines(self, test_hash: str) -> str:
        attempt = self._latest_candidate_replay_attempt(test_hash)
        if not attempt:
            return ""
        compact = self._compact_candidate_replay(attempt)
        lines = ["candidate_replay:"]
        for field in (
            "status",
            "candidate_pass",
            "candidate_clean_fail",
            "verdict",
            "passed",
            "failed",
            "errors",
            "failure_category",
            "infra_reason",
        ):
            lines.append(f"  {field}: {compact.get(field, '')}")
        if compact.get("candidate_pass"):
            lines.append(
                "  next_candidate_action: The current test also passes on the candidate patch. "
                "Strengthen issue-backed public-behavior triggers or assertions that target the candidate's "
                "changed branch/blind spot. If no stronger issue-backed discriminator exists, call "
                f"{self.config.structured_test_tool} again with the same generated-test artifact and oracle to confirm."
            )
        elif compact.get("candidate_clean_fail"):
            lines.append(
                "  next_candidate_action: Candidate replay cleanly failed, so this test distinguishes the "
                "candidate from the buggy base behavior under the current public oracle."
            )
        elif compact.get("status") == "candidate_apply_failure":
            lines.append(
                "  next_candidate_action: Candidate patch could not be applied in replay; keep focusing on "
                "issue-backed public behavior and inspect the patch only as diagnostic context."
            )
        return "\n".join(lines) + "\n\n"

    @staticmethod
    def _oracle_feedback_lines(oracle_quality: dict) -> str:
        if not oracle_quality:
            return ""
        compact = GeneratedTestSubmitAgent._compact_oracle_quality(oracle_quality)
        contract = compact.get("contract") or {}
        flags = ", ".join(compact.get("quality_flags") or []) or "none"
        tags = ", ".join(compact.get("grounding_tags") or []) or "none"
        lines = ["oracle:"]
        for field in ORACLE_CONTRACT_FIELDS:
            lines.append(f"  {field}: {contract.get(field) or ''}")
        lines.append(f"  quality_flags: {flags}")
        lines.append(f"  grounding_tags: {tags}")
        if compact.get("quality_flags"):
            lines.append(
                "  next_oracle_action: Fill missing x/y/evidence/public_integration_path or replace "
                "implementation-derived expectations with externally observable behavior."
            )
        return "\n".join(lines) + "\n\n"

    @staticmethod
    def _normalize_relpath(path: str) -> str:
        text = str(path or "").strip()
        if text.startswith("/testbed/"):
            text = text[len("/testbed/") :]
        text = text.lstrip("./")
        return str(PurePosixPath(text))

    @staticmethod
    def _ignored_change(path: str) -> bool:
        parts = PurePosixPath(path).parts
        if any(part in {"__pycache__", ".pytest_cache", ".hypothesis", ".mypy_cache"} for part in parts):
            return True
        return path.endswith((".pyc", ".pyo", ".coverage"))
