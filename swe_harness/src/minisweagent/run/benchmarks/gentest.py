#!/usr/bin/env python3
"""Generate regression tests as a native mini-SWE-agent benchmark.

This runner treats test generation as a first-class mini-swe-agent task:

- the submit protocol is enforced by GeneratedTestSubmitAgent;
- every sample gets a normal mini-swe-agent trajectory;
- validation labels are written as structured metadata;
- base-fail and gold-pass are kept separate so weak signals cannot masquerade
  as resolved labels.
"""

from __future__ import annotations

import ast
import base64
import concurrent.futures
import copy
import hashlib
import json
import multiprocessing
import os
import random
import re
import shlex
import time
import traceback
from pathlib import Path
from queue import Empty
from typing import Any

import typer
from datasets import load_dataset

from minisweagent.agents import get_agent
from minisweagent.agents.extra.generated_test_submit import (
    classify_generated_test_output,
    evaluate_oracle_contract,
    validate_execution_contract,
)
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.benchmarks.swerebench import (
    DATASET_MAPPING as SWEBENCH_DATASET_MAPPING,
    _resolve_per_instance_api_base,
    get_sb_environment,
)
from minisweagent.utils.log import add_file_handler, logger
from minisweagent.utils.serialize import UNSET, recursive_merge


app = typer.Typer(rich_markup_mode="rich", add_completion=False)
DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "gentest.yaml"
TEST_FILE = "test_model_gen.py"
DEFAULT_CONDA_ENV = "testbed"
SCALESWE_DATASET = "AweAI-Team/Scale-SWE"
REBENCH_V2_DATASET = SWEBENCH_DATASET_MAPPING["rebench_v2"]
SWE_BENCH_MULTILINGUAL_DATASET = SWEBENCH_DATASET_MAPPING["multilingual"]
DATASET_MAPPING = {**SWEBENCH_DATASET_MAPPING, "scaleswe": SCALESWE_DATASET}

ACTIVATE = (
    "source /opt/miniconda3/bin/activate 2>/dev/null; "
    "source /opt/conda/bin/activate 2>/dev/null; "
    f"if conda activate {DEFAULT_CONDA_ENV} 2>/dev/null; then :; "
    f"else echo 'gentest_activation_failure: {DEFAULT_CONDA_ENV}' >&2; exit 127; fi; "
    "export PYTHONPATH=/testbed:/testbed/src:${PYTHONPATH:-};"
)


def _is_scaleswe_instance(instance: dict | None) -> bool:
    return bool((instance or {}).get("dataset") == SCALESWE_DATASET)


def _is_rebench_v2_instance(instance: dict | None) -> bool:
    return bool((instance or {}).get("dataset") == REBENCH_V2_DATASET)


def _is_multilingual_instance(instance: dict | None) -> bool:
    return bool((instance or {}).get("dataset") == SWE_BENCH_MULTILINGUAL_DATASET)


def _link_workdir_to_testbed(env, workdir: str, expected_commit: str, dataset_name: str) -> None:
    quoted_workdir = shlex.quote(workdir)
    linked = env.execute(
        {
            "command": (
                "if [ -e /testbed ] && [ ! -L /testbed ]; then "
                f"echo '{dataset_name} adapter refuses to replace existing /testbed' >&2; exit 73; fi; "
                f"ln -sfn {quoted_workdir} /testbed && git -C {quoted_workdir} rev-parse HEAD"
            )
        },
        cwd=workdir,
        timeout=120,
    )
    if linked.get("returncode") != 0:
        output = str(linked.get("output") or "") + str(linked.get("exception_info") or "")
        raise RuntimeError(f"{dataset_name} /testbed adapter failed: {output[-2000:]}")

    output_lines = str(linked.get("output") or "").strip().splitlines()
    actual_commit = output_lines[-1] if output_lines else ""
    if expected_commit and actual_commit != expected_commit:
        raise RuntimeError(
            f"{dataset_name} base commit mismatch: expected {expected_commit}, got {actual_commit}"
        )


def get_gentest_environment(config: dict, instance: dict):
    """Create the dataset sandbox while preserving gentest's /testbed contract."""
    if not _is_scaleswe_instance(instance) and not _is_rebench_v2_instance(instance):
        return get_sb_environment(config, instance)

    workdir = str(instance.get("workdir") or "").strip()
    if not workdir.startswith("/") or workdir == "/":
        raise ValueError(f"Invalid gentest workdir for {instance.get('dataset')}: {workdir!r}")

    if _is_rebench_v2_instance(instance):
        image_name = (
            instance.get("image_name")
            or instance.get("image_url")
            or instance.get("docker_image")
        )
        if not image_name:
            raise ValueError(f"No image field found in ReBench V2 instance {instance.get('instance_id')}")
        routed_instance = {**instance, "image_name": image_name}
        config.setdefault("environment", {})["cwd"] = workdir
        env = get_sb_environment(config, routed_instance)
        try:
            _link_workdir_to_testbed(
                env,
                workdir,
                str(instance.get("base_commit") or "").strip(),
                "ReBench V2",
            )
            return env
        except Exception:
            env.cleanup()
            raise

    from minisweagent.run.benchmarks.scaleswe import get_scaleswe_docker_image_name

    env_config = config.setdefault("environment", {})
    env_config["environment_class"] = env_config.get("environment_class", "docker")
    image_name = get_scaleswe_docker_image_name(instance)
    if env_config["environment_class"] in ["docker", "swerex_modal", "azure_modal"]:
        env_config["image"] = image_name
    elif env_config["environment_class"] in ["singularity", "contree"]:
        env_config["image"] = "docker://" + image_name
    env_config["cwd"] = workdir

    env = get_environment(env_config)
    try:
        setup_timeout = int(env_config.get("sandbox_timeout") or 600)
        pre_commands = instance.get("pre_commands")
        if isinstance(pre_commands, list):
            pre_commands = " && ".join(str(command) for command in pre_commands if str(command).strip())
        if pre_commands:
            prepared = env.execute(
                {"command": str(pre_commands).replace("\\n", "\n")},
                cwd=workdir,
                timeout=setup_timeout,
            )
            if prepared.get("returncode") != 0:
                output = str(prepared.get("output") or "") + str(prepared.get("exception_info") or "")
                raise RuntimeError(f"Scale-SWE pre_commands failed: {output[-2000:]}")

        expected_commit = str(
            instance.get("parent_commit") or instance.get("base_commit") or ""
        ).strip()
        _link_workdir_to_testbed(env, workdir, expected_commit, "Scale-SWE")
        return env
    except Exception:
        env.cleanup()
        raise


def get_gentest_agent_environment(env, instance: dict):
    """Apply the SWE-bench activation contract only to images that use it."""
    if _is_rebench_v2_instance(instance):
        return env
    if _is_multilingual_instance(instance):
        from minisweagent.run.benchmarks.swebench_multilingual_verify_azure_modal import (
            _get_repo_language,
        )

        if _get_repo_language(str(instance.get("repo") or "")) != "py":
            return env
    return ActivatingEnv(env, activation_for_instance(instance))


IMPORT_LINE = re.compile(r"^\s*(?:from\s+[\w.]+\s+import\s+.+|import\s+.+)$")

NON_TEST_EXTS = (
    ".json",
    ".png",
    ".csv",
    ".txt",
    ".md",
    ".jpg",
    ".jpeg",
    ".pkl",
    ".zip",
    ".parquet",
    ".feather",
    ".svg",
    ".npy",
    ".yml",
    ".yaml",
    ".toml",
)
DEFAULT_GENERATED_PYTEST_CMD = "python -m pytest --color=no -rA --tb=no -p no:cacheprovider"
SELF_TEST_TAIL_LINES = 240
FALLBACK_TEST_CMD_BY_STYLE = {
    "django_unittest": "./tests/runtests.py --verbosity 2",
    "sympy_bin_test": "bin/test -C --verbose",
    "pytest": DEFAULT_GENERATED_PYTEST_CMD,
}


class ActivatingEnv:
    """Prefix every agent command with the instance-specific conda activation."""

    def __init__(self, inner, prefix: str = ACTIVATE):
        self._inner = inner
        self._prefix = prefix

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def execute(self, action, *args, **kwargs):
        wrapped = dict(action)
        wrapped["command"] = f"{self._prefix} {action.get('command', '')}"
        return self._inner.execute(wrapped, *args, **kwargs)


class GitHistoryGuardEnv:
    """Block agent commands that mine git history for the fix (gold-free enforcement).

    Layer-3 behavioural guard complementing sanitize_git_history's physical prune: even if a
    stray ref survives, commands like `git log --all` / `git show <sha>` return a canned
    refusal instead of executing, and the attempt is visible in the trajectory. Legitimate
    working-tree git (`git add -AN`, `git diff`, `git status`, `git reset`) is untouched.
    """

    _REFUSAL = (
        "<returncode>1</returncode>\n<output>\nBLOCKED: inspecting git history/other commits to "
        "find the fix is not allowed for this task. Derive the corrected behavior from the issue "
        "text and the current buggy code only. (git add/diff/status/reset on the working tree are "
        "fine.)\n</output>"
    )

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def execute(self, action, *args, **kwargs):
        if _GIT_LEAK_RE.search(str(action.get("command", ""))):
            return {"output": self._REFUSAL, "returncode": 1, "exception_info": ""}
        return self._inner.execute(action, *args, **kwargs)


def shell(env, command: str, *, timeout: int | None = None) -> dict:
    return env.execute({"command": command}, timeout=timeout)


def conda_env_name(instance: dict | None = None) -> str:
    environment = (instance or {}).get("environment") or ""
    if isinstance(environment, dict):
        name = str(environment.get("name") or "").strip()
        if name:
            return name
        prefix = str(environment.get("prefix") or "")
    else:
        text = str(environment)
        match = re.search(r"(?m)^name:\s*([A-Za-z0-9_.-]+)\s*$", text)
        if match:
            return match.group(1)
        prefix = text
    match = re.search(r"(?m)^prefix:\s*/opt/(?:conda|miniconda3)/envs/([A-Za-z0-9_.-]+)\s*$", prefix)
    if match:
        return match.group(1)
    match = re.search(r"/(?:conda|miniconda3)/envs/([A-Za-z0-9_.-]+)(?:\s|$)", prefix)
    if match:
        return match.group(1)
    return ""


def conda_env_candidates(instance: dict | None = None) -> list[str]:
    name = conda_env_name(instance)
    candidates = []
    if name and name != DEFAULT_CONDA_ENV:
        candidates.append(name)
    candidates.append(DEFAULT_CONDA_ENV)
    return candidates


def activation_for_instance(instance: dict | None = None) -> str:
    candidates = conda_env_candidates(instance)
    activate = " || ".join(f"conda activate {shlex.quote(name)} 2>/dev/null" for name in candidates)
    names = ",".join(candidates)
    return (
        "source /opt/miniconda3/bin/activate 2>/dev/null; "
        "source /opt/conda/bin/activate 2>/dev/null; "
        f"if {activate}; then :; "
        "elif ! command -v conda >/dev/null 2>&1; then :; "  # no conda (e.g. Scale-SWE): use system python
        f"else echo 'gentest_activation_failure: {names}' >&2; exit 127; fi; "
        "export PYTHONPATH=/testbed:/testbed/src:${PYTHONPATH:-};"
    )


def list_from_json_or_obj(value) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [text]
        return list_from_json_or_obj(parsed)
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def write_file(env, path: str, text: str, *, timeout: int = 30) -> None:
    payload = base64.b64encode((text or "").encode("utf-8")).decode("ascii")
    qpath = shlex.quote(path)
    shell(env, f"printf %s '{payload}' | base64 -d > {qpath}", timeout=timeout)


def _normalize_test_file_path(filename: str = TEST_FILE) -> str:
    text = str(filename or TEST_FILE).strip()
    if text.startswith("/testbed/"):
        text = text[len("/testbed/") :]
    return str(Path(text.lstrip("./")).as_posix())


def execution_contract_hash(code_or_patch: str, filename: str, test_command: str) -> str:
    payload = "\0".join((code_or_patch or "", _normalize_test_file_path(filename), (test_command or "").strip()))
    return hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()


def write_generated_test_file(env, filename: str, text: str, *, timeout: int = 30) -> None:
    filename = _normalize_test_file_path(filename)
    parent = filename.rsplit("/", 1)[0] if "/" in filename else ""
    if parent:
        shell(env, f"cd /testbed && mkdir -p {shlex.quote(parent)}", timeout=timeout)
    write_file(env, f"/testbed/{filename}", text, timeout=timeout)


def read_test_file(env, filename: str = TEST_FILE) -> str:
    qname = shlex.quote(_normalize_test_file_path(filename))
    out = shell(env, f"cd /testbed && base64 -w0 {qname} 2>/dev/null", timeout=30)
    blob = (out.get("output") or "").strip()
    if (out.get("returncode") or 0) != 0 or not blob:
        return ""
    try:
        return base64.b64decode(blob).decode("utf-8", "replace")
    except Exception:
        return ""


def read_repo_file(env, relpath: str) -> str:
    relpath = str(relpath or "").strip()
    if relpath.startswith("/testbed/"):
        relpath = relpath[len("/testbed/") :]
    relpath = relpath.lstrip("./")
    if not relpath:
        return ""
    qpath = shlex.quote(relpath)
    out = shell(env, f"cd /testbed && base64 -w0 {qpath} 2>/dev/null", timeout=30)
    blob = (out.get("output") or "").strip()
    if (out.get("returncode") or 0) != 0 or not blob:
        return ""
    try:
        return base64.b64decode(blob).decode("utf-8", "replace")
    except Exception:
        return ""


def reset_base(env) -> None:
    # `git checkout -- .` only reverts tracked-file worktree edits; it leaves
    # staged changes, new commits, and build artifacts (e.g. `pip install -e`
    # rewriting sources, or the agent committing) that shift later gold-patch
    # hunks by large offsets and break `git apply`. Hard-reset to HEAD and clean
    # everything (incl. ignored) so gold applies against a pristine base.
    shell(env, "cd /testbed && git reset --hard HEAD && git clean -fdxq", timeout=120)


# Commands the agent must not use to look up the fix from git history. Enforced by
# GitHistoryGuardEnv below AND, physically, by sanitize_git_history at env creation.
_GIT_LEAK_RE = re.compile(
    r"""\bgit\b[^\n;|&]*?          # a git invocation ...
        (?:\blog\b[^\n;|&]*?--all  # git log --all / --branches / --remotes
          |\blog\b[^\n;|&]*?--branches
          |\blog\b[^\n;|&]*?--remotes
          |\blog\b[^\n;|&]*?--grep # git log --grep=<issue keyword>
          |\bshow\b\s+[0-9a-fA-F]{7,40}   # git show <full/short commit sha>
          |\bdiff\b[^\n;|&]*?\b[0-9a-fA-F]{7,40}\b  # git diff <sha> ...
          |\bbranch\b[^\n;|&]*?-a          # git branch -a / -r
          |\bbranch\b[^\n;|&]*?-r
          |\bcat-file\b
          |\brev-list\b[^\n;|&]*?--all
          |\bfor-each-ref\b)
    """,
    re.VERBOSE,
)


def sanitize_git_history(env, *, timeout: int = 180) -> dict:
    """Physically sever /testbed's git history so the fix is unreachable (gold-free base).

    SWE-bench images ship the full repo; `git reset --hard <base>` only moves the worktree
    and leaves the fix commit, other branches, and the origin remote in `.git`, so an agent
    can `git log --all` / `git show <fix_sha>` and copy the answer. This removes every ref
    except the current HEAD, drops the remote, expires the reflog, and prunes, so objects
    unreachable from HEAD (the fix and its tests) are physically gone. HEAD's own commit SHA
    is preserved (detached or on a branch), so `git reset --hard <base_commit>` elsewhere in
    the harness keeps working.
    """
    success_marker = "GENTEST_GIT_HISTORY_SANITIZED"
    script = (
        "set -eu; cd /testbed; "
        "git rev-parse --is-inside-work-tree >/dev/null; "
        "head_ref=$(git symbolic-ref -q HEAD 2>/dev/null || true); "
        "for remote in $(git remote); do git remote remove \"$remote\"; done; "
        # Keep only the ref HEAD currently points at (empty when detached -> all refs dropped,
        # while HEAD itself still pins its commit). Feed all deletions through checked update-ref
        # input instead of masking individual failures.
        "git for-each-ref --format='%(refname)' | while IFS= read -r ref; do "
        "  if [ -z \"$head_ref\" ] || [ \"$ref\" != \"$head_ref\" ]; then "
        "    printf 'delete %s\\n' \"$ref\"; "
        "  fi; "
        "done | git update-ref --stdin; "
        "git reflog expire --expire=now --expire-unreachable=now --all; "
        "git gc --prune=now; git prune --expire=now; "
        # Fail if a remote, extra ref, reflog entry, or unreachable object survived. A successful
        # return code alone is insufficient because sandbox transports can truncate shell output.
        "test -z \"$(git remote)\"; "
        "remaining_refs=$(git for-each-ref --format='%(refname)'); "
        "if [ -n \"$head_ref\" ]; then test \"$remaining_refs\" = \"$head_ref\"; "
        "else test -z \"$remaining_refs\"; fi; "
        "test -z \"$(git reflog show --all 2>/dev/null)\"; "
        "unreachable=$(git fsck --full --unreachable --no-reflogs 2>&1); "
        "test -z \"$unreachable\" || { printf '%s\\n' \"$unreachable\" >&2; exit 86; }; "
        f"printf '{success_marker}\\n'"
    )
    result = shell(env, script, timeout=timeout)
    output = str(result.get("output") or "")
    exception_info = str(result.get("exception_info") or "")
    if result.get("returncode") != 0:
        details = (output + "\n" + exception_info).strip()[-1000:]
        raise RuntimeError(
            f"git-history sanitization command failed with returncode={result.get('returncode')}: {details}"
        )
    if success_marker not in output:
        raise RuntimeError("git-history sanitization postcondition marker is missing")
    return result


def restore_env(env, instance: dict, timeout: int, *, activation: str | None = None) -> None:
    install_config = instance.get("install_config", {}) or {}
    if isinstance(install_config, str):
        try:
            install_config = json.loads(install_config)
        except json.JSONDecodeError:
            install_config = {}
    commands = []
    eval_commands = install_config.get("eval_commands") or []
    if isinstance(eval_commands, str):
        eval_commands = [eval_commands]
    commands.extend(eval_commands)
    install = install_config.get("install")
    if install:
        commands.extend(install if isinstance(install, list) else [install])
    script = " && ".join(str(cmd).strip() for cmd in commands if str(cmd).strip())
    if script:
        activation = activation or activation_for_instance(instance)
        shell(env, f"cd /testbed && ( {activation} {script} ) 2>&1 | tail -80", timeout=timeout)


def run_generated_test(
    env,
    instance: dict,
    filename: str,
    timeout: int,
    *,
    test_command: str | None = None,
    activation: str | None = None,
) -> dict:
    test_command = str(test_command or "").strip() or test_command_for_instance(instance, filename)
    logical_command = f"cd /testbed && {test_command}"
    activation = activation or activation_for_instance(instance)
    script = (
        "log=$(mktemp /tmp/gentest-selftest.XXXXXX.log) || exit 125; "
        f"( {activation} {test_command} ) > \"$log\" 2>&1; "
        "rc=$?; "
        f"tail -{SELF_TEST_TAIL_LINES} \"$log\"; "
        "printf '\\n<GENTEST_OUTPUT_LOG>%s</GENTEST_OUTPUT_LOG>\\n' \"$log\"; "
        "printf '<GENTEST_COMMAND_RC>%s</GENTEST_COMMAND_RC>\\n' \"$rc\"; "
        "exit \"$rc\""
    )
    harness_command = f"cd /testbed && bash -lc {shlex.quote(script)}"
    result = shell(env, harness_command, timeout=timeout)
    classified = classify_generated_test_output(logical_command, result)
    classified["harness_command"] = harness_command
    return classified


def _generated_test_patch(filename: str, text: str) -> str:
    filename = _normalize_test_file_path(filename)
    lines = (text or "").splitlines()
    body = "\n".join(f"+{line}" for line in lines)
    if body:
        body += "\n"
    return (
        f"diff --git a/{filename} b/{filename}\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        f"+++ b/{filename}\n"
        f"@@ -0,0 +1,{len(lines)} @@\n"
        f"{body}"
    )


def run_generated_test_official(
    env,
    instance: dict,
    filename: str,
    code: str,
    timeout: int,
    *,
    test_command: str | None = None,
    include_dataset_test_patch: bool = False,
    skip_install: bool = False,
) -> dict:
    """Run a generated test through the dataset's canonical official eval script/parser."""
    if _is_scaleswe_instance(instance):
        write_generated_test_file(env, filename, code)
        classified = run_generated_test(
            env,
            instance,
            filename,
            timeout,
            test_command=test_command,
        )
        classified["official_verify_format"] = True
        classified["official_verifier_kind"] = "scaleswe_execution_contract"
        return classified

    if instance.get("install_config"):
        from minisweagent.run.benchmarks.swerebench_verify_azure_modal import (
            END_TEST_OUTPUT,
            START_TEST_OUTPUT,
            _upload_file,
            build_eval_script,
            grade_eval_output,
        )
        official_kind = "swe_rebench"
    else:
        from minisweagent.run.benchmarks.swebench_verify_azure_modal_cli import (
            END_TEST_OUTPUT,
            START_TEST_OUTPUT,
            _upload_file,
            build_eval_script,
            grade_eval_output,
        )
        official_kind = "swebench"

    generated_patch = _generated_test_patch(filename, code)
    dataset_test_patch = str(instance.get("test_patch") or "") if include_dataset_test_patch else ""
    combined_test_patch = "\n".join(
        patch.rstrip("\n") for patch in (dataset_test_patch, generated_patch) if patch.strip()
    ) + "\n"
    official_instance = copy.deepcopy(instance)
    official_instance["test_patch"] = combined_test_patch
    selected_command = str(test_command or "").strip() or test_command_for_instance(instance, filename)
    logical_command = f"cd /testbed && {selected_command}"
    build_kwargs = {"test_command_override": selected_command}
    if official_kind == "swe_rebench":
        build_kwargs["environment_prepared"] = skip_install
    else:
        build_kwargs["skip_install"] = skip_install
    eval_script = build_eval_script(official_instance, **build_kwargs)
    _upload_file(env, "/tmp/test_patch.diff", combined_test_patch, timeout=60)
    _upload_file(env, "/tmp/eval.sh", eval_script, timeout=30)
    output = shell(env, "bash /tmp/eval.sh 2>&1", timeout=timeout)
    full_output = str(output.get("output") or "")
    test_output = full_output
    if START_TEST_OUTPUT in full_output and END_TEST_OUTPUT in full_output:
        test_output = full_output.split(START_TEST_OUTPUT, 1)[1].split(END_TEST_OUTPUT, 1)[0]
    classified = classify_generated_test_output(
        logical_command,
        {**output, "output": test_output},
    )
    classified["official_verify_format"] = True
    classified["official_verifier_kind"] = official_kind
    classified["official_grade"] = grade_eval_output(
        official_instance,
        full_output,
        exit_code=int(output.get("returncode") if output.get("returncode") is not None else -1),
        patch_applied=True,
    )
    classified["official_output_tail"] = full_output[-2000:]
    return classified


def _run_multilingual_test_patch_official(
    env,
    instance: dict,
    test_patch: str,
    test_command: str,
    timeout: int,
) -> dict:
    from minisweagent.run.benchmarks.swebench_multilingual_verify_azure_modal import (
        _upload_file,
        build_gentest_eval_script,
        classify_gentest_output,
    )

    official_instance = copy.deepcopy(instance)
    official_instance["test_patch"] = test_patch.rstrip("\n") + "\n"
    eval_script = build_gentest_eval_script(official_instance, test_command)
    _upload_file(env, "/tmp/test_patch.diff", official_instance["test_patch"], timeout=60)
    _upload_file(env, "/tmp/eval.sh", eval_script, timeout=30)
    output = shell(env, "bash /tmp/eval.sh 2>&1", timeout=timeout)
    classified = classify_gentest_output(
        official_instance,
        f"cd /testbed && {test_command}",
        output,
    )
    classified["official_verify_format"] = True
    classified["official_verifier_kind"] = "swebench_multilingual"
    classified["official_output_tail"] = str(output.get("output") or "")[-2000:]
    return classified


def run_test_patch_official(
    env,
    instance: dict,
    test_patch: str,
    test_command: str,
    timeout: int,
    *,
    skip_install: bool = True,
) -> dict:
    """Resolve-style verify: apply an incremental test_patch (may edit existing test files) via the
    dataset's OFFICIAL eval script, run `test_command`, return classify output. gold-free: the
    official eval_script handles conda activation, install, repo reset, and `git apply` of
    `instance['test_patch']`; we do NOT call grade_eval_output (that needs gold FAIL_TO_PASS).

    Env is a dedicated verifier sandbox, kept warm across calls. Set skip_install=True after the
    first prepared run so deps are not reinstalled every time.
    """
    if _is_multilingual_instance(instance):
        return _run_multilingual_test_patch_official(
            env,
            instance,
            test_patch,
            test_command,
            timeout,
        )

    if _is_scaleswe_instance(instance):
        write_file(env, "/tmp/test_patch.diff", test_patch.rstrip("\n") + "\n", timeout=60)
        logical_command = f"cd /testbed && {test_command}"
        command = (
            f"{activation_for_instance(instance)} "
            "cd /testbed && git apply --whitespace=nowarn /tmp/test_patch.diff && "
            f"{test_command}"
        )
        output = shell(env, command, timeout=timeout)
        classified = classify_generated_test_output(logical_command, output)
        classified["official_verify_format"] = True
        classified["official_verifier_kind"] = "scaleswe_execution_contract"
        classified["official_output_tail"] = str(output.get("output") or "")[-2000:]
        return classified

    if instance.get("install_config"):
        from minisweagent.run.benchmarks.swerebench_verify_azure_modal import (
            END_TEST_OUTPUT,
            START_TEST_OUTPUT,
            _upload_file,
            build_eval_script,
        )
        official_kind = "swe_rebench"
    else:
        from minisweagent.run.benchmarks.swebench_verify_azure_modal_cli import (
            END_TEST_OUTPUT,
            START_TEST_OUTPUT,
            _upload_file,
            build_eval_script,
        )
        official_kind = "swebench"

    official_instance = copy.deepcopy(instance)
    official_instance["test_patch"] = test_patch.rstrip("\n") + "\n"
    logical_command = f"cd /testbed && {test_command}"
    build_kwargs = {"test_command_override": test_command}
    if official_kind == "swe_rebench":
        build_kwargs["environment_prepared"] = skip_install
    else:
        build_kwargs["skip_install"] = skip_install
    eval_script = build_eval_script(official_instance, **build_kwargs)
    _upload_file(env, "/tmp/test_patch.diff", official_instance["test_patch"], timeout=60)
    _upload_file(env, "/tmp/eval.sh", eval_script, timeout=30)
    output = shell(env, "bash /tmp/eval.sh 2>&1", timeout=timeout)
    full_output = str(output.get("output") or "")
    test_output = full_output
    if START_TEST_OUTPUT in full_output and END_TEST_OUTPUT in full_output:
        test_output = full_output.split(START_TEST_OUTPUT, 1)[1].split(END_TEST_OUTPUT, 1)[0]
    classified = classify_generated_test_output(logical_command, {**output, "output": test_output})
    classified["official_verify_format"] = True
    classified["official_verifier_kind"] = official_kind
    classified["official_output_tail"] = full_output[-2000:]
    return classified


def run_shared_test_patch_official(
    env,
    instance: dict,
    test_patch: str,
    test_command: str,
    timeout: int,
    *,
    skip_install: bool,
    restore_agent_workspace: bool,
) -> dict:
    """Run official verification transactionally in the agent sandbox.

    No model action may run concurrently with this function. The worktree is reset to the
    immutable dataset base before verification. While the agent is still active, its submitted
    patch is restored afterwards; verifier-only files are always removed before returning.
    """
    restore_path = "/tmp/gentest_shared_restore.diff"
    verifier_paths = (restore_path, "/tmp/test_patch.diff", "/tmp/eval.sh")
    base_commit = str(
        instance.get("_persistent_baseline_commit")
        or instance.get("base_commit")
        or instance.get("parent_commit")
        or "HEAD"
    )
    reset_command = (
        f"cd /testbed && git reset --hard {shlex.quote(base_commit)} "
        "&& git clean -fdq"
    )
    workspace_reset_attempted = False
    verify_error: Exception | None = None
    verify_traceback = None
    classified = None
    restoration_errors = []

    try:
        if restore_agent_workspace:
            write_file(env, restore_path, test_patch, timeout=60)
            uploaded = shell(env, f"test -s {shlex.quote(restore_path)}", timeout=30)
            if uploaded.get("returncode") != 0:
                raise RuntimeError("could not persist the agent patch before shared verification")

        workspace_reset_attempted = True
        reset_result = shell(env, reset_command, timeout=120)
        if reset_result.get("returncode") != 0:
            raise RuntimeError(
                "shared verifier could not reset to the dataset base: "
                f"{str(reset_result.get('output') or '')[-1000:]}"
            )
        classified = run_test_patch_official(
            env,
            instance,
            test_patch,
            test_command,
            timeout,
            skip_install=skip_install,
        )
    except Exception as exc:  # noqa: BLE001
        verify_error = exc
        verify_traceback = exc.__traceback__
    finally:
        if workspace_reset_attempted:
            try:
                reset_result = shell(env, reset_command, timeout=120)
                if reset_result.get("returncode") != 0:
                    restoration_errors.append(
                        "post-verify reset failed: " + str(reset_result.get("output") or "")[-1000:]
                    )
                elif restore_agent_workspace:
                    apply_result = shell(
                        env,
                        f"cd /testbed && git apply --whitespace=nowarn {shlex.quote(restore_path)}",
                        timeout=120,
                    )
                    if apply_result.get("returncode") != 0:
                        restoration_errors.append(
                            "agent patch restore failed: "
                            + str(apply_result.get("output") or "")[-1000:]
                        )
            except Exception as exc:  # noqa: BLE001
                restoration_errors.append(
                    f"agent workspace restoration raised {type(exc).__name__}: {exc}"
                )
        try:
            cleanup_result = shell(
                env,
                "rm -f " + " ".join(shlex.quote(path) for path in verifier_paths),
                timeout=30,
            )
            if cleanup_result.get("returncode") != 0:
                restoration_errors.append(
                    "verifier context cleanup failed: "
                    + str(cleanup_result.get("output") or "")[-1000:]
                )
        except Exception as exc:  # noqa: BLE001
            restoration_errors.append(
                f"verifier context cleanup raised {type(exc).__name__}: {exc}"
            )

    if restoration_errors:
        if verify_error is not None:
            restoration_errors.insert(0, f"verify failed: {verify_error}")
        raise RuntimeError("shared verify transaction failed: " + "; ".join(restoration_errors))
    if verify_error is not None:
        raise verify_error.with_traceback(verify_traceback)
    return classified


def apply_patch_text(env, patch_text: str, *, timeout: int = 120) -> tuple[bool, str]:
    if not (patch_text or "").strip():
        return False, "empty patch"
    write_file(env, "/tmp/gentest_patch.diff", patch_text, timeout=30)
    errors = []
    for command in (
        "cd /testbed && git apply --verbose /tmp/gentest_patch.diff",
        "cd /testbed && git apply --verbose --reject /tmp/gentest_patch.diff",
        "cd /testbed && patch --batch --fuzz=5 -p1 -i /tmp/gentest_patch.diff",
    ):
        out = shell(env, command, timeout=timeout)
        text = (out.get("output") or "") + (out.get("exception_info") or "")
        if (out.get("returncode") or 0) == 0:
            return True, ""
        errors.append(text[-800:])
    return False, "\n".join(errors)[-1600:]


def apply_test_patch_for_alignment(env, instance: dict, *, enabled: bool, phase: str) -> dict:
    status = {
        "enabled": bool(enabled),
        "phase": phase,
        "has_test_patch": bool((instance.get("test_patch") or "").strip()),
        "changed_paths": _changed_paths_from_patch(instance.get("test_patch", "") or ""),
        "test_paths": _changed_test_paths(instance),
        "applied": False,
        "reason": "",
        "error": "",
    }
    if not enabled:
        return status
    test_patch = instance.get("test_patch", "") or ""
    if not test_patch.strip():
        status["reason"] = "missing_test_patch"
        return status
    applied, apply_error = apply_patch_text(env, test_patch)
    status["applied"] = applied
    if not applied:
        status["reason"] = "test_patch_apply_failure"
        status["error"] = (apply_error or "")[-1600:]
    return status


def is_clean_base_failure(result: dict) -> bool:
    return bool(result.get("clean_fail"))


def infra_failure_info(result: dict) -> dict:
    return {
        "infra_failure": bool(result.get("infra_failure")),
        "infra_reason": result.get("infra_reason") or "",
        "failure_category": result.get("failure_category") or "",
        "next_action": result.get("next_action") or "",
        "missing_module": result.get("missing_module") or "",
        "missing_symbol": result.get("missing_symbol") or "",
        "missing_fixture": result.get("missing_fixture") or "",
        "output_log_path": result.get("output_log_path") or "",
    }


def _split_f2p_node(node: str) -> tuple[str, str, str]:
    """Split a FAIL_TO_PASS node into (path, class_name, test_name) as far as the format allows."""
    text = str(node or "").strip()
    if not text:
        return "", "", ""
    django = re.match(r"^(?P<test>[\w.]+)\s+\((?P<dotted>[\w.]+)\)$", text)
    if django:
        dotted = django.group("dotted").split(".")
        return "", dotted[-1] if dotted else "", django.group("test")
    parts = [part for part in text.split("::") if part]
    if len(parts) == 1:
        return "", "", parts[0]
    path = parts[0] if parts[0].endswith(".py") else ""
    test_name = parts[-1].split("[", 1)[0]
    class_name = parts[-2] if len(parts) >= 3 else ""
    return path, class_name, test_name


def gold_agreement(
    instance: dict | None,
    generated_test_paths: list[str] | None,
    generated_test_names: list[str] | None,
) -> dict:
    """Read-only agreement between a generated test and the dataset's gold F2P test patch.

    `generated_test_names` may be plain or `Class.test_name` qualified. Nothing here gates or
    scores anything; it records how close the generated test lands to where the maintainer
    actually put the regression test.
    """
    gold_paths = _changed_test_paths(instance) or [
        path
        for path in _changed_paths_from_patch((instance or {}).get("_gentest_gold_test_patch", ""))
        if not path.endswith(NON_TEST_EXTS)
    ]
    gen_paths = [_normalize_test_file_path(p) for p in (generated_test_paths or []) if str(p or "").strip()]
    gold_path_set = {_normalize_test_file_path(p) for p in gold_paths}
    path_overlap = sorted(gold_path_set.intersection(gen_paths))

    nodes = [str(n) for n in list_from_json_or_obj((instance or {}).get("FAIL_TO_PASS"))]
    split_nodes = [_split_f2p_node(node) for node in nodes]
    gold_classes = {cls for _, cls, _ in split_nodes if cls}
    gold_tests = {test for _, _, test in split_nodes if test}

    gen_classes, gen_tests = set(), set()
    for name in generated_test_names or []:
        text = str(name or "").strip()
        if not text:
            continue
        head, _, tail = text.rpartition(".")
        if head:
            gen_classes.add(head.split(".")[-1])
        gen_tests.add(tail or text)

    return {
        "gold_test_paths": sorted(gold_path_set),
        "generated_test_paths": sorted(set(gen_paths)),
        "path_overlap": path_overlap,
        "path_match": bool(path_overlap),
        "gold_f2p_nodes": nodes[:20],
        "gold_f2p_classes": sorted(gold_classes),
        "gold_f2p_test_names": sorted(gold_tests),
        "generated_test_classes": sorted(gen_classes),
        "generated_test_names": sorted(gen_tests),
        "class_overlap": sorted(gold_classes.intersection(gen_classes)),
        "class_match": bool(gold_classes.intersection(gen_classes)),
        "test_name_overlap": sorted(gold_tests.intersection(gen_tests)),
        "test_name_match": bool(gold_tests.intersection(gen_tests)),
    }


def _submitted_test_names(info_gate: dict | None, code: str) -> list[str]:
    """Return test names from the submitted gate, falling back to source analysis."""
    shape = (info_gate or {}).get("test_shape")
    if isinstance(shape, dict):
        names = shape.get("all_test_names") or shape.get("test_names")
        if isinstance(names, list) and names:
            return [str(name) for name in names]
    return list(semantic_summary(code).get("tests") or [])


def validate_generated_test(
    env,
    instance: dict,
    code: str,
    *,
    filename: str,
    test_command: str | None = None,
    test_timeout: int,
    gold_eval: bool,
    oracle: dict | None = None,
    apply_test_patch_at_start: bool = False,
    official_verify_format: bool = False,
    official_skip_install: bool | None = None,
    base_result: dict | None = None,
    base_test_patch: dict | None = None,
    generated_test_names: list[str] | None = None,
) -> dict:
    validation_started = time.perf_counter()
    semantic = semantic_summary(code)
    quality_flags = static_quality_flags(code, semantic)
    oracle_quality = evaluate_oracle_contract(oracle, code)
    validation = {
        "has_test": bool((code or "").strip()),
        "test_code": code or "",
        "test_file_path": _normalize_test_file_path(filename),
        "test_command": str(test_command or "").strip(),
        "base_fail_gold_pass": False,
        "semantic": semantic,
        "static_analysis_scope": "full_file",
        "quality_flags": quality_flags,
        "oracle_quality": oracle_quality,
        "oracle_quality_flags": oracle_quality.get("quality_flags") or [],
        "oracle_grounding_tags": oracle_quality.get("grounding_tags") or [],
        "gold_agreement": gold_agreement(
            instance,
            [filename],
            generated_test_names if generated_test_names is not None else semantic.get("tests") or [],
        ),
        "test_patch": {"enabled": bool(apply_test_patch_at_start)},
        "timings": {"base_reused": base_result is not None},
    }

    def finish() -> dict:
        validation["timings"]["total_sec"] = time.perf_counter() - validation_started
        return validation

    if not validation["has_test"]:
        validation["label"] = "missing_test"
        return finish()

    base_started = time.perf_counter()
    if base_result is not None:
        expected_hash = hashlib.sha256((code or "").encode("utf-8", "replace")).hexdigest()
        if base_result.get("test_hash") != expected_hash:
            raise ValueError("reused base self-test does not match the generated test hash")
        if test_command:
            expected_execution_hash = execution_contract_hash(code, filename, test_command)
            if base_result.get("execution_hash") != expected_execution_hash:
                raise ValueError("reused base self-test does not match the generated execution contract")
        base = copy.deepcopy(base_result)
        validation["test_patch"]["base"] = copy.deepcopy(
            base_test_patch
            or {
                "enabled": bool(apply_test_patch_at_start),
                "phase": "base",
                "applied": bool(apply_test_patch_at_start),
                "reason": "",
                "error": "",
                "reused": True,
            }
        )
    else:
        reset_base(env)
        if official_verify_format:
            applied_base_test_patch = {
                "enabled": bool(apply_test_patch_at_start),
                "phase": "base",
                "applied": bool(apply_test_patch_at_start),
                "reason": "",
                "error": "",
                "official_verify_format": True,
            }
        else:
            applied_base_test_patch = apply_test_patch_for_alignment(
                env,
                instance,
                enabled=apply_test_patch_at_start,
                phase="base",
            )
        validation["test_patch"]["base"] = applied_base_test_patch
        if applied_base_test_patch["reason"]:
            validation["label"] = f"base_{applied_base_test_patch['reason']}"
            validation["infra_failure"] = True
            validation["infra_failure_phase"] = "base_test_patch"
            validation["timings"]["base_sec"] = time.perf_counter() - base_started
            return finish()
        if official_verify_format:
            base = run_generated_test_official(
                env,
                instance,
                filename,
                code,
                test_timeout,
                test_command=test_command,
                include_dataset_test_patch=apply_test_patch_at_start,
                skip_install=official_skip_install,
            )
        else:
            write_generated_test_file(env, filename, code)
            run_kwargs = {"test_command": test_command} if test_command else {}
            base = run_generated_test(env, instance, filename, test_timeout, **run_kwargs)
    validation["timings"]["base_sec"] = time.perf_counter() - base_started
    validation["base"] = base
    validation["base_clean_fail"] = is_clean_base_failure(base)
    validation["base"].update(infra_failure_info(base))

    if gold_eval:
        gold_started = time.perf_counter()
        try:
            # Always reset to a clean base checkout before applying the gold
            # patch: the base-verify phase leaves the generated test file (and
            # other churn) in /testbed, which makes git apply of the gold patch
            # fail on any overlapping path (verified on Scale-SWE: same patch
            # applies cleanly on a fresh parent commit). The generated test is
            # re-written below before the gold run.
            reset_base(env)
            applied, apply_error = apply_patch_text(env, instance.get("patch", "") or "")
            validation["gold_applied"] = applied
            if applied:
                if official_verify_format:
                    gold_test_patch = {
                        "enabled": bool(apply_test_patch_at_start),
                        "phase": "gold",
                        "applied": bool(apply_test_patch_at_start),
                        "reason": "",
                        "error": "",
                        "official_verify_format": True,
                    }
                else:
                    gold_test_patch = apply_test_patch_for_alignment(
                        env,
                        instance,
                        enabled=apply_test_patch_at_start,
                        phase="gold",
                    )
                validation["test_patch"]["gold"] = gold_test_patch
                if gold_test_patch["reason"]:
                    validation["label"] = f"gold_{gold_test_patch['reason']}"
                    validation["gold_pass"] = False
                    validation["infra_failure"] = True
                    validation["infra_failure_phase"] = "gold_test_patch"
                    return finish()
                if official_verify_format:
                    gold = run_generated_test_official(
                        env,
                        instance,
                        filename,
                        code,
                        test_timeout,
                        test_command=test_command,
                        include_dataset_test_patch=apply_test_patch_at_start,
                        skip_install=official_skip_install,
                    )
                else:
                    write_generated_test_file(env, filename, code)
                    run_kwargs = {"test_command": test_command} if test_command else {}
                    gold = run_generated_test(env, instance, filename, test_timeout, **run_kwargs)
                validation["gold"] = gold
                validation["gold"].update(infra_failure_info(gold))
                validation["gold_pass"] = gold.get("verdict") == "pass"
            else:
                validation["gold_apply_error"] = apply_error
                validation["gold_apply_infra_failure"] = True
                validation["gold_pass"] = False
        finally:
            validation["timings"]["gold_sec"] = time.perf_counter() - gold_started
    else:
        validation["gold_applied"] = None
        validation["gold_pass"] = None

    if validation["base"].get("infra_failure"):
        base_validation_label = "base_infra_failure"
    elif not validation["base_clean_fail"]:
        base_validation_label = "base_not_clean_fail"
    elif quality_flags:
        base_validation_label = "base_fail_with_quality_flags"
    else:
        base_validation_label = "base_clean_fail"
    validation["base_validation_label"] = base_validation_label

    gold_validated_base_labels = {
        "base_clean_fail",
        "base_fail_with_quality_flags",
        "base_infra_failure",
    }
    validation["gold_validated"] = bool(
        validation.get("gold_pass") is True and base_validation_label in gold_validated_base_labels
    )
    validation["base_fail_gold_pass"] = bool(
        validation["base_clean_fail"] and validation.get("gold_pass") is True
    )

    if validation["gold_validated"]:
        validation["label"] = "gold_validated"
    elif validation.get("gold_apply_infra_failure"):
        validation["label"] = "gold_apply_failure"
    elif isinstance(validation.get("gold"), dict) and validation["gold"].get("infra_failure"):
        validation["label"] = "gold_infra_failure"
    elif base_validation_label != "base_clean_fail":
        validation["label"] = base_validation_label
    elif validation.get("gold_pass") is False:
        validation["label"] = "gold_failed"
    else:
        validation["label"] = "base_fail_only"
    validation["infra_failure"] = bool(
        base_validation_label == "base_infra_failure"
        or validation["label"] in {"gold_apply_failure", "gold_infra_failure"}
    )
    validation["infra_failure_phase"] = (
        "base"
        if base_validation_label == "base_infra_failure"
        else "gold_apply"
        if validation["label"] == "gold_apply_failure"
        else "gold"
        if validation["label"] == "gold_infra_failure"
        else ""
    )
    return finish()


def semantic_summary(code: str) -> dict:
    out = {"parse_ok": False, "tests": [], "assert_count": 0, "source_inspection": False}
    try:
        tree = ast.parse(code or "")
    except SyntaxError as exc:
        out["parse_error"] = str(exc)[:200]
        return out
    out["parse_ok"] = True
    source_names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            out["tests"].append(node.name)
        elif isinstance(node, ast.Assert):
            out["assert_count"] += 1
        elif isinstance(node, ast.Call) and _call_name(node.func).split(".")[-1].startswith("assert"):
            out["assert_count"] += 1
        if isinstance(node, ast.Assign) and _looks_like_source_read(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    source_names.add(target.id)
    out["source_inspection"] = _assert_uses_names(tree, source_names)
    return out


def static_quality_flags(code: str, semantic: dict | None = None) -> list[str]:
    flags: set[str] = set()
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return ["syntax_error"]
    semantic = semantic or semantic_summary(code)
    if not semantic.get("tests"):
        flags.add("no_test_function")
    if not semantic.get("assert_count"):
        flags.add("no_assertion")
    if semantic.get("source_inspection"):
        flags.add("source_file_inspection")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            leaf = name.split(".")[-1]
            if leaf in {"fail"}:
                flags.add("pytest_fail_oracle")
            if leaf in {"skip", "xfail"}:
                flags.add("skip_or_xfail")
        elif isinstance(node, ast.With):
            for item in node.items:
                if _is_broad_exception_context(item.context_expr):
                    flags.add("broad_exception_oracle")
    return sorted(flags)


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _looks_like_source_read(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    name = _call_name(node.func)
    if name.endswith("read_text") or name in {"open", "Path", "pathlib.Path"}:
        return True
    return any(
        isinstance(arg, ast.Constant)
        and isinstance(arg.value, str)
        and arg.value.endswith((".py", ".js", ".ts", ".java", ".go", ".rs"))
        for arg in node.args
    )


def _assert_uses_names(tree: ast.AST, names: set[str]) -> bool:
    if not names:
        return False
    for node in ast.walk(tree):
        exprs = []
        if isinstance(node, ast.Assert):
            exprs.append(node.test)
        elif isinstance(node, ast.Call) and _call_name(node.func).split(".")[-1].startswith("assert"):
            exprs.extend(node.args)
        for expr in exprs:
            if any(isinstance(child, ast.Name) and child.id in names for child in ast.walk(expr)):
                return True
    return False


def _is_broad_exception_context(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    name = _call_name(node.func)
    if name not in {"raises", "pytest.raises"} and not name.endswith(".assertRaises"):
        return False
    return bool(node.args and _contains_broad_exception(node.args[0]))


def _contains_broad_exception(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id in {"Exception", "BaseException"}
    if isinstance(node, ast.Attribute):
        return node.attr in {"Exception", "BaseException"}
    if isinstance(node, ast.Tuple):
        return any(_contains_broad_exception(item) for item in node.elts)
    return False


def generated_test_runner_for_instance(instance: dict | None = None, filename: str = TEST_FILE) -> dict:
    base_cmd, source = _base_test_cmd_for_instance(instance)
    kind = _runner_kind(base_cmd)
    target = _target_path_for_generated_file(instance, filename)
    source_file = _normalize_test_file_path(filename)
    module = _django_module_from_path(target)
    directive = module if kind == "django" else target
    if kind == "pytest":
        base_cmd = _ensure_pytest_color_off(base_cmd)
    prep_commands = _prep_commands_for_target(source_file, target)
    runner_command = _format_runner_command(base_cmd, target=target, directive=directive, module=module)
    return {
        "command": " && ".join([*prep_commands, runner_command]),
        "runner_command": runner_command,
        "prep_commands": prep_commands,
        "kind": kind,
        "target": target,
        "directive": directive,
        "base_cmd": base_cmd,
        "source": source,
    }


def test_command_for_instance(instance: dict | None = None, filename: str = TEST_FILE) -> str:
    return generated_test_runner_for_instance(instance, filename)["command"]


def test_style_guidance_for_instance(instance: dict | None = None, filename: str = TEST_FILE) -> str:
    runner = generated_test_runner_for_instance(instance, filename)
    note = _mirror_note(runner, filename)
    if runner["kind"] == "django":
        return (
            "The harness runs Django's unittest runner. Write unittest/Django TestCase or SimpleTestCase "
            f"classes with test_* methods; bare pytest functions may not be collected.{note}"
        )
    if runner["kind"] == "sympy":
        return f"The harness runs SymPy's bin/test runner. Avoid pytest-only fixtures.{note}"
    return f"Write a pytest-style test_* function in the generated file.{note}"


def _install_config(instance: dict | None = None) -> dict:
    install_config = (instance or {}).get("install_config") or {}
    if isinstance(install_config, str):
        try:
            parsed = json.loads(install_config)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return install_config if isinstance(install_config, dict) else {}


def _coerce_test_cmd(value) -> str:
    if not value:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return " ".join(str(v).strip() for v in value if str(v).strip()).strip()
    return str(value).strip()


def _changed_paths_from_patch(test_patch: str) -> list[str]:
    paths = []
    for line in (test_patch or "").splitlines():
        if line.startswith("+++ b/"):
            path = line[6:].strip()
            if path and path != "/dev/null":
                paths.append(path)
    return paths


def _changed_test_paths(instance: dict | None = None) -> list[str]:
    return [p for p in _changed_paths_from_patch((instance or {}).get("test_patch", "")) if not p.endswith(NON_TEST_EXTS)]


def _changed_python_test_paths(instance: dict | None = None) -> list[str]:
    return [p for p in _changed_test_paths(instance) if p.endswith(".py")]


def _looks_like_django_unittest_node(node) -> bool:
    return bool(re.search(r"\s+\([A-Za-z_][^)]*\.[^)]*\)$", str(node or "")))


def _infer_runner_style(instance: dict | None = None) -> str:
    nodes = [str(n) for n in list_from_json_or_obj((instance or {}).get("FAIL_TO_PASS"))]
    if any(_looks_like_django_unittest_node(n) for n in nodes):
        return "django_unittest"
    py_paths = _changed_python_test_paths(instance)
    if any(p.startswith("sympy/") and "/tests/" in p for p in py_paths) and not any("::" in n for n in nodes):
        return "sympy_bin_test"
    return "pytest"


def _swebench_spec_test_cmd(instance: dict | None = None) -> str:
    repo = (instance or {}).get("repo")
    version = (instance or {}).get("version")
    if not repo or version is None:
        return ""
    try:
        from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
    except Exception:
        return ""
    specs = (MAP_REPO_VERSION_TO_SPECS.get(repo) or {}).get(version) or (
        MAP_REPO_VERSION_TO_SPECS.get(repo) or {}
    ).get(str(version)) or {}
    return _coerce_test_cmd(specs.get("test_cmd"))


def _base_test_cmd_for_instance(instance: dict | None = None) -> tuple[str, str]:
    cmd = _coerce_test_cmd((instance or {}).get("_gentest_test_command_override"))
    if cmd:
        return cmd, "gentest.test_command_override"
    cmd = _coerce_test_cmd(_install_config(instance).get("test_cmd"))
    if cmd:
        return cmd, "install_config.test_cmd"
    cmd = _swebench_spec_test_cmd(instance)
    if cmd:
        return cmd, "swebench_specs.test_cmd"
    style = _infer_runner_style(instance)
    return FALLBACK_TEST_CMD_BY_STYLE.get(style, DEFAULT_GENERATED_PYTEST_CMD), f"inferred:{style}"


def _runner_kind(base_cmd: str) -> str:
    text = (base_cmd or "").lower()
    if "runtests.py" in text:
        return "django"
    if re.search(r"(^|[;&|()\s])bin/test(\s|$)", text):
        return "sympy"
    return "pytest"


def _target_path_for_generated_file(instance: dict | None = None, filename: str = TEST_FILE) -> str:
    instance = instance or {}
    name = _normalize_test_file_path(filename).rsplit("/", 1)[-1]
    changed = _changed_python_test_paths(instance)
    if changed and "/" in changed[0]:
        return f"{changed[0].rsplit('/', 1)[0]}/{name}"

    base_cmd, _ = _base_test_cmd_for_instance(instance)
    if _runner_kind(base_cmd) != "django":
        return name

    changed_test_dirs = {
        path.rsplit("/", 1)[0]
        for path in _changed_paths_from_patch(instance.get("test_patch", ""))
        if path.startswith("tests/") and "/" in path
    }
    if len(changed_test_dirs) == 1:
        return f"{changed_test_dirs.pop()}/{name}"

    django_apps = set()
    for node in list_from_json_or_obj(instance.get("FAIL_TO_PASS")):
        match = re.search(r"\(([^()]+)\)$", str(node))
        if not match:
            continue
        app = match.group(1).split(".", 1)[0]
        if re.fullmatch(r"[A-Za-z_]\w*", app):
            django_apps.add(app)
    if len(django_apps) == 1:
        return f"tests/{django_apps.pop()}/{name}"
    return name


def generated_test_file_for_instance(instance: dict | None = None, filename: str = TEST_FILE) -> str:
    return _target_path_for_generated_file(instance, filename)


def _django_module_from_path(path: str) -> str:
    module = path[:-3] if path.endswith(".py") else path
    if module.startswith("tests/"):
        module = module[len("tests/") :]
    return module.replace("/", ".")


def _prep_commands_for_target(source: str, target: str) -> list[str]:
    if target == source:
        return []
    parent = target.rsplit("/", 1)[0] if "/" in target else ""
    commands = []
    if parent:
        commands.append(f"mkdir -p {shlex.quote(parent)}")
    commands.append(f"cp {shlex.quote(source)} {shlex.quote(target)}")
    return commands


def _ensure_pytest_color_off(base_cmd: str) -> str:
    if "pytest" not in (base_cmd or "").lower() or "--color" in base_cmd:
        return base_cmd
    return f"{base_cmd} --color=no"


def _format_runner_command(base_cmd: str, *, target: str, directive: str, module: str) -> str:
    values = {
        "test_file": target,
        "test_target": directive,
        "test_module": module,
        "generated_test": target,
    }
    if any(f"{{{name}}}" in base_cmd for name in values):
        return base_cmd.format(**values)
    return f"{base_cmd} {shlex.quote(directive)}".strip()


def _mirror_note(runner: dict, filename: str) -> str:
    source = _normalize_test_file_path(filename)
    if runner["target"] == source:
        return ""
    return f" The harness copies /testbed/{source} to /testbed/{runner['target']} before running tests."


def reconstruct_added_files(test_patch: str) -> dict[str, list[str]]:
    files: dict[str, list[str]] = {}
    current = None
    for line in (test_patch or "").splitlines():
        match = re.match(r"^\+\+\+ b/(.+)$", line)
        if match:
            current = match.group(1).strip()
            files.setdefault(current, [])
            continue
        if current is not None and line.startswith("+") and not line.startswith("+++"):
            files[current].append(line[1:])
    return files


def extract_example(test_patch: str, node: str, mode: str) -> tuple[str, str]:
    if mode == "none" or not test_patch:
        return node or "(none)", ""
    files = reconstruct_added_files(test_patch)
    path = (node or "").split("::")[0]
    test_name = (node or "").split("::")[-1] if "::" in (node or "") else ""
    lines = files.get(path)
    if not lines and files:
        path, lines = next(iter(files.items()))
    if not lines:
        return node or path or "(none)", ""
    full = "\n".join(lines)
    if mode == "full" or not test_name:
        return node or path, full[:8000]
    start = next((i for i, line in enumerate(lines) if re.match(rf"\s*(async\s+)?def {re.escape(test_name)}\b", line)), None)
    if start is None:
        return node or path, full[:8000]
    base_indent = len(lines[start]) - len(lines[start].lstrip())
    end = len(lines)
    for idx in range(start + 1, len(lines)):
        line = lines[idx]
        indent = len(line) - len(line.lstrip())
        if line.strip() and indent <= base_indent and re.match(r"\s*(def |class |@)", line):
            end = idx
            break
    imports = [line for line in lines[:start] if re.match(r"\s*(import |from )", line)]
    return node or path, "\n".join(imports + ([""] if imports else []) + lines[start:end])[:8000]


def extract_import_lines(source: str, *, max_imports: int = 20) -> list[str]:
    source = source or ""
    imports: list[str] = []
    try:
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant):
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(ast.unparse(node).strip())
                continue
            if imports:
                break
        if imports:
            return _dedupe_imports(imports[:max_imports])
    except SyntaxError:
        pass

    for raw in source.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if IMPORT_LINE.match(raw):
            imports.append(re.sub(r"\s+", " ", line))
            if len(imports) >= max_imports:
                break
            continue
        if imports:
            break
    return _dedupe_imports(imports)


def _dedupe_imports(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        clean = (line or "").strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
    return out


def _candidate_gold_test_path(instance: dict, example_node: str | None) -> str:
    path = (example_node or "").split("::")[0].strip()
    if path.endswith(".py"):
        return path
    changed = _changed_python_test_paths(instance)
    return changed[0] if changed else ""


def current_test_import_starter(env, instance: dict, example_node: str | None) -> tuple[list[str], str]:
    path = _candidate_gold_test_path(instance, example_node)
    if not path:
        return [], "current_file:none"
    try:
        imports = extract_import_lines(read_repo_file(env, path))
    except Exception as exc:  # noqa: BLE001
        return [], f"current_file_error:{path}:{type(exc).__name__}:{str(exc)[:160]}"
    return imports, f"current_file:{path}:imports={len(imports)}"


def gold_test_import_starter(env, instance: dict, example_node: str | None) -> tuple[str, list[str], str]:
    imports: list[str] = []
    status = "no_test_patch"
    test_patch = instance.get("test_patch", "") or ""
    path = _candidate_gold_test_path(instance, example_node)

    if path and test_patch.strip():
        try:
            reset_base(env)
            applied, apply_error = apply_patch_text(env, test_patch)
            if applied:
                imports = extract_import_lines(read_repo_file(env, path))
                status = f"gold_file:{path}:imports={len(imports)}"
            else:
                status = f"test_patch_apply_failed:{(apply_error or '')[-160:]}"
        except Exception as exc:  # noqa: BLE001
            status = f"gold_file_error:{type(exc).__name__}:{str(exc)[:160]}"
        finally:
            try:
                reset_base(env)
            except Exception:
                pass

    if not imports and test_patch.strip():
        try:
            node = example_node or path or ""
            _, example_src = extract_example(test_patch, node, "full")
            imports = extract_import_lines(example_src)
            if imports:
                status = f"test_patch_added_lines:imports={len(imports)}"
            elif status == "no_test_patch":
                status = "test_patch_added_lines:imports=0"
        except Exception as exc:  # noqa: BLE001
            if status == "no_test_patch":
                status = f"test_patch_added_lines_error:{type(exc).__name__}:{str(exc)[:160]}"

    starter = ("\n".join(imports) + "\n\n") if imports else ""
    return starter, imports, status


def scaffold_starter_for_runner(runner: dict, imports: list[str]) -> tuple[list[str], str, str]:
    kind = runner.get("kind")
    if kind == "django":
        imports = _ensure_django_testcase_import(imports)
        base_class = _django_testcase_base(imports)
        scaffold = (
            f"class GeneratedRegressionTest({base_class}):\n"
            "    def test_regression(self):\n"
            "        pass\n"
        )
        return imports, scaffold, f"django_unittest:{base_class}"
    if kind in {"pytest", "sympy"}:
        return imports, "def test_regression():\n    pass\n", f"{kind}:function"
    return imports, "", "none"


def _ensure_django_testcase_import(imports: list[str]) -> list[str]:
    if any(re.search(r"\b(SimpleTestCase|TestCase|TransactionTestCase)\b", line) for line in imports):
        return imports
    return _dedupe_imports(["from django.test import SimpleTestCase", *imports])


def _django_testcase_base(imports: list[str]) -> str:
    text = "\n".join(imports)
    if re.search(r"\bTransactionTestCase\b", text):
        return "TransactionTestCase"
    if re.search(r"\bTestCase\b", text):
        return "TestCase"
    return "SimpleTestCase"


def starter_code_for_sample(env, instance: dict, opts: dict, example_node: str | None) -> tuple[str, list[str], str]:
    starter = opts.get("starter", "gold_imports")
    if starter == "runner_scaffold":
        test_file = generated_test_file_for_instance(instance, opts.get("test_file", TEST_FILE))
        runner = generated_test_runner_for_instance(instance, test_file)
        if runner["kind"] != "django":
            return "", [], f"runner_scaffold:{runner['kind']}:disabled"
        imports = ["from django.test import TestCase"]
        scaffold = (
            "class GeneratedRegressionTest(TestCase):\n"
            "    def test_regression(self):\n"
            "        pass\n"
        )
        return (
            "\n".join(imports) + "\n\n" + scaffold,
            imports,
            "runner_scaffold:django_unittest:TestCase",
        )
    if starter == "gold_imports":
        current_imports, current_status = current_test_import_starter(env, instance, example_node)
        _, gold_imports, gold_status = gold_test_import_starter(env, instance, example_node)
        imports = _dedupe_imports([*current_imports, *gold_imports])
        test_file = generated_test_file_for_instance(instance, opts.get("test_file", TEST_FILE))
        imports, scaffold, scaffold_status = scaffold_starter_for_runner(
            generated_test_runner_for_instance(instance, test_file),
            imports,
        )
        starter_text = "\n".join(imports)
        if starter_text and scaffold:
            starter_text += "\n\n"
        starter_text += scaffold
        if starter_text and not starter_text.endswith("\n"):
            starter_text += "\n"
        status = f"{current_status};{gold_status};scaffold:{scaffold_status}"
        return starter_text, imports, status
    return "", [], "disabled"


def load_ids(path: Path | None, *, limit: int = 0) -> list[str] | None:
    if path is None:
        return None
    if path.suffix == ".jsonl":
        ids = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            iid = row.get("instance_id") or (row.get("metadata") or {}).get("instance_id")
            if iid:
                ids.append(iid)
    else:
        data = json.loads(path.read_text())
        ids = [str(x.get("instance_id") if isinstance(x, dict) else x) for x in data]
    return ids[:limit] if limit else ids


def existing_keys(results_jsonl: Path) -> set[tuple[str, int]]:
    if not results_jsonl.exists():
        return set()
    seen: set[tuple[str, int]] = set()
    for line in results_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            seen.add((row["instance_id"], int(row.get("sample_idx", 0))))
        except Exception:
            continue
    return seen


def sample_error_record(instance: dict, sample_idx: int, opts: dict, error: str, traceback_text: str = "") -> dict:
    requested_test_file = opts.get("test_file", TEST_FILE)
    record = {
        "instance_id": instance.get("instance_id", ""),
        "sample_idx": int(sample_idx),
        "test_filename": (
            _normalize_test_file_path(requested_test_file)
            if opts.get("execution_contract_mode")
            else generated_test_file_for_instance(instance, requested_test_file)
        ),
        "requested_test_filename": requested_test_file,
        "error": error,
        "traceback": traceback_text,
    }
    if int(opts.get("sample_retries") or 0) > 0:
        retry_attempt = int(opts.get("retry_attempt") or 0)
        record["retry_attempt"] = retry_attempt
        if opts.get("output_dir"):
            instance_id = instance.get("instance_id", "")
            retry_suffix = f".retry_{retry_attempt}" if retry_attempt else ""
            record["trajectory_path"] = str(
                Path(opts["output_dir"])
                / instance_id
                / f"{instance_id}.sample_{int(sample_idx)}{retry_suffix}.traj.json"
            )
    return record


def run_sample_in_env(instance: dict, config: dict, opts: dict, env, verify_env=None) -> dict:
    instance_id = instance["instance_id"]
    sample_idx = int(opts["sample_idx"])
    output_dir = Path(opts["output_dir"])
    requested_test_file = opts["test_file"]
    execution_contract_mode = bool((config.get("agent") or {}).get("require_execution_contract"))
    test_file = (
        _normalize_test_file_path(requested_test_file)
        if execution_contract_mode
        else generated_test_file_for_instance(instance, requested_test_file)
    )
    traj_dir = output_dir / instance_id
    retry_attempt = int(opts.get("retry_attempt") or 0)
    retry_suffix = f".retry_{retry_attempt}" if retry_attempt else ""
    traj_path = traj_dir / f"{instance_id}.sample_{sample_idx}{retry_suffix}.traj.json"
    traj_dir.mkdir(parents=True, exist_ok=True)

    agent = None
    record: dict[str, Any] = {
        "instance_id": instance_id,
        "sample_idx": sample_idx,
        "test_filename": test_file,
        "requested_test_filename": requested_test_file,
        "agent_inferred_execution_contract": execution_contract_mode,
        "prompt_mode": opts["prompt_mode"],
        "example_shot": opts["example_shot"],
        "starter": opts["starter"],
        "test_command_override": str(opts.get("test_command_override") or ""),
        "use_dataset_test_patch": bool(opts.get("use_dataset_test_patch", True)),
        "apply_test_patch_at_start": bool(opts.get("apply_test_patch_at_start")),
        "isolated_verify": verify_env is not None,
        "official_verify_format": verify_env is not None,
        "conda_env_name": conda_env_name(instance) or DEFAULT_CONDA_ENV,
        "conda_env_candidates": conda_env_candidates(instance),
    }
    if int(opts.get("sample_retries") or 0) > 0:
        record["retry_attempt"] = retry_attempt
        record["trajectory_path"] = str(traj_path)
    try:
        if execution_contract_mode and (
            opts.get("prompt_mode") != "zero_shot"
            or opts.get("example_shot") != "none"
            or opts.get("starter") != "empty"
            or opts.get("use_dataset_test_patch")
            or opts.get("apply_test_patch_at_start")
            or opts.get("test_command_override")
        ):
            raise ValueError(
                "agent-inferred execution-contract mode requires zero-shot, empty starter, no dataset test patch, "
                "and no test-command override"
            )
        if opts.get("reset_before_sample", True):
            reset_base(env)
        model = get_model(config=config.get("model", {}))

        chosen = ""
        example_node, example_src = "", ""
        if execution_contract_mode:
            starter_text, starter_imports, starter_status = "", [], "disabled:agent_inferred_execution_contract"
        else:
            all_f2p = list_from_json_or_obj(instance.get("FAIL_TO_PASS"))
            if all_f2p and (opts["prompt_mode"] == "one_shot_gold" or opts["starter"] == "gold_imports"):
                rng = random.Random(f"{opts['seed']}:{instance_id}:{sample_idx}")
                chosen = str(rng.choice(all_f2p))
            example_node, example_src = extract_example(
                instance.get("test_patch", "") or "",
                chosen,
                opts["example_shot"],
            )
            if opts["prompt_mode"] != "one_shot_gold":
                example_node, example_src = "", ""
            starter_text, starter_imports, starter_status = starter_code_for_sample(
                verify_env or env,
                instance,
                opts,
                chosen,
            )
        if verify_env is not None or execution_contract_mode:
            test_patch_start = {
                "enabled": bool(opts.get("apply_test_patch_at_start")),
                "phase": "generation",
                "applied": False,
                "reason": "",
                "error": "",
                "isolated_to_verifier": verify_env is not None,
            }
        else:
            test_patch_start = apply_test_patch_for_alignment(
                env,
                instance,
                enabled=bool(opts.get("apply_test_patch_at_start")),
                phase="generation",
            )
        record["test_patch_start"] = test_patch_start
        if test_patch_start["reason"]:
            record["error"] = f"{test_patch_start['phase']}_{test_patch_start['reason']}"
            if test_patch_start["error"]:
                record["test_patch_apply_error"] = test_patch_start["error"]
            return record
        if starter_text or not execution_contract_mode:
            write_generated_test_file(env, test_file, starter_text)
        record["starter_example_node"] = chosen
        record["starter_status"] = starter_status
        record["starter_imports"] = starter_imports
        record["starter_code_chars"] = len(starter_text)

        agent_config = copy.deepcopy(config.get("agent", {}))
        agent_config.setdefault("agent_class", "generated_test_submit")
        agent_config["output_path"] = traj_path
        agent_config["test_file"] = test_file
        allowed_generated_paths = [] if execution_contract_mode else [test_file]
        if verify_env is None and opts.get("apply_test_patch_at_start"):
            allowed_generated_paths.extend(_changed_paths_from_patch(instance.get("test_patch", "") or ""))
        agent_config["allowed_generated_paths"] = list(dict.fromkeys(allowed_generated_paths))
        if not execution_contract_mode:
            agent_config["self_test_command"] = f"cd /testbed && {test_command_for_instance(instance, test_file)}"
        agent_config["self_test_timeout"] = int(opts["test_timeout"])
        agent_config["initial_test_hash"] = hashlib.sha256(
            starter_text.encode("utf-8", "replace")
        ).hexdigest()
        if verify_env is not None:

            def run_isolated_self_test(test_code: str, selected_test_file: str, selected_test_command: str) -> dict:
                reset_base(verify_env)
                return run_generated_test_official(
                    verify_env,
                    instance,
                    selected_test_file if execution_contract_mode else test_file,
                    test_code,
                    int(opts["test_timeout"]),
                    test_command=selected_test_command if execution_contract_mode else None,
                    include_dataset_test_patch=bool(opts.get("apply_test_patch_at_start")),
                )

            agent_config["self_test_runner"] = run_isolated_self_test
        agent = get_agent(
            model,
            ActivatingEnv(env, activation_for_instance(instance)),
            agent_config,
            default_type="generated_test_submit",
        )
        info = agent.run(
            task=instance.get("problem_statement", ""),
            instance_id=instance_id,
            sample_idx=sample_idx,
            prompt_mode=opts["prompt_mode"],
            example_node=example_node,
            example_src=example_src,
            test_file=test_file,
            test_style_guidance=(
                "" if execution_contract_mode else test_style_guidance_for_instance(instance, test_file)
            ),
            starter_status=starter_status,
            starter_imports=starter_imports,
            starter_code=starter_text,
        )
        record["agent_exit_status"] = info.get("exit_status", "")
        record["agent_submission_chars"] = len(info.get("submission", "") or "")
        record["agent_calls"] = getattr(agent, "n_calls", 0)
        record["agent_cost"] = round(float(getattr(agent, "cost", 0.0)), 4)
        info_gate = info.get("gentest_gate") if isinstance(info.get("gentest_gate"), dict) else {}
        record["gate"] = getattr(agent, "gate_info", {})
        record["submitted_gate"] = info_gate
        record["example_node"] = example_node

        if execution_contract_mode:
            execution_contract, contract_error = validate_execution_contract(
                info_gate.get("test_file"),
                info_gate.get("test_command"),
                info_gate.get("command_evidence"),
            )
            if contract_error:
                record["error"] = f"missing_or_invalid_execution_contract: {contract_error}"
                return record
            selected_test_file = execution_contract["test_file_path"]
            selected_test_command = execution_contract["test_command"]
            command_evidence = execution_contract["command_evidence"]
        else:
            selected_test_file = test_file
            selected_test_command = ""
            command_evidence = ""
        record["test_filename"] = selected_test_file
        record["generated_test_path"] = selected_test_file
        record["generated_test_command"] = selected_test_command
        record["command_evidence"] = command_evidence
        code = read_test_file(env, selected_test_file) or info.get("submission", "")
        oracle_quality = info_gate.get("oracle_quality") if isinstance(info_gate.get("oracle_quality"), dict) else {}
        oracle_contract = oracle_quality.get("contract") if isinstance(oracle_quality.get("contract"), dict) else {}
        record["oracle"] = oracle_contract
        record["oracle_quality"] = evaluate_oracle_contract(oracle_contract, code)
        record["test_code"] = code
        validation_env = verify_env or env
        latest_self_test = (
            info_gate.get("latest_self_test")
            if isinstance(info_gate.get("latest_self_test"), dict)
            else {}
        )
        code_hash = hashlib.sha256(code.encode("utf-8", "replace")).hexdigest()
        selected_execution_hash = (
            execution_contract_hash(code, selected_test_file, selected_test_command)
            if execution_contract_mode and selected_test_command
            else ""
        )
        record["execution_hash"] = selected_execution_hash
        reused_base_result = (
            latest_self_test
            if verify_env is not None
            and latest_self_test.get("test_hash") == code_hash
            and (
                not execution_contract_mode
                or latest_self_test.get("execution_hash") == selected_execution_hash
            )
            else None
        )
        reused_base_test_patch = None
        if reused_base_result is not None:
            reused_base_test_patch = {
                "enabled": bool(opts.get("apply_test_patch_at_start")),
                "phase": "base",
                "applied": bool(opts.get("apply_test_patch_at_start")),
                "reason": "",
                "error": "",
                "isolated_to_verifier": True,
                "official_verify_format": True,
                "reused": True,
            }
        record["validation"] = validate_generated_test(
            validation_env,
            instance,
            code,
            filename=selected_test_file,
            test_command=selected_test_command or None,
            test_timeout=int(opts["test_timeout"]),
            gold_eval=bool(opts["gold_eval"]),
            oracle=oracle_contract,
            apply_test_patch_at_start=bool(opts.get("apply_test_patch_at_start")),
            official_verify_format=verify_env is not None,
            base_result=reused_base_result,
            base_test_patch=reused_base_test_patch,
            generated_test_names=_submitted_test_names(info_gate, code),
        )
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        record["traceback"] = traceback.format_exc()[-4000:]
    finally:
        try:
            if agent is not None:
                agent.save(
                    traj_path,
                    {
                        "info": {
                            "instance_id": instance_id,
                            "sample_idx": sample_idx,
                            "gentest_record": _record_for_traj(record),
                        },
                        "instance_id": instance_id,
                    },
                )
        except Exception:
            pass
    return record


def run_sample_group_iter(instance: dict, config: dict, opts: dict, sample_indices: list[int]):
    instance = copy.deepcopy(instance)
    if not opts.get("use_dataset_test_patch", True):
        # Keep the gold test patch out of generation, but preserve it under a private key so the
        # post-hoc gold_agreement diagnostic can still see where the maintainer put the test.
        instance["_gentest_gold_test_patch"] = instance.get("test_patch") or ""
        instance["test_patch"] = ""
    test_command_override = str(opts.get("test_command_override") or "").strip()
    if test_command_override:
        instance["_gentest_test_command_override"] = test_command_override
    instance_id = instance["instance_id"]
    cfg = copy.deepcopy(config)
    _resolve_per_instance_api_base(cfg, instance_id)
    env = None
    verify_env = None
    completed = 0
    try:
        if opts.get("isolated_verify"):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                environment_futures = [
                    executor.submit(get_gentest_environment, copy.deepcopy(cfg), instance),
                    executor.submit(get_gentest_environment, copy.deepcopy(cfg), instance),
                ]
            created_environments = []
            create_error = None
            for future in environment_futures:
                try:
                    created_environments.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    create_error = create_error or exc
            if create_error is not None:
                for created_environment in created_environments:
                    try:
                        created_environment.cleanup()
                    except Exception:
                        pass
                raise create_error
            env, verify_env = created_environments
        else:
            env = get_gentest_environment(copy.deepcopy(cfg), instance)

        if os.environ.get("GENTEST_RESTORE_ENV_ONCE", "").strip().lower() in ("1", "true", "yes"):
            for environment in (env, verify_env):
                if environment is not None:
                    restore_env(
                        environment,
                        instance,
                        int(opts["install_timeout"]),
                        activation=activation_for_instance(instance),
                    )
        for pos, sample_idx in enumerate(sample_indices):
            sample_opts = {**opts, "sample_idx": int(sample_idx), "reset_before_sample": pos > 0}
            record = run_sample_in_env(instance, cfg, sample_opts, env, verify_env)
            completed += 1
            yield record
    except Exception as exc:  # noqa: BLE001
        traceback_text = traceback.format_exc()[-4000:]
        for sample_idx in sample_indices[completed:]:
            yield sample_error_record(
                instance,
                int(sample_idx),
                opts,
                f"{type(exc).__name__}: {str(exc)[:300]}",
                traceback_text,
            )
    finally:
        try:
            if env is not None:
                env.cleanup()
        except Exception:
            pass
        try:
            if verify_env is not None:
                verify_env.cleanup()
        except Exception:
            pass


def run_sample(instance: dict, config: dict, opts: dict) -> dict:
    records = list(run_sample_group_iter(instance, config, opts, [int(opts["sample_idx"])]))
    if records:
        return records[0]
    return sample_error_record(instance, int(opts["sample_idx"]), opts, "MissingResult: sample group produced no record")


def _record_for_traj(record: dict) -> dict:
    compact = dict(record)
    if "test_code" in compact:
        compact["test_code_chars"] = len(compact.get("test_code") or "")
        compact.pop("test_code", None)
    validation = compact.get("validation")
    if isinstance(validation, dict) and "test_code" in validation:
        validation = dict(validation)
        validation["test_code_chars"] = len(validation.get("test_code") or "")
        validation.pop("test_code", None)
        compact["validation"] = validation
    return compact


def starter_code(instance: dict, opts: dict) -> str:
    starter = opts.get("starter", "gold_imports")
    if starter != "gold_imports":
        return ""
    all_f2p = list_from_json_or_obj(instance.get("FAIL_TO_PASS"))
    node = str(all_f2p[0]) if all_f2p else ""
    _, example_src = extract_example(instance.get("test_patch", "") or "", node, "full")
    imports = extract_import_lines(example_src)
    return "\n".join(imports) + ("\n\n" if imports else "")


def _run_sample_subprocess(instance: dict, config: dict, opts: dict, result_queue: multiprocessing.Queue) -> None:
    try:
        result_queue.put(run_sample(instance, config, opts))
    except Exception as exc:  # noqa: BLE001
        result_queue.put(
            sample_error_record(
                instance,
                int(opts.get("sample_idx", 0)),
                opts,
                f"{type(exc).__name__}: {str(exc)[:300]}",
                traceback.format_exc()[-4000:],
            )
        )


def _run_sample_group_subprocess(
    instance: dict,
    config: dict,
    opts: dict,
    sample_indices: list[int],
    result_queue: multiprocessing.Queue,
) -> None:
    emitted: set[int] = set()
    try:
        for record in run_sample_group_iter(instance, config, opts, sample_indices):
            emitted.add(int(record.get("sample_idx", 0)))
            result_queue.put(record)
    except Exception as exc:  # noqa: BLE001
        traceback_text = traceback.format_exc()[-4000:]
        for sample_idx in sample_indices:
            if int(sample_idx) not in emitted:
                result_queue.put(
                    sample_error_record(
                        instance,
                        int(sample_idx),
                        opts,
                        f"{type(exc).__name__}: {str(exc)[:300]}",
                        traceback_text,
                    )
                )
    finally:
        result_queue.put({"__done__": True})


def process_sample(instance: dict, config: dict, opts: dict) -> dict:
    timeout = int(opts.get("instance_timeout") or 0)
    if timeout <= 0:
        return run_sample(instance, config, opts)
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_run_sample_subprocess, args=(instance, config, opts, queue), daemon=True)
    proc.start()
    deadline = time.monotonic() + timeout
    while proc.is_alive():
        try:
            record = queue.get_nowait()
            proc.join(timeout=5)
            return record
        except Empty:
            pass
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        proc.join(timeout=min(1.0, remaining))
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive() and hasattr(proc, "kill"):
            proc.kill()
            proc.join(timeout=5)
        return sample_error_record(
            instance,
            int(opts["sample_idx"]),
            opts,
            f"WallClockTimeout: exceeded {timeout}s",
        )
    try:
        return queue.get_nowait()
    except Empty:
        return sample_error_record(
            instance,
            int(opts["sample_idx"]),
            opts,
            "MissingResult: child produced no record",
        )


def sample_retry_reason(record: dict) -> str:
    validation = record.get("validation")
    if not isinstance(validation, dict):
        return "missing_validation"
    if validation.get("infra_failure") is True:
        return "infra_failure"
    base_label = validation.get("base_validation_label")
    if base_label != "base_clean_fail":
        return str(base_label or validation.get("label") or "missing_base_validation_label")
    return ""


def process_sample_with_retries(instance: dict, config: dict, opts: dict) -> dict:
    max_retries = max(0, int(opts.get("sample_retries") or 0))
    if max_retries == 0:
        return process_sample(instance, config, opts)

    attempts = []
    record = {}
    retry_reason = ""
    for retry_attempt in range(max_retries + 1):
        attempt_opts = {**opts, "retry_attempt": retry_attempt}
        record = process_sample(instance, config, attempt_opts)
        retry_reason = sample_retry_reason(record)
        validation = record.get("validation") if isinstance(record.get("validation"), dict) else {}
        attempts.append(
            {
                "attempt": retry_attempt,
                "trajectory_path": record.get("trajectory_path", ""),
                "label": validation.get("label"),
                "base_validation_label": validation.get("base_validation_label"),
                "base_clean_fail": validation.get("base_clean_fail"),
                "infra_failure": validation.get("infra_failure"),
                "retry_reason": retry_reason,
                "agent_exit_status": record.get("agent_exit_status"),
                "agent_calls": record.get("agent_calls"),
                "error": record.get("error"),
            }
        )
        if not retry_reason:
            break

    record["sample_retry"] = {
        "configured_retries": max_retries,
        "attempts": len(attempts),
        "retries_used": len(attempts) - 1,
        "exhausted": bool(retry_reason and len(attempts) == max_retries + 1),
        "final_retry_reason": retry_reason,
        "attempt_summaries": attempts,
    }
    return record


def process_sample_group(instance: dict, config: dict, opts: dict, sample_indices: list[int]) -> list[dict]:
    sample_indices = [int(sample_idx) for sample_idx in sample_indices]
    timeout = int(opts.get("instance_timeout") or 0)
    if timeout <= 0:
        return list(run_sample_group_iter(instance, config, opts, sample_indices))

    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue(maxsize=max(1, min(len(sample_indices) + 1, 128)))
    proc = ctx.Process(
        target=_run_sample_group_subprocess,
        args=(instance, config, opts, sample_indices, queue),
        daemon=True,
    )
    proc.start()
    deadline = time.monotonic() + timeout * max(1, len(sample_indices))
    records: list[dict] = []
    emitted: set[int] = set()
    done = False

    while True:
        try:
            item = queue.get(timeout=0.5)
        except Empty:
            item = None
        if isinstance(item, dict):
            if item.get("__done__"):
                done = True
                break
            records.append(item)
            emitted.add(int(item.get("sample_idx", 0)))

        if not proc.is_alive():
            break
        if time.monotonic() >= deadline:
            break

    timed_out = proc.is_alive() and not done
    if timed_out:
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive() and hasattr(proc, "kill"):
            proc.kill()
            proc.join(timeout=5)
    else:
        proc.join(timeout=5)

    while True:
        try:
            item = queue.get_nowait()
        except Empty:
            break
        if isinstance(item, dict):
            if item.get("__done__"):
                done = True
                continue
            records.append(item)
            emitted.add(int(item.get("sample_idx", 0)))

    if timed_out:
        reason = f"GroupWallClockTimeout: exceeded {timeout * max(1, len(sample_indices))}s"
    elif not done:
        reason = "MissingResult: child produced no group completion marker"
    else:
        reason = ""
    if reason:
        for sample_idx in sample_indices:
            if sample_idx not in emitted:
                records.append(sample_error_record(instance, sample_idx, opts, reason))

    order = {sample_idx: pos for pos, sample_idx in enumerate(sample_indices)}
    return sorted(records, key=lambda record: order.get(int(record.get("sample_idx", 0)), len(order)))


def build_config(config_spec: list[str], *, model: str | None, model_class: str | None, environment_class: str | None) -> dict:
    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append(
        {
            "environment": {"environment_class": environment_class or UNSET},
            "model": {"model_name": model or UNSET, "model_class": model_class or UNSET},
        }
    )
    return recursive_merge(*configs)


def summarize_record(record: dict) -> str:
    validation = record.get("validation") or {}
    gate = record.get("gate") or {}
    return (
        f"{record.get('instance_id')}#{record.get('sample_idx')} "
        f"exit={record.get('agent_exit_status')} "
        f"base={validation.get('base_clean_fail')} gold={validation.get('gold_pass')} "
        f"base_fail_gold_pass={validation.get('base_fail_gold_pass')} "
        f"label={validation.get('label')} "
        f"gate_self_tested={(gate.get('last_gate') or {}).get('self_tested', gate.get('self_test_commands') != [])} "
        f"error={record.get('error')}"
    )


# fmt: off
@app.command()
def main(
    output: Path = typer.Option(..., "-o", "--output", help="Output directory for JSONL results and trajectories"),
    subset: str = typer.Option("rebench", "--subset", help="Dataset subset or dataset path"),
    split: str = typer.Option("filtered", "--split", help="Dataset split"),
    ids: Path | None = typer.Option(None, "--ids", help="Optional JSON/JSONL file of instance ids"),
    limit: int = typer.Option(0, "--limit", help="Limit selected ids"),
    workers: int = typer.Option(1, "-w", "--workers", help="Parallel samples"),
    n_samples: int = typer.Option(1, "-n", "--n-samples", help="Samples per instance"),
    reuse_sandbox_samples: bool = typer.Option(
        False,
        "--reuse-sandbox-samples/--no-reuse-sandbox-samples",
        help="Reuse one sandbox per instance across that instance's samples.",
    ),
    isolated_verify: bool = typer.Option(
        True,
        "--isolated-verify/--shared-verify",
        help="Run generated-test feedback and validation in a separate sandbox using the official eval format.",
    ),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Regenerate records already in results JSONL"),
    results_jsonl: Path | None = typer.Option(None, "--results-jsonl", help="Defaults to <output>/gentest_results.jsonl"),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config", help="mini-swe-agent config specs"),
    model: str | None = typer.Option(None, "-m", "--model", help="Override model name"),
    model_class: str | None = typer.Option(None, "--model-class", help="Override model class"),
    environment_class: str | None = typer.Option(None, "--environment-class", help="Override environment class"),
    prompt_mode: str = typer.Option("zero_shot", "--prompt-mode", help="zero_shot or one_shot_gold"),
    example_shot: str = typer.Option("none", "--example-shot", help="node, full, or none"),
    starter: str = typer.Option(
        "empty",
        "--starter",
        help="gold_imports, runner_scaffold, or empty",
    ),
    test_file: str = typer.Option(TEST_FILE, "--test-file", help="Generated test file name"),
    test_command_override: str = typer.Option(
        "",
        "--test-command-override",
        help="Override the project test runner while still targeting the generated test file.",
    ),
    use_dataset_test_patch: bool = typer.Option(
        False,
        "--use-dataset-test-patch/--ignore-dataset-test-patch",
        help="Allow gentest to inspect and optionally apply the dataset test_patch.",
    ),
    seed: int = typer.Option(0, "--seed", help="Seed for choosing one-shot examples"),
    gold_eval: bool = typer.Option(True, "--gold-eval/--no-gold-eval", help="Run offline gold-patch validation"),
    apply_test_patch_at_start: bool = typer.Option(
        False,
        "--apply-test-patch-at-start/--no-apply-test-patch-at-start",
        help="Apply the dataset test_patch before agent exploration and validation for oracle-aligned experiments.",
    ),
    test_timeout: int = typer.Option(240, "--test-timeout", help="Generated test timeout"),
    install_timeout: int = typer.Option(600, "--install-timeout", help="Install/restore timeout"),
    instance_timeout: int = typer.Option(1800, "--instance-timeout", help="Hard timeout per sample; 0 disables"),
    sample_retries: int = typer.Option(
        0,
        "--sample-retries",
        min=0,
        help="Retry a sample in fresh sandboxes when base validation is not clean or validation reports infra.",
    ),
) -> None:
    # fmt: on
    config = build_config(config_spec, model=model, model_class=model_class, environment_class=environment_class)
    execution_contract_mode = bool((config.get("agent") or {}).get("require_execution_contract"))
    if sample_retries > 0 and reuse_sandbox_samples:
        raise typer.BadParameter("--sample-retries requires --no-reuse-sandbox-samples")
    if apply_test_patch_at_start and not use_dataset_test_patch:
        raise typer.BadParameter("--apply-test-patch-at-start requires --use-dataset-test-patch")
    if execution_contract_mode:
        unsafe_generation_options = []
        if prompt_mode != "zero_shot":
            unsafe_generation_options.append("--prompt-mode must be zero_shot")
        if example_shot != "none":
            unsafe_generation_options.append("--example-shot must be none")
        if starter != "empty":
            unsafe_generation_options.append("--starter must be empty")
        if use_dataset_test_patch:
            unsafe_generation_options.append("--ignore-dataset-test-patch is required")
        if apply_test_patch_at_start:
            unsafe_generation_options.append("--no-apply-test-patch-at-start is required")
        if test_command_override.strip():
            unsafe_generation_options.append("--test-command-override must be empty")
        if unsafe_generation_options:
            raise typer.BadParameter(
                "agent-inferred execution-contract mode forbids evaluation-derived generation inputs: "
                + "; ".join(unsafe_generation_options)
            )
    output.mkdir(parents=True, exist_ok=True)
    add_file_handler(output / "minisweagent_gentest.log")
    results_path = results_jsonl or (output / "gentest_results.jsonl")
    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset {dataset_path} split={split}")
    ds = {}
    for row in load_dataset(dataset_path, split=split):
        instance = dict(row)
        instance.setdefault("dataset", dataset_path)
        ds[instance["instance_id"]] = instance

    selected = load_ids(ids, limit=limit)
    selected_ids = [iid for iid in (selected or list(ds)) if iid in ds]
    if limit and selected is None:
        selected_ids = selected_ids[:limit]

    done = set() if redo_existing else existing_keys(results_path)
    work = []
    work_by_instance: dict[str, list[int]] = {}
    for iid in selected_ids:
        for sample_idx in range(n_samples):
            if (iid, sample_idx) not in done:
                work.append((iid, sample_idx))
                work_by_instance.setdefault(iid, []).append(sample_idx)
    work_groups = [(iid, sample_indices) for iid, sample_indices in work_by_instance.items()]

    settings = {
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": dataset_path,
        "split": split,
        "selected_instances": len(selected_ids),
        "todo_samples": len(work),
        "n_samples": n_samples,
        "reuse_sandbox_samples": reuse_sandbox_samples,
        "isolated_verify": isolated_verify,
        "agent_inferred_execution_contract": execution_contract_mode,
        "prompt_mode": prompt_mode,
        "example_shot": example_shot,
        "starter": starter,
        "test_file": test_file,
        "test_command_override": test_command_override,
        "use_dataset_test_patch": use_dataset_test_patch,
        "gold_eval": gold_eval,
        "apply_test_patch_at_start": apply_test_patch_at_start,
        "sample_retries": sample_retries,
        "workers": workers,
        "config_spec": config_spec,
    }
    (output / "gentest_settings.json").write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n")
    print(
        f"dataset={dataset_path} selected={len(selected_ids)} todo={len(work)} existing={len(done)} "
        f"output={output} results={results_path} workers={workers} prompt_mode={prompt_mode} "
        f"starter={starter} test_command_override={test_command_override!r} "
        f"use_dataset_test_patch={use_dataset_test_patch} gold_eval={gold_eval} "
        f"apply_test_patch_at_start={apply_test_patch_at_start} "
        f"reuse_sandbox_samples={reuse_sandbox_samples} isolated_verify={isolated_verify} "
        f"sample_retries={sample_retries}",
        flush=True,
    )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    base_opts = {
        "output_dir": str(output),
        "prompt_mode": prompt_mode,
        "example_shot": example_shot,
        "starter": starter,
        "test_file": test_file,
        "test_command_override": test_command_override,
        "use_dataset_test_patch": use_dataset_test_patch,
        "seed": seed,
        "gold_eval": gold_eval,
        "apply_test_patch_at_start": apply_test_patch_at_start,
        "test_timeout": test_timeout,
        "install_timeout": install_timeout,
        "instance_timeout": instance_timeout,
        "isolated_verify": isolated_verify,
        "sample_retries": sample_retries,
        "execution_contract_mode": execution_contract_mode,
    }
    with results_path.open("a", buffering=1) as handle:
        if reuse_sandbox_samples:
            completed = 0
            if workers <= 1:
                for iid, sample_indices in work_groups:
                    for record in process_sample_group(ds[iid], config, base_opts, sample_indices):
                        completed += 1
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        print(f"[{completed}/{len(work)}] {summarize_record(record)}", flush=True)
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(process_sample_group, ds[iid], config, base_opts, sample_indices): (
                            iid,
                            sample_indices,
                        )
                        for iid, sample_indices in work_groups
                    }
                    for future in concurrent.futures.as_completed(futures):
                        iid, sample_indices = futures[future]
                        try:
                            records = future.result()
                        except Exception as exc:  # noqa: BLE001
                            records = [
                                sample_error_record(
                                    ds[iid],
                                    sample_idx,
                                    base_opts,
                                    f"{type(exc).__name__}: {str(exc)[:300]}",
                                    traceback.format_exc()[-4000:],
                                )
                                for sample_idx in sample_indices
                            ]
                        for record in records:
                            completed += 1
                            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                            print(f"[{completed}/{len(work)}] {summarize_record(record)}", flush=True)
        elif workers <= 1:
            for idx, (iid, sample_idx) in enumerate(work, 1):
                opts = {**base_opts, "sample_idx": sample_idx}
                record = process_sample_with_retries(ds[iid], config, opts)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"[{idx}/{len(work)}] {summarize_record(record)}", flush=True)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {}
                for iid, sample_idx in work:
                    opts = {**base_opts, "sample_idx": sample_idx}
                    futures[executor.submit(process_sample_with_retries, ds[iid], config, opts)] = (iid, sample_idx)
                for idx, future in enumerate(concurrent.futures.as_completed(futures), 1):
                    try:
                        record = future.result()
                    except Exception as exc:  # noqa: BLE001
                        iid, sample_idx = futures[future]
                        record = {
                            "instance_id": iid,
                            "sample_idx": sample_idx,
                            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                            "traceback": traceback.format_exc()[-4000:],
                        }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    print(f"[{idx}/{len(work)}] {summarize_record(record)}", flush=True)


if __name__ == "__main__":
    app()
