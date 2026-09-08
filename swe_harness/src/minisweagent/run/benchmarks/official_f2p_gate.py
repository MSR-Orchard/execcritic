"""Dataset-aware persistent official evaluator for F2P repair.

Each sample owns one verifier sandbox.  The sandbox runs the dataset's official
environment setup once, then every focused or full grade hard-resets the
repository to ``base_commit`` before applying the submitted patch and hidden
test patch.  Dependency state outside the repository remains stable.
"""

from __future__ import annotations

import copy
import os
import shlex
import tempfile
from pathlib import Path
from typing import Any


def environment_config(timeout: int) -> dict[str, Any]:
    """Build the official Azure verifier configuration."""
    return {
        "base_url": os.environ.get("SANDBOX_BASE_URL", "http://localhost:8000"),
        "api_key": os.environ.get("SANDBOX_API_KEY", ""),
        "block_network": False,
        "cpu": "2",
        "memory": "8Gi",
        "timeout": 600,
        "sandbox_timeout": max(int(timeout), 600),
        "heartbeat_interval": 120,
        "login_shell": True,
        "env": {
            "PAGER": "cat",
            "MANPAGER": "cat",
            "LESS": "-R",
            "PIP_PROGRESS_BAR": "off",
            "TQDM_DISABLE": "1",
        },
    }


def _verifier(instance: dict[str, Any]):
    if instance.get("_swebench_pro"):
        from minisweagent.run.benchmarks import swebench_pro_verify_azure_modal

        return "swebench_pro", swebench_pro_verify_azure_modal
    if instance.get("dataset") in {"nebius/SWE-rebench", "nebius/SWE-rebench-V2"}:
        from minisweagent.run.benchmarks import swerebench_verify_azure_modal

        return "swerebench", swerebench_verify_azure_modal
    if instance.get("dataset") == "AweAI-Team/Scale-SWE":
        from minisweagent.run.benchmarks import scaleswe_verify_azure_modal

        return "scaleswe", scaleswe_verify_azure_modal

    # Combined training JSONL files can omit dataset provenance.  Prebuilt
    # SWE-rebench rows still carry their image and complete install/test spec.
    install_config = instance.get("install_config")
    if (
        not instance.get("dataset")
        and (instance.get("image_name") or instance.get("image_url"))
        and isinstance(install_config, dict)
        and install_config.get("test_cmd")
    ):
        from minisweagent.run.benchmarks import swerebench_verify_azure_modal

        return "swerebench", swerebench_verify_azure_modal

    if any(instance.get(key) for key in ("f2p_script", "f2p_patch", "task_type")):
        from minisweagent.run.benchmarks import scaleswe_verify_azure_modal

        return "scaleswe", scaleswe_verify_azure_modal

    from minisweagent.run.benchmarks import swebench_verify_azure_modal

    return "swebench", swebench_verify_azure_modal


def _prepare_instance(
    instance: dict[str, Any],
    *,
    verifier_name: str,
    fail_to_pass: list[str],
    pass_to_pass: list[str],
) -> dict[str, Any]:
    official_instance = copy.deepcopy(instance)
    official_instance["FAIL_TO_PASS"] = list(fail_to_pass)
    official_instance["PASS_TO_PASS"] = list(pass_to_pass)
    if verifier_name == "swerebench":
        image_name = official_instance.get("image_name") or official_instance.get("image_url")
        if not image_name:
            raise ValueError("SWE-rebench instance is missing image_name/image_url")
        if str(image_name).startswith("mirror.gcr.io/"):
            image_name = str(image_name)[len("mirror.gcr.io/") :]
        official_instance["image_name"] = image_name
    elif verifier_name == "swebench_pro":
        official_instance["_swebench_pro_selected_nodes"] = list(fail_to_pass)
    elif verifier_name == "swebench":
        official_instance["_swebench_selected_nodes"] = list(fail_to_pass)
    return official_instance


class PersistentOfficialGate:
    """One prepared official verifier reused for every focused and full grade."""

    def __init__(self, instance: dict[str, Any], *, timeout: int) -> None:
        self.instance = copy.deepcopy(instance)
        self.verifier_name, self.verifier = _verifier(self.instance)
        prepared = _prepare_instance(
            self.instance,
            verifier_name=self.verifier_name,
            fail_to_pass=[],
            pass_to_pass=[],
        )
        self.env_config = environment_config(timeout)
        self.env = self.verifier.create_environment(prepared, self.env_config)
        if self.verifier_name == "swebench_pro":
            workdir = "/app"
        elif self.verifier_name == "scaleswe":
            workdir = str(self.instance.get("workdir") or "/")
        else:
            workdir = "/testbed"
        requested_base = str(self.instance.get("base_commit") or "HEAD")
        try:
            baseline = self.env.execute(
                {
                    "command": (
                        f"cd {shlex.quote(workdir)} && "
                        f"git rev-parse {shlex.quote(requested_base)}^{{commit}}"
                    )
                },
                timeout=60,
            )
            self.baseline_commit = str(baseline.get("output") or "").strip()
            if baseline.get("returncode") != 0 or not self.baseline_commit:
                raise RuntimeError(
                    "official verifier base_commit is unavailable: "
                    f"{str(baseline.get('output') or '')[-1000:]}"
                )
            reset = self.env.execute(
                {
                    "command": (
                        f"cd {shlex.quote(workdir)} && "
                        f"git reset --hard {shlex.quote(self.baseline_commit)} && git clean -fdq"
                    )
                },
                timeout=120,
            )
            if reset.get("returncode") != 0:
                raise RuntimeError(
                    "official verifier initial reset failed: "
                    f"{str(reset.get('output') or '')[-1000:]}"
                )
            prepare = getattr(self.verifier, "prepare_environment_for_evaluation", None)
            if prepare is not None:
                prepare(self.env, prepared, timeout=timeout)
        except BaseException:
            self.env.cleanup()
            raise
        self.closed = False

    def evaluate(
        self,
        patch: str,
        *,
        fail_to_pass: list[str],
        pass_to_pass: list[str],
        timeout: int,
    ) -> dict[str, Any]:
        if self.closed:
            raise RuntimeError("persistent official gate is already closed")
        official_instance = _prepare_instance(
            self.instance,
            verifier_name=self.verifier_name,
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
        )
        official_instance["_persistent_baseline_commit"] = self.baseline_commit
        with tempfile.TemporaryDirectory(prefix="slime_official_f2p_gate_") as temporary:
            logs_dir = Path(temporary)
            evaluate_kwargs = {"logs_dir": logs_dir}
            if self.verifier_name == "swerebench":
                evaluate_kwargs["environment_prepared"] = True
            result = self.verifier.evaluate_instance_in_environment(
                official_instance,
                patch,
                self.env,
                environment_config(timeout),
                **evaluate_kwargs,
            )
            output_path = logs_dir / f"{official_instance['instance_id']}_log.txt"
            output = output_path.read_text(errors="replace") if output_path.exists() else ""
        return {
            **result,
            "output": output,
            "official_verifier": self.verifier_name,
            "persistent_verifier": True,
        }

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.env.cleanup()


def passed_count(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, (list, tuple, set)):
        return len(value)
    return 0


def normalize_result(
    result: dict[str, Any],
    *,
    fail_to_pass: list[str],
    pass_to_pass: list[str],
) -> dict[str, Any]:
    """Map all official verifier result shapes to the repair reward schema."""
    f2p_passed = passed_count(result.get("fail_to_pass_passed"))
    p2p_passed = passed_count(result.get("pass_to_pass_passed"))
    resolved = bool(result.get("resolved"))
    error = str(result.get("error") or "")
    if not result.get("patch_applied"):
        resolved = False
        error = error or "official verifier failed to apply submitted patch"
    return {
        **result,
        "passed": resolved,
        "resolved": resolved,
        "resolution": "RESOLVED_FULL" if resolved else "RESOLVED_NO",
        "f2p_total": len(fail_to_pass),
        "f2p_passed": f2p_passed,
        "p2p_total": len(pass_to_pass),
        "p2p_passed": p2p_passed,
        "tests_run": len(fail_to_pass) + len(pass_to_pass),
        "tests_passed": f2p_passed + p2p_passed,
        "error": error,
    }
