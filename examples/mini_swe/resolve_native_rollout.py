"""Native direct-SGLang resolve-style test_patch generator (RL rollout).

Fork of ``gentest_native_rollout`` whose ALGORITHM is aligned 1:1 with the offline
``swe_harness/gentest_v2/batch_resolve.py`` eval so training and eval run the same protocol:

  * agent: selected by ``RESOLVE_OVERLAY``. The v11/behavior_contract paths use
    ``BashTestPatchResolveAgent`` with only the standard bash tool, ``/tmp/test_command``,
    ``/tmp/test.patch``, and mini-swe-agent's native submission marker. behavior-contract additionally requires
    ``/tmp/test_contract.json`` and exact test-node selectors. The legacy v10
    ``TestPatchResolveAgent`` path remains available for reproducibility.
  * candidate-free AND gold-free during generation: no starter code, no apply_test_patch_at_start,
    and no gold F2P / gold test_command reaches the model. The configured patch-classification
    reward may read candidate and Gold metadata only after generation.

The trainable layer (``_SGLangGentestModel``: token ids, loss mask, rollout logprobs, routed
experts) is imported unchanged from ``gentest_native_rollout`` so RL token alignment stays a single
source of truth. Only ``_generate_once`` and the reward mapping differ.

If the configured custom reward does not produce a value, the fallback reuses ``gentest_reward``
via a synthesized gold-free label:
  Submitted & base_clean_fail  -> label ``base_fail_only``      (default 0.5)
  Submitted & not clean_fail   -> label ``base_not_clean_fail`` (default -1.0)
  not Submitted (LimitsExceeded/void) -> GENTEST_UNSUBMITTED_REWARD (default -1.0)
The fallback cannot emit ``gold_validated``; Gold-aware scoring belongs to the separate custom
reward path.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Any

from minisweagent.agents.extra.bash_testpatch_resolve import BashTestPatchResolveAgent
from minisweagent.agents.extra.testpatch_resolve import TestPatchResolveAgent
from minisweagent.run.benchmarks.gentest import (
    GitHistoryGuardEnv,
    get_gentest_agent_environment,
    run_shared_test_patch_official,
    run_test_patch_official,
    sanitize_git_history,
)

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample

# Reuse the entire trainable RL infrastructure from the v6 native rollout unchanged.
from .gentest_native_rollout import _gentest_rollout_metrics  # metrics reader is label/status generic
from .gentest_native_rollout import (
    DEFAULT_MAX_ALL_TOKENS,
    _build_rollout_tensors,
    _compute_reward_before_environment_cleanup,
    _create_environment,
    _create_isolated_environments,
    _env_flag,
    _env_int,
    _fill_placeholder_trajectory,
    _has_format_error,
    _instance_from_sample,
    _is_retryable_infrastructure_error,
    _load_gentest_config,
    _load_official_verifier_config,
    _record_error,
    _record_timing,
    _run_blocking,
    _run_blocking_before_environment_cleanup,
    _SGLangGentestModel,
    _trajectory_root,
    _zero_loss_mask,
    configure_blocking_executor,
)
from .gentest_reward import reward_from_record
from .swe_wrapper_v2 import EnvironmentUnavailable

logger = logging.getLogger(__name__)

RESOLVE_EXTRA_TOOLS = ["grounding_plan", "write_test_patch", "submit_test_patch"]

_DEFAULT_RESOLVE_OVERLAY = "swe_harness/gentest_v2/versions/behavior_contract.yaml"
_RESOLVE_OVERLAY_CACHE: tuple[Path, dict[str, Any]] | None = None
_RESOLVE_AGENT_CLASSES = {
    "testpatch_resolve": TestPatchResolveAgent,
    "bash_testpatch_resolve": BashTestPatchResolveAgent,
}


def _resolve_overlay() -> tuple[Path, dict[str, Any]]:
    """Load the complete resolve overlay used by both the agent and model."""
    global _RESOLVE_OVERLAY_CACHE
    path = Path(os.environ.get("RESOLVE_OVERLAY", _DEFAULT_RESOLVE_OVERLAY)).expanduser().resolve()
    if _RESOLVE_OVERLAY_CACHE is not None and _RESOLVE_OVERLAY_CACHE[0] == path:
        return path, copy.deepcopy(_RESOLVE_OVERLAY_CACHE[1])

    import yaml

    if not path.is_file():
        raise FileNotFoundError(f"Resolve overlay does not exist: {path}")
    overlay = yaml.safe_load(path.read_text()) or {}
    if not isinstance(overlay, dict):
        raise ValueError(f"Resolve overlay must contain a YAML mapping: {path}")
    _RESOLVE_OVERLAY_CACHE = (path, overlay)
    return path, copy.deepcopy(overlay)


def _resolve_runtime_config(
    config: dict[str, Any],
) -> tuple[type, dict[str, Any], dict[str, Any], str, str, Path]:
    """Merge the selected overlay and return the concrete protocol implementation."""
    overlay_path, overlay = _resolve_overlay()
    overlay_agent = dict(overlay.get("agent", {}) or {})
    agent_class_name = str(overlay_agent.pop("agent_class", "testpatch_resolve"))
    try:
        agent_class = _RESOLVE_AGENT_CLASSES[agent_class_name]
    except KeyError as exc:
        supported = ", ".join(sorted(_RESOLVE_AGENT_CLASSES))
        raise ValueError(
            f"Unsupported resolve agent_class {agent_class_name!r} in {overlay_path}; supported: {supported}"
        ) from exc

    # The resolve templates reference only {{task}}, unlike the v6 gentest templates. Model
    # settings are also overlay-owned so v11/behavior_contract format errors and empty extra-tool lists take effect.
    agent_cfg = copy.deepcopy(config.get("agent", {}) or {})
    agent_cfg.update(overlay_agent)
    model_cfg = copy.deepcopy(config.get("model", {}) or {})
    model_cfg.update(copy.deepcopy(overlay.get("model", {}) or {}))

    # Keep each protocol's model-facing tool surface exact even if the transport base config has
    # gentest defaults. ``tools_for_names([])`` still exposes the standard bash tool.
    if agent_class_name == "bash_testpatch_resolve":
        model_cfg["extra_tools"] = []
        contract_flags = (
            bool(agent_cfg.get("require_test_contract")),
            bool(agent_cfg.get("require_exact_test_selector")),
        )
        if any(contract_flags) and not all(contract_flags):
            raise ValueError(
                "bash_testpatch_resolve must enable require_test_contract and " "require_exact_test_selector together"
            )
        protocol = "behavior_contract" if all(contract_flags) else "v11_native_bash"
    else:
        model_cfg["extra_tools"] = list(RESOLVE_EXTRA_TOOLS)
        protocol = "v10_structured_tools"
    return agent_class, agent_cfg, model_cfg, agent_class_name, protocol, overlay_path


def _resolve_label(exit_status: str, base_clean_fail: Any) -> str:
    """Map the resolve gate onto a gentest_reward label. Gold-free: never emits gold_validated."""
    if exit_status == "sandbox_infra_error":
        return "sandbox_infra_error"
    if exit_status != "Submitted":
        return "missing_test"
    return "base_fail_only" if bool(base_clean_fail) else "base_not_clean_fail"


async def _create_resolve_environments(
    workspace_config: dict[str, Any],
    verifier_config: dict[str, Any],
    instance: dict[str, Any],
    timings: dict[str, Any],
    *,
    shared_verify: bool,
):
    """Create either one shared sandbox or the default isolated workspace/verifier pair."""
    if not shared_verify:
        return await _create_isolated_environments(
            workspace_config,
            verifier_config,
            instance,
            timings,
        )

    started = time.perf_counter()
    with _record_timing(timings, "workspace_environment_create_sec"):
        workspace_env = await _create_environment(copy.deepcopy(workspace_config), instance)
    timings["verify_environment_create_sec"] = 0.0
    timings["environment_create_sec"] = time.perf_counter() - started
    return workspace_env, workspace_env


async def _cleanup_resolve_environments(
    workspace_env,
    verify_env,
    timings: dict[str, Any],
    *,
    instance_id: str,
) -> None:
    """Clean each owned sandbox exactly once, including when shared verify aliases both roles."""
    environments = (
        (("workspace", workspace_env),)
        if verify_env is workspace_env
        else (
            ("workspace", workspace_env),
            ("verifier", verify_env),
        )
    )

    async def cleanup_all() -> None:
        for role, environment in environments:
            if environment is None:
                continue
            try:
                with _record_timing(timings, f"{role}_environment_cleanup_sec"):
                    await _run_blocking(environment.cleanup)
            except Exception as cleanup_exc:  # noqa: BLE001
                logger.warning(
                    "[RESOLVE_GENERATE] %s cleanup failed for %s: %s",
                    role,
                    instance_id,
                    cleanup_exc,
                )

    cleanup_task = asyncio.create_task(cleanup_all())
    try:
        await asyncio.shield(cleanup_task)
    except asyncio.CancelledError:
        # A hard tail cut must not interrupt sandbox deletion halfway through.
        await cleanup_task
        raise


async def generate(args, sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False) -> Sample:
    """Slime custom-generate entrypoint with whole-environment retry (mirrors v6 native)."""
    default_workers = max(
        2,
        2 * int(getattr(args, "rollout_batch_size", 1) or 1) * int(getattr(args, "n_samples_per_prompt", 1) or 1),
    )
    configure_blocking_executor(_env_int("GENTEST_NATIVE_WORKERS", default_workers))

    resource_level = 1
    max_retries = max(1, _env_int("GENTEST_MAX_ENV_RETRIES", 2))
    retry_wait = _env_int("GENTEST_ENV_RETRY_WAIT", 10)
    for attempt in range(1, max_retries + 1):
        try:
            sample = await _generate_once(args, sample, sampling_params, resource_level, evaluation=evaluation)
            gentest_record = sample.metadata.get("gentest_record") if isinstance(sample.metadata, dict) else None
            if isinstance(gentest_record, dict):
                gentest_record["generate_attempts"] = attempt
                gentest_record["resource_level"] = resource_level
            if _has_format_error(sample):
                reward_key = getattr(args, "reward_key", None)
                sample.reward = {reward_key: 0.0} if reward_key else 0.0
            return sample
        except Exception as exc:  # noqa: BLE001
            retryable = _is_retryable_infrastructure_error(exc)
            logger.error(
                "[RESOLVE_GENERATE] attempt %s/%s failed sample=%s retryable=%s: %s",
                attempt,
                max_retries,
                sample.index,
                retryable,
                exc,
                exc_info=True,
            )
            if retryable and attempt < max_retries:
                resource_level += 1
                sample.reward = None
                await asyncio.sleep(retry_wait)
                continue
            instance = _instance_from_sample(sample)
            existing = sample.metadata.get("gentest_record") if isinstance(sample.metadata, dict) else None
            record = (
                existing
                if isinstance(existing, dict)
                else _record_error(instance, sample, f"{type(exc).__name__}: {str(exc)[:300]}", traceback.format_exc())
            )
            record["retryable_infrastructure_error"] = retryable
            if retryable:
                record["agent_exit_status"] = "sandbox_infra_error"
                record["sandbox_infra_error"] = True
            record["generate_attempts"] = attempt
            sample.metadata.update(
                {
                    "gentest_record": record,
                    "exit_status": record.get("agent_exit_status", "Error"),
                    "sandbox_infra_error": retryable,
                }
            )
            sample.status = Sample.Status.ABORTED
            sample.remove_sample = retryable
            _fill_placeholder_trajectory(args, sample, instance)
            _zero_loss_mask(sample)
            # Aborted (dead sandbox / infra): zero loss mask, excluded from GRPO group stats.
            # Reward 0.0 (not the -1.0 unsubmitted penalty) so it isn't mistaken for a bad trajectory.
            reward = 0.0
            reward_key = getattr(args, "reward_key", None)
            sample.reward = {"raw_reward": reward}
            if reward_key and reward_key != "raw_reward":
                sample.reward[reward_key] = reward
            return sample


async def _generate_once(
    args, sample: Sample, sampling_params: dict[str, Any], resource_level: int, evaluation: bool = False
) -> Sample:
    start_time = time.time()
    state = GenerateState(args)
    tokenizer = state.tokenizer
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    rollout_id = getattr(state, "rollout_id", None)

    instance = _instance_from_sample(sample)
    instance_id = str(instance["instance_id"])
    config = _load_gentest_config(instance_id)
    verifier_config = _load_official_verifier_config(instance_id)
    env_config = config.setdefault("environment", {})
    verifier_env_config = verifier_config.setdefault("environment", {})
    if resource_level > 1:
        for c in (env_config, verifier_env_config):
            c["cpu"] = str(resource_level * 2)
            c["memory"] = f"{resource_level * 8}Gi"

    trajectory_dir = _trajectory_root()
    if rollout_id is not None:
        trajectory_dir = trajectory_dir / f"rl_step_{rollout_id}"
    trajectory_dir = trajectory_dir / instance_id
    traj_path = trajectory_dir / f"{instance_id}.sample_{int(sample.index or 0)}.traj.json"

    agent_class, overlay_agent_cfg, model_cfg, agent_class_name, protocol, overlay_path = _resolve_runtime_config(
        config
    )
    install_timeout = _env_int("RESOLVE_INSTALL_TIMEOUT", 1200)
    step_limit = _env_int("RESOLVE_STEP_LIMIT", int(overlay_agent_cfg.get("step_limit", 60)))
    time_limit = _env_int("RESOLVE_TIME_LIMIT", int(overlay_agent_cfg.get("time_limit", 0)))
    shared_verify = _env_flag("RESOLVE_SHARED_VERIFY", False)

    env = verify_env = agent = model = None
    record: dict[str, Any] = {
        "instance_id": instance_id,
        "sample_idx": int(sample.index or 0),
        "algorithm": "resolve_testpatch",
        "protocol": protocol,
        "agent_class": agent_class_name,
        "resolve_overlay": str(overlay_path),
        "gold_free": True,
        "official_verify_format": True,
        "shared_verify": shared_verify,
        "isolated_verify": not shared_verify,
        "rl_step": rollout_id,
        "timings": {},
    }
    phase_timings = record["timings"]
    try:
        env, verify_env = await _create_resolve_environments(
            config,
            verifier_config,
            instance,
            phase_timings,
            shared_verify=shared_verify,
        )

        # Gold-free enforcement: swerebench images ship the full upstream repo, so `git log --all`
        # / `git show <fix_sha>` can reach the fix commit and its tests. Physically prune unreachable
        # history on both the agent workspace and the (isolated) verifier so neither the agent nor
        # the reward path can leak the answer. Behaviour guard (GitHistoryGuardEnv) is a second layer.
        sanitize_enabled = _env_flag("GENTEST_SANITIZE_GIT_HISTORY", True)
        history_guard_enabled = _env_flag("GENTEST_GIT_HISTORY_GUARD", True)
        record["git_history_sanitize_requested"] = sanitize_enabled
        record["git_history_guard_enabled"] = history_guard_enabled
        if protocol == "behavior_contract" and not history_guard_enabled:
            raise EnvironmentUnavailable("strict behavior-contract requires the runtime git-history guard")
        if sanitize_enabled:
            sanitize_errors = []
            with _record_timing(phase_timings, "git_sanitize_sec"):
                for _sanitize_env in ({id(env): env, id(verify_env): verify_env}).values():
                    try:
                        await _run_blocking_before_environment_cleanup(sanitize_git_history, _sanitize_env)
                    except Exception as sanitize_exc:  # noqa: BLE001
                        sanitize_errors.append(f"{type(sanitize_exc).__name__}: {sanitize_exc}")
                        logger.warning(
                            "[RESOLVE_GENERATE] git-history sanitize failed for %s: %s",
                            instance_id,
                            sanitize_exc,
                        )
            record["git_history_sanitized"] = not sanitize_errors
            if sanitize_errors and protocol == "behavior_contract":
                raise EnvironmentUnavailable(
                    "strict behavior-contract git-history sanitization failed: " + "; ".join(sanitize_errors)
                )
        else:
            record["git_history_sanitized"] = False
            if protocol == "behavior_contract":
                raise EnvironmentUnavailable("strict behavior-contract requires physical git-history sanitization")

        max_all_tokens = _env_int(
            "GENTEST_MAX_ALL_TOKENS", int(getattr(args, "rollout_max_context_len", None) or DEFAULT_MAX_ALL_TOKENS)
        )
        model = _SGLangGentestModel(
            model_cfg,
            tokenizer=tokenizer,
            url=url,
            sampling_params=sampling_params,
            llm_timeout=_env_int("GENTEST_TIMEOUT_LLM_INFERENCE", 300),
            max_llm_attempts=_env_int("GENTEST_MAX_LLM_ATTEMPTS", 2),
            max_all_tokens=max_all_tokens,
            use_rollout_routing_replay=bool(getattr(args, "use_rollout_routing_replay", False)),
            num_layers=int(args.num_layers),
            moe_router_topk=int(getattr(args, "moe_router_topk", 0) or 0),
            num_experts=int(getattr(args, "num_experts", 0) or 0),
            abort_check=lambda: state.aborted,
        )

        warmed = {"done": False}

        def verify_runner(test_patch: str, test_command: str) -> dict:
            # First call installs deps (skip_install=False), later calls reuse the warm env. Gold
            # NEVER enters here. Shared mode resets transactionally and then restores agent edits.
            skip = warmed["done"]
            if shared_verify:
                result = run_shared_test_patch_official(
                    env,
                    instance,
                    test_patch,
                    test_command,
                    install_timeout,
                    skip_install=skip,
                    restore_agent_workspace=True,
                )
            else:
                result = run_test_patch_official(
                    verify_env,
                    instance,
                    test_patch,
                    test_command,
                    install_timeout,
                    skip_install=skip,
                )
            warmed["done"] = True
            return result

        # Bridge the blocking verify_runner onto the shared executor so it does not stall the loop.
        def blocking_verify_runner(test_patch: str, test_command: str) -> dict:
            fut = asyncio.run_coroutine_threadsafe(_run_blocking(verify_runner, test_patch, test_command), loop)
            return fut.result()

        loop = asyncio.get_running_loop()

        # Resolve agent templates come from the resolve overlay (system/instance_template that only
        # reference {{task}}), NOT the v6 gentest.yaml agent block (whose template needs v6-only
        # vars like max_generated_test_cases). Fall back to the v6 agent block only for scalars.
        agent_cfg = copy.deepcopy(config.get("agent", {}) or {})
        agent_cfg.update(overlay_agent_cfg)
        agent_cfg["step_limit"] = step_limit
        if time_limit > 0:
            agent_cfg["time_limit"] = time_limit
        agent_cfg["output_path"] = traj_path
        # Agent sees: activation prefix -> git-history guard -> sandbox. The guard refuses
        # `git log --all`/`git show <sha>`-style history mining even if a ref survived the prune.
        agent_env = get_gentest_agent_environment(env, instance)
        if history_guard_enabled:
            agent_env = GitHistoryGuardEnv(agent_env)
        agent = agent_class(
            model,
            agent_env,
            verify_runner=blocking_verify_runner,
            **agent_cfg,
        )

        with _record_timing(phase_timings, "agent_sec"):
            info = await _run_blocking_before_environment_cleanup(
                agent.run, str(instance.get("problem_statement", ""))
            )
        phase_timings["model"] = dict(model.timings)

        exit_status = str(info.get("exit_status") or "Unknown")
        if exit_status == "sandbox_infra_error":
            raise EnvironmentUnavailable(
                str(info.get("sandbox_infra_reason") or "resolve sandbox infrastructure error")
            )
        gate = info.get("resolve_gate") if isinstance(info.get("resolve_gate"), dict) else {}
        base_clean_fail = gate.get("base_clean_fail")
        submission = str(info.get("submission") or "")

        # Full-trajectory training tensors (resolve ends at the model's own submit/exit).
        with _record_timing(phase_timings, "build_rollout_tensors_sec"):
            (
                prompt_token_ids,
                init_prompt_text,
                response_token_ids,
                loss_mask,
                rollout_log_probs,
                rollout_routed_experts,
                response,
            ) = await _run_blocking_before_environment_cleanup(_build_rollout_tensors, model, None)

        label = _resolve_label(exit_status, base_clean_fail)
        sandbox_infra_error = exit_status == "sandbox_infra_error"
        record.update(
            {
                "agent_exit_status": exit_status,
                "agent_calls": agent.n_calls,
                "agent_submission_chars": len(submission),
                "base_clean_fail": bool(base_clean_fail) if base_clean_fail is not None else None,
                "sandbox_infra_error": sandbox_infra_error,
                "infra_failure": sandbox_infra_error,
                "raw_reward": 0.0 if sandbox_infra_error else None,
                "test_command": gate.get("test_command", ""),
                "behavior_contract": gate.get("behavior_contract"),
                "grounding_plan": gate.get("grounding_plan"),
                "test_patch": submission,
                "validation": {
                    "label": label,
                    "base_clean_fail": bool(base_clean_fail),
                    "infra_failure": sandbox_infra_error,
                },
                "trajectory_path": str(traj_path),
                "total_time": time.time() - start_time,
            }
        )

        sample.tokens = prompt_token_ids + response_token_ids
        sample.response_length = len(response_token_ids)
        sample.response = response
        sample.loss_mask = loss_mask
        sample.prompt = init_prompt_text
        sample.rollout_log_probs = rollout_log_probs
        sample.rollout_routed_experts = rollout_routed_experts
        if exit_status == "Submitted":
            sample.status = Sample.Status.COMPLETED
        elif exit_status in {"LimitsExceeded", "TimeExceeded"}:
            sample.status = Sample.Status.TRUNCATED
        else:
            sample.status = Sample.Status.ABORTED
        sample.metadata.update(
            {
                "instance_id": instance_id,
                "instance": instance,
                "rl_step": rollout_id,
                "trajectory_path": str(traj_path),
                # patch_classification reads the generated test artifact via test_code/submission;
                # for resolve that artifact is the git-diff test_patch (may edit existing files).
                "test_code": submission,
                "test_patch": submission,
                "submission": submission,
                "generated_test_command": gate.get("test_command", ""),
                "test_command": gate.get("test_command", ""),
                "behavior_contract": gate.get("behavior_contract"),
                "gentest_record": record,
                "exit_status": exit_status,
                "sandbox_infra_error": sandbox_infra_error,
                "shared_verify": shared_verify,
                "isolated_verify": not shared_verify,
                "format_error_count": 0,
                "patch_classification_verifier_warmed": warmed["done"],
            }
        )

        if sandbox_infra_error:
            sample.status = Sample.Status.ABORTED
            sample.remove_sample = True
            _zero_loss_mask(sample)
            reward_key = getattr(args, "reward_key", None)
            sample.reward = {"raw_reward": 0.0}
            if reward_key and reward_key != "raw_reward":
                sample.reward[reward_key] = 0.0

        # Reward: reuse the v6 patch_classification path verbatim (0/0.05/0.2/0.5/1.0). Generation
        # stayed gold-free; reward MAY read gold (apply gold fix to judge gold_pass + classify
        # candidate patches) — gold never enters the rollout tokens/context. Reuses the SAME warm
        # verifier sandbox before cleanup. In shared mode, the reward's official verifies use the
        # same reset transaction and do not restore the agent workspace after generation is over.
        reward_started = time.perf_counter()
        reward_sandbox_reused = await _compute_reward_before_environment_cleanup(
            args, sample, verify_env, evaluation=evaluation
        )
        if reward_sandbox_reused:
            phase_timings["reward_sec"] = time.perf_counter() - reward_started
            record["reward_sandbox_reused"] = True
        if sample.reward is None:
            # patch_classification not active (mode/custom_rm_path mismatch): fall back to the
            # gold-free label reward so the sample still carries a signal.
            reward = reward_from_record(record)
            reward_key = getattr(args, "reward_key", None)
            sample.reward = {reward_key: reward} if reward_key else reward

        assert len(sample.loss_mask) == sample.response_length
        assert sample.rollout_log_probs is None or len(sample.rollout_log_probs) == sample.response_length
        if sample.rollout_routed_experts is not None:
            assert sample.rollout_routed_experts.shape[0] == len(sample.tokens) - 1

        trajectory_dir.mkdir(parents=True, exist_ok=True)
        record["total_time"] = time.time() - start_time
        with _record_timing(phase_timings, "trajectory_save_sec"):
            agent.save(
                traj_path,
                {
                    "info": {
                        "instance_id": instance_id,
                        "sample_idx": int(sample.index or 0),
                        "gentest_record": record,
                    },
                    "instance": instance,
                    "rl_step": rollout_id,
                    "prompt_token_ids": prompt_token_ids,
                    "response_token_ids": response_token_ids,
                    "loss_mask": sample.loss_mask,
                },
            )
        logger.info(
            "[RESOLVE_GENERATE] done %s sample=%s status=%s label=%s reward=%s base_fail=%s",
            instance_id,
            sample.index,
            sample.status.value,
            label,
            sample.reward,
            base_clean_fail,
        )
        return sample
    except Exception as exc:
        record.update(
            _record_error(instance, sample, f"{type(exc).__name__}: {str(exc)[:300]}", traceback.format_exc())
        )
        sample.metadata.update({"instance_id": instance_id, "gentest_record": record})
        sample.reward = reward_from_record(record)
        sample.status = Sample.Status.ABORTED
        raise
    finally:
        if model is not None:
            model.close()
        await _cleanup_resolve_environments(
            env,
            verify_env,
            phase_timings,
            instance_id=instance_id,
        )
        phase_timings["total_sec"] = time.time() - start_time
        record["total_time"] = phase_timings["total_sec"]


def append_rollout_metrics(_rollout_id, _args, samples, rollout_extra_metrics, rollout_time) -> bool:
    """Add resolve rollout metrics (reuses the label/status-generic v6 reader), then continue."""
    if rollout_extra_metrics is not None:
        rollout_extra_metrics.update(_gentest_rollout_metrics(samples, rollout_time))
    return False
