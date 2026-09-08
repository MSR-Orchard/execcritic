"""Outcome reward for the solve -> F2P-guided repair rollout.

Raw sample reward:

* 1.5: the round-0 submission passes the persistent verifier's official F2P + P2P suite;
* 1.0: a submission after entering repair passes the official F2P + P2P suite;
* 0.2: official evaluation is unresolved, but the final submitted patch passed
  the selected in-rollout F2P repair gate;
* 0.1: the sample made a valid submission but did not pass that repair gate;
* 0.0: no valid submission or an integrity violation such as modifying tests.

The reward model returns ``raw_reward`` and ``reward`` plus four cumulative
binary outcome components: ``submit_reward``, ``repair_reward``,
``pass_reward``, and ``direct_reward``. ``center_rewards_by_group`` replaces
``reward`` with the mean-centered training value for its GRPO prompt group; it
intentionally does not divide by the group standard deviation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from slime.utils.types import Sample

from .swe_reward import (
    SWE_TIMEOUT_REWARD_TOTAL,
    extract_patch_from_output,
    is_valid_patch,
    run_tests_in_docker,
)

logger = logging.getLogger(__name__)


def _score(name: str, default: float, *, maximum: float | None = 1.0) -> float:
    value = float(os.environ.get(name, default))
    if not math.isfinite(value) or value < 0.0 or (maximum is not None and value > maximum):
        interval = f"[0, {maximum:g}]" if maximum is not None else "[0, infinity)"
        raise ValueError(f"{name} must be in {interval}, got {value}")
    return value


def _repair_metadata(sample: Sample) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    repair = metadata.get("f2p_repair")
    return metadata, repair if isinstance(repair, dict) else {}


def _test_list(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value]


def _write_reward_details(sample: Sample, details: dict[str, Any]) -> None:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    trajectory_path = metadata.get("trajectory_path")
    if not trajectory_path:
        return

    path = Path(str(trajectory_path))
    reward_path = path.with_suffix(".reward.json")
    try:
        reward_path.write_text(json.dumps(details, ensure_ascii=False, indent=2))
    except OSError as exc:
        logger.warning("Failed to write repair reward details to %s: %s", reward_path, exc)


def _record_reward(
    sample: Sample,
    *,
    reward: float,
    reason: str,
    submit_reward: float = 0.0,
    repair_reward: float = 0.0,
    pass_reward: float = 0.0,
    direct_reward: float = 0.0,
    official: dict[str, Any] | None = None,
) -> dict[str, float]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    sample.metadata = metadata
    official = official or {}
    details = {
        "raw_reward": reward,
        # ``reward`` is overwritten with the group-centered training reward
        # in ``center_rewards_by_group``.
        "reward": reward,
        "submit_reward": submit_reward,
        "repair_reward": repair_reward,
        "pass_reward": pass_reward,
        "direct_reward": direct_reward,
        "reason": reason,
        "official_resolved": bool(official.get("resolved")),
        "official_resolution": official.get("resolution", ""),
        "official_f2p_passed": official.get("f2p_passed", 0),
        "official_f2p_total": official.get("f2p_total", 0),
        "official_p2p_passed": official.get("p2p_passed", 0),
        "official_p2p_total": official.get("p2p_total", 0),
        "official_eval_seconds": float(
            official.get("reward_eval_seconds", official.get("reward_eval_time", 0.0)) or 0.0
        ),
        "official_error": str(official.get("error") or "")[-1000:],
    }
    metadata["repair_reward"] = details
    _write_reward_details(sample, details)

    logger.info(
        "[REPAIR_REWARD] instance=%s reward=%.3f submit=%.1f repair=%.1f pass=%.1f direct=%.1f reason=%s",
        metadata.get("instance_id", "unknown"),
        reward,
        submit_reward,
        repair_reward,
        pass_reward,
        direct_reward,
        reason,
    )
    return {
        "raw_reward": reward,
        "reward": reward,
        "submit_reward": submit_reward,
        "repair_reward": repair_reward,
        "pass_reward": pass_reward,
        "direct_reward": direct_reward,
    }


async def _reward_one(args, sample: Sample, **kwargs) -> dict[str, float]:
    del args, kwargs
    full_reward = _score("SWE_REPAIR_FULL_REWARD", 1.0)
    direct_score = _score("SWE_REPAIR_DIRECT_REWARD", full_reward, maximum=None)
    f2p_reward = _score("SWE_REPAIR_F2P_REWARD", 0.2)
    submit_score = _score("SWE_REPAIR_SUBMIT_REWARD", 0.1)
    zero_reward = _score("SWE_REPAIR_ZERO_REWARD", 0.0)

    metadata, repair = _repair_metadata(sample)
    forbidden = repair.get("final_forbidden_test_paths") or metadata.get("final_forbidden_test_paths") or []
    if forbidden:
        return _record_reward(sample, reward=zero_reward, reason="forbidden_test_modification")

    if not repair.get("had_submission"):
        return _record_reward(sample, reward=zero_reward, reason="no_valid_submission")

    final_output = str(metadata.get("final_output") or "")
    patch = extract_patch_from_output(final_output)
    if not final_output or not is_valid_patch(patch):
        return _record_reward(sample, reward=zero_reward, reason="invalid_or_empty_patch")

    precomputed_official = repair.get("official_verify")
    if isinstance(precomputed_official, dict) and precomputed_official:
        official = dict(precomputed_official)
    else:
        official_started = time.monotonic()
        try:
            official = await asyncio.wait_for(
                run_tests_in_docker(
                    instance_id=str(metadata.get("instance_id") or "unknown"),
                    patch=patch,
                    repo=str(metadata.get("repo") or ""),
                    base_commit=str(metadata.get("base_commit") or ""),
                    fail_to_pass=_test_list(metadata.get("FAIL_TO_PASS")),
                    pass_to_pass=_test_list(metadata.get("PASS_TO_PASS")),
                    instance=metadata,
                ),
                timeout=SWE_TIMEOUT_REWARD_TOTAL,
            )
        except asyncio.TimeoutError:
            official_eval_seconds = time.monotonic() - official_started
            logger.error(
                "[REPAIR_REWARD] official evaluation timed out after %.2fs (limit=%ss) for instance=%s",
                official_eval_seconds,
                SWE_TIMEOUT_REWARD_TOTAL,
                metadata.get("instance_id", "unknown"),
            )
            return _record_reward(
                sample,
                reward=f2p_reward if repair.get("f2p_passed") else submit_score,
                reason="official_eval_timeout",
                submit_reward=1.0,
                repair_reward=float(bool(repair.get("f2p_passed"))),
                official={
                    "error": f"official evaluation timed out after {SWE_TIMEOUT_REWARD_TOTAL}s",
                    "reward_eval_seconds": official_eval_seconds,
                },
            )
        official["reward_eval_seconds"] = time.monotonic() - official_started
    if official.get("error"):
        return _record_reward(
            sample,
            reward=f2p_reward if repair.get("f2p_passed") else submit_score,
            reason="official_eval_error",
            submit_reward=1.0,
            repair_reward=float(bool(repair.get("f2p_passed"))),
            official=official,
        )
    if official.get("resolved"):
        repair_rounds_started = int(
            repair.get("repair_rounds_started", repair.get("repair_rounds_used", 0)) or 0
        )
        direct_pass = repair_rounds_started == 0
        return _record_reward(
            sample,
            reward=direct_score if direct_pass else full_reward,
            reason="official_resolved_direct" if direct_pass else "official_resolved",
            submit_reward=1.0,
            repair_reward=1.0,
            pass_reward=1.0,
            direct_reward=float(direct_pass),
            official=official,
        )
    if repair.get("f2p_passed"):
        return _record_reward(
            sample,
            reward=f2p_reward,
            reason="repair_f2p_passed",
            submit_reward=1.0,
            repair_reward=1.0,
            official=official,
        )
    return _record_reward(
        sample,
        reward=submit_score,
        reason="official_unresolved",
        submit_reward=1.0,
        official=official,
    )


async def reward_func(
    args,
    sample: Sample | list[Sample],
    **kwargs,
) -> dict[str, float] | list[dict[str, float]]:
    """Slime custom-RM entrypoint supporting both single and batched dispatch."""
    if isinstance(sample, list):
        return await asyncio.gather(*(_reward_one(args, item, **kwargs) for item in sample))
    return await _reward_one(args, sample, **kwargs)


def center_rewards_by_group(args, samples: list[Sample], **kwargs):
    """Mean-center non-aborted rewards within each GRPO prompt group.

    Aborted rollouts are missing observations rather than zero-reward policy
    outcomes. They remain in ``raw_rewards`` for accounting, but receive zero
    centered reward and do not contribute to the group baseline.
    """
    del kwargs
    reward_key = getattr(args, "reward_key", None)
    if reward_key != "raw_reward":
        raise ValueError(f"repair reward components require --reward-key raw_reward, got {reward_key!r}")

    raw_rewards = []
    for sample in samples:
        value = None if sample.reward is None else sample.get_reward_value(args)
        if value is None:
            if sample.status != Sample.Status.ABORTED:
                raise ValueError(
                    "Missing reward for non-aborted repair sample "
                    f"index={sample.index} group_index={sample.group_index} status={sample.status.value}"
                )
            reward_components = _record_reward(
                sample,
                reward=_score("SWE_REPAIR_ZERO_REWARD", 0.0),
                reason="rollout_aborted",
            )
            sample.reward = reward_components
            value = reward_components[reward_key]
        raw_rewards.append(float(value))
    group_size = int(getattr(args, "n_samples_per_prompt", 1))
    if group_size <= 0:
        raise ValueError(f"n_samples_per_prompt must be positive, got {group_size}")

    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for position, sample in enumerate(samples):
        key = (
            ("group_index", int(sample.group_index))
            if sample.group_index is not None
            else ("position_chunk", position // group_size)
        )
        groups[key].append(position)

    centered = [0.0] * len(samples)
    for positions in groups.values():
        valid_positions = [
            position for position in positions if samples[position].status != Sample.Status.ABORTED
        ]
        mean = (
            sum(raw_rewards[position] for position in valid_positions) / len(valid_positions)
            if valid_positions
            else 0.0
        )
        for position in positions:
            sample = samples[position]
            centered[position] = (
                0.0 if sample.status == Sample.Status.ABORTED else raw_rewards[position] - mean
            )
            if not isinstance(sample.reward, dict):
                raise TypeError(
                    "repair reward post-processing expected reward components, "
                    f"got {type(sample.reward).__name__} for sample index={sample.index}"
                )
            sample.reward["reward"] = centered[position]

            metadata = sample.metadata
            if not isinstance(metadata, dict):
                metadata = {}
                sample.metadata = metadata
            metadata["repair_centered_reward"] = centered[position]
            details = metadata.get("repair_reward")
            if isinstance(details, dict):
                details["reward"] = centered[position]
                _write_reward_details(sample, details)

    logger.info(
        "[REPAIR_REWARD] centered %d samples across %d GRPO groups",
        len(samples),
        len(groups),
    )
    return raw_rewards, centered
