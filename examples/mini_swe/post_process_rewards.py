import logging
import math
import os
from typing import Any

from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def naive_reinforce(args, samples: list[Sample], **kwargs):
    """Return raw rewards without any normalization. Replaces None rewards with 0.0."""
    raw_rewards = [sample.get_reward_value(args) for sample in samples]

    none_count = sum(1 for r in raw_rewards if r is None)
    if none_count > 0:
        logger.warning(
            f"Found {none_count} samples with None rewards (e.g., aborted/failed samples). "
            "Replacing with 0.0."
        )
        raw_rewards = [r if r is not None else 0.0 for r in raw_rewards]

    return raw_rewards, raw_rewards


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _reward_component(sample: Sample, key: str) -> float:
    reward = getattr(sample, "reward", None)
    if isinstance(reward, dict):
        return _as_float(reward.get(key), 0.0)
    return 0.0


def _validation_label(sample: Sample) -> str:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    record = metadata.get("gentest_record") if isinstance(metadata.get("gentest_record"), dict) else {}
    validation = record.get("validation") if isinstance(record.get("validation"), dict) else {}
    return str(validation.get("label") or "")


def _agent_calls(sample: Sample) -> int | None:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    record = metadata.get("gentest_record") if isinstance(metadata.get("gentest_record"), dict) else {}
    calls = record.get("agent_calls")
    try:
        return int(calls)
    except (TypeError, ValueError):
        return None


def _apply_same_reward_turn_penalty(samples: list[Sample], raw_rewards: list[float], group_size: int) -> list[float]:
    """Halve ``raw_reward`` for over-exploring winners within each rollout group.

    For samples whose ``raw_reward`` lands in a penalized tier (default {1.0, 0.5, 0.2}), find the
    minimum turn count among same-tier siblings in the same group; any sample using more than
    ``+threshold`` turns beyond that minimum gets ``raw_reward *= factor``. Only ``raw_reward`` is
    touched — ``base_reward``/``gold_reward`` diagnostics stay intact. Disabled unless
    ``SWE_TURN_PENALTY_ENABLE=1``.
    """
    if os.environ.get("SWE_TURN_PENALTY_ENABLE", "0") != "1":
        return raw_rewards

    threshold = int(_as_float(os.environ.get("SWE_TURN_PENALTY_TURN_MARGIN"), 10.0))
    factor = _as_float(os.environ.get("SWE_TURN_PENALTY_FACTOR"), 0.5)
    tiers_raw = os.environ.get("SWE_TURN_PENALTY_TIERS", "1.0,0.5,0.2")
    tiers = set()
    for piece in tiers_raw.split(","):
        piece = piece.strip()
        if piece:
            tiers.add(round(_as_float(piece), 2))

    if len(raw_rewards) % group_size == 0:
        group_bounds = [(start, start + group_size) for start in range(0, len(raw_rewards), group_size)]
    else:
        group_bounds = [(0, len(raw_rewards))]

    penalized = list(raw_rewards)
    penalized_count = 0
    eligible_count = 0
    for start, end in group_bounds:
        # Minimum turn count per penalized reward tier within this group.
        tier_min_turn: dict[float, int] = {}
        for idx in range(start, end):
            tier = round(penalized[idx], 2)
            if tier not in tiers:
                continue
            calls = _agent_calls(samples[idx])
            if calls is None:
                continue
            if tier not in tier_min_turn or calls < tier_min_turn[tier]:
                tier_min_turn[tier] = calls
        for idx in range(start, end):
            tier = round(penalized[idx], 2)
            if tier not in tiers or tier not in tier_min_turn:
                continue
            eligible_count += 1
            if samples[idx].metadata is None:
                samples[idx].metadata = {}
            samples[idx].metadata["turn_penalty_eligible"] = True
            samples[idx].metadata["turn_penalty_applied"] = False
            calls = _agent_calls(samples[idx])
            if calls is None or calls <= tier_min_turn[tier] + threshold:
                continue
            new_value = penalized[idx] * factor
            penalized[idx] = new_value
            penalized_count += 1
            samples[idx].metadata["turn_penalty_applied"] = True
            if isinstance(samples[idx].reward, dict):
                samples[idx].reward["raw_reward"] = new_value

    if eligible_count:
        logger.info(
            "turn_penalty: eligible=%d penalized=%d (%.1f%%) factor=%.2f margin=%d tiers=%s",
            eligible_count,
            penalized_count,
            100.0 * penalized_count / eligible_count,
            factor,
            threshold,
            sorted(tiers),
        )
    return penalized


def patch_classification_raw_reward_normalization(args, samples: list[Sample], **kwargs):
    """Normalize precomputed ``raw_reward`` values within each rollout group."""
    raw_rewards = [_reward_component(sample, "raw_reward") for sample in samples]
    group_size = int(getattr(args, "n_samples_per_prompt", 1))
    if group_size <= 0:
        raise ValueError(f"n_samples_per_prompt must be positive, got {group_size}")

    raw_rewards = _apply_same_reward_turn_penalty(samples, raw_rewards, group_size)

    if len(raw_rewards) % group_size == 0:
        reward_groups = [
            raw_rewards[start : start + group_size]
            for start in range(0, len(raw_rewards), group_size)
        ]
    else:
        reward_groups = [raw_rewards]

    rewards_normalization = bool(getattr(args, "rewards_normalization", True))
    std_normalization = bool(getattr(args, "grpo_std_normalization", True))

    # Aborted samples (e.g. dead-sandbox trajectories) carry a placeholder reward
    # and a zeroed loss mask; they must not shift the group mean/std that the other
    # samples' advantages are centered on. Compute group statistics over the
    # non-aborted members only, and give each aborted sample a normalized reward of 0.
    excluded = [sample.status == Sample.Status.ABORTED or bool(sample.remove_sample) for sample in samples]

    normalized_rewards = []
    offset = 0
    for reward_group in reward_groups:
        group_excluded = excluded[offset : offset + len(reward_group)]
        offset += len(reward_group)
        if not rewards_normalization:
            normalized_rewards.extend(0.0 if drop else reward for reward, drop in zip(reward_group, group_excluded))
            continue
        live = [reward for reward, drop in zip(reward_group, group_excluded) if not drop]
        if not live:
            normalized_rewards.extend(0.0 for _ in reward_group)
            continue
        group_mean = sum(live) / len(live)
        group_std = 0.0
        if std_normalization and len(live) > 1:
            variance = sum((r - group_mean) ** 2 for r in live) / (len(live) - 1)
            group_std = math.sqrt(variance)
        for r, drop in zip(reward_group, group_excluded):
            if drop:
                normalized_rewards.append(0.0)
                continue
            centered = r - group_mean
            if std_normalization and len(live) > 1:
                centered = centered / (group_std + 1e-6)
            normalized_rewards.append(centered)

    for sample, normalized_reward in zip(samples, normalized_rewards, strict=True):
        if isinstance(sample.reward, dict):
            sample.reward["normalized_reward"] = normalized_reward
        if sample.metadata is None:
            sample.metadata = {}
        sample.metadata["patch_classification_normalized_reward"] = normalized_reward

    included_raw_rewards = [reward for reward, drop in zip(raw_rewards, excluded) if not drop]
    included_normalized_rewards = [reward for reward, drop in zip(normalized_rewards, excluded) if not drop]
    logger.info(
        "patch_classification_raw_reward_normalization: total=%d excluded=%d raw_mean=%.4f normalized_mean=%.4f",
        len(raw_rewards),
        sum(excluded),
        sum(included_raw_rewards) / len(included_raw_rewards) if included_raw_rewards else 0.0,
        sum(included_normalized_rewards) / len(included_normalized_rewards) if included_normalized_rewards else 0.0,
    )

    return raw_rewards, normalized_rewards


def patch_classification_rule_based(args, samples: list[Sample], **kwargs):
    """Rule-based shaping with group-mean-centered training rewards.

    Reward rule:
      - 1.0 if it passes with the gold patch;
      - 0.1 if it fails on base but does not pass with the gold patch;
      - 0.0 for base_not_clean_fail;
      - 0.0 if it passes on the base repo or otherwise lacks base-fail signal.
    """
    base_score = _as_float(os.environ.get("SWE_PATCH_CLASSIFICATION_POST_REWARD_BASE_SCORE"), 0.1)
    success_score = _as_float(os.environ.get("SWE_PATCH_CLASSIFICATION_POST_REWARD_SUCCESS_SCORE"), 1.0)

    reward_key = getattr(args, "reward_key", None)

    shaped_rewards = []
    zero_count = 0
    base_only_count = 0
    base_not_clean_count = 0
    success_count = 0

    for sample in samples:
        base_reward = _reward_component(sample, "base_reward")
        gold_reward = _reward_component(sample, "gold_reward")
        label = _validation_label(sample)

        if gold_reward > 0.0:
            reward = success_score
            success_count += 1
        elif label == "base_not_clean_fail":
            reward = 0.0
            base_not_clean_count += 1
        elif base_reward > 0.0:
            reward = base_score
            base_only_count += 1
        else:
            reward = 0.0
            zero_count += 1
        if sample.metadata is None:
            sample.metadata = {}
        sample.metadata["patch_classification_post_reward"] = reward
        # Write the shaped scalar back onto sample.reward[reward_key] so wandb
        # metrics (zero_std, reward histograms) read the same {0.0, 0.1, 1.0}
        # signal that is actually optimized. base_reward/gold_reward stay intact
        # as raw diagnostic components.
        if reward_key is not None and isinstance(sample.reward, dict):
            sample.reward[reward_key] = reward
        shaped_rewards.append(reward)

    logger.info(
        (
            "patch_classification_rule_based: total=%d zero=%d base_only=%d "
            "base_not_clean=%d success=%d"
        ),
        len(shaped_rewards),
        zero_count,
        base_only_count,
        base_not_clean_count,
        success_count,
    )
    group_size = int(getattr(args, "n_samples_per_prompt", 1))
    if group_size <= 0:
        raise ValueError(f"n_samples_per_prompt must be positive, got {group_size}")

    if len(shaped_rewards) % group_size == 0:
        reward_groups = [
            shaped_rewards[start : start + group_size]
            for start in range(0, len(shaped_rewards), group_size)
        ]
    else:
        reward_groups = [shaped_rewards]

    centered_rewards = []
    for reward_group in reward_groups:
        group_mean = sum(reward_group) / len(reward_group)
        centered_rewards.extend(reward - group_mean for reward in reward_group)

    return shaped_rewards, centered_rewards
