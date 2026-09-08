"""Reward helpers for mini_swe generated-test rollouts."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from slime.utils.types import Sample

logger = logging.getLogger(__name__)

DEFAULT_LABEL_REWARDS = {
    "gold_validated": 1.0,
    "base_fail_only": 0.5,
    "gold_failed": 0.0,
    "base_fail_with_quality_flags": 0.0,
    "base_not_clean_fail": -1.0,
    "base_infra_failure": -1.0,
    "gold_apply_failure": -1.0,
    "gold_infra_failure": -1.0,
    "missing_test": -1.0,
}


def _label_reward_map() -> dict[str, float]:
    raw = os.environ.get("GENTEST_REWARD_LABELS_JSON")
    if not raw:
        return dict(DEFAULT_LABEL_REWARDS)
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Invalid GENTEST_REWARD_LABELS_JSON=%r: %s", raw, exc)
        return dict(DEFAULT_LABEL_REWARDS)
    rewards = dict(DEFAULT_LABEL_REWARDS)
    for key, value in values.items():
        try:
            rewards[str(key)] = float(value)
        except (TypeError, ValueError):
            logger.warning("Ignoring non-numeric gentest reward override %r=%r", key, value)
    return rewards


def reward_from_record(record: dict[str, Any] | None) -> float:
    record = record or {}
    exit_status = str(record.get("agent_exit_status") or "")
    if exit_status != "Submitted":
        return float(os.environ.get("GENTEST_UNSUBMITTED_REWARD", "-1.0"))

    validation = record.get("validation") if isinstance(record.get("validation"), dict) else {}
    label = str(validation.get("label") or "")
    rewards = _label_reward_map()
    return float(rewards.get(label, os.environ.get("GENTEST_DEFAULT_REWARD", "-1.0")))


async def reward_func(args, sample: Sample | list[Sample], **kwargs):
    if isinstance(sample, list):
        return [await reward_func(args, item, **kwargs) for item in sample]
    if sample.reward is not None:
        return sample.reward
    record = sample.metadata.get("gentest_record") if isinstance(sample.metadata, dict) else {}
    return reward_from_record(record)
