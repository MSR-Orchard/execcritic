"""Dynamic sampling filters for mini_swe generated-test rollouts."""

from __future__ import annotations

import torch

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.types import Sample


def check_no_aborted(args, samples: list[Sample], **kwargs):
    if any(sample.status == Sample.Status.ABORTED for sample in samples):
        return DynamicFilterOutput(keep=False, reason="aborted")
    return DynamicFilterOutput(keep=True)


def mask_aborted(args, samples: list[Sample], **kwargs):
    """Keep the group but remove aborted samples from the training loss."""
    del args, kwargs
    for sample in samples:
        if sample.status != Sample.Status.ABORTED:
            continue
        sample.remove_sample = True
        sample.loss_mask = [0] * sample.response_length
    return DynamicFilterOutput(keep=True)


def check_no_aborted_nonzero_std_and_pos_reward(args, samples: list[Sample], **kwargs):
    if any(sample.status == Sample.Status.ABORTED for sample in samples):
        return DynamicFilterOutput(keep=False, reason="aborted")
    if any(sample.reward is None for sample in samples):
        return DynamicFilterOutput(keep=False, reason="missing_reward")

    rewards = [float(sample.get_reward_value(args)) for sample in samples]
    reward_tensor = torch.tensor(rewards, dtype=torch.float64)
    if reward_tensor.std() <= 1e-6:
        return DynamicFilterOutput(keep=False, reason=f"zero_std_{round(rewards[0], 1)}")
    if not any(reward > 0 for reward in rewards):
        return DynamicFilterOutput(keep=False, reason="no_positive_reward")
    return DynamicFilterOutput(keep=True)
