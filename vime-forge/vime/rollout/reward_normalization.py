"""Reward normalization for rollouts that fan out into training segments.

One agent trial may emit several :class:`~vime.utils.types.Sample` rows after
trajectory reconstruction. Those rows are training segments, not independent
GRPO samples. Advantage statistics therefore have to be computed once per
unique trial and then broadcast back to every segment from that trial.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any

import torch

from vime.utils.types import Sample


def normalize_rewards_by_trajectory(
    args: Any,
    samples: list[Sample],
    raw_rewards: list[float],
) -> list[float]:
    """Normalize rewards per prompt group over unique rollout/trial ids.

    ``Sample.group_index`` identifies the prompt group and ``rollout_id``
    identifies one sampled trial. Multiple segments with the same pair must
    carry the same outcome reward. The returned list preserves sample order
    and broadcasts one trial advantage to all of its sibling segments.

    Legacy samples without ``group_index`` retain the old shape-based behavior
    so loading historical non-agent rollout dumps remains possible.
    """
    if len(samples) != len(raw_rewards):
        raise ValueError(f"samples/rewards length mismatch: {len(samples)} != {len(raw_rewards)}")
    if not samples:
        return []
    if any(sample.group_index is None for sample in samples):
        return _legacy_shape_normalize(args, raw_rewards)

    # group_index -> rollout_id -> (reward, original sample positions)
    groups: OrderedDict[int, OrderedDict[Any, tuple[float, list[int]]]] = OrderedDict()
    for position, (sample, raw_reward) in enumerate(zip(samples, raw_rewards, strict=True)):
        group_index = int(sample.group_index)
        rollout_id: Any = sample.rollout_id if sample.rollout_id is not None else sample.index
        # A completely unannotated trial cannot have known siblings. Treat it
        # as a unique row rather than accidentally merging unrelated samples.
        if rollout_id is None:
            rollout_id = ("sample_position", position)

        reward = float(raw_reward)
        trials = groups.setdefault(group_index, OrderedDict())
        prior = trials.get(rollout_id)
        if prior is None:
            trials[rollout_id] = (reward, [position])
            continue

        prior_reward, positions = prior
        if not math.isclose(reward, prior_reward, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError(
                "segments from one trial must share a reward: "
                f"group_index={group_index}, rollout_id={rollout_id!r}, "
                f"rewards={prior_reward!r} and {reward!r}"
            )
        positions.append(position)

    normalized = [0.0] * len(samples)
    use_std = (
        args.advantage_estimator in {"grpo", "gspo", "cispo"}
        and bool(getattr(args, "grpo_std_normalization", False))
    )
    for trials in groups.values():
        trial_rewards = torch.tensor([reward for reward, _ in trials.values()], dtype=torch.float)
        trial_advantages = trial_rewards - trial_rewards.mean()
        # torch.std's default unbiased estimator is undefined for one trial.
        # A singleton prompt group has zero centered advantage either way.
        if use_std and len(trial_rewards) > 1:
            trial_advantages = trial_advantages / (trial_advantages.std() + 1e-6)

        for advantage, (_reward, positions) in zip(trial_advantages.tolist(), trials.values(), strict=True):
            for position in positions:
                normalized[position] = float(advantage)
    return normalized


def _legacy_shape_normalize(args: Any, raw_rewards: list[float]) -> list[float]:
    """Preserve the pre-AgentTrial fallback for historical unannotated rows."""
    rewards = torch.tensor(raw_rewards, dtype=torch.float)
    expected = int(args.n_samples_per_prompt) * int(args.rollout_batch_size)
    if rewards.shape[-1] == expected:
        rewards = rewards.reshape(-1, int(args.n_samples_per_prompt))
    else:
        rewards = rewards.view(-1, rewards.shape[-1])
    rewards = rewards - rewards.mean(dim=-1, keepdim=True)
    if (
        args.advantage_estimator in {"grpo", "gspo", "cispo"}
        and bool(getattr(args, "grpo_std_normalization", False))
        and rewards.shape[-1] > 1
    ):
        rewards = rewards / (rewards.std(dim=-1, keepdim=True) + 1e-6)
    return rewards.flatten().tolist()


__all__ = ["normalize_rewards_by_trajectory"]
