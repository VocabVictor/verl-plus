"""Reward helpers mirroring the ms-swift reward interface."""

from .registry import (
    CaptionReward,
    GainReward,
    RewardFunction,
    RewardSpec,
    register_reward,
    list_registered_rewards,
    resolve_reward_cls,
    instantiate_reward,
    build_reward_specs,
    reward_specs_to_dict,
)

__all__ = [
    "CaptionReward",
    "GainReward",
    "RewardFunction",
    "RewardSpec",
    "register_reward",
    "list_registered_rewards",
    "resolve_reward_cls",
    "instantiate_reward",
    "build_reward_specs",
    "reward_specs_to_dict",
]
