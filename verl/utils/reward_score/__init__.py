"""Compatibility layer for reward utilities.

The reward system now mirrors ModelScope's ms-swift implementation. Reward
functions should be configured explicitly via ``reward_model.reward_funcs``.
"""

from verl.utils.import_utils import deprecated
from verl.reward import (
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
    "RewardFunction",
    "RewardSpec",
    "register_reward",
    "list_registered_rewards",
    "resolve_reward_cls",
    "instantiate_reward",
    "build_reward_specs",
    "reward_specs_to_dict",
    "default_compute_score",
]


def default_compute_score(*_args, **_kwargs):
    raise RuntimeError(
        "default_compute_score has been removed. Configure reward_model.reward_funcs "
        "with explicit entries or supply a custom reward module."
    )


@deprecated("verl.reward.register_reward")
def _default_compute_score(*args, **kwargs):
    return default_compute_score(*args, **kwargs)
