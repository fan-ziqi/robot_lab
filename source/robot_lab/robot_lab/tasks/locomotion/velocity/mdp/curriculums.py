# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def command_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], reward_term_name: str, max_curriculum: float = 1.0
) -> None:
    """Curriculum based on the tracking reward of the robot when commanded to move at a desired velocity.

    This term is used to increase the range of commands when the robot's tracking reward is above 80% of the
    maximum.

    Returns:
        The cumulative increase in velocity command range.
    """
    episode_sums = env.reward_manager._episode_sums[reward_term_name]
    reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    delta_range = torch.tensor([-0.1, 0.1], device=env.device)
    if not hasattr(env, "delta_lin_vel"):
        env.delta_lin_vel = torch.tensor(0.0, device=env.device)
    # If the tracking reward is above 80% of the maximum, increase the range of commands
    if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
        lin_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        lin_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        base_velocity_ranges.lin_vel_x = torch.clamp(lin_vel_x + delta_range, -max_curriculum, max_curriculum).tolist()
        base_velocity_ranges.lin_vel_y = torch.clamp(lin_vel_y + delta_range, -max_curriculum, max_curriculum).tolist()
        env.delta_lin_vel = torch.clamp(env.delta_lin_vel + delta_range[1], 0.0, max_curriculum)
    return env.delta_lin_vel
