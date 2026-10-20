# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit lazy exports; local names take precedence over upstream fallbacks."""

__all__ = [
    "DiscreteCommandController",
    "DiscreteCommandControllerCfg",
    "GaitReward",
    "UniformThresholdVelocityCommand",
    "UniformThresholdVelocityCommandCfg",
    "action_mirror",
    "action_sync",
    "ang_vel_xy_l2",
    "base_height_l2",
    "command_levels_ang_vel",
    "command_levels_lin_vel",
    "feet_air_time",
    "feet_air_time_positive_biped",
    "feet_air_time_variance_penalty",
    "feet_contact",
    "feet_contact_without_cmd",
    "feet_distance_xy_exp",
    "feet_distance_y_exp",
    "feet_height",
    "feet_height_body",
    "feet_slide",
    "feet_stumble",
    "flat_orientation_l2",
    "is_env_assigned_to_terrain",
    "is_robot_on_terrain",
    "joint_mirror",
    "joint_pos_penalty",
    "joint_pos_rel_without_wheel",
    "joint_power",
    "lin_vel_z_l2",
    "phase",
    "reset_root_state_uniform",
    "stand_still",
    "track_ang_vel_z_exp",
    "track_ang_vel_z_world_exp",
    "track_lin_vel_xy_exp",
    "track_lin_vel_xy_yaw_frame_exp",
    "undesired_contacts",
    "upward",
    "wheel_vel_penalty",
]

from isaaclab.envs.mdp import *

from isaaclab_tasks.core.velocity.mdp import *

from .commands import (
    DiscreteCommandController,
    DiscreteCommandControllerCfg,
    UniformThresholdVelocityCommand,
    UniformThresholdVelocityCommandCfg,
)
from .curriculums import command_levels_ang_vel, command_levels_lin_vel
from .events import reset_root_state_uniform
from .observations import joint_pos_rel_without_wheel, phase
from .rewards import (
    GaitReward,
    action_mirror,
    action_sync,
    ang_vel_xy_l2,
    base_height_l2,
    feet_air_time,
    feet_air_time_positive_biped,
    feet_air_time_variance_penalty,
    feet_contact,
    feet_contact_without_cmd,
    feet_distance_xy_exp,
    feet_distance_y_exp,
    feet_height,
    feet_height_body,
    feet_slide,
    feet_stumble,
    flat_orientation_l2,
    joint_mirror,
    joint_pos_penalty,
    joint_power,
    lin_vel_z_l2,
    stand_still,
    track_ang_vel_z_exp,
    track_ang_vel_z_world_exp,
    track_lin_vel_xy_exp,
    track_lin_vel_xy_yaw_frame_exp,
    undesired_contacts,
    upward,
    wheel_vel_penalty,
)
from .utils import is_env_assigned_to_terrain, is_robot_on_terrain
