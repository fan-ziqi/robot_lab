# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit lazy exports; local names take precedence over upstream fallbacks."""

from .commands import (
    UniformThresholdVelocityCommand,
    UniformThresholdVelocityCommandCfg,
    DiscreteCommandController,
    DiscreteCommandControllerCfg,
)
from .curriculums import command_levels_lin_vel, command_levels_ang_vel
from .events import randomize_rigid_body_inertia, randomize_com_positions, reset_root_state_uniform
from .observations import joint_pos_rel_without_wheel, phase
from .rewards import (
    track_lin_vel_xy_exp,
    track_ang_vel_z_exp,
    track_lin_vel_xy_yaw_frame_exp,
    track_ang_vel_z_world_exp,
    joint_power,
    stand_still,
    joint_pos_penalty,
    wheel_vel_penalty,
    GaitReward,
    joint_mirror,
    action_mirror,
    action_sync,
    feet_air_time,
    feet_air_time_positive_biped,
    feet_air_time_variance_penalty,
    feet_contact,
    feet_contact_without_cmd,
    feet_stumble,
    feet_distance_y_exp,
    feet_distance_xy_exp,
    feet_height,
    feet_height_body,
    feet_slide,
    upward,
    base_height_l2,
    lin_vel_z_l2,
    ang_vel_xy_l2,
    undesired_contacts,
    flat_orientation_l2,
)
from .utils import is_env_assigned_to_terrain, is_robot_on_terrain

from isaaclab.envs.mdp import *
from isaaclab_tasks.core.velocity.mdp import *
