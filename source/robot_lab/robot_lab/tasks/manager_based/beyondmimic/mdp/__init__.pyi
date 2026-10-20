# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Explicit lazy exports; local names take precedence over upstream fallbacks."""

from .commands import MotionLoader, MotionCommand, MotionCommandCfg
from .events import randomize_joint_default_pos
from .observations import (
    robot_anchor_ori_w,
    robot_anchor_lin_vel_w,
    robot_anchor_ang_vel_w,
    robot_body_pos_b,
    robot_body_ori_b,
    motion_anchor_pos_b,
    motion_anchor_ori_b,
)
from .rewards import (
    motion_global_anchor_position_error_exp,
    motion_global_anchor_orientation_error_exp,
    motion_relative_body_position_error_exp,
    motion_relative_body_orientation_error_exp,
    motion_global_body_linear_velocity_error_exp,
    motion_global_body_angular_velocity_error_exp,
    feet_contact_time,
)
from .terminations import (
    bad_anchor_pos,
    bad_anchor_pos_z_only,
    bad_anchor_ori,
    bad_motion_body_pos,
    bad_motion_body_pos_z_only,
)

from isaaclab.envs.mdp import *
