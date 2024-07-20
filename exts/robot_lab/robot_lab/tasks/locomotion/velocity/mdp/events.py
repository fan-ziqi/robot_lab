# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable different events.

Events include anything related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`omni.isaac.lab.managers.EventTermCfg` object to enable
the event introduced by the function.
"""

from __future__ import annotations

import torch
import warnings
from typing import TYPE_CHECKING, Literal

import carb
import omni.physics.tensors.impl.api as physx

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.actuators import ImplicitActuator
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.terrains import TerrainImporter

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

from robot_lab.utils.wrappers.rsl_rl.datasets.motion_loader import AMPLoader
from robot_lab.tasks.locomotion.velocity.manager_based_rl_amp_env import ManagerBasedRLAmpEnv


def reset_root_state_amp(
    env: ManagerBasedRLAmpEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    frames = env.amp_loader.get_full_frame_batch(len(env_ids))
    # base position
    positions = AMPLoader.get_root_pos_batch(frames)
    positions = positions + root_states[:, 0:3]
    positions[:, :2] = positions[:, :2] + env.scene.env_origins[env_ids, :2]
    orientations = AMPLoader.get_root_rot_batch(frames)
    # base velocities
    lin_vel = math_utils.quat_rotate(orientations, AMPLoader.get_linear_vel_batch(frames))
    ang_vel = math_utils.quat_rotate(orientations, AMPLoader.get_angular_vel_batch(frames))
    velocities = torch.cat([lin_vel, ang_vel], dim=-1)
    velocities = velocities + root_states[:, 7:13]

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)



def reset_joints_amp(
    env: ManagerBasedRLAmpEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    frames = env.amp_loader.get_full_frame_batch(len(env_ids))
    joint_pos = AMPLoader.get_joint_pose_batch(frames)
    joint_vel = AMPLoader.get_joint_vel_batch(frames)

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

