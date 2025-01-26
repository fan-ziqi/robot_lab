# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def flip_ang_vel_around_axis(
    env: ManagerBasedRLEnv, axis: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    axis_dir, axis_num = (1 if axis[0] == "+" else -1), {"x": 0, "y": 1, "z": 2}[axis[1]]
    current_time = env.episode_length_buf * env.step_dt
    ang_vel = axis_dir * asset.data.root_com_ang_vel_b[:, axis_num].clamp(max=7.2, min=-7.2)
    return ang_vel * torch.logical_and(current_time > 0.5, current_time < 1.0)


def flip_ang_vel_z(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.abs(asset.data.root_com_ang_vel_b[:, 2])


def flip_lin_vel_z(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    current_time = env.episode_length_buf * env.step_dt
    lin_vel = asset.data.root_com_lin_vel_b[:, 2].clamp(max=3)
    return lin_vel * torch.logical_and(current_time > 0.5, current_time < 0.75)


def flip_orientation_control(
    env: ManagerBasedRLEnv, axis: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    axis_vector = [(1 if axis[0] == "+" else -1) * a for a in {"x": [1, 0, 0], "y": [0, 1, 0]}[axis[1]]]
    current_time = env.episode_length_buf * env.step_dt
    phase = (current_time - 0.5).clamp(min=0, max=0.5)
    quat = math_utils.quat_from_angle_axis(
        4 * phase * torch.pi, torch.tensor(axis_vector, device=env.device, dtype=torch.float)
    )
    desired_base_quat = math_utils.quat_mul(
        quat, torch.tensor(asset.cfg.init_state.rot, device=env.device).reshape(1, -1).repeat(env.num_envs, 1)
    )
    inv_desired_base_quat = math_utils.quat_inv(desired_base_quat)
    desired_projected_gravity = math_utils.transform_points(
        points=asset.data.GRAVITY_VEC_W.unsqueeze(1), quat=inv_desired_base_quat
    ).squeeze(1)
    orientation_diff = torch.sum(torch.square(asset.data.projected_gravity_b - desired_projected_gravity), dim=1)
    return orientation_diff


def flip_feet_height_before_backflip(
    env: ManagerBasedRLEnv, foot_radius: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    feet_height = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2] - foot_radius
    current_time = env.episode_length_buf * env.step_dt
    return feet_height.clamp(min=0).sum(dim=1) * (
        current_time < 0.5
    )  # torch.logical_or(current_time < 0.5, current_time > 1.0)


def flip_height_control(
    env: ManagerBasedRLEnv, target_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    current_time = env.episode_length_buf * env.step_dt
    height_diff = torch.square(target_height - asset.data.root_link_pos_w[:, 2]) * torch.logical_or(
        current_time < 0.4, current_time > 1.4
    )
    return height_diff


def flip_actions_symmetry(env: ManagerBasedRLEnv) -> torch.Tensor:
    actions_diff = torch.square(env.action_manager.action[:, 0] + env.action_manager.action[:, 3])
    actions_diff += torch.square(env.action_manager.action[:, 1:3] - env.action_manager.action[:, 4:6]).sum(dim=-1)
    actions_diff += torch.square(env.action_manager.action[:, 6] + env.action_manager.action[:, 9])
    actions_diff += torch.square(env.action_manager.action[:, 7:9] - env.action_manager.action[:, 10:12]).sum(dim=-1)
    return actions_diff


def flip_gravity_axis(
    env: ManagerBasedRLEnv, axis: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    axis_num = {"x": 0, "y": 1, "z": 2}[axis[1]]
    return torch.square(asset.data.projected_gravity_b[:, axis_num])
