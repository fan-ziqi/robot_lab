# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def handstand_feet_height_exp(
    env: ManagerBasedRLEnv,
    std: float,
    target_height: dict[str, float],
    foot_name: dict[str, str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    # Extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    state_commands = env.command_manager.get_command("state_command").to(torch.int64)

    # 生成 keys 和 values
    target_height_keys = torch.tensor(list(map(int, target_height.keys())), device=env.device, dtype=torch.long)
    target_height_values = torch.tensor(list(target_height.values()), device=env.device, dtype=torch.float32)

    # 创建索引查找表
    target_height_map = torch.full((target_height_keys.max() + 1,), 0.0, device=env.device, dtype=torch.float32)
    target_height_map[target_height_keys] = target_height_values

    # 通过索引查找目标高度
    target_heights = target_height_map[state_commands.clamp(0, target_height_keys.max())]

    # 处理 foot_name -> body_id
    foot_name_keys = torch.tensor(list(map(int, foot_name.keys())), device=env.device, dtype=torch.long)
    foot_name_values = [asset.find_bodies(name) for name in foot_name.values()]

    # 处理 body_id 可能是嵌套列表的情况
    foot_body_ids = torch.full((foot_name_keys.max() + 1,), -1, device=env.device, dtype=torch.long)
    for k, body_id_list in zip(foot_name_keys, foot_name_values):
        if body_id_list and isinstance(body_id_list[0], list):
            foot_body_ids[k] = body_id_list[0][0]
        elif body_id_list:
            foot_body_ids[k] = body_id_list[0]

    # 查找对应的 body_id
    body_ids = foot_body_ids[state_commands.clamp(0, foot_name_keys.max())].unsqueeze(1)

    # 获取足部高度
    feet_height = asset.data.body_pos_w[torch.arange(env.num_envs, device=env.device).unsqueeze(1), body_ids, 2]

    # 计算误差
    feet_height_error = torch.sum(torch.square(feet_height - target_heights[:, None]), dim=1)

    return torch.exp(-feet_height_error / std**2)


def handstand_feet_on_air(
    env: ManagerBasedRLEnv, foot_name: dict[str, str], sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    # 提取传感器数据
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    state_commands = env.command_manager.get_command("state_command").to(torch.int64)

    # 预处理 foot_name 映射，提高查找效率
    foot_name_keys = torch.tensor(list(map(int, foot_name.keys())), device=env.device, dtype=torch.long)
    foot_name_values = list(foot_name.values())

    # 预分配 body_id 索引表
    max_command = foot_name_keys.max().item() if len(foot_name_keys) > 0 else 0
    body_id_map = torch.full((max_command + 1,), -1, device=env.device, dtype=torch.long)

    # 填充索引表
    for key, name in zip(foot_name_keys, foot_name_values):
        body_ids = contact_sensor.find_bodies(name)
        if body_ids:
            body_id_map[key] = body_ids[0] if isinstance(body_ids[0], int) else body_ids[0][0]

    # 通过索引查找 body_id
    body_ids = body_id_map[state_commands.clamp(0, max_command)].unsqueeze(1)

    # 计算 first_air（避免重复计算）
    first_air = contact_sensor.compute_first_air(env.step_dt)

    # 获取当前环境的索引
    env_indices = torch.arange(env.num_envs, device=env.device).unsqueeze(1)

    # 直接索引
    first_air_vals = first_air[env_indices, body_ids]

    # 计算奖励
    reward = torch.all(first_air_vals, dim=1).float()
    return reward


def handstand_feet_air_time(
    env: ManagerBasedRLEnv, foot_name: dict[str, str], sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    # 提取传感器数据
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    state_commands = env.command_manager.get_command("state_command").to(torch.int64)

    # 预处理 foot_name 映射，提高查找效率
    foot_name_keys = torch.tensor(list(map(int, foot_name.keys())), device=env.device, dtype=torch.long)
    foot_name_values = list(foot_name.values())

    # 预分配 body_id 索引表
    max_command = foot_name_keys.max().item() if len(foot_name_keys) > 0 else 0
    body_id_map = torch.full((max_command + 1,), -1, device=env.device, dtype=torch.long)

    # 填充索引表
    for key, name in zip(foot_name_keys, foot_name_values):
        body_ids = contact_sensor.find_bodies(name)
        if body_ids:
            body_id_map[key] = body_ids[0] if isinstance(body_ids[0], int) else body_ids[0][0]

    # 通过索引查找 body_id
    body_ids = body_id_map[state_commands.clamp(0, max_command)].unsqueeze(1)

    # 计算接触时间（避免重复计算）
    first_contact = contact_sensor.compute_first_contact(env.step_dt)

    # 获取历史离地时间
    last_air_time = contact_sensor.data.last_air_time

    # 使用索引直接查找
    env_indices = torch.arange(env.num_envs, device=env.device).unsqueeze(1)
    first_contact_vals = first_contact[env_indices, body_ids]
    last_air_time_vals = last_air_time[env_indices, body_ids]

    # 计算奖励
    reward = torch.sum((last_air_time_vals - threshold) * first_contact_vals, dim=1)
    return reward


def handstand_orientation_l2(
    env: ManagerBasedRLEnv,
    target_gravity: dict[str, list[float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    # Extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    state_commands = env.command_manager.get_command("state_command").to(torch.int64)

    # 生成 keys 和 values
    target_gravity_keys = torch.tensor(list(map(int, target_gravity.keys())), device=env.device, dtype=torch.long)
    target_gravity_values = torch.tensor(list(target_gravity.values()), device=env.device, dtype=torch.float32)

    # 获取最大索引值
    max_command = target_gravity_keys.max().item() if len(target_gravity_keys) > 0 else 0

    # 创建查找表，默认值为 [0, 0, -1]（标准重力方向）
    default_gravity = torch.tensor([0.0, 0.0, -1.0], device=env.device, dtype=torch.float32)
    target_gravity_map = default_gravity.repeat(max_command + 1, 1)  # shape: (max_command + 1, 3)

    # 填充 target_gravity_map
    target_gravity_map[target_gravity_keys] = target_gravity_values

    # 通过索引查找目标重力方向
    target_gravity_tensor = target_gravity_map[state_commands.clamp(0, max_command)]

    # 计算误差
    return torch.sum(torch.square(asset.data.projected_gravity_b - target_gravity_tensor), dim=1)


def track_lin_vel_world_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_com_lin_vel_w[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_world_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2] - asset.data.root_com_ang_vel_w[:, 2]
    )
    return torch.exp(-ang_vel_error / std**2)
