from __future__ import annotations

import torch

from robot_lab.tasks.locomotion.velocity.manager_based_rl_amp_env import ManagerBasedRLAmpEnv

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import quat_rotate


def reset_amp(
    env: ManagerBasedRLAmpEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    frames = env.unwrapped.amp_loader.get_full_frame_batch(len(env_ids))

    # Step1: reset_root_state
    # base position
    root_pos = env.unwrapped.amp_loader.get_root_pos_batch(frames)
    root_pos[:, :2] = root_pos[:, :2] + env.scene.env_origins[env_ids, :2]
    root_orn = env.unwrapped.amp_loader.get_root_rot_batch(frames)  # xyzw
    # Func quat_rotate() and Isaacsim/IsaacLab all need wxyz
    root_orn = torch.cat((root_orn[:, -1].unsqueeze(1), root_orn[:, :-1]), dim=1)
    # base velocities
    lin_vel = quat_rotate(root_orn, env.unwrapped.amp_loader.get_linear_vel_batch(frames))
    ang_vel = quat_rotate(root_orn, env.unwrapped.amp_loader.get_angular_vel_batch(frames))
    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([root_pos, root_orn], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.cat([lin_vel, ang_vel], dim=-1), env_ids=env_ids)

    # Step2: reset_joints
    joint_pos = env.unwrapped.amp_loader.get_joint_pose_batch(frames)
    joint_vel = env.unwrapped.amp_loader.get_joint_vel_batch(frames)
    # Isaac Sim uses breadth-first joint ordering, while Isaac Gym uses depth-first joint ordering
    joint_pos = env.unwrapped.amp_loader.reorder_from_isaacgym_to_isaacsim_tool(joint_pos)
    joint_vel = env.unwrapped.amp_loader.reorder_from_isaacgym_to_isaacsim_tool(joint_vel)
    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)
    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
