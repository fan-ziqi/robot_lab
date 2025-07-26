# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2025 Linden
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply

from .g1_amp_env_cfg import G1AmpDanceEnvCfg
from .motions import MotionLoader


class G1AmpEnv(DirectRLEnv):
    cfg: G1AmpDanceEnvCfg

    def __init__(self, cfg: G1AmpDanceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # action offset and scale
        dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0]
        dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self.action_offset = 0.5 * (dof_upper_limits + dof_lower_limits)
        self.action_scale = dof_upper_limits - dof_lower_limits

        # load motion
        self._motion_loader = MotionLoader(motion_file=self.cfg.motion_file, device=self.device)
        # self._motion_loader.resample(self.cfg.sim.dt, kind="linear")

        # DOF and key body indexes
        key_body_names = [
            "left_shoulder_yaw_link",
            "right_shoulder_yaw_link",
            "left_elbow_link",
            "right_elbow_link",
            "right_rubber_hand",
            "left_rubber_hand",
            "right_ankle_roll_link",
            "left_ankle_roll_link",
            "torso_link",
            "right_hip_yaw_link",
            "left_hip_yaw_link",
            "right_knee_link",
            "left_knee_link",
        ]

        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in key_body_names]
        self.motion_dof_indexes = self._motion_loader.get_dof_index(self.robot.data.joint_names)
        self.motion_ref_body_index = self._motion_loader.get_body_index([self.cfg.reference_body])[0]
        self.motion_key_body_indexes = self._motion_loader.get_body_index(key_body_names)

        # reconfigure AMP observation space according to the number of observations and create the buffer
        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
        )

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        # add ground plane
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        target = self.action_offset + self.action_scale * self.actions
        self.robot.set_joint_position_target(target)

    def _get_observations(self) -> dict:
        # build task observation

        # calculate progress: current episode step / max step, shape [num_envs, 1]
        progress = (self.episode_length_buf.squeeze(-1).float() / (self.max_episode_length - 1)).unsqueeze(-1)
        # convert to relative coordinates, keep consistent with reference action observation
        root_pos_relative = self.robot.data.body_pos_w[:, self.ref_body_index] - self.scene.env_origins
        key_body_pos_relative = self.robot.data.body_pos_w[:, self.key_body_indexes] - self.scene.env_origins.unsqueeze(
            1
        )
        obs = compute_obs(
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            root_pos_relative,
            self.robot.data.body_quat_w[:, self.ref_body_index],
            # self.robot.data.body_lin_vel_w[:, self.ref_body_index],
            # self.robot.data.body_ang_vel_w[:, self.ref_body_index],
            key_body_pos_relative,
            progress,
        )

        # update AMP observation history
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        # build AMP observation
        self.amp_observation_buffer[:, 0] = obs.clone()
        self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # ================= imitation reward ==========================
        with torch.no_grad():
            # get reference action at current time
            current_times = (self.episode_length_buf * self.physics_dt).cpu().numpy()
            # sample reference action data
            (
                ref_dof_positions,
                ref_dof_velocities,
                ref_body_positions,
                ref_body_rotations,
                _,
                _,
            ) = self._motion_loader.sample(num_samples=self.num_envs, times=current_times)

            # get reference joint angles and velocities
            ref_joint_pos = ref_dof_positions[:, self.motion_dof_indexes]
            ref_joint_vel = ref_dof_velocities[:, self.motion_dof_indexes]

            # get body point position and root orientation
            ref_root_pos = ref_body_positions[:, self.motion_ref_body_index]
            ref_root_quat = ref_body_rotations[:, self.motion_ref_body_index]

        # 1. joint angle imitation reward
        joint_pos_error = torch.square(self.robot.data.joint_pos - ref_joint_pos).sum(dim=-1)
        rew_joint_pos = exp_reward_with_floor(
            joint_pos_error, self.cfg.rew_imitation_joint_pos, self.cfg.imitation_sigma_joint_pos, floor=4.0
        )
        rew_joint_pos = torch.clamp(rew_joint_pos, min=-1.0)  # avoid joint position over-penalty

        # 2. joint velocity imitation reward
        joint_vel_error = torch.square(self.robot.data.joint_vel - ref_joint_vel).sum(dim=-1)
        rew_joint_vel = exp_reward_with_floor(
            joint_vel_error, self.cfg.rew_imitation_joint_vel, self.cfg.imitation_sigma_joint_vel, floor=6.0
        )
        rew_joint_vel = torch.clamp(rew_joint_vel, min=-1.0)  # avoid over-penalty, minimum -2.0

        # 3. root position imitation reward
        # convert robot current position to relative position to environment origin, compare with reference position
        current_relative_pos = self.robot.data.body_pos_w[:, self.ref_body_index] - self.scene.env_origins
        pos_err = torch.square(current_relative_pos - ref_root_pos).sum(dim=-1)
        rew_pos = exp_reward_with_floor(pos_err, self.cfg.rew_imitation_pos, self.cfg.imitation_sigma_pos, floor=4.0)
        rew_pos = torch.clamp(rew_pos, min=-1.0)  # avoid position error over-penalty

        # 4. root orientation imitation reward
        quat_dot = torch.abs(torch.sum(self.robot.data.body_quat_w[:, self.ref_body_index] * ref_root_quat, dim=-1))
        ang_err = 2 * torch.arccos(torch.clamp(quat_dot, -1.0, 1.0))
        rew_rot = self.cfg.rew_imitation_rot * torch.exp(-torch.square(ang_err) / (self.cfg.imitation_sigma_rot**2))

        # 5. total imitation reward
        imitation_reward = rew_joint_pos + rew_joint_vel + rew_pos + rew_rot

        # ================= basic reward (call the original compute_rewards function) ==========================
        basic_reward, basic_reward_log = compute_rewards(
            self.cfg.rew_termination,
            self.cfg.rew_action_l2,
            self.cfg.rew_joint_pos_limits,
            self.cfg.rew_joint_acc_l2,
            self.cfg.rew_joint_vel_l2,
            self.reset_terminated,
            self.actions,
            self.robot.data.joint_pos,
            self.robot.data.soft_joint_pos_limits,
            self.robot.data.joint_acc,
            self.robot.data.joint_vel,
        )

        # ================= total reward ==========================
        total_reward = imitation_reward + basic_reward

        # ============== log ================================
        log_dict = {
            # imitation learning reward
            "rew_imitation": imitation_reward.mean().item(),
            "rew_joint_pos": rew_joint_pos.mean().item(),
            "rew_joint_vel": rew_joint_vel.mean().item(),
            "rew_pos": rew_pos.mean().item(),
            "rew_rot": rew_rot.mean().item(),
            "error_joint_pos": joint_pos_error.mean().item(),
            "error_joint_vel": joint_vel_error.mean().item(),
            "error_root_pos": pos_err.mean().item(),
            "error_ang": ang_err.mean().item(),
            "total_reward": total_reward.mean().item(),
        }

        # add basic reward log
        for key, value in basic_reward_log.items():
            if isinstance(value, torch.Tensor):
                log_dict[key] = value.mean().item()
            else:
                log_dict[key] = float(value)

        self.extras["log"] = log_dict

        # directly record to TensorBoard (if agent is available)
        if hasattr(self, "_skrl_agent") and getattr(self, "_skrl_agent", None) is not None:
            try:
                agent = getattr(self, "_skrl_agent")
                for k, v in log_dict.items():
                    agent.track_data(f"Reward / {k}", v)
            except Exception:
                pass

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.early_termination:
            died = self.robot.data.body_pos_w[:, self.ref_body_index, 2] < self.cfg.termination_height
        else:
            died = torch.zeros_like(time_out)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if self.cfg.reset_strategy == "default":
            root_state, joint_pos, joint_vel = self._reset_strategy_default(env_ids)
        elif self.cfg.reset_strategy.startswith("random"):
            start = "start" in self.cfg.reset_strategy
            root_state, joint_pos, joint_vel = self._reset_strategy_random(env_ids, start)
        else:
            raise ValueError(f"Unknown reset strategy: {self.cfg.reset_strategy}")

        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # reset strategies

    def _reset_strategy_default(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        return root_state, joint_pos, joint_vel

    def _reset_strategy_random(
        self, env_ids: torch.Tensor, start: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # sample random motion times (or zeros if start is True)
        num_samples = env_ids.shape[0]
        times = np.zeros(num_samples) if start else self._motion_loader.sample_times(num_samples)
        # sample random motions
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)

        # get root transforms (the humanoid torso)
        motion_torso_index = self._motion_loader.get_body_index([self.cfg.reference_body])[0]
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = body_positions[:, motion_torso_index] + self.scene.env_origins[env_ids]
        root_state[:, 2] += 0.05  # lift the humanoid slightly to avoid collisions with the ground
        root_state[:, 3:7] = body_rotations[:, motion_torso_index]
        root_state[:, 7:10] = body_linear_velocities[:, motion_torso_index]
        root_state[:, 10:13] = body_angular_velocities[:, motion_torso_index]
        # get DOFs state
        dof_pos = dof_positions[:, self.motion_dof_indexes]
        dof_vel = dof_velocities[:, self.motion_dof_indexes]

        # update AMP observation
        amp_observations = self.collect_reference_motions(num_samples, times)
        self.amp_observation_buffer[env_ids] = amp_observations.view(num_samples, self.cfg.num_amp_observations, -1)

        return root_state, dof_pos, dof_vel

    # env methods

    def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None) -> torch.Tensor:
        # sample random motion times (or use the one specified)
        if current_times is None:
            current_times = self._motion_loader.sample_times(num_samples)
        times = (
            np.expand_dims(current_times, axis=-1)
            - self._motion_loader.dt * np.arange(0, self.cfg.num_amp_observations)
        ).flatten()
        # get motions
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)
        # compute AMP observation
        progress = (
            torch.as_tensor(times, device=dof_positions.device, dtype=dof_positions.dtype).unsqueeze(-1)
            / self._motion_loader.duration
        )
        amp_observation = compute_obs(
            dof_positions[:, self.motion_dof_indexes],
            dof_velocities[:, self.motion_dof_indexes],
            body_positions[:, self.motion_ref_body_index],
            body_rotations[:, self.motion_ref_body_index],
            # body_linear_velocities[:, self.motion_ref_body_index],
            # body_angular_velocities[:, self.motion_ref_body_index],
            body_positions[:, self.motion_key_body_indexes],
            progress,
        )
        return amp_observation.view(-1, self.amp_observation_size)


@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_apply(q, ref_tangent)
    normal = quat_apply(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)


@torch.jit.script
def compute_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    root_rotations: torch.Tensor,
    # root_linear_velocities: torch.Tensor,
    # root_angular_velocities: torch.Tensor,
    key_body_positions: torch.Tensor,
    progress: torch.Tensor,
) -> torch.Tensor:
    obs = torch.cat(
        (
            dof_positions,
            dof_velocities,
            root_positions[:, 2:3],  # root body height
            quaternion_to_tangent_and_normal(root_rotations),
            # root_linear_velocities,
            # root_angular_velocities,
            (key_body_positions - root_positions.unsqueeze(-2)).view(key_body_positions.shape[0], -1),
            progress,
        ),
        dim=-1,
    )
    return obs


@torch.jit.script
def compute_rewards(
    rew_scale_termination: float,
    rew_scale_action_l2: float,
    rew_scale_joint_pos_limits: float,
    rew_scale_joint_acc_l2: float,
    rew_scale_joint_vel_l2: float,
    reset_terminated: torch.Tensor,
    actions: torch.Tensor,
    joint_pos: torch.Tensor,
    soft_joint_pos_limits: torch.Tensor,
    joint_acc: torch.Tensor,
    joint_vel: torch.Tensor,
):
    rew_termination = rew_scale_termination * reset_terminated.float()
    rew_action_l2 = rew_scale_action_l2 * torch.sum(torch.square(actions), dim=1)

    out_of_limits = -(joint_pos - soft_joint_pos_limits[:, :, 0]).clip(max=0.0)
    out_of_limits += (joint_pos - soft_joint_pos_limits[:, :, 1]).clip(min=0.0)
    rew_joint_pos_limits = rew_scale_joint_pos_limits * torch.sum(out_of_limits, dim=1)

    rew_joint_acc_l2 = rew_scale_joint_acc_l2 * torch.sum(torch.square(joint_acc), dim=1)
    rew_joint_vel_l2 = rew_scale_joint_vel_l2 * torch.sum(torch.square(joint_vel), dim=1)
    total_reward = rew_termination + rew_action_l2 + rew_joint_pos_limits + rew_joint_acc_l2 + rew_joint_vel_l2

    log = {
        "pub_termination": (rew_termination).mean(),
        "pub_action_l2": (rew_action_l2).mean(),
        "pub_joint_pos_limits": (rew_joint_pos_limits).mean(),
        "pub_joint_acc_l2": (rew_joint_acc_l2).mean(),
        "pub_joint_vel_l2": (rew_joint_vel_l2).mean(),
    }
    return total_reward, log


@torch.jit.script
def exp_reward_with_floor(error: torch.Tensor, weight: float, sigma: float, floor: float = 3.0) -> torch.Tensor:
    """
    piecewise exponential reward function: large error region use linear, small error region use exponential

    Args:
        error: error value (already squared error)
        weight: reward weight
        sigma: standard deviation parameter of exponential function
        floor: threshold, unit is sigma² multiple

    Returns:
        piecewise exponential reward value
    """
    sigma_sq = sigma * sigma
    threshold = floor * sigma_sq

    # exponential part at threshold and gradient
    exp_val_at_threshold = weight * torch.exp(-floor)
    linear_slope = weight / sigma_sq * torch.exp(-floor)  # ensure first-order continuous

    # large error region: use linear penalty (keep negative slope)
    linear_reward = exp_val_at_threshold - linear_slope * (error - threshold)

    # small error region: use exponential reward
    exp_reward = weight * torch.exp(-error / sigma_sq)

    # choose the corresponding reward function based on the error size
    return torch.where(error > threshold, linear_reward, exp_reward)
