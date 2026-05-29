# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Script to collect chirp excitation data for PACE system identification."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pace agent for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Pace-Anymal-D-v0", help="Name of the task.")
parser.add_argument("--min_frequency", type=float, default=0.1, help="Minimum frequency for the chirp signal in Hz.")
parser.add_argument("--max_frequency", type=float, default=10.0, help="Maximum frequency for the chirp signal in Hz.")
parser.add_argument("--duration", type=float, default=20.0, help="Duration of the chirp signal in seconds.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import robot_lab.tasks  # noqa: F401
import torch
from robot_lab.utils.paths import project_root
from torch import pi

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment

    articulation = env.unwrapped.scene["robot"]

    joint_order = env_cfg.sim2real.joint_order
    joint_ids = torch.tensor(
        [articulation.joint_names.index(name) for name in joint_order], device=env.unwrapped.device
    )

    armature = torch.tensor([0.1] * len(joint_ids), device=env.unwrapped.device).unsqueeze(0)
    damping = torch.tensor([4.5] * len(joint_ids), device=env.unwrapped.device).unsqueeze(0)
    friction = torch.tensor([0.05] * len(joint_ids), device=env.unwrapped.device).unsqueeze(0)  # coulomb friction
    bias = torch.tensor([0.05] * 12, device=env.unwrapped.device).unsqueeze(0)
    time_lag = torch.tensor([[5]], dtype=torch.int, device=env.unwrapped.device)
    env.reset()

    articulation.write_joint_armature_to_sim(armature, joint_ids=joint_ids, env_ids=torch.arange(len(armature)))
    articulation.data.default_joint_armature[:, joint_ids] = armature
    articulation.write_joint_viscous_friction_coefficient_to_sim(
        damping, joint_ids=joint_ids, env_ids=torch.arange(len(damping))
    )
    articulation.data.default_joint_viscous_friction_coeff[:, joint_ids] = damping
    # note: modeling coulomb friction if joint_friction = joint_dynamic_friction
    articulation.write_joint_friction_coefficient_to_sim(friction, joint_ids=joint_ids, env_ids=torch.tensor([0]))
    articulation.data.default_joint_friction_coeff[:, joint_ids] = friction
    articulation.write_joint_dynamic_friction_coefficient_to_sim(
        friction, joint_ids=joint_ids, env_ids=torch.tensor([0])
    )
    articulation.data.default_joint_dynamic_friction_coeff[:, joint_ids] = friction
    drive_types = articulation.actuators.keys()
    for drive_type in drive_types:
        drive_indices = articulation.actuators[drive_type].joint_indices
        if isinstance(drive_indices, slice):
            all_idx = torch.arange(joint_ids.shape[0], device=joint_ids.device)
            drive_indices = all_idx[drive_indices]
        comparison_matrix = joint_ids.unsqueeze(1) == drive_indices.unsqueeze(0)
        drive_joint_idx = torch.argmax(comparison_matrix.int(), dim=0)
        articulation.actuators[drive_type].update_time_lags(time_lag)
        articulation.actuators[drive_type].update_encoder_bias(bias[:, drive_joint_idx])
        articulation.actuators[drive_type].reset(torch.arange(env.unwrapped.num_envs))

    data_dir = project_root() / "data" / env_cfg.sim2real.robot_name

    # Create a chirp signal for each action dimension

    duration = args_cli.duration  # seconds
    sample_rate = 1 / env.unwrapped.sim.get_physics_dt()  # Hz
    num_steps = int(duration * sample_rate)
    t = torch.linspace(0, duration, steps=num_steps, device=env.unwrapped.device)
    f0 = args_cli.min_frequency  # Hz
    f1 = args_cli.max_frequency  # Hz

    # Linear chirp: phase = 2*pi*(f0*t + (f1-f0)/(2*duration)*t^2)
    phase = 2 * pi * (f0 * t + ((f1 - f0) / (2 * duration)) * t**2)
    chirp_signal = torch.sin(phase)

    trajectory = torch.zeros((num_steps, len(joint_ids)), device=env.unwrapped.device)
    trajectory[:, :] = chirp_signal.unsqueeze(-1)
    trajectory_directions = torch.tensor(
        [1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0], device=env.unwrapped.device
    )

    trajectory_bias = torch.tensor([0.0, 0.4, 0.8] * 4, device=env.unwrapped.device)
    trajectory_scale = torch.tensor([0.25, 0.5, -2.0] * 4, device=env.unwrapped.device)
    trajectory[:, joint_ids] = (
        (trajectory[:, joint_ids] + trajectory_bias.unsqueeze(0))
        * trajectory_directions.unsqueeze(0)
        * trajectory_scale.unsqueeze(0)
    )

    articulation.write_joint_position_to_sim(trajectory[0, :].unsqueeze(0) + bias[0, joint_ids])
    articulation.write_joint_velocity_to_sim(torch.zeros((1, len(joint_ids)), device=env.unwrapped.device))

    counter = 0
    # simulate environment
    dof_pos_buffer = torch.zeros(num_steps, len(joint_ids), device=env.unwrapped.device)
    dof_target_pos_buffer = torch.zeros(num_steps, len(joint_ids), device=env.unwrapped.device)
    time_data = t
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute actions
            dof_pos_buffer[counter, :] = (
                env.unwrapped.scene.articulations["robot"].data.joint_pos[0, joint_ids] - bias[0]
            )
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            actions = trajectory[counter % num_steps, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
            # apply actions
            obs, _, _, _, _ = env.step(actions)
            dof_target_pos_buffer[counter, :] = env.unwrapped.scene.articulations["robot"]._data.joint_pos_target[
                0, joint_ids
            ]
            counter += 1
            if counter % 400 == 0:
                print(f"[INFO]: Step {counter / sample_rate} seconds")
            if counter >= num_steps:
                break

    # close the simulator
    env.close()

    from time import sleep

    sleep(1)  # wait a bit for everything to settle

    (data_dir).mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "time": time_data.cpu(),
            "dof_pos": dof_pos_buffer.cpu(),
            "des_dof_pos": dof_target_pos_buffer.cpu(),
        },
        data_dir / "chirp_data.pt",
    )

    import matplotlib.pyplot as plt

    for i in range(len(joint_ids)):
        plt.figure()
        plt.plot(t.cpu().numpy(), dof_pos_buffer[:, i].cpu().numpy(), label=f"{joint_order[i]} pos")
        plt.plot(
            t.cpu().numpy(),
            dof_target_pos_buffer[:, i].cpu().numpy(),
            label=f"{joint_order[i]} target",
            linestyle="dashed",
        )
        plt.title(f"Joint {joint_order[i]} Trajectory")
        plt.xlabel("Time [s]")
        plt.ylabel("Joint position [rad]")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
