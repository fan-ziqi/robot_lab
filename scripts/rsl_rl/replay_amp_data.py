"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import os
import torch
import numpy as np
from omni.isaac.lab.utils.math import quat_rotate

# Import extensions to set up environment tasks
import robot_lab.tasks  # noqa: F401

# from rsl_rl.runners import OnPolicyRunner
from robot_lab.utils.wrappers.rsl_rl.runners import OnPolicyRunner

from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

from robot_lab.utils.wrappers.rsl_rl.datasets.motion_loader import AMPLoader


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # make a smaller scene for play
    env_cfg.scene.num_envs = 1
    # spawn the robot randomly in the grid (instead of their terrain levels)
    env_cfg.scene.terrain.max_init_terrain_level = None
    # reduce the number of terrains to save memory
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 1
        env_cfg.scene.terrain.terrain_generator.num_cols = 1
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # disable randomization for play
    env_cfg.observations.policy.enable_corruption = False
    # remove random pushing
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.push_robot = None

    env_cfg.amp_num_preload_transitions = 1
    # env_cfg.algorithm.amp_replay_buffer_size = 2

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # reset environment
    obs, _ = env.get_observations()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            env_ids = torch.tensor([0], device=env.unwrapped.device)
            t = 0.0
            traj_idx = 0
            while traj_idx < len(env.unwrapped.amp_loader.trajectory_lens):
                actions = torch.zeros((env_cfg.scene.num_envs, env.unwrapped.num_actions), device=env.unwrapped.device)

                if (t + env.unwrapped.amp_loader.time_between_frames + env_cfg.sim.dt) >= env.unwrapped.amp_loader.trajectory_lens[traj_idx]:
                    print(f"finish traj {traj_idx}")
                    traj_idx += 1
                    t = 0
                else:
                    t += env_cfg.sim.dt

                frames = env.unwrapped.amp_loader.get_full_frame_at_time_batch(np.array([traj_idx]), np.array([t]))
                positions = AMPLoader.get_root_pos_batch(frames)
                orientations = AMPLoader.get_root_rot_batch(frames)
                lin_vel = quat_rotate(orientations, AMPLoader.get_linear_vel_batch(frames))
                ang_vel = quat_rotate(orientations, AMPLoader.get_angular_vel_batch(frames))
                velocities = torch.cat([lin_vel, ang_vel], dim=-1)
                env.unwrapped.robot.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
                env.unwrapped.robot.write_root_velocity_to_sim(velocities, env_ids=env_ids)
                joint_pos = AMPLoader.get_joint_pose_batch(frames)
                joint_vel = AMPLoader.get_joint_vel_batch(frames)
                joint_pos_limits = env.unwrapped.robot.data.soft_joint_pos_limits[env_ids]
                joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
                joint_vel_limits = env.unwrapped.robot.data.soft_joint_vel_limits[env_ids]
                joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)
                env.unwrapped.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

                # print("---")
                # foot_pos_amp = AMPLoader.get_tar_toe_pos_local_batch(frames)
                # print(env.unwrapped.get_amp_observations()[env_ids.item(), 12:24])
                # print(foot_pos_amp[0])

                # env stepping
                obs, _, _, _ = env.step(actions)

                # camera follow
                env.unwrapped.viewport_camera_controller.set_view_env_index(env_index=0)
                lookat = [env.unwrapped.robot.data.root_pos_w[env_ids.item(), i].cpu().item() for i in range(3)]
                eye_offset = [2, 2, 2]
                pairs = zip(lookat, eye_offset)
                eye = [x + y for x, y in pairs]
                env.unwrapped.viewport_camera_controller.update_view_location(eye, lookat)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
