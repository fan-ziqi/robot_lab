# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Script to play a checkpoint if an RL agent from CusRL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate an RL agent with CusRL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to load for playing.")
parser.add_argument(
    "--stochastic",
    action="store_true",
    default=False,
    help="Whether to run the agent in stochastic mode.",
)
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import cusrl

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import DirectMARLEnvCfg  # noqa: F401
from isaaclab.envs import DirectRLEnvCfg  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import robot_lab.tasks  # noqa: F401


# @hydra_task_config(args_cli.task, "cusrl_cfg_entry_point")
# def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg):
def main():
    """Play with CusRL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "cusrl_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 50

    # # set the environment seed
    # # note: certain randomizations occur in the environment initialization so we set the seed here
    cusrl.set_global_seed(args_cli.seed)
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # spawn the robot randomly in the grid (instead of their terrain levels)
    env_cfg.scene.terrain.max_init_terrain_level = None
    # reduce the number of terrains to save memory
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # disable randomization for play
    env_cfg.observations.policy.enable_corruption = False
    # remove random pushing
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.curriculum.command_levels = None

    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False
        config = Se2KeyboardCfg(
            v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1],
            v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
            omega_z_sensitivity=env_cfg.commands.base_velocity.ranges.ang_vel_z[1],
        )
        controller = Se2Keyboard(config)
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: torch.tensor(controller.advance(), dtype=torch.float32).unsqueeze(0).to(env.device),
        )

    if args_cli.checkpoint is None:
        args_cli.checkpoint = os.path.join("logs", "cusrl", agent_cfg.experiment_name)
    trial = cusrl.Trial(args_cli.checkpoint)
    if trial is not None:
        log_dir = trial.home
    else:
        # specify directory for logging videos
        log_dir = os.path.join("logs", "cusrl", agent_cfg.experiment_name)
        log_dir = os.path.abspath(log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # create player from cusrl
    player = cusrl.Player(
        environment=cusrl.environment.IsaacLabEnvAdapter(env),
        agent=agent_cfg.agent_factory.override(device=args_cli.device),
        checkpoint_path=trial,
        deterministic=not args_cli.stochastic,
    )

    # run playing loop
    player.run_playing_loop()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
