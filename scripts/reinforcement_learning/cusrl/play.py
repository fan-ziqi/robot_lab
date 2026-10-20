# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Script to play a checkpoint if an RL agent from CusRL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

import warp as wp

wp.config.enable_backward = False

from isaaclab.app import AppLauncher


# add argparse arguments
def positive_int(value: str) -> int:
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


parser = argparse.ArgumentParser(description="Evaluate an RL agent with CusRL.")
parser.add_argument(
    "--max_steps", type=positive_int, default=None, help="Stop playback after this many environment steps."
)
parser.add_argument("--disable-export", action="store_true", help="Skip ONNX and JIT policy exports.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="cusrl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
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
parser.add_argument("--headless", action="store_true", help="Run without a viewer (alias for --viz none).")
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
if args_cli.headless and args_cli.visualizer is None:
    args_cli.visualizer = ["none"]
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
    if args_cli.visualizer is None:
        args_cli.visualizer = ["kit"]

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import cusrl
import gymnasium as gym
import torch
from cusrl.environment.isaaclab import TrainerCfg

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,  # noqa: F401
    DirectRLEnvCfg,  # noqa: F401
    ManagerBasedRLEnvCfg,  # noqa: F401
    VideoRecorderCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm

from isaaclab_tasks.utils import resolve_task_config

import robot_lab.tasks  # noqa: F401  # isort: skip

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rl_utils import camera_follow


class CameraFollowPlayerHook(cusrl.Player.Hook):
    def step(self, step: int, transition: dict, metrics: dict):
        camera_follow(self.player.environment)


class PlaybackStepCounter(cusrl.Player.Hook):
    """Count callbacks after the environment and agent complete each step."""

    def __init__(self):
        self.completed_steps = 0

    def step(self, step: int, transition: dict, metrics: dict):
        self.completed_steps += 1


def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: TrainerCfg):
    """Play with CusRL-RL agent."""
    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    cusrl.set_global_seed(args_cli.seed)

    # modify environment configurations based on CLI args
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 50
    env_cfg.sim.use_fabric = not args_cli.disable_fabric
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
    env_cfg.curriculum.command_levels_lin_vel = None
    env_cfg.curriculum.command_levels_ang_vel = None

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

    # Configure recording before the environment creates its recorder manager.
    if args_cli.video:
        env_cfg.video_recorders = [
            VideoRecorderCfg(
                source="visualizer:kit",
                output_dir=os.path.join(log_dir, "videos", "play"),
                video_length=args_cli.video_length,
                video_interval=0,
            )
        ]

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # create player from cusrl
    player = cusrl.Player(
        environment=cusrl.environment.IsaacLabEnvAdapter(env),
        agent=agent_cfg.agent_factory.override(device=args_cli.device),
        checkpoint_path=trial,
        deterministic=not args_cli.stochastic,
        num_steps=args_cli.max_steps,
    )

    if not args_cli.disable_export:
        export_model_dir = os.path.join(log_dir, "exported")
        player.agent.export(output_dir=export_model_dir, target_format="onnx", verbose=args_cli.verbose)
        player.agent.export(output_dir=export_model_dir, target_format="jit", verbose=args_cli.verbose)

    if args_cli.keyboard:
        player.register_hook(CameraFollowPlayerHook())

    step_counter = PlaybackStepCounter()
    player.register_hook(step_counter)

    # run playing loop
    player.run_playing_loop()
    print(f"[robot_lab] playback_steps={step_counter.completed_steps} requested_max_steps={args_cli.max_steps}")
    if args_cli.max_steps is not None and step_counter.completed_steps != args_cli.max_steps:
        raise RuntimeError(f"Playback completed {step_counter.completed_steps} steps; expected {args_cli.max_steps}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main(*resolve_task_config(args_cli.task, args_cli.agent, overrides=hydra_args))
    # close sim app
    simulation_app.close()
