# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Script to train RL agent with CusRL."""

import argparse
import os
import sys

import warp as wp

wp.config.enable_backward = False

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_rl.entrypoints.common import apply_video_recording, pre_launch_video_config

from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

import robot_lab.tasks  # noqa: F401  # isort: skip

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# add argparse arguments
def positive_int(value: str) -> int:
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


parser = argparse.ArgumentParser(description="Train an RL agent with CusRL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="cusrl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--run_name", type=str, default=None, help="Name of the run for logging.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to load for resuming training.")
parser.add_argument("--logger", type=str, default="tensorboard", help="Logger to use for training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--save_interval", type=positive_int, default=None, help="Checkpoint interval in training iterations."
)
parser.add_argument("--autocast", nargs="?", const=True, help="Datatype for automatic mixed precision.")
parser.add_argument("--compile", action="store_true", help="Whether to use `torch.compile` for optimization.")

# append simulation launcher cli args
parser.add_argument("--headless", action="store_true", help="Run without a viewer (alias for --viz none).")
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
if args_cli.headless and args_cli.visualizer is None:
    args_cli.visualizer = ["none"]

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
    if args_cli.visualizer is None:
        args_cli.visualizer = ["kit"]

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# set distributed to True if LOCAL_RANK is set
if os.environ.get("LOCAL_RANK") is not None:
    args_cli.distributed = True
    args_cli.device = f"cuda:{os.environ['LOCAL_RANK']}"

"""Rest everything follows."""

from dataclasses import replace
from datetime import datetime

import cusrl
import gymnasium as gym
import torch
from cusrl.environment.isaaclab import TrainerCfg

from isaaclab.envs import (
    DirectMARLEnvCfg,  # noqa: F401
    DirectRLEnvCfg,  # noqa: F401
    ManagerBasedRLEnvCfg,  # noqa: F401
    multi_agent_to_single_agent,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: TrainerCfg):
    """Train with CusRL agent."""
    from isaaclab.envs import DirectMARLEnv

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    if args_cli.save_interval is not None:
        agent_cfg.save_interval = args_cli.save_interval

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    cusrl.set_global_seed(args_cli.seed)
    env_cfg.scene.num_envs = int(env_cfg.scene.num_envs / cusrl.utils.distributed.world_size())

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "cusrl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args_cli.run_name is not None:
        log_dir = f"{log_dir}_{args_cli.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # Configure recording before the environment creates its recorder manager.
    if args_cli.video and cusrl.utils.is_main_process():
        apply_video_recording(env_cfg, log_dir, args_cli, subdir="train")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # create trainer from cusrl
    trainer = cusrl.Trainer(
        environment=cusrl.environment.IsaacLabEnvAdapter(env),
        agent_factory=replace(
            agent_cfg.agent_factory, device=env.unwrapped.device, autocast=args_cli.autocast, compile=args_cli.compile
        ),
        logger_factory=cusrl.make_logger_factory(args_cli.logger, log_dir, name=None),
        num_iterations=agent_cfg.max_iterations,
        checkpoint_interval=agent_cfg.save_interval,
        checkpoint_path=args_cli.checkpoint,
    )

    # run training
    trainer.run_training_loop()

    # close the simulator
    env.close()


if __name__ == "__main__":
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, overrides=hydra_args)
    pre_launch_video_config(env_cfg, args_cli)
    with launch_simulation(env_cfg, args_cli):
        main(env_cfg, agent_cfg)
