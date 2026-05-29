# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import torch

# Resolve project root from this script's location instead of importing
# robot_lab.utils.paths, which would pull in the full IsaacLab task registry
# and require launching the Isaac Sim app for an offline plotting tool.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# add argparse arguments
parser = argparse.ArgumentParser(description="Pace agent for Isaac Lab environments.")
parser.add_argument("--folder_name", type=str, default=None, help="Name of the folder to use.")
parser.add_argument("--mean_name", type=str, default=None, help="Name of the parameters file to use.")
parser.add_argument("--robot_name", type=str, default="anymal_d_sim", help="Name of the robot.")
parser.add_argument("--plot_trajectory", action="store_true", help="Whether to plot the trajectory.")
parser.add_argument("--plot_score", action="store_true", help="Whether to plot the score over iterations.")

args = parser.parse_args()
folder_name = args.folder_name
mean_name = args.mean_name
robot_name = args.robot_name
plot_trajectory = args.plot_trajectory
plot_score = args.plot_score

log_dir = _PROJECT_ROOT / "logs" / "pace" / robot_name

if not log_dir.exists():
    raise FileNotFoundError(f"No logs for robot {robot_name} under {log_dir}")

_pattern = re.compile(r"^mean_(\d+)\.pt$")


def find_latest_params(root: Path):
    best = None  # tuple (int, Path)
    for p in root.rglob("mean_*.pt"):
        m = _pattern.match(p.name)
        if not m:
            continue
        num = int(m.group(1))
        if best is None or num > best[0]:
            best = (num, p)
    return None if best is None else best[1], best[0]


# if no folder_name given, pick the most recent run folder for the robot
if not folder_name:
    robot_dir = log_dir
    candidates = [p for p in robot_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run folders found under {robot_dir}")
    latest_folder = max(candidates, key=lambda p: p.stat().st_mtime)
    folder_name = latest_folder.name

# now point log_dir at the chosen robot/run folder
log_dir = log_dir / folder_name

if mean_name is None:
    params_path, params_num = find_latest_params(log_dir)
else:
    params_path = log_dir / mean_name
    params_num = int(_pattern.match(mean_name).group(1))
    if not params_path.exists():
        raise FileNotFoundError(f"Given params file {params_path} does not exist")
if params_path is None:
    raise FileNotFoundError(f"No mean_*.pt files found under {log_dir}")
print(f"Latest params file: {params_path}")

mean = torch.load(params_path)
config = torch.load(log_dir / "config.pt")

joint_order = config["joint_order"]
trajectories = torch.load(log_dir / "best_trajectory.pt")  # time x joints
real_trajectories = config["dof_pos"]  # time x joints
target_trajectories = config["des_dof_pos"]  # time x joints
time = config["time"]  # time

print(f"Best parameter set: {mean}")
print(f"Armature params: {mean[: len(joint_order)]}")
print(f"Viscous friction params: {mean[len(joint_order) : 2 * len(joint_order)]}")
print(f"Static/dynamic friction params: {mean[2 * len(joint_order) : 3 * len(joint_order)]}")
print(f"Encoder bias params: {mean[3 * len(joint_order) : 4 * len(joint_order)]}")
print(f"Delay param: {mean[-1].item()}")
encoder_bias = mean[3 * len(joint_order) : 4 * len(joint_order)]  # extract encoder bias

if plot_score:
    try:
        progress = torch.load(log_dir / "progress.pt")
        print("Loaded optimization progress.")
    except FileNotFoundError:
        progress = None
        print("No optimization progress file found. Skipping score plot.")
        plot_score = False

if plot_score:
    plt.figure()
    plt.title("CMA-ES Score over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    data = torch.min(progress["scores_buffer"][: params_num + 1], dim=1).values.cpu().numpy()
    plt.semilogy(data)
    plt.xlim(0, params_num)
    # plt.ylim(0, None)
    plt.grid()
    plt.show()

if plot_trajectory:
    for i in range(len(joint_order)):
        plt.figure(figsize=(8, 4.5))
        plt.plot(
            time, trajectories[:, i].cpu().numpy() - encoder_bias[i].item(), c="tab:orange", label="Sim", linewidth=2
        )  # in encoder frame
        plt.plot(time, real_trajectories[:, i].cpu().numpy(), label="Real", c="tab:green", linestyle="--", linewidth=2)
        plt.plot(time, target_trajectories[:, i].cpu().numpy(), c="grey", label="Target", linestyle="--", alpha=0.5)
        plt.title(f"Joint {joint_order[i]}")  # Use joint names from config
        plt.xlabel("Time [s]")
        plt.ylabel("Joint position [rad]")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

print("Plotting complete.")
