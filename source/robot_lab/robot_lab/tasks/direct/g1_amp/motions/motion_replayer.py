# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2025 Linden
# SPDX-License-Identifier: BSD 3-Clause

"""
Motion Replayer for G1 Humanoid Robot

This script replays motion data for the G1 humanoid robot in Isaac Sim.

Usage:
    python motion_replayer.py [options]

Options:
    --motion MOTION_FILE    Motion data file to replay (default: G1_dance.npz)
    --device DEVICE       Device to run simulation on (default: cuda:0)

Example:
    # Replay a motion file
    python motion_replayer.py --motion G1_dance.npz
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse

from isaaclab.app import AppLauncher

# Command line arguments
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--motion", type=str, default="g1_dance.npz")
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from motion_loader import MotionLoader

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

from robot_lab.assets.unitree import UNITREE_G1_29DOF_CFG

# Load motion data and get dt
motion = MotionLoader(args_cli.motion, device=args_cli.device)
motion.resample(0.005, kind="linear")
num_frames = motion.num_frames
print(f"motion.dt: {motion.dt}")
print(f"motion.num_frames: {motion.num_frames}")

# Find the index for the root body, typically 'pelvis'
try:
    print(f"Searching for 'pelvis' in the following list of body names: {motion.body_names}")
    root_idx = motion.body_names.index("pelvis")
    print(f"Found root body 'pelvis' at index: {root_idx}")
except (ValueError, AttributeError):
    print("\nError: Could not find 'pelvis' in the motion file's body_names.")
    print("The motion replayer requires 'pelvis' to be defined as the root body.")
    simulation_app.close()
    sys.exit(1)

# Configure simulation with dt matching motion.dt


sim_cfg = sim_utils.SimulationCfg(
    dt=motion.dt,
    device=args_cli.device,
    gravity=(0.0, 0.0, -9.81),  # Explicitly set gravity
    render_interval=1,  # Render every physics step
    enable_scene_query_support=True,
    use_fabric=True,
    physx=sim_utils.PhysxCfg(
        solver_type=1,  # TGS solver
        min_position_iteration_count=8,  # Increase solver iterations
        max_position_iteration_count=8,
        min_velocity_iteration_count=4,  # Add velocity iterations
        max_velocity_iteration_count=4,
        enable_ccd=True,  # Enable continuous collision detection
        enable_stabilization=True,  # Enable additional stabilization
        bounce_threshold_velocity=0.2,  # Lower threshold for more stable contacts
        friction_offset_threshold=0.04,  # Increase friction threshold
        friction_correlation_distance=0.025,  # Increase correlation distance
    ),
)
sim = sim_utils.SimulationContext(sim_cfg)
sim.set_camera_view([3.0, 3.0, 3.0], [0.0, 0.0, 0.0])

# Configure scene
scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
scene_cfg.robot = UNITREE_G1_29DOF_CFG.replace(prim_path="/World/Robot")
scene = InteractiveScene(scene_cfg)

# Add DomeLight for illumination
light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
light_cfg.func("/World/Light", light_cfg)

# Add black ground plane at -0.5m
ground_cfg = sim_utils.GroundPlaneCfg(color=(0.0, 0.0, -0.5))
# ground_cfg.func("/World/ground", ground_cfg)

ground_cfg.func(
    "/World/ground",
    ground_cfg,
    translation=(0.0, 0.0, -0.5),  # 这里是 (x, y, z)
)
# Reset simulation
sim.reset()
scene.reset()

robot = scene["robot"]
# Align joint order from motion file to robot's joint order
motion_dof_indices = motion.get_dof_index(robot.joint_names)

print("\nStarting G1 motion replay loop...")
print("Tip: Close window or press Ctrl+C to exit")

try:
    # Continue motion replay loop
    while simulation_app.is_running():
        for i in range(num_frames):
            if not simulation_app.is_running():
                break

            # Get current frame's joint and root states (aligned order!)
            joint_pos = motion.dof_positions[i, motion_dof_indices].unsqueeze(0)
            joint_vel = motion.dof_velocities[i, motion_dof_indices].unsqueeze(0)
            root_pos = motion.body_positions[i, root_idx].unsqueeze(0)
            root_rot = motion.body_rotations[i, root_idx].unsqueeze(0)
            root_vel = motion.body_linear_velocities[i, root_idx].unsqueeze(0)
            root_ang_vel = motion.body_angular_velocities[i, root_idx].unsqueeze(0)
            root_state = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)

            # Write to simulation
            robot.write_root_link_pose_to_sim(root_state[:, :7], torch.tensor([0], device=args_cli.device))
            robot.write_root_com_velocity_to_sim(root_state[:, 7:], torch.tensor([0], device=args_cli.device))
            robot.write_joint_state_to_sim(joint_pos, joint_vel, None, torch.tensor([0], device=args_cli.device))

            # Step simulation (strict dt synchronization)
            scene.update(dt=sim.get_physics_dt())
            scene.write_data_to_sim()
            sim.step(render=True)

except KeyboardInterrupt:
    print("\nProgram interrupted by user")

finally:
    simulation_app.close()
