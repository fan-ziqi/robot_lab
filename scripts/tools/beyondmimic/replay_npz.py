# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    python replay_npz.py -f path_to_motion.npz
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.app import add_launcher_args, launch_simulation

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument("--file", "-f", type=str, required=True)

# append simulation launcher cli args
parser.add_argument("--headless", action="store_true", help="Run without a visualizer unless --viz is explicit.")
add_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
if args_cli.headless and not args_cli.visualizer:
    args_cli.visualizer = ["none"]

"""Rest everything follows."""

##
# Pre-defined configs
##
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from robot_lab.assets.unitree import UNITREE_G1_29DOF_CFG

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.scene import InteractiveScene


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    from robot_lab.tasks.manager_based.beyondmimic.mdp import MotionLoader

    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    motion = MotionLoader(
        args_cli.file,
        torch.tensor([0], dtype=torch.long, device=sim.device),
        sim.device,
        body_names=[robot.body_names[0]],
        joint_names=robot.joint_names,
        num_robot_bodies=robot.num_bodies,
    )
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

    # Simulation loop
    while sim.is_running():
        time_steps += 1
        reset_ids = time_steps >= motion.time_step_total
        time_steps[reset_ids] = 0

        root_pose = robot.data.default_root_pose.torch.clone()
        root_pose[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins
        root_pose[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
        root_velocity = torch.cat(
            (motion.body_lin_vel_w[time_steps][:, 0], motion.body_ang_vel_w[time_steps][:, 0]), dim=-1
        )

        robot.write_root_link_pose_to_sim_index(root_pose=root_pose)
        robot.write_root_com_velocity_to_sim_index(root_velocity=root_velocity)
        robot.write_joint_state_to_sim_index(
            position=motion.joint_pos[time_steps], velocity=motion.joint_vel[time_steps]
        )
        scene.write_data_to_sim()
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim_dt)

        pos_lookat = root_pose[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, physics=PhysxCfg())
    sim_cfg.dt = 0.02
    with launch_simulation(sim_cfg, args_cli):
        from isaaclab.scene import InteractiveScene
        from isaaclab.sim import SimulationContext

        sim = SimulationContext(sim_cfg)
        scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
