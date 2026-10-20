# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2025 Linden
# SPDX-License-Identifier: BSD 3-Clause

"""Replay a G1 AMP motion archive using the Isaac Lab simulation launcher."""

import argparse
from pathlib import Path

import torch
import warp as wp

wp.config.enable_backward = False

from isaaclab_physx.physics import PhysxCfg

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import DomeLightCfg, GroundPlaneCfg, SimulationCfg
from isaaclab.utils import configclass, replace

from robot_lab.assets.unitree import UNITREE_G1_29DOF_CFG
from robot_lab.tasks.direct.g1_amp.motions.motion_loader import MotionLoader


@configclass
class MotionReplaySceneCfg(InteractiveSceneCfg):
    """Robot, ground and lighting for motion playback."""

    robot = replace(UNITREE_G1_29DOF_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.5)),
        spawn=GroundPlaneCfg(color=(0.0, 0.0, 0.0)),
    )
    light = AssetBaseCfg(prim_path="/World/Light", spawn=DomeLightCfg(intensity=2000.0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", default=str(Path(__file__).with_name("g1_dance1_subject2_30.npz")))
    parser.add_argument(
        "--max_steps", type=int, default=0, help="Stop after this many frames; zero replays indefinitely."
    )
    add_launcher_args(parser)
    args = parser.parse_args()
    motion = MotionLoader(args.motion, device=args.device)
    motion.resample(0.005)
    root_id = motion.get_body_index(["pelvis"])[0]
    sim_cfg = SimulationCfg(dt=motion.dt, device=args.device, physics=PhysxCfg())
    with launch_simulation(sim_cfg, args):
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene

        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view([3.0, 3.0, 3.0], [0.0, 0.0, 0.0])
        scene = InteractiveScene(MotionReplaySceneCfg(num_envs=1, env_spacing=2.0))
        sim.reset()
        scene.reset()
        robot = scene["robot"]
        joint_ids = motion.get_dof_index(robot.joint_names)
        step = 0
        while sim.is_running() and (args.max_steps == 0 or step < args.max_steps):
            frame = step % motion.num_frames
            root_pose = torch.cat((motion.body_positions[frame, root_id], motion.body_rotations[frame, root_id]))
            root_velocity = torch.cat(
                (motion.body_linear_velocities[frame, root_id], motion.body_angular_velocities[frame, root_id])
            )
            robot.write_root_link_pose_to_sim_index(root_pose=root_pose.unsqueeze(0))
            robot.write_root_link_velocity_to_sim_index(root_velocity=root_velocity.unsqueeze(0))
            robot.write_joint_state_to_sim_index(
                position=motion.dof_positions[frame, joint_ids].unsqueeze(0),
                velocity=motion.dof_velocities[frame, joint_ids].unsqueeze(0),
            )
            scene.write_data_to_sim()
            sim.render()
            scene.update(sim_cfg.dt)
            step += 1


if __name__ == "__main__":
    main()
