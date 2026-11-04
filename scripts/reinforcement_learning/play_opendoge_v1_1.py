"""Replay LainLab OpenDog V1.1 exports in Isaac Lab on flat ground."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import warp as wp

wp.config.enable_backward = False

from isaaclab.app import add_launcher_args, launch_simulation

import robot_lab  # noqa: F401  # isort: skip
from robot_lab.policies.opendoge_v1_1 import (
    JOINT_NAMES,
    OBS_DIMS,
    SKILLS,
    OnnxPolicy,
    PolicyState,
    imu_linear_velocity,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=OBS_DIMS, default="flat")
    parser.add_argument("--skill", choices=SKILLS, default="walk", help="Skill for the multi-skill export.")
    parser.add_argument("--policy", type=Path, help="Override the selected profile's ONNX export.")
    parser.add_argument("--initial-pose", choices=["auto", "stand", "side", "back"], default="auto")
    parser.add_argument("--command", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("VX", "VY", "YAW"))
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--report", type=Path, help="Write replay metrics as JSON.")
    parser.add_argument("--physics", default="newton_mjwarp", choices=["newton_mjwarp", "isaacsim_physx", "ovphysx"])
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.duration <= 0 or not np.isfinite(args.duration) or not np.isfinite(args.command).all():
        parser.error("duration must be positive and command values must be finite")

    # Fail on a missing/incompatible model before starting the physics runtime.
    if args.profile != "skills" and args.skill != "walk":
        parser.error("--skill requires --profile skills")
    if args.profile == "getup" and any(args.command):
        parser.error("getup uses a zero command")
    policy = OnnxPolicy(args.profile, args.policy)
    state = PolicyState(args.profile, skill=args.skill)
    recovering = args.profile == "getup" or (args.profile == "skills" and args.skill == "getup")
    pose = ("side" if recovering else "stand") if args.initial_pose == "auto" else args.initial_pose

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.utils import instantiate

    from isaaclab_tasks.core.velocity.velocity_env_cfg import RoughPhysicsCfg

    from robot_lab.assets.opendoge_v1_1 import OPENDOGE_V1_1_CFG

    cfg = sim_utils.SimulationCfg(dt=0.005, device=args.device, physics=getattr(RoughPhysicsCfg(), args.physics))
    cfg.use_newton_actuators = False  # Evaluate explicit PD at each physics step.
    with launch_simulation(cfg, args) as physics_cfg:
        cfg.physics = physics_cfg
        sim = sim_utils.SimulationContext(cfg)
        scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
        scene_cfg.ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.CuboidCfg(
                size=(200.0, 200.0, 0.1),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.8),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
        )
        scene_cfg.robot = OPENDOGE_V1_1_CFG.copy()
        if pose != "stand":
            scene_cfg.robot.init_state.pos = (0.0, 0.0, 0.08)
            scene_cfg.robot.init_state.rot = (2**-0.5, 2**-0.5, 0.0, 0.0) if pose == "side" else (0.0, 1.0, 0.0, 0.0)
        jumping = args.profile == "skills" and args.skill == "jump"
        if jumping:
            scene_cfg.feet = ContactSensorCfg(prim_path="/World/Robot/.*_foot", update_period=0.005)
        scene_cfg.light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2500.0))
        scene = instantiate(scene_cfg)
        sim.set_camera_view([1.1, 1.1, 0.7], [0.0, 0.0, 0.15])
        sim.reset()
        robot = scene["robot"]
        joint_ids, names = robot.find_joints(JOINT_NAMES, preserve_order=True)
        if names != JOINT_NAMES:
            raise ValueError(f"Unexpected robot joints: {names}")
        robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch.clone())
        robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch.clone())
        robot.reset()
        scene.update(cfg.dt)
        targets = robot.data.default_joint_pos.torch.clone()
        steps = 0
        heights, velocities, uprights, alignments = [], [], [], []
        airborne_steps = 0
        max_actions = 0.0
        start_xy = robot.data.root_pos_w.torch[0, :2].cpu().numpy().copy()
        with torch.inference_mode():
            while sim.is_running() and steps < round(args.duration / cfg.dt):
                if steps % 2 == 0:
                    angular_velocity = robot.data.root_link_ang_vel_b.torch[0].cpu().numpy()
                    linear_velocity = imu_linear_velocity(
                        robot.data.root_link_lin_vel_b.torch[0].cpu().numpy(), angular_velocity
                    )
                    # The only terrain in this replay is the z=0 plane: every downward
                    # terrain-only ray has exactly this height, independent of yaw/grid order.
                    scan = (
                        np.full(165, float(robot.data.root_link_pos_w.torch[0, 2])) if args.profile == "rough" else None
                    )
                    observation = state.observe(
                        args.command,
                        linear_velocity,
                        angular_velocity,
                        robot.data.projected_gravity_b.torch[0].cpu().numpy(),
                        robot.data.joint_pos.torch[0, joint_ids].cpu().numpy(),
                        robot.data.joint_vel.torch[0, joint_ids].cpu().numpy(),
                        height_scan=scan,
                    )
                    output = policy(observation)
                    targets[0, joint_ids] = torch.as_tensor(state.accept_action(output), device=sim.device)
                    max_actions = max(max_actions, float(np.max(np.abs(output))))
                robot.actuators.target_command.set_position_index(value=targets)
                scene.write_data_to_sim()
                sim.step()
                scene.update(cfg.dt)
                steps += 1
                height = float(robot.data.root_link_pos_w.torch[0, 2])
                gravity = robot.data.projected_gravity_b.torch[0].cpu().numpy()
                heights.append(height)
                uprights.append(float(-gravity[2]))
                alignments.append(float(gravity[0]))
                velocities.append(robot.data.root_link_lin_vel_b.torch[0].cpu().numpy().copy())
                if not np.isfinite([height, *gravity, *velocities[-1]]).all():
                    raise RuntimeError("Non-finite robot state during replay")
                if steps % 2 == 0 and jumping:
                    forces = scene["feet"].data.net_normal_forces_w.torch[0]
                    if len(forces) != 4:
                        raise RuntimeError("Jump feedback needs exactly four foot bodies")
                    contact = (torch.linalg.vector_norm(forces, dim=-1) > 1e-5).cpu().numpy()
                    airborne_steps += int(not contact.any())
                    state.advance(0.01, height, contact)
        report = {
            "policy": str(policy.path),
            "profile": args.profile,
            "skill": args.skill if args.profile == "skills" else None,
            "initial_pose": pose,
            "terrain": "plane",
            "command": args.command,
            "physics": type(physics_cfg).__name__,
            "steps": steps,
            "simulated_seconds": steps * cfg.dt,
            "min_base_height": min(heights) if heights else None,
            "final_base_height": heights[-1] if heights else None,
            "max_base_height": max(heights) if heights else None,
            "final_upright": uprights[-1] if uprights else None,
            "final_handstand_alignment": alignments[-1] if alignments else None,
            "handstand_fraction": float(np.mean((np.array(alignments) > 0.9) & (np.array(heights) > 0.19)))
            if heights
            else None,
            "jump_airborne_control_steps": airborne_steps if jumping else None,
            "jump_left_ground": bool(state.left_ground) if jumping else None,
            "upright_fraction": float(np.mean(np.array(uprights) > 0.9)) if uprights else None,
            "standing_fraction": float(np.mean((np.array(uprights) > 0.9) & (np.array(heights) > 0.13)))
            if heights
            else None,
            "first_standing_seconds": next(
                (
                    i * cfg.dt
                    for i, (h, u) in enumerate(zip(heights, uprights, strict=True), start=1)
                    if h > 0.13 and u > 0.9
                ),
                None,
            ),
            "max_abs_action": max_actions,
            "samples_below_0_08m": sum(h < 0.08 for h in heights),
            "mean_body_velocity": np.mean(velocities, axis=0).tolist() if velocities else None,
            "displacement_xy": (robot.data.root_pos_w.torch[0, :2].cpu().numpy() - start_xy).tolist(),
        }
        print(json.dumps(report, indent=2))
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
