"""Replay the original OpenDoge HIM export in Isaac Lab, independently of RSL-RL."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import warp as wp

wp.config.enable_backward = False

from isaaclab.app import add_launcher_args, launch_simulation

import robot_lab  # noqa: F401  # isort: skip
from robot_lab.policies.opendoge_legacy import DEFAULT_POLICY, JOINT_NAMES, LegacyOnnxPolicy, LegacyPolicyState


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--command", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("VX", "VY", "YAW"))
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--report", type=Path, help="Write replay metrics as JSON.")
    parser.add_argument("--physics", default="newton_mjwarp", choices=["newton_mjwarp", "isaacsim_physx", "ovphysx"])
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.duration <= 0 or not np.isfinite(args.duration) or not np.isfinite(args.command).all():
        parser.error("duration must be positive and command values must be finite")

    # Fail on a missing/incompatible model before starting the physics runtime.
    policy = LegacyOnnxPolicy(args.policy)
    state = LegacyPolicyState()

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils import instantiate

    from isaaclab_tasks.core.velocity.velocity_env_cfg import RoughPhysicsCfg

    from robot_lab.assets.opendoge import OPENDOGE_CFG

    cfg = sim_utils.SimulationCfg(dt=0.005, device=args.device, physics=getattr(RoughPhysicsCfg(), args.physics))
    cfg.use_newton_actuators = False  # Evaluate explicit PD each physics step, like the original training loop.
    with launch_simulation(cfg, args) as physics_cfg:
        cfg.physics = physics_cfg
        sim = sim_utils.SimulationContext(cfg)
        scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
        scene_cfg.ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.CuboidCfg(
                size=(200.0, 200.0, 0.1),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
        )
        scene_cfg.robot = OPENDOGE_CFG
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
        heights, velocities = [], []
        start_xy = robot.data.root_pos_w.torch[0, :2].cpu().numpy().copy()
        with torch.inference_mode():
            while sim.is_running() and steps < round(args.duration / cfg.dt):
                if steps % 2 == 0:
                    observation = state.observe(
                        args.command,
                        robot.data.root_ang_vel_b.torch[0].cpu().numpy(),
                        robot.data.projected_gravity_b.torch[0].cpu().numpy(),
                        robot.data.joint_pos.torch[0, joint_ids].cpu().numpy(),
                        robot.data.joint_vel.torch[0, joint_ids].cpu().numpy(),
                    )
                    targets[0, joint_ids] = torch.as_tensor(state.accept_action(policy(observation)), device=sim.device)
                robot.actuators.target_command.set_position_index(value=targets)
                scene.write_data_to_sim()
                sim.step()
                scene.update(cfg.dt)
                steps += 1
                height = float(robot.data.root_pos_w.torch[0, 2])
                heights.append(height)
                velocities.append(robot.data.root_lin_vel_b.torch[0].cpu().numpy().copy())
                if not np.isfinite(height):
                    raise RuntimeError("Non-finite robot state during replay")
        report = {
            "policy": str(args.policy),
            "command": args.command,
            "physics": type(physics_cfg).__name__,
            "steps": steps,
            "simulated_seconds": steps * cfg.dt,
            "min_base_height": min(heights) if heights else None,
            "final_base_height": heights[-1] if heights else None,
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
