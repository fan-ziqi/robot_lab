"""Framework-neutral DR / event term factories — IsaacLab.

Each factory returns an :class:`EventTermCfg` ready to drop into a manager
dict. Kwargs use IsaacLab-style names; mjlab side's ``events.py`` mirrors the
signatures and translates the body to mjlab's DR module.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg, SceneEntityCfg

from robot_lab.framework.isaaclab import mdp_aliases as _mdp


def randomize_geom_friction(
    asset_cfg: SceneEntityCfg,
    static_range: tuple[float, float] = (0.3, 1.0),
    dynamic_range: tuple[float, float] = (0.3, 0.8),
    restitution_range: tuple[float, float] = (0.0, 0.4),
    num_buckets: int = 64,
) -> EventTermCfg:
    return EventTermCfg(
        func=_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": asset_cfg,
            "static_friction_range": static_range,
            "dynamic_friction_range": dynamic_range,
            "restitution_range": restitution_range,
            "num_buckets": num_buckets,
        },
    )


def randomize_body_mass(
    asset_cfg: SceneEntityCfg,
    mass_range: tuple[float, float],
    operation: str = "add",
    recompute_inertia: bool = True,
) -> EventTermCfg:
    return EventTermCfg(
        func=_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": asset_cfg,
            "mass_distribution_params": mass_range,
            "operation": operation,
            "recompute_inertia": recompute_inertia,
        },
    )


def randomize_body_com(
    asset_cfg: SceneEntityCfg,
    com_range: dict[str, tuple[float, float]],
) -> EventTermCfg:
    return EventTermCfg(
        func=_mdp.randomize_rigid_body_com,
        mode="startup",
        params={"asset_cfg": asset_cfg, "com_range": com_range},
    )


def apply_external_force_torque(
    asset_cfg: SceneEntityCfg,
    force_range: tuple[float, float] = (-10.0, 10.0),
    torque_range: tuple[float, float] = (-10.0, 10.0),
) -> EventTermCfg:
    return EventTermCfg(
        func=_mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": asset_cfg,
            "force_range": force_range,
            "torque_range": torque_range,
        },
    )


def randomize_actuator_gains(
    asset_cfg: SceneEntityCfg,
    stiffness_range: tuple[float, float] = (0.5, 2.0),
    damping_range: tuple[float, float] = (0.5, 2.0),
    operation: str = "scale",
    distribution: str = "uniform",
) -> EventTermCfg:
    return EventTermCfg(
        func=_mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": asset_cfg,
            "stiffness_distribution_params": stiffness_range,
            "damping_distribution_params": damping_range,
            "operation": operation,
            "distribution": distribution,
        },
    )


def randomize_reset_joints(
    position_range: tuple[float, float] = (1.0, 1.0),
    velocity_range: tuple[float, float] = (0.0, 0.0),
    by: str = "scale",
) -> EventTermCfg:
    func = _mdp.reset_joints_by_scale if by == "scale" else _mdp.reset_joints_by_offset
    return EventTermCfg(
        func=func,
        mode="reset",
        params={"position_range": position_range, "velocity_range": velocity_range},
    )


def randomize_reset_base(
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
) -> EventTermCfg:
    return EventTermCfg(
        func=_mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": pose_range, "velocity_range": velocity_range},
    )


def push_robot(
    velocity_range: dict[str, tuple[float, float]],
    interval_range_s: tuple[float, float] = (5.0, 10.0),
) -> EventTermCfg:
    return EventTermCfg(
        func=_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=interval_range_s,
        params={"velocity_range": velocity_range},
    )


__all__ = [
    "apply_external_force_torque",
    "push_robot",
    "randomize_actuator_gains",
    "randomize_body_com",
    "randomize_body_mass",
    "randomize_geom_friction",
    "randomize_reset_base",
    "randomize_reset_joints",
]
