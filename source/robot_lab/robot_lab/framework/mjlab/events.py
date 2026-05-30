"""Framework-neutral DR / event term factories — mjlab.

Mirror of ``robot_lab.framework.isaaclab.events``. Same kwargs; bodies wire
into mjlab's DR namespace (``mjlab.envs.mdp.dr.*``) via the aliases in
``mdp_aliases``.
"""

from __future__ import annotations

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from robot_lab.framework.mjlab import mdp_aliases as _mdp


def randomize_geom_friction(
    asset_cfg: SceneEntityCfg,
    static_range: tuple[float, float] = (0.3, 1.0),
    dynamic_range: tuple[float, float] = (0.3, 0.8),
    restitution_range: tuple[float, float] = (0.0, 0.4),
    num_buckets: int = 64,
) -> EventTermCfg:
    # mjlab's geom_friction takes a single (lo, hi) range (single-coefficient
    # MuJoCo friction model). Drop dynamic_range / restitution_range; the IL
    # signature compat keeps callers framework-neutral but only static_range
    # is consumed.
    del dynamic_range, restitution_range, num_buckets
    return EventTermCfg(
        mode="startup",
        func=_mdp.randomize_rigid_body_material,
        params={
            "asset_cfg": asset_cfg,
            "operation": "abs",
            "ranges": static_range,
            "shared_random": True,
        },
    )


def randomize_body_mass(
    asset_cfg: SceneEntityCfg,
    mass_range: tuple[float, float],
    operation: str = "add",
    recompute_inertia: bool = True,
) -> EventTermCfg:
    del recompute_inertia  # mjlab handles inertia automatically.
    return EventTermCfg(
        mode="startup",
        func=_mdp.randomize_rigid_body_mass,
        params={"asset_cfg": asset_cfg, "operation": operation, "ranges": mass_range},
    )


def randomize_body_com(
    asset_cfg: SceneEntityCfg,
    com_range: dict[str, tuple[float, float]],
) -> EventTermCfg:
    # mjlab body_com_offset takes `ranges=(lo, hi)` and `axes=(0,1,2)`. IL's
    # com_range is a dict {x: (lo,hi), y: (lo,hi), z: (lo,hi)}; we collapse to
    # the per-axis ranges by taking the bounding (min lo, max hi).
    los = [com_range[k][0] for k in ("x", "y", "z") if k in com_range]
    his = [com_range[k][1] for k in ("x", "y", "z") if k in com_range]
    bounding = (min(los) if los else 0.0, max(his) if his else 0.0)
    return EventTermCfg(
        mode="startup",
        func=_mdp.randomize_rigid_body_com,
        params={"asset_cfg": asset_cfg, "ranges": bounding, "axes": (0, 1, 2)},
    )


def apply_external_force_torque(
    asset_cfg: SceneEntityCfg,
    force_range: tuple[float, float] = (-10.0, 10.0),
    torque_range: tuple[float, float] = (-10.0, 10.0),
) -> EventTermCfg:
    return EventTermCfg(
        mode="reset",
        func=_mdp.apply_external_force_torque,
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
    # mjlab pd_gains takes kp_range / kd_range; map IL's stiffness/damping.
    return EventTermCfg(
        mode="reset",
        func=_mdp.randomize_actuator_gains,
        params={
            "asset_cfg": asset_cfg,
            "kp_range": stiffness_range,
            "kd_range": damping_range,
            "operation": operation,
            "distribution": distribution,
        },
    )


def randomize_reset_joints(
    position_range: tuple[float, float] = (1.0, 1.0),
    velocity_range: tuple[float, float] = (0.0, 0.0),
    by: str = "scale",
) -> EventTermCfg:
    if by == "scale":
        # mjlab has only by-offset. The framework_required stub raises if invoked.
        func = _mdp.reset_joints_by_scale
    else:
        func = _mdp.reset_joints_by_offset
    return EventTermCfg(
        mode="reset",
        func=func,
        params={"position_range": position_range, "velocity_range": velocity_range},
    )


def randomize_reset_base(
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
) -> EventTermCfg:
    return EventTermCfg(
        mode="reset",
        func=_mdp.reset_root_state_uniform,
        params={"pose_range": pose_range, "velocity_range": velocity_range},
    )


def push_robot(
    velocity_range: dict[str, tuple[float, float]],
    interval_range_s: tuple[float, float] = (5.0, 10.0),
) -> EventTermCfg:
    return EventTermCfg(
        mode="interval",
        func=_mdp.push_by_setting_velocity,
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
