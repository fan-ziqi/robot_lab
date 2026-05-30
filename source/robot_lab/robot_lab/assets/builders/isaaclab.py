"""IsaacLab-specific asset wiring helpers.

Per-robot ``assets/<vendor>.py`` factories pass a small set of neutral
parameters (paths, init pose, actuator wiring) into ``build_articulation_cfg``
and the IL-side ArticulationCfg is constructed here.
"""

from __future__ import annotations

from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg


def build_articulation_cfg(
    *,
    urdf_path: str,
    init_pos: tuple[float, float, float],
    init_joint_pos: dict[str, float],
    actuators: dict[str, Any],
    soft_joint_pos_limit_factor: float = 0.9,
    rigid_props: sim_utils.RigidBodyPropertiesCfg | None = None,
    articulation_props: sim_utils.ArticulationRootPropertiesCfg | None = None,
) -> ArticulationCfg:
    """Build an IsaacLab ArticulationCfg from neutral robot data."""
    return ArticulationCfg(
        spawn=sim_utils.UrdfFileCfg(
            fix_base=False,
            merge_fixed_joints=True,
            asset_path=urdf_path,
            activate_contact_sensors=True,
            rigid_props=rigid_props
            or sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=articulation_props
            or sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0, damping=0
                )
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=init_pos,
            joint_pos=init_joint_pos,
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
        actuators=actuators,
    )


__all__ = ["build_articulation_cfg"]
