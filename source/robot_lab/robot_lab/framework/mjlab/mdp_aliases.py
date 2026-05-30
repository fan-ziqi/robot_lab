"""Re-export mjlab MDP functions used by robot_lab business code.

The full list comes from the Phase 0 audit
(``docs/superpowers/specs/2026-05-30-dual-framework-audit.md``); this file
mirrors ``robot_lab.framework.isaaclab.mdp_aliases``. Names are aligned with
the IsaacLab side. Where a name is not directly available in mjlab, a thin
stub raising ``FrameworkRequiredError`` is provided so business code can
still construct the cfg term — but the term must be registered with
``framework_required="isaaclab"`` or the call will error at term execution.
"""

from __future__ import annotations

from typing import Any

# Common MDP — all `same` per audit Table 2.
from mjlab.envs.mdp import (
    action_rate_l2,
    apply_external_force_torque,
    base_ang_vel,
    base_lin_vel,
    flat_orientation_l2,
    generated_commands,
    height_scan,
    is_alive,
    is_terminated,
    joint_acc_l2,
    joint_pos_limits,
    joint_pos_rel,
    joint_torques_l2,
    joint_vel_l2,
    joint_vel_rel,
    last_action,
    projected_gravity,
    push_by_setting_velocity,
    reset_joints_by_offset,
    reset_root_state_uniform,
    time_out,
)

# Velocity-task MDP from mjlab.
from mjlab.tasks.velocity.mdp import feet_air_time, illegal_contact

# Aliased: mjlab uses different names — re-export under the IsaacLab-canonical name.
from mjlab.tasks.velocity.mdp.curriculums import terrain_levels_vel
from mjlab.tasks.velocity.mdp.rewards import (
    track_angular_velocity as track_ang_vel_z_exp,
    track_linear_velocity as track_lin_vel_xy_exp,
)
from mjlab.tasks.velocity.mdp.terminations import (
    out_of_terrain_bounds as terrain_out_of_bounds,
)

# DR module — IsaacLab calls them randomize_rigid_body_*; mjlab calls them
# dr.geom_friction / dr.body_mass / dr.body_com_offset / dr.pd_gains. Aliased.
from mjlab.envs.mdp.dr import body_com_offset as randomize_rigid_body_com
from mjlab.envs.mdp.dr import body_mass as randomize_rigid_body_mass
from mjlab.envs.mdp.dr import geom_friction as randomize_rigid_body_material
from mjlab.envs.mdp.dr import pd_gains as randomize_actuator_gains

# framework_required-isaaclab: not present in mjlab. Stub raises if invoked.
from robot_lab.framework.spec import FrameworkRequiredError


def reset_joints_by_scale(*args: Any, **kwargs: Any) -> Any:
    raise FrameworkRequiredError(
        symbol="reset_joints_by_scale", required="isaaclab", current="mjlab"
    )


__all__ = [
    "action_rate_l2",
    "apply_external_force_torque",
    "base_ang_vel",
    "base_lin_vel",
    "feet_air_time",
    "flat_orientation_l2",
    "generated_commands",
    "height_scan",
    "illegal_contact",
    "is_alive",
    "is_terminated",
    "joint_acc_l2",
    "joint_pos_limits",
    "joint_pos_rel",
    "joint_torques_l2",
    "joint_vel_l2",
    "joint_vel_rel",
    "last_action",
    "projected_gravity",
    "push_by_setting_velocity",
    "randomize_actuator_gains",
    "randomize_rigid_body_com",
    "randomize_rigid_body_mass",
    "randomize_rigid_body_material",
    "reset_joints_by_offset",
    "reset_joints_by_scale",
    "reset_root_state_uniform",
    "terrain_levels_vel",
    "terrain_out_of_bounds",
    "time_out",
    "track_ang_vel_z_exp",
    "track_lin_vel_xy_exp",
]
