"""Re-export upstream IsaacLab MDP functions used by robot_lab business code.

The full list comes from the Phase 0 audit
(``docs/superpowers/specs/2026-05-30-dual-framework-audit.md``); this file is
the **only** place ``from isaaclab.envs.mdp import ...`` happens for the
IsaacLab backend. The mjlab counterpart lives in
``robot_lab.framework.mjlab.mdp_aliases``.

Symbols classified ``framework_required-isaaclab`` in the audit (e.g.
``base_pos_z``, ``joint_effort``, ``joint_deviation_l1``, ``undesired_contacts``,
``modify_reward_weight``, ``feet_air_time_positive_biped``) are NOT re-exported
here; they remain reachable as ``isaaclab.envs.mdp.<name>`` for IsaacLab-only
business code. The framework public API only re-exports symbols that are
shared (`same`) or aliased (canonical name maps to IsaacLab name verbatim).
"""

from __future__ import annotations

# Common MDP — all `same` classification per audit Table 2.
from isaaclab.envs.mdp import (
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
    randomize_actuator_gains,
    randomize_rigid_body_com,
    randomize_rigid_body_mass,
    randomize_rigid_body_material,
    reset_joints_by_offset,
    reset_joints_by_scale,
    reset_root_state_uniform,
    time_out,
)

# Velocity-task MDP from isaaclab_tasks.
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import (
    feet_air_time,
    illegal_contact,
    terrain_levels_vel,
    terrain_out_of_bounds,
    track_ang_vel_z_exp,
    track_lin_vel_xy_exp,
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
