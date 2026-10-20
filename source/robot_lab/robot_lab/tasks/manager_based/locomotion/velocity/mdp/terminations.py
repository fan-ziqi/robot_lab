# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026 robot_lab contributors
# SPDX-License-Identifier: Apache-2.0
"""Physical validity checks for the opt-in Newton velocity environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def invalid_physics_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_root_lin_vel: float = 50.0,
    max_root_ang_vel: float = 100.0,
    max_joint_vel: float = 500.0,
) -> torch.Tensor:
    """Reset non-finite or exploding worlds before computing observations.

    Velocity limits apply to vector magnitudes (root) and absolute joint
    velocities. Positions are checked for finiteness, without restricting
    terrain coordinates or continuous wheel joint angles. No host synchronization
    or solver-buffer inspection is needed on the normal step path.
    """
    data = env.scene[asset_cfg.name].data
    root = data.root_state_w.torch
    joint_pos = data.joint_pos.torch
    joint_vel = data.joint_vel.torch
    body_pos = data.body_pos_w.torch
    invalid = ~torch.isfinite(root).all(dim=-1)
    invalid |= ~torch.isfinite(joint_pos).all(dim=-1)
    invalid |= ~torch.isfinite(joint_vel).all(dim=-1)
    invalid |= ~torch.isfinite(body_pos).all(dim=(-2, -1))
    invalid |= torch.linalg.vector_norm(root[:, 7:10], dim=-1) > max_root_lin_vel
    invalid |= torch.linalg.vector_norm(root[:, 10:13], dim=-1) > max_root_ang_vel
    invalid |= (joint_vel.abs() > max_joint_vel).any(dim=-1)
    return invalid
