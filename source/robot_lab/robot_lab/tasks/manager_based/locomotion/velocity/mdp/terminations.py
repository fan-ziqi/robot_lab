# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Custom termination terms for locomotion velocity environments."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def bad_orientation(
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate episodes when the base orientation deviates from handstand pose.

    Args:
        env: RL 环境。
        threshold: 对 projected_gravity_z 取绝对值的阈值；大于该值视为姿态错误。
        asset_cfg: 指定机器人资产（默认为 "robot"）。

    Returns:
        torch.Tensor: bool 掩码，True 表示对应环境应终止。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    projected_gravity_z = asset.data.projected_gravity_b[:, 2]
    return torch.abs(projected_gravity_z) > threshold
