# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Terrain-specific extensions of the upstream event terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg

from .utils import is_env_assigned_to_terrain

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class reset_root_state_uniform(mdp.reset_root_state_uniform):
    """Use upstream root resets, keeping pit environments at their default pose."""

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
        pose_range: dict[str, tuple[float, float]],
        velocity_range: dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ):
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=env.device)[env_ids or slice(None)]
        assigned_to_pits = is_env_assigned_to_terrain(env, "pits")[env_ids]
        pit_env_ids = env_ids[assigned_to_pits]
        other_env_ids = env_ids[~assigned_to_pits]
        if pit_env_ids.numel():
            super().__call__(env, pit_env_ids, {}, {}, asset_cfg)
        if other_env_ids.numel():
            super().__call__(env, other_env_ids, pose_range, velocity_range, asset_cfg)
