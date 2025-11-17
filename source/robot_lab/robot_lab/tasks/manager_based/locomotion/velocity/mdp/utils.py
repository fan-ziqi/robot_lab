# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for terrain-aware operations."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _get_terrain_column_range(terrain_cfg, terrain_name: str, device) -> tuple[int, int] | None:
    """Helper function to calculate column range for a terrain type.

    Args:
        terrain_cfg: The terrain generator configuration.
        terrain_name: Name of the terrain.
        device: Torch device.

    Returns:
        Tuple of (col_start, col_end) or None if terrain not found.
    """
    if terrain_cfg.sub_terrains is None or terrain_name not in terrain_cfg.sub_terrains:
        return None

    sub_terrain_names = list(terrain_cfg.sub_terrains.keys())
    proportions = torch.tensor(
        [sub_cfg.proportion for sub_cfg in terrain_cfg.sub_terrains.values()],
        device=device
    )
    proportions = proportions / proportions.sum()
    cumsum_props = torch.cumsum(proportions, dim=0)

    terrain_idx = sub_terrain_names.index(terrain_name)
    # Use round() instead of int() to properly allocate columns
    col_start = round((0.0 if terrain_idx == 0 else cumsum_props[terrain_idx - 1].item()) * terrain_cfg.num_cols)
    col_end = round(cumsum_props[terrain_idx].item() * terrain_cfg.num_cols)

    return (col_start, col_end)


def is_env_assigned_to_terrain(env: ManagerBasedEnv, terrain_name: str) -> torch.Tensor:
    """Check which environments are initially assigned to the specified terrain type.

    Each environment is assigned to a specific terrain cell at initialization.
    This function returns a mask indicating which environments were assigned to the given terrain type.

    Args:
        env: The environment instance.
        terrain_name: Name of the terrain to check (e.g., "pits", "stairs").

    Returns:
        Boolean tensor of shape (num_envs,) where True means the environment is assigned to this terrain.
    """
    # Check if terrain and terrain generator are available
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None or not hasattr(terrain, "terrain_types"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if terrain.cfg.terrain_type != "generator" or terrain.cfg.terrain_generator is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    terrain_cfg = terrain.cfg.terrain_generator
    col_range = _get_terrain_column_range(terrain_cfg, terrain_name, env.device)
    if col_range is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    col_start, col_end = col_range
    # terrain_types directly stores column indices, so just check if they're in range
    return (terrain.terrain_types >= col_start) & (terrain.terrain_types < col_end)


def is_robot_on_terrain(env: ManagerBasedEnv, terrain_name: str, asset_name: str = "robot") -> torch.Tensor:
    """Check which robots are currently standing on the specified terrain type.

    This function calculates which terrain grid cell each robot is on based on its world position,
    then checks if that cell's terrain type matches the specified terrain.

    Args:
        env: The environment instance.
        terrain_name: Name of the terrain to check (e.g., "pits", "stairs").
        asset_name: Name of the robot asset. Defaults to "robot".

    Returns:
        Boolean tensor of shape (num_envs,) where True means the robot is currently on this terrain.
    """
    # Check if terrain and terrain generator are available
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None or not hasattr(terrain, "terrain_types"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if terrain.cfg.terrain_type != "generator" or terrain.cfg.terrain_generator is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    terrain_cfg = terrain.cfg.terrain_generator
    col_range = _get_terrain_column_range(terrain_cfg, terrain_name, env.device)
    if col_range is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    col_start, col_end = col_range

    # Get robot positions in world frame
    asset = env.scene[asset_name]
    robot_pos_w = asset.data.root_pos_w[:, :2]  # [num_envs, 2] (x, y)

    # Get terrain grid information
    terrain_origins = terrain.terrain_origins  # [num_rows, num_cols, 3]
    num_rows, num_cols, _ = terrain_origins.shape

    # Use terrain_origins to directly compute which cell each robot is in
    # terrain_origins[r, c, :2] is the center of cell (r, c)
    # We need to find the closest terrain origin for each robot

    # Reshape terrain_origins for distance calculation
    terrain_origins_2d = terrain_origins[:, :, :2].reshape(num_rows * num_cols, 2)  # [num_rows*num_cols, 2]

    # Calculate distances from each robot to all terrain origins
    distances = torch.cdist(robot_pos_w, terrain_origins_2d)  # [num_envs, num_rows*num_cols]

    # Find the closest terrain origin for each robot
    closest_flat_idx = torch.argmin(distances, dim=1)  # [num_envs]

    # Convert flat index to column index
    # flat_idx = row * num_cols + col
    col_idx = closest_flat_idx % num_cols  # [num_envs]

    # Check if the robot's current terrain column is in the specified terrain's range
    return (col_idx >= col_start) & (col_idx < col_end)
