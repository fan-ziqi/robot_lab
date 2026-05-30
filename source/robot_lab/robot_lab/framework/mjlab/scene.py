"""Framework-neutral scene helpers — mjlab.

mjlab assembles scenes via ``SceneCfg(terrain=..., entities={...}, sensors=())``.
There is no cfg-level sky_light analog — lights live inside each entity's MJCF.
make_sky_light() returns ``None`` so the velocity_env_cfg can blindly call it.
"""

from __future__ import annotations

from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG


def make_rough_terrain() -> TerrainEntityCfg:
    return TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
    )


def make_flat_terrain() -> TerrainEntityCfg:
    return TerrainEntityCfg(terrain_type="plane")


def make_sky_light() -> None:
    """No-op on mjlab: sky/lights are configured via the MjSpec per entity."""
    return None


__all__ = ["make_flat_terrain", "make_rough_terrain", "make_sky_light"]
