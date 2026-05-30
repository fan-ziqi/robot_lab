"""mjlab-specific asset wiring helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg


def build_entity_cfg(
    *,
    mjcf_path: str | Path,
    init_pos: tuple[float, float, float],
    init_joint_pos: dict[str, float],
    actuators: tuple[Any, ...],
    soft_joint_pos_limit_factor: float = 0.9,
    collisions: tuple[Any, ...] = (),
) -> EntityCfg:
    """Build a mjlab EntityCfg from neutral robot data."""
    mjcf_path = Path(mjcf_path)

    def _spec_fn() -> mujoco.MjSpec:
        # mjlab 1.4+ resolves mesh paths relative to the MJCF file location;
        # no explicit update_assets call is needed.
        return mujoco.MjSpec.from_file(str(mjcf_path))

    return EntityCfg(
        spec_fn=_spec_fn,
        init_state=EntityCfg.InitialStateCfg(
            pos=init_pos,
            joint_pos=init_joint_pos,
            joint_vel={".*": 0.0},
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=actuators,
            soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
        ),
        collisions=collisions,
    )


__all__ = ["build_entity_cfg"]
