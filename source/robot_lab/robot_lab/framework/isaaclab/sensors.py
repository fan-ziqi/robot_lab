"""Framework-neutral sensor factories that produce IsaacLab cfg objects.

Business code calls these factories with **engine-agnostic** kwargs; the
factory writes the IsaacLab-specific fields (``prim_path``, USD asset paths).
Mirrored for mjlab in ``robot_lab.framework.mjlab.sensors``.
"""

from __future__ import annotations

from isaaclab.sensors import ContactSensorCfg as _IsaacContactSensorCfg
from isaaclab.sensors import RayCasterCfg, patterns


def make_height_scan_sensor(
    *,
    name: str,
    body_name: str,
    pattern_size: tuple[float, float] = (1.6, 1.0),
    pattern_resolution: float = 0.1,
    max_distance: float = 5.0,
    update_period: float | None = None,
    debug_vis: bool = False,
) -> RayCasterCfg:
    del name  # IsaacLab keys the sensor by its scene attribute name; the kwarg
              # is for parity with the mjlab factory which requires `name`.
    cfg = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + body_name,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=pattern_resolution, size=pattern_size
        ),
        debug_vis=debug_vis,
        mesh_prim_paths=["/World/ground"],
    )
    if update_period is not None:
        cfg.update_period = update_period
    cfg.max_distance = max_distance
    return cfg


def make_contact_sensor(
    *,
    name: str,
    body_pattern: str = ".*",
    history_length: int = 3,
    track_air_time: bool = True,
) -> _IsaacContactSensorCfg:
    del name  # see make_height_scan_sensor
    return _IsaacContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + body_pattern,
        history_length=history_length,
        track_air_time=track_air_time,
    )


__all__ = ["make_contact_sensor", "make_height_scan_sensor"]
