"""Framework-neutral sensor factories that produce mjlab cfg objects.

Mirror of ``robot_lab.framework.isaaclab.sensors``. Same kwargs, returns mjlab
``RayCastSensorCfg`` / ``ContactSensorCfg``.
"""

from __future__ import annotations

from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg as _MjContactSensorCfg,
    GridPatternCfg,
    ObjRef,
    RayCastSensorCfg,
)


def make_height_scan_sensor(
    *,
    name: str,
    body_name: str,
    pattern_size: tuple[float, float] = (1.6, 1.0),
    pattern_resolution: float = 0.1,
    max_distance: float = 5.0,
    update_period: float | None = None,
    debug_vis: bool = False,
) -> RayCastSensorCfg:
    del update_period  # mjlab raycasts at sim rate; no per-sensor update period.
    return RayCastSensorCfg(
        name=name,
        frame=ObjRef(type="body", name=body_name, entity="robot"),
        ray_alignment="yaw",
        pattern=GridPatternCfg(size=pattern_size, resolution=pattern_resolution),
        max_distance=max_distance,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        debug_vis=debug_vis,
    )


def make_contact_sensor(
    *,
    name: str,
    body_pattern: str = ".*",
    history_length: int = 3,
    track_air_time: bool = True,
) -> _MjContactSensorCfg:
    return _MjContactSensorCfg(
        name=name,
        primary=ContactMatch(mode="body", entity="robot", pattern=body_pattern),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=track_air_time,
        history_length=history_length,
    )


__all__ = ["make_contact_sensor", "make_height_scan_sensor"]
