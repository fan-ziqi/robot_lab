# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026 robot_lab contributors
# SPDX-License-Identifier: Apache-2.0
"""Opt-in Newton configuration using measured, exact PhysX public name order."""

import hashlib
import json
from collections import Counter
from pathlib import Path


def exact_names(names, label: str) -> tuple[str, ...]:
    """Accept a measured sequence of unique literal names, never a convention guess."""
    if not isinstance(names, (list, tuple)) or not names:
        raise ValueError(f"{label} must be a nonempty list or tuple of exact names")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError(f"{label} contains an empty or non-string name")
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise ValueError(f"{label} contains duplicate names: {duplicates}")
    return tuple(names)


def name_order_indices(expected, actual) -> tuple[int, ...]:
    """Return indices mapping actual values into expected order; require exact sets."""
    expected = exact_names(expected, "expected names")
    actual = exact_names(actual, "actual names")
    missing, extra = sorted(set(expected) - set(actual)), sorted(set(actual) - set(expected))
    if missing or extra:
        raise ValueError(f"Name mismatch: missing={missing}; extra={extra}")
    lookup = {name: index for index, name in enumerate(actual)}
    return tuple(lookup[name] for name in expected)


def load_order_profile(cfg, key: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Load measured names for a verified URDF identity or exact USD URI.

    Receipt paths and hashes are provenance metadata: remote deployments need
    the small profile and asset, rather than the uncommitted measurement receipt.
    USD URI identity does not verify remote or composed asset bytes. No USD URI
    is read or downloaded here; runtime ordering checks remain necessary.
    """
    profile_path = Path(__file__).with_name("newton_ordering_profiles.json")
    profiles = json.loads(profile_path.read_text())
    if profiles.get("schema_version") != 1:
        raise ValueError(f"Unsupported ordering-profile schema in {profile_path}")
    if key not in profiles["profiles"]:
        raise KeyError(f"Unknown measured ordering profile: {key}")
    profile = profiles["profiles"][key]
    spawn = cfg.scene.robot.spawn
    asset_kind = profile.get("asset_kind", "urdf")
    if asset_kind == "urdf":
        asset = Path(spawn.asset_path)
        digest = hashlib.sha256(asset.read_bytes()).hexdigest()
        if digest != profile["asset_sha256"]:
            raise ValueError(
                f"Ordering profile {key} asset SHA256 mismatch: "
                f"expected={profile['asset_sha256']}, actual={digest}, asset={asset}"
            )
        if spawn.merge_fixed_joints != profile["merge_fixed_joints"]:
            raise ValueError(
                f"Ordering profile {key} merge_fixed_joints mismatch: "
                f"expected={profile['merge_fixed_joints']}, actual={spawn.merge_fixed_joints}"
            )
        if not asset.as_posix().endswith("/" + profile["asset_relative_path"]):
            raise ValueError(f"Ordering profile {key} asset relative path mismatch: {asset}")
    elif asset_kind == "usd":
        uri = profile.get("asset_uri")
        if not isinstance(uri, str) or not uri.strip():
            raise ValueError(f"Ordering profile {key} requires a nonempty USD asset_uri")
        actual_uri = getattr(spawn, "usd_path", None)
        if actual_uri != uri:
            raise ValueError(f"Ordering profile {key} USD URI mismatch: expected={uri!r}, actual={actual_uri!r}")
    else:
        raise ValueError(f"Ordering profile {key} has unknown asset_kind: {asset_kind!r}")
    return exact_names(profile["joint_names"], f"{key} joint_names"), exact_names(
        profile["body_names"], f"{key} body_names"
    )


def _contact_entity_updates(cfg, references):
    """Resolve sensor entity selections against measured PhysX sensor order."""
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils.string import resolve_matching_names

    if not isinstance(references, dict):
        raise ValueError("contact_sensor_body_names must be a sensor-name mapping")
    references = {name: exact_names(names, f"{name} contact sensor reference") for name, names in references.items()}
    if any(not isinstance(name, str) or not name.strip() or name == "robot" for name in references):
        raise ValueError("Contact sensor reference keys must be nonempty sensor names, excluding robot")
    updates, visited = [], set()

    def indices(ids, size):
        if isinstance(ids, slice):
            for bound in (ids.start, ids.stop, ids.step):
                if bound is not None and (isinstance(bound, bool) or not isinstance(bound, int)):
                    raise ValueError(f"Invalid body index slice: {ids}")
            if (
                ids.step == 0
                or (ids.start is not None and not 0 <= ids.start < size)
                or (ids.stop is not None and not 0 <= ids.stop <= size)
            ):
                raise ValueError(f"Out-of-range body index slice: {ids}, reference size={size}")
            result = list(range(size))[ids]
        elif isinstance(ids, (list, tuple)):
            result = list(ids)
        else:
            raise ValueError(f"Unsupported explicit body_ids type: {type(ids).__name__}")
        if not result or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < size for i in result):
            raise ValueError(f"Empty or invalid body_ids selection: {result}, reference size={size}")
        if len(set(result)) != len(result):
            raise ValueError(f"Duplicate body_ids selection: {result}")
        return result

    def walk(value):
        if value is None or isinstance(value, (str, bytes, bool, int, float, slice)):
            return
        if id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, SceneEntityCfg):
            if value.name not in references:
                return
            reference = references[value.name]
            if value.body_names is not None:
                selected_ids, selected_names = resolve_matching_names(
                    value.body_names, reference, preserve_order=value.preserve_order
                )
                ids_unset = isinstance(value.body_ids, slice) and value.body_ids == slice(None)
                if not ids_unset and indices(value.body_ids, len(reference)) != selected_ids:
                    raise ValueError(f"Inconsistent body_names/body_ids for contact sensor {value.name}")
            else:
                selected_names = [reference[i] for i in indices(value.body_ids, len(reference))]
            selected_names = list(exact_names(selected_names, f"{value.name} selected sensor bodies"))
            updates.append((value, selected_names))
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
        elif hasattr(value, "__dict__") and not callable(value):
            for item in vars(value).values():
                walk(item)

    for name in ("observations", "rewards", "events", "terminations", "curriculum"):
        walk(getattr(cfg, name, None))
    return updates


def configure_newton(
    cfg,
    joint_names,
    body_names,
    njmax=256,
    nconmax=128,
    selected=False,
    *,
    contact_sensor_body_names=None,
    disabled_event_names=(),
):
    """Add a measured-order Newton alternative, preserving the original PhysX object.

    ``selected=True`` converts an already resolved diagnostic config directly.
    It does not opt a registered task into Newton support. Reward functions,
    weights, actions and timing are unchanged. Explicit event disabling applies
    only to the Newton branch. Optional contact-selector
    metadata remapping preserves PhysX selection and ordering, and selects
    explicit measured body names only in the Newton branch.
    """
    from copy import deepcopy

    from isaaclab_physx.physics import PhysxCfg

    from isaaclab.managers import TerminationTermCfg
    from isaaclab.sensors import RayCasterCfg

    from isaaclab_tasks.utils import preset

    from .mdp.terminations import invalid_physics_state
    from .velocity_env_cfg import VelocityPhysicsCfg

    joints, bodies = exact_names(joint_names, "joint_names"), exact_names(body_names, "body_names")
    if not isinstance(disabled_event_names, (list, tuple)):
        raise ValueError("disabled_event_names must be a list or tuple of unique event names")
    disabled_events = exact_names(disabled_event_names, "disabled_event_names") if disabled_event_names else ()
    event_updates = []
    for name in disabled_events:
        events = getattr(cfg, "events", None)
        if events is None or not hasattr(events, name) or getattr(events, name) is None:
            raise ValueError(f"Cannot disable missing or inactive event: {name}")
        event_updates.append((name, getattr(events, name)))
    contact_updates = (
        _contact_entity_updates(cfg, contact_sensor_body_names) if contact_sensor_body_names is not None else []
    )
    if any(isinstance(limit, bool) or not isinstance(limit, int) or limit < 1 for limit in (njmax, nconmax)):
        raise ValueError("njmax and nconmax must be positive integers")
    physx = cfg.sim.physics
    if not isinstance(physx, PhysxCfg):
        raise TypeError(f"Expected original resolved PhysxCfg, got {type(physx).__module__}:{type(physx).__qualname__}")
    physics = VelocityPhysicsCfg(default=physx, isaacsim_physx=physx)
    # configclass construction copies mutable arguments; retain the authored
    # PhysX object itself as well as its complete field values.
    physics.default = physx
    physics.isaacsim_physx = physx
    physics.newton_mjwarp.solver_cfg.njmax = njmax
    physics.newton_mjwarp.solver_cfg.nconmax = nconmax
    cfg.sim.physics = physics.newton_mjwarp if selected else physics
    guard = TerminationTermCfg(func=invalid_physics_state, time_out=False)
    original_terminations = cfg.terminations
    newton_terminations = deepcopy(original_terminations)
    newton_terminations.invalid_physics_state = guard
    cfg.terminations = (
        newton_terminations if selected else preset(default=original_terminations, newton_mjwarp=newton_terminations)
    )
    robot = cfg.scene.robot
    robot.joint_ordering = joints if selected else preset(default=robot.joint_ordering, newton_mjwarp=joints)
    robot.body_ordering = bodies if selected else preset(default=robot.body_ordering, newton_mjwarp=bodies)
    for sensor in vars(cfg.scene).values():
        if isinstance(sensor, RayCasterCfg):
            sensor.global_world_only = (
                True if selected else preset(default=sensor.global_world_only, newton_mjwarp=True)
            )
    for entity, names in contact_updates:
        entity.body_names = names if selected else preset(default=entity.body_names, newton_mjwarp=names)
        entity.body_ids = slice(None) if selected else preset(default=entity.body_ids, newton_mjwarp=slice(None))
        entity.preserve_order = True if selected else preset(default=entity.preserve_order, newton_mjwarp=True)
    for name, original in event_updates:
        setattr(cfg.events, name, None if selected else preset(default=original, newton_mjwarp=None))
    from isaaclab.managers import EventTermCfg, SceneEntityCfg

    from .mdp.events import floor_body_mass

    original_events = cfg.events
    newton_events = deepcopy(original_events)
    newton_events.mass_floor = EventTermCfg(
        func=floor_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot"), "min_mass": 0.05, "min_inertia": 8e-6},
    )
    cfg.events = newton_events if selected else preset(default=original_events, newton_mjwarp=newton_events)
    return cfg


def configure_newton_from_profile(cfg, key: str, njmax=256, nconmax=128):
    """Configure an explicitly selected measured profile, including optional sensor order."""
    joints, bodies = load_order_profile(cfg, key)
    profile = json.loads(Path(__file__).with_name("newton_ordering_profiles.json").read_text())["profiles"][key]
    return configure_newton(
        cfg,
        joints,
        bodies,
        njmax=njmax,
        nconmax=nconmax,
        contact_sensor_body_names=profile.get("contact_sensor_body_names"),
        disabled_event_names=profile.get("disabled_event_names", ()),
    )
