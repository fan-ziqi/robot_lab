"""Contact-sensor data accessors — IsaacLab.

Business MDP code (e.g. ``feet_air_time``) calls these instead of touching
``ContactSensor.data.<field>`` directly, so field-name and shape differences
between backends are confined to this file.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv


def _sensor(env: ManagerBasedRLEnv, name: str):
    return env.scene.sensors[name]


def current_air_time(env: ManagerBasedRLEnv, sensor_name: str) -> torch.Tensor:
    return _sensor(env, sensor_name).data.current_air_time


def last_air_time(env: ManagerBasedRLEnv, sensor_name: str) -> torch.Tensor:
    return _sensor(env, sensor_name).data.last_air_time


def current_contact_time(env: ManagerBasedRLEnv, sensor_name: str) -> torch.Tensor:
    return _sensor(env, sensor_name).data.current_contact_time


def last_contact_time(env: ManagerBasedRLEnv, sensor_name: str) -> torch.Tensor:
    return _sensor(env, sensor_name).data.last_contact_time


def net_forces_w(env: ManagerBasedRLEnv, sensor_name: str) -> torch.Tensor:
    """Per-body net contact force in world frame, shape ``[B, P, 3]``."""
    return _sensor(env, sensor_name).data.net_forces_w


def net_forces_history(env: ManagerBasedRLEnv, sensor_name: str) -> torch.Tensor:
    """Per-body net contact force history, shape ``[B, H, P, 3]``."""
    return _sensor(env, sensor_name).data.net_forces_w_history


def first_contact(env: ManagerBasedRLEnv, sensor_name: str, dt: float) -> torch.Tensor:
    return _sensor(env, sensor_name).compute_first_contact(dt)


__all__ = [
    "current_air_time",
    "current_contact_time",
    "first_contact",
    "last_air_time",
    "last_contact_time",
    "net_forces_history",
    "net_forces_w",
]
