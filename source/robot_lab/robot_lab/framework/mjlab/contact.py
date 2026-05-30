"""Contact-sensor data accessors — mjlab.

Mirror of ``robot_lab.framework.isaaclab.contact``. Same return shapes:
``net_forces_w`` returns ``[B, P, 3]`` (mjlab's ``force`` is ``[B, P, S, 3]``
with ``S=1`` slot, so we squeeze). ``net_forces_history`` aligns to
``[B, H, P, 3]``.
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv


def _sensor(env: ManagerBasedRlEnv, name: str):
    return env.scene.sensors[name]


def current_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    return _sensor(env, sensor_name).data["current_air_time"]


def last_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    return _sensor(env, sensor_name).data["last_air_time"]


def current_contact_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    return _sensor(env, sensor_name).data["current_contact_time"]


def last_contact_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    return _sensor(env, sensor_name).data["last_contact_time"]


def net_forces_w(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Per-body net contact force, shape ``[B, P, 3]`` (slot dim squeezed)."""
    force = _sensor(env, sensor_name).data["force"]  # [B, P, S, 3], S=1
    return force.squeeze(-2)


def net_forces_history(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Per-body net contact force history, shape ``[B, H, P, 3]``."""
    history = _sensor(env, sensor_name).data["force_history"]  # [B, H, P, S, 3]
    return history.squeeze(-2)


def first_contact(env: ManagerBasedRlEnv, sensor_name: str, dt: float) -> torch.Tensor:
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
