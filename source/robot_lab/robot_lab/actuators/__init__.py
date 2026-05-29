# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Custom actuator models used by robot_lab tasks.

PACE actuator model: see pace_actuator.py / pace_actuator_cfg.py.
"""

from . import pace_actuator
from .pace_actuator import PaceDCMotor
from .pace_actuator_cfg import PaceDCMotorCfg

__all__ = [
    "PaceDCMotor",
    "PaceDCMotorCfg",
    "pace_actuator",
]
