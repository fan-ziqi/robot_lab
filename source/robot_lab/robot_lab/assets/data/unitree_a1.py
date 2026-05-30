# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Framework-neutral constants for the Unitree A1 quadruped.

Holds asset paths, the canonical initial pose, and actuator parameters in a
form that both the IsaacLab and mjlab builders can consume. No framework
imports here on purpose -- this module is loaded under either backend.

Note: The MJCF file may not yet ship with the asset bundle. The mjlab
builder defers file existence checks until the env is instantiated, so
referencing ``MJCF_PATH`` here is safe at import time.
"""

from __future__ import annotations

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

##
# Asset paths.
##

URDF_PATH = f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/unitree/a1_description/urdf/a1.urdf"
MJCF_PATH = f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/unitree/a1_description/mjcf/a1.xml"

##
# Initial state.
##

INIT_POS: tuple[float, float, float] = (0.0, 0.0, 0.38)

INIT_JOINT_POS: dict[str, float] = {
    ".*L_hip_joint": 0.0,
    ".*R_hip_joint": -0.0,
    "F.*_thigh_joint": 0.8,
    "R.*_thigh_joint": 0.8,
    ".*_calf_joint": -1.5,
}

##
# Actuator parameters (DC motor on IL, builtin position actuator on mjlab).
# Specs from https://www.trossenrobotics.com/a1-quadruped#specifications
##

EFFORT_LIMIT = 33.5
SATURATION_EFFORT = 33.5
VELOCITY_LIMIT = 21.0
STIFFNESS = 20.0
DAMPING = 0.5
FRICTION = 0.0

JOINT_NAMES_EXPR: tuple[str, ...] = (".*_joint",)


__all__ = [
    "DAMPING",
    "EFFORT_LIMIT",
    "FRICTION",
    "INIT_JOINT_POS",
    "INIT_POS",
    "JOINT_NAMES_EXPR",
    "MJCF_PATH",
    "SATURATION_EFFORT",
    "STIFFNESS",
    "URDF_PATH",
    "VELOCITY_LIMIT",
]
