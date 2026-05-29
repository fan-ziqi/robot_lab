# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

from __future__ import annotations
import torch

from isaaclab.utils import configclass

from isaaclab.actuators import DCMotorCfg
from robot_lab.actuators import pace_actuator


@configclass
class PaceDCMotorCfg(DCMotorCfg):
    """Configuration for Pace DC Motor actuator model.

    This class extends the base DCMotorCfg with Pace-specific parameters.
    """
    class_type: type = pace_actuator.PaceDCMotor
    encoder_bias: dict[str, float] | float | None = 0.0
    max_delay: torch.int | None = 0
