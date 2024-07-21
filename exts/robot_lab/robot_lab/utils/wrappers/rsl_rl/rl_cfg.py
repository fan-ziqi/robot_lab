# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from omni.isaac.lab.utils import configclass
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl.rl_cfg import RslRlPpoAlgorithmCfg


@configclass
class RslRlAmpPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the AMP PPO algorithm."""

    amp_replay_buffer_size: int = MISSING
    """The size of the replay buffer for AMP"""
