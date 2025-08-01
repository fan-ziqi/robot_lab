# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
AMP G1 dance environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="RobotLab-Isaac-G1-AMP-Dance-Direct-v0",
    entry_point=f"{__name__}.g1_amp_env:G1AmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_env_cfg:G1AmpDanceEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_dance_amp_cfg.yaml",
    },
)
