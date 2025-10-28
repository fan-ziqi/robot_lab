# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents, flat_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-Unitree-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.UnitreeG1BeyondMimicFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1BeyondMimicFlatPPORunnerCfg",
    },
)
