# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-Amp-Unitree-A1-v0",
    entry_point="robot_lab.tasks.locomotion.velocity.config.quadruped.unitree_a1_amp.env.manager_based_rl_amp_env:ManagerBasedRLAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeA1AmpFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1AmpFlatPPORunnerCfg",
    },
)

# Rough is not supported yet.
# gym.register(
#     id="RobotLab-Isaac-Velocity-Rough-Amp-Unitree-A1-v0",
#     entry_point="robot_lab.tasks.locomotion.velocity.config.quadruped.unitree_a1_amp.env.manager_based_rl_amp_env:ManagerBasedRLAmpEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeA1AmpRoughEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1AmpRoughPPORunnerCfg",
#     },
# )
