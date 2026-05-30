# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from robot_lab.framework import register_task

from . import agents

##
# Register tasks for the active framework.
##

register_task(
    task_id="RobotLab-Velocity-Flat-MagicLab-Dog-W-v0",
    env_cfg=f"{__name__}.flat_env_cfg:MagicDogWFlatEnvCfg",
    agent_cfg_entries={
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MagicDogWFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:MagicDogWFlatTrainerCfg",
    },
    framework_required="isaaclab",
)

register_task(
    task_id="RobotLab-Velocity-Rough-MagicLab-Dog-W-v0",
    env_cfg=f"{__name__}.rough_env_cfg:MagicDogWRoughEnvCfg",
    agent_cfg_entries={
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MagicDogWRoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:MagicDogWRoughTrainerCfg",
    },
    framework_required="isaaclab",
)
