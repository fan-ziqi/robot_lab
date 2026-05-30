# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from robot_lab.framework import register_task

from . import agents

##
# Register tasks for the active framework.
##

register_task(
    task_id="RobotLab-Velocity-Flat-DDTRobot-Tita-v0",
    env_cfg=f"{__name__}.flat_env_cfg:DDTRobotTitaFlatEnvCfg",
    agent_cfg_entries={
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DDTRobotTitaFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:DDTRobotTitaFlatTrainerCfg",
    },
    framework_required="isaaclab",
)

register_task(
    task_id="RobotLab-Velocity-Rough-DDTRobot-Tita-v0",
    env_cfg=f"{__name__}.rough_env_cfg:DDTRobotTitaRoughEnvCfg",
    agent_cfg_entries={
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DDTRobotTitaRoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:DDTRobotTitaRoughTrainerCfg",
    },
    framework_required="isaaclab",
)
