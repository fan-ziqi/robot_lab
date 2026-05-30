# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from robot_lab.framework import register_task

from . import agents

##
# Register tasks for the active framework.
##

register_task(
    task_id="RobotLab-Velocity-Rough-RoboParty-ATOM01-v0",
    env_cfg=f"{__name__}.rough_env_cfg:RoboPartyATOM01RoughEnvCfg",
    agent_cfg_entries={
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoboPartyATOM01RoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:RoboPartyATOM01RoughTrainerCfg",
    },
    framework_required="isaaclab",
)

register_task(
    task_id="RobotLab-Velocity-Flat-RoboParty-ATOM01-v0",
    env_cfg=f"{__name__}.flat_env_cfg:RoboPartyATOM01FlatEnvCfg",
    agent_cfg_entries={
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoboPartyATOM01FlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:RoboPartyATOM01FlatTrainerCfg",
    },
    framework_required="isaaclab",
)
