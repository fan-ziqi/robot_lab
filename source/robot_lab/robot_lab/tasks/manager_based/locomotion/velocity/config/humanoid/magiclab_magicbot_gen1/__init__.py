# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from robot_lab.framework import register_task

from . import agents

##
# Register tasks for the active framework.
##

register_task(
    task_id="RobotLab-Velocity-Rough-MagicLab-Bot-Gen1-v0",
    env_cfg=f"{__name__}.rough_env_cfg:MagicLabBotGen1RoughEnvCfg",
    agent_cfg_entries={
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MagicLabBotGen1RoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:MagicLabBotGen1RoughTrainerCfg",
    },
    framework_required="isaaclab",
)

register_task(
    task_id="RobotLab-Velocity-Flat-MagicLab-Bot-Gen1-v0",
    env_cfg=f"{__name__}.flat_env_cfg:MagicLabBotGen1FlatEnvCfg",
    agent_cfg_entries={
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MagicLabBotGen1FlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:MagicLabBotGen1FlatTrainerCfg",
    },
    framework_required="isaaclab",
)
