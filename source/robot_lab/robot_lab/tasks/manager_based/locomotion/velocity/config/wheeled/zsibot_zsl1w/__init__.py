# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from robot_lab.framework import register_task

from . import agents

##
# Register tasks for the active framework.
##

register_task(
    task_id="RobotLab-Velocity-Flat-Zsibot-ZSL1W-v0",
    env_cfg=f"{__name__}.flat_env_cfg:ZsibotZSL1WFlatEnvCfg",
    agent_cfg_entries={
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ZsibotZSL1WFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:ZsibotZSL1WFlatTrainerCfg",
    },
    framework_required="isaaclab",
)

register_task(
    task_id="RobotLab-Velocity-Rough-Zsibot-ZSL1W-v0",
    env_cfg=f"{__name__}.rough_env_cfg:ZsibotZSL1WRoughEnvCfg",
    agent_cfg_entries={
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ZsibotZSL1WRoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:ZsibotZSL1WRoughTrainerCfg",
    },
    framework_required="isaaclab",
)
