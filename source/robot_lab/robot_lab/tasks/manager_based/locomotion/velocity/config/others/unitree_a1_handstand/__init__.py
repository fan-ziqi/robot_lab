# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from robot_lab.framework import register_task

from . import agents

##
# Register tasks for the active framework.
##

register_task(
    task_id="RobotLab-Velocity-Flat-HandStand-Unitree-A1-v0",
    env_cfg=f"{__name__}.flat_env_cfg:UnitreeA1HandStandFlatEnvCfg",
    agent_cfg_entries={
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1HandStandFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeA1HandStandFlatTrainerCfg",
    },
    framework_required="isaaclab",
)

register_task(
    task_id="RobotLab-Velocity-Rough-HandStand-Unitree-A1-v0",
    env_cfg=f"{__name__}.rough_env_cfg:UnitreeA1HandStandRoughEnvCfg",
    agent_cfg_entries={
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1HandStandRoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeA1HandStandRoughTrainerCfg",
    },
    framework_required="isaaclab",
)
