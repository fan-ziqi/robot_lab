# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from robot_lab.framework import register_task

from . import agents

##
# Register tasks. framework_required="isaaclab" because the env_cfg classes
# inherit IL-style nested @configclass; mjlab-native cfg lands in a follow-up.
# env_cfg is a "module:Class" string so the cfg module is loaded lazily by
# IsaacLab's task registry — under mjlab the registration is skipped before
# any IL-only imports fire.
##

register_task(
    task_id="RobotLab-BeyondMimic-Flat-Unitree-G1-v0",
    env_cfg=f"{__name__}.flat_env_cfg:UnitreeG1BeyondMimicFlatEnvCfg",
    agent_cfg_entries={
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1BeyondMimicFlatPPORunnerCfg",
    },
    framework_required="isaaclab",
)
