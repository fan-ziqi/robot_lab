# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from robot_lab.framework import register_task

from . import agents

##
# Register tasks for the active framework. Per spec §12, task IDs are unified
# to `RobotLab-Velocity-...-v0` (no `Isaac-` infix); the old form is not
# aliased. The mjlab side requires the env_cfg classes to evaluate under
# mjlab — for now, the existing IL-style nested-@configclass cfgs (which
# inherit ``LocomotionVelocityRoughEnvCfg``) only work on IsaacLab. So we
# register IL-only here. A separate mjlab-native env cfg is the Phase 7+
# follow-up.
##

register_task(
    task_id="RobotLab-Velocity-Flat-Unitree-A1-v0",
    env_cfg=f"{__name__}.flat_env_cfg:UnitreeA1FlatEnvCfg",
    agent_cfg_entries={
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1FlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeA1FlatTrainerCfg",
    },
    framework_required="isaaclab",
)

register_task(
    task_id="RobotLab-Velocity-Rough-Unitree-A1-v0",
    env_cfg=f"{__name__}.rough_env_cfg:UnitreeA1RoughEnvCfg",
    agent_cfg_entries={
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1RoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeA1RoughTrainerCfg",
    },
    framework_required="isaaclab",
)
