# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from robot_lab.framework import register_task

from . import agents

##
# Register tasks for the active framework. Per spec §12, task IDs are unified
# to `RobotLab-Velocity-...-v0` (no `Isaac-` infix).
##

register_task(
    task_id="RobotLab-Velocity-Flat-Anymal-D-v0",
    env_cfg=f"{__name__}.flat_env_cfg:AnymalDFlatEnvCfg",
    agent_cfg_entries={
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalDFlatPPORunnerCfg",
        "rsl_rl_recurrent_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalDFlatPPORunnerRecurrentCfg",
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:AnymalDFlatDistillationRunnerCfg"
        ),
        "rsl_rl_distillation_recurrent_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:AnymalDFlatDistillationRunnerRecurrentCfg"
        ),
        "rsl_rl_with_symmetry_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalDFlatPPORunnerWithSymmetryCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
    framework_required="isaaclab",
)

register_task(
    task_id="RobotLab-Velocity-Flat-Anymal-D-Play-v0",
    env_cfg=f"{__name__}.flat_env_cfg:AnymalDFlatEnvCfg_PLAY",
    agent_cfg_entries={
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalDFlatPPORunnerCfg",
        "rsl_rl_recurrent_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalDFlatPPORunnerRecurrentCfg",
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:AnymalDFlatDistillationRunnerCfg"
        ),
        "rsl_rl_distillation_recurrent_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:AnymalDFlatDistillationRunnerRecurrentCfg"
        ),
        "rsl_rl_with_symmetry_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalDFlatPPORunnerWithSymmetryCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
    framework_required="isaaclab",
)
