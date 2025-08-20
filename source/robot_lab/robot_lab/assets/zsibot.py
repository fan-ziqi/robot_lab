# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg  # noqa: F401
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR
from robot_lab.assets.utils.usd_converter import (  # noqa: F401
    mjcf_to_usd,
    spawn_from_lazy_usd,
    urdf_to_usd,
    xacro_to_usd,
)

##
# Configuration
##

ZSIBOT_ZSL1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        func=spawn_from_lazy_usd,
        usd_path=urdf_to_usd(  # type: ignore
            file_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/zsibot/zsl1_description/urdf/zsl1.urdf",
            output_usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/zsibot/zsl1_description/usd/zsl1.usd",
            merge_joints=True,
            fix_base=False,
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*_ABAD_JOINT": 0.0,
            ".*_HIP_JOINT": 0.8,
            ".*_KNEE_JOINT": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[".*_ABAD_JOINT", ".*_HIP_JOINT", ".*_KNEE_JOINT"],
            effort_limit=28,
            velocity_limit=28,
            stiffness=20.0,
            damping=0.7,
            friction=0.0,
            min_delay=0,  # physics time steps (min: 5.0*0=00.0ms)
            max_delay=5,  # physics time steps (max: 5.0*5=25.0ms)
        ),
    },
)

ZSIBOT_ZSL1W_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        func=spawn_from_lazy_usd,
        usd_path=urdf_to_usd(  # type: ignore
            file_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/zsibot/zsl1w_description/urdf/zsl1w.urdf",
            output_usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/zsibot/zsl1w_description/usd/zsl1w.usd",
            merge_joints=True,
            fix_base=False,
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*_ABAD_JOINT": 0.0,
            ".*_HIP_JOINT": 0.8,
            ".*_KNEE_JOINT": -1.5,
            ".*_FOOT_JOINT": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[".*_ABAD_JOINT", ".*_HIP_JOINT", ".*_KNEE_JOINT"],
            effort_limit=28,
            velocity_limit=28,
            stiffness=20.0,
            damping=0.7,
            friction=0.0,
            min_delay=0,  # physics time steps (min: 5.0*0=00.0ms)
            max_delay=5,  # physics time steps (max: 5.0*5=25.0ms)
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_FOOT_JOINT"],
            effort_limit_sim=28,
            velocity_limit_sim=28,
            stiffness=0.0,
            damping=0.7,
            friction=0.0,
        ),
    },
)
"""Configuration of zsibot."""
