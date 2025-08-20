# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Configuration for loong robots.
"""

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


OPENLOONG_LOONG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        func=spawn_from_lazy_usd,
        usd_path=urdf_to_usd(  # type: ignore
            file_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/openloong/loong_description/urdf/loong.urdf",
            output_usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/openloong/loong_description/usd/loong.usd",
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
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.2),
        joint_pos={
            # left leg
            "J_hip_l_roll": 0.0,
            "J_hip_l_yaw": 0.0,
            "J_hip_l_pitch": 0.2,
            "J_knee_l_pitch": -0.5,
            "J_ankle_l_pitch": 0.3,
            "J_ankle_l_roll": 0.0,
            # right leg
            "J_hip_r_roll": -0.0,
            "J_hip_r_yaw": 0.0,
            "J_hip_r_pitch": 0.2,
            "J_knee_r_pitch": -0.5,
            "J_ankle_r_pitch": 0.3,
            "J_ankle_r_roll": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "actuators": IdealPDActuatorCfg(
            joint_names_expr=[".*"],
            stiffness={
                "J_hip_.*_roll": 400.0,
                "J_hip_.*_yaw": 200.0,
                "J_hip_.*_pitch": 400.0,
                "J_knee_.*_pitch": 400.0,
                "J_ankle_.*_pitch": 120.0,
                "J_ankle_.*_roll": 120.0,
            },
            damping={
                "J_hip_.*_roll": 2.0,
                "J_hip_.*_yaw": 2.0,
                "J_hip_.*_pitch": 2.0,
                "J_knee_.*_pitch": 4.0,
                "J_ankle_.*_pitch": 0.5,
                "J_ankle_.*_roll": 0.5,
            },
            min_delay=0,  # physics time steps (min: 5.0*0=00.0ms)
            max_delay=5,  # physics time steps (max: 5.0*5=25.0ms)
        ),
    },
)
"""Configuration for the loong Humanoid robot."""
