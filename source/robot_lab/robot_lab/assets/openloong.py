# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Configuration for loong robots."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

##
# Configuration
##


OPENLOONG_LOONG_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/openloong/loong_description/urdf/loong.urdf",
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
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
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
        "actuators": ImplicitActuatorCfg(
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
        ),
    },
)
"""Configuration for the loong Humanoid robot."""
