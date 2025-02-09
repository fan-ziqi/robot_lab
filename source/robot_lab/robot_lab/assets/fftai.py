# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Configuration for FFTAI robots.

The following configurations are available:

* :obj:`FFTAI_GR1T1_CFG`: FFTAI GR1T1 humanoid robot

Reference: https://github.com/FFTAI
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

##
# Configuration
##


FFTAI_GR1T1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/FFTAI/GR1T1/GR1T1.usd",
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
        pos=(0.0, 0.0, 0.93),
        joint_pos={
            # left leg
            "l_hip_roll": 0.0,
            "l_hip_yaw": 0.0,
            "l_hip_pitch": -0.2618,
            "l_knee_pitch": 0.5236,
            "l_ankle_pitch": -0.2618,
            "l_ankle_roll": 0.0,
            # right leg
            "r_hip_roll": -0.0,
            "r_hip_yaw": 0.0,
            "r_hip_pitch": -0.2618,
            "r_knee_pitch": 0.5236,
            "r_ankle_pitch": -0.2618,
            "r_ankle_roll": 0.0,
            # waist
            "waist_yaw": 0.0,
            "waist_pitch": 0.0,
            "waist_roll": 0.0,
            # head
            "head_yaw": 0.0,
            "head_pitch": 0.0,
            "head_roll": 0.0,
            # left arm
            "l_shoulder_pitch": 0.0,
            "l_shoulder_roll": 0.2,
            "l_shoulder_yaw": 0.0,
            "l_elbow_pitch": -0.3,
            "l_wrist_yaw": 0.0,
            "l_wrist_roll": 0.0,
            "l_wrist_pitch": 0.0,
            # right arm
            "r_shoulder_pitch": 0.0,
            "r_shoulder_roll": -0.2,
            "r_shoulder_yaw": 0.0,
            "r_elbow_pitch": -0.3,
            "r_wrist_yaw": 0.0,
            "r_wrist_roll": 0.0,
            "r_wrist_pitch": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "actuators": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness={
                ".*_hip_roll": 251.625,
                ".*_hip_yaw": 362.5214,
                ".*_hip_pitch": 200,
                ".*_knee_pitch": 200,
                ".*_ankle_pitch": 10.9805,
                ".*_ankle_roll": 0.25,
                "waist_yaw": 362.5214,
                "waist_pitch": 362.5214,
                "waist_roll": 362.5214,
                "head_yaw": 10.0,
                "head_pitch": 10.0,
                "head_roll": 10.0,
                ".*_shoulder_pitch": 92.85,
                ".*_shoulder_roll": 92.85,
                ".*_shoulder_yaw": 112.06,
                ".*_elbow_pitch": 112.06,
                ".*_wrist_yaw": 10.0,
                ".*_wrist_roll": 10.0,
                ".*_wrist_pitch": 10.0,
            },
            damping={
                ".*_hip_roll": 14.72,
                ".*_hip_yaw": 10.0833,
                ".*_hip_pitch": 11,
                ".*_knee_pitch": 11,
                ".*_ankle_pitch": 0.5991,
                ".*_ankle_roll": 0.01,
                "waist_yaw": 10.0833,
                "waist_pitch": 10.0833,
                "waist_roll": 10.0833,
                "head_yaw": 1.0,
                "head_pitch": 1.0,
                "head_roll": 1.0,
                ".*_shoulder_pitch": 2.575,
                ".*_shoulder_roll": 2.575,
                ".*_shoulder_yaw": 3.1,
                ".*_elbow_pitch": 3.1,
                ".*_wrist_yaw": 1.0,
                ".*_wrist_roll": 1.0,
                ".*_wrist_pitch": 1.0,
            },
        ),
    },
)
"""Configuration for the FFTAI GR1T1 Humanoid robot."""


FFTAI_GR1T1_LOWER_LIMB_CFG = FFTAI_GR1T1_CFG.copy()
FFTAI_GR1T1_LOWER_LIMB_CFG.spawn.usd_path = f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/FFTAI/GR1T1/GR1T1_lower_limb.usd"
FFTAI_GR1T1_LOWER_LIMB_CFG.actuators = (
    {
        "actuators": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness={
                ".*_hip_roll": 114,
                ".*_hip_yaw": 86,
                ".*_hip_pitch": 229,
                ".*_knee_pitch": 229,
                ".*_ankle_pitch": 30.5,
            },
            damping={
                ".*_hip_roll": 114 / 15,
                ".*_hip_yaw": 86 / 15,
                ".*_hip_pitch": 229 / 15,
                ".*_knee_pitch": 229 / 15,
                ".*_ankle_pitch": 30.5 / 15,
            },
        ),
    },
)
"""Configuration for the FFTAI GR1T1 Humanoid robot with fixed upper limb."""


FFTAI_GR1T2_CFG = FFTAI_GR1T1_CFG.copy()
FFTAI_GR1T2_CFG.spawn.usd_path = f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/FFTAI/GR1T2/GR1T2.usd"
"""Configuration for the FFTAI GR1T1 Humanoid robot."""


FFTAI_GR1T2_LOWER_LIMB_CFG = FFTAI_GR1T1_LOWER_LIMB_CFG.copy()
FFTAI_GR1T2_LOWER_LIMB_CFG.spawn.usd_path = f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/FFTAI/GR1T2/GR1T2_lower_limb.usd"
"""Configuration for the FFTAI GR1T2 Humanoid robot with fixed upper limb."""
