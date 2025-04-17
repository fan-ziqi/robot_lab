# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

ROBOTERA_XBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/RobotEra/Xbot/xbot.usd",
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
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.95),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_leg_roll_joint",
                ".*_leg_yaw_joint",
                ".*_leg_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit={
                ".*_leg_roll_joint": 100,
                ".*_leg_yaw_joint": 100,
                ".*_leg_pitch_joint": 250,
                ".*_knee_joint": 250,
            },
            velocity_limit=12,
            stiffness={
                ".*_leg_roll_joint": 200,
                ".*_leg_yaw_joint": 200,
                ".*_leg_pitch_joint": 350,
                ".*_knee_joint": 350,
            },
            damping=10.0,
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit=100,
            velocity_limit=12.0,
            stiffness=15.0,
            damping=10.0,
            armature=0.01,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_.*"],
            effort_limit=100,
            velocity_limit=12.0,
            stiffness=200.0,
            damping=10.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_arm_yaw_joint",
                ".*_elbow_pitch_joint",
                ".*_elbow_yaw_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit={
                ".*_shoulder_pitch_joint": 80,
                ".*_shoulder_roll_joint": 80,
                ".*_arm_yaw_joint": 50,
                ".*_elbow_pitch_joint": 50,
                ".*_elbow_yaw_joint": 50,
                ".*_wrist_roll_joint": 50,
                ".*_wrist_yaw_joint": 50,
            },
            velocity_limit=7.0,
            stiffness=100.0,
            damping=10.0,
            armature=0.01,
        ),
    },
)
"""Configuration for the RobotEra Xbot Humanoid robot."""
