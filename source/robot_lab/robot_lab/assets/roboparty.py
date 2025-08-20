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


ATOM01_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        func=spawn_from_lazy_usd,
        usd_path=urdf_to_usd(  # type: ignore
            file_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/roboparty/atom01_description/urdf/atom01.urdf",
            output_usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/roboparty/atom01_description/usd/atom01.usd",
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
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.7),
        joint_pos={
            "left_thigh_pitch_joint": -0.2,
            "left_knee_joint": 0.4,
            "left_ankle_pitch_joint": -0.2,
            "left_arm_pitch_joint": 0.1,
            "left_arm_roll_joint": 0.07,
            "left_elbow_pitch_joint": 1.0,
            "right_thigh_pitch_joint": -0.2,
            "right_knee_joint": 0.4,
            "right_ankle_pitch_joint": -0.2,
            "right_arm_pitch_joint": 0.1,
            "right_arm_roll_joint": -0.07,
            "right_elbow_pitch_joint": 1.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_thigh_yaw_joint",
                ".*_thigh_roll_joint",
                ".*_thigh_pitch_joint",
                ".*_knee_joint",
                ".*torso.*",
            ],
            stiffness={
                ".*_thigh_yaw_joint": 100.0,
                ".*_thigh_roll_joint": 100.0,
                ".*_thigh_pitch_joint": 100.0,
                ".*_knee_joint": 150.0,
                ".*torso.*": 150.0,
            },
            damping={
                ".*_thigh_yaw_joint": 3.0,
                ".*_thigh_roll_joint": 3.0,
                ".*_thigh_pitch_joint": 3.0,
                ".*_knee_joint": 5.0,
                ".*torso.*": 5.0,
            },
            armature=0.01,
            min_delay=0,  # physics time steps (min: 5.0*0=00.0ms)
            max_delay=5,  # physics time steps (max: 5.0*5=25.0ms)
        ),
        "feet": IdealPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=40.0,
            damping=1.5,
            armature=0.01,
            min_delay=0,  # physics time steps (min: 5.0*0=00.0ms)
            max_delay=5,  # physics time steps (max: 5.0*5=25.0ms)
        ),
        "shoulders": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_arm_pitch_joint",
                ".*_arm_roll_joint",
                ".*_arm_yaw_joint",
            ],
            stiffness=60.0,
            damping=2.0,
            armature=0.01,
            min_delay=0,  # physics time steps (min: 5.0*0=00.0ms)
            max_delay=5,  # physics time steps (max: 5.0*5=25.0ms)
        ),
        "arms": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_elbow_pitch_joint",
                ".*_elbow_yaw_joint",
            ],
            stiffness={
                ".*_elbow_pitch_joint": 40.0,
                ".*_elbow_yaw_joint": 20.0,
            },
            damping={
                ".*_elbow_pitch_joint": 1.5,
                ".*_elbow_yaw_joint": 1.0,
            },
            armature=0.01,
            min_delay=0,  # physics time steps (min: 5.0*0=00.0ms)
            max_delay=5,  # physics time steps (max: 5.0*5=25.0ms)
        ),
    },
)
