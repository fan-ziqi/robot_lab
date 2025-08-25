# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Configuration for DDT robots.
Reference: https://github.com/DDTRobot
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg  # noqa: F401
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

##
# Configuration
##


"""Configuration of DDT TITA using DC motor.
"""

DDTROBOT_TITA_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/ddt/tita_description/urdf/tita.urdf",
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
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.3),
        joint_pos={
            "joint_left_leg_1": 0.0,
            "joint_right_leg_1": 0.0,
            "joint_left_leg_2": 0.8,
            "joint_right_leg_2": 0.8,
            "joint_left_leg_3": -1.5,
            "joint_right_leg_3": -1.5,
            "joint_left_leg_4": 0.0,
            "joint_right_leg_4": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip": DCMotorCfg(
            joint_names_expr=["joint_left_leg_1", "joint_right_leg_1"],
            effort_limit=60.0,
            saturation_effort=60.0,
            velocity_limit=25.0,
            stiffness=40.0,
            damping=1.0,
            friction=0.0,
        ),
        "thigh": DCMotorCfg(
            joint_names_expr=["joint_left_leg_2", "joint_right_leg_2"],
            effort_limit=60.0,
            saturation_effort=60.0,
            velocity_limit=25.0,
            stiffness=40.0,
            damping=1.0,
            friction=0.0,
        ),
        "calf": DCMotorCfg(
            joint_names_expr=["joint_left_leg_3", "joint_right_leg_3"],
            effort_limit=60.0,
            saturation_effort=60.0,
            velocity_limit=25.0,
            stiffness=40.0,
            damping=1.0,
            friction=0.0,
        ),
        "wheel": DCMotorCfg(
            joint_names_expr=["joint_left_leg_4", "joint_right_leg_4"],
            effort_limit=15.0,
            saturation_effort=15.0,
            velocity_limit=20.0,
            stiffness=0.0,
            damping=1.0,
            friction=0.0,
        ),
    },
)
"""Configuration of DDT TITA using DC motor.
"""
