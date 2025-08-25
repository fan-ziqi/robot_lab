# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Configuration for magiclab robots.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

##
# Configuration
##

MAGICLAB_BOT_GEN1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=False,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/magiclab/magicbot-Gen1/urdf/MAGICBOT.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=3.0,
            max_angular_velocity=3.0,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.92),
        joint_pos={
            "JOINT_HIP_ROLL_.*": 0.0,
            "JOINT_HIP_YAW_.*": 0.0,
            "JOINT_HIP_PITCH_.*": -0.4,
            "JOINT_KNEE_PITCH_.*": 0.8,
            "JOINT_ANKLE_PITCH_.*": -0.45,
            "JOINT_ANKLE_ROLL_.*": 0.0,
            "joint_.*a1": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "JOINT_HIP_ROLL_.*",
                "JOINT_HIP_YAW_.*",
                "JOINT_HIP_PITCH_.*",
                "JOINT_KNEE_PITCH_.*",
            ],
            effort_limit_sim=300,
            velocity_limit_sim=100.0,
            stiffness={
                "JOINT_HIP_PITCH_.*": 200.0,
                "JOINT_HIP_ROLL_.*": 150.0,
                "JOINT_HIP_YAW_.*": 150.0,
                "JOINT_KNEE_PITCH_.*": 200.0,
            },
            damping={
                "JOINT_HIP_PITCH_.*": 5.0,
                "JOINT_HIP_ROLL_.*": 5.0,
                "JOINT_HIP_YAW_.*": 5.0,
                "JOINT_KNEE_PITCH_.*": 5.0,
            },
            armature={
                "JOINT_HIP_.*": 0.01,
                "JOINT_KNEE_.*": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=20,
            joint_names_expr=["JOINT_ANKLE_PITCH_.*", "JOINT_ANKLE_ROLL_.*"],
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "joint_.*a1",
            ],
            effort_limit_sim=300,
            velocity_limit_sim=100.0,
            stiffness=40.0,
            damping=10.0,
            armature={
                "joint_.*a1": 0.01,
            },
        ),
    },
)


MAGICLAB_BOT_Z1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=False,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/magiclab/magicbot-Z1/urdf/MagicBotZ1.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=3.0,
            max_angular_velocity=3.0,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            "JOINT_HIP_ROLL_.*": 0.0,
            "JOINT_HIP_YAW_.*": 0.0,
            "JOINT_HIP_PITCH_.*": -0.35,
            "JOINT_KNEE_PITCH_.*": 0.7,
            "JOINT_ANKLE_PITCH_.*": -0.35,
            "JOINT_ANKLE_ROLL_.*": 0.0,
            "joint_.*a1": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "JOINT_HIP_ROLL_.*",
                "JOINT_HIP_YAW_.*",
                "JOINT_HIP_PITCH_.*",
                "JOINT_KNEE_PITCH_.*",
            ],
            effort_limit_sim=300,
            velocity_limit_sim=100.0,
            stiffness={
                "JOINT_HIP_PITCH_.*": 200.0,
                "JOINT_HIP_ROLL_.*": 150.0,
                "JOINT_HIP_YAW_.*": 150.0,
                "JOINT_KNEE_PITCH_.*": 200.0,
            },
            damping={
                "JOINT_HIP_PITCH_.*": 5.0,
                "JOINT_HIP_ROLL_.*": 5.0,
                "JOINT_HIP_YAW_.*": 5.0,
                "JOINT_KNEE_PITCH_.*": 5.0,
            },
            armature={
                "JOINT_HIP_.*": 0.01,
                "JOINT_KNEE_.*": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=20,
            joint_names_expr=["JOINT_ANKLE_PITCH_.*", "JOINT_ANKLE_ROLL_.*"],
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "joint_.*a1",
            ],
            effort_limit_sim=300,
            velocity_limit_sim=100.0,
            stiffness=40.0,
            damping=10.0,
            armature={
                "joint_.*a1": 0.01,
            },
        ),
    },
)


MAGICDOG_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=False,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/magiclab/magicdog/urdf/magicdog.urdf",
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
        pos=(0.0, 0.0, 0.5),
        joint_pos={
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": 0.0,
            "F.*_thigh_joint": 0.6683,
            "R.*_thigh_joint": 0.6683,
            ".*_calf_joint": -1.312,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[".*"],
            effort_limit=25.0,
            saturation_effort=25.0,
            velocity_limit=22.0,
            stiffness=30.0,
            damping=1.0,
            friction=0.0,
        ),
    },
)


MAGICDOG_W_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/magiclab/magicdog_w/urdf/magicdog_w.urdf",
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
        pos=(0.0, 0.0, 0.5),
        joint_pos={
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": 0.0,
            "F.*_thigh_joint": 1.0,
            "R.*_thigh_joint": 1.0,
            ".*_calf_joint": -1.8,
            ".*_wheel_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=["^(?!.*_foot_joint).*"],
            effort_limit=37.5,
            saturation_effort=37.5,
            velocity_limit=15.0,
            stiffness=30.0,
            damping=1.0,
            friction=0.0,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            effort_limit_sim=15.0,
            velocity_limit_sim=35.0,
            stiffness=0.0,
            damping=0.2,
            friction=0.0,
        ),
    },
)
