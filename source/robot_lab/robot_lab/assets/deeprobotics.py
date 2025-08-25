# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

DEEPROBOTICS_LITE3_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/deeprobotics/lite3_description/urdf/lite3.urdf",
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
        pos=(0.0, 0.0, 0.35),
        joint_pos={
            ".*HipX_joint": 0.0,
            ".*HipY_joint": -0.8,
            ".*Knee_joint": 1.6,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "Hip": DCMotorCfg(
            joint_names_expr=[".*_Hip[X,Y]_joint"],
            effort_limit=24.0,
            saturation_effort=24.0,
            velocity_limit=26.2,
            stiffness=30.0,
            damping=0.5,
            friction=0.0,
        ),
        "Knee": DCMotorCfg(
            joint_names_expr=[".*_Knee_joint"],
            effort_limit=36.0,
            saturation_effort=36.0,
            velocity_limit=17.3,
            stiffness=30.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)

DEEPROBOTICS_M20_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/deeprobotics/m20_description/urdf/m20.urdf",
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
        pos=(0.0, 0.0, 0.52),
        joint_pos={
            ".*hipx_joint": 0.0,
            "f[l,r]_hipy_joint": -0.6,
            "h[l,r]_hipy_joint": 0.6,
            "f[l,r]_knee_joint": 1.0,
            "h[l,r]_knee_joint": -1.0,
            ".*wheel_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "joint": DCMotorCfg(
            joint_names_expr=[".*hipx_joint", ".*hipy_joint", ".*knee_joint"],
            effort_limit=76.4,
            saturation_effort=76.4,
            velocity_limit=22.4,
            stiffness=80.0,
            damping=2.0,
            friction=0.0,
        ),
        "wheel": DCMotorCfg(
            joint_names_expr=[".*_wheel_joint"],
            effort_limit=21.6,
            saturation_effort=21.6,
            velocity_limit=79.3,
            stiffness=0.0,
            damping=0.6,
            friction=0.0,
        ),
    },
)
