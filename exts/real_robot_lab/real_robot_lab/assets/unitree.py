"""Configuration for Unitree robots.

The following configurations are available:

* :obj:`UNITREE_A1_CFG`: Unitree A1 robot with DC motor model for the legs
* :obj:`G1_CFG`: G1 humanoid robot

Reference: https://github.com/unitreerobotics/unitree_ros
"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import DCMotorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from real_robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR


##
# Configuration
##


UNITREE_A1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/Unitree/A1/a1.usd",
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
        pos=(0.0, 0.0, 0.42),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=33.5,
            saturation_effort=33.5,
            velocity_limit=21.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)
"""Configuration of Unitree A1 using DC motor.

Note: Specifications taken from: https://www.trossenrobotics.com/a1-quadruped#specifications
"""

# G1_CFG = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/Unitree/G1/g1.usd",
#         activate_contact_sensors=True,
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             disable_gravity=False,
#             retain_accelerations=False,
#             linear_damping=0.0,
#             angular_damping=0.0,
#             max_linear_velocity=1000.0,
#             max_angular_velocity=1000.0,
#             max_depenetration_velocity=1.0,
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
#         ),
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 0.74),
#         joint_pos={
#             ".*_hip_pitch_joint": -0.20,
#             ".*_knee_joint": 0.42,
#             ".*_ankle_pitch_joint": -0.23,
#             ".*_elbow_pitch_joint": 0.87,
#             "left_shoulder_roll_joint": 0.16,
#             "left_shoulder_pitch_joint": 0.35,
#             "right_shoulder_roll_joint": -0.16,
#             "right_shoulder_pitch_joint": 0.35,
#             "left_one_joint": 1.0,
#             "right_one_joint": -1.0,
#             "left_two_joint": 0.52,
#             "right_two_joint": -0.52,
#         },
#         joint_vel={".*": 0.0},
#     ),
#     soft_joint_pos_limit_factor=0.9,
#     actuators={
#         "legs": ImplicitActuatorCfg(
#             joint_names_expr=[
#                 ".*_hip_yaw_joint",
#                 ".*_hip_roll_joint",
#                 ".*_hip_pitch_joint",
#                 ".*_knee_joint",
#                 "torso_joint",
#             ],
#             effort_limit=300,
#             velocity_limit=100.0,
#             stiffness={
#                 ".*_hip_yaw_joint": 150.0,
#                 ".*_hip_roll_joint": 150.0,
#                 ".*_hip_pitch_joint": 200.0,
#                 ".*_knee_joint": 200.0,
#                 "torso_joint": 200.0,
#             },
#             damping={
#                 ".*_hip_yaw_joint": 5.0,
#                 ".*_hip_roll_joint": 5.0,
#                 ".*_hip_pitch_joint": 5.0,
#                 ".*_knee_joint": 5.0,
#                 "torso_joint": 5.0,
#             },
#             armature={
#                 ".*_hip_.*": 0.01,
#                 ".*_knee_joint": 0.01,
#                 "torso_joint": 0.01,
#             },
#         ),
#         "feet": ImplicitActuatorCfg(
#             effort_limit=20,
#             joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
#             stiffness=20.0,
#             damping=2.0,
#             armature=0.01,
#         ),
#         "arms": ImplicitActuatorCfg(
#             joint_names_expr=[
#                 ".*_shoulder_pitch_joint",
#                 ".*_shoulder_roll_joint",
#                 ".*_shoulder_yaw_joint",
#                 ".*_elbow_pitch_joint",
#                 ".*_elbow_roll_joint",
#                 ".*_five_joint",
#                 ".*_three_joint",
#                 ".*_six_joint",
#                 ".*_four_joint",
#                 ".*_zero_joint",
#                 ".*_one_joint",
#                 ".*_two_joint",
#             ],
#             effort_limit=300,
#             velocity_limit=100.0,
#             stiffness=40.0,
#             damping=10.0,
#             armature={
#                 ".*_shoulder_.*": 0.01,
#                 ".*_elbow_.*": 0.01,
#                 ".*_five_joint": 0.001,
#                 ".*_three_joint": 0.001,
#                 ".*_six_joint": 0.001,
#                 ".*_four_joint": 0.001,
#                 ".*_zero_joint": 0.001,
#                 ".*_one_joint": 0.001,
#                 ".*_two_joint": 0.001,
#             },
#         ),
#     },
# )
# """Configuration for the Unitree G1 Humanoid robot."""


# G1_MINIMAL_CFG = G1_CFG.copy()
# G1_MINIMAL_CFG.spawn.usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1_minimal.usd"
# """Configuration for the Unitree G1 Humanoid robot with fewer collision meshes.

# This configuration removes most collision meshes to speed up simulation.
# """
