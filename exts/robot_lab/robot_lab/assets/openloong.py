"""Configuration for OPENLOONG robots.

The following configurations are available:

* :obj:`OPENLOONG_OPENLOONG_CFG`: OPENLOONG OPENLOONG humanoid robot

Reference: https://www.openloong.org.cn/cn
"""

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

##
# Configuration
##


OPENLOONG_OPENLOONG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/OpenLoong/OpenLoong.usd",
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
        pos=(0.0, 0.0, 1.09),
        joint_pos={
            #left_leg
           'l_hip_roll' : 0.,  
           'l_hip_yaw' : 0. ,   
           'l_hip_pitch' : 0.,         
           'l_knee_pitch' : 0.0,       
           'l_ankle_pitch': 0.,
           'l_ankle_roll': 0.,  
           #right_leg  
           'r_hip_roll' : 0., 
           'r_hip_yaw' : 0., 
           'r_hip_pitch' : 0.,                                       
           'r_knee_pitch' : 0.0,                                             
           'r_ankle_pitch': 0.,
           'r_ankle_roll': 0.,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "actuators": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness={
                '.*_hip_roll': 251.625,
                '.*_hip_pitch': 200,
                '.*_hip_yaw': 362.5214,
                '.*_knee_pitch': 200,
                '.*_ankle_pitch': 10.9805,
                '.*_ankle_roll': 10.9805,
            },
            damping={
                '.*_hip_roll': 10,
                '.*_hip_pitch': 10,
                '.*_hip_yaw': 10,
                '.*_knee_pitch':10,
                '.*_ankle_pitch': 10,
                '.*_ankle_roll': 10,                   
            },
        ),
    },
)
"""Configuration for the OPENLOONG OPENLOONG Humanoid robot."""
