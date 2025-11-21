"""MyDog robot configuration (Fixed based on thunder_nohead.urdf analysis)."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

# 确保路径指向正确
MYDOG_URDF_PATH = f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/myrobots/mydog/urdf/thunder_nohead.urdf"
MYDOG_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=False,
        asset_path=MYDOG_URDF_PATH,
        activate_contact_sensors=True,
        # [关键修复] 显式添加 joint_drive 配置以通过验证
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55), 
        # 关节初始角度
        joint_pos={
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": -0.7,
            "FR_calf_joint": 1.4,
            "FR_foot_joint": 0.0,

            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.7,
            "FL_calf_joint": -1.4,
            "FL_foot_joint": 0.0,

            "RR_hip_joint": 0.0,
            "RR_thigh_joint": 0.7,
            "RR_calf_joint": -1.4,
            "RR_foot_joint": 0.0,

            "RL_hip_joint": 0.0,
            "RL_thigh_joint": -0.7,
            "RL_calf_joint": 1.4,
            "RL_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # 腿部：位置控制
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            stiffness=80.0,   
            damping=2.0,
            effort_limit_sim=100.0,
            velocity_limit_sim=20.0,
        ),
        # 轮子：速度控制
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_foot_joint"],
            stiffness=0.0,   
            damping=4.0,    
            effort_limit_sim=50.0,
            velocity_limit_sim=50.0,
        ),
    },
)