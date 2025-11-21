"""MyDog robot configuration (Fixed based on thunder_nohead.urdf analysis)."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from robot_lab.robot_lab_data import ROBOT_LAB_DATA_DIR 

# 确保路径指向正确
MYDOG_URDF_PATH = f"{ROBOT_LAB_DATA_DIR}/Robots/mydog/mydog_description/urdf/thunder_nohead.urdf"

MYDOG_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=False,
        asset_path=MYDOG_URDF_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            # 适当增加阻尼以提高稳定性
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
        pos=(0.0, 0.0, 0.55), # 稍微调高一点，避免出生时陷入地面
        # --- 根据 URDF 限位修正后的关节角度 ---
        joint_pos={
            # FR (右前): Thigh限位[-3.14, 0], Calf限位[0, 3.14]
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": -0.7,  # 修改为负值
            "FR_calf_joint": 1.4,    # 修改为正值
            "FR_foot_joint": 0.0,

            # FL (左前): Thigh限位[0, 3.14], Calf限位[-3.14, 0]
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.7,   # 正值 (保持不变)
            "FL_calf_joint": -1.4,   # 负值 (保持不变)
            "FL_foot_joint": 0.0,

            # RR (右后): Thigh限位[0, 3.14], Calf限位[-3.14, 3.14]
            # 注意：RR和FR的大腿限位在URDF中定义方向相反，需顺应URDF
            "RR_hip_joint": 0.0,
            "RR_thigh_joint": 0.7,   # 正值
            "RR_calf_joint": -1.4,   # 负值 (通常后腿向后弯)
            "RR_foot_joint": 0.0,

            # RL (左后): Thigh限位[-3.14, 0], Calf限位[-3.14, 3.14]
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": -0.7,  # 修改为负值
            "RL_calf_joint": 1.4,    # 正值
            "RL_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # 普通关节驱动 (Hips, Thighs, Calves)
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            # 修正刚度：根据机器人约40kg的重量设置
            stiffness=80.0,   
            damping=2.0,
            effort_limit_sim=100.0,
            velocity_limit_sim=20.0,
        ),
        # 如果你的脚是轮子并且需要驱动，取消下面的注释：
        # "wheels": ImplicitActuatorCfg(
        #     joint_names_expr=[".*_foot_joint"],
        #     stiffness=0.0, # 轮子通常是速度控制，位置刚度设为0
        #     damping=5.0,   # 提供一点阻尼
        #     velocity_limit_sim=20.0,
        # ),
    },
)