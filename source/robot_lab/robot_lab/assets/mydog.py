"""
MyDog robot configuration.

本文件定义了用于 IsaacLab/Isaac Sim 的 `ArticulationCfg`，用于在仿真中实例化
“MyDog / thunder_nohead” 机器人。配置内容包括：URDF 路径、初始位姿、关节初始角、
物理属性和 actuator（驱动器）分组。注释中说明了每个字段的作用，便于在训练或评估
时对齐观测与动作。

使用说明：上层环境或任务代码会引用 `MYDOG_CFG` 来 spawn 机器人；如果你要改变
哪些关节被控制（例如去掉 foot 关节），请修改 `actuators` 中的 `joint_names_expr`。
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR


# URDF 文件路径（使用扩展的资产目录常量拼接）
MYDOG_URDF_PATH = f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/myrobots/mydog/urdf/thunder_nohead.urdf"


# ArticulationCfg: 描述如何在仿真中实例化和驱动该机器人
MYDOG_CFG = ArticulationCfg(
    # spawn: 指定 URDF 以及转换参数
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,  # False 表示机器人有浮动基座（可整体移动）
        merge_fixed_joints=True,  # 合并固定关节以提高仿真性能
        asset_path=MYDOG_URDF_PATH,
        activate_contact_sensors=True,  # 启用接触传感器（常用于接地检测等）
        # joint_drive: 明确指定转换时的驱动类型和 PD 增益（帮助通过验证）
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
        # 刚体属性：阻尼/速度上限等，影响仿真稳定性与行为
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        # articulation 根节点的仿真求解器参数
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),

    # init_state: 初始位姿与关节状态
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),  # 机器人在世界坐标的初始位置 (x, y, z)
        # joint_pos: 明确为每个关节设置初始角度（也可使用正则表达式匹配）
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
        joint_vel={".*": 0.0},  # 所有关节速度初始为 0
    ),

    # 系统级的软限位因子（用于限制关节移动范围的软边界）
    soft_joint_pos_limit_factor=0.9,

    # actuators: 定义哪些关节由哪些 actuator 组控制，以及各组的参数
    actuators={
        # legs: 控制腿部的 hip/thigh/calf 三个关节（位置/力矩驱动）
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            # stiffness/damping 为隐式驱动的 PD 特性（较高 stiffness 意味着更刚的伺服行为）
            stiffness=25.0,
            damping=0.5,
            effort_limit_sim=23.5,  # 参考GO2W，避免过大力矩导致不稳定
            velocity_limit_sim=30.0,  # 提高速度限制以获得更灵活的运动
        ),
        # wheels: 将 foot_joint 视为轮子进行速度驱动
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_foot_joint"],
            stiffness=0.0,  # 轮子不需要高刚度
            damping=0.5,
            effort_limit_sim=20.0,  # 参考GO2W，轮子力矩限制
            velocity_limit_sim=30.0,  # 参考GO2W，适当的轮子转速
        ),
    },
)