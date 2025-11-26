# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers import SceneEntityCfg, TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from .rough_env_cfg import MyDogRoughEnvCfg


@configclass
class MyDogFlatEnvCfg(MyDogRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # 覆盖奖励
        
        self.rewards.base_height_l2.params["sensor_cfg"] = None
       
        
        # 强制平面地形
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        
        # 禁用高度扫描
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        
        # 禁用课程
        self.curriculum.terrain_levels = None

        # 删除权重为0的奖励
        if self.__class__.__name__ == "MyDogFlatEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class MyDogHandstandFlatEnvCfg(MyDogFlatEnvCfg):
    """专用于平地倒立训练的配置，启用手倒立奖励并关闭与行走相冲突的项。"""

    def __post_init__(self):
        super().__post_init__()

        # 仅在原地训练，关闭速度/朝向指令
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.heading_command = False

        # 移除与行走相关的奖励，避免与倒立目标冲突
        self.rewards.track_lin_vel_xy_exp.weight = 0
        self.rewards.track_ang_vel_z_exp.weight = 0
        self.rewards.feet_contact_without_cmd.weight = 0
        self.rewards.wheel_vel_penalty.weight = 0
        self.rewards.stand_still.weight = 0
        self.rewards.joint_pos_penalty.weight = 0
        self.rewards.upward.weight = 0
        self.rewards.contact_forces.weight = 0

        # 放宽 root 相关惩罚，重点关注倒立姿态
        self.rewards.lin_vel_z_l2.weight = -0.5
        self.rewards.ang_vel_xy_l2.weight = -0.01
        self.rewards.base_height_l2.weight = 0

        # ------------------------------Handstand Rewards------------------------------
        handstand_type = "back"  # 使用前腿支撑倒立
        if handstand_type == "front":
            air_foot_pattern = "F.*_foot"
            knee_patterns = ["F.*(thigh|calf).*"]
            target_gravity = [-1.0, 0.0, 0.0]
        else:
            air_foot_pattern = "R.*_foot"
            knee_patterns = ["R.*(thigh|calf).*"]
            target_gravity = [1.0, 0.0, 0.0]

        self.rewards.handstand_orientation_l2.weight = -1.0
        self.rewards.handstand_orientation_l2.params["target_gravity"] = target_gravity

        self.rewards.handstand_feet_height_exp.weight = 8.0
        self.rewards.handstand_feet_height_exp.params["asset_cfg"].body_names = [air_foot_pattern]
        self.rewards.handstand_feet_height_exp.params["target_height"] = 0.55

        self.rewards.handstand_feet_on_air.weight = 6.0
        self.rewards.handstand_feet_on_air.params["sensor_cfg"].body_names = [air_foot_pattern]
        self.rewards.handstand_feet_on_air.params["threshold"] = 5.0
        self.rewards.handstand_feet_on_air.params["knee_body_names"] = knee_patterns

        self.rewards.handstand_feet_air_time.weight = 4.0
        self.rewards.handstand_feet_air_time.params["sensor_cfg"].body_names = [air_foot_pattern]
        self.rewards.handstand_feet_air_time.params["threshold"] = 0.3
        self.rewards.handstand_feet_air_time.params["knee_body_names"] = knee_patterns
        self.rewards.handstand_feet_air_time.params["contact_force_threshold"] = 5.0

        # ------------------------------Events------------------------------
        # 关闭复位随机化，保持每次 episode 初始姿态一致
        #self.events.randomize_reset_base = None

        # ------------------------------Terminations------------------------------
        # self.terminations.bad_orientation = DoneTerm(
        #     func=mdp.bad_orientation,
        #     params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 0.8},
        # )

        # 删除权重为0的奖励
        if self.__class__.__name__ == "MyDogHandstandFlatEnvCfg":
            self.disable_zero_weight_rewards()
