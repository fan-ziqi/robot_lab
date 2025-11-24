# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

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