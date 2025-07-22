# Copyright (c) 2024-2025 Yankai Xiang
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .rough_env_cfg import DDTTitaRoughEnvCfg
import isaaclab.terrains as terrain_gen

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
    },
)
@configclass
class DDTTitaFlatEnvCfg(DDTTitaRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        # change terrain to flat
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = COBBLESTONE_ROAD_CFG
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "DDTTitaFlatEnvCfg":
            self.disable_zero_weight_rewards()
