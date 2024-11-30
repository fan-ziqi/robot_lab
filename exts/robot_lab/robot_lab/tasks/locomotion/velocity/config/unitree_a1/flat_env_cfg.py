from omni.isaac.lab.utils import configclass

from .rough_env_cfg import UnitreeA1RoughEnvCfg


@configclass
class UnitreeA1FlatEnvCfg(UnitreeA1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.base_height_l2.weight = -2
        self.rewards.base_height_rough_l2.weight = 0
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeA1FlatEnvCfg":
            self.disable_zero_weight_rewards()
