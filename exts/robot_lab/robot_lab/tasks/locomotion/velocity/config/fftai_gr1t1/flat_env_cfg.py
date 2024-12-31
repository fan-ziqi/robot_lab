from omni.isaac.lab.utils import configclass

from .rough_env_cfg import FFTAIGR1T1RoughEnvCfg


@configclass
class FFTAIGR1T1FlatEnvCfg(FFTAIGR1T1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.8
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "FFTAIGR1T1FlatEnvCfg":
            self.disable_zero_weight_rewards()
