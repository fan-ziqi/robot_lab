from omni.isaac.lab.utils import configclass

from .rough_env_cfg import UnitreeA1AmpRoughEnvCfg


@configclass
class UnitreeA1AmpFlatEnvCfg(UnitreeA1AmpRoughEnvCfg):
    def __post_init__(self):
        # Temporarily not run disable_zerow_eight_rewards() in parent class to override rewards
        self._run_disable_zero_weight_rewards = False
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # Now executing disable_zerow_eight_rewards()
        self._run_disable_zero_weight_rewards = True
        if self._run_disable_zero_weight_rewards:
            self.disable_zero_weight_rewards()
