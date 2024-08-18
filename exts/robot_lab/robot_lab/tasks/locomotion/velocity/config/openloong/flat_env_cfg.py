from omni.isaac.lab.utils import configclass

from .rough_env_cfg import openloong_12_RoughEnvCfg


@configclass
class openloong_12_FlatEnvCfg(openloong_12_RoughEnvCfg):
    def __post_init__(self):
        # Temporarily not run disable_zerow_eight_rewards() in parent class to override rewards
        self._run_disable_zero_weight_rewards = False
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.5
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


class openloong_12_FlatEnvCfg_PLAY(openloong_12_FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
