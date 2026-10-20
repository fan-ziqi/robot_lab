# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .rough_env_cfg import AnymalDRoughEnvCfg


@configclass
class AnymalDFlatEnvCfg(AnymalDRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "AnymalDFlatEnvCfg":
            self.disable_zero_weight_rewards()

        # Only the independently validated flat class opts into Newton.
        if type(self) is AnymalDFlatEnvCfg:
            from robot_lab.tasks.manager_based.locomotion.velocity.newton_support import (
                configure_newton,
                load_order_profile,
            )

            joint_names, body_names = load_order_profile(self, "anymal_d")
            configure_newton(self, joint_names, body_names, njmax=256, nconmax=128)


@configclass
class AnymalDFlatEnvCfg_PLAY(AnymalDFlatEnvCfg):
    """Playback configuration for the existing AnyMal D Play registration."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
        self.events.randomize_apply_external_force_torque = None
        self.events.push_robot = None
        self.disable_zero_weight_rewards()

        if type(self) is AnymalDFlatEnvCfg_PLAY:
            from robot_lab.tasks.manager_based.locomotion.velocity.newton_support import configure_newton_from_profile

            configure_newton_from_profile(self, "anymal_d", njmax=256, nconmax=128)
