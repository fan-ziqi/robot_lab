# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robot_lab.tasks.locomotion.velocity.mdp as mdp
from robot_lab.tasks.locomotion.velocity.config.unitree_a1_handstand.env import rewards
from robot_lab.tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

##
# Pre-defined configs
##
# use cloud assets
from isaaclab_assets.robots.unitree import UNITREE_A1_CFG  # isort: skip

# use local assets
# from robot_lab.assets.unitree import UNITREE_A1_CFG  # isort: skip


@configclass
class UnitreeA1HandStandRewardsCfg(RewardsCfg):
    """Reward terms for the MDP."""

    handstand_feet_height_exp = RewTerm(
        func=rewards.handstand_feet_height_exp,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "target_height": 0.0, "std": math.sqrt(0.25)},
    )

    handstand_feet_on_air = RewTerm(
        func=rewards.handstand_feet_on_air,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
        },
    )

    handstand_feet_air_time = RewTerm(
        func=rewards.handstand_feet_air_time,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "threshold": 5.0,
        },
    )

    handstand_orientation_l2 = RewTerm(
        func=rewards.handstand_orientation_l2,
        weight=0.0,
        params={
            "target_gravity": [],
        },
    )


@configclass
class UnitreeA1HandStandRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: UnitreeA1HandStandRewardsCfg = UnitreeA1HandStandRewardsCfg()

    base_link_name = "trunk"
    foot_link_name = ".*_foot"

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.episode_length_s = 10.0

        # ------------------------------Sence------------------------------
        # switch robot to unitree-a1
        self.scene.robot = UNITREE_A1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # ------------------------------Observations------------------------------
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}

        # ------------------------------Events------------------------------
        self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        self.events.randomize_rigid_body_mass = None
        self.events.randomize_com_positions = None
        self.events.randomize_apply_external_force_torque = None

        # ------------------------------Rewards------------------------------
        # General
        # UNUESD self.rewards.is_alive.weight = 0
        self.rewards.is_terminated.weight = 0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = 0
        self.rewards.ang_vel_xy_l2.weight = 0
        self.rewards.flat_orientation_l2.weight = 0
        self.rewards.base_height_l2.weight = 0
        self.rewards.base_height_l2.params["target_height"] = 0.35
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penaltie
        self.rewards.joint_torques_l2.weight = -0.0002
        # UNUESD self.rewards.joint_vel_l1.weight = 0.0
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_acc_l2.weight = -2.5e-7
        # self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -0.1, [".*_hip_joint"])
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_vel_limits.weight = 0

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.05
        # UNUESD self.rewards.action_l2.weight = 0.0

        # Contact sensor
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [".*_thigh"]
        self.rewards.contact_forces.weight = 0
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_world_xy_exp
        self.rewards.track_ang_vel_z_exp.func = mdp.track_ang_vel_world_z_exp

        # Others
        self.rewards.feet_air_time.weight = 0
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = 0
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.joint_power.weight = -2e-5
        self.rewards.stand_still_when_zero_command.weight = 0

        # HandStand
        handstand_type = "back"  # which leg on air, can be "front", "back", "left", "right"
        if handstand_type == "front":
            air_foot_name = "F.*_foot"
            self.rewards.handstand_orientation_l2.weight = -1.0
            self.rewards.handstand_orientation_l2.params["target_gravity"] = [-1.0, 0.0, 0.0]
            self.rewards.handstand_feet_height_exp.params["target_height"] = 0.5
        elif handstand_type == "back":
            air_foot_name = "R.*_foot"
            self.rewards.handstand_orientation_l2.weight = -1.0
            self.rewards.handstand_orientation_l2.params["target_gravity"] = [1.0, 0.0, 0.0]
            self.rewards.handstand_feet_height_exp.params["target_height"] = 0.5
        elif handstand_type == "left":
            air_foot_name = ".*L_foot"
            self.rewards.handstand_orientation_l2.weight = 0
            self.rewards.handstand_orientation_l2.params["target_gravity"] = [0.0, -1.0, 0.0]
            self.rewards.handstand_feet_height_exp.params["target_height"] = 0.3
        elif handstand_type == "right":
            air_foot_name = ".*R_foot"
            self.rewards.handstand_orientation_l2.weight = 0
            self.rewards.handstand_orientation_l2.params["target_gravity"] = [0.0, 1.0, 0.0]
            self.rewards.handstand_feet_height_exp.params["target_height"] = 0.3
        self.rewards.handstand_feet_height_exp.weight = 10
        self.rewards.handstand_feet_height_exp.params["asset_cfg"].body_names = [air_foot_name]
        self.rewards.handstand_feet_on_air.weight = 1.0
        self.rewards.handstand_feet_on_air.params["sensor_cfg"].body_names = [air_foot_name]
        self.rewards.handstand_feet_air_time.weight = 1.0
        self.rewards.handstand_feet_air_time.params["sensor_cfg"].body_names = [air_foot_name]

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeA1HandStandRoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]

        # ------------------------------Commands------------------------------
        # self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        # self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # self.commands.base_velocity.heading_command = False
        # self.commands.base_velocity.debug_vis = False
