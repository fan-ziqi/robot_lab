# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils import configclass

import robot_lab.tasks.locomotion.velocity.mdp as mdp
from robot_lab.tasks.locomotion.velocity.config.unitree_a1_flip.env import observations, rewards
from robot_lab.tasks.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    ObservationsCfg,
    RewardsCfg,
)

##
# Pre-defined configs
##
# use cloud assets
from omni.isaac.lab_assets.unitree import UNITREE_GO2_CFG  # isort: skip

# use local assets
# from robot_lab.assets.unitree import UNITREE_A1_CFG  # isort: skip


@configclass
class UnitreeA1FlipRewardsCfg(RewardsCfg):
    """Reward terms for the MDP."""

    flip_ang_vel_around_axis = RewTerm(
        func=rewards.flip_ang_vel_around_axis,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "axis": str,  # axis to flip around, can be "+x", "-x", "+y", "-y"
        },
    )

    flip_ang_vel_z = RewTerm(
        func=rewards.flip_ang_vel_z,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    flip_lin_vel_z = RewTerm(
        func=rewards.flip_lin_vel_z,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    flip_orientation_control = RewTerm(
        func=rewards.flip_orientation_control,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "axis": str,  # axis to flip around, can be "+x", "-x", "+y", "-y"
        },
    )

    flip_feet_height_before_backflip = RewTerm(
        func=rewards.flip_feet_height_before_backflip,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "foot_radius": float,
        },
    )

    flip_height_control = RewTerm(
        func=rewards.flip_height_control,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_height": float,
        },
    )

    flip_actions_symmetry = RewTerm(
        func=rewards.flip_actions_symmetry,
        weight=0.0,
    )

    flip_gravity_axis = RewTerm(
        func=rewards.flip_gravity_axis,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "axis": str,  # axis to punish, can be "+x", "-x", "+y", "-y"
        },
    )

    flip_feet_distance = RewTerm(
        func=mdp.feet_distance,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "stance_width": float,
        },
    )


@configclass
class UnitreeA1FlipObservationsCfg(ObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        # last_last_action = ObsTerm(
        #     func=observations.last_last_action,
        #     scale=1.0,
        # )

        phase = ObsTerm(
            func=observations.phase,
            scale=1.0,
        )

        def __post_init__(self):
            super().__post_init__()

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        base_pos_z = ObsTerm(func=mdp.base_pos_z, scale=1.0)
        # last_last_action = ObsTerm(func=observations.last_last_action)
        phase = ObsTerm(func=observations.phase, scale=1.0)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class UnitreeA1FlipRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: UnitreeA1FlipRewardsCfg = UnitreeA1FlipRewardsCfg()
    observations: UnitreeA1FlipObservationsCfg = UnitreeA1FlipObservationsCfg()

    base_link_name = "base"
    foot_link_name = ".*_foot"

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.episode_length_s = 2.0
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.eye = (1.0, 2.0, 1.0)
        self.viewer.resolution = (1920, 1080)

        # ------------------------------Sence------------------------------
        # switch robot to unitree-a1
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        self.scene.robot.actuators["base_legs"].stiffness = 70.0
        self.scene.robot.actuators["base_legs"].damping = 3.0
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
        self.observations.policy.velocity_commands = None
        # self.observations.policy.height_scan = None
        self.observations.critic.velocity_commands = None

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}

        # ------------------------------Events------------------------------
        self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

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
        self.rewards.joint_torques_l2.weight = 0
        # UNUESD self.rewards.joint_vel_l1.weight = 0.0
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_acc_l2.weight = 0
        # self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -0.1, [".*_hip_joint"])
        self.rewards.joint_pos_limits.weight = 0
        self.rewards.joint_vel_limits.weight = 0

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.001
        # UNUESD self.rewards.action_l2.weight = 0.0

        # Contact sensor
        self.rewards.undesired_contacts.weight = -10
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = 0
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0

        # Others
        self.rewards.feet_air_time.weight = 0
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = 0
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.joint_power.weight = 0
        self.rewards.stand_still_when_zero_command.weight = 0

        # Flip
        flip_type = "front"  # can be "front", "back", "left", "right"
        if flip_type == "front":
            flip_axis = "+y"
        elif flip_type == "back":
            flip_axis = "-y"
        elif flip_type == "left":
            flip_axis = "-x"
        elif flip_type == "right":
            flip_axis = "+x"
        self.rewards.flip_feet_height_before_backflip.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.flip_feet_distance.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.flip_ang_vel_around_axis.weight = 10.0  # 5.0
        self.rewards.flip_ang_vel_around_axis.params["axis"] = flip_axis
        self.rewards.flip_ang_vel_z.weight = -1.0  # -1.0
        self.rewards.flip_lin_vel_z.weight = 40.0  # 20.0
        self.rewards.flip_orientation_control.weight = -1.0  # -1.0
        self.rewards.flip_orientation_control.params["axis"] = flip_axis
        self.rewards.flip_feet_height_before_backflip.weight = -30  # -30.0
        self.rewards.flip_feet_height_before_backflip.params["foot_radius"] = 0.02
        self.rewards.flip_height_control.weight = -10.0  # -10.0
        self.rewards.flip_height_control.params["target_height"] = 0.3
        self.rewards.flip_actions_symmetry.weight = 0
        self.rewards.flip_gravity_axis.weight = -10.0  # -10.0
        self.rewards.flip_gravity_axis.params["axis"] = flip_axis
        self.rewards.flip_feet_distance.weight = -1.0  # -1.0
        self.rewards.flip_feet_distance.params["stance_width"] = 0.30

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeA1FlipRoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact = None
        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
        #     self.base_link_name,
        #     ".*_hip",
        #     ".*_thigh",
        #     ".*_calf",
        # ]

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.resampling_time_range = (4.0, 4.0)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.debug_vis = False
