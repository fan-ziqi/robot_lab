# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robot_lab.tasks.locomotion.velocity.mdp as mdp
from robot_lab.tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import G1_MINIMAL_CFG  # isort: skip


@configclass
class UnitreeG1RewardsCfg(RewardsCfg):
    """Reward terms for the MDP."""

    feet_air_time_biped = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "threshold": 0.4,
        },
    )


@configclass
class UnitreeG1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: UnitreeG1RewardsCfg = UnitreeG1RewardsCfg()

    base_link_name = "torso_link"
    foot_link_name = ".*_ankle_roll_link"
    # fmt: off
    joint_names = [
        "left_hip_pitch_joint",          # 0  L_LEG_HIP_PITCH
        "left_hip_roll_joint",           # 1  L_LEG_HIP_ROLL
        "left_hip_yaw_joint",            # 2  L_LEG_HIP_YAW
        "left_knee_joint",               # 3  L_LEG_KNEE
        "left_ankle_pitch_joint",        # 4  L_LEG_ANKLE_B
        "left_ankle_roll_joint",         # 5  L_LEG_ANKLE_A
        "right_hip_pitch_joint",         # 6  R_LEG_HIP_PITCH
        "right_hip_roll_joint",          # 7  R_LEG_HIP_ROLL
        "right_hip_yaw_joint",           # 8  R_LEG_HIP_YAW
        "right_knee_joint",              # 9  R_LEG_KNEE
        "right_ankle_pitch_joint",       # 10 R_LEG_ANKLE_B
        "right_ankle_roll_joint",        # 11 R_LEG_ANKLE_A
        "torso_joint",                   # 12 WAIST_YAW
        "left_shoulder_pitch_joint",     # 15 L_SHOULDER_PITCH
        "left_shoulder_roll_joint",      # 16 L_SHOULDER_ROLL
        "left_shoulder_yaw_joint",       # 17 L_SHOULDER_YAW
        "left_elbow_pitch_joint",        # 18 L_ELBOW
        "left_elbow_roll_joint",         # 19 L_WRIST_ROLL
        "right_shoulder_pitch_joint",    # 22 R_SHOULDER_PITCH
        "right_shoulder_roll_joint",     # 23 R_SHOULDER_ROLL
        "right_shoulder_yaw_joint",      # 24 R_SHOULDER_YAW
        "right_elbow_pitch_joint",       # 25 R_ELBOW
        "right_elbow_roll_joint",        # 26 R_WRIST_ROLL
    ]
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ------------------------------Observations------------------------------
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names

        # ------------------------------Events------------------------------
        self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        # ------------------------------Rewards------------------------------
        # General
        self.rewards.is_terminated.weight = -200.0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = 0
        self.rewards.ang_vel_xy_l2.weight = -0.1
        self.rewards.flat_orientation_l2.weight = -0.2
        self.rewards.base_height_l2.weight = 0
        self.rewards.base_height_l2.params["target_height"] = 0
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penaltie
        self.rewards.joint_torques_l2.weight = -1.5e-7
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = [".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_acc_l2.weight = -1.25e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = [".*_hip_.*", ".*_knee_joint"]
        self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -0.1, [".*hip_yaw.*", ".*hip_roll.*"])
        self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_arms_l1", -0.1, [".*shoulder.*", ".*elbow.*"])
        self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_torso_l1", -0.1, ["torso_joint"])
        self.rewards.joint_pos_limits.weight = -1.0
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_power.weight = 0
        self.rewards.stand_still_without_cmd.weight = 0
        self.rewards.joint_pos_penalty.weight = -1.0
        self.rewards.joint_mirror.weight = 0
        self.rewards.joint_mirror.params["mirror_joints"] = [["left_(hip|knee|ankle).*", "right_(hip|knee|ankle).*"]]

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.action_mirror.weight = 0
        self.rewards.action_mirror.params["mirror_joints"] = [["left_(hip|knee|ankle).*", "right_(hip|knee|ankle).*"]]

        # Contact sensor
        self.rewards.undesired_contacts.weight = 0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = 0
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_xy_yaw_frame_exp
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.func = mdp.track_ang_vel_z_world_exp

        # Others
        self.rewards.feet_air_time_biped.weight = 0.25
        self.rewards.feet_air_time_biped.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.2
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height.params["target_height"] = 0.05
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = 0
        self.rewards.feet_height_body.params["target_height"] = -0.2
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.upward.weight = 1.0

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeG1RoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
