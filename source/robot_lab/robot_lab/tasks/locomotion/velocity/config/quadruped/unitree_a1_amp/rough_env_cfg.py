# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import glob

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

import robot_lab.tasks.locomotion.velocity.mdp as mdp
from robot_lab.tasks.locomotion.velocity.config.unitree_a1_amp.env.events import reset_amp
from robot_lab.tasks.locomotion.velocity.velocity_env_cfg import (
    EventCfg,
    LocomotionVelocityRoughEnvCfg,
    ObservationsCfg,
)
from robot_lab.third_party.amp_utils import AMP_UTILS_DIR

##
# Pre-defined configs
##
# use cloud assets
# from isaaclab_assets.robots.unitree import UNITREE_A1_CFG  # isort: skip
# use local assets
from robot_lab.assets.unitree import UNITREE_A1_CFG  # isort: skip


@configclass
class UnitreeA1AmpObservationsCfg(ObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class AmpCfg(ObsGroup):
        """Observations for Amp group."""

        base_pos_z = ObsTerm(func=mdp.base_pos_z, scale=1.0, clip=(-100.0, 100.0))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=1.0, clip=(-100.0, 100.0))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=1.0, clip=(-100.0, 100.0))
        joint_pos = ObsTerm(func=mdp.joint_pos, scale=1.0, clip=(-100.0, 100.0))
        joint_vel = ObsTerm(func=mdp.joint_vel, scale=1.0, clip=(-100.0, 100.0))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    amp: AmpCfg = AmpCfg()


@configclass
class UnitreeA1AmpEventCfg(EventCfg):
    """Configuration for events."""

    reset_amp = EventTerm(func=reset_amp, mode="reset")


@configclass
class UnitreeA1AmpRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    observations: UnitreeA1AmpObservationsCfg = UnitreeA1AmpObservationsCfg()
    events: UnitreeA1AmpEventCfg = UnitreeA1AmpEventCfg()

    base_link_name = "base"
    foot_link_name = ".*_foot"

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

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
        self.observations.policy.base_ang_vel = None
        self.observations.policy.height_scan = None
        self.observations.critic = None

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
        self.rewards.body_lin_acc_l2.weight = 0

        # Joint penaltie
        self.rewards.joint_torques_l2.weight = 0
        # UNUESD self.rewards.joint_vel_l1.weight = 0.0
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_acc_l2.weight = 0
        self.rewards.joint_pos_limits.weight = 0
        self.rewards.joint_vel_limits.weight = 0

        # Action penalties
        self.rewards.action_rate_l2.weight = 0
        # UNUESD self.rewards.action_l2.weight = 0.0

        # Contact sensor
        self.rewards.undesired_contacts.weight = 0
        self.rewards.contact_forces.weight = 0

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 1.5 * 1.0 / (0.005 * 6)
        self.rewards.track_ang_vel_z_exp.weight = 0.5 * 1.0 / (0.005 * 6)

        # Others
        self.rewards.feet_air_time.weight = 0
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_slide.weight = 0
        self.rewards.joint_power.weight = 0
        self.rewards.stand_still_without_cmd.weight = 0

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeA1AmpRoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 2.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.57, 1.57)

        # ------------------------------AMP------------------------------
        self.urdf_path = f"{AMP_UTILS_DIR}/models/a1/urdf/a1.urdf"
        self.ee_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        self.base_name = self.base_link_name
        self.reference_state_initialization = True
        self.amp_motion_files = glob.glob(f"{AMP_UTILS_DIR}/motion_files/mocap_motions_a1/*")
        self.amp_num_preload_transitions = 2000000
        self.amp_replay_buffer_size = 1000000
