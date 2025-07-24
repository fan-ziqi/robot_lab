# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import inspect
import math
import sys
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment='yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    height_scanner_base = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment='yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.1, 0.1)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformThresholdVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformThresholdVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True, clip=None, preserve_order=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=1.0,
        )
        # joint_effort = ObsTerm(
        #     func=mdp.joint_effort,
        #     clip=(-100, 100),
        #     scale=0.01,
        # )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    randomize_rigid_body_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 64,
        },
    )

    randomize_rigid_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    randomize_rigid_body_inertia = EventTerm(
        func=mdp.randomize_rigid_body_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "inertia_distribution_params": (0.5, 1.5),
            "operation": "scale",
        },
    )

    randomize_com_positions = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # reset
    randomize_apply_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "force_range": (-10.0, 10.0),
            "torque_range": (-10.0, 10.0),
        },
    )

    randomize_reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        # func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.5, 2.0),
            "damping_distribution_params": (0.5, 2.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    randomize_reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    # interval
    randomize_push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # General
    is_terminated = RewTerm(func=mdp.is_terminated, weight=0.0)

    # Root penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=0.0)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "sensor_cfg": SceneEntityCfg("height_scanner_base"),
            "target_height": 0.0,
        },
    )
    body_lin_acc_l2 = RewTerm(
        func=mdp.body_lin_acc_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="")},
    )

    # Joint penalties
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )

    def create_joint_deviation_l1_rewterm(self, attr_name, weight, joint_names_pattern):
        rew_term = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=weight,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=joint_names_pattern)},
        )
        setattr(self, attr_name, rew_term)

    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    joint_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "soft_ratio": 1.0},
    )
    joint_power = RewTerm(
        func=mdp.joint_power,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )

    stand_still_without_cmd = RewTerm(
        func=mdp.stand_still_without_cmd,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )

    joint_pos_penalty = RewTerm(
        func=mdp.joint_pos_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
        },
    )

    wheel_vel_penalty = RewTerm(
        func=mdp.wheel_vel_penalty,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=""),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "command_name": "base_velocity",
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
        },
    )

    joint_mirror = RewTerm(
        func=mdp.joint_mirror,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [["FR.*", "RL.*"], ["FL.*", "RR.*"]],
        },
    )

    action_mirror = RewTerm(
        func=mdp.action_mirror,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [["FR.*", "RL.*"], ["FL.*", "RR.*"]],
        },
    )

    action_sync = RewTerm(
        func=mdp.action_sync,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_groups": [
                ["FR_hip_joint", "FL_hip_joint", "RL_hip_joint", "RR_hip_joint"],
                ["FR_thigh_joint", "FL_thigh_joint", "RL_thigh_joint", "RR_thigh_joint"],
                ["FR_calf_joint", "FL_calf_joint", "RL_calf_joint", "RR_calf_joint"],
            ],
        },
    )

    # Action penalties
    applied_torque_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=0.0)
    # smoothness_1 = RewTerm(func=mdp.smoothness_1, weight=0.0)  # Same as action_rate_l2
    # smoothness_2 = RewTerm(func=mdp.smoothness_2, weight=0.0)  # Unvaliable now

    # Contact sensor
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "threshold": 1.0,
        },
    )
    contact_forces = RewTerm(
        func=mdp.contact_forces,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=""), "threshold": 100.0},
    )

    # Velocity-tracking rewards
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=0.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # Others
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "mode_time": 0.3,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
        },
    )

    feet_air_time_variance = RewTerm(
        func=mdp.feet_air_time_variance_penalty,
        weight=0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="")},
    )

    feet_gait = RewTerm(
        func=mdp.GaitReward,
        weight=0.0,
        params={
            "std": math.sqrt(0.5),
            "command_name": "base_velocity",
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
            "synced_feet_pair_names": (("", ""), ("", "")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )

    feet_contact = RewTerm(
        func=mdp.feet_contact,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "command_name": "base_velocity",
            "expect_contact_num": 2,
        },
    )

    feet_contact_without_cmd = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "command_name": "base_velocity",
        },
    )

    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
        },
    )

    feet_height = RewTerm(
        func=mdp.feet_height,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "tanh_mult": 2.0,
            "target_height": 0.05,
            "command_name": "base_velocity",
        },
    )

    feet_height_body = RewTerm(
        func=mdp.feet_height_body,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "tanh_mult": 2.0,
            "target_height": -0.3,
            "command_name": "base_velocity",
        },
    )

    feet_distance_y_exp = RewTerm(
        func=mdp.feet_distance_y_exp,
        weight=0.0,
        params={
            "std": math.sqrt(0.25),
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "stance_width": float,
        },
    )

    # feet_distance_xy_exp = RewTerm(
    #     func=mdp.feet_distance_xy_exp,
    #     weight=0.0,
    #     params={
    #         "std": math.sqrt(0.25),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=""),
    #         "stance_length": float,
    #         "stance_width": float,
    #     },
    # )

    upward = RewTerm(func=mdp.upward, weight=0.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # MDP terminations
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # command_resample
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )

    # Contact sensor
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=""), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

    command_levels = CurrTerm(
        func=mdp.command_levels_vel,
        params={
            "reward_term_name": "track_lin_vel_xy_exp",
            "range_multiplier": (0.1, 1.0),
        },
    )


##
# Environment configuration
##


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

    def disable_zero_weight_rewards(self):
        """If the weight of rewards is 0, set rewards to None"""
        for attr in dir(self.rewards):
            if not attr.startswith("__"):
                reward_attr = getattr(self.rewards, attr)
                if not callable(reward_attr) and reward_attr.weight == 0:
                    setattr(self.rewards, attr, None)


def create_obsgroup_class(class_name, terms, enable_corruption=False, concatenate_terms=True):
    """
    Dynamically create and register a ObsGroup class based on the given configuration terms.

    :param class_name: Name of the configuration class.
    :param terms: Configuration terms, a dictionary where keys are term names and values are term content.
    :param enable_corruption: Whether to enable corruption for the observation group. Defaults to False.
    :param concatenate_terms: Whether to concatenate the observation terms in the group. Defaults to True.
    :return: The dynamically created class.
    """
    # Dynamically determine the module name
    module_name = inspect.getmodule(inspect.currentframe()).__name__

    # Define the post-init function
    def post_init_wrapper(self):
        setattr(self, "enable_corruption", enable_corruption)
        setattr(self, "concatenate_terms", concatenate_terms)

    # Dynamically create the class using ObsGroup as the base class
    terms["__post_init__"] = post_init_wrapper
    dynamic_class = configclass(type(class_name, (ObsGroup,), terms))

    # Custom serialization and deserialization
    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    # Add custom serialization methods to the class
    dynamic_class.__getstate__ = __getstate__
    dynamic_class.__setstate__ = __setstate__

    # Place the class in the global namespace for accessibility
    globals()[class_name] = dynamic_class

    # Register the dynamic class in the module's dictionary
    if module_name in sys.modules:
        sys.modules[module_name].__dict__[class_name] = dynamic_class
    else:
        raise ImportError(f"Module {module_name} not found.")

    # Return the class for external instantiation
    return dynamic_class
