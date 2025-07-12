# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2025 Linden
# SPDX-License-Identifier: BSD 3-Clause

from __future__ import annotations

import os
from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from robot_lab.assets.unitree import UNITREE_G1_29DOF_CFG

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")


@configclass
class G1AmpEnvCfg(DirectRLEnvCfg):
    """Humanoid AMP environment config (base class)."""

    # reward
    rew_termination = -0
    rew_action_l2 = -0
    rew_joint_pos_limits = -0
    rew_joint_acc_l2 = -0
    rew_joint_vel_l2 = -0

    # env
    episode_length_s = 10.0
    decimation = 2

    # spaces
    observation_space = 71 + 3 * 10
    action_space = 29
    state_space = 0
    num_amp_observations = 2
    amp_observation_space = 71 + 3 * 10

    early_termination = True
    termination_height = 0.5

    motion_file: str = MISSING
    reference_body = "pelvis"
    reset_strategy = "random"  # default, random, random-start
    """Strategy to be followed when resetting each environment (humanoid's pose and joint states).

    * default: pose and joint states are set to the initial state of the asset.
    * random: pose and joint states are set by sampling motions at random, uniform times.
    * random-start: pose and joint states are set by sampling motion at the start (time zero).
    """

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = UNITREE_G1_29DOF_CFG.replace(prim_path="/World/envs/env_.*/Robot")


@configclass
class G1AmpDanceEnvCfg(G1AmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "g1_dance.npz")


@configclass
class G1AmpWalkEnvCfg(G1AmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "g1_walk.npz")
