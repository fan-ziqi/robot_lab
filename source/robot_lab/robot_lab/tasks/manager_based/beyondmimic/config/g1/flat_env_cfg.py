# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import os

from isaaclab.utils import configclass

from robot_lab.assets.unitree import UNITREE_G1_29DOF_ACTION_SCALE, UNITREE_G1_29DOF_CFG
from robot_lab.tasks.manager_based.beyondmimic.tracking_env_cfg import BeyondMimicEnvCfg


@configclass
class UnitreeG1BeyondMimicFlatEnvCfg(BeyondMimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = UNITREE_G1_29DOF_ACTION_SCALE
        self.commands.motion.motion_file = f"{os.path.dirname(__file__)}/motion/G1_Take_102.bvh_60hz.npz"
        # self.commands.motion.motion_file = f"{os.path.dirname(__file__)}/motion/G1_gangnam_style_V01.bvh_60hz.npz"
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]

        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None

        self.episode_length_s = 30.0
