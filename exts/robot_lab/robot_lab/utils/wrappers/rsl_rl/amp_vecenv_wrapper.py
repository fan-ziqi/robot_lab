# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to configure an :class:`ManagerBasedRLEnv` instance to RSL-RL vectorized environment.

The following example shows how to wrap an environment for RSL-RL:

.. code-block:: python

    from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

    env = RslRlVecEnvWrapper(env)

"""


import gymnasium as gym
import torch

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

from omni.isaac.lab.envs import DirectRLEnv, ManagerBasedRLEnv

from omni.isaac.lab.assets import Articulation



class RslRlAmpVecEnvWrapper(RslRlVecEnvWrapper):
    """Wraps for AMP"""

    def __init__(self, env: ManagerBasedRLEnv):
        super().__init__(env)

        # _robot = Articulation(self.unwrapped.cfg.scene.robot)
        # joint_pos_limits = _robot.data.soft_joint_pos_limits[0]
        # print(joint_pos_limits)
        # joint_pos_limits[..., 0]
        # joint_pos_limits[..., 1]

    """
    Properties
    """

    def get_amp_observations(self) -> torch.Tensor:
        return self.unwrapped.get_amp_observations()

