import gymnasium as gym
import torch

from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper


class RslRlAmpVecEnvWrapper(RslRlVecEnvWrapper):
    """Wraps for AMP"""

    def __init__(self, env: ManagerBasedRLEnv):
        super().__init__(env)

        self.amp_loader = self.unwrapped.amp_loader

    """
    Properties
    """

    def get_amp_observations(self) -> torch.Tensor:
        return self.unwrapped.get_amp_observations()
