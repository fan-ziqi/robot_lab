import numpy as np
import torch
from typing import Tuple

_EPS = np.finfo(float).eps * 4.0


class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        """
        Calculates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class Normalizer(RunningMeanStd):
    def __init__(self, input_dim, epsilon=1e-4, clip_obs=10.0):
        super().__init__(shape=input_dim)
        self.epsilon = epsilon
        self.clip_obs = clip_obs

    def normalize(self, input):
        return np.clip((input - self.mean) / np.sqrt(self.var + self.epsilon), -self.clip_obs, self.clip_obs)

    def normalize_torch(self, input, device):
        mean_torch = torch.tensor(self.mean, device=device, dtype=torch.float32)
        std_torch = torch.sqrt(torch.tensor(self.var + self.epsilon, device=device, dtype=torch.float32))
        return torch.clamp((input - mean_torch) / std_torch, -self.clip_obs, self.clip_obs)

    def update_normalizer(self, rollouts, expert_loader):
        policy_data_generator = rollouts.feed_forward_generator_amp(None, mini_batch_size=expert_loader.batch_size)
        expert_data_generator = expert_loader.dataset.feed_forward_generator_amp(expert_loader.batch_size)

        for expert_batch, policy_batch in zip(expert_data_generator, policy_data_generator):
            self.update(torch.vstack(tuple(policy_batch) + tuple(expert_batch)).cpu().numpy())


def quaternion_slerp(q0, q1, fraction, spin=0, shortestpath=True):
    """Batch quaternion spherical linear interpolation."""

    out = torch.zeros_like(q0)

    zero_mask = torch.isclose(fraction, torch.zeros_like(fraction)).squeeze()
    ones_mask = torch.isclose(fraction, torch.ones_like(fraction)).squeeze()
    out[zero_mask] = q0[zero_mask]
    out[ones_mask] = q1[ones_mask]

    d = torch.sum(q0 * q1, dim=-1, keepdim=True)
    dist_mask = (torch.abs(torch.abs(d) - 1.0) < _EPS).squeeze()
    out[dist_mask] = q0[dist_mask]

    if shortestpath:
        d_old = torch.clone(d)
        d = torch.where(d_old < 0, -d, d)
        q1 = torch.where(d_old < 0, -q1, q1)

    angle = torch.acos(d) + spin * torch.pi
    angle_mask = (torch.abs(angle) < _EPS).squeeze()
    out[angle_mask] = q0[angle_mask]

    final_mask = torch.logical_or(zero_mask, ones_mask)
    final_mask = torch.logical_or(final_mask, dist_mask)
    final_mask = torch.logical_or(final_mask, angle_mask)
    final_mask = torch.logical_not(final_mask)

    isin = 1.0 / angle
    q0 *= torch.sin((1.0 - fraction) * angle) * isin
    q1 *= torch.sin(fraction * angle) * isin
    q0 += q1
    out[final_mask] = q0[final_mask]
    return out
