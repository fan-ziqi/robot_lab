# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = torch.pi * env.episode_length_buf[:, None] * env.step_dt / 2
    phase_tensor = torch.cat(
        [
            torch.sin(phase),
            torch.cos(phase),
            torch.sin(phase / 2),
            torch.cos(phase / 2),
            torch.sin(phase / 4),
            torch.cos(phase / 4),
        ],
        dim=-1,
    )
    return phase_tensor


def last_last_action(env: ManagerBasedEnv) -> torch.Tensor:
    return env.action_manager.prev_action
