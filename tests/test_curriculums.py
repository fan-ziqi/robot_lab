# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Repeated resets before stepping must not repeatedly shrink command ranges."""

from types import SimpleNamespace

import pytest
import torch

from robot_lab.tasks.manager_based.locomotion.velocity.mdp.curriculums import (
    command_levels_ang_vel,
    command_levels_lin_vel,
)


@pytest.mark.parametrize(
    "curriculum,field,expected",
    [(command_levels_lin_vel, "lin_vel_x", [-0.1, 0.1]), (command_levels_ang_vel, "ang_vel_z", [-0.2, 0.2])],
)
def test_command_curriculum_initializes_ranges_only_once(curriculum, field, expected):
    ranges = SimpleNamespace(lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-2.0, 2.0))
    env = SimpleNamespace(
        device="cpu",
        common_step_counter=0,
        max_episode_length=100,
        max_episode_length_s=2.0,
        command_manager=SimpleNamespace(get_term=lambda name: SimpleNamespace(cfg=SimpleNamespace(ranges=ranges))),
        reward_manager=SimpleNamespace(
            _episode_sums={"tracking": torch.zeros(2)},
            get_term_cfg=lambda name: SimpleNamespace(weight=1.0),
        ),
    )
    for _ in range(2):
        curriculum(env, torch.arange(2), "tracking")
        assert getattr(ranges, field) == pytest.approx(expected)
