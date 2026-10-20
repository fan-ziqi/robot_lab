# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Motion archives retain their physical timing and quaternion convention."""

import numpy as np
import pytest
import torch

from robot_lab.tasks.direct.g1_amp.motions.motion_loader import MotionLoader


@pytest.fixture
def motion_archive(tmp_path):
    def write(order="xyzw"):
        frames = np.linspace(0.0, 1.0, 5, dtype=np.float32)
        rotations = np.zeros((5, 1, 4), dtype=np.float32)
        rotations[:, 0, 2] = np.sin(frames * np.pi / 2)
        rotations[:, 0, 3] = np.cos(frames * np.pi / 2)
        if order == "wxyz":
            rotations = rotations[..., [3, 0, 1, 2]]
        path = tmp_path / "motion.npz"
        np.savez(
            path,
            fps=4,
            quaternion_order=order,
            dof_names=["joint"],
            body_names=["pelvis"],
            dof_positions=frames[:, None],
            dof_velocities=np.ones((5, 1), dtype=np.float32),
            body_positions=np.repeat(frames[:, None, None], 3, axis=-1),
            body_rotations=rotations,
            body_linear_velocities=np.ones((5, 1, 3), dtype=np.float32),
            body_angular_velocities=np.zeros((5, 1, 3), dtype=np.float32),
        )
        return MotionLoader(str(path), "cpu")

    return write


@pytest.mark.parametrize("order", ["xyzw", "wxyz"])
def test_sampling_interpolates_and_clamps_motion_time(motion_archive, order):
    motion = motion_archive(order)
    # A nonlinear trajectory distinguishes adjacent-frame interpolation from extrapolation.
    motion.dof_positions.square_()
    positions, _, _, rotations, _, _ = motion.sample(3, times=np.array([-0.25, 0.1875, 1.25]))
    torch.testing.assert_close(positions[:, 0], torch.tensor([0.0, 0.046875, 1.0]))
    angles = torch.tensor([0.0, 0.1875, 1.0]) * torch.pi / 2
    expected = torch.stack((torch.zeros(3), torch.zeros(3), angles.sin(), angles.cos()), dim=-1)
    torch.testing.assert_close(rotations[:, 0], expected, atol=1e-6, rtol=1e-5)


def test_resampling_preserves_the_declared_time_step(motion_archive):
    motion = motion_archive()
    motion.resample(0.3)
    torch.testing.assert_close(motion.dof_positions[:, 0], torch.tensor([0.0, 0.3, 0.6, 0.9]))
    assert motion.dt == pytest.approx(0.3)
    assert motion.duration == pytest.approx(0.9)


def test_csv_angular_velocity_is_expressed_in_world_axes():
    from robot_lab.tasks.direct.g1_amp.motions.csv2npz import compute_angular_velocity

    previous = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0])
    following = np.array([0.5, 0.5, 0.5, 0.5])
    np.testing.assert_allclose(compute_angular_velocity(previous, following, 1.0), [0.0, 0.0, np.pi / 2], atol=1e-7)


@pytest.mark.parametrize("body_ids", [slice(None, None, 2), torch.tensor([0, 2])])
def test_feet_height_accepts_manager_body_selections(body_ids):
    from types import SimpleNamespace

    from isaaclab.managers import SceneEntityCfg

    from robot_lab.tasks.manager_based.locomotion.velocity.mdp.rewards import feet_height_body

    def proxy(value):
        return SimpleNamespace(torch=torch.tensor(value, dtype=torch.float32))

    data = SimpleNamespace(
        body_pos_w=proxy([[[0, 0, -2], [0, 0, -3], [0, 0, -4]]]),
        body_lin_vel_w=proxy([[[1, 0, 0], [1, 0, 0], [1, 0, 0]]]),
        root_pos_w=proxy([[0, 0, 0]]),
        root_lin_vel_w=proxy([[0, 0, 0]]),
        root_quat_w=proxy([[0, 0, 0, 1]]),
        projected_gravity_b=proxy([[0, 0, -1]]),
    )
    env = SimpleNamespace(
        scene={"robot": SimpleNamespace(data=data)},
        num_envs=1,
        device="cpu",
        command_manager=SimpleNamespace(get_command=lambda name: torch.ones((1, 3))),
    )
    result = feet_height_body(env, "base_velocity", SceneEntityCfg("robot", body_ids=body_ids), -1.0, 1.0)
    torch.testing.assert_close(result, 10.0 * torch.tanh(torch.ones(1)))
