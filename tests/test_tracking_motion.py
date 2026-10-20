# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Named tracking archives keep their joint and body association across importers."""

import numpy as np
import pytest
import torch

from robot_lab.tasks.manager_based.beyondmimic.mdp.commands import MotionLoader


def test_named_motion_resolves_recorded_joint_and_body_order(tmp_path):
    path = tmp_path / "tracking.npz"
    np.savez(
        path,
        fps=[50],
        quaternion_order="xyzw",
        joint_names=["b", "a"],
        body_names=["foot", "pelvis"],
        joint_pos=[[20, 10], [21, 11]],
        joint_vel=[[2, 1], [2, 1]],
        body_pos_w=[[[1, 0, 0], [2, 0, 0]]] * 2,
        body_quat_w=[[[0, 0, 0, 1], [0, 0, 0, 1]]] * 2,
        body_lin_vel_w=np.zeros((2, 2, 3)),
        body_ang_vel_w=np.zeros((2, 2, 3)),
    )
    motion = MotionLoader(str(path), [0], body_names=["pelvis"], joint_names=["a", "b"])
    torch.testing.assert_close(motion.joint_pos, torch.tensor([[10.0, 20.0], [11.0, 21.0]]))
    torch.testing.assert_close(motion.body_pos_w[..., 0], torch.full((2, 1), 2.0))
    assert motion.fps == 50.0

    with pytest.raises(ValueError, match="Reconvert"):
        MotionLoader(str(path), [0], num_robot_bodies=3)
