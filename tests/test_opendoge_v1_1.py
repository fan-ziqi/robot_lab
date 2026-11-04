"""Catch observation/axis/action mismatches between LainLab and Isaac Lab."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest


def test_flat_input_keeps_unscaled_velocities_and_command_last():
    from robot_lab.policies.opendoge_v1_1 import PolicyState

    state = PolicyState("flat")
    # Literal V1.1 stance, alternating left/right leg signs.
    q = np.array([0, 0.8, -1.6, 0, -0.8, 1.6, 0, 0.8, -1.6, 0, -0.8, 1.6])
    obs = state.observe([0.3, -0.2, 0.4], [1, 2, 3], [4, 5, 6], [0, 0, -1], q + 0.1, np.arange(12))
    assert obs.shape == (1, 48)
    np.testing.assert_allclose(obs[0, :9], [1, 2, 3, 4, 5, 6, 0, 0, -1])
    np.testing.assert_allclose(obs[0, 9:21], 0.1, atol=1e-7)
    np.testing.assert_allclose(obs[0, 21:33], np.arange(12))
    np.testing.assert_equal(obs[0, 33:45], 0)
    np.testing.assert_allclose(obs[0, 45:48], [0.3, -0.2, 0.4])


def test_actions_are_leg_major_unclipped_offsets_and_reset_previous_action():
    from robot_lab.policies.opendoge_v1_1 import PolicyState

    state = PolicyState("flat")
    targets = state.accept_action(np.arange(12, dtype=np.float32)[None])
    np.testing.assert_allclose(targets, [0, 1.05, -1.1, 0.75, 0.2, 2.85, 1.5, 2.55, 0.4, 2.25, 1.7, 4.35], atol=1e-6)
    obs = state.observe([0] * 3, [0] * 3, [0] * 3, [0, 0, -1], state.default_angles, [0] * 12)
    np.testing.assert_equal(obs[0, 33:45], np.arange(12))
    state.reset()
    np.testing.assert_equal(state.action, 0)
    with pytest.raises(ValueError):
        state.accept_action(np.full((1, 12), np.nan))
    with pytest.raises(ValueError):
        state.observe([0] * 3, [0] * 3, [0] * 3, [0, 0, -1], [0] * 11, [0] * 12)


def test_rough_scan_uses_sensor_height_over_hit_divided_by_five():
    from robot_lab.policies.opendoge_v1_1 import PolicyState

    state = PolicyState("rough")
    args = ([0] * 3, [0] * 3, [0] * 3, [0, 0, -1], state.default_angles, [0] * 12)
    with pytest.raises(ValueError):
        state.observe(*args)
    obs = state.observe(*args, height_scan=np.full(165, 0.15))
    assert obs.shape == (1, 213)
    np.testing.assert_allclose(obs[0, 48:], 0.03)


def test_skills_command_phase_and_height_qualified_contact_latch():
    from robot_lab.policies.opendoge_v1_1 import PolicyState

    state = PolicyState("skills", skill="jump")
    args = ([0.1, 0.2, 0.3], [0] * 3, [0] * 3, [0, 0, -1], state.default_angles, [0] * 12)
    obs = state.observe(*args)
    np.testing.assert_allclose(obs[0, 45:], [0, 1, 0, 0.1, 0.2, 0.3, 0, 0, 0, 1, 0, 1])
    state.advance(0.01, 0.15, [True] * 4)
    state.advance(0.01, 0.16, [False] * 4)  # gait suspension is not takeoff
    assert not state.left_ground
    state.advance(0.605, 0.18, [False] * 4)
    obs = state.observe(*args)
    np.testing.assert_allclose(obs[0, 45:48], [1, 0, 1], atol=1e-6)
    np.testing.assert_allclose(obs[0, 55:57], [1, 0], atol=1e-6)
    state.reset()
    assert not state.left_ground
    stationary = PolicyState("skills", skill="handstand")
    stationary.advance(0.625, 0.15, [True] * 4)
    obs = stationary.observe(*args)
    np.testing.assert_allclose(obs[0, 45:], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1])


def test_imu_velocity_includes_site_offset_cross_product():
    from robot_lab.policies.opendoge_v1_1 import imu_linear_velocity

    np.testing.assert_allclose(imu_linear_velocity([1, 2, 3], [2, 3, 4]), [1.09, 1.94, 3])


def test_v11_configuration_removes_solver_speed_caps_without_mutating_legacy():
    from robot_lab.assets.opendoge import OPENDOGE_CFG
    from robot_lab.assets.opendoge_v1_1 import OPENDOGE_V1_1_CFG

    # The source's built-in position actuators saturate torque, not speed.
    # actuator_velocity_limit alone does not override the solver's URDF cap.
    assert OPENDOGE_V1_1_CFG.actuators["legs"].joint_velocity_limit == 1e9
    assert OPENDOGE_V1_1_CFG.spawn.asset_path != OPENDOGE_CFG.spawn.asset_path
    changed = OPENDOGE_V1_1_CFG.copy()
    changed.init_state.joint_pos["FL_thigh_joint"] = 0.4
    assert OPENDOGE_V1_1_CFG.init_state.joint_pos["FL_thigh_joint"] == pytest.approx(0.8)
    assert OPENDOGE_CFG.init_state.joint_pos[".*_thigh_joint"] == 0.6


def test_assets_and_all_bundled_exports_are_self_contained():
    from robot_lab.policies.opendoge_v1_1 import ASSET_DIR, OnnxPolicy, PolicyState

    urdf = ASSET_DIR / "urdf/opendoge.urdf"
    root = ET.parse(urdf).getroot()
    for mesh in root.findall(".//mesh"):
        assert (urdf.parent / mesh.attrib["filename"]).is_file()
    assert len([j for j in root.findall("joint") if j.attrib["type"] == "revolute"]) == 12
    pytest.importorskip("onnxruntime")
    for profile in ("flat", "getup", "rough", "skills"):
        state = PolicyState(profile)
        obs = state.observe(
            [0] * 3,
            [0] * 3,
            [0] * 3,
            [0, 0, -1],
            state.default_angles,
            [0] * 12,
            height_scan=np.full(165, 0.158) if profile == "rough" else None,
        )
        action = OnnxPolicy(profile)(obs)
        assert action.shape == (1, 12)
        assert np.isfinite(action).all()
    from robot_lab.policies.opendoge_legacy import DEFAULT_POLICY

    with pytest.raises(ValueError):
        OnnxPolicy("flat", DEFAULT_POLICY)
