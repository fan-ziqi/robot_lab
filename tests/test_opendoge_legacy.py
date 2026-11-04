"""The legacy exported policy retains its training-time input and action contract."""

import numpy as np
import pytest


def test_observations_keep_command_first_layout_and_newest_first_history():
    from robot_lab.policies.opendoge_legacy import LegacyPolicyState

    state = LegacyPolicyState()
    first = state.observe([1, 2, 3], [4, 8, 12], [0, 0, -1], state.default_angles + 0.2, np.ones(12))
    np.testing.assert_allclose(first[0, :9], [2, 4, 0.75, 1, 2, 3, 0, 0, -1])
    np.testing.assert_allclose(first[0, 9:21], 0.2, atol=1e-7)
    np.testing.assert_allclose(first[0, 21:33], 0.05)
    np.testing.assert_equal(first[0, 33:], 0)
    state.accept_action(np.full((1, 12), 200, dtype=np.float32))
    second = state.observe([0, 0, 0], [0, 0, 0], [0, 0, -1], state.default_angles, np.zeros(12))
    np.testing.assert_equal(second[0, 33:45], 100)
    np.testing.assert_equal(second[0, 45:90], first[0, :45])
    state.reset()
    np.testing.assert_equal(state.history, 0)
    np.testing.assert_equal(state.action, 0)


def test_targets_use_original_stance_and_reject_invalid_policy_outputs():
    from robot_lab.policies.opendoge_legacy import LegacyPolicyState

    state = LegacyPolicyState()
    np.testing.assert_allclose(state.accept_action(np.ones((1, 12))), np.tile([0.3, 0.9, -1.2], 4))
    with pytest.raises(ValueError):
        state.accept_action(np.full((1, 12), np.nan))
    with pytest.raises(ValueError):
        state.accept_action(np.zeros((1, 11)))


def test_migrated_assets_and_policy_are_real_and_self_contained():
    import xml.etree.ElementTree as ET

    from robot_lab.assets import ROBOT_LAB_ASSETS_DIR
    from robot_lab.policies.opendoge_legacy import DEFAULT_POLICY, LegacyOnnxPolicy, LegacyPolicyState

    urdf = ROBOT_LAB_ASSETS_DIR / "Robots/opendoge/urdf/Opendoge.urdf"
    root = ET.parse(urdf).getroot()
    for mesh in root.findall(".//mesh"):
        assert (urdf.parent / mesh.attrib["filename"]).is_file()
    assert len([j for j in root.findall("joint") if j.attrib["type"] == "revolute"]) == 12
    pytest.importorskip("onnxruntime", reason="Install the legacy-policy extra for ONNX inference")
    policy = LegacyOnnxPolicy(DEFAULT_POLICY)
    state = LegacyPolicyState()
    obs = state.observe([0, 0, 0], [0, 0, 0], [0, 0, -1], state.default_angles, np.zeros(12))
    actions = policy(obs)
    assert actions.shape == (1, 12)
    assert np.isfinite(actions).all()
