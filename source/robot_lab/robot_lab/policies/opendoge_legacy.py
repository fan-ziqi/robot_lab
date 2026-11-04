"""Original OpenDoge HIM ONNX inference contract (not the V1.1 robot)."""

from pathlib import Path

import numpy as np

from robot_lab.assets import ROBOT_LAB_ASSETS_DIR

JOINT_NAMES = [f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR") for joint in ("hip", "thigh", "calf")]
DEFAULT_POLICY = ROBOT_LAB_ASSETS_DIR / "Robots/opendoge/policies/flat_opendoge_9000_omni.onnx"


class LegacyPolicyState:
    """Single-robot history, newest frame first, reset to zeros between episodes."""

    def __init__(self):
        self.default_angles = np.tile(np.array([0.0, 0.6, -1.5], dtype=np.float32), 4)
        self.history = np.zeros((1, 270), dtype=np.float32)
        self.action = np.zeros(12, dtype=np.float32)

    def reset(self):
        self.history.fill(0)
        self.action.fill(0)

    def observe(self, command, angular_velocity, gravity, joint_pos, joint_vel):
        frame = np.concatenate(
            (
                np.asarray(command) * [2.0, 2.0, 0.25],
                np.asarray(angular_velocity) * 0.25,
                gravity,
                np.asarray(joint_pos) - self.default_angles,
                np.asarray(joint_vel) * 0.05,
                self.action,
            )
        ).astype(np.float32)
        if frame.shape != (45,) or not np.isfinite(frame).all():
            raise ValueError("Expected one finite 45-value OpenDoge observation")
        self.history[:, 45:] = self.history[:, :-45].copy()
        self.history[0, :45] = frame
        return np.clip(self.history, -100.0, 100.0).copy()

    def accept_action(self, output):
        action = np.asarray(output, dtype=np.float32)
        if action.shape != (1, 12) or not np.isfinite(action).all():
            raise ValueError("Expected finite ONNX actions with shape (1, 12)")
        self.action[:] = np.clip(action[0], -100.0, 100.0)
        return self.default_angles + self.action * 0.30


class LegacyOnnxPolicy:
    """CPU ONNX Runtime inference; validate the exported model before launching simulation."""

    def __init__(self, path: str | Path = DEFAULT_POLICY):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1 or inputs[0].shape[-1] != 270 or outputs[0].shape[-1] != 12:
            raise ValueError("Expected a legacy HIM model with 270 inputs and 12 actions")
        self.input_name = inputs[0].name

    def __call__(self, observation):
        return self.session.run(None, {self.input_name: np.asarray(observation, dtype=np.float32)})[0]
