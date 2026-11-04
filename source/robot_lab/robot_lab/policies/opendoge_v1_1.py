"""LainLab V1.1 exported-policy contract, independent of the mjlab runtime.

See assets/data/Robots/opendoge_v1_1 for source revision and licenses.
ONNX includes the trained observation normalizer; inputs here are physical units.
"""

from pathlib import Path

import numpy as np

from robot_lab.assets import ROBOT_LAB_ASSETS_DIR

ASSET_DIR = ROBOT_LAB_ASSETS_DIR / "Robots/opendoge_v1_1"
JOINT_NAMES = [f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR") for joint in ("hip", "thigh", "calf")]
SKILLS = ("walk", "handstand", "getup", "jump")
OBS_DIMS = {"flat": 48, "getup": 48, "rough": 213, "skills": 57}


def _vector(value, size, name):
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise ValueError(f"Expected {size} finite values for {name}")
    return vector


def imu_linear_velocity(link_velocity, angular_velocity):
    """MuJoCo velocimeter at the V1.1 IMU site (3 cm above base origin)."""
    return _vector(link_velocity, 3, "linear velocity") + np.cross(
        _vector(angular_velocity, 3, "angular velocity"), np.array([0, 0, 0.03], dtype=np.float32)
    )


class PolicyState:
    """Single-robot proprioception, previous raw action and multi-skill clock."""

    def __init__(self, profile="flat", *, skill="walk"):
        if profile not in OBS_DIMS or skill not in SKILLS:
            raise ValueError("Unknown V1.1 policy profile or skill")
        self.profile, self.skill = profile, skill
        self.default_angles = np.array([0, 0.8, -1.6, 0, -0.8, 1.6] * 2, dtype=np.float32)
        self.action = np.zeros(12, dtype=np.float32)
        self.reset()

    def reset(self):
        self.action.fill(0)
        self.phase = 0.0
        self.left_ground = False
        self.grounded_once = False

    def advance(self, dt, base_height, feet_contact):
        """Update command after a control step, as LainLab's command manager does."""
        if not np.isfinite([dt, base_height]).all() or dt < 0:
            raise ValueError("Expected finite height and non-negative time step")
        contact = np.asarray(feet_contact, dtype=bool)
        if contact.shape != (4,):
            raise ValueError("Expected four foot contact flags")
        if self.profile == "skills" and self.skill == "jump":
            self.phase = (self.phase + dt / 2.5) % 1.0
            airborne = not contact.any()
            self.left_ground |= self.grounded_once and airborne and base_height > 0.166
            self.grounded_once |= not airborne

    def observe(self, command, linear_velocity, angular_velocity, gravity, joint_pos, joint_vel, *, height_scan=None):
        command = _vector(command, 3, "command")
        phase = np.array([np.sin(2 * np.pi * self.phase), np.cos(2 * np.pi * self.phase)], dtype=np.float32)
        command_block = command
        if self.profile == "getup" or (self.profile == "skills" and self.skill in ("handstand", "getup")):
            command_block = np.zeros(3, dtype=np.float32)
        elif self.profile == "skills" and self.skill == "jump":
            command_block = np.array([*phase, float(self.left_ground)], dtype=np.float32)
        frame = np.concatenate(
            (
                _vector(linear_velocity, 3, "linear velocity"),
                _vector(angular_velocity, 3, "angular velocity"),
                _vector(gravity, 3, "gravity"),
                _vector(joint_pos, 12, "joint position") - self.default_angles,
                _vector(joint_vel, 12, "joint velocity"),
                self.action,
                command_block,
            )
        )
        if self.profile == "rough":
            frame = np.concatenate((frame, _vector(height_scan, 165, "height scan") / 5.0))
        elif self.profile == "skills":
            token = np.zeros(4, dtype=np.float32)
            token[SKILLS.index(self.skill)] = 1
            twist = command if self.skill == "jump" else np.zeros(3, dtype=np.float32)
            frame = np.concatenate((frame, twist, token, phase))
        return frame.astype(np.float32)[None]

    def accept_action(self, output):
        action = np.asarray(output, dtype=np.float32)
        if action.shape != (1, 12) or not np.isfinite(action).all():
            raise ValueError("Expected finite ONNX actions with shape (1, 12)")
        # LainLab's runner and JointPositionAction both leave actions unclipped.
        self.action[:] = action[0]
        return self.default_angles + self.action * 0.25


class OnnxPolicy:
    """Load an export with its embedded normalizer, without importing mjlab/RSL-RL."""

    def __init__(self, profile="flat", path: str | Path | None = None):
        import onnxruntime as ort

        if profile not in OBS_DIMS:
            raise ValueError("Unknown V1.1 policy profile")
        self.path = Path(path) if path is not None else ASSET_DIR / f"policies/{profile}/policy.onnx"
        self.width = OBS_DIMS[profile]
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.path), sess_options=options, providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (
            len(inputs) != 1
            or len(outputs) != 1
            or inputs[0].shape != [1, self.width]
            or outputs[0].shape != [1, 12]
            or inputs[0].type != "tensor(float)"
        ):
            raise ValueError(f"Expected a V1.1 {profile} model with {self.width} inputs and 12 actions")
        self.input_name = inputs[0].name

    def __call__(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (1, self.width) or not np.isfinite(observation).all():
            raise ValueError(f"Expected finite observation with shape (1, {self.width})")
        return self.session.run(None, {self.input_name: observation})[0]
