# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
Humanoid Motion Data Simple Converter

This script only converts CSV motion data directly to NPZ format.

USAGE:
    python csv2npz.py --file motion.csv --urdf robot.urdf --mesh-dir meshes --output motion.npz

DESCRIPTION:
    1. Read raw CSV motion data
    2. Directly save as NPZ format, keeping the original frame rate

INPUT:
    - CSV file (joints + root pose)
    - URDF file
    - mesh directory

OUTPUT:
    - NPZ file, same content as original data_convert.py

REQUIREMENTS:
    - numpy
    - pandas
    - pinocchio
"""

import argparse
from pathlib import Path

import numpy as np


def quaternion_inverse(q):
    # Input q: (w, x, y, z), returns its inverse.
    w, x, y, z = q
    norm_sq = w * w + x * x + y * y + z * z
    if norm_sq < 1e-8:
        norm_sq = 1e-8
    return np.array([w, -x, -y, -z], dtype=q.dtype) / norm_sq


def quaternion_multiply(q1, q2):
    # Input/output: (w, x, y, z)
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z], dtype=q1.dtype)


def compute_angular_velocity(q_prev, q_next, dt, eps=1e-8):
    """
    Compute angular velocity from adjacent quaternions (w, x, y, z):
      - Relative rotation q_rel = inv(q_prev) * q_next
      - Extract rotation angle and axis from q_rel
      - Return (angle / dt) * axis
    """
    q_inv = quaternion_inverse(q_prev)
    # World angular velocity: the relative rotation is expressed in world axes.
    q_rel = quaternion_multiply(q_next, q_inv)
    norm_q_rel = np.linalg.norm(q_rel)
    if norm_q_rel < eps:
        return np.zeros(3, dtype=np.float32)
    q_rel /= norm_q_rel
    if q_rel[0] < 0:
        q_rel = -q_rel
    w = np.clip(q_rel[0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    sin_half = np.sqrt(1.0 - w * w)
    if sin_half < eps:
        return np.zeros(3, dtype=np.float32)
    axis = q_rel[1:] / sin_half
    return (angle / dt) * axis


def build_pin_robot(urdf_path, mesh_dir):
    """
    Load URDF file and construct a pin.RobotWrapper with free-flyer.
    Args:
        urdf_path: Path to the URDF file
        mesh_dir: Directory containing associated mesh files
    Returns:
        robot (pin.RobotWrapper)
    """
    import pinocchio as pin

    robot = pin.RobotWrapper.BuildFromURDF(urdf_path, mesh_dir, pin.JointModelFreeFlyer())
    return robot


def main():
    parser = argparse.ArgumentParser(description="Convert a G1 motion CSV to an AMP archive.")
    parser.add_argument("--file", required=True, help="Input CSV: root xyz, quaternion xyzw, then 29 joints.")
    parser.add_argument("--urdf", required=True, help="G1 URDF used by the retargeted motion.")
    parser.add_argument("--mesh-dir", required=True, help="Directory containing the URDF mesh packages.")
    parser.add_argument("--output", required=True, help="Output NPZ path.")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    args = parser.parse_args()
    if args.fps <= 0 or args.start_frame < 0:
        parser.error("fps must be positive and start-frame must be nonnegative")
    import pandas as pd
    import pinocchio as pin

    csv_file = args.file
    df = pd.read_csv(csv_file, header=None)
    start_idx = args.start_frame
    end_idx = args.end_frame
    data_orig = df.iloc[start_idx:end_idx].to_numpy(dtype=np.float32)
    N = data_orig.shape[0]
    if N < 2 or data_orig.shape[1] != 36:
        raise ValueError("Motion CSV must contain at least two selected frames and 36 columns.")
    print(f"Loading CSV: {csv_file}, frame range [{start_idx}:{end_idx}], total {N} frames.")

    # Root and joint data
    root_data = data_orig[:, :7]  # (N, 7)
    joint_data = data_orig[:, 7:]  # (N, D)

    # Original sampling rate
    fps = args.fps
    dt = 1.0 / fps

    # Joint names
    joint_names = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
    dof_names = np.array(joint_names, dtype=np.str_)

    # Joint positions
    dof_positions = joint_data.copy()  # (N, D)

    # Joint velocities
    dof_velocities = np.zeros_like(dof_positions)
    dof_velocities[1:-1] = (dof_positions[2:] - dof_positions[:-2]) / (2 * dt)
    dof_velocities[0] = (dof_positions[1] - dof_positions[0]) / dt
    dof_velocities[-1] = (dof_positions[-1] - dof_positions[-2]) / dt

    # Body link names
    body_names = [
        "pelvis",
        "left_shoulder_yaw_link",
        "right_shoulder_yaw_link",
        "left_elbow_link",
        "right_elbow_link",
        "right_rubber_hand",
        "left_rubber_hand",
        "right_ankle_roll_link",
        "left_ankle_roll_link",
        "torso_link",
        "right_hip_yaw_link",
        "left_hip_yaw_link",
        "right_knee_link",
        "left_knee_link",
    ]
    body_names = np.array(body_names, dtype=np.str_)
    B = len(body_names)

    body_positions = np.zeros((N, B, 3), dtype=np.float32)
    body_rotations = np.zeros((N, B, 4), dtype=np.float32)

    # Pinocchio forward kinematics
    urdf_path = args.urdf
    mesh_dir = args.mesh_dir
    robot = build_pin_robot(urdf_path, mesh_dir)
    model = robot.model
    data_pk = robot.data
    nq = model.nq
    if (7 + joint_data.shape[1]) != nq:
        print(
            f"Warning: CSV columns={7 + joint_data.shape[1]}, but pinocchio nq={nq}, may need to check or adjust script"
            " parsing."
        )
    q_pin = pin.neutral(model)
    for i in range(N):
        q_pin[0:3] = root_data[i, 0:3]
        q_pin[3:7] = root_data[i, 3:7]
        dofD = joint_data.shape[1]
        q_pin[7 : 7 + dofD] = joint_data[i, :]
        pin.forwardKinematics(model, data_pk, q_pin)
        pin.updateFramePlacements(model, data_pk)
        for j, link_name in enumerate(body_names):
            fid = model.getFrameId(link_name)
            link_tf = data_pk.oMf[fid]
            body_positions[i, j, :] = link_tf.translation
            quat_xyzw = pin.Quaternion(link_tf.rotation)
            body_rotations[i, j, :] = np.array([quat_xyzw.w, quat_xyzw.x, quat_xyzw.y, quat_xyzw.z], dtype=np.float32)

    # Linear velocities
    body_linear_velocities = np.zeros_like(body_positions)
    body_linear_velocities[1:-1] = (body_positions[2:] - body_positions[:-2]) / (2 * dt)
    body_linear_velocities[0] = (body_positions[1] - body_positions[0]) / dt
    body_linear_velocities[-1] = (body_positions[-1] - body_positions[-2]) / dt

    # Angular velocities
    body_angular_velocities = np.zeros((N, B, 3), dtype=np.float32)
    for j in range(B):
        quats = body_rotations[:, j, :]
        angular_vels = np.zeros((N, 3), dtype=np.float32)
        if N > 1:
            angular_vels[0] = compute_angular_velocity(quats[0], quats[1], dt)
            angular_vels[-1] = compute_angular_velocity(quats[-2], quats[-1], dt)
        for k in range(1, N - 1):
            av1 = compute_angular_velocity(quats[k - 1], quats[k], dt)
            av2 = compute_angular_velocity(quats[k], quats[k + 1], dt)
            angular_vels[k] = 0.5 * (av1 + av2)
        body_angular_velocities[:, j, :] = angular_vels

    # Save
    data_dict = {
        "fps": fps,
        "dof_names": dof_names,
        "body_names": body_names,
        "dof_positions": dof_positions,
        "dof_velocities": dof_velocities,
        "body_positions": body_positions,
        "body_rotations": body_rotations[..., [1, 2, 3, 0]],
        "quaternion_order": "xyzw",
        "body_linear_velocities": body_linear_velocities,
        "body_angular_velocities": body_angular_velocities,
    }
    out_filename = Path(args.output)
    out_filename.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_filename, **data_dict)
    print(f"Conversion completed, data saved to {out_filename}")
    print("fps:", fps)
    print("dof_names:", dof_names.shape)
    print("body_names:", body_names.shape)
    print("dof_positions:", dof_positions.shape)
    print("dof_velocities:", dof_velocities.shape)
    print("body_positions:", body_positions.shape)
    print("body_rotations:", body_rotations.shape)
    print("body_linear_velocities:", body_linear_velocities.shape)
    print("body_angular_velocities:", body_angular_velocities.shape)


if __name__ == "__main__":
    main()
