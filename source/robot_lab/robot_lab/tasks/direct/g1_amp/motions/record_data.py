"""
Motion Data Recording Tool

This script provides functionality for recording and managing motion data from robots.
It includes classes and functions for:
- Recording motion data in real-time
- Saving motion data to npz files
- Loading motion data from npz files
- Managing motion data structure
- Converting joint orders between different formats

Usage:
    As a module:
        from record_data import MotionRecorder, MotionData
        
        recorder = MotionRecorder(robot, motion_dof_indices, fps=60)
        recorder.start_recording()
        # ... record frames ...
        recorder.stop_recording()
        recorder.save_data("output.npz")
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from typing import List, Dict, Optional
import os
from dataclasses import dataclass
from datetime import datetime

@dataclass
class MotionData:
    fps: int
    dof_names: np.ndarray
    body_names: np.ndarray
    dof_positions: np.ndarray
    dof_velocities: np.ndarray
    body_positions: np.ndarray
    body_rotations: np.ndarray
    body_linear_velocities: np.ndarray
    body_angular_velocities: np.ndarray


def smooth_motion_data(motion_data: MotionData, window_size: int = 3) -> MotionData:
    """Apply smoothing to motion data to reduce jitter.
    
    Args:
        motion_data: Original motion data
        window_size: Size of smoothing window
        
    Returns:
        Smoothed motion data
    """
    # Create a copy of the data
    smoothed = MotionData(
        fps=motion_data.fps,
        dof_names=motion_data.dof_names,
        body_names=motion_data.body_names,
        dof_positions=motion_data.dof_positions.copy(),
        dof_velocities=motion_data.dof_velocities.copy(),
        body_positions=motion_data.body_positions.copy(),
        body_rotations=motion_data.body_rotations.copy(),
        body_linear_velocities=motion_data.body_linear_velocities.copy(),
        body_angular_velocities=motion_data.body_angular_velocities.copy()
    )
    
    # Apply smoothing to positions and velocities
    for i in range(window_size, len(smoothed.dof_positions) - window_size):
        # Smooth joint positions
        smoothed.dof_positions[i] = np.mean(
            motion_data.dof_positions[i-window_size:i+window_size+1], 
            axis=0
        )
        # Smooth joint velocities
        smoothed.dof_velocities[i] = np.mean(
            motion_data.dof_velocities[i-window_size:i+window_size+1], 
            axis=0
        )
        # Smooth body positions
        smoothed.body_positions[i] = np.mean(
            motion_data.body_positions[i-window_size:i+window_size+1], 
            axis=0
        )
        # Smooth body rotations (using quaternion averaging)
        smoothed.body_rotations[i] = np.mean(
            motion_data.body_rotations[i-window_size:i+window_size+1], 
            axis=0
        )
        # Normalize quaternions
        smoothed.body_rotations[i] = smoothed.body_rotations[i] / np.linalg.norm(
            smoothed.body_rotations[i], 
            axis=-1, 
            keepdims=True
        )
    
    return smoothed

class MotionRecorder:
    """Motion data recorder for robot movements"""

    def __init__(self, robot, dof_names_to_record: List[str],
                 fps: int = 60, device: str = "cuda", smoothing_window: int = 3):
        """
        Args:
            robot: Robot object for metadata
            dof_names_to_record: List of joint names to record.
            fps: Target frame rate (used to calculate dt)
            device: Device to use (cuda/cpu)
            smoothing_window: Window size for motion smoothing
        """
        self.robot = robot
        self.fps = int(fps)
        self.dt = 1.0 / self.fps
        self.device = device
        self.smoothing_window = smoothing_window

        # Get names directly from robot to avoid empty arrays
        self.dof_names = np.asarray(dof_names_to_record, dtype=np.str_)
        self.body_names = np.asarray(robot.body_names, dtype=np.str_)

        try:
            self.root_body_idx = list(robot.body_names).index('pelvis')
        except ValueError:
            raise ValueError("The robot asset must have a body named 'pelvis' to be used with MotionRecorder.")

        # Create a mapping from the robot's full joint list to the desired recording list
        robot_dof_map = {name: i for i, name in enumerate(robot.joint_names)}
        self.dof_indices = np.array([robot_dof_map[name] for name in dof_names_to_record], dtype=np.int32)
        
        self.recorded_frames: list[dict] = []
        self.is_recording = False

    # 其余接口保持不变 --------------------------
    def start_recording(self):
        self.recorded_frames.clear()
        self.is_recording = True

    def stop_recording(self):
        self.is_recording = False

    def record_frame(self, frame_idx: int):
        """Record single frame data (no need to pass robot/motion_dof_indices externally)"""
        if not self.is_recording:
            return None

        # --- Joints ---
        dof_pos = self.robot.data.joint_pos[0].cpu().numpy()
        dof_vel = self.robot.data.joint_vel[0].cpu().numpy()

        # --- Root ---
        root_pos = self.robot.data.body_pos_w[0, self.root_body_idx].cpu().numpy()
        root_rot = self.robot.data.body_quat_w[0, self.root_body_idx].cpu().numpy()
        root_lin_vel = self.robot.data.body_lin_vel_w[0, self.root_body_idx].cpu().numpy()
        root_ang_vel = self.robot.data.body_ang_vel_w[0, self.root_body_idx].cpu().numpy()

        # --- Full body ---
        body_pos = self.robot.data.body_pos_w[0].cpu().numpy()
        body_rot = self.robot.data.body_quat_w[0].cpu().numpy()
        body_lin_vel = self.robot.data.body_lin_vel_w[0].cpu().numpy()
        body_ang_vel = self.robot.data.body_ang_vel_w[0].cpu().numpy()

        self.recorded_frames.append(
            dict(
                frame_idx=frame_idx,
                dof_positions=dof_pos[self.dof_indices],
                dof_velocities=dof_vel[self.dof_indices],
                root_position=root_pos,
                root_rotation=root_rot,
                root_linear_velocity=root_lin_vel,
                root_angular_velocity=root_ang_vel,
                body_positions=body_pos,
                body_rotations=body_rot,
                body_linear_velocities=body_lin_vel,
                body_angular_velocities=body_ang_vel,
            )
        )
        return self.recorded_frames[-1]

    def get_recorded_data(self) -> MotionData | None:
        if not self.recorded_frames:
            return None

        n = len(self.recorded_frames)
        d = len(self.dof_indices)
        b = len(self.body_names)

        # Pre-allocate arrays
        dof_pos = np.zeros((n, d), np.float32)
        dof_vel = np.zeros_like(dof_pos)
        body_pos = np.zeros((n, b, 3), np.float32)
        body_rot = np.zeros((n, b, 4), np.float32)
        body_lin = np.zeros_like(body_pos)
        body_ang = np.zeros_like(body_pos)

        for i, f in enumerate(self.recorded_frames):
            dof_pos[i] = f["dof_positions"]
            dof_vel[i] = f["dof_velocities"]
            body_pos[i] = f["body_positions"]
            body_rot[i] = f["body_rotations"]
            body_lin[i] = f["body_linear_velocities"]
            body_ang[i] = f["body_angular_velocities"]

        motion_data = MotionData(
            fps=self.fps,
            dof_names=self.dof_names,
            body_names=self.body_names,
            dof_positions=dof_pos,
            dof_velocities=dof_vel,
            body_positions=body_pos,
            body_rotations=body_rot,
            body_linear_velocities=body_lin,
            body_angular_velocities=body_ang,
        )
        
        # Apply smoothing if window size > 1
        if self.smoothing_window > 1:
            motion_data = smooth_motion_data(motion_data, self.smoothing_window)
            
        return motion_data

    def save_data(self, out_file: str, data: MotionData | None = None) -> bool:
        if data is None:
            data = self.get_recorded_data()
        if data is None:
            print("No data to save")
            return False

        os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
        try:
            np.savez(
                out_file,
                fps=data.fps,
                dof_names=data.dof_names,
                body_names=data.body_names,
                dof_positions=data.dof_positions,
                dof_velocities=data.dof_velocities,
                body_positions=data.body_positions,
                body_rotations=data.body_rotations,
                body_linear_velocities=data.body_linear_velocities,
                body_angular_velocities=data.body_angular_velocities,
                record_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
            print(f"Data saved to {out_file}")
            return True
        except Exception as e:
            print("Error saving data:", e)
            return False


def load_motion_data(file_path: str) -> Optional[MotionData]:
    """Load motion data file
    
    Args:
        file_path: Path to .npz file
        
    Returns:
        MotionData object if successful, None otherwise
    """
    try:
        data = np.load(file_path)
        return MotionData(
            fps=int(data['fps']),
            dof_names=data['dof_names'],
            body_names=data['body_names'],
            dof_positions=data['dof_positions'],
            dof_velocities=data['dof_velocities'],
            body_positions=data['body_positions'],
            body_rotations=data['body_rotations'],
            body_linear_velocities=data['body_linear_velocities'],
            body_angular_velocities=data['body_angular_velocities']
        )
    except Exception as e:
        print(f"Error loading data file: {e}")
        return None
