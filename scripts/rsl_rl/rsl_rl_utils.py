# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import torch

import carb

import isaaclab.utils.math as math_utils


def sub_keyboard_event(event, cmd_vel, lin_vel=1.0, ang_vel=1.0) -> bool:
    """
    This function is subscribed to keyboard events and updates the velocity commands.
    """
    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        # Update the velocity commands for the first environment (0th index)
        if event.input.name == "W":
            cmd_vel[0] = torch.tensor([lin_vel, 0, 0], dtype=torch.float32)
        elif event.input.name == "S":
            cmd_vel[0] = torch.tensor([-lin_vel, 0, 0], dtype=torch.float32)
        elif event.input.name == "A":
            cmd_vel[0] = torch.tensor([0, lin_vel, 0], dtype=torch.float32)
        elif event.input.name == "D":
            cmd_vel[0] = torch.tensor([0, -lin_vel, 0], dtype=torch.float32)
        elif event.input.name == "J":
            cmd_vel[0] = torch.tensor([0, 0, ang_vel], dtype=torch.float32)
        elif event.input.name == "L":
            cmd_vel[0] = torch.tensor([0, 0, -ang_vel], dtype=torch.float32)

    # Reset velocity commands on key release
    elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
        cmd_vel.zero_()

    return True


def camera_follow(env):
    if not hasattr(camera_follow, "smooth_camera_positions"):
        camera_follow.smooth_camera_positions = []
    robot_pos = env.unwrapped.scene["robot"].data.root_pos_w[0]
    robot_quat = env.unwrapped.scene["robot"].data.root_quat_w[0]
    camera_offset = torch.tensor([-5.0, 0.0, 1.0], dtype=torch.float32, device=env.device)
    camera_pos = math_utils.transform_points(
        camera_offset.unsqueeze(0), pos=robot_pos.unsqueeze(0), quat=robot_quat.unsqueeze(0)
    ).squeeze(0)
    camera_pos[2] = torch.clamp(camera_pos[2], min=0.0)
    window_size = 50
    camera_follow.smooth_camera_positions.append(camera_pos)
    if len(camera_follow.smooth_camera_positions) > window_size:
        camera_follow.smooth_camera_positions.pop(0)
    smooth_camera_pos = torch.mean(torch.stack(camera_follow.smooth_camera_positions), dim=0)
    env.unwrapped.viewport_camera_controller.set_view_env_index(env_index=0)
    env.unwrapped.viewport_camera_controller.update_view_location(
        eye=smooth_camera_pos.cpu().numpy(), lookat=robot_pos.cpu().numpy()
    )
