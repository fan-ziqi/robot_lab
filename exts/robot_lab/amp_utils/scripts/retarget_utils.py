import numpy as np
from pybullet_utils import transformations
import pybullet

from rsl_rl_extension.datasets import pose3d

POS_SIZE = 3
ROT_SIZE = 4


def get_root_pos(pose):
    return pose[0:POS_SIZE]


def get_root_rot(pose):
    return pose[POS_SIZE : (POS_SIZE + ROT_SIZE)]


def calc_heading(q):
    """Returns the heading of a rotation q, specified as a quaternion.

    The heading represents the rotational component of q along the vertical
    axis (z axis).

    Args:
      q: A quaternion that the heading is to be computed from.

    Returns:
      An angle representing the rotation about the z axis.

    """
    ref_dir = np.array([1, 0, 0])
    rot_dir = pose3d.QuaternionRotatePoint(ref_dir, q)
    heading = np.arctan2(rot_dir[1], rot_dir[0])
    return heading


def calc_heading_rot(q):
    """Return a quaternion representing the heading rotation of q along the vertical axis (z axis).

    Args:
      q: A quaternion that the heading is to be computed from.

    Returns:
      A quaternion representing the rotation about the z axis.

    """
    heading = calc_heading(q)
    q_heading = transformations.quaternion_about_axis(heading, [0, 0, 1])
    return q_heading


def get_joint_limits(robot):
    num_joints = pybullet.getNumJoints(robot)
    joint_limit_low = []
    joint_limit_high = []

    for i in range(num_joints):
        joint_info = pybullet.getJointInfo(robot, i)
        joint_type = joint_info[2]

        if (
            joint_type == pybullet.JOINT_PRISMATIC
            or joint_type == pybullet.JOINT_REVOLUTE
        ):
            joint_limit_low.append(joint_info[8])
            joint_limit_high.append(joint_info[9])

    return joint_limit_low, joint_limit_high


def set_pose(robot, pose):
    num_joints = pybullet.getNumJoints(robot)
    root_pos = get_root_pos(pose)
    root_rot = get_root_rot(pose)
    pybullet.resetBasePositionAndOrientation(robot, root_pos, root_rot)

    for j in range(num_joints):
        j_info = pybullet.getJointInfo(robot, j)
        j_state = pybullet.getJointStateMultiDof(robot, j)

        j_pose_idx = j_info[3]
        j_pose_size = len(j_state[0])
        j_vel_size = len(j_state[1])

        if j_pose_size > 0:
            j_pose = pose[j_pose_idx : (j_pose_idx + j_pose_size)]
            j_vel = np.zeros(j_vel_size)
            pybullet.resetJointStateMultiDof(robot, j, j_pose, j_vel)


def output_motion(frames, out_filename, motion_weight, frame_duration):
    with open(out_filename, "w") as f:
        f.write("{\n")
        f.write('"LoopMode": "Wrap",\n')
        f.write('"FrameDuration": ' + str(frame_duration) + ",\n")
        f.write('"EnableCycleOffsetPosition": true,\n')
        f.write('"EnableCycleOffsetRotation": true,\n')
        f.write('"MotionWeight": ' + str(motion_weight) + ",\n")
        f.write("\n")

        f.write('"Frames":\n')

        f.write("[")
        for i in range(frames.shape[0]):
            curr_frame = frames[i]

            if i != 0:
                f.write(",")
            f.write("\n  [")

            for j in range(frames.shape[1]):
                curr_val = curr_frame[j]
                if j != 0:
                    f.write(", ")
                f.write("%.5f" % curr_val)

            f.write("]")

        f.write("\n]")
        f.write("\n}")

    return


###########################
## Visualization methods ##
###########################
def set_linear_vel_pos(linear_vel, robot_idx, unique_id=None):
    ray_start = np.array([0.0, 0.0, 0.25])
    ray_end = ray_start + linear_vel / 3.0

    if unique_id is not None:
        return pybullet.addUserDebugLine(
            lineFromXYZ=ray_start,
            lineToXYZ=ray_end,
            lineWidth=4,
            replaceItemUniqueId=unique_id,
            lineColorRGB=[1, 0, 0],
            parentObjectUniqueId=robot_idx,
        )
    else:
        return pybullet.addUserDebugLine(
            lineFromXYZ=ray_start,
            lineToXYZ=ray_end,
            lineWidth=4,
            lineColorRGB=[1, 0, 0],
            parentObjectUniqueId=robot_idx,
        )


def set_angular_vel_pos(angular_vel, robot_idx, unique_id=None):
    ray_start = np.array([0.0, 0.0, 0.0])
    ray_end = ray_start + angular_vel

    if unique_id is not None:
        return pybullet.addUserDebugLine(
            lineFromXYZ=ray_start,
            lineToXYZ=ray_end,
            lineWidth=4,
            replaceItemUniqueId=unique_id,
            lineColorRGB=[1, 0, 0],
            parentObjectUniqueId=robot_idx,
        )
    else:
        return pybullet.addUserDebugLine(
            lineFromXYZ=ray_start,
            lineToXYZ=ray_end,
            lineWidth=4,
            lineColorRGB=[1, 0, 0],
            parentObjectUniqueId=robot_idx,
        )


def build_markers(num_markers, special_idx, special_colors):
    marker_radius = 0.01
    markers = []
    for i in range(num_markers):
        if i in special_idx:
            color = special_colors[special_idx.index(i)]
        else:
            color = [1, 1, 1, 1]

        virtual_shape_id = pybullet.createVisualShape(
            shapeType=pybullet.GEOM_SPHERE, radius=marker_radius, rgbaColor=color
        )
        body_id = pybullet.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=virtual_shape_id,
            basePosition=[0, 0, 0],
            useMaximalCoordinates=True,
        )
        markers.append(body_id)
    return markers


def update_marker_pos(marker_pos, marker_ids):
    num_markers = len(marker_ids)
    assert num_markers == marker_pos.shape[0]

    for i in range(num_markers):
        curr_id = marker_ids[i]
        curr_pos = marker_pos[i]
        pybullet.resetBasePositionAndOrientation(curr_id, curr_pos, [0, 0, 0, 1])
    return marker_ids


def build_coordinate_frame():
    dir_line_1 = pybullet.addUserDebugLine([0, 0, 0], [1, 1, 1], [1, 0, 0])
    dir_line_2 = pybullet.addUserDebugLine([0, 0, 0], [1, 1, 1], [1, 0, 0])
    dir_line_3 = pybullet.addUserDebugLine([0, 0, 0], [1, 1, 1], [1, 0, 0])
    return [dir_line_1, dir_line_2, dir_line_3]


def update_coordinate_frame(lines, origin, H, R, scale=0.25):
    for i in range(3):
        color = [0, 0, 0]
        color[i] = 1
        lines[i] = pybullet.addUserDebugLine(
            lineFromXYZ=origin,
            lineToXYZ=H + R.dot(np.array(color)) * scale,
            replaceItemUniqueId=lines[i],
            lineColorRGB=color,
        )
    return lines
