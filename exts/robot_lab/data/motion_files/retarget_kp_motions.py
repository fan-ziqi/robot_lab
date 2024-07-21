"""Retarget motions from keypoint (.txt) files."""

import os
import inspect

import time

import numpy as np
from pyquaternion import Quaternion

import isaacgym
from legged_gym.envs import *
import legged_gym.utils.kinematics.urdf as pk

from rsl_rl.datasets import pose3d
from pybullet_utils import transformations
import pybullet
import pybullet_data as pd

from retarget_utils import *
import retarget_config as config

POS_SIZE = 3
ROT_SIZE = 4
JOINT_POS_SIZE = 12
TAR_TOE_POS_LOCAL_SIZE = 12
LINEAR_VEL_SIZE = 3
ANGULAR_VEL_SIZE = 3
JOINT_VEL_SIZE = 12
TAR_TOE_VEL_LOCAL_SIZE = 12

DEFAULT_ROT = np.array([0, 0, 0, 1])
FORWARD_DIR = np.array([1, 0, 0])

GROUND_URDF_FILENAME = "plane_implicit.urdf"

# reference motion
FRAME_DURATION = 0.01677
REF_COORD_ROT = transformations.quaternion_from_euler(0.5 * np.pi, 0, 0)  # 坐标系变换
REF_POS_OFFSET = np.array([0, 0, 0])
REF_ROOT_ROT = transformations.quaternion_from_euler(0, 0, 0.47 * np.pi)  #

REF_PELVIS_JOINT_ID = 0
REF_NECK_JOINT_ID = 3

REF_TOE_JOINT_IDS = [10, 15, 19, 23]
REF_HIP_JOINT_IDS = [6, 11, 16, 20]

chain_foot_fl = pk.build_serial_chain_from_urdf(
    open(config.URDF_FILENAME).read(), config.FL_FOOT_NAME
)
chain_foot_fr = pk.build_serial_chain_from_urdf(
    open(config.URDF_FILENAME).read(), config.FR_FOOT_NAME
)
chain_foot_rl = pk.build_serial_chain_from_urdf(
    open(config.URDF_FILENAME).read(), config.HL_FOOT_NAME
)
chain_foot_rr = pk.build_serial_chain_from_urdf(
    open(config.URDF_FILENAME).read(), config.HR_FOOT_NAME
)


def build_markers(num_markers):
    marker_radius = 0.02

    markers = []
    for i in range(num_markers):
        if (
            (i == REF_NECK_JOINT_ID)
            or (i == REF_PELVIS_JOINT_ID)
            or (i in REF_HIP_JOINT_IDS)
        ):
            col = [0, 0, 1, 1]
        elif i in REF_TOE_JOINT_IDS:
            col = [1, 0, 0, 1]
        else:
            col = [0, 1, 0, 1]

        virtual_shape_id = pybullet.createVisualShape(
            shapeType=pybullet.GEOM_SPHERE, radius=marker_radius, rgbaColor=col
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


def get_joint_pose(pose):
    return pose[(POS_SIZE + ROT_SIZE) : (POS_SIZE + ROT_SIZE + JOINT_POS_SIZE)]


def get_tar_toe_pos_local(pose):
    return pose[
        (POS_SIZE + ROT_SIZE + JOINT_POS_SIZE) : (
            POS_SIZE + ROT_SIZE + JOINT_POS_SIZE + TAR_TOE_POS_LOCAL_SIZE
        )
    ]


def get_linear_vel(pose):
    return pose[
        (POS_SIZE + ROT_SIZE + JOINT_POS_SIZE + TAR_TOE_POS_LOCAL_SIZE) : (
            POS_SIZE
            + ROT_SIZE
            + JOINT_POS_SIZE
            + TAR_TOE_POS_LOCAL_SIZE
            + LINEAR_VEL_SIZE
        )
    ]


def get_angular_vel(pose):
    return pose[
        (
            POS_SIZE
            + ROT_SIZE
            + JOINT_POS_SIZE
            + TAR_TOE_POS_LOCAL_SIZE
            + LINEAR_VEL_SIZE
        ) : (
            POS_SIZE
            + ROT_SIZE
            + JOINT_POS_SIZE
            + TAR_TOE_POS_LOCAL_SIZE
            + LINEAR_VEL_SIZE
            + ANGULAR_VEL_SIZE
        )
    ]


def get_joint_vel(pose):
    return pose[
        (
            POS_SIZE
            + ROT_SIZE
            + JOINT_POS_SIZE
            + TAR_TOE_POS_LOCAL_SIZE
            + LINEAR_VEL_SIZE
            + ANGULAR_VEL_SIZE
        ) : (
            POS_SIZE
            + ROT_SIZE
            + JOINT_POS_SIZE
            + TAR_TOE_POS_LOCAL_SIZE
            + LINEAR_VEL_SIZE
            + ANGULAR_VEL_SIZE
            + JOINT_VEL_SIZE
        )
    ]


def get_tar_toe_vel_local(pose):
    return pose[
        (
            POS_SIZE
            + ROT_SIZE
            + JOINT_POS_SIZE
            + TAR_TOE_POS_LOCAL_SIZE
            + LINEAR_VEL_SIZE
            + ANGULAR_VEL_SIZE
            + JOINT_VEL_SIZE
        ) : (
            POS_SIZE
            + ROT_SIZE
            + JOINT_POS_SIZE
            + TAR_TOE_POS_LOCAL_SIZE
            + LINEAR_VEL_SIZE
            + ANGULAR_VEL_SIZE
            + JOINT_VEL_SIZE
            + TAR_TOE_VEL_LOCAL_SIZE
        )
    ]


def set_root_pos(root_pos, pose):
    pose[0:POS_SIZE] = root_pos
    return


def set_root_rot(root_rot, pose):
    pose[POS_SIZE : (POS_SIZE + ROT_SIZE)] = root_rot
    return


def set_joint_pose(joint_pose, pose):
    pose[(POS_SIZE + ROT_SIZE) :] = joint_pose
    return


def set_maker_pos(marker_pos, marker_ids):
    num_markers = len(marker_ids)
    assert num_markers == marker_pos.shape[0]

    for i in range(num_markers):
        curr_id = marker_ids[i]
        curr_pos = marker_pos[i]

        pybullet.resetBasePositionAndOrientation(curr_id, curr_pos, DEFAULT_ROT)

    return


def set_foot_marker_pos(marker_pos, robot_idx, unique_ids=None):
    marker_pos = marker_pos.reshape(4, 3)
    new_unique_ids = []

    for foot_pos, unique_id in zip(marker_pos, unique_ids):
        if unique_id is not None:
            new_unique_ids.append(
                pybullet.addUserDebugLine(
                    lineFromXYZ=foot_pos - np.array([0.0, 0.0, 0.04]),
                    lineToXYZ=foot_pos + np.array([0.0, 0.0, 0.04]),
                    lineWidth=4,
                    replaceItemUniqueId=unique_id,
                    lineColorRGB=[1, 0, 0],
                    parentObjectUniqueId=robot_idx,
                )
            )
        else:
            new_unique_ids.append(
                pybullet.addUserDebugLine(
                    lineFromXYZ=foot_pos - np.array([0.0, 0.0, 0.04]),
                    lineToXYZ=foot_pos + np.array([0.0, 0.0, 0.04]),
                    lineWidth=4,
                    lineColorRGB=[1, 0, 0],
                    parentObjectUniqueId=robot_idx,
                )
            )
    return new_unique_ids


def process_ref_joint_pos_data(joint_pos):
    proc_pos = joint_pos.copy()
    num_pos = joint_pos.shape[0]

    for i in range(num_pos):
        curr_pos = proc_pos[i]
        curr_pos = pose3d.QuaternionRotatePoint(curr_pos, REF_COORD_ROT)
        curr_pos = pose3d.QuaternionRotatePoint(curr_pos, REF_ROOT_ROT)
        curr_pos = curr_pos * config.REF_POS_SCALE + REF_POS_OFFSET
        proc_pos[i] = curr_pos

    return proc_pos


def retarget_root_pose(ref_joint_pos):
    pelvis_pos = ref_joint_pos[REF_PELVIS_JOINT_ID]
    neck_pos = ref_joint_pos[REF_NECK_JOINT_ID]

    left_shoulder_pos = ref_joint_pos[REF_HIP_JOINT_IDS[0]]
    right_shoulder_pos = ref_joint_pos[REF_HIP_JOINT_IDS[1]]
    left_hip_pos = ref_joint_pos[REF_HIP_JOINT_IDS[2]]
    right_hip_pos = ref_joint_pos[REF_HIP_JOINT_IDS[3]]

    forward_dir = neck_pos - pelvis_pos
    forward_dir += config.FORWARD_DIR_OFFSET
    forward_dir = forward_dir / np.linalg.norm(forward_dir)

    delta_shoulder = left_shoulder_pos - right_shoulder_pos
    delta_hip = left_hip_pos - right_hip_pos
    dir_shoulder = delta_shoulder / np.linalg.norm(delta_shoulder)
    dir_hip = delta_hip / np.linalg.norm(delta_hip)

    left_dir = 0.5 * (dir_shoulder + dir_hip)

    up_dir = np.cross(forward_dir, left_dir)
    up_dir = up_dir / np.linalg.norm(up_dir)

    left_dir = np.cross(up_dir, forward_dir)
    left_dir[2] = 0.0  # make the base more stable
    left_dir = left_dir / np.linalg.norm(left_dir)

    rot_mat = np.array(
        [
            [forward_dir[0], left_dir[0], up_dir[0], 0],
            [forward_dir[1], left_dir[1], up_dir[1], 0],
            [forward_dir[2], left_dir[2], up_dir[2], 0],
            [0, 0, 0, 1],
        ]
    )

    root_pos = 0.5 * (pelvis_pos + neck_pos)
    # root_pos = 0.25 * (left_shoulder_pos + right_shoulder_pos + left_hip_pos + right_hip_pos)
    root_rot = transformations.quaternion_from_matrix(rot_mat)
    root_rot = transformations.quaternion_multiply(root_rot, config.INIT_ROT)
    root_rot = root_rot / np.linalg.norm(root_rot)

    return root_pos, root_rot


def retarget_pose(robot, default_pose, ref_joint_pos):
    # 获取关节限制
    joint_lim_low, joint_lim_high = get_joint_limits(robot)
    joint_lim_low = [i * -1 for i in joint_lim_high]
    joint_lim_high = [i * -1 for i in joint_lim_high]
    # print(joint_lim_low)
    # print(joint_lim_high)

    root_pos, root_rot = retarget_root_pose(ref_joint_pos)
    root_pos += config.SIM_ROOT_OFFSET

    pybullet.resetBasePositionAndOrientation(robot, root_pos, root_rot)

    inv_init_rot = transformations.quaternion_inverse(config.INIT_ROT)
    heading_rot = calc_heading_rot(
        transformations.quaternion_multiply(root_rot, inv_init_rot)
    )

    tar_toe_pos = []
    for i in range(len(REF_TOE_JOINT_IDS)):
        ref_toe_id = REF_TOE_JOINT_IDS[i]
        ref_hip_id = REF_HIP_JOINT_IDS[i]
        sim_hip_id = config.SIM_HIP_JOINT_IDS[i]
        toe_offset_local = config.SIM_TOE_OFFSET_LOCAL[i]
        ref_toe_pos = ref_joint_pos[ref_toe_id]
        ref_hip_pos = ref_joint_pos[ref_hip_id]

        hip_link_state = pybullet.getLinkState(
            robot, sim_hip_id, computeForwardKinematics=True
        )
        sim_hip_pos = np.array(hip_link_state[4])

        toe_offset_world = pose3d.QuaternionRotatePoint(toe_offset_local, heading_rot)

        ref_hip_toe_delta = ref_toe_pos - ref_hip_pos
        sim_tar_toe_pos = sim_hip_pos + ref_hip_toe_delta
        sim_tar_toe_pos[2] = ref_toe_pos[2]
        sim_tar_toe_pos += toe_offset_world

        tar_toe_pos.append(sim_tar_toe_pos)

    joint_pose = pybullet.calculateInverseKinematics2(
        robot,
        config.SIM_TOE_JOINT_IDS,
        tar_toe_pos,
        jointDamping=config.JOINT_DAMPING,
        lowerLimits=joint_lim_low,
        upperLimits=joint_lim_high,
        restPoses=default_pose,
    )
    joint_pose = np.array(joint_pose)
    # print(joint_pose)

    tar_toe_pos_local = np.squeeze(
        np.concatenate(
            [
                chain_foot_fl.forward_kinematics(joint_pose[:3]).get_matrix()[:, :3, 3],
                chain_foot_fr.forward_kinematics(joint_pose[3:6]).get_matrix()[
                    :, :3, 3
                ],
                chain_foot_rl.forward_kinematics(joint_pose[6:9]).get_matrix()[
                    :, :3, 3
                ],
                chain_foot_rr.forward_kinematics(joint_pose[9:12]).get_matrix()[
                    :, :3, 3
                ],
            ],
            axis=-1,
        )
    )

    pose = np.concatenate([root_pos, root_rot, joint_pose, tar_toe_pos_local])

    return pose


def update_camera(robot):
    base_pos = np.array(pybullet.getBasePositionAndOrientation(robot)[0])
    [yaw, pitch, dist] = pybullet.getDebugVisualizerCamera()[8:11]
    pybullet.resetDebugVisualizerCamera(dist, yaw, pitch, base_pos)
    return


def load_ref_data(JOINT_POS_FILENAME, FRAME_START, FRAME_END):
    joint_pos_data = np.loadtxt(JOINT_POS_FILENAME, delimiter=",")

    start_frame = 0 if (FRAME_START is None) else FRAME_START
    end_frame = joint_pos_data.shape[0] if (FRAME_END is None) else FRAME_END
    joint_pos_data = joint_pos_data[start_frame:end_frame]

    return joint_pos_data


def retarget_motion(robot, joint_pos_data):
    num_frames = joint_pos_data.shape[0]

    time_between_frames = FRAME_DURATION

    for f in range(num_frames - 1):
        # Current robot pose.
        ref_joint_pos = joint_pos_data[f]
        # ref_joint_pos = np.reshape(ref_joint_pos, [-1, POS_SIZE])
        # ref_joint_pos = process_ref_joint_pos_data(ref_joint_pos)
        curr_pose = retarget_pose(robot, config.DEFAULT_JOINT_POSE, ref_joint_pos)
        set_pose(robot, curr_pose)

        # Next robot pose.
        next_ref_joint_pos = joint_pos_data[f + 1]
        # next_ref_joint_pos = np.reshape(next_ref_joint_pos, [-1, POS_SIZE])
        # next_ref_joint_pos = process_ref_joint_pos_data(next_ref_joint_pos)
        next_pose = retarget_pose(robot, config.DEFAULT_JOINT_POSE, next_ref_joint_pos)

        if f == 0:
            pose_size = (
                curr_pose.shape[-1]
                + LINEAR_VEL_SIZE
                + ANGULAR_VEL_SIZE
                + JOINT_POS_SIZE
                + TAR_TOE_VEL_LOCAL_SIZE
            )
            new_frames = np.zeros([num_frames - 1, pose_size])

        # Linear velocity in base frame.
        del_linear_vel = (
            np.array((get_root_pos(next_pose) - get_root_pos(curr_pose)))
            / time_between_frames
        )
        r = pybullet.getMatrixFromQuaternion(get_root_rot(curr_pose))
        del_linear_vel = np.matmul(del_linear_vel, np.array(r).reshape(3, 3))

        # Angular velocity in base frame.
        curr_quat = get_root_rot(curr_pose)
        next_quat = get_root_rot(next_pose)
        diff_quat = Quaternion.distance(
            Quaternion(curr_quat[3], curr_quat[0], curr_quat[1], curr_quat[2]),
            Quaternion(next_quat[3], next_quat[0], next_quat[1], next_quat[2]),
        )
        del_angular_vel = pybullet.getDifferenceQuaternion(
            get_root_rot(curr_pose), get_root_rot(next_pose)
        )
        axis, _ = pybullet.getAxisAngleFromQuaternion(del_angular_vel)
        del_angular_vel = np.array(axis) * (diff_quat * 2) / time_between_frames
        # del_angular_vel = pybullet.getDifferenceQuaternion(get_root_rot(curr_pose), get_root_rot(next_pose))
        # del_angular_vel = np.array(pybullet.getEulerFromQuaternion(del_angular_vel)) / time_between_frames
        inv_init_rot = transformations.quaternion_inverse(config.INIT_ROT)
        _, base_orientation_quat_from_init = pybullet.multiplyTransforms(
            positionA=(0, 0, 0),
            orientationA=inv_init_rot,
            positionB=(0, 0, 0),
            orientationB=get_root_rot(curr_pose),
        )
        _, inverse_base_orientation = pybullet.invertTransform(
            [0, 0, 0], base_orientation_quat_from_init
        )
        del_angular_vel, _ = pybullet.multiplyTransforms(
            positionA=(0, 0, 0),
            orientationA=(inverse_base_orientation),
            positionB=del_angular_vel,
            orientationB=(0, 0, 0, 1),
        )

        joint_velocity = (
            np.array(get_joint_pose(next_pose) - get_joint_pose(curr_pose))
            / time_between_frames
        )
        toe_velocity = (
            np.array(
                get_tar_toe_pos_local(next_pose) - get_tar_toe_pos_local(curr_pose)
            )
            / time_between_frames
        )

        curr_pose = np.concatenate(
            [curr_pose, del_linear_vel, del_angular_vel, joint_velocity, toe_velocity]
        )

        new_frames[f] = curr_pose

    new_frames[:, 0:2] -= new_frames[0, 0:2]

    return new_frames


def main(argv):
    p = pybullet
    # p.connect(p.GUI, options=f"--width=1920 --height=1080")
    if config.VISUALIZE_RETARGETING:
        p.connect(
            p.GUI, options='--width=1920 --height=1080 --mp4="test.mp4" --mp4fps=60'
        )
    else:
        p.connect(
            p.DIRECT, options='--width=1920 --height=1080 --mp4="test.mp4" --mp4fps=60'
        )
    p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
    pybullet.setAdditionalSearchPath(pd.getDataPath())

    output_dir = config.OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for mocap_motion in config.MOCAP_MOTIONS:
        pybullet.resetSimulation()
        pybullet.setGravity(0, 0, 0)

        ground = pybullet.loadURDF(
            GROUND_URDF_FILENAME
        )  # pylint: disable=unused-variable
        robot = pybullet.loadURDF(
            config.URDF_FILENAME,
            config.INIT_POS,
            config.INIT_ROT,
            flags=p.URDF_MAINTAIN_LINK_ORDER,
        )
        # robot = pybullet.loadMJCF(
        #     config.URDF_FILENAME,
        #     config.INIT_POS,
        #     config.INIT_ROT,
        #     flags=p.URDF_MAINTAIN_LINK_ORDER)
        # Set robot to default pose to bias knees in the right direction.
        set_pose(
            robot,
            np.concatenate(
                [config.INIT_POS, config.INIT_ROT, config.DEFAULT_JOINT_POSE]
            ),
        )

        print(f"Re-targeting {mocap_motion}")

        p.removeAllUserDebugItems()
        # 读取每一行为一个数组
        joint_pos_data = load_ref_data(
            mocap_motion[1], mocap_motion[2], mocap_motion[3] + 1
        )

        if "reverse" in mocap_motion[0]:
            joint_pos_data = np.flip(joint_pos_data, axis=0)

        # 每3个数构建一个数组
        joint_pos_data = joint_pos_data.reshape(joint_pos_data.shape[0], -1, POS_SIZE)

        # 对数据进行坐标变换
        for i in range(joint_pos_data.shape[0]):
            joint_pos_data[i] = process_ref_joint_pos_data(joint_pos_data[i])

        # 调整足端位置
        for ref_toe_joint_id in REF_TOE_JOINT_IDS:
            # 对于关节姿态数据中脚尖关节的最后一个分量（通常表示 z 轴位置），减去该关节在所有帧中的最小值。这个操作将使所有帧中的脚尖位置的最小值为零，相当于把整个运动序列提升到使最低点在地面上。
            joint_pos_data[:, ref_toe_joint_id, -1] -= np.min(
                joint_pos_data[:, ref_toe_joint_id, -1]
            )
            print(joint_pos_data[:, ref_toe_joint_id, -1])
            # 在上一步基础上，再加上一个常数 config.TOE_HEIGHT_OFFSET。这个常数用来提升所有帧中的脚尖位置，从而增加脚尖与地面的距离。这种调整可能用于确保模型在模拟时不会与地面产生碰撞等情况。
            joint_pos_data[:, ref_toe_joint_id, -1] += config.TOE_HEIGHT_OFFSET

        retarget_frames = retarget_motion(robot, joint_pos_data)
        joint_pos_data = joint_pos_data[:-1, :]

        output_file = os.path.join(output_dir, f"{mocap_motion[0]}.txt")
        output_motion(retarget_frames, output_file, mocap_motion[4], FRAME_DURATION)

        if config.VISUALIZE_RETARGETING:
            num_markers = joint_pos_data.shape[1]
            marker_ids = build_markers(num_markers)
            foot_line_unique_ids = [None] * 4
            linear_vel_unique_id = None
            angular_vel_unique_id = None

            f = 0
            num_frames = joint_pos_data.shape[0]

            for repeat in range(1 * num_frames):
                time_start = time.time()

                f_idx = f % num_frames
                print("Frame {:d}".format(f_idx))

                ref_joint_pos = joint_pos_data[f_idx]
                # ref_joint_pos = np.reshape(ref_joint_pos, [-1, POS_SIZE])
                # ref_joint_pos = process_ref_joint_pos_data(ref_joint_pos)

                pose = retarget_frames[f_idx]

                set_pose(robot, pose)
                set_maker_pos(ref_joint_pos, marker_ids)
                foot_line_unique_ids = set_foot_marker_pos(
                    get_tar_toe_pos_local(pose), robot, foot_line_unique_ids
                )
                linear_vel_unique_id = set_linear_vel_pos(
                    get_linear_vel(pose), robot, linear_vel_unique_id
                )
                angular_vel_unique_id = set_angular_vel_pos(
                    get_angular_vel(pose), robot, angular_vel_unique_id
                )

                update_camera(robot)
                p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
                f += 1

                time_end = time.time()
                sleep_dur = FRAME_DURATION - (time_end - time_start)
                sleep_dur = max(0, sleep_dur)

                time.sleep(sleep_dur)
                # time.sleep(0.5) # jp hack
            for m in marker_ids:
                p.removeBody(m)

            p.removeAllUserDebugItems()
            marker_ids = []

    pybullet.disconnect()

    return


if __name__ == "__main__":
    main(None)
    # tf.app.run(main)
