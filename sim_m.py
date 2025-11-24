import os

os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
os.environ["__VK_LAYER_NV_optimus"] = "NVIDIA_only"

import numpy as np
import mujoco
import mujoco_viewer
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from pynput import keyboard as pynput_keyboard

# DOF configuration
#set like this be
#FR_hip_joint, FR_thigh_joint, FR_calf_joint,
#FL_hip_joint, FL_thigh_joint, FL_calf_joint,
#RR_hip_joint, RR_thigh_joint, RR_calf_joint,
#RL_hip_joint, RL_thigh_joint, RL_calf_joint,
#FR_foot_joint, FL_foot_joint, RR_foot_joint, RL_foot_joint

dof_vel = [6,7,8, 10,11,12, 14,15,16, 18,19,20, 9,13,17,21]
#set like this:
#FR_hip_joint, FR_thigh_joint, FR_calf_joint,
#FL_hip_joint, FL_thigh_joint, FL_calf_joint,
#RR_hip_joint, RR_thigh_joint, RR_calf_joint,
#RL_hip_joint, RL_thigh_joint, RL_calf_joint,
#FR_foot_joint, FL_foot_joint, RR_foot_joint, RL_foot_joint
dof_ids = [7,8,9, 11,12,13, 15,16,17, 19,20,21,10,14,18,22]


class SimConfig:
    def __init__(self, dt=0.001, decimation=20, sim_duration=60.0):
        self.sim_duration = sim_duration
        self.dt = dt
        self.decimation = decimation

class RobotConfig:
    def __init__(self):
        # Joint names (order matches action order)
        self.joint_names = [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint"
        ]
        
        # Individual joint stiffness (kp) [Nm/rad]
        # NOTE: Hip values increased for better tracking (differs from training)
        self.kp = {
            "FR_hip_joint": 2000.0, "FL_hip_joint": 2000.0,
            "RR_hip_joint": 2000.0, "RL_hip_joint": 2000.0,
            "FR_thigh_joint": 2000.0, "FL_thigh_joint": 2000.0,
            "RR_thigh_joint": 2000.0, "RL_thigh_joint": 2000.0,
            "FR_calf_joint": 2000.0, "FL_calf_joint": 2000.0,
            "RR_calf_joint": 2000.0, "RL_calf_joint": 2000.0,
            "FR_foot_joint": 2000.0, "FL_foot_joint": 1000.0,
            "RR_foot_joint": 1000.0, "RL_foot_joint": 1000.0,
        }
        
        # Individual joint damping (kd) [Nm·s/rad]
        # NOTE: Hip values increased for better tracking (differs from training)
        self.kd = {
            "FR_hip_joint": 10.0, "FL_hip_joint": 10.0,
            "RR_hip_joint": 10.0, "RL_hip_joint": 10.0,
            "FR_thigh_joint": 10.0, "FL_thigh_joint": 10.0,
            "RR_thigh_joint": 10.0, "RL_thigh_joint": 10.0,
            "FR_calf_joint": 15.0, "FL_calf_joint": 15.0,
            "RR_calf_joint": 15.0, "RL_calf_joint": 15.0,
            "FR_foot_joint": 0.1, "FL_foot_joint": 0.1,
            "RR_foot_joint": 0.1, "RL_foot_joint": 0.1,
        }
        
        # Individual joint effort limits [Nm]
        self.tau_limit = {
            "FR_hip_joint": 120.0, "FL_hip_joint": 120.0,
            "RR_hip_joint": 120.0, "RL_hip_joint": 120.0,
            "FR_thigh_joint": 120.0, "FL_thigh_joint": 120.0,
            "RR_thigh_joint": 120.0, "RL_thigh_joint": 120.0,
            "FR_calf_joint": 120.0, "FL_calf_joint": 120.0,
            "RR_calf_joint": 120.0, "RL_calf_joint": 120.0,
            "FR_foot_joint": 60.0, "FL_foot_joint": 60.0,
            "RR_foot_joint": 60.0, "RL_foot_joint": 60.0,
        }
        
        # Action scaling
        # Action scales (must match training config!)
        self.hip_scale = 0.125            # hip joints
        self.thigh_calf_scale = 0.25     # thigh/calf joints  
        self.wheel_scale = 10.0           # wheel joints
        
        # Initial height
        self.init_height = 0.55          # [m]
        
        # Convert dicts to arrays (ordered by joint_names)
        self.kp_array = np.array([self.kp[name] for name in self.joint_names])
        self.kd_array = np.array([self.kd[name] for name in self.joint_names])
        self.tau_limit_array = np.array([self.tau_limit[name] for name in self.joint_names])
    
    def print_parameters(self):
        """Print all PD control parameters for verification"""
        print("\n" + "="*70)
        print("PD Control Parameters Configuration")
        print("="*70)
        print(f"{'Joint Name':<20} {'Kp [Nm/rad]':>12} {'Kd [Nm·s/rad]':>15} {'τ_limit [Nm]':>15}")
        print("-"*70)
        for name in self.joint_names:
            print(f"{name:<20} {self.kp[name]:>12.1f} {self.kd[name]:>15.1f} {self.tau_limit[name]:>15.1f}")
        print("="*70 + "\n")

class Config:
    def __init__(self):
        self.sim_config = SimConfig()
        self.robot_config = RobotConfig()

cfg = Config()

# Default joint angles
# default_joint_angles = {
#     "FR_hip_joint": 0.0, "FL_hip_joint": 0.0, "RR_hip_joint": 0.0, "RL_hip_joint": 0.0,
#     "FR_thigh_joint": -0.64, "FL_thigh_joint": 0.64, 
#     "RR_thigh_joint": 0.64,  "RL_thigh_joint": -0.64,
#     "FR_calf_joint": 1.6,   "FL_calf_joint": -1.6,
#     "RR_calf_joint": -1.6,  "RL_calf_joint": 1.6,
#     "FR_foot_joint": 0.0, "FL_foot_joint": 0.0, "RR_foot_joint": 0.0, "RL_foot_joint": 0.0,
# }
default_joint_angles = {
    # "FR_hip_joint": -0.1, "FL_hip_joint": 0.1, "RR_hip_joint": 0.1, "RL_hip_joint": -0.1,
    # "FR_thigh_joint": -0.8, "FL_thigh_joint": 0.8, 
    # "RR_thigh_joint": 0.8,  "RL_thigh_joint": -0.8,
    # "FR_calf_joint": 1.8,   "FL_calf_joint": -1.8,
    # "RR_calf_joint": -1.8,  "RL_calf_joint": 1.8,
    # "FR_foot_joint": 0.0, "FL_foot_joint": 0.0, "RR_foot_joint": 0.0, "RL_foot_joint": 0.0,
            "FR_hip_joint": -0.1,
            "FR_thigh_joint": -0.8,
            "FR_calf_joint": 1.8,
            "FL_hip_joint": 0.1,
            "FL_thigh_joint": 0.8,
            "FL_calf_joint": -1.8,
            "RR_hip_joint": 0.1,
            "RR_thigh_joint": 0.8,
            "RR_calf_joint": -1.8,
            "RL_hip_joint": -0.1,
            "RL_thigh_joint": -0.8,
            "RL_calf_joint": 1.8,
            "FR_foot_joint": 0.0, "FL_foot_joint": 0.0, "RR_foot_joint": 0.0, "RL_foot_joint": 0.0,
}
default_angle = np.array([
    default_joint_angles["FR_hip_joint"], default_joint_angles["FR_thigh_joint"], default_joint_angles["FR_calf_joint"],
    default_joint_angles["FL_hip_joint"], default_joint_angles["FL_thigh_joint"], default_joint_angles["FL_calf_joint"],
    default_joint_angles["RR_hip_joint"], default_joint_angles["RR_thigh_joint"], default_joint_angles["RR_calf_joint"],
    default_joint_angles["RL_hip_joint"], default_joint_angles["RL_thigh_joint"], default_joint_angles["RL_calf_joint"],
    default_joint_angles["FR_foot_joint"], default_joint_angles["FL_foot_joint"], 
    default_joint_angles["RR_foot_joint"], default_joint_angles["RL_foot_joint"]
], dtype=np.double)

# 单帧观测模式：移除历史堆叠，策略每次仅接收当前 1 帧 obs


class PDTuner:
    """实时 PD 参数调节器（按关节类型分组）"""
    def __init__(self, robot_config):
        self.cfg = robot_config
        self.joint_names = robot_config.joint_names

        # 关节类型分组
        self.joint_types = ["hip", "thigh", "calf", "foot"]
        self.selected_type = 0  # 当前选中的关节类型

        # 调节步长
        self.kp_step = 10.0
        self.kd_step = 1.0

        # 构建关节类型到索引的映射
        self.type_to_indices = {
            "hip": [0, 3, 6, 9],      # FR_hip, FL_hip, RR_hip, RL_hip
            "thigh": [1, 4, 7, 10],   # FR_thigh, FL_thigh, RR_thigh, RL_thigh
            "calf": [2, 5, 8, 11],    # FR_calf, FL_calf, RR_calf, RL_calf
            "foot": [12, 13, 14, 15]  # FR_foot, FL_foot, RR_foot, RL_foot
        }

    def select_next_type(self):
        """切换到下一个关节类型"""
        self.selected_type = (self.selected_type + 1) % len(self.joint_types)
        self._print_status()

    def select_prev_type(self):
        """切换到上一个关节类型"""
        self.selected_type = (self.selected_type - 1) % len(self.joint_types)
        self._print_status()

    def increase_kp(self):
        """增加当前选中类型所有关节的 Kp"""
        joint_type = self.joint_types[self.selected_type]
        indices = self.type_to_indices[joint_type]
        for idx in indices:
            name = self.joint_names[idx]
            self.cfg.kp[name] += self.kp_step
            self.cfg.kp_array[idx] += self.kp_step
        self._print_status()

    def decrease_kp(self):
        """减少当前选中类型所有关节的 Kp"""
        joint_type = self.joint_types[self.selected_type]
        indices = self.type_to_indices[joint_type]
        for idx in indices:
            name = self.joint_names[idx]
            self.cfg.kp[name] = max(0, self.cfg.kp[name] - self.kp_step)
            self.cfg.kp_array[idx] = max(0, self.cfg.kp_array[idx] - self.kp_step)
        self._print_status()

    def increase_kd(self):
        """增加当前选中类型所有关节的 Kd"""
        joint_type = self.joint_types[self.selected_type]
        indices = self.type_to_indices[joint_type]
        for idx in indices:
            name = self.joint_names[idx]
            self.cfg.kd[name] += self.kd_step
            self.cfg.kd_array[idx] += self.kd_step
        self._print_status()

    def decrease_kd(self):
        """减少当前选中类型所有关节的 Kd"""
        joint_type = self.joint_types[self.selected_type]
        indices = self.type_to_indices[joint_type]
        for idx in indices:
            name = self.joint_names[idx]
            self.cfg.kd[name] = max(0, self.cfg.kd[name] - self.kd_step)
            self.cfg.kd_array[idx] = max(0, self.cfg.kd_array[idx] - self.kd_step)
        self._print_status()

    def _print_status(self):
        """打印当前选中类型的参数"""
        joint_type = self.joint_types[self.selected_type]
        indices = self.type_to_indices[joint_type]

        print(f"\n=== Selected Type: {joint_type.upper()} ===")
        for idx in indices:
            name = self.joint_names[idx]
            print(f"  [{idx:2d}] {name:<18} Kp={self.cfg.kp[name]:>7.1f}  Kd={self.cfg.kd[name]:>6.1f}")

    def print_all_params(self):
        """打印所有关节的 PD 参数（按类型分组）"""
        print("\n" + "="*70)
        print("Current PD Parameters (by Joint Type)")
        print("="*70)

        for joint_type in self.joint_types:
            indices = self.type_to_indices[joint_type]
            marker = " *SELECTED*" if joint_type == self.joint_types[self.selected_type] else ""
            print(f"\n{joint_type.upper()}{marker}:")
            print(f"  {'Idx':<4} {'Joint Name':<18} {'Kp':>10} {'Kd':>10}")
            print("  " + "-"*44)
            for idx in indices:
                name = self.joint_names[idx]
                print(f"  {idx:<4} {name:<18} {self.cfg.kp[name]:>10.1f} {self.cfg.kd[name]:>10.1f}")
        print("="*70 + "\n")


# 全局 PDTuner 实例（稍后初始化）
pd_tuner = None


class Cmd:
    def __init__(self):
        # Velocity command range
        self.vx = 2.0    # [-4.5, 4.5] m/s
        self.vy = 1.0    # [-1.5, 1.5] m/s  
        self.dyaw = 0.0  # [-1.0, 1.0] rad/s

        self.vx_step = 0.1
        self.vy_step = 0.1
        self.vyaw_step = 0.1
        self.vx_max = 2.0
        self.vy_max = 1.0
        self.vyaw_max = 1.0

    def increase_vx(self):
        self.vx = min(self.vx + self.vx_step, self.vx_max)

    def decrease_vx(self):
        self.vx = max(self.vx - self.vx_step, -self.vx_max)

    def increase_vy(self):
        self.vy = min(self.vy + self.vy_step, self.vy_max)

    def decrease_vy(self):
        self.vy = max(self.vy - self.vy_step, -self.vy_max)

    def increase_vyaw(self):
        self.dyaw = min(self.dyaw + self.vyaw_step, self.vyaw_max)

    def decrease_vyaw(self):
        self.dyaw = max(self.dyaw - self.vyaw_step, -self.vyaw_max)

    def stop_all(self):
        self.vx = 0.0
        self.vy = 0.0
        self.dyaw = 0.0

    def stop_vx(self):
        self.vx = 0.0

    def stop_vy(self):
        self.vy = 0.0

    def stop_vyaw(self):
        self.dyaw = 0.0

vel_cmd = Cmd()


def start_keyboard_listener():
    from pynput import keyboard as pynput_keyboard

    def on_press(key):
        global pd_tuner
        try:
            k = key.char.lower()  # 普通字母键
        except AttributeError:
            k = None

        # 速度控制键
        if k == 'w':
            vel_cmd.increase_vx()
            print(f"Forward: vx={vel_cmd.vx:.2f}")
        elif k == 's':
            vel_cmd.decrease_vx()
            print(f"Backward: vx={vel_cmd.vx:.2f}")
        elif k == 'a':
            vel_cmd.increase_vy()
            print(f"Left: vy={vel_cmd.vy:.2f}")
        elif k == 'd':
            vel_cmd.decrease_vy()
            print(f"Right: vy={vel_cmd.vy:.2f}")
        elif k == 'q':
            vel_cmd.increase_vyaw()
            print(f"CCW: vyaw={vel_cmd.dyaw:.2f}")
        elif k == 'e':
            vel_cmd.decrease_vyaw()
            print(f"CW: vyaw={vel_cmd.dyaw:.2f}")
        elif k == 'x':
            vel_cmd.stop_vx()
            print("Stop vx")
        elif k == 'c':
            vel_cmd.stop_vy()
            print("Stop vy")
        elif k == 'v':
            vel_cmd.stop_vyaw()
            print("Stop vyaw")

        # PD 调参热键（按关节类型）
        elif k == '[':
            if pd_tuner:
                pd_tuner.select_prev_type()
        elif k == ']':
            if pd_tuner:
                pd_tuner.select_next_type()
        elif k == 'p':
            if pd_tuner:
                pd_tuner.print_all_params()

        # 特殊键控制
        if key == pynput_keyboard.Key.space:
            vel_cmd.stop_all()
            print("Emergency stop")
        elif key == pynput_keyboard.Key.up:
            if pd_tuner:
                pd_tuner.increase_kp()
        elif key == pynput_keyboard.Key.down:
            if pd_tuner:
                pd_tuner.decrease_kp()
        elif key == pynput_keyboard.Key.right:
            if pd_tuner:
                pd_tuner.increase_kd()
        elif key == pynput_keyboard.Key.left:
            if pd_tuner:
                pd_tuner.decrease_kd()

    listener = pynput_keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()

def get_obs(data, vel_cmd, last_action, debug=False):
    q = data.qpos[dof_ids].astype(np.double) - default_angle
    # q += np.random.uniform(-0.01, 0.01, q.shape)
    
    dq = data.qvel[dof_vel].astype(np.double) * 0.05
    q[-4:] = 0.0   # zero wheel joints

    imu_quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r_imu = R.from_quat(imu_quat)
    proj = r_imu.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)

    gyro_local = data.sensor('angular-velocity').data.astype(np.double)
    base_quat = data.qpos[3:7][[1, 2, 3, 0]].astype(np.double)
    r_base = R.from_quat(base_quat)

    gyro_world = r_imu.apply(gyro_local)
    gyro = r_base.apply(gyro_world, inverse=True) * 0.25

    obs = np.concatenate([
        gyro, proj,
        np.array([vel_cmd.vx, vel_cmd.vy, vel_cmd.dyaw]),
        q, dq, last_action
    ]).astype(np.float32)

    if debug:
        print(f"gyro: {gyro}, proj: {proj}, q: {q}, dq: {dq}")

    return obs

class PDController:
    def __init__(self, kp, kd, tau_limit):
        self.kp = kp
        self.kd = kd
        self.tau_limit = tau_limit

    def compute(self, target_q, q, target_dq, dq):
        tau = self.kp * (target_q - q) + self.kd * (target_dq - dq)
        return np.clip(tau, -self.tau_limit, self.tau_limit)

def scale_action(raw_action, cfg):
    scaled = np.zeros_like(raw_action)
    for i in range(4):
        base = i * 3
        scaled[base+0] = raw_action[base+0] * cfg.robot_config.hip_scale
        scaled[base+1] = raw_action[base+1] * cfg.robot_config.thigh_calf_scale
        scaled[base+2] = raw_action[base+2] * cfg.robot_config.thigh_calf_scale
    # 轮子速度（XML文件已修复，不需要反转）
    scaled[12:] = raw_action[12:] * cfg.robot_config.wheel_scale
    return scaled

def plot_joint_data(plot_data):
    """Plot two figures: 1) Target vs Actual, 2) Error vs Torque"""
    joint_names = [
        "FR_hip", "FR_thigh", "FR_calf",
        "FL_hip", "FL_thigh", "FL_calf",
        "RR_hip", "RR_thigh", "RR_calf",
        "RL_hip", "RL_thigh", "RL_calf",
        "FR_wheel", "FL_wheel", "RR_wheel", "RL_wheel"
    ]
    
    time = np.array(plot_data['time'])
    
    # Figure 1: Target vs Actual
    fig1, axes1 = plt.subplots(4, 4, figsize=(16, 12))
    fig1.suptitle('Target vs Actual (Position/Velocity)', fontsize=16, fontweight='bold')
    
    for i in range(16):
        row = i // 4
        col = i % 4
        ax = axes1[row, col]
        
        targets = np.array(plot_data['targets'][i])
        actuals = np.array(plot_data['actuals'][i])
        
        # Plot target (blue dashed) and actual (green solid)
        ax.plot(time, targets, 'b--', linewidth=1.5, label='Target', alpha=0.8)
        ax.plot(time, actuals, 'g-', linewidth=1.5, label='Actual', alpha=0.8)
        
        if i < 12:
            ax.set_ylabel('Position [rad]', fontsize=8)
        else:
            ax.set_ylabel('Velocity [rad/s]', fontsize=8)
        
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=7)
        ax.set_title(joint_names[i], fontsize=10, fontweight='bold')
        ax.set_xlabel('Time [s]', fontsize=8)
        ax.tick_params(axis='x', labelsize=7)
    
    plt.tight_layout()
    fig1.savefig('joint_tracking.png', dpi=150, bbox_inches='tight')
    print(f"✓ Figure 1 saved: joint_tracking.png")
    
    # Figure 2: Error vs Torque
    fig2, axes2 = plt.subplots(4, 4, figsize=(16, 12))
    fig2.suptitle('Error and Torque Output', fontsize=16, fontweight='bold')
    
    for i in range(16):
        row = i // 4
        col = i % 4
        ax = axes2[row, col]
        
        errors = np.array(plot_data['errors'][i])
        torques = np.array(plot_data['torques'][i])
        
        # Plot error (blue)
        ax.plot(time, errors, 'b-', linewidth=1.5, label='Error', alpha=0.8)
        if i < 12:
            ax.set_ylabel('Error [rad]', color='b', fontsize=8)
        else:
            ax.set_ylabel('Error [rad/s]', color='b', fontsize=8)
        ax.tick_params(axis='y', labelcolor='b', labelsize=7)
        ax.grid(True, alpha=0.3)
        
        # Plot torque on secondary axis (red)
        ax2 = ax.twinx()
        ax2.plot(time, torques, 'r-', linewidth=1.5, label='Torque', alpha=0.8)
        ax2.set_ylabel('Torque [Nm]', color='r', fontsize=8)
        ax2.tick_params(axis='y', labelcolor='r', labelsize=7)
        
        ax.set_title(joint_names[i], fontsize=10, fontweight='bold')
        ax.set_xlabel('Time [s]', fontsize=8)
        ax.tick_params(axis='x', labelsize=7)
    
    plt.tight_layout()
    fig2.savefig('joint_error_torque.png', dpi=150, bbox_inches='tight')
    print(f"✓ Figure 2 saved: joint_error_torque.png")
    
    plt.show()


class RealtimePlotter:
    """实时 PD 效果可视化 - 显示目标值和实际值"""
    def __init__(self, window_size=500, update_interval=10):
        self.window_size = window_size
        self.update_interval = update_interval
        self.step_count = 0

        # 滚动数据缓冲区
        from collections import deque
        self.time_buffer = deque(maxlen=window_size)
        self.target_buffers = [deque(maxlen=window_size) for _ in range(16)]
        self.actual_buffers = [deque(maxlen=window_size) for _ in range(16)]

        # 初始化图表
        plt.ion()  # 交互模式
        self.fig, self.axes = plt.subplots(4, 4, figsize=(15, 11))
        self.fig.suptitle('Real-time PD Control Monitor | Blue=Target, Green=Actual', fontsize=13, fontweight='bold')

        self.target_lines = []
        self.actual_lines = []

        joint_names = ["FR_hip", "FR_thigh", "FR_calf",
                       "FL_hip", "FL_thigh", "FL_calf",
                       "RR_hip", "RR_thigh", "RR_calf",
                       "RL_hip", "RL_thigh", "RL_calf",
                       "FR_whl", "FL_whl", "RR_whl", "RL_whl"]

        for i in range(16):
            ax = self.axes[i // 4, i % 4]

            # 目标值曲线（蓝色虚线）
            line_target, = ax.plot([], [], 'b--', linewidth=1.5, label='Target', alpha=0.8)

            # 实际值曲线（绿色实线）
            line_actual, = ax.plot([], [], 'g-', linewidth=1.5, label='Actual', alpha=0.8)

            ax.set_title(joint_names[i], fontsize=10, fontweight='bold')

            # Y轴标签
            if i < 12:
                ax.set_ylabel('Pos [rad]', fontsize=8, color='k')
            else:
                ax.set_ylabel('Vel [rad/s]', fontsize=8, color='k')

            ax.set_xlabel('Time [s]', fontsize=7)
            ax.tick_params(axis='y', labelsize=7)
            ax.tick_params(axis='x', labelsize=7)
            ax.grid(True, alpha=0.3)

            # 添加图例
            ax.legend(loc='upper right', fontsize=7)

            self.target_lines.append(line_target)
            self.actual_lines.append(line_actual)

        plt.tight_layout()
        self.fig.canvas.draw()
        plt.pause(0.01)

    def update(self, time, target_q, actual_q, actual_dq):
        """每个控制周期调用 - 更新目标值和实际值曲线"""
        self.step_count += 1
        self.time_buffer.append(time)

        for i in range(16):
            if i < 12:
                # 腿部关节：显示位置
                self.target_buffers[i].append(target_q[i])
                self.actual_buffers[i].append(actual_q[i])
            else:
                # 轮子关节：显示速度
                self.target_buffers[i].append(target_q[i])
                self.actual_buffers[i].append(actual_dq[i])

        # 每 update_interval 步更新图表
        if self.step_count % self.update_interval == 0:
            time_data = list(self.time_buffer)
            for i in range(16):
                target_data = list(self.target_buffers[i])
                actual_data = list(self.actual_buffers[i])

                # 更新曲线数据
                self.target_lines[i].set_data(time_data, target_data)
                self.actual_lines[i].set_data(time_data, actual_data)

                ax = self.axes[i // 4, i % 4]
                if len(time_data) > 1:
                    ax.set_xlim(time_data[0], time_data[-1])

                    # 动态调整 Y 轴范围（位置/速度）
                    if len(target_data) > 0 and len(actual_data) > 0:
                        all_vals = target_data + actual_data
                        val_min, val_max = min(all_vals), max(all_vals)
                        margin = max(0.1, (val_max - val_min) * 0.15)
                        ax.set_ylim(val_min - margin, val_max + margin)

            try:
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
            except Exception:
                pass  # 忽略绘图错误

    def close(self):
        """关闭图表"""
        plt.ioff()
        plt.close(self.fig)


def run_mujoco(policy, mujoco_model_path, sim_duration, dt, decimation, debug=False, plot=False, keyboard_control=False, realtime_plot=False):
    global pd_tuner

    model = mujoco.MjModel.from_xml_path(mujoco_model_path)
    model.opt.timestep = dt
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)
    viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = 1
    viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 1

    # 初始化 PDTuner
    pd_tuner = PDTuner(cfg.robot_config)

    # 初始化实时绘图器
    rt_plotter = None
    if realtime_plot:
        rt_plotter = RealtimePlotter(window_size=500, update_interval=5)
        print("\nRealtime plot enabled")

    if keyboard_control:
        print("\nKeyboard controls:")
        print("  Velocity: W/S (vx), A/D (vy), Q/E (yaw), Space (stop)")
        print("  PD Tuning: [/] (select type: hip/thigh/calf/foot)")
        print("             Up/Down (Kp), Left/Right (Kd), P (print all)\n")

    # Set initial state
    data.qpos[:3] = [0, 0, cfg.robot_config.init_height]
    data.qpos[3:7] = [1, 0, 0, 0]  # quaternion [w, x, y, z]
    data.qpos[dof_ids] = default_angle.copy()
    data.qvel[:] = 0.0

    target_q = default_angle.copy()
    action = np.zeros(16, dtype=np.float32)
    last_action = np.zeros(16, dtype=np.float32)

    # Data recording for plotting
    if plot:
        plot_data = {
            'time': [],
            'targets': [[] for _ in range(16)],
            'actuals': [[] for _ in range(16)],
            'errors': [[] for _ in range(16)],
            'torques': [[] for _ in range(16)]
        }
    steps = int(sim_duration / dt)
    try:
        for step in tqdm(range(steps), desc="Simulating..."):
            if step % decimation == 0:
                # 构建单帧观测张量 (1, obs_dim)
                obs = get_obs(data, vel_cmd, last_action, debug=debug)
                obs_tensor = torch.from_numpy(obs).to(dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    raw_action = policy(obs_tensor).cpu().numpy().squeeze()
                action[:] = raw_action
                if debug:
                    print("raw_action:", action)
                if step > 100:
                    scaled_action = scale_action(action, cfg)
                    target_q = scaled_action + default_angle
                else:
                    target_q = default_angle
                last_action = action.copy()

            q = data.qpos[dof_ids]
            dq = data.qvel[dof_vel]

            # 使用实时更新的 PD 参数（可通过键盘调节）
            kp = cfg.robot_config.kp_array
            kd = cfg.robot_config.kd_array
            tau_limit = cfg.robot_config.tau_limit_array

            tau = np.zeros(16)

            # Leg joints (0-11): position control
            tau[:12] = kp[:12] * (target_q[:12] - q[:12]) #+ kd[:12] * (0 - dq[:12])

            # Wheel joints (12-15): velocity control (target_q stores target velocity for wheels)
            tau[12:] = kd[12:] * (target_q[12:] - dq[12:])
            #tau[:]=0
            # tau[14]=5
            #tau[15]=5
            print("target_q[12:]:", target_q[:])
            print("dq[12:]:", dq[:])
            tau = np.clip(tau, -tau_limit, tau_limit)

            # 计算误差（用于绘图）
            errors = np.zeros(16)
            errors[:12] = target_q[:12] - q[:12]  # 位置误差
            errors[12:] = target_q[12:] - dq[12:]  # 速度误差

            # Record data for plotting (every decimation steps)
            if plot and step % decimation == 0:
                plot_data['time'].append(step * dt)
                for i in range(12):
                    plot_data['targets'][i].append(target_q[i])
                    plot_data['actuals'][i].append(q[i])
                    plot_data['errors'][i].append(errors[i])
                    plot_data['torques'][i].append(tau[i])
                for i in range(12, 16):
                    plot_data['targets'][i].append(target_q[i])
                    plot_data['actuals'][i].append(dq[i])
                    plot_data['errors'][i].append(errors[i])
                    plot_data['torques'][i].append(tau[i])

            # 实时绘图更新
            if rt_plotter is not None and step % decimation == 0:
                rt_plotter.update(step * dt, target_q, q, dq)

            # Apply computed torques to actuators
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)

            if step % decimation == 0:
                viewer.render()

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    finally:
        viewer.close()

        # 关闭实时绘图器
        if rt_plotter is not None:
            rt_plotter.close()

        # Plot results (事后绘图)
        if plot and len(plot_data['time']) > 0:
            plot_joint_data(plot_data)

        # 打印最终 PD 参数（如果有修改）
        if keyboard_control:
            print("\nFinal PD Parameters:")
            pd_tuner.print_all_params()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Mujoco deployment')
    parser.add_argument('--model-path', type=str, default='/home/liu/Desktop/robot_lab/source/robot_lab/data/Robots/myrobots/mydog/mjcf/thunder2_v1.xml',
                        help='Path to MuJoCo XML model. Available terrains: thunder2_v1.xml (complex), thunder2_v1_simple.xml (stairs only)')
    parser.add_argument('--policy-path', type=str, default='/home/liu/Desktop/robot_lab/logs/rsl_rl/mydog_rough/2025-11-21_17-47-39/exported/policy.pt')
    parser.add_argument('--duration', type=float, default=120.0)
    parser.add_argument('--dt', type=float, default=0.001)
    parser.add_argument('--decimation', type=int, default=5)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--vx', type=float, default=0.0, help='Forward velocity command [-4.5, 4.5] m/s')
    parser.add_argument('--vy', type=float, default=0.0, help='Lateral velocity command [-1.5, 1.5] m/s')
    parser.add_argument('--vyaw', type=float, default=0.0, help='Yaw velocity command [-1.0, 1.0] rad/s')
    parser.add_argument('--plot', action='store_true', help='Generate plots of joint errors and torques after simulation')
    parser.add_argument('--realtime-plot', action='store_true', help='Enable realtime PD control visualization')
    parser.add_argument('--keyboard', action='store_true', help='Enable keyboard control (WASD/QE + PD tuning)')

    args = parser.parse_args()
    args.keyboard = True  # 默认启用键盘控制
    cfg.sim_config.dt = args.dt
    cfg.sim_config.decimation = args.decimation

    vel_cmd.vx = args.vx
    vel_cmd.vy = args.vy
    vel_cmd.dyaw = args.vyaw

    # Print PD control parameters
    cfg.robot_config.print_parameters()

    print(f"Loading policy from: {args.policy_path}")
    policy = torch.jit.load(args.policy_path)
    print(f"Velocity command: vx={vel_cmd.vx:.2f}, vy={vel_cmd.vy:.2f}, vyaw={vel_cmd.dyaw:.2f}")

    if args.keyboard:
        start_keyboard_listener()
        print("\nKeyboard control enabled:")
        print("  Velocity: W/S (vx), A/D (vy), Q/E (yaw), Space (stop)")
        print("  PD Tuning: [/] (select type: hip/thigh/calf/foot)")
        print("             Up/Down (Kp), Left/Right (Kd), P (print all)")

    if args.plot:
        print("\nPost-simulation plot enabled")

    if args.realtime_plot:
        print("Realtime plot enabled")

    print()
    run_mujoco(policy, args.model_path, args.duration, args.dt, args.decimation,
               debug=args.debug, plot=args.plot, keyboard_control=args.keyboard,
               realtime_plot=args.realtime_plot)