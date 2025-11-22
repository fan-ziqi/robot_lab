import numpy as np
import mujoco
from mujoco import viewer
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
            "FR_hip_joint": 80.0, "FL_hip_joint": 80.0,
            "RR_hip_joint": 80.0, "RL_hip_joint": 80.0,
            "FR_thigh_joint": 100.0, "FL_thigh_joint": 100.0,
            "RR_thigh_joint": 100.0, "RL_thigh_joint": 100.0,
            "FR_calf_joint": 120.0, "FL_calf_joint": 120.0,
            "RR_calf_joint": 120.0, "RL_calf_joint": 120.0,
            "FR_foot_joint": 0.0, "FL_foot_joint": 0.0,
            "RR_foot_joint": 0.0, "RL_foot_joint": 0.0,
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
            "FR_foot_joint": 1.0, "FL_foot_joint": 1.0,
            "RR_foot_joint": 1.0, "RL_foot_joint": 1.0,
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
        self.wheel_scale = 10.0          # wheel joints
        
        # Initial height
        self.init_height = 0.40          # [m]
        
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
            "FR_hip_joint": 0.1,
            "FR_thigh_joint": -0.7,
            "FR_calf_joint": 1.4,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.7,
            "FL_calf_joint": -1.4,
            "RR_hip_joint": 0.0,
            "RR_thigh_joint": 0.7,
            "RR_calf_joint": -1.4,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": -0.7,
            "RL_calf_joint": 1.4,
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

HISTORY_LEN = 5  # Must match training configuration


class ObservationHistory:
    """Maintain a FIFO buffer of observations for HIM-style policies."""

    def __init__(self, history_len: int, frame_dim: int):
        self.history_len = history_len
        self.frame_dim = frame_dim
        self.buffer = np.zeros((history_len, frame_dim), dtype=np.float32)
        self.count = 0

    def push(self, obs: np.ndarray):
        if self.count < self.history_len:
            self.buffer[self.count] = obs
            self.count += 1
        else:
            self.buffer[:-1] = self.buffer[1:]
            self.buffer[-1] = obs

    def get_tensor(self) -> torch.Tensor:
        if self.count == 0:
            padded = self.buffer.copy()
        elif self.count < self.history_len:
            padded = self.buffer.copy()
            last = padded[self.count - 1]
            for idx in range(self.count, self.history_len):
                padded[idx] = last
        else:
            padded = self.buffer
        flat = padded.reshape(1, -1)
        return torch.from_numpy(flat).to(dtype=torch.float32)


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
    def on_press(key):
        try:
            k = key.char.lower()  # 普通字符键
        except AttributeError:
            k = None  # 特殊键没有 .char

        # 移动控制
        if k == 'w':
            vel_cmd.increase_vx()
            print(f"⬆ Forward: vx={vel_cmd.vx:.2f}")
        elif k == 's':
            vel_cmd.decrease_vx()
            print(f"⬇ Backward: vx={vel_cmd.vx:.2f}")
        elif k == 'a':
            vel_cmd.increase_vy()
            print(f"⬅ Left: vy={vel_cmd.vy:.2f}")
        elif k == 'd':
            vel_cmd.decrease_vy()
            print(f"➡ Right: vy={vel_cmd.vy:.2f}")
        elif k == 'q':
            vel_cmd.increase_vyaw()
            print(f"↺ CCW: vyaw={vel_cmd.dyaw:.2f}")
        elif k == 'e':
            vel_cmd.decrease_vyaw()
            print(f"↻ CW: vyaw={vel_cmd.dyaw:.2f}")
        elif k == 'x':
            vel_cmd.stop_vx()
        elif k == 'c':
            vel_cmd.stop_vy()
        elif k == 'v':
            vel_cmd.stop_vyaw()

        # 特殊键
        if key == pynput_keyboard.Key.space:
            vel_cmd.stop_all()
            print("⏹ Emergency stop")

def get_obs(data, vel_cmd, last_action, debug=False):
    q = data.qpos[dof_ids].astype(np.double) - default_angle
    q += np.random.uniform(-0.01, 0.01, q.shape)
    
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

def run_mujoco(policy, mujoco_model_path, sim_duration, dt, decimation, debug=False, plot=False, keyboard_control=False):
    model = mujoco.MjModel.from_xml_path(mujoco_model_path)
    model.opt.timestep = dt
    data = mujoco.MjData(model)
    print(1)
    #   读取速度

    mujoco.mj_step(model, data)
    print(1)
    viewer_instance = viewer.launch_passive(model, data)
    print(1)
    if keyboard_control:
        print("\n⌨️ Keyboard control active (W/S/A/D, Q/E, space/X/C/V)\n")

    # Set initial state
    data.qpos[:3] = [0, 0, cfg.robot_config.init_height]
    data.qpos[3:7] = [1, 0, 0, 0]  # quaternion [w, x, y, z]
    data.qpos[dof_ids] = default_angle.copy()
    data.qvel[:] = 0.0
    print(data.qvel)
    print(data.qvel)
    target_q = default_angle.copy()
    action = np.zeros(16, dtype=np.float32)
    last_action = np.zeros(16, dtype=np.float32)

    # Vectorized PD control parameters
    kp_array = cfg.robot_config.kp_array
    kd_array = cfg.robot_config.kd_array
    tau_limit_array = cfg.robot_config.tau_limit_array

    # Data recording for plotting
    if plot:
        plot_data = {
            'time': [],
            'targets': [[] for _ in range(16)],
            'actuals': [[] for _ in range(16)],
            'errors': [[] for _ in range(16)],
            'torques': [[] for _ in range(16)]
        }

    obs_history = None
    steps = int(sim_duration / dt)
   
    try:
        for step in tqdm(range(steps), desc="Simulating..."):
            if step % decimation == 0:
                obs = get_obs(data, vel_cmd, last_action, debug=debug)
                if obs_history is None:
                    obs_history = ObservationHistory(HISTORY_LEN, obs.shape[0])
                obs_history.push(obs)
                with torch.no_grad():
                    obs_tensor = obs_history.get_tensor()
                    print("obs_tensor: ", obs_tensor)
                    raw_action = policy(obs_tensor).cpu().numpy().squeeze()
                action[:] = raw_action
                print("action: ", action)
                print("================================================")
                if step > 100:
                    scaled_action = scale_action(action, cfg)
                    target_q = scaled_action + default_angle
                else:
                    target_q = default_angle
                last_action = action.copy()

            q = data.qpos[dof_ids]
            dq = data.qvel[dof_vel]


            tau = np.zeros(16)
            
            # Leg joints (0-11): position control
            tau[:12] = kp_array[:12] * (target_q[:12] - q[:12]) + kd_array[:12] * (0 - dq[:12])
            
            # Wheel joints (12-15): velocity control (target_q stores target velocity for wheels)
            tau[12:] = kd_array[12:] * (target_q[12:] - dq[12:])
            
            tau = np.clip(tau, -tau_limit_array, tau_limit_array)
            
            # Record data for plotting (every decimation steps)
            if plot and step % decimation == 0:
                plot_data['time'].append(step * dt)
                # For leg joints (0-11): position control
                for i in range(12):
                    plot_data['targets'][i].append(target_q[i])
                    plot_data['actuals'][i].append(q[i])
                    plot_data['errors'][i].append(target_q[i] - q[i])
                    plot_data['torques'][i].append(tau[i])
                # For wheel joints (12-15): velocity control
                for i in range(12, 16):
                    plot_data['targets'][i].append(target_q[i])
                    plot_data['actuals'][i].append(dq[i])
                    plot_data['errors'][i].append(target_q[i] - dq[i])
                    plot_data['torques'][i].append(tau[i])
            
            # Debug mode disabled when plotting (too much output)
            # if debug and step % (decimation * 10) == 0:
            #     pass
            
            # Apply computed torques to actuators
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)
    
            # if step % decimation == 0:
            #     viewer_instance.render()

    except KeyboardInterrupt:
        print("Simulation interrupted by user")
    finally:
        viewer.close()
        
        # Plot results
        if plot and len(plot_data['time']) > 0:
            plot_joint_data(plot_data)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Mujoco deployment')
    parser.add_argument('--model-path', type=str, default='/home/bsrl/hongsenpang/simulation/thunder2v1/mjcf/thunder2_v1.xml')
    parser.add_argument('--policy-path', type=str, default='/home/bsrl/hongsenpang/simulation/thunder2v1/policy/policy.pt')
    parser.add_argument('--duration', type=float, default=60.0)
    parser.add_argument('--dt', type=float, default=0.001)
    parser.add_argument('--decimation', type=int, default=10)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--vx', type=float, default=0.0, help='Forward velocity command [-4.5, 4.5] m/s')
    parser.add_argument('--vy', type=float, default=0.0, help='Lateral velocity command [-1.5, 1.5] m/s')
    parser.add_argument('--vyaw', type=float, default=0.0, help='Yaw velocity command [-1.0, 1.0] rad/s')
    parser.add_argument('--plot', action='store_true', help='Generate plots of joint errors and torques')
    parser.add_argument('--keyboard', action='store_true', help='Enable keyboard control (WASD/QE)')
    
    args = parser.parse_args()

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
        print("Keyboard control enabled: W/S (vx), A/D (vy), Q/E (yaw), space to stop")
    if args.plot:
        print("Plot mode enabled: will generate joint_analysis.png after simulation\n")
    
    run_mujoco(policy, args.model_path, args.duration, args.dt, args.decimation,
               debug=args.debug, plot=args.plot, keyboard_control=args.keyboard)
