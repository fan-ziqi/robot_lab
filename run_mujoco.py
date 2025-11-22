import numpy as np
import mujoco
import mujoco.viewer
import torch
import time
import sys
import argparse
from scipy.spatial.transform import Rotation as R
from pynput import keyboard

# ==========================================
# 1. 机器人与仿真配置 (Robot & Sim Config)
# ==========================================

class RobotConfig:
    def __init__(self):
        # 关节名称 (必须与 XML 和训练时的顺序严格一致)
        self.joint_names = [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint"
        ]
        
        # 默认站立角度 (来自您的 mydog.py / user input)
        self.default_joint_angles = {
            "FR_hip_joint": -0.1, "FR_thigh_joint": -0.8, "FR_calf_joint": 1.8,
            "FL_hip_joint": 0.1,  "FL_thigh_joint": 0.8,  "FL_calf_joint": -1.8,
            "RR_hip_joint": 0.1,  "RR_thigh_joint": 0.8,  "RR_calf_joint": -1.8,
            "RL_hip_joint": -0.1, "RL_thigh_joint": -0.8, "RL_calf_joint": 1.8,
            "FR_foot_joint": 0.0, "FL_foot_joint": 0.0, "RR_foot_joint": 0.0, "RL_foot_joint": 0.0,
        }

        # 动作缩放 (Action Scales - 必须匹配 rough_env_cfg.py)
        self.leg_pos_scale = 0.25       # 腿部位置缩放
        self.wheel_vel_scale = 10.0     # 轮子速度缩放

        # 初始高度 (用于重置)
        self.init_height = 0.45

        # PD 增益 (MuJoCo 专用，用于模拟高刚度 Actuator)
        # 腿部 (位置控制)
        self.kp_leg = 5.0 
        self.kd_leg = 5.0
        # 轮子 (速度控制)
        self.kp_wheel = 1.0
        self.kd_wheel = 0.0 

        # 力矩限制
        self.tau_limit_leg = 100.0
        self.tau_limit_wheel = 60.0

# 仿真参数
SIM_DT = 0.0002      # 5kHz 物理步长 (防炸飞关键)
DECIMATION = 100     # 控制频率分频: 0.0002 * 100 = 0.02s (50Hz 控制频率)
RAMP_UP_TIME = 1.5   # 软启动时间 (秒)
HISTORY_LEN = 5      # 与训练一致的历史帧长度（HIM）

# ==========================================
# 2. 辅助函数 (Helpers)
# ==========================================

def get_joint_indices(model, joint_names):
    """动态获取关节索引，避免硬编码"""
    dof_ids = [] # qpos 索引
    dof_vel = [] # qvel 索引
    for name in joint_names:
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j_id == -1:
            raise ValueError(f"❌ XML 中找不到关节: {name}")
        dof_ids.append(model.jnt_qposadr[j_id])
        dof_vel.append(model.jnt_dofadr[j_id])
    return np.array(dof_ids), np.array(dof_vel)

def get_obs(data, cmd, last_action, dof_ids, dof_vel, default_angle):
    """计算 57 维观测向量 (严格对齐 Isaac Lab)"""
    # 1. 坐标系转换 (World -> Body)
    base_quat = data.qpos[3:7] # [w, x, y, z]
    # Scipy 需要 [x, y, z, w]
    rot = R.from_quat(base_quat[[1, 2, 3, 0]]) 
    
    # 2. 观测项计算
    # A. Base Ang Vel (Body Frame)
    ang_vel_w = data.qvel[3:6]
    ang_vel_b = rot.apply(ang_vel_w, inverse=True)
    
    # B. Projected Gravity (Body Frame)
    gravity_vec = np.array([0.0, 0.0, -1.0])
    proj_grav = rot.apply(gravity_vec, inverse=True)
    
    # C. Commands (vx, vy, w_yaw)
    commands = cmd
    
    # D. Joint Pos (Relative to default)
    q_pos_rel = data.qpos[dof_ids] - default_angle
    
    # E. Joint Vel
    q_vel = data.qvel[dof_vel]
    
    # F. Last Action
    actions = last_action

    # 3. 拼接 (3+3+3+16+16+16 = 57)
    obs = np.concatenate([
        ang_vel_b,      # 3
        proj_grav,      # 3
        commands,       # 3
        q_pos_rel,      # 16
        q_vel,          # 16
        actions         # 16
    ]).astype(np.float32)
    
    return obs


class ObservationHistory:
    """简单的 FIFO 历史观测缓冲，配合 HIM 风格策略使用。"""
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

    def get_tensor(self):
        # 返回 torch.Tensor，shape = (1, history_len * frame_dim)
        import torch
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

# ==========================================
# 3. 键盘控制 (Keyboard Control)
# ==========================================

class CommandInterface:
    def __init__(self):
        self.cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.exit_flag = False
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()
        print("\n🎮 控制说明:")
        print("  W/S: 前进/后退  A/D: 左右横移  Q/E: 转向")
        print("  Space: 急停归零  Esc: 退出程序\n")

    def on_press(self, key):
        try:
            if hasattr(key, 'char') and key.char:
                k = key.char.lower()
                step_lin = 0.1
                step_ang = 0.1
                
                if k == 'w': self.cmd[0] += step_lin
                elif k == 's': self.cmd[0] -= step_lin
                elif k == 'a': self.cmd[1] += step_lin
                elif k == 'd': self.cmd[1] -= step_lin
                elif k == 'q': self.cmd[2] += step_ang
                elif k == 'e': self.cmd[2] -= step_ang
                elif k == 'x': pass # 修复 x 键报错
                
                # 限制范围
                self.cmd[0] = np.clip(self.cmd[0], -2.0, 2.0)
                self.cmd[1] = np.clip(self.cmd[1], -1.0, 1.0)
                self.cmd[2] = np.clip(self.cmd[2], -1.0, 1.0)
                
                print(f"\r指令: vx={self.cmd[0]:.1f}, vy={self.cmd[1]:.1f}, w={self.cmd[2]:.1f}   ", end="")

            elif key == keyboard.Key.space:
                self.cmd[:] = 0.0
                print("\r指令: [急停]                                   ", end="")
            elif key == keyboard.Key.esc:
                self.exit_flag = True
                
        except AttributeError:
            pass

# ==========================================
# 4. 主循环 (Main Loop)
# ==========================================

def main(args):
    cfg = RobotConfig()
    
    # 1. 加载模型
    print(f"Loading Model: {args.model}")
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    
    # 强制设置小步长以保证高 PD 增益下的稳定
    model.opt.timestep = SIM_DT 
    print(f"Physics Step: {SIM_DT}s")

    # 2. 获取关节索引 & 默认角度数组
    dof_ids, dof_vel = get_joint_indices(model, cfg.joint_names)
    default_angle = np.array([cfg.default_joint_angles[n] for n in cfg.joint_names])
    
    # 3. 加载策略 (TorchScript)
    print(f"Loading Policy: {args.policy}")
    device = torch.device("cpu") # 部署通常用 CPU 即可
    try:
        policy = torch.jit.load(args.policy, map_location=device)
        policy.eval()
    except Exception as e:
        print(f"❌ 无法加载策略文件: {e}")
        return

    # 4. 初始化状态
    # 设置初始姿态 (站立)
    data.qpos[0:3] = [0, 0, cfg.init_height] # 躯干高度
    data.qpos[3:7] = [1, 0, 0, 0]            # 姿态
    data.qpos[dof_ids] = default_angle       # 关节角度
    mujoco.mj_step(model, data) # 刷新一下状态

    cmd_interface = CommandInterface()
    last_action = np.zeros(16, dtype=np.float32)
    
    # 关节分组索引
    idx_leg = np.array([0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14])
    idx_wheel = np.array([3, 7, 11, 15])

    # 5. 仿真循环
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        
        while viewer.is_running() and not cmd_interface.exit_flag:
            loop_start = time.time()
            
            # --- A. 软启动 (Ramp Up) ---
            # 前几秒内，PD 增益从 0 慢慢增加，防止机器人一出生就炸飞
            run_time = time.time() - start_time
            ramp_factor = min(run_time / RAMP_UP_TIME, 1.0)
            
            # --- B. 策略推理 (每 DECIMATION 步执行一次) ---
            # 简单处理：这里每步物理都推理有点浪费，但在 Python 循环中为了逻辑简单，
            # 我们通常每步都算，或者用计数器。为了最流畅的效果，这里使用计数器。
            # 但为了代码最简，我们先每步都跑 (MuJoCo很快)，或者您可以加计数器。
            # 这里演示每 100 步物理步 (0.02s) 推理一次的标准做法：
            
            # (简化版：实时计算，不分频，保证响应最快，Python开销可接受)
            # 1. 获取观测
            obs = get_obs(data, cmd_interface.cmd, last_action, dof_ids, dof_vel, default_angle)
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            
            # 2. 推理
            with torch.no_grad():
                if run_time < 0.5: # 前0.5秒不输出动作，只靠PD维持初始姿态
                    raw_action = np.zeros(16, dtype=np.float32)
                else:
                    raw_action = policy(obs_tensor).numpy().flatten()
            
            last_action = raw_action

            # --- C. 动作缩放 & PD 控制 ---
            # 腿部目标 = 默认 + 动作 * 缩放
            target_q_leg = default_angle[idx_leg] + raw_action[idx_leg] * cfg.leg_pos_scale
            # 轮子目标 = 动作 * 缩放 (速度)
            target_v_wheel = raw_action[idx_wheel] * cfg.wheel_vel_scale

            # 获取当前状态
            curr_q = data.qpos[dof_ids]
            curr_v = data.qvel[dof_vel]

            # 计算力矩
            tau = np.zeros(16)
            
            # 腿部 PD: Kp * (target - current) - Kd * velocity
            tau[idx_leg] = (cfg.kp_leg * ramp_factor) * (target_q_leg - curr_q[idx_leg]) - \
                           (cfg.kd_leg * ramp_factor) * curr_v[idx_leg]
            
            # 轮子 P: Kp * (target_vel - current_vel)
            # 注意：轮子用 kp_wheel 作为速度环的 P 增益
            tau[idx_wheel] = (cfg.kp_wheel * ramp_factor) * (target_v_wheel - curr_v[idx_wheel])

            # 力矩限幅
            tau[idx_leg] = np.clip(tau[idx_leg], -cfg.tau_limit_leg, cfg.tau_limit_leg)
            tau[idx_wheel] = np.clip(tau[idx_wheel], -cfg.tau_limit_wheel, cfg.tau_limit_wheel)

            # --- D. 物理步进 ---
            data.ctrl[:] = tau # 注意：MuJoCo ctrl 索引通常对应 DOF 索引
            mujoco.mj_step(model, data)
            viewer.sync()

            # --- E. 帧率控制 ---
            # 保持仿真不超速
            elapsed = time.time() - loop_start
            if elapsed < SIM_DT:
                time.sleep(SIM_DT - elapsed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 设置默认路径，您可以修改这里
    parser.add_argument('--model', type=str, default='source/robot_lab/data/Robots/myrobots/mydog/mjcf/thunder2_v1.xml')
    parser.add_argument('--policy', type=str, default='logs/rsl_rl/mydog_rough/2025-11-21_17-47-39/exported/policy.pt')
    
    args = parser.parse_args()
    main(args)