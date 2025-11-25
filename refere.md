下面的代码基于 Isaac Gym 和 legged_gym 框架，目的是通过强化学习训练机器狗完成倒立任务。
我现在基于issaclab 框架训练四足轮腿机器狗完成倒立，你只能参考其思路，并在我现在的框架上完成该思路的代码
✔ 当你输出代码时必须遵守

必须兼容 IsaacLab 用法（如 obs/state/reward 计算方式）

不得直接照搬 legged_gym 路径或 API

不要输出伪代码，必须可运行

支持 GPU batch 操作（torch）

尽量减少 for 循环，提高张量并行效率

✔ 当你输出新函数时必须

写清楚输入、输出说明

注释逻辑、变量含义

允许我复制即可直接运行
在这个 legged_robot.py 文件中，除了通用的移动和稳定性奖励（如速度跟踪、能量惩罚）外，有4个专门针对倒立（Handstand）设计的核心 Reward 函数。

1. _reward_handstand_feet_height_exp (足部高度奖励)

这是引导机器人将后腿抬起的最直接的奖励信号。

    原理：

        通过正则匹配找到所有足部（feet）的刚体索引。

        获取这些足部的当前 Z 轴坐标（高度）。

        计算当前高度与配置文件中设定的 target_height（目标高度）之间的误差。

        使用高斯核函数（指数形式）将误差转化为奖励值：Reward=exp(−std2error2​)。

    作用：

        当足部高度越接近目标高度，奖励越高。

        利用指数函数的特性，当误差很大时奖励几乎为0，只有当机器人开始尝试抬腿接近目标时，奖励才会急剧增加，从而引导策略网络学习“抬腿”动作。

2. _reward_handstand_feet_on_air (足部悬空奖励 - 改进版)

这个函数是一个二值（0或1）或概率奖励，用于强制机器人保持特定部位离地。代码中存在两个版本，生效的是改进版，专门为了防止机器人“作弊”（例如用膝盖跪地）。

    原理：

        检测脚部：检查所有足部（feet）是否接触地面（接触力 > 1.0）。

        检测膝盖：通过正则 r'.*(Knee|THIGH|SHANK).*' 检查膝盖、大腿或小腿是否接触地面。

        逻辑判断：
        Reward=(所有脚未触地)×(所有膝盖未触地)

    作用：

        防止机器人偷懒：在强化学习中，机器人很容易学会“跪着”或者“坐着”把前脚抬起来，这算不上倒立。

        该奖励强制要求只有前肢（手）着地，后腿和膝盖必须全部悬空，才能获得奖励。

3. _reward_handstand_feet_air_time (空中持续时间奖励)

这个奖励鼓励机器人保持倒立姿态的时间越长越好，而不是跳一下就下来。

    原理：

        维护一个计时器 self.feet_air_time，记录足部连续离地的时间。

        触发时刻：当足部落地的那一瞬间（First Contact），结算一次奖励。

        计算公式：Reward=∑(air_time−threshold)。即滞空时间越长，落地时给的奖励越大。

        膝盖惩罚（关键机制）：如果在滞空期间或落地瞬间膝盖接触了地面 (any_knee_contact)，则将奖励强行置为 0。

    作用：

        鼓励动态平衡：只有长时间保持倒立不倒（且不跪地），才能在最终结算时获得高分。

4. _reward_handstand_orientation_l2 (倒立姿态/方向奖励)

除了把脚抬起来，机器人的躯干还需要保持垂直，这个函数用于约束身体的角度。

    原理：

        self.projected_gravity 是重力向量在机器人基座坐标系下的投影。

            如果机器人平趴，重力投影通常是 (0, 0, -1)。

            如果机器人垂直倒立，重力投影可能是 (1, 0, 0) 或 (-1, 0, 0)（取决于机器人坐标轴定义）。

        计算当前投影向量与 target_gravity（在 config 中定义，例如设为倒立时的理想向量）之间的 L2 距离（欧氏距离的平方）。

    作用：

        通过惩罚姿态偏差，引导机器人调整躯干的 Pitch（俯仰角），使其保持笔直的倒立姿态，防止身体过度倾斜或翻转。

总结：倒立训练的逻辑链条

这四个函数共同构成了一个完整的训练逻辑：

    Shape（引导）: _reward_handstand_feet_height_exp 告诉机器人“把脚举高”。

    Constraint（约束）: _reward_handstand_feet_on_air 告诉机器人“别用膝盖作弊，只能用手撑”。

    Balance（平衡）: _reward_handstand_orientation_l2 告诉机器人“身体要摆正，和重力线对其”。

    Sustain（维持）: _reward_handstand_feet_air_time 告诉机器人“坚持住这个姿势，越久越好”。
def _reward_handstand_feet_height_exp(self):
        feet_indices = [i for i, name in enumerate(self.rigid_body_names) if re.match(self.cfg.params.feet_name_reward["feet_name"], name)]
        # print(feet_indices)
        # print("Rigid body pos shape:", self.rigid_body_pos.shape)
        feet_indices_tensor = torch.tensor(feet_indices, dtype=torch.long, device=self.rigid_body_pos.device)
        # feet_indices_tensor = torch.tensor(feet_indices, dtype=torch.long, device=self.rigid_body_pos.device)
        foot_pos = self.rigid_body_pos[:, feet_indices_tensor, :]
        feet_height = foot_pos[..., 2]
        # print(feet_height)
        target_height = self.cfg.params.handstand_feet_height_exp["target_height"]
        std = self.cfg.params.handstand_feet_height_exp["std"]
        feet_height_error = torch.sum((feet_height - target_height) ** 2, dim=1)
        # print(torch.exp(-feet_height_error / (std**2)))
        return torch.exp(-feet_height_error / (std**2))
def _reward_handstand_feet_on_air(self):
        """
        改进版：同时检查脚部和膝盖的接触状态
        """
        # 1. 获取脚部索引（原有逻辑）
        feet_indices = [i for i, name in enumerate(self.rigid_body_names) 
                    if re.match(self.cfg.params.feet_name_reward["feet_name"], name)]
        feet_indices_tensor = torch.tensor(feet_indices, dtype=torch.long, device=self.rigid_body_pos.device)
        
        # 2. 获取膝盖/腿部其他可能接触地面的部位索引
        knee_indices = [i for i, name in enumerate(self.rigid_body_names) 
                    if re.match(r'.*(Knee|THIGH|SHANK).*', name.lower())]  # 匹配膝盖、大腿、小腿等
        knee_indices_tensor = torch.tensor(knee_indices, dtype=torch.long, device=self.rigid_body_pos.device)
        
        # 3. 检查脚部接触
        feet_contact = torch.norm(self.contact_forces[:, feet_indices_tensor, :], dim=-1) > 1.0
        
        # 4. 检查膝盖接触
        knee_contact = torch.norm(self.contact_forces[:, knee_indices_tensor, :], dim=-1) > 1.0
        
        # 5. 奖励条件：所有脚部未接触 AND 所有膝盖未接触
        reward = ((~feet_contact).float().prod(dim=1) * 
                (~knee_contact).float().prod(dim=1))
        
        return reward
 def _reward_handstand_feet_air_time(self):
        """
        改进版：计算手倒立时足部空中时间奖励，同时惩罚膝盖接触
        """
        threshold = self.cfg.params.handstand_feet_air_time["threshold"]

        # 获取脚部索引
        feet_indices = [i for i, name in enumerate(self.rigid_body_names) if re.match(self.cfg.params.feet_name_reward["feet_name"], name)]
        feet_indices_tensor = torch.tensor(feet_indices, dtype=torch.long, device=self.device)
        
        # 获取膝盖索引
        knee_indices = [i for i, name in enumerate(self.rigid_body_names) 
                    if re.match(r'.*(Knee|THIGH|SHANK).*', name.lower())]
        knee_indices_tensor = torch.tensor(knee_indices, dtype=torch.long, device=self.device)

        # 计算脚部接触状态
        feet_contact = self.contact_forces[:, feet_indices_tensor, 2] > 1.0  # (batch_size, num_feet)
        
        # 计算膝盖接触状态
        knee_contact = self.contact_forces[:, knee_indices_tensor, 2] > 1.0  # (batch_size, num_knees)
        any_knee_contact = knee_contact.any(dim=1)  # 任意膝盖接触就惩罚

        # 初始化状态变量（保持原有逻辑）
        if not hasattr(self,"last_contacts") or self.last_contacts.shape != feet_contact.shape:
            self.last_contacts = torch.zeros_like(feet_contact, dtype=torch.bool, device=feet_contact.device)
            
        if not hasattr(self,"feet_air_time") or self.feet_air_time.shape != feet_contact.shape:
            self.feet_air_time = torch.zeros_like(feet_contact, dtype=torch.float, device=feet_contact.device)
        
        # 原有悬空时间计算逻辑
        contact_filt = torch.logical_or(feet_contact, self.last_contacts)
        self.last_contacts = feet_contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt
        
        # 计算基础悬空时间奖励
        rew_airTime = torch.sum((self.feet_air_time - threshold) * first_contact, dim=1)
        
        # 添加膝盖接触惩罚：有膝盖接触时奖励为0
        rew_airTime = rew_airTime * (~any_knee_contact).float()
        
        self.feet_air_time *= ~contact_filt
        
        return rew_airTime
def _reward_handstand_orientation_l2(self):
        """
        姿态奖励：
        1. 使用 self.projected_gravity（机器人基座坐标系下的重力投影）来评估姿态。
        2. 目标重力方向通过配置传入（例如 [1, 0, 0] 表示目标为竖直向上）。
        3. 对比当前和目标重力方向的 L2 距离，偏差越大惩罚越大。
        """
        target_gravity = torch.tensor(
            self.cfg.params.handstand_orientation_l2["target_gravity"],
            device=self.device
        )

        return torch.sum((self.projected_gravity - target_gravity) ** 2, dim=1)