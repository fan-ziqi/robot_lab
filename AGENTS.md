# 仓库指南

## 项目概述

**robot_lab** (v2.3.0) 是一个基于 Isaac Lab 2.3.0 的机器人强化学习扩展库。它为各种机器人平台（四足机器人、轮式机器人、人形机器人）提供模块化的强化学习训练框架，使用 Isaac Sim 中的 GPU 加速仿真。并在sim_m.py中使用mujuco进行模型的验证，调试pid等。


## 项目现在目的

模仿其他机器人环境配置等，导入我自己的四足轮腿机器人模型，配置资产，训练环境，以及agents,增加rewards，训练调试以满足可以上台阶的需求，并在sim_m.py进行mujucoo中进行调试pid，并验证效果
其中我自己的四足轮腿机器人模型地址：/home/liu/Desktop/robot_lab/source/robot_lab/data/Robots/myrobots/mydog
配置资产地址：/home/liu/Desktop/robot_lab/source/robot_lab/robot_lab/assets/mydog.py
训练环境及reward策略：/home/liu/Desktop/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/myrobots_mydogw/rough_env_cfg.py
ppo_cfg：/home/liu/Desktop/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/myrobots_mydogw/agents/rsl_rl_ppo_cfg.py

## 代码规定
关键代码需要写中文注释
## 项目结构与模块组织
Isaac Lab 扩展位于 。核心 Python 包放在 ，其中  定义机器人模型， 则包含 direct、manager_based 和 beyondmimic 环境。训练入口位于 ， 包含 URDF/MJCF 转换与动作预处理等工具。机器人网格与 URDF/MJCF 文件保存在 。Docker 相关文件在 ，文档资产在 ，运行时产物写入  与 。
## 架构概览

### 配置层次结构

项目使用三层配置继承系统：

1. **基础配置** (`velocity_env_cfg.py`)
   - 定义通用环境结构（场景、命令、动作、观测、事件、奖励、终止条件、课程学习）
   - 位置：`tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`

2. **机器人特定 Rough 配置** (`rough_env_cfg.py`)
   - 继承基础配置并针对特定机器人定制
   - 使用地形生成器（多难度级别）
   - 包含高度扫描器用于地形感知
   - 启用完整领域随机化（质量、摩擦力等）
   - 位置：`config/<type>/<robot>/rough_env_cfg.py`

3. **机器人特定 Flat 配置** (`flat_env_cfg.py`)
   - 继承 Rough 配置
   - 覆盖地形类型为平面（`terrain_type = "plane"`）
   - 禁用高度扫描器
   - 适用于初始策略学习

这种继承模式允许在训练模式之间轻松切换。

### 目录结构关键组件

```
robot_lab/
├── source/robot_lab/robot_lab/
│   ├── assets/                    # 机器人定义 (ArticulationCfg)
│   │   ├── unitree.py            # Unitree 机器人配置
│   │   ├── mydog.py              # 自定义机器人配置
│   │   └── ...                   # 其他制造商
│   │
│   ├── tasks/manager_based/locomotion/velocity/
│   │   ├── velocity_env_cfg.py   # 基础环境配置
│   │   ├── mdp/                  # MDP 组件（观测、奖励、事件等）
│   │   └── config/               # 机器人特定配置
│   │       ├── quadruped/        # 四足机器人
│   │       ├── wheeled/          # 轮式机器人
│   │       ├── humanoid/         # 人形机器人
│   │       └── others/           # 特殊任务
│   │
│   └── data/Robots/              # 机器人网格/URDF/MJCF 文件
│       ├── unitree/
│       ├── myrobots/
│       └── ...
│
├── scripts/
│   ├── reinforcement_learning/
│   │   ├── rsl_rl/              # RSL-RL 训练（默认）
│   │   ├── cusrl/               # CusRL 训练（实验性）
│   │   └── skrl/                # SKRL 训练（实验性）
│   └── tools/                   # 工具脚本
│
├── logs/                        # 训练日志和检查点
└── outputs/                     # Hydra 输出
```

### 机器人资产配置系统

每个机器人制造商有自己的 Python 文件（例如 `assets/unitree.py`、`assets/mydog.py`），定义：
- `ArticulationCfg`：描述如何生成和控制机器人
- URDF/MJCF 路径（指向 `data/Robots/`）
- 初始姿态和关节配置
- 执行器配置（刚度/阻尼参数）
- 关节组（例如轮式机器人的 "legs"、"wheels"）

**轮式机器人模式示例**（mydog）：
```python
MYDOG_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(...),
    init_state=ArticulationCfg.InitialStateCfg(...),
    actuators={
        "legs": ImplicitActuatorCfg(stiffness=..., damping=...),    # 位置控制
        "wheels": ImplicitActuatorCfg(stiffness=..., damping=...),  # 速度控制
    }
)
```

### 环境注册系统

环境使用 Gym 注册表系统在 `__init__.py` 文件中注册：

命名约定：`RobotLab-Isaac-<Task>-<Terrain>-<Robot>-v<Version>`

示例：
```python
gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Unitree-A1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeA1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1RoughPPORunnerCfg",
    },
)
```

### 观测-动作架构

**观测**分为两组：
- **Policy Group**：智能体的观测（带噪声/损坏）
  - 基座速度（线性/角速度）
  - 投影重力
  - 关节位置/速度
  - 前一个动作
  - 高度扫描（用于粗糙地形）
  - 速度命令

- **Critic Group**：价值函数观测（无噪声）
  - 与 policy 相同结构但无损坏

**动作**根据机器人类型定义：
- **四足机器人**：关节位置目标（缩放后）
- **轮式机器人**：混合动作
  - 腿部关节位置控制
  - 轮子关节速度控制
- **人形机器人**：关节位置目标（特殊处理）

### 奖励系统

模块化奖励系统，包含 70+ 个奖励项，按类别组织：
- 根部惩罚：姿态、高度、加速度
- 关节惩罚：力矩、速度、加速度、限制
- 动作惩罚：平滑性、速率限制
- 接触惩罚：不希望的接触、过度力
- 速度跟踪：指数跟踪奖励
- 步态奖励：足部空中时间、接触模式、对称性
- 特殊奖励：向上姿态、轮子行为

权重可设为 0 并通过 `disable_zero_weight_rewards()` 自动移除。

### 领域随机化事件

事件在不同模式触发：
- **Startup**：材料属性、质量、惯性、质心随机化
- **Reset**：关节状态、基座姿态、执行器增益、外力
- **Interval**：在回合期间推动机器人

这实现了鲁棒的 sim-to-real 迁移。

### MuJoCo 集成

项目包含 MuJoCo 支持：
- **MJCF 文件**：位于 `data/Robots/*/mjcf/`
- **仿真脚本**：`sim_m.py` 和 `run_mujoco.py` 运行独立 MuJoCo 仿真
  - 使用来自 Isaac Lab 的训练策略
  - 实现可配置增益的 PD 控制
  - 用于在不同物理引擎中测试策略

**关键差异**：MuJoCo 使用不同的关节排序和索引（dof_ids 映射）。

**MuJoCo 脚本关键参数**：
- `SIM_DT`：物理步长（默认 0.0002s，5kHz）
- `DECIMATION`：控制频率分频（默认 100，即 50Hz 控制频率）
- `RAMP_UP_TIME`：软启动时间（默认 1.5s）
- PD 增益：`kp_leg`、`kd_leg`（腿部）和 `kp_wheel`、`kd_wheel`（轮子）

## 添加新机器人的工作流程

1. **添加资产文件**
   - 将 URDF/网格文件添加到 `data/Robots/<manufacturer>/<robot>/`
   - 在 `assets/<manufacturer>.py` 中创建 `ArticulationCfg`

2. **创建任务配置**
   - 在 `config/<type>/<robot>/` 创建目录
   - 编写 `rough_env_cfg.py`（继承基础配置并定制）
   - 编写 `flat_env_cfg.py`（继承 rough 配置并覆盖地形）

3. **配置智能体**
   - 在 `agents/` 中创建 RL 智能体配置（`rsl_rl_ppo_cfg.py`、`cusrl_ppo_cfg.py`）

4. **注册环境**
   - 在 `__init__.py` 中使用 `gym.register()` 注册环境
   - 遵循命名约定

5. **验证**
   - 运行 `python scripts/tools/list_envs.py` 确认注册
   - 用小数量环境测试训练（`--num_envs 64`）

## 训练最佳实践

1. **先在平面地形训练**以实现初始学习
2. **使用课程学习**应对粗糙地形
3. **迭代调整奖励权重**
4. **监控 TensorBoard**：`tensorboard --logdir=logs`
5. **频繁在 play 模式测试**
6. **使用领域随机化**实现鲁棒性
7. **日志位置**：`logs/rsl_rl/<task_name>/<timestamp>/`
8. **检查点频率**：每 100 次迭代保存

## 重要架构模式

### SceneEntityCfg 模式
```python
SceneEntityCfg("robot", joint_names=".*_hip_joint")
```
在整个代码库中使用正则表达式匹配引用场景元素。

### Configclass 装饰器
```python
@configclass
class MyEnvCfg(BaseEnvCfg):
    ...
```
将 dataclass 转换为与 Isaac Lab 兼容的配置对象。

### 智能体配置分离
- 环境配置定义任务
- 智能体配置（在 `agents/` 中）定义学习算法
- 清晰的分离允许交换 RL 算法

### MDP 模块系统
- 所有 MDP 函数在 `mdp/` 子目录中
- 按以下组织：观测、奖励、事件、命令、课程
- 通过通配符导入：`from .mdp import *`

## 当前 MyDog 机器人配置

**MyDog** 是展示轮式腿式模式的自定义机器人：
- **资产**：`assets/mydog.py` - 定义带腿和轮子执行器的 `MYDOG_CFG`
- **URDF**：`data/Robots/myrobots/mydog/urdf/thunder_nohead.urdf`
- **MJCF**：`data/Robots/myrobots/mydog/mjcf/thunder2_v1.xml`
- **任务配置**：`config/wheeled/myrobots_mydogw/rough_env_cfg.py`
  - 混合动作空间：腿部位置控制，轮子速度控制
  - 特殊观测：排除轮子位置（无限旋转）
  - 轮子特定奖励和约束
- **当前分支**：`feature/mydog`
- **最近训练**：`logs/rsl_rl/mydog_rough/` 和 `mydog_flat/`

## 构建、测试与开发命令


## 编码风格与命名约定
目标平台为 Linux 上的 Python 3.11，使用 4 空格缩进并由  限制 120 列宽。导入使用  （含自定义区段）组织，通过  触发 flake8、codespell 与 pyright 基础类型检查。环境命名遵循  约定，并与  下的目录结构保持一致。文件名使用 snake_case；仅为 Isaac/Omniverse API 或基于模式的标识符保留 camelCase。为每个模块添加简洁的文档字符串，描述机器人、任务或智能体配置。

## 测试指南
当新增环境或修改注册表时，将  视为快速冒烟测试。强化学习回归依赖使用固定随机种子的 ；度量写入 ，并在训练稳定性相关时分享 tensorboard 日志。使用  捕获可视化内容并记录 GPU 数量与  声明。BeyondMimic 或 AMP 路径需要重新生成  动作，并在提交 PR 前通过  进行验证。

## 提交与 PR 指南
近期提交混合了精简单行与传统  前缀。推荐使用 （例如 ）并提及相关 issue（如 ）。本地运行 pre-commit，排除生成的检查点或日志，确保新资产存放在正确的  品牌目录。PR 需要说明场景、列出验证命令、注明硬件（GPU 数量、Isaac Sim 版本），并附上保存于  的截图或短视频以说明任何可视化变更。添加动作或网格时请引用外部数据集或许可证。

## 安全与配置提示
不要将 、个人 Omniverse 路径或专有网格归档提交到仓库。提供下载指引，而不是提交供应商数据。共享检查点时，通过制品存储发布并仅在 PR 中链接；上传前请清理日志中的内部主机名或凭证。
