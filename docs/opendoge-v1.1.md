# OpenDog V1.1 模型和 LainLab 策略迁移

模型和四套策略来自 LainLab `feat/opendoge`，固定版本
`7f6df34c67c455a80b9d296344a53e8a969cd765`。资产和权重已复制进 robot_lab，
运行时不依赖 LainLab 或 OpenDoge_train 目录，也不需要安装 mjlab。
旧版模型和 `play_opendoge_legacy.py` 仍可独立使用。

## 运行

在 robot_lab 目录安装依赖：ONNX 运行时与旧版共用 `legacy-policy` extra。

```powershell
uv sync --extra importers --extra legacy-policy
$env:WARP_CACHE_PATH="$PWD/logs/warp-cache"
.venv/Scripts/python.exe scripts/reinforcement_learning/play_opendoge_v1_1.py --viz newton_gl --duration 60 --command 0.3 0 0
```

Windows 可以双击 `run-opendoge-v1.1.cmd` 打开 Newton Viewer，默认前进 0.3 m/s。
窗口由本机图形会话显示；无图形环境请使用 `--viz none`。
命令行示例：

```powershell
# 平地策略，保存 20 秒指标
.venv/Scripts/python.exe scripts/reinforcement_learning/play_opendoge_v1_1.py --viz none --duration 20 --command 0.3 0 0 --report logs/v11-forward.json
# 起身策略：默认从侧躺开始，也可指定 --initial-pose back
.venv/Scripts/python.exe scripts/reinforcement_learning/play_opendoge_v1_1.py --profile getup --viz newton_gl
# 同一个多技能网络：walk、handstand、getup、jump
.venv/Scripts/python.exe scripts/reinforcement_learning/play_opendoge_v1_1.py --profile skills --skill handstand --viz newton_gl
.venv/Scripts/python.exe scripts/reinforcement_learning/play_opendoge_v1_1.py --profile skills --skill jump --viz newton_gl
# 地形策略目前在平面上检查推理和控制，不代表通过崎岖地形验收
.venv/Scripts/python.exe scripts/reinforcement_learning/play_opendoge_v1_1.py --profile rough --viz newton_gl --command 0.3 0 0
```

`--policy` 可以覆盖同一接口的 ONNX 文件。`--profile` 决定观测格式，不能仅凭
48 维输入区分平地和起身权重，需要正确选择 profile。旧版 270 维权重会被拒绝。
getup 和静态技能使用零速度指令。`--initial-pose stand|side|back` 可指定初始姿态。

## 接口和模型

可复用资产配置：`robot_lab.assets.opendoge_v1_1.OPENDOGE_V1_1_CFG`。
模型约 4.5914 kg，12 个驱动关节，左右腿轴向和默认姿态与旧版不同。
左右腿默认姿态分别为 `[0,.8,-1.6]` 和 `[0,-.8,1.6]`，基座初高 .158 m。

| 项目 | V1.1 接口 |
|---|---|
| 关节与动作顺序 | FL、FR、RL、RR，各 hip/thigh/calf；运行时按关节名映射 |
| 平地/起身观测 | 机身线速度 3、角速度 3、投影重力 3、关节位置偏差 12、关节速度 12、上次原始动作 12、指令 3，共 48 |
| 地形观测 | 48 + 165 个高度样本，样本值为传感器高度减地形高度，再除以 5 |
| 多技能观测 | 48 + jump twist 3 + 技能 one-hot 4 + phase sin/cos 2，共 57 |
| 归一化 | 包含在原始 ONNX 内；外部不重复归一化，也不使用旧版缩放 |
| IMU | 基座原点上方 .03 m；线速度补偿 angular_velocity × offset |
| 动作 | 默认关节角 + 原始动作 × .25，不裁剪原始动作 |
| 控制 | dt .005 s，decimation 2，200 Hz 物理 / 100 Hz 策略 |
| 电机 | 显式 PD 12/.5，armature .005，每腿力矩上限 6/6/9 Nm |

多技能时钟仅在 jump 下以 2.5 秒为周期前进；jump 指令块是
`[phase_sin,phase_cos,left_ground]`，速度放在单独 twist 块。
left_ground 需要真实足底接触反馈：先接触地面，再全脚离地且机身高于 .166 m 才置位，
保持至重置；其他技能的 phase 为 `[0,1]`。

## 验证范围

本次提供模型接入和现有导出策略的 Isaac Lab 回放入口。复制的 `.pt` 保留原始训练
信息，尚未接入 robot_lab 的训练/续训 runner。原始奖励、随机化、课程和训练环境尚未移植。
没有自动进行技能间切换；通过参数选择单项技能。

默认后端为 Newton MJWarp。它与原始 mjlab MuJoCo 的接触模型、显式/内置 PD 和
求解步长存在差异；URDF 入口不包含 MJCF 特有的足底 solimp/condim 设置。
源 MJCF 随资产保留作为对照。Isaac Sim PhysX / OvPhysX 参数入口存在，但未实测。

rough 回放的场景只有 z=0 平面；165 个 terrain-only 垂直射线高度由真实机身高度
解析计算，未接入崎岖地形生成或非平面扫描。此模式用于接口验证。

本机验证结果见文末及 `logs/opendoge-v11-*.json`。

## 本机验证（2026-11-05）

以下为单机器人、固定初态、无观测噪声/外力的 Newton MJWarp 回放；每项 20 秒。
侧躺/仰躺测试并非原项目随机跌倒分布的成功率测评。

| 场景 | 实测 |
|---|---|
| 平地站立 | 始终直立；最终高度 .148 m；水平漂移约 2.5 mm |
| 平地前进 .3 m/s | 平均前向 .259 m/s；世界坐标位移 [5.174,.206] m |
| 独立起身策略：侧躺/仰躺 | 分别 .45/.52 s 首次恢复站立，最终高度均 .156 m |
| rough 策略在平面上 | 平均前向 .284 m/s；世界坐标侧移 -1.132 m，跟踪存在偏差 |
| 多技能 walk，指令 .3 m/s | 平均机身前向 .252 m/s；世界坐标侧移 1.772 m，跟踪存在偏差 |
| 多技能 getup，侧躺 | .48 s 首次站立；最终高度 .156 m |
| 多技能 handstand | 最终高度 .218 m，重力对齐 .995；98.45% 采样满足对齐>.9 且高度>.19 m；水平漂移约 .40 m |
| 多技能 jump | 最高机身 .211 m，较参考站高 .151 m 上升约 .060 m；最终 .156 m 且直立 |

所有回放均正常退出。动作、状态和 ONNX 输出通过有限值检查。
四套 ONNX 与原始 checkpoint 的归一化 + ELU MLP 在同一测试输入上的最大绝对误差
分别为 1.32e-6、9.54e-7、4.71e-6、2.51e-6。
完整 pytest：21 通过、4 项因未安装可选 cusrl 跳过；Ruff 检查和格式检查通过。
已知依赖弃用警告不影响这些检查。独立代码审查未发现默认 Newton 回放的阻断问题；
审查发现的求解器速度限幅遗漏已修正并增加回归测试。

这些结果支持模型加载和导出策略回放链路可用；速度、航向和技能漂移仍需进一步校准。
尚未验证随机跌倒、多地形、技能切换、实机或训练收敛。

### 提速测试

随后在同一平面场景下对 flat/rough 策略逐档测试，每档 20 秒、固定站立初态：

| 前进指令 m/s | flat 实际平均前向 m/s | rough 实际平均前向 m/s |
|---|---|---|
| 1.0 | .943 | .975 |
| 1.5 | 1.341 | 1.465 |
| 2.0 | 1.478 | 1.942 |
| 2.5 | 1.558 | 2.180 |
| 3.0 | 摔倒 | 摔倒 |

截至本轮测试，保持直立的最高实测平均速度约 2.18 m/s（rough 策略在平面上），
但该档 20 秒的世界坐标位移为 [40.73,-14.28] m，存在明显航向偏差。
flat 对更高速度指令的跟踪接近饱和。本轮不确定精确失稳边界，
也不代表所有初态、扰动下的可靠速度上限。
仿真没有电机转速—力矩曲线、电池等硬件约束，因此不能把这些速度解释为实机极速。
逐档报告与汇总保存在 `logs/opendoge-speed-sweep/`。
