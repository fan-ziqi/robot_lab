# 旧版 OpenDoge 模型与策略回放

本次迁移原版 OpenDoge（不是 OpenDog V1.1），提供独立 Isaac Lab 回放入口。
模型和 ONNX 已复制到项目内，运行时不需要原 OpenDoge_train 目录。

在 `robot_lab` 目录下安装所需依赖：

```powershell
uv sync --extra importers --extra legacy-policy
```

无界面回放 20 秒，指令为前进 0.3 m/s：

```powershell
uv run --extra importers --extra legacy-policy python scripts/reinforcement_learning/play_opendoge_legacy.py --viz none --duration 20 --command 0.3 0 0 --report logs/opendoge-forward.json
```

可视化回放（Newton OpenGL）：

```powershell
uv run --extra importers --extra legacy-policy python scripts/reinforcement_learning/play_opendoge_legacy.py --viz newton_gl --duration 60 --command 0.3 0 0
```

`--policy` 可指定其他同机器人、同观测接口的旧版 ONNX。
默认使用 `flat_opendoge_9000_omni.onnx`。初始指令默认为零。
脚本按指定时长退出；JSON 记录最低/最终机身高度、低于 0.08 m 的采样数、平均身体坐标速度和水平位移。
成功运行不等于策略站稳或精确跟踪，需结合这些指标和可视化判断。

在已准备好的 Isaac Sim 独立环境中，可选择 `--physics isaacsim_physx --viz kit`。
默认使用 Newton MJWarp，接触与求解器不同于原 Isaac Gym PhysX；原训练效果不保证直接复现。

## 兼容参数

| 参数 | 值 |
|---|---|
| 关节顺序 | FL、FR、RL、RR，各 hip/thigh/calf；运行时按名称映射 |
| 默认站姿 | 每腿 `[0, 0.6, -1.5]` rad |
| 观测排列 | command × `[2,2,0.25]`、角速度 × 0.25、投影重力、关节位置偏差、关节速度 × 0.05、上次动作 |
| 历史 | 45 × 6 = 270，最新在前，初始全零 |
| 控制 | 200 Hz 仿真、100 Hz 推理，显式 PD 12/0.5 |
| 动作缩放 | 0.30，动作裁剪 ±100 |
| 力矩上限 | 每腿 6/6/9 Nm |
| armature | 0.005 |

资产配置：`robot_lab.assets.opendoge.OPENDOGE_CFG`。
可在后续任务中复用该配置；当前入口用于旧 HIM 策略推理，尚不支持续训其 `.pt` checkpoint。
源部署文件 `opendoge.yaml` 混用了 V1.1 参数，本迁移以旧版训练配置和旧版浏览器回放参数为准。
源许可原文随资产保存在 `assets/data/Robots/opendoge/SOURCE_LICENSE`。

## 本机验证（2026-11-04）

Newton MJWarp，原版 ONNX 未修改：

| 场景 | 仿真时长 | 最低/最终机身高度 | 结果 |
|---|---|---|---|
| 零速度站立 | 20 s | 0.150 / 0.153 m | 未低于 0.08 m，水平漂移约 2 mm |
| 前进指令 0.3 m/s | 20 s | 0.150 / 0.159 m | 未低于 0.08 m，平均机身前向速度 0.073 m/s，有侧向漂移 |

模型加载、ONNX 推理和控制循环已跑通；速度跟踪尚未达到原训练效果。
尚未实测 Isaac Sim PhysX 后端。完整日志指标在 `logs/opendoge-legacy-standing.json`
和 `logs/opendoge-legacy-forward.json`。

自动测试：14 通过，4 项可选 cusrl 测试因缺少依赖跳过；Ruff 检查通过。
沙箱内验证将 `WARP_CACHE_PATH` 指向项目的 `logs/warp-cache`，避开系统缓存目录写权限限制。
