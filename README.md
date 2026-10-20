# robot_lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-6.2%20(optional)-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-3.0.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://docs.python.org/3/whatsnew/3.12.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Overview

**robot_lab** is a RL extension library for robots, based on IsaacLab. It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

The table below lists representative environments. Use `isaaclab list_envs` for all task IDs and agent configurations:

| Category   | Robot Model         | Environment Name (<ENV_NAME>)                                      | Screenshot |
|------------|---------------------|------------------------------------------------------------|------------|
| **Quadruped** | [Anymal D](https://www.anybotics.com/robotics/anymal) | Isaac-Velocity-Flat-Anymal-D-v0 | <img src="./docs/imgs/anymal_d.png" alt="anymal_d" width="75"> |
|            | [Unitree Go2](https://www.unitree.com/go2) | RobotLab-Isaac-Velocity-Rough-Unitree-Go2-v0 | <img src="./docs/imgs/unitree_go2.png" alt="unitree_go2" width="75"> |
|            | [Unitree B2](https://www.unitree.com/b2) | RobotLab-Isaac-Velocity-Rough-Unitree-B2-v0 | <img src="./docs/imgs/unitree_b2.png" alt="unitree_b2" width="75"> |
|            | [Unitree A1](https://www.unitree.com/a1) | RobotLab-Isaac-Velocity-Rough-Unitree-A1-v0 | <img src="./docs/imgs/unitree_a1.png" alt="unitree_a1" width="75"> |
|            | [Deeprobotics Lite3](https://www.deeprobotics.cn/robot/index/product1.html) | RobotLab-Isaac-Velocity-Rough-Deeprobotics-Lite3-v0 | <img src="./docs/imgs/deeprobotics_lite3.png" alt="Lite3" width="75"> |
|            | [Zsibot ZSL1](https://www.zsibot.com/zsl1) | RobotLab-Isaac-Velocity-Rough-Zsibot-ZSL1-v0 | <img src="./docs/imgs/zsibot_zsl1.png" alt="zsibot_zsl1" width="75"> |
|            | [Magiclab MagicDog](https://www.magiclab.top/dog) | RobotLab-Isaac-Velocity-Rough-MagicLab-Dog-v0 | <img src="./docs/imgs/magiclab_magicdog.png" alt="magiclab_magicdog" width="75"> |
|            | [Agibot D1](https://www.agibot.com/) | RobotLab-Isaac-Velocity-Rough-Agibot-D1-v0 | <img src="./docs/imgs/agibot_d1.png" alt="magiclab_magicdog" width="75">  |
| **Wheeled** | [Unitree Go2W](https://www.unitree.com/go2-w) | RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0 | <img src="./docs/imgs/unitree_go2w.png" alt="unitree_go2w" width="75"> |
|            | [Unitree B2W](https://www.unitree.com/b2-w) | RobotLab-Isaac-Velocity-Rough-Unitree-B2W-v0 | <img src="./docs/imgs/unitree_b2w.png" alt="unitree_b2w" width="75"> |
|            | [Deeprobotics M20](https://www.deeprobotics.cn/robot/index/lynx.html) | RobotLab-Isaac-Velocity-Rough-Deeprobotics-M20-v0 | <img src="./docs/imgs/deeprobotics_m20.png" alt="deeprobotics_m20" width="75"> |
|            | [DDTRobot Tita](https://directdrive.com/product_TITA) | RobotLab-Isaac-Velocity-Rough-DDTRobot-Tita-v0 | <img src="./docs/imgs/ddtrobot_tita.png" alt="ddtrobot_tita" width="75"> |
|            | [Zsibot ZSL1W](https://www.zsibot.com/zsl1) | RobotLab-Isaac-Velocity-Rough-Zsibot-ZSL1W-v0 | <img src="./docs/imgs/zsibot_zsl1w.png" alt="zsibot_zsl1w" width="75"> |
|            | [Magiclab MagicDog-W](https://www.magiclab.top/dog-w) | RobotLab-Isaac-Velocity-Rough-MagicLab-Dog-W-v0 | <img src="./docs/imgs/magiclab_magicdog_w.png" alt="magiclab_magicdog_w" width="75"> |
| **Humanoid** | [Unitree G1](https://www.unitree.com/g1) | RobotLab-Isaac-Velocity-Rough-Unitree-G1-v0 | <img src="./docs/imgs/unitree_g1.png" alt="unitree_g1" width="75"> |
|             | [Unitree H1](https://www.unitree.com/h1) | RobotLab-Isaac-Velocity-Rough-Unitree-H1-v0 | <img src="./docs/imgs/unitree_h1.png" alt="unitree_h1" width="75"> |
|             | [FFTAI GR1T1](https://www.fftai.com/products-gr1) | RobotLab-Isaac-Velocity-Rough-FFTAI-GR1T1-v0 | <img src="./docs/imgs/fftai_gr1t1.png" alt="fftai_gr1t1" width="75"> |
|             | [FFTAI GR1T2](https://www.fftai.com/products-gr1) | RobotLab-Isaac-Velocity-Rough-FFTAI-GR1T2-v0 | <img src="./docs/imgs/fftai_gr1t2.png" alt="fftai_gr1t2" width="75"> |
|             | [Booster T1](https://www.boosterobotics.com/) | RobotLab-Isaac-Velocity-Rough-Booster-T1-v0 | <img src="./docs/imgs/booster_t1.png" alt="booster_t1" width="75"> |
|             | [RobotEra Xbot](https://www.robotera.com/) | RobotLab-Isaac-Velocity-Rough-RobotEra-Xbot-v0 | <img src="./docs/imgs/robotera_xbot.png" alt="robotera_xbot" width="75"> |
|             | [Openloong Loong](https://www.openloong.net/) | RobotLab-Isaac-Velocity-Rough-Openloong-Loong-v0 | <img src="./docs/imgs/openloong_loong.png" alt="openloong_loong" width="75"> |
|             | [RoboParty ATOM01](https://roboparty.cn/) | RobotLab-Isaac-Velocity-Rough-RoboParty-ATOM01-v0 | <img src="./docs/imgs/roboparty_atom01.png" alt="roboparty_atom01" width="75"> |
|             | [Magiclab MagicBot-Gen1](https://www.magiclab.top/human) | RobotLab-Isaac-Velocity-Rough-MagicLab-Bot-Gen1-v0 | <img src="./docs/imgs/magiclab_magicbot_gen1.png" alt="magiclab_magicbot_gen1" width="75"> |
|             | [Magiclab MagicBot-Z1](https://www.magiclab.top/z1) | RobotLab-Isaac-Velocity-Rough-MagicLab-Bot-Z1-v0 | <img src="./docs/imgs/magiclab_magicbot_z1.png" alt="magiclab_magicbot_z1" width="75"> |

> [!NOTE]
> If you want to run policy in gazebo or real robot, please use [rl_sar](https://github.com/fan-ziqi/rl_sar) project.
>
> Discuss in [Github Discussion](https://github.com/fan-ziqi/robot_lab/discussions) or [Discord](http://www.robotsfan.com/dc_robot_lab).

## Version Dependency

| robot_lab Version | Isaac Lab Version             | Isaac Sim Version         |
|------------------ | ----------------------------- | ------------------------- |
| `main` branch     | `v3.0.0`                      | Isaac Sim 6.2 (optional for Newton) |
| `v2.3.2`          | `v2.3.2`                      | Isaac Sim 4.5 / 5.0 / 5.1 |
| `v2.2.2`          | `v2.2.1`                      | Isaac Sim 4.5 / 5.0       |
| `v2.1.1`          | `v2.1.1`                      | Isaac Sim 4.5             |
| `v1.1`            | `v1.4.1`                      | Isaac Sim 4.2             |

## Installation

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). This source project follows Isaac Lab's
external-project template and expects an Isaac Lab **v3.0.0** checkout beside it:

```text
workspaces/
├── IsaacLab/     # v3.0.0
└── robot_lab/
```

For a new workspace:

```bash
git clone --branch v3.0.0 --depth 1 https://github.com/isaac-sim/IsaacLab.git
git clone https://github.com/fan-ziqi/robot_lab.git
cd robot_lab
uv sync --extra importers
uv run --extra importers isaaclab list_envs --show_presets
```

If the checkouts already exist, verify the Isaac Lab version and run `uv sync --extra importers`
from this repository. Relative source paths are declared in `pyproject.toml`; update them if the
Isaac Lab directory moves. Commit `uv.lock` with dependency changes.

For an existing, fully provisioned Isaac Lab environment, install only this extension:

```bash
uv pip install --no-deps -e .
uv run --active --no-project isaaclab list_envs
```

### Newton without Isaac Sim

Velocity tasks use Isaac Lab's `RoughPhysicsCfg` and default to Newton MJWarp. The `importers`
extra provides URDF/MJCF conversion without installing the full Isaac Sim runtime or Kit.

```bash
uv run --extra importers isaaclab zero_agent --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 --num_envs 4 --max_steps 100 --viz none physics=newton_mjwarp
uv run --extra importers isaaclab train --rl_library rsl_rl --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 --viz none physics=newton_mjwarp
uv run --extra importers isaaclab play --rl_library rsl_rl --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 --checkpoint latest --num_envs 4 --viz newton_gl physics=newton_mjwarp
```

The standalone importer retains the namespace `isaacsim.asset.importer`. It converts assets and
registers USD schemas without `SimulationApp`. Project initialization imports it before the first
USD stage is created when standalone importers are available.

### Isaac Sim PhysX

Use a separate environment for Isaac Sim. Its regular `isaacsim` package shadows the standalone
importer namespace, so URDF import in that environment requires Kit.

```bash
UV_PROJECT_ENVIRONMENT=.venv-isaacsim uv sync --extra isaacsim
UV_PROJECT_ENVIRONMENT=.venv-isaacsim uv run --extra isaacsim isaaclab zero_agent --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 --num_envs 4 --max_steps 100 --viz none physics=isaacsim_physx
UV_PROJECT_ENVIRONMENT=.venv-isaacsim uv run --extra isaacsim isaaclab train --rl_library rsl_rl --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 --viz none physics=isaacsim_physx
UV_PROJECT_ENVIRONMENT=.venv-isaacsim uv run --extra isaacsim isaaclab play --rl_library rsl_rl --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 --checkpoint latest --num_envs 4 --viz kit physics=isaacsim_physx
```

When Newton runs in the full Isaac Sim environment, add `--viz kit` for URDF import. USD assets can
run without Kit unless another selected component requires it. The AMP and BeyondMimic tasks
currently configure Isaac Sim PhysX directly.

## Train, play and record

Use the upstream CLI and select the RL library with `--rl_library`. RSL-RL is included; install
`--extra skrl` for tasks with a skrl agent configuration. Select a task and its available libraries
with `isaaclab list_envs`.

```bash
# Zero or random actions
uv run --extra importers isaaclab random_agent --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 --num_envs 4 --max_steps 100 --viz none physics=newton_mjwarp

# Choose a checkpoint explicitly, or use latest/best
uv run --extra importers isaaclab play --rl_library rsl_rl --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 --checkpoint /path/to/model.pt --viz newton_gl physics=newton_mjwarp

# Resume training
uv run --extra importers isaaclab train --rl_library rsl_rl --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 --checkpoint /path/to/model.pt --viz none physics=newton_mjwarp

# Record a bounded playback through the selected visualizer
uv run --extra importers --extra video isaaclab play --rl_library rsl_rl --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 --checkpoint latest --video --video_length 200 --viz newton_gl physics=newton_mjwarp

# Multiple GPUs
uv run --extra importers isaaclab train_multigpu --rl_library rsl_rl --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 --num_gpus 2 --viz none physics=newton_mjwarp

# Training logs
uv run tensorboard --logdir logs
```

The scripts under `scripts/reinforcement_learning/train.py`, `play.py`, and the existing `rsl_rl`
and `skrl` directories delegate to upstream entry points. The compatibility scripts still accept
`--headless`; new commands use `--viz none`.

### CusRL

CusRL is a project integration and uses its own scripts. Select its extra together with the runtime:

```bash
uv run --extra importers --extra cusrl python scripts/reinforcement_learning/cusrl/train.py --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 --viz none physics=newton_mjwarp
uv run --extra importers --extra cusrl python scripts/reinforcement_learning/cusrl/play.py --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 --checkpoint /path/to/model.pt --viz newton_gl physics=newton_mjwarp
```

### Motion tracking and AMP

Acquire the motion data under its original license. Convert retargeted G1 CSV files for BeyondMimic
with the repository tool, then set the resulting archive in the environment configuration:

```bash
UV_PROJECT_ENVIRONMENT=.venv-isaacsim uv run --extra isaacsim python scripts/tools/beyondmimic/csv_to_npz.py -f /path/to/motion.csv --input_fps 60 --viz none --exit_after_save
UV_PROJECT_ENVIRONMENT=.venv-isaacsim uv run --extra isaacsim python scripts/tools/beyondmimic/replay_npz.py -f /path/to/motion.npz --viz kit
UV_PROJECT_ENVIRONMENT=.venv-isaacsim uv run --extra isaacsim isaaclab train --rl_library rsl_rl --task RobotLab-Isaac-BeyondMimic-Flat-Unitree-G1-v0 --viz none env.commands.motion.motion_file=/path/to/motion.npz
```

For G1 AMP, use a motion archive containing joint names, body names, positions, rotations and
velocities. See the [motion format and conversion tools](source/robot_lab/robot_lab/tasks/direct/g1_amp/motions/README.md).

```bash
UV_PROJECT_ENVIRONMENT=.venv-isaacsim uv run --extra isaacsim --extra skrl isaaclab train --rl_library skrl --task RobotLab-Isaac-G1-AMP-Dance-Direct-v0 --algorithm AMP --viz none
UV_PROJECT_ENVIRONMENT=.venv-isaacsim uv run --extra isaacsim --extra skrl isaaclab play --rl_library skrl --task RobotLab-Isaac-G1-AMP-Dance-Direct-v0 --algorithm AMP --checkpoint latest --num_envs 4 --viz kit
```

### Symmetry, distillation and handstand

ANYmal-D registers separate RSL-RL agent configurations for symmetry and distillation:

```bash
uv run --extra importers isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Flat-Anymal-D-v0 --agent rsl_rl_with_symmetry_cfg_entry_point --viz none physics=newton_mjwarp agent.algorithm.symmetry_cfg.use_data_augmentation=true
uv run --extra importers isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Flat-Anymal-D-v0 --agent rsl_rl_distillation_cfg_entry_point --checkpoint /path/to/teacher/model.pt --viz none physics=newton_mjwarp
uv run --extra importers isaaclab train --rl_library rsl_rl --task RobotLab-Isaac-Velocity-Flat-HandStand-Unitree-A1-v0 --viz none physics=newton_mjwarp
```

For playback in a custom scene, use `scripts/reinforcement_learning/rsl_rl/play_cs.py` with
`--map /path/to/scene.usd`, the task, checkpoint and backend arguments.

## Add a robot or task

Tasks follow the Gymnasium API. Common velocity configuration and MDP terms belong to the task
family; robot-specific scenes, registrations and agent configurations belong to `config`:

```text
source/robot_lab/robot_lab/
├── assets/
│   ├── data/Robots/               # Packaged project-owned URDFs, meshes and textures
│   └── unitree.py                # Asset configuration
└── tasks/manager_based/locomotion/velocity/
    ├── velocity_env_cfg.py        # Shared task and upstream physics presets
    ├── mdp/                      # Project-specific terms and upstream lazy exports
    └── config/quadruped/unitree_a1/
        ├── __init__.py           # Gymnasium registrations
        ├── rough_env_cfg.py
        ├── flat_env_cfg.py
        └── agents/
```

Use `ROBOT_LAB_ASSETS_DIR` from `robot_lab.assets` for asset paths. It resolves correctly in an
editable checkout and an installed wheel. Register new tasks in their package's `__init__.py` with
module-path environment and agent entry points, following the existing registrations. The
`isaaclab.tasks` package entry point discovers them without copying launchers.

Use one mechanical asset for both backends and composable upstream schema configurations for
backend-specific properties. Keep `run_asset_transformer=True` and `run_multi_physics_conversion=True`
for URDF/MJCF import. Set solver capacities directly in the task's
`self.sim.physics.newton_mjwarp.solver_cfg`. Follow the official
[asset preparation guide](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/docs/source/how-to/prepare_asset_for_newton.rst)
and [2.x to 3.0 migration guide](https://isaac-sim.github.io/IsaacLab/develop/source/isaaclab/migrating.html).

Assets use each backend's native public joint/body ordering by default. For checkpoint transfer,
configure `ArticulationCfg.joint_ordering` and `body_ordering` through the upstream ordering API.
Checkpoints trained with the previous explicit ordering profiles or unmerged body layouts require
their original conventions. A1 velocity tasks randomize physical trunk/link masses while preserving the foot bodies.
G1 uses its physical pelvis as the root; velocity tasks merge fixed sensor frames.
New BeyondMimic archives store joint/body names and resolve motion data by name. Reconvert legacy
archives if their body layout differs, using an output FPS matching the task control rate.

Validate initial states, contacts, observations, rewards and repeated resets in both backends before
training. Smoke tests and short training do not establish policy convergence or transfer quality.

## Development

```bash
uv run pytest
uv run pre-commit run --all-files
uv run --extra importers isaaclab --editor
```

The editor command selects the active interpreter and writes the local Pyright configuration. The
`setup_python_env` VS Code task invokes it. Select that interpreter in VS Code or Cursor; see the
upstream [editor setup](https://isaac-sim.github.io/IsaacLab/develop/source/isaaclab/installation.html).
If ROS injects incompatible pytest plugins into the environment, run tests with
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest`.

The optional `robot_lab.ui_extension_example` module is a Kit UI example; task discovery and
training do not load it.

## Docker

Build an Isaac Lab v3.0.0 base image using its Docker workflow, then set `ISAACLAB_BASE_IMAGE`
in `docker/.env.base` to that image. The extension image installs this project into the base
image's provisioned environment:

```bash
cd docker
docker compose --env-file .env.base --file docker-compose.yaml build robot-lab
docker compose --env-file .env.base --file docker-compose.yaml run --rm robot-lab
# Inside the container, use the existing environment:
uv run --active --no-project isaaclab list_envs
```

## Citation

Please cite the following if you use this code or parts of it:

```
@software{fan-ziqi2024robot_lab,
  author = {Ziqi Fan},
  title = {robot_lab: RL Extension Library for Robots, Based on IsaacLab.},
  url = {https://github.com/fan-ziqi/robot_lab},
  year = {2024}
}
```

## Acknowledgements

The project uses some code from the following open-source code repositories:

- [linden713/humanoid_amp](https://github.com/linden713/humanoid_amp)
- [HybridRobotics/whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking)
