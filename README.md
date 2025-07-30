# robot_lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.2.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Overview

**robot_lab** is a RL extension library for robots, based on IsaacLab. It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

> [!NOTE]
> If you want to run policy in gazebo or real robot, please use [rl_sar](https://github.com/fan-ziqi/rl_sar) project.
>
> Discuss in [Github Discussion](https://github.com/fan-ziqi/robot_lab/discussions).

## Version Dependency

| robot_lab Version | Isaac Lab Version             | Isaac Sim Version       |
|------------------ | ----------------------------- | ----------------------- |
| `main` branch     | `main` branch                 | Isaac Sim 5.0           |
| `v2.2.0`          | `v2.2.0`                      | Isaac Sim 4.5 / 5.0     |
| `v2.1.1`          | `v2.1.1`                      | Isaac Sim 4.5           |
| `v1.1`            | `v1.4.1`                      | Isaac Sim 4.2           |

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone this repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

  ```bash
  git clone https://github.com/fan-ziqi/robot_lab.git
  ```

- Using a python interpreter that has Isaac Lab installed, install the library

  ```bash
  python -m pip install -e source/robot_lab
  ```

- Verify that the extension is correctly installed by running the following command to print all the available environments in the extension:

  ```bash
  python scripts/tools/list_envs.py
  ```

<details>

<summary>Set up IDE (Optional, click to expand)</summary>

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu. When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory. The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse. This helps in indexing all the python modules for intelligent suggestions while writing code.

</details>

<details>

<summary>Setup as Omniverse Extension (Optional, click to expand)</summary>

We provide an example UI extension that will load upon enabling your extension defined in `source/robot_lab/robot_lab/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of your repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon** (☰), then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to `robot_lab/source`
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon** (☰), then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

</details>

## Docker setup

<details>

<summary>Click to expand</summary>

### Building Isaac Lab Base Image

Currently, we don't have the Docker for Isaac Lab publicly available. Hence, you'd need to build the docker image
for Isaac Lab locally by following the steps [here](https://isaac-sim.github.io/IsaacLab/main/source/deployment/index.html).

Once you have built the base Isaac Lab image, you can check it exists by doing:

```bash
docker images

# Output should look something like:
#
# REPOSITORY                       TAG       IMAGE ID       CREATED          SIZE
# isaac-lab-base                   latest    28be62af627e   32 minutes ago   18.9GB
```

### Building robot_lab Image

Following above, you can build the docker container for this project. It is called `robot-lab`. However,
you can modify this name inside the [`docker/docker-compose.yaml`](docker/docker-compose.yaml).

```bash
cd docker
docker compose --env-file .env.base --file docker-compose.yaml build robot-lab
```

You can verify the image is built successfully using the same command as earlier:

```bash
docker images

# Output should look something like:
#
# REPOSITORY                       TAG       IMAGE ID       CREATED             SIZE
# robot-lab                        latest    00b00b647e1b   2 minutes ago       18.9GB
# isaac-lab-base                   latest    892938acb55c   About an hour ago   18.9GB
```

### Running the container

After building, the usual next step is to start the containers associated with your services. You can do this with:

```bash
docker compose --env-file .env.base --file docker-compose.yaml up
```

This will start the services defined in your `docker-compose.yaml` file, including robot-lab.

If you want to run it in detached mode (in the background), use:

```bash
docker compose --env-file .env.base --file docker-compose.yaml up -d
```

### Interacting with a running container

If you want to run commands inside the running container, you can use the `exec` command:

```bash
docker exec --interactive --tty -e DISPLAY=${DISPLAY} robot-lab /bin/bash
```

### Shutting down the container

When you are done or want to stop the running containers, you can bring down the services:

```bash
docker compose --env-file .env.base --file docker-compose.yaml down
```

This stops and removes the containers, but keeps the images.

</details>

## Try examples

You can use the following commands to run all environments:

RSL-RL:

```bash
# Train
python scripts/rsl_rl/base/train.py --task=<ENV_NAME> --headless

# Play
python scripts/rsl_rl/base/play.py --task=<ENV_NAME>
```

CusRL (**Experimental**:​​ Hydra/fix-seed/view-follow/policy-export not supported yet):

```bash
# Train
python scripts/cusrl/train.py --task=<ENV_NAME> --headless

# Play
python scripts/cusrl/play.py --task=<ENV_NAME>
```

The table below lists all available environments:

<table>
  <thead>
    <tr>
      <th style="text-align:center;">Category</th>
      <th>Robot Model</th>
      <th>Environment Name (ID)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:center;" rowspan="7">Quadruped</td>
      <td>Anymal D</td>
      <td>RobotLab-Isaac-Velocity-Rough-Anymal-D-v0</td>
    </tr>
    <tr>
      <td>Unitree Go2</td>
      <td>RobotLab-Isaac-Velocity-Rough-Unitree-Go2-v0</td>
    </tr>
    <tr>
      <td>Unitree B2</td>
      <td>RobotLab-Isaac-Velocity-Rough-Unitree-B2-v0</td>
    </tr>
    <tr>
      <td>Unitree A1</td>
      <td>RobotLab-Isaac-Velocity-Rough-Unitree-A1-v0</td>
    </tr>
    <tr>
      <td>Unitree A1 HandStand</td>
      <td>RobotLab-Isaac-Velocity-Flat-HandStand-Unitree-A1-v0</td>
    </tr>
    <tr>
      <td>Deeprobotics Lite3</td>
      <td>RobotLab-Isaac-Velocity-Rough-Deeprobotics-Lite3-v0</td>
    </tr>
    <tr>
      <td>Deeprobotics X30</td>
      <td>RobotLab-Isaac-Velocity-Rough-Deeprobotics-X30-v0</td>
    </tr>
    <tr>
      <td style="text-align:center;" rowspan="4">Wheeled</td>
      <td>Unitree Go2W</td>
      <td>RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0</td>
    </tr>
    <tr>
      <td>Unitree B2W</td>
      <td>RobotLab-Isaac-Velocity-Rough-Unitree-B2W-v0</td>
    </tr>
    <tr>
      <td>Deeprobotics M20</td>
      <td>RobotLab-Isaac-Velocity-Rough-Deeprobotics-M20-v0</td>
    </tr>
    <tr>
      <td>DDTRobot Tita</td>
      <td>RobotLab-Isaac-Velocity-Rough-DDTRobot-Tita-v0</td>
    </tr>
    <tr>
      <td style="text-align:center;" rowspan="7">Humanoid</td>
      <td>Unitree G1</td>
      <td>RobotLab-Isaac-Velocity-Rough-Unitree-G1-v0</td>
    </tr>
    <tr>
      <td>Unitree H1</td>
      <td>RobotLab-Isaac-Velocity-Rough-Unitree-H1-v0</td>
    </tr>
    <tr>
      <td>FFTAI GR1T1</td>
      <td>RobotLab-Isaac-Velocity-Rough-FFTAI-GR1T1-v0</td>
    </tr>
    <tr>
      <td>FFTAI GR1T2</td>
      <td>RobotLab-Isaac-Velocity-Rough-FFTAI-GR1T2-v0</td>
    </tr>
    <tr>
      <td>Booster T1</td>
      <td>RobotLab-Isaac-Velocity-Rough-Booster-T1-v0</td>
    </tr>
    <tr>
      <td>RobotEra Xbot</td>
      <td>RobotLab-Isaac-Velocity-Rough-RobotEra-Xbot-v0</td>
    </tr>
    <tr>
      <td>Openloong Loong</td>
      <td>RobotLab-Isaac-Velocity-Rough-Openloong-Loong-v0</td>
    </tr>
  </tbody>
</table>

Train AMP Dance for Unitree G1

```bash
# Train
python scripts/skrl/train.py --task RobotLab-Isaac-G1-AMP-Dance-Direct-v0 --algorithm AMP --headless

# Play
python scripts/skrl/play.py --task RobotLab-Isaac-G1-AMP-Dance-Direct-v0 --algorithm AMP --num_envs 32
```

> [!NOTE]
> If you want to control a **SINGLE ROBOT** with the keyboard during playback, add `--keyboard` at the end of the play script.
>
> ```
> Key bindings:
> ====================== ========================= ========================
> Command                Key (+ve axis)            Key (-ve axis)
> ====================== ========================= ========================
> Move along x-axis      Numpad 8 / Arrow Up       Numpad 2 / Arrow Down
> Move along y-axis      Numpad 4 / Arrow Right    Numpad 6 / Arrow Left
> Rotate along z-axis    Numpad 7 / Z              Numpad 9 / X
> ====================== ========================= ========================
> ```

* You can change `Rough` to `Flat` in the above configs.
* Record video of a trained agent (requires installing `ffmpeg`), add `--video --video_length 200`
* Play/Train with 32 environments, add `--num_envs 32`
* Play on specific folder or checkpoint, add `--load_run run_folder_name --checkpoint model.pt`
* Resume training from folder or checkpoint, add `--resume --load_run run_folder_name --checkpoint model.pt`
* To train with multiple GPUs, use the following command, where --nproc_per_node represents the number of available GPUs:
    ```bash
    python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 scripts/rsl_rl/base/train.py --task=<ENV_NAME> --headless --distributed
    ```
* To scale up training beyond multiple GPUs on a single machine, it is also possible to train across multiple nodes. To train across multiple nodes/machines, it is required to launch an individual process on each node.

    For the master node, use the following command, where --nproc_per_node represents the number of available GPUs, and --nnodes represents the number of nodes:
    ```bash
    python -m torch.distributed.run --nproc_per_node=2 --nnodes=2 --node_rank=0 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=localhost:5555 scripts/rsl_rl/base/train.py --task=<ENV_NAME> --headless --distributed
    ```
    Note that the port (`5555`) can be replaced with any other available port.
    For non-master nodes, use the following command, replacing `--node_rank` with the index of each machine:
    ```bash
    python -m torch.distributed.run --nproc_per_node=2 --nnodes=2 --node_rank=1 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=ip_of_master_machine:5555 scripts/rsl_rl/base/train.py --task=<ENV_NAME> --headless --distributed
    ```

## Add your own robot

This repository supports direct import of URDF, XACRO, and MJCF robot models without requiring pre-conversion to USD format.

```python
from robot_lab.assets.utils.usd_converter import (  # noqa: F401
    mjcf_to_usd,
    spawn_from_lazy_usd,
    urdf_to_usd,
    xacro_to_usd,
)

YOUR_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # for urdf
        func=spawn_from_lazy_usd,
        usd_path=urdf_to_usd(  # type: ignore
            file_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/your_robot/your_robot.urdf",
            merge_joints=True,
            fix_base=False,
        ),
        # for xacro
        func=spawn_from_lazy_usd,
        usd_path=xacro_to_usd(  # type: ignore
            file_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/your_robot/your_robot.xacro",
            merge_joints=True,
            fix_base=False,
        ),
        # for mjcf
        func=spawn_from_lazy_usd,
        usd_path=mjcf_to_usd(  # type: ignore
            file_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/your_robot/your_robot.xml",
            import_sites=True,
            fix_base=False,
        ),
        # ... other configuration parameters ...
    ),
    # ... other configuration parameters ...
)
```

Check your model import compatibility using:

```bash
python scripts/tools/check_robot.py {urdf,mjcf,xacro} path_to_your_model_file
```

Using the core framework developed as part of Isaac Lab, we provide various learning environments for robotics research.
These environments follow the `gym.Env` API from OpenAI Gym version `0.21.0`. The environments are registered using
the Gym registry.

Each environment's name is composed of `Isaac-<Task>-<Robot>-v<X>`, where `<Task>` indicates the skill to learn
in the environment, `<Robot>` indicates the embodiment of the acting agent, and `<X>` represents the version of
the environment (which can be used to suggest different observation or action spaces).

The environments are configured using either Python classes (wrapped using `configclass` decorator) or through
YAML files. The template structure of the environment is always put at the same level as the environment file
itself. However, its various instances are included in directories within the environment directory itself.
This looks like as follows:

```tree
source/robot_lab/tasks/manager_based/locomotion/
├── __init__.py
└── velocity
    ├── config
    │   └── unitree_a1
    │       ├── agent  # <- this is where we store the learning agent configurations
    │       ├── __init__.py  # <- this is where we register the environment and configurations to gym registry
    │       ├── flat_env_cfg.py
    │       └── rough_env_cfg.py
    ├── __init__.py
    └── velocity_env_cfg.py  # <- this is the base task configuration
```

The environments are then registered in the `source/robot_lab/tasks/manager_based/locomotion/velocity/config/unitree_a1/__init__.py`:

```python
gym.register(
    id="RobotLab-Isaac-Velocity-Flat-Unitree-A1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeA1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1FlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeA1FlatTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Unitree-A1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeA1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1RoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeA1RoughTrainerCfg",
    },
)
```

## Tensorboard

To view tensorboard, run:

```bash
tensorboard --logdir=logs
```

## Code formatting

A pre-commit template is given to automatically format the code.

To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing. In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

**Note: Replace `<path-to-isaac-lab>` with your own IsaacLab path.**

```json
{
    "python.analysis.extraPaths": [
        "${workspaceFolder}/source/robot_lab",
        "/<path-to-isaac-lab>/source/isaaclab",
        "/<path-to-isaac-lab>/source/isaaclab_assets",
        "/<path-to-isaac-lab>/source/isaaclab_mimic",
        "/<path-to-isaac-lab>/source/isaaclab_rl",
        "/<path-to-isaac-lab>/source/isaaclab_tasks",
    ]
}
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
