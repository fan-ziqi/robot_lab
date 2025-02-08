# robot_lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.0.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

## Overview

**robot_lab** is an extension project based on Isaac Lab. It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

> [!IMPORTANT]
> This code repository is currently at **[v2.0](https://github.com/fan-ziqi/robot_lab/releases/tag/v2.0)**, which corresponds to **[IsaacLab-v2.0.0](https://github.com/isaac-sim/IsaacLab/releases/tag/v2.0.0)**.
>
> If you prefer to use **[IsaacLab-v1.4.1](https://github.com/isaac-sim/IsaacLab/releases/tag/v1.4.1)**, please refer to the **[v1.1](https://github.com/fan-ziqi/robot_lab/releases/tag/v1.1)** release of this repository.

> [!NOTE]
> If you want to run policy in gazebo or real robot, please use [rl_sar](https://github.com/fan-ziqi/rl_sar) project.
>
> [Click to discuss on Discord](https://discord.gg/vmVjkhVugU)

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

### Base Locomotion

Anymal D

```bash
# Train
python scripts/rsl_rl/base/train.py --task RobotLab-Isaac-Velocity-Rough-Anymal-D-v0 --headless
# Play
python scripts/rsl_rl/base/play.py --task RobotLab-Isaac-Velocity-Rough-Anymal-D-v0
```

Unitree A1

```bash
# Train
python scripts/rsl_rl/base/train.py --task RobotLab-Isaac-Velocity-Rough-Unitree-A1-v0 --headless
# Play
python scripts/rsl_rl/base/play.py --task RobotLab-Isaac-Velocity-Rough-Unitree-A1-v0
```

Unitree Go2

```bash
# Train
python scripts/rsl_rl/base/train.py --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2-v0 --headless
# Play
python scripts/rsl_rl/base/play.py --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2-v0
```

Unitree Go2W

```bash
# Train
python scripts/rsl_rl/base/train.py --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0 --headless
# Play
python scripts/rsl_rl/base/play.py --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0
```

Unitree B2

```bash
# Train
python scripts/rsl_rl/base/train.py --task RobotLab-Isaac-Velocity-Rough-Unitree-B2-v0 --headless
# Play
python scripts/rsl_rl/base/play.py --task RobotLab-Isaac-Velocity-Rough-Unitree-B2-v0
```

Unitree B2W

```bash
# Train
python scripts/rsl_rl/base/train.py --task RobotLab-Isaac-Velocity-Rough-Unitree-B2W-v0 --headless
# Play
python scripts/rsl_rl/base/play.py --task RobotLab-Isaac-Velocity-Rough-Unitree-B2W-v0
```

FFTAI GR1T1

```bash
# Train
python scripts/rsl_rl/base/train.py --task RobotLab-Isaac-Velocity-Rough-FFTAI-GR1T1-v0 --headless
# Play
python scripts/rsl_rl/base/play.py --task RobotLab-Isaac-Velocity-Rough-FFTAI-GR1T1-v0
```

Unitree H1

```bash
# Train
python scripts/rsl_rl/base/train.py --task RobotLab-Isaac-Velocity-Rough-Unitree-H1-v0 --headless
# Play
python scripts/rsl_rl/base/play.py --task RobotLab-Isaac-Velocity-Rough-Unitree-H1-v0
```

Unitree G1

```bash
# Train
python scripts/rsl_rl/base/train.py --task RobotLab-Isaac-Velocity-Rough-Unitree-G1-v0 --headless
# Play
python scripts/rsl_rl/base/play.py --task RobotLab-Isaac-Velocity-Rough-Unitree-G1-v0
```

The above configs are rough, you can change `Rough` to `Flat`

**Note**

* Record video of a trained agent (requires installing `ffmpeg`), add `--video --video_length 200`
* Play/Train with 32 environments, add `--num_envs 32`
* Play on specific folder or checkpoint, add `--load_run run_folder_name --checkpoint model.pt`
* Resume training from folder or checkpoint, add `--resume --load_run run_folder_name --checkpoint model.pt`

### AMP Locomotion

The code for AMP training refers to [AMP_for_hardware](https://github.com/Alescontrela/AMP_for_hardware)

Unitree A1

```bash
# Retarget motion files
python source/robot_lab/robot_lab/third_party/amp_utils/scripts/retarget_kp_motions.py
# Replay AMP data
python scripts/rsl_rl/amp/replay_amp_data.py --task RobotLab-Isaac-Velocity-Flat-Amp-Unitree-A1-v0
# Train
python scripts/rsl_rl/amp/train.py --task RobotLab-Isaac-Velocity-Flat-Amp-Unitree-A1-v0 --headless
# Play
python scripts/rsl_rl/amp/play.py --task RobotLab-Isaac-Velocity-Flat-Amp-Unitree-A1-v0
```

### HandStand

Supports four directions of handstand, use `handstand_type` in the configuration file to switch.

Unitree A1

```bash
# Train
python scripts/rsl_rl/base/train.py --task RobotLab-Isaac-Velocity-Flat-HandStand-Unitree-A1-v0 --headless
# Play
python scripts/rsl_rl/base/play.py --task RobotLab-Isaac-Velocity-Flat-HandStand-Unitree-A1-v0
```

## Add your own robot

For example, to generate Unitree A1 usd file:

```bash
python scripts/tools/convert_urdf.py a1.urdf source/robot_lab/data/Robots/Unitree/A1/a1.usd  --merge-join
```

Check [import_new_asset](https://docs.robotsfan.com/isaaclab/source/how-to/import_new_asset.html) for detail

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
source/robot_lab/tasks/locomotion/
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

The environments are then registered in the `source/robot_lab/tasks/locomotion/velocity/config/unitree_a1/__init__.py`:

```python
gym.register(
    id="RobotLab-Isaac-Velocity-Flat-Unitree-A1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeA1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1FlatPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Unitree-A1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeA1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1RoughPPORunnerCfg",
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

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/robot_lab"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```

## Citation

Please cite the following if you use this code or parts of it:

```
@software{fan-ziqi2024robot_lab,
  author = {fan-ziqi},
  title = {robot_lab: An extension project based on Isaac Lab.},
  url = {https://github.com/fan-ziqi/robot_lab},
  year = {2024}
}
```
