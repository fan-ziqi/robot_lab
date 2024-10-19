# robot_lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.2.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-1.2.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Overview

**robot_lab** is an extension project based on Isaac Lab. It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

If you want to run policy in gazebo or real robot, please use [rl_sar](https://github.com/fan-ziqi/rl_sar) project.

Todo:

- [x] AMP training
- [ ] VAE training code
- [x] Sim to Sim transfer(Gazebo)
- [ ] Sim to Real transfer(Unitree A1)

[Click to discuss on Discord](https://discord.gg/vmVjkhVugU)

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/source/setup/installation/index.html). We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone the repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

  ```bash
  git clone https://github.com/fan-ziqi/robot_lab.git
  ```

- Using a python interpreter that has Isaac Lab installed, install the library

  ```bash
  python -m pip install -e ./exts/robot_lab
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

We provide an example UI extension that will load upon enabling your extension defined in `exts/robot_lab/robot_lab/ui_extension_example.py`. For more information on UI extensions, enable and check out the source code of the `omni.isaac.ui_template` extension and refer to the introduction on [Isaac Sim Workflows 1.2.3. GUI](https://docs.omniverse.nvidia.com/isaacsim/latest/introductory_tutorials/tutorial_intro_workflows.html#gui).

To enable your extension, follow these steps:

1. **Add the search path of your repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon** (☰), then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to `robot_lab/exts`
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory (`IsaacLab/source/extensions`)
    - Click on the **Hamburger Icon** (☰), then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

</details>

## Try examples

FFTAI GR1T1

```bash
# Train
python scripts/rsl_rl/train.py --task RobotLab-Isaac-Velocity-Flat-FFTAI-GR1T1-v0 --headless
# Play
python scripts/rsl_rl/play.py --task RobotLab-Isaac-Velocity-Flat-FFTAI-GR1T1-v0
```

Anymal D

```bash
# Train
python scripts/rsl_rl/train.py --task RobotLab-Isaac-Velocity-Flat-Anymal-D-v0 --headless
# Play
python scripts/rsl_rl/play.py --task RobotLab-Isaac-Velocity-Flat-Anymal-D-v0
```

Unitree A1

```bash
# Train
python scripts/rsl_rl/train.py --task RobotLab-Isaac-Velocity-Flat-Unitree-A1-v0 --headless
# Play
python scripts/rsl_rl/play.py --task RobotLab-Isaac-Velocity-Flat-Unitree-A1-v0
```

Unitree Go2W (Unvalible for now)

```bash
# Train
python scripts/rsl_rl/train.py --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2W-v0 --headless
# Play
python scripts/rsl_rl/play.py --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2W-v0
```

Unitree H1

```bash
# Train
python scripts/rsl_rl/train.py --task RobotLab-Isaac-Velocity-Flat-Unitree-H1-v0 --headless
# Play
python scripts/rsl_rl/play.py --task RobotLab-Isaac-Velocity-Flat-Unitree-H1-v0
```

The above configs are flat, you can change Flat to Rough

**Note**

* Record video of a trained agent (requires installing `ffmpeg`), add `--video --video_length 200`
* Play/Train with 32 environments, add `--num_envs 32`
* Play on specific folder or checkpoint, add `--load_run run_folder_name --checkpoint model.pt`
* Resume training from folder or checkpoint, add `--resume --load_run run_folder_name --checkpoint model.pt`

## AMP training

The code for AMP training refers to [AMP_for_hardware](https://github.com/Alescontrela/AMP_for_hardware)

Unitree A1

```bash
# Retarget motion files
python exts/robot_lab/amp_utils/scripts/retarget_kp_motions.py
# Replay AMP data
python scripts/rsl_rl/replay_amp_data.py --task RobotLab-Isaac-Velocity-Flat-Amp-Unitree-A1-v0
# Train
python scripts/rsl_rl/train_amp.py --task RobotLab-Isaac-Velocity-Flat-Amp-Unitree-A1-v0 --headless
# Play
python scripts/rsl_rl/play_amp.py --task RobotLab-Isaac-Velocity-Flat-Amp-Unitree-A1-v0
```

## Add your own robot

For example, to generate Unitree A1 usd file:

```bash
python scripts/tools/convert_urdf.py a1.urdf exts/robot_lab/data/Robots/Unitree/A1/a1.usd  --merge-join
```

Check [import_new_asset](https://docs.robotsfan.com/isaaclab/source/how-to/import_new_asset.html) for detail

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
        "<path-to-ext-repo>/exts/robot_lab"
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
  title = {{robot_lab: An extension project based on Isaac Lab.}},
  url = {https://github.com/fan-ziqi/robot_lab},
  year = {2024}
}
```
