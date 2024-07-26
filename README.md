# robot_lab

robot_lab is an extension project based on the IsaacLab framework. It has a clear code framework and you can quickly migrate to your own robot.

[Click to discuss on Discord](https://discord.gg/vmVjkhVugU)

## Get Ready

You need to install `Isaac Lab`.

## Installation

Using a python interpreter that has Isaac Lab installed, install the library

```bash
python -m pip install -e ./exts/robot_lab
```

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

Unitree H1

```bash
# Train
python scripts/rsl_rl/train.py --task RobotLab-Isaac-Velocity-Flat-Unitree-H1-v0 --headless
# Play
python scripts/rsl_rl/play.py --task RobotLab-Isaac-Velocity-Flat-Unitree-H1-v0
```

The above configs are flat, you can change Flat to Rough

## AMP training

Unitree A1

```bash
# Retarget motion files
python exts/robot_lab/amp_utils/scripts/retarget_kp_motions.py
# Train
python scripts/rsl_rl/train_amp.py --task RobotLab-Isaac-Velocity-Flat-Amp-Unitree-A1-v0 --headless
# Play
python scripts/rsl_rl/play_amp.py --task RobotLab-Isaac-Velocity-Flat-Amp-Unitree-A1-v0
# Replay AMP data
python scripts/rsl_rl/replay_amp_data.py --task RobotLab-Isaac-Velocity-Flat-Amp-Unitree-A1-v0
```

## Add your own robot

To convert urdf, you need to run `convert_urdf.py` of dir `IsaacLab`

For example, to generate A1 usd file:

```bash
./isaaclab.sh -p source/standalone/tools/convert_urdf.py <YOUR_ROBOT>.urdf source/extensions/omni.isaac.lab_assets/data/Robots/<YOUR_ROBOT>/<YOUR_ROBOT>.usd --merge-join
```

Check [import_new_asset](https://docs.robotsfan.com/isaaclab/source/how-to/import_new_asset.html) for detail
